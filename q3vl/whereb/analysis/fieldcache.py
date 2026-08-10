"""Re-run a frozen checkpoint over a handful of samples and keep their fields.

``per_sample.jsonl`` carries scalars only, by design -- 896 samples x 7 contexts
of ``(s, m_low, m_hi)`` would be gigabytes per eval.  But three of the failure
mechanisms in :mod:`attribution` are statements *about the fields*:

* is the direction wrong?  ``cos(w_dir, w_dir*)``;
* is ``s`` wrong or is ``rho`` wrong?  swap one oracle component in at a time and
  see which swap repairs the mask;
* is the hi tier losing what the low tier had, on a single-primitive fit?

So this module re-runs the checkpoint on **just the tail samples** (a few dozen),
which is also what the panels need in order to draw a prediction at all.  It is
the only part of the tool that needs a GPU, it is optional, and everything it
produces is additive: without it the report still ships, with those mechanisms
marked ``not_tested`` rather than silently absent.

Two invariants copied from the eval path, because a field measured differently
from the gate it is supposed to explain explains nothing:

* ``float32`` everywhere on the analytic path (``require_dtype=torch.float32``),
  the review-blocker-B4 rule;
* the oracle mask is recomputed from the published Where-A latent through the
  *same* ``phi_dir`` and the *same* single guided upsample, exactly as
  ``evaluate._oracle_mask`` does.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import torch

__all__ = ["FieldCache", "build_field_cache"]


class FieldCache:
    """Reader for what :func:`build_field_cache` wrote."""

    def __init__(self, root: str | Path):
        self.root = Path(root)
        self.scalars: dict[str, dict[str, Any]] = {}
        path = self.root / "fields.jsonl"
        if path.exists():
            with path.open("r", encoding="utf-8") as fh:
                for line in fh:
                    if line.strip():
                        r = json.loads(line)
                        self.scalars[r["sample_id"]] = r

    def __contains__(self, sample_id: str) -> bool:
        return sample_id in self.scalars

    def arrays(self, sample_id: str) -> dict[str, np.ndarray]:
        p = self.root / f"{sample_id}.npz"
        if not p.exists():
            return {}
        with np.load(p) as z:
            return {k: z[k].astype(np.float32) for k in z.files}

    def facts(self) -> dict[str, Any]:
        meta = self.root / "cache_meta.json"
        return json.loads(meta.read_text()) if meta.exists() else {}


def _cos(a: torch.Tensor, b: torch.Tensor) -> float:
    a, b = a.reshape(-1).double(), b.reshape(-1).double()
    n = float(a.norm() * b.norm())
    return float(a @ b) / n if n else 0.0


@torch.no_grad()
def build_field_cache(
    sample_ids: Sequence[str],
    *,
    out_dir: str | Path,
    checkpoint: Path,
    arm: str,
    split: str = "V_where",
    context: str = "generated",
    device: str = "cuda",
    vlm_checkpoint: Path | None = None,
    model_dir: Path | None = None,
    attn: str = "flash_attention_2",
    dtype: str = "bfloat16",
    batch_size: int = 2,
    store_hi: bool = True,
) -> dict[str, Any]:
    """Run ``checkpoint`` over ``sample_ids`` and write the fields to ``out_dir``.

    Returns a provenance dict (checkpoint step, basis digest, n samples).
    """
    from q3vl.train.collator import Sft2SegCollator
    from q3vl.train.modeling import load_model, load_processor
    from q3vl.where.readout import apply_readout
    from q3vl.whereb.config import (
        BASIS_ARM, GENCTX_DIR, MODEL_DIR, ORACLE_NAMESPACE, SFT_CHECKPOINT,
        WHERE_A_MASKVIEW_DIR, WHERE_A_ORACLE_DIR, arm_config,
    )
    from q3vl.whereb.data import BatchBuilder, open_dataset
    from q3vl.whereb.fields import load_basis, predict_fields
    from q3vl.whereb.hiddens import FrozenVLM
    from q3vl.whereb.metrics import active_primitive_count, soft_iou_value
    from q3vl.whereb.model import WhereBModel
    from q3vl.whereb.stores import GenContextStore, OracleStore

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    cfg = arm_config(arm)
    keep = list(dict.fromkeys(sample_ids))

    processor, _special = load_processor(str(model_dir or MODEL_DIR), 2048)
    collator = Sft2SegCollator(processor, max_length=2048, system_prompt=None)
    vlm_model = load_model(str(vlm_checkpoint or SFT_CHECKPOINT),
                           attn_implementation=attn, dtype=dtype).to(device).eval()
    vlm = FrozenVLM(vlm_model, processor, device=device)
    basis = load_basis(BASIS_ARM).to(device)

    ds, info = open_dataset(split, maskview_root=WHERE_A_MASKVIEW_DIR)
    wanted = set(keep)
    ds.refs = [r for r in ds.refs if r.sample_id in wanted]
    missing = wanted - {r.sample_id for r in ds.refs}

    oracle_root = Path(WHERE_A_ORACLE_DIR) / BASIS_ARM / ORACLE_NAMESPACE / split
    oracle = OracleStore(oracle_root)
    genctx = GenContextStore(Path(GENCTX_DIR) / split)
    builder = BatchBuilder(collator, vlm, basis, cfg, oracle=oracle,
                           genctx=genctx, device=device)

    model = WhereBModel(cfg).to(device)
    ck = torch.load(checkpoint, map_location="cpu", weights_only=False)
    model.load_state_dict(ck["model"])
    model.eval()

    rows: list[dict[str, Any]] = []
    order = [i for i in range(len(ds))]
    for lo in range(0, len(order), batch_size):
        samples = [ds[i] for i in order[lo:lo + batch_size]]
        batch = builder.build(samples, [context] * len(samples))
        out = model(**batch.inputs)
        for j, tgt in enumerate(batch.targets):
            params = {k: v.float() for k, v in out.select(j).items()}
            gh, gw = tgt["grid_h"], tgt["grid_w"]
            f = predict_fields(tgt["phi_dir"].float(), params, cfg.readout, gh, gw,
                               guide_hi=tgt["guide_hi"].float(),
                               up_cfg=cfg.upsample, require_dtype=torch.float32)
            gt_hi = tgt["mask_hi"].float()
            rho_pred = {k: v for k, v in params.items()
                        if k not in ("w0", "w_raw", "alpha_raw")}
            row: dict[str, Any] = {
                "sample_id": tgt["sample_id"], "context": context,
                "iou_pred": soft_iou_value(f["m_hi"].reshape(gt_hi.shape), gt_hi),
                "active_primitives": active_primitive_count(cfg.readout, rho_pred),
                "alpha": float(f["alpha"]), "w0": float(f["w0"]),
                "s_low_std": float(f["s_low"].std(unbiased=False)),
            }
            arrays: dict[str, np.ndarray] = {
                "s_low": f["s_low"].reshape(gh, gw).cpu().numpy().astype(np.float16),
                "m_low": f["m_low"].reshape(gh, gw).cpu().numpy().astype(np.float16),
            }
            if store_hi:
                arrays["m_hi"] = f["m_hi"].reshape(gt_hi.shape).cpu().numpy().astype(np.float16)

            lat = builder.oracle.latent(tgt["sample_id"], cfg.readout) if not tgt["is_global"] else None
            if lat is not None:
                from q3vl.where.basis import alpha_of, w_dir_of

                lat = lat.to(tgt["phi_dir"].device)
                o_params = {"w0": lat.w0, "w_raw": lat.w_raw,
                            "alpha_raw": lat.alpha_raw, **lat.rho}
                o_params = {k: v.to(tgt["phi_dir"].device).float()
                            for k, v in o_params.items()}
                fo = predict_fields(tgt["phi_dir"].float(), o_params, cfg.readout,
                                    gh, gw, guide_hi=tgt["guide_hi"].float(),
                                    up_cfg=cfg.upsample, require_dtype=torch.float32)
                rho_star = {k: v for k, v in o_params.items()
                            if k not in ("w0", "w_raw", "alpha_raw")}
                # one component swapped in at a time: whichever swap repairs the
                # mask is the component that was broken
                m_os_pr = apply_readout(cfg.readout, fo["s_hi"].reshape(-1), rho_pred)
                m_ps_or = apply_readout(cfg.readout, f["s_hi"].reshape(-1), rho_star)
                row.update({
                    "iou_oracle": soft_iou_value(fo["m_hi"].reshape(gt_hi.shape), gt_hi),
                    "iou_oracle_s_pred_rho": soft_iou_value(
                        m_os_pr.reshape(gt_hi.shape), gt_hi),
                    "iou_pred_s_oracle_rho": soft_iou_value(
                        m_ps_or.reshape(gt_hi.shape), gt_hi),
                    "w_dir_cos": _cos(w_dir_of(params["w_raw"]),
                                      w_dir_of(lat.w_raw.float())),
                    "alpha_oracle": float(alpha_of(lat.alpha_raw.float())),
                    "s_star_std": float(fo["s_low"].std(unbiased=False)),
                })
                arrays["s_star_low"] = fo["s_low"].reshape(gh, gw).cpu().numpy().astype(np.float16)
                arrays["m_star_low"] = fo["m_low"].reshape(gh, gw).cpu().numpy().astype(np.float16)
            else:
                row["oracle"] = None
            np.savez_compressed(out_dir / f"{tgt['sample_id']}.npz", **arrays)
            rows.append(row)

    with (out_dir / "fields.jsonl").open("w", encoding="utf-8") as fh:
        for r in rows:
            fh.write(json.dumps(r, ensure_ascii=False) + "\n")
    meta = {
        "checkpoint": str(checkpoint), "step": ck.get("step"), "arm": arm,
        "readout": cfg.readout, "split": split, "context": context,
        "n_requested": len(keep), "n_written": len(rows),
        "missing_sample_ids": sorted(missing),
        "basis_digest": basis.digest(),
        "checkpoint_basis_digest": ck.get("basis_digest"),
        "dataset": info,
    }
    if ck.get("basis_digest") and ck["basis_digest"] != basis.digest():
        raise RuntimeError(
            f"checkpoint was trained against basis {ck['basis_digest'][:12]} but "
            f"{basis.digest()[:12]} is loaded; the fields would not be the ones "
            "the eval scored"
        )
    (out_dir / "cache_meta.json").write_text(
        json.dumps(meta, indent=2, ensure_ascii=False), encoding="utf-8")
    return meta
