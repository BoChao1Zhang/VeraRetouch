"""Protocol 5.4 / 5.6 -- four-context evaluation, strata and the gate board.

    "Every checkpoint must report the four contexts separately, never as one
     average: GT / generated / null / shuffled."

Each context is a full pass over the selection split.  Per-sample rows go to
``per_sample.jsonl`` (protocol 13.1) and carry their stratum keys, so the
``image.upscaled`` and ``winner_confidence`` breakdowns the reviewer asked for
are a group-by rather than a second run.

The "relative to the per-image oracle" gates need the oracle mask *in the same
space as the prediction*: it is recomputed here from the published Where-A
latent through the same ``phi_dir`` and the same single guided upsample, so the
ratio compares two masks and not two pipelines.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Callable, Iterable, Sequence

import torch

from .config import (
    EXTRA_STRATA_KEYS,
    PAIRED_AREA_RATIO_RANGE,
    STRATA_KEYS,
    ArmConfig,
)
from .context import CONTEXT_MODES, SHUFFLED
from .data import BatchBuilder, WhereBDataset
from .fields import predict_fields
from .metrics import (
    active_primitive_bucket,
    active_primitive_count,
    arm_metrics,
    attribution_section,
    antonym_invariance,
    center_prior_field,
    evaluate_gates,
    gt_area_k,
    hard_iou,
    instruction_paired_delta,
    paired_delta,
    sample_metrics,
    topk_mask,
    summarise,
)
from .model import WhereBModel
from .prefetch import SamplePrefetcher, prefetch_warmers

__all__ = ["evaluate_context", "evaluate_arm", "strata_report", "write_per_sample"]


def _chunks(xs: Sequence[int], n: int) -> Iterable[list[int]]:
    for i in range(0, len(xs), n):
        yield list(xs[i:i + n])


@torch.no_grad()
def evaluate_context(
    model: WhereBModel,
    builder: BatchBuilder,
    dataset: WhereBDataset,
    arm_cfg: ArmConfig,
    mode: str,
    *,
    batch_size: int = 4,
    limit: int | None = None,
    with_oracle: bool = True,
    prefetch_workers: int | None = None,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    model.eval()
    idx = list(range(len(dataset) if limit is None else min(limit, len(dataset))))
    rows: list[dict[str, Any]] = []
    skipped: list[dict[str, str]] = []
    # kept only for the same-image paired difference (A-5); ~6 KB per sample
    fields: dict[str, dict[str, Any]] = {}

    # PERF-1: the periodic eval walks the split four times and used to pay the
    # same serial NFS latency as training did (~8.5 min per checkpoint).  Same
    # order, same samples -- only built ahead of the forward.
    chunks = ([(i, mode) for i in chunk] for chunk in _chunks(idx, batch_size))
    supply = SamplePrefetcher(dataset, chunks, workers=prefetch_workers,
                              warm=prefetch_warmers(builder))
    for _chunk, built in supply:
        samples = []
        for s in built:
            if mode == SHUFFLED and builder.shuffle_index is not None:
                if builder.shuffle_index.partner_of(s.sample_id) is None:
                    skipped.append({"sample_id": s.sample_id, "reason": "no_shuffle_partner"})
                    continue
            samples.append(s)
        if not samples:
            continue
        batch = builder.build(samples, [mode] * len(samples))
        out = model(**batch.inputs)
        for j, tgt in enumerate(batch.targets):
            params = {k: v.float() for k, v in out.select(j).items()}
            # float32 everywhere, exactly as the training path (review blocker
            # B4): the gate must measure the function training optimises.
            f = predict_fields(
                tgt["phi_dir"].float(), params, arm_cfg.readout,
                tgt["grid_h"], tgt["grid_w"],
                guide_hi=tgt["guide_hi"].float(), up_cfg=arm_cfg.upsample,
                require_dtype=torch.float32,
            )
            m_or = grid_or = None
            if with_oracle and tgt.get("has_oracle"):
                m_or, grid_or = _oracle_mask(builder, tgt, arm_cfg)
            # amendment A-5: the grid-level and centre-prior columns are computed
            # on the F_pre grid, where the matched-area top-k rule makes two
            # fields comparable (both own exactly k cells).
            gh, gw = tgt["grid_h"], tgt["grid_w"]
            met = sample_metrics(
                f["m_hi"].reshape(tgt["mask_hi"].shape), tgt["mask_hi"].float(),
                s_pred=f["s_low"], s_star=tgt.get("s_star"), m_oracle=m_or,
                grid_pred=f["m_low"].reshape(gh, gw),
                grid_gt=tgt["mask_low"].reshape(gh, gw).float(),
                grid_oracle=(grid_or.reshape(gh, gw) if grid_or is not None else None),
            )
            ctx = batch.contexts[j]
            met.update({
                "sample_id": tgt["sample_id"], "context": mode,
                "context_provenance": ctx.provenance,
                "format_failure": ctx.format_failure,
                "instruction_swapped": ctx.instruction is not None,
                "n_where_tokens": ctx.n_tokens,
                "has_oracle": bool(tgt.get("has_oracle")),
                **{k: tgt["meta"].get(k) for k in STRATA_KEYS},
            })
            # WEVAL-1: the stratum is computed from the PREDICTED rho, so it can
            # only be filled here, where `params` exists -- taking it from
            # `tgt["meta"]` above (as the other strata do) silently yields None
            # for every row and `strata_report` then reports {"None": 896}.
            n_act = active_primitive_count(arm_cfg.readout, params)
            met["active_primitive_count"] = n_act
            met["active_primitive_bucket"] = active_primitive_bucket(n_act)
            # NOT setdefault: STRATA_KEYS already inserted the key, so a meta
            # dict whose render_mode is absent would leave it as None and the
            # row would vanish from *both* the local and the global aggregate
            # (review nit N5).
            met["render_mode"] = ("global" if tgt["is_global"]
                                  else met.get("render_mode") or "local")
            if met["render_mode"] not in ("local", "global"):
                raise AssertionError(
                    f"{tgt['sample_id']}: render_mode={met['render_mode']!r}; "
                    "summarise() would drop this row from every stratum"
                )
            rows.append(met)
            fields[tgt["sample_id"]] = {
                "grid_pred": f["m_low"].detach().reshape(gh, gw).cpu(),
                "grid_gt": tgt["mask_low"].detach().reshape(gh, gw).float().cpu(),
                "source_image_id": tgt["meta"].get("source_image_id"),
                "instruction": tgt["meta"].get("instruction"),
            }

    summary = summarise(rows)
    summary["instruction_paired"] = _instruction_paired(fields)
    summary["n_skipped"] = len(skipped)
    summary["skipped"] = skipped[:50]
    summary["context"] = mode
    summary["format_stats"] = builder.stats().get(mode)
    return rows, summary


def _paired_rows(fields: dict[str, dict[str, Any]], *, use_prior: bool
                 ) -> list[dict[str, Any]]:
    """Cross-score every same-image pair, either with the real field or with the
    zero-parameter centre prior (which is the calibration column, F-B1)."""
    by_img: dict[str, list[str]] = {}
    for sid, f in fields.items():
        by_img.setdefault(str(f["source_image_id"]), []).append(sid)
    rows = []
    for img, members in by_img.items():
        if len(members) < 2:
            continue
        for i, sid in enumerate(members):
            partner = members[(i + 1) % len(members)]
            a, b = fields[sid], fields[partner]
            if a["grid_gt"].shape != b["grid_gt"].shape:
                continue                       # different geometry, not comparable
            if torch.equal(a["grid_gt"] > 0.5, b["grid_gt"] > 0.5):
                continue                       # same target region: no contrast
            k = gt_area_k(a["grid_gt"])
            k_b = gt_area_k(b["grid_gt"])
            if k == 0 or k_b == 0:
                continue
            field = (center_prior_field(*a["grid_gt"].shape[-2:],
                                        dtype=a["grid_pred"].dtype)
                     if use_prior else a["grid_pred"])
            pred_bin = topk_mask(field, k)
            rows.append({
                "sample_id": sid, "source_image_id": img, "partner": partner,
                "self_iou": hard_iou(pred_bin, a["grid_gt"]),
                "cross_iou": hard_iou(pred_bin, b["grid_gt"]),
                "area_ratio": k / k_b,
            })
    return rows


def _instruction_paired(fields: dict[str, dict[str, Any]]) -> dict[str, Any]:
    """Same-image paired difference, **calibrated against the centre prior**.

    The raw difference is confounded by GT area (review blocker F-B1).  The
    prediction is binarised to ``k = |GT_A|`` cells, so

        IoU(pred_k, GT_B) <= |GT_A| / |GT_B|

    and whenever the partner's region is larger the cross score is mechanically
    capped, making ``self - cross`` positive for free.  Measured by the reviewer
    on a zero-information centre-prior field: +0.0000 at equal area, +0.1343 at
    36-vs-324, and **+0.7590** on a concentric pair whose two GTs share a centre
    and differ only in area.

    So the same cross-scoring pipeline is run a second time with the field
    replaced by the centre prior, and the reported headline is the **calibrated**
    difference ``delta_field - delta_prior``: whatever the area effect gives the
    real field, it gives the zero-parameter baseline too, and subtracting cancels
    it.  The area-balanced subset (``|GT_A|/|GT_B|`` within
    :data:`PAIRED_AREA_RATIO_RANGE`) is reported alongside as a second, more
    conservative view.
    """
    lo, hi = PAIRED_AREA_RATIO_RANGE
    field_rows = _paired_rows(fields, use_prior=False)
    prior_rows = _paired_rows(fields, use_prior=True)

    def diffs(rows):
        return [r["self_iou"] - r["cross_iou"] for r in rows]

    def block(f_rows, p_rows) -> dict[str, Any]:
        if not f_rows:
            return {"n_pairs": 0, "delta": None, "delta_center_prior": None,
                    "calibrated_delta": None, "p_value": None}
        raw = instruction_paired_delta(f_rows)
        prior = instruction_paired_delta(p_rows)
        cal = paired_delta(diffs(f_rows), diffs(p_rows))
        return {
            "n_pairs": len(f_rows),
            "delta": raw["delta"], "p_value": raw["p_value"],
            "delta_center_prior": prior["delta"],
            # the headline: what the field earns OVER the zero-parameter baseline
            "calibrated_delta": cal["delta"],
            "calibrated_p_value": cal["p_value"],
            "calibrated_ci95": cal["ci95"],
            "n_groups": raw.get("n_groups"),
        }

    keep = [i for i, r in enumerate(field_rows) if lo <= r["area_ratio"] <= hi]
    ratios = sorted(r["area_ratio"] for r in field_rows)
    return {
        "all_pairs": block(field_rows, prior_rows),
        "area_balanced": block([field_rows[i] for i in keep],
                               [prior_rows[i] for i in keep]),
        "area_ratio_range": list(PAIRED_AREA_RATIO_RANGE),
        "area_ratio_observed": (
            {"min": ratios[0], "p50": ratios[len(ratios) // 2], "max": ratios[-1]}
            if ratios else None),
        "n_pairs_area_balanced": len(keep),
        "n_pairs_total": len(field_rows),
        "note": (
            "report `calibrated_delta` (field minus centre prior); the raw "
            "`delta` is area-confounded and a zero-information field can reach "
            "+0.76 on a concentric pair (review F-B1). Reported column, not a "
            "gate (F-B2)."
        ),
    }


def _oracle_mask(builder: BatchBuilder, tgt: dict[str, Any], arm_cfg: ArmConfig):
    lat = builder.oracle.latent(tgt["sample_id"], arm_cfg.readout)
    if lat is None:
        return None, None
    params = {"w0": lat.w0, "w_raw": lat.w_raw, "alpha_raw": lat.alpha_raw,
              **{k: v for k, v in lat.rho.items()}}
    params = {k: v.to(tgt["phi_dir"].device).float() for k, v in params.items()}
    f = predict_fields(
        tgt["phi_dir"].float(), params, arm_cfg.readout, tgt["grid_h"], tgt["grid_w"],
        guide_hi=tgt["guide_hi"].float(), up_cfg=arm_cfg.upsample,
        require_dtype=torch.float32,
    )
    return f["m_hi"].reshape(tgt["mask_hi"].shape), f["m_low"]


def strata_report(rows: Sequence[dict[str, Any]],
                  keys: Sequence[str] = STRATA_KEYS + EXTRA_STRATA_KEYS,
                  ) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for key in keys:
        groups: dict[str, list[dict[str, Any]]] = {}
        for r in rows:
            groups.setdefault(str(r.get(key)), []).append(r)
        out[key] = {k: summarise(v) for k, v in sorted(groups.items())}
    return out


def write_per_sample(rows: Iterable[dict[str, Any]], path: Path) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as fh:
        for r in rows:
            fh.write(json.dumps(r, ensure_ascii=False) + "\n")
    return path


def evaluate_arm(
    model: WhereBModel,
    builder: BatchBuilder,
    dataset: WhereBDataset,
    arm_cfg: ArmConfig,
    *,
    contexts: Sequence[str] = CONTEXT_MODES,
    batch_size: int = 4,
    limit: int | None = None,
    out_dir: Path | None = None,
    resources: dict[str, Any] | None = None,
    progress: Callable[[str], None] | None = None,
    prefetch_workers: int | None = None,
) -> dict[str, Any]:
    """The full protocol 5.6 board for one checkpoint."""
    per_context: dict[str, dict[str, Any]] = {}
    all_rows: list[dict[str, Any]] = []
    # nit N8: the builder is reused across every 500-step eval, so its format
    # counters would otherwise accumulate and this eval's metrics.json would
    # report the run-to-date rates instead of this checkpoint's.
    if hasattr(builder, "format_stats"):
        builder.format_stats = {}
    for mode in contexts:
        if progress:
            progress(mode)
        rows, summary = evaluate_context(
            model, builder, dataset, arm_cfg, mode,
            batch_size=batch_size, limit=limit,
            prefetch_workers=prefetch_workers,
        )
        per_context[mode] = summary
        all_rows.extend(rows)

    res = dict(resources or {})
    res.setdefault("n_trainable_params", model.n_trainable())
    metrics = arm_metrics(per_context, main_context="generated"
                          if "generated" in per_context else contexts[0],
                          resources=res)
    metrics["arm"] = arm_cfg.arm
    metrics["structure"] = arm_cfg.structure
    metrics["readout"] = arm_cfg.readout
    metrics["gate"] = evaluate_gates(metrics)
    # invariance control: joined per sample across the gt and antonym boards
    metrics["antonym_invariance"] = antonym_invariance(all_rows)
    metrics["strata"] = {
        mode: strata_report([r for r in all_rows if r["context"] == mode])
        for mode in per_context
    }
    if builder.shuffle_index is not None:
        metrics["shuffle_coverage"] = builder.shuffle_index.coverage()
    if out_dir is not None:
        out_dir = Path(out_dir)
        write_per_sample(all_rows, out_dir / "per_sample.jsonl")
        (out_dir / "metrics.json").write_text(
            json.dumps(metrics, indent=2, ensure_ascii=False), encoding="utf-8"
        )
        # ruling D-B1 + the reviewer's addendum: the "soft-IoU is both the
        # dominant loss term and the first selection key" consequence ships with
        # every board, filled in with this arm's own numbers, so REPORT.md
        # cannot quietly present it as independent evidence.
        (out_dir / "ATTRIBUTION.md").write_text(
            attribution_section(metrics), encoding="utf-8"
        )
    return metrics
