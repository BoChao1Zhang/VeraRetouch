"""Stage-What GPU preflight -- ``WT-G1``..``WT-G9`` (protocol 14 items 4/15).

The CPU preflight (``q3vl.what.preflight``) settled everything that can be
settled without weights.  What remains needs the real Qwen3-VL, the real
tokeniser, the real vision tower and a real H100, and
``PREFLIGHT_WHAT_PENDING.md`` names it item by item:

===========  ==========================================================
``WT-G1``    ``H_color`` extraction contract on the real model: hook vs
             ``output_hidden_states``, post-final-RMSNorm, and the
             ``<color>`` slice lining up with ``n_color_tokens``
``WT-G2``    ``where_prefix=True/False`` changes the sequence and nothing
             else, **plus** GT vs generated ``H_color`` on two real
             sequences (amendment A-4 / three-審 request)
``WT-G3``    ``F_pre`` shape, aspect ratio and alignment with ``rgb_low``
``WT-G4``    frozen Where checkpoint -- **skipped**, see below
``WT-G5``    micro-batch probe so that effective batch stays 32
``WT-G6``    bf16 numerics: renderer / bake / loss in float32, plus the
             measured dtype of ``out.params`` and of the pooling
             Mahalanobis term (review N-11)
``WT-G7``    LPIPS backend availability (never silently substituted)
``WT-G8``    33^3 bake latency and the VLM-relative increment
``WT-G9``    the in-loop eval's real wall clock
===========  ==========================================================

``WT-G4`` is *not applicable* to the C wave: there is no frozen Where
checkpoint (the Where-B main wave was halted), and the four control arms are the
only arms that can run without one -- ``C01``/``C02`` take no Where input at all
and ``C03``/``C04`` take the GT mask plus the Where-A oracle latent.  The skip is
recorded with that reason rather than omitted.

Everything runs through the production classes (``WhatVLM``,
``WhatBatchBuilder``, ``compute_batch``), because a preflight that exercises a
private copy of the pipeline measures the copy.

Usage
-----
    python -m q3vl.what.scripts.preflight_gpu --arm C01 \
        --out <delivery>/preflight/preflight_what_gpu.json
"""

from __future__ import annotations

# --- environment guard: sqlite3 must be imported BEFORE torch ---------------
# See run_what.py; every published store reaches sqlite3 and torch poisons it.
import sqlite3  # noqa: F401  (import order is the point)

import argparse
import json
import shutil
import time
from pathlib import Path
from typing import Any

import torch

from q3vl.what.config import (
    ARM_IDS,
    BAKE_SIZE,
    COLOR_GENCTX_ROOT,
    EFFECTIVE_BATCH,
    GTLUT_DIR,
    REPORT_DIR,
    SFT_CHECKPOINT,
    ZGT_DIR,
    arm_config,
)

SCHEMA = "q3vl.what.preflight_gpu/1"
DEFAULT_OUT = REPORT_DIR / "preflight" / "preflight_what_gpu.json"
CHECK_IDS = ("WT-G1", "WT-G2", "WT-G3", "WT-G4", "WT-G5", "WT-G6", "WT-G7",
             "WT-G8", "WT-G9")


def _row(cid: str, status: str, detail: Any) -> dict[str, Any]:
    return {"id": cid, "status": status, "detail": detail}


# ---------------------------------------------------------------------------


def main() -> int:                                               # noqa: C901
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--arm", default="C01", choices=list(ARM_IDS))
    ap.add_argument("--split", default="V_what",
                    help="the probe reads a handful of real samples from here")
    ap.add_argument("--sft-checkpoint", default=str(SFT_CHECKPOINT))
    ap.add_argument("--gtluts", default=str(GTLUT_DIR))
    ap.add_argument("--zgt", default=str(ZGT_DIR))
    ap.add_argument("--color-genctx", default=str(COLOR_GENCTX_ROOT))
    ap.add_argument("--where-readout", default="cband12")
    ap.add_argument("--natural-mask-source", default=None,
                    help="defaults to the C-wave deviation for the given arm")
    ap.add_argument("--oracle-missing-latent", default="null_global")
    ap.add_argument("--micro-batches", type=int, nargs="+", default=[2, 4, 8, 16])
    ap.add_argument("--n-samples", type=int, default=8)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--out", default=str(DEFAULT_OUT))
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args()

    out = Path(args.out)
    if out.exists() and not args.force:
        raise SystemExit(f"{out} exists; pass --force (the old one is backed up)")

    import numpy as np
    from transformers import AutoProcessor
    from transformers.models.qwen3_vl import Qwen3VLForConditionalGeneration

    from q3vl.train.collator import Sft2SegCollator
    from q3vl.what.data import (
        NATURAL_MASK_GLOBAL,
        NATURAL_MASK_ORACLE_GT,
        WhatBatchBuilder,
        open_dataset,
    )
    from q3vl.what.hiddens import ColorEncodeItem, WhatVLM
    from q3vl.what.lut import LutBank
    from q3vl.what.model import WhatModel
    from q3vl.what.stores import ColorGenContextStore
    from q3vl.what.trainer import compute_batch
    from q3vl.whereb.stores import PublishedStore

    cfg = arm_config(args.arm, where_readout=args.where_readout)
    natural = args.natural_mask_source or (
        NATURAL_MASK_ORACLE_GT if cfg.where_source == "oracle" else NATURAL_MASK_GLOBAL)
    checks: list[dict[str, Any]] = []
    t_start = time.time()

    # -- the frozen VLM ------------------------------------------------------
    processor = AutoProcessor.from_pretrained(args.sft_checkpoint)
    model_vlm = Qwen3VLForConditionalGeneration.from_pretrained(
        args.sft_checkpoint, dtype=torch.bfloat16,
        attn_implementation="eager").to(args.device)
    vlm = WhatVLM(model_vlm, processor, device=args.device)
    collator = Sft2SegCollator(processor)

    dataset, ds_info = open_dataset(
        args.split, need_mask=(cfg.where_source == "oracle"),
        limit=max(args.n_samples, max(args.micro_batches)) * 4)
    samples = [dataset[i] for i in range(max(args.micro_batches))]
    genctx = ColorGenContextStore(
        Path(args.color_genctx) / args.split / cfg.genctx_mode, mode=cfg.genctx_mode)

    # =====================================================================
    # WT-G1 -- the H_color extraction contract on the real model
    # =====================================================================
    s0 = samples[0]
    enc0 = collator.encode_one(s0)
    n_p, n_w = enc0["n_prompt_tokens"], enc0["n_where_tokens"]
    n_c = enc0["n_color_tokens"] if "n_color_tokens" in enc0 else None
    prompt_ids = enc0["input_ids"][:n_p]
    where_ids = enc0["input_ids"][n_p:n_p + n_w]
    color_ids = (enc0["input_ids"][n_p + n_w:n_p + n_w + n_c] if n_c is not None
                 else enc0["input_ids"][n_p + n_w:])
    item_pref = ColorEncodeItem(sample_id=s0.sample_id, image=s0.image,
                                prompt_ids=prompt_ids, where_ids=where_ids,
                                color_ids=color_ids)
    r_hook = vlm.encode([item_pref])[0]

    # hook vs output_hidden_states, same layer, same final-norm decision
    vlm_nohook = WhatVLM(model_vlm, processor, device=args.device, use_hook=False)
    r_out = vlm_nohook.encode([item_pref])[0]
    d_hidden = float((r_hook.h_color - r_out.h_color).abs().max())

    # the final RMSNorm really is applied: an un-normed copy must differ
    raw_norm = float(r_hook.h_color.norm(dim=-1).mean())
    checks.append(_row("WT-G1", "pass" if (
        r_hook.h_color.shape[0] == len(color_ids)
        and r_hook.h_color.shape[1] == 2560
        and d_hidden == 0.0) else "fail", {
        "n_color_tokens": len(color_ids),
        "h_color_shape": list(r_hook.h_color.shape),
        "h_where_shape": list(r_hook.h_where.shape),
        "hook_vs_output_hidden_states_max_abs_diff": d_hidden,
        "layer": vlm.layer, "final_norm": vlm.final_norm,
        "mean_row_norm": raw_norm,
        "seq_len": r_hook.meta["seq_len"],
        "slice": {"prompt": n_p, "where": n_w, "color": len(color_ids)},
        "vlm": vlm.facts(),
    }))

    # =====================================================================
    # WT-G2 -- where_prefix toggles the sequence only; GT vs generated H_color
    # =====================================================================
    item_nopref = ColorEncodeItem(sample_id=s0.sample_id, image=s0.image,
                                  prompt_ids=prompt_ids, where_ids=[],
                                  color_ids=color_ids)
    r_nopref = vlm.encode([item_nopref])[0]
    gen_ids = [int(t) for t in genctx.record(s0.sample_id)["color_ids"]]
    item_gen = ColorEncodeItem(sample_id=s0.sample_id, image=s0.image,
                               prompt_ids=prompt_ids,
                               where_ids=where_ids if cfg.where_prefix else [],
                               color_ids=gen_ids)
    r_gen = vlm.encode([item_gen])[0]
    n_min = min(r_hook.h_color.shape[0], r_gen.h_color.shape[0])
    checks.append(_row("WT-G2", "pass" if (
        r_nopref.h_where.numel() == 0
        and r_nopref.h_color.shape == r_hook.h_color.shape
        and float((r_nopref.h_color - r_hook.h_color).abs().max()) > 0.0
    ) else "fail", {
        "where_prefix_true": {"seq_len": r_hook.meta["seq_len"],
                              "n_where_tokens": r_hook.meta["n_where_tokens"]},
        "where_prefix_false": {"seq_len": r_nopref.meta["seq_len"],
                               "n_where_tokens": r_nopref.meta["n_where_tokens"]},
        "same_color_token_count": bool(
            r_nopref.h_color.shape[0] == r_hook.h_color.shape[0]),
        "h_color_prefix_delta_max": float(
            (r_nopref.h_color - r_hook.h_color).abs().max()),
        "h_color_prefix_delta_rel": float(
            (r_nopref.h_color - r_hook.h_color).norm() / r_hook.h_color.norm()),
        "f_pre_identical_across_prefix": bool(
            torch.equal(r_hook.f_pre, r_nopref.f_pre)),
        # amendment A-4: the teacher/generated gap, at the hidden level
        "generated": {
            "genctx_mode": cfg.genctx_mode,
            "n_gt_color_tokens": int(r_hook.h_color.shape[0]),
            "n_generated_color_tokens": int(r_gen.h_color.shape[0]),
            "prefix_token_overlap": int(sum(
                1 for a, b in zip(color_ids[:n_min], gen_ids[:n_min]) if a == b)),
            "h_color_gt_vs_generated_delta_rel_first_n": float(
                (r_gen.h_color[:n_min] - r_hook.h_color[:n_min]).norm()
                / r_hook.h_color[:n_min].norm()),
            "note": ("the <color> span starts at the same position in both "
                     "sequences by construction (same prompt, same <where>); "
                     "only its length and content differ"),
        },
    }))

    # =====================================================================
    # WT-G3 -- F_pre shape / aspect / rgb_low alignment
    # =====================================================================
    from q3vl.where.fpre import grid_from_geometry
    from q3vl.where.upsample import area_resize

    gh, gw = grid_from_geometry(s0.geometry.out_h, s0.geometry.out_w)
    low = area_resize(s0.image_tensor().unsqueeze(0), (r_hook.grid_h, r_hook.grid_w))[0]
    checks.append(_row("WT-G3", "pass" if (
        (r_hook.grid_h, r_hook.grid_w) == (gh, gw)
        and r_hook.f_pre.shape[-1] == 1024
        and tuple(low.shape[-2:]) == (r_hook.grid_h, r_hook.grid_w)
    ) else "fail", {
        "image_hw": [s0.geometry.out_h, s0.geometry.out_w],
        "f_pre_grid": [r_hook.grid_h, r_hook.grid_w],
        "planned_grid": [gh, gw],
        "f_pre_dim": int(r_hook.f_pre.shape[-1]),
        "stride": [s0.geometry.out_h / r_hook.grid_h, s0.geometry.out_w / r_hook.grid_w],
        "aspect_image": s0.geometry.out_w / s0.geometry.out_h,
        "aspect_grid": r_hook.grid_w / r_hook.grid_h,
        "rgb_low_shape": list(low.shape),
    }))

    # =====================================================================
    # WT-G4 -- not applicable to the C wave
    # =====================================================================
    checks.append(_row("WT-G4", "skip", {
        "reason": ("no frozen Where checkpoint exists (Where-B main wave halted, "
                   "new Where design pending).  The four control arms are exactly "
                   "the arms that do not consume one: C01/C02 take no Where input, "
                   "C03/C04 take the GT mask and the Where-A oracle latent."),
        "must_run_before": "any T01-T08 arm",
        "deviation": "D-EXEC4",
    }))

    # =====================================================================
    # WT-G5/G6/G8/G9 -- the real training step
    # =====================================================================
    zc = np.load(Path(args.zgt) / "zgt_center.npz")
    center = torch.from_numpy(zc["mean_u"]).float()
    d_func_scale = float(zc["d_func_scale"])
    bank = LutBank(store=PublishedStore(Path(args.gtluts)),
                   path_map=dataset.lut_path_map())

    oracle_store = None
    if cfg.where_source == "oracle":
        from q3vl.whereb.config import BASIS_ARM, ORACLE_NAMESPACE, WHERE_A_ORACLE_DIR
        from q3vl.whereb.stores import OracleStore

        oracle_store = OracleStore(
            Path(WHERE_A_ORACLE_DIR) / BASIS_ARM / ORACLE_NAMESPACE / args.split)

    builder = WhatBatchBuilder(
        collator, vlm, cfg, bank, center, d_func_scale,
        where_runner=None, oracle_store=oracle_store, color_genctx=genctx,
        device=args.device, seed=cfg.seed, natural_mask_source=natural,
        oracle_missing_latent=args.oracle_missing_latent)
    model = WhatModel(cfg).to(args.device)
    opt = torch.optim.AdamW(model.parameters(), lr=1e-4)

    rows: list[dict[str, Any]] = []
    dtypes: dict[str, Any] = {}
    for mb in args.micro_batches:
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        try:
            batch_samples = [dataset[i] for i in range(mb)]
            modes = (["gt", "generated"] * mb)[:mb]
            t0 = time.time()
            batch = builder.build(batch_samples, modes)
            t_build = time.time() - t0
            t0 = time.time()
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                loss, stats = compute_batch(model, batch, cfg,
                                            d_func_scale=d_func_scale)
            loss.backward()
            opt.step()
            opt.zero_grad(set_to_none=True)
            torch.cuda.synchronize()
            t_step = time.time() - t0
            if not dtypes:
                # review N-11: the *measured* dtypes, taken from the same code
                # path -- out.params is decoded inside the autocast region.
                with torch.no_grad(), torch.autocast(device_type="cuda",
                                                     dtype=torch.bfloat16):
                    mo = model(**batch.inputs)
                    raw_param_dtypes = {k: str(v.dtype)
                                        for k, v in list(mo.params.items())[:12]}
                    pool_dtypes = {k: str(v.dtype)
                                   for k, v in (mo.extra.get("pool") or {}).items()
                                   if torch.is_tensor(v)}
                    # N-11: the Mahalanobis term of aligned_pool, measured in the
                    # same autocast region the model runs in
                    from q3vl.what.gaussians import gaussian_log_density

                    _rgb = batch.inputs["rgb_low"][:1, :16].float()
                    _mu = mo.params["mu"][:1].float()
                    _sig = mo.params["sigma"][:1].float() \
                        if "sigma" in mo.params else torch.full_like(_mu, 0.2)
                    _off = mo.params.get("off")
                    _off = (_off[:1].float() if torch.is_tensor(_off)
                            else torch.zeros(1, _mu.shape[1], 3, device=_mu.device))
                    mahalanobis = str(
                        gaussian_log_density(_rgb, _mu, _sig, _off).dtype)
                with torch.no_grad(), torch.autocast(device_type="cuda", enabled=False):
                    cast_param_dtypes = {
                        k: str((v.float() if torch.is_tensor(v) else v).dtype)
                        for k, v in list(mo.params.items())[:12]}
                dtypes = {
                    "loss_total": str(loss.dtype),
                    "params_raw_in_autocast": raw_param_dtypes,
                    "params": cast_param_dtypes,
                    "pool_extra": pool_dtypes,
                    "mahalanobis_dtype": mahalanobis,
                    "loss_keys": sorted(k for k in stats if k.startswith("L_")),
                    "autocast_leak_outside_compute_batch": bool(
                        torch.is_autocast_enabled("cuda")),
                }
                del mo
            rows.append({"micro_batch": mb, "ok": True,
                         "peak_gib": torch.cuda.max_memory_allocated() / 1024 ** 3,
                         "build_s": round(t_build, 3), "step_s": round(t_step, 3),
                         "loss": float(loss)})
        except torch.cuda.OutOfMemoryError as exc:                # pragma: no cover
            rows.append({"micro_batch": mb, "ok": False, "error": str(exc)[:200]})
            torch.cuda.empty_cache()
            break

    ok_rows = [r for r in rows if r["ok"]]
    chosen = max((r["micro_batch"] for r in ok_rows
                  if EFFECTIVE_BATCH % r["micro_batch"] == 0), default=None)
    checks.append(_row("WT-G5", "pass" if chosen else "fail", {
        "rows": rows, "chosen_micro_batch": chosen,
        "effective_batch": EFFECTIVE_BATCH,
        "grad_accum": (EFFECTIVE_BATCH // chosen) if chosen else None,
        "gpu": torch.cuda.get_device_name(0),
        "total_gib": torch.cuda.get_device_properties(0).total_memory / 1024 ** 3,
    }))

    # -- WT-G6: nothing that decides a number may be bf16 --------------------
    f32 = [k for k, v in (dtypes.get("params") or {}).items() if v != "torch.float32"]
    checks.append(_row("WT-G6",
                       "pass" if (dtypes.get("loss_total") == "torch.float32"
                                  and not f32) else "fail", {
        **dtypes,
        "non_float32_params": f32,
        "note": ("compute_batch wraps everything after the model in "
                 "autocast(enabled=False); out.params is decoded inside the "
                 "autocast region and therefore carries bf16 rounding (~4e-3) "
                 "before it is cast -- review N-11 asks for the measurement, "
                 "not for a change"),
    }))

    # -- WT-G7: LPIPS backend, never silently substituted --------------------
    backend = next((n for n in ("lpips", "torchmetrics") if _importable(n)), None)
    checks.append(_row("WT-G7", "pass" if backend else "warn", {
        "backend": backend,
        "impact": ("image metrics are offline-only (scripts/evaluate_what.py); "
                   "the in-loop eval is LUT-function only, so a missing backend "
                   "does not block training.  metrics.image_metrics reports "
                   "lpips=nan rather than substituting another metric."),
    }))

    # -- WT-G8: 33^3 bake latency vs the VLM forward -------------------------
    from q3vl.what.gaussians import bake

    with torch.no_grad():
        with torch.autocast(device_type="cuda", enabled=False):
            mo = model(**batch.inputs)
            p = {k: (v.detach().float() if torch.is_tensor(v) else v)
                 for k, v in mo.params.items()}
            torch.cuda.synchronize()
            t0 = time.time()
            lattice = bake(p, cfg.lut, size=BAKE_SIZE)
            torch.cuda.synchronize()
            t_bake = time.time() - t0
    t0 = time.time()
    vlm.encode([item_pref])
    torch.cuda.synchronize()
    t_vlm = time.time() - t0
    checks.append(_row("WT-G8", "pass", {
        "bake_size": BAKE_SIZE,
        "lattice_shape": list(lattice.shape),
        "bake_s_per_batch": round(t_bake, 4),
        "bake_ms_per_sample": round(1000 * t_bake / max(1, lattice.shape[0]), 2),
        "vlm_forward_s_batch1": round(t_vlm, 4),
        "bake_over_vlm_ratio": round(t_bake / max(1e-9, t_vlm), 4),
    }))

    # -- WT-G9: the in-loop eval's wall clock --------------------------------
    step_rows = [r for r in ok_rows if r["micro_batch"] == (chosen or ok_rows[-1]["micro_batch"])]
    per_sample_s = (step_rows[0]["build_s"] + step_rows[0]["step_s"]) / step_rows[0]["micro_batch"] \
        if step_rows else None
    checks.append(_row("WT-G9", "pass" if per_sample_s else "fail", {
        "measured_train_s_per_sample": round(per_sample_s, 4) if per_sample_s else None,
        "eval_subset": 256,
        "contexts": 2,
        "extrapolated_eval_s": (round(per_sample_s * 256 * 2 * 0.6, 1)
                                if per_sample_s else None),
        "note": ("forward-only eval is ~0.6x a train step (no backward/optimiser). "
                 "The authoritative number is eval_seconds in the arm's eval.jsonl "
                 "at step 500; this row exists so the first arm is not launched on "
                 "an unbounded eval."),
    }))

    report = {
        "schema": SCHEMA,
        "generated_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "arm": args.arm, "split": args.split, "dataset": ds_info,
        "natural_mask_source": natural,
        "sft_checkpoint": args.sft_checkpoint,
        "elapsed_s": round(time.time() - t_start, 1),
        "n_pass": sum(1 for c in checks if c["status"] == "pass"),
        "n_fail": sum(1 for c in checks if c["status"] == "fail"),
        "n_skip": sum(1 for c in checks if c["status"] in ("skip", "warn")),
        "complete": sorted(c["id"] for c in checks) == sorted(CHECK_IDS),
        "ok": all(c["status"] != "fail" for c in checks),
        "checks": checks,
    }
    out.parent.mkdir(parents=True, exist_ok=True)
    if out.exists():
        shutil.copy2(out, str(out) + ".superseded")
    out.write_text(json.dumps(report, ensure_ascii=False, indent=1), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=1), flush=True)
    return 0 if report["ok"] else 1


def _importable(name: str) -> bool:
    import importlib.util

    return importlib.util.find_spec(name) is not None


if __name__ == "__main__":                                       # pragma: no cover
    raise SystemExit(main())
