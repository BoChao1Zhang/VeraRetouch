#!/usr/bin/env python
"""Offline full evaluation -- the basis of the protocol 12.4 selection.  NOT RUN YET.

Review NF-2's third point: amendment A-4's two boards had no producer.  The
in-loop pass (:mod:`q3vl.what.evalloop`) is a **bounded proxy** -- a fixed
256-sample subset, LUT-function metrics only -- whose job is to keep the right
checkpoint files alive and to give a 500-step readout.  *This* is the
measurement selection reads:

* the complete ``V_what``, not a subset;
* both contexts (amendment A-4 item 2), never averaged;
* protocol 12.1 LUT metrics **and** 12.2 image metrics -- ``I_out = I_in +
  m (T(I_in) - I_in)`` against ``I_tar``, partitioned into mask interior / 3px
  boundary band / exterior and stratified;
* protocol 12.3's anti-collapse and causal controls;
* ``main_board`` (generated context, gate as a filter), ``ceiling_board`` for
  ``C03``/``C04``, and ``context_report`` for the teacher/generated gap.

``I_tar`` enters here and **only** here.  Protocol 9.5 keeps it out of
``L_what``; ``WhatDataset.load_target_image`` exists for this script and the
training path has no call site for it.

Selection discipline is enforced by the library, not by this script's caller:
``main_board`` raises ``PermissionError`` on any split outside ``V_what``, so
opening ``T_final`` has to be a deliberate ``--allow-split`` argument, recorded
in the output.

Deviation **D-EXEC5** (task card EXEC-5, 2026-08-10) -- evaluating the C wave
before a Where checkpoint exists:

* ``--where-checkpoint`` is optional here for the same reason it is optional in
  ``run_what.py`` (deviation D-EXEC4): the four control arms have no Where input
  to lose.  A ``predicted`` arm still refuses.  Without one, ``--where-readout``
  and ``--natural-mask-source`` must be **explicit**, and they are cross-checked
  against the arm's own ``run_setup.json`` -- evaluating an arm on a different
  natural-query distribution than it trained on is exactly the kind of silent
  mismatch this campaign keeps getting bitten by.
* ``--composite-mask`` is **required, with no default**.  Protocol 12.2's
  composite uses the frozen Where mask for every arm including C01/C02, so that
  the controls are not confounded with a mask ablation.  Under D-EXEC4 there is
  no frozen mask, so the only mask that exists for all four arms is the GT one,
  and taking it is a *declared* choice, recorded in the report and in every
  row's ``composite_mask``.  ``model`` reproduces the pre-EXEC-5 behaviour (GT
  for C03/C04, all-ones for C01/C02) and is kept only to make that confound
  reproducible, never as a default.

Usage (submit through the GPU queue, which supplies D-20; never nohup by hand):
    q submit eval_C01 0 /home/bc/data/runs/what/evaluate/C01/evaluate.log \
        --gate /home/bc/data/runs/what/C01/what_final.pt \
        --ready 'progress' -- bash waves/what_c_eval.sh C01
"""

from __future__ import annotations

# --- environment guard: sqlite3 must be imported BEFORE torch ---------------
import sqlite3  # noqa: F401  (import order is the point)

import argparse
import json
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch

from q3vl.what.config import (
    COLOR_GENCTX_ROOT,
    CONTEXT_GENERATED,
    CONTEXT_GT,
    GTLUT_DIR,
    RUN_ROOT,
    SFT_CHECKPOINT,
    ZGT_DIR,
    arm_config,
)
from q3vl.what.data import (
    NATURAL_MASK_FROZEN,
    NATURAL_MASK_ORACLE_GT,
    NATURAL_MASK_SOURCES,
    ORACLE_MISSING_POLICIES,
    ORACLE_MISSING_REJECT,
    WhatBatchBuilder,
    WhereRunner,
    open_dataset,
)
from q3vl.what.evaluate import (
    arm_metrics,
    ceiling_board,
    context_report,
    main_board,
    sample_row,
    write_per_sample,
)
from q3vl.what.hiddens import WhatVLM
from q3vl.what.metrics import style_diagnostics
from q3vl.what.model import WhatModel
from q3vl.what.stores import ColorGenContextStore


#: which mask composes ``I_out = I_in + m (T(I_in) - I_in)`` for protocol 12.2.
#: ``gt`` is the same mask for every arm (what the protocol asks for, once the
#: frozen Where mask does not exist); ``model`` is whatever the arm's *input*
#: carried, which differs across arms and therefore confounds the control.
COMPOSITE_GT, COMPOSITE_MODEL, COMPOSITE_ONES = "gt", "model", "ones"
COMPOSITE_MASKS = (COMPOSITE_GT, COMPOSITE_MODEL, COMPOSITE_ONES)


def _load_checkpoint(path: Path):
    ck = torch.load(path, map_location="cpu", weights_only=False)
    cfg = arm_config(ck["arm"])
    model = WhatModel(cfg)
    model.load_state_dict(ck["model"])
    return cfg, model, int(ck.get("step", 0))


def lpips_closure(net):
    """``lpips.LPIPS`` expects **[-1, 1]**; every image in this package is [0, 1].

    Wired as a named function rather than a lambda inside ``main`` so that the
    rescale is testable: ``preflight/wt_g7_lpips_closure.json`` recorded the
    convention (``input_convention: "[-1,1] ... the caller must rescale"``) and a
    missing ``x * 2 - 1`` would not crash -- it would quietly report the LPIPS of
    a half-contrast pair, which is the silent-failure shape the campaign's s-cache
    contract warns about.
    """
    def _lpips(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
        return net(a * 2 - 1, b * 2 - 1).mean()

    return _lpips


def _composite_mask(sample, model_m_hi, *, composite: str, where_source: str,
                    device: str):
    """The 12.2 composite mask for one sample, plus its audit label."""
    if composite == COMPOSITE_GT:
        m = getattr(sample, "mask_hi", None)
        if m is None:                      # global edits have no ROI at all
            return None, "ones_global"
        return m.to(device), "gt"
    if composite == COMPOSITE_MODEL:
        if model_m_hi is None:
            return None, "ones"
        return model_m_hi.to(device), ("gt" if where_source == "oracle"
                                       else "predicted")
    return None, "ones"


@torch.no_grad()
def evaluate_checkpoint(model, builder, dataset, cfg, *, context: str,
                        micro_batch: int, device: str, lpips_fn=None,
                        full_grid: bool = False, limit: int | None = None,
                        composite: str = COMPOSITE_GT, step: int = 0,
                        log_every: int = 32
                        ) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """One (checkpoint, context) pass over the whole split, with image metrics."""
    model.eval().to(device)
    model.collect_pool_stats = False
    n = len(dataset) if limit is None else min(limit, len(dataset))
    rows: list[dict[str, Any]] = []
    z_styles, u_gts = [], []
    t0 = time.time()
    # D-20 (3): a job that prints nothing for an hour cannot be told apart from a
    # job that died on line one.  These lines are what the queue's --ready
    # pattern matches, so they must appear *after* real work, not before it.
    print(f"evaluate_what: begin arm={cfg.arm} step={step} context={context} "
          f"n={n} micro_batch={micro_batch} composite_mask={composite} "
          f"lpips={lpips_fn is not None} full_grid={full_grid}", flush=True)
    for start in range(0, n, micro_batch):
        idx = list(range(start, min(start + micro_batch, n)))
        samples = [dataset[i] for i in idx]
        batch = builder.build(samples, [context] * len(samples))
        out = model(**batch.inputs)
        params = {k: (v.float() if torch.is_tensor(v) else v)
                  for k, v in out.params.items()}
        z_styles.append(out.z_style.detach().float().cpu())
        u_gts.append(torch.stack([t["u_gt"] for t in batch.targets]).float().cpu())
        m_hi = batch.inputs["where"].m_hi
        for b, (s, tgt) in enumerate(zip(samples, batch.targets)):
            one = {k: (v[b:b + 1] if torch.is_tensor(v) and v.shape[0] == len(idx)
                       else v) for k, v in params.items()}
            i_in = s.image_tensor().to(device)
            i_tar = dataset.load_target_image(s).to(device)
            mask, mask_source = _composite_mask(
                s, None if m_hi is None else m_hi[b], composite=composite,
                where_source=cfg.where_source, device=device)
            row = sample_row(one, tgt, lut_cfg=cfg.lut,
                             table=builder.bank.get(s.lut_id),
                             gt_interp=builder.gt_interp,
                             i_in=i_in, i_tar=i_tar, mask=mask,
                             mask_source=mask_source, full_grid=full_grid,
                             lpips_fn=lpips_fn)
            row["context"] = context
            rows.append(row)
        done = len(rows)
        if start == 0 or done % log_every < micro_batch or done == n:
            print(f"evaluate_what: {cfg.arm} step={step} {context} progress "
                  f"{done}/{n} elapsed={time.time() - t0:.1f}s", flush=True)
    diagnostics = style_diagnostics(torch.cat(z_styles), u_gt=torch.cat(u_gts))
    return rows, diagnostics


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--checkpoints", nargs="+", required=True)
    ap.add_argument("--split", default="V_what")
    ap.add_argument("--allow-split", nargs="+", default=["V_what"],
                    help="splits selection may run on; opening T_final must be "
                         "deliberate (protocol 12.4)")
    ap.add_argument("--where-checkpoint", default=None)
    # --- deviation D-EXEC5: evaluating the C wave before a Where checkpoint ---
    ap.add_argument("--where-readout", default=None,
                    help="taken from the Where checkpoint when there is one; "
                         "required to be explicit when there is not (it sizes "
                         "the rho token and selects the oracle fit)")
    ap.add_argument("--natural-mask-source", default=None,
                    choices=list(NATURAL_MASK_SOURCES),
                    help="must be the one the arm TRAINED with (amendment A-3 / "
                         "deviation D-EXEC4); cross-checked against the arm's "
                         "run_setup.json")
    ap.add_argument("--oracle-missing-latent", default=ORACLE_MISSING_REJECT,
                    choices=list(ORACLE_MISSING_POLICIES))
    ap.add_argument("--composite-mask", required=True, choices=list(COMPOSITE_MASKS),
                    help="protocol 12.2's composite mask.  No default: with no "
                         "frozen Where mask the choice changes what the image "
                         "metrics mean, and 'model' silently gives C01/C02 a "
                         "different renderer than C03/C04")
    ap.add_argument("--log-every", type=int, default=32,
                    help="progress line cadence (D-20: substantive output)")
    ap.add_argument("--sft-checkpoint", default=str(SFT_CHECKPOINT))
    ap.add_argument("--gtluts", default=str(GTLUT_DIR))
    ap.add_argument("--zgt", default=str(ZGT_DIR))
    ap.add_argument("--color-genctx", default=str(COLOR_GENCTX_ROOT))
    ap.add_argument("--out-dir", default=str(RUN_ROOT / "evaluate"))
    ap.add_argument("--micro-batch", type=int, default=4)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--full-grid", action="store_true",
                    help="also evaluate the complete 33^3 lattice (protocol 12.1)")
    ap.add_argument("--lpips", action="store_true")
    args = ap.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    t0 = time.time()

    # -- D-EXEC5: everything that can refuse must refuse before the VLM loads --
    if not args.where_checkpoint and not args.where_readout:
        raise SystemExit(
            "no --where-checkpoint, so --where-readout is required (it sizes the "
            "rho token and selects which Where-A oracle fit C03/C04 read); do not "
            "let it default silently -- deviation D-EXEC4/D-EXEC5.")
    natural = args.natural_mask_source or (
        NATURAL_MASK_FROZEN if args.where_checkpoint else None)
    if natural is None:
        raise SystemExit(
            "no --where-checkpoint, so amendment A-3's "
            f"natural_mask_source={NATURAL_MASK_FROZEN!r} cannot be honoured.  "
            "Pass the value the arm TRAINED with (its run_setup.json records it "
            "under deviation.natural_mask_source) -- evaluating an arm on a "
            "different natural-query distribution than it trained on changes the "
            "'natural' half of every LUT metric.")
    for ck_path in args.checkpoints:
        setup_path = Path(ck_path).parent / "run_setup.json"
        if not setup_path.exists():
            print(f"evaluate_what: note -- no {setup_path}, cannot cross-check "
                  "natural_mask_source against the run that produced this "
                  "checkpoint", flush=True)
            continue
        setup = json.loads(setup_path.read_text(encoding="utf-8"))
        trained = ((setup.get("deviation") or {}).get("natural_mask_source")
                   or NATURAL_MASK_FROZEN)
        if trained != natural:
            raise SystemExit(
                f"{ck_path}: trained with natural_mask_source={trained!r} but "
                f"--natural-mask-source={natural!r}.  The natural query half is "
                "drawn from that mask, so the two settings do not measure the "
                "same thing (D-EXEC4 / amendment A-3).")

    from transformers import AutoProcessor
    from transformers.models.qwen3_vl import Qwen3VLForConditionalGeneration

    from q3vl.train.collator import Sft2SegCollator
    from q3vl.what.scripts.run_what import build_bank, load_center

    processor = AutoProcessor.from_pretrained(args.sft_checkpoint)
    vlm_model = Qwen3VLForConditionalGeneration.from_pretrained(
        args.sft_checkpoint, dtype=torch.bfloat16,
        attn_implementation="eager").to(args.device)
    vlm = WhatVLM(vlm_model, processor, device=args.device)
    collator = Sft2SegCollator(processor)
    center, d_func_scale = load_center(Path(args.zgt))
    where_runner = (WhereRunner.from_checkpoint(Path(args.where_checkpoint),
                                                device=args.device)
                    if args.where_checkpoint else None)
    readout = where_runner.readout if where_runner else args.where_readout

    lpips_fn = None
    if args.lpips:
        import lpips as lpips_lib  # optional dependency (WT-G7)

        # WT-G7 closure record: lpips 0.1.4, net='alex', weights bundled in the
        # wheel.  The [0,1] -> [-1,1] rescale lives in `lpips_closure`.
        lpips_fn = lpips_closure(lpips_lib.LPIPS(net="alex").to(args.device))

    candidates: list[dict[str, Any]] = []
    per_checkpoint: dict[str, Any] = {}
    for ck_path in args.checkpoints:
        cfg, model, step = _load_checkpoint(Path(ck_path))
        cfg = arm_config(cfg.arm, where_readout=readout)
        if where_runner is None and cfg.where_source == "predicted":
            raise SystemExit(
                f"{cfg.arm} needs --where-checkpoint: a WC-conditioned arm has "
                "no Where input without it, so there is nothing to evaluate.")
        # the GT mask is loaded when the model needs it (oracle arms), when the
        # supervision mask needs it, or when it is the 12.2 composite mask
        need_mask = (cfg.where_source == "oracle"
                     or natural == NATURAL_MASK_ORACLE_GT
                     or args.composite_mask == COMPOSITE_GT)
        dataset, ds_info = open_dataset(
            args.split, need_mask=need_mask, limit=args.limit)
        genctx = ColorGenContextStore(
            Path(args.color_genctx) / args.split / cfg.genctx_mode,
            mode=cfg.genctx_mode)
        genctx.assert_covers([dataset.refs[i].sample_id
                              for i in range(len(dataset))])
        oracle_store = None
        if cfg.where_source == "oracle":
            from q3vl.whereb.config import (
                BASIS_ARM,
                ORACLE_NAMESPACE,
                WHERE_A_ORACLE_DIR,
            )
            from q3vl.whereb.stores import OracleStore

            # published layout is <oracle>/<basis arm>/<namespace>/<split>
            # (q3vl.whereb.config; run_where_b.py reads it that way).  The old
            # <oracle>/<split> resolved to nothing -- fixed 2026-08-10.
            oracle_store = OracleStore(
                Path(WHERE_A_ORACLE_DIR) / BASIS_ARM / ORACLE_NAMESPACE / args.split)
        builder = WhatBatchBuilder(
            collator, vlm, cfg, build_bank(Path(args.gtluts), dataset),
            center, d_func_scale, where_runner=where_runner,
            oracle_store=oracle_store, color_genctx=genctx,
            device=args.device, seed=cfg.seed,
            natural_mask_source=natural,
            oracle_missing_latent=args.oracle_missing_latent)

        ck_out: dict[str, Any] = {"path": ck_path, "arm": cfg.arm, "step": step}
        for context in (CONTEXT_GT, CONTEXT_GENERATED):
            rows, diagnostics = evaluate_checkpoint(
                model, builder, dataset, cfg, context=context,
                micro_batch=args.micro_batch, device=args.device,
                lpips_fn=lpips_fn, full_grid=args.full_grid, limit=args.limit,
                composite=args.composite_mask, step=step,
                log_every=args.log_every)
            tag = f"{cfg.arm}_step{step}_{context}"
            write_per_sample(rows, out_dir / f"per_sample_{tag}.jsonl")
            m = arm_metrics(rows, arm=cfg.arm, step=step,
                            n_trainable=model.n_trainable(), context=context)
            m["style_diagnostics"] = diagnostics       # protocol 12.3
            m["dataset"] = ds_info
            candidates.append(m)
            ck_out[context] = {k: m.get(k) for k in
                               ("local_image_de00_median", "lut_de00_p90",
                                "boundary_de00_median", "gate_pass")}
        per_checkpoint[f"{cfg.arm}_step{step}"] = ck_out
        if builder.oracle_latent_stats["null_global"]:
            print(f"evaluate_what: oracle latents -- {builder.oracle_latent_stats}",
                  flush=True)

    board = main_board(candidates, split=args.split, allow=tuple(args.allow_split))
    deviation = None
    if where_runner is None:
        deviation = {
            "id": "D-EXEC5",
            "ruling": "task card EXEC-5 (2026-08-10), inherits D-EXEC4",
            "what": ("no frozen Where checkpoint exists, so (a) the natural "
                     "query half is weighted by the arm's own declared source "
                     "rather than by amendment A-3's frozen m_pred, and (b) "
                     "protocol 12.2's composite cannot use the frozen Where "
                     "mask"),
            "natural_mask_source": natural,
            "composite_mask": args.composite_mask,
            "consequence": ("image metrics are read on the declared composite "
                            "mask; re-run against the frozen mask once a Where "
                            "checkpoint is selected"),
        }
    report = {
        "split": args.split,
        "n_checkpoints": len(args.checkpoints),
        "kind": "offline_full_v_what",
        "main_board": board,
        "ceiling_board": ceiling_board(candidates),
        "context_report": context_report(candidates),
        "per_checkpoint": per_checkpoint,
        "lpips": bool(args.lpips),
        "full_grid": bool(args.full_grid),
        "composite_mask": args.composite_mask,
        "natural_mask_source": natural,
        "oracle_missing_latent": args.oracle_missing_latent,
        "where": {"path": args.where_checkpoint, "readout": readout},
        "deviation": deviation,
        "elapsed_s": round(time.time() - t0, 1),
    }
    (out_dir / f"evaluate_{args.split}.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=1), encoding="utf-8")
    (out_dir / f"candidates_{args.split}.jsonl").write_text(
        "\n".join(json.dumps(c, ensure_ascii=False, sort_keys=True)
                  for c in candidates) + "\n", encoding="utf-8")
    print(json.dumps({k: v for k, v in report.items() if k != "per_checkpoint"},
                     ensure_ascii=False, indent=1), flush=True)
    return 0


if __name__ == "__main__":                                       # pragma: no cover
    raise SystemExit(main())
