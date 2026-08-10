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

Usage (D-20: rm -f the log first, then verify with ``ps -p <PID>``, never pgrep):
    rm -f /home/bc/data/runs/what/evaluate_V_what.log
    nohup python -m q3vl.what.scripts.evaluate_what \
        --checkpoints /home/bc/data/runs/what/T01/what_*.pt \
        --where-checkpoint <frozen where> \
        > /home/bc/data/runs/what/evaluate_V_what.log 2>&1 &
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
from q3vl.what.data import WhatBatchBuilder, WhereRunner, open_dataset
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


def _load_checkpoint(path: Path):
    ck = torch.load(path, map_location="cpu", weights_only=False)
    cfg = arm_config(ck["arm"])
    model = WhatModel(cfg)
    model.load_state_dict(ck["model"])
    return cfg, model, int(ck.get("step", 0))


@torch.no_grad()
def evaluate_checkpoint(model, builder, dataset, cfg, *, context: str,
                        micro_batch: int, device: str, lpips_fn=None,
                        full_grid: bool = False, limit: int | None = None
                        ) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """One (checkpoint, context) pass over the whole split, with image metrics."""
    model.eval().to(device)
    model.collect_pool_stats = False
    n = len(dataset) if limit is None else min(limit, len(dataset))
    rows: list[dict[str, Any]] = []
    z_styles, u_gts = [], []
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
            mask = None if m_hi is None else m_hi[b].to(device)
            mask_source = ("gt" if cfg.where_source == "oracle"
                           else ("predicted" if mask is not None else "ones"))
            row = sample_row(one, tgt, lut_cfg=cfg.lut,
                             table=builder.bank.get(s.lut_id),
                             gt_interp=builder.gt_interp,
                             i_in=i_in, i_tar=i_tar, mask=mask,
                             mask_source=mask_source, full_grid=full_grid,
                             lpips_fn=lpips_fn)
            row["context"] = context
            rows.append(row)
    diagnostics = style_diagnostics(torch.cat(z_styles), u_gt=torch.cat(u_gts))
    return rows, diagnostics


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--checkpoints", nargs="+", required=True)
    ap.add_argument("--split", default="V_what")
    ap.add_argument("--allow-split", nargs="+", default=["V_what"],
                    help="splits selection may run on; opening T_final must be "
                         "deliberate (protocol 12.4)")
    ap.add_argument("--where-checkpoint", required=True)
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
    where_runner = WhereRunner.from_checkpoint(Path(args.where_checkpoint),
                                               device=args.device)

    lpips_fn = None
    if args.lpips:
        import lpips as lpips_lib  # optional dependency (WT-G7)

        net = lpips_lib.LPIPS(net="alex").to(args.device)
        lpips_fn = lambda a, b: net(a * 2 - 1, b * 2 - 1).mean()   # noqa: E731

    candidates: list[dict[str, Any]] = []
    per_checkpoint: dict[str, Any] = {}
    for ck_path in args.checkpoints:
        cfg, model, step = _load_checkpoint(Path(ck_path))
        cfg = arm_config(cfg.arm, where_readout=where_runner.readout)
        dataset, ds_info = open_dataset(
            args.split, need_mask=(cfg.where_source == "oracle"), limit=args.limit)
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
            device=args.device, seed=cfg.seed)

        ck_out: dict[str, Any] = {"path": ck_path, "arm": cfg.arm, "step": step}
        for context in (CONTEXT_GT, CONTEXT_GENERATED):
            rows, diagnostics = evaluate_checkpoint(
                model, builder, dataset, cfg, context=context,
                micro_batch=args.micro_batch, device=args.device,
                lpips_fn=lpips_fn, full_grid=args.full_grid, limit=args.limit)
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

    board = main_board(candidates, split=args.split, allow=tuple(args.allow_split))
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
