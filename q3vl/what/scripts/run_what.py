"""Run one Stage-What arm (protocol 8 / 10.4 / 11).  NOT EXECUTED.

One arm per GPU (protocol 11's ``T1``-``C2`` waves).  Everything this script does
before the first optimiser step is a refusal check:

* the preflight of protocol 14 must pass;
* the published generated ``<color>`` context (amendment A-4) must exist, be in
  the mode this arm requires, and cover every sample of the split;
* the frozen Where checkpoint must exist and be the *same* one for every arm --
  its digest goes into ``run_setup.json`` and a mismatch with a previous arm's
  record is a hard stop, because "the Where checkpoint is exactly the same and
  frozen for every What arm" (protocol 6) is not something to hope for;
* ``mean_train_u`` must exist: an uncentred ``z_gt`` is a different target;
* the GT LUT bank must resolve every ``lut_id`` in the split.

Submission discipline (CLAUDE.md D-20) is the caller's, not this script's: ``rm
-f`` the log first, then ``ps -p $PID``, then ``tail`` for real output, and only
then write ``job.marker``.  ``pgrep`` is banned for liveness.

Usage
-----
    python -m q3vl.what.scripts.run_what --arm T01 --where-checkpoint <path> \
        --run-dir /home/bc/data/runs/what/T01
"""

from __future__ import annotations

# --- environment guard: sqlite3 must be imported BEFORE torch ---------------
# Verified 2026-08-05 in the campaign env (/home/bc/envs/q3vl_sft):
#   import torch; import sqlite3  -> ImportError, libstdc++ CXXABI_1.3.15 not found
#   import sqlite3; import torch  -> fine
# torch loads a libstdc++ that shadows the one `_sqlite3`'s dependency chain
# (libicui18n) needs, so any process that touches torch first can never open a
# published shard afterwards -- `q3vl.data.shardio` imports sqlite3, and every
# store in this campaign goes through it.  Importing it first costs nothing and
# inoculates the whole process.  This is campaign-wide, not Stage-What specific:
# `q3vl.whereb.stores` sits on the same chain (see NOTES R6).
import sqlite3  # noqa: F401  (import order is the point)

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch

from q3vl.what.config import (
    ARM_IDS,
    COLOR_GENCTX_ROOT,
    GTLUT_DIR,
    RUN_ROOT,
    SFT_CHECKPOINT,
    TrainConfig,
    ZGT_DIR,
    arm_config,
)
from q3vl.what.stores import ColorGenContextStore
from q3vl.what.data import WhatBatchBuilder, WhereRunner, open_dataset
from q3vl.what.hiddens import WhatVLM
from q3vl.what.lut import LutBank
from q3vl.what.model import WhatModel
from q3vl.what.preflight import run_what_preflight
from q3vl.what.provenance import assert_where_consistency, file_sha256
from q3vl.what.trainer import WhatTrainer


def load_center(root: Path) -> tuple[torch.Tensor, float]:
    """``(mean_train_u, C)`` -- the two published train-set constants.

    ``C`` is amendment A-2's ``d_func`` scale.  Both come out of the same file so
    that a run cannot pick up a centre from one corpus and a scale from another.
    """
    p = Path(root) / "zgt_center.npz"
    if not p.exists():
        raise FileNotFoundError(
            f"{p} does not exist -- run scripts/make_zgt_center.py first. "
            "Training against an uncentred SRHT code is a different target "
            "(protocol 7.1) and would not be comparable across arms.")
    z = np.load(p)
    if "d_func_scale" not in z:
        raise KeyError(
            f"{p} predates amendment A-2 and carries no d_func scale C. "
            "Re-run scripts/make_zgt_center.py; L_style_dist may not fall back "
            "to a per-batch scale.")
    return torch.from_numpy(z["mean_u"]).float(), float(z["d_func_scale"])


def build_bank(gtluts: Path, dataset) -> LutBank:
    path_map = dataset.lut_path_map()
    root = Path(gtluts)
    if (root / "manifest.json").exists():
        from q3vl.whereb.stores import PublishedStore

        return LutBank(store=PublishedStore(root), path_map=path_map)
    return LutBank(path_map=path_map)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--arm", required=True, choices=ARM_IDS)
    ap.add_argument("--split", default="train")
    ap.add_argument("--where-checkpoint", default=None)
    ap.add_argument("--sft-checkpoint", default=str(SFT_CHECKPOINT))
    ap.add_argument("--gtluts", default=str(GTLUT_DIR))
    ap.add_argument("--zgt", default=str(ZGT_DIR))
    ap.add_argument("--color-genctx", default=str(COLOR_GENCTX_ROOT),
                    help="root of the published generated-<color> context "
                         "(amendment A-4); one dir per (split, mode)")
    ap.add_argument("--run-dir", default=None)
    ap.add_argument("--run-root", default=str(RUN_ROOT),
                    help="where the other arms' run_setup.json live (B-6 scan)")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--micro-batch", type=int, default=4)
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--skip-preflight", action="store_true")
    args = ap.parse_args()

    cfg = arm_config(args.arm)
    run_dir = Path(args.run_dir or (RUN_ROOT / args.arm))
    run_dir.mkdir(parents=True, exist_ok=True)

    if not args.skip_preflight:
        rep = run_what_preflight(run_dir / "preflight", arm=args.arm, with_data=True,
                                 limit=args.limit)
        if not rep.ok:
            raise SystemExit(
                "preflight failed; protocol 14: fix the implementation and redo "
                "the preflight, do not write smoke numbers into the results\n"
                + json.dumps(rep.to_dict(), indent=1))

    # -- frozen inputs -------------------------------------------------------
    # Amendment A-3: the frozen Where checkpoint is needed by **every** arm, not
    # only the `predicted` ones -- its m_pred weights the natural query half of
    # the loss for all twelve.  C01/C02 simply do not pass its output to the model.
    if not args.where_checkpoint:
        raise SystemExit(
            f"{args.arm} needs --where-checkpoint.  Protocol 6 freezes one Where "
            "checkpoint for every What arm, and amendment A-3 makes its m_pred the "
            "supervision mask of all twelve arms including C01-C04."
        )
    where_runner = WhereRunner.from_checkpoint(Path(args.where_checkpoint),
                                               device=args.device)
    ck = torch.load(args.where_checkpoint, map_location="cpu", weights_only=False)
    digest = file_sha256(args.where_checkpoint)
    # B-6: hard stop before anything expensive happens
    provenance = assert_where_consistency(
        Path(args.run_root), args.arm, digest, args.where_checkpoint)
    where_facts = {
        "source": cfg.where_source, "path": args.where_checkpoint,
        "where_arm": ck["arm"], "step": ck.get("step"),
        "basis_digest": ck.get("basis_digest"),
        "checkpoint_sha256": digest,
        "used_for": ("model input + supervision mask" if cfg.where_source == "predicted"
                     else "supervision mask only (amendment A-3)"),
        **provenance,
    }
    cfg = arm_config(args.arm, where_readout=where_runner.readout)

    oracle_store = None
    if cfg.where_source == "oracle":
        from q3vl.whereb.config import WHERE_A_ORACLE_DIR
        from q3vl.whereb.stores import OracleStore

        oracle_store = OracleStore(Path(WHERE_A_ORACLE_DIR) / args.split)
        where_facts["oracle_root"] = str(WHERE_A_ORACLE_DIR)

    from transformers import AutoProcessor
    from transformers.models.qwen3_vl import Qwen3VLForConditionalGeneration

    processor = AutoProcessor.from_pretrained(args.sft_checkpoint)
    model_vlm = Qwen3VLForConditionalGeneration.from_pretrained(
        args.sft_checkpoint, dtype=torch.bfloat16,
        attn_implementation="eager").to(args.device)
    vlm = WhatVLM(model_vlm, processor, device=args.device)

    from q3vl.train.collator import Sft2SegCollator

    collator = Sft2SegCollator(processor)
    dataset, ds_info = open_dataset(
        args.split, need_mask=(cfg.where_source == "oracle"), limit=args.limit)
    center, d_func_scale = load_center(Path(args.zgt))

    # amendment A-4: training is 50/50 teacher/generated, so the published
    # generated <color> context must exist and must cover the whole split before
    # the first step.  A coverage hole discovered mid-run has no legal repair --
    # falling back to the GT span is exactly what A-4 forbids.
    genctx_root = Path(args.color_genctx) / args.split / cfg.genctx_mode
    color_genctx = ColorGenContextStore(genctx_root, mode=cfg.genctx_mode)
    genctx_coverage = color_genctx.assert_covers(
        [dataset.refs[i].sample_id for i in range(len(dataset))])

    builder = WhatBatchBuilder(
        collator, vlm, cfg, build_bank(Path(args.gtluts), dataset),
        center, d_func_scale, where_runner=where_runner,
        oracle_store=oracle_store, color_genctx=color_genctx,
        device=args.device, seed=cfg.seed)

    model = WhatModel(cfg)
    tcfg = TrainConfig(arm=args.arm, micro_batch=args.micro_batch)
    trainer = WhatTrainer(model, builder, dataset, cfg, tcfg, run_dir=run_dir,
                          device=args.device)

    setup = trainer.setup()
    setup.update({"dataset": ds_info, "where": where_facts,
                  "sft_checkpoint": args.sft_checkpoint,
                  "gtluts": args.gtluts, "zgt": args.zgt,
                  "color_genctx": {"root": str(genctx_root),
                                   "mode": cfg.genctx_mode,
                                   "coverage": genctx_coverage,
                                   "summary": color_genctx.summary()},
                  "started_at": time.strftime("%Y-%m-%dT%H:%M:%S")})
    (run_dir / "run_setup.json").write_text(
        json.dumps(setup, ensure_ascii=False, indent=1), encoding="utf-8")
    print(json.dumps(setup, ensure_ascii=False, indent=1), flush=True)

    trainer.train()
    return 0


if __name__ == "__main__":                                       # pragma: no cover
    raise SystemExit(main())
