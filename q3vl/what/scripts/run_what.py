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
import hashlib
from dataclasses import replace
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
from q3vl.what.boundary import require_color_boundary_scan
from q3vl.what.data import (
    NATURAL_MASK_FROZEN,
    NATURAL_MASK_GLOBAL,
    NATURAL_MASK_ORACLE_GT,
    NATURAL_MASK_SOURCES,
    ORACLE_MISSING_POLICIES,
    ORACLE_MISSING_REJECT,
    WhatBatchBuilder,
    WhereRunner,
    open_dataset,
)
from q3vl.what.evalloop import build_eval_subset, eval_subset_rows, make_eval_fn
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


def _config_digest(setup: dict, subset_manifest: dict) -> str:
    """One digest over the things that define what this run measured.

    Includes the eval subset digest, so "which 256 samples the online proxy ran
    on" cannot drift between arms without the digest changing.
    """
    material = {
        "arm": setup.get("arm"),
        "train": setup.get("train"),
        "where": {k: setup.get("where", {}).get(k)
                  for k in ("checkpoint_sha256", "where_arm", "step")},
        "d_func_scale": setup.get("d_func_scale"),
        "genctx_mode": setup.get("builder", {}).get("genctx_mode"),
        # a run whose supervision mask came from somewhere else did not measure
        # the same thing, so it must not share a digest (deviation D-EXEC4)
        "natural_mask_source": setup.get("builder", {}).get("natural_mask_source"),
        "oracle_missing_latent": setup.get("builder", {}).get("oracle_missing_latent"),
        "eval_subset_digest": subset_manifest.get("digest"),
        "eval_subset_strata": subset_manifest.get("strata_keys"),
    }
    blob = json.dumps(material, ensure_ascii=False, sort_keys=True).encode()
    return hashlib.sha256(blob).hexdigest()


def _env_facts() -> dict:
    """git commit + interpreter + GPU -- the campaign's delivery spec asks every
    run to carry them, and an arm that started four hours after its wave partner
    is exactly the one whose code version nobody can reconstruct later."""
    import platform
    import subprocess

    def _git(*a: str) -> str | None:
        try:
            return subprocess.run(["git", *a], cwd=Path(__file__).resolve().parents[3],
                                  capture_output=True, text=True,
                                  timeout=10).stdout.strip() or None
        except (OSError, subprocess.SubprocessError):        # pragma: no cover
            return None

    return {
        "git_commit": _git("rev-parse", "HEAD"),
        "git_dirty": bool(_git("status", "--porcelain")),
        "python": platform.python_version(),
        "torch": torch.__version__,
        "gpu": (torch.cuda.get_device_name(0) if torch.cuda.is_available() else None),
        "hostname": platform.node(),
    }


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
    ap.add_argument("--eval-split", default="V_what",
                    help="split the in-loop eval subset is drawn from (NF-2)")
    ap.add_argument("--eval-subset-size", type=int, default=None)
    ap.add_argument("--boundary-report", default=None,
                    help="the <color> boundary full-corpus scan report (N-24); "
                         "defaults to the published location")
    ap.add_argument("--skip-preflight", action="store_true")
    # --- deviation D-EXEC4: running the C wave before a Where checkpoint ----
    ap.add_argument("--natural-mask-source", default=NATURAL_MASK_FROZEN,
                    choices=list(NATURAL_MASK_SOURCES),
                    help="which mask weights the natural query half (protocol "
                         "9.1 / amendment A-3).  Anything but the default is a "
                         "declared deviation and is written into run_setup.json")
    ap.add_argument("--oracle-missing-latent", default=ORACLE_MISSING_REJECT,
                    choices=list(ORACLE_MISSING_POLICIES),
                    help="what an oracle arm does with a sample that has no "
                         "Where-A fit (by construction: the global samples)")
    ap.add_argument("--where-readout", default=None,
                    help="Where readout that sizes the rho token / selects the "
                         "oracle fit.  Taken from the checkpoint when there is "
                         "one; required to be explicit when there is not")
    ap.add_argument("--oracle-root", default=None,
                    help="Where-A oracle latents; defaults to "
                         "<ORACLE_DIR>/<basis arm>/<namespace>")
    ap.add_argument("--keep-last", default=None,
                    help="rolling checkpoint budget; 'none' disables deletion "
                         "entirely (review N-26 recommends it for the real runs)")
    args = ap.parse_args()

    cfg = arm_config(args.arm)
    run_dir = Path(args.run_dir or (RUN_ROOT / args.arm))
    run_dir.mkdir(parents=True, exist_ok=True)

    # N-24: gt_color_context *raises* past the boundary, so one over-long sample
    # would kill an arm mid-epoch.  The scan is a cheap record-only pass; it has
    # to have happened, and to have passed, before anything expensive starts.
    boundary = require_color_boundary_scan(args.boundary_report)

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
    deviation = None
    if not args.where_checkpoint:
        # D-EXEC4: the C wave runs before a Where checkpoint exists.  Only the
        # four control arms can: a `predicted` arm has no model input without one.
        if cfg.where_source == "predicted":
            raise SystemExit(
                f"{args.arm} needs --where-checkpoint.  Protocol 6 freezes one "
                "Where checkpoint for every What arm and a WC-conditioned main "
                "arm has no Where input without it."
            )
        if args.natural_mask_source == NATURAL_MASK_FROZEN:
            raise SystemExit(
                f"{args.arm}: no --where-checkpoint, so amendment A-3's "
                "natural_mask_source='frozen_m_pred' cannot be honoured.  "
                "Declare the deviation explicitly with --natural-mask-source "
                f"{{{NATURAL_MASK_ORACLE_GT}|{NATURAL_MASK_GLOBAL}}} -- it is "
                "written into run_setup.json and every per-sample row."
            )
        if not args.where_readout:
            raise SystemExit(
                f"{args.arm}: --where-readout is required without a checkpoint "
                "(it sizes the rho token and selects which Where-A oracle fit "
                "C03/C04 read); do not let it default silently."
            )
        where_runner = None
        digest = None
        provenance = assert_where_consistency(
            Path(args.run_root), args.arm, None, None)
        deviation = {
            "id": "D-EXEC4",
            "ruling": "main agent, task card EXEC-4 item 3 (2026-08-10)",
            "what": ("amendment A-3's twelve-arm unified natural sampling is "
                     "suspended for the C wave: no Where checkpoint exists "
                     "(Where-B main wave halted, new Where design pending)"),
            "natural_mask_source": args.natural_mask_source,
            "consequence": ("the C arms' loss query-colour distribution is not "
                            "the one the T arms will get; if the re-calibrated "
                            "D-W10 rule requires it, the C wave is re-run"),
        }
        where_facts = {
            "source": cfg.where_source, "path": None, "where_arm": None,
            "step": None, "basis_digest": None, "checkpoint_sha256": None,
            "used_for": "not used (deviation D-EXEC4)",
            "readout": args.where_readout,
            **provenance,
        }
        cfg = arm_config(args.arm, where_readout=args.where_readout)
    else:
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
            "readout": where_runner.readout,
            "used_for": ("model input + supervision mask"
                         if cfg.where_source == "predicted"
                         else "supervision mask only (amendment A-3)"),
            **provenance,
        }
        cfg = arm_config(args.arm, where_readout=where_runner.readout)

    oracle_store = None
    if cfg.where_source == "oracle":
        from q3vl.whereb.config import (
            BASIS_ARM,
            ORACLE_NAMESPACE,
            WHERE_A_ORACLE_DIR,
        )
        from q3vl.whereb.stores import OracleStore

        # VERIFIED 2026-08-10: the published layout is
        # <oracle>/<basis arm>/<namespace>/<split> (q3vl.whereb.config's own
        # comment, and run_where_b.py reads it that way).  The previous
        # <oracle>/<split> here resolved to nothing.
        oracle_root = Path(args.oracle_root
                           or (Path(WHERE_A_ORACLE_DIR) / BASIS_ARM / ORACLE_NAMESPACE))
        oracle_store = OracleStore(oracle_root / args.split)
        where_facts["oracle_root"] = str(oracle_root)
        where_facts["oracle_readout"] = cfg.where_readout
        where_facts["oracle_missing_latent"] = args.oracle_missing_latent

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
        device=args.device, seed=cfg.seed,
        natural_mask_source=args.natural_mask_source,
        oracle_missing_latent=args.oracle_missing_latent)

    model = WhatModel(cfg)
    tcfg = TrainConfig(arm=args.arm, micro_batch=args.micro_batch)
    if args.eval_subset_size is not None:
        tcfg = replace(tcfg, eval_subset_size=args.eval_subset_size)
    if args.keep_last is not None:
        # review N-26: `none` keeps every checkpoint (~44 GB/arm) and removes the
        # last way the online proxy's ranking can drop a checkpoint the offline
        # board would have chosen.
        keep = None if str(args.keep_last).lower() in ("none", "null", "0") \
            else int(args.keep_last)
        tcfg = replace(tcfg, keep_last=keep)

    # --- NF-2: the in-loop evaluation -------------------------------------
    # A fixed deterministic subset of V_what, both contexts, LUT-function metrics
    # only.  This is what implements protocol 10.4's eval_steps, activates B-2's
    # checkpoint protection (which is inert without an eval report) and produces
    # amendment A-4's two boards every 500 steps.
    eval_dataset, eval_ds_info = open_dataset(
        args.eval_split, need_mask=(cfg.where_source == "oracle"))
    eval_genctx = ColorGenContextStore(
        Path(args.color_genctx) / args.eval_split / cfg.genctx_mode,
        mode=cfg.genctx_mode)
    eval_genctx.assert_covers(
        [eval_dataset.refs[i].sample_id for i in range(len(eval_dataset))])
    # the oracle store is keyed by sample_id inside ONE split's publication, so
    # the eval split needs its own -- reusing the train store would raise on
    # every V_what sample (found 2026-08-10 while wiring the C wave).
    eval_oracle_store = None
    if cfg.where_source == "oracle":
        from q3vl.whereb.stores import OracleStore as _OracleStore

        eval_oracle_store = _OracleStore(
            Path(where_facts["oracle_root"]) / args.eval_split)
    eval_builder = WhatBatchBuilder(
        collator, vlm, cfg, build_bank(Path(args.gtluts), eval_dataset),
        center, d_func_scale, where_runner=where_runner,
        oracle_store=eval_oracle_store, color_genctx=eval_genctx,
        device=args.device, seed=cfg.seed,
        natural_mask_source=args.natural_mask_source,
        oracle_missing_latent=args.oracle_missing_latent)
    subset_idx, subset_manifest = build_eval_subset(
        eval_subset_rows(eval_dataset, getattr(eval_dataset, "maskviews", None)),
        n=tcfg.eval_subset_size)
    (run_dir / "eval_subset.json").write_text(
        json.dumps(subset_manifest, ensure_ascii=False, indent=1), encoding="utf-8")
    eval_fn = make_eval_fn(
        model, eval_builder, eval_dataset, subset_idx, cfg,
        n_trainable=model.n_trainable(), micro_batch=tcfg.eval_micro_batch,
        subset_manifest=subset_manifest, out_dir=run_dir)

    trainer = WhatTrainer(model, builder, dataset, cfg, tcfg, run_dir=run_dir,
                          device=args.device, eval_fn=eval_fn)

    setup = trainer.setup()
    setup.update({"dataset": ds_info, "where": where_facts,
                  "deviation": deviation,
                  "sft_checkpoint": args.sft_checkpoint,
                  "gtluts": args.gtluts, "zgt": args.zgt,
                  "color_genctx": {"root": str(genctx_root),
                                   "mode": cfg.genctx_mode,
                                   "coverage": genctx_coverage,
                                   "summary": color_genctx.summary()},
                  "color_boundary_scan": boundary,
                  "eval": {"split": args.eval_split, "dataset": eval_ds_info,
                           "subset": {k: v for k, v in subset_manifest.items()
                                      if k != "sample_ids"},
                           "subset_manifest": str(run_dir / "eval_subset.json"),
                           "kind": "online_subset_lut_function",
                           "note": ("LUT-function metrics only; the full V_what "
                                    "with image metrics is scripts/evaluate_what.py "
                                    "and is what selection reads")},
                  "started_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
                  "env": _env_facts()})
    # the subset content is part of the run's identity (NF-2 ruling item 1)
    setup["config_digest"] = _config_digest(setup, subset_manifest)
    (run_dir / "run_setup.json").write_text(
        json.dumps(setup, ensure_ascii=False, indent=1), encoding="utf-8")
    print(json.dumps(setup, ensure_ascii=False, indent=1), flush=True)

    state = trainer.train()

    # run_setup.json is written before the first batch, so its builder facts are
    # the *declared* ones.  The same facts after the epoch carry the measured
    # counters -- the teacher/generated format stats and, for C03/C04, how many
    # samples took the null oracle latent.  Declared and measured are two files
    # on purpose: a silent divergence between them is the thing worth seeing.
    (run_dir / "run_facts_final.json").write_text(json.dumps({
        "arm": args.arm,
        "finished_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "steps": getattr(state, "step", None),
        "builder": builder.facts(),
        "eval_builder": eval_builder.facts(),
        "config_digest": setup["config_digest"],
    }, ensure_ascii=False, indent=1), encoding="utf-8")
    return 0


if __name__ == "__main__":                                       # pragma: no cover
    raise SystemExit(main())
