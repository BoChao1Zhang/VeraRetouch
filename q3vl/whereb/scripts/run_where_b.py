#!/usr/bin/env python
"""FULL-SCALE JOB -- NOT RUN YET.  One Where-B main arm (W01..W08) end to end.

Protocol 5.3 + 10.3: one arm per GPU, 1 epoch, effective batch 32, AdamW 2e-4,
warmup 3%, cosine, clip 1.0, bf16, eval/save every 500 optimizer steps.  The
frozen VLM and the frozen ``BA-3-Joint`` basis are loaded, never trained.

Preconditions, all of which this script asserts rather than assumes:

* the Base SFT checkpoint exists (``F_pre`` and ``H_where`` come from it);
* Where-A has published ``BA-3-Joint``'s ``B.npy`` + ``basis.json``;
* oracle latents exist for the training split and for ``V_where``
  (``scripts/make_oracle_latents.py``);
* the generated ``<where>`` context has been produced for both splits
  (``scripts/make_generated_context.py``) -- there is no GT fallback.

Usage (D-20: rm -f the log first, then verify with ``ps -p <PID>``, never pgrep):
    rm -f /home/bc/data/runs/where_b/W01/train.log
    CUDA_VISIBLE_DEVICES=0 nohup python -m q3vl.whereb.scripts.run_where_b \
        --arm W01 > /home/bc/data/runs/where_b/W01/train.log 2>&1 &
"""

from __future__ import annotations

# --- environment guard: sqlite3 must be imported BEFORE torch ---------------
# Verified 2026-08-05 in the campaign env (/home/bc/envs/q3vl_sft):
#   import torch; import sqlite3  -> ImportError, libstdc++ CXXABI_1.3.15 not found
#   import sqlite3; import torch  -> fine
# torch loads a libstdc++ that shadows the one `_sqlite3`'s dependency chain
# (libicui18n) needs, so any process that touches torch first can never open a
# published shard afterwards.  This job reaches sqlite3 twice over --
# `q3vl.data.shardio` (every published store and every packer) and
# `q3vl.where.maskdata` (the live mask locator opens a build catalog) -- so it
# would die on its first store access without this line.  Importing it first
# costs nothing and inoculates the whole process.  Campaign-wide bug R6, found
# by WHAT-IMPL; the guard belongs in entry points only (a guard inside a library
# module makes that module unimportable in any torch-first process, which is
# strictly worse -- tried and reverted).
import sqlite3  # noqa: F401  (import order is the point)

import argparse
import json
from pathlib import Path

import torch

from q3vl.train.collator import Sft2SegCollator
from q3vl.train.modeling import load_model, load_processor
from q3vl.where.config import CBAND_NORMALIZATION
from q3vl.whereb.config import (
    ARM_IDS,
    SPLIT_DIR,
    assert_read_mount,
    BASIS_ARM,
    CURVE_Z_HI,
    CURVE_Z_LO,
    CURVE_Z_N,
    GENCTX_DIR,
    MODEL_DIR,
    ORACLE_NAMESPACE,
    REPORT_DIR,
    RUN_ROOT,
    SFT_CHECKPOINT,
    WHERE_A_MASKVIEW_DIR,
    WHERE_A_ORACLE_DIR,
    ArmConfig,
    TrainConfig,
    arm_config,
)
from q3vl.whereb.context import ShuffleIndex
from q3vl.whereb.data import BatchBuilder, WhereBDataset, open_dataset
from q3vl.whereb.evaluate import evaluate_arm
from q3vl.whereb.fields import load_basis
from q3vl.whereb.hiddens import FrozenVLM
from q3vl.whereb.model import WhereBModel
from q3vl.whereb.preflight import _env
from q3vl.whereb.stores import GenContextStore, MaskViewStore, OracleStore
from q3vl.whereb.trainer import WhereBTrainer, probe_micro_batch


def assert_genctx_coverage(dataset: WhereBDataset, genctx: GenContextStore,
                           split: str) -> dict:
    """Nit N2: every sample must already have a generated ``<where>`` record.

    ``BalancedContextSampler`` assigns contexts by index up front, so a single
    missing record surfaces as a ``KeyError`` hours into training.  There is no
    GT fallback by design (protocol 5.4), so the only correct response is to
    refuse to start.
    """
    want = {ref.sample_id for ref in dataset.refs}
    have = genctx.sample_ids
    missing = sorted(want - have)
    info = {"split": split, "n_dataset": len(want), "n_genctx": len(have),
            "n_missing": len(missing), "missing_head": missing[:10]}
    if missing:
        raise SystemExit(
            f"{split}: {len(missing)} samples have no generated <where> record "
            f"(e.g. {missing[:3]}). Run scripts/make_generated_context.py for this "
            "split first -- there is deliberately no GT fallback (protocol 5.4)."
        )
    return info


def assert_oracle_contract(oracle: OracleStore, readout: str, split: str,
                           *, n_probe: int = 8) -> dict:
    """The consumer half of the s-cache contract, applied to the oracle latents.

    ``L_curve`` compares Where-B's own ``R(z; rho_pred)`` -- recomputed by
    ``fields.oracle_fields`` through ``apply_readout`` under
    ``CBAND_NORMALIZATION`` -- against ``rho*`` fit by Where-A.  If the two sides
    used different CBand normalisations, or a different z-grid, the loss is not a
    comparison of the same function and nothing downstream says so.  The
    published record declares both, so this asserts on it instead of assuming.

    It also proves the store actually *reads*: a manifest can be complete while
    the payload namespace is the wrong one (see ``config.ORACLE_NAMESPACE``).
    """
    ids = sorted(oracle.sample_ids)
    if not ids:
        raise SystemExit(f"{split}: oracle store {oracle.root} has no members")
    probe = ids[:: max(1, len(ids) // n_probe)][:n_probe]
    norms: set[str] = set()
    curve_lens: set[int] = set()
    for sid in probe:
        fit = (oracle.payload(sid).get("fits") or {}).get(readout)
        if fit is None:
            raise SystemExit(
                f"{split}/{sid}: no {readout!r} fit in {oracle.root}. Where-B is "
                "supervised from the S5 namespace; check config.ORACLE_NAMESPACE."
            )
        if "curve" not in fit or "cband_normalization" not in fit:
            raise SystemExit(
                f"{split}/{sid}: the oracle record in {oracle.root} carries no "
                "'curve'/'cband_normalization'. That is the Where-A arms' own "
                "evaluate() payload (ceiling evidence), not the S5 supervision "
                "namespace -- see config.ORACLE_NAMESPACE."
            )
        norms.add(str(fit["cband_normalization"]))
        curve_lens.add(len(fit["curve"]))
        z = fit.get("curve_z")
        if z is not None and (len(z) != CURVE_Z_N
                              or abs(z[0] - CURVE_Z_LO) > 1e-9
                              or abs(z[-1] - CURVE_Z_HI) > 1e-9):
            raise SystemExit(
                f"{split}/{sid}: oracle curve grid is [{z[0]}, {z[-1]}] with "
                f"{len(z)} points, protocol 5.5 fixes "
                f"linspace({CURVE_Z_LO}, {CURVE_Z_HI}, {CURVE_Z_N})"
            )
    if norms != {CBAND_NORMALIZATION}:
        raise SystemExit(
            f"{split}: oracle latents declare cband_normalization={sorted(norms)}, "
            f"Where-B recomputes r*(z) under {CBAND_NORMALIZATION!r}. L_curve would "
            "compare two different functions."
        )
    if curve_lens != {CURVE_Z_N}:
        raise SystemExit(f"{split}: oracle curve lengths {sorted(curve_lens)} != {CURVE_Z_N}")
    return {"root": str(oracle.root), "readout": readout,
            "n_probed": len(probe), "probe_head": probe[:3],
            "cband_normalization": sorted(norms), "curve_points": sorted(curve_lens),
            "curve_z_grid": {"lo": CURVE_Z_LO, "hi": CURVE_Z_HI, "n": CURVE_Z_N},
            "facts": oracle.facts()}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--arm", required=True, choices=list(ARM_IDS))
    ap.add_argument("--checkpoint", default=str(SFT_CHECKPOINT))
    ap.add_argument("--model-dir", default=str(MODEL_DIR))
    ap.add_argument("--basis-arm", default=BASIS_ARM)
    ap.add_argument("--oracle-root", default=str(WHERE_A_ORACLE_DIR),
                    help="Where-A oracle root; the split is read from "
                         "<root>/<basis-arm>/<oracle-namespace>/<split>")
    ap.add_argument("--oracle-namespace", default=ORACLE_NAMESPACE,
                    help="'s5' is the only namespace that carries curve + "
                         "cband_normalization for every split (see config)")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--dtype", default="bfloat16")
    ap.add_argument("--attn", default="flash_attention_2")
    ap.add_argument("--micro-batch", type=int, default=0, help="0 = probe")
    ap.add_argument("--train-limit", type=int, default=None)
    ap.add_argument("--eval-limit", type=int, default=None)
    ap.add_argument("--out-root", default=str(RUN_ROOT))
    ap.add_argument("--train-split", default="train")
    ap.add_argument("--eval-split", default="V_where")
    args = ap.parse_args()

    run_dir = Path(args.out_root) / args.arm
    run_dir.mkdir(parents=True, exist_ok=True)

    # Fail fast if the read mount is gone: an unattended arm must never start
    # against the hard rw mount, where a stalled read is an unkillable D state
    # (CLAUDE.md 2026-08-10). Probes at depth -- the mount point itself answers
    # from cache long after the server is gone.
    mount_info = assert_read_mount(SPLIT_DIR, WHERE_A_ORACLE_DIR, GENCTX_DIR)
    print(json.dumps({"read_mount": mount_info}, indent=2), flush=True)

    processor, special_ids = load_processor(args.model_dir, 2048)
    collator = Sft2SegCollator(processor, max_length=2048, system_prompt=None)
    vlm_model = load_model(args.checkpoint, attn_implementation=args.attn,
                           dtype=args.dtype).to(args.device).eval()
    vlm = FrozenVLM(vlm_model, processor, device=args.device)
    basis = load_basis(args.basis_arm).to(args.device)

    oracle_base = Path(args.oracle_root) / args.basis_arm / args.oracle_namespace

    def make(split: str, limit):
        ds, info = open_dataset(split, limit=limit,
                                maskview_root=WHERE_A_MASKVIEW_DIR)
        oracle = OracleStore(oracle_base / split)
        genctx = GenContextStore(GENCTX_DIR / split)
        info["genctx"] = assert_genctx_coverage(ds, genctx, split)
        info["oracle"] = assert_oracle_contract(oracle, cfg.readout, split)
        return ds, oracle, genctx, info

    cfg: ArmConfig = arm_config(args.arm)
    train_ds, train_oracle, train_gen, train_info = make(args.train_split, args.train_limit)
    eval_ds, eval_oracle, eval_gen, eval_info = make(args.eval_split, args.eval_limit)
    # the shuffled control swaps instruction *and* where text as a pair, so the
    # index carries both and refuses a partner that is missing either (B3)
    eval_shuffle = ShuffleIndex(eval_ds.shuffle_records(), seed=0)

    model = WhereBModel(cfg).to(args.device)

    train_builder = BatchBuilder(collator, vlm, basis, cfg, oracle=train_oracle,
                                 genctx=train_gen, device=args.device)
    eval_builder = BatchBuilder(collator, vlm, basis, cfg, oracle=eval_oracle,
                                genctx=eval_gen, shuffle_index=eval_shuffle,
                                device=args.device)

    micro = args.micro_batch
    probe = None
    if not micro:
        probe = probe_micro_batch(train_builder, train_ds, model, cfg, device=args.device)
        micro = probe["chosen"] or 2
    tcfg = TrainConfig(arm=args.arm, micro_batch=micro)

    def eval_fn(step: int) -> dict:
        return evaluate_arm(
            model, eval_builder, eval_ds, cfg, batch_size=max(2, micro),
            out_dir=run_dir / f"eval_step{step}",
            resources={"n_trainable_params": model.n_trainable()},
        )

    trainer = WhereBTrainer(model, train_builder, train_ds, cfg, tcfg,
                            run_dir=run_dir, device=args.device, eval_fn=eval_fn)
    setup = trainer.setup()
    setup.update({"env": _env(), "special_token_ids": special_ids,
                  "checkpoint": args.checkpoint, "micro_batch_probe": probe,
                  "splits": {"train": args.train_split, "eval": args.eval_split},
                  "datasets": {"train": train_info, "eval": eval_info},
                  "shuffle_coverage": eval_shuffle.coverage(),
                  "read_mount": mount_info,
                  "oracle_coverage": {
                      "train": train_oracle.coverage(
                          [r.sample_id for r in train_ds.refs], cfg.readout),
                      "eval": eval_oracle.coverage(
                          [r.sample_id for r in eval_ds.refs], cfg.readout)}})
    (run_dir / "run_setup.json").write_text(json.dumps(setup, indent=2, ensure_ascii=False))
    print(json.dumps(setup, indent=2, ensure_ascii=False), flush=True)

    state = trainer.train()

    final = evaluate_arm(model, eval_builder, eval_ds, cfg,
                         batch_size=max(2, micro), out_dir=run_dir / "eval_final",
                         resources={"n_trainable_params": model.n_trainable(),
                                    "peak_memory_gib": (
                                        torch.cuda.max_memory_allocated() / 2**30
                                        if torch.cuda.is_available() else None)})
    rd = Path(REPORT_DIR)
    rd.mkdir(parents=True, exist_ok=True)
    (rd / f"arm_{args.arm}.json").write_text(json.dumps({
        "setup": setup, "final": final, "best_recorded": trainer.best(),
        "n_steps": state.step, "format_stats": {
            "train": train_builder.stats(), "eval": eval_builder.stats()},
    }, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps({"arm": args.arm, "gate": final["gate"],
                      "local_soft_iou_median": final.get("local_soft_iou_median")},
                     indent=2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
