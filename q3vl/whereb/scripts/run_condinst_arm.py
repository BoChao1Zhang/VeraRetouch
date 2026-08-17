"""EPR-023 entry: CondInst conditional-convolution head (``--arm CONDINST``).

Form mirrors ``run_uniq4b_arm.py``: this wrapper parses its own ``--condinst-*``
flags with ``parse_known_args``, freezes them into
``<out-root>/<run-name>/config/condinst_setup.json`` (with the sha256 of
``condinst.py`` and of this file), installs the arm seams, and hands everything
else to ``run_amort_arm.main(rest)`` verbatim.

The faithful recipe is the default of every flag, so

    python -m q3vl.whereb.scripts.run_condinst_arm --run-name condinst_faithful

is the main arm and each ablation row of proposal §4 is one flag away.

What the wrapper injects into ``rest`` (each only when the caller did not pass
it, and each recorded in ``condinst_setup.json``):

``--arm CONDINST``        the registry name
``--no-semantic-head``    B-4: all four families go through this head
``--no-sim-field``        ST_LANG parts are not built
``--no-film``             ditto
``--scheduler warmup_multistep``  detectron2's ``WarmupMultiStepLR``
``--max-grad-norm 0``     CLIP_GRADIENTS.ENABLED False (defaults.py:580)
``--pixgt-source``        ``render`` for ``--condinst-gt raster``, ``cgt1024``
                          for ``--condinst-gt cgt``
``--cond-readout``        when ``--condinst-ctrl`` is ``pool`` / ``query``

It also installs the proposal's first-row witness: the FIRST aggregated training
micro-batch must carry ``L_dice`` (``L_bce`` under the loss ablation) and, with
``--geom-reg-weight > 0``, ``L_geom``.  The check runs on the columns that are
actually produced, so "the loss was defined and never logged" cannot pass.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

#: extra keys merged into ``condinst_setup.json`` by a caller that wraps this
#: entry in turn (same seam as ``run_uniq4b_arm.EXTRA_SETUP``)
EXTRA_SETUP: dict[str, Any] = {}


def _has(rest: list[str], flag: str) -> bool:
    return any(a == flag or a.startswith(flag + "=") for a in rest)


def _ensure(rest: list[str], flag: str, value: str | None,
            injected: dict[str, Any]) -> None:
    """Add ``flag [value]`` unless the caller already passed it."""
    if _has(rest, flag):
        return
    rest.append(flag)
    if value is not None:
        rest.append(str(value))
    injected[flag] = value if value is not None else True


def _peek(rest: list[str]) -> argparse.Namespace:
    """Read the base entry's flags this wrapper needs WITHOUT consuming them."""
    p = argparse.ArgumentParser(add_help=False)
    p.add_argument("--out-root", default="/home/bc/data/runs/where_b")
    p.add_argument("--run-name", default=None)
    p.add_argument("--seed", type=int, default=20260810)
    p.add_argument("--scheduler", default="warmup_multistep")
    p.add_argument("--max-steps", type=int, default=1200)
    p.add_argument("--cond-readout", default="seg_where")
    p.add_argument("--checkpoint", default="")
    known, _ = p.parse_known_args(rest)
    return known


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(add_help=False)
    # -- head (proposal §3 入口) -------------------------------------------
    ap.add_argument("--condinst-mask-out-stride", type=int, default=4,
                    help="MASK_OUT_STRIDE (defaults.py:228); the upsample factor "
                         "is 16 / this")
    ap.add_argument("--condinst-head-channels", type=int, default=8,
                    help="MASK_HEAD.CHANNELS (defaults.py:238)")
    ap.add_argument("--condinst-head-layers", type=int, default=3,
                    help="MASK_HEAD.NUM_LAYERS (defaults.py:239)")
    ap.add_argument("--condinst-no-rel-coords", action="store_true",
                    help="ablation ③: DISABLE_REL_COORDS (defaults.py:241); the "
                         "dynamic head goes 169 -> 153 parameters")
    ap.add_argument("--condinst-mask-branch-channels", type=int, default=128,
                    help="MASK_BRANCH.CHANNELS (defaults.py:246)")
    ap.add_argument("--condinst-mask-branch-convs", type=int, default=4,
                    help="MASK_BRANCH.NUM_CONVS (defaults.py:248)")
    ap.add_argument("--condinst-ctrl", default="seg", choices=["seg", "pool", "query"],
                    help="controller input: seg = <seg_where> h_cond (main arm), "
                         "pool = <where> span mean-pool (row ⑥), query = a "
                         "learnable query cross-attending the span (row ⑩); an "
                         "alias of --cond-readout")
    # -- supervision --------------------------------------------------------
    ap.add_argument("--condinst-gt", default="raster", choices=["raster", "cgt"],
                    help="ablation ④: analytic re-render vs .cgt area-resize")
    ap.add_argument("--condinst-mask-loss", default="dice", choices=["dice", "bce"],
                    help="ablation ②")
    ap.add_argument("--condinst-center-sup", action="store_true",
                    help="explicit centroid supervision on the controller's "
                         "centre output (the main arm does NOT supervise it, "
                         "matching CondInst)")
    ap.add_argument("--condinst-center-weight", type=float, default=0.05,
                    help="weight of --condinst-center-sup (NOVEL: no upstream value)")
    ap.add_argument("--geom-reg-weight", type=float, default=0.0,
                    help="ablation ①: geometry-parameter regression head "
                         "(0 = the branch is not built at all)")
    ap.add_argument("--condinst-geom-hidden", type=int, default=256)
    ap.add_argument("--condinst-geom-route-loss", default="ce", choices=["ce", "none"],
                    help="NOVEL: the proposal registers a 3-way route logit and "
                         "its accuracy column but no loss for it; 'ce' trains it "
                         "inside L_geom, 'none' leaves the column at the "
                         "zero-init argmax")
    ap.add_argument("--condinst-geom-angle-mask-ratio", type=float, default=0.0,
                    help="D-1 (UNRESOLVED): mask the angle columns when "
                         "max(rx,ry)/min(rx,ry) is below this. 0 = the "
                         "conservative default, no masking, recorded as such")
    # -- optimizer / schedule ----------------------------------------------
    ap.add_argument("--optimizer", default="sgd", choices=["sgd", "adamw", "adam"],
                    help="detectron2 build_optimizer default = SGD")
    ap.add_argument("--condinst-lr", type=float, default=0.01,
                    help="SOLVER.BASE_LR (Base-CondInst.yaml); D-2: NOT linearly "
                         "rescaled for the campaign's effective batch 32")
    ap.add_argument("--condinst-weight-decay", type=float, default=1e-4,
                    help="defaults.py:538 (norm layers get 0.0, defaults.py:541; "
                         "bias follows WEIGHT_DECAY, defaults.py:577)")
    ap.add_argument("--lr-milestones", default="",
                    help="absolute steps; empty = carried over from "
                         "(60000, 80000)/90000, i.e. 800,1067 at 1200 steps")
    ap.add_argument("--lr-gamma", type=float, default=0.1,
                    help="defaults.py:543")
    ap.add_argument("--warmup-iters", type=int, default=-1,
                    help="<0 = carried over from WARMUP_ITERS 1000/90000, "
                         "i.e. 13 at 1200 steps")
    ap.add_argument("--warmup-factor", type=float, default=1.0 / 1000.0,
                    help="defaults.py:549")
    return ap


def config_from(own: argparse.Namespace, peek: argparse.Namespace):
    from q3vl.whereb.amort.condinst import CondInstConfig

    return CondInstConfig(
        mask_out_stride=own.condinst_mask_out_stride,
        head_channels=own.condinst_head_channels,
        head_layers=own.condinst_head_layers,
        disable_rel_coords=bool(own.condinst_no_rel_coords),
        mask_branch_channels=own.condinst_mask_branch_channels,
        mask_branch_num_convs=own.condinst_mask_branch_convs,
        ctrl=own.condinst_ctrl,
        gt=own.condinst_gt,
        mask_loss=own.condinst_mask_loss,
        center_sup=bool(own.condinst_center_sup),
        center_weight=own.condinst_center_weight,
        geom_reg_weight=own.geom_reg_weight,
        geom_hidden=own.condinst_geom_hidden,
        geom_route_loss=own.condinst_geom_route_loss,
        geom_angle_mask_ratio=own.condinst_geom_angle_mask_ratio,
        optimizer=own.optimizer,
        lr=own.condinst_lr,
        weight_decay=own.condinst_weight_decay,
        scheduler=peek.scheduler,
        lr_milestones=own.lr_milestones,
        lr_gamma=own.lr_gamma,
        warmup_iters=own.warmup_iters,
        warmup_factor=own.warmup_factor,
        seed=peek.seed,
    )


def install_first_step_witness(cfg) -> None:
    """The proposal's first-row witness, on the columns actually logged.

    ``trainer.compute_micro_batch`` calls ``aggregate`` to build the ``L_*``
    columns that become the ``steps.jsonl`` row, so wrapping it here checks the
    real thing on the first training micro-batch instead of trusting that the
    term was named correctly.
    """
    import q3vl.whereb.amort.trainer as T
    from q3vl.whereb.amort.condinst import assert_first_step_columns

    real = T.aggregate
    state = {"done": False}

    def checked(losses):
        total, stats = real(losses)
        if not state["done"]:
            state["done"] = True
            rep = assert_first_step_columns(stats, cfg)
            print(f"CONDINST first-step witness OK: {rep}", flush=True)
        return total, stats

    T.aggregate = checked


def main(argv: list[str] | None = None) -> int:
    own, rest = build_parser().parse_known_args(argv)
    peek = _peek(rest)
    cfg = config_from(own, peek)

    from q3vl.whereb.amort import arms as A
    from q3vl.whereb.amort import condinst

    condinst.set_config(cfg)
    A.ARM_KWARGS.clear()
    A.ARM_KWARGS["cfg"] = cfg

    # -- the flags the batch's B-4 口径 fixes -------------------------------
    injected: dict[str, Any] = {}
    if _has(rest, "--arm"):
        got = _peek_value(rest, "--arm")
        if got != "CONDINST":
            raise SystemExit(f"run_condinst_arm is --arm CONDINST, got --arm {got}")
    else:
        _ensure(rest, "--arm", "CONDINST", injected)
    # The B-4 routing flag, and the ONLY one of the three `--newarm-legacy-
    # routing` is about.  `AmortModel` already forces `with_semantic=False` for
    # every new arm (model.py:97-101); pinning it here unconditionally makes
    # `--newarm-legacy-routing` a mixed 口径 (legacy routing asked for, semantic
    # head still off) instead of the row it names.  Same shape as
    # `run_matte_arm.py:127-129` / `run_liifhead_arm.py:140-143`.
    if not _has(rest, "--newarm-legacy-routing"):
        _ensure(rest, "--no-semantic-head", None, injected)
    else:
        injected["--no-semantic-head"] = (
            "NOT injected: --newarm-legacy-routing restores the semantic head")
    _ensure(rest, "--no-sim-field", None, injected)
    _ensure(rest, "--no-film", None, injected)
    _ensure(rest, "--scheduler", cfg.scheduler, injected)
    _ensure(rest, "--max-grad-norm", "0", injected)
    _ensure(rest, "--pixgt-source",
            "render" if cfg.gt == "raster" else "cgt1024", injected)
    if cfg.readout_kind:
        _ensure(rest, "--cond-readout", cfg.readout_kind, injected)

    total_steps = int(peek.max_steps or 0)
    schedule = {"kind": cfg.scheduler,
                "milestones": cfg.milestones(total_steps),
                "gamma": cfg.lr_gamma,
                "warmup_steps": cfg.warmup(total_steps),
                "warmup_factor": cfg.warmup_factor,
                "total_steps": total_steps}

    here = Path(__file__).resolve()
    mod = here.parent.parent / "amort" / "condinst.py"
    setup = {
        "arm": "CONDINST",
        "proposal": "experiments/prs/EPR-023_relcoord-dynamic-head/PROPOSAL.md",
        "reference": "CondInst arXiv 2003.05664 / aim-uofa/AdelaiDet",
        "config": cfg.to_dict(),
        "cli_flags": vars(own),
        "injected_flags": injected,
        "schedule": schedule,
        "optimizer": {"type": cfg.optimizer, "lr": cfg.lr,
                      "momentum": cfg.momentum, "nesterov": cfg.nesterov,
                      "weight_decay": cfg.weight_decay,
                      "weight_decay_norm": cfg.weight_decay_norm,
                      "grouping": "norm_bias (detectron2: norm wd 0, bias wd 1e-4)"},
        "base_checkpoint": peek.checkpoint,
        "readout": {"ctrl": cfg.ctrl,
                    "cond_readout": (cfg.readout_kind or peek.cond_readout),
                    "note": ("--condinst-ctrl query is served as a learnable-query "
                             "cross-attention over the <where> span rows; the "
                             "shared readout module has no 'xattn' kind, so the "
                             "span is fed with kind where_span_pool and the head "
                             "reads the rows itself")},
        "deviations": {
            "geom_route_loss": cfg.geom_route_loss,
            "geom_angle_mask": "disabled (D-1 unresolved)",
            "batch_lr_rule": "D-2: BASE_LR 0.01 copied, not linearly rescaled",
            "precision": "D-3: head fp32 (no_autocast), frozen base bf16",
            "dice_as_target": ("dice is the optimisation target, per the user's "
                               "2026-08-14 exemption for this batch of ports"),
        },
        **EXTRA_SETUP,
        "condinst_sha256": hashlib.sha256(mod.read_bytes()).hexdigest(),
        "wrapper_sha256": hashlib.sha256(here.read_bytes()).hexdigest(),
    }
    A.ARM_SETUP.clear()
    A.ARM_SETUP["condinst"] = setup

    cfg_dir = None
    if peek.run_name:
        cfg_dir = Path(peek.out_root) / peek.run_name / "config"
        cfg_dir.mkdir(parents=True, exist_ok=True)
        (cfg_dir / "condinst_setup.json").write_text(
            json.dumps(setup, indent=2, default=str), encoding="utf-8")
        # The per-arm loss record, beside the shared `loss_preregistration.json`
        # `run_amort_arm` builds from THIS SAME hook
        # (`condinst.loss_preregistration`, arms.RUN_REQUIRED_HOOKS): dice IS
        # the optimisation target here, which the live arms' record denies.
        (cfg_dir / "loss_preregistration_condinst.json").write_text(
            json.dumps(condinst.loss_preregistration(own), indent=2,
                       default=str), encoding="utf-8")

    install_first_step_witness(cfg)
    print(f"CONDINST ctrl={cfg.ctrl} rel_coords={not cfg.disable_rel_coords} "
          f"theta={cfg.head_channels}ch x {cfg.head_layers}L "
          f"gt={cfg.gt} loss={cfg.mask_loss} geom_reg={cfg.geom_reg_weight} "
          f"opt={cfg.optimizer}@{cfg.lr} sched={schedule['kind']}"
          f"{schedule['milestones']} warmup={schedule['warmup_steps']} "
          "(seams installed)", flush=True)

    from q3vl.whereb.scripts.run_amort_arm import main as base_main

    rc = base_main(rest)

    # enrich the frozen record with what only the run knows: the resolved token
    # ids, the readout counts and the genwhere cache's checkpoint field
    if cfg_dir is not None:
        try:
            run_setup = json.loads((cfg_dir / "run_setup.json").read_text(encoding="utf-8"))
            setup["resolved"] = run_setup.get("new_arm")
            (cfg_dir / "condinst_setup.json").write_text(
                json.dumps(setup, indent=2, default=str), encoding="utf-8")
        except (OSError, ValueError) as exc:  # noqa: BLE001 -- reported, not fatal
            print(f"condinst_setup enrichment skipped: {exc}", flush=True)
    return rc


def _peek_value(rest: list[str], flag: str) -> str | None:
    for i, a in enumerate(rest):
        if a == flag:
            return rest[i + 1] if i + 1 < len(rest) else None
        if a.startswith(flag + "="):
            return a.split("=", 1)[1]
    return None


if __name__ == "__main__":
    raise SystemExit(main())
