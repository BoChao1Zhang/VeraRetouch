"""EPR-021 entry: the LIIF implicit-coordinate head (``--arm LIIF``).

Form follows ``run_uniq4b_arm.py:29-144``: this script parses its own
``--liif-*`` / ``--geom-reg-*`` flags with ``parse_known_args``, fills the
registry seams (``arms.ARM_KWARGS`` / ``arms.ARM_SETUP`` and
``liifhead.CFG``), writes ``config/liif_setup.json`` next to the run, and hands
everything else to the frozen ``run_amort_arm.main``.

What it forces onto the delegated command line (each one is a written-down part
of the recipe, and each is skipped if the caller already passed it):

    --arm LIIF                  the registry name (arms.py:76)
    --scheduler multistep       MultiStepLR (yaml L54-57 / train_liif.py L83)
    --max-grad-norm 0           no clipping (train_liif.py L114-116)
    --lr 1e-4                   yaml L50-53; keeps cfg.learning_rate and the
                                arm's OptimizerSpec telling the same story
    --pixgt-source render|maskhi  from --liif-gt closed|cgt
    --no-semantic-head / --no-sim-field / --no-film
                                the batch's B-4 口径: all four families go
                                through the new head.  NOT applied when the
                                caller asked for --newarm-legacy-routing (the
                                pre-registered alternative row).

The readout flags are the shared ones and pass straight through: the base entry
spells them ``--cond-readout {seg_where,where_span_pool,where_close,
color_close,im_end,qtok}`` (its own ``--readout`` is the P1 field readout and
predates this batch), plus ``--readout-qtok`` / ``--readout-nseg``.

Selecting nothing here changes nothing anywhere: with no ``--liif-*`` flag the
module's defaults are the reference recipe, and no other arm imports this file.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any


#: the base this arm is pre-registered on (proposal §1 "语言条件读出"): the
#: two ``<seg_*>`` tokens are supervised on the v2seg SFT product and nowhere
#: else.  Passing ``--checkpoint`` explicitly overrides it.
V2SEG_BASE = "/home/bc/data/runs/q3vl_base_sft_v2seg_20260814/checkpoint-4976"


def _has(rest: list[str], flag: str) -> bool:
    return any(a == flag or a.startswith(flag + "=") for a in rest)


def _value_of(rest: list[str], flag: str, default: str | None = None) -> str | None:
    for i, a in enumerate(rest):
        if a == flag and i + 1 < len(rest):
            return rest[i + 1]
        if a.startswith(flag + "="):
            return a.split("=", 1)[1]
    return default


def _check_base(rest: list[str]) -> None:
    """Refuse a ``seg_where`` read-out on a base that does not supervise it.

    Proposal §1 "依赖项", written down: the arm does not start before the v2seg
    product is on disk (and its genwhere cache regenerated -- that half is
    asserted by ``run_amort_arm._assert_genctx_checkpoint``).  待决策 (k)'s
    alternative, ``--cond-readout im_end``, runs on checkpoint-4976 and is
    therefore allowed through.
    """
    from q3vl.whereb.readout import READOUT_NEEDS_V2SEG

    kind = _value_of(rest, "--cond-readout", "seg_where") or "seg_where"
    ckpt = _value_of(rest, "--checkpoint", V2SEG_BASE) or V2SEG_BASE
    if kind not in READOUT_NEEDS_V2SEG:
        return
    if not Path(ckpt).exists():
        raise SystemExit(
            f"--cond-readout {kind} reads a token that only the v2seg SFT "
            f"product supervises, and {ckpt} does not exist yet.  Wait for it "
            "(and for the genwhere cache to be regenerated on it), or run the "
            "pre-registered alternative row: --cond-readout im_end with an "
            "explicit --checkpoint on the 4976 base (proposal 待决策 (k)).")
    if "v2seg" not in str(ckpt):
        print(f"WARNING: --cond-readout {kind} on a base whose path does not "
              f"say v2seg ({ckpt}); the two seg tokens are only supervised "
              "there.", flush=True)


def _install_amount_ref(k: int) -> None:
    """Pin the ``linear`` family's ``amount`` reference grid to ``k x k``."""
    import q3vl.whereb.amort.pixgt as _pix

    base = _pix.PixGTProvider
    if getattr(base, "_liif_amount_ref", None) == (k, k):
        return

    class _LIIFPixGTProvider(base):                     # type: ignore[misc]
        _liif_amount_ref = (k, k)

        def __init__(self, *a, amount_ref_hw=None, **kw):
            super().__init__(*a, amount_ref_hw=(amount_ref_hw or (k, k)), **kw)

    _pix.PixGTProvider = _LIIFPixGTProvider
    print(f"LIIF: linear amount reference grid pinned to {k}x{k} "
          "(proposal 待决策 (f))", flush=True)


def main(argv: list[str] | None = None) -> int:
    from q3vl.whereb.amort import arms as _arms
    from q3vl.whereb.amort import liifhead as L

    ap = argparse.ArgumentParser(add_help=False)
    L.add_arguments(ap)
    own, rest = ap.parse_known_args(argv)

    cfg = L.cfg_from_args(own)
    L.CFG.update(cfg)

    # -- the delegated command line ----------------------------------------
    legacy = _has(rest, "--newarm-legacy-routing")
    if not _has(rest, "--arm"):
        rest = ["--arm", "LIIF", *rest]
    elif _value_of(rest, "--arm") != "LIIF":
        raise SystemExit(
            f"run_liifhead_arm delegates to --arm LIIF; got "
            f"--arm {_value_of(rest, '--arm')!r}")
    if not _has(rest, "--scheduler"):
        rest += ["--scheduler", "multistep"]
    if not _has(rest, "--max-grad-norm"):
        rest += ["--max-grad-norm", "0"]
    if not _has(rest, "--lr"):
        rest += ["--lr", repr(float(L.DEFAULTS["lr"]))]
    else:
        # one source of truth: the OptimizerSpec follows the resolved --lr
        L.CFG["lr"] = float(_value_of(rest, "--lr", L.DEFAULTS["lr"]))
    if not _has(rest, "--pixgt-source"):
        rest += ["--pixgt-source", "maskhi" if cfg["gt"] == "cgt" else "render"]
    if not _has(rest, "--checkpoint"):
        rest += ["--checkpoint", V2SEG_BASE]
    if not legacy:
        for flag in ("--no-semantic-head", "--no-sim-field", "--no-film"):
            if not _has(rest, flag):
                rest.append(flag)

    # -- the written-down base dependency (proposal §1, 待决策 (k)) ----------
    _check_base(rest)

    # -- 待决策 (f): the linear family's `amount` reference grid -------------
    # `PixGTProvider` is constructed inside `run_amort_arm.main` and its
    # default is "raw_mean measured on the grid being rendered".  The proposal
    # pins a 256x256 reference grid instead, so the seam is installed here --
    # the same kind of pre-delegation seam `run_uniq4b_arm.py:90-91` uses.
    # `--liif-amount-ref 0` restores the shared default.
    if int(own.liif_amount_ref) > 0:
        _install_amount_ref(int(own.liif_amount_ref))

    # -- head kwargs reach build_head through the registry seam -------------
    head_kwargs: dict[str, Any] = {
        k: cfg[k] for k in (
            "feat_dim", "cond_dim", "hidden", "local_ensemble", "feat_unfold",
            "cell_decode", "sample_q", "scale_min", "scale_max", "eval_bsize",
            "loss", "pix_diag", "geom_reg_weight", "geom_reg_type_weight",
            "geom_angle_mask_ratio", "geom_band_rx")
    }
    _arms.ARM_KWARGS.update(head_kwargs)

    # a CPU probe: the parameter counts that go into the run record, and a
    # construction failure that surfaces before the VLM is loaded
    probe = L.build_head(in_dim=1024, text_dim=2560, args=None, **head_kwargs)
    params = probe.n_params()
    L.CFG["expected_loss_columns"] = probe.expected_loss_columns()

    setup: dict[str, Any] = {
        "arm": "LIIF",
        "epr": "EPR-021",
        "reference": {
            "paper": "Learning Continuous Image Representation with Local "
                     "Implicit Image Function (arXiv 2012.09161)",
            "repo": "github.com/yinboc/liif",
            "config": "configs/train-div2k/train_edsr-baseline-liif.yaml",
        },
        "config": {k: (list(v) if isinstance(v, tuple) else v)
                   for k, v in cfg.items()},
        "params": params,
        "imnet_in_dim": probe.imnet_in_dim,
        "loss_columns": probe.expected_loss_columns(),
        "decode": {
            "query_grid": f"{cfg['scale_max']}x (gh, gw) cell centres",
            "cell": "2/qh, 2/qw",
            "activation": ("sigmoid" if cfg["loss"] == "bce"
                           else "clamp(0.5*y+0.5, 0, 1)  (test.py:73)"),
            "projection": "area_resize -> (gh, gw)  (data.py:685-686)",
        },
        "optimizer": {"type": "adam", "lr": L.CFG.get("lr", L.DEFAULTS["lr"]),
                      "weight_decay": 0.0, "grouping": "none",
                      "grad_clip": "off"},
        "scheduler": {"kind": "multistep", "fractions": [0.2, 0.4, 0.6, 0.8],
                      "gamma": 0.5, "warmup_steps": 1},
        "readout_flags": {
            "cond_readout": _value_of(rest, "--cond-readout", "seg_where"),
            "readout_qtok": _value_of(rest, "--readout-qtok", "0"),
            "readout_nseg": _value_of(rest, "--readout-nseg", "1"),
            "note": "the proposal writes --readout; the shared entry spells it "
                    "--cond-readout (its own --readout is the P1 field readout)",
        },
        "base_checkpoint": _value_of(rest, "--checkpoint"),
        "geom_reg": (None if cfg["geom_reg_weight"] <= 0 else {
            "weight": cfg["geom_reg_weight"],
            "type_weight": cfg["geom_reg_type_weight"],
            "angle_mask_ratio": cfg["geom_angle_mask_ratio"],
            "band_rx_threshold": cfg["geom_band_rx"],
            "criterion_column": L.GEOM_CRITERION}),
        "b4_all_four_families_through_new_head": not legacy,
        "delegated_flags": [a for a in rest if a.startswith("--")],
        "cli_flags": vars(own),
        "deviations_from_reference": [
            "row 7: 1x1 Conv 1024->64 replaces the trained EDSR encoder",
            "row 8: imnet input gains the 64-d language condition",
            "row 9: out_dim 1 (alpha) instead of 3 (RGB)",
            "rows 14/15: no LR crop, no augmentation",
            "row 17: milestones carried over as fractions of the run",
            "row 18: effective batch 32 (step-matching protocol)",
            "row 20: the encoder is frozen; only adapter+cond+imnet train",
            "row 21: checkpoint selection = quick-eval gate + "
            "local_soft_iou_median (never a val loss / PSNR)",
            "sampling: n_pts = min(sample_q, qh*qw) -- the query grid can hold "
            "fewer than 2304 cells at s ~ 1, where LIIF's 48s x 48s crop "
            "never can",
        ],
    }

    peek = argparse.ArgumentParser(add_help=False)
    peek.add_argument("--out-root", default="/home/bc/data/runs/where_b")
    peek.add_argument("--run-name", default=None)
    loc, _ = peek.parse_known_args(rest)
    # same derivation as run_amort_arm.py:409-410, so the runtime assertion
    # finds steps.jsonl even when the caller did not name the run
    run_dir = Path(loc.out_root) / (loc.run_name or "amort_LIIF")
    L.CFG["run_dir"] = str(run_dir)
    cfg_dir = run_dir / "config"
    cfg_dir.mkdir(parents=True, exist_ok=True)
    here = Path(__file__).resolve()
    head_py = here.parent.parent / "amort" / "liifhead.py"
    setup["run_dir"] = str(run_dir)
    setup["liifhead_sha256"] = hashlib.sha256(head_py.read_bytes()).hexdigest()
    setup["wrapper_sha256"] = hashlib.sha256(here.read_bytes()).hexdigest()
    (cfg_dir / "liif_setup.json").write_text(
        json.dumps(setup, indent=2, default=str), encoding="utf-8")
    # The per-arm loss record, beside the shared `loss_preregistration.json`
    # `run_amort_arm` writes from this same hook (arms.RUN_REQUIRED_HOOKS):
    # LIIF's reference recipe is ONE term on the sampled points, not the live
    # arms' seven-term stack.
    (cfg_dir / "loss_preregistration_liif.json").write_text(
        json.dumps(L.loss_preregistration(own), indent=2, default=str),
        encoding="utf-8")

    _arms.ARM_SETUP.update({"liif": setup})

    print(f"LIIF head: {json.dumps({'params': params, 'in_dim': probe.imnet_in_dim, 'loss': cfg['loss'], 'scale_max': cfg['scale_max'], 'geom_reg': cfg['geom_reg_weight'], 'gt': cfg['gt']})}",
          flush=True)
    print(f"LIIF delegating: {' '.join(rest)}", flush=True)

    from q3vl.whereb.scripts.run_amort_arm import main as base_main

    rc = base_main(rest)

    # Post-run runtime assertion (the in-run one fires at the first quick eval,
    # from liifhead.criteria_columns): the pre-registered loss columns really
    # are in steps.jsonl, first row.
    steps = run_dir / "steps.jsonl"
    if steps.exists():
        with steps.open(encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                if line:
                    rep = L.assert_first_step_row(
                        json.loads(line),
                        expected=L.CFG["expected_loss_columns"])
                    print(f"LIIF loss-column assertion {json.dumps(rep)}",
                          flush=True)
                    break
    return rc


if __name__ == "__main__":
    raise SystemExit(main())
