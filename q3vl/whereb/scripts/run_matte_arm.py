"""EPR-022 MATTE entry: ViTMatte Detail_Capture + MattingCriterion on F_pre.

Wrapper form mirrors ``run_uniq4b_arm.py``: parse the arm's own flags, install
the one seam the arm needs, fill the ``arms`` module's hand-off slots, then pass
everything else verbatim to the frozen ``run_amort_arm``.

    python -m q3vl.whereb.scripts.run_matte_arm --run-name amort_MATTE_a

What this wrapper does, and nothing else:

1. parses ``--matte-*`` (defined once, in :func:`q3vl.whereb.amort.matte.add_arguments`);
2. installs the PixGT image seam so the ConvStream can read the spec-5 RGB --
   ``AmortSampleInputs`` has no ``img_pix`` field (shared-infrastructure gap,
   proposal §3 ④), and this is the seam idiom ``run_uniq4b_arm.py:90-91`` uses;
3. injects the ViTMatte recipe's defaults into the base entry's argv **only when
   the caller did not pass them**: ``--arm MATTE``, ``--lr 5e-4``,
   ``--weight-decay 0.1``, ``--scheduler multistep``, ``--pixgt-source`` from
   ``--matte-gt-source``, ``--pixgt-fallback cgt1024``, and the B-4 routing
   switches ``--no-semantic-head --no-sim-field --no-film``;
4. refuses to start when the v2seg base is not on disk (proposal D-11: the
   ``<seg_where>`` readout does not exist before it);
5. writes ``config/matte_setup.json`` with every resolved flag and the sha256 of
   both this file and ``amort/matte.py``;
6. after the run, re-checks the pre-registered assertion (b): the FIRST row of
   ``steps.jsonl`` carries an ``L_*`` column for every configured loss term.
   ``matte.compute_loss`` already asserts this at source on every micro-batch;
   this is the proposal's literal file-level form of the same check.

Flag spelling note: the proposal writes ``--readout seg_where``.  On the shared
entry that name was already taken by P1's field readout, so the shared
infrastructure spells it ``--cond-readout`` (``run_amort_arm.py`` "EPR-018..023
shared flags").  This wrapper passes ``--cond-readout`` straight through; it is
NOT renamed here, so there is exactly one spelling in the run record.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

#: proposal §1 "依赖项": the base the ``<seg_where>`` readout needs
V2SEG_CHECKPOINT = "/home/bc/data/runs/q3vl_base_sft_v2seg_20260814/checkpoint-4976"
#: proposal §3 ③′
B4_ROUTING_FLAGS = ("--no-semantic-head", "--no-sim-field", "--no-film")


def _present(rest: list[str], name: str) -> bool:
    return any(a == name or a.startswith(name + "=") for a in rest)


def _default(rest: list[str], name: str, value: str | None = None) -> None:
    """Append ``name [value]`` unless the caller already passed ``name``."""
    if _present(rest, name):
        return
    rest.append(name)
    if value is not None:
        rest.append(value)


def _first_step_row(path: Path) -> dict | None:
    try:
        with Path(path).open(encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                if line:
                    return json.loads(line)
    except (OSError, ValueError):
        return None
    return None


def check_first_step_row(row: dict | None, losses) -> dict[str, Any]:
    """Pre-registered assertion (b), on the written ``steps.jsonl`` first row."""
    from q3vl.whereb.amort.matte import TERM_OF

    want = [f"L_{TERM_OF[n]}" for n in losses]
    if row is None:
        return {"ok": False, "required": want, "present": [],
                "reason": "steps.jsonl is absent or empty (no training step ran)"}
    present = sorted(k for k in row if k.startswith("L_"))
    missing = [c for c in want if c not in row]
    return {"ok": not missing, "required": want, "present": present,
            "missing": missing, "step": row.get("step")}


def main(argv: list[str] | None = None) -> int:
    from q3vl.whereb.amort import arms as _arms
    from q3vl.whereb.amort import matte

    ap = argparse.ArgumentParser(add_help=False)
    matte.add_arguments(ap)
    ap.add_argument("--matte-base", default=V2SEG_CHECKPOINT,
                    help="the v2seg base this arm's <seg_where> readout needs "
                         "(proposal §1 依赖项 / D-11); injected as --checkpoint "
                         "when the caller passes none")
    ap.add_argument("--matte-allow-missing-base", action="store_true",
                    help="D-11 alternative: start on whatever --checkpoint says "
                         "even though the v2seg product is not on disk")
    own, rest = ap.parse_known_args(argv)

    losses = matte.parse_losses(own.matte_losses)

    # -- pre-registered guard (c), the half that is decidable before start ---
    if _present(rest, "--no-pixgt"):
        raise SystemExit(
            "--no-pixgt is incompatible with --arm MATTE: the whole criterion is "
            "pixel-resolution, and EPR-022's guard (c) pre-registers the "
            "`gt_pix_source` counts and the analytic spot check.  A run without "
            "a pixel-GT provider cannot produce either.")

    # -- the seam ----------------------------------------------------------
    seam = matte.install_image_seam()

    # -- argv defaults (only where the caller was silent) ------------------
    _default(rest, "--arm", "MATTE")
    _default(rest, "--lr", repr(matte.VITMATTE_LR))
    _default(rest, "--weight-decay", repr(matte.VITMATTE_WD))
    _default(rest, "--scheduler", "multistep")
    _default(rest, "--pixgt-source",
             "cgt1024" if own.matte_gt_source == "cgt" else "render")
    # proposal §3 ④: the semantic family and any geometry miss fall back to the
    # .cgt path (the .maskhi shard is the same GT and stays the IO alternative)
    _default(rest, "--pixgt-fallback", "cgt1024")
    if not _present(rest, "--newarm-legacy-routing"):
        for flag in B4_ROUTING_FLAGS:
            _default(rest, flag)
    injected_base = False
    if not _present(rest, "--checkpoint"):
        rest += ["--checkpoint", own.matte_base]
        injected_base = True
        if not Path(own.matte_base).is_dir() and not own.matte_allow_missing_base:
            raise SystemExit(
                f"v2seg base not found: {own.matte_base}\n"
                "EPR-022 §1 依赖项 / D-11 writes this dependency down: the "
                "<seg_where> readout does not exist before the v2seg product is "
                "on disk AND the genwhere cache has been regenerated against it. "
                "Pass --checkpoint <other base> together with a --cond-readout "
                "that does not need the seg tokens (im_end / where_close / "
                "color_close), or --matte-allow-missing-base to override.")

    # -- hand-off slots ----------------------------------------------------
    _arms.ARM_ARGS = own
    setup = {
        "arm": "MATTE",
        "reference": "hustvl/ViTMatte @ main (raw files read 2026-08-14)",
        "matte": {
            "cond": own.matte_cond, "res": float(own.matte_res),
            "norm": own.matte_norm, "losses": list(losses),
            "gt_source": own.matte_gt_source,
            "img_source_flag": own.matte_img_source,
            "precision": own.matte_precision,
            "gt_audit_n": own.matte_gt_audit,
            "gt_audit_tol": own.matte_gt_audit_tol,
            "control_seed": own.matte_control_seed,
            "grad_sigma": own.matte_grad_sigma,
            "pix_diag": not own.matte_no_pix_diag,
        },
        "vitmatte_literals": {
            "norm_const": matte.NORM_CONST,
            "grad_sparsity": matte.GRAD_SPARSITY,
            "lap_max_levels": matte.LAP_MAX_LEVELS,
            "pixel_mean": list(matte.PIXEL_MEAN), "pixel_std": list(matte.PIXEL_STD),
            "lr": matte.VITMATTE_LR, "weight_decay": matte.VITMATTE_WD,
            "betas": list(matte.VITMATTE_BETAS),
            "schedule_values": list(matte.VITMATTE_VALUES),
            "schedule_milestone_fracs": [m / matte.VITMATTE_MAX_ITER
                                         for m in matte.VITMATTE_MILESTONES],
            "warmup_frac": matte.VITMATTE_WARMUP_ITERS / matte.VITMATTE_MAX_ITER,
            "warmup_factor": matte.VITMATTE_WARMUP_FACTOR,
            "reference_max_iter": matte.VITMATTE_MAX_ITER,
        },
        "decisions": {
            "D-1_warmup": "proportional (round(steps * 250/134687))",
            "D-2_norm_const": "literal 262144",
            "D-3_batch": "repo effective_batch (32), not ViTMatte's 16",
            "D-4_grad_clip": "repo max_grad_norm=1.0 (ViTMatte does not declare)",
            "D-8_is_fake": "gt_pix := 0, all four terms, L_*_fake logged apart",
            "D-9_audit": f"mean-abs <= {own.matte_gt_audit_tol} over the first "
                         f"{own.matte_gt_audit} analytic samples, fallback counted",
            "D-10_film": "Linear(2560,256)+ReLU+Linear(256,2*sum(fusion_out)), "
                         "last layer zero-init",
            "D-12_seg_color": "not consumed",
        },
        "image_seam": seam,
        "injected_base_checkpoint": injected_base,
        # matte.builder_kwargs() sets the BUILDER's want_hi so gt_hi / guide_hi
        # are carried (the D-9 audit's comparison raster, its fallback, and the
        # degraded luma path).  The TRAINER's --want-hi is left alone: the arm
        # reads those tensors off the sample, not off forward_geo's guide_hi.
        "builder_want_hi": True,
        "trainer_want_hi": _present(rest, "--want-hi"),
        "b4_routing_flags": (list(B4_ROUTING_FLAGS)
                             if not _present(rest, "--newarm-legacy-routing")
                             else []),
        "delegated_argv": list(rest),
    }
    here = Path(__file__).resolve()
    mfile = here.parent.parent / "amort" / "matte.py"
    setup["matte_sha256"] = hashlib.sha256(mfile.read_bytes()).hexdigest()
    setup["wrapper_sha256"] = hashlib.sha256(here.read_bytes()).hexdigest()
    _arms.ARM_SETUP.clear()
    _arms.ARM_SETUP.update({"matte": setup})

    peek = argparse.ArgumentParser(add_help=False)
    peek.add_argument("--out-root", default="/home/bc/data/runs/where_b")
    peek.add_argument("--run-name", default=None)
    peek.add_argument("--arm", default="MATTE")
    peek.add_argument("--eval-only", action="store_true")
    loc, _ = peek.parse_known_args(rest)
    run_dir = Path(loc.out_root) / (loc.run_name or f"amort_{loc.arm}")
    (run_dir / "config").mkdir(parents=True, exist_ok=True)
    (run_dir / "config" / "matte_setup.json").write_text(
        json.dumps(setup, indent=2, default=str), encoding="utf-8")
    # The per-arm loss record, beside the shared `loss_preregistration.json`
    # `run_amort_arm` writes from this same hook (arms.RUN_REQUIRED_HOOKS): the
    # four ViTMatte terms, equal weight, and nothing from the seven-term stack.
    (run_dir / "config" / "loss_preregistration_matte.json").write_text(
        json.dumps(matte.loss_preregistration(own), indent=2, default=str),
        encoding="utf-8")

    print(f"MATTE cond={own.matte_cond} res={own.matte_res} norm={own.matte_norm} "
          f"losses={list(losses)} gt={own.matte_gt_source} "
          f"img={own.matte_img_source} seam={seam} (installed)", flush=True)

    from q3vl.whereb.scripts.run_amort_arm import main as base_main

    rc = base_main(rest)

    # -- pre-registered assertion (b), on the written file ------------------
    if loc.eval_only:
        rep = {"skipped": "eval_only (no training steps in this run)"}
    else:
        rep = check_first_step_row(_first_step_row(run_dir / "steps.jsonl"), losses)
    (run_dir / "config" / "matte_step_assertion.json").write_text(
        json.dumps(rep, indent=2), encoding="utf-8")
    print(f"MATTE steps.jsonl first-row assertion {json.dumps(rep)}", flush=True)
    if rep.get("ok") is False:
        why = rep.get("reason") or f"the first row is missing {rep.get('missing')}"
        raise SystemExit(
            f"EPR-022 runtime assertion (b) FAILED: {why}.  The ported loss "
            "terms are this arm's pre-registered loss columns; a board built on "
            "a run that cannot show them is not publishable.")
    return rc


if __name__ == "__main__":
    raise SystemExit(main())
