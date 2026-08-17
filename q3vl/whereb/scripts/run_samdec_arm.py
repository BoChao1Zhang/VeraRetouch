"""EPR-019 SAMDEC entry: SAM's mask decoder from scratch on the frozen F_pre.

Thin wrapper in the shape ``run_uniq4b_arm.py`` established: it owns the
``--samdec-*`` flags, fills the arm-registry seams (``samdec.VARIANT`` -- which
every ``*_from_args`` hook reads -- and ``arms.ARM_SETUP``, which lands in
``run_setup.json``), derives the three shared flags this recipe implies, and
delegates everything else verbatim to the frozen ``run_amort_arm.main``.

Derived flags (each only when the caller did not pass it, and each recorded in
``config/samdec_setup.json`` next to the value it was derived from):

``--arm SAMDEC``
    the registry name; a different ``--arm`` is refused rather than overridden.
``--pixgt-source cgt1024`` / ``render``
    ``--samdec-gt png`` (the default) supervises all four families from the
    ``.cgt.png`` at short side 1024, area-downsampled to the decoder's 4x grid;
    ``--samdec-gt raster`` is ablation ④, the analytic re-render for the three
    geometric families (semantic has no geometry and is counted as a fallback by
    ``PixGTProvider.facts()``).
``--scheduler multistep`` + ``--lr`` / ``--weight-decay``
    SAM §A's recipe: linear warmup then x0.1 twice (the milestones come from
    ``samdec.scheduler_kwargs``), lr 8e-4, wd 0.1.  ``optimizer_spec`` sets the
    optimiser itself; the two CLI values are mirrored so ``run_setup.json``'s
    ``train`` block and its ``optimizer_spec`` block cannot disagree.

Everything the arm needs beyond that is already in the shared entry: the readout
(``--cond-readout seg_where`` by default), the B-4 routing (all four families
through the new head, ``SemanticHead`` not built, ``CondEncoder`` frozen), and
the frozen criterion path.

Typical launch (v2seg base + its regenerated genwhere cache; the entry refuses
to start when the cache was made on another base)::

    python -m q3vl.whereb.scripts.run_samdec_arm \\
      --checkpoint /home/bc/data/runs/q3vl_base_sft_v2seg_20260814/checkpoint-4976 \\
      --run-name amort_SAMDEC --max-steps 1200
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

#: ``--samdec-gt`` -> the shared ``--pixgt-source`` value it implies
GT_TO_PIXGT: dict[str, str] = {"png": "cgt1024", "raster": "render"}

#: ``--samdec-sched`` -> ``make_scheduler`` kind (``q3vl/where/calibrate.py:129``).
#: ``sam_step`` is SAM §A's step-wise decay; ``multistep``'s body after the
#: linear warmup is ``gamma ** #(milestones passed)``, i.e. the same schedule.
SCHED_TO_KIND: dict[str, str] = {"sam_step": "multistep", "cosine": "cosine"}


def _has(rest: list[str], flag: str) -> bool:
    return any(a == flag or a.startswith(flag + "=") for a in rest)


def _peek(rest: list[str], flag: str) -> str | None:
    for i, a in enumerate(rest):
        if a == flag:
            return rest[i + 1] if i + 1 < len(rest) else None
        if a.startswith(flag + "="):
            return a.split("=", 1)[1]
    return None


def plan(argv: list[str] | None = None) -> tuple[Any, list[str], dict[str, Any]]:
    """``(own_args, delegated_argv, setup_record)`` -- pure, so it is testable.

    Refuses, rather than silently overriding, every case where an explicit flag
    contradicts the recipe: a run whose config says one GT source and whose
    supervision came from another is exactly the kind of board this campaign
    cannot audit afterwards.
    """
    from q3vl.whereb.amort import samdec

    ap = argparse.ArgumentParser(add_help=False)
    samdec.add_arguments(ap)
    own, rest = ap.parse_known_args(argv)

    arm = _peek(rest, "--arm")
    if arm is None:
        rest = ["--arm", "SAMDEC", *rest]
    elif arm != "SAMDEC":
        raise SystemExit(
            f"run_samdec_arm is the {samdec.ARM} entry; got --arm {arm!r}. "
            "Use run_amort_arm.py directly for the other arms.")

    want_pixgt = GT_TO_PIXGT[own.samdec_gt]
    got_pixgt = _peek(rest, "--pixgt-source")
    if got_pixgt is None:
        rest += ["--pixgt-source", want_pixgt]
    elif got_pixgt != want_pixgt:
        raise SystemExit(
            f"--samdec-gt {own.samdec_gt} means --pixgt-source {want_pixgt} "
            f"(EPR-019 §3.3, ablation ④), but --pixgt-source {got_pixgt} was "
            "passed.  Pick one; a run whose record and whose supervision "
            "disagree is unreadable.")
    if _has(rest, "--no-pixgt"):
        raise SystemExit(
            "--no-pixgt removes the pixel GT this arm is supervised on "
            "(the decoder's native 4x output); it cannot be used with SAMDEC.")

    want_kind = SCHED_TO_KIND[own.samdec_sched]
    got_kind = _peek(rest, "--scheduler")
    if got_kind is None:
        rest += ["--scheduler", want_kind]
    elif got_kind != want_kind:
        raise SystemExit(
            f"--samdec-sched {own.samdec_sched} means --scheduler {want_kind}; "
            f"--scheduler {got_kind} was passed.")
    if not _has(rest, "--lr"):
        rest += ["--lr", str(float(own.samdec_lr))]
    if not _has(rest, "--weight-decay"):
        rest += ["--weight-decay", str(float(own.samdec_wd))]

    variant = samdec.set_variant(**samdec.variant_from_args(own))
    setup = {
        "arm": samdec.ARM,
        "criteria": list(samdec.CRITERIA),
        "variant": dict(variant),
        "loss": samdec.loss_config(None).to_dict(),
        "derived_flags": {"pixgt_source": want_pixgt, "scheduler": want_kind,
                          "lr": float(own.samdec_lr),
                          "weight_decay": float(own.samdec_wd)},
        "schedule_source": {
            "warmup_iters_of_90k": samdec.SAM_WARMUP_ITERS,
            "milestone_fracs": list(samdec.SAM_MILESTONE_FRACS),
            "gamma": samdec.SAM_GAMMA,
            "note": "SAM paper §A, carried over proportionally (U4 step matching)",
        },
        "pretrained_weights_loaded": False,
        "sources": list(samdec.SAM_SOURCES),
        "cli_flags": {k: v for k, v in sorted(vars(own).items())},
    }
    return own, rest, setup


def _write_setup(rest: list[str], setup: dict[str, Any]) -> Path | None:
    """``config/samdec_setup.json`` + the sha256 of the two files this arm owns."""
    peek = argparse.ArgumentParser(add_help=False)
    peek.add_argument("--out-root", default="/home/bc/data/runs/where_b")
    peek.add_argument("--run-name", default=None)
    loc, _ = peek.parse_known_args(rest)
    if not loc.run_name:
        return None
    run_dir = Path(loc.out_root) / loc.run_name
    cfg_dir = run_dir / "config"
    cfg_dir.mkdir(parents=True, exist_ok=True)
    here = Path(__file__).resolve()
    head = here.parent.parent / "amort" / "samdec.py"
    (cfg_dir / "samdec_setup.json").write_text(json.dumps({
        **setup,
        "samdec_sha256": hashlib.sha256(head.read_bytes()).hexdigest(),
        "wrapper_sha256": hashlib.sha256(here.read_bytes()).hexdigest(),
    }, indent=2), encoding="utf-8")
    # The per-arm loss record, beside the shared `loss_preregistration.json`
    # `run_amort_arm` writes from this same hook (arms.RUN_REQUIRED_HOOKS).
    # SAM's criterion carries `w_dice = 1.0`, so the live arms' form -- which
    # is what that file used to carry unconditionally -- was false here.
    from q3vl.whereb.amort import samdec

    # `None`: `plan()` has already pushed the resolved `--samdec-*` values into
    # `samdec.VARIANT`, which is the single source both this file and
    # `run_amort_arm`'s shared record read.
    (cfg_dir / "loss_preregistration_samdec.json").write_text(
        json.dumps(samdec.loss_preregistration(None), indent=2,
                   default=str), encoding="utf-8")
    return run_dir


def _check_steps(run_dir: Path | None) -> dict[str, Any] | None:
    """Post-run witness check: the FIRST ``steps.jsonl`` row carries the four
    pre-registered training columns.

    ``samdec.compute_loss`` already asserts them on every micro-batch, so this
    is the belt-and-braces read of what actually landed on disk -- and it fails
    the job loudly instead of leaving a plausible-looking board behind.
    """
    from q3vl.whereb.amort.samdec import assert_steps_row

    if run_dir is None:
        return None
    p = run_dir / "steps.jsonl"
    if not p.exists():
        return None
    with p.open(encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if line:
                return assert_steps_row(json.loads(line))
    return None


def main(argv: list[str] | None = None) -> int:
    own, rest, setup = plan(argv)
    # the registry's ``run_setup.json`` seam: everything below lands in
    # ``setup["new_arm"]["samdec"]`` next to the readout facts, the pixel-GT
    # facts and the optimizer spec, frozen with the run's source sha256s.
    from q3vl.whereb.amort import arms as _arms

    _arms.ARM_SETUP.clear()
    _arms.ARM_SETUP.update({"samdec": setup})
    run_dir = _write_setup(rest, setup)
    print(f"SAMDEC variant {json.dumps(setup['variant'], default=str)}", flush=True)
    print(f"SAMDEC derived {json.dumps(setup['derived_flags'])} "
          "(SAM decoder from scratch, no pretrained weights)", flush=True)

    from q3vl.whereb.scripts.run_amort_arm import main as base_main

    rc = base_main(rest)
    if rc == 0:
        witness = _check_steps(run_dir)
        if witness is not None:
            print(f"SAMDEC step witness {json.dumps(witness)}", flush=True)
    return rc


if __name__ == "__main__":
    raise SystemExit(main())
