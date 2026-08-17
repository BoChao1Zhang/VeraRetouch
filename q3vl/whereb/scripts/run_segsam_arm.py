"""EPR-018 entry: LISA [SEG]-token -> SAM mask decoder (``--arm SEGSAM``).

Wrapper of the frozen ``run_amort_arm``, shaped like ``run_uniq4b_arm.py`` /
``run_prnd_arm.py``: it owns the ``--segsam-*`` flag surface, fills the arm
registry's setup seam, pins the shared flags the recipe implies, installs the
arm's extra publication assertion, and hands everything else over verbatim.

Shared flags this wrapper pins (each only when the caller did not pass it, and
each recorded in ``run_setup.json`` -> ``new_arm.segsam.pinned_shared_flags``)::

    --arm SEGSAM               the registry name
    --no-sim-field             LISA's decoder sees the image embedding and the
    --no-film                  text prompt and nothing else (proposal 3 (2))
    --no-semantic-head         all four families through one decoder (3 (3));
                               NOT pinned under --newarm-legacy-routing, whose
                               whole point is to restore the semantic head
    --scheduler linear         DeepSpeed WarmupDecayLR (train_ds.py:279-287)
    --pixgt-source <X>         derived from --segsam-gt
    --pixgt-fallback <Y>       semantic / geometry misses -> .cgt (3 (6))

Everything else -- base checkpoint, splits, steps, batch, seed, the
``--cond-readout`` / ``--readout-qtok`` / ``--readout-nseg`` readout-ablation
group -- belongs to ``run_amort_arm`` and passes through untouched.  The
optimiser (AdamW 3e-4 / wd 0 / betas 0.9,0.95) and the warmup (24 steps at the
1200-step horizon) come from the arm module's ``optimizer_spec`` /
``scheduler_kwargs`` hooks, which the entry script already calls.

Example (the main arm, once v2seg and its genwhere cache are in place)::

    python -m q3vl.whereb.scripts.run_segsam_arm \\
        --run-name amort_SEGSAM_main \\
        --checkpoint /home/bc/data/runs/q3vl_base_sft_v2seg_20260814/checkpoint-4976 \\
        --cond-readout seg_where --max-steps 1200
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any


def _pinned_flag(rest: list[str], flag: str) -> bool:
    return any(a == flag or a.startswith(flag + "=") for a in rest)


def _pin(rest: list[str], flag: str, value: str | None, pinned: dict[str, Any],
         *, why: str) -> list[str]:
    """Add ``flag [value]`` unless the caller already passed it.  Never silent:
    every pin, and every caller override of a pin, lands in ``pinned``."""
    if any(a == flag or a.startswith(flag + "=") for a in rest):
        pinned[flag] = {"pinned": False, "reason": "passed by the caller", "why": why}
        return rest
    pinned[flag] = {"pinned": True, "value": value, "why": why}
    return rest + ([flag] if value is None else [flag, value])


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(add_help=False)
    from q3vl.whereb.amort import segsam

    segsam.add_arguments(ap)
    own, rest = ap.parse_known_args(argv)
    if any(a in ("-h", "--help") for a in rest):
        # `--help` belongs to run_amort_arm's parser (this one has
        # add_help=False so the rest passes through), but a caller asking for
        # help must still see the flags THIS entry owns.
        print(ap.format_help(), flush=True)
    opts = segsam.configure(own)

    from q3vl.whereb.amort import arms as _arms

    # NOT `_arms.ARM_ARGS = own`: `ARM_ARGS` is what `build_head` is handed and
    # this namespace has no `--seed`, so the head's initialisation seed would
    # silently fall back to the default on any run that changed it.  The
    # `--segsam-*` values travel in `segsam.OPTIONS`, which every hook reads --
    # one record, three namespaces, no way for them to disagree.
    _arms.ARM_SETUP.update(segsam.setup_record())

    pinned: dict[str, Any] = {}
    if not any(a == "--arm" or a.startswith("--arm=") for a in rest):
        rest = ["--arm", "SEGSAM"] + rest
        pinned["--arm"] = {"pinned": True, "value": "SEGSAM",
                           "why": "registry name"}
    rest = _pin(rest, "--no-sim-field", None, pinned,
                why="LISA's decoder takes the image embedding and the text "
                    "prompt only; extra stem channels are not part of the port")
    rest = _pin(rest, "--no-film", None, pinned,
                why="the conditioning enters through text_hidden_fcs, not FiLM")
    # The B-4 routing flag, and the ONLY one of the three that
    # `--newarm-legacy-routing` is about.  `AmortModel` already forces
    # `with_semantic=False` for every new arm (model.py:97-101), so pinning it
    # here unconditionally would make the legacy-routing row impossible to run:
    # `--newarm-legacy-routing --no-semantic-head` is the mixed 口径 (legacy
    # routing asked for, semantic head still off).  Same shape as
    # `run_matte_arm.py:127-129` / `run_liifhead_arm.py:140-143`.
    if not _pinned_flag(rest, "--newarm-legacy-routing"):
        rest = _pin(rest, "--no-semantic-head", None, pinned,
                    why="all four families through one decoder, as LISA runs "
                        "one head over its whole dataset (proposal 3 (3))")
    else:
        pinned["--no-semantic-head"] = {
            "pinned": False, "reason": "--newarm-legacy-routing",
            "why": "the legacy-routing row restores the semantic head; pinning "
                   "this flag as well would be a mixed 口径"}
    rest = _pin(rest, "--scheduler", "linear", pinned,
                why="DeepSpeed WarmupDecayLR: linear warmup, then linear decay "
                    "to 0 (train_ds.py:279-287)")
    src, fb = segsam.GT_TO_PIXGT[str(opts["gt"])]
    rest = _pin(rest, "--pixgt-source", src, pinned, why=f"--segsam-gt {opts['gt']!r}")
    rest = _pin(rest, "--pixgt-fallback", fb, pinned,
                why="semantic declares no geometry and geometry misses fall "
                    "back to .cgt, area-resized (proposal 3 (6))")
    _arms.ARM_SETUP["segsam"]["pinned_shared_flags"] = pinned

    # --- where the run will keep its books ----------------------------------
    # Read BEFORE the assertion is installed: the assertion needs the path.
    # `run_amort_arm.py:537-538` is the rule being mirrored -- an absent
    # `--run-name` means `amort_<ARM>`, not "no run directory".
    peek = argparse.ArgumentParser(add_help=False)
    peek.add_argument("--out-root", default="/home/bc/data/runs/where_b")
    peek.add_argument("--run-name", default=None)
    peek.add_argument("--max-steps", type=int, default=1200)
    peek.add_argument("--eval-only", action="store_true")
    loc, _ = peek.parse_known_args(rest)
    steps_path = (Path(loc.out_root) / (loc.run_name or f"amort_{segsam.ARM}")
                  / "steps.jsonl")

    # --- the arm's extra publication assertion ------------------------------
    # `run_amort_arm._finish_board` does `from ... import assert_criteria_ran`
    # INSIDE the function, so replacing the module attribute here reaches it --
    # and it runs before `metrics.json` is written, which is what "refuse to
    # publish" has to mean.  Same seam `run_uniq4b_arm.py` uses for its swaps;
    # the sha256 freeze below covers this file.
    #
    # `steps_path` is passed on every call because the OTHER call site --
    # `evaluate.evaluate_arm` (`evaluate.py:660`), the one that writes
    # `metrics.json` three lines later -- hands over `board` and `arm` and no
    # steps row at all.  The 2026-08-15 smoke failed there with "columns
    # present: []" while `steps.jsonl` row 1 carried both columns: the row was
    # never looked up, not missing.  `--eval-only` is read from argv rather
    # than from the board because that call site has not built
    # `deep_supervision_check` yet either.
    import q3vl.whereb.amort.evaluate as _ev

    _base_assert = _ev.assert_criteria_ran

    def _assert_with_segsam(board, arm, *, head_facts=None, steps_row=None):
        report = _base_assert(board, arm, head_facts=head_facts,
                              steps_row=steps_row)
        if arm == segsam.ARM:
            skipped = bool(loc.eval_only
                           or (board.get("deep_supervision_check") or {}).get("skipped")
                           or board.get("eval_only"))
            report["segsam_publication"] = segsam.assert_publishable(
                board, steps_row=steps_row, eval_only=skipped,
                steps_path=steps_path)
        return report

    _ev.assert_criteria_ran = _assert_with_segsam

    # --- the arm's own pre-registration record, frozen with the run ---------
    if loc.run_name:
        cfg_dir = Path(loc.out_root) / loc.run_name / "config"
        cfg_dir.mkdir(parents=True, exist_ok=True)
        here = Path(__file__).resolve()
        mod = here.parent.parent / "amort" / "segsam.py"
        (cfg_dir / "segsam_setup.json").write_text(json.dumps({
            **segsam.setup_record()["segsam"],
            "warmup_at_this_horizon": segsam.warmup_for(loc.max_steps),
            "total_steps": loc.max_steps,
            "cli_flags": {k: getattr(own, f"segsam_{k}", None)
                          for k in segsam.DEFAULTS},
            "pinned_shared_flags": pinned,
            "segsam_sha256": hashlib.sha256(mod.read_bytes()).hexdigest(),
            "wrapper_sha256": hashlib.sha256(here.read_bytes()).hexdigest(),
        }, indent=2, default=str), encoding="utf-8")
        # The per-arm loss record.  Since 2026-08-14 `run_amort_arm` builds
        # `loss_preregistration.json` from THIS SAME hook
        # (`segsam.loss_preregistration`, arms.RUN_REQUIRED_HOOKS), so the two
        # files cannot disagree; this one is kept because the delivery contract
        # and the existing boards name it.
        (cfg_dir / "loss_preregistration_segsam.json").write_text(
            json.dumps(segsam.loss_form(float(opts["bce_weight"]),
                                        float(opts["dice_weight"])),
                       indent=2), encoding="utf-8")

    print(f"SEGSAM weights={'scratch' if opts['scratch'] else opts['weights']} "
          f"frozen_decoder={bool(opts['frozen_decoder'])} "
          f"dice={opts['dice_weight']} bce={opts['bce_weight']} "
          f"gt={opts['gt']} sup={opts['sup']} "
          f"pixgt={src}/{fb} (seams installed)", flush=True)

    from q3vl.whereb.scripts.run_amort_arm import main as base_main

    return base_main(rest)


if __name__ == "__main__":
    raise SystemExit(main())
