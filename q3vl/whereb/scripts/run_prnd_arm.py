"""EPR-020 entry: PointRend point-sampled head (``--arm PRND``).

Wrapper of the frozen ``run_amort_arm``, shaped like ``run_uniq4b_arm.py``: it
owns the ``--prnd-*`` flag surface, records it in ``prnd.OPTIONS`` and
``arms.ARM_SETUP``, pins the shared flags the recipe implies, installs the arm's
two extra publication assertions, and then hands everything else over verbatim.

Shared flags this wrapper pins (each only when the caller did not pass it, and
each recorded in ``run_setup.json`` -> ``new_arm.prnd``)::

    --arm PRND                 the registry name
    --want-hi                  gt_hi / pixel GT into the batch and the loss
    --pixgt-source <X>         derived from --prnd-gt (maskhi|cgt1024|analytic)
    --max-grad-norm 0          CLIP_GRADIENTS.ENABLED = False (defaults.py:580)
    --scheduler warmup_multistep     LR_SCHEDULER_NAME (defaults.py:526), SGD only

Everything else -- base checkpoint, splits, steps, batch, seed, the
``--cond-readout`` / ``--readout-qtok`` / ``--readout-nseg`` readout-ablation
group -- belongs to ``run_amort_arm`` and is passed through untouched.

The optimiser (SGD 0.01 / momentum 0.9 / wd 1e-4, norm-exempt decay) and the
schedule (WarmupMultiStepLR, warmup 18, milestones 738/1015, gamma 0.1) come
from the arm module's ``optimizer_spec`` / ``scheduler_kwargs`` hooks, which the
entry script already calls -- they are not re-implemented here.

Example (the main arm, once v2seg + its genwhere cache are in place)::

    python -m q3vl.whereb.scripts.run_prnd_arm \\
        --run-name amort_PRND_main \\
        --checkpoint /home/bc/data/runs/q3vl_base_sft_v2seg_20260814/checkpoint-4976 \\
        --cond-readout seg_where --max-steps 1200
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any


def _pin(rest: list[str], flag: str, value: str | None, pinned: dict[str, Any],
         *, why: str) -> list[str]:
    """Add ``flag [value]`` unless the caller already passed it.  Never silent:
    every pin, and every caller override of a pin, lands in ``pinned``."""
    if any(a == flag or a.startswith(flag + "=") for a in rest):
        pinned[flag] = {"pinned": False, "reason": "passed by the caller", "why": why}
        return rest
    pinned[flag] = {"pinned": True, "value": value, "why": why}
    return rest + ([flag] if value is None else [flag, value])


def install_publication_assert(*, steps_path: Any = None,
                               eval_only: bool | None = None) -> bool:
    """Install the arm's two extra publication assertions.

    Thin delegate: the implementation lives in
    :func:`q3vl.whereb.amort.prnd.install_publication_assert`, which
    ``builder_kwargs`` also calls, so the guard holds even when the operator
    launches ``run_amort_arm --arm PRND`` directly.  Idempotent either way.

    ``steps_path`` / ``eval_only`` are the run facts the shared
    ``evaluate.assert_criteria_ran`` call site (``evaluate.py:660``) does not
    pass; this wrapper installs BEFORE ``builder_kwargs`` runs, so it has to
    hand them over itself or the first installer would win with neither.
    """
    from q3vl.whereb.amort import prnd

    return prnd.install_publication_assert(steps_path=steps_path,
                                           eval_only=eval_only)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(add_help=False)
    from q3vl.whereb.amort import prnd

    prnd.add_arguments(ap)
    own, rest = ap.parse_known_args(argv)
    if any(a in ("-h", "--help") for a in rest):
        # this parser has add_help=False so `--help` reaches run_amort_arm; show
        # the arm's own flags first rather than leaving them undocumented there
        ap.print_help()
        print()
    opts = prnd.configure(own)

    from q3vl.whereb.amort import arms as _arms

    # NOT `_arms.ARM_ARGS = own`: `ARM_ARGS` is what `build_head` is handed, and
    # this namespace has no `--seed`, so the point sampler's `cfg.seed + 314`
    # would silently fall back to the default seed on any run that changed it.
    # The `--prnd-*` values travel in `prnd.OPTIONS` instead, which every hook
    # reads -- one record, three namespaces, no way for them to disagree.
    record = prnd.setup_record()
    _arms.ARM_SETUP.update(record)

    pinned: dict[str, Any] = {}
    if not any(a == "--arm" or a.startswith("--arm=") for a in rest):
        rest = ["--arm", "PRND"] + rest
        pinned["--arm"] = {"pinned": True, "value": "PRND", "why": "registry name"}
    rest = _pin(rest, "--want-hi", None, pinned,
                why="the coarse dense loss and the point labels are at pixel "
                    "resolution (proposal §3 ④)")
    rest = _pin(rest, "--pixgt-source",
                prnd.GT_TO_PIXGT_SOURCE[str(opts["gt"])], pinned,
                why=f"--prnd-gt {opts['gt']!r}")
    rest = _pin(rest, "--max-grad-norm", "0", pinned,
                why="CLIP_GRADIENTS.ENABLED = False (d2 defaults.py:580); the "
                    "gnorm column is still logged")
    if str(opts["optimizer"]) == "sgd":
        rest = _pin(rest, "--scheduler", "warmup_multistep", pinned,
                    why="LR_SCHEDULER_NAME = WarmupMultiStepLR "
                        "(d2 defaults.py:526)")
    record["prnd"]["pinned_shared_flags"] = pinned

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
    steps_path = (Path(loc.out_root) / (loc.run_name or f"amort_{prnd.ARM}")
                  / "steps.jsonl")

    install_publication_assert(steps_path=steps_path, eval_only=loc.eval_only)

    # --- the arm's own pre-registration record, frozen with the run ---------
    if loc.run_name:
        cfg_dir = Path(loc.out_root) / loc.run_name / "config"
        cfg_dir.mkdir(parents=True, exist_ok=True)
        here = Path(__file__).resolve()
        mod = here.parent.parent / "amort" / "prnd.py"
        (cfg_dir / "prnd_setup.json").write_text(json.dumps({
            **record["prnd"],
            "schedule_at_this_horizon": prnd.schedule_for(loc.max_steps),
            "cli_flags": {k: getattr(own, f"prnd_{k}") for k in prnd.DEFAULTS},
            "prnd_sha256": hashlib.sha256(mod.read_bytes()).hexdigest(),
            "wrapper_sha256": hashlib.sha256(here.read_bytes()).hexdigest(),
        }, indent=2, default=str), encoding="utf-8")
        # The per-arm loss record.  Since 2026-08-14 `run_amort_arm` builds
        # `loss_preregistration.json` from THIS SAME hook
        # (`prnd.loss_preregistration` -> `loss_form`,
        # arms.RUN_REQUIRED_HOOKS), so the two files cannot disagree; this one
        # is kept because the delivery contract and the existing boards name it.
        (cfg_dir / "loss_preregistration_prnd.json").write_text(
            json.dumps(prnd.loss_form(), indent=2), encoding="utf-8")

    print(f"PRND points={opts['train_points']} oversample={opts['oversample']} "
          f"importance={opts['importance']} subdiv={opts['subdiv_steps']}x"
          f"{opts['subdiv_points']} point_loss={opts['point_loss']} "
          f"gt={opts['gt']} opt={opts['optimizer']} "
          f"no_subdivision={bool(opts['no_subdivision'])} (seams installed)",
          flush=True)

    from q3vl.whereb.scripts.run_amort_arm import main as base_main

    return base_main(rest)


if __name__ == "__main__":
    raise SystemExit(main())
