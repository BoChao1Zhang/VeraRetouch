"""UNIQ wave-3 entry (EPR-011 P / PIXQ / FQ) -- wrapper over the frozen entry.

Freeze contract: `run_amort_arm.py` and the amort modules queued arms A-E load
must not change while they run.  This wrapper therefore parses its own three
flags, installs the two documented seams from `amort.uniq2`, records its own
setup sidecar, and delegates everything else verbatim to the frozen
`run_amort_arm.main`.  Both patched names are re-read at call time by their
frozen call sites (imports inside functions), which is what makes the seam
sufficient.

Usage: python -m q3vl.whereb.scripts.run_uniq2_arm \
           [--uniq2-presence | --uniq2-pix-attn | --uniq2-family-queries] \
           <every run_amort_arm argument>   # --arm UNIQ expected
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(add_help=False)
    ap.add_argument("--uniq2-presence", action="store_true")
    ap.add_argument("--uniq2-pix-attn", action="store_true")
    ap.add_argument("--uniq2-family-queries", action="store_true")
    own, rest = ap.parse_known_args(argv)

    import q3vl.whereb.amort.losses as alosses
    import q3vl.whereb.amort.model as amodel
    from q3vl.whereb.amort import uniq2

    uniq2.VARIANT.update(presence=own.uniq2_presence,
                         pix_attn=own.uniq2_pix_attn,
                         family_queries=own.uniq2_family_queries)
    amodel.AmortModel = uniq2.AmortModelV2
    alosses.uniq_wta_loss = uniq2.dispatching_wta

    # sidecar: the frozen entry's run_setup.json cannot know these flags, so
    # they are recorded next to it before a single step runs
    peek = argparse.ArgumentParser(add_help=False)
    peek.add_argument("--out-root", default="/home/bc/data/runs/where_b")
    peek.add_argument("--run-name", default=None)
    loc, _ = peek.parse_known_args(rest)
    if loc.run_name:
        cfg_dir = Path(loc.out_root) / loc.run_name / "config"
        cfg_dir.mkdir(parents=True, exist_ok=True)
        here = Path(__file__).resolve()
        u2 = here.parent.parent / "amort" / "uniq2.py"
        (cfg_dir / "uniq2_setup.json").write_text(json.dumps({
            "variant": dict(uniq2.VARIANT),
            "seams": ["amort.model.AmortModel -> uniq2.AmortModelV2",
                      "amort.losses.uniq_wta_loss -> uniq2.dispatching_wta"],
            "uniq2_sha256": hashlib.sha256(u2.read_bytes()).hexdigest(),
            "wrapper_sha256": hashlib.sha256(here.read_bytes()).hexdigest(),
        }, indent=2), encoding="utf-8")
    print(f"UNIQ2 variant {json.dumps(uniq2.VARIANT)} (seams installed)",
          flush=True)

    from q3vl.whereb.scripts.run_amort_arm import main as base_main

    return base_main(rest)


if __name__ == "__main__":
    raise SystemExit(main())
