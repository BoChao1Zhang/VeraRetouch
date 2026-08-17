"""UNIQ wave-7 entry: K-Net gated kernel-update stages (EPR-014) -- NEW FILE.

Thin wrapper over :mod:`run_uniq4b_arm`: it parses only the ``--uniq5-*``
flags, installs ``UniQ5Head`` as the head class through the wrapper's
``EXTRA_HEAD_CLS`` seam, and hands everything else over verbatim.  Every other
seam (token machinery, language-only LoRA, sha256 config freeze) is the
uniq4b one, unduplicated.

``--uniq5-stages 0`` constructs no stage module and leaves the forward on the
UniQ4Head path, i.e. bit-identical to the ST_LANG baseline.
"""

from __future__ import annotations

import argparse
import hashlib
from pathlib import Path


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(add_help=False)
    ap.add_argument("--uniq5-stages", type=int, default=0,
                    help="EPR-014: K-Net kernel-update stages after the "
                         "existing dot product (0 = baseline; K-Net uses 3)")
    ap.add_argument("--uniq5-hard-thr", type=float, default=0.5,
                    help="EPR-014: mask binarisation threshold for the group "
                         "feature (K-Net hard_mask_thr = 0.5)")
    ap.add_argument("--uniq5-soft-feedback", action="store_true",
                    help="EPR-014 ablation: feed the sigmoid field back "
                         "directly instead of binarising it")
    ap.add_argument("--uniq5-per-stage-to-mask", action="store_true",
                    help="EPR-014 ablation: one to_mask per stage instead of "
                         "K-Net-style sharing")
    ap.add_argument("--uniq5-no-query-attn", action="store_true",
                    help="EPR-014 ablation: drop the query-to-query attention")
    ap.add_argument("--uniq5-last-stage-only", action="store_true",
                    help="EPR-014 ablation: supervise only the final stage")
    own, rest = ap.parse_known_args(argv)

    from q3vl.whereb.scripts import run_uniq4b_arm

    if own.uniq5_stages:
        from q3vl.whereb.amort.uniq5 import UniQ5Head

        run_uniq4b_arm.EXTRA_HEAD_CLS = UniQ5Head
        run_uniq4b_arm.EXTRA_HEAD_KWARGS = {
            "n_stages": int(own.uniq5_stages),
            "hard_thr": float(own.uniq5_hard_thr),
            "soft_feedback": bool(own.uniq5_soft_feedback),
            "per_stage_to_mask": bool(own.uniq5_per_stage_to_mask),
            "stage_query_attn": not own.uniq5_no_query_attn,
            "stage_supervision": not own.uniq5_last_stage_only,
        }
        u5 = Path(__file__).resolve().parent.parent / "amort" / "uniq5.py"
        run_uniq4b_arm.EXTRA_SETUP = {
            "uniq5_stages": own.uniq5_stages,
            "uniq5_sha256": hashlib.sha256(u5.read_bytes()).hexdigest(),
            "uniq5_wrapper_sha256": hashlib.sha256(
                Path(__file__).resolve().read_bytes()).hexdigest(),
        }
    print(f"UNIQ5 stages={own.uniq5_stages} (K-Net kernel update)", flush=True)
    return run_uniq4b_arm.main(rest)


if __name__ == "__main__":
    raise SystemExit(main())
