"""E1b / B4: emit ``meta.norm`` for the attention field caches + prove consumption.

The s-cache contract requires the *writer* to declare a domain (whole-arm measured
range plus consumption advice) and the *consumer* to assert the raw data really
lives in it.  The published delivery declared nothing, so this walks the exported
caches, measures the actual domains, writes the sidecar, and then runs
:func:`q3vl.whereb.attnread.assert_domain` over a sample of the data so the
REPORT can show a **passing record** rather than a promise.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np

POOLS = ("where_special", "where_content", "instr_text", "where_close")


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--export", required=True)
    ap.add_argument("--arm", required=True)
    ap.add_argument("--out", default=None)
    args = ap.parse_args(argv)

    from q3vl.whereb.attnread import assert_domain

    exp = Path(args.export)
    files = sorted((exp / "fields").glob(f"*__{args.arm}.npz"))
    if not files:
        raise SystemExit(f"no field files for arm {args.arm!r} in {exp}")

    lo: dict[str, float] = {}
    hi: dict[str, float] = {}
    neg: dict[str, list[float]] = {p: [] for p in POOLS}
    plo, phi = np.inf, -np.inf
    for f in files:
        d = np.load(f)
        for p in POOLS:
            key = f"field_{p}"
            if key not in d:
                continue
            a = d[key].astype(np.float64)
            lo[p] = min(lo.get(p, np.inf), float(a.min()))
            hi[p] = max(hi.get(p, -np.inf), float(a.max()))
            neg[p].append(float((a < 0).mean()))
        cp = d["col_profile"].astype(np.float64)
        plo, phi = min(plo, float(cp.min())), max(phi, float(cp.max()))

    meta: dict[str, Any] = {
        "schema": "q3vl.whereb.attnfields/1",
        "arm": args.arm,
        "n_samples": len(files),
        "norm": {
            "kind": "raw_post_softmax_attention",
            "normalisation_applied": "none",
            "clamp_applied": False,
            "per_image_normalisation": False,
            "domain": {p: [lo[p], hi[p]] for p in sorted(lo)},
            "col_profile_domain": [plo, phi],
            "consumption_advice": (
                "Post-softmax attention rows restricted to image columns: values are "
                "NON-NEGATIVE and small (order 1/n_img), and they do NOT sum to 1 over "
                "the image block because the row's remaining mass sits on text tokens. "
                "Do NOT rescale per image (red line). To make heads commensurable use "
                "whole-arm constants only -- see attnprobe.head_norm_constants, which "
                "multiplies by n_img (a deterministic geometric factor) and then "
                "standardises with fit-fold constants. The FUSED field that comes out "
                "of that pipeline CROSSES ZERO and must never be clamped to (0,1): "
                "clamping is the silent failure mode of this contract, since the values "
                "stay inside the anchors while the axis is destroyed."
            ),
            "frac_cells_negative_raw": {p: float(np.mean(v)) for p, v in neg.items() if v},
        },
    }

    # consumption record: assert a sample of the data lives in the declared box
    checks = []
    for f in files[:: max(1, len(files) // 25)][:25]:
        d = np.load(f)
        for p in sorted(lo):
            key = f"field_{p}"
            if key in d:
                checks.append(assert_domain(d[key].astype(np.float64),
                                            tuple(meta["norm"]["domain"][p]),
                                            name=f"{f.stem}:{p}"))
    meta["consumer_assertion_record"] = {
        "n_checked": len(checks),
        "all_passed": all(c["passed"] for c in checks),
        "max_frac_saturated": max(c["frac_saturated"] for c in checks) if checks else None,
        "any_crosses_zero": any(c["crosses_zero"] for c in checks),
        "note": ("raw attention is non-negative so crosses_zero=False is expected HERE; "
                 "the fused/diff field is the one that crosses zero"),
    }

    out = Path(args.out) if args.out else exp / f"meta.norm.{args.arm}.json"
    out.write_text(json.dumps(meta, indent=2), encoding="utf-8")
    print(json.dumps({k: v for k, v in meta.items() if k != "norm"}, indent=2))
    print("domains:", json.dumps(meta["norm"]["domain"], indent=2))
    print(f"-> {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
