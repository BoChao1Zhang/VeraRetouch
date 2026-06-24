"""Data-driven calibration of the deterministic over-exposed / dead-black veto thresholds.

The hand-picked thresholds in qa._det_flags were guesses. Instead: measure the distribution of
clip/luma metrics on the PRESET-CLEAN data (the rendered probe previews of pass_c=1 presets — i.e.
real, accepted preset applications), and set the over-exposed / dead-black cutoffs at the extreme
tail (clean data rarely exceeds them => exceeding them = clearly broken). Then cross-validate the
metric against the VLM judge's HILIGHTCLIP / SHADOWCRUSH vetoes on the pilot.

CLI:
  python -m construct.calibrate dist  [--n 3000]            # distribution + suggested thresholds
  python -m construct.calibrate xval  /tmp/r2v4/r2_calib.json [...]   # metric vs VLM-veto agreement
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from concurrent.futures import ThreadPoolExecutor

import numpy as np

from dataset_build.source_qa import db
from .qa import _stats


def _sample_previews(n: int) -> list:
    conn = db.connect()
    rows = conn.execute(
        "SELECT p.after_path FROM preset_previews p JOIN assets a ON a.asset_id=p.asset_id "
        "WHERE a.asset_type='preset' AND a.pass_c=1 AND p.after_path IS NOT NULL "
        "ORDER BY p.id DESC LIMIT %s", (n * 2,)).fetchall()
    conn.close()
    paths = [r["after_path"] for r in rows if r["after_path"] and os.path.exists(r["after_path"])]
    return paths[:n]


def dist(n: int) -> dict:
    paths = _sample_previews(n)
    print(f"sampling {len(paths)} preset-clean preview renders…")
    with ThreadPoolExecutor(max_workers=16) as ex:
        stats = [s for s in ex.map(lambda p: _safe(p), paths) if s]
    hi = np.array([s["hi"] for s in stats]); lo = np.array([s["lo"] for s in stats])
    lum = np.array([s["luma"] for s in stats]); cf = np.array([s["cf"] for s in stats])
    def pct(a, ps): return {f"p{p}": round(float(np.percentile(a, p)), 3) for p in ps}
    out = {
        "n": len(stats),
        "hi_clip": pct(hi, [50, 90, 99, 99.5, 99.9]),
        "lo_crush": pct(lo, [50, 90, 99, 99.5, 99.9]),
        "luma": {**pct(lum, [0.5, 1, 50, 99, 99.5]), "mean": round(float(lum.mean()), 1)},
        "cf": pct(cf, [50, 90, 99, 99.5]),
    }
    # suggested thresholds: clear OVEREXP / DEADBLACK = beyond the clean p99.5 tail
    sugg = {
        "OVEREXP_hi": round(float(np.percentile(hi, 99.5)), 3),
        "OVEREXP_luma": round(float(np.percentile(lum, 99.5)), 1),
        "DEADBLACK_lo": round(float(np.percentile(lo, 99.5)), 3),
        "DEADBLACK_luma": round(float(np.percentile(lum, 0.5)), 1),
    }
    out["suggested_thresholds"] = sugg
    print(json.dumps(out, ensure_ascii=False, indent=1))
    json.dump(out, open("/home/bc/data/datasets/vera_directionA_1M/preset_bank_full/det_calib.json", "w"),
              ensure_ascii=False, indent=1)
    return out


def _safe(p):
    try:
        return _stats(p)
    except Exception:  # noqa: BLE001
        return None


def xval(files: list) -> None:
    """Agreement between deterministic det-flags and the VLM veto on the same dim, from pilot data."""
    R = []
    for f in files:
        R += json.load(open(f))["results"]
    pairs = [("HILIGHTCLIP", "overexp"), ("SHADOWCRUSH", "crush"), ("SATCLIP", "oversat")]
    # each scored variant stores det (metric flags) + vlm_veto (model). compute confusion per dim.
    from collections import Counter
    conf = {dim: Counter() for dim, _ in pairs}
    for r in R:
        for p, s in r["scores"].items():
            if not s or not s.get("reliable"):
                continue
            det = set(s.get("det", [])); vv = set(s.get("vlm_veto", []))
            for dim, _ in pairs:
                m, v = dim in det, dim in vv
                conf[dim][("M" if m else "·") + ("V" if v else "·")] += 1
    print("metric(M) vs VLM(V) agreement per veto dim (MV=both, M·=metric-only, ·V=vlm-only, ··=neither):")
    for dim, _ in pairs:
        c = conf[dim]
        both, mo, vo = c["MV"], c["M·"], c["·V"]
        denom = both + mo + vo
        print(f"  {dim:12s} both={both} metric_only={mo} vlm_only={vo}  "
              f"agree_when_either={'%.2f'%(both/denom) if denom else 'n/a'}")


def main() -> None:
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    d = sub.add_parser("dist"); d.add_argument("--n", type=int, default=3000)
    x = sub.add_parser("xval"); x.add_argument("files", nargs="+")
    a = ap.parse_args()
    if a.cmd == "dist":
        dist(a.n)
    else:
        xval(a.files)


if __name__ == "__main__":
    main()
