#!/usr/bin/env python3
"""EPR-050 v3: stratified 20-row montage pick.

Kept out of epr050_build_degradation.py on purpose: that tool's sha256 is frozen
into run_args.json, and editing it after the run would make every later resume of
the same journal fail the identity guard.  This script only reads pairs.jsonl and
the already-written assets, and draws the sheet with the build tool's own
montage() so the panel layout and the absolute error colour scale are identical.

Selection (deterministic, content-key ordered):
  1. take `--ctrl` control-pool rows,
  2. guarantee `--per-geom` rows for each of subject / linear / radial,
  3. fill the remainder from the main pool in content-key order.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from epr050_build_degradation import ckey, load_config, montage  # noqa: E402

GEOMS = ["subject", "linear", "radial"]        # fallback for an empty journal only


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="/home/bc/data/builds/epr050-degrade-20260825/v3")
    ap.add_argument("--montage-dir",
                    default="/home/bc/VeraRetouch/docs/assets/epr050_degrade_20260825")
    ap.add_argument("--rows", type=int, default=20)
    ap.add_argument("--ctrl", type=int, default=6)
    ap.add_argument("--per-geom", type=int, default=2)
    ap.add_argument("--tag", default="v3")
    # the build module reads err_vmax (the absolute heat-map ceiling) from
    # the experiment TOML, so the sheet has to load the same config
    ap.add_argument("--config", default=None,
                    help="defaults to the config recorded in <out>/run_args.json")
    a = ap.parse_args()

    out = Path(a.out)
    cfg = a.config
    if cfg is None:
        man = json.loads((out / "run_args.json").read_text())
        cfg = man.get("config_path")
        if not cfg:
            raise SystemExit("no --config and run_args.json has no config_path")
    load_config(Path(cfg))
    rows = [json.loads(l) for l in open(out / "pairs.jsonl")]
    rows.sort(key=lambda r: ckey(r["id"]))

    pick, ids = [], set()

    def take(r):
        if r["id"] not in ids and len(pick) < a.rows:
            ids.add(r["id"])
            pick.append(r)

    for r in [x for x in rows if x["pool"] == "ctrl"][: a.ctrl]:
        take(r)
    # v3.6: the geometry families come from the journal, not from a literal --
    # v3.5 renamed subject -> semantic and added `band`, and a hard-coded list
    # would silently guarantee zero rows for the families it does not name.
    fams = sorted({r["mask"]["geom"] for r in rows}) or GEOMS
    for g in fams:                        # floor per geometry, control rows count
        have = sum(x["mask"]["geom"] == g for x in pick)
        for r in [x for x in rows if x["mask"]["geom"] == g][: max(0, a.per_geom - have)]:
            take(r)
    for r in [x for x in rows if x["pool"] == "main"]:
        take(r)

    pick.sort(key=lambda r: (r["pool"], r["mask"]["geom"], r["id"]))
    md = Path(a.montage_dir)
    md.mkdir(parents=True, exist_ok=True)
    p = md / f"montage_{a.tag}_n{len(pick)}.jpg"
    montage(pick, out / "assets", p)

    cen = {}
    for r in pick:
        k = f"{r['pool']}/{r['mask']['geom']}"
        cen[k] = cen.get(k, 0) + 1
    print(f"montage: {p}\n  rows {len(pick)}  census {json.dumps(cen, ensure_ascii=False)}")
    print("  ids: " + ", ".join(r["id"] for r in pick))


if __name__ == "__main__":
    main()
