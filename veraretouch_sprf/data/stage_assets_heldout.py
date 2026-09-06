# 源自 experiments/prs/EPR-051_masked-restore-production/stage0/sprf/vlmsft/stage_assets_heldout.py @ 2026-09-06（原件未入 git；原 sha256 见 veraretouch_sprf/PROVENANCE.md），逐字复制/仅改导入与路径解析（EPR-052）。
"""EPR-051 / SFT+ADAPT -- stage the held-out y images for R1@heldout-d6 and R2@4,560.

The held-out side has no cot100k selection plan (that campaign is train-only), so
the sample list is taken from the SAME loader the evaluator uses -- T0.load_shards
filtered against snapshot_newdata_v3.heldout_ids -- and y is written out via
T0.load_pair.  Using the evaluator's own loader (rather than re-deriving tar
offsets) guarantees the image the student sees is bit-identical to the y the
executor scores against.

E-heldout: load_shards re-derives and inflates the held-out set (~35,571); the
filter is asserted with an EQUALITY to 4,560, never a subset check.

x0 is never written.  Only y.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

from veraretouch_sprf import _paths as _P  # EPR-052：替代原 sys.path 注入块
_P.ensure_sys_path()
SPRF = _P.SPRF_LEGACY

import numpy as np  # noqa: E402
import torch  # noqa: E402
from PIL import Image  # noqa: E402

from veraretouch_sprf.train import train_align_time as TA  # noqa: E402

T0, EC, ST = TA.T0, TA.EC, TA.ST


def die(m):
    print(f"FATAL: {m}", flush=True)
    raise SystemExit(2)


def safe_name(key: str) -> str:
    return key.replace("|", "__")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--backbone-run", default="/home/bc/data/runs/epr051_sprf/arm_clut_full")
    ap.add_argument("--heldout-ids",
                    default="/home/bc/VeraRetouch/experiments/prs/EPR-051_masked-restore-production/stage0/snapshot_newdata_v3.heldout_ids.json")
    ap.add_argument("--out", default="/home/bc/data/runs/epr051_vlmsft/assets_y_heldout")
    ap.add_argument("--depth-filter", type=int, default=0)
    ap.add_argument("--limit", type=int, default=0)
    args = ap.parse_args()

    run = Path(args.backbone_run)
    ra = json.loads((run / "run_args.json").read_text())
    cd = ra["config"]
    n_steps = int(ra["data_law"]["n_steps"])
    # MUST precede load_shards: train_stage0.compact_row strips the journal down
    # to what the alpha rebuild needs and DROPS `luts`/`grids`.  SPRF patches it
    # (train_align_time.py:1435 does the same) so the blobs keep LUT identity.
    # Without this the rows load fine and silently have no lut ids.
    ST.install_compact_row(T0)
    T0.ASSET_MODE = cd["data"]["asset_source"]
    bcfg = T0.Cfg(Path(ra["config_path"]))
    samples, blobs, _ = T0.load_shards(bcfg)
    want = set(json.loads(Path(args.heldout_ids).read_text()))
    held = sorted([s for s in samples if s["heldout"] and s["id"] in want],
                  key=lambda s: (s["id"], s["depth"] or 0))
    if len(held) != 4560:
        die(f"held-out filter produced {len(held)} != 4560 (E-heldout equality)")
    print(f"[heldout] {len(held)} samples after equality assertion", flush=True)
    if args.depth_filter:
        held = [s for s in held if (s["depth"] or n_steps) == args.depth_filter]
        print(f"[heldout] depth d{args.depth_filter}: {len(held)}", flush=True)
    if args.limit:
        held = held[: args.limit]

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    index = {}
    inv = EC.InvLutSource(cd["edit"]["inv_cache_dir"], int(cd["edit"]["grid"]))
    for i, s in enumerate(held):
        row = json.loads(blobs[s["id"]])
        x0, y = T0.load_pair(s["shard"], row, s["after_asset"])
        key = f"{s['id']}|d{s['depth'] if s['depth'] is not None else 'full'}"
        arr = (y.clamp(0, 1).cpu().numpy() * 255.0 + 0.5).astype(np.uint8)
        dst = out / f"{safe_name(key)}.png"
        Image.fromarray(arr).save(dst, format="PNG", optimize=False)
        steps = row["steps"]
        luts = row["luts"]
        grids = row["grids"]
        if not (len(steps) == len(luts) == len(grids) == n_steps):
            die(f"{key}: steps/luts/grids lengths {len(steps)}/{len(luts)}/"
                f"{len(grids)} != n_steps {n_steps}")
        index[key] = dict(png=str(dst), bytes=dst.stat().st_size,
                          sha256=hashlib.sha256(dst.read_bytes()).hexdigest(),
                          shard=s["shard"], depth=s["depth"], geom=s["geom"],
                          rec_band=s["rec_band"], id=s["id"],
                          chain=[dict(k=k, kind=steps[k]["kind"], lut=luts[k],
                                      grid=int(grids[k]))
                                 for k in range(n_steps)],
                          cot_step_to_chain_k={str(p): n_steps - p
                                               for p in range(1, n_steps + 1)},
                          calib_s=float(row["calib"]["s"]))
        if (i + 1) % 200 == 0 or i + 1 == len(held):
            print(f"  staged {i+1}/{len(held)}", flush=True)

    kinds = {}
    for r in index.values():
        t = " ".join(c["kind"] for c in r["chain"])
        kinds[t] = kinds.get(t, 0) + 1
    payload = dict(n=len(index), out_dir=str(out), chain_kind_orders=kinds,
                   depth_filter=args.depth_filter or None, index=index)
    ip = out / "assets_index.json"
    ip.write_text(json.dumps(payload))
    (out / "keys.json").write_text(json.dumps(sorted(index)))
    print(f"[done] n={len(index)} index={ip}", flush=True)
    print("[chain kind orders] " + json.dumps(kinds), flush=True)
    print(f"[bytes] {sum(r['bytes'] for r in index.values())/2**30:.2f} GiB", flush=True)
    print("HELDOUT_STAGE_DONE", flush=True)


if __name__ == "__main__":
    main()
