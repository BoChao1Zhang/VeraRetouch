"""Stage 1 (campaign env): put everything AceTone needs on disk.

Written once, read by both inference stages, so that the external code never
touches this repository's loaders and this repository never runs inside the
external environment.

Products, all under ``<out>/inputs``::

    rows.jsonl         one line per scored sample: sample_id, lut_id, task_type,
                       minor, winner_confidence, image path, target path
    luts32.npz         lut_id -> (32,32,32,3) float32, the GT bank LUT put
                       through AceTone's own ``resize_lut`` (grid[b,g,r] order)
    luts_native.json   lut_id -> native grid size / preset path / format
    images/<sid>.png   I  -- the input image, short side 512, area_resize
    targets/<sid>.png  I* -- ``bank.f_star_image(I, a, lut_id)`` at the same
                       size, i.e. the board's own GT after-image
    a_rows.json        the ``A_rows`` report
    cube_parity.json   AceTone's ``read_cube_file`` vs ``dataset_build.lut_io``

``I*`` is written 8-bit because that is what a VLM processor consumes; no
published number is computed from these PNGs -- scoring recomputes ``I*`` in
float from the bank.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

from q3vl.whatb.evaldata import SampleStore
from q3vl.whatb.lutdata import LutBank
from q3vl.whatb.scripts import run_carrier_arm as R

from . import bridge
from .rowset import assert_rows, eval_rows


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", default="/home/bc/data/runs/what_b/acetone_inputs")
    ap.add_argument("--limit", type=int, default=0, help="smoke only")
    ap.add_argument("--no-images", action="store_true",
                    help="row C does not need the PNGs")
    args = ap.parse_args(argv)

    out = Path(args.out)
    (out / "images").mkdir(parents=True, exist_ok=True)
    (out / "targets").mkdir(parents=True, exist_ok=True)

    a_rows = assert_rows()
    rows = eval_rows()
    if args.limit:
        rows = rows[: args.limit]
    print(json.dumps({"A_rows": a_rows}, indent=2), flush=True)
    (out / "a_rows.json").write_text(json.dumps(a_rows, indent=2), encoding="utf-8")

    bank = LutBank()
    minors = R.record_fields(rows)

    # -- the cube-reader parity, array level (no interpolation in the way) ----
    parity = []
    seen = 0
    for r in rows:
        entry = bank.entry(r.lut_id)
        if entry.suffix != ".cube":
            continue
        parity.append(bridge.cube_reader_parity(entry.path))
        seen += 1
        if seen >= 8:
            break
    (out / "cube_parity.json").write_text(json.dumps(parity, indent=2),
                                          encoding="utf-8")
    print(json.dumps({"A_axis_cube_reader_parity_max":
                      max((p["max_abs_delta"] for p in parity), default=None)}),
          flush=True)

    # -- the 32^3 resample --------------------------------------------------- #
    lut_ids = sorted({r.lut_id for r in rows})
    luts32: dict[str, np.ndarray] = {}
    native: dict[str, dict] = {}
    for i, lid in enumerate(lut_ids):
        grid = bank.grid_numpy(lid)
        luts32[lid] = bridge.resize_lut_acetone(grid, 32)
        native[lid] = {"size": int(grid.shape[0]),
                       "path": str(bank.entry(lid).path),
                       "format": bank.entry(lid).suffix}
        if i % 50 == 0:
            print(f"[resample] {i}/{len(lut_ids)}", flush=True)
    np.savez_compressed(out / "luts32.npz", **luts32)
    (out / "luts_native.json").write_text(json.dumps(native, indent=2),
                                          encoding="utf-8")

    # -- rows + images ------------------------------------------------------- #
    store = SampleStore("V_what")
    with (out / "rows.jsonl").open("w", encoding="utf-8") as fh:
        for i, r in enumerate(rows):
            rec = {"sample_id": r.sample_id, "lut_id": r.lut_id,
                   "task_type": r.task_type,
                   "winner_confidence": r.winner_confidence,
                   "source_image_id": r.source_image_id,
                   "minor": (minors.get(r.sample_id) or {}).get("minor"),
                   "image": str(out / "images" / f"{r.sample_id}.png"),
                   "target": str(out / "targets" / f"{r.sample_id}.png")}
            fh.write(json.dumps(rec) + "\n")
            if args.no_images:
                continue
            img, alpha = store.load(r)
            i_star = bank.f_star_image(img, alpha, r.lut_id)
            _save(img, out / "images" / f"{r.sample_id}.png")
            _save(i_star, out / "targets" / f"{r.sample_id}.png")
            if i % 25 == 0:
                print(f"[images] {i}/{len(rows)}", flush=True)

    print(json.dumps({"out": str(out), "n_rows": len(rows),
                      "n_lut_ids": len(lut_ids)}, indent=2), flush=True)
    return 0


def _save(img: torch.Tensor, path: Path) -> None:
    from PIL import Image

    arr = (img.clamp(0, 1).permute(1, 2, 0).cpu().numpy() * 255.0
           ).round().astype(np.uint8)
    Image.fromarray(arr).save(path)


if __name__ == "__main__":
    sys.exit(main())
