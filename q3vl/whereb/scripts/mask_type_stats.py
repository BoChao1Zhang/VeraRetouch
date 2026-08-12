"""mask_type census + semantic-class .cgt inspection (P1/P3 routing prerequisite).

Why this exists
---------------
The eight-slot local plan is ``{"radial": 2, "semantic": 2, "band": 2, "linear": 2}``
(``dataset_build/src/construct/canonical_masks.py:21``), but **the family is not
carried by any Where-B-visible field**: the split index, the ``.rec.json`` and the
maskview ``meta`` all drop it.  The only surviving label is ``slot_id`` in the
source build's ``.vrmeta.json``, reachable through ``MaskResolver`` with a
non-default suffix.

Two questions this answers, both gating the P1/P3 head design:

1. **How much of local training is the semantic family?**  Above ~10% the spatial
   head cannot be a pure geometry regressor -- radial/band/linear are smooth
   ellipses and ramps, semantic is an object silhouette, and one conv tower
   asked to do both is the gradient conflict the two-path design exists to avoid.
2. **Is the semantic ``.cgt`` actually the object contour, or an ellipse
   approximation?**  This decides whether the semantic head can be supervised
   directly on ``.cgt`` or has to be re-routed onto SAM3 pseudo-labels.
   ``_semantic_alpha`` (``canonical_masks.py:79-87``) blurs the hard instance
   mask and re-multiplies by it, so it *should* hug the silhouette -- but that is
   a code reading, and the card asks for eyes on the actual rasters.
"""

from __future__ import annotations

import argparse
import collections
import json
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np


def family_of(slot_id: str) -> str:
    return str(slot_id).rsplit("-", 1)[0] if slot_id else "unknown"


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--splits", nargs="*", default=["V_where", "train"])
    ap.add_argument("--out", required=True)
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--workers", type=int, default=24)
    ap.add_argument("--n-viz", type=int, default=20)
    ap.add_argument("--viz-split", default="V_where")
    args = ap.parse_args(argv)

    t0 = time.time()
    from q3vl.where.maskdata import MaskResolver
    from q3vl.whereb.data import open_dataset

    out_dir = Path(args.out)
    (out_dir / "viz_masktype").mkdir(parents=True, exist_ok=True)

    report: dict = {"splits": {}}
    keep: dict[str, list] = {}

    for split in args.splits:
        ds, _ = open_dataset(split, need_mask=False, limit=args.limit)
        rows = ds.meta_rows()
        local = [i for i, r in enumerate(rows) if r.get("render_mode") == "local"]
        print(f"{split}: {len(local)} local of {len(rows)}", flush=True)

        res = MaskResolver(verify="none", suffix=".vrmeta.json")
        recs = [ds.record(i) for i in local]

        def one(rec):
            try:
                vm = json.loads(res.read_bytes(res.resolve(rec)).decode())
                return rec["sample_id"], vm.get("slot_id"), rec
            except Exception as exc:                       # noqa: BLE001
                return rec["sample_id"], None, f"{type(exc).__name__}: {exc}"

        got = []
        with ThreadPoolExecutor(max_workers=args.workers) as ex:
            for n, r in enumerate(ex.map(one, recs)):
                got.append(r)
                if (n + 1) % 5000 == 0:
                    print(f"  [{n+1}/{len(recs)}] {time.time()-t0:.0f}s", flush=True)

        fam = collections.Counter()
        slot = collections.Counter()
        by_build = collections.defaultdict(collections.Counter)
        fails = 0
        for sid, s_id, rec in got:
            if s_id is None:
                fails += 1
                continue
            f = family_of(s_id)
            fam[f] += 1
            slot[s_id] += 1
            by_build[rec["build"]][f] += 1
        n_ok = sum(fam.values())
        report["splits"][split] = {
            "n_local": len(local), "n_resolved": n_ok, "n_failed": fails,
            "family_counts": dict(fam),
            "family_fracs": {k: v / n_ok for k, v in fam.items()} if n_ok else {},
            "slot_counts": dict(slot),
            "by_build": {b: dict(c) for b, c in sorted(by_build.items())},
            "semantic_frac": fam.get("semantic", 0) / n_ok if n_ok else None,
        }
        print(f"  {split}: " + "  ".join(
            f"{k} {v} ({100*v/n_ok:.1f}%)" for k, v in sorted(fam.items())), flush=True)
        if split == args.viz_split:
            keep[split] = [(sid, s_id, rec) for sid, s_id, rec in got
                           if s_id and family_of(s_id) == "semantic"]

    # ---- semantic .cgt inspection -----------------------------------------
    sem = keep.get(args.viz_split, [])[: args.n_viz]
    if sem:
        print(f"rendering {len(sem)} semantic .cgt panels", flush=True)
        _viz_semantic(out_dir / "viz_masktype", sem, args.viz_split)
        report["semantic_viz"] = {
            "split": args.viz_split, "n": len(sem),
            "sample_ids": [s for s, _, _ in sem],
        }

    (out_dir / "mask_type_stats.json").write_text(
        json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"done in {(time.time()-t0)/60:.1f} min -> {out_dir}", flush=True)
    return 0


def _viz_semantic(viz_dir: Path, sem, split: str) -> None:
    """Image + .cgt alpha + contour overlay, so "hugs the object" is checkable.

    No per-image min-max: the alpha is already in [0,1] and is drawn on a fixed
    0..1 scale, which is the whole point -- a soft ellipse and a tight silhouette
    must not be normalised into looking alike.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from PIL import Image

    from q3vl.where.maskdata import MaskResolver

    mres = MaskResolver(verify="none")            # default suffix = .cgt.png
    for sid, slot_id, rec in sem:
        try:
            raw = mres.read_bytes(mres.resolve(rec))
            alpha = np.asarray(Image.open(__import__("io").BytesIO(raw)).convert("L"),
                               dtype=np.float64) / 255.0
            ipath = rec["image"]["origin"]
            img = None
            for key in ("path", "file", "root"):
                p = ipath.get(key) if isinstance(ipath, dict) else None
                if p and Path(str(p)).is_file():
                    img = np.asarray(Image.open(str(p)).convert("RGB"))
                    break
        except Exception as exc:                            # noqa: BLE001
            print(f"  !! {sid}: {exc}", flush=True)
            continue

        ncol = 3 if img is not None else 2
        fig, ax = plt.subplots(1, ncol, figsize=(5.2 * ncol, 4.4))
        ax = np.atleast_1d(ax)
        for a in ax:
            a.set_xticks([]); a.set_yticks([])
        k = 0
        if img is not None:
            ax[k].imshow(img); ax[k].set_title("source image", fontsize=9); k += 1
        im = ax[k].imshow(alpha, cmap="magma", vmin=0, vmax=1)
        ax[k].set_title(f"{slot_id}  .cgt alpha (fixed 0..1)", fontsize=9)
        fig.colorbar(im, ax=ax[k], fraction=0.03)
        k += 1
        ax[k].imshow(alpha, cmap="gray", vmin=0, vmax=1)
        ax[k].contour(alpha, levels=[0.05, 0.5, 0.95], colors=["cyan", "lime", "red"],
                      linewidths=0.8)
        ax[k].set_title("contours 0.05 / 0.50 / 0.95", fontsize=9)
        frac = float((alpha > 0.5).mean())
        soft = float(((alpha > 0.05) & (alpha < 0.95)).mean())
        fig.suptitle(f"{split}  {sid}  {slot_id}   area(>0.5)={frac:.3f}  "
                     f"soft-band frac={soft:.3f}", fontsize=10)
        fig.tight_layout()
        fig.savefig(viz_dir / f"semantic_{sid}.png", dpi=105)
        plt.close(fig)


if __name__ == "__main__":
    raise SystemExit(main())
