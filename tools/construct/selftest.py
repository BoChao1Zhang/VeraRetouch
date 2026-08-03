"""Runnable self-check for the INF-2 construct generator (real-data smoke).

Checks (NOTES.md §8; split checks reworked in wave-1.5 for review B3):
  1. S-split: default splitter is backed by the frozen T1 side-table
     (spot-check vs direct sqlite queries); deprecated inline fallback
     REFUSES to start while the side-table exists; distribution sanity
  2. idx.jsonl ranged extraction vs recorded sha256 (real shard, 12 members)
  3. RGB<->HSV vs colorsys; sRGB<->linear round trip
  4. composite identity: O==I where m==0; O==T1(I) where m==1 (hard mask)
  5. geometric area solver: families x alpha x feather within +-1%
  6. hard-control pwl is programmatically non-monotone (200 draws)
  7. manifest replay: same seed -> bitwise identical arrays
  8. L5 / L6 / L7 constraint fields present and in range on real samples
  9. (--manifests) every manifest source_id's split == T1 side-table entry
     (hard regression gate, zero tolerance)

Usage:
  python -m tools.construct.selftest --catalog <catalog.jsonl> \
      [--manifests sanity/train/manifest.jsonl sanity/val/manifest.jsonl]
"""

from __future__ import annotations

import argparse
import colorsys
import hashlib
import json
import sys
from collections import Counter
from pathlib import Path

import numpy as np

import sqlite3

from . import masks as M
from . import transforms as T
from .generate import GenContext, gen_one, binary_iou
from .splits import DEFAULT_SPLIT_DB, make_splitter

PASS, FAIL = "PASS", "FAIL"
RESULTS: list[tuple[str, str, str]] = []


def check(name: str, ok: bool, detail: str = ""):
    RESULTS.append((name, PASS if ok else FAIL, detail))
    print(f"[{PASS if ok else FAIL}] {name} {detail}", flush=True)


def t1_split(catalog: Path):
    # 1. default splitter must be backed by the frozen T1 side-table;
    #    spot-check 200 sampled rows against direct sqlite queries
    splitter, rule = make_splitter()
    con = sqlite3.connect(f"file:{DEFAULT_SPLIT_DB}?mode=ro", uri=True)
    rows = con.execute(
        "SELECT source_id, split FROM sources ORDER BY source_id").fetchall()
    con.close()
    rng = np.random.default_rng(20260803)
    picks = rng.choice(len(rows), size=min(200, len(rows)), replace=False)
    n_ok = sum(int(splitter(rows[int(p)][0]) == rows[int(p)][1]) for p in picks)
    check("split.table_backed",
          rule.startswith("t1_side_table") and n_ok == len(picks),
          f"rule={rule} spotcheck={n_ok}/{len(picks)}")
    # 2. deprecated inline fallback must refuse to start while the table exists
    refused, msg = False, ""
    try:
        make_splitter(use_inline=True)
    except RuntimeError as e:
        refused, msg = True, str(e)[:60]
    check("split.inline_guard_refuses", refused and DEFAULT_SPLIT_DB.exists(),
          msg)
    # 3. catalog sources through the table splitter: sane coverage
    sources = set()
    with open(catalog) as fh:
        for line in fh:
            sources.add(json.loads(line)["source_id"])
    dist = Counter(splitter(s) for s in sources)
    n = sum(dist.values())
    known = n - dist["unknown"]
    ok = known > 100 and 0.80 <= dist["train"] / known <= 0.97
    check("split.distribution", ok,
          f"n={n} train={dist['train']} val={dist['val']} "
          f"test={dist['test']} unknown={dist['unknown']}")
    return dist


def t9_manifest_vs_table(manifests: list[str]):
    """Hard gate (review B3): every manifest row's source split must match the
    T1 side-table AND equal the manifest's own split tag. Zero tolerance."""
    splitter, rule = make_splitter()
    ok_all = True
    details = []
    for mp in manifests:
        n = bad = 0
        with open(mp, "r", encoding="utf-8") as fh:
            for line in fh:
                r = json.loads(line)
                n += 1
                if splitter(r["source_id"]) != r["split"]:
                    bad += 1
        ok_all &= (bad == 0 and n > 0)
        details.append(f"{Path(mp).parent.name}:{n - bad}/{n}")
    check("split.manifest_vs_table", ok_all,
          f"rule={rule} " + " ".join(details))


def t2_ranged_read(catalog: Path, root: Path):
    rows = [json.loads(l) for l in open(catalog)][:200]
    rng = np.random.default_rng(0)
    picks = rng.choice(len(rows), size=min(4, len(rows)), replace=False)
    n_ok = n_all = 0
    for p in picks:
        row = rows[int(p)]
        bdir = root / row["build"] / row["batch"]
        idx = sorted((bdir / "indexes").glob("shard-*.idx.jsonl"))[0]
        lines = [json.loads(l) for l in open(idx)][:3]
        for ir in lines:
            raw = M.read_member(bdir / "shards" / f"{ir['shard']}.tar",
                                ir["offset_data"], ir["length"])
            n_all += 1
            n_ok += int(hashlib.sha256(raw).hexdigest() == ir["sha256"])
    check("shards.ranged_read_sha256", n_ok == n_all and n_all >= 12,
          f"{n_ok}/{n_all}")


def t3_colorspaces():
    rng = np.random.default_rng(1)
    pts = rng.random((2000, 3))
    ours = T.rgb_to_hsv(pts.reshape(1, -1, 3)).reshape(-1, 3)
    ref = np.array([colorsys.rgb_to_hsv(*p) for p in pts])
    err_hsv = float(np.abs(ours - ref).max())
    back = T.hsv_to_rgb(ours.reshape(1, -1, 3)).reshape(-1, 3)
    err_rt = float(np.abs(back - pts).max())
    check("color.hsv_vs_colorsys", err_hsv < 1e-6 and err_rt < 1e-6,
          f"maxerr={err_hsv:.2e} roundtrip={err_rt:.2e}")
    x = rng.random((64, 64, 3))
    rt = T.linear_to_srgb(T.srgb_to_linear(x))
    check("color.srgb_roundtrip", float(np.abs(rt - x).max()) < 1e-5)


def t4_composite():
    rng = np.random.default_rng(2)
    img = rng.random((96, 128, 3)).astype(np.float32)
    mask = np.zeros((96, 128), dtype=np.float32)
    mask[20:60, 30:90] = 1.0
    spec = T.sample_transform("gamma", 2, rng)
    edited = T.apply_transform(img, spec)
    out = T.composite(img, mask, edited)
    ok0 = bool(np.array_equal(out[mask == 0], img[mask == 0]))
    ok1 = bool(np.array_equal(out[mask == 1], edited[mask == 1]))
    check("composite.identity_outside", ok0)
    check("composite.exact_inside", ok1)


def t5_area_solver():
    sampler = M.GeomMaskSampler()
    rng = np.random.default_rng(3)
    h, w = 683, 1024
    worst, n_fail = 0.0, 0
    for fam in M.GEOM_FAMILIES:
        for alpha in M.AREAS:
            for f in M.FEATHERS:
                gm = sampler.sample(fam, h, w, rng, alpha, f)
                if gm is None:
                    n_fail += 1
                    continue
                worst = max(worst, abs(gm.alpha_achieved - alpha))
    check("geom.area_within_1pct", n_fail == 0 and worst <= M.AREA_TOL,
          f"worst={worst:.4f} fails={n_fail}/64")


def t6_pwl():
    rng = np.random.default_rng(4)
    bad = 0
    for _ in range(200):
        amp = T.TIERS["hardpwl"][int(rng.integers(4))]
        xs, ys = T.make_hard_pwl(rng, amp)
        if bool(np.all(np.diff(ys) >= 0)):
            bad += 1
    check("hardpwl.non_monotone_200", bad == 0, f"monotone={bad}")


def t7_replay(catalog: Path, root: Path):
    bank1 = M.SemanticBank(catalog, root=root)
    bank2 = M.SemanticBank(catalog, root=root)
    ok_all, details = True, []
    for level, i in [("L4", 3), ("L1", 0), ("L6", 1)]:
        r1, a1 = gen_one(GenContext(bank1, "train", 123), level, i)
        r2, a2 = gen_one(GenContext(bank2, "train", 123), level, i)
        same = set(a1) == set(a2) and all(np.array_equal(a1[k], a2[k]) for k in a1)
        same = same and json.dumps(r1, sort_keys=True) == json.dumps(r2, sort_keys=True)
        ok_all &= same
        details.append(f"{level}[{i}]={'ok' if same else 'MISMATCH'}")
    check("replay.bitwise", ok_all, " ".join(details))


def t8_negative_controls(catalog: Path, root: Path):
    bank = M.SemanticBank(catalog, root=root)
    ctx = GenContext(bank, "train", 321)
    r5, a5 = gen_one(ctx, "L5", 0)
    ok5 = r5["mismatch_iou"] < 0.5 and "maskrender" in a5 and \
        binary_iou(a5["mask"], a5["maskrender"]) == r5["mismatch_iou"]
    check("L5.mismatch_fields", ok5,
          f"iou={r5['mismatch_iou']:.3f} met={r5['iou_constraint_met']}")
    r6, a6 = gen_one(ctx, "L6", 0)
    ok6 = {"maska", "maskb"} <= set(a6) and 0.0 <= r6["overlap_iou"] <= 1.0 \
        and isinstance(r6["transform"], list) and len(r6["transform"]) == 2 \
        and r6["transform"][0]["class"] != r6["transform"][1]["class"]
    check("L6.dual_mask_fields", ok6,
          f"iou={r6['overlap_iou']:.3f} met={r6['iou_constraint_met']}")
    r7, _ = gen_one(ctx, "L7", 0)
    # comparison boundary: ordinary linear masks on the same image pool
    sampler = M.GeomMaskSampler()
    rng = np.random.default_rng(5)
    ref_dists = []
    for k in range(6):
        row, img = ctx.load_image_only(ctx.sources[k % len(ctx.sources)], rng)
        ih, iw = img.shape[:2]
        gm = sampler.sample("linear", ih, iw, rng, 0.4, 0)
        if gm is None:
            continue
        th = gm.params["theta"]
        t0 = gm.params["t0"]
        cx = (t0 - 0) / max(np.cos(th), 1e-9) if abs(np.cos(th)) > 0.5 else img.shape[1] / 2
        from .generate import cross_boundary_color_dist
        h, w = img.shape[:2]
        p = (min(max(cx, 60), w - 60), h / 2)
        d = cross_boundary_color_dist(img, p, th)
        if np.isfinite(d):
            ref_dists.append(d)
    ref_med = float(np.median(ref_dists)) if ref_dists else float("nan")
    ok7 = r7["homog_window_std"] <= 0.06 and np.isfinite(r7["cross_boundary_rgb_dist"])
    check("L7.homogeneity_fields", ok7,
          f"std={r7['homog_window_std']:.4f} cross={r7['cross_boundary_rgb_dist']:.4f} "
          f"ref_linear_median={ref_med:.4f}")


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--catalog", required=True)
    ap.add_argument("--root", default=M.DEFAULT_ROOT)
    ap.add_argument("--out-json", default=None)
    ap.add_argument("--manifests", nargs="*", default=None,
                    help="manifest.jsonl paths for the split hard gate")
    args = ap.parse_args(argv)
    catalog, root = Path(args.catalog), Path(args.root)

    t1_split(catalog)
    t2_ranged_read(catalog, root)
    t3_colorspaces()
    t4_composite()
    t5_area_solver()
    t6_pwl()
    t7_replay(catalog, root)
    t8_negative_controls(catalog, root)
    if args.manifests:
        t9_manifest_vs_table(args.manifests)

    n_fail = sum(1 for _, s, _ in RESULTS if s == FAIL)
    print(f"\nselftest: {len(RESULTS) - n_fail}/{len(RESULTS)} passed", flush=True)
    if args.out_json:
        with open(args.out_json, "w", encoding="utf-8") as fh:
            json.dump([{"name": n, "status": s, "detail": d} for n, s, d in RESULTS],
                      fh, ensure_ascii=False, indent=2)
    return 1 if n_fail else 0


if __name__ == "__main__":
    sys.exit(main())
