"""INF-2 capacity-ladder generator (L0-L7).

Levels (PLAN §3 second-rung table; L5/L6/L7 designs documented in NOTES.md §5
before implementation):

  L0 global            m = 1 everywhere
  L1 binary semantic   cgt >= 0.5, hard edge
  L2 same data recipe as L1 (independent seed stream; the L1/L2 difference is
     the s source at experiment time, not the data)
  L3 soft matte        raw cgt soft mask (+ gaussian feather tier)
  L4 geometric         radial/linear/elliptical/vignette, feather+area tiers
  L5 mismatch (negative control): rendered with a radial mask, but the
     *provided* GT mask is the (mismatched) semantic cgt; IoU < 0.3 enforced
  L6 two overlapping soft masks, two different transforms applied in sequence
  L7 boundary cutting through a color-homogeneous region (linear boundary
     pinned through a low-variance point; expected FAIL unless (x,y) leaks)

Usage:
  python -m tools.construct.generate build-catalog --out CACHE/source_catalog.jsonl
  python -m tools.construct.generate generate --catalog ... --out-dir ... \
      --split train --per-level 200 --seed 20260802 --viz-dir ...
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw
from scipy.ndimage import gaussian_filter, uniform_filter

from . import masks as M
from . import transforms as T

GENERATOR_VERSION = "t4construct_v2"
LEVELS = ["L0", "L1", "L2", "L3", "L4", "L5", "L6", "L7"]
LEVEL_IDS = {lv: k for k, lv in enumerate(LEVELS)}

COMBOS24 = [(cls, t) for t in range(4) for cls in T.ALL_CLASSES]      # 6x4
COMBOS20 = [(cls, t) for t in range(4) for cls in T.STD_CLASSES]      # 5x4
PAIRS20 = [(a, b) for a in T.STD_CLASSES for b in T.STD_CLASSES if a != b]

F_SOFT = [8, 24]
A_MID = [0.15, 0.40]
L7_HOMOG_STD_MAX = 0.06
L5_IOU_MAX = 0.30
L6_IOU_RANGE = (0.10, 0.55)


# ----------------------------------------------------------------------------
# helpers
# ----------------------------------------------------------------------------

def git_commit() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=Path(__file__).resolve().parents[2],
            text=True).strip()
    except Exception:
        return "unknown"


def to_u8(arr: np.ndarray) -> np.ndarray:
    return np.clip(np.round(arr * 255.0), 0, 255).astype(np.uint8)


def save_rgb(arr: np.ndarray, path: Path, fmt: str):
    img = Image.fromarray(to_u8(arr))
    if fmt == "jpg":
        img.save(path, quality=95)
    else:
        img.save(path)


def save_mask(mask: np.ndarray, path: Path):
    Image.fromarray(to_u8(mask), mode="L").save(path)


def binary_iou(a: np.ndarray, b: np.ndarray, thr: float = 0.5) -> float:
    ab, bb = a >= thr, b >= thr
    inter = float(np.logical_and(ab, bb).sum())
    union = float(np.logical_or(ab, bb).sum())
    return inter / union if union > 0 else 0.0


def find_homogeneous_point(img: np.ndarray, rng: np.random.Generator):
    """Lowest-local-std point (16px window @ 1/4 res) in the central 60%."""
    g = T.luma(img)
    q = g[::4, ::4]
    m1 = uniform_filter(q, size=16, mode="nearest")
    m2 = uniform_filter(q * q, size=16, mode="nearest")
    std = np.sqrt(np.maximum(m2 - m1 * m1, 0.0))
    h, w = std.shape
    my, mx = int(h * 0.2), int(w * 0.2)
    central = std[my:h - my, mx:w - mx]
    order = np.argsort(central, axis=None)[:32]
    pick = int(order[int(rng.integers(len(order)))])
    cy, cx = np.unravel_index(pick, central.shape)
    val = float(central[cy, cx])
    if val > L7_HOMOG_STD_MAX:
        return None
    return float((cx + mx) * 4), float((cy + my) * 4), val


def cross_boundary_color_dist(img: np.ndarray, p_xy, theta: float,
                              radius: int = 40) -> float:
    """|mean sRGB| distance between the two half-discs across the boundary."""
    x0, y0 = p_xy
    h, w = img.shape[:2]
    x1, x2 = max(0, int(x0 - radius)), min(w, int(x0 + radius + 1))
    y1, y2 = max(0, int(y0 - radius)), min(h, int(y0 + radius + 1))
    sub = img[y1:y2, x1:x2]
    X, Y = np.meshgrid(np.arange(x1, x2) - x0, np.arange(y1, y2) - y0)
    disc = X * X + Y * Y <= radius * radius
    side = (X * np.cos(theta) + Y * np.sin(theta)) > 0
    s1, s2 = disc & side, disc & ~side
    if s1.sum() < 32 or s2.sum() < 32:
        return float("inf")
    return float(np.linalg.norm(sub[s1].mean(axis=0) - sub[s2].mean(axis=0)))


# ----------------------------------------------------------------------------
# generation context
# ----------------------------------------------------------------------------

class GenContext:
    def __init__(self, bank: M.SemanticBank, split: str, master_seed: int):
        self.bank = bank
        self.split = split
        self.master_seed = master_seed
        self.sampler = M.GeomMaskSampler()
        self.sources = bank.sources(split)
        if not self.sources:
            raise RuntimeError(f"no sources for split={split}")
        # per-level deterministic shuffle so consecutive indices spread over
        # the whole pool (the sorted list clusters ppr10k_* first)
        self._perms = {
            lv: np.random.default_rng(
                np.random.SeedSequence([master_seed, LEVEL_IDS[lv], 0xA5A5])
            ).permutation(len(self.sources))
            for lv in LEVELS
        }

    def rng_for(self, level: str, i: int) -> np.random.Generator:
        return np.random.default_rng(
            np.random.SeedSequence([self.master_seed, LEVEL_IDS[level], i]))

    def pick_source(self, level: str, i: int, attempt: int) -> str:
        n = len(self.sources)
        return self.sources[int(self._perms[level][(i + attempt * 7919) % n])]

    def load_semantic(self, source_id: str, rng: np.random.Generator):
        """-> (row, img, cgt) from a semantic-slot candidate, or None."""
        rows = self.bank.semantic_rows_of(source_id)
        if not rows:
            return None
        start = int(rng.integers(len(rows)))
        img = None
        for k in range(len(rows)):
            row = rows[(start + k) % len(rows)]
            if img is None:
                img = self.bank.load_image(row)
            cgt = self.bank.load_cgt(row, img.shape[:2])
            if self.bank.cgt_admissible(cgt):
                return row, img, cgt
        return None

    def load_image_only(self, source_id: str, rng: np.random.Generator):
        rows = self.bank.rows_of(source_id)
        row = rows[int(rng.integers(len(rows)))]
        return row, self.bank.load_image(row)


# ----------------------------------------------------------------------------
# per-level generators: return (record_extras, arrays) or None to retry source
# arrays keys: in / out / mask (+ maskrender | maska+maskb)
# ----------------------------------------------------------------------------

def _apply_single(img, mask, cls, tier, rng):
    spec = T.sample_transform(cls, tier, rng)
    out = T.composite(img, mask, T.apply_transform(img, spec))
    return spec, out


def gen_one(ctx: GenContext, level: str, i: int, max_source_attempts: int = 60):
    rng = ctx.rng_for(level, i)
    for attempt in range(max_source_attempts):
        source_id = ctx.pick_source(level, i, attempt)
        res = _gen_with_source(ctx, level, i, rng, source_id)
        if res is not None:
            record, arrays = res
            record["source_attempts"] = attempt + 1
            return record, arrays
        # burn a draw so retries do not replay the identical stream
        rng = ctx.rng_for(level, i)
        for _ in range(attempt + 1):
            rng.random()
    raise RuntimeError(f"{level}[{i}]: no usable source after {max_source_attempts} attempts")


def _gen_with_source(ctx, level, i, rng, source_id):
    bank, sampler = ctx.bank, ctx.sampler

    if level in ("L1", "L2", "L3", "L5"):
        got = ctx.load_semantic(source_id, rng)
        if got is None:
            return None
        row, img, cgt = got
    else:
        row, img = ctx.load_image_only(source_id, rng)
        cgt = None
    h, w = img.shape[:2]

    rec: dict = {
        "level": level, "source_id": source_id,
        "provenance": {"build": row["build"], "batch": row["batch"],
                       "sample_id": row["sample_id"],
                       "slot_id": row.get("slot_id")},
        "img_hw": [h, w],
    }

    if level == "L0":
        cls, tier = COMBOS24[i % 24]
        mask = np.ones((h, w), dtype=np.float32)
        spec, out = _apply_single(img, mask, cls, tier, rng)
        rec.update(mask_kind="global_ones", feather_px=None, feather_kind=None,
                   alpha_target=None, alpha_achieved=1.0, transform=spec)
        return rec, {"in": img, "out": out, "mask": mask}

    if level in ("L1", "L2"):
        assert cgt is not None
        cls, tier = COMBOS24[i % 24]
        mask = (cgt >= 0.5).astype(np.float32)
        if not (M.CGT_VALID_RANGE[0] <= float(mask.mean()) <= M.CGT_VALID_RANGE[1]):
            return None
        spec, out = _apply_single(img, mask, cls, tier, rng)
        rec.update(mask_kind="semantic_binary", feather_px=0, feather_kind="none",
                   alpha_target=None, alpha_achieved=float(mask.mean()),
                   transform=spec, cgt_mean=float(cgt.mean()))
        return rec, {"in": img, "out": out, "mask": mask}

    if level == "L3":
        assert cgt is not None
        cls, tier = COMBOS24[i % 24]
        sigma = M.FEATHERS[i % 4]
        mask = cgt if sigma == 0 else np.clip(
            gaussian_filter(cgt, sigma=sigma), 0.0, 1.0).astype(np.float32)
        spec, out = _apply_single(img, mask, cls, tier, rng)
        rec.update(mask_kind="semantic_soft", feather_px=sigma,
                   feather_kind="gaussian_sigma",
                   alpha_target=None, alpha_achieved=float(mask.mean()),
                   transform=spec, cgt_mean=float(cgt.mean()))
        return rec, {"in": img, "out": out, "mask": mask}

    if level == "L4":
        family = M.GEOM_FAMILIES[i % 4]
        feather = M.FEATHERS[(i // 4) % 4]
        alpha = M.AREAS[(i // 16) % 4]
        cls, tier = COMBOS24[i % 24]
        gm = sampler.sample(family, h, w, rng, alpha, feather)
        if gm is None:
            return None
        spec, out = _apply_single(img, gm.mask, cls, tier, rng)
        rec.update(mask_kind=f"geom_{family}", geom_params=gm.params,
                   feather_px=feather, feather_kind="smoothstep_halfwidth",
                   alpha_target=alpha,
                   alpha_achieved=gm.alpha_achieved, transform=spec)
        return rec, {"in": img, "out": out, "mask": gm.mask}

    if level == "L5":
        assert cgt is not None
        feather = M.FEATHERS[i % 4]
        alpha = M.AREAS[(i // 4) % 4]
        cls, tier = COMBOS20[i % 20]
        best = None
        for _ in range(6):
            gm = sampler.sample("radial", h, w, rng, alpha, feather)
            if gm is None:
                continue
            iou = binary_iou(gm.mask, cgt)
            if best is None or iou < best[1]:
                best = (gm, iou)
            if iou < L5_IOU_MAX:
                break
        if best is None:
            return None
        gm, iou = best
        spec, out = _apply_single(img, gm.mask, cls, tier, rng)
        rec.update(mask_kind="mismatch_semantic_gt", geom_params=gm.params,
                   feather_px=feather, feather_kind="smoothstep_halfwidth",
                   alpha_target=alpha,
                   alpha_achieved=float(cgt.mean()),
                   render_alpha_achieved=gm.alpha_achieved,
                   mismatch_iou=iou, iou_constraint_met=iou < L5_IOU_MAX,
                   transform=spec, expected="FAIL(negative control)")
        return rec, {"in": img, "out": out, "mask": cgt, "maskrender": gm.mask}

    if level == "L6":
        fa, fb = F_SOFT[i % 2], F_SOFT[(i // 2) % 2]
        aa, ab = A_MID[(i // 4) % 2], A_MID[(i // 8) % 2]
        cls_a, cls_b = PAIRS20[i % 20]
        ma = sampler.sample("radial", h, w, rng, aa, fa)
        if ma is None:
            return None
        best = None
        for _ in range(8):
            mb = sampler.sample("elliptical", h, w, rng, ab, fb)
            if mb is None:
                continue
            iou = binary_iou(ma.mask, mb.mask)
            dist = 0.0 if L6_IOU_RANGE[0] <= iou <= L6_IOU_RANGE[1] else \
                min(abs(iou - L6_IOU_RANGE[0]), abs(iou - L6_IOU_RANGE[1]))
            if best is None or dist < best[2]:
                best = (mb, iou, dist)
            if dist == 0.0:
                break
        if best is None:
            return None
        mb, iou, dist = best
        spec_a = T.sample_transform(cls_a, int(rng.integers(4)), rng)
        spec_b = T.sample_transform(cls_b, int(rng.integers(4)), rng)
        o1 = T.composite(img, ma.mask, T.apply_transform(img, spec_a))
        out = T.composite(o1, mb.mask, T.apply_transform(o1, spec_b))
        union = np.maximum(ma.mask, mb.mask)
        rec.update(mask_kind="dual_overlap_soft",
                   geom_params={"a": ma.params, "b": mb.params},
                   feather_px=[fa, fb], feather_kind="smoothstep_halfwidth",
                   alpha_target=[aa, ab],
                   alpha_achieved=[ma.alpha_achieved, mb.alpha_achieved],
                   overlap_iou=iou, iou_constraint_met=dist == 0.0,
                   transform=[spec_a, spec_b],
                   compose_order="a_then_b",
                   expected="FAIL(rank limit)")
        return rec, {"in": img, "out": out, "mask": union,
                     "maska": ma.mask, "maskb": mb.mask}

    if level == "L7":
        feather = [0, 2][i % 2]
        cls, tier = COMBOS20[i % 20]
        pt = find_homogeneous_point(img, rng)
        if pt is None:
            return None
        x0, y0, homog_std = pt
        thetas = rng.uniform(0.0, 2.0 * np.pi, size=8)
        dists = [cross_boundary_color_dist(img, (x0, y0), th) for th in thetas]
        j = int(np.argmin(dists))
        theta, cross_dist = float(thetas[j]), float(dists[j])
        t0 = x0 * np.cos(theta) + y0 * np.sin(theta)
        gm = sampler.sample("linear", h, w, rng, 0.0, feather,
                            force_theta=theta, force_t0=float(t0))
        if gm is None or not (0.03 <= gm.alpha_achieved <= 0.97):
            return None
        spec, out = _apply_single(img, gm.mask, cls, tier, rng)
        rec.update(mask_kind="linear_through_flat_region", geom_params=gm.params,
                   feather_px=feather, feather_kind="smoothstep_halfwidth",
                   alpha_target=None,
                   alpha_achieved=gm.alpha_achieved,
                   homog_point=[x0, y0], homog_window_std=homog_std,
                   cross_boundary_rgb_dist=cross_dist,
                   transform=spec, expected="FAIL(pass=xy leak bug)")
        return rec, {"in": img, "out": out, "mask": gm.mask}

    raise ValueError(f"unknown level {level}")


# ----------------------------------------------------------------------------
# batch generation + viz
# ----------------------------------------------------------------------------

def generate_split(catalog: str, out_dir: str, split: str, per_level: int,
                   seed: int, levels: list[str], img_format: str,
                   viz_dir: str | None, root: str = M.DEFAULT_ROOT,
                   split_table_path: str | None = None) -> dict:
    # default: SemanticBank/make_splitter load the frozen T1 side-table
    # (tools/data_splits/splits.sqlite3); --split-table only overrides the file
    bank = M.SemanticBank(catalog, root=root,
                          split_table_path=split_table_path)
    ctx = GenContext(bank, split, seed)
    out_root = Path(out_dir) / split
    out_root.mkdir(parents=True, exist_ok=True)
    manifest_path = out_root / "manifest.jsonl"
    commit = git_commit()
    stats: dict = {"split": split, "per_level": per_level, "seed": seed,
                   "levels": {}, "n_sources": len(ctx.sources)}
    ext = "jpg" if img_format == "jpg" else "png"

    with open(manifest_path, "w", encoding="utf-8") as mf:
        for level in levels:
            lv_dir = out_root / level
            lv_dir.mkdir(parents=True, exist_ok=True)
            t0 = time.time()
            lv_stats: dict[str, float] = {
                "n": 0, "area_hits": 0, "area_targeted": 0,
                "source_attempts": 0, "iou_ok": 0, "iou_all": 0}
            for i in range(per_level):
                rec, arrays = gen_one(ctx, level, i)
                uid = f"{level}_{split}_{i:04d}"
                files = {}
                for key, arr in arrays.items():
                    if key in ("in", "out"):
                        fn = f"{uid}_{key}.{ext}"
                        save_rgb(arr, lv_dir / fn, img_format)
                    else:
                        fn = f"{uid}_{key}.png"
                        save_mask(arr, lv_dir / fn)
                    files[key] = f"{level}/{fn}"
                rec.update(
                    schema=GENERATOR_VERSION, split=split, index=i, uid=uid,
                    master_seed=seed, seed_ints=[seed, LEVEL_IDS[level], i],
                    split_rule=bank.split_rule,
                    img_format=img_format, files=files, git_commit=commit,
                )
                mf.write(json.dumps(rec, ensure_ascii=False) + "\n")
                lv_stats["n"] += 1
                lv_stats["source_attempts"] += rec.get("source_attempts", 1)
                at = rec.get("alpha_target")
                if isinstance(at, (int, float)):
                    # for L5 the area target applies to the *render* mask
                    ach = rec.get("render_alpha_achieved", rec["alpha_achieved"])
                    lv_stats["area_targeted"] += 1
                    if abs(ach - at) <= M.AREA_TOL:
                        lv_stats["area_hits"] += 1
                if "iou_constraint_met" in rec:
                    lv_stats["iou_all"] += 1
                    lv_stats["iou_ok"] += int(bool(rec["iou_constraint_met"]))
            lv_stats["seconds"] = round(time.time() - t0, 1)
            stats["levels"][level] = lv_stats
            print(f"[{split}/{level}] {per_level} samples in {lv_stats['seconds']}s",
                  flush=True)

    with open(out_root / "gen_stats.json", "w", encoding="utf-8") as fh:
        json.dump(stats, fh, ensure_ascii=False, indent=2)

    if viz_dir:
        make_grids(manifest_path, out_root, Path(viz_dir), split)
    return stats


def make_grids(manifest_path: Path, data_root: Path, viz_dir: Path, split: str,
               thumb_w: int = 340):
    viz_dir.mkdir(parents=True, exist_ok=True)
    rows_by_level: dict[str, list[dict]] = {}
    with open(manifest_path, "r", encoding="utf-8") as fh:
        for line in fh:
            r = json.loads(line)
            rows_by_level.setdefault(r["level"], []).append(r)
    for level, rows in rows_by_level.items():
        picks = [rows[0], rows[len(rows) // 2], rows[-1]]
        tiles, labels = [], []
        for r in picks:
            trip = []
            for key in ("in", "mask", "out"):
                img = Image.open(data_root / r["files"][key]).convert("RGB")
                s = thumb_w / img.width
                trip.append(img.resize((thumb_w, max(1, round(img.height * s))),
                                       Image.Resampling.BILINEAR))
            tiles.append(trip)
            tr = r["transform"]
            tdesc = (f"{tr['class']}@t{tr['tier']}" if isinstance(tr, dict)
                     else "+".join(f"{t['class']}@t{t['tier']}" for t in tr))
            feather = r.get("feather_px")
            a = r["alpha_achieved"]
            adesc = ("/".join(f"{x:.2f}" for x in a) if isinstance(a, list)
                     else f"{a:.2f}")
            labels.append(f"{r['uid']}  {tdesc}  a={adesc} f={feather}")
        pad, header = 6, 22
        th = max(t.height for trip in tiles for t in trip)
        gw = 3 * thumb_w + 4 * pad
        gh = header + 3 * (th + header + pad) + pad
        grid = Image.new("RGB", (gw, gh), (24, 24, 24))
        draw = ImageDraw.Draw(grid)
        for c, name in enumerate(["input", "mask", "target(GT)"]):
            draw.text((pad + c * (thumb_w + pad) + 4, 4), name, fill=(230, 230, 230))
        y = header
        for r_i, trip in enumerate(tiles):
            draw.text((pad + 4, y + 2), labels[r_i], fill=(180, 220, 180))
            y += header
            for c, img in enumerate(trip):
                grid.paste(img, (pad + c * (thumb_w + pad), y))
            y += th + pad
        out = viz_dir / f"{level}_{split}_grid.png"
        grid.save(out)
        print(f"grid -> {out}", flush=True)


# ----------------------------------------------------------------------------
# CLI
# ----------------------------------------------------------------------------

def main(argv=None):
    ap = argparse.ArgumentParser(prog="tools.construct.generate")
    sub = ap.add_subparsers(dest="cmd", required=True)

    ap_cat = sub.add_parser("build-catalog", help="scan l-series shards -> source catalog")
    ap_cat.add_argument("--out", required=True)
    ap_cat.add_argument("--root", default=M.DEFAULT_ROOT)
    ap_cat.add_argument("--builds", nargs="*", default=None)
    ap_cat.add_argument("--batches-per-build", type=int, default=4)

    ap_gen = sub.add_parser("generate", help="generate ladder samples")
    ap_gen.add_argument("--catalog", required=True)
    ap_gen.add_argument("--out-dir", required=True)
    ap_gen.add_argument("--split", choices=["train", "val", "test"], default="train")
    ap_gen.add_argument("--per-level", type=int, default=200)
    ap_gen.add_argument("--seed", type=int, default=20260802)
    ap_gen.add_argument("--levels", nargs="*", default=LEVELS)
    ap_gen.add_argument("--img-format", choices=["jpg", "png"], default="jpg")
    ap_gen.add_argument("--viz-dir", default=None)
    ap_gen.add_argument("--root", default=M.DEFAULT_ROOT)
    ap_gen.add_argument("--split-table", default=None,
                        help="override split side-table file (sqlite3/csv/jsonl); "
                             "default: frozen T1 tools/data_splits/splits.sqlite3")

    args = ap.parse_args(argv)
    if args.cmd == "build-catalog":
        stats = M.build_catalog(args.out, root=args.root, builds=args.builds,
                                batches_per_build=args.batches_per_build)
        print(json.dumps(stats, ensure_ascii=False, indent=2))
    elif args.cmd == "generate":
        stats = generate_split(args.catalog, args.out_dir, args.split,
                               args.per_level, args.seed, args.levels,
                               args.img_format, args.viz_dir, root=args.root,
                               split_table_path=args.split_table)
        print(json.dumps(stats, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
