"""Mask v2 — Route 2: SAM3 semantic region + LUT (numpy), composited.

LUTs have NO Lightroom develop form — they are ALWAYS applied in numpy (trilinear). So there is no
"real LR render" to stay faithful to: compositing a numpy-LUT into a SAM3 semantic region is the
faithful path for LUTs (and the only way to use an external pixel mask, which LR develop can't take).

  vlemb -> a LUT whose grade suits the source ; SAM3 -> a real semantic region (sky / building /
  subject / ...) ; apply the LUT globally in numpy ; alpha-composite it into the SAM3 region.
  I_tar = source*(1-m) + lut_render*m.  C_GT = the SAM3 soft mask.

SAM3 masks are read from the precomputed cache (mask_cache.CachedMasker; ~55% of b_quality=3 sources
are cached). Cache misses need a SAM3 run in the monetgpt_sam3 env (deferred to the full-build
precompute); the pilot uses cached sources only.

CLI:  python -m construct.mask_sam3 verify [--n 8]
"""
from __future__ import annotations

import argparse
import os
import uuid

import numpy as np
from PIL import Image, ImageFile

from dataset_build.source_qa import config, db
from dataset_build.mask_cache import path_key
from . import render

ImageFile.LOAD_TRUNCATED_IMAGES = True
_CACHE = "/home/bc/data/datasets/vera_directionA_1M/sam3_cache"
_SKIP = {"background", "foreground", "image", "photo", "picture"}   # non-semantic / whole-frame
_CN = {"sky": "天空", "building": "建筑", "buildings": "建筑", "water": "水面", "sea": "海面",
       "person": "人物", "people": "人物", "subject": "主体", "face": "面部", "tree": "树木",
       "trees": "树木", "grass": "草地", "mountain": "山", "mountains": "山", "cloud": "云",
       "clouds": "云", "flower": "花", "flowers": "花", "car": "车", "road": "路", "snow": "雪",
       "rock": "岩石", "rocks": "岩石", "plant": "植物", "plants": "植物", "wall": "墙",
       "window": "窗", "hair": "头发", "skin": "皮肤", "foreground": "前景", "river": "河",
       "lake": "湖", "forest": "森林", "sand": "沙", "beach": "沙滩", "sunset": "夕阳"}


def _cache_dir(source: str) -> str:
    return os.path.join(_CACHE, path_key(source))


def has_cache(source: str) -> bool:
    d = _cache_dir(source)
    return os.path.isdir(d) and any(f.endswith(".png") for f in os.listdir(d))


def _load_mask(png: str):
    m = np.asarray(Image.open(png).convert("L"), "float32") / 255.0
    return m


def candidate_concepts(source: str, k: int = 4) -> list:
    """Salient SAM3 regions for a source, read directly from the cached per-concept PNGs (no
    regions.json in this cache). Returns [(slug, png_path, area)] mid-sized, biggest first."""
    d = _cache_dir(source)
    if not os.path.isdir(d):
        return []
    out = []
    for f in os.listdir(d):
        if not f.endswith(".png"):
            continue
        slug = f[:-4]
        if slug.lower() in _SKIP:
            continue
        try:
            area = float((_load_mask(os.path.join(d, f)) > 0.5).mean())
        except Exception:  # noqa: BLE001
            continue
        if 0.04 <= area <= 0.85:
            out.append((slug, os.path.join(d, f), area))
    out.sort(key=lambda x: -x[2])
    return out[:k]


def make_lut_local_sample(source: str, lut_feat: dict, concept: str, mask_png: str,
                          out_dir: str) -> dict | None:
    """Apply the LUT (numpy) and composite it into the SAM3 region for `concept`. Saves I_tar + C_GT."""
    res = render.render_preset(lut_feat["path"], "lut", lut_feat.get("fmt"), source)
    if not res.get("ok"):
        return None
    lut_im = Image.open(res["after_path"]).convert("RGB")
    src_im = Image.open(source).convert("RGB").resize(lut_im.size)
    mimg = Image.open(mask_png).convert("L").resize(lut_im.size)
    m = np.asarray(mimg, "float32") / 255.0
    if float(m.max()) < 0.1:
        return None
    a = m[..., None].astype("float32")
    comp = np.asarray(src_im, "float32") * (1 - a) + np.asarray(lut_im, "float32") * a
    os.makedirs(out_dir, exist_ok=True)
    muid = uuid.uuid4().hex[:16]
    out_path = os.path.join(config.RENDER_STAGE, f"sam3lut_{muid}.jpg")
    Image.fromarray(np.clip(comp, 0, 255).astype("uint8")).save(out_path, "JPEG", quality=95)
    cgt_path = os.path.join(out_dir, f"cgt_{muid}.png")
    Image.fromarray((m * 255).astype("uint8")).save(cgt_path)
    return {"after_path": out_path, "cgt_path": cgt_path, "mask_unit_id": muid,
            "concept": concept, "concept_cn": _CN.get(concept.lower(), concept),
            "area": float((m > 0.5).mean())}


def verify(n: int = 8) -> None:
    from .bank import PresetBank
    bank = PresetBank.load("/home/bc/data/datasets/vera_directionA_1M/preset_bank_full")
    luts = [f for f in bank.feats if f["kind"] == "lut"]
    conn = db.connect()
    srcs = [r["path"] for r in conn.execute(
        "SELECT path FROM assets WHERE asset_type='image' AND b_quality=3 AND dup_of IS NULL "
        "ORDER BY asset_id LIMIT 400").fetchall()]
    conn.close()
    srcs = [s for s in srcs if os.path.exists(s) and has_cache(s)]
    print(f"{len(srcs)} cached sources; {len(luts)} LUTs in bank")
    import random
    rng = random.Random(0)
    ok = 0
    for i in range(n):
        src = srcs[rng.randrange(len(srcs))]
        cons = candidate_concepts(src, 4)
        if not cons:
            print(f"[{i+1}] no salient concept"); continue
        slug, png, area = cons[0]; lut = luts[rng.randrange(len(luts))]
        s = make_lut_local_sample(src, lut, slug, png, config.RENDER_STAGE)
        if not s:
            print(f"[{i+1}] {slug}: render/mask fail"); continue
        # localization: LUT change should follow the SAM3 mask
        src_im = np.asarray(Image.open(src).convert("RGB").resize((256, 256)), "float32")
        out_im = np.asarray(Image.open(s["after_path"]).convert("RGB").resize((256, 256)), "float32")
        diff = np.abs(out_im - src_im).mean(-1)
        m = np.asarray(Image.open(png).convert("L").resize((256, 256)), "float32") / 255.0
        din = diff[m > 0.5].mean() if (m > 0.5).any() else 0
        dout = diff[m <= 0.5].mean() if (m <= 0.5).any() else 1
        ratio = (din + 1e-3) / (dout + 1e-3)
        passed = ratio > 3.0 and din > 3.0
        ok += passed
        print(f"[{i+1}] {s['concept_cn']:6s} area={s['area']:.2f} in/out={ratio:4.1f} in={din:5.1f} "
              f"{'OK' if passed else 'WEAK'}")
    print(f"\nSAM3+LUT round-trip: {ok}/{n} (LUT confined to semantic region)")


def main() -> None:
    ap = argparse.ArgumentParser()
    v = ap.add_subparsers(dest="cmd", required=True).add_parser("verify")
    v.add_argument("--n", type=int, default=8)
    a = ap.parse_args()
    if a.cmd == "verify":
        verify(a.n)


if __name__ == "__main__":
    main()
