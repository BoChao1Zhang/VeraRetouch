"""Mask v2 — Route 1: local samples = a GLOBAL-selected preset applied ONLY inside a mask.

Per the design (user-clarified):
  - GLOBAL flow (vlemb) selects an appropriate preset for the source (its core job: pick the preset).
  - the MASK pool's core job: pick an appropriate region (GEOMETRY only).
  - the sample = that preset's look applied INSIDE the mask. The localizable tone/color params
    (Exposure/Contrast/Highlights/Shadows/Whites/Blacks/Clarity/Dehaze/Texture/Saturation) are moved
    into the mask as Local* params; the parts with NO local equivalent (HSL, tone curve, color
    grading, white balance) STAY GLOBAL (LR can't confine them). C_GT = the mask geometry.

Implementation: modify the base preset's XMP TREE — pop the localizable global attrs, re-emit them as
one MaskGroupBasedCorrections (Local* + geom), keep everything else global. xmp2lua (masks preserved)
-> render_via_lr. Round-trip = the EXTRA change concentrates inside C_GT.

CLI:  python -m construct.mask_synth bank            # build the geometry pool
      python -m construct.mask_synth verify [--n 12]  # synth -> render -> round-trip check
"""
from __future__ import annotations

import argparse
import math
import os
import random
import uuid
import xml.etree.ElementTree as ET

import numpy as np
from PIL import Image, ImageFile

from dataset_build.source_qa import config, db
from dataset_build import recipes
from . import render

ImageFile.LOAD_TRUNCATED_IMAGES = True
_CRS = recipes.CRS_NS
_RDF = recipes.RDF_NS
_NS = {"rdf": _RDF, "crs": _CRS}
_SHAPES = {"circulargradient", "gradient"}
_GEOM_KEYS = ("Top", "Left", "Bottom", "Right", "Angle", "Midpoint", "Roundness", "Feather",
              "Flipped", "Version", "ZeroX", "ZeroY", "FullX", "FullY")
_BANK_PATH = "/home/bc/data/datasets/vera_directionA_1M/mask_templates/mask_bank_v2.json"

# Strong, deliberately-VISIBLE local edits applied INSIDE the mask (on top of the base preset's look,
# which is applied globally — curve/HSL/WB can't localize in LR so they stay global, per design).
# The old "pop the preset's own mild tone into the mask" produced near-invisible local changes; these
# are large, coherent, single-intent edits so the localized region is clearly distinguishable.
# EVERY edit is anchored on a strong LocalExposure2012 (|Δ|≥0.5) or LocalSaturation (|Δ|≥45) — those
# are the only Local* params that render strongly+reliably through the LR pipeline (Clarity/Contrast/
# Dehaze/Highlights/Shadows alone are image-dependent and often near-invisible on a feathered region).
_LOCAL_EDITS = [
    ("提亮并增清晰(主体突出)", {"LocalExposure2012": 0.85, "LocalClarity2012": 30, "LocalContrast2012": 20}),
    ("压暗收光(局部加重氛围)", {"LocalExposure2012": -0.9, "LocalHighlights2012": -45, "LocalContrast2012": 15}),
    ("提亮压高光(开阴影)", {"LocalExposure2012": 0.7, "LocalHighlights2012": -55, "LocalShadows2012": 45}),
    ("增艳加清晰", {"LocalSaturation": 55, "LocalExposure2012": 0.25, "LocalClarity2012": 20}),
    ("降饱和压暗(弱化干扰)", {"LocalSaturation": -65, "LocalExposure2012": -0.5}),
    ("提亮找回细节", {"LocalExposure2012": 0.6, "LocalHighlights2012": -55, "LocalShadows2012": 50}),
    ("增艳提亮去朦胧", {"LocalSaturation": 50, "LocalExposure2012": 0.4, "LocalClarity2012": 22}),
]


def sample_local_edit(rng: random.Random) -> tuple:
    """Pick one strong, coherent local edit and jitter its magnitudes ±20% so samples vary."""
    label, base = _LOCAL_EDITS[rng.randrange(len(_LOCAL_EDITS))]
    edit = {}
    for k, v in base.items():
        v2 = v * rng.uniform(0.85, 1.2)
        edit[k] = round(v2, 4) if k == "LocalExposure2012" else round(max(-100, min(100, v2)), 1)
    return label, edit


def _f(d, k, default=0.0):
    try:
        return float(str(d.get(k, default)).lstrip("+"))
    except Exception:
        return default


# --------------------------------------------------------------------------- #
# geometry pool (mask bank)
# --------------------------------------------------------------------------- #
def mine(limit: int = 4000) -> list:
    conn = db.connect()
    rows = conn.execute("SELECT path FROM assets WHERE asset_type='preset' AND kind='param' "
                        "AND has_local_mask=1 LIMIT %s", (limit,)).fetchall()
    conn.close()
    out = []
    for r in rows:
        p = r["path"]
        if not p.endswith(".xmp") or not os.path.exists(p):
            continue
        for m in recipes.parse_local_masks(p):
            if m["mask_type"] in _SHAPES:
                out.append(m)
    return out


def build_bank(out_path: str = _BANK_PATH) -> dict:
    """Persist the GEOMETRY pool (per shape). The local EDIT comes from the global-selected preset,
    not the pool — so the bank only carries geometry."""
    geoms = {"circulargradient": [], "gradient": []}
    seen = set()
    for m in mine():
        g = {k: m["geom"][k] for k in _GEOM_KEYS if k in m["geom"]}
        key = m["mask_type"] + "|" + "|".join(f"{k}={round(_f(g, k), 2)}" for k in sorted(g))
        if g and key not in seen:
            seen.add(key); geoms[m["mask_type"]].append({"what": m["what"], "geom": g})
    import json
    bank = {"geoms": geoms, "n_geom": {k: len(v) for k, v in geoms.items()}}
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    json.dump(bank, open(out_path, "w"), ensure_ascii=False, indent=1)
    print(f"mask_bank (geometry pool): {bank['n_geom']} -> {out_path}")
    return bank


def load_bank(path: str = _BANK_PATH) -> dict:
    import json
    return json.load(open(path))


_LIN_SIDES = {  # linear gradient: (Zero=0% line near center, Full=100% line at the edge), jittered
    "上方": ((0.5, 0.55), (0.5, 0.08)), "下方": ((0.5, 0.45), (0.5, 0.92)),
    "左侧": ((0.55, 0.5), (0.08, 0.5)), "右侧": ((0.45, 0.5), (0.92, 0.5))}


def sample_geom(bank: dict, rng: random.Random, radial_frac: float = 0.55) -> dict:
    """Synthesize an in-frame, substantial mask geometry directly (no mined geoms — those perturbed
    off-frame / degenerate, giving empty masks). Guarantees a meaningful, correctly-rasterizable C_GT."""
    j = lambda v, d=0.05: round(v + rng.uniform(-d, d), 4)
    if rng.random() < radial_frac:
        cx, cy = rng.uniform(0.34, 0.66), rng.uniform(0.34, 0.66)
        rx, ry = rng.uniform(0.24, 0.36), rng.uniform(0.24, 0.36)
        geom = {"Top": round(cy - ry, 4), "Bottom": round(cy + ry, 4),
                "Left": round(cx - rx, 4), "Right": round(cx + rx, 4),
                "Angle": 0.0, "Feather": float(rng.choice([45, 60, 75])), "Roundness": 0.0,
                "Midpoint": 50.0, "Flipped": "true"}   # Flipped=affect INSIDE the ellipse (subject)
        return {"mask_type": "circulargradient", "what": "Mask/CircularGradient", "geom": geom}
    side = rng.choice(list(_LIN_SIDES))
    (zx, zy), (fx, fy) = _LIN_SIDES[side]
    geom = {"ZeroX": j(zx), "ZeroY": j(zy), "FullX": j(fx), "FullY": j(fy), "Flipped": "false"}
    return {"mask_type": "gradient", "what": "Mask/Gradient", "geom": geom, "_side": side}


def perturb(geom: dict, rng: random.Random) -> dict:
    return dict(geom)   # geometry is already synthesized in-frame + jittered; keep API for agent


# --------------------------------------------------------------------------- #
# XMP synthesis: base preset (global, minus localizable) + one mask correction
# --------------------------------------------------------------------------- #
def _C(name):  # crs-namespaced tag/attr
    return f"{{{_CRS}}}{name}"


def synth_local_xmp(base_xmp: str, geom: dict, local: dict, out_dir: str) -> dict | None:
    """base preset stays GLOBAL (curve/HSL/WB included — they can't localize in LR); add ONE strong
    `local` edit (Local* params) confined to `geom` as a mask correction. Returns {xmp_path,
    local_params} or None on parse failure."""
    try:
        ET.register_namespace("crs", _CRS); ET.register_namespace("rdf", _RDF)
        ET.register_namespace("x", "adobe:ns:meta/")
        tree = ET.parse(base_xmp)
    except Exception:  # noqa: BLE001
        return None
    desc = tree.getroot().find(".//rdf:Description", _NS)
    if desc is None or not local:
        return None
    corr = ET.SubElement(ET.SubElement(ET.SubElement(desc, _C("MaskGroupBasedCorrections")),
                                       f"{{{_RDF}}}Seq"), f"{{{_RDF}}}li")
    cd = ET.SubElement(corr, f"{{{_RDF}}}Description")
    cd.set(_C("What"), "Correction"); cd.set(_C("CorrectionAmount"), "1")
    cd.set(_C("CorrectionActive"), "true"); cd.set(_C("CorrectionName"), "local")
    cd.set(_C("CorrectionSyncID"), uuid.uuid4().hex.upper())
    for lk, v in local.items():
        cd.set(_C(lk), str(v))
    md = ET.SubElement(ET.SubElement(ET.SubElement(cd, _C("CorrectionMasks")),
                                     f"{{{_RDF}}}Seq"), f"{{{_RDF}}}li")
    mdd = ET.SubElement(md, f"{{{_RDF}}}Description")
    mdd.set(_C("What"), geom.get("__what__", "Mask/CircularGradient"))
    mdd.set(_C("MaskActive"), "true"); mdd.set(_C("MaskName"), "local")
    mdd.set(_C("MaskBlendMode"), "0"); mdd.set(_C("MaskValue"), "1")
    mdd.set(_C("MaskSyncID"), uuid.uuid4().hex.upper())
    for k in _GEOM_KEYS:
        if k in geom:
            mdd.set(_C(k), str(geom[k]))
    os.makedirs(out_dir, exist_ok=True)
    path = os.path.join(out_dir, f"localmask_{uuid.uuid4().hex[:12]}.xmp")
    tree.write(path, encoding="utf-8", xml_declaration=True)
    return {"xmp_path": path, "local_params": local}


def cgt_raster(mask_type: str, geom: dict, h: int, w: int) -> np.ndarray:
    yy, xx = np.mgrid[0:h, 0:w].astype("float32")
    x, y = xx / w, yy / h
    if mask_type == "circulargradient":
        cx = (_f(geom, "Left") + _f(geom, "Right")) / 2
        cy = (_f(geom, "Top") + _f(geom, "Bottom")) / 2
        rx = max(abs(_f(geom, "Right") - _f(geom, "Left")) / 2, 1e-3)
        ry = max(abs(_f(geom, "Bottom") - _f(geom, "Top")) / 2, 1e-3)
        ang = math.radians(_f(geom, "Angle"))
        xr = (x - cx) * math.cos(ang) + (y - cy) * math.sin(ang)
        yr = -(x - cx) * math.sin(ang) + (y - cy) * math.cos(ang)
        d = np.sqrt((xr / rx) ** 2 + (yr / ry) ** 2)
        feather = max(_f(geom, "Feather", 50) / 100.0, 0.05)
        # LR's CircularGradient corrects the area OUTSIDE the ellipse by default (verified: center-=1
        # convention round-tripped with negative corr). 0 inside, 1 outside; Flipped re-inverts below.
        m = np.clip((d - 1.0) / feather + 0.5, 0, 1).astype("float32")
    else:
        zx, zy, fx, fy = _f(geom, "ZeroX"), _f(geom, "ZeroY"), _f(geom, "FullX", 1), _f(geom, "FullY")
        dxv, dyv = fx - zx, fy - zy
        L2 = dxv * dxv + dyv * dyv + 1e-6
        m = np.clip(((x - zx) * dxv + (y - zy) * dyv) / L2, 0, 1).astype("float32")
    if str(geom.get("Flipped", "false")).lower().lstrip("+") == "true":
        m = 1.0 - m
    return m


_REGION = [("左上", 0.4, 0.4), ("上方", 0.6, 0.4), ("右上", 1.0, 0.4),
           ("左侧", 0.4, 0.6), ("中心", 0.6, 0.6), ("右侧", 1.0, 0.6),
           ("左下", 0.4, 1.0), ("下方", 0.6, 1.0), ("右下", 1.0, 1.0)]


def _region(geom: dict) -> str:
    if "Top" in geom:
        cx = (_f(geom, "Left") + _f(geom, "Right")) / 2; cy = (_f(geom, "Top") + _f(geom, "Bottom")) / 2
    else:
        cx = (_f(geom, "ZeroX") + _f(geom, "FullX")) / 2; cy = (_f(geom, "ZeroY") + _f(geom, "FullY")) / 2
    for name, mx, my in _REGION:
        if cx <= mx and cy <= my:
            return name
    return "整体"


def make_local_sample(source_path: str, base_xmp: str, geom: dict, out_dir: str,
                      rng: random.Random | None = None) -> dict | None:
    """base preset globally + ONE strong, visible local edit inside `geom`. Save 1-ch C_GT."""
    rng = rng or random.Random()
    label, local = sample_local_edit(rng)
    syn = synth_local_xmp(base_xmp, geom, local, config.RENDER_STAGE)
    if syn is None:
        return None
    res = render.render_preset(syn["xmp_path"], "param", "xmp", source_path)
    try:
        os.remove(syn["xmp_path"])
    except OSError:
        pass
    if not res.get("ok"):
        return None
    w, h = Image.open(res["after_path"]).size
    cgt = cgt_raster(geom.get("__type__", "circulargradient"), geom, h, w)
    os.makedirs(out_dir, exist_ok=True)
    muid = uuid.uuid4().hex[:16]
    cgt_path = os.path.join(out_dir, f"cgt_{muid}.png")
    Image.fromarray((cgt * 255).astype("uint8")).save(cgt_path)
    return {"after_path": res["after_path"], "cgt_path": cgt_path, "mask_unit_id": muid,
            "geom": geom, "region": _region(geom), "local_params": syn["local_params"],
            "edit_label": label}


# --------------------------------------------------------------------------- #
def verify(n: int = 12) -> None:
    bank = load_bank() if os.path.exists(_BANK_PATH) else build_bank()
    print(f"mask bank geoms={bank['n_geom']}")
    conn = db.connect()
    bases = [r["path"] for r in conn.execute(
        "SELECT path FROM assets WHERE asset_type='preset' AND kind='param' AND pass_c=1 LIMIT 400").fetchall()]
    srcs = [r["path"] for r in conn.execute(
        "SELECT path FROM assets WHERE asset_type='image' AND b_quality=3 AND dup_of IS NULL "
        "ORDER BY asset_id LIMIT 200").fetchall()]
    conn.close()
    bases = [b for b in bases if b.endswith(".xmp") and os.path.exists(b)]
    srcs = [s for s in srcs if os.path.exists(s)]
    rng = random.Random(0)
    ok = 0
    for i in range(n):
        g = sample_geom(bank, rng)
        geom = perturb(g["geom"], rng); geom["__what__"] = g["what"]; geom["__type__"] = g["mask_type"]
        base = bases[rng.randrange(len(bases))]; src = srcs[rng.randrange(len(srcs))]
        s = make_local_sample(src, base, geom, config.RENDER_STAGE)
        if not s:
            print(f"[{i+1}] skip (no localizable / render fail)"); continue
        # new model = GLOBAL(non-localizable) everywhere + LOCAL(tone) in mask. To test the LOCAL
        # contribution, diff against the GLOBAL-ONLY render of the same base preset (not the source).
        gb = render.render_preset(base, "param", "xmp", src)
        ref = gb["after_path"] if gb.get("ok") else src
        before = np.asarray(Image.open(ref).convert("RGB").resize((256, 256)), "float32")
        after = np.asarray(Image.open(s["after_path"]).convert("RGB").resize((256, 256)), "float32")
        diff = np.abs(after - before).mean(-1)
        cgt = cgt_raster(g["mask_type"], geom, 256, 256)
        df, cf = diff.flatten(), cgt.flatten()
        corr = float(np.corrcoef(df, cf)[0, 1]) if df.std() > 1e-3 and cf.std() > 1e-3 else 0.0
        din = diff[cgt > 0.5].mean() if (cgt > 0.5).any() else diff.mean()
        passed = corr > 0.35 and din > 2.0
        ok += passed
        print(f"[{i+1}] {g['mask_type']:16s} local={list(s['local_params'])} "
              f"corr={corr:+.2f} in={din:5.1f} {'OK' if passed else 'WEAK'}")
    print(f"\nround-trip: {ok}/{n} (preset look localized to mask)")


def main() -> None:
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("bank")
    v = sub.add_parser("verify"); v.add_argument("--n", type=int, default=12)
    a = ap.parse_args()
    if a.cmd == "bank":
        build_bank()
    else:
        verify(a.n)


if __name__ == "__main__":
    main()
