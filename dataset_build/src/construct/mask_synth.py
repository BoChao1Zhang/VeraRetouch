"""Mask v4 Route 1: a selected complete preset applied only inside a subject-aware mask.

Per the design (user-locked 2026-07-11):
  - GLOBAL flow (vlemb) selects an appropriate preset for the source (its core job: pick the preset).
  - subject_geom.sample_plan yields the per-source composition (1 radial + 1 semantic +
    2 band + 4 linear with four-direction polling; bisect fills; no-subject -> all bisect).
  - the complete preset is rendered by the local GPU pipeline, including tone curve/HSL/WB/grade;
  - the rendered branch is composited back with a per-variant alpha (exact lerp), so mask-exterior
    pixels remain the source image. C_GT is exactly that alpha; the render backend writes it
    (contract: docs/LOCAL_PIPELINE_V4_CONTRACT.md §2) — no duplicate CPU raster here.

The v2 geometry bank and the legacy global-preset-plus-Local* XMP synthesis were deleted
(2026-07-11); ``cgt_raster`` stays as the golden CPU raster used by parity checks and tests.

CLI:  python -m construct.mask_synth verify [--n 8]   # plan -> render -> round-trip check
"""
from __future__ import annotations

import argparse
import math
import os
import random
import uuid

import numpy as np
from PIL import Image, ImageFile

from dataset_build.source_qa import config, db
from . import render

ImageFile.LOAD_TRUNCATED_IMAGES = True


def _f(d, k, default=0.0):
    try:
        return float(str(d.get(k, default)).lstrip("+"))
    except Exception:
        return default


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
    # Generated local-preset CGT easing; matches raster_alpha(..., smoothstep=True).
    m = m * m * (3.0 - 2.0 * m)
    if str(geom.get("Flipped", "false")).lower().lstrip("+") == "true":
        m = 1.0 - m
    return m


_REGION = [("左上", 0.4, 0.4), ("上方", 0.6, 0.4), ("右上", 1.0, 0.4),
           ("左侧", 0.4, 0.6), ("中心", 0.6, 0.6), ("右侧", 1.0, 0.6),
           ("左下", 0.4, 1.0), ("下方", 0.6, 1.0), ("右下", 1.0, 1.0)]


def _region_at(cx: float, cy: float) -> str:
    for name, mx, my in _REGION:
        if cx <= mx and cy <= my:
            return name
    return "整体"


def _region(geom: dict) -> str:
    if "Top" in geom:
        cx = (_f(geom, "Left") + _f(geom, "Right")) / 2; cy = (_f(geom, "Top") + _f(geom, "Bottom")) / 2
    else:
        cx = (_f(geom, "ZeroX") + _f(geom, "FullX")) / 2; cy = (_f(geom, "ZeroY") + _f(geom, "FullY")) / 2
    return _region_at(cx, cy)


def _spec_region(spec: dict) -> str:
    if spec.get("mask_type") == "semantic":
        a = np.asarray(spec["alpha"], np.float32)
        ys, xs = np.nonzero(a > 0.5)
        if not len(xs):
            return "整体"
        h, w = a.shape
        return _region_at(float(xs.mean()) / w, float(ys.mean()) / h)
    return _region(spec.get("geom") or {})


def make_local_samples(source_path: str, base_preset: dict, plan: list,
                       out_dir: str, save_cgt: bool = True, store: bool = True) -> list:
    """Localize one complete base preset into a subject_geom plan in one render call.

    ``plan`` entries are subject_geom.sample_plan specs (geometry or semantic).
    C_GT PNGs are produced by the render backend at ``cgt_path`` (contract §2)；
    预览级（两级政策的 768 初筛）传 save_cgt=False 跳过 C_GT 产出。"""
    if not plan:
        return []
    os.makedirs(out_dir, exist_ok=True)
    specs, prepared = [], []
    for idx, spec in enumerate(plan):
        muid = uuid.uuid4().hex[:16]
        payload = {k: spec[k] for k in ("mask_type", "geom", "alpha", "amount") if k in spec}
        if save_cgt:
            payload["cgt_path"] = os.path.join(out_dir, f"cgt_{muid}.png")
        specs.append(payload)
        prepared.append((idx, muid, spec))

    res = render.render_local_preset_variants(
        base_preset["path"], base_preset.get("fmt") or "xmp", source_path, specs,
        preset_id=base_preset.get("preset_id"), store=store)
    rows = res.get("results") or []
    out = []
    for (idx, muid, spec), row in zip(prepared, rows):
        if not row.get("ok") or not row.get("after_path"):
            continue
        out.append({"after_path": row["after_path"], "cgt_path": row.get("cgt_path"),
                    "mask_unit_id": muid, "spec": spec, "geom": spec.get("geom"),
                    "region": _spec_region(spec), "blend_mode": "exact",
                    "engine": row.get("engine"), "variant_index": idx})
    return out


def local_candidate(base: dict, spec: dict, row: dict, qa_score) -> dict:
    """tier 消费的 local 候选记录（agent 与 local_pipeline 共用，结构即下游契约）。"""
    return {"preset_id": row["mask_unit_id"], "kind": "local_preset",
            "fmt": base.get("fmt"), "preset_path": base["path"],
            "content_hash": base.get("preset_content_hash"),
            "after_path": row["after_path"], "qa": qa_score,
            "local": {"route": "geom", "mask_unit_id": row["mask_unit_id"],
                      "mask_type": spec["mask_type"], "mode": spec.get("_mode"),
                      "subject": spec.get("_subject"), "apply": spec.get("_apply"),
                      "feather": spec.get("_feather"),
                      # 语义变体无几何：stub 供 tier 等下游统一读取，GT 在 cgt_path
                      "geom": spec.get("geom") or {"semantic": True},
                      "cgt_path": row["cgt_path"], "region": row["region"],
                      "blend_mode": row["blend_mode"], "engine": row.get("engine"),
                      "base_preset_id": base["preset_id"], "base_preset_path": base["path"],
                      "base_preset_content_hash": base.get("preset_content_hash")}}


# --------------------------------------------------------------------------- #
def verify(n: int = 8) -> None:
    """plan -> render -> round-trip: diff 与 backend 产出的 C_GT 应强相关且 mask 外不动。"""
    from . import subject_geom

    conn = db.connect()
    bases = [r["path"] for r in conn.execute(
        "SELECT path FROM assets WHERE asset_type='preset' AND kind='param' AND pass_c=1 "
        "AND has_local_mask=0 LIMIT 400").fetchall()]
    srcs = [r["path"] for r in conn.execute(
        "SELECT path FROM assets WHERE asset_type='image' AND b_quality=3 AND dup_of IS NULL "
        "ORDER BY asset_id LIMIT 400").fetchall()]
    conn.close()
    bases = [b for b in bases if b.endswith(".xmp") and os.path.exists(b)]
    srcs = [s for s in srcs if os.path.exists(s)]
    rng = random.Random(0)
    ok = tried = 0
    for src in srcs:
        if tried >= n:
            break
        try:
            plan = subject_geom.sample_plan(src, rng)
        except subject_geom.SubjectCacheMiss:
            continue   # 未预计算的源图跳过（verify 只吃已重建 cache 的）
        tried += 1
        base = bases[rng.randrange(len(bases))]
        rows = make_local_samples(src, {"path": base, "fmt": "xmp"}, plan[:2],
                                  config.RENDER_STAGE)
        for s in rows:
            before = np.asarray(Image.open(src).convert("RGB").resize((256, 256)), "float32")
            after = np.asarray(Image.open(s["after_path"]).convert("RGB").resize((256, 256)),
                               "float32")
            diff = np.abs(after - before).mean(-1)
            cgt = np.asarray(Image.open(s["cgt_path"]).convert("L").resize((256, 256)),
                             "float32") / 255.0
            df, cf = diff.flatten(), cgt.flatten()
            corr = float(np.corrcoef(df, cf)[0, 1]) if df.std() > 1e-3 and cf.std() > 1e-3 else 0.0
            din = diff[cgt > 0.5].mean() if (cgt > 0.5).any() else diff.mean()
            dout = diff[cgt < 0.02].mean() if (cgt < 0.02).any() else 0.0
            passed = corr > 0.35 and din > 2.0 and dout < 3.0
            ok += passed
            print(f"{os.path.basename(src)} {s['spec']['mask_type']:16s} "
                  f"engine={s.get('engine')} corr={corr:+.2f} in={din:5.1f} out={dout:4.1f} "
                  f"{'OK' if passed else 'WEAK'}")
    print(f"\nround-trip variants passed: {ok} (sources tried: {tried})")


def main() -> None:
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    v = sub.add_parser("verify"); v.add_argument("--n", type=int, default=8)
    a = ap.parse_args()
    verify(a.n)


if __name__ == "__main__":
    main()
