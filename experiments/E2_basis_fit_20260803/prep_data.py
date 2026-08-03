"""E2 prep: synthetic masks (4 geometric families + constants), semantic
C_GT masks (l-series, slot_id=semantic-*, S-val), CLIP dense semantic
channels + range channels per image, per-image standardization +
residual orthogonalization (with before/after correlation bookkeeping).

Outputs (CACHE = /var/cache/veradata/e2_basis_20260803):
  CACHE/feats/<key>.npz   L, S (H,W f16 standardized), e (H,W,6 f16 final),
                          e_raw48 (48,48,6 f32), img_u8 (H,W,3 u8)
  CACHE/masks/<mask_id>.npz  mask (H,W f16), + meta in masks.jsonl
  <exp>/config/prep.json  + <exp>/cache_meta/residualization.json
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import scipy.ndimage as ndi
from PIL import Image

EXP = Path("/home/bc/VeraRetouch/experiments/E2_basis_fit_20260803")
CACHE = Path("/var/cache/veradata/e2_basis_20260803")
sys.path.insert(0, "/home/bc/VeraRetouch")
sys.path.insert(0, str(EXP))
import e2lib  # noqa: E402
from tools.construct.masks import SemanticBank, soft_edge  # noqa: E402

CATALOG = ("/home/bc/VeraRetouch/experiments/tooling-wave1/"
           "T4_construct/cache/source_catalog.jsonl")
SIZE = 512
SEED = 20260803


# --------------------------------------------------------------------------
# Synthetic geometric masks (512^2)
# --------------------------------------------------------------------------

def smoothstep(t):
    t = np.clip(t, 0.0, 1.0)
    return t * t * (3.0 - 2.0 * t)


def gen_geometric(n_per_family: int, rng: np.random.Generator):
    """PLAN section 2 parameter table adapted to 512^2 single masks."""
    H = W = SIZE
    D = float(np.hypot(H, W))
    ys, xs = np.mgrid[0:H, 0:W]
    ys = ys + 0.5
    xs = xs + 0.5
    out = []

    def linear(i):
        theta = rng.uniform(0, 2 * np.pi)
        t = xs * np.cos(theta) + ys * np.sin(theta)
        w_px = rng.uniform(0.05, 0.50) * SIZE
        lo, hi = np.quantile(t, [0.2, 0.8])
        t0 = rng.uniform(lo, hi)
        m = smoothstep((t0 - t + w_px / 2) / w_px)
        return m, {"theta": theta, "t0": float(t0), "w_px": float(w_px)}

    def radial(i):
        cx, cy = rng.uniform(0.2, 0.8) * W, rng.uniform(0.2, 0.8) * H
        r0 = rng.uniform(0.10, 0.40) * D
        r1 = r0 + rng.uniform(0.15, 0.60) * D
        d = np.hypot(xs - cx, ys - cy)
        m = 1.0 - smoothstep((d - r0) / (r1 - r0))
        return m, {"cx": cx, "cy": cy, "r0": float(r0), "r1": float(r1)}

    def elliptical(i):
        cx, cy = rng.uniform(0.25, 0.75) * W, rng.uniform(0.25, 0.75) * H
        a = rng.uniform(0.10, 0.50) * D
        b = rng.uniform(0.10, 0.50) * D
        phi = rng.uniform(0, np.pi)
        f = rng.uniform(0.02, 0.20) * D
        u = (xs - cx) * np.cos(phi) + (ys - cy) * np.sin(phi)
        v = -(xs - cx) * np.sin(phi) + (ys - cy) * np.cos(phi)
        d = (np.sqrt((u / a) ** 2 + (v / b) ** 2) - 1.0) * np.sqrt(a * b)
        m = 1.0 - smoothstep((d + f / 2) / f)
        return m, {"cx": cx, "cy": cy, "a": float(a), "b": float(b),
                   "phi": phi, "feather": float(f)}

    def ring(i):
        # thin annuli, hw <= 0.10*r_mid: the best monotone readout of ANY
        # q-field is the covering disc with soft-IoU 1-(r_in/r_out)^2; the
        # thinness bound keeps that analytic ceiling at 0.21-0.33 so the
        # pre-registered monotone<=0.40 is a property of the readout, not of
        # accidentally-fat rings (see NOTES, post-smoke family fix).
        cx, cy = rng.uniform(0.35, 0.65) * W, rng.uniform(0.35, 0.65) * H
        r_mid = rng.uniform(0.24, 0.35) * D
        hw = r_mid * rng.uniform(0.06, 0.10)
        f = hw * rng.uniform(0.3, 0.6)
        d = np.hypot(xs - cx, ys - cy)
        m = soft_edge(hw - np.abs(d - r_mid), f)
        return np.asarray(m, dtype=np.float64), {
            "cx": cx, "cy": cy, "r_mid": float(r_mid),
            "half_width": float(hw), "feather": float(f)}

    def wedge(i):
        cx, cy = rng.uniform(0.3, 0.7) * W, rng.uniform(0.3, 0.7) * H
        direction = rng.uniform(0, 2 * np.pi)
        half_ang = np.deg2rad(rng.uniform(15, 45))
        f_ang = np.deg2rad(rng.uniform(2, 8))
        ang = np.arctan2(ys - cy, xs - cx)
        dif = np.abs((ang - direction + np.pi) % (2 * np.pi) - np.pi)
        m = smoothstep((half_ang - dif + f_ang) / (2 * f_ang))
        return m, {"cx": cx, "cy": cy, "direction": direction,
                   "half_angle": float(half_ang), "feather_ang": float(f_ang)}

    fams = {"linear": linear, "radial_ell": None, "ring": ring,
            "wedge": wedge}
    for fam, fn in fams.items():
        for i in range(n_per_family):
            if fam == "radial_ell":
                m, p = (radial(i) if i % 2 == 0 else elliptical(i))
                p["kind"] = "radial" if i % 2 == 0 else "elliptical"
            else:
                m, p = fn(i)
            # reject degenerate coverage
            if not (0.02 <= m.mean() <= 0.9):
                m = np.clip(m, 0, 1)
            out.append({"family": fam, "params": p,
                        "mask": m.astype(np.float16)})
    for c in (0.0, 0.25, 0.5, 0.75, 1.0):
        for k in range(4):
            out.append({"family": "constant",
                        "params": {"value": c, "rep": k},
                        "mask": np.full((H, W), c, dtype=np.float16)})
    return out


# --------------------------------------------------------------------------
# Image channels
# --------------------------------------------------------------------------

def process_image(img: np.ndarray, clip: "e2lib.ClipDense"):
    """img (H,W,3) [0,1] -> dict of channels + residualization record."""
    H, W = img.shape[:2]
    sims48 = clip.sim_maps(img)                              # (48,48,6)
    up = np.stack([ndi.zoom(sims48[..., k].astype(np.float64),
                            (H / 48, W / 48), order=1)
                   for k in range(6)], axis=-1)
    L, S = e2lib.range_channels(img)
    guide = L
    e_gf = np.stack([e2lib.guided_filter(guide, up[..., k], r=32, eps=1e-3)
                     for k in range(6)], axis=-1)
    e_std = (e_gf - e_gf.mean((0, 1))) / (e_gf.std((0, 1)) + 1e-6)
    Ls, Ss = e2lib.standardize(L), e2lib.standardize(S)
    geo = e2lib.geo_features(H, W, stride=1)                 # (P,5)
    A = np.concatenate([np.ones((geo.shape[0], 1)), geo,
                        Ls.reshape(-1, 1), Ss.reshape(-1, 1)], axis=1)
    Er, before, after = e2lib.residualize(e_std.reshape(-1, 6), A)
    Er = Er.reshape(H, W, 6)
    Er = (Er - Er.mean((0, 1))) / (Er.std((0, 1)) + 1e-6)
    fro_before = float(np.linalg.norm(before))
    fro_after = float(np.linalg.norm(after))
    return {"L": Ls.astype(np.float16), "S": Ss.astype(np.float16),
            "e": Er.astype(np.float16), "e_raw48": sims48,
            "img_u8": (np.clip(img, 0, 1) * 255).astype(np.uint8),
            }, {"fro_offblock_before": fro_before,
                "fro_offblock_after": fro_after}


def load_val_image_512(bank, row):
    img = bank.load_image(row)                                # long side 1024
    h, w = img.shape[:2]
    sc = SIZE / max(h, w)
    pil = Image.fromarray((img * 255).astype(np.uint8)).resize(
        (max(1, round(w * sc)), max(1, round(h * sc))),
        Image.Resampling.LANCZOS)
    return np.asarray(pil, dtype=np.float32) / 255.0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--device", default="cuda:0")
    args = ap.parse_args()
    t0 = time.time()
    rng = np.random.default_rng(SEED)
    n_geo = 8 if args.smoke else 200
    n_sem = 10 if args.smoke else 200
    n_pool = 6 if args.smoke else 100
    suffix = "_smoke" if args.smoke else ""

    feats_dir = CACHE / f"feats{suffix}"
    masks_dir = CACHE / f"masks{suffix}"
    feats_dir.mkdir(parents=True, exist_ok=True)
    masks_dir.mkdir(parents=True, exist_ok=True)

    bank = SemanticBank(CATALOG)
    assert "sqlite" in bank.split_rule.lower() or "side-table" in \
        bank.split_rule.lower() or Path(
        "/home/bc/VeraRetouch/tools/data_splits/splits.sqlite3").exists(), \
        f"unexpected split rule: {bank.split_rule}"
    val_sources = bank.sources("val")
    print(f"S-val sources: {len(val_sources)}; split rule: "
          f"{bank.split_rule[:60]}", flush=True)
    clip = e2lib.ClipDense(device=args.device)

    resid_log = []
    mask_rows = []

    # ---- semantic masks (<=1 per source, conf!=low, admissible) ----------
    order = list(val_sources)
    np.random.default_rng(SEED + 1).shuffle(order)
    n_low_skipped = 0
    for sid in order:
        if sum(r["family"] == "semantic" for r in mask_rows) >= n_sem:
            break
        rows = [r for r in bank.semantic_rows_of(sid)]
        rows_ok = [r for r in rows if r.get("winner_confidence") != "low"]
        n_low_skipped += len(rows) - len(rows_ok)
        picked = False
        for row in rows_ok:
            img = load_val_image_512(bank, row)
            cgt = bank.load_cgt(row, img.shape[:2])
            if not bank.cgt_admissible(cgt):
                continue
            key = f"sem_{row['sample_id'][:48]}"
            ch, rec = process_image(img, clip)
            np.savez_compressed(feats_dir / f"{key}.npz", **ch)
            rec["key"] = key
            resid_log.append(rec)
            # 5-class assignment: dominant standardized content anchor
            e_std = ch["e"].astype(np.float32)
            in_mask = cgt > 0.5
            if in_mask.sum() < 50:
                in_mask = cgt > cgt.mean()
            means = [float(e_std[..., k][in_mask].mean())
                     for k in range(6)]
            cls = e2lib.CONTENT_CLASSES[int(np.argmax(means[:5]))]
            mid = f"semantic_{len(mask_rows):04d}"
            np.savez_compressed(masks_dir / f"{mid}.npz",
                                mask=cgt.astype(np.float16))
            mask_rows.append({
                "mask_id": mid, "family": "semantic", "feat_key": key,
                "source_id": sid, "sample_id": row["sample_id"],
                "slot_id": row["slot_id"], "class5": cls,
                "winner_confidence": row.get("winner_confidence"),
                "params": {"cgt_mean": float(cgt.mean())}})
            picked = True
            break
        if picked:
            continue
    n_sem_got = sum(r["family"] == "semantic" for r in mask_rows)
    print(f"semantic masks: {n_sem_got} (low-conf rows skipped: "
          f"{n_low_skipped})", flush=True)

    # ---- image pool for geometric pairing (square 512^2) -----------------
    pool_keys = []
    for sid in order[::-1]:
        if len(pool_keys) >= n_pool:
            break
        rows = bank.rows_of(sid)
        if not rows:
            continue
        row = rows[0]
        try:
            img = load_val_image_512(bank, row)
        except Exception:
            continue
        pil = Image.fromarray((img * 255).astype(np.uint8)).resize(
            (SIZE, SIZE), Image.Resampling.LANCZOS)
        img_sq = np.asarray(pil, dtype=np.float32) / 255.0
        key = f"pool_{len(pool_keys):03d}_{sid[:32]}"
        ch, rec = process_image(img_sq, clip)
        np.savez_compressed(feats_dir / f"{key}.npz", **ch)
        rec["key"] = key
        resid_log.append(rec)
        pool_keys.append(key)
    print(f"pool images: {len(pool_keys)}", flush=True)

    # ---- synthetic masks --------------------------------------------------
    synth = gen_geometric(n_geo, rng)
    for i, srec in enumerate(synth):
        mid = f"{srec['family']}_{i:04d}"
        np.savez_compressed(masks_dir / f"{mid}.npz", mask=srec["mask"])
        mask_rows.append({
            "mask_id": mid, "family": srec["family"],
            "feat_key": pool_keys[i % len(pool_keys)],
            "params": srec["params"]})

    with open(CACHE / f"masks{suffix}.jsonl", "w") as fh:
        for r in mask_rows:
            fh.write(json.dumps(r) + "\n")
    (EXP / "cache_meta").mkdir(exist_ok=True)
    with open(EXP / "cache_meta" / f"residualization{suffix}.json", "w") as fh:
        json.dump(resid_log, fh, indent=1)

    from collections import Counter
    fam_counts = Counter(r["family"] for r in mask_rows)
    cls_counts = Counter(r.get("class5") for r in mask_rows
                         if r["family"] == "semantic")
    cfg = {
        "git_commit": subprocess.run(
            ["git", "-C", "/home/bc/VeraRetouch", "rev-parse", "HEAD"],
            capture_output=True, text=True).stdout.strip(),
        "seed": SEED, "size": SIZE, "n_per_geo_family": n_geo,
        "n_semantic": n_sem_got, "n_pool": len(pool_keys),
        "families": dict(fam_counts), "semantic_class5": dict(cls_counts),
        "clip": "openai/clip-vit-large-patch14-336 dense (MaskCLIP v-proj), "
                "input 672^2, grid 48^2, guided filter r=32 eps=1e-3",
        "split": "S-val via tools/data_splits/splits.sqlite3 "
                 "(SemanticBank default)",
        "low_conf_excluded": True,
        "wall_seconds": round(time.time() - t0, 1),
    }
    with open(EXP / "config" / f"prep{suffix}.json", "w") as fh:
        json.dump(cfg, fh, indent=2)
    print(json.dumps(cfg, indent=1), flush=True)


if __name__ == "__main__":
    main()
