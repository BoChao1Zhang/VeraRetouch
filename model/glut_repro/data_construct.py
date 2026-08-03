"""D-CONSTRUCT pair loader + oracle-s assembly for G3 (Gate D3, collapse channel).

Data: experiments/tooling-wave1/T4_construct/sanity/{train,val}  (D-CONSTRUCT,
S-train / S-val, 8 levels x 200/24).  G3 uses L1 (semantic binary mask) and L4
(geometric soft mask).

Oracle s (task card: "s = GT 掩膜下采样对齐"), matching the INF-5 scache
convention exactly:
    mask uint8/255  --PIL BOX (area-weighted)-->  32x32 float  (the CACHED s)
    32x32           --bilinear--------------->    HxW          (the RENDERED s)

The 32x32 stage is not cosmetic: Delta_shuffle requires identical s shapes
across images (collapse_probes refuses silent resize), and the low-resolution
bottleneck is what the real VLM readout will produce.

Upsampling is plain bilinear, NOT guided (tools/scache/upsample.py).  Guided
upsampling injects the input image into s, which would inflate Delta_const /
Delta_shuffle for reasons unrelated to whether the model uses the s axis
(NOTES decision 3).

Training tensors: a uniform random pixel subsample per image (preserves the
natural in-mask area fraction alpha; no in-mask oversampling -- oversampling
would change the very cost/benefit balance the gate is measuring).
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass, field

import sys

import numpy as np
from PIL import Image

_REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
CONSTRUCT_ROOT = os.path.join(
    _REPO, "experiments/tooling-wave1/T4_construct/sanity")
S_SIZE = 32

sys.path.insert(0, os.path.join(_REPO, "tools", "construct"))
import transforms as T  # noqa: E402

# G3 primary dataset ("fixed" arm).  The as-shipped D-CONSTRUCT L1+L4 mixes 40
# distinct transform identities (class|tier|sign); s encodes only WHERE, never
# WHICH, so E[y|x,s] regresses to near-identity and the best possible f(x,s)
# scores Delta_shuffle* = +0.5 dB full-image / NEGATIVE in-mask -- i.e. the
# mixed set cannot express the >=3 dB Gate-D3 criterion at all (NOTES decision
# 1, measured).  We therefore re-render the target of the SAME (source, mask)
# pairs with one pinned transform, using the generator's own primitives
# (tools/construct/transforms.py: O = (1-m) I + m T1(I), verbatim).
FIXED_SPEC = {"class": "exposure", "tier": 2, "amp": 0.60, "sign": 1,
              "dev": 0.60}

# Intermediate difficulty ("tiered"): the transform CLASS and SIGN are pinned,
# but the AMPLITUDE tier varies per pair and is NOT encoded in s.  s therefore
# still says exactly WHERE, and the best any f(x,s) can do is apply the mean
# amplitude -- a partially identifiable task that sits between `fixed` (s is
# fully sufficient) and `mixed` (s is nearly worthless).  Without this middle
# rung a "no collapse" verdict on `fixed` alone would not generalise.
TIERED_TIERS = [0.15, 0.30, 0.60, 1.20]      # transforms.TIERS["exposure"]


def tiered_spec(pair) -> dict:
    """Deterministic per-pair amplitude tier (stable across runs/machines)."""
    import hashlib
    uid = pair.uid if hasattr(pair, "uid") else str(pair)
    t = int(hashlib.md5(uid.encode()).hexdigest(), 16) % 4
    return {"class": "exposure", "tier": t, "amp": TIERED_TIERS[t],
            "sign": 1, "dev": TIERED_TIERS[t]}


# ---------------------------------------------------------------------------
# Manifest
# ---------------------------------------------------------------------------

@dataclass
class Pair:
    uid: str
    level: str
    split: str
    root: str
    files: dict
    transform: dict
    mask_kind: str
    alpha: float
    img_hw: tuple
    meta: dict = field(default_factory=dict)

    def path(self, key: str) -> str:
        return os.path.join(self.root, self.files[key])

    @property
    def tkey(self) -> str:
        """Transform identity: class|tier|sign (the thing s does NOT encode)."""
        t = self.transform
        return f"{t['class']}|{t['tier']}|{t['sign']}"


def load_pairs(split: str = "train", levels=("L1", "L4"),
               root: str = CONSTRUCT_ROOT,
               tkeys: set | None = None) -> list[Pair]:
    """Read manifest.jsonl of one split, keep the requested levels (and,
    optionally, only the requested transform identities)."""
    mpath = os.path.join(root, split, "manifest.jsonl")
    out = []
    with open(mpath) as f:
        for line in f:
            if not line.strip():
                continue
            r = json.loads(line)
            if r["level"] not in levels:
                continue
            p = Pair(uid=r["uid"], level=r["level"], split=r["split"],
                     root=os.path.join(root, split), files=r["files"],
                     transform=r["transform"], mask_kind=r["mask_kind"],
                     alpha=float(r.get("alpha_achieved") or 0.0),
                     img_hw=tuple(r["img_hw"]),
                     meta={"source_id": r["source_id"],
                           "feather_px": r.get("feather_px"),
                           "feather_kind": r.get("feather_kind")})
            if tkeys is not None and p.tkey not in tkeys:
                continue
            out.append(p)
    return sorted(out, key=lambda p: p.uid)


# ---------------------------------------------------------------------------
# s field
# ---------------------------------------------------------------------------

def mask_to_s32(mask_path: str, size: int = S_SIZE) -> np.ndarray:
    """uint8 mask -> (size,size) float32 in [0,1], PIL BOX area-weighted
    (identical recipe to tools/scache/oracle.py)."""
    m = Image.open(mask_path).convert("L")
    small = m.resize((size, size), Image.Resampling.BOX)
    return np.asarray(small, dtype=np.float32) / 255.0


def s32_to_full(s32: np.ndarray, hw: tuple) -> np.ndarray:
    """(32,32) -> (H,W) bilinear, align_corners=False semantics.

    Implemented with numpy (no torch dependency at data time) using the same
    half-pixel centre convention torch's bilinear interpolate uses.
    """
    h, w = int(hw[0]), int(hw[1])
    sh, sw = s32.shape
    yy = (np.arange(h, dtype=np.float64) + 0.5) * sh / h - 0.5
    xx = (np.arange(w, dtype=np.float64) + 0.5) * sw / w - 0.5
    yy = np.clip(yy, 0.0, sh - 1.0)
    xx = np.clip(xx, 0.0, sw - 1.0)
    y0 = np.floor(yy).astype(np.int64)
    x0 = np.floor(xx).astype(np.int64)
    y1 = np.minimum(y0 + 1, sh - 1)
    x1 = np.minimum(x0 + 1, sw - 1)
    wy = (yy - y0).astype(np.float32)[:, None]
    wx = (xx - x0).astype(np.float32)[None, :]
    a = s32[np.ix_(y0, x0)]
    b = s32[np.ix_(y0, x1)]
    c = s32[np.ix_(y1, x0)]
    d = s32[np.ix_(y1, x1)]
    top = a * (1 - wx) + b * wx
    bot = c * (1 - wx) + d * wx
    return (top * (1 - wy) + bot * wy).astype(np.float32)


def read_pair(p: Pair, s_size: int = S_SIZE, fixed_spec: dict | None = None):
    """-> (x HxWx3 f32 [0,1], y HxWx3 f32, s32 (S,S) f32, s HxW f32).

    fixed_spec=None      -> y is the shipped target (mixed transform identities).
    fixed_spec=dict      -> y is re-rendered as O = (1-m) I + m T1(I) with that
                            one spec, using the generator's own primitives and
                            the SHIPPED mask at full resolution (uint8/255).
    fixed_spec=callable  -> spec_fn(pair) -> dict, resolved per pair (tiered).
    """
    if callable(fixed_spec):
        fixed_spec = fixed_spec(p)
    x = np.asarray(Image.open(p.path("in")).convert("RGB"),
                   dtype=np.float32) / 255.0
    if fixed_spec is None:
        y = np.asarray(Image.open(p.path("out")).convert("RGB"),
                       dtype=np.float32) / 255.0
    else:
        m = read_mask(p)
        y = T.composite(x, m, T.apply_transform(x, fixed_spec))
        # the shipped pipeline round-trips through 8-bit PNG; match it so the
        # target is representable and PSNR is not inflated by float precision
        y = (np.round(np.clip(y, 0, 1) * 255.0) / 255.0).astype(np.float32)
    s32 = mask_to_s32(p.path("mask"), s_size)
    s = s32_to_full(s32, x.shape[:2])
    return x, y, s32, s


def read_mask(p: Pair) -> np.ndarray:
    return np.asarray(Image.open(p.path("mask")).convert("L"),
                      dtype=np.float32) / 255.0


# ---------------------------------------------------------------------------
# Pixel-level training cache
# ---------------------------------------------------------------------------

def build_pixel_cache(pairs: list[Pair], px_per_img: int = 30000,
                      seed: int = 0, s_size: int = S_SIZE,
                      fixed_spec: dict | None = None,
                      verbose: bool = True) -> dict:
    """Uniform random pixel subsample per image.

    Returns dict of arrays:
      x   (M,3) f16   input RGB
      y   (M,3) f16   target RGB
      s   (M,)  f16   oracle s at the pixel (32x32 -> bilinear -> HxW)
      img (M,)  i32   pair index
      s32 (n,S,S) f16 the cached low-res s fields (for Delta_shuffle)
      uids, tkeys, levels, alphas
    """
    rng = np.random.default_rng(seed)
    xs, ys, ss, ii, s32s = [], [], [], [], []
    for k, p in enumerate(pairs):
        x, y, s32, s = read_pair(p, s_size, fixed_spec)
        h, w = x.shape[:2]
        n = h * w
        take = min(px_per_img, n)
        idx = rng.choice(n, size=take, replace=False)
        xs.append(x.reshape(-1, 3)[idx].astype(np.float16))
        ys.append(y.reshape(-1, 3)[idx].astype(np.float16))
        ss.append(s.reshape(-1)[idx].astype(np.float16))
        ii.append(np.full(take, k, dtype=np.int32))
        s32s.append(s32.astype(np.float16))
        if verbose and (k + 1) % 100 == 0:
            print(f"[data_construct] {k+1}/{len(pairs)} pairs cached",
                  flush=True)
    return {
        "x": np.concatenate(xs), "y": np.concatenate(ys),
        "s": np.concatenate(ss), "img": np.concatenate(ii),
        "s32": np.stack(s32s),
        "uids": np.array([p.uid for p in pairs]),
        "tkeys": np.array([p.tkey for p in pairs]),
        "levels": np.array([p.level for p in pairs]),
        "alphas": np.array([p.alpha for p in pairs], dtype=np.float32),
    }


def save_cache(cache: dict, path: str) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    tmp = path + ".tmp.npz"
    np.savez(tmp, **cache)
    os.replace(tmp, path)


def load_cache(path: str) -> dict:
    with np.load(path, allow_pickle=False) as z:
        return {k: z[k] for k in z.files}
