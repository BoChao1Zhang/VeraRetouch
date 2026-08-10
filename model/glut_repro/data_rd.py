"""RD-STD / RD-E data assembly: the full D-CONSTRUCT capacity ladder L0-L7.

Extends `data_construct.py` (G3's loader, reused verbatim for the manifest
dataclass, the BOX->32x32 oracle-s recipe and the bilinear upsampler) to all
eight levels, including the two levels whose manifest carries more than one
mask / more than one transform:

  L5  mismatch negative control -- rendered with `maskrender` (a radial mask),
      the mask HANDED TO THE MODEL is `mask` (the semantic cgt, IoU<0.30 with
      the render mask).  Oracle s therefore points at the wrong place *by
      design*; the evaluation partition still uses the render mask, because the
      question "did the model reproduce the edit" is about where the edit is.
  L6  two overlapping soft masks (`maska`, `maskb`) with two different
      transforms applied in sequence; `mask` is their union and is both the
      oracle-s source and the evaluation partition.

Two target variants (the G3 / D-26 precedent, extended per level):

  variant="mixed"  the as-shipped target.  Each level mixes up to 40 distinct
                   transform identities (class|tier|sign over the 6x4 COMBOS24
                   grid with randomized sign) while s encodes only WHERE and
                   never WHICH.  Any f(x,s) is then bounded by the conditional
                   mean E[y|x,s], which for near-symmetric signs collapses back
                   to near-identity.  G3 measured the ceiling of Delta_shuffle
                   on this variant at +0.32 dB -- i.e. the >=3 dB criterion is
                   NOT expressible on it.  Kept as the literal-task-card control.
  variant="fixed"  same source image, same mask(s), same `T.composite` primitive
                   -- only the target is re-rendered with ONE pinned transform
                   per level (L6: one pinned PAIR).  s becomes fully sufficient
                   and the >=3 dB criterion becomes reachable.  Ceiling measured
                   per level by `ceiling_rd.py` before any training (task card).

Oracle s (RO-0): mask uint8/255 -> [0,1] linearly, no per-image normalization of
any kind (red line).  Two resolutions:
  s_res="c32"  mask --PIL BOX (area weighted)--> 32x32  --bilinear--> HxW
               (the INF-5 production cache caliber; a common shape across
               images is what makes Delta_shuffle legal at all)
  s_res="full" mask at native resolution, no bottleneck (upper reference: tells
               apart "the renderer cannot" from "the 32x32 cache cannot")
"""

from __future__ import annotations

import os
import sys

import numpy as np
from PIL import Image

import json

from model.glut_repro.data_construct import (  # noqa: F401  (re-exported)
    CONSTRUCT_ROOT,
    S_SIZE,
    Pair,
    load_cache,
    mask_to_s32,
    s32_to_full,
    save_cache,
)

_REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.insert(0, os.path.join(_REPO, "tools", "construct"))
import transforms as T  # noqa: E402

LEVELS = ("L0", "L1", "L2", "L3", "L4", "L5", "L6", "L7")

# ---------------------------------------------------------------------------
# The pinned transforms of the `fixed` variant.
#
# Single-transform levels reuse G3's spec verbatim (exposure +0.60 EV, tier 2)
# so the two experiments' `fixed` numbers are directly comparable.  L6 needs a
# PAIR of distinct transforms or its two overlapping masks would be
# indistinguishable from one union mask, which is exactly the rank question the
# level exists to ask; the pair is (exposure +0.60 EV on mask A, then white
# balance delta=+0.12 on mask B), both taken from the generator's own tier
# tables (tools/construct/transforms.TIERS).
# ---------------------------------------------------------------------------
FIXED_SPEC = {"class": "exposure", "tier": 2, "amp": 0.60, "sign": 1,
              "dev": 0.60}
FIXED_SPEC_B = {"class": "wb", "tier": 2, "amp": 0.12, "sign": 1,
                "r_gain": 1.12, "b_gain": 0.88}


def fixed_specs(level: str) -> list:
    return [FIXED_SPEC, FIXED_SPEC_B] if level == "L6" else [FIXED_SPEC]


# ---------------------------------------------------------------------------
# Manifest reading
#
# data_construct.load_pairs only ever saw L1/L4, where `transform` is one dict
# and `alpha_achieved` one float.  L6 ships a LIST of two transforms and a list
# of two alphas, which makes that loader raise.  Rather than editing G3's
# delivered file (its runs are on record and must stay reproducible), the
# ladder gets its own reader here.
# ---------------------------------------------------------------------------

class RDPair(Pair):
    @property
    def tkey(self) -> str:
        """Transform identity (class|tier|sign) -- the thing s does NOT encode.
        Multi-transform levels join their components with '+'."""
        t = self.transform
        ts = t if isinstance(t, list) else [t]
        return "+".join(f"{u['class']}|{u['tier']}|{u['sign']}" for u in ts)


def load_pairs(split: str = "train", levels=LEVELS,
               root: str = CONSTRUCT_ROOT) -> list:
    out = []
    with open(os.path.join(root, split, "manifest.jsonl")) as f:
        for line in f:
            if not line.strip():
                continue
            r = json.loads(line)
            if r["level"] not in levels:
                continue
            a = r.get("alpha_achieved") or 0.0
            out.append(RDPair(
                uid=r["uid"], level=r["level"], split=r["split"],
                root=os.path.join(root, split), files=r["files"],
                transform=r["transform"], mask_kind=r["mask_kind"],
                alpha=float(np.mean(a) if isinstance(a, list) else a),
                img_hw=tuple(r["img_hw"]),
                meta={"source_id": r["source_id"],
                      "feather_px": r.get("feather_px"),
                      "feather_kind": r.get("feather_kind"),
                      "mismatch_iou": r.get("mismatch_iou"),
                      "overlap_iou": r.get("overlap_iou"),
                      "expected": r.get("expected")}))
    return sorted(out, key=lambda p: p.uid)


# ---------------------------------------------------------------------------
# Per-level mask semantics
# ---------------------------------------------------------------------------

def render_mask_keys(level: str) -> list:
    """Which manifest mask file(s) the TARGET was rendered with."""
    if level == "L5":
        return ["maskrender"]
    if level == "L6":
        return ["maska", "maskb"]
    return ["mask"]


def s_mask_key(level: str) -> str:
    """Which mask file is handed to the model as oracle s.  Always `mask`:
    for L5 that is deliberately the mismatched one (negative control)."""
    return "mask"


def _read_mask(p: Pair, key: str) -> np.ndarray:
    return np.asarray(Image.open(p.path(key)).convert("L"),
                      dtype=np.float32) / 255.0


def eval_mask(p: Pair) -> np.ndarray:
    """Mask used to partition in / boundary-band / out for masked PSNR.

    = where the edit actually is.  For L5 that is the render mask, NOT the
    (mismatched) mask given to the model; for L6 the union of the two."""
    keys = render_mask_keys(p.level)
    m = _read_mask(p, keys[0])
    for k in keys[1:]:
        m = np.maximum(m, _read_mask(p, k))
    return m


# ---------------------------------------------------------------------------
# Pair reading
# ---------------------------------------------------------------------------

def _render_fixed(x: np.ndarray, p: Pair) -> np.ndarray:
    """Re-render the target with the pinned spec(s), using the generator's own
    primitives (tools/construct/transforms.composite: O = (1-m) I + m T1(I)),
    the shipped full-resolution masks, and the shipped 8-bit round-trip."""
    keys = render_mask_keys(p.level)
    specs = fixed_specs(p.level)
    out = x
    for key, spec in zip(keys, specs):
        m = _read_mask(p, key)
        out = T.composite(out, m, T.apply_transform(out, spec))
    return (np.round(np.clip(out, 0, 1) * 255.0) / 255.0).astype(np.float32)


def read_pair(p: Pair, variant: str = "fixed", s_size: int = S_SIZE,
              s_res: str = "c32"):
    """-> dict with x, y, s_cache, s_full, mask (evaluation partition).

    s_cache is what the harness shuffles across images (uniform shape);
    s_full is what the renderer consumes.  With s_res="full" the two differ in
    shape and the shuffle is done by the caller (see train_rd.shuffle_full)."""
    x = np.asarray(Image.open(p.path("in")).convert("RGB"),
                   dtype=np.float32) / 255.0
    if variant == "mixed":
        y = np.asarray(Image.open(p.path("out")).convert("RGB"),
                       dtype=np.float32) / 255.0
    elif variant == "fixed":
        y = _render_fixed(x, p)
    else:
        raise ValueError(f"unknown variant {variant!r}")
    sm_path = p.path(s_mask_key(p.level))
    if s_res == "c32":
        s_cache = mask_to_s32(sm_path, s_size)
        s_full = s32_to_full(s_cache, x.shape[:2])
    elif s_res == "full":
        s_full = np.asarray(Image.open(sm_path).convert("L"),
                            dtype=np.float32) / 255.0
        s_cache = s_full
    else:
        raise ValueError(f"unknown s_res {s_res!r}")
    return {"x": x, "y": y, "s_cache": s_cache, "s_full": s_full,
            "mask": eval_mask(p), "uid": p.uid, "level": p.level,
            "alpha": p.alpha}


# ---------------------------------------------------------------------------
# Pixel-level training cache
# ---------------------------------------------------------------------------

def build_pixel_cache(pairs: list, variant: str, px_per_img: int = 30000,
                      seed: int = 0, s_size: int = S_SIZE,
                      s_res: str = "c32", verbose: bool = True) -> dict:
    """Uniform random pixel subsample per image.

    Uniform, NOT in-mask-stratified: over-sampling the mask would change the
    very cost/benefit balance the capacity ladder is measuring (PLAN's
    hard-mining stratification belongs to the Stage-2 recipe, not here).
    """
    rng = np.random.default_rng(seed)
    xs, ys, ss, ii = [], [], [], []
    for k, p in enumerate(pairs):
        d = read_pair(p, variant, s_size, s_res)
        h, w = d["x"].shape[:2]
        n = h * w
        take = min(px_per_img, n)
        idx = rng.choice(n, size=take, replace=False)
        xs.append(d["x"].reshape(-1, 3)[idx].astype(np.float16))
        ys.append(d["y"].reshape(-1, 3)[idx].astype(np.float16))
        ss.append(d["s_full"].reshape(-1)[idx].astype(np.float16))
        ii.append(np.full(take, k, dtype=np.int32))
        if verbose and (k + 1) % 100 == 0:
            print(f"[data_rd] {k+1}/{len(pairs)} pairs cached", flush=True)
    return {
        "x": np.concatenate(xs), "y": np.concatenate(ys),
        "s": np.concatenate(ss), "img": np.concatenate(ii),
        "uids": np.array([p.uid for p in pairs]),
        "tkeys": np.array([p.tkey for p in pairs]),
        "levels": np.array([p.level for p in pairs]),
        "alphas": np.array([p.alpha for p in pairs], dtype=np.float32),
    }


def even_subset(pairs: list, n: int) -> list:
    """Evenly spaced over the uid-sorted list -- NEVER the first n.

    (G3 lesson: uids sort level-major, so a prefix of a multi-level list is
    100% one level.  Within a single level a prefix is still biased, because
    the generator's transform grid cycles with the index: COMBOS24[i % 24]
    means indices 0..7 are only 8 of the 24 class x tier cells.)
    """
    pairs = list(pairs)
    if n >= len(pairs):
        return pairs
    idx = np.linspace(0, len(pairs) - 1, n).round().astype(int)
    return [pairs[i] for i in dict.fromkeys(idx.tolist())]
