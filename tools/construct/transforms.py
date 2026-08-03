"""Color transforms for the INF-2 construct generator.

All transforms operate on float32 sRGB images in [0, 1], shape (H, W, 3).
Composite rule (PLAN §3 second rung): ``O = (1 - m) * T0(I) + m * T1(I)``
with T0 = identity (conservative default, NOTES.md D2).

Five classes x 4 amplitude tiers + hard control (3-interior-node non-monotone
piecewise linear). Tier values follow the authoritative inline table of
PLAN §3 "合成数据（权威数值表，2026-08-03 内联定稿）" (decision D-06):
exposure/wb/hue tiers are magnitudes with randomized sign; sat/gamma tiers are
the literal table values (sign implied by the value itself). hardpwl node
displacement tiers are not specified by the PLAN table and keep the NOTES §4
defaults.
"""

from __future__ import annotations

import numpy as np

# ----------------------------------------------------------------------------
# Tier tables (PLAN §3 2026-08-03 inline table, D-06; hardpwl per NOTES.md §4)
# ----------------------------------------------------------------------------

STD_CLASSES = ["exposure", "wb", "sat", "hue", "gamma"]
HARD_CLASS = "hardpwl"
ALL_CLASSES = STD_CLASSES + [HARD_CLASS]

TIERS = {
    "exposure": [0.15, 0.30, 0.60, 1.20],  # |dEV| stops, gain 2^dEV (sign rnd)
    "wb": [0.03, 0.06, 0.12, 0.24],        # delta on diag(1+d,1,1-d) (sign rnd)
    "sat": [0.70, 0.85, 1.15, 1.40],       # k literal, out = Y + k (in - Y)
    "hue": [5.0, 10.0, 20.0, 40.0],        # |degrees| hue rotation (sign rnd)
    "gamma": [0.80, 0.90, 1.10, 1.25],     # gamma literal, out = x^gamma
    "hardpwl": [0.10, 0.18, 0.28, 0.40],   # max node displacement
}

# ----------------------------------------------------------------------------
# Color-space primitives (verified in selftest.py against colorsys and by
# round-trip identity; sRGB EOTF per IEC 61966-2-1 textbook formula)
# ----------------------------------------------------------------------------


def srgb_to_linear(x: np.ndarray) -> np.ndarray:
    x = np.clip(x, 0.0, 1.0)
    return np.where(x <= 0.04045, x / 12.92, ((x + 0.055) / 1.055) ** 2.4)


def linear_to_srgb(x: np.ndarray) -> np.ndarray:
    x = np.clip(x, 0.0, None)
    return np.where(x <= 0.0031308, x * 12.92, 1.055 * np.power(x, 1.0 / 2.4) - 0.055)


def rgb_to_hsv(img: np.ndarray) -> np.ndarray:
    """Vectorized RGB->HSV, all channels in [0,1] (h in turns)."""
    r, g, b = img[..., 0], img[..., 1], img[..., 2]
    maxc = np.max(img, axis=-1)
    minc = np.min(img, axis=-1)
    v = maxc
    rng = maxc - minc
    s = np.where(maxc > 0, rng / np.where(maxc > 0, maxc, 1.0), 0.0)
    safe = np.where(rng > 0, rng, 1.0)
    rc = (maxc - r) / safe
    gc = (maxc - g) / safe
    bc = (maxc - b) / safe
    h = np.where(
        maxc == r, bc - gc, np.where(maxc == g, 2.0 + rc - bc, 4.0 + gc - rc)
    )
    h = np.where(rng > 0, (h / 6.0) % 1.0, 0.0)
    return np.stack([h, s, v], axis=-1)


def hsv_to_rgb(hsv: np.ndarray) -> np.ndarray:
    h, s, v = hsv[..., 0], hsv[..., 1], hsv[..., 2]
    i = np.floor(h * 6.0)
    f = h * 6.0 - i
    p = v * (1.0 - s)
    q = v * (1.0 - s * f)
    t = v * (1.0 - s * (1.0 - f))
    i = i.astype(np.int64) % 6
    out = np.empty(hsv.shape, dtype=hsv.dtype)
    conds = [i == k for k in range(6)]
    r = np.select(conds, [v, q, p, p, t, v])
    g = np.select(conds, [t, v, v, q, p, p])
    b = np.select(conds, [p, p, t, v, v, q])
    out[..., 0], out[..., 1], out[..., 2] = r, g, b
    return out


_LUMA = np.array([0.2126, 0.7152, 0.0722], dtype=np.float64)  # Rec.709


def luma(img: np.ndarray) -> np.ndarray:
    return img @ _LUMA


# ----------------------------------------------------------------------------
# Transform sampling / application
# ----------------------------------------------------------------------------


def sample_transform(cls: str, tier: int, rng: np.random.Generator) -> dict:
    """Draw concrete parameters for (class, tier). Fully recorded.

    exposure/wb/hue: tier value = magnitude, sign randomized.
    sat/gamma: tier value = literal k / gamma from the PLAN table (D-06);
    sign field records sign(value - 1) for stratification.
    """
    amp = TIERS[cls][tier]
    sign = 1 if rng.random() < 0.5 else -1
    if cls in ("sat", "gamma"):
        sign = 1 if amp >= 1.0 else -1
    spec: dict = {"class": cls, "tier": tier, "amp": amp, "sign": sign}
    if cls == "exposure":
        spec["dev"] = sign * amp
    elif cls == "wb":
        spec["r_gain"] = 1.0 + sign * amp
        spec["b_gain"] = 1.0 - sign * amp
    elif cls == "sat":
        spec["k"] = amp
    elif cls == "hue":
        spec["degrees"] = sign * amp
    elif cls == "gamma":
        spec["gamma"] = amp
    elif cls == HARD_CLASS:
        xs, ys = make_hard_pwl(rng, amp)
        spec["xs"] = [round(float(x), 6) for x in xs]
        spec["ys"] = [round(float(y), 6) for y in ys]
    else:
        raise ValueError(f"unknown transform class {cls}")
    return spec


def make_hard_pwl(rng: np.random.Generator, amp: float, max_tries: int = 64):
    """3 interior nodes, endpoints (0,0)/(1,1), guaranteed non-monotone."""
    for _ in range(max_tries):
        xs_in = np.sort(rng.uniform(0.15, 0.85, size=3))
        if np.min(np.diff(xs_in)) < 0.08:
            continue
        ys_in = np.clip(xs_in + rng.uniform(-amp, amp, size=3), 0.0, 1.0)
        xs = np.concatenate([[0.0], xs_in, [1.0]])
        ys = np.concatenate([[0.0], ys_in, [1.0]])
        if not _is_monotone(xs, ys):
            return xs, ys
        # force one descending segment
        j = int(rng.integers(1, 4))
        ys[j] = max(0.0, ys[j - 1] - 0.6 * amp - 0.02)
        if not _is_monotone(xs, ys):
            return xs, ys
    raise RuntimeError("failed to build a non-monotone pwl curve")


def _is_monotone(xs: np.ndarray, ys: np.ndarray) -> bool:
    return bool(np.all(np.diff(ys) >= 0.0))


def is_non_monotone(spec: dict) -> bool:
    return not _is_monotone(np.asarray(spec["xs"]), np.asarray(spec["ys"]))


def apply_transform(img: np.ndarray, spec: dict) -> np.ndarray:
    """Apply T1 to a float32 sRGB [0,1] image. Returns a new array."""
    cls = spec["class"]
    x = img.astype(np.float64, copy=False)
    if cls == "exposure":
        lin = srgb_to_linear(x) * (2.0 ** spec["dev"])
        out = linear_to_srgb(lin)
    elif cls == "wb":
        lin = srgb_to_linear(x)
        lin = lin * np.array([spec["r_gain"], 1.0, spec["b_gain"]])
        out = linear_to_srgb(lin)
    elif cls == "sat":
        y = luma(x)[..., None]
        out = y + spec["k"] * (x - y)
    elif cls == "hue":
        hsv = rgb_to_hsv(x)
        hsv[..., 0] = (hsv[..., 0] + spec["degrees"] / 360.0) % 1.0
        out = hsv_to_rgb(hsv)
    elif cls == "gamma":
        out = np.power(np.clip(x, 0.0, 1.0), spec["gamma"])
    elif cls == HARD_CLASS:
        xs = np.asarray(spec["xs"])
        ys = np.asarray(spec["ys"])
        out = np.interp(x, xs, ys)
    else:
        raise ValueError(f"unknown transform class {cls}")
    return np.clip(out, 0.0, 1.0).astype(np.float32)


def composite(img: np.ndarray, mask: np.ndarray, edited: np.ndarray) -> np.ndarray:
    """O = (1 - m) I + m T1(I)  (T0 = identity)."""
    m = mask[..., None].astype(np.float32)
    return (1.0 - m) * img + m * edited
