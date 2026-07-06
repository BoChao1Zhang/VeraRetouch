# lens-exp common utilities: manifest, probe targets, box parsing, metrics helpers
import os, re, json, random, glob
import numpy as np
import cv2

LR_ROOT = os.path.expanduser("~/data/datasets/ArtEdit-Bench/Lr")
RESULTS = os.path.expanduser("~/VeraRetouch/lens_exp_results")
DUMPS = os.path.join(RESULTS, "dumps")

BOX_RE = re.compile(r"<box>\s*([\d.]+)\s+([\d.]+)\s+([\d.]+)\s+([\d.]+)\s*</box>")

TOKENS = ["light", "colortemp", "colormixer"]


def parse_boxes(text):
    return [tuple(float(v) for v in m) for m in BOX_RE.findall(text)]


def build_manifest(n_per_lang=150, n_box_per_lang=50, seed=0):
    """Stratified sample: per language, prefer n_box_per_lang box-containing samples."""
    rng = random.Random(seed)
    rows = []
    for lang in ["CN", "EN"]:
        ids = sorted(os.listdir(os.path.join(LR_ROOT, lang)), key=lambda s: int(s))
        boxed, plain = [], []
        for sid in ids:
            d = os.path.join(LR_ROOT, lang, sid)
            try:
                txt = open(os.path.join(d, "user_want.txt"), encoding="utf-8").read().strip()
            except Exception:
                continue
            if not (os.path.exists(os.path.join(d, "input.jpg")) and os.path.exists(os.path.join(d, "gt.jpg"))):
                continue
            (boxed if BOX_RE.search(txt) else plain).append((sid, txt))
        rng.shuffle(boxed); rng.shuffle(plain)
        nb = min(n_box_per_lang, len(boxed))
        sel = boxed[:nb] + plain[: n_per_lang - nb]
        for sid, txt in sel:
            d = os.path.join(LR_ROOT, lang, sid)
            rows.append(dict(
                key=f"{lang}_{sid}", lang=lang, sid=sid,
                input_path=os.path.join(d, "input.jpg"),
                gt_path=os.path.join(d, "gt.jpg"),
                prompt=txt, has_box=int(bool(BOX_RE.search(txt))),
            ))
    rows.sort(key=lambda r: r["key"])
    return rows


def load_manifest(path=None):
    path = path or os.path.join(RESULTS, "manifest.json")
    with open(path, encoding="utf-8") as f:
        return json.load(f)


# ---------------- probe targets from (input, gt) ----------------

def _lab(img_bgr):
    lab = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2LAB).astype(np.float32)
    L = lab[..., 0] * (100.0 / 255.0)
    a = lab[..., 1] - 128.0
    b = lab[..., 2] - 128.0
    return L, a, b


def _cct_mired(img_bgr):
    """McCamy CCT from mean RGB, returned in mired (1e6/K) for linearity."""
    rgb = img_bgr[..., ::-1].reshape(-1, 3).mean(axis=0) / 255.0
    r, g, b = np.maximum(rgb, 1e-6)
    X = 0.4124 * r + 0.3576 * g + 0.1805 * b
    Y = 0.2126 * r + 0.7152 * g + 0.0722 * b
    Z = 0.0193 * r + 0.1192 * g + 0.9505 * b
    s = X + Y + Z
    x, y = X / s, Y / s
    n = (x - 0.3320) / (0.1858 - y + 1e-9)
    cct = 449.0 * n ** 3 + 3525.0 * n ** 2 + 6823.3 * n + 5520.33
    cct = float(np.clip(cct, 1000, 25000))
    return 1e6 / cct


HUE_NAMES = ["red", "orange", "yellow", "green", "aqua", "blue", "purple", "magenta"]
# hue bucket centers in OpenCV hue units (0-180): red~0, orange~15, yellow~30, green~60, aqua~90, blue~120, purple~140, magenta~160
HUE_CENTERS = np.array([0, 15, 30, 60, 90, 120, 140, 160], dtype=np.float32)


def _hue_bucket_masks(hsv):
    h = hsv[..., 0].astype(np.float32)
    s = hsv[..., 1].astype(np.float32) / 255.0
    v = hsv[..., 2].astype(np.float32) / 255.0
    # distance on hue circle (0-180)
    d = np.abs(h[..., None] - HUE_CENTERS[None, None, :])
    d = np.minimum(d, 180.0 - d)
    idx = d.argmin(axis=-1)
    chroma_w = (s > 0.10) & (v > 0.05)  # only chromatic pixels vote
    return idx, chroma_w


def compute_targets(input_path, gt_path, max_side=512):
    """Return dict token -> 1D target vector, computed on downscaled pair."""
    a = cv2.imread(input_path, cv2.IMREAD_COLOR)
    b = cv2.imread(gt_path, cv2.IMREAD_COLOR)
    if a is None or b is None:
        raise IOError(f"cannot read {input_path} / {gt_path}")
    if a.shape[:2] != b.shape[:2]:
        b = cv2.resize(b, (a.shape[1], a.shape[0]), interpolation=cv2.INTER_AREA)
    scale = max_side / max(a.shape[:2])
    if scale < 1.0:
        dsize = (int(a.shape[1] * scale), int(a.shape[0] * scale))
        a = cv2.resize(a, dsize, interpolation=cv2.INTER_AREA)
        b = cv2.resize(b, dsize, interpolation=cv2.INTER_AREA)

    La, aa, ba = _lab(a)
    Lb, ab_, bb = _lab(b)
    hi = La >= np.percentile(La, 75)
    lo = La <= np.percentile(La, 25)

    light = np.array([
        Lb.mean() - La.mean(),                    # global mean-L shift
        Lb[hi].mean() - La[hi].mean(),            # highlights shift
        Lb[lo].mean() - La[lo].mean(),            # shadows shift
        Lb.std() - La.std(),                      # contrast shift
        np.percentile(Lb, 95) - np.percentile(La, 95),
        np.percentile(Lb, 5) - np.percentile(La, 5),
    ], dtype=np.float32)

    hsv_a = cv2.cvtColor(a, cv2.COLOR_BGR2HSV)
    hsv_b = cv2.cvtColor(b, cv2.COLOR_BGR2HSV)
    sat_a = hsv_a[..., 1].astype(np.float32) / 255.0
    sat_b = hsv_b[..., 1].astype(np.float32) / 255.0
    colortemp = np.array([
        ab_.mean() - aa.mean(),                   # a* shift (green-magenta / tint)
        bb.mean() - ba.mean(),                    # b* shift (blue-yellow / warmth)
        sat_b.mean() - sat_a.mean(),              # global saturation shift
        _cct_mired(b) - _cct_mired(a),            # CCT shift in mired
    ], dtype=np.float32)

    idx, w = _hue_bucket_masks(hsv_a)
    va = hsv_a[..., 2].astype(np.float32) / 255.0
    vb = hsv_b[..., 2].astype(np.float32) / 255.0
    mixer = np.zeros(16, dtype=np.float32)
    for k in range(8):
        m = (idx == k) & w
        if m.sum() >= 50:
            mixer[k] = (sat_b[m] - sat_a[m]).mean()      # per-hue saturation shift
            mixer[8 + k] = (vb[m] - va[m]).mean()        # per-hue luminance shift
    return {"light": light, "colortemp": colortemp, "colormixer": mixer}


TARGET_NAMES = {
    "light": ["dL_mean", "dL_highlight", "dL_shadow", "dL_std", "dL_p95", "dL_p5"],
    "colortemp": ["da_mean", "db_mean", "dSat_mean", "dCCT_mired"],
    "colormixer": [f"dSat_{h}" for h in HUE_NAMES] + [f"dLum_{h}" for h in HUE_NAMES],
}


# ---------------- box -> patch grid mapping ----------------

def box_to_patch_mask(box, img_w, img_h, grid):
    """box normalized on original image -> boolean mask on grid x grid patch layout
    of the center-padded square image."""
    S = max(img_w, img_h)
    ox = (S - img_w) / 2.0
    oy = (S - img_h) / 2.0
    x1, y1, x2, y2 = box
    px1 = (x1 * img_w + ox) / S * grid
    px2 = (x2 * img_w + ox) / S * grid
    py1 = (y1 * img_h + oy) / S * grid
    py2 = (y2 * img_h + oy) / S * grid
    m = np.zeros((grid, grid), dtype=bool)
    xa, xb = int(np.floor(px1)), int(np.ceil(px2))
    ya, yb = int(np.floor(py1)), int(np.ceil(py2))
    m[max(0, ya):min(grid, yb), max(0, xa):min(grid, xb)] = True
    return m


def image_valid_patch_mask(img_w, img_h, grid):
    """patches that fall on actual image content (not the pad bars)."""
    return box_to_patch_mask((0, 0, 1, 1), img_w, img_h, grid)
