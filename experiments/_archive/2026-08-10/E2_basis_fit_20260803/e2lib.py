"""E2 shared library: 14-dim mask basis (PLAN v2 section 1.3), readouts,
soft-IoU, per-mask L-BFGS fitting, CLIP dense semantic channels.

Basis (single-axis v1, PLAN section 1.3):
    q(p) = w0 + alpha * (w_dir . phi_dir(p)),  ||w_dir||=1, alpha>=0
    s    = 3 * tanh(q / 3)
    phi  = [1] + [x, y, P2(x), P2(y), x*y]  (Legendre, centered / short side)
         + [L, S] (range)  + [e1..e6] (semantic, per-image standardized +
           residual-orthogonalized against the geometric+range block)

Readouts:
    monotone: m = sigmoid(g * s + b)
    bandpass: m = exp(-0.5 * ((s - mu) / sigma)^2), sigma bounded via sigmoid

Loss = 1 - softIoU_minmax, softIoU_minmax = sum(min(m,t)) / sum(max(m,t))
(exact recovery of a *soft* target scores 1.0, which is what the >=0.97
pre-registered thresholds semantically require; the product-form soft IoU is
also reported as a secondary metric).
"""

from __future__ import annotations

import numpy as np
import torch

# --------------------------------------------------------------------------
# Geometry features (Legendre, centered, normalized by half short side)
# --------------------------------------------------------------------------

GEO_NAMES = ["x", "y", "P2x", "P2y", "xy"]
CUBIC_NAMES = ["P3x", "P3y", "P2x_y", "x_P2y"]
RANGE_NAMES = ["L", "S"]
SEM_NAMES = [f"e{i}" for i in range(1, 7)]
DIR_NAMES_14 = GEO_NAMES + RANGE_NAMES + SEM_NAMES          # 13 dirs + [1]

ANCHORS = {
    "sky": ["the sky", "clouds in the sky", "a photo of the sky"],
    "skin": ["human skin", "a person", "a person's face",
             "a portrait photo of a person"],
    "foliage": ["green foliage", "trees and plants", "leaves of a plant"],
    "water": ["water", "a lake or a river", "waves of the sea"],
    "architecture": ["a building", "architecture", "an urban street scene"],
    "subject": ["the main subject of the photo",
                "the foreground subject of the photo",
                "the most salient object in the photo"],
}
ANCHOR_ORDER = ["sky", "skin", "foliage", "water", "architecture", "subject"]
CONTENT_CLASSES = ["sky", "skin", "foliage", "water", "architecture"]


def norm_coords(h: int, w: int, stride: int = 1):
    """Centered pixel coords normalized by half the SHORT side -> short side
    spans [-1, 1]."""
    half = min(h, w) / 2.0
    ys = (np.arange(0, h, stride) + 0.5 - h / 2.0) / half
    xs = (np.arange(0, w, stride) + 0.5 - w / 2.0) / half
    X, Y = np.meshgrid(xs, ys)
    return X.astype(np.float64), Y.astype(np.float64)


def P2(t):
    return 0.5 * (3.0 * t * t - 1.0)


def P3(t):
    return 0.5 * (5.0 * t ** 3 - 3.0 * t)


def geo_features(h: int, w: int, stride: int = 1, cubic: bool = False):
    """(P, 5) or (P, 9) float64 matrix of direction features (no constant)."""
    X, Y = norm_coords(h, w, stride)
    cols = [X, Y, P2(X), P2(Y), X * Y]
    if cubic:
        cols += [P3(X), P3(Y), P2(X) * Y, X * P2(Y)]
    return np.stack([c.reshape(-1) for c in cols], axis=1)


def monomial_features(h: int, w: int, stride: int = 1):
    X, Y = norm_coords(h, w, stride)
    cols = [np.ones_like(X), X, Y, X * X, Y * Y, X * Y]
    return np.stack([c.reshape(-1) for c in cols], axis=1)


def gram_cond(F: np.ndarray) -> float:
    G = (F.T @ F) / F.shape[0]
    return float(np.linalg.cond(G))


# --------------------------------------------------------------------------
# Range channels + per-image standardization / residualization
# --------------------------------------------------------------------------

def range_channels(img: np.ndarray):
    """img: (H,W,3) sRGB [0,1] -> L (Rec.709 luma), S (HSV saturation)."""
    L = (0.2126 * img[..., 0] + 0.7152 * img[..., 1]
         + 0.0722 * img[..., 2])
    mx = img.max(axis=-1)
    mn = img.min(axis=-1)
    S = np.where(mx > 1e-6, (mx - mn) / np.maximum(mx, 1e-6), 0.0)
    return L.astype(np.float64), S.astype(np.float64)


def standardize(v: np.ndarray, eps: float = 1e-6):
    return (v - v.mean()) / (v.std() + eps)


def residualize(E: np.ndarray, A: np.ndarray):
    """Orthogonalize semantic block E (P,6) against block A (P,k) via lstsq.
    Returns (E_resid, corr_before, corr_after) where corr_* is the (k,6)
    normalized cross-correlation of standardized columns."""

    def _corr(A_, E_):
        As = (A_ - A_.mean(0)) / (A_.std(0) + 1e-9)
        Es = (E_ - E_.mean(0)) / (E_.std(0) + 1e-9)
        return (As.T @ Es) / A_.shape[0]

    before = _corr(A, E)
    coef, *_ = np.linalg.lstsq(A, E, rcond=None)
    Er = E - A @ coef
    after = _corr(A, Er)
    return Er, before, after


# --------------------------------------------------------------------------
# Guided filter (edge-aware smoothing of basis channels; PLAN 1.3 line 73)
# --------------------------------------------------------------------------

def _box(a: np.ndarray, r: int):
    """Box mean with window 2r+1 (reflect padding), separable cumsum."""
    from scipy.ndimage import uniform_filter
    return uniform_filter(a, size=2 * r + 1, mode="reflect")


def guided_filter(guide: np.ndarray, src: np.ndarray, r: int = 32,
                  eps: float = 1e-3):
    mI = _box(guide, r)
    mP = _box(src, r)
    corr = _box(guide * src, r)
    var = _box(guide * guide, r) - mI * mI
    a = (corr - mI * mP) / (var + eps)
    b = mP - a * mI
    return _box(a, r) * guide + _box(b, r)


# --------------------------------------------------------------------------
# soft-IoU
# --------------------------------------------------------------------------

def soft_iou_minmax_np(a: np.ndarray, b: np.ndarray, eps: float = 1e-6):
    return float(np.minimum(a, b).sum() / (np.maximum(a, b).sum() + eps))


def soft_iou_prod_np(a: np.ndarray, b: np.ndarray, eps: float = 1e-6):
    inter = (a * b).sum()
    return float(inter / (a.sum() + b.sum() - inter + eps))


# --------------------------------------------------------------------------
# Per-mask L-BFGS fit
# --------------------------------------------------------------------------

def _forward(params: dict, Phi: torch.Tensor, readout: str):
    u = params["u"]
    w_dir = u / (u.norm() + 1e-12)
    alpha = torch.nn.functional.softplus(params["a_raw"])
    q = params["w0"] + alpha * (Phi @ w_dir)
    s = 3.0 * torch.tanh(q / 3.0)
    if readout == "monotone":
        m = torch.sigmoid(params["g"] * s + params["b"])
    elif readout == "bandpass":
        # logistic band sigma(k(s-mu+h)) - sigma(k(s-mu-h)): a passband with
        # a flat top, the single-band analogue of what the renderer's
        # M=12-Gaussian s-axis mixture can realize (a lone Gaussian cannot
        # form the plateau of a feathered ring; measured med IoU 0.78).
        h = 0.02 + 2.48 * torch.sigmoid(params["h_raw"])
        k = 1.0 + 39.0 * torch.sigmoid(params["k_raw"])
        m = (torch.sigmoid(k * (s - params["mu"] + h))
             - torch.sigmoid(k * (s - params["mu"] - h)))
    elif readout == "gauss":
        sigma = 0.05 + 2.95 * torch.sigmoid(params["sig_raw"])
        m = torch.exp(-0.5 * ((s - params["mu"]) / sigma) ** 2)
    else:
        raise ValueError(readout)
    return m, s, alpha


def _loss(params, Phi, t, readout):
    m, _, _ = _forward(params, Phi, readout)
    iou = torch.minimum(m, t).sum() / (torch.maximum(m, t).sum() + 1e-6)
    return 1.0 - iou


def _lsq_init(Phi_np: np.ndarray, t_np: np.ndarray):
    """Least-squares on logit(target) -> (w0, alpha, u)."""
    z = np.log(np.clip(t_np, 1e-3, 1 - 1e-3) /
               (1 - np.clip(t_np, 1e-3, 1 - 1e-3)))
    A = np.concatenate([np.ones((Phi_np.shape[0], 1)), Phi_np], axis=1)
    coef, *_ = np.linalg.lstsq(A, z, rcond=None)
    w0 = float(coef[0])
    v = coef[1:]
    a = float(np.linalg.norm(v))
    u = v / a if a > 1e-8 else np.ones_like(v) / np.sqrt(len(v))
    return w0, max(a, 1e-3), u


def softplus_inv(y: float) -> float:
    y = max(y, 1e-6)
    return float(y + np.log(-np.expm1(-y))) if y < 20 else y


def fit_mask(Phi_fit: np.ndarray, t_fit: np.ndarray,
             Phi_eval: np.ndarray, t_eval: np.ndarray,
             readout: str, seed: int = 0, n_random: int = 6,
             max_iter: int = 120,
             extra_starts: list[dict] | None = None) -> dict:
    """Fit one mask; returns best params + eval IoUs.

    Restarts: LSQ-informed + n_random random + caller-provided informed
    starts (e.g. centroid-radial). Tie-break (within 1e-4 loss): smallest
    alpha (degeneracy check for constant masks; PLAN: alpha init ~0,
    alpha=0 must stay reachable).
    """
    rng = np.random.default_rng(seed)
    D = Phi_fit.shape[1]
    Phi = torch.from_numpy(Phi_fit.astype(np.float64))
    t = torch.from_numpy(t_fit.astype(np.float64))

    w0_i, a_i, u_i = _lsq_init(Phi_fit, t_fit)
    s_lsq = 3.0 * np.tanh((w0_i + a_i * (Phi_fit @ u_i)) / 3.0)
    mu_w = float((s_lsq * t_fit).sum() / (t_fit.sum() + 1e-6))
    sd_w = float(np.sqrt((t_fit * (s_lsq - mu_w) ** 2).sum()
                         / (t_fit.sum() + 1e-6)) + 0.1)

    starts = []
    if readout == "monotone":
        starts.append({"w0": w0_i, "a": a_i, "u": u_i, "g": 1.0, "b": 0.0})
        for k in range(n_random):
            starts.append({"w0": 0.0, "a": 1.0,
                           "u": rng.normal(size=D), "g": float((-1) ** k * 3.0),
                           "b": 0.0})
        starts.append({"w0": 0.0, "a": 0.05,
                       "u": rng.normal(size=D), "g": 2.0, "b": 0.0})
    else:
        starts.append({"w0": w0_i, "a": a_i, "u": u_i,
                       "mu": mu_w, "sig": min(max(sd_w, 0.1), 2.0)})
        for k in range(n_random):
            starts.append({"w0": 0.0, "a": 1.0, "u": rng.normal(size=D),
                           "mu": float(rng.uniform(-2, 2)),
                           "sig": float(rng.uniform(0.2, 1.5))})
        starts.append({"w0": 0.0, "a": 0.05, "u": rng.normal(size=D),
                       "mu": 0.0, "sig": 0.5})
    for st in (extra_starts or []):
        starts.append(st)

    best = None
    for st in starts:
        params = {
            "w0": torch.tensor(st["w0"], dtype=torch.float64,
                               requires_grad=True),
            "a_raw": torch.tensor(softplus_inv(st["a"]), dtype=torch.float64,
                                  requires_grad=True),
            "u": torch.tensor(np.asarray(st["u"], dtype=np.float64),
                              requires_grad=True),
        }
        if readout == "monotone":
            params["g"] = torch.tensor(st["g"], dtype=torch.float64,
                                       requires_grad=True)
            params["b"] = torch.tensor(st["b"], dtype=torch.float64,
                                       requires_grad=True)
        else:
            params["mu"] = torch.tensor(st["mu"], dtype=torch.float64,
                                        requires_grad=True)
            if readout == "bandpass":
                hr = float(np.log(max((st["sig"] - 0.02) / 2.48, 1e-4)
                                  / max(1 - (st["sig"] - 0.02) / 2.48,
                                        1e-4)))
                params["h_raw"] = torch.tensor(hr, dtype=torch.float64,
                                               requires_grad=True)
                kr = float(np.log((8.0 - 1.0) / 39.0
                                  / (1 - (8.0 - 1.0) / 39.0)))
                params["k_raw"] = torch.tensor(kr, dtype=torch.float64,
                                               requires_grad=True)
            else:
                sr = float(np.log(max((st["sig"] - 0.05) / 2.95, 1e-4)
                                  / max(1 - (st["sig"] - 0.05) / 2.95,
                                        1e-4)))
                params["sig_raw"] = torch.tensor(sr, dtype=torch.float64,
                                                 requires_grad=True)
        opt = torch.optim.LBFGS(list(params.values()), max_iter=max_iter,
                                history_size=20, line_search_fn="strong_wolfe",
                                tolerance_grad=1e-9, tolerance_change=1e-11)

        def closure():
            opt.zero_grad()
            loss = _loss(params, Phi, t, readout)
            loss.backward()
            return loss

        try:
            opt.step(closure)
        except Exception:
            continue
        with torch.no_grad():
            loss = float(_loss(params, Phi, t, readout))
            alpha = float(torch.nn.functional.softplus(params["a_raw"]))
        cand = {"loss": loss, "alpha": alpha,
                "state": {k: v.detach().clone() for k, v in params.items()}}
        if (best is None or cand["loss"] < best["loss"] - 1e-4
                or (abs(cand["loss"] - best["loss"]) <= 1e-4
                    and cand["alpha"] < best["alpha"])):
            best = cand

    assert best is not None, "all restarts failed"
    # evaluate at full resolution
    with torch.no_grad():
        Phi_e = torch.from_numpy(Phi_eval.astype(np.float64))
        m_e, s_e, alpha = _forward(best["state"], Phi_e, readout)
    m_np = m_e.numpy()
    st = best["state"]
    # complement IoU keeps the metric meaningful for near-empty targets
    # (an all-zero target scores 0 under min/max IoU even at exact recovery)
    out = {
        "loss_fit": best["loss"],
        "iou_minmax": soft_iou_minmax_np(m_np, t_eval),
        "iou_minmax_comp": soft_iou_minmax_np(1 - m_np, 1 - t_eval),
        "mae": float(np.abs(m_np - t_eval).mean()),
        "iou_prod": soft_iou_prod_np(m_np, t_eval),
        "pred_std": float(m_np.std()),
        "pred_mean": float(m_np.mean()),
        "alpha": float(alpha),
        "w0": float(st["w0"]),
        "w_dir": (st["u"] / st["u"].norm()).numpy().tolist(),
    }
    if readout == "monotone":
        out["g"] = float(st["g"])
        out["b"] = float(st["b"])
    elif readout == "bandpass":
        out["mu"] = float(st["mu"])
        out["h"] = float(0.02 + 2.48 * torch.sigmoid(st["h_raw"]))
        out["k"] = float(1.0 + 39.0 * torch.sigmoid(st["k_raw"]))
    else:
        out["mu"] = float(st["mu"])
        out["sigma"] = float(0.05 + 2.95 * torch.sigmoid(st["sig_raw"]))
    return out


def radial_starts(Phi_fit: np.ndarray, t_fit: np.ndarray,
                  readout: str) -> list[dict]:
    """Centroid-radial informed starts. Assumes dir-columns start with
    [x, y, P2(x), P2(y), xy, ...]. Builds q ~ -((x-a)^2 + (y-b)^2) around the
    target centroid (x^2 = (2*P2(x)+1)/3), scaled so s spans ~[-2, 2]; for
    bandpass, mu/sigma from the target-weighted s statistics."""
    D = Phi_fit.shape[1]
    if D < 5:
        return []
    tw = t_fit.sum() + 1e-6
    a = float((t_fit * Phi_fit[:, 0]).sum() / tw)
    b = float((t_fit * Phi_fit[:, 1]).sum() / tw)
    v = np.zeros(D)
    v[0], v[1], v[2], v[3] = 2 * a, 2 * b, -2.0 / 3.0, -2.0 / 3.0
    u = v / np.linalg.norm(v)
    z = Phi_fit @ u
    alpha0 = 2.0 / (z.std() + 1e-6)
    w0 = -alpha0 * float(z.mean())
    s = 3.0 * np.tanh((w0 + alpha0 * z) / 3.0)
    mu_w = float((t_fit * s).sum() / tw)
    sd_w = float(np.sqrt((t_fit * (s - mu_w) ** 2).sum() / tw)) + 0.05
    out = []
    if readout == "bandpass":
        for sig in (min(max(sd_w, 0.1), 1.5), 0.3):
            out.append({"w0": w0, "a": alpha0, "u": u.copy(),
                        "mu": mu_w, "sig": sig})
    else:
        for g in (3.0, -3.0):
            out.append({"w0": w0, "a": alpha0, "u": u.copy(),
                        "g": g, "b": -g * mu_w})
    return out


def predict_mask(row: dict, Phi: np.ndarray, readout: str) -> np.ndarray:
    """Rebuild m-hat from a stored result row (for viz)."""
    w_dir = np.asarray(row["w_dir"])
    q = row["w0"] + row["alpha"] * (Phi @ w_dir)
    s = 3.0 * np.tanh(q / 3.0)
    if readout == "monotone":
        return 1.0 / (1.0 + np.exp(-(row["g"] * s + row["b"])))
    if readout == "bandpass":
        sg = lambda z: 1.0 / (1.0 + np.exp(-z))  # noqa: E731
        return (sg(row["k"] * (s - row["mu"] + row["h"]))
                - sg(row["k"] * (s - row["mu"] - row["h"])))
    return np.exp(-0.5 * ((s - row["mu"]) / row["sigma"]) ** 2)


# --------------------------------------------------------------------------
# CLIP dense semantic channels (MaskCLIP-style value-projection readout)
# --------------------------------------------------------------------------

class ClipDense:
    """openai/clip-vit-large-patch14-336 dense patch features projected to
    the joint space (MaskCLIP trick: final block attention replaced by
    identity over the value path), cosine sims against 6 fixed text anchors.
    """

    def __init__(self, device: str = "cuda:0",
                 model_id: str = "openai/clip-vit-large-patch14-336",
                 input_px: int = 672):
        from transformers import CLIPModel, CLIPProcessor
        self.device = torch.device(device)
        self.model = CLIPModel.from_pretrained(
            model_id, local_files_only=True).eval().to(self.device)
        self.proc = CLIPProcessor.from_pretrained(model_id,
                                                  local_files_only=True)
        self.input_px = input_px
        with torch.no_grad():
            embs = []
            for name in ANCHOR_ORDER:
                tok = self.proc(text=ANCHORS[name], return_tensors="pt",
                                padding=True).to(self.device)
                e = self.model.get_text_features(**tok)
                e = torch.nn.functional.normalize(e, dim=-1).mean(0)
                embs.append(torch.nn.functional.normalize(e, dim=-1))
            self.text = torch.stack(embs)          # (6, 768)

    @torch.no_grad()
    def sim_maps(self, img: np.ndarray) -> np.ndarray:
        """img: (H,W,3) sRGB [0,1] -> (g,g,6) float32 cosine sims,
        g = input_px/14."""
        from PIL import Image
        pil = Image.fromarray((np.clip(img, 0, 1) * 255).astype(np.uint8))
        pil = pil.resize((self.input_px, self.input_px),
                         Image.Resampling.LANCZOS)
        px = self.proc(images=pil, return_tensors="pt",
                       do_resize=False, do_center_crop=False)
        pv = px["pixel_values"].to(self.device)

        vm = self.model.vision_model
        # manual embedding with bicubic pos-emb interpolation (transformers
        # 4.36 CLIPVisionEmbeddings has no interpolate_pos_encoding kwarg)
        pe = vm.embeddings
        patch = pe.patch_embedding(pv)                 # (1, dim, g, g)
        g = patch.shape[-1]
        patch = patch.flatten(2).transpose(1, 2)       # (1, g*g, dim)
        cls_tok = pe.class_embedding.reshape(1, 1, -1)
        pos = pe.position_embedding.weight             # (577, dim)
        g0 = int(np.sqrt(pos.shape[0] - 1))
        grid_pos = pos[1:].reshape(1, g0, g0, -1).permute(0, 3, 1, 2)
        if g != g0:
            grid_pos = torch.nn.functional.interpolate(
                grid_pos, size=(g, g), mode="bicubic", align_corners=False)
        grid_pos = grid_pos.permute(0, 2, 3, 1).reshape(g * g, -1)
        hidden = torch.cat([cls_tok, patch], dim=1) + torch.cat(
            [pos[:1], grid_pos], dim=0).unsqueeze(0)
        hidden = vm.pre_layrnorm(hidden)
        for layer in vm.encoder.layers[:-1]:
            hidden = layer(hidden, None, None)[0]
        last = vm.encoder.layers[-1]
        hn = last.layer_norm1(hidden)
        v = last.self_attn.v_proj(hn)
        attn_out = last.self_attn.out_proj(v)      # identity attention
        hidden = hidden + attn_out
        hidden = hidden + last.mlp(last.layer_norm2(hidden))
        hidden = vm.post_layernorm(hidden)
        feats = self.model.visual_projection(hidden[:, 1:])   # (1, g*g, 768)
        feats = torch.nn.functional.normalize(feats, dim=-1)
        sims = feats[0] @ self.text.T                          # (g*g, 6)
        g = int(np.sqrt(sims.shape[0]))
        return sims.reshape(g, g, 6).float().cpu().numpy()
