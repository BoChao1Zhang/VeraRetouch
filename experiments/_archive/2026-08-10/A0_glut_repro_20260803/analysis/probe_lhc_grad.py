"""Numerical proof of the two L_hc defects (task 1, root-cause evidence).

`losses.l_hc` computes  L = mean( C_gt * (1 - <h_pred, h_gt>) ) in ABSOLUTE
CIELab units with
    h_pred = (a_p, b_p) / C_p,   C_p = sqrt(a_p^2 + b_p^2 + 1e-6)
so the analytic per-sample gradient is
    dL/d(a_p,b_p) = -(C_gt / C_p) * (h_gt - cos_h * h_pred),
    |dL/d(a_p,b_p)| = C_gt * sin(delta_h) / C_p .

Two independent defects follow:
  D1  1/C_pred singularity: C_p floors at 1e-3 (the eps inside the sqrt), so a
      near-neutral prediction emits a gradient up to ~1e5x a typical one.
  D2  unit mismatch: even away from the singularity, |dL_hc/d rgb| ~ 1e2..1e3
      because a,b are O(100) and d(a,b)/d(rgb) is O(1e2..1e3), while
      |dL_rec/d rgb| = O(1).  With lambda = 10 the auxiliary term outweighs
      the reconstruction objective by orders of magnitude, and Adam's
      per-parameter normalisation makes the tiny L_rec contribution
      numerically irrelevant -> the `full` arm optimises HUE ONLY.

Measurements are taken w.r.t. the PRE-CLAMP model output f (the gradient
backprop actually delivers, so the clamp mask is honoured) at two states:
  * GLUT-original init (f = 2x, ~88% of samples clamped);
  * a converged A0 rec-arm checkpoint (representative of the whole run).

Output: analysis/lhc_grad_probe.json
"""

from __future__ import annotations

import json
import os
import sys

import numpy as np
import torch

_REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
if _REPO not in sys.path:
    sys.path.insert(0, _REPO)

from model.glut_repro import data, losses  # noqa: E402
from model.glut_repro.ablate_a0 import l_hc_stable  # noqa: E402
from model.glut_repro.model import BatchedGLUT  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
CKPT = os.path.join(HERE, "..", "runs", "rec", "ckpt_chunk000.pt")


def grad_wrt_f(loss_fn, w: float, f: torch.Tensor, gt: torch.Tensor):
    """d(w*loss(clamp(f)))/df -- the true training gradient (clamp masked)."""
    fv = f.detach().clone().requires_grad_(True)
    lv = w * loss_fn(fv.clamp(0, 1), gt)
    (g,) = torch.autograd.grad(lv, fv)
    return g, float(lv)


def state_stats(f: torch.Tensor, y: torch.Tensor, tag: str) -> dict:
    """Per-sample gradient magnitudes + batch-level term balance."""
    P = f.shape[0]
    lab_p = losses.srgb_to_lab_torch(f.clamp(0, 1))
    cp = torch.sqrt(lab_p[..., 1] ** 2 + lab_p[..., 2] ** 2)
    unclamped = ((f > 0) & (f < 1)).all(-1)
    out: dict = {"tag": tag, "n_samples": int(P), "occupancy": {
        "frac_unclamped": float(unclamped.float().mean()),
        "frac_pred_chroma_lt_1": float((cp < 1.0).float().mean()),
        "frac_singular_active(C<1 & unclamped)":
            float(((cp < 1.0) & unclamped).float().mean()),
        "singular_samples_per_1024_batch":
            float(((cp < 1.0) & unclamped).float().mean()) * 1024,
    }}

    # per-sample gradient magnitude, binned by predicted chroma
    cpn = cp.detach().cpu().numpy()
    edges = np.array([0.0, 0.1, 1.0, 3.0, 10.0, 30.0, 1e9])
    for name, fn in (("paper", losses.l_hc), ("stable", l_hc_stable)):
        g, lv = grad_wrt_f(fn, 1.0, f, y)
        gm = (g.norm(dim=-1) * P).detach().cpu().numpy()   # un-average
        bins = []
        for i in range(len(edges) - 1):
            sel = (cpn >= edges[i]) & (cpn < edges[i + 1]) & \
                  unclamped.cpu().numpy()
            if sel.sum() == 0:
                continue
            bins.append({"C_pred": f"[{edges[i]},{edges[i+1]})",
                         "n_unclamped": int(sel.sum()),
                         "grad_median": float(np.median(gm[sel])),
                         "grad_max": float(gm[sel].max())})
        out[f"per_sample_grad_{name}"] = {
            "loss_value": lv,
            "median": float(np.median(gm)),
            "p99.9": float(np.percentile(gm, 99.9)),
            "max": float(gm.max()),
            "max_over_median": float(gm.max() / max(np.median(gm), 1e-12)),
            "bins_by_pred_chroma_unclamped_only": bins,
        }

    # batch-level: how big is each objective term's gradient (bs=1024 draws)
    bs, n_draw = 1024, 300
    acc = {k: [] for k in ("L_rec", "10*L_hc_paper", "10*L_hc_stable",
                           "0.001*R_sparse_equiv")}
    gen = torch.Generator(device=f.device).manual_seed(0)
    for _ in range(n_draw):
        sel = torch.randint(0, P, (bs,), device=f.device, generator=gen)
        fb, yb = f[sel], y[sel]
        for nm, fn, w in (("L_rec", losses.l_rec, 1.0),
                          ("10*L_hc_paper", losses.l_hc, 10.0),
                          ("10*L_hc_stable", l_hc_stable, 10.0)):
            g, _ = grad_wrt_f(fn, w, fb, yb)
            acc[nm].append(float(g.norm()))
    del acc["0.001*R_sparse_equiv"]
    bal = {}
    for nm, lst in acc.items():
        a = np.asarray(lst)
        bal[nm] = {"p50": float(np.median(a)), "p99": float(np.percentile(a, 99)),
                   "max": float(a.max()),
                   "max_over_p50": float(a.max() / max(np.median(a), 1e-12))}
    ref = bal["L_rec"]["p50"]
    for nm in bal:
        bal[nm]["ratio_to_L_rec_p50"] = bal[nm]["p50"] / ref
    bal["_note"] = ("gradient of each objective term w.r.t. the pre-clamp "
                    "model output f, batch size 1024, 300 draws")
    out["batch_grad_norm_wrt_f"] = bal
    return out


def main() -> None:
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    lut_id = "e18__e18_001615"          # median-difficulty LUT of the A0 set
    gt_all = np.load(data._cache_path("a0", lut_id, "train"))
    colors = data.train_colors()
    rng = np.random.default_rng(0)
    idx = rng.choice(colors.shape[0], 1 << 19, replace=False)
    x = torch.from_numpy(colors[idx]).to(dev)
    y = torch.from_numpy(gt_all[idx].astype(np.float32)).to(dev)

    res: dict = {"lut_id": lut_id}

    # (1) GLUT-original init, f = 2x
    m = BatchedGLUT(1, 32).to(dev)
    with torch.no_grad():
        f0 = m(x.unsqueeze(0))[0]
    res["at_glut_init"] = state_stats(f0, y, "glut_init(f=2x)")

    # (2) converged rec-arm checkpoint for the same LUT
    ck = torch.load(CKPT, map_location=dev, weights_only=False)
    ids = ck["lut_ids"]
    if lut_id in ids:
        k = ids.index(lut_id)
        mm = BatchedGLUT(len(ids), ck["n_gaussians"]).to(dev)
        mm.load_state_dict(ck["state_dict"])
        with torch.no_grad():
            fk = mm(x.unsqueeze(0).expand(len(ids), -1, -1))[k]
        res["at_rec_converged"] = state_stats(fk, y, "rec_arm_converged")
        res["at_rec_converged"]["ckpt"] = os.path.relpath(CKPT, _REPO)
    else:
        res["at_rec_converged"] = {"error": f"{lut_id} not in {CKPT}"}

    # (3) lambda calibration: what weight makes 10*L_hc a perturbation?
    for state in ("at_glut_init", "at_rec_converged"):
        b = res.get(state, {}).get("batch_grad_norm_wrt_f")
        if not b:
            continue
        res[state]["lambda_for_10pct_of_L_rec"] = {
            "paper_formula": 10.0 * 0.1 / b["10*L_hc_paper"]["ratio_to_L_rec_p50"],
            "chroma_floored": 10.0 * 0.1 / b["10*L_hc_stable"]["ratio_to_L_rec_p50"],
        }

    with open(os.path.join(HERE, "lhc_grad_probe.json"), "w") as f:
        json.dump(res, f, indent=1)
    print(json.dumps(res, indent=1))


if __name__ == "__main__":
    main()
