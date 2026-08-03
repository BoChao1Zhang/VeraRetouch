"""E1b: truncated SVD of the 3,522 production-preset 33^3 LUT library.

Rank-r reconstruction -> per-LUT mean dE00 over the 33^3 grid -> cross-LUT
p50/p90/p99 curves. Pre-registered prediction: knee at r in [32, 64].

Method
- X: (3522, 107811) float64, rows = flattened (33,33,33,3) LUT tables.
- Truncated SVD via Gram eigendecomposition: G = X X^T, eigh -> U;
  rank-r reconstruction X_r = U_r U_r^T X (exact economy-SVD truncation).
- dE00 computed on GPU (torch port of cubelib's sRGB->Lab->CIEDE2000),
  validated against tools/cube/cubelib.delta_e00 before use (gate 5e-3).
- Primary metric: per-LUT mean dE00 over the 35,937 grid points; report
  p50/p90/p99 across the 3,522 LUTs. Secondary: pooled per-point percentiles.
- Variants: plain (primary, spec wording "flatten -> truncated SVD") and
  mean-centered (secondary, PCA convention).

Runtime: minutes on one GPU.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import torch

REPO = Path("/home/bc/VeraRetouch")
sys.path.insert(0, str(REPO / "tools" / "cube"))
import cubelib  # noqa: E402

NPY_DIR = Path("/var/cache/veradata/dcube/npy33")
USED = REPO / "experiments/tooling-wave1/cube/inventory/used_presets.txt"
OUT = REPO / "experiments/E1b_svd_20260803"
RANKS = [1, 2, 4, 8, 12, 16, 24, 32, 40, 48, 56, 64, 80, 96, 128,
         160, 192, 256, 384, 512]
GATE_P90, GATE_P99 = 1.0, 2.0  # E1-analogous gate on per-LUT mean dE00


# ---------------------------------------------------------------------------
# Torch color: sRGB -> Lab -> CIEDE2000 (constants pulled from colour-science)
# ---------------------------------------------------------------------------

def _colour_constants():
    import colour
    cs = colour.models.RGB_COLOURSPACE_sRGB
    M = np.asarray(cs.matrix_RGB_to_XYZ, dtype=np.float64)
    wp_xy = np.asarray(cs.whitepoint, dtype=np.float64)
    XYZ_n = np.asarray(colour.xy_to_XYZ(wp_xy), dtype=np.float64)
    return M, XYZ_n


def srgb_to_lab_t(rgb: torch.Tensor, M: torch.Tensor,
                  XYZ_n: torch.Tensor) -> torch.Tensor:
    """rgb: (...,3) float64 in [0,1] gamma sRGB -> Lab (D65)."""
    c = rgb.clamp(0.0, 1.0)
    lin = torch.where(c <= 0.04045, c / 12.92,
                      ((c + 0.055) / 1.055) ** 2.4)
    xyz = lin @ M.T
    t = xyz / XYZ_n
    d = 6.0 / 29.0
    f = torch.where(t > d ** 3, t ** (1.0 / 3.0),
                    t / (3 * d * d) + 4.0 / 29.0)
    fx, fy, fz = f[..., 0], f[..., 1], f[..., 2]
    L = 116.0 * fy - 16.0
    a = 500.0 * (fx - fy)
    b = 200.0 * (fy - fz)
    return torch.stack([L, a, b], dim=-1)


def ciede2000_t(lab1: torch.Tensor, lab2: torch.Tensor) -> torch.Tensor:
    """CIEDE2000 (kL=kC=kH=1). lab*: (...,3) float64."""
    L1, a1, b1 = lab1[..., 0], lab1[..., 1], lab1[..., 2]
    L2, a2, b2 = lab2[..., 0], lab2[..., 1], lab2[..., 2]
    C1 = torch.hypot(a1, b1)
    C2 = torch.hypot(a2, b2)
    Cbar = 0.5 * (C1 + C2)
    c7 = Cbar ** 7
    G = 0.5 * (1.0 - torch.sqrt(c7 / (c7 + 25.0 ** 7)))
    a1p = (1.0 + G) * a1
    a2p = (1.0 + G) * a2
    C1p = torch.hypot(a1p, b1)
    C2p = torch.hypot(a2p, b2)
    h1p = torch.rad2deg(torch.atan2(b1, a1p)) % 360.0
    h2p = torch.rad2deg(torch.atan2(b2, a2p)) % 360.0
    dLp = L2 - L1
    dCp = C2p - C1p
    dh = h2p - h1p
    dh = torch.where(dh > 180.0, dh - 360.0, dh)
    dh = torch.where(dh < -180.0, dh + 360.0, dh)
    dh = torch.where((C1p * C2p) == 0.0, torch.zeros_like(dh), dh)
    dHp = 2.0 * torch.sqrt(C1p * C2p) * torch.sin(torch.deg2rad(dh) / 2.0)
    Lbp = 0.5 * (L1 + L2)
    Cbp = 0.5 * (C1p + C2p)
    hsum = h1p + h2p
    hdiff = torch.abs(h1p - h2p)
    hbp = torch.where(hdiff > 180.0,
                      torch.where(hsum < 360.0, (hsum + 360.0) / 2.0,
                                  (hsum - 360.0) / 2.0),
                      hsum / 2.0)
    hbp = torch.where((C1p * C2p) == 0.0, hsum, hbp)  # colour convention
    T = (1.0 - 0.17 * torch.cos(torch.deg2rad(hbp - 30.0))
         + 0.24 * torch.cos(torch.deg2rad(2.0 * hbp))
         + 0.32 * torch.cos(torch.deg2rad(3.0 * hbp + 6.0))
         - 0.20 * torch.cos(torch.deg2rad(4.0 * hbp - 63.0)))
    dtheta = 30.0 * torch.exp(-(((hbp - 275.0) / 25.0) ** 2))
    cb7 = Cbp ** 7
    RC = 2.0 * torch.sqrt(cb7 / (cb7 + 25.0 ** 7))
    SL = 1.0 + (0.015 * (Lbp - 50.0) ** 2
                / torch.sqrt(20.0 + (Lbp - 50.0) ** 2))
    SC = 1.0 + 0.045 * Cbp
    SH = 1.0 + 0.015 * Cbp * T
    RT = -torch.sin(torch.deg2rad(2.0 * dtheta)) * RC
    return torch.sqrt((dLp / SL) ** 2 + (dCp / SC) ** 2 + (dHp / SH) ** 2
                      + RT * (dCp / SC) * (dHp / SH))


def validate_de00(device: torch.device, M: torch.Tensor, XYZ_n: torch.Tensor,
                  n: int = 200_000, seed: int = 0) -> float:
    rng = np.random.default_rng(seed)
    a = rng.random((n, 3))
    b = np.clip(a + rng.normal(0, 0.08, (n, 3)), 0, 1)
    ref = cubelib.delta_e00(a, b)
    ta = torch.from_numpy(a).to(device)
    tb = torch.from_numpy(b).to(device)
    got = ciede2000_t(srgb_to_lab_t(ta, M, XYZ_n),
                      srgb_to_lab_t(tb, M, XYZ_n)).cpu().numpy()
    return float(np.abs(got - ref).max())


# ---------------------------------------------------------------------------
# Main pipeline
# ---------------------------------------------------------------------------

def load_matrix(ids: list[str], workers: int = 16) -> np.ndarray:
    from concurrent.futures import ThreadPoolExecutor
    X = np.empty((len(ids), 33 * 33 * 33 * 3), dtype=np.float32)

    def _load(i):
        X[i] = np.load(NPY_DIR / f"{ids[i]}.npy").reshape(-1)
    with ThreadPoolExecutor(max_workers=workers) as ex:
        list(ex.map(_load, range(len(ids))))
    return X


def run_variant(Xt: torch.Tensor, lab_orig: torch.Tensor, name: str,
                M: torch.Tensor, XYZ_n: torch.Tensor, ranks: list[int],
                center: bool) -> dict:
    """Xt: (m, n) float64 on GPU. Returns metrics dict; also per-rank per-LUT
    mean dE00 for the last-needed analysis stored in the dict."""
    m = Xt.shape[0]
    mu = Xt.mean(dim=0, keepdim=True) if center else torch.zeros(
        (1, Xt.shape[1]), dtype=Xt.dtype, device=Xt.device)
    Xc = Xt - mu
    G = Xc @ Xc.T                                  # (m, m) float64
    evals, U = torch.linalg.eigh(G)                # ascending
    evals = torch.flip(evals, dims=[0]).clamp_min(0.0)
    U = torch.flip(U, dims=[1])
    svals = torch.sqrt(evals)
    energy = (evals / evals.sum()).cumsum(0)

    per_rank = {}
    curves = {"rank": [], "p50": [], "p90": [], "p99": [], "max": [],
              "pooled_p50": [], "pooled_p90": [], "pooled_p99": []}
    for r in ranks:
        Ur = U[:, :r]
        Xr = (Ur @ (Ur.T @ Xc) + mu).clamp(0.0, 1.0)
        lab_r = srgb_to_lab_t(Xr.reshape(m, -1, 3), M, XYZ_n)
        de = ciede2000_t(lab_orig, lab_r)          # (m, 35937)
        per_lut_mean = de.mean(dim=1)
        per_rank[r] = per_lut_mean.cpu().numpy()
        q = torch.quantile(per_lut_mean,
                           torch.tensor([0.5, 0.9, 0.99], dtype=Xt.dtype,
                                        device=Xt.device))
        # pooled per-point percentiles (sample 8M points to bound sort cost)
        flat = de.reshape(-1)
        idx = torch.randint(flat.numel(), (8_000_000,), device=flat.device)
        pooled = torch.quantile(flat[idx],
                                torch.tensor([0.5, 0.9, 0.99], dtype=Xt.dtype,
                                             device=Xt.device))
        curves["rank"].append(r)
        curves["p50"].append(float(q[0]))
        curves["p90"].append(float(q[1]))
        curves["p99"].append(float(q[2]))
        curves["max"].append(float(per_lut_mean.max()))
        curves["pooled_p50"].append(float(pooled[0]))
        curves["pooled_p90"].append(float(pooled[1]))
        curves["pooled_p99"].append(float(pooled[2]))
        print(f"[{name}] r={r:4d} p50={q[0]:.4f} p90={q[1]:.4f} "
              f"p99={q[2]:.4f} max={per_lut_mean.max():.4f}", flush=True)

    # knee: smallest r meeting the E1-analogous gate
    knee = None
    for i, r in enumerate(curves["rank"]):
        if curves["p90"][i] < GATE_P90 and curves["p99"][i] < GATE_P99:
            knee = r
            break
    return {
        "curves": curves,
        "knee_rank_gate": knee,
        "singular_values": svals.cpu().numpy().tolist(),
        "energy_cumsum_first64": energy[:64].cpu().numpy().tolist(),
        "rank_at_energy": {
            str(tau): int((energy < tau).sum().item()) + 1
            for tau in (0.9, 0.95, 0.99, 0.999)},
        "_per_rank_per_lut_mean": per_rank,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--smoke", action="store_true",
                    help="200 LUTs, ranks [1,8,32] only")
    args = ap.parse_args()
    t0 = time.time()
    dev = torch.device(args.device)

    used = [line.strip() for line in open(USED) if line.strip()]
    ids = [cubelib.preset_slug(p) for p in used]
    missing = [i for i in ids if not (NPY_DIR / f"{i}.npy").exists()]
    assert not missing, f"missing npy for {len(missing)} ids: {missing[:5]}"
    ranks = RANKS
    if args.smoke:
        ids = ids[::18][:200]
        ranks = [1, 8, 32]
    print(f"presets: {len(ids)}", flush=True)

    X = load_matrix(ids)
    Mnp, XYZn_np = _colour_constants()
    M = torch.from_numpy(Mnp).to(dev)
    XYZ_n = torch.from_numpy(XYZn_np).to(dev)

    de_gate = validate_de00(dev, M, XYZ_n)
    print(f"torch dE00 vs cubelib max|diff| = {de_gate:.2e}", flush=True)
    assert de_gate < 5e-3, "torch CIEDE2000 port disagrees with cubelib"

    Xt = torch.from_numpy(X).to(dev, dtype=torch.float64)
    lab_orig = srgb_to_lab_t(Xt.reshape(len(ids), -1, 3), M, XYZ_n)

    out = {"n_presets": len(ids), "ranks": ranks,
           "de00_port_max_abs_diff": de_gate,
           "gate": {"p90": GATE_P90, "p99": GATE_P99},
           "eval_basis": "33^3 grid points, per-LUT mean dE00",
           "variants": {}}
    per_rank_store = {}
    for name, center in [("plain", False), ("centered", True)]:
        res = run_variant(Xt, lab_orig, name, M, XYZ_n, ranks, center)
        per_rank_store[name] = res.pop("_per_rank_per_lut_mean")
        out["variants"][name] = res

    out["wall_seconds"] = round(time.time() - t0, 1)
    OUT.mkdir(parents=True, exist_ok=True)
    suffix = "_smoke" if args.smoke else ""
    with open(OUT / f"metrics{suffix}.json", "w") as fh:
        json.dump(out, fh, indent=2)
    np.savez_compressed(
        OUT / f"per_lut_mean_de00{suffix}.npz",
        ids=np.array(ids),
        **{f"{v}_r{r}": arr for v, d in per_rank_store.items()
           for r, arr in d.items()})

    # config snapshot
    cfg = {
        "git_commit": subprocess.run(
            ["git", "-C", str(REPO), "rev-parse", "HEAD"],
            capture_output=True, text=True).stdout.strip(),
        "npy_dir": str(NPY_DIR),
        "used_presets": str(USED),
        "n_presets": len(ids),
        "ranks": ranks,
        "dtype": "float64 (Gram eigh + reconstruction + dE00)",
        "device": args.device,
        "torch": torch.__version__,
        "numpy": np.__version__,
        "argv": sys.argv,
    }
    with open(OUT / "config" / f"run{suffix}.json", "w") as fh:
        json.dump(cfg, fh, indent=2)
    print(f"done in {out['wall_seconds']}s -> {OUT}/metrics{suffix}.json",
          flush=True)


if __name__ == "__main__":
    main()
