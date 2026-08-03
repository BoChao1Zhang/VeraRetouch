"""E1 per-LUT direct Adam overfit engine (PLAN v2, Stage-0 / first level).

Protocol items implemented (PLAN section "第一级" steps 3-4):
  * mu init      : |f(x)-x|-weighted k-means on train colors (per LUT)
  * b init       : per-cluster weighted mean residual (y - x); M=0, G=I, g=0
                   -> f(x) = x + sum w_i b_i at step 0 (identity passthrough)
  * sigma anneal : Cholesky diagonal lower bound decays 0.30 -> 0.02 over the
                   first 60% of steps (log-space clamp after every step), then
                   floor 0.005 (numerical guard) for the rest
  * density ctrl : every 500 steps, dead primitives (probe weight-mass share
                   < 0.1/N of total) are relocated to the highest-DeltaE-proxy
                   residual probe colors; opacity/M/b reset; Adam state of the
                   relocated slices zeroed
  * loss         : L_rec (L1) on unclamped f(x)
  * report       : effective primitive count (alive share), relocation stats
                   (PLAN red flag: alive < 0.8N -> fix the optimizer first)

Batched over B LUTs at a fixed N (independent params/GT per LUT).
"""

from __future__ import annotations

import time

import numpy as np
import torch

from model.glut_repro.model import BatchedGLUT
from model.glut_repro.losses import l_rec


# ---------------------------------------------------------------------------
# Weighted k-means init (CPU, sklearn)
# ---------------------------------------------------------------------------

def weighted_kmeans_init(colors: np.ndarray, gt: np.ndarray, n: int,
                         sample: int = 24576, seed: int = 0
                         ) -> tuple[np.ndarray, np.ndarray]:
    """Returns (mu (n,3), bias (n,3)) for one LUT.

    k-means on a residual-weighted subsample of train colors; bias = weighted
    mean residual (gt - x) of each cluster (zero for empty clusters).
    """
    from sklearn.cluster import KMeans
    rng = np.random.default_rng(seed)
    res = gt.astype(np.float32) - colors
    wall = np.linalg.norm(res, axis=1) + 1e-3
    p = wall / wall.sum()
    idx = rng.choice(colors.shape[0], size=min(sample, colors.shape[0]),
                     replace=False, p=p)
    xs, ws, rs = colors[idx], wall[idx], res[idx]
    km = KMeans(n_clusters=n, n_init=1, max_iter=25,
                random_state=seed).fit(xs, sample_weight=ws)
    mu = km.cluster_centers_.astype(np.float32)
    bias = np.zeros((n, 3), dtype=np.float32)
    for k in range(n):
        m = km.labels_ == k
        if m.any():
            wk = ws[m]
            bias[k] = (rs[m] * wk[:, None]).sum(0) / wk.sum()
    return np.clip(mu, 0.0, 1.0), bias


# ---------------------------------------------------------------------------
# Fit engine
# ---------------------------------------------------------------------------

class E1Fitter:
    SIGMA_HI = 0.30
    SIGMA_LO = 0.02
    SIGMA_FINAL_FLOOR = 0.005
    DENSITY_EVERY = 500
    DEAD_SHARE = 0.1          # dead if mass share < DEAD_SHARE / N

    def __init__(self, colors: np.ndarray, gts: np.ndarray, n_gaussians: int,
                 device: str = "cuda", seed: int = 0, steps: int = 3000,
                 bs: int = 8192, lr: float = 5e-3, probe: int = 32768):
        """colors: (P,3) f32 shared; gts: (B,P,3) f16 per-LUT GT."""
        torch.manual_seed(seed)
        self.device = device
        self.steps = steps
        self.bs = bs
        self.B, self.P = gts.shape[0], colors.shape[0]
        self.N = n_gaussians
        self.x_all = torch.from_numpy(colors).to(device)
        self.y_all = torch.from_numpy(gts).to(device)
        self.model = BatchedGLUT(self.B, n_gaussians).to(device)
        # per-LUT weighted k-means init (CPU)
        mus, biases = [], []
        for b in range(self.B):
            mu, bias = weighted_kmeans_init(
                colors, gts[b], n_gaussians, seed=seed + b)
            mus.append(mu)
            biases.append(bias)
        self.model.init_e1_residual(
            torch.from_numpy(np.stack(mus)).to(device),
            torch.from_numpy(np.stack(biases)).to(device),
            sigma=self.SIGMA_HI)
        self.opt = torch.optim.Adam(self.model.parameters(), lr=lr)
        self.sched = torch.optim.lr_scheduler.CosineAnnealingLR(
            self.opt, T_max=steps, eta_min=lr * 0.01)
        self.g = torch.Generator(device=device)
        self.g.manual_seed(seed + 7)
        pr = torch.randperm(self.P, generator=torch.Generator().manual_seed(
            seed + 11))[:probe]
        self.probe_idx = pr.to(device)
        self.stats = {"relocations": [], "n_relocated_total": 0}
        self.log: list[dict] = []

    # -- sigma annealing ----------------------------------------------------
    def _sigma_floor(self, step: int) -> float:
        t_anneal = 0.6 * self.steps
        if step <= t_anneal:
            frac = step / t_anneal
            return float(self.SIGMA_HI *
                         (self.SIGMA_LO / self.SIGMA_HI) ** frac)
        return self.SIGMA_FINAL_FLOOR

    @torch.no_grad()
    def _apply_sigma_clamp(self, step: int) -> None:
        lo = float(np.log(self._sigma_floor(step)))
        hi = float(np.log(2.0))
        self.model.chol_log_diag.clamp_(lo, hi)

    # -- density control ----------------------------------------------------
    @torch.no_grad()
    def _density_control(self, step: int) -> None:
        m = self.model
        xp = self.x_all[self.probe_idx].unsqueeze(0).expand(self.B, -1, -1)
        yp = torch.gather(self.y_all, 1, self.probe_idx.view(1, -1, 1)
                          .expand(self.B, -1, 3)).float()
        mass = m.weight_mass(xp)                          # (B,N)
        pred = m.predict(xp)
        err = (pred - yp).abs().mean(-1)                  # (B,Pp) L1 proxy
        dead = mass < (self.DEAD_SHARE / self.N)          # (B,N)
        n_dead = int(dead.sum())
        if n_dead == 0:
            self.stats["relocations"].append({"step": step, "n": 0})
            return
        sigma_new = float(np.log(max(self._sigma_floor(step) * 1.5,
                                     self.SIGMA_LO)))
        adam = self.opt.state
        for b in range(self.B):
            di = torch.nonzero(dead[b]).flatten()
            if di.numel() == 0:
                continue
            k = di.numel()
            top = torch.topk(err[b], k).indices           # k worst probe pts
            xstar = xp[b, top]                            # (k,3)
            jitter = torch.randn(k, 3, device=self.device,
                                 generator=self.g) * 0.01
            m.mu[b, di] = (xstar + jitter).clamp(0, 1)
            m.chol_log_diag[b, di] = sigma_new
            m.chol_off[b, di] = 0.0
            m.opacity_raw[b, di] = 1.0
            m.M[b, di] = 0.0
            # payload: target residual at the relocation point (global adds x)
            m.b[b, di] = yp[b, top] - xstar
            # zero Adam state on relocated slices
            for p in (m.mu, m.chol_log_diag, m.chol_off, m.opacity_raw,
                      m.M, m.b):
                st = adam.get(p)
                if st and "exp_avg" in st:
                    st["exp_avg"][b, di] = 0
                    st["exp_avg_sq"][b, di] = 0
        self.stats["relocations"].append({"step": step, "n": n_dead})
        self.stats["n_relocated_total"] += n_dead

    # -- main loop ----------------------------------------------------------
    def fit(self, log_every: int = 500) -> dict:
        t0 = time.time()
        for step in range(self.steps):
            if step > 0 and step % self.DENSITY_EVERY == 0:
                self._density_control(step)
            idx = torch.randint(0, self.P, (self.B, self.bs),
                                device=self.device, generator=self.g)
            x = self.x_all[idx]
            y = torch.gather(self.y_all, 1,
                             idx.unsqueeze(-1).expand(-1, -1, 3)).float()
            loss = l_rec(self.model(x), y)
            self.opt.zero_grad(set_to_none=True)
            loss.backward()
            self.opt.step()
            self.sched.step()
            self._apply_sigma_clamp(step)
            if step % log_every == 0 or step == self.steps - 1:
                self.log.append({"step": step, "loss": float(loss.detach()),
                                 "sigma_floor": self._sigma_floor(step),
                                 "sec": round(time.time() - t0, 1)})
        # final alive stats
        with torch.no_grad():
            xp = self.x_all[self.probe_idx].unsqueeze(0).expand(self.B, -1, -1)
            mass = self.model.weight_mass(xp)
            alive = (mass >= (self.DEAD_SHARE / self.N)).sum(-1)  # (B,)
        self.stats["alive_final"] = alive.cpu().tolist()
        self.stats["alive_frac"] = [a / self.N for a in
                                    self.stats["alive_final"]]
        return self.stats

    @torch.no_grad()
    def predict_colors(self, colors: np.ndarray,
                       chunk: int = 1 << 16) -> np.ndarray:
        m = self.model.eval()
        out = np.empty((self.B, colors.shape[0], 3), dtype=np.float32)
        for i in range(0, colors.shape[0], chunk):
            j = min(i + chunk, colors.shape[0])
            x = torch.from_numpy(colors[i:j]).to(self.device)
            out[:, i:j] = m.predict(
                x.unsqueeze(0).expand(self.B, -1, -1)).cpu().numpy()
        return out
