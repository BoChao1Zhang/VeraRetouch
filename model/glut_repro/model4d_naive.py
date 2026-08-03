"""G3 / Gate D3 — naive 4D GLUT vs the R-2 minimal anti-collapse arm.

Purpose (PLAN v2 section 1.2 + section 5 "Gate D3"): decide empirically whether the
NAIVE 4D extension really walks the zero-cost escape channel under a pure
reconstruction loss, i.e. whether R-2 (mu_s anchoring + bounded sigma_s) and
R-3 must be on from day one.

The 3D reproduction (model.py, A0 anchor) is NOT touched.

--------------------------------------------------------------------- geometry
Addressing is 4D, payload is unchanged (M_i in R^{3x3}, b_i in R^3, global
G in R^{3x3}, g in R^3 -- all acting on RGB only):

    u        = (s, r, g, b)                      <- s FIRST, see below
    p_i(u)   = (2 pi)^{-2} |Sigma_i|^{-1/2} exp(-1/2 (u-mu_i)^T Sigma_i^{-1} (u-mu_i))
    w_i(u)   = p_i(u) o_i / (sum_j p_j(u) o_j + eps)        eps = 1e-6
    f(x,s)   = sum_i w_i(u) (M_i x + b_i) + (G x + g)

Sigma_i = L_i L_i^T with L_i lower triangular 4x4.  **s is placed FIRST in the
coordinate order on purpose**: then Var(s) = L[0,0]^2 exactly, so L[0,0] IS the
marginal sigma_s.  That is the quantity PLAN section 1.2 defines the degenerate set
D over ("all Gaussians share one s marginal"), the quantity R-2 bounds, and the
quantity this experiment plots.  With s last it would be
L[3,0]^2+L[3,1]^2+L[3,2]^2+L[3,3]^2 and neither bounding nor logging it would
mean what the plan says.  The three sub-diagonal entries of column 0
(L[1,0], L[2,0], L[3,0]) are the s-RGB cross-covariances -- kept, never
factorized (R-2 explicitly requires "保留 s–RGB 交叉协方差（不可因子分解）").

------------------------------------------------------------------------- arms
arm="naive"     mu_s LEARNABLE, initialised to 0.5 for every Gaussian, and
                sigma_s = exp(raw), free and unbounded (bare exp).
                Initialising every mu_s to the same value puts step 0 exactly
                ON the degenerate set D -- the symmetric saddle of PLAN 1.2
                item 1.  This is deliberate: it is the initialisation a naive
                "just copy GLUT's isotropic init into 4D" implementation gets.
                (Bare exp is a project red line for production arms.  This arm
                exists precisely to measure what the red line is protecting
                against; NOTES decision 4.)

arm="anchored"  R-2 minimal version: mu_s is a NON-learnable buffer on a K=6
                uniform grid over the s range, assigned round-robin (i % K), and
                sigma_s = lo + (hi-lo) * sigmoid(raw) with [lo,hi]=[0.025,0.30].
                Everything else (RGB geometry, payload, init, optimizer) is
                bit-identical to the naive arm, so the two sigma_s trajectories
                differ only by the s-axis parameterization.

--------------------------------------------------------------------------- init
Both arms: mu_c on the uniform RGB grid (paper A.1), sigma_c = 0.15 isotropic,
cross terms 0, opacity 1.0, M_i = I, b_i = 0, **G = 0, g = 0** -> f(x,s) = x at
step 0 (identity passthrough).  G=0 rather than GLUT's G=I is the project red
line ("全局仿射 G 初始化 = 0 不是 I，否则 f(x)=2x"); the G=I copy lives only in
the A0 reproduction.
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn

from model.glut_repro.model import uniform_grid_mu

EPS_W = 1e-6
LOG_2PI = math.log(2.0 * math.pi)

SIGMA_S_MIN = 0.025          # R-2 bounds (PLAN section 2 table, row R-2)
SIGMA_S_MAX = 0.30
K_ANCHOR = 6                 # R-2: mu_s fixed K=6 grid
GAMMA_MU = 0.15              # R-3 reference diversity floor for std(mu_s)


def anchor_grid(k: int = K_ANCHOR, lo: float = 0.0,
                hi: float = 1.0) -> torch.Tensor:
    """K uniform anchors spanning the s range, endpoints included."""
    return torch.linspace(lo, hi, k)


class GLUT4D(nn.Module):
    """Single 4D GLUT (one global renderer shared by every image).

    Parameters (N Gaussians):
      mu_c          (N,3)  RGB centers                     learnable
      mu_s          (N,)   s centers            learnable | buffer (anchored)
      log_sigma_c   (N,3)  log Cholesky diagonal of the RGB block
      sigma_s_raw   (N,)   exp(raw) | sigmoid-bounded      -> marginal sigma_s
      chol_off      (N,6)  sub-diagonal entries, order
                           (L10, L20, L30, L21, L31, L32)
                           = (r-s, g-s, b-s, g-r, b-r, b-g)
      opacity_raw   (N,)   clamp[0,1] in forward
      M (N,3,3), b (N,3), G (3,3), g (3)   payload, RGB only
    """

    ARMS = ("naive", "anchored")

    def __init__(self, n_gaussians: int, arm: str = "naive",
                 k_anchor: int = K_ANCHOR,
                 sigma_c_init: float = 0.15, sigma_s_init: float = 0.15,
                 sigma_s_min: float = SIGMA_S_MIN,
                 sigma_s_max: float = SIGMA_S_MAX,
                 s_lo: float = 0.0, s_hi: float = 1.0,
                 mu_s_init: float = 0.5):
        super().__init__()
        if arm not in self.ARMS:
            raise ValueError(f"arm must be one of {self.ARMS}, got {arm!r}")
        self.N = int(n_gaussians)
        self.arm = arm
        self.k_anchor = int(k_anchor)
        self.sigma_s_min = float(sigma_s_min)
        self.sigma_s_max = float(sigma_s_max)
        self.s_lo, self.s_hi = float(s_lo), float(s_hi)
        N = self.N

        self.mu_c = nn.Parameter(torch.empty(N, 3))
        self.log_sigma_c = nn.Parameter(torch.empty(N, 3))
        self.chol_off = nn.Parameter(torch.empty(N, 6))
        self.opacity_raw = nn.Parameter(torch.empty(N))
        self.M = nn.Parameter(torch.empty(N, 3, 3))
        self.b = nn.Parameter(torch.empty(N, 3))
        self.G = nn.Parameter(torch.empty(3, 3))
        self.g = nn.Parameter(torch.empty(3))

        if arm == "naive":
            # free, learnable, every center on the same value == on D
            self.mu_s = nn.Parameter(torch.full((N,), float(mu_s_init)))
            self.sigma_s_raw = nn.Parameter(
                torch.full((N,), math.log(sigma_s_init)))
        else:
            anchors = anchor_grid(self.k_anchor, s_lo, s_hi)
            self.register_buffer(
                "mu_s", anchors[torch.arange(N) % self.k_anchor].clone())
            self.sigma_s_raw = nn.Parameter(
                torch.full((N,), self._inv_bounded(sigma_s_init)))
        self.init_payload(sigma_c_init)

    # ------------------------------------------------------------------ init
    def _inv_bounded(self, sigma: float) -> float:
        lo, hi = self.sigma_s_min, self.sigma_s_max
        t = min(max((sigma - lo) / (hi - lo), 1e-4), 1.0 - 1e-4)
        return math.log(t / (1.0 - t))

    @torch.no_grad()
    def init_payload(self, sigma_c: float = 0.15) -> None:
        """mu_c uniform grid, iso sigma_c, cross 0, o=1, M=I, b=0, G=0, g=0
        -> f(x,s) = x at step 0 (red line: G init 0, not I)."""
        self.mu_c.copy_(uniform_grid_mu(self.N))
        self.log_sigma_c.fill_(math.log(sigma_c))
        self.chol_off.zero_()
        self.opacity_raw.fill_(1.0)
        self.M.copy_(torch.eye(3).expand(self.N, 3, 3))
        self.b.zero_()
        self.G.zero_()
        self.g.zero_()

    # -------------------------------------------------------------- geometry
    def sigma_s(self) -> torch.Tensor:
        """Marginal std of the s axis, (N,).  naive: bare exp (unbounded);
        anchored: sigmoid-bounded to [sigma_s_min, sigma_s_max]."""
        if self.arm == "naive":
            return torch.exp(self.sigma_s_raw)
        lo, hi = self.sigma_s_min, self.sigma_s_max
        return lo + (hi - lo) * torch.sigmoid(self.sigma_s_raw)

    def mu4(self) -> torch.Tensor:
        """(N,4) centers in (s, r, g, b) order."""
        return torch.cat([self.mu_s.unsqueeze(-1), self.mu_c], dim=-1)

    def chol(self) -> torch.Tensor:
        """Lower-triangular L, (N,4,4), coordinate order (s, r, g, b)."""
        sig_s = self.sigma_s()
        dc = torch.exp(self.log_sigma_c)
        o = self.chol_off
        z = torch.zeros_like(sig_s)
        return torch.stack([
            torch.stack([sig_s, z, z, z], dim=-1),
            torch.stack([o[:, 0], dc[:, 0], z, z], dim=-1),   # r-s cross
            torch.stack([o[:, 1], o[:, 3], dc[:, 1], z], dim=-1),   # g-s
            torch.stack([o[:, 2], o[:, 4], o[:, 5], dc[:, 2]], dim=-1),  # b-s
        ], dim=-2)

    def log_density(self, u: torch.Tensor) -> torch.Tensor:
        """log p_i(u) for u (P,4) -> (N,P).  Forced fp32 (dossier 2.4 row 3).

        The Mahalanobis term solves L z = (u - mu) by explicit 4x4 forward
        substitution rather than torch.linalg.solve_triangular: identical
        arithmetic, but it keeps the whole thing in elementwise kernels
        (~4x faster here, since N*P tiny triangular solves are launch-bound).
        ci_checks_4d check 2 pins the result against
        MultivariateNormal.log_prob.
        """
        u32 = u.float()
        diff = u32.unsqueeze(0) - self.mu4().float().unsqueeze(1)   # (N,P,4)
        d0, d1, d2, d3 = diff.unbind(-1)                            # (N,P) each
        # L entries taken straight from the parameters (same L as chol(), but
        # without materializing the (N,4,4) tensor in the hot path)
        sg = self.sigma_s().float().unsqueeze(-1)                   # (N,1)
        dc = torch.exp(self.log_sigma_c.float())                    # (N,3)
        off = self.chol_off.float()                                 # (N,6)
        z0 = d0 / sg
        z1 = (d1 - off[:, 0:1] * z0) / dc[:, 0:1]
        z2 = (d2 - off[:, 1:2] * z0 - off[:, 3:4] * z1) / dc[:, 1:2]
        z3 = (d3 - off[:, 2:3] * z0 - off[:, 4:5] * z1
              - off[:, 5:6] * z2) / dc[:, 2:3]
        maha = z0 * z0 + z1 * z1 + z2 * z2 + z3 * z3                # (N,P)
        log_det = 2.0 * (torch.log(sg.squeeze(-1))
                         + self.log_sigma_c.float().sum(-1))        # (N,)
        return -2.0 * LOG_2PI - 0.5 * log_det.unsqueeze(-1) - 0.5 * maha

    def weights(self, u: torch.Tensor) -> torch.Tensor:
        """Eq.2 weights, (N,P).  Exact same value as
        `p*o / (sum p*o + eps)` but evaluated with the per-pixel log-max
        factored out, so a diverging or collapsing sigma cannot overflow:

            w_i = o_i e^{l_i-m} / (sum_j o_j e^{l_j-m} + eps e^{-m})

        As all densities vanish (the escape channel), e^{-m} saturates and the
        weights go to 0 -- which is the correct limit, not a NaN.
        """
        ld = self.log_density(u)                                    # (N,P)
        o = self.opacity_raw.clamp(0.0, 1.0).float().unsqueeze(-1)
        m = ld.max(dim=0, keepdim=True).values                      # (1,P)
        num = torch.exp(ld - m) * o
        eps_term = torch.exp(torch.clamp(-m, max=80.0)) * EPS_W
        return num / (num.sum(dim=0, keepdim=True) + eps_term)

    # --------------------------------------------------------------- forward
    def forward(self, x: torch.Tensor, s: torch.Tensor) -> torch.Tensor:
        """f(x,s), unclamped.  x (P,3), s (P,) -> (P,3)."""
        x32 = x.float()
        u = torch.cat([s.float().unsqueeze(-1), x32], dim=-1)
        w = self.weights(u)                                         # (N,P)
        # sum_i w_i (M_i x + b_i) == (sum_i w_i M_i) x + sum_i w_i b_i.
        # Contracting the Gaussian index first keeps the intermediate at
        # (P,3,3) instead of (N,P,3) -- identical arithmetic, ~N/3 less memory.
        A = torch.einsum("np,nij->pij", w, self.M.float())          # (P,3,3)
        mix = torch.einsum("pij,pj->pi", A, x32) + w.T @ self.b.float()
        glob = x32 @ self.G.float().T + self.g.float()
        return mix + glob

    def predict(self, x: torch.Tensor, s: torch.Tensor) -> torch.Tensor:
        return self.forward(x, s).clamp(0.0, 1.0)

    # ----------------------------------------------------------- diagnostics
    @torch.no_grad()
    def sigma_s_stats(self) -> dict:
        sig = self.sigma_s().detach().float().cpu()
        q = torch.quantile(sig, torch.tensor([0.05, 0.25, 0.5, 0.75, 0.95]))
        eps = 0.01 * (SIGMA_S_MAX - SIGMA_S_MIN)
        return {
            "q05": float(q[0]), "q25": float(q[1]), "q50": float(q[2]),
            "q75": float(q[3]), "q95": float(q[4]),
            "mean": float(sig.mean()), "std": float(sig.std()),
            "min": float(sig.min()), "max": float(sig.max()),
            # fraction beyond the R-2 upper bound: for the anchored arm this is
            # M3's "贴上界" statistic; for the naive arm it is the divergence
            # counter (how many Gaussians have already left the legal range).
            "frac_ge_r2_max": float((sig >= SIGMA_S_MAX - eps).float().mean()),
            "frac_le_r2_min": float((sig <= SIGMA_S_MIN + eps).float().mean()),
        }

    @torch.no_grad()
    def mu_s_stats(self) -> dict:
        mu = self.mu_s.detach().float().cpu()
        q = torch.quantile(mu, torch.tensor([0.05, 0.25, 0.5, 0.75, 0.95]))
        return {
            "q05": float(q[0]), "q25": float(q[1]), "q50": float(q[2]),
            "q75": float(q[3]), "q95": float(q[4]),
            "mean": float(mu.mean()), "std": float(mu.std()),
            "min": float(mu.min()), "max": float(mu.max()),
            # R-3 diversity: std(mu_s) below gamma_mu is what the hinge would
            # penalise.  Reported, never optimised, in this experiment.
            "below_gamma_mu": bool(float(mu.std()) < GAMMA_MU),
        }

    @torch.no_grad()
    def s_sensitivity(self, x: torch.Tensor, s: torch.Tensor,
                      delta: float = 0.1) -> float:
        """E || f(x, s+d) - f(x, s) ||_2 over probe pixels.

        The perturbation always points away from the nearer end of the s range,
        so |ds| = delta exactly and no probe is silently clipped to 0.
        """
        mid = 0.5 * (self.s_lo + self.s_hi)
        step = torch.where(s <= mid, torch.full_like(s, delta),
                           torch.full_like(s, -delta))
        f0 = self.forward(x, s)
        f1 = self.forward(x, (s + step).clamp(self.s_lo, self.s_hi))
        return float((f1 - f0).norm(dim=-1).mean())

    def n_params(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
