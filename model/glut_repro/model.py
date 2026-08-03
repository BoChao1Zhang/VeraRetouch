"""Batched single-style GLUT model (Eq.1-3, IMPL_DOSSIER 2.2).

A "batch" here is a batch of B independent LUT models trained simultaneously
(each with its own parameters and its own GT).  B=1 recovers the plain
single-GLUT of the paper.  Everything runs in fp32; the density computation is
explicitly forced to fp32 even under autocast (dossier 2.4, row 3).

Parameterization (per LUT, N Gaussians, 22N+12 params):
  mu           (N,3)    RGB centers
  chol_log_diag(N,3)    log of Cholesky diagonal  (dossier 2.4: log/exp param,
                        init log(0.15))
  chol_off     (N,3)    below-diagonal entries (l21, l31, l32), init 0
  opacity_raw  (N,)     raw + clamp[0,1] in forward (dossier 2.4), init 1.0
  M            (N,3,3)  local affine matrix, GLUT-original init = I
  b            (N,3)    local affine bias,   init 0
  G            (3,3)    global affine matrix, GLUT-original init = I
  g            (3,)     global affine bias,   init 0

Forward (Eq.1-3):
  p_i(x) = (2*pi)^{-3/2} |Sigma_i|^{-1/2} exp(-0.5 * d_i(x))
  w_i(x) = p_i(x) o_i / (sum_j p_j(x) o_j + eps),  eps = 1e-6
  f(x)   = sum_i w_i(x) (M_i x + b_i) + (G x + g)
  output = clamp(f(x), 0, 1)   # clamp applied for eval; loss uses raw f(x)
                               # by default (see train_a0 NOTES)
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn

EPS_W = 1e-6
LOG_2PI = math.log(2.0 * math.pi)


def uniform_grid_mu(n: int) -> torch.Tensor:
    """Regular grid uniformly covering [0,1]^3 (paper A.1).

    For non-cube n, use the smallest k with k^3 >= n and take n evenly spaced
    points of the k^3 ordering (deterministic; paper reports Uniform 45.47 vs
    Random 45.45, i.e. the choice is worth <0.05 dB).
    """
    k = 1
    while k ** 3 < n:
        k += 1
    ax = torch.linspace(0.0, 1.0, k)
    r, g, b = torch.meshgrid(ax, ax, ax, indexing="ij")
    pts = torch.stack([r, g, b], dim=-1).reshape(-1, 3)
    if pts.shape[0] == n:
        return pts
    idx = torch.linspace(0, pts.shape[0] - 1, n).round().long()
    return pts[idx]


class BatchedGLUT(nn.Module):
    """B independent single-style GLUT models."""

    def __init__(self, batch: int, n_gaussians: int):
        super().__init__()
        self.B = batch
        self.N = n_gaussians
        B, N = batch, n_gaussians
        self.mu = nn.Parameter(torch.empty(B, N, 3))
        self.chol_log_diag = nn.Parameter(torch.empty(B, N, 3))
        self.chol_off = nn.Parameter(torch.empty(B, N, 3))
        self.opacity_raw = nn.Parameter(torch.empty(B, N))
        self.M = nn.Parameter(torch.empty(B, N, 3, 3))
        self.b = nn.Parameter(torch.empty(B, N, 3))
        self.G = nn.Parameter(torch.empty(B, 3, 3))
        self.g = nn.Parameter(torch.empty(B, 3))
        self.init_glut_original()

    # ------------------------------------------------------------------
    # Initializations
    # ------------------------------------------------------------------
    @torch.no_grad()
    def init_glut_original(self, sigma: float = 0.15) -> None:
        """GLUT paper A.1 init, copied verbatim for the A0 anchor:
        mu uniform grid, Sigma iso sigma (log-Cholesky), opacity 1.0,
        local AND global affines = identity matrix + zero bias.
        (Yes, this makes f(x) ~= 2x at init; that is the paper's own choice.
        Our G=0 change is an RD-arm modification, out of A0 scope.)"""
        grid = uniform_grid_mu(self.N)
        self.mu.copy_(grid.unsqueeze(0).expand(self.B, -1, -1))
        self.chol_log_diag.fill_(math.log(sigma))
        self.chol_off.zero_()
        self.opacity_raw.fill_(1.0)
        eye = torch.eye(3)
        self.M.copy_(eye.expand(self.B, self.N, 3, 3))
        self.b.zero_()
        self.G.copy_(eye.expand(self.B, 3, 3))
        self.g.zero_()

    @torch.no_grad()
    def init_e1_residual(self, mu: torch.Tensor, bias: torch.Tensor,
                         sigma: float = 0.3) -> None:
        """E1 direct-fit init: f(x) = x + sum_i w_i b_i at step 0.

        mu   (B,N,3): weighted-kmeans centers (|f(x)-x|-weighted, PLAN step 3)
        bias (B,N,3): per-cluster weighted mean residual (y - x)
        Local M = 0 (payload starts at cluster-constant residual), global = I.
        """
        self.mu.copy_(mu)
        self.chol_log_diag.fill_(math.log(sigma))
        self.chol_off.zero_()
        self.opacity_raw.fill_(1.0)
        self.M.zero_()
        self.b.copy_(bias)
        self.G.copy_(torch.eye(3, device=self.G.device).expand(self.B, 3, 3))
        self.g.zero_()

    # ------------------------------------------------------------------
    # Forward pieces
    # ------------------------------------------------------------------
    def _chol(self) -> torch.Tensor:
        """Lower-triangular Cholesky factors, (B,N,3,3)."""
        B, N = self.B, self.N
        L = self.mu.new_zeros(B, N, 3, 3)
        d = torch.exp(self.chol_log_diag)          # positive diagonal
        L[..., 0, 0] = d[..., 0]
        L[..., 1, 1] = d[..., 1]
        L[..., 2, 2] = d[..., 2]
        L[..., 1, 0] = self.chol_off[..., 0]       # l21
        L[..., 2, 0] = self.chol_off[..., 1]       # l31
        L[..., 2, 1] = self.chol_off[..., 2]       # l32
        return L

    def log_density(self, x: torch.Tensor) -> torch.Tensor:
        """log p_i(x), full normalized Gaussian density.  x: (B,P,3).
        Returns (B,N,P).  Forced fp32 (dossier 2.4 row 3)."""
        x32 = x.float()
        L = self._chol().float()                                  # (B,N,3,3)
        diff = x32.unsqueeze(1) - self.mu.float().unsqueeze(2)    # (B,N,P,3)
        # solve L z = diff  -> z, Mahalanobis d = |z|^2
        z = torch.linalg.solve_triangular(
            L.unsqueeze(2), diff.unsqueeze(-1), upper=False)      # (B,N,P,3,1)
        maha = z.squeeze(-1).pow(2).sum(-1)                       # (B,N,P)
        # log|Sigma| = 2 * sum log diag(L)
        log_det = 2.0 * self.chol_log_diag.float().sum(-1)        # (B,N)
        return -1.5 * LOG_2PI - 0.5 * log_det.unsqueeze(-1) - 0.5 * maha

    def weights(self, x: torch.Tensor) -> torch.Tensor:
        """w_i(x) (Eq.2): density * opacity, normalized with eps.  (B,N,P)."""
        p = torch.exp(self.log_density(x))                        # (B,N,P) fp32
        o = self.opacity_raw.clamp(0.0, 1.0).float().unsqueeze(-1)
        num = p * o
        return num / (num.sum(dim=1, keepdim=True) + EPS_W)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """f(x) (Eq.3), unclamped.  x: (B,P,3) -> (B,P,3)."""
        w = self.weights(x)                                       # (B,N,P)
        local = torch.einsum("bnij,bpj->bnpi", self.M.float(), x.float()) \
            + self.b.float().unsqueeze(2)                         # (B,N,P,3)
        mix = torch.einsum("bnp,bnpi->bpi", w, local)
        glob = torch.einsum("bij,bpj->bpi", self.G.float(), x.float()) \
            + self.g.float().unsqueeze(1)
        return mix + glob

    def predict(self, x: torch.Tensor) -> torch.Tensor:
        """Eval output = clamp(f(x), 0, 1)."""
        return self.forward(x).clamp(0.0, 1.0)

    # ------------------------------------------------------------------
    # Introspection (E1 density control / dossier B.5 style stats)
    # ------------------------------------------------------------------
    @torch.no_grad()
    def weight_mass(self, x: torch.Tensor) -> torch.Tensor:
        """Per-primitive share of total mixing weight on probe x: (B,N),
        rows sum to ~1."""
        w = self.weights(x)                                       # (B,N,P)
        m = w.sum(-1)
        return m / (m.sum(-1, keepdim=True) + 1e-12)

    def n_params_per_lut(self) -> int:
        return 22 * self.N + 12
