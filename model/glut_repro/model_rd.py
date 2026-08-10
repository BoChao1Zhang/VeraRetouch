"""RD-STD (R-1+R-2+R-3+R-7+R-10) and RD-E (R-11) renderers, plus their
same-N / same-family 3D controls.

Four arms, all sharing one training driver (train_rd.py) and one evaluation
caliber (tools/harness):

  g3d    3D Gaussian control      N=32, 22N+12 = 716 params.  The "同 N 3D"
                                  the PLAN L-ladder compares against.
  gstd   RD-STD                   4D anchored Gaussians (R-2) + zero-init
                                  s-gated payload modulation (R-1); trained
                                  with the R-3 diversity hinge and the
                                  GECO-managed (R-7) s-response constraint
                                  that realises R-10(b)/(c).  2530 params.
  lut3d  3D LUT control           17^3 x 3 = 14,739 params.  Structurally
                                  s-independent -> Delta_const / Delta_shuffle
                                  are identically 0, which doubles as a
                                  built-in negative control on the probes.
  lut4d  RD-E                     17^3 x 5 x 3 = 73,695 params quadrilinear 4D
                                  LUT (PLAN R-11's "73.7K" to the digit).

Red lines honoured here (see ci_checks_rd.py, which pins each of them):
  * global affine G initialised to 0, never I  -> f(x,s) = x at step 0
  * sigma_s bounded by a sigmoid, never a bare exp
  * NO smoothing regularizer on the s axis anywhere (the RD-E TV term is
    applied to the RGB axes only; the s-axis TV weight defaults to 0)
  * the per-pixel operator sees (x, s) and nothing else -- no (x,y), no
    neighbourhood, no ordering.  The R-1 MLP is a function of *s alone*, so at
    any fixed s the model is still sum_i w_i(x,s)(M~_i x + b~_i) + Gx + g, i.e.
    an affine mixture over RGB and bakeable to a 3D LUT slice.
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn

from model.glut_repro.model4d_naive import (
    GLUT4D, GAMMA_MU, K_ANCHOR, SIGMA_S_MAX, SIGMA_S_MIN, anchor_grid,
    uniform_grid_mu)

# R-3 thresholds.
#
# gamma_mu is written in PLAN as the absolute number 0.15, but 0.15 is only
# meaningful relative to the spread of the s axis: PLAN 1.2 puts s in [-3,3]
# (grid std ~2.11, where 0.15 is 7% and the hinge silently never fires), while
# oracle mask s lives in [0,1] (grid std 0.374, where 0.15 is 40%).  We
# therefore carry the RELATIVE form, authorised by the main agent, and note
# that on this experiment's [0,1] range it reproduces PLAN's own number to
# three decimals: 0.4 * std(linspace(0,1,6)) = 0.1497.
GAMMA_MU_REL = 0.4
GAMMA_LAMBDA = 0.15          # on log(sigma_s); PLAN gives gamma_mu only, this
                             # is pre-registered (NOTES)


def gamma_mu_for(anchors: torch.Tensor) -> float:
    """R-3's mu floor, as a fraction of the anchor grid's own spread."""
    return float(GAMMA_MU_REL * anchors.float().std())


# ===========================================================================
# R-1: zero-init s-gated payload modulation
# ===========================================================================

class SModulation(nn.Module):
    """f_i = (1 + gamma_i(s)) * (M_i x + b_i) + beta_i(s), zero at init.
    (out_dim=3 reuses the same trunk for R-5's magnitude-only Delta_i(s).)

    PLAN section 2.1 row R-1 verbatim: Fourier(16) -> shared MLP -> elementwise
    product with a per-Gaussian code z_i(16) -> linear head to [gamma; beta],
    head initialised to all zeros; the mixture weights w_i are NOT touched.

    Zero-init means step 0 reproduces the un-modulated 4D GLUT exactly, so the
    arm degrades gracefully rather than starting from a perturbed model
    (ControlNet-style "switch s off but keep the gradient alive" -- PLAN 1.2:
    zero-init switches s OFF with a live gradient, normalization cancellation
    divides s OUT with a dead one; the two are not the same thing).

    Input is s and nothing else.  z_i carries the per-Gaussian identity, so the
    shared trunk is evaluated once per pixel instead of N times.
    """

    def __init__(self, n_gaussians: int, n_freq: int = 8, hidden: int = 32,
                 code: int = 16, out_dim: int = 6):
        super().__init__()
        self.N = int(n_gaussians)
        self.code = int(code)
        self.out_dim = int(out_dim)
        # Fourier features: [sin(pi 2^k s), cos(pi 2^k s)], k=0..n_freq-1 -> 16
        self.register_buffer(
            "freqs", math.pi * torch.pow(2.0, torch.arange(n_freq).float()))
        self.trunk = nn.Sequential(
            nn.Linear(2 * n_freq, hidden), nn.GELU(),
            nn.Linear(hidden, code))
        self.z = nn.Parameter(torch.empty(self.N, code))
        self.head = nn.Linear(code, self.out_dim)   # [gamma(3); beta(3)] or [d(3)]
        self.reset_parameters()

    @torch.no_grad()
    def reset_parameters(self) -> None:
        for m in self.trunk:
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                m.bias.zero_()
        self.z.normal_(0.0, 1.0 / math.sqrt(self.code))
        self.head.weight.zero_()                # <- the zero init of R-1
        self.head.bias.zero_()

    def s_code(self, s: torch.Tensor) -> torch.Tensor:
        """(P,) -> (P, code).  The shared trunk, evaluated once per pixel."""
        ang = s.float().unsqueeze(-1) * self.freqs                 # (P,F)
        feat = torch.cat([torch.sin(ang), torch.cos(ang)], dim=-1)  # (P,2F)
        return self.trunk(feat)

    def raw(self, s: torch.Tensor) -> torch.Tensor:
        """(P,) -> (N, P, out_dim).

        head(code_s(p) * z_i) = sum_k W[:,k] z_i[k] code_s[p,k] + bias, so the
        (N,P,code) intermediate is never materialised: fold z into W first.
        Arithmetic is identical (ci_checks_rd check 8b pins it against the
        literal formulation)."""
        cs = self.s_code(s)                                        # (P,C)
        wz = self.head.weight.unsqueeze(0) * self.z.unsqueeze(1)   # (N,O,C)
        return torch.einsum("nkc,pc->npk", wz, cs) + self.head.bias

    def forward(self, s: torch.Tensor):
        """(P,) -> gamma (N,P,3), beta (N,P,3)."""
        gb = self.raw(s)
        return gb[..., :3], gb[..., 3:]


# ===========================================================================
# Gaussian arms
# ===========================================================================

def bspline_basis(s: torch.Tensor, k: int) -> torch.Tensor:
    """R-4's B-spline basis on the s axis: (P,) -> (P, k).

    Clamped uniform B-spline basis of degree d = min(3, k-1) (Cox-de Boor),
    the standard basis of a B-spline function expansion: it is a partition of
    unity, so with the k>=1 coefficient tensors zero-initialised the payload
    starts at exactly M_i^0.

    k = 1 is special-cased to beta_1(s) = s.  The degree-0 clamped basis with
    one function is the constant 1, which would make the K=1 rung s-BLIND and
    turn the whole K sweep into "0 vs 3 vs 5" -- the opposite of what the rung
    is for.  s is the minimal one-degree-of-freedom s dependence (NOTES).
    """
    p = s.shape[0]
    if k == 1:
        return s.reshape(p, 1)
    d = min(3, k - 1)
    n_int = k - d - 1
    knots = torch.cat([
        torch.zeros(d + 1, device=s.device, dtype=s.dtype),
        (torch.arange(1, n_int + 1, device=s.device, dtype=s.dtype)
         / (n_int + 1)) if n_int > 0 else s.new_zeros(0),
        torch.ones(d + 1, device=s.device, dtype=s.dtype)])
    u = s.clamp(0.0, 1.0 - 1e-6).reshape(p, 1)
    # degree 0
    lo = knots[:k + d].reshape(1, -1)
    hi = knots[1:k + d + 1].reshape(1, -1)
    b = ((u >= lo) & (u < hi)).to(s.dtype)
    for deg in range(1, d + 1):
        m = k + d - deg
        d1 = knots[deg:deg + m] - knots[:m]
        d2 = knots[deg + 1:deg + 1 + m] - knots[1:1 + m]
        t1 = torch.where(d1 > 0, (u - knots[:m]) / d1.clamp(min=1e-12),
                         torch.zeros_like(u))
        t2 = torch.where(d2 > 0,
                         (knots[deg + 1:deg + 1 + m] - u) / d2.clamp(min=1e-12),
                         torch.zeros_like(u))
        b = t1 * b[:, :m] + t2 * b[:, 1:m + 1]
    return b


class GLUTRD(nn.Module):
    """One class, five payload/addressing mechanisms, one payload layout, so
    that every arm below differs from RD-STD in exactly one place.

    "3d"      3D control.  s deleted from the addressing, no modulation.
              22N+12 = 716 params -- the "同 N 3D" the PLAN ladder compares to.
    "std"     RD-STD, R-1: 4D anchored addressing (R-2: mu_s a non-learnable
              K=6 grid, sigma_s = 0.025+0.275*sigmoid(raw), s-RGB cross
              covariance kept and never factorized) + zero-init s-gated payload
              modulation f_i = (1+gamma_i(s))(M_i x + b_i) + beta_i(s).
    "mag"     RD-A, R-5: same addressing, but the payload is DoRA-decomposed
              M_i = m_i * V_i/||V_i||_col with V (the direction) independent of
              s and only the magnitude modulated, m_i(s) = m_i^0 (1+Delta_i(s)),
              Delta zero-init, and NO additive beta.  The single-variable
              contrast against "std" is the project's core science question:
              does s have to change the transform's DIRECTION or only its
              STRENGTH?  Side product: m_i^0 reads out "how large the model
              thinks the effect is".
    "lowrank" RD-C, R-4: geometry is RGB-only (3D addressing, no s at all in
              w_i); the entire s dependence is a rank-K payload expansion
              M_i(s) = M_i^0 + sum_k beta_k(s) M_i^k over a B-spline basis,
              k>=1 zero-init.  Params 22N+12 + 12NK, i.e. 1100 / 1868 / 2636
              for K = 1 / 3 / 5 -- PLAN's own "1.1-2.6K" to the digit.
    "unnorm"  RD-D, R-6: the normalisation denominator sums over RGB ONLY and
              the s gate g_i(s) = exp(-(s-mu_i)^2/2 sigma_i^2) enters the
              numerator alone, WITHOUT the (2 pi sigma^2)^-1/2 prefactor (PLAN
              C2: keeping the prefactor reverses the degeneration direction).
              s dependence is then formally impossible to cancel.  mu_s is
              learnable here (that is what makes 780 = PLAN's count), which is
              also the only arm where R-3's mu hinge can actually do work.
    """

    MODES = ("3d", "std", "mag", "lowrank", "unnorm")
    S_FREE_GEOMETRY = ("3d", "lowrank")

    def __init__(self, n_gaussians: int = 32, mode: str = "std",
                 sigma_c_init: float = 0.15, rank_k: int = 3,
                 sigma_s_init: float = 0.15):
        super().__init__()
        if mode not in self.MODES:
            raise ValueError(f"mode must be one of {self.MODES}")
        self.mode = mode
        self.N = int(n_gaussians)
        self.rank_k = int(rank_k)
        # The 4D core owns geometry + payload for every mode; the s row/column
        # of its Cholesky is simply never consulted where s is not addressed.
        self.core = GLUT4D(n_gaussians, arm="anchored",
                           sigma_c_init=sigma_c_init,
                           sigma_s_init=sigma_s_init)
        self.gamma_mu = gamma_mu_for(anchor_grid(K_ANCHOR))
        if mode in self.S_FREE_GEOMETRY:
            # drop the s-axis geometry so the count is exactly 22N+12 (+ payload)
            del self.core.sigma_s_raw
            self.core.register_buffer("sigma_s_raw",
                                      torch.zeros(self.N), persistent=False)
            self.core.chol_off = nn.Parameter(
                self.core.chol_off.detach()[:, 3:].clone())   # (N,3): rgb only
        if mode == "unnorm":
            # R-6: RGB-only denominator -> the 4D Cholesky's s row is unused;
            # the gate carries its own (mu_s, sigma_s), mu_s LEARNABLE and
            # initialised ON the K=6 anchor grid (never all-equal: that is
            # exactly the degenerate set D of PLAN 1.2).
            self.core.chol_off = nn.Parameter(
                self.core.chol_off.detach()[:, 3:].clone())
            anchors = anchor_grid(K_ANCHOR)
            self.mu_s_free = nn.Parameter(
                anchors[torch.arange(self.N) % K_ANCHOR].clone())
        self.mod = None
        if mode == "std":
            self.mod = SModulation(self.N, out_dim=6)
        elif mode == "mag":
            self.mod = SModulation(self.N, out_dim=3)
            self.V = nn.Parameter(torch.eye(3).expand(self.N, 3, 3).clone())
            self.m0 = nn.Parameter(torch.ones(self.N, 3))
            del self.core.M                      # V + m0 replace M
        elif mode == "lowrank":
            self.Mk = nn.Parameter(torch.zeros(self.rank_k, self.N, 3, 3))
            self.bk = nn.Parameter(torch.zeros(self.rank_k, self.N, 3))
        if mode == "unnorm":
            # With an UN-normalised local branch, sum_i w_i(x,s) < 1, so the
            # usual (M=I, G=0) pairing gives f(x,s) = x * sum_i w_i ~ 0.3x --
            # a contraction, not the identity, and the arm would spend its
            # whole budget climbing back.  The invariant the red line protects
            # is "f(x,s) = x at init, never 2x"; the only way to have it here
            # is to put the identity in the global branch and start the local
            # one at zero.  Same pairing model.py's init_e1_residual already
            # uses for the same reason, and M=0 makes f(x)=2x impossible.
            with torch.no_grad():
                self.core.M.zero_()
                self.core.b.zero_()
                self.core.G.copy_(torch.eye(3))
                self.core.g.zero_()

    # ---------------------------------------------------------------- pieces
    def weights(self, x: torch.Tensor, s: torch.Tensor) -> torch.Tensor:
        if self.mode in ("std", "mag"):
            u = torch.cat([s.float().unsqueeze(-1), x.float()], dim=-1)
            return self.core.weights(u)
        w = self._weights3d(x)
        if self.mode == "unnorm":
            # R-6: the RGB-normalised weight is MULTIPLIED by an un-normalised
            # s gate.  sum_i w_i != 1 by design -- that is the whole point:
            # with s absent from the denominator there is no algebraic way for
            # the s dependence to cancel.  No (2 pi sigma^2)^-1/2 prefactor.
            d = (s.float().unsqueeze(0) - self.mu_s_free.float().unsqueeze(-1))
            sg = self.sigma_s().float().unsqueeze(-1)
            w = w * torch.exp(-0.5 * (d / sg) ** 2)
        return w

    def _weights3d(self, x: torch.Tensor) -> torch.Tensor:
        """Eq.2 weights in 3D, same log-max-factored form as GLUT4D.weights."""
        c = self.core
        x32 = x.float()
        diff = x32.unsqueeze(0) - c.mu_c.float().unsqueeze(1)      # (N,P,3)
        d0, d1, d2 = diff.unbind(-1)
        dc = torch.exp(c.log_sigma_c.float())                      # (N,3)
        off = c.chol_off.float()                                   # (N,3)
        z0 = d0 / dc[:, 0:1]
        z1 = (d1 - off[:, 0:1] * z0) / dc[:, 1:2]
        z2 = (d2 - off[:, 1:2] * z0 - off[:, 2:3] * z1) / dc[:, 2:3]
        maha = z0 * z0 + z1 * z1 + z2 * z2
        log_det = 2.0 * c.log_sigma_c.float().sum(-1)
        ld = -1.5 * math.log(2.0 * math.pi) - 0.5 * log_det.unsqueeze(-1) \
            - 0.5 * maha
        o = c.opacity_raw.clamp(0.0, 1.0).float().unsqueeze(-1)
        m = ld.max(dim=0, keepdim=True).values
        num = torch.exp(ld - m) * o
        eps_term = torch.exp(torch.clamp(-m, max=80.0)) * 1e-6
        return num / (num.sum(dim=0, keepdim=True) + eps_term)

    def payload(self, x: torch.Tensor, s: torch.Tensor) -> torch.Tensor:
        """(N,P,3): the per-Gaussian local transform of every pixel."""
        c = self.core
        if self.mode == "mag":
            # DoRA: M_i = m_i * V_i / ||V_i||_col, direction frozen w.r.t. s,
            # magnitude modulated.  M_i(s) x = Vdir_i @ (mag_i(s) * x).
            vdir = self.V.float() / self.V.float().norm(dim=1, keepdim=True) \
                .clamp(min=1e-8)                                    # (N,3,3)
            mag = self.m0.float().unsqueeze(1) * (1.0 + self.mod.raw(s))
            return torch.einsum("nij,npj->npi", vdir, mag * x.unsqueeze(0)) \
                + c.b.float().unsqueeze(1)
        base = torch.einsum("nij,pj->npi", c.M.float(), x) \
            + c.b.float().unsqueeze(1)                              # (N,P,3)
        if self.mode == "lowrank":
            beta = bspline_basis(s.float(), self.rank_k)            # (P,K)
            for k in range(self.rank_k):
                inc = torch.einsum("nij,pj->npi", self.Mk[k].float(), x) \
                    + self.bk[k].float().unsqueeze(1)
                base = base + beta[:, k].reshape(1, -1, 1) * inc
        elif self.mode == "std":
            gamma, bet = self.mod(s)
            base = (1.0 + gamma) * base + bet
        return base

    # --------------------------------------------------------------- forward
    def forward(self, x: torch.Tensor, s: torch.Tensor) -> torch.Tensor:
        c = self.core
        x32 = x.float()
        w = self.weights(x32, s)                                    # (N,P)
        local = self.payload(x32, s)                                # (N,P,3)
        mix = (w.unsqueeze(-1) * local).sum(0)
        glob = x32 @ c.G.float().T + c.g.float()
        return mix + glob

    def predict(self, x: torch.Tensor, s: torch.Tensor) -> torch.Tensor:
        return self.forward(x, s).clamp(0.0, 1.0)

    # ----------------------------------------------------------- diagnostics
    def sigma_s(self) -> torch.Tensor:
        return self.core.sigma_s()

    @torch.no_grad()
    def sigma_s_stats(self) -> dict:
        return self.core.sigma_s_stats()

    def mu_s(self) -> torch.Tensor:
        return self.mu_s_free if self.mode == "unnorm" else self.core.mu_s

    @torch.no_grad()
    def mu_s_stats(self) -> dict:
        mu = self.mu_s().detach().float().cpu()
        q = torch.quantile(mu, torch.tensor([0.05, 0.5, 0.95]))
        return {"q05": float(q[0]), "q50": float(q[1]), "q95": float(q[2]),
                "mean": float(mu.mean()), "std": float(mu.std()),
                "min": float(mu.min()), "max": float(mu.max()),
                "learnable": self.mode == "unnorm",
                "gamma_mu": self.gamma_mu,
                "below_gamma_mu": bool(float(mu.std()) < self.gamma_mu)}

    def diversity_hinge(self) -> tuple:
        """R-3, along the Gaussian index: relu(gamma_mu - std(mu_s))
                                       + relu(gamma_lambda - std(log sigma_s)).

        gamma_mu is the RELATIVE form (0.4 * std(anchor grid) = 0.1497 on the
        [0,1] oracle-mask range, i.e. PLAN's 0.15 to three decimals; on PLAN
        1.2's [-3,3] the literal 0.15 would be 7% of the grid spread and the
        hinge would silently never fire).

        Under R-2 (modes "std"/"mag") mu_s is a frozen grid, so the first term
        is structurally 0 -- reported anyway, because "the hinge is pinned at
        100% activation" is R-3's own failure criterion and we have to be able
        to say which term did it.  Mode "unnorm" is the one arm where mu_s is
        learnable and the mu term can actually do work.  Modes with s-free
        geometry have neither quantity and return 0.
        """
        dev = self.core.mu_c.device
        if self.mode in self.S_FREE_GEOMETRY:
            z = torch.zeros((), device=dev)
            return z, z, z
        h_mu = torch.relu(self.gamma_mu - self.mu_s().float().std())
        h_lam = torch.relu(
            GAMMA_LAMBDA - torch.log(self.sigma_s().float()).std())
        return h_mu + h_lam, h_mu.detach(), h_lam.detach()

    @torch.no_grad()
    def payload_stats(self) -> dict:
        """R-5's advertised side product: m_i^0 = "how large the model thinks
        the effect is", plus the realised magnitude modulation range."""
        if self.mode != "mag":
            return {}
        m0 = self.m0.detach().float().cpu()
        ss = torch.linspace(0.0, 1.0, 21, device=self.m0.device)
        d = self.mod.raw(ss).detach().float().cpu()          # (N,21,3)
        return {"m0_mean": float(m0.mean()), "m0_std": float(m0.std()),
                "m0_min": float(m0.min()), "m0_max": float(m0.max()),
                "delta_absmax": float(d.abs().max()),
                "delta_range_mean": float((d.max(1).values
                                           - d.min(1).values).mean())}

    def n_params(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


# ===========================================================================
# RD-E: quadrilinear 4D LUT (R-11)
# ===========================================================================

class LUT4D(nn.Module):
    """s as the 4th lattice axis of a plain lookup table, 17^3 x n_s x 3.

    Quadrilinear = the tensor product of four 1-D linear interpolations, i.e.
    the 4-D generalisation of the trilinear read of a .cube: 2^4 = 16 corners,
    weight = prod over axes of (1-t) or t.  Written out from the definition; no
    external implementation is copied (DATA_ASSIGNMENT calls for a
    self-implementation in the shape of SA-LUT's clut4d, IMPL_DOSSIER appendix B).

    n_s = 5 is RD-E (73,695 params = PLAN's "73.7K").
    n_s = 1 collapses the s axis analytically (the interpolation weight on the
    single slice is 1 for every s) and gives the 3D LUT control with 14,739
    params -- and therefore Delta_const = Delta_shuffle = 0 by construction.

    Identity init: table[si, r, g, b] = (r,g,b)/(n_rgb-1).  Trilinear
    interpolation of a linear table is exact, so f(x,s) = x at step 0 to
    machine precision (no epsilon offset, unlike the Gaussian arms).
    """

    def __init__(self, n_rgb: int = 17, n_s: int = 5):
        super().__init__()
        self.n_rgb, self.n_s = int(n_rgb), int(n_s)
        ax = torch.linspace(0.0, 1.0, self.n_rgb)
        r, g, b = torch.meshgrid(ax, ax, ax, indexing="ij")
        ident = torch.stack([r, g, b], dim=-1)                     # (K,K,K,3)
        self.table = nn.Parameter(
            ident.unsqueeze(0).repeat(self.n_s, 1, 1, 1, 1).clone())

    # ------------------------------------------------------------------ read
    def forward(self, x: torch.Tensor, s: torch.Tensor) -> torch.Tensor:
        k, ns = self.n_rgb, self.n_s
        x32 = x.float().clamp(0.0, 1.0)
        c = x32 * (k - 1)
        i0 = c.floor().clamp(0, k - 2).long() if k > 1 else c.long() * 0
        t = (c - i0.float())                                        # (P,3)
        i1 = (i0 + 1).clamp(max=k - 1)
        if ns > 1:
            cs = (s.float().clamp(0.0, 1.0) * (ns - 1))
            j0 = cs.floor().clamp(0, ns - 2).long()
            ts = cs - j0.float()
            j1 = j0 + 1
        else:
            j0 = torch.zeros_like(s, dtype=torch.long)
            j1, ts = j0, torch.zeros_like(s)
        flat = self.table.reshape(ns * k * k * k, 3)
        out = torch.zeros(x32.shape[0], 3, device=x32.device, dtype=torch.float32)
        for ds in (0, 1):
            js, ws = (j0, 1.0 - ts) if ds == 0 else (j1, ts)
            if ns == 1 and ds == 1:
                continue
            for dr in (0, 1):
                ir = i0[:, 0] if dr == 0 else i1[:, 0]
                wr = (1.0 - t[:, 0]) if dr == 0 else t[:, 0]
                for dg in (0, 1):
                    ig = i0[:, 1] if dg == 0 else i1[:, 1]
                    wg = (1.0 - t[:, 1]) if dg == 0 else t[:, 1]
                    for db in (0, 1):
                        ib = i0[:, 2] if db == 0 else i1[:, 2]
                        wb = (1.0 - t[:, 2]) if db == 0 else t[:, 2]
                        idx = ((js * k + ir) * k + ig) * k + ib
                        wgt = (ws * wr * wg * wb).unsqueeze(-1)
                        out = out + wgt * flat.index_select(0, idx)
        return out

    def predict(self, x: torch.Tensor, s: torch.Tensor) -> torch.Tensor:
        return self.forward(x, s).clamp(0.0, 1.0)

    # ------------------------------------------------------------------- reg
    def tv(self, w_rgb: float = 1e-4, w_s: float = 0.0) -> torch.Tensor:
        """Total variation on the LATTICE.

        w_s defaults to 0 and must stay 0 for any headline RD-E number:
        "s 轴禁平滑正则" is a project red line, and here it is also a
        methodological one -- RD-E is the veto gate on the whole renderer
        programme, so smoothing along s would make "no s information" a
        self-fulfilling result.  The knob exists only for the PLAN's
        "lambda -> inf degenerates to 3D" appendix sweep.
        """
        t = self.table
        loss = t.new_zeros(())
        if w_rgb:
            for ax in (1, 2, 3):
                loss = loss + w_rgb * (t.diff(dim=ax)).abs().mean()
        if w_s and self.n_s > 1:
            loss = loss + w_s * (t.diff(dim=0)).abs().mean()
        return loss

    @torch.no_grad()
    def cell_visits(self, x: torch.Tensor, s: torch.Tensor) -> float:
        """Fraction of lattice cells whose 16-corner stencil is ever touched."""
        k, ns = self.n_rgb, self.n_s
        i = (x.float().clamp(0, 1) * (k - 1)).round().long()
        j = (s.float().clamp(0, 1) * max(ns - 1, 1)).round().long()
        code = ((j * k + i[:, 0]) * k + i[:, 1]) * k + i[:, 2]
        return float(torch.unique(code).numel()) / float(ns * k ** 3)

    def n_params(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


# ===========================================================================
# R-7: GECO
# ===========================================================================

class GECO:
    """Lagrangian auto-weighting of the s-response constraint (PLAN R-7).

    Source of the algorithm (verified first-hand against the original, NOTES
    section 0): Rezende & Viola, "Generalized ELBO with Constrained
    Optimization, GECO", 3rd Workshop on Bayesian Deep Learning, NeurIPS 2018
    (arXiv:1810.00597).  Algorithm 1 verbatim:

        initialise lambda = 1
        C_hat_t   = batch average of the constraint
        C_ma_t    = alpha C_ma_{t-1} + (1-alpha) C_hat_t     (C_ma_0 = C_hat_0)
        C_t       = C_hat_t + StopGradient(C_ma_t - C_hat_t)
        lambda_t <- lambda_{t-1} exp(prop. to C_t)      i.e.  dlog(lambda) ~ C_t

    with alpha = 0.99 (the paper's stated constraint moving-average parameter).
    The paper's constraint table includes the lower-bound form "kappa - q", so
    C = tau - D(s) below is one of its own templates, not an invention.

    The paper does not state the log-lambda step size (it writes "prop. to"),
    nor a cap on lambda; both are pre-registered here (NOTES): the constraint is
    normalised by tau so the step is dimensionless, step = 0.01 per iteration
    clipped to +-0.05, and lambda is clamped to [1e-4, 100].  PLAN R-7's own
    failure criterion is "beta pinned at its ceiling and the constraint still
    unmet", which is exactly what the cap makes observable.
    """

    def __init__(self, tau: float, alpha: float = 0.99, step: float = 0.01,
                 lam_init: float = 1.0, lam_min: float = 1e-4,
                 lam_max: float = 100.0, max_dlog: float = 0.05):
        self.tau, self.alpha, self.step = float(tau), float(alpha), float(step)
        self.lam = float(lam_init)
        self.lam_min, self.lam_max = float(lam_min), float(lam_max)
        self.max_dlog = float(max_dlog)
        self.ma: float | None = None
        self.pinned_steps = 0
        self.n_steps = 0

    def constraint(self, response: torch.Tensor) -> torch.Tensor:
        """C = tau - D  (<= 0 means "the model responds to s enough")."""
        return self.tau - response

    def term(self, c: torch.Tensor) -> torch.Tensor:
        """lambda * C_t with C_t = C_hat + sg(C_ma - C_hat) (Algorithm 1)."""
        c_hat = c
        v = float(c_hat.detach())
        self.ma = v if self.ma is None else \
            self.alpha * self.ma + (1.0 - self.alpha) * v
        c_t = c_hat + (self.ma - v)              # the sg() part is a constant
        return self.lam * c_t

    @torch.no_grad()
    def update(self) -> None:
        if self.ma is None:
            return
        d = max(-self.max_dlog,
                min(self.max_dlog, self.step * self.ma / self.tau))
        self.lam = float(min(self.lam_max,
                             max(self.lam_min, self.lam * math.exp(d))))
        self.n_steps += 1
        if self.lam >= self.lam_max * (1 - 1e-9):
            self.pinned_steps += 1

    def state(self) -> dict:
        return {"lambda": self.lam, "C_ma": self.ma, "tau": self.tau,
                "pinned_frac": (self.pinned_steps / self.n_steps
                                if self.n_steps else 0.0)}


def s_response(model, x: torch.Tensor, s: torch.Tensor,
               delta: float = 0.5, s_lo: float = 0.0,
               s_hi: float = 1.0) -> torch.Tensor:
    """R-10(b)/(c): mean |f(x, s+d) - f(x, s)|, d pointing away from the nearer
    end of the range so |d| = delta exactly and no probe is silently clipped.

    Differentiable, and the quantity the GECO constraint is written on.  Note
    it is a *perturbation* response, not a *shuffle* response: on L0 every
    image has s == 1, so a shuffle-based proxy would be identically 0 and would
    drive lambda to its ceiling on a level whose whole point is that s must
    cost nothing.  A perturbation response is satisfiable at zero
    reconstruction cost there, which is exactly R-7's stated job ("化解强制敏感
    vs 优雅退化的矛盾").

    The magnitude is capped at 1.0 per channel before averaging.  Without the
    cap the quantity is not on the scale tau is written in: evaluating a pixel
    at an s it never co-occurs with is an extrapolation, and the raw
    (unclamped) mixture can return values in the hundreds there, which makes
    any finite tau vacuously satisfied.  Capping at one full RGB range keeps
    "tau = 0.02" readable as "2% of full scale on average" and keeps the
    gradient alive exactly in the regime that matters.
    """
    mid = 0.5 * (s_lo + s_hi)
    step = torch.where(s <= mid, torch.full_like(s, delta),
                       torch.full_like(s, -delta))
    f0 = model(x, s)
    f1 = model(x, (s + step).clamp(s_lo, s_hi))
    return (f1 - f0).abs().clamp(max=1.0).mean()


# ===========================================================================
# factory
# ===========================================================================

ARMS = ("g3d", "gstd", "ga", "gc1", "gc3", "gc5", "gd", "lut3d", "lut4d")

#  arm -> (EXPERIMENTS_v3 row, PLAN mechanism)
ARM_DOC = {
    "g3d": ("control", "3D Gaussian, same N (PLAN ladder's 同 N 3D)"),
    "gstd": ("RD-STD", "R-1+R-2+R-3+R-7+R-10"),
    "ga": ("RD-A", "R-5 magnitude-only (DoRA) + R-2+R-3+R-7+R-10"),
    "gc1": ("RD-C K=1", "R-4 low-rank payload, RGB-only geometry, +R-7+R-10"),
    "gc3": ("RD-C K=3", "R-4 low-rank payload, RGB-only geometry, +R-7+R-10"),
    "gc5": ("RD-C K=5", "R-4 low-rank payload, RGB-only geometry, +R-7+R-10"),
    "gd": ("RD-D", "R-6 un-normalised s gate + R-3+R-7+R-10"),
    "lut3d": ("control", "3D LUT 17^3 (structurally s-free)"),
    "lut4d": ("RD-E", "R-11 quadrilinear 4D LUT 17^3 x 5"),
}


def build_arm(arm: str, n_gaussians: int = 32, n_rgb: int = 17,
              n_s: int = 5) -> nn.Module:
    if arm == "g3d":
        return GLUTRD(n_gaussians, mode="3d")
    if arm == "gstd":
        return GLUTRD(n_gaussians, mode="std")
    if arm == "ga":
        return GLUTRD(n_gaussians, mode="mag")
    if arm.startswith("gc"):
        return GLUTRD(n_gaussians, mode="lowrank", rank_k=int(arm[2:]))
    if arm == "gd":
        return GLUTRD(n_gaussians, mode="unnorm")
    if arm == "lut3d":
        return LUT4D(n_rgb=n_rgb, n_s=1)
    if arm == "lut4d":
        return LUT4D(n_rgb=n_rgb, n_s=n_s)
    raise ValueError(f"unknown arm {arm!r} (want one of {ARMS})")


def uses_s(arm: str) -> bool:
    """Arms whose output can depend on s at all.  g3d / lut3d cannot, which is
    why their Delta_const and Delta_shuffle are exact zeros and double as an
    implementation self-check on the probes."""
    return arm not in ("g3d", "lut3d")


def has_hinge(arm: str) -> bool:
    """R-3 applies where the s axis lives in the Gaussian geometry."""
    return arm in ("gstd", "ga", "gd")


def has_geco(arm: str) -> bool:
    """R-7/R-10 apply to every Gaussian arm that can see s."""
    return arm in ("gstd", "ga", "gc1", "gc3", "gc5", "gd")
