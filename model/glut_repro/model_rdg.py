"""RD-G: transformer parameter generators for the GLUT renderer.

The renderer is NOT re-invented here.  `render()` is the functional form of
`model.BatchedGLUT.forward` (Eq.1-3 of the GLUT paper, IMPL_DOSSIER 2.2) with
the parameters supplied per sample by a generator instead of being free
nn.Parameters; `ci_checks_rdg.check_render_matches_batched_glut` pins it against
BatchedGLUT to <1e-5 on random inputs, so "same rendering core" is a test, not a
claim.  The only structural addition is the existence gate g_i of PLAN 1.1,
which enters the SAME place as the opacity (the mixture numerator) -- see
`render()` for why that placement is forced by PLAN 1.4's identity CI.

Three generator families, all writing into the identical parameter layout and
all sharing one image tokenizer, so the only thing that varies across arms is
the map (condition -> ~1.1K renderer parameters):

  MLPGen        CGLUT section 3.2 shape (IMPL_DOSSIER 2.2): 64-d style code, a
                3-Linear shared encoder, one head per parameter class
                (mean head 2 Linear, local colour head 3 Linear, ...).
                width=128 reproduces the paper's "Large"; width is the only
                knob, so `mlp_wide` is the capacity-matched control that
                separates "transformer" from "more parameters".
  TransformerGen  PLAN 1.4: N primitive queries + 1 global query, K/V = patch
                tokens + 4 register tokens (Darcet et al., arXiv:2309.16588,
                verified 2026-08-03), learnable-Fourier positional encoding
                added to the keys, cross-attention only at the PLAN's layer
                indices, ModLN with per-sublayer parameters driven by the
                global style code, and energy routing over {L/2, 3L/4, L} with
                an entropy floor so the router cannot collapse onto the last
                layer.
                G-Lite d=256 L=4 cross@{1,3}; G-Base d=384 L=6 cross@{1,3,5}.

Red lines pinned by ci_checks_rdg:
  * output head is a single zero-initialised Linear (weight AND bias), so every
    arm starts at f(x) = x exactly -- G is 0, never I (PLAN 1.4).
  * sigma comes from 0.02 + 0.48*sigmoid(z), never a bare exp.
  * the per-pixel operator sees x and nothing else: no (x,y), no neighbourhood,
    no MLP on colour, no ordering.  Everything spatial happens in the
    generator, which runs once per image and emits a bakeable LUT.
  * there is no s axis in Stage-1, hence no s-axis regulariser of any kind.
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from model.glut_repro.model import uniform_grid_mu

EPS_W = 1e-6
LOG_2PI = math.log(2.0 * math.pi)

SIGMA_LO = 0.02          # PLAN 1.4: s_i = 0.02 + 0.48*sigmoid(zs)
SIGMA_SPAN = 0.48
N_PRIM_OUT = 23          # mu3 + chol_diag3 + chol_off3 + o1 + g1 + M9 + b3
N_GLOB_OUT = 12          # G9 + c3


# ===========================================================================
# functional renderer
# ===========================================================================

def render(p: dict, x: torch.Tensor) -> torch.Tensor:
    """f(x) for a batch of per-sample GLUT parameter sets.  x: (B,P,3).

    p: mu (B,N,3), sigma (B,N,3) positive, off (B,N,3), opacity (B,N),
       gate (B,N), M (B,N,3,3), b (B,N,3), G (B,3,3), c (B,3).

    Identical arithmetic to BatchedGLUT.forward: full normalised Gaussian
    density, opacity-weighted normalisation with eps=1e-6, local affine mixture
    plus a global affine.  Two differences of form, neither of value:
      * the triangular solve is written out as forward substitution (the
        model4d_naive trick) instead of torch.linalg.solve_triangular;
      * the log-max is factored out of the softmax-like normaliser so a
        collapsing sigma underflows to w->0 instead of NaN.
    Both are pinned by CI against the reference implementation.

    The existence gate multiplies the mixture numerator next to the opacity.
    That placement is forced: PLAN 1.4 requires max dE00(f(x),x) < 1e-4 at the
    zero-initialised head, and at that point every g_i is the same sigmoid(4);
    a common factor cancels in the normaliser (identity preserved) whereas a
    payload gate would leave f(x) = 0.982 x (dE00 ~ 1, CI fails).
    """
    x32 = x.float()
    B, P, _ = x32.shape
    mu, sig, off = p["mu"].float(), p["sigma"].float(), p["off"].float()
    diff = x32.unsqueeze(1) - mu.unsqueeze(2)                    # (B,N,P,3)
    d0, d1, d2 = diff.unbind(-1)
    z0 = d0 / sig[..., 0:1]
    z1 = (d1 - off[..., 0:1] * z0) / sig[..., 1:2]
    z2 = (d2 - off[..., 1:2] * z0 - off[..., 2:3] * z1) / sig[..., 2:3]
    maha = z0 * z0 + z1 * z1 + z2 * z2                           # (B,N,P)
    log_det = 2.0 * torch.log(sig).sum(-1)                       # (B,N)
    ld = -1.5 * LOG_2PI - 0.5 * log_det.unsqueeze(-1) - 0.5 * maha
    og = (p["opacity"].float() * p["gate"].float()).unsqueeze(-1)  # (B,N,1)
    m = ld.max(dim=1, keepdim=True).values                       # (B,1,P)
    num = torch.exp(ld - m) * og
    eps_term = torch.exp(torch.clamp(-m, max=80.0)) * EPS_W
    w = num / (num.sum(dim=1, keepdim=True) + eps_term)          # (B,N,P)
    # sum_i w_i (M_i x + b_i) == (sum_i w_i M_i) x + sum_i w_i b_i
    A = torch.einsum("bnp,bnij->bpij", w, p["M"].float())        # (B,P,3,3)
    mix = torch.einsum("bpij,bpj->bpi", A, x32) \
        + torch.einsum("bnp,bnj->bpj", w, p["b"].float())
    glob = torch.einsum("bij,bpj->bpi", p["G"].float(), x32) \
        + p["c"].float().unsqueeze(1)
    return mix + glob


def n_render_params(n: int) -> int:
    """23N + 12 (GLUT's 22N+12 plus the PLAN 1.1 existence gate)."""
    return 23 * n + 12


SIGMA_S_MIN, SIGMA_S_MAX = 0.025, 0.30      # R-2 bounds (PLAN section 2, R-2)
K_ANCHOR = 6


def render4d(p: dict, x: torch.Tensor, s: torch.Tensor) -> torch.Tensor:
    """Stage-2 renderer: the same mixture with s prepended as coordinate 0.

    Geometry, ordering and bounds follow `model4d_naive.GLUT4D(arm="anchored")`
    exactly -- s FIRST so that L[0,0] IS the marginal sigma_s (the quantity
    PLAN 1.2 defines the degenerate set over), the three sub-diagonal entries of
    column 0 kept as s-RGB cross covariances and never factorised (R-2), mu_s a
    NON-learnable K=6 grid, sigma_s = 0.025 + 0.275*sigmoid(raw) (bounded, R-2).
    x (B,P,3), s (B,P) -> (B,P,3).
    """
    x32, s32 = x.float(), s.float()
    d0 = s32.unsqueeze(1) - p["mu_s"].float().unsqueeze(-1)      # (B,N,P)
    diff = x32.unsqueeze(1) - p["mu"].float().unsqueeze(2)       # (B,N,P,3)
    d1, d2, d3 = diff.unbind(-1)
    ss = p["sigma_s"].float().unsqueeze(-1)                      # (B,N,1)
    dc = p["sigma"].float()                                      # (B,N,3)
    off = p["off6"].float()                                      # (B,N,6)
    z0 = d0 / ss
    z1 = (d1 - off[..., 0:1] * z0) / dc[..., 0:1]
    z2 = (d2 - off[..., 1:2] * z0 - off[..., 3:4] * z1) / dc[..., 1:2]
    z3 = (d3 - off[..., 2:3] * z0 - off[..., 4:5] * z1
          - off[..., 5:6] * z2) / dc[..., 2:3]
    maha = z0 * z0 + z1 * z1 + z2 * z2 + z3 * z3
    log_det = 2.0 * (torch.log(ss.squeeze(-1)) + torch.log(dc).sum(-1))
    ld = -2.0 * LOG_2PI - 0.5 * log_det.unsqueeze(-1) - 0.5 * maha
    og = (p["opacity"].float() * p["gate"].float()).unsqueeze(-1)
    m = ld.max(dim=1, keepdim=True).values
    num = torch.exp(ld - m) * og
    eps_term = torch.exp(torch.clamp(-m, max=80.0)) * EPS_W
    w = num / (num.sum(dim=1, keepdim=True) + eps_term)
    A = torch.einsum("bnp,bnij->bpij", w, p["M"].float())
    mix = torch.einsum("bpij,bpj->bpi", A, x32) \
        + torch.einsum("bnp,bnj->bpj", w, p["b"].float())
    glob = torch.einsum("bij,bpj->bpi", p["G"].float(), x32) \
        + p["c"].float().unsqueeze(1)
    return mix + glob


# ===========================================================================
# zero-initialised output head (PLAN 1.4, verbatim)
# ===========================================================================

class ParamHead(nn.Module):
    """z (raw head output) -> renderer parameters.

        mu_i    = anchor_i + r * tanh(z_mu)          anchor: Stage-0 grid/kmeans
        sigma_i = 0.02 + 0.48 * sigmoid(z_sigma)     bounded, never a bare exp
        off_i   = 0.1 * z_off
        M_i     = I + 0.1 * z_M ;  b_i = 0.1 * z_b
        o_i     = sigmoid(z_o - 2) ; g_i = sigmoid(z_g + 4)
        G       = 0.1 * z_G       ;  c   = 0.1 * z_c      (G is 0, never I)

    `free=True` drops every one of those parameterisations (raw linear outputs,
    sigma via a bare exp, G initialised to I) and is the PLAN's mandatory
    "no-hardening free regression" control.
    """

    def __init__(self, n_gaussians: int, anchors: torch.Tensor | None = None,
                 anchor_radius: float = 1.0 / 3.0, free: bool = False):
        super().__init__()
        self.N = int(n_gaussians)
        self.free = bool(free)
        self.r = float(anchor_radius)
        a = uniform_grid_mu(self.N) if anchors is None else anchors
        self.register_buffer("anchors", a.float().clone())

    def forward(self, z_prim: torch.Tensor, z_glob: torch.Tensor) -> dict:
        """z_prim (B,N,23), z_glob (B,12) -> parameter dict."""
        B = z_prim.shape[0]
        zmu, zsg, zof = z_prim[..., 0:3], z_prim[..., 3:6], z_prim[..., 6:9]
        zo, zg = z_prim[..., 9], z_prim[..., 10]
        zM, zb = z_prim[..., 11:20], z_prim[..., 20:23]
        eye = torch.eye(3, device=z_prim.device, dtype=z_prim.dtype)
        if self.free:
            mu = zmu
            sigma = torch.exp(zsg).clamp(1e-3, 10.0)   # bare exp (the control)
            off = zof
            o = torch.sigmoid(zo)
            g = torch.ones_like(zg)
            M = zM.view(B, self.N, 3, 3) + eye
            b = zb
            G = z_glob[:, :9].view(B, 3, 3) + eye      # G = I (the anti-pattern)
            c = z_glob[:, 9:12]
        else:
            mu = self.anchors.unsqueeze(0) + self.r * torch.tanh(zmu)
            sigma = SIGMA_LO + SIGMA_SPAN * torch.sigmoid(zsg)
            off = 0.1 * zof
            o = torch.sigmoid(zo - 2.0)
            g = torch.sigmoid(zg + 4.0)
            M = eye + 0.1 * zM.view(B, self.N, 3, 3)
            b = 0.1 * zb
            G = 0.1 * z_glob[:, :9].view(B, 3, 3)
            c = 0.1 * z_glob[:, 9:12]
        return {"mu": mu, "sigma": sigma, "off": off, "opacity": o, "gate": g,
                "M": M, "b": b, "G": G, "c": c,
                "z_prim": z_prim, "z_glob": z_glob}


N_PRIM_OUT_4D = 27       # mu3 + sigma_c3 + sigma_s1 + off6 + o1 + g1 + M9 + b3


class ParamHead4D(ParamHead):
    """Stage-2 head: everything ParamHead does, plus the s axis under R-2.

    mu_s is a NON-learnable K=6 uniform grid assigned round robin (so it is not
    an output of the generator at all -- R-2 anchoring is structural, not a
    penalty), and sigma_s is sigmoid-bounded to [0.025, 0.30].  The s-RGB cross
    covariances (columns 0 of the 4x4 Cholesky) are generated and never
    factorised out.  Zero-init still gives f(x,s) = x for every s.
    """

    def __init__(self, n_gaussians: int, anchors=None,
                 anchor_radius: float = 1.0 / 3.0, free: bool = False,
                 k_anchor: int = K_ANCHOR, s_lo: float = 0.0,
                 s_hi: float = 1.0):
        super().__init__(n_gaussians, anchors, anchor_radius, free)
        grid = torch.linspace(s_lo, s_hi, k_anchor)
        self.register_buffer(
            "mu_s", grid[torch.arange(self.N) % k_anchor].clone())

    # z_prim layout (27): 0:3 mu | 3:6 sigma_c | 6 sigma_s | 7:13 off6
    #                     13 o   | 14 g        | 15:24 M   | 24:27 b
    def forward(self, z_prim: torch.Tensor, z_glob: torch.Tensor) -> dict:
        B = z_prim.shape[0]
        z23 = torch.cat([z_prim[..., 0:6], z_prim[..., 7:10],
                         z_prim[..., 13:15], z_prim[..., 15:27]], dim=-1)
        base = super().forward(z23, z_glob)
        base.pop("off")                       # 3-D-only field, not used in 4D
        if self.free:
            base["sigma_s"] = torch.exp(z_prim[..., 6]).clamp(1e-3, 10.0)
            base["off6"] = z_prim[..., 7:13]
        else:
            base["sigma_s"] = SIGMA_S_MIN + (SIGMA_S_MAX - SIGMA_S_MIN) * \
                torch.sigmoid(z_prim[..., 6])
            # full 4x4 lower-triangular sub-diagonals, order
            # (L10,L20,L30,L21,L31,L32) = (r-s, g-s, b-s, g-r, b-r, b-g)
            base["off6"] = 0.1 * z_prim[..., 7:13]
        base["mu_s"] = self.mu_s.unsqueeze(0).expand(B, -1)
        base["z_prim"] = z_prim
        return base


def zero_linear(in_f: int, out_f: int) -> nn.Linear:
    """The single zero-initialised Linear of PLAN 1.4 (no MLP stacking)."""
    lin = nn.Linear(in_f, out_f)
    nn.init.zeros_(lin.weight)
    nn.init.zeros_(lin.bias)
    return lin


# ===========================================================================
# shared image tokenizer (identical module in every arm)
# ===========================================================================

class PairTokenizer(nn.Module):
    """(I_in, after) 6-channel stack -> patch tokens + a global style code.

    A 16x pixel-shuffle-free conv stem; at R=128 that is an 8x8 = 64 token grid.
    Deliberately small and identical across arms: this experiment measures the
    generator, so the encoder must not be a confound.  Its parameter count is
    reported separately from the generator's in every table.
    """

    def __init__(self, dim: int = 256, style: int = 128, in_ch: int = 6):
        super().__init__()
        self.dim, self.style_dim = dim, style
        self.stem = nn.Sequential(
            nn.Conv2d(in_ch, 64, 4, 4), nn.GELU(),
            nn.Conv2d(64, 128, 2, 2), nn.GELU(),
            nn.Conv2d(128, dim, 2, 2))
        self.norm = nn.LayerNorm(dim)
        self.to_style = nn.Sequential(nn.Linear(dim, style), nn.GELU(),
                                      nn.Linear(style, style))
        # condition dropout target (PLAN: p=0.15 -> a LEARNABLE constant)
        self.null_tokens = nn.Parameter(torch.zeros(1, 1, dim))
        self.null_style = nn.Parameter(torch.zeros(1, style))

    def forward(self, img: torch.Tensor, drop: torch.Tensor | None = None):
        """img (B,6,R,R) in [0,1]; drop (B,) bool -> use the null condition."""
        h = self.stem(img)                                   # (B,D,g,g)
        B, D, gh, gw = h.shape
        tok = self.norm(h.flatten(2).transpose(1, 2))        # (B,T,D)
        sty = self.to_style(tok.mean(1))                     # (B,S)
        if drop is not None and bool(drop.any()):
            m = drop.view(B, 1, 1).to(tok.dtype)
            tok = tok * (1 - m) + self.null_tokens.expand_as(tok) * m
            sty = sty * (1 - m[:, :, 0]) + \
                self.null_style.expand_as(sty) * m[:, :, 0]
        return tok, sty, (gh, gw)


# ===========================================================================
# arm 1: CGLUT-style MLP generator
# ===========================================================================

class MLPGen(nn.Module):
    """IMPL_DOSSIER 2.2 CGLUT section 3.2 generator, image-conditioned.

    CGLUT's own condition is a per-style lookup embedding E in R^{L x 64}, which
    cannot generalise past the styles it was trained on; the 64-d code here is
    produced from the image pair instead (PLAN 1.5 step 1, "E -> Proj(embed)"),
    which is the only change needed to put the MLP and the transformer on the
    same task.  Everything downstream is the paper's shape.
    """

    def __init__(self, n_gaussians: int, tok_dim: int = 256, width: int = 128,
                 code: int = 64, four_d: bool = False):
        super().__init__()
        self.N, self.W = int(n_gaussians), int(width)
        self.four_d = bool(four_d)
        self.n_chol = 10 if four_d else 6   # 4D: sigma_c3 + sigma_s1 + off6
        W = self.W
        self.to_code = nn.Linear(tok_dim, code)
        self.trunk = nn.Sequential(
            nn.Linear(code, W), nn.ReLU(inplace=True),
            nn.Linear(W, W), nn.ReLU(inplace=True),
            nn.Linear(W, W), nn.ReLU(inplace=True))
        self.mu_head = nn.Sequential(nn.Linear(W, W), nn.ReLU(inplace=True),
                                     zero_linear(W, self.N * 3))
        self.col_head = nn.Sequential(nn.Linear(W, W), nn.ReLU(inplace=True),
                                      nn.Linear(W, W), nn.ReLU(inplace=True),
                                      zero_linear(W, self.N * 12))
        self.chol_head = zero_linear(W, self.N * self.n_chol)
        self.op_head = zero_linear(W, self.N * 2)
        self.glob_head = zero_linear(W, N_GLOB_OUT)

    def forward(self, tok: torch.Tensor, sty: torch.Tensor, grid=None):
        e = self.to_code(tok.mean(1))
        h = self.trunk(e)
        B, N = tok.shape[0], self.N
        mu = self.mu_head(h).view(B, N, 3)
        col = self.col_head(h).view(B, N, 12)
        ch = self.chol_head(h).view(B, N, self.n_chol)
        op = self.op_head(h).view(B, N, 2)
        z_prim = torch.cat([mu, ch, op, col], dim=-1)
        return [(z_prim, self.glob_head(h))], None


# ===========================================================================
# arm 2: transformer decoder generator (PLAN 1.4)
# ===========================================================================

class ModLN(nn.Module):
    """LayerNorm with per-sublayer FiLM parameters driven by the style code.

    PLAN 1.4: "全局 style 走 ModLN 每子层独立参数（可插值性是功能，删=回退）".
    Zero-initialised projection so the module starts as a plain LayerNorm.
    """

    def __init__(self, dim: int, style: int):
        super().__init__()
        self.norm = nn.LayerNorm(dim, elementwise_affine=False)
        self.proj = nn.Linear(style, 2 * dim)
        nn.init.zeros_(self.proj.weight)
        nn.init.zeros_(self.proj.bias)

    def forward(self, x: torch.Tensor, sty: torch.Tensor) -> torch.Tensor:
        scale, shift = self.proj(sty).unsqueeze(1).chunk(2, dim=-1)
        return self.norm(x) * (1.0 + scale) + shift


class LearnableFourierPE(nn.Module):
    """Learnable Fourier features of the patch coordinate, added to the keys.

    PLAN 1.4: "位置编码 learnable Fourier 加 key".  Frequencies are learned
    (a Linear on the 2-d normalised coordinate), then [sin, cos] is projected
    to the model width.
    """

    def __init__(self, dim: int, n_freq: int = 32):
        super().__init__()
        self.freq = nn.Linear(2, n_freq, bias=False)
        nn.init.normal_(self.freq.weight, std=8.0)
        self.out = nn.Linear(2 * n_freq, dim)

    def forward(self, gh: int, gw: int, device, dtype) -> torch.Tensor:
        ys = torch.linspace(-1, 1, gh, device=device, dtype=dtype)
        xs = torch.linspace(-1, 1, gw, device=device, dtype=dtype)
        gy, gx = torch.meshgrid(ys, xs, indexing="ij")
        pos = torch.stack([gx, gy], -1).view(-1, 2)
        a = self.freq(pos)
        return self.out(torch.cat([torch.sin(a), torch.cos(a)], -1))


class DecoderLayer(nn.Module):
    def __init__(self, dim: int, style: int, heads: int, cross: bool,
                 mlp_ratio: int = 4):
        super().__init__()
        self.n1 = ModLN(dim, style)
        self.self_attn = nn.MultiheadAttention(dim, heads, batch_first=True)
        self.cross = cross
        if cross:
            self.n2 = ModLN(dim, style)
            self.cross_attn = nn.MultiheadAttention(dim, heads,
                                                    batch_first=True)
        self.n3 = ModLN(dim, style)
        self.mlp = nn.Sequential(nn.Linear(dim, mlp_ratio * dim), nn.GELU(),
                                 nn.Linear(mlp_ratio * dim, dim))

    def forward(self, q, kv, sty):
        h = self.n1(q, sty)
        q = q + self.self_attn(h, h, h, need_weights=False)[0]
        if self.cross:
            h = self.n2(q, sty)
            q = q + self.cross_attn(h, kv, kv, need_weights=False)[0]
        q = q + self.mlp(self.n3(q, sty))
        return q


class TransformerGen(nn.Module):
    """PLAN 1.4 generator.  n_q = N primitive queries + 1 global query.

    The 5 mask queries and the curve query of PLAN 1.4's token list are NOT
    instantiated: neither the mask basis nor the 1-D pre-curve is part of the
    Stage-1 cube objective, so they would be unsupervised dead tokens (NOTES
    decision 3).  Everything else is the spec.
    """

    def __init__(self, n_gaussians: int, dim: int = 256, depth: int = 4,
                 heads: int = 8, cross_at=(1, 3), tok_dim: int = 256,
                 style: int = 128, n_register: int = 4,
                 route_at: tuple = None, four_d: bool = False):
        super().__init__()
        self.N, self.dim, self.depth = int(n_gaussians), dim, depth
        self.n_q = self.N + 1
        self.query = nn.Parameter(torch.randn(1, self.n_q, dim) * 0.02)
        self.kv_proj = nn.Linear(tok_dim, dim)
        self.kv_norm = nn.LayerNorm(dim)
        self.pe = LearnableFourierPE(dim)
        self.register = nn.Parameter(torch.randn(1, n_register, dim) * 0.02)
        cross_set = {c - 1 for c in cross_at}          # PLAN uses 1-based
        self.layers = nn.ModuleList([
            DecoderLayer(dim, style, heads, i in cross_set)
            for i in range(depth)])
        if route_at is None:
            route_at = tuple(sorted({max(1, depth // 2), max(1, (3 * depth) // 4),
                                     depth}))
        self.route_at = tuple(route_at)               # 1-based layer indices
        self.route_logit = nn.Parameter(torch.zeros(len(self.route_at)))
        self.out_norm = nn.LayerNorm(dim)
        self.prim_head = zero_linear(
            dim, N_PRIM_OUT_4D if four_d else N_PRIM_OUT)
        self.glob_head = zero_linear(dim, N_GLOB_OUT)

    def _head(self, h):
        h = self.out_norm(h)
        return self.prim_head(h[:, :self.N]), self.glob_head(h[:, self.N])

    def route_entropy(self) -> torch.Tensor:
        p = torch.softmax(self.route_logit, 0)
        return -(p * torch.log(p + 1e-9)).sum()

    def forward(self, tok: torch.Tensor, sty: torch.Tensor, grid):
        B = tok.shape[0]
        gh, gw = grid
        kv = self.kv_proj(tok)
        kv = kv + self.pe(gh, gw, kv.device, kv.dtype).unsqueeze(0)
        kv = torch.cat([kv, self.register.expand(B, -1, -1)], dim=1)
        kv = self.kv_norm(kv)
        q = self.query.expand(B, -1, -1)
        taps = []
        for i, layer in enumerate(self.layers):
            q = layer(q, kv, sty)
            if (i + 1) in self.route_at:
                taps.append(q)
        wts = torch.softmax(self.route_logit, 0)
        fused = sum(w * t for w, t in zip(wts, taps))
        outs = [self._head(t) for t in taps]           # per-tap aux heads
        return [self._head(fused)] + outs, self.route_entropy()


# ===========================================================================
# full arm = tokenizer + generator + head
# ===========================================================================

ARMS = {
    # name        : (kind, kwargs)
    "mlp":        ("mlp", dict(width=128)),
    "mlp_wide":   ("mlp", dict(width=880)),
    "gtiny":      ("tf", dict(dim=192, depth=2, heads=6, cross_at=(1, 2))),
    "glite":      ("tf", dict(dim=256, depth=4, heads=8, cross_at=(1, 3))),
    "gbase":      ("tf", dict(dim=384, depth=6, heads=8, cross_at=(1, 3, 5))),
}


class RDGModel(nn.Module):
    def __init__(self, arm: str, n_gaussians: int = 48, tok_dim: int = 256,
                 style: int = 128, anchors: torch.Tensor | None = None,
                 free: bool = False, in_ch: int = 6):
        super().__init__()
        if arm not in ARMS:
            raise ValueError(f"unknown arm {arm!r}; have {sorted(ARMS)}")
        kind, kw = ARMS[arm]
        self.arm, self.N = arm, int(n_gaussians)
        self.tokenizer = PairTokenizer(tok_dim, style, in_ch)
        if kind == "mlp":
            self.gen = MLPGen(n_gaussians, tok_dim=tok_dim, **kw)
        else:
            self.gen = TransformerGen(n_gaussians, tok_dim=tok_dim,
                                      style=style, **kw)
        self.head = ParamHead(n_gaussians, anchors=anchors, free=free)

    def params_from_image(self, img, drop=None):
        tok, sty, grid = self.tokenizer(img, drop)
        outs, extra = self.gen(tok, sty, grid)
        return [self.head(zp, zg) for zp, zg in outs], extra

    def forward(self, img, x, drop=None):
        plist, extra = self.params_from_image(img, drop)
        return [render(p, x) for p in plist], plist, extra

    # ---------------------------------------------------------------- counts
    def n_params(self) -> dict:
        def c(m):
            return sum(p.numel() for p in m.parameters() if p.requires_grad)
        return {"tokenizer": c(self.tokenizer), "generator": c(self.gen),
                "head": c(self.head), "total": c(self)}


# ===========================================================================
# bake / export (E22, "烘焙一致性从第一天当一等指标")
# ===========================================================================

def bake_cube(p: dict, size: int = 33, chunk: int = 8) -> torch.Tensor:
    """Evaluate the generated renderer on the uniform size^3 grid -> (B,K,K,K,3).

    This is the deliverable of PLAN 1.6: the exportable object is a standard 3-D
    .cube, obtained by resampling the anisotropic Gaussian mixture onto a
    uniform lattice.  The resampling loss is exactly what E22 must quantify, so
    it is measured, never assumed away.
    """
    dev = p["mu"].device
    ax = torch.linspace(0.0, 1.0, size, device=dev)
    r, g, b = torch.meshgrid(ax, ax, ax, indexing="ij")
    grid = torch.stack([r, g, b], -1).reshape(1, -1, 3)
    B = p["mu"].shape[0]
    out = []
    for i in range(0, B, chunk):
        j = min(i + chunk, B)
        sub = {k: (v[i:j] if torch.is_tensor(v) and v.shape[0] == B else v)
               for k, v in p.items() if torch.is_tensor(v)}
        out.append(render(sub, grid.expand(j - i, -1, -1)).clamp(0, 1))
    return torch.cat(out).view(B, size, size, size, 3)


def tetra_lookup(table: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
    """Tetrahedral interpolation of table[r,g,b] at x.  table (B,K,K,K,3),
    x (B,P,3) -> (B,P,3).

    PLAN 1.6: the bake evaluation MUST read the exported cube back with the
    host's interpolator, and .cube hosts use tetrahedral, not the trilinear the
    Gaussian mixture was trained under.  Written from the standard 6-simplex
    decomposition of the unit cube (Kasson et al.); ci_checks_rdg pins it
    against colour-science's table_interpolation_tetrahedral.
    """
    B, K = table.shape[0], table.shape[1]
    xc = x.clamp(0, 1) * (K - 1)
    i0 = xc.floor().clamp(0, K - 2).long()
    f = xc - i0.float()
    fr, fg, fb = f.unbind(-1)
    flat = table.reshape(B, K * K * K, 3)

    def at(dr, dg, db):
        idx = ((i0[..., 0] + dr) * K + (i0[..., 1] + dg)) * K + (i0[..., 2] + db)
        return torch.gather(flat, 1, idx.unsqueeze(-1).expand(-1, -1, 3))

    c000, c111 = at(0, 0, 0), at(1, 1, 1)
    out = torch.zeros_like(c000)
    # six tetrahedra of the unit cube, standard ordering on (fr, fg, fb)
    cases = [
        (fr >= fg) & (fg >= fb), (fr >= fb) & (fb > fg), (fb > fr) & (fr >= fg),
        (fg > fr) & (fr >= fb), (fb > fg) & (fg > fr), (fg >= fb) & (fb > fr),
    ]
    verts = [
        ((1, 0, 0), (1, 1, 0), fr, fg, fb),
        ((1, 0, 0), (1, 0, 1), fr, fb, fg),
        ((0, 0, 1), (1, 0, 1), fb, fr, fg),
        ((0, 1, 0), (1, 1, 0), fg, fr, fb),
        ((0, 0, 1), (0, 1, 1), fb, fg, fr),
        ((0, 1, 0), (0, 1, 1), fg, fb, fr),
    ]
    for cond, (v1, v2, a, b_, c_) in zip(cases, verts):
        m = cond.unsqueeze(-1)
        val = ((1 - a).unsqueeze(-1) * c000 + (a - b_).unsqueeze(-1) * at(*v1)
               + (b_ - c_).unsqueeze(-1) * at(*v2) + c_.unsqueeze(-1) * c111)
        out = torch.where(m, val, out)
    return out


# ===========================================================================
# CIEDE2000 (torch), used for every headline number
# ===========================================================================

_XN = 0.3127 / 0.3290
_ZN = (1.0 - 0.3127 - 0.3290) / 0.3290
_M_RGB2XYZ = [[0.4123907992659595, 0.3575843393838780, 0.1804807884018343],
              [0.2126390058715104, 0.7151686787677559, 0.0721923153607337],
              [0.0193308187155918, 0.1191947797946259, 0.9505321522496607]]


def srgb_to_lab(rgb: torch.Tensor) -> torch.Tensor:
    rgb = rgb.clamp(0.0, 1.0).float()
    lin = torch.where(rgb <= 0.04045, rgb / 12.92,
                      ((rgb + 0.055) / 1.055) ** 2.4)
    xyz = lin @ lin.new_tensor(_M_RGB2XYZ).T
    xr, yr, zr = xyz[..., 0] / _XN, xyz[..., 1], xyz[..., 2] / _ZN
    eps, kappa = 216.0 / 24389.0, 24389.0 / 27.0

    def fn(t):
        return torch.where(t > eps, t.clamp(min=1e-12) ** (1.0 / 3.0),
                           (kappa * t + 16.0) / 116.0)
    fx, fy, fz = fn(xr), fn(yr), fn(zr)
    return torch.stack([116.0 * fy - 16.0, 500.0 * (fx - fy),
                        200.0 * (fy - fz)], -1)


def delta_e00(rgb1: torch.Tensor, rgb2: torch.Tensor) -> torch.Tensor:
    """CIEDE2000 between two sRGB tensors (...,3) -> (...).  kL=kC=kH=1."""
    l1 = srgb_to_lab(rgb1)
    l2 = srgb_to_lab(rgb2)
    L1, a1, b1 = l1.unbind(-1)
    L2, a2, b2 = l2.unbind(-1)
    C1 = torch.sqrt(a1 * a1 + b1 * b1)
    C2 = torch.sqrt(a2 * a2 + b2 * b2)
    Cb = 0.5 * (C1 + C2)
    Cb7 = Cb ** 7
    G = 0.5 * (1 - torch.sqrt(Cb7 / (Cb7 + 25.0 ** 7)))
    a1p, a2p = (1 + G) * a1, (1 + G) * a2
    C1p = torch.sqrt(a1p * a1p + b1 * b1)
    C2p = torch.sqrt(a2p * a2p + b2 * b2)
    h1p = torch.rad2deg(torch.atan2(b1, a1p)) % 360.0
    h2p = torch.rad2deg(torch.atan2(b2, a2p)) % 360.0
    dLp = L2 - L1
    dCp = C2p - C1p
    prod = C1p * C2p
    dh = h2p - h1p
    dhp = torch.where(prod == 0, torch.zeros_like(dh),
                      torch.where(dh > 180.0, dh - 360.0,
                                  torch.where(dh < -180.0, dh + 360.0, dh)))
    dHp = 2.0 * torch.sqrt(prod.clamp(min=0)) * torch.sin(
        torch.deg2rad(dhp) / 2.0)
    Lbp = 0.5 * (L1 + L2)
    Cbp = 0.5 * (C1p + C2p)
    hsum = h1p + h2p
    hdiff = torch.abs(h1p - h2p)
    hbp = torch.where(
        prod == 0, hsum,
        torch.where(hdiff <= 180.0, 0.5 * hsum,
                    torch.where(hsum < 360.0, 0.5 * (hsum + 360.0),
                                0.5 * (hsum - 360.0))))
    T = (1 - 0.17 * torch.cos(torch.deg2rad(hbp - 30.0))
         + 0.24 * torch.cos(torch.deg2rad(2 * hbp))
         + 0.32 * torch.cos(torch.deg2rad(3 * hbp + 6.0))
         - 0.20 * torch.cos(torch.deg2rad(4 * hbp - 63.0)))
    dtheta = 30.0 * torch.exp(-(((hbp - 275.0) / 25.0) ** 2))
    Cbp7 = Cbp ** 7
    Rc = 2.0 * torch.sqrt(Cbp7 / (Cbp7 + 25.0 ** 7))
    Lm = (Lbp - 50.0) ** 2
    Sl = 1.0 + 0.015 * Lm / torch.sqrt(20.0 + Lm)
    Sc = 1.0 + 0.045 * Cbp
    Sh = 1.0 + 0.015 * Cbp * T
    Rt = -torch.sin(torch.deg2rad(2 * dtheta)) * Rc
    return torch.sqrt((dLp / Sl) ** 2 + (dCp / Sc) ** 2 + (dHp / Sh) ** 2
                      + Rt * (dCp / Sc) * (dHp / Sh))
