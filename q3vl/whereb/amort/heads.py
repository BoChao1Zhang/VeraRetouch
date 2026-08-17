"""P1 and P3' -- two ways to turn one frozen forward into a mask field.

Both arms share every input and every loss; they differ *only* in what sits
between the conv tower and the mask.  That is the whole point: the A/B answers
"is the Phi-71 intermediate layer an asset or a liability in a feed-forward
chain", which E5 left open by proving only that Phi-71 is not a *bottleneck*
(it reproduces any field handed to it, decode-minus-prior Delta ~ 0).

    P1   tower -> spatially-aligned grid codes -> w(71) -> frozen D -> m
    P3'  tower -> field  -> guided upsample -> m                (no Phi at all)

Design constraints that are not negotiable here
-----------------------------------------------

**No coordinate channels, no positional encoding.**  DELTA §5.6 forbids them
explicitly, "to stop the head learning the centre prior itself", and E3 then
measured exactly that failure on the trained arms: corr(output, centre prior)
0.64 > corr(output, GT) 0.47.  A head that is *structurally* unable to address
absolute position cannot rediscover a centre blob.  Padding is ``reflect`` for
the same reason -- zero padding is a well-known back door through which a CNN
learns absolute position from the frame.

**No single pooled vector on the path to w.**  W01/W02 funnelled everything
through one attention-pooled ``(B, dim)`` vector; RESEARCH §2 M5 names that the
suspected (b)-class bottleneck and P1's brief says "spatially-aligned grid codes
organised by Phi support, single-vector pooling forbidden".  :class:`CoeffHead`
therefore contracts a per-coefficient spatial map against that coefficient's own
basis column -- coefficient *j* is read out over the support of ``phi_dir[:, j]``
and nothing else.

The pooled path survives only for the handful of genuinely global scalars
(``w0``, ``alpha``, and the readout's ``rho`` -- 4 numbers for ``band``), which
are global by construction of ``R(s; rho)``.  That asymmetry is deliberate and
is the pre-registered second-round ablation ("pooled single-vector control arm").
"""

from __future__ import annotations

import math
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F

from q3vl.where.config import PHI_DIR_DIM, S_SCALE
from q3vl.where.readout import inv_bounded_sigmoid, param_shapes

from ..heads import rho_numel, rho_split

__all__ = ["FiLM", "ConvTower", "CoeffHead", "PooledCoeffHead", "P1Head",
           "P3PrimeHead", "SemanticHead", "CondEncoder", "ShapeDistHead"]


def _softplus_inv(y: float) -> float:
    return float(y + math.log(-math.expm1(-y))) if y < 20 else float(y)


# --- conditioning -----------------------------------------------------------

class CondEncoder(nn.Module):
    """``pooled <where> hidden (2560)`` + orientation/shape word ids -> cond vector.

    The orientation/shape words are parsed from the **instruction text**, never
    from the generated ``<where>`` span.  DELTA §7 is unambiguous: the noun part
    of ``<where>`` is trustworthy (P-W5 proved word specificity) but its geometry
    part is not -- local samples reproduce the GT ``edit scope`` clause verbatim
    0/30 times and err systematically toward "large centred ellipse", which is
    the very failure mode being treated.  Feeding that back in would close a loop
    around the bug.
    """

    def __init__(self, text_dim: int = 2560, n_words: int = 32, word_dim: int = 32,
                 out_dim: int = 256):
        super().__init__()
        self.proj = nn.Sequential(nn.LayerNorm(text_dim), nn.Linear(text_dim, out_dim))
        self.words = nn.EmbeddingBag(n_words, word_dim, mode="sum")
        self.out_dim = out_dim + word_dim
        nn.init.zeros_(self.words.weight)

    def forward(self, h_where: torch.Tensor, h_mask: torch.Tensor | None,
                word_ids: torch.Tensor, word_offsets: torch.Tensor) -> torch.Tensor:
        if h_mask is not None:
            m = h_mask.float().unsqueeze(-1)
            pooled = (h_where * m).sum(1) / m.sum(1).clamp_min(1.0)
        else:
            pooled = h_where.mean(1)
        return torch.cat([self.proj(pooled), self.words(word_ids, word_offsets)], dim=-1)


class FiLM(nn.Module):
    """Per-channel affine modulation.  ``gamma`` starts at 1, ``beta`` at 0.

    Zero-init on the projection (not on the bias) makes the block an identity at
    step 0, so the conditioning switches on smoothly instead of scrambling the
    tower before it has learned anything.
    """

    def __init__(self, cond_dim: int, channels: int):
        super().__init__()
        self.to_scale_shift = nn.Linear(cond_dim, 2 * channels)
        nn.init.zeros_(self.to_scale_shift.weight)
        nn.init.zeros_(self.to_scale_shift.bias)
        self.channels = channels

    def forward(self, x: torch.Tensor, cond: torch.Tensor) -> torch.Tensor:
        g, b = self.to_scale_shift(cond).chunk(2, dim=-1)
        g = g.reshape(-1, self.channels, 1, 1)
        b = b.reshape(-1, self.channels, 1, 1)
        return x * (1.0 + g) + b


def apply_inject(codes: torch.Tensor, inject=None) -> torch.Tensor:
    """Add a conditioning residual to the tower's penultimate features.

    This is the injection point of proposal tap A: ``codes`` is exactly the
    tensor the field convolution reads, so a residual here can reshape the
    field without touching the stem's input channels -- which is what makes the
    module addable to an already-trained checkpoint (the broadcast form widens
    ``stem.weight`` and cannot be resumed at all).

    ``inject`` is a callable rather than a tensor so the *caller* owns which
    module is applied and with what code; the head stays agnostic.
    """
    return codes if inject is None else codes + inject(codes)


class _Block(nn.Module):
    """Pre-norm residual conv block with reflect padding and optional FiLM."""

    def __init__(self, ch: int, cond_dim: int | None = None):
        super().__init__()
        self.norm = nn.GroupNorm(8, ch)
        self.conv1 = nn.Conv2d(ch, ch, 3, padding=1, padding_mode="reflect")
        self.conv2 = nn.Conv2d(ch, ch, 3, padding=1, padding_mode="reflect")
        self.film = FiLM(cond_dim, ch) if cond_dim else None
        nn.init.zeros_(self.conv2.weight)
        nn.init.zeros_(self.conv2.bias)

    def forward(self, x: torch.Tensor, cond: torch.Tensor | None = None) -> torch.Tensor:
        h = F.gelu(self.norm(x))
        h = self.conv1(h)
        if self.film is not None and cond is not None:
            h = self.film(h, cond)
        h = self.conv2(F.gelu(h))
        return x + h


class ConvTower(nn.Module):
    """``F_pre (+ dense similarity field) -> (C, gh, gw)`` codes.  ~2M params.

    FiLM is injected into the **deep** blocks only (``film_from`` onward), which
    is the geometry path's registered wiring; the semantic path injects early
    and owns its own tower.
    """

    def __init__(self, in_dim: int = 1024, extra_ch: int = 1, ch: int = 128,
                 n_blocks: int = 6, cond_dim: int | None = 288, film_from: int = 3):
        super().__init__()
        self.stem = nn.Conv2d(in_dim + extra_ch, ch, 1)
        self.blocks = nn.ModuleList(
            _Block(ch, cond_dim if i >= film_from else None) for i in range(n_blocks)
        )
        self.out_norm = nn.GroupNorm(8, ch)
        self.ch = ch
        self.extra_ch = extra_ch

    def forward(self, feat: torch.Tensor, extra: torch.Tensor | None,
                cond: torch.Tensor | None) -> torch.Tensor:
        x = feat if extra is None else torch.cat([feat, extra], dim=1)
        x = self.stem(x)
        for blk in self.blocks:
            x = blk(x, cond)
        return F.gelu(self.out_norm(x))


# --- P1: grid codes -> w ----------------------------------------------------

class CoeffHead(nn.Module):
    """``(C, gh, gw)`` codes + ``phi_dir (P, 71)`` -> ``w_raw (71,)``.

    ``w_raw[j] = sum_p phi_dir[p, j] * a_j(p) / P``

    One learned scalar map per coefficient, each contracted against **its own**
    basis column.  Three properties follow, and all three are the reason this
    shape was chosen over "pool then project":

    * no single vector is ever formed, so the (b)-class bottleneck cannot exist;
    * coefficient *j*'s gradient is spatially localised to where ``phi_dir[:, j]``
      has support, which is what "organised by Phi support" means operationally;
    * the map is exactly the adjoint of the analytic decoder, so at
      ``a_j = const`` the head reproduces a plain projection -- a sane, non-
      degenerate starting point rather than an arbitrary linear layer.

    ``w_raw`` is never supervised.  It is a direction: ``w_dir = w_raw/||w_raw||``
    downstream, so the sign-canonicalisation flip boundary that E2 found sitting
    under 3.92% of training targets is structurally irrelevant here (the mask is
    invariant to it, NOTES §7.5).
    """

    def __init__(self, ch: int, n_coeff: int = PHI_DIR_DIM, seed: int = 0):
        super().__init__()
        self.n_coeff = n_coeff
        self.to_maps = nn.Conv2d(ch, n_coeff, 1)
        nn.init.zeros_(self.to_maps.weight)
        g = torch.Generator(device="cpu").manual_seed(seed)
        w0 = torch.randn(n_coeff, generator=g)
        # bias -> a_j(p) == const_j at step 0 -> w_raw == Phi^T const, a unit-ish
        # direction rather than the origin (||w_raw||=0 has a 1e12 gradient).
        with torch.no_grad():
            self.to_maps.bias.copy_(w0 / w0.norm())

    def forward(self, codes: torch.Tensor, phi_dir: torch.Tensor) -> torch.Tensor:
        maps = self.to_maps(codes)                       # (1, 71, gh, gw)
        p = maps.shape[-2] * maps.shape[-1]
        a = maps.reshape(self.n_coeff, p)                # (71, P)
        return (a * phi_dir.transpose(0, 1)).sum(-1) / p


class _GlobalScalars(nn.Module):
    """``w0``, ``alpha_raw`` and ``rho`` -- the genuinely global parameters.

    ``R(s; rho)`` is defined as a global map from the scalar field to the mask,
    so these have nowhere spatial to live.  They are pooled, and the informed
    biases are copied from the Where-A oracle's own start point so that step 0
    is a sane band rather than a random one.
    """

    def __init__(self, ch: int, readout: str):
        super().__init__()
        self.readout = readout
        self.n_rho = rho_numel(readout)
        self.proj = nn.Linear(ch, 2 + self.n_rho)
        with torch.no_grad():
            self.proj.weight.zero_()
            bias = torch.empty(2 + self.n_rho)
            bias[0] = 0.0                        # w0
            bias[1] = _softplus_inv(1.0)         # alpha ~= 1
            bias[2:] = self._rho_bias(readout)
            self.proj.bias.copy_(bias)

    @staticmethod
    def _rho_bias(readout: str) -> torch.Tensor:
        from ..heads import RhoOutput

        return RhoOutput.default_bias(readout)

    def forward(self, codes: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, dict]:
        pooled = codes.mean(dim=(-2, -1)).reshape(1, -1)
        y = self.proj(pooled)[0]
        return y[0], y[1], rho_split(self.readout, y[2:])


class PooledCoeffHead(nn.Module):
    """The registered **control**: pool to one vector, then project to w(71).

    This is W01/W02's shape, kept deliberately: RESEARCH §2 M5 names single-vector
    pooling as the suspected (b)-class bottleneck, and no experiment in the
    literature or in this campaign has isolated it from the loss change.  Running
    it against :class:`CoeffHead` under an otherwise identical arm is that
    missing experiment.
    """

    def __init__(self, ch: int, n_coeff: int = PHI_DIR_DIM, seed: int = 0):
        super().__init__()
        self.proj = nn.Linear(ch, n_coeff)
        g = torch.Generator(device="cpu").manual_seed(seed)
        w0 = torch.randn(n_coeff, generator=g)
        with torch.no_grad():
            self.proj.weight.zero_()
            self.proj.bias.copy_(w0 / w0.norm())

    def forward(self, codes: torch.Tensor, phi_dir: torch.Tensor) -> torch.Tensor:
        return self.proj(codes.mean(dim=(-2, -1)).reshape(1, -1))[0]


class P1Head(nn.Module):
    """Tower -> ``(w0, w_raw, alpha_raw, rho)``; the caller renders through D."""

    def __init__(self, readout: str = "band", ch: int = 128, in_dim: int = 1024,
                 extra_ch: int = 1, n_blocks: int = 6, cond_dim: int | None = 288,
                 seed: int = 0, pooled_w: bool = False):
        super().__init__()
        self.tower = ConvTower(in_dim, extra_ch, ch, n_blocks, cond_dim)
        self.coeff = (PooledCoeffHead(ch, seed=seed) if pooled_w
                      else CoeffHead(ch, seed=seed))
        self.globals = _GlobalScalars(ch, readout)
        self.readout = readout
        self.pooled_w = pooled_w

    def forward(self, feat: torch.Tensor, extra: torch.Tensor | None,
                cond: torch.Tensor | None, phi_dir: torch.Tensor,
                inject=None) -> dict[str, Any]:
        codes = apply_inject(self.tower(feat, extra, cond), inject)
        w0, alpha_raw, rho = self.globals(codes)
        return {"w0": w0, "w_raw": self.coeff(codes, phi_dir),
                "alpha_raw": alpha_raw, **rho}


# --- P3': tower -> field ----------------------------------------------------

class P3PrimeHead(nn.Module):
    """Tower -> a bounded scalar field, upsampled by the same guided filter.

    The field is squashed with ``S_SCALE*tanh(./S_SCALE)`` so it is *structurally*
    inside ``UpsampleConfig.domain == (-3, 3)``.  That is not decoration: the
    s-cache contract's second failure mode is a field silently clamped to the
    domain edge, which leaves the orphan/þdomain checks quiet while the axis is
    already gone.  A tanh cannot be clamped, so the failure cannot occur.

    ``gain`` is learnable (init 2.0) because ``sigmoid(3) = 0.953`` would
    otherwise cap the predicted mask below the soft GT's upper values and put a
    floor under the BCE term for reasons that have nothing to do with placement.
    """

    def __init__(self, ch: int = 128, in_dim: int = 1024, extra_ch: int = 1,
                 n_blocks: int = 6, cond_dim: int | None = 288, gain: float = 2.0):
        super().__init__()
        self.tower = ConvTower(in_dim, extra_ch, ch, n_blocks, cond_dim)
        self.to_field = nn.Conv2d(ch, 1, 1)
        nn.init.zeros_(self.to_field.weight)
        nn.init.zeros_(self.to_field.bias)
        self.gain = nn.Parameter(torch.tensor(float(gain)))

    def forward(self, feat: torch.Tensor, extra: torch.Tensor | None,
                cond: torch.Tensor | None, inject=None,
                inject_logit=None) -> torch.Tensor:
        """``inject`` is proposal tap B, ``inject_logit`` is tap A.

        Tap A is a *logit* residual, not a feature residual: §2.2(4) reads
        ``logits_final = logits_uncond + tanh(gamma) . logit_cond`` where
        ``logit_cond`` is the rank-1 hypernetwork dotted against the penultimate
        features.  It therefore has to be applied here, between ``to_field`` and
        the tanh squash, and cannot be folded into ``apply_inject``.

        Note which tensor tap A reads: the features **after** tap B, i.e. exactly
        the tensor ``to_field`` consumes.  With both taps zero-initialised the
        ordering is unobservable at step 0; afterwards "U = the dense head's
        penultimate features" is the honest reading of §2.2(4).
        """
        codes = apply_inject(self.tower(feat, extra, cond), inject)
        raw = self.to_field(codes)
        if inject_logit is not None:
            raw = raw + inject_logit(codes)
        return S_SCALE * torch.tanh(raw / S_SCALE)

    def mask_of(self, s_field: torch.Tensor) -> torch.Tensor:
        return torch.sigmoid(self.gain * s_field)


class ShapeDistHead(nn.Module):
    """SHAPE3: predict a **distance-like** field, then threshold it.

    ``m = sigmoid((s - t) / width)`` where ``s`` is a free scalar field trained
    toward ``|grad s| = 1`` (soft eikonal) and ``(t, width)`` are two per-sample
    scalars read from pooled features.

    Why this is a *shape* fix rather than another smoothness fix: level sets of
    a true distance field are, by construction, the offset curves of one
    another.  A smoothness penalty can make a blob smooth, but nothing forces
    its iso-contours to be a coherent family -- which is exactly the failure the
    user identified by eye (smooth, but not an ellipse).  Regularity here is a
    property of the parameterisation, not something the loss has to win.

    ``(t, width)`` are two scalars from pooled features, **not** a return to the
    71-dim global coefficient regression -- the spatial content stays in ``s``.
    """

    def __init__(self, ch: int = 128, in_dim: int = 1024, extra_ch: int = 1,
                 n_blocks: int = 6, cond_dim: int | None = 288):
        super().__init__()
        self.tower = ConvTower(in_dim, extra_ch, ch, n_blocks, cond_dim)
        self.to_s = nn.Conv2d(ch, 1, 1)
        nn.init.zeros_(self.to_s.weight)
        nn.init.zeros_(self.to_s.bias)
        self.profile = nn.Linear(ch, 2)
        with torch.no_grad():
            self.profile.weight.zero_()
            # level 0, width ~ softplus(0)=0.69 -- a soft but not flat profile
            self.profile.bias.copy_(torch.tensor([0.0, 0.0]))

    def forward(self, feat, extra, cond, inject=None):
        codes = apply_inject(self.tower(feat, extra, cond), inject)
        s = self.to_s(codes)
        p = self.profile(codes.mean(dim=(-2, -1)).reshape(1, -1))[0]
        t, w_raw = p[0], p[1]
        width = torch.nn.functional.softplus(w_raw) + 1e-2
        return s, t, width

    @staticmethod
    def mask_of(s, t, width):
        return torch.sigmoid((s - t) / width)


# --- the semantic path ------------------------------------------------------

class SemanticHead(nn.Module):
    """A small U-Net for the ``semantic`` family only (17.5% of local train).

    Separate path, not a shared trunk, and the project's own data is what decided
    it (NOTES §2): the geometric families are large-support, wide-soft-edge fields
    (area 0.29-0.73, soft band 23-87%) while ``semantic`` is a small-support,
    sharp-edged silhouette (area 0.169, soft band 4.8%).  Opposite gradient
    demands.

    It regresses the ``.cgt`` **soft alpha** directly rather than a binarised
    mask (NOTES §5-C conservative default): the target is an edit-falloff region
    whose 4.8% soft band is real signal, and thresholding it first would throw
    that away.  FiLM injects **early** here, mirroring the registered wiring.
    """

    def __init__(self, in_dim: int = 1024, extra_ch: int = 1, ch: int = 96,
                 cond_dim: int | None = 288, gain: float = 2.0):
        super().__init__()
        self.stem = nn.Conv2d(in_dim + extra_ch, ch, 1)
        self.film = FiLM(cond_dim, ch) if cond_dim else None
        self.enc1 = _Block(ch)
        self.enc2 = _Block(ch)
        self.mid = _Block(ch)
        self.dec1 = _Block(ch)
        self.dec2 = _Block(ch)
        self.out_norm = nn.GroupNorm(8, ch)
        self.to_field = nn.Conv2d(ch, 1, 1)
        nn.init.zeros_(self.to_field.weight)
        nn.init.zeros_(self.to_field.bias)
        self.gain = nn.Parameter(torch.tensor(float(gain)))

    def forward(self, feat: torch.Tensor, extra: torch.Tensor | None,
                cond: torch.Tensor | None) -> torch.Tensor:
        x = self.stem(feat if extra is None else torch.cat([feat, extra], dim=1))
        if self.film is not None and cond is not None:
            x = self.film(x, cond)          # early injection (semantic path)
        e1 = self.enc1(x)
        d = F.avg_pool2d(e1, 2, ceil_mode=True)
        e2 = self.enc2(d)
        m = self.mid(e2)
        u = F.interpolate(m, size=e1.shape[-2:], mode="bilinear", align_corners=False)
        y = self.dec2(self.dec1(u + e1))
        raw = self.to_field(F.gelu(self.out_norm(y)))
        return S_SCALE * torch.tanh(raw / S_SCALE)

    def mask_of(self, s_field: torch.Tensor) -> torch.Tensor:
        return torch.sigmoid(self.gain * s_field)
