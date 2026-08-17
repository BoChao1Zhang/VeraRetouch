"""PCH-Full / PCH-Lite -- the §2.2 injection module, built as specified.

This is the **complete** module of proposal §2.2 (AMD-4: all three extraction
arms share it, unconditionally on what the simplified module measured).  The
simplified injector in :mod:`q3vl.whereb.amort.pch` stays untouched as
**PCH-v0** (AMD-2) because EPR-001/002/003/007/008 are its control series and
their module sha is part of those deliveries.

Structure, and where each piece comes from (project rule: every borrowed
component names the file it was read out of; anything invented is marked NOVEL)

    (1) code -> prototype tokens
        E_shape(5,d) E_dir(9,d) E_ext(7,d) group embedding tables
          anchor: SurgicalSAM `surgicalSAM/model.py::Learnable_Prototypes`
                  (`nn.Embedding(num_classes, feat_dim)`, returned as `.weight`)
        t_g    = LayerNorm( sum_i c_i E_g[i] / max(sum_i c_i, 1) )
        t_cont = LayerNorm( W_c c_cont ),  W_c: (d,3)
        T_g    = conf_g t_g + (1-conf_g) null_g
          anchor: SAM `modeling/prompt_encoder.py` -- `not_a_point_embed` /
                  `no_mask_embed` are exactly this: a *learned* embedding standing
                  in for "this prompt is absent", never a zero vector
        tokens = [m; T_shape; T_dir; T_ext; T_cont]     m = field token
          anchor: SAM `modeling/mask_decoder.py` prepends `iou_token`+`mask_tokens`
                  to the sparse prompts and reads the answer off `hs[:, 0]`
    (2) K = LayerNorm(Conv1x1 C_feat->d (F_dense)); key_pe = PositionEmbeddingRandom(d//2)
          anchor: SAM `modeling/prompt_encoder.py::PositionEmbeddingRandom`
    (3) TwoWayTransformer(depth, d, heads, mlp, downsample_rate)
          anchor: SAM `modeling/transformer.py` (ported below, see `_TwoWay*`)
          DEVIATION (NOVEL): both cross-attentions take a `valid_grid`
          key-padding mask.  SAM has no padding to mask; this campaign's red line
          is that pad cells never participate, and F-LMM's measured 53-74% quality
          loss from unmasked pad is the reason it is not optional.
    (4) tap A -- rank-1 hypernetwork (main path)
        w = MLP(d, d, 32, 3)(hs[:,0]);  U = 1x1(C_feat->32)(dense penultimate)
        logit_cond   = w . U                       (a per-cell dot product)
        logits_final = logits_uncond + tanh(gamma) logit_cond,  gamma init 0
          anchor: SAM `modeling/mask_decoder.py::predict_masks` --
                  `masks = (hyper_in @ upscaled_embedding.view(b, c, h*w))`
          DEVIATION: SAM's hypernet output width is `transformer_dim // 8` = 32
                  against a 32-channel upscaled embedding; our tower is 128
                  channels (§0.1-3, `heads.py::apply_inject`), so a 1x1 adapter
                  brings it to 32 rather than the proposal's assumed 1024->32.
    (5) tap B -- dense residual
        F'_dense = F_dense + zero_conv(W_out keys)
          anchor: ControlNet `cldm/cldm.py::make_zero_conv` =
                  `zero_module(conv_nd(dims, ch, ch, 1, padding=0))`
        Lite collapses W_out and zero_conv into ONE zero-initialised 1x1 (D-11),
        so the rank-1 tap's escape hatch does not disappear with the capacity.

Two guarantees the module is built around, because the arms resume a trained
checkpoint and a perturbation at step 0 would be indistinguishable from an
injection effect:

* **structural** -- tanh(0)=0 and a zero-initialised zero_conv make step 0 an
  exact no-op, bit for bit (:meth:`PCHFull.assert_zero_at_init`);
* **distributional** -- conf=0 falls back to a *learned* null constant, so an
  ABSTAIN sample is served by a trained path rather than by noise; and with
  conf=0 the output is provably independent of the code's content
  (:meth:`PCHFull.assert_null_is_content_free`).

Forbidden inside this module, by red line: any per-image min-max or softmax
field normalisation (READ's mechanism is explicitly not transplanted), and any
broadcast-constant injection.
"""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F

from .geocode import (GROUP_NAMES, GROUP_SIZES, ContNorm, GeoCode,
                      contract_facts)

__all__ = ["PCHFullConfig", "PCHFull", "PCHSession", "PositionEmbeddingRandom",
           "TwoWayTransformer"]


# --- SAM ports ---------------------------------------------------------------

class MLP(nn.Module):
    """SAM `modeling/mask_decoder.py::MLP` -- ReLU between layers, linear out."""

    def __init__(self, input_dim: int, hidden_dim: int, output_dim: int,
                 num_layers: int):
        super().__init__()
        self.num_layers = num_layers
        h = [hidden_dim] * (num_layers - 1)
        self.layers = nn.ModuleList(
            nn.Linear(n, k) for n, k in zip([input_dim] + h, h + [output_dim]))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for i, layer in enumerate(self.layers):
            x = F.relu(layer(x)) if i < self.num_layers - 1 else layer(x)
        return x


class MLPBlock(nn.Module):
    """SAM `modeling/common.py::MLPBlock`."""

    def __init__(self, embedding_dim: int, mlp_dim: int, act=nn.ReLU):
        super().__init__()
        self.lin1 = nn.Linear(embedding_dim, mlp_dim)
        self.lin2 = nn.Linear(mlp_dim, embedding_dim)
        self.act = act()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.lin2(self.act(self.lin1(x)))


class PositionEmbeddingRandom(nn.Module):
    """SAM `modeling/prompt_encoder.py::PositionEmbeddingRandom`, verbatim.

    Random spatial frequencies in a **registered buffer**, so the encoding is
    part of the checkpoint and a resumed arm keeps the geometry it was trained
    with.  ``forward((h, w))`` returns ``C x H x W`` with ``C = 2 * num_pos_feats``.
    """

    def __init__(self, num_pos_feats: int = 64, scale: float | None = None):
        super().__init__()
        if scale is None or scale <= 0.0:
            scale = 1.0
        self.register_buffer("positional_encoding_gaussian_matrix",
                             scale * torch.randn((2, num_pos_feats)))

    def _pe_encoding(self, coords: torch.Tensor) -> torch.Tensor:
        coords = 2 * coords - 1
        coords = coords @ self.positional_encoding_gaussian_matrix.to(coords.dtype)
        coords = 2 * math.pi * coords
        return torch.cat([torch.sin(coords), torch.cos(coords)], dim=-1)

    def forward(self, size: tuple[int, int]) -> torch.Tensor:
        h, w = size
        device = self.positional_encoding_gaussian_matrix.device
        grid = torch.ones((h, w), device=device, dtype=torch.float32)
        y_embed = (grid.cumsum(dim=0) - 0.5) / h
        x_embed = (grid.cumsum(dim=1) - 0.5) / w
        pe = self._pe_encoding(torch.stack([x_embed, y_embed], dim=-1))
        return pe.permute(2, 0, 1)                       # C x H x W


class Attention(nn.Module):
    """SAM `modeling/transformer.py::Attention` + a key-padding mask.

    The mask is the one deliberate addition (see module docstring): keys marked
    invalid receive ``-inf`` logits, so their softmax weight is exactly zero
    rather than merely small.
    """

    def __init__(self, embedding_dim: int, num_heads: int,
                 downsample_rate: int = 1):
        super().__init__()
        self.embedding_dim = embedding_dim
        self.internal_dim = embedding_dim // downsample_rate
        self.num_heads = num_heads
        if self.internal_dim % num_heads:
            raise ValueError("num_heads must divide the internal dim")
        self.q_proj = nn.Linear(embedding_dim, self.internal_dim)
        self.k_proj = nn.Linear(embedding_dim, self.internal_dim)
        self.v_proj = nn.Linear(embedding_dim, self.internal_dim)
        self.out_proj = nn.Linear(self.internal_dim, embedding_dim)
        #: filled by the last forward -- the observable the pad-mask test reads
        self.last_attn: torch.Tensor | None = None

    def _sep(self, x: torch.Tensor) -> torch.Tensor:
        b, n, c = x.shape
        return x.reshape(b, n, self.num_heads, c // self.num_heads).transpose(1, 2)

    @staticmethod
    def _rec(x: torch.Tensor) -> torch.Tensor:
        b, nh, n, c = x.shape
        return x.transpose(1, 2).reshape(b, n, nh * c)

    def forward(self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor,
                key_padding_mask: torch.Tensor | None = None,
                keep_attn: bool = False) -> torch.Tensor:
        q = self._sep(self.q_proj(q))
        k = self._sep(self.k_proj(k))
        v = self._sep(self.v_proj(v))
        attn = (q @ k.permute(0, 1, 3, 2)) / math.sqrt(q.shape[-1])
        if key_padding_mask is not None:
            # True = valid.  (B, N_k) -> (B, 1, 1, N_k)
            m = key_padding_mask[:, None, None, :]
            attn = attn.masked_fill(~m, float("-inf"))
        attn = torch.softmax(attn, dim=-1)
        if key_padding_mask is not None:
            # a query whose every key is masked would produce NaN; there is no
            # such query here (a sample always has valid cells), but the guard
            # keeps an all-pad row from poisoning the whole field silently
            attn = torch.nan_to_num(attn, nan=0.0)
        self.last_attn = attn.detach() if keep_attn else None
        return self.out_proj(self._rec(attn @ v))


class TwoWayAttentionBlock(nn.Module):
    """SAM `modeling/transformer.py::TwoWayAttentionBlock` + pad masking."""

    def __init__(self, embedding_dim: int, num_heads: int, mlp_dim: int = 2048,
                 activation=nn.ReLU, attention_downsample_rate: int = 2,
                 skip_first_layer_pe: bool = False):
        super().__init__()
        self.self_attn = Attention(embedding_dim, num_heads)
        self.norm1 = nn.LayerNorm(embedding_dim)
        self.cross_attn_token_to_image = Attention(
            embedding_dim, num_heads, downsample_rate=attention_downsample_rate)
        self.norm2 = nn.LayerNorm(embedding_dim)
        self.mlp = MLPBlock(embedding_dim, mlp_dim, activation)
        self.norm3 = nn.LayerNorm(embedding_dim)
        self.norm4 = nn.LayerNorm(embedding_dim)
        self.cross_attn_image_to_token = Attention(
            embedding_dim, num_heads, downsample_rate=attention_downsample_rate)
        self.skip_first_layer_pe = skip_first_layer_pe

    def forward(self, queries, keys, query_pe, key_pe, key_padding_mask=None,
                keep_attn: bool = False):
        if self.skip_first_layer_pe:
            queries = self.self_attn(q=queries, k=queries, v=queries)
        else:
            q = queries + query_pe
            queries = queries + self.self_attn(q=q, k=q, v=queries)
        queries = self.norm1(queries)

        q = queries + query_pe
        k = keys + key_pe
        queries = queries + self.cross_attn_token_to_image(
            q=q, k=k, v=keys, key_padding_mask=key_padding_mask,
            keep_attn=keep_attn)
        queries = self.norm2(queries)

        queries = queries + self.mlp(queries)
        queries = self.norm3(queries)

        q = queries + query_pe
        k = keys + key_pe
        attn_out = self.cross_attn_image_to_token(q=k, k=q, v=queries)
        if key_padding_mask is not None:
            # the *queries* of this direction are the image cells, so masking
            # keys is not enough: a pad cell must not accumulate a residual it
            # would then feed back through tap B
            attn_out = attn_out * key_padding_mask[..., None].to(attn_out.dtype)
        keys = keys + attn_out
        keys = self.norm4(keys)
        return queries, keys


class TwoWayTransformer(nn.Module):
    """SAM `modeling/transformer.py::TwoWayTransformer` + pad masking."""

    def __init__(self, depth: int, embedding_dim: int, num_heads: int,
                 mlp_dim: int, activation=nn.ReLU,
                 attention_downsample_rate: int = 2):
        super().__init__()
        self.depth = depth
        self.layers = nn.ModuleList(
            TwoWayAttentionBlock(embedding_dim, num_heads, mlp_dim, activation,
                                 attention_downsample_rate,
                                 skip_first_layer_pe=(i == 0))
            for i in range(depth))
        self.final_attn_token_to_image = Attention(
            embedding_dim, num_heads, downsample_rate=attention_downsample_rate)
        self.norm_final_attn = nn.LayerNorm(embedding_dim)

    def forward(self, keys: torch.Tensor, key_pe: torch.Tensor,
                point_embedding: torch.Tensor,
                key_padding_mask: torch.Tensor | None = None,
                keep_attn: bool = False):
        """``keys`` is (B, N, C) here rather than SAM's (B, C, H, W): the caller
        already has a flattened grid and flattening twice would only invite a
        transpose bug."""
        queries = point_embedding
        for layer in self.layers:
            queries, keys = layer(queries, keys, point_embedding, key_pe,
                                  key_padding_mask=key_padding_mask,
                                  keep_attn=keep_attn)
        q = queries + point_embedding
        k = keys + key_pe
        queries = queries + self.final_attn_token_to_image(
            q=q, k=k, v=keys, key_padding_mask=key_padding_mask,
            keep_attn=keep_attn)
        queries = self.norm_final_attn(queries)
        return queries, keys


# --- configuration ----------------------------------------------------------

@dataclass(frozen=True)
class PCHFullConfig:
    """§2.2 Full, and §2.2/D-11 Lite.

    ``feat_dim`` is the tower's channel count at the injection point.  §0.1-3
    settled it at **128** by reading `heads.py::apply_inject` (`ConvTower(ch=128)`),
    not the 1024 the proposal's prose assumed; the 1x1 adapters below are the
    "!= 32 -> add a 1x1" clause of §2.2(4) applied to the real number.
    """

    feat_dim: int = 128
    dim: int = 256
    depth: int = 2
    n_heads: int = 8
    mlp_dim: int = 2048
    downsample_rate: int = 2
    hyper_dim: int = 32          #: rank of tap A == adapted U channel count
    hyper_layers: int = 3
    tap_a: bool = True
    tap_b: bool = True
    tap_b_lite: bool = False     #: D-11: one zero-init 1x1 is both projection and zero_conv
    pe_scale: float = 1.0
    variant: str = "full"

    @staticmethod
    def full(**kw) -> "PCHFullConfig":
        return PCHFullConfig(**{**dict(variant="full"), **kw})

    @staticmethod
    def lite(**kw) -> "PCHFullConfig":
        """d=128 / heads=4 / depth=1 / mlp=256 / cross-internal 64; hypernet
        MLP(128,128,32,2); tap B-lite (D-11)."""
        base = dict(dim=128, depth=1, n_heads=4, mlp_dim=256,
                    downsample_rate=2, hyper_layers=2, tap_b_lite=True,
                    variant="lite")
        return PCHFullConfig(**{**base, **kw})

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


# --- the module -------------------------------------------------------------

class PCHSession:
    """One sample's injection, with the transformer run at most once.

    The two taps read different tensors at different points of the head (tap B
    modifies the tower's penultimate features; tap A dots the hypernetwork
    against what the field convolution then sees), so they cannot be one call --
    but they share the transformer output, and running it twice would double the
    module's cost for nothing.
    """

    def __init__(self, pch: "PCHFull", code: GeoCode | None,
                 valid_grid: torch.Tensor | None = None):
        self.pch = pch
        self.code = code
        self.valid_grid = valid_grid
        self.hs: torch.Tensor | None = None
        self.keys: torch.Tensor | None = None
        self._shape: tuple[int, int, int, int] | None = None

    # -- tap B ------------------------------------------------------------
    def residual(self, feat: torch.Tensor) -> torch.Tensor:
        hs, keys = self.pch._run(feat, self.code, self.valid_grid)
        self.hs, self.keys, self._shape = hs, keys, tuple(feat.shape)
        return self.pch._tap_b(feat, keys, self.valid_grid)

    # -- tap A ------------------------------------------------------------
    def logit(self, feat: torch.Tensor) -> torch.Tensor:
        if self.hs is None:
            hs, keys = self.pch._run(feat, self.code, self.valid_grid)
            self.hs, self.keys, self._shape = hs, keys, tuple(feat.shape)
        return self.pch._tap_a(feat, self.hs)


class PCHFull(nn.Module):
    """``(dense features, GeoCode) -> (feature residual, logit residual)``."""

    def __init__(self, cfg: PCHFullConfig | None = None,
                 cont_norm: ContNorm | None = None):
        super().__init__()
        self.cfg = cfg or PCHFullConfig.full()
        c = self.cfg
        d = c.dim

        # (1) prototypes.  nn.Embedding at its default init, exactly as
        # SurgicalSAM's Learnable_Prototypes and SAM's prompt tokens leave it.
        self.e_shape = nn.Embedding(GROUP_SIZES["shape"], d)
        self.e_dir = nn.Embedding(GROUP_SIZES["dir"], d)
        self.e_ext = nn.Embedding(GROUP_SIZES["ext"], d)
        self.w_cont = nn.Linear(3, d)
        self.null_g = nn.Embedding(len(GROUP_NAMES), d)
        self.field_token = nn.Embedding(1, d)
        self.ln_group = nn.ModuleList(nn.LayerNorm(d) for _ in GROUP_NAMES)

        # (2) keys
        self.k_proj = nn.Conv2d(c.feat_dim, d, 1)
        self.k_norm = nn.LayerNorm(d)
        self.pe = PositionEmbeddingRandom(d // 2, scale=c.pe_scale)

        # (3) two-way transformer
        self.transformer = TwoWayTransformer(
            depth=c.depth, embedding_dim=d, num_heads=c.n_heads,
            mlp_dim=c.mlp_dim, attention_downsample_rate=c.downsample_rate)

        # (4) tap A -- rank-1 hypernetwork
        if c.tap_a:
            self.hyper = MLP(d, d, c.hyper_dim, c.hyper_layers)
            self.u_adapt = nn.Conv2d(c.feat_dim, c.hyper_dim, 1)
            self.gamma = nn.Parameter(torch.zeros(()))     # tanh(0) = 0
        # (5) tap B -- ControlNet-style zero conv
        if c.tap_b:
            if c.tap_b_lite:
                self.tap_b_out = nn.Conv2d(d, c.feat_dim, 1)
                _zero(self.tap_b_out)
            else:
                self.w_out = nn.Linear(d, c.feat_dim)
                self.zero_conv = nn.Conv2d(c.feat_dim, c.feat_dim, 1)
                _zero(self.zero_conv)

        # meta.norm for c_cont, carried in the checkpoint so a resumed arm
        # cannot be scored under a different standardisation than it trained on
        n = cont_norm or ContNorm()
        self.register_buffer("cont_mean", torch.tensor(n.mean, dtype=torch.float32))
        self.register_buffer("cont_std", torch.tensor(n.std, dtype=torch.float32))
        self.cont_norm = n

        #: Delta-domain recorder (§2.4: the assertion ships with the delivery).
        #: Plain attributes, not buffers -- these are observations about a run,
        #: not weights, and putting them in the state_dict would make two runs
        #: with identical weights compare unequal.
        self.reset_domain()

    def set_cont_norm(self, norm: ContNorm) -> None:
        """Install the arm constants for ``c_cont`` (§2.4 / s-cache contract).

        Stored in buffers so the standardisation travels with the checkpoint: an
        arm scored later under different constants is scored on a different
        input than it trained on, and nothing downstream would show it.
        """
        self.cont_norm = norm
        with torch.no_grad():
            self.cont_mean.copy_(torch.tensor(norm.mean, dtype=torch.float32))
            self.cont_std.copy_(torch.tensor(norm.std, dtype=torch.float32))

    # -- reporting ---------------------------------------------------------
    def reset_domain(self) -> None:
        self._dom: dict[str, dict[str, float]] = {}

    def _record(self, kind: str, t: torch.Tensor) -> None:
        v = t.detach()
        if not torch.isfinite(v).all():
            raise AssertionError(
                f"PCH produced a non-finite {kind}; every downstream field and "
                "every Delta computed from it would be meaningless")
        d = self._dom.setdefault(kind, {"n": 0.0, "min": float("inf"),
                                        "max": float("-inf"), "absmax": 0.0,
                                        "absmean_sum": 0.0})
        d["n"] += 1
        d["min"] = min(d["min"], float(v.min()))
        d["max"] = max(d["max"], float(v.max()))
        d["absmax"] = max(d["absmax"], float(v.abs().max()))
        d["absmean_sum"] += float(v.abs().mean())

    def domain_report(self) -> dict[str, Any]:
        """The Delta-logit / Delta-feature domains, for the delivery folder.

        §2.4: *any* "the injection did nothing" claim must first show this
        record.  Without it, a dead tap and a genuinely null effect are the same
        number.
        """
        out: dict[str, Any] = {"gamma": float(self.gamma.detach())
                               if self.cfg.tap_a else None,
                               "tanh_gamma": float(torch.tanh(self.gamma.detach()))
                               if self.cfg.tap_a else None}
        for kind, d in self._dom.items():
            n = max(d["n"], 1.0)
            out[kind] = {"n_calls": int(d["n"]), "min": d["min"], "max": d["max"],
                         "absmax": d["absmax"], "absmean": d["absmean_sum"] / n}
        return out

    def n_params(self) -> int:
        return sum(p.numel() for p in self.parameters())

    def facts(self) -> dict[str, Any]:
        return {
            "module": "PCH-Full (proposal §2.2)" if self.cfg.variant == "full"
                      else "PCH-Lite (proposal §2.2 / D-11)",
            "pch": self.cfg.to_dict(),
            "n_params": self.n_params(),
            "params_by_part": self._params_by_part(),
            "contract": contract_facts(),
            "cont_norm": self.cont_norm.to_dict(),
            "zero_init_output": True,
            "null_path": "conf=0 -> learned null_g constant (not zero); the "
                         "exact no-op guarantee is tanh(gamma=0) + zero_conv",
            "per_image_normalisation": "none (red line)",
        }

    def _params_by_part(self) -> dict[str, int]:
        groups: dict[str, int] = {}
        for name, p in self.named_parameters():
            groups[name.split(".")[0]] = groups.get(name.split(".")[0], 0) + p.numel()
        return groups

    # -- code -> tokens ----------------------------------------------------
    def tokens_of(self, code: GeoCode | None, *, device=None, dtype=None
                  ) -> torch.Tensor:
        """``(1, 5, d)`` = [field token; T_shape; T_dir; T_ext; T_cont]."""
        device = device or self.field_token.weight.device
        dtype = dtype or self.field_token.weight.dtype
        if code is None:
            code = GeoCode.null(device=device)
        code = code.to(device=device, dtype=dtype)
        code.validate()

        tables = {"shape": self.e_shape, "dir": self.e_dir, "ext": self.e_ext}
        toks = []
        for gi, g in enumerate(GROUP_NAMES):
            if g == "cont":
                raw = code.c_cont
                self.cont_norm.assert_in_domain(raw)
                z = (raw - self.cont_mean.to(raw)) / self.cont_std.to(raw).clamp_min(1e-6)
                t_g = self.ln_group[gi](self.w_cont(z))
            else:
                c = code.group(g)                       # (n_g,)
                # sum_i c_i E[i] / max(sum_i c_i, 1): a mean over the ACTIVE
                # slots when several fire, and the plain embedding when one
                # does, so a two-hot code is not twice as loud as a one-hot one
                w = c / torch.clamp(c.sum(), min=1.0)
                t_g = self.ln_group[gi](w @ tables[g].weight)
            null = self.null_g.weight[gi]
            conf = code.conf[gi]
            toks.append(conf * t_g + (1.0 - conf) * null)
        return torch.cat([self.field_token.weight, torch.stack(toks)],
                         dim=0).unsqueeze(0)

    # -- internals ---------------------------------------------------------
    def _keys_of(self, feat: torch.Tensor):
        b, ch, h, w = feat.shape
        if ch != self.cfg.feat_dim:
            raise AssertionError(
                f"PCH was built for a {self.cfg.feat_dim}-channel injection "
                f"point and got {ch}; §0.1-3 fixed this at 128 by reading "
                "heads.py::apply_inject -- a silent mismatch here would train a "
                "module against the wrong tensor")
        k = self.k_proj(feat).reshape(b, self.cfg.dim, h * w).transpose(1, 2)
        k = self.k_norm(k)
        pe = self.pe((h, w)).to(k.dtype).reshape(1, self.cfg.dim, h * w)
        return k, pe.transpose(1, 2).expand(b, -1, -1)

    def _run(self, feat: torch.Tensor, code: GeoCode | None,
             valid_grid: torch.Tensor | None, keep_attn: bool = False):
        keys, key_pe = self._keys_of(feat)
        tokens = self.tokens_of(code, device=feat.device, dtype=feat.dtype)
        tokens = tokens.expand(feat.shape[0], -1, -1)
        mask = None
        if valid_grid is not None:
            mask = valid_grid.reshape(feat.shape[0], -1).bool()
            if not bool(mask.any()):
                raise AssertionError("valid_grid marks every cell invalid")
        return self.transformer(keys, key_pe, tokens, key_padding_mask=mask,
                                keep_attn=keep_attn)

    def _tap_a(self, feat: torch.Tensor, hs: torch.Tensor) -> torch.Tensor:
        """``tanh(gamma) * (w . U)`` -- the rank-1 logit residual."""
        if not self.cfg.tap_a:
            return torch.zeros(feat.shape[0], 1, *feat.shape[-2:],
                               device=feat.device, dtype=feat.dtype)
        b, _, h, w = feat.shape
        wv = self.hyper(hs[:, 0])                              # (B, hyper_dim)
        u = self.u_adapt(feat).reshape(b, self.cfg.hyper_dim, h * w)
        logit_cond = torch.bmm(wv.unsqueeze(1), u).reshape(b, 1, h, w)
        out = torch.tanh(self.gamma) * logit_cond
        self._record("delta_logit", out)
        return out

    def _tap_b(self, feat: torch.Tensor, keys: torch.Tensor,
               valid_grid: torch.Tensor | None) -> torch.Tensor:
        if not self.cfg.tap_b:
            return torch.zeros_like(feat)
        b, _, h, w = feat.shape
        if self.cfg.tap_b_lite:
            x = keys.transpose(1, 2).reshape(b, self.cfg.dim, h, w)
            out = self.tap_b_out(x)
        else:
            x = self.w_out(keys).transpose(1, 2).reshape(b, self.cfg.feat_dim, h, w)
            out = self.zero_conv(x)
        if valid_grid is not None:
            out = out * valid_grid.reshape(b, 1, h, w).to(out.dtype)
        self._record("delta_feature", out)
        return out

    # -- public API --------------------------------------------------------
    def session(self, code: GeoCode | None,
                valid_grid: torch.Tensor | None = None) -> PCHSession:
        return PCHSession(self, code, valid_grid)

    def forward(self, feat: torch.Tensor, code: GeoCode | None,
                valid_grid: torch.Tensor | None = None
                ) -> tuple[torch.Tensor, torch.Tensor]:
        """``(feature residual, logit residual)`` for one call site that wants both."""
        s = self.session(code, valid_grid)
        res = s.residual(feat)
        return res, s.logit(feat + res)

    # -- runtime assertions ------------------------------------------------
    @torch.no_grad()
    def assert_zero_at_init(self, feat: torch.Tensor, code: GeoCode) -> dict[str, Any]:
        """Step 0 is an exact no-op: both taps return exact zeros.

        Called by the harness before training, not only by the unit test: the
        arms resume a trained checkpoint, and "the module perturbed step 0" is
        indistinguishable in the board from "the injection helped/hurt".
        """
        res, dlog = self.forward(feat, code)
        nz_r = int(torch.count_nonzero(res))
        nz_l = int(torch.count_nonzero(dlog))
        if nz_r or nz_l:
            raise AssertionError(
                f"PCH is not a no-op at initialisation: {nz_r} non-zero feature "
                f"residual cells and {nz_l} non-zero logit cells.  The zero-init "
                "of zero_conv / gamma is the entire reason this module may be "
                "added to a resumed checkpoint")
        return {"zero_at_init": True, "n_cells": int(res.numel())}

    @torch.no_grad()
    def assert_null_is_content_free(self, feat: torch.Tensor,
                                    code_a: GeoCode, code_b: GeoCode
                                    ) -> dict[str, Any]:
        """With conf=0 the output must not depend on the code's content.

        This is the *post-training* form of the fallback claim, and the one that
        can actually fail: at init everything is zero for structural reasons, so
        a leak from c_disc into the null path would hide until the module had
        trained.  Two different codes, both with conf=0, must give bit-identical
        outputs.
        """
        z = torch.zeros(len(GROUP_NAMES))
        a = code_a.replace(conf=z.clone(), valid=False)
        b = code_b.replace(conf=z.clone(), valid=False)
        ra, la = self.forward(feat, a)
        rb, lb = self.forward(feat, b)
        if not (torch.equal(ra, rb) and torch.equal(la, lb)):
            raise AssertionError(
                "conf=0 does not degenerate to the null constant: two different "
                "codes produced different outputs, so code content is leaking "
                "past the confidence gate and M3/ABSTAIN samples are not on the "
                "path they are reported to be on")
        return {"null_is_content_free": True}


def _zero(m: nn.Module) -> None:
    """ControlNet `cldm/cldm.py::make_zero_conv` -> `zero_module`: every
    parameter to zero, so the branch starts as an exact identity."""
    for p in m.parameters():
        nn.init.zeros_(p)
