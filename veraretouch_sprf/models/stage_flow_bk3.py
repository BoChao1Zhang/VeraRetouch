#!/usr/bin/env python3
# 源自 experiments/prs/EPR-051_masked-restore-production/stage0/sprf/stage_flow_bk3.py @ 2026-09-06（原件未入 git；原 sha256 见 veraretouch_sprf/PROVENANCE.md），逐字复制/仅改导入与路径解析（EPR-052）。
"""EPR-051 · BK-ABL-v2 · 追加两臂（E1：`stage_flow_bk.py` / `stage_flow_bk2.py` 被在跑作业 sha 冻结，不改）。

  canonfilm  每层 → FFN 残差块内「激活后」FiLM      CanonCGT `Estimator_blocks.py`
             `Point_Feed_Forward_FiLM` L122-148、`FiLM_Layer` L56-69、`EncoderBlock.forward` L21-23（x = x + ffn(x, g)）
  adagn      每层 FiLM → GroupNorm 后 scale/shift   RDBM `code/networks.py` `Block` L126-142（proj → GroupNorm(8) → x·(scale+1)+shift → SiLU），
             `ResnetBlock.mlp = SiLU → Linear(time_emb_dim, 2·dim_out)` L148-151、`chunk(2)` L164

共享外壳、逆表、负控制、features、A12 探针接口全部复用 `stage_flow_bk.BkSprfModel`。
逐臂「搬 / 改 / 待决策」见 NOTES_backend.md D-bk11。
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from veraretouch_sprf.models import stage_flow_bk as SFB
from veraretouch_sprf.data.stage_targets import die

edit_spec = SFB.edit_spec
BK_ARMS = ("canonfilm", "adagn")
_n = SFB._n


class FiLMLayer(nn.Module):
    """CanonCGT `FiLM_Layer` L56-69 逐字：scale/shift 各 Linear(dim,dim)-GELU-Linear(dim,hidden)。默认初始化。"""

    def __init__(self, dim: int, hidden_dim: int):
        super().__init__()
        self.scale = nn.Sequential(nn.Linear(dim, dim), nn.GELU(), nn.Linear(dim, hidden_dim))
        self.shift = nn.Sequential(nn.Linear(dim, dim), nn.GELU(), nn.Linear(dim, hidden_dim))

    def forward(self, x):
        return self.scale(x), self.shift(x)


class PointFFNFiLM(nn.Module):
    """CanonCGT `Point_Feed_Forward_FiLM` L122-148：norm → ffn1(1×1, dim→4dim) → GELU →
    res·scale+shift（scale/shift = FiLM_Layer(style)）→ ffn2(1×1, 4dim→dim)；外层 x = x + ffn(x, g)（L23）。
    改：`GroupNorm(1, dim)`（原码注释「equivalent with LayerNorm」，但在 [b,c,h,w] 上含空间统计）
    → 逐像素通道 LayerNorm（逐像素契约：一个像素的输出不依赖同批其他像素）。style 维 = hf 维（384）。
    """

    def __init__(self, dim: int, style_dim: int, ratio: int):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        hidden = dim * int(ratio)
        self.ffn1 = nn.Conv1d(dim, hidden, 1)
        self.act = nn.GELU()
        self.ffn2 = nn.Conv1d(hidden, dim, 1)
        self.FiLM = FiLMLayer(style_dim, hidden)

    def forward(self, feat, style):                      # feat (B,C,P)
        x = self.norm(feat.transpose(1, 2)).transpose(1, 2)
        res = self.act(self.ffn1(x))
        scale, shift = self.FiLM(style)
        res = res * scale.unsqueeze(-1) + shift.unsqueeze(-1)
        return self.ffn2(res)


class CanonFilmBackend(nn.Module):
    """首层 1×1(19→W)（CanonCGT 的 token 来自 backbone；本栈无空间，首层投影同 FULL conv1 位）；
    8 个 FFN-FiLM 残差块（style = hf = [h ⊕ e_m]）；heads 同 FULL（action 零初始化 ⇒ A1）。"""

    def __init__(self, core: SFB.FullCondCore, in_ch: int, width: int, n_layers: int,
                 ratio: int, layer_ckpt: bool):
        super().__init__()
        self.core = core
        self.conv_in = nn.Conv1d(int(in_ch), int(width), 1)
        self.blocks = nn.ModuleList([PointFFNFiLM(int(width), core.hf_dim, ratio)
                                     for _ in range(int(n_layers))])
        self.action, self.clean = SFB._heads(width)
        self.ckpt = bool(layer_ckpt)
        self.width, self.n_layers, self.ratio = int(width), int(n_layers), int(ratio)

    def base(self, c, edits=None):
        return self.core.base(c).unsqueeze(1)

    def macs_per_pixel(self) -> dict:
        w, L = self.width, self.n_layers
        blk = 2 * w * w * self.ratio
        return dict(per_block=blk, blocks=L * blk, total=L * blk + self.conv_in.in_channels * w + 2 * w * 3)

    def forward(self, base, feat, z, y, st, sid, depth, s, ed, m):
        h, hf = self.core(base[:, 0], st, sid, depth, s, ed)
        x = self.conv_in(feat)
        for blk in self.blocks:
            def layer(xx, gg, blk=blk):
                return xx + blk(xx, gg)
            x = SFB._ckpt(layer, self.ckpt, x, hf)
        return self.action(x), self.clean(x)

    def a12_probe(self):
        d = self.core.hidden
        return dict(name="blocks[0].FiLM.scale[0].weight[:, hidden:hidden+edit_latent]",
                    param=self.blocks[0].FiLM.scale[0].weight, cols=slice(d, d + self.core.edit_latent))

    def param_groups(self) -> dict:
        return dict(conditioner_core=_n(self.core) - _n(self.core.edit_enc),
                    edit_encoder=_n(self.core.edit_enc), conv_in=_n(self.conv_in),
                    ffn=sum(_n(b.ffn1) + _n(b.ffn2) + _n(b.norm) for b in self.blocks),
                    film_layers=sum(_n(b.FiLM) for b in self.blocks),
                    action_head=_n(self.action), clean_head=_n(self.clean))


class AdaGNBackend(nn.Module):
    """RDBM `Block` L126-142：x = proj(x); x = GroupNorm(groups)(x); x = x·(scale+1)+shift; x = SiLU(x)；
    scale, shift = `SiLU → Linear(cond, 2·dim_out)`(cond).chunk(2)（L148-151, L164）。
    改：proj 3×3 WS-Conv → 本栈原有 1×1 conv（不加权重标准化，待决策）；cond = hf（[h ⊕ e_m]）而非时间步嵌入；
    GroupNorm 统计只在**逐像素的通道组**上（逐像素契约；RDBM 的 GN 含空间统计）。FiLM 头去掉；其余同 FULL。
    """

    def __init__(self, core: SFB.FullCondCore, in_ch: int, width: int, n_layers: int,
                 groups: int, layer_ckpt: bool):
        super().__init__()
        self.core = core
        self.convs = SFB._full_convs(in_ch, width, n_layers)
        self.G = int(groups)
        if int(width) % self.G:
            die(f"width {width} 不能被 gn_groups {groups} 整除")
        self.gn_weight = nn.ParameterList([nn.Parameter(torch.ones(int(width))) for _ in range(int(n_layers))])
        self.gn_bias = nn.ParameterList([nn.Parameter(torch.zeros(int(width))) for _ in range(int(n_layers))])
        self.mlps = nn.ModuleList([nn.Sequential(nn.SiLU(), nn.Linear(core.hf_dim, 2 * int(width)))
                                   for _ in range(int(n_layers))])
        self.action, self.clean = SFB._heads(width)
        self.ckpt = bool(layer_ckpt)
        self.width, self.n_layers = int(width), int(n_layers)
        self.eps = 1e-5

    def base(self, c, edits=None):
        return self.core.base(c).unsqueeze(1)

    def _gn_pixelwise(self, x, i):                       # x (B,C,P)：每像素在 G 个通道组内归一化
        B, C, P = x.shape
        xg = x.view(B, self.G, C // self.G, P)
        mu = xg.mean(dim=2, keepdim=True)
        var = xg.var(dim=2, keepdim=True, unbiased=False)
        xg = (xg - mu) * torch.rsqrt(var + self.eps)
        return xg.view(B, C, P) * self.gn_weight[i].view(1, C, 1) + self.gn_bias[i].view(1, C, 1)

    def forward(self, base, feat, z, y, st, sid, depth, s, ed, m):
        h, hf = self.core(base[:, 0], st, sid, depth, s, ed)
        x = feat
        for i in range(self.n_layers):
            scale, shift = self.mlps[i](hf).chunk(2, dim=1)

            def layer(xx, sc, sh, i=i):
                u = self._gn_pixelwise(self.convs[i](xx), i)
                return F.silu(u * (sc.unsqueeze(-1) + 1) + sh.unsqueeze(-1))
            x = SFB._ckpt(layer, self.ckpt, x, scale, shift)
        return self.action(x), self.clean(x)

    def a12_probe(self):
        d = self.core.hidden
        return dict(name="mlps[0][1].weight[:, hidden:hidden+edit_latent]",
                    param=self.mlps[0][1].weight, cols=slice(d, d + self.core.edit_latent))

    def param_groups(self) -> dict:
        return dict(conditioner_core=_n(self.core) - _n(self.core.edit_enc),
                    edit_encoder=_n(self.core.edit_enc), trunk_convs=_n(self.convs),
                    gn_affine=sum(p.numel() for p in self.gn_weight) + sum(p.numel() for p in self.gn_bias),
                    scale_shift_mlps=_n(self.mlps), action_head=_n(self.action), clean_head=_n(self.clean))


class BkSprfModel(SFB.SprfModelBase if hasattr(SFB, "SprfModelBase") else SFB.SF.SprfModel):
    """外壳同 stage_flow_bk.BkSprfModel（复制其 __init__ 骨架，只换臂表）。"""

    def __init__(self, cond_in: int, cfg, n_steps: int, alpha_mode: str, depth_values):
        super().__init__(cond_in, cfg, n_steps, alpha_mode, depth_values)
        if self.action_backend != "mlp" or self.edit_condition != "inv_lut" or self.lut_id_condition != "off":
            die("BK 臂只定义在 mlp / inv_lut(oracle_lut) / lut_id off 上")
        del self.cond
        del self.net
        arm = cfg.str_("bk", "arm", BK_ARMS)
        self.arm = arm
        width = cfg.int_("flow", "width")
        n_layers = cfg.int_("flow", "n_layers")
        ec = self.edit
        layer_ckpt = cfg.bool_("bk", "layer_checkpoint")
        core = SFB.FullCondCore(cond_in, cfg.int_("flow", "cond_hidden"), cfg.int_("flow", "cond_layers"),
                                n_stage_types=int(n_steps), n_stage_ids=int(n_steps),
                                n_depths=int(n_steps) + 1, edit_in=ec["descriptor_dim"],
                                edit_hidden=ec["enc_hidden"], edit_layers=ec["enc_layers"],
                                edit_latent=ec["latent_dim"])
        if arm == "canonfilm":
            bk = CanonFilmBackend(core, self.in_ch, width, n_layers, cfg.int_("bk", "ffn_ratio"), layer_ckpt)
        else:
            bk = AdaGNBackend(core, self.in_ch, width, n_layers, cfg.int_("bk", "gn_groups"), layer_ckpt)
        self.bk = bk
        self.cond = bk
        self.net = None

    @property
    def edit_encoder(self) -> nn.Module:
        return self.bk.core.edit_enc

    forward = SFB.BkSprfModel.forward
    param_counts = SFB.BkSprfModel.param_counts
