#!/usr/bin/env python3
# 源自 experiments/prs/EPR-051_masked-restore-production/stage0/sprf/stage_flow_bk4.py @ 2026-09-06（原件未入 git；原 sha256 见 veraretouch_sprf/PROVENANCE.md），逐字复制/仅改导入与路径解析（EPR-052）。
"""EPR-051 · BK-ABL-v2 · 叠加臂（E1：bk/bk2/bk3 均被已提交作业 sha 冻结，另开本文件）。

  adagn_ff          = BK-ADAGN（RDBM GN 后 scale/shift，stage_flow_bk3.AdaGNBackend 原样）
                      + 首层输入 z、y 的 8 频正弦编码（MetaDC PositionalEncoding，stage_flow_bk.FFBackend._pe 原样）
  adagn_ff_affhead  = adagn_ff + 线性头 → 3×4 仿射（恒等初始化）+ 0.1·tanh 残差（MetaDC，stage_flow_bk.AffHeadBackend 原样）

每一级只比上一级多一处改动；其余（条件器、GN、scale/shift MLP、逐层重算、clean 头）逐字复用。
"""
from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from veraretouch_sprf.models import stage_flow_bk as SFB
from veraretouch_sprf.models import stage_flow_bk3 as SFB3
from veraretouch_sprf.data.stage_targets import die

edit_spec = SFB.edit_spec
BK_ARMS = ("adagn_ff", "adagn_ff_affhead")
_n = SFB._n


class AdaGNFFBackend(SFB3.AdaGNBackend):
    def __init__(self, core, in_ch, width, n_layers, groups, layer_ckpt, n_freq, affine_head):
        self.n_freq = int(n_freq)
        pe_ch = 3 + 3 * 2 * self.n_freq
        in_ch2 = 2 * pe_ch + (int(in_ch) - 6)
        super().__init__(core, in_ch2, width, n_layers, groups, layer_ckpt)
        self.register_buffer("freq_bands", 2 ** torch.linspace(0, self.n_freq - 1, self.n_freq),
                             persistent=False)
        self.in_ch_raw, self.in_ch = int(in_ch), int(in_ch2)
        self.affine_head = bool(affine_head)
        if self.affine_head:
            del self.action
            self.head = nn.Conv1d(int(width), 15, 1)          # MetaDC mlp_head (Linear(hidden,15))
            nn.init.zeros_(self.head.weight)
            with torch.no_grad():
                self.head.bias.fill_(0)
                self.head.bias[0] = 1.0                        # 3×4 行主序恒等：0/5/10（见 D-bk3 affhead）
                self.head.bias[5] = 1.0
                self.head.bias[10] = 1.0

    _pe = SFB.FFBackend._pe                                    # MetaDC PositionalEncoding 逐字

    def forward(self, base, feat, z, y, st, sid, depth, s, ed, m):
        f = torch.cat([self._pe(feat[:, :3]), self._pe(feat[:, 3:6]), feat[:, 6:]], dim=1)
        if f.shape[1] != self.in_ch:
            die(f"adagn_ff 首层通道 {f.shape[1]} != {self.in_ch}")
        h, hf = self.core(base[:, 0], st, sid, depth, s, ed)
        x = f
        for i in range(self.n_layers):                         # == AdaGNBackend.forward 的层循环（逐字）
            scale, shift = self.mlps[i](hf).chunk(2, dim=1)

            def layer(xx, sc, sh, i=i):
                u = self._gn_pixelwise(self.convs[i](xx), i)
                return F.silu(u * (sc.unsqueeze(-1) + 1) + sh.unsqueeze(-1))
            x = SFB._ckpt(layer, self.ckpt, x, scale, shift)
        if not self.affine_head:
            return self.action(x), self.clean(x)
        out = self.head(x)                                     # == AffHeadBackend.forward 的头（逐字）
        B, _, P = out.shape
        M = out[:, :12].view(B, 3, 4, P)
        zt = z.permute(0, 2, 1)
        zh = torch.cat([zt, torch.ones(B, 1, P, device=z.device, dtype=z.dtype)], dim=1)
        out_matrix = (M * zh.unsqueeze(1)).sum(dim=2)
        final = out_matrix + torch.tanh(out[:, 12:]) * 0.1
        return final - zt, self.clean(x)

    def param_groups(self) -> dict:
        d = dict(conditioner_core=_n(self.core) - _n(self.core.edit_enc),
                 edit_encoder=_n(self.core.edit_enc), trunk_convs=_n(self.convs),
                 gn_affine=sum(p.numel() for p in self.gn_weight) + sum(p.numel() for p in self.gn_bias),
                 scale_shift_mlps=_n(self.mlps), clean_head=_n(self.clean))
        d["affine_head" if self.affine_head else "action_head"] = _n(self.head if self.affine_head else self.action)
        return d


class BkSprfModel(SFB.SF.SprfModel):
    def __init__(self, cond_in: int, cfg, n_steps: int, alpha_mode: str, depth_values):
        super().__init__(cond_in, cfg, n_steps, alpha_mode, depth_values)
        if self.action_backend != "mlp" or self.edit_condition != "inv_lut" or self.lut_id_condition != "off":
            die("BK 臂只定义在 mlp / inv_lut(oracle_lut) / lut_id off 上")
        del self.cond
        del self.net
        arm = cfg.str_("bk", "arm", BK_ARMS)
        self.arm = arm
        ec = self.edit
        core = SFB.FullCondCore(cond_in, cfg.int_("flow", "cond_hidden"), cfg.int_("flow", "cond_layers"),
                                n_stage_types=int(n_steps), n_stage_ids=int(n_steps),
                                n_depths=int(n_steps) + 1, edit_in=ec["descriptor_dim"],
                                edit_hidden=ec["enc_hidden"], edit_layers=ec["enc_layers"],
                                edit_latent=ec["latent_dim"])
        bk = AdaGNFFBackend(core, self.in_ch, cfg.int_("flow", "width"), cfg.int_("flow", "n_layers"),
                            cfg.int_("bk", "gn_groups"), cfg.bool_("bk", "layer_checkpoint"),
                            cfg.int_("bk", "n_freq"), affine_head=(arm == "adagn_ff_affhead"))
        self.bk = bk
        self.cond = bk
        self.net = None

    @property
    def edit_encoder(self) -> nn.Module:
        return self.bk.core.edit_enc

    forward = SFB.BkSprfModel.forward
    param_counts = SFB.BkSprfModel.param_counts
