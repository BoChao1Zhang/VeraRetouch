#!/usr/bin/env python3
# 源自 experiments/prs/EPR-051_masked-restore-production/stage0/sprf/stage_flow_bk.py @ 2026-09-06（原件未入 git；原 sha256 见 veraretouch_sprf/PROVENANCE.md），逐字复制/仅改导入与路径解析（EPR-052）。
"""EPR-051 · stage0 · sprf · BK-ABL-v2 分支模型文件（E1：`stage_flow.py` 一字不改）。

六个 backend 臂，每臂只替换 FULL 基座（`stage_flow.SprfModel`：条件 MLP 求和 → 每层
图级 FiLM 零初始化 → 8 层 1×1 conv w=512 SiLU → 线性头零初始化）的**一块**：

  pxfilm    图级 FiLM → 逐像素 FiLM             MetaDC-INR `model.py` film_gen L85-92 / L144-163
  loramoe   每层 FiLM → W x + Σ_i α_i B_i A_i x   FAPE-IR `modeling_fapeir_denoise_tower_moe.py`
                                                  add_lora_moe_to_linear L19-112 / forward_with_lora_moe L155-243
  ditblk    每层 → 逐像素 DiT 块（adaLN(τ)+cross-attn+SwiGLU）
                                                  RFMSR `lightningdit.py` LightningDiTBlock L207-274,
                                                  TimestepEmbedder L171-201, MultiHeadCrossAttention L25-84,
                                                  initialize_weights L393-417；`swiglu_ffn.py`；`rmsnorm.py`
  affhead   线性头 → 3×4 仿射 + 0.1·tanh 残差    MetaDC-INR `model.py` _initialize_weights L101-111, forward L165-181
  clutflow  8 层主干 → 8 基 LUT 混合 + F_θ       FlowLUT arXiv:2509.23608 式 (4)-(10) + §III-D F_θ；
                                                  三线性 = IA-3DLUT `models.py` TrilinearInterpolation L289 的
                                                  grid_sample 等价形；零 LUT 残差参数化 = Generator3DLUT_zero L277
  ff        首层输入 → 8 频正弦编码              MetaDC-INR `model.py` PositionalEncoding L26-50（color_pe 8 freq）

共享契约（与 FULL 逐字相同）：逐像素输入 [z,y,β_obs,β_state,λ]（alpha_full = 19 通道）；
输出 ĥ,x̂0 ∈ R³；action 末层零初始化 ⇒ A1 整链 bit-exact 恒等；`model(base, z, y, ahat,
beta, alphas, lam, m, depth, s, lut_id, edit_m)` 签名不变。**唯一的接口扩展**：
`model.cond.base(c, edits_full)`——ditblk 的 K/V token 需要整链 6 个编辑，其余臂忽略第二参。
`base` 恒为 (B, T, H) 三维张量（沿 dim0 可切片、可经 checkpoint 透传）。

逐臂「搬了哪段 / 改了哪段 / 待决策」见 NOTES_backend.md D-bk3。
"""
from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.checkpoint

from veraretouch_sprf.models import stage_flow as SF
from veraretouch_sprf.data.stage_targets import die

BK_ARMS = ("pxfilm", "loramoe", "ditblk", "affhead", "clutflow", "ff")

# 共享栈从本模块取的原件符号（与 stage_flow 同一对象）
edit_spec = SF.edit_spec
EDIT_CONDITIONS, EDIT_CONTRACTS, EDIT_CHANNELS = SF.EDIT_CONDITIONS, SF.EDIT_CONTRACTS, SF.EDIT_CHANNELS
alpha_channels, pointwise_in_channels = SF.alpha_channels, SF.pointwise_in_channels


def _ckpt(fn, enabled: bool, *args):
    """训练期逐层重算（数学与梯度不变，只换显存）；无梯度时直通。"""
    if enabled and torch.is_grad_enabled():
        return torch.utils.checkpoint.checkpoint(fn, *args, use_reentrant=False)
    return fn(*args)


def _n(mod) -> int:
    return 0 if mod is None else sum(p.numel() for p in mod.parameters())


# --------------------------------------------------------------------------- #
# FULL 条件器核心（stage_flow.StageConditioner L100-177 逐字搬，去掉 film 头）
# --------------------------------------------------------------------------- #
class FullCondCore(nn.Module):
    """trunk(c) + emb_type + emb_stage + emb_depth + s_proj → SiLU → h；hf = [h, edit_enc(edit)]。"""

    def __init__(self, cond_in: int, hidden: int, cond_layers: int,
                 n_stage_types: int, n_stage_ids: int, n_depths: int,
                 edit_in: int, edit_hidden: int, edit_layers: int, edit_latent: int):
        super().__init__()
        if cond_layers < 1:
            die(f"cond_layers = {cond_layers} 必须 >= 1")
        seq: list[nn.Module] = []
        d = int(cond_in)
        for _ in range(int(cond_layers)):
            seq += [nn.Linear(d, hidden), nn.SiLU()]
            d = hidden
        self.trunk = nn.Sequential(*seq)
        self.emb_type = nn.Embedding(int(n_stage_types), hidden)
        self.emb_stage = nn.Embedding(int(n_stage_ids), hidden)
        self.emb_depth = nn.Embedding(int(n_depths), hidden)
        self.s_proj = nn.Linear(1, hidden)
        self.edit_latent = int(edit_latent)
        if self.edit_latent < 1:
            die("BK 臂只定义在 edit.condition = inv_lut（oracle_lut）契约上")
        if int(edit_layers) < 2:
            die(f"edit_layers = {edit_layers} 必须 >= 2（2-3 层 MLP）")
        es: list[nn.Module] = []
        ed = int(edit_in)
        for _ in range(int(edit_layers) - 1):
            es += [nn.Linear(ed, int(edit_hidden)), nn.SiLU()]
            ed = int(edit_hidden)
        es += [nn.Linear(ed, self.edit_latent)]
        self.edit_enc = nn.Sequential(*es)
        self.hidden = int(hidden)
        self.n_stage_types, self.n_stage_ids, self.n_depths = (
            int(n_stage_types), int(n_stage_ids), int(n_depths))

    @property
    def hf_dim(self) -> int:
        return self.hidden + self.edit_latent

    def base(self, c: torch.Tensor) -> torch.Tensor:
        return self.trunk(c)

    def forward(self, base: torch.Tensor, stage_type, stage_id, depth, s, edit):
        for name, t, n in (("stage_type", stage_type, self.n_stage_types),
                           ("stage_id", stage_id, self.n_stage_ids),
                           ("depth", depth, self.n_depths)):
            if int(t.min()) < 0 or int(t.max()) >= n:
                die(f"{name} 索引 {int(t.min())}..{int(t.max())} 越出 0..{n-1}")
        h = (base + self.emb_type(stage_type) + self.emb_stage(stage_id)
             + self.emb_depth(depth) + self.s_proj(s.view(-1, 1)))
        h = F.silu(h)
        if edit is None:
            die("edit_condition 已开但没有传 edit 描述子（条件注入接线缺失）")
        hf = torch.cat([h, self.edit_enc(edit)], dim=-1)
        return h, hf


def _full_convs(in_ch: int, width: int, n_layers: int) -> nn.ModuleList:
    return nn.ModuleList([nn.Conv1d(int(in_ch) if i == 0 else int(width), int(width), 1)
                          for i in range(int(n_layers))])


def _heads(width: int):
    action = nn.Conv1d(int(width), 3, 1)
    clean = nn.Conv1d(int(width), 3, 1)
    nn.init.zeros_(action.weight)               # A1 的来源（stage_flow.py:195-196）
    nn.init.zeros_(action.bias)
    return action, clean


# --------------------------------------------------------------------------- #
# 臂 1：pxfilm —— 逐像素 FiLM（MetaDC film_gen）
# --------------------------------------------------------------------------- #
class PxFilmBackend(nn.Module):
    """γ_p,β_p = film_gen([hf ⊕ f1_p])，film_gen = Linear-SiLU-Linear（末层 zeros_），
    作用于第 2..L 层（首层特征 f1 = SiLU(conv1(feat)) 不调制，作为 film 输入）。

    MetaDC `model.py` L85-90 film_gen = Linear(cond,2H)-SiLU-Linear(2H, params)；
    L102-103 末层 weight/bias zeros_；L146-147 对 (B·H·W) 行逐像素算；
    L155-157 `h = layer(h); h = h*(1+g)+b; h = silu(h)`（conv 后、激活前，与 FULL 同位）。
    Linear 对逐像素行 == 1×1 conv 对 (B,C,P)，这里用 conv1d 形式，权重同一份。
    [hf ⊕ f1] 的第一层按块拆成 W1a·hf（逐样本）+ W1b·f1（逐像素）——数学同 cat 后一次 Linear，
    省去 (B, hf+W, P) 的拼接张量。第二层按层切片逐层算（同一权重矩阵的行块），数学同一次算全部。
    """

    def __init__(self, core: FullCondCore, in_ch: int, width: int, n_layers: int,
                 fg_hidden: int, layer_ckpt: bool):
        super().__init__()
        if n_layers < 2:
            die("pxfilm 需要 n_layers >= 2（首层不调制，后续各层逐像素 FiLM）")
        self.core = core
        self.convs = _full_convs(in_ch, width, n_layers)
        self.action, self.clean = _heads(width)
        self.width, self.n_layers, self.ckpt = int(width), int(n_layers), bool(layer_ckpt)
        self.fg_hidden = int(fg_hidden)
        self.film_gen = nn.Sequential(
            nn.Linear(core.hf_dim + self.width, self.fg_hidden), nn.SiLU(),
            nn.Linear(self.fg_hidden, 2 * self.width * (self.n_layers - 1)))
        nn.init.zeros_(self.film_gen[-1].weight)
        nn.init.zeros_(self.film_gen[-1].bias)

    def base(self, c, edits=None):
        return self.core.base(c).unsqueeze(1)

    def macs_per_pixel(self) -> int:
        w, L = self.width, self.n_layers
        base = self.convs[0].in_channels * w + (L - 1) * w * w + 2 * w * 3
        extra = w * self.fg_hidden + self.fg_hidden * 2 * w * (L - 1)
        return dict(base_trunk_and_heads=base, pxfilm_extra=extra, total=base + extra)

    def forward(self, base, feat, z, y, st, sid, depth, s, ed, m):
        h, hf = self.core(base[:, 0], st, sid, depth, s, ed)
        W1, b1 = self.film_gen[0].weight, self.film_gen[0].bias
        W2, b2 = self.film_gen[2].weight, self.film_gen[2].bias
        d_hf = hf.shape[1]
        f1 = F.silu(self.convs[0](feat))                              # (B,W,P)
        # u = SiLU(W1·[hf ⊕ f1] + b1) = SiLU(conv1d(f1, W1[:, hf:]) + (W1[:, :hf]·hf + b1))
        u = F.silu(F.conv1d(f1, W1[:, d_hf:].unsqueeze(-1))
                   + (hf @ W1[:, :d_hf].t() + b1).unsqueeze(-1))     # (B,fg,P)
        hcur = f1
        w = self.width
        for i in range(1, self.n_layers):
            rows = slice(2 * w * (i - 1), 2 * w * i)

            def layer(hh, uu, conv=self.convs[i], rows=rows):
                g = F.conv1d(uu, W2[rows].unsqueeze(-1), b2[rows])   # (B,2W,P)
                return F.silu(conv(hh) * (1.0 + g[:, :w]) + g[:, w:])
            hcur = _ckpt(layer, self.ckpt, hcur, u)
        return self.action(hcur), self.clean(hcur)

    def a12_probe(self):
        d = self.core.hidden
        return dict(name="film_gen[0].weight[:, hidden:hidden+edit_latent]",
                    param=self.film_gen[0].weight, cols=slice(d, d + self.core.edit_latent))

    def param_groups(self) -> dict:
        return dict(conditioner_core=_n(self.core) - _n(self.core.edit_enc),
                    edit_encoder=_n(self.core.edit_enc), trunk_convs=_n(self.convs),
                    film_gen=_n(self.film_gen), action_head=_n(self.action),
                    clean_head=_n(self.clean))


# --------------------------------------------------------------------------- #
# 臂 2：loramoe —— y = Wx + Σ_i α_i·(alpha/r)·B_i A_i x，α = Softmax(W_t·hf)
# --------------------------------------------------------------------------- #
class LoraMoeBackend(nn.Module):
    """FAPE-IR `add_lora_moe_to_linear`：lora_downs = Linear(in, r, bias=False) 逐专家
    kaiming_uniform_(a=√5)（L107）；lora_ups = Linear(r, out, bias=False) zeros_（L108）；
    lora_scales = alpha / rank（L36）；text_gate = Linear(D, E, bias=False) normal std 0.01（L112）；
    text_prior = softmax(text_gate(h))（L205-206，无温度）；
    `forward_with_lora_moe` L157/L238-242：original + Σ_i w_i · ups_i(downs_i(x)) · scale_i。
    这里每层 1×1 conv 即「Linear」，E 个专家的 downs/ups 各自拼成一次 conv（数学同逐专家求和）；
    路由输入 = FULL 的 FiLM 输入 hf = [h ⊕ e_m]（编辑 hidden state）。FiLM 去掉。
    """

    def __init__(self, core: FullCondCore, in_ch: int, width: int, n_layers: int,
                 n_experts: int, rank: int, lora_alpha: float, gate_std: float,
                 layer_ckpt: bool):
        super().__init__()
        self.core = core
        self.convs = _full_convs(in_ch, width, n_layers)
        self.action, self.clean = _heads(width)
        self.E, self.r = int(n_experts), int(rank)
        self.scale = float(lora_alpha) / float(rank)
        self.width, self.n_layers, self.ckpt = int(width), int(n_layers), bool(layer_ckpt)
        self.downs = nn.ModuleList()
        self.ups = nn.ModuleList()
        self.gates = nn.ModuleList()
        for i in range(self.n_layers):
            cin = int(in_ch) if i == 0 else int(width)
            dn = nn.Conv1d(cin, self.E * self.r, 1, bias=False)
            for e in range(self.E):                                   # 逐专家 kaiming_uniform_
                nn.init.kaiming_uniform_(dn.weight[e * self.r:(e + 1) * self.r], a=math.sqrt(5))
            up = nn.Conv1d(self.E * self.r, int(width), 1, bias=False)
            nn.init.zeros_(up.weight)                                 # B_i 零初始化（A1）
            g = nn.Linear(core.hf_dim, self.E, bias=False)
            nn.init.normal_(g.weight, std=float(gate_std))
            self.downs.append(dn)
            self.ups.append(up)
            self.gates.append(g)

    def base(self, c, edits=None):
        return self.core.base(c).unsqueeze(1)

    def forward(self, base, feat, z, y, st, sid, depth, s, ed, m):
        h, hf = self.core(base[:, 0], st, sid, depth, s, ed)
        hcur = feat
        for i in range(self.n_layers):
            alpha = torch.softmax(self.gates[i](hf), dim=-1)          # (B,E)
            wexp = alpha.repeat_interleave(self.r, dim=1) * self.scale  # (B,E·r)

            def layer(hh, we, conv=self.convs[i], dn=self.downs[i], up=self.ups[i]):
                a = dn(hh) * we.unsqueeze(-1)
                return F.silu(conv(hh) + up(a))
            hcur = _ckpt(layer, self.ckpt, hcur, wexp)
        return self.action(hcur), self.clean(hcur)

    def a12_probe(self):
        d = self.core.hidden
        return dict(name="gates[0].weight[:, hidden:hidden+edit_latent]",
                    param=self.gates[0].weight, cols=slice(d, d + self.core.edit_latent))

    def param_groups(self) -> dict:
        return dict(conditioner_core=_n(self.core) - _n(self.core.edit_enc),
                    edit_encoder=_n(self.core.edit_enc), trunk_convs=_n(self.convs),
                    lora_downs=_n(self.downs), lora_ups=_n(self.ups), gates=_n(self.gates),
                    action_head=_n(self.action), clean_head=_n(self.clean))


# --------------------------------------------------------------------------- #
# 臂 3：ditblk —— 逐像素 LightningDiTBlock（RFMSR）
# --------------------------------------------------------------------------- #
class RMSNorm(nn.Module):
    """RFMSR `rmsnorm.py` 逐字。"""

    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def _norm(self, x):
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)

    def forward(self, x):
        return self._norm(x.float()).type_as(x) * self.weight


class SwiGLUFFN(nn.Module):
    """RFMSR `swiglu_ffn.py` 逐字（去 torch.compile）。"""

    def __init__(self, in_features: int, hidden_features: int, bias: bool = True):
        super().__init__()
        self.w12 = nn.Linear(in_features, 2 * hidden_features, bias=bias)
        self.w3 = nn.Linear(hidden_features, in_features, bias=bias)

    def forward(self, x):
        x1, x2 = self.w12(x).chunk(2, dim=-1)
        return self.w3(F.silu(x1) * x2)


class TimestepEmbedder(nn.Module):
    """RFMSR `lightningdit.py` L171-201 逐字（cos 在前、sin 在后）。"""

    def __init__(self, hidden_size: int, frequency_embedding_size: int = 256):
        super().__init__()
        self.frequency_embedding_size = frequency_embedding_size
        self.mlp = nn.Sequential(nn.Linear(frequency_embedding_size, hidden_size, bias=True),
                                 nn.SiLU(), nn.Linear(hidden_size, hidden_size, bias=True))

    @staticmethod
    def timestep_embedding(t, dim, max_period=10000):
        half = dim // 2
        freqs = torch.exp(-math.log(max_period) * torch.arange(0, half, dtype=torch.float32)
                          / half).to(device=t.device)
        args = t[:, None].float() * freqs[None]
        emb = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
        if dim % 2:
            emb = torch.cat([emb, torch.zeros_like(emb[:, :1])], dim=-1)
        return emb

    def forward(self, t):
        return self.mlp(self.timestep_embedding(t, self.frequency_embedding_size))


class MultiHeadCrossAttention(nn.Module):
    """RFMSR L25-84（fused SDPA 路径；qk_norm = RMSNorm(head_dim)）。"""

    def __init__(self, d_model: int, num_heads: int, qk_norm: bool):
        super().__init__()
        if d_model % num_heads:
            die("d_model must be divisible by num_heads")
        self.num_heads, self.head_dim = int(num_heads), int(d_model) // int(num_heads)
        self.q_linear = nn.Linear(d_model, d_model)
        self.k_linear = nn.Linear(d_model, d_model)
        self.v_linear = nn.Linear(d_model, d_model)
        self.proj = nn.Linear(d_model, d_model)
        self.q_norm = RMSNorm(self.head_dim) if qk_norm else nn.Identity()
        self.k_norm = RMSNorm(self.head_dim) if qk_norm else nn.Identity()

    def forward(self, x, cond):
        B, N, C = x.shape
        Bc, Nc, _ = cond.shape
        q = self.q_linear(x).view(B, N, self.num_heads, self.head_dim).permute(0, 2, 1, 3)
        k = self.k_linear(cond).view(Bc, Nc, self.num_heads, self.head_dim).permute(0, 2, 1, 3)
        v = self.v_linear(cond).view(Bc, Nc, self.num_heads, self.head_dim).permute(0, 2, 1, 3)
        q, k = self.q_norm(q), self.k_norm(k)
        x = F.scaled_dot_product_attention(q, k, v, attn_mask=None, dropout_p=0.0)
        return self.proj(x.permute(0, 2, 1, 3).contiguous().view(B, N, C))


def modulate_adasin(x, shift, scale):
    """RFMSR L91-95：x·(1+scale)+shift，shift/scale 形 (B,1,C)。"""
    return x * (1 + scale) + shift


class DitBlock(nn.Module):
    """RFMSR `LightningDiTBlock` L207-274，Attention 槽位换成对条件 token 的 cross-attn
    （逐像素契约：无像素间自注意力），独立的无门 cross_attn 支不搬。
    x += gate_msa · CA(modulate(norm1(x))，tokens)；x += gate_mlp · SwiGLU(modulate(norm2(x)))。
    """

    def __init__(self, hidden: int, num_heads: int, mlp_ratio: float, qk_norm: bool):
        super().__init__()
        self.norm1 = RMSNorm(hidden)
        self.norm2 = RMSNorm(hidden)
        self.attn = MultiHeadCrossAttention(hidden, num_heads, qk_norm)
        self.mlp = SwiGLUFFN(hidden, int(2 / 3 * int(hidden * mlp_ratio)))
        self.scale_shift_table = nn.Parameter(torch.randn(6, hidden) / hidden ** 0.5)

    def forward(self, x, c, tokens):
        B = x.shape[0]
        shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = (
            self.scale_shift_table[None] + c.reshape(B, 6, -1)).chunk(6, dim=1)
        x = x + gate_msa * self.attn(modulate_adasin(self.norm1(x), shift_msa, scale_msa), tokens)
        x = x + gate_mlp * self.mlp(modulate_adasin(self.norm2(x), shift_mlp, scale_mlp))
        return x


class DitBlkBackend(nn.Module):
    """8 个逐像素 DiT 块，宽度 = FULL width。adaLN 只吃 τ(m)（t_embedder → t_block）；
    K/V = {W_c c, W_e e_j + τ(j)} 7 token → LayerNorm → mlp_ca（RFMSR L363-372 / L463-466）。
    初始化按 RFMSR `initialize_weights` L393-417（xavier / t 系 normal 0.02 / table randn/√H），
    **门零初始化**（任务卡；DiT adaLN-Zero）：t_block 末层与 table 的 gate 行置零。
    """

    def __init__(self, cond_in: int, in_ch: int, width: int, n_layers: int,
                 num_heads: int, mlp_ratio: float, qk_norm: bool, n_tokens_edit: int,
                 edit_in: int, edit_hidden: int, edit_layers: int, edit_latent: int,
                 encdim_ratio: int, t_std: float, layer_ckpt: bool):
        super().__init__()
        W = int(width)
        self.width, self.n_layers, self.ckpt = W, int(n_layers), bool(layer_ckpt)
        self.K = int(n_tokens_edit)
        self.edit_latent = int(edit_latent)
        es: list[nn.Module] = []
        ed = int(edit_in)
        for _ in range(int(edit_layers) - 1):
            es += [nn.Linear(ed, int(edit_hidden)), nn.SiLU()]
            ed = int(edit_hidden)
        es += [nn.Linear(ed, self.edit_latent)]
        self.edit_enc = nn.Sequential(*es)                    # 与 FULL 同形的编辑编码器
        self.tok_c = nn.Linear(int(cond_in), W)
        self.tok_e = nn.Linear(self.edit_latent, W)
        self.tok_norm = nn.LayerNorm(W)
        self.mlp_ca = nn.Sequential(nn.Linear(W, W * int(encdim_ratio)), nn.GELU(approximate="tanh"),
                                    nn.Linear(W * int(encdim_ratio), W))
        self.x_embedder = nn.Conv1d(int(in_ch), W, 1)
        self.t_embedder = TimestepEmbedder(W)
        self.t_block = nn.Sequential(nn.SiLU(), nn.Linear(W, 6 * W, bias=True))
        self.blocks = nn.ModuleList([DitBlock(W, num_heads, mlp_ratio, qk_norm)
                                     for _ in range(self.n_layers)])
        self.action, self.clean = _heads(W)
        # ---- RFMSR initialize_weights ----
        def _basic_init(mod):
            if isinstance(mod, nn.Linear):
                nn.init.xavier_uniform_(mod.weight)
                if mod.bias is not None:
                    nn.init.constant_(mod.bias, 0)
        for mod in (self.tok_c, self.tok_e, self.mlp_ca, self.blocks, self.t_block, self.edit_enc):
            mod.apply(_basic_init)
        wx = self.x_embedder.weight.data
        nn.init.xavier_uniform_(wx.view(wx.shape[0], -1))
        nn.init.constant_(self.x_embedder.bias, 0)
        nn.init.normal_(self.t_embedder.mlp[0].weight, std=t_std)
        nn.init.normal_(self.t_embedder.mlp[2].weight, std=t_std)
        nn.init.normal_(self.t_block[1].weight, std=t_std)
        # 门零初始化：chunk 顺序 (shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp)
        with torch.no_grad():
            for gi in (2, 5):
                self.t_block[1].weight[gi * W:(gi + 1) * W].zero_()
                self.t_block[1].bias[gi * W:(gi + 1) * W].zero_()
                for blk in self.blocks:
                    blk.scale_shift_table[gi].zero_()
        # 重新钉死 action 头零初始化（_basic_init 只碰 Linear，conv 头未动；这里再断言一次）
        nn.init.zeros_(self.action.weight)
        nn.init.zeros_(self.action.bias)

    def base(self, c, edits):
        """(B,1152) c 与 (B,K,D) 编辑描述子 → (B, 1+K, W) 条件 token（整链算一次）。"""
        if edits is None:
            die("ditblk 的 base() 需要整链编辑描述子（K/V token）")
        with torch.autocast("cuda", enabled=False):
            c = c.float()
            e = self.edit_enc(edits.float())                                  # (B,K,latent)
            j = torch.arange(1, self.K + 1, device=c.device, dtype=torch.float32)
            tau = self.t_embedder(j)                                           # (K,W)
            tok = torch.cat([self.tok_c(c).unsqueeze(1), self.tok_e(e) + tau[None]], dim=1)
            return self.mlp_ca(self.tok_norm(tok))

    def forward(self, base, feat, z, y, st, sid, depth, s, ed, m):
        c0 = self.t_block(self.t_embedder(m.float()))                          # (B,6W)
        x = self.x_embedder(feat).transpose(1, 2)                              # (B,P,W)
        for blk in self.blocks:
            x = _ckpt(blk, self.ckpt, x, c0, base)
        x = x.transpose(1, 2)
        return self.action(x), self.clean(x)

    def a12_probe(self):
        return dict(name="tok_e.weight (W_e: edit latent -> token)", param=self.tok_e.weight,
                    cols=None)

    def param_groups(self) -> dict:
        return dict(edit_encoder=_n(self.edit_enc), token_proj=_n(self.tok_c) + _n(self.tok_e)
                    + _n(self.tok_norm) + _n(self.mlp_ca), x_embedder=_n(self.x_embedder),
                    t_embedder=_n(self.t_embedder), t_block=_n(self.t_block),
                    dit_blocks=_n(self.blocks), action_head=_n(self.action),
                    clean_head=_n(self.clean))


# --------------------------------------------------------------------------- #
# 臂 4：affhead —— 3×4 仿射 + 0.1·tanh 残差双路头（MetaDC）
# --------------------------------------------------------------------------- #
class FullFilmTrunk(nn.Module):
    """FULL 的 FiLM 头 + 8 层 1×1 conv（stage_flow L137-140 / L183-205 逐字），供 affhead / ff 复用。"""

    def __init__(self, core: FullCondCore, in_ch: int, width: int, n_layers: int):
        super().__init__()
        self.core = core
        self.film = nn.Linear(core.hf_dim, 2 * int(width) * int(n_layers))
        nn.init.zeros_(self.film.weight)
        nn.init.zeros_(self.film.bias)
        self.convs = _full_convs(in_ch, width, n_layers)
        self.width, self.n_layers = int(width), int(n_layers)

    def forward(self, base0, feat, st, sid, depth, s, ed):
        h, hf = self.core(base0, st, sid, depth, s, ed)
        g = self.film(hf).view(-1, self.n_layers, 2, self.width)
        gamma, beta = 1.0 + g[:, :, 0], g[:, :, 1]
        x = feat
        for i, conv in enumerate(self.convs):
            x = F.silu(conv(x) * gamma[:, i].unsqueeze(-1) + beta[:, i].unsqueeze(-1))
        return x


class AffHeadBackend(nn.Module):
    """MetaDC `mlp_head = Linear(hidden, 15)`：weight zeros_，bias[0,4,8]=1（L104-111）；
    forward L166-179：M = out[:12]→(3,4)，out_matrix = M·[x;1]，final = out_matrix + tanh(detail)·0.1。
    这里 x = 当前状态 z，ĥ = final − z（初始 M=I、detail=0 ⇒ ĥ≡0，bit-exact）。clean 头同 FULL。
    """

    def __init__(self, core: FullCondCore, in_ch: int, width: int, n_layers: int):
        super().__init__()
        self.trunk = FullFilmTrunk(core, in_ch, width, n_layers)
        self.head = nn.Conv1d(int(width), 15, 1)
        nn.init.zeros_(self.head.weight)
        with torch.no_grad():
            self.head.bias.fill_(0)
            # 3×4 行主序 view(3,4)：恒等 = 索引 0/5/10（M[0,0],M[1,1],M[2,2]）。
            # MetaDC 原码 L108-110 写的是 bias[0,4,8]=1，而其 L171 是 view(...,3,4)，
            # 索引 4/8 落在 M[1,0]/M[2,0]（三通道都取 x_r），并非恒等；任务卡要求
            # 「M 初始化恒等」（A1），故按恒等取 0/5/10，差异记 NOTES D-bk3。
            self.head.bias[0] = 1.0
            self.head.bias[5] = 1.0
            self.head.bias[10] = 1.0
        self.clean = nn.Conv1d(int(width), 3, 1)

    def base(self, c, edits=None):
        return self.trunk.core.base(c).unsqueeze(1)

    def forward(self, base, feat, z, y, st, sid, depth, s, ed, m):
        x = self.trunk(base[:, 0], feat, st, sid, depth, s, ed)
        out = self.head(x)                                          # (B,15,P)
        B, _, P = out.shape
        M = out[:, :12].view(B, 3, 4, P)
        zt = z.permute(0, 2, 1)                                     # (B,3,P)
        zh = torch.cat([zt, torch.ones(B, 1, P, device=z.device, dtype=z.dtype)], dim=1)
        out_matrix = (M * zh.unsqueeze(1)).sum(dim=2)               # M·[z;1] (B,3,P)
        final = out_matrix + torch.tanh(out[:, 12:]) * 0.1
        return final - zt, self.clean(x)

    def a12_probe(self):
        d = self.trunk.core.hidden
        return dict(name="film.weight[:, hidden:hidden+edit_latent]", param=self.trunk.film.weight,
                    cols=slice(d, d + self.trunk.core.edit_latent))

    @property
    def core(self):
        return self.trunk.core

    def param_groups(self) -> dict:
        return dict(conditioner_core=_n(self.core) - _n(self.core.edit_enc),
                    edit_encoder=_n(self.core.edit_enc), film=_n(self.trunk.film),
                    trunk_convs=_n(self.trunk.convs), affine_head=_n(self.head),
                    clean_head=_n(self.clean))


# --------------------------------------------------------------------------- #
# 臂 5：clutflow —— 8 基 LUT 混合 + F_θ([z, y−z])（FlowLUT）
# --------------------------------------------------------------------------- #
class ClutFlowBackend(nn.Module):
    """FlowLUT 式 (4)(5)：w = Softmax(Linear_N(ReLU(Linear_1(F_global))))，F_global → hf；
    式 (6)：I_LUT = Σ_i w_i L_i(z)，N=8，D=33；§III-D F_θ = Conv(6→64)-ReLU-Conv(64→64)-ReLU-Conv(64→3)-Tanh
    （3×3 → 1×1，逐像素契约），输入 [z, y−z]（式 (8)(9) 的 [I, R]）；
    ĥ = Σ_i w_i L_i(z) − z + F_θ。
    L_i = identity + R_i（IA-3DLUT `Generator3DLUT_identity` + `Generator3DLUT_zero` 的和式），
    R_i 零初始化；恒等部分解析消去：Σ_i w_i L_i(z) − z = T(Σ_i w_i R_i, z)（三线性对 LUT 线性、Σw=1），
    使初始 ĥ 为**精确零**（三线性插值恒等 LUT 只到浮点舍入，达不到 bit-exact）。
    三线性 = `TrilinearInterpolation`（binsize=1/(D−1)、角点 8 邻插值）的 grid_sample(align_corners=True) 等价形。
    F_θ 末层零初始化（tanh(0)=0）；clean 头 = Conv1d(64→3) 接 F_θ 倒数第二层（FULL 的 clean 头同位）。
    """

    def __init__(self, core: FullCondCore, n_luts: int, lut_dim: int, w_hidden: int,
                 f_width: int):
        super().__init__()
        self.core = core
        self.N, self.D = int(n_luts), int(lut_dim)
        self.w_mlp = nn.Sequential(nn.Linear(core.hf_dim, int(w_hidden)), nn.ReLU(),
                                   nn.Linear(int(w_hidden), self.N))
        self.lut_res = nn.Parameter(torch.zeros(self.N, 3, self.D, self.D, self.D))
        fw = int(f_width)
        self.f1 = nn.Conv1d(6, fw, 1)
        self.f2 = nn.Conv1d(fw, fw, 1)
        self.f3 = nn.Conv1d(fw, 3, 1)
        nn.init.zeros_(self.f3.weight)
        nn.init.zeros_(self.f3.bias)
        self.clean = nn.Conv1d(fw, 3, 1)

    def base(self, c, edits=None):
        return self.core.base(c).unsqueeze(1)

    def forward(self, base, feat, z, y, st, sid, depth, s, ed, m):
        h, hf = self.core(base[:, 0], st, sid, depth, s, ed)
        w = torch.softmax(self.w_mlp(hf), dim=-1)                       # (B,N)
        rmix = torch.einsum("bn,ncijk->bcijk", w, self.lut_res)         # (B,3,D,D,D)
        grid = (2.0 * z - 1.0).view(z.shape[0], z.shape[1], 1, 1, 3)    # (B,P,1,1,3)，(r,g,b)
        lut_term = F.grid_sample(rmix, grid, mode="bilinear", padding_mode="border",
                                 align_corners=True).view(z.shape[0], 3, z.shape[1])
        x6 = torch.cat([z, y - z], dim=-1).permute(0, 2, 1)             # (B,6,P)
        u = F.relu(self.f1(x6))
        u = F.relu(self.f2(u))
        dflow = torch.tanh(self.f3(u))
        return lut_term + dflow, self.clean(u)

    def a12_probe(self):
        d = self.core.hidden
        return dict(name="w_mlp[0].weight[:, hidden:hidden+edit_latent]", param=self.w_mlp[0].weight,
                    cols=slice(d, d + self.core.edit_latent))

    def param_groups(self) -> dict:
        return dict(conditioner_core=_n(self.core) - _n(self.core.edit_enc),
                    edit_encoder=_n(self.core.edit_enc), w_mlp=_n(self.w_mlp),
                    lut_bank=self.lut_res.numel(), f_theta=_n(self.f1) + _n(self.f2) + _n(self.f3),
                    clean_head=_n(self.clean))


# --------------------------------------------------------------------------- #
# 臂 6：ff —— z、y 8 频正弦编码（MetaDC PositionalEncoding，include_input=True）
# --------------------------------------------------------------------------- #
class FFBackend(nn.Module):
    """MetaDC L26-50：freq_bands = 2**linspace(0, F−1, F)；out = [x] + [sin(x·f·π), cos(x·f·π)]_f。
    color_pe F=8：3 → 3+3·16 = 51 通道；z、y 各 51，其余 13 通道原样 ⇒ 首层 115 通道。
    """

    def __init__(self, core: FullCondCore, in_ch: int, width: int, n_layers: int,
                 n_freq: int):
        super().__init__()
        self.n_freq = int(n_freq)
        self.register_buffer("freq_bands", 2 ** torch.linspace(0, self.n_freq - 1, self.n_freq),
                             persistent=False)
        self.pe_ch = 3 + 3 * 2 * self.n_freq
        self.in_ch_raw = int(in_ch)
        self.in_ch = 2 * self.pe_ch + (int(in_ch) - 6)
        self.trunk = FullFilmTrunk(core, self.in_ch, width, n_layers)
        self.action, self.clean = _heads(width)

    def base(self, c, edits=None):
        return self.trunk.core.base(c).unsqueeze(1)

    def _pe(self, v):                                                # (B,3,P) -> (B,51,P)
        out = [v]
        for f in self.freq_bands:
            xf = v * (f * math.pi)
            out += [torch.sin(xf), torch.cos(xf)]
        return torch.cat(out, dim=1)

    def forward(self, base, feat, z, y, st, sid, depth, s, ed, m):
        f = torch.cat([self._pe(feat[:, :3]), self._pe(feat[:, 3:6]), feat[:, 6:]], dim=1)
        if f.shape[1] != self.in_ch:
            die(f"ff 首层通道 {f.shape[1]} != {self.in_ch}")
        x = self.trunk(base[:, 0], f, st, sid, depth, s, ed)
        return self.action(x), self.clean(x)

    def a12_probe(self):
        d = self.trunk.core.hidden
        return dict(name="film.weight[:, hidden:hidden+edit_latent]", param=self.trunk.film.weight,
                    cols=slice(d, d + self.trunk.core.edit_latent))

    @property
    def core(self):
        return self.trunk.core

    def param_groups(self) -> dict:
        return dict(conditioner_core=_n(self.core) - _n(self.core.edit_enc),
                    edit_encoder=_n(self.core.edit_enc), film=_n(self.trunk.film),
                    trunk_convs=_n(self.trunk.convs), action_head=_n(self.action),
                    clean_head=_n(self.clean))


# --------------------------------------------------------------------------- #
# 模型
# --------------------------------------------------------------------------- #
class BkSprfModel(SF.SprfModel):
    """FULL 的 SprfModel 外壳（features / 逆表 / 负控制 / stage_ids 原件），
    `cond` 与 `net` 换成本臂 backend；`cond.base(c, edits_full)`。"""

    def __init__(self, cond_in: int, cfg, n_steps: int, alpha_mode: str, depth_values):
        super().__init__(cond_in, cfg, n_steps, alpha_mode, depth_values)
        if self.action_backend != "mlp":
            die("BK 臂只定义在 flow.action_backend = mlp 上")
        if self.edit_condition != "inv_lut":
            die("BK 臂只定义在 edit.condition = inv_lut（oracle_lut）契约上")
        if self.lut_id_condition != "off":
            die("BK 臂不支持 lut_id_condition = embed")
        del self.cond
        del self.net
        arm = cfg.str_("bk", "arm", BK_ARMS)
        self.arm = arm
        width = cfg.int_("flow", "width")
        n_layers = cfg.int_("flow", "n_layers")
        hidden = cfg.int_("flow", "cond_hidden")
        cond_layers = cfg.int_("flow", "cond_layers")
        ec = self.edit
        layer_ckpt = cfg.bool_("bk", "layer_checkpoint")

        def core():
            return FullCondCore(cond_in, hidden, cond_layers, n_stage_types=int(n_steps),
                                n_stage_ids=int(n_steps), n_depths=int(n_steps) + 1,
                                edit_in=ec["descriptor_dim"], edit_hidden=ec["enc_hidden"],
                                edit_layers=ec["enc_layers"], edit_latent=ec["latent_dim"])
        if arm == "pxfilm":
            bk = PxFilmBackend(core(), self.in_ch, width, n_layers,
                               cfg.int_("bk", "film_gen_hidden"), layer_ckpt)
        elif arm == "loramoe":
            bk = LoraMoeBackend(core(), self.in_ch, width, n_layers,
                                cfg.int_("bk", "n_experts"), cfg.int_("bk", "rank"),
                                cfg.num("bk", "lora_alpha"), cfg.num("bk", "gate_init_std"),
                                layer_ckpt)
        elif arm == "ditblk":
            bk = DitBlkBackend(cond_in, self.in_ch, width, n_layers,
                               cfg.int_("bk", "num_heads"), cfg.num("bk", "mlp_ratio"),
                               cfg.bool_("bk", "qk_norm"), int(n_steps),
                               ec["descriptor_dim"], ec["enc_hidden"], ec["enc_layers"],
                               ec["latent_dim"], cfg.int_("bk", "encdim_ratio"),
                               cfg.num("bk", "t_init_std"), layer_ckpt)
        elif arm == "affhead":
            bk = AffHeadBackend(core(), self.in_ch, width, n_layers)
        elif arm == "clutflow":
            bk = ClutFlowBackend(core(), cfg.int_("bk", "n_luts"), cfg.int_("bk", "lut_dim"),
                                 cfg.int_("bk", "w_hidden"), cfg.int_("bk", "f_width"))
        elif arm == "ff":
            bk = FFBackend(core(), self.in_ch, width, n_layers, cfg.int_("bk", "n_freq"))
        else:
            die(f"未知 bk.arm {arm!r}")
        self.bk = bk
        self.cond = bk                     # 共享栈调用的是 model.cond.base(...)
        self.net = None

    # 供 A12b：编辑编码器
    @property
    def edit_encoder(self) -> nn.Module:
        return self.bk.edit_enc if self.arm == "ditblk" else self.bk.core.edit_enc

    def forward(self, base, z, y, alpha_hat, beta_state, alphas, lam, m, depth, s,
                lut_id=None, edit=None):
        if lut_id is not None:
            die("lut_id_condition = off 却传了 lut_id —— 推理端无 LUT 身份")
        st, sid = self.stage_ids(m)
        ed = self.edit_descriptor(edit)                       # (B,D) 当前阶段
        feat = self.features(z, y, alpha_hat, beta_state, alphas, lam, m)
        h, x0 = self.bk(base, feat, z, y, st, sid, depth, s, ed, m)
        return h.permute(0, 2, 1), x0.permute(0, 2, 1)

    def param_counts(self) -> dict:
        d = dict(arm=self.arm, total=_n(self), action_backend="mlp",
                 edit_condition=self.edit_condition, edit_contract=self.edit_contract,
                 edit_descriptor_dim=self.edit["descriptor_dim"],
                 edit_latent_dim=self.edit["latent_dim"], by_module=self.bk.param_groups())
        if hasattr(self.bk, "macs_per_pixel"):
            d["macs_per_pixel"] = self.bk.macs_per_pixel()
        return d
