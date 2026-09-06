#!/usr/bin/env python3
# 源自 experiments/prs/EPR-051_masked-restore-production/stage0/sprf/stage_flow.py @ 2026-09-06（原件未入 git；原 sha256 见 veraretouch_sprf/PROVENANCE.md），逐字复制/仅改导入与路径解析（EPR-052）。
"""EPR-051 · stage0 · arm=sprf · T2 第 2 层：StageConditioner + PointwiseActionNet。

模型图（PROPOSAL_sprf.md §2）：

  c(冻结 SigLIP2 缓存) ──► StageConditioner MLP ──► FiLM(γ, β) 逐层
        + stage_type learned embedding(6 类)
        + stage_id 嵌入 + depth 嵌入 + 标量 s
  逐像素输入 [z, y, β_obs, (β_state | α_1..α_K), λ]
        ──► PointwiseActionNet（n_layers 层 1×1 conv，w=width，SiLU，逐层 FiLM）
        ──► action 头 ĥ∈R³（线性，**末层零初始化**）
        └──► clean 头 x̂0∈R³（只做辅助监督/诊断）

条件三档（`alpha_mode`，逐档是递增信息的阶梯，β_obs = α̂ 恒在）：
  alpha_hat   [z(3), y(3), α̂(1), λ(1)]                       = 8 通道
  alpha_m     [z(3), y(3), α̂(1), β_state=α_m(1), λ(1)]       = 9 通道
  alpha_full  [z(3), y(3), α̂(1), α_1..α_K(K), λ(1)]          = 8 + K 通道

不用正弦时间编码（无连续全局 t）：时间/阶段信息全部走 embedding + λ 标量。
零初始化契约：action 头 weight/bias 全零 ⇒ 每阶段 ĥ≡0 ⇒ z 不变 ⇒ 整条 d 步链
的输出与 y 逐位相同（断言 1）。FiLM 头也零初始化 ⇒ 初始 γ≡1、β≡0。
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from veraretouch_sprf.data.stage_targets import ALPHA_MODES, die

BACKENDS = ("mlp", "g4d")

# X-COND：显式编辑信息注入的两个 oracle 档 + off。
EDIT_CONDITIONS = ("off", "inv_lut", "ref_pair")
EDIT_CONTRACTS = ("none", "oracle_lut", "oracle_ref")
# 描述子的通道口径（两臂**逐字相同**，只有来源不同 —— J-C3 要的就是这个差）：
#   inv_lut   通道 0..2 = L^{-1}(y_j) − y_j（GN 求逆），通道 3 = valid
#   ref_pair  通道 0..2 = 演示对在该格内的平均位移，通道 3 = 占用（归一化计数）
EDIT_CHANNELS = 4


def edit_spec(cfg) -> dict:
    """读 `[edit]` 节。**整节缺失 = off**（六个主臂的 TOML 里没有这一节，
    它们的 sha 已经跟着完成的 run 冻结了，不能回填）。

    只要这一节存在，每个键都按 A9 严格取值（缺键 die，代码里没有默认值兜底）。
    `descriptor_dim` 不是 TOML 键，是 grid³ × 通道数 的**导出量**。
    """
    if "edit" not in cfg.d:
        return dict(condition="off", contract="none", latent_dim=0, enc_hidden=0,
                    enc_layers=0, grid=0, descriptor_dim=0, section_present=False)
    cond = cfg.str_("edit", "condition", EDIT_CONDITIONS)
    contract = cfg.str_("edit", "contract", EDIT_CONTRACTS)
    want = {"off": "none", "inv_lut": "oracle_lut", "ref_pair": "oracle_ref"}[cond]
    if contract != want:
        die(f"[edit] condition = {cond!r} 的条件契约必须是 {want!r}，写的是 "
            f"{contract!r} —— 契约标注不能与实际注入的信息不符")
    g = cfg.int_("edit", "grid")
    if g < 2:
        die(f"[edit] grid = {g} 必须 >= 2")
    lat = cfg.int_("edit", "latent_dim")
    if cond == "off":
        if lat != 0:
            die(f"[edit] condition = off 时 latent_dim 必须为 0，写的是 {lat}")
        return dict(condition=cond, contract=contract, latent_dim=0, enc_hidden=0,
                    enc_layers=0, grid=g, descriptor_dim=0, section_present=True)
    if lat < 1:
        die(f"[edit] latent_dim = {lat} 必须 >= 1")
    return dict(condition=cond, contract=contract, latent_dim=lat,
                enc_hidden=cfg.int_("edit", "enc_hidden"),
                enc_layers=cfg.int_("edit", "enc_layers"),
                grid=g, descriptor_dim=g ** 3 * EDIT_CHANNELS,
                section_present=True)


def alpha_channels(alpha_mode: str, n_steps: int) -> int:
    """α 相关的逐像素通道数。

    `alpha_full` = 2K（N3 裁决的 19 维口径，d=6 时 3+3+6+6+1）：
    前 K 个是 β_obs = 完整 α_1..α_K（可观测的全 α 场），
    后 K 个是 β_state = 当前阶段的 one-hot × α（第 m 位放 α_m、其余 0），
    这样「有哪些阶段」与「现在走到第几阶段、它的 α 多大」分两组显式给出。
    """
    if alpha_mode not in ALPHA_MODES:
        die(f"alpha_mode {alpha_mode!r} 不在 {ALPHA_MODES}")
    return {"alpha_hat": 1, "alpha_m": 2, "alpha_full": 2 * int(n_steps)}[alpha_mode]


def pointwise_in_channels(alpha_mode: str, n_steps: int) -> int:
    return 3 + 3 + alpha_channels(alpha_mode, n_steps) + 1


class StageConditioner(nn.Module):
    """冻结条件 c + 阶段身份 -> 每层 FiLM 参数。

    `base(c)` 与阶段无关，可跨阶段缓存只算一次（π0 的条件缓存口径）；
    `film(...)` 每阶段一次，只是 embedding 相加 + 一个线性头。
    """

    def __init__(self, cond_in: int, hidden: int, cond_layers: int,
                 width: int, n_layers: int, n_stage_types: int, n_stage_ids: int,
                 n_depths: int, lut_id_vocab: int = 0, edit_in: int = 0,
                 edit_hidden: int = 0, edit_layers: int = 0, edit_latent: int = 0):
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
        # 仅 lut_id_condition="embed"（K5 诊断臂）时存在；0 号是 null/unknown token。
        self.emb_lut = (nn.Embedding(int(lut_id_vocab), hidden)
                        if int(lut_id_vocab) > 0 else None)
        self.s_proj = nn.Linear(1, hidden)
        # X-COND：显式编辑信息注入。小 MLP 把逐阶段的编辑描述子压到 latent，
        # 与 c 侧的 h **拼接**后一起进 FiLM 头（原 c 的 hidden 维一位不动，
        # 新增维度只走 FiLM）。末层普通 init；FiLM 头仍然零初始化，所以
        # 断言 A1（零初始化下整条链逐位恒等）不受影响。
        self.edit_latent = int(edit_latent)
        if self.edit_latent > 0:
            if int(edit_layers) < 2:
                die(f"edit_layers = {edit_layers} 必须 >= 2（2-3 层 MLP）")
            es: list[nn.Module] = []
            ed = int(edit_in)
            for _ in range(int(edit_layers) - 1):
                es += [nn.Linear(ed, int(edit_hidden)), nn.SiLU()]
                ed = int(edit_hidden)
            es += [nn.Linear(ed, self.edit_latent)]
            self.edit_enc = nn.Sequential(*es)
        else:
            self.edit_enc = None
        self.film = nn.Linear(hidden + self.edit_latent,
                              2 * int(width) * int(n_layers))
        nn.init.zeros_(self.film.weight)
        nn.init.zeros_(self.film.bias)          # 初始 gamma=1, beta=0
        self.width, self.n_layers = int(width), int(n_layers)
        self.n_stage_types = int(n_stage_types)
        self.n_stage_ids = int(n_stage_ids)
        self.n_depths = int(n_depths)

    def base(self, c: torch.Tensor) -> torch.Tensor:
        return self.trunk(c)

    def forward(self, base: torch.Tensor, stage_type: torch.Tensor,
                stage_id: torch.Tensor, depth: torch.Tensor, s: torch.Tensor,
                lut_id: torch.Tensor | None = None,
                edit: torch.Tensor | None = None
                ) -> tuple[torch.Tensor, torch.Tensor]:
        for name, t, n in (("stage_type", stage_type, self.n_stage_types),
                           ("stage_id", stage_id, self.n_stage_ids),
                           ("depth", depth, self.n_depths)):
            if int(t.min()) < 0 or int(t.max()) >= n:
                die(f"{name} 索引 {int(t.min())}..{int(t.max())} 越出 0..{n-1}")
        h = (base + self.emb_type(stage_type) + self.emb_stage(stage_id)
             + self.emb_depth(depth) + self.s_proj(s.view(-1, 1)))
        if self.emb_lut is not None:
            if lut_id is None:
                die("lut_id_condition 已开但没有传 lut_id（诊断臂接线缺失）")
            h = h + self.emb_lut(lut_id)
        elif lut_id is not None:
            die("lut_id_condition = off 却传了 lut_id —— 推理端无 LUT 身份")
        h = F.silu(h)
        # X-COND：编辑 latent 只在 FiLM 头入口拼接，不改 h 本身（B-g4d 的 θ 头
        # 吃的仍然是原来的 hidden 维 h，接口不动）。
        if self.edit_enc is not None:
            if edit is None:
                die("edit_condition 已开但没有传 edit 描述子（条件注入接线缺失）")
            hf = torch.cat([h, self.edit_enc(edit)], dim=-1)
        else:
            if edit is not None:
                die("edit_condition = off 却传了 edit 描述子 —— 条件契约不符")
            hf = h
        g = self.film(hf).view(-1, self.n_layers, 2, self.width)
        # 第三个返回值是阶段条件的隐状态：mlp 后端用不到，B-g4d 的 θ 头吃它。
        return 1.0 + g[:, :, 0], g[:, :, 1], h


class PointwiseActionNet(nn.Module):
    """n_layers 层 1×1 conv（w=width，SiLU，逐层 FiLM）+ action 头 + clean 头。"""

    def __init__(self, in_ch: int, width: int, n_layers: int):
        super().__init__()
        if not 1 <= int(n_layers):
            die(f"n_layers = {n_layers} 必须 >= 1")
        self.convs = nn.ModuleList(
            [nn.Conv1d(int(in_ch) if i == 0 else int(width), int(width), 1)
             for i in range(int(n_layers))])
        self.action = nn.Conv1d(int(width), 3, 1)
        self.clean = nn.Conv1d(int(width), 3, 1)
        nn.init.zeros_(self.action.weight)      # 断言 1 的来源
        nn.init.zeros_(self.action.bias)
        self.width, self.n_layers = int(width), int(n_layers)

    def forward(self, feat: torch.Tensor, gamma: torch.Tensor,
                beta: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """`feat` (B,C,P)，`gamma`/`beta` (B,n_layers,W) -> 两个 (B,3,P)。"""
        h = feat
        for i, conv in enumerate(self.convs):
            h = F.silu(conv(h) * gamma[:, i].unsqueeze(-1) + beta[:, i].unsqueeze(-1))
        return self.action(h), self.clean(h)


class SprfModel(nn.Module):
    """StageConditioner + PointwiseActionNet + 两头；逐像素特征拼装也在这里。

    `alpha_contract = oracle`：网络看到什么 α 由 `alpha_mode` 决定，
    但更新式里用的 β_m 永远是 journal 重建 × 标定 s 的真实 β 场。
    """

    def __init__(self, cond_in: int, cfg, n_steps: int, alpha_mode: str,
                 depth_values: list[int]):
        super().__init__()
        width = cfg.int_("flow", "width")
        n_layers = cfg.int_("flow", "n_layers")
        hidden = cfg.int_("flow", "cond_hidden")
        cond_layers = cfg.int_("flow", "cond_layers")
        lut_cond = cfg.str_("flow", "lut_id_condition", ("off", "embed"))
        lut_vocab = cfg.int_("flow", "lut_id_vocab")
        # 预注册的层数集合放在 TOML 里（提案容量节 v2：主臂 w512×8，w256×4–6 是
        # 容量下界消融行），代码里不写死数字。
        allowed = [int(v) for v in cfg.list_("flow", "n_layers_allowed", int)]
        if n_layers not in allowed:
            die(f"[flow] n_layers = {n_layers} 不在预注册集合 {allowed}")
        if alpha_mode not in ALPHA_MODES:
            die(f"[flow] alpha_mode {alpha_mode!r} 不在 {ALPHA_MODES}")
        self.alpha_mode = alpha_mode
        self.lut_id_condition = lut_cond
        self.lut_id_vocab = int(lut_vocab)
        self.n_steps = int(n_steps)
        self.in_ch = pointwise_in_channels(alpha_mode, n_steps)
        ec = edit_spec(cfg)
        self.edit_condition = ec["condition"]
        self.edit_contract = ec["contract"]
        self.cond = StageConditioner(cond_in, hidden, cond_layers, width, n_layers,
                                     n_stage_types=int(n_steps),
                                     n_stage_ids=int(n_steps),
                                     n_depths=int(n_steps) + 1,
                                     lut_id_vocab=(self.lut_id_vocab
                                                   if lut_cond == "embed" else 0),
                                     edit_in=ec["descriptor_dim"],
                                     edit_hidden=ec["enc_hidden"],
                                     edit_layers=ec["enc_layers"],
                                     edit_latent=ec["latent_dim"])
        self.net = PointwiseActionNet(self.in_ch, width, n_layers)
        self.edit = ec
        if ec["condition"] == "inv_lut":
            # 逆表按 lut_id 一次性预计算（L_m 是池属性，与图像/β/s 无关），
            # 训练时按 journal 的 lut_id 查表。**非持久 buffer**：不进 ckpt
            # （304 MiB × 每次存盘），指纹校验由训练入口在启动时做。
            self.register_buffer("inv_table", torch.zeros(0), persistent=False)
        # T5 B-g4d：只换 action 的表示，SPRF 的结构/损失/判据一概不动。
        self.action_backend = cfg.str_("flow", "action_backend", BACKENDS)
        if self.action_backend == "g4d":
            from veraretouch_sprf.models import stage_backend_g4d as BG
            g4d_cfg = BG.G4DConfig(mode=cfg.str_("g4d", "mode"),
                                   n_gauss=cfg.int_("g4d", "n_gauss"),
                                   cond_dim=hidden, hidden=cfg.int_("g4d", "hidden"),
                                   init_seed=cfg.int_("g4d", "init_seed"))
            self.g4d_head = BG.G4DActionHead(hidden, g4d_cfg,
                                             cfg.int_("g4d", "point_chunk"))
        else:
            self.g4d_head = None
        self.depth_values = sorted(int(d) for d in depth_values)

    # -- 逐像素特征 -------------------------------------------------------- #
    def features(self, z: torch.Tensor, y: torch.Tensor, alpha_hat: torch.Tensor,
                 beta_state: torch.Tensor, alphas: torch.Tensor,
                 lam: torch.Tensor, m: torch.Tensor | None = None) -> torch.Tensor:
        """全部形状 (B,P,·) -> (B,C,P)。λ 是逐样本标量，广播成逐像素通道。"""
        p = z.shape[1]
        if self.alpha_mode == "alpha_full":
            # β_obs = 完整 α 场 (B,K,P)；β_state = one-hot(m) × α (B,K,P)
            if m is None:
                die("alpha_full 需要当前阶段 m 才能构造 β_state 的 one-hot")
            k = alphas.shape[1]
            oh = torch.zeros_like(alphas)
            oh.scatter_(1, (m - 1).view(-1, 1, 1).expand(-1, 1, alphas.shape[2]),
                        torch.gather(alphas, 1,
                                     (m - 1).view(-1, 1, 1)
                                     .expand(-1, 1, alphas.shape[2])))
            parts = [z, y, alphas.permute(0, 2, 1), oh.permute(0, 2, 1)]
        else:
            parts = [z, y, alpha_hat.unsqueeze(-1)]
            if self.alpha_mode == "alpha_m":
                parts.append(beta_state.unsqueeze(-1))
        parts.append(lam.view(-1, 1, 1).expand(-1, p, 1).to(z.dtype))
        feat = torch.cat(parts, dim=-1)
        if feat.shape[-1] != self.in_ch:
            die(f"逐像素输入通道 {feat.shape[-1]} != 预期 {self.in_ch}")
        return feat.permute(0, 2, 1).contiguous()

    # -- X-COND：编辑描述子 -------------------------------------------------- #
    def load_inv_table(self, table: torch.Tensor) -> None:
        """装载 `(N_lut, grid³, 4)` 的逆表（`precompute_inv_lut.py` 的产物）。"""
        if self.edit_condition != "inv_lut":
            die(f"edit.condition = {self.edit_condition!r} 却在装 LUT 逆表")
        want = self.edit["descriptor_dim"]
        flat = table.reshape(table.shape[0], -1)
        if flat.shape[1] != want:
            die(f"逆表描述子维度 {flat.shape[1]} != [edit] grid³×通道 {want}")
        # 末尾追加一行**全零**的 null 描述子：Δ_edit_null 负控制用它，
        # 这样 inv_lut 档的编辑来源始终是 long 行号（类型守卫不被绕开）。
        z = torch.zeros((1, flat.shape[1]), dtype=flat.dtype)
        self.edit_null_row = int(flat.shape[0])
        self.inv_table = torch.cat([flat, z], dim=0).to(
            next(self.parameters()).device, torch.float32)

    def edit_descriptor(self, src: torch.Tensor | None) -> torch.Tensor | None:
        """逐阶段的编辑来源 -> 逐阶段描述子 (B, descriptor_dim)。

        `inv_lut`   `src` 是 (B,) long 的 bank 行号 -> 查预计算逆表（**不逐图求逆**：
                    L_m 是池属性，与图像 / β / s 无关）。
        `ref_pair`  `src` 已经是 (B, descriptor_dim) 的 float 描述子（DataLoader
                    在 worker 里从演示对现算），原样返回。
        """
        if self.edit_condition == "off":
            if src is not None:
                die("edit.condition = off 却收到编辑来源")
            return None
        if src is None:
            die(f"edit.condition = {self.edit_condition!r} 但没收到编辑来源")
        if self.edit_condition == "inv_lut":
            if src.dtype != torch.long:
                die(f"inv_lut 的编辑来源必须是 long 行号，收到 {src.dtype}")
            if int(self.inv_table.numel()) == 0:
                die("inv_lut 逆表未装载（load_inv_table 没被调用）")
            n = int(self.inv_table.shape[0])
            if int(src.min()) < 0 or int(src.max()) >= n:
                die(f"lut 行号 {int(src.min())}..{int(src.max())} 越出 0..{n - 1}")
            return self.inv_table[src]
        if src.dtype == torch.long:
            die("ref_pair 的编辑来源必须是 float 描述子，收到 long")
        if src.shape[-1] != self.edit["descriptor_dim"]:
            die(f"ref_pair 描述子维度 {src.shape[-1]} != "
                f"{self.edit['descriptor_dim']}")
        return src.to(next(self.parameters()).dtype)

    def null_edit(self, src: torch.Tensor | None) -> torch.Tensor | None:
        """Δ_edit_null 负控制：与 `src` 同形的「零编辑」来源。

        `inv_lut` 指向逆表末尾那行全零（类型仍是 long）；`ref_pair` 直接给零
        描述子。两档的 null 在**描述子空间**里是同一个东西（全零），所以
        两臂的 Δ_edit_null 可以并排读。
        """
        if src is None:
            return None
        if src.dtype == torch.long:
            return torch.full_like(src, int(self.edit_null_row))
        return torch.zeros_like(src)

    def roll_edit(self, src: torch.Tensor | None) -> torch.Tensor | None:
        """Δ_edit_roll 负控制：沿**阶段轴**循环移一位。

        样本还是那个样本、编辑集合还是那一组、编码器容量一位不差 —— 只有
        「哪个阶段配哪个编辑」错了。它与 Δ_edit_null 分工：null 问「用没用编辑
        信息」，roll 问「用没用**对**的那条编辑」。
        """
        if src is None:
            return None
        return torch.roll(src, shifts=1, dims=1)

    def stage_ids(self, m: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """m ∈ 1..K -> (stage_type, stage_id) 索引，均为 m-1。

        本数据律下每条链的 `STEP_KIND` 顺序固定（`rebuild_forward` 已断言），
        所以 stage_type 与 stage_id 在数值上重合；两个 embedding 按提案分别保留。
        """
        return m - 1, m - 1

    def forward(self, base: torch.Tensor, z: torch.Tensor, y: torch.Tensor,
                alpha_hat: torch.Tensor, beta_state: torch.Tensor,
                alphas: torch.Tensor, lam: torch.Tensor, m: torch.Tensor,
                depth: torch.Tensor, s: torch.Tensor,
                lut_id: torch.Tensor | None = None,
                edit: torch.Tensor | None = None):
        st, sid = self.stage_ids(m)
        gamma, beta, cond_h = self.cond(base, st, sid, depth, s, lut_id,
                                        self.edit_descriptor(edit))
        h, x0 = self.net(self.features(z, y, alpha_hat, beta_state, alphas, lam, m),
                         gamma, beta)
        x0 = x0.permute(0, 2, 1)
        if self.g4d_head is not None:
            # action 走 G4D 载体；clean 头仍来自共享干（辅助监督口径不变）。
            h_g4d, _ = self.g4d_head(cond_h, z, s)
            return h_g4d, x0
        return h.permute(0, 2, 1), x0

    def param_counts(self) -> dict:
        def n(mod):
            return sum(p.numel() for p in mod.parameters())
        d = dict(conditioner=n(self.cond), action_net=n(self.net),
                 action_head=n(self.net.action), clean_head=n(self.net.clean),
                 total=n(self), action_backend=self.action_backend,
                 edit_condition=self.edit_condition,
                 edit_contract=self.edit_contract,
                 edit_encoder=(n(self.cond.edit_enc)
                               if self.cond.edit_enc is not None else 0),
                 edit_descriptor_dim=self.edit["descriptor_dim"],
                 edit_latent_dim=self.edit["latent_dim"])
        if self.g4d_head is not None:
            d["g4d_theta_head"] = n(self.g4d_head.theta)
            d["g4d_carrier"] = n(self.g4d_head.carrier)
            d["g4d_n_theta"] = int(self.g4d_head.n_theta)
        return d
