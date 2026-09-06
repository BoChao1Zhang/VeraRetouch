#!/usr/bin/env python3
# 源自 experiments/prs/EPR-051_masked-restore-production/stage0/sprf/train_sprf_bk_core4.py @ 2026-09-06（原件未入 git；原 sha256 见 veraretouch_sprf/PROVENANCE.md），逐字复制/仅改导入与路径解析（EPR-052）。
"""EPR-051 · stage0 · arm=sprf · T2 训练/推理入口。

四要素在 `PROPOSAL_sprf.md`；本文件只实现那四样。

  数据    与 train_stage0 完全同一条：`snapshot_newdata_v2` 的 20 个分片，
          per-(id, depth) clean，`winner_confidence=low` 不进，split 两权威
          （frozen_table 优先，其余 sha1 bucket），held-out 由
          `sha1("<salt>:"+source_id)%100 >= holdout_bucket_min` 划出。
          载入/切分/守卫全部 **import train_stage0 的原件**，不重写。
  模型    冻结 SigLIP2 条件 c（复用现缓存）-> StageConditioner -> FiLM
          -> PointwiseActionNet -> action 头（零初始化）+ clean 头；
          逆序每阶段一次 Euler 更新 z ← where(β_m==0, z, z+β_m·ĥ)。
  损失    L_core = L_prev + 0.25·L_action + 0.1·L_clean + 0.01·L_bound；
          预算后段 +0.5·L_rollout（batch 比例 25%→50% curriculum）；
          最后 30% +0.05·L_q50（Q45–Q55 分位带内像素均值）。
          权重为 0 的项还在算 -> die。
  优化器  AdamW lr 3e-4 / betas(0.9,0.999) / wd 0.01 / cosine warmup auto /
          grad_clip 1.0 / batch 64 / bf16 只包 conditioner trunk。

预注册运行时断言（启动或触发即 die），逐条见 `SMOKE.md`：
  A1 step-0 恒等（跨越整条 d 步链，torch.equal）
  A2 teacher 抽查（每 N 个 batch 一个样本，独立重算，折线 G-A1 / 一步 G-A2）
  A3 config 自动 diff（臂间只允许 4 个预注册键不同）
  A4 MetricLedger.assert_wired（新列调用数 == 被评样本数）
  A5 NFE 计数 == depth × nfe_per_stage
  A6 no-LUT 生产 restore() 接口（删禁入键后输出逐位一致）
  A7 held-out 隔离（目标构造逐 batch 断言）
  A8 sha256 冻结（第一步前进 run_args.json）
  A9 实验语义数字全在 TOML（Cfg 严格取值，缺键 die）
  A11 eval_at 死键守卫：该键未接线，非空即 die（「定义了没接线」红线）
  A10 锚点契约：rollout/推理起点 = uint8 资产 y，teacher 采样 = fp32 链态 r^m，
      两者不混用，差值必须 <= [flow] anchor_quant_gap_max（D-quant 裁决）


----------------------------------------------------------------------
LOSSABL 分支文件 core2（EPR-051 任务卡 3b · 损失归因消融）
  来源: train_sprf.py（被在跑的 SPRF_CLUTFULL_S1 sha 冻结，一行不改；E1 教训）
  差异: 仅核心四项 prev/action/clean/bound 的 book.add 由无条件改为按 `core` 开门，
        使 λ=0 的项真跳过计算而不是撞断言停机。数学在 λ!=0 时逐位不变。
  入口: train_sprf_lossabl2.py（装 archive/subset 猴补丁后转调本文件 main()）
  与 core（v1）的差异: 仅 A12a/A12b 判读时机由 step1/2 后移到 step2/3
        （裁决 2026-09-02(a)）。core v1 保持字节冻结, FULL 臂在用, 不得改动。
----------------------------------------------------------------------
BK 分支文件 core（EPR-051 任务卡 BK-ABL-v2 · backend 消融）
  core4（叠加臂）：与 train_sprf_bk_core.py 的差异 = import stage_flow_bk4 + `--skip-final-eval`（ckpt_last 落盘后写 metrics_train.json 并退出，final 由 batch_eval_bk.py 出）
  来源: train_sprf_lossabl_core2.py（被已完成/在跑的 LOSSABL 臂 sha 冻结，一字不改；E1）
  差异（BK 分支：这是本文件与 core2 的全部差异）:
    1. import stage_flow_bk / stage_solver_bk（分支模型与分支求解器）；模型类 BkSprfModel
    2. model.cond.base(feat, edits_desc)：整链编辑描述子进 base（ditblk 的 K/V token）；
       Δ_edit_null / Δ_edit_roll 控制列按被替换的编辑**重算 base**
    3. A12a 判读对象改为各臂「编辑 latent 进入 backend 的首个可学习投影」（model.bk.a12_probe），
       A12b 仍为编辑编码器；时机 step 2 / step 3（与 core2 同）
    4. 日志加 ms_per_step / gpu_peak_reserved_gb；metrics 加 params.by_module 与 train_seconds
    5. frozen_sha256 加 stage_flow_bk / stage_solver_bk（stage_flow / stage_solver 仍记原件）
  损失、课程、优化器、评测列、断言 A1–A11 逐字不变（损失全开，λ 全非零由入口断言）。
----------------------------------------------------------------------
"""
from __future__ import annotations

import argparse
import copy
import hashlib
import json
import math
import os
import random
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset

HERE = Path(__file__).resolve().parent
from veraretouch_sprf import _paths as _P  # EPR-052：替代原 sys.path 注入块
_P.ensure_sys_path()
STAGE0 = _P.STAGE0
REPO = _P.REPO

from veraretouch_sprf.data import train_stage0 as T0                       # noqa: E402  数据/切分/评测口径原件
import epr050_build_degradation as B            # noqa: E402
from veraretouch_sprf.data import stage_targets as ST                      # noqa: E402
from veraretouch_sprf.models import stage_flow_bk4 as SF                     # noqa: E402  BK 分支模型（叠加臂 adagn_ff / adagn_ff_affhead）
from veraretouch_sprf.solver import stage_solver_bk as SS                    # noqa: E402  BK 分支求解器
from veraretouch_sprf.models import edit_cond as EC                           # noqa: E402  X-COND 条件注入
from q3vl.whatb import lutdata as LUTDATA       # noqa: E402

LEVEL = 255.0
SELECT_METRICS = ("heldout_model_linf8_p50",)
# 臂间唯一允许不同的实验语义键（A3）
ARM_KEYS = ("flow.path_mode", "flow.endpoint_probability", "flow.alpha_mode",
            "solver.nfe_per_stage",
            # X-COND：条件注入节整节进「允许键集」。六个主臂的 TOML 里没有 [edit]
            # 这一节（它们的 sha 已随完成的 run 冻结，不回填），所以组内两两 diff
            # 里根本不会出现这些键；只有 X-COND 的两臂之间会。
            "edit.condition", "edit.contract", "edit.latent_dim",
            "edit.enc_hidden", "edit.enc_layers", "edit.grid",
            "edit.inv_cache_dir", "edit.ref_pool_n", "edit.ref_size",
            "edit.ref_salt", "edit.ref_hist_bins")
# 带 p50/p95/p99 的误差列（其余列只有一个标量 value）
ERR_COLUMNS = ("model", "identity", "exact_solve", "model_midpoint",
               "active_only_headline", "stage_all", "stage_active", "stage_beta1",
               "cycle", "de00", "de00_identity")
# 只有在对应开关打开时才存在的列（首版都关；缺省并在 metrics 里注明，不出假数）
OPTIONAL_COLUMNS = ("true_lut_condition_delta", "predicted_lut_accuracy")


def die(msg: str):
    raise SystemExit(f"train_sprf: {msg}")


class SprfLedger(T0.MetricLedger):
    """`MetricLedger` + 「结构性稀疏层」白名单。

    基类有一条规则：某列在**每个**被评样本上都 missing 就停机（「该列对这份数据
    不可用，不要报空」）。`stage_beta1` 不属于那种情况——它被逐样本算了（调用数
    照样 == 被评样本数），只是 β = 场 × s，s≠1 的链上**没有** β==1 的像素
    （mix_alpha 的 `alpha==1` 分支不触发），所以这一层在那些链上天然为空。

    白名单只放宽「全体 missing」这一条，调用计数与单一来源两条守卫不变；
    放行的列名必须显式写在 `[eval] sparse_ok_columns` 里，并在输出里带上
    `n_available`，稀疏本身可见，不被遮掩。
    """

    def __init__(self, columns, strata, sparse_ok):
        super().__init__(columns, strata)
        unknown = [c for c in sparse_ok if c not in columns]
        if unknown:
            die(f"[eval] sparse_ok_columns 里的 {unknown} 不在 columns 中")
        self.sparse_ok = set(sparse_ok)

    def assert_wired(self, out: dict) -> None:
        stash = {}
        for c in self.sparse_ok:
            if self.missing.get(c) == self.expected:
                stash[c] = self.missing[c]
                self.missing[c] = 0
        try:
            super().assert_wired(out)
        finally:
            self.missing.update(stash)
        out["sparse_ok_columns"] = sorted(self.sparse_ok)
        out["column_available"] = {c: self.expected - n
                                   for c, n in self.missing.items()}


# --------------------------------------------------------------------------- #
# A3 config 自动 diff
# --------------------------------------------------------------------------- #
def flatten_toml(d: dict, prefix: str = "") -> dict:
    out = {}
    for k, v in d.items():
        key = f"{prefix}{k}"
        if isinstance(v, dict):
            out.update(flatten_toml(v, key + "."))
        else:
            out[key] = v
    return out


def assert_arm_diff(cfg: T0.Cfg, peers: list[str], identity_keys: list[str]) -> dict:
    """两臂 config 除 ARM_KEYS 外逐键相同（run 身份键除外，显式列在 TOML 里）。"""
    import tomllib
    mine = flatten_toml(cfg.d)
    report = []
    for p in peers:
        pp = Path(p)
        if not pp.is_absolute():                  # 相对路径按 config 自己的目录解析
            pp = Path(cfg.path).resolve().parent / p
        if not pp.is_file():
            die(f"[arms] peer_configs 里的 {pp} 不存在")
        theirs = flatten_toml(tomllib.loads(pp.read_text()))
        keys = set(mine) | set(theirs)
        diff = sorted(k for k in keys if mine.get(k, KeyError) != theirs.get(k, KeyError))
        unexpected = [k for k in diff
                      if k not in ARM_KEYS and k not in set(identity_keys)]
        if unexpected:
            die(f"config 自动 diff 失败：与 {pp.name} 除预注册键 {list(ARM_KEYS)} "
                f"（+ 身份键 {list(identity_keys)}）外还差 {unexpected}")
        report.append(dict(peer=str(pp), differing_keys=diff))
    return dict(arm_keys=list(ARM_KEYS), identity_keys=list(identity_keys),
                peers=report)


# --------------------------------------------------------------------------- #
# 数据集：一条 = 一对 prefix 的 n_pix 个像素 + 中间态 + h* 目标
# --------------------------------------------------------------------------- #
class SprfPixels(Dataset):
    """`PrefixPixels` 的 SPRF 版：多返回 r^(0..K) 与逐阶段 h*。

    LUT 求值与 `mix_alpha` 都是逐像素算子，所以在 `n_pix` 个采样像素上走链
    与「全图走链再取这些像素」逐位相同（`check_rebuild_equiv.py` 断言）。
    """

    def __init__(self, samples, blobs, n_pix: int, n_steps: int, bank_dir: str,
                 lut_cache: int, lut_vocab: int, edit_src=None):
        # `edit_src` = X-COND 的逐阶段编辑来源提供者（None 即 edit.condition=off）。
        self.edit_src = edit_src
        self.samples = samples
        self.blobs = blobs
        self.n_pix = int(n_pix)
        self.n_steps = int(n_steps)
        self.bank_dir = bank_dir
        self.lut_cache = int(lut_cache)
        self.lut_vocab = int(lut_vocab)
        self._bank = None

    @property
    def bank(self) -> ST.LutVolumes:
        if self._bank is None:                    # 每 worker 进程一份
            self._bank = ST.LutVolumes(self.bank_dir, self.lut_cache)
        return self._bank

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, i: int):
        s = self.samples[i]
        row = json.loads(self.blobs[s["id"]])
        x0, y = T0.load_pair(s["shard"], row, s["after_asset"])
        a = T0.alpha_fields(row, x0)                       # (K,H,W)，已乘 s
        npx = x0.shape[0] * x0.shape[1]
        idx = torch.randint(0, npx, (self.n_pix,))
        mask = T0.depth_mask(s["depth"], self.n_steps)     # (K,)
        alphas = a.reshape(self.n_steps, -1)[:, idx] * mask.unsqueeze(-1)
        x_px = x0.reshape(-1, 3)[idx]
        r, hstar = ST.build_intermediates(x_px, alphas, row["luts"], self.bank)
        depth = self.n_steps if s["depth"] is None else int(s["depth"])
        return dict(x=x_px, y=y.reshape(-1, 3)[idx], alphas=alphas,
                    r=r, hstar=hstar, mask=mask,
                    depth=torch.tensor(depth, dtype=torch.long),
                    s=torch.tensor(float(row["calib"]["s"])),
                    lut_ids=torch.tensor(
                        [ST.lut_id_token(n, self.lut_vocab) for n in row["luts"]],
                        dtype=torch.long),
                    grids=torch.tensor([int(g) for g in row["grids"]],
                                       dtype=torch.long),
                    idx=idx, key=T0.feature_key(s), sid=s["id"], index=i,
                    **({} if self.edit_src is None
                       else dict(edit=self.edit_src.for_row(row, x0, mask))))


def collate(batch: list[dict]) -> dict:
    keys = ("x", "y", "alphas", "r", "hstar", "mask", "depth", "s", "idx",
            "lut_ids", "grids")
    if "edit" in batch[0]:
        keys = keys + ("edit",)
    out = {k: torch.stack([b[k] for b in batch]) for k in keys}
    out["key"] = [b["key"] for b in batch]
    out["sid"] = [b["sid"] for b in batch]
    out["index"] = torch.tensor([b["index"] for b in batch])
    return out


# --------------------------------------------------------------------------- #
# 损失开关板：权重为 0 的项还在算 -> die
# --------------------------------------------------------------------------- #
class LossBook:
    def __init__(self, weights: dict[str, float]):
        self.w = dict(weights)
        self.reset(set())

    def reset(self, active: set[str]) -> None:
        self.active = set(active)
        self.computed: set[str] = set()
        self.parts: dict[str, float] = {}

    def add(self, name: str, value: torch.Tensor) -> torch.Tensor:
        if name not in self.w:
            die(f"损失项 {name!r} 未在 [loss] 预注册")
        if self.w[name] == 0.0:
            die(f"λ==0 的损失项 {name!r} 还在被计算 —— 预注册禁止")
        if name not in self.active:
            die(f"损失项 {name!r} 不在本步的 curriculum 激活集 {sorted(self.active)}")
        self.computed.add(name)
        self.parts[name] = float(value.detach())
        return self.w[name] * value

    def finish(self) -> None:
        if self.computed != self.active:
            die(f"损失接线校验失败：本步激活 {sorted(self.active)}，实算 "
                f"{sorted(self.computed)}")


# --------------------------------------------------------------------------- #
# teacher 抽查（A2）：独立重算一遍，验折线 G-A1 与一步 G-A2
# --------------------------------------------------------------------------- #
def teacher_spotcheck(sample: dict, blob: bytes, idx: torch.Tensor,
                      r_batch: torch.Tensor, h_batch: torch.Tensor,
                      bank: ST.LutVolumes, n_steps: int, device: str,
                      tol: float, rhos: list[float], m_pick: int) -> dict:
    row = json.loads(blob)
    x0, _ = T0.load_pair(sample["shard"], row, sample["after_asset"])
    a = T0.alpha_fields(row, x0)
    mask = T0.depth_mask(sample["depth"], n_steps)
    alphas = a.reshape(n_steps, -1)[:, idx] * mask.unsqueeze(-1)
    x_px = x0.reshape(-1, 3)[idx].to(device)
    r, hstar = ST.build_intermediates(x_px, alphas.to(device), row["luts"], bank,
                                      device=device)
    # 逐位比对只在**与目标构造同一设备**上成立：同一样本 GPU 与 CPU 的 fp32 前向链
    # 差 1.19e-07 ~ 1.79e-07（preflight NOTES D3/D9 已实测），所以 teacher_device
    # 必须等于 DataLoader worker 用的 cpu，否则退化成 tol 比对并如实记录。
    same_r = bool(torch.equal(r, r_batch.to(device)))
    same_h = bool(torch.equal(hstar, h_batch.to(device)))
    bitexact_required = (torch.device(device).type == "cpu")
    if bitexact_required and not (same_r and same_h):
        die(f"teacher 抽查 {sample['id']}: 批内中间态与独立重算不逐位相同 "
            f"(r={same_r}, h*={same_h})")
    if not bitexact_required:
        for nm, p, q in (("r", r, r_batch.to(device)),
                         ("hstar", hstar, h_batch.to(device))):
            e = float((p - q).abs().max())
            if not e <= tol:
                die(f"teacher 抽查 {sample['id']}: {nm} 与独立重算差 {e:.6e} "
                    f"> tol {tol:g}")
    al = alphas.to(device)
    a2 = 0.0
    for m in range(1, n_steps + 1):
        e = float((r[m] + al[m - 1].unsqueeze(-1) * hstar[m - 1] - r[m - 1])
                  .abs().max())
        a2 = max(a2, e)
        if not e <= tol:
            die(f"G-A2 FAILED {sample['id']} m={m}: max|err| = {e:.6e} > tol {tol:g}")
    vols = [bank.get(n, device, x_px.dtype) for n in row["luts"]]
    a1 = 0.0
    m = int(m_pick)
    for rho in rhos:
        als = [al[k].view(1, -1, 1) if k < m - 1 else
               (al[k].view(1, -1, 1) * rho if k == m - 1 else
                torch.zeros_like(al[k].view(1, -1, 1))) for k in range(n_steps)]
        z = ST.chain_with(als, vols, x_px.unsqueeze(0))[0]
        lin = (1.0 - rho) * r[m - 1] + rho * r[m]
        e = float((z - lin).abs().max())
        a1 = max(a1, e)
        if not e <= tol:
            die(f"G-A1 FAILED {sample['id']} m={m} rho={rho}: max|err| = {e:.6e} "
                f"> tol {tol:g}")
    return dict(id=sample["id"], m=m, g_a1_max=a1, g_a2_max=a2,
                device=str(device), bitexact_required=bitexact_required,
                r_bit_exact=same_r, hstar_bit_exact=same_h)


# --------------------------------------------------------------------------- #
# 前向：一步（训练）与整链（rollout / 评测）
# --------------------------------------------------------------------------- #
def cond_base(model, feat, edits, amp_dt, scope):
    """bf16 只包 conditioner trunk；其后（含 FiLM/action/更新）一律 fp32。
    BK 分支：整链编辑描述子进 base（ditblk 用；其余臂忽略）。"""
    ed = model.edit_descriptor(edits)
    if amp_dt is not None and scope == "trunk_only":
        with torch.autocast("cuda", dtype=amp_dt):
            base = model.cond.base(feat, ed)
        return base.float()
    return model.cond.base(feat, ed)


def reverse_reference(r: torch.Tensor, depth: torch.Tensor,
                      n_stages: int) -> torch.Tensor:
    """逆序参考折线：`ref[:, j]` = 走完 j 个逆序阶段后的**真实**状态。

    solver 的循环是 `m = n_stages..1`，而 `β_m = 0`（m > 该样本 depth）的阶段
    是空转。所以批内混深度时，走了 j 次循环真正生效的阶段数是
    `eff = clamp(j - (n_stages - depth), 0, depth)`，参考态是 `r^(depth - eff)`
    —— 不是 `r^(depth - j)`。评测里 `n_stages == depth`，两者恰好重合，
    所以这个偏移只在训练 rollout（B4 逐段监督）上才暴露出来。
    """
    ar = torch.arange(r.shape[0], device=r.device)
    lead = int(n_stages) - depth                      # 前面空转的阶段数
    out = []
    for j in range(int(n_stages) + 1):
        eff = torch.clamp(j - lead, min=0)
        eff = torch.minimum(eff, depth)
        out.append(r[ar, depth - eff])
    return torch.stack(out, dim=1)


# --------------------------------------------------------------------------- #
# 评测
# --------------------------------------------------------------------------- #
def err_stats(e: torch.Tensor) -> dict:
    return T0.err_stats(e)


def _stage_mask_frac(mtr: dict):
    """N4：逐阶段 β_m==0 像素逐位不动的比例（全阶段汇总）。无 β==0 像素则 None。"""
    d = mtr.get("stage_mask_bitexact") or {}
    tot = sum(v["n_beta_zero"] for v in d.values())
    ok = sum(v["n_bit_exact"] for v in d.values())
    return (ok / tot) if tot else float("nan")


def _corr_across_stages(kappa_by_stage: dict, drift_by_stage: dict):
    """N5：跨阶段的 Pearson corr(log κ_p50, rollout_drift)。阶段数 <3 给 None。"""
    ms = [m for m in kappa_by_stage
          if kappa_by_stage.get(m) and m in drift_by_stage
          and drift_by_stage[m] is not None]
    if len(ms) < 3:
        return None
    lk = np.log(np.array([max(kappa_by_stage[m]["p50"], 1e-12) for m in ms]))
    dr = np.array([drift_by_stage[m] for m in ms], dtype=float)
    lk = lk - lk.mean()
    dr = dr - dr.mean()
    den = float(np.linalg.norm(lk) * np.linalg.norm(dr))
    return float((lk * dr).sum() / den) if den > 1e-12 else None


@torch.no_grad()
def quick_eval(model, samples, blobs, feats, cfg, device, ledger, n_steps,
               bank, path_mode, nfe_per_stage, exact_source, tokens, diag,
               shuffle_salt, sabotage=None, edit_src=None, de00_max_px=0) -> dict:
    """held-out 恢复误差 + 提案第 3 节的全部预注册守卫列。只出数字。

    列（`[eval] columns`，逐列 assert_wired）：
      误差列  model / identity / exact_solve / model_midpoint /
              active_only_headline / stage_{all,active,beta1}[m] / cycle[m]
      标量列  rollout_drift[m] / mask_leak_max / mask_bitexact_count /
              stage_oob_fraction[m] / stage_oob_max[m] /
              condition_number_by_stage[m]（与 e_stage 同像素相关列）/
              nfe_d_vs_2d / latency_by_depth / Δ_const / Δ_shuffle
      可选列  true_lut_condition_delta（lut_id_condition="embed" 才有）、
              predicted_lut_accuracy（挂 LUT 辅助分类头才有）；首版都不注册，
              在 metrics 的 `columns_declared_unavailable` 里注明，不出假数。
    """
    n_pix = cfg.int_("eval", "pixels_per_sample")
    ledger.arm(len(samples))
    model.eval()
    recs: list[dict] = []
    keys = [T0.feature_key(s) for s in samples]
    n = len(samples)
    lut_cond = model.lut_id_condition
    # B5：Δ_shuffle 的错位对象必须跨 source_id（同源不同 depth 的条件近乎等价，
    # 用它做负控制会系统性低估 Δ）。规则与 salt 都在 TOML 里。
    partner = ST.shuffle_partner_index([s["id"] for s in samples], shuffle_salt,
                                       T0.source_id_of)

    def want(col):
        """该列在本臂是否要算：既要已注册（结构性不可用的列根本不在 columns
        里），也要没被 --sabotage-metric 故意跳过。"""
        return col in ledger.columns and sabotage != col

    def note_err(col, e, source):
        """一列误差：ledger 记一次，返回 err_stats。"""
        ledger.note(col, source=source)
        return T0.err_stats(e)

    def note_val(col, value, source, extra=None):
        miss = value is None or (isinstance(value, float) and math.isnan(value))
        ledger.note(col, source=source, missing=bool(miss))
        d = dict(value=(None if miss else float(value)), source=source)
        if extra:
            d.update(extra)
        return d

    for i, s in enumerate(samples):
        row = json.loads(blobs[s["id"]])
        x0, y = T0.load_pair(s["shard"], row, s["after_asset"])
        a = T0.alpha_fields(row, x0)
        npx = x0.shape[0] * x0.shape[1]
        idx = (torch.arange(npx) if n_pix <= 0 else
               torch.arange(0, npx, max(1, npx // n_pix))[:n_pix])
        mask = T0.depth_mask(s["depth"], n_steps)
        alphas = (a.reshape(n_steps, -1)[:, idx] * mask.unsqueeze(-1)).to(device)
        xb = x0.reshape(-1, 3)[idx].to(device).unsqueeze(0)
        yb = y.reshape(-1, 3)[idx].to(device).unsqueeze(0)
        alphas = alphas.unsqueeze(0)
        d_int = n_steps if s["depth"] is None else int(s["depth"])
        depth = torch.tensor([d_int], device=device)
        sc = torch.tensor([float(row["calib"]["s"])], device=device)
        union = ST.union_mask_of(alphas)
        ahat = ST.compose_alpha_hat(alphas)
        feat = T0.gather_feats(feats, [keys[i]], tokens, device)
        edits_full = (None if edit_src is None else
                      edit_src.for_row(row, x0, mask).unsqueeze(0).to(device))
        base = model.cond.base(feat, model.edit_descriptor(edits_full))
        n_stages = SS.stages_for(path_mode, d_int)
        lut_ids = (torch.tensor(
            [[ST.lut_id_token(nm, model.lut_id_vocab) for nm in row["luts"]]],
            dtype=torch.long, device=device) if lut_cond == "embed" else None)
        # X-COND：逐阶段编辑来源（oracle）。inv_lut 递 (1,K) long 行号，
        # ref_pair 递 (1,K,D) float 描述子；两者都不进 ProductionBatch。
        edits = edits_full

        r, hstar = ST.build_intermediates(xb[0], alphas[0], row["luts"], bank,
                                          device=device)
        r, hstar = r.unsqueeze(0), hstar.unsqueeze(0)
        ref = reverse_reference(r, depth, n_stages)

        if device.startswith("cuda"):
            torch.cuda.synchronize()
        t_lat = time.time()
        pred, mtr = SS.rollout_with_metrics(
            model, base, yb, ahat, alphas, union, depth, sc, path_mode,
            n_stages, 1, ref_states=ref, lut_ids=lut_ids,
            stage_mask_check=True, edits=edits)
        if device.startswith("cuda"):
            torch.cuda.synchronize()
        latency_ms = (time.time() - t_lat) * 1000.0
        pred2, mtr2 = SS.rollout_with_metrics(
            model, base, yb, ahat, alphas, union, depth, sc, path_mode,
            n_stages, 2, ref_states=ref, lut_ids=lut_ids, edits=edits)
        if mtr["nfe"] != n_stages or mtr2["nfe"] != n_stages * 2:
            die(f"A5 NFE 计数失败: {mtr['nfe']}/{mtr2['nfe']} vs d={n_stages}")

        cols: dict[str, dict] = {}
        e_model = T0.linf8(pred, xb)
        if want("model"):
            cols["model"] = note_err("model", e_model, "uint8_asset")
        if want("identity"):
            cols["identity"] = note_err("identity", T0.linf8(yb, xb), "uint8_asset")
        if want("exact_solve"):
            cols["exact_solve"] = T0.exact_column(s, idx, xb, ledger, exact_source,
                                                  device)
        if want("model_midpoint"):
            cols["model_midpoint"] = note_err("model_midpoint",
                                              T0.linf8(pred2, xb), "uint8_asset")
        # ---- ΔE00（构造工具 de00 的原件；确定性等距子采样到 de00_max_px）---- #
        de_m = de_i = None
        if want("de00") or want("de00_frac_le5"):
            de_m = EC.de00_pixels(pred[0], xb[0], de00_max_px, B.de00)
        if want("de00_identity") or want("de00_identity_frac_le5"):
            de_i = EC.de00_pixels(yb[0], xb[0], de00_max_px, B.de00)
        if want("de00"):
            cols["de00"] = note_err("de00", torch.from_numpy(de_m).float(),
                                    "ciede2000_vs_clean")
        if want("de00_frac_le5"):
            cols["de00_frac_le5"] = note_val(
                "de00_frac_le5", float((de_m <= 5.0).mean()), "ciede2000_vs_clean",
                dict(n_px=int(de_m.size), criterion="ΔE00 <= 5"))
        if want("de00_identity"):
            cols["de00_identity"] = note_err(
                "de00_identity", torch.from_numpy(de_i).float(),
                "ciede2000_vs_clean")
        if want("de00_identity_frac_le5"):
            cols["de00_identity_frac_le5"] = note_val(
                "de00_identity_frac_le5", float((de_i <= 5.0).mean()),
                "ciede2000_vs_clean",
                dict(n_px=int(de_i.size), criterion="ΔE00 <= 5"))

        act = union[0] > 0
        if want("active_only_headline"):
            cols["active_only_headline"] = note_err(
                "active_only_headline", e_model[0][act], "uint8_asset_active")

        # ---- 掩膜守卫：union_mask==0 的像素必须与 y 逐位相同 ------------------ #
        off = ~act
        n_off = int(off.sum())
        if want("mask_leak_max"):
            leak = (float((pred[0][off] - yb[0][off]).abs().max()) if n_off else 0.0)
            cols["mask_leak_max"] = note_val("mask_leak_max", leak, "off_union")
        if want("mask_bitexact_count"):
            eq = (int((pred[0][off] == yb[0][off]).all(dim=-1).sum())
                  if n_off else 0)
            cols["mask_bitexact_count"] = note_val(
                "mask_bitexact_count", float(eq), "off_union",
                dict(n_off_union=n_off,
                     frac=(eq / n_off if n_off else None),
                     criterion="union_mask==0 像素与 y 逐位相同"))

        # ---- 逐阶段：stage 三分层 / cycle / condition number ----------------- #
        st_all, st_act, st_b1, cyc = [], [], [], []
        by_stage = {"stage_all": {}, "stage_active": {}, "stage_beta1": {},
                    "cycle": {}, "condition_number_by_stage": {}}
        kap_all, corr_by = [], {}
        cn_px = min(int(diag["cn_max_px"]), int(idx.numel()))
        for m_idx in range(1, n_stages + 1):
            m = torch.tensor([m_idx], device=device)
            lam = torch.zeros(1, device=device)
            z, beta, h_tgt, _ = ST.path_states(path_mode, r, hstar, alphas, union,
                                               depth, m, lam)
            lid = None if lut_ids is None else lut_ids[:, m_idx - 1]
            ed = None if edits is None else edits[:, m_idx - 1]
            h_hat, _ = model(base, z, yb, ahat, beta, alphas, lam, m, depth, sc,
                             lid, ed)
            e_stage = T0.linf8(h_hat, h_tgt)[0]              # (P,)
            b0 = beta[0]
            sel_act, sel_b1 = b0 > 0, b0 == 1
            st_all.append(e_stage)
            st_act.append(e_stage[sel_act])
            st_b1.append(e_stage[sel_b1])
            for nm, v in (("stage_all", e_stage), ("stage_active", e_stage[sel_act]),
                          ("stage_beta1", e_stage[sel_b1])):
                by_stage[nm][m_idx] = (T0.err_stats(v) if v.numel() else None)

            # cycle：模型走一步逆序，再用真实前向算子走回来（只做指标，不反传）
            z_prev = SS._update(z, beta, h_hat)
            if path_mode == "chain":
                vol = bank.get(row["luts"][m_idx - 1], device, z.dtype)
                fwd = LUTDATA.mix_alpha(z_prev[0],
                                        ST.apply_lut_px(vol, z_prev[0]),
                                        beta[0].unsqueeze(-1))
            else:                       # linear / one_shot：路径自带前向 = 逆更新的反向
                fwd = z_prev[0] - beta[0].unsqueeze(-1) * h_tgt[0]
            e_cyc = T0.linf8(fwd.unsqueeze(0), z)[0][sel_act]
            cyc.append(e_cyc)
            by_stage["cycle"][m_idx] = T0.err_stats(e_cyc) if e_cyc.numel() else None

            # condition number：κ₂((1-β)I + βJ_L)，与 e_stage 同像素的相关列
            if diag["cn_on"] and path_mode == "chain" and cn_px > 0:
                sub = slice(0, cn_px)
                vol = bank.get(row["luts"][m_idx - 1], device, z.dtype)
                J = ST.fd_jacobian(vol, r[0, m_idx - 1][sub].unsqueeze(0),
                                   int(row["grids"][m_idx - 1]), diag["fd_lo"],
                                   diag["fd_hi"], diag["fd_min_h"])
                kap, valid, _ = ST.condition_number(J, b0[sub], diag["svd_chunk"],
                                                    diag["sigma_floor"])
                kv = kap[valid]
                by_stage["condition_number_by_stage"][m_idx] = (
                    T0.err_stats(kv) if kv.numel() else None)
                if kv.numel() > 2:
                    kap_all.append(kv)
                    ev = e_stage[sub][valid]
                    lk = torch.log(kv.clamp_min(1e-12))
                    lk = lk - lk.mean()
                    ev2 = ev - ev.mean()
                    den = (lk.norm() * ev2.norm()).clamp_min(1e-12)
                    corr_by[m_idx] = float((lk * ev2).sum() / den)

        for nm, parts in (("stage_all", st_all), ("stage_active", st_act),
                          ("stage_beta1", st_b1), ("cycle", cyc)):
            if not want(nm):
                continue
            # β==1 这一层在某些链上可以整条为空（β 已乘全局强度 s，s≠1 时
            # 「场==1」并不给出 β==1）。空就如实记 missing，不编数字。
            keep = [p for p in parts if p.numel()]
            flat = torch.cat(keep) if keep else None
            if flat is None or flat.numel() == 0:
                ledger.note(nm, source="teacher_state", missing=True)
                cols[nm] = dict(missing=True, source="teacher_state")
            else:
                cols[nm] = note_err(nm, flat, "teacher_state")
                cols[nm]["by_stage"] = by_stage[nm]

        if want("condition_number_by_stage"):
            kv = torch.cat(kap_all) if kap_all else None
            cols["condition_number_by_stage"] = note_val(
                "condition_number_by_stage",
                (float(torch.quantile(kv.float()[:1 << 22], 0.5)) if kv is not None
                 and kv.numel() else float("nan")),
                "fd_jacobian",
                dict(by_stage=by_stage["condition_number_by_stage"],
                     corr_log_kappa_vs_stage_err=corr_by,
                     corr_log_kappa_vs_rollout_drift=_corr_across_stages(
                         by_stage["condition_number_by_stage"],
                         mtr["rollout_drift_by_stage"]),
                     sigma_floor=diag["sigma_floor"], n_pixels=cn_px,
                     note=("与 e_stage 同像素；σ_min < sigma_floor 的像素显式排除，"
                           "不做 clamp 兜底（preflight B1）")))

        for col, val, src, extra in (
                ("rollout_drift", mtr["rollout_drift"], "euler_d",
                 dict(by_stage=mtr["rollout_drift_by_stage"])),
                ("stage_oob_fraction", mtr["stage_oob_fraction"], "euler_d",
                 dict(by_stage=mtr["stage_oob_fraction_by_stage"])),
                ("stage_oob_max", mtr["stage_oob_max"], "euler_d",
                 dict(by_stage=mtr["stage_oob_max_by_stage"])),
                ("latency_by_depth", latency_ms, "wall_clock_ms",
                 dict(depth=d_int, n_pixels=int(idx.numel()))),
                ("stage_mask_bitexact", _stage_mask_frac(mtr), "per_stage_beta0",
                 dict(by_stage=mtr["stage_mask_bitexact"],
                      criterion="每阶段 β_m==0 的像素经该步更新后逐位不动"
                                "（G-A3 口径）"))):
            if not want(col):
                continue
            cols[col] = note_val(col, val, src, extra)

        # 条件性负控制（c 是图像条件，不是文本：const = 零向量，
        # shuffle = **跨 source_id** 的确定性错位，B5）
        p50_model = cols["model"]["p50"] if "model" in cols else float("nan")
        for col, ctrl in (("delta_const", torch.zeros_like(feat)),
                          ("delta_shuffle",
                           T0.gather_feats(feats, [keys[partner[i]]], tokens,
                                           device))):
            if not want(col):
                continue
            pc, _ = SS.rollout_with_metrics(
                model, model.cond.base(ctrl, model.edit_descriptor(edits)), yb, ahat,
                alphas, union, depth, sc,
                path_mode, n_stages, 1, lut_ids=lut_ids, edits=edits)
            v = T0.err_stats(T0.linf8(pc, xb))
            cols[col] = note_val(col, v["p50"] - p50_model, "cond_control",
                                 dict(control_p50=v["p50"]))
        if want("nfe_d_vs_2d"):
            cols["nfe_d_vs_2d"] = note_val(
                "nfe_d_vs_2d", cols["model_midpoint"]["p50"] - p50_model,
                "euler_vs_midpoint",
                dict(nfe_euler=mtr["nfe"], nfe_midpoint=mtr2["nfe"]))
        # K5 诊断行：只有真的挂了 LUT-ID 条件才出数，否则该列根本不注册
        if want("true_lut_condition_delta"):
            null_ids = torch.zeros_like(lut_ids)
            pn, _ = SS.rollout_with_metrics(
                model, base, yb, ahat, alphas, union, depth, sc, path_mode,
                n_stages, 1, lut_ids=null_ids, edits=edits)
            vn = T0.err_stats(T0.linf8(pn, xb))
            cols["true_lut_condition_delta"] = note_val(
                "true_lut_condition_delta", vn["p50"] - p50_model,
                "true_id_vs_null_id", dict(no_id_p50=vn["p50"]))

        # ---- 编辑侧负控制（X-COND）：条件是编辑描述子本身，不是 SigLIP 的 c --- #
        # null 问「用没用编辑信息」，roll 问「用没用**对**的那条编辑」。
        # 两者都不改样本、不改容量，只改喂进去的编辑描述子。
        for col, mk in (("delta_edit_null", model.null_edit),
                        ("delta_edit_roll", model.roll_edit)):
            if not want(col):
                continue
            if edits is None:
                die(f"{col} 已注册但 edit.condition = off —— 该列无处可算")
            ed_ctrl = mk(edits)          # BK 分支：被替换的编辑也进 base（token 侧）
            pc, _ = SS.rollout_with_metrics(
                model, model.cond.base(feat, model.edit_descriptor(ed_ctrl)), yb, ahat,
                alphas, union, depth, sc, path_mode,
                n_stages, 1, lut_ids=lut_ids, edits=ed_ctrl)
            vc = T0.err_stats(T0.linf8(pc, xb))
            cols[col] = note_val(col, vc["p50"] - p50_model, "edit_control",
                                 dict(control_p50=vc["p50"]))

        recs.append(dict(id=s["id"], depth=d_int, geom=s["geom"],
                         rec_band=s["rec_band"], major=s["major"],
                         n_eval_pixels=int(idx.numel()),
                         union_frac=float(union.mean()),
                         alpha_hat_mean=float(ahat.mean()),
                         y_quant_gap=float((r[0, d_int] - yb[0]).abs().max()),
                         cols=cols))
    model.train()

    def med(col, field):
        v = [rr["cols"][col][field] for rr in recs
             if col in rr["cols"] and rr["cols"][col].get(field) is not None]
        return float(np.median(np.asarray(v, dtype=float))) if v else float("nan")

    overall = {}
    for c in ledger.columns:
        overall[c] = ({f: med(c, f) for f in ("p50", "p95", "p99")}
                      if c in ERR_COLUMNS else dict(value=med(c, "value")))
    by: dict[str, dict] = {}
    for stratum in ledger.strata:
        groups: dict[str, list] = {}
        for rr in recs:
            groups.setdefault(str(rr[stratum]), []).append(rr)
        by[stratum] = {}
        for k, v in sorted(groups.items()):
            cell = dict(n=len(v))
            for c in ledger.columns:
                fields = ("p50", "p95") if c in ERR_COLUMNS else ("value",)
                got = {}
                for f in fields:
                    vals = [r2["cols"][c][f] for r2 in v
                            if c in r2["cols"] and r2["cols"][c].get(f) is not None]
                    got[f] = float(np.median(vals)) if vals else float("nan")
                cell[c] = got
            by[stratum][k] = cell
    out = dict(n=len(recs),
               metric="8-bit L-inf over RGB, |x_hat - before| * 255; per-sample "
                      "p50/p95/p99, then the MEDIAN across evaluated samples",
               err_columns=[c for c in ledger.columns if c in ERR_COLUMNS],
               scalar_columns=[c for c in ledger.columns if c not in ERR_COLUMNS],
               exact_solve_source=exact_source,
               column_missing={c: n for c, n in ledger.missing.items()},
               column_sources={c: sorted(v) for c, v in ledger.sources.items()},
               overall=overall, by=by, per_sample=recs)
    ledger.assert_wired(out)
    return out


def print_eval(r: dict) -> None:
    print(f"\n=== held-out, median across {r['n']} samples "
          f"({r.get('eval_kind')} eval, step {r.get('step')}) ===", flush=True)
    for c, v in r["overall"].items():
        if v.get("status") == "not_registered":
            print(f"{c:16s} not_registered（本臂结构性不适用）")
        elif "p50" in v:
            print(f"{c:16s} p50={v['p50']:9.3f} p95={v['p95']:9.3f} "
                  f"p99={v['p99']:9.3f}  missing={r['column_missing'].get(c, 0)}")
        else:
            print(f"{c:16s} value={v['value']!r}  "
                  f"missing={r['column_missing'].get(c, 0)}")


# --------------------------------------------------------------------------- #
def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--smoke", action="store_true",
                    help="冒烟：只验断言接线与损失有限值，数字不可读")
    ap.add_argument("--limit-samples", type=int, default=0,
                    help="冒烟用：train / held-out 各截到 N 条")
    ap.add_argument("--stop-after", type=int, default=0)
    ap.add_argument("--skip-final-eval", action="store_true",
                    help="ckpt_last 落盘后不跑逐样本 final eval；写 metrics_train.json 退出（final 由 batch_eval_bk.py 出）")
    ap.add_argument("--sabotage-metric", default=None,
                    help="故意跳过一个预注册列；A4 必须触发（守卫验证用）")
    a = ap.parse_args()

    cfg = T0.Cfg(Path(a.config))
    out_dir = Path(cfg.str_("run", "out_dir"))
    device = cfg.str_("run", "device")
    seed = cfg.int_("run", "seed")
    cfg_max_steps = cfg.int_("run", "max_steps")
    one_pass_cap = cfg.bool_("run", "one_pass_cap")
    cfg_log_every = cfg.int_or_auto("run", "log_every")
    cfg_eval_every = cfg.int_or_auto("run", "eval_every")
    cfg_ckpt_every = cfg.int_or_auto("run", "ckpt_every")
    log_div = cfg.int_("run", "auto_log_divisor")
    eval_div = cfg.int_("run", "auto_eval_divisor")
    eval_at = sorted({int(x) for x in cfg.list_("run", "eval_at", int)})
    # 「定义了没接线」红线：eval_at 被读取/校验/写进 run_args，但**从未**在训练
    # 循环里触发。在真正接线之前，非空即停机——绝不允许它静默变成 no-op。
    # 中间 eval 由 [run] eval_every（auto = budget//10）提供，末次是全量 held-out。
    if eval_at:
        die(f"[run] eval_at = {eval_at} 非空，但该键在训练循环里**没有接线**"
            "（读取/校验/落 run_args 都有，触发没有）。这是「定义了没接线」红线，"
            "接线之前禁止使用；中间 eval 请用 [run] eval_every。")
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cuda.matmul.allow_tf32 = False

    # ---- A3 config 自动 diff ------------------------------------------------ #
    arm_name = cfg.str_("arm", "name")
    peers = cfg.list_("arm", "peer_configs")
    identity_keys = cfg.list_("arm", "identity_keys")
    diff_report = assert_arm_diff(cfg, peers, identity_keys)

    # ---- A8 sha256 冻结 ----------------------------------------------------- #
    frozen = dict(
        train_sprf=T0.sha256_file(Path(__file__)),
        stage_targets=T0.sha256_file(Path(ST.__file__)),
        stage_flow=T0.sha256_file(_P.src("stage_flow.py")),
        stage_solver=T0.sha256_file(_P.src("stage_solver.py")),
        stage_flow_bk=T0.sha256_file(Path(SF.__file__)),
        stage_solver_bk=T0.sha256_file(Path(SS.__file__)),
        train_stage0=T0.sha256_file(Path(T0.__file__)),
        config=cfg.sha256,
        lutdata=T0.sha256_file(Path(LUTDATA.__file__)),
        epr050_build_degradation=T0.sha256_file(Path(B.__file__)))
    allowed = set(cfg.list_("guard", "build_tool_sha256_allowed"))
    if frozen["epr050_build_degradation"] not in allowed:
        die(f"epr050_build_degradation.py sha {frozen['epr050_build_degradation'][:12]}"
            " 不在 [guard] build_tool_sha256_allowed")
    want_lut = cfg.str_("guard", "lutdata_sha256")
    if frozen["lutdata"] != want_lut:
        die(f"q3vl/whatb/lutdata.py sha {frozen['lutdata'][:12]} != [guard] "
            f"lutdata_sha256 {want_lut[:12]} —— 渲染律在运行下被改了")

    # ---- 数据（全部 import 原件） -------------------------------------------- #
    ST.install_compact_row(T0)
    T0.ASSET_MODE = cfg.str_("data", "asset_source", ("dir", "tar", "auto"))
    samples, blobs, index_meta = T0.load_shards(cfg)
    law = T0.bind_build_config(samples,
                               cfg.list_("guard", "build_config_sha256_allowed"))
    n_steps = law["n_steps"]
    train = [s for s in samples if not s["heldout"]]
    held = [s for s in samples if s["heldout"]]
    if not held:
        die("held-out 为空；检查 [data] split_salt / holdout_bucket_min")
    train.sort(key=lambda s: (s["id"], s["depth"] or 0))
    held.sort(key=lambda s: (s["id"], s["depth"] or 0))
    if a.limit_samples:
        train = train[:a.limit_samples]
        held = held[:a.limit_samples]
        samples = train + held
    heldout_ids = {s["id"] for s in held}
    guard = ST.HeldoutGuard(heldout_ids)          # A7
    print(f"samples: {len(samples)} ({len(train)} train / {len(held)} held-out), "
          f"chain length {n_steps}, step_order {law['step_kind']}", flush=True)

    inputs = cfg.list_("encoder", "inputs")
    feats, cache_meta = T0.build_feature_cache(cfg, samples, device, inputs)
    tokens = int(cache_meta["tokens"])
    in_dim = cache_meta["dim"] * (1 if tokens else len(inputs))

    # ---- 模型 --------------------------------------------------------------- #
    path_mode = cfg.str_("flow", "path_mode", ST.PATH_MODES)
    endpoint_probability = cfg.num("flow", "endpoint_probability")
    alpha_mode = cfg.str_("flow", "alpha_mode", ST.ALPHA_MODES)
    # D-quant 裁决：两个起点是预注册契约键，只实现这一种组合，配别的立即停机。
    rollout_anchor = cfg.str_("flow", "rollout_anchor", ("asset_y",))
    teacher_anchor = cfg.str_("flow", "teacher_anchor", ("chain_r",))
    nfe_per_stage = cfg.int_("solver", "nfe_per_stage")
    clamp_lo = cfg.num("solver", "clamp_lo")
    clamp_hi = cfg.num("solver", "clamp_hi")
    depth_values = cfg.list_("data", "depth_values", int)
    model = SF.BkSprfModel(in_dim, cfg, n_steps, alpha_mode, depth_values).to(device)
    pcount = model.param_counts()

    lr = cfg.num("optim", "lr")
    wd = cfg.num("optim", "weight_decay")
    betas = tuple(cfg.list_("optim", "betas", float))
    batch = cfg.int_("optim", "batch")
    cfg_warmup = cfg.int_or_auto("optim", "warmup_steps")
    warmup_frac = cfg.num("optim", "warmup_frac")
    warmup_max = cfg.int_("optim", "warmup_max")
    schedule = cfg.str_("optim", "schedule", ("cosine", "constant"))
    clip = cfg.num("optim", "grad_clip")
    amp = cfg.str_("optim", "autocast", ("bf16", "fp16", "off"))
    scope = cfg.str_("optim", "autocast_scope", ("trunk_only", "off"))
    amp_dt = {"bf16": torch.bfloat16, "fp16": torch.float16}.get(amp)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=wd, betas=betas)

    one_pass = math.ceil(len(train) / batch)
    max_steps = min(cfg_max_steps, one_pass) if one_pass_cap else cfg_max_steps
    log_every = T0.derive_cadence(cfg_log_every, max_steps, log_div, "[run] log_every")
    eval_every = T0.derive_cadence(cfg_eval_every, max_steps, eval_div,
                                   "[run] eval_every")
    ckpt_every = T0.derive_cadence(cfg_ckpt_every, max_steps, eval_div,
                                   "[run] ckpt_every", allow_zero=True)
    warmup = (min(warmup_max, int(warmup_frac * max_steps)) if cfg_warmup is None
              else cfg_warmup)
    if warmup >= max_steps:
        die(f"[optim] warmup_steps = {warmup} 不小于有效预算 {max_steps}")
    for e in eval_at:
        if e < 1 or e > max_steps:
            die(f"[run] eval_at 的 {e} 越出 1..{max_steps}")

    # ---- 损失权重与 curriculum ---------------------------------------------- #
    weights = {n: cfg.num("loss", f"w_{n}")
               for n in ("prev", "action", "clean", "bound", "rollout", "q50")}
    book = LossBook(weights)
    h_bound = cfg.num("loss", "action_abs_bound")
    grad_ckpt = cfg.bool_("optim", "rollout_grad_checkpoint")
    ch_eps = cfg.num("loss", "charbonnier_eps")
    sl1_beta = cfg.num("loss", "smooth_l1_beta")
    roll_start = cfg.num("loss", "rollout_start_frac")
    roll_step = cfg.num("loss", "rollout_step_frac")
    roll_bf0 = cfg.num("loss", "rollout_batch_frac_start")
    roll_bf1 = cfg.num("loss", "rollout_batch_frac_end")
    q50_start = cfg.num("loss", "q50_start_frac")
    q50_lo = cfg.num("loss", "q50_band_lo")
    q50_hi = cfg.num("loss", "q50_band_hi")
    if q50_start < roll_start:
        die(f"[loss] q50_start_frac {q50_start} < rollout_start_frac {roll_start}："
            "L_q50 建立在 rollout 预测上，顺序不能反")
    core = [n for n in ("prev", "action", "clean", "bound") if weights[n] != 0.0]

    n_pix = cfg.int_("loss", "pixels_per_sample")
    bank_dir = cfg.str_("data", "lut_bank_dir")
    lut_cache = cfg.int_("data", "lut_cache_size")
    bank_main = ST.LutVolumes(bank_dir, lut_cache)

    # ---- X-COND：显式编辑信息注入（oracle 条件源） -------------------------- #
    # 假设「模型不知道编辑方向」。两档 oracle：inv_lut（直接给编辑）/
    # ref_pair（给编辑的演示对）。两档描述子同形 (grid³,4)，编码器同构。
    espec = SF.edit_spec(cfg)
    edit_src = None
    edit_meta = dict(condition=espec["condition"], contract=espec["contract"],
                     section_present=espec["section_present"],
                     latent_dim=espec["latent_dim"], grid=espec["grid"],
                     descriptor_dim=espec["descriptor_dim"])
    build_cfg_sha = T0.sha256_file(Path(law["config_path"]))
    if espec["condition"] == "inv_lut":
        inv = EC.InvLutSource(cfg.str_("edit", "inv_cache_dir"), espec["grid"])
        edit_meta["inv_lut"] = inv.assert_fingerprint(
            frozen["epr050_build_degradation"], build_cfg_sha, bank_dir)
        model.load_inv_table(inv.load_table())
        edit_src = inv
        print(f"[edit] inv_lut 逆表 {inv.dir} | {len(inv.names)} 支 LUT | "
              f"求逆失败格 {inv.totals['frac_fail_overall'] * 100:.1f}%"
              f"（**不吸收**，逐格在 valid 通道里）", flush=True)
    elif espec["condition"] == "ref_pair":
        pool, psrc, pmeta = EC.build_ref_pool(
            train, blobs, cfg.int_("edit", "ref_pool_n"),
            cfg.int_("edit", "ref_size"))
        edit_src = EC.RefPairSource(pool, psrc, cfg.str_("edit", "ref_salt"),
                                    espec["grid"], bank_dir, lut_cache)
        edit_meta["ref_pair"] = pmeta
        print(f"[edit] ref_pair 参考池 {pmeta['n_ref']} 张 "
              f"{pmeta['size']}×{pmeta['size']}（train 侧，跨 source 选取）",
              flush=True)
    if espec["condition"] != "off":
        print(f"[edit] condition_contract = {espec['contract']} —— 本臂是 oracle "
              "诊断行，生产 restore() 已被拒；数字不可与生产臂混读", flush=True)

    # ---- 评测口径（沿用 train_stage0） ---------------------------------------- #
    columns = cfg.list_("eval", "columns")
    strata = cfg.list_("eval", "strata")
    # 可选列：开关没开就**不注册**（而不是注册后每个样本都 missing 出假数）。
    declared_unavailable = {}
    if model.lut_id_condition == "off":
        declared_unavailable["true_lut_condition_delta"] = (
            "flow.lut_id_condition = \"off\"：首版不挂 LUT-ID 条件，K5 诊断行"
            "需另跑 lut_id_condition = \"embed\" 的 oracle 臂")
    declared_unavailable["predicted_lut_accuracy"] = (
        "首版不挂 LUT 辅助分类头（提案：`若挂 LUT 辅助分类头`）")
    bad = [c for c in OPTIONAL_COLUMNS
           if c in columns and c in declared_unavailable]
    if bad:
        die(f"[eval] columns 注册了不可用的列 {bad}：{declared_unavailable}")
    # 结构性不可用：κ₂((1-β)I + βJ_L) 是**真实退化链**第 m 阶段算子的性质，
    # linear / one_shot 路径没有「第 m 步的 LUT」，这一列在那两臂上不存在。
    # `[eval] columns` 是 A3 的逐键相同项，所以不能在 TOML 里各写各的：
    # 预注册的全表照写，真正注册的列在这里按臂扣除，并把原因落进 run_args/metrics。
    if path_mode != "chain":
        declared_unavailable["condition_number_by_stage"] = (
            f'flow.path_mode = "{path_mode}"：该路径没有逐阶段 LUT 算子，'
            "κ₂ 与「同像素相关列」无定义（chain 臂上照常出数）")
    registered = [c for c in columns if c not in declared_unavailable]
    dropped = [c for c in columns if c in declared_unavailable]
    if dropped:
        print(f"[eval] 本臂不注册的列 {dropped}："
              f"{ {k: declared_unavailable[k] for k in dropped} }", flush=True)
    columns = registered
    ledger = SprfLedger(columns, strata,
                        cfg.list_("eval", "sparse_ok_columns"))
    diag = dict(cn_on=cfg.bool_("diag", "condition_number_enabled"),
                cn_max_px=cfg.int_("diag", "condition_number_max_pixels"),
                svd_chunk=cfg.int_("diag", "svd_chunk"),
                sigma_floor=cfg.num("diag", "sigma_floor"),
                fd_lo=cfg.num("diag", "fd_clamp_lo"),
                fd_hi=cfg.num("diag", "fd_clamp_hi"),
                fd_min_h=cfg.num("diag", "fd_min_h"))
    select_metric = cfg.str_("eval", "select_metric", SELECT_METRICS)
    exact_source = cfg.str_("eval", "exact_solve_source", ("journal", "asset"))
    de00_max_px = cfg.int_("eval", "de00_max_pixels") if "de00_max_pixels" in \
        cfg.d.get("eval", {}) else 0
    de00_cols = [c for c in cfg.d["eval"]["columns"] if c.startswith("de00")]
    if de00_cols and de00_max_px < 1:
        die(f"[eval] 注册了 {de00_cols} 却没给 de00_max_pixels —— ΔE00 走 skimage，"
            "全量像素不可行，子采样上限必须是预注册键")
    interval_n = cfg.int_("eval", "interval_max_samples")
    final_n = cfg.int_("eval", "final_max_samples")
    subset_salt = cfg.str_("eval", "subset_salt")
    interval_set = T0.eval_subset(held, interval_n, subset_salt)
    final_set = T0.eval_subset(held, final_n, subset_salt)

    shuffle_salt = cfg.str_("eval", "shuffle_salt")
    shuffle_rule = cfg.str_("eval", "shuffle_rule",
                            ("cross_source_sha1_next",))
    # N9④：未知的 sabotage 列名要按 A4 的口径停机，而不是评测里抛 KeyError。
    if a.sabotage_metric is not None and a.sabotage_metric not in columns:
        die(f"--sabotage-metric {a.sabotage_metric!r} 不在本臂注册列 {columns}"
            " —— A4：判据列名对不上，停机")
    tcheck_every = cfg.int_("assert", "teacher_check_every")
    tol = cfg.num("assert", "teacher_tol")
    rhos = cfg.list_("assert", "teacher_rho", float)
    # 目标构造发生在 DataLoader worker 的 CPU 上；逐位比对必须同设备（D3/D9）。
    tdev = cfg.str_("assert", "teacher_device", ("cpu", "cuda:0"))

    out_dir.mkdir(parents=True, exist_ok=True)
    started = time.strftime("%Y%m%dT%H%M%S%z")
    rap = out_dir / "run_args.json"
    if rap.exists():
        rap = out_dir / f"run_args.{started}.json"

    # ---- 启动断言 A1 / A5 / A6 ---------------------------------------------- #
    ds = SprfPixels(train, blobs, n_pix, n_steps, bank_dir, lut_cache,
                    model.lut_id_vocab, edit_src)
    probe = collate([ds[i] for i in range(min(2, len(train)))])
    guard.assert_batch(probe["sid"], "startup probe")
    pf = T0.gather_feats(feats, probe["key"], tokens, device)
    p_al = probe["alphas"].to(device)
    p_y = probe["y"].to(device)
    p_union = ST.union_mask_of(p_al)
    p_depth = probe["depth"].to(device)
    p_s = probe["s"].to(device)
    d_probe = int(p_depth.max())
    full_batch = SS.ProductionBatch(
        cond=pf, y=p_y, alphas=p_al, union=p_union, depth=p_depth, s=p_s,
        lut=None, lut_id=None, before=probe["x"].to(device),
        intermediate_gt=probe["r"].to(device), forward_operator=None)
    strip_batch = SS.ProductionBatch(
        {k: full_batch[k] for k in SS.PRODUCTION_KEYS})
    # X-COND：编辑描述子**从参数递进来**，永不进 batch —— 所以 A6「删掉禁入键
    # 后输出逐位一致」在 oracle 臂上仍然是一条有效断言（禁入键里也加了 edit）。
    p_edit = None if edit_src is None else probe["edit"].to(device)
    ork = espec["condition"] != "off"
    model.eval()
    with torch.no_grad():
        out_full = SS.restore(model, full_batch, path_mode, nfe_per_stage, d_probe,
                              clamp_lo, clamp_hi, edits=p_edit, oracle_ok=ork)
        out_strip = SS.restore(model, strip_batch, path_mode, nfe_per_stage, d_probe,
                               clamp_lo, clamp_hi, edits=p_edit, oracle_ok=ork)
        out_e1 = SS.restore(model, strip_batch, path_mode, 1, d_probe,
                            clamp_lo, clamp_hi, edits=p_edit, oracle_ok=ork)
    if ork:
        # 生产接口守卫必须**真的**拒绝这一臂（不是「写了没接线」）。
        try:
            SS.restore(model, strip_batch, path_mode, nfe_per_stage, d_probe,
                       clamp_lo, clamp_hi, edits=p_edit)
        except SystemExit:
            pass
        else:
            die("A6 FAILED: edit.condition != off 的 oracle 臂，生产 restore() "
                "没有拒绝 —— 守卫没接线")
    model.train()
    a6_ok = bool(torch.equal(out_full, out_strip))
    if not a6_ok:
        die("A6 FAILED: 删除 lut/lut_id/before/intermediate_gt/forward_operator 后 "
            "restore() 输出不逐位一致")
    # A1 按 action backend 分口径：
    #   mlp  action 头零初始化 ⇒ ĥ 恒等于 0 ⇒ 逐位恒等（torch.equal）。
    #   g4d  g4d 的恒等 init 只保证 sum_i w_i ≈ 1（G4DGenerator 自己的 docstring
    #        写的是 f(x,s) ~= x），所以是**两界**：绝对值界 + 8bit 量化级界。
    a1_absmax = float((out_e1 - p_y).abs().max())
    a1_levels = int(((out_e1 * LEVEL).round() - (p_y * LEVEL).round()).abs().max())
    if model.action_backend == "mlp":
        a1_ok = bool(torch.equal(out_e1, p_y))
        if not a1_ok:
            die(f"A1 FAILED (mlp): 零初始化下整条链输出与 y 不逐位相同 "
                f"(max|diff| = {a1_absmax:.3e})")
        a1 = dict(backend="mlp", criterion="torch.equal", bit_exact=True,
                  abs_max=a1_absmax, level_max=a1_levels)
    else:
        tol = cfg.num("assert", "a1_g4d_abs_tol")
        lvl = cfg.int_("assert", "a1_g4d_max_level_diff")
        a1_ok = (a1_absmax <= tol) and (a1_levels <= lvl)
        if not a1_ok:
            die(f"A1 FAILED (g4d) 两界: max|diff| = {a1_absmax:.3e} (界 {tol:g}), "
                f"max 8bit 级差 = {a1_levels} (界 {lvl}) —— 恒等 init 没落在界内")
        a1 = dict(backend="g4d", criterion="two-bound (abs + 8bit level)",
                  bit_exact=bool(torch.equal(out_e1, p_y)),
                  abs_max=a1_absmax, abs_tol=tol,
                  level_max=a1_levels, level_tol=lvl)
    a5 = dict(depth=d_probe, nfe_per_stage=nfe_per_stage,
              n_stages=SS.stages_for(path_mode, d_probe),
              expected=SS.stages_for(path_mode, d_probe) * nfe_per_stage)

    # ---- A10 锚点契约（D-quant 裁决）：两个起点各司其职，禁混 ---------------- #
    # rollout loss 与推理起点 = uint8 资产 y（与部署一致）；
    # teacher-forced 采样 = fp32 链态 r^m。两者差一个已知量化位（≈0.5/255）。
    p_r = probe["r"].to(device)
    p_hstar = probe["hstar"].to(device)
    m_d = p_depth.clone()
    lam0 = torch.zeros(p_depth.shape[0], device=device)
    z_teacher = ST.path_states(path_mode, p_r, p_hstar, p_al, p_union,
                               p_depth, m_d, lam0)[0]
    r_at_depth = p_r[torch.arange(p_r.shape[0], device=device), p_depth]
    if not torch.equal(z_teacher, r_at_depth):
        die("A10 FAILED: teacher_anchor 不是 fp32 链态 —— λ=0/m=d 的路径状态与 "
            f"r^d 不逐位相同 (max|diff| = "
            f"{float((z_teacher - r_at_depth).abs().max()):.3e})")
    # rollout/推理起点是资产 y：A1 已证明零初始化下整条链的输出与 p_y 逐位相同，
    # 而 p_y 就是 <id>.d{k}.after.png 的采样像素，所以这条与 A1 同源，不重复断言。
    quant_gap = float((r_at_depth - p_y).abs().max())
    gap_max = cfg.num("flow", "anchor_quant_gap_max")
    if not quant_gap <= gap_max:
        die(f"A10 FAILED: 链态 r^d 与资产 y 的量化差 {quant_gap:.6e} > "
            f"[flow] anchor_quant_gap_max {gap_max:g} —— 两个锚点不再是"
            "「同一条链差一个量化位」，口径已漂移")
    a10 = dict(rollout_anchor=rollout_anchor, teacher_anchor=teacher_anchor,
               teacher_is_chain_state=True, rollout_is_asset_y=a1_ok,
               y_quant_gap_max=quant_gap, y_quant_gap_bound=gap_max,
               note="rollout/推理起点 = uint8 资产 y；teacher 采样 = fp32 链态 r^m；"
                    "两者不混用，差值是已知量化差")
    print(f"[assert] A1 step-0 恒等 OK；A5 NFE {a5['expected']} = "
          f"{a5['n_stages']}×{nfe_per_stage} OK；A6 no-LUT restore OK；"
          f"A10 锚点契约 OK (量化差 {quant_gap:.3e} <= {gap_max:g})", flush=True)

    (rap).write_text(json.dumps(dict(
        epr="EPR-051/stage0/sprf", arm=arm_name,
        tool=str(Path(__file__).resolve()), config_path=str(cfg.path),
        config=cfg.d, data_law=law, index=index_meta, encoder=cache_meta,
        frozen_sha256=frozen, arm_diff=diff_report,
        assertions=dict(A1_step0_identity=a1_ok, A1_detail=a1, A5_nfe=a5,
                        A6_no_lut_restore=a6_ok,
                        A7_heldout_guard=guard.state(),
                        A10_anchor_contract=a10),
        edit=edit_meta,
        model=dict(in_dim=in_dim, tokens=tokens, alpha_mode=alpha_mode,
                   path_mode=path_mode,
                   endpoint_probability=endpoint_probability,
                   nfe_per_stage=nfe_per_stage,
                   pointwise_in_channels=model.in_ch, params=pcount),
        schedule=dict(config_max_steps=cfg_max_steps, one_pass_steps=one_pass,
                      effective_max_steps=max_steps, warmup=warmup,
                      log_every=log_every, eval_every=eval_every,
                      ckpt_every=ckpt_every, eval_at=eval_at),
        eval=dict(select_metric=select_metric, exact_solve_source=exact_source,
                  columns_registered=columns,
                  columns_preregistered=cfg.d["eval"]["columns"],
                  columns_declared_unavailable=declared_unavailable,
                  strata=strata,
                  interval_n=len(interval_set), final_n=len(final_set)),
        invocation=dict(smoke=bool(a.smoke), limit_samples=int(a.limit_samples),
                        stop_after=int(a.stop_after),
                        sabotage_metric=a.sabotage_metric),
        started=started), ensure_ascii=False, indent=1))
    print(f"model: in_dim={in_dim} in_ch={model.in_ch} params={pcount} "
          f"budget {max_steps} (one pass {one_pass}); cadence "
          f"{log_every}/{eval_every}/{ckpt_every}; warmup {warmup}; "
          f"eval {len(interval_set)}/{len(final_set)} of {len(held)}; "
          f"run_args -> {rap.name}", flush=True)

    nw = cfg.int_("data", "num_workers")
    dl = DataLoader(ds, batch_size=batch, shuffle=True, num_workers=nw,
                    collate_fn=collate, drop_last=len(train) > batch,
                    worker_init_fn=T0.worker_init,
                    persistent_workers=nw > 0, pin_memory=True)

    steps_path = out_dir / "steps.jsonl"
    inter_path = out_dir / "metrics_intermediate.jsonl"
    if steps_path.exists():
        die(f"{steps_path} 已存在（D-20：file exists 不是噪声）；先挪走")

    def run_eval(step: int, which: str, sabotage=None) -> dict:
        subset = interval_set if which == "interval" else final_set
        r = quick_eval(model, subset, blobs, feats, cfg, device, ledger, n_steps,
                       bank_main, path_mode, nfe_per_stage, exact_source, tokens,
                       diag, shuffle_salt, sabotage=sabotage, edit_src=edit_src,
                       de00_max_px=de00_max_px)
        r["step"] = step
        r["eval_kind"] = which
        # D-cn 裁决：本臂结构性不注册的列，在 metrics 里**显式**写
        # "not_registered"（不是缺省空值/ NaN），overall 与每个分层格子都写，
        # 读表时一眼能分清「这一列在本臂不存在」与「算了但是没数」。
        for c, why in declared_unavailable.items():
            if c not in cfg.d["eval"]["columns"]:
                continue                      # 预注册全表里都没有的列，不用交代
            r["overall"][c] = dict(status="not_registered", reason=why)
            for stratum in r["by"].values():
                for cell in stratum.values():
                    cell[c] = dict(status="not_registered")
        r["columns_not_registered"] = {
            c: why for c, why in declared_unavailable.items()
            if c in cfg.d["eval"]["columns"]}
        return r

    def save_ckpt(path: Path, step: int) -> None:
        T0._atomic_save(dict(step=step, model=model.state_dict(),
                             optimizer=opt.state_dict(), rng=T0.rng_state(),
                             config_sha256=cfg.sha256,
                             frozen_sha256=frozen), path)

    step = 0
    best_key = None
    t0 = time.time()
    t_last = [time.time(), 0]          # BK 分支：ms/step 窗口计时
    epoch = 0
    interrupted = False
    stop_at = step + a.stop_after if a.stop_after else None
    history: list[dict] = []
    spot: list[dict] = []
    # A12：条件注入的**运行时**接线断言（「定义了没接线」红线）。分两拍，
    # 因为 FiLM 头零初始化 ⇒ 第 1 步 edit_enc 的梯度**恒为 0**（∂L/∂edit_lat
    # 经过全零的 film.weight），这不是没接线，是零初始化契约的必然结果：
    #   A12a  第 2 步：film.weight 的**编辑 latent 那几列**梯度非零
    #                  ⇒ 编辑 latent 确实到达了 FiLM 头；
    #   A12b  第 3 步：edit_enc 自身参数梯度非零且无 None
    #                  ⇒ 编码器确实在被训练。
    #   （本文件 = core2：判读时机由 step1/2 后移一拍到 step2/3，见循环内注释。）
    a12: dict = {}
    gen = torch.Generator(device=device)
    gen.manual_seed(seed)

    try:
        while step < max_steps and not interrupted:
            for b in dl:
                if step >= max_steps or interrupted:
                    break
                guard.assert_batch(b["sid"], f"step {step}")     # A7 逐步
                for g in opt.param_groups:
                    g["lr"] = T0.lr_at(step, lr, warmup, max_steps, schedule)
                x = b["x"].to(device, non_blocking=True)
                y = b["y"].to(device, non_blocking=True)
                al = b["alphas"].to(device, non_blocking=True)
                r = b["r"].to(device, non_blocking=True)
                hs = b["hstar"].to(device, non_blocking=True)
                depth = b["depth"].to(device, non_blocking=True)
                sc = b["s"].to(device, non_blocking=True)
                feat = T0.gather_feats(feats, b["key"], tokens, device)
                union = ST.union_mask_of(al)
                ahat = ST.compose_alpha_hat(al)
                nb = x.shape[0]

                m, lam = ST.build_stage_path_sample(depth, endpoint_probability, gen)
                z, beta, h_tgt, prev_tgt = ST.path_states(
                    path_mode, r, hs, al, union, depth, m, lam)
                ed = None if edit_src is None else b["edit"].to(device,
                                                               non_blocking=True)
                base = cond_base(model, feat, ed, amp_dt, scope)
                lid = (b["lut_ids"].to(device).gather(1, (m - 1).view(-1, 1))
                       .squeeze(1) if model.lut_id_condition == "embed" else None)
                ed_m = None if ed is None else ed[torch.arange(nb, device=device),
                                                  m - 1]
                h_hat, x0_hat = model(base, z, y, ahat, beta, al, lam, m, depth,
                                      sc, lid, ed_m)

                frac = step / max(1, max_steps)
                active = set(core)
                roll_on = weights["rollout"] != 0.0 and frac >= roll_start
                q50_on = weights["q50"] != 0.0 and frac >= q50_start
                if roll_on:
                    active.add("rollout")
                if q50_on:
                    active.add("q50")
                book.reset(active)

                # B1：λ 采样点上的一步预测是 z + (1-λ)·β_m·ĥ，目标是**真实**
                # r^(m-1)（`prev_tgt`），不是 z+β·h* 的外推点。
                u3 = union.unsqueeze(-1)
                b3 = beta.unsqueeze(-1)
                act3 = (b3 > 0)                      # B2：L_action 的像素集 = β_m>0
                lam3 = lam.view(-1, 1, 1)
                z_prev = z + (1.0 - lam3) * b3 * h_hat
                # ---- LOSSABL 分支：这是本文件与 train_sprf.py 的**唯一**差异 ----
                # 原件这四项是**无条件** book.add，所以 λ=0 会撞上 LossBook 的
                # 「λ==0 的损失项还在被计算」断言直接停机（train_sprf.py:251）。
                # 损失归因消融要求 λ=0 的项**真跳过计算**，这里按 `core`
                # （train_sprf.py:902 已按 weights[n]!=0 筛过）逐项开门 —— 与
                # rollout/q50 原有的 `if roll_on:` / `if q50_on:` 是同一条机械。
                # 张量表达式写在 if 体内，λ=0 时连 charbonnier/smooth_l1 都不建图。
                # z_prev 不在门内：它同时是 step 日志的 stage_linf8 诊断量
                # （见下方 T0.linf8(z_prev...)），不是损失项本身。
                loss = torch.zeros((), device=device, dtype=torch.float32)
                if "prev" in core:
                    loss = loss + book.add("prev", ST.masked_mean(
                        ST.charbonnier(z_prev - prev_tgt, ch_eps), u3))
                if "action" in core:
                    loss = loss + book.add("action", ST.masked_mean(
                        nn.functional.smooth_l1_loss(h_hat, h_tgt, reduction="none",
                                                     beta=sl1_beta), act3))
                if "clean" in core:
                    loss = loss + book.add("clean", ST.masked_mean(
                        ST.charbonnier(x0_hat - x, ch_eps), u3))
                # B3：bound 用 ReLU(|ĥ|-1) 的**平方**
                if "bound" in core:
                    loss = loss + book.add("bound", ST.masked_mean(
                        (h_hat.abs() - h_bound).clamp_min(0.0) ** 2, u3))
                # ---- LOSSABL 分支差异结束 ----

                if roll_on:
                    # N1：两段阶跃（不是线性 ramp）—— 前段 25%，后段 50%
                    bf = roll_bf0 if frac < roll_step else roll_bf1
                    k = max(1, int(round(bf * nb)))
                    sl = slice(0, k)
                    n_st = SS.stages_for(path_mode, int(depth[sl].max()))
                    ref_tr = reverse_reference(r[sl], depth[sl], n_st)
                    roll_pred, rmtr = SS.rollout_with_metrics(
                        model, base[sl], y[sl], ahat[sl], al[sl], union[sl],
                        depth[sl], sc[sl], path_mode, n_st, nfe_per_stage,
                        clamp_lo=clamp_lo, clamp_hi=clamp_hi,
                        return_trace=True, checkpoint=grad_ckpt,
                        edits=(None if ed is None else ed[sl]))
                    u3r = union[sl].unsqueeze(-1)
                    # B4：逐段监督 —— 每段 r̂^(m-1) 对真实 r^(m-1)，w_m ≡ 1
                    seg = [ST.masked_mean(ST.charbonnier(zj - ref_tr[:, j], ch_eps),
                                          u3r)
                           for j, zj in enumerate(rmtr["trace"], start=1)]
                    loss = loss + book.add(
                        "rollout", torch.stack(seg).mean())
                    if q50_on:
                        # N9③：e_p 是逐**像素**的 max-channel 误差
                        e_px = (roll_pred - x[sl]).abs().amax(-1)
                        sel = e_px[union[sl] > 0]
                        if sel.numel() == 0:
                            die("L_q50: union 掩膜为空，分位带无定义")
                        lo = torch.quantile(sel.float(), q50_lo).detach()
                        hi = torch.quantile(sel.float(), q50_hi).detach()
                        band = sel[(sel >= lo) & (sel <= hi)]
                        loss = loss + book.add(
                            "q50", band.mean() if band.numel() else sel.mean() * 0.0)
                book.finish()

                if not torch.isfinite(loss):
                    die(f"step {step}: loss = {float(loss)} —— NaN/Inf 停机")
                opt.zero_grad(set_to_none=True)
                loss.backward()
                # ---- A12 判读时机：step>=1（即从第 2 个优化步起）----
                # 裁决 2026-09-02(a)：A12a 六臂统一改在 step 2 评估。
                # 原因（已核实，非推测）：action 头零初始化
                # （stage_flow.py:195-196，A1「step-0 恒等」的来源）
                # ⇒ 第 1 步 ∂h_hat/∂trunk = W_action = 0，经 action 那条路
                # 到 FiLM 头的梯度恒为 0；第 1 步唯一的梯度载体是**非零初始化**
                # 的 clean 头。于是 λ_clean=0 时 A12a 在第 1 步必然读到 0.0 ——
                # 那不是「没接线」，是唯一载体被消融掉了。跳过第 1 步后，
                # action 头已更新一次（非零），A12a 对六臂都可判且判据不变。
                # 断言时机不是实验语义：损失/模型/数据/优化器逐键未动。
                if edit_src is not None and step >= 1 and len(a12) < 2:
                    # BK 分支：A12a 判读对象 = 本臂「编辑 latent 进入 backend 的首个
                    # 可学习投影」（model.bk.a12_probe()：pxfilm=film_gen[0] 编辑列，
                    # loramoe=gates[0] 编辑列，ditblk=tok_e，affhead/ff=film 编辑列，
                    # clutflow=w_mlp[0] 编辑列）；A12b = 编辑编码器。时机与 core2 同。
                    enc = model.edit_encoder
                    pr = model.bk.a12_probe()
                    if "a12a_edit_path" not in a12:
                        g = pr["param"].grad
                        gs = (None if g is None else
                              (g[:, pr["cols"]] if pr["cols"] is not None else g))
                        v = 0.0 if gs is None else float(gs.abs().sum())
                        a12["a12a_edit_path"] = dict(
                            step=step + 1, grad_abs_sum=v, probe=pr["name"],
                            arm=model.arm)
                        if not v > 0.0:
                            die("A12a FAILED: 编辑 latent 进入 backend 的首个可学习投影"
                                f"梯度为 0 ({a12['a12a_edit_path']}) —— 条件注入"
                                "「定义了没接线」")
                    elif "a12b_edit_encoder" not in a12:
                        n_none = sum(1 for q in enc.parameters() if q.grad is None)
                        v = math.sqrt(sum(float((q.grad ** 2).sum())
                                          for q in enc.parameters()
                                          if q.grad is not None))
                        a12["a12b_edit_encoder"] = dict(
                            step=step + 1, grad_l2=v, params_without_grad=n_none,
                            n_params=sum(q.numel() for q in enc.parameters()))
                        if n_none or not v > 0.0:
                            die("A12b FAILED: 编辑编码器没有拿到梯度 "
                                f"({a12['a12b_edit_encoder']}) —— 条件注入"
                                "「定义了没接线」")
                        print(f"[assert] A12 条件注入接线 OK: "
                              f"{a12['a12a_edit_path']['probe']} |grad| = "
                              f"{a12['a12a_edit_path']['grad_abs_sum']:.3e}; "
                              f"edit_enc grad_l2 = {v:.3e}", flush=True)
                gn = float(nn.utils.clip_grad_norm_(model.parameters(), clip))
                if not math.isfinite(gn):
                    die(f"step {step}: grad norm = {gn} —— NaN/Inf 停机")
                opt.step()
                step += 1

                if step % tcheck_every == 0 or step == 1:        # A2
                    j = step % nb
                    si = int(b["index"][j])
                    spot.append(teacher_spotcheck(
                        train[si], blobs[train[si]["id"]], b["idx"][j],
                        b["r"][j], b["hstar"][j], bank_main, n_steps, tdev,
                        tol, rhos, m_pick=1 + (step % n_steps)))
                    print(f"  [teacher] step {step} {spot[-1]['id']} "
                          f"G-A1={spot[-1]['g_a1_max']:.3e} "
                          f"G-A2={spot[-1]['g_a2_max']:.3e}", flush=True)

                if step % log_every == 0 or step == 1 or step == max_steps:
                    with torch.no_grad():
                        e = T0.linf8(z_prev.detach().float(), prev_tgt)
                        rec = dict(step=step, epoch=epoch, loss=float(loss),
                                   parts=dict(book.parts),
                                   active=sorted(active),
                                   stage_linf8_p50=T0.quantile(e, 0.5),
                                   stage_linf8_p95=T0.quantile(e, 0.95),
                                   lam_zero_frac=float((lam == 0).float().mean()),
                                   union_frac=float(union.mean()),
                                   lr=opt.param_groups[0]["lr"], grad_norm=gn,
                                   seconds=round(time.time() - t0, 1),
                                   ms_per_step=round(1000.0 * (time.time() - t_last[0])
                                                     / max(1, step - t_last[1]), 1),
                                   gpu_peak_gb=round(
                                       torch.cuda.max_memory_allocated() / 2 ** 30, 3),
                                   gpu_peak_reserved_gb=round(
                                       torch.cuda.max_memory_reserved() / 2 ** 30, 3))
                        t_last[0], t_last[1] = time.time(), step
                        history.append(rec)
                    T0.append_jsonl(steps_path, rec)
                    print(f"  step {step}/{max_steps} loss={rec['loss']:.6f} "
                          f"parts={ {k: round(v, 5) for k, v in rec['parts'].items()} } "
                          f"lr={rec['lr']:.2e} gn={gn:.3f} "
                          f"peak={rec['gpu_peak_gb']}GB "
                          f"reserved={rec['gpu_peak_reserved_gb']}GB "
                          f"ms/step={rec['ms_per_step']}", flush=True)
                if eval_every and step % eval_every == 0:
                    rr = run_eval(step, "interval")
                    print_eval(rr)
                    T0.append_jsonl(inter_path, {k: v for k, v in rr.items()
                                                 if k != "per_sample"})
                    key = rr["overall"]["model"]["p50"]
                    if best_key is None or key < best_key:
                        best_key = key
                        save_ckpt(out_dir / "ckpt_best.pt", step)
                if ckpt_every and step % ckpt_every == 0:
                    save_ckpt(out_dir / "ckpt_last.pt", step)
                if stop_at is not None and step >= stop_at:
                    save_ckpt(out_dir / "ckpt_last.pt", step)
                    interrupted = True
                    break
            epoch += 1
    except T0.ReadFailed as exc:
        save_ckpt(out_dir / "ckpt_last.pt", step)
        T0.append_jsonl(steps_path, dict(event="io_abort", step=step,
                                         error=str(exc)))
        die(f"read error at step {step}: {exc}；ckpt_last.pt 已落盘")

    save_ckpt(out_dir / "ckpt_last.pt", step)
    if a.skip_final_eval:
        (out_dir / "metrics_train.json").write_text(json.dumps(dict(
            epr="EPR-051/stage0/sprf", arm=arm_name, generated=time.strftime("%Y-%m-%dT%H:%M:%S%z"),
            note="--skip-final-eval：逐样本 final eval 未跑；final 数字由 batch_eval_bk.py 的 metrics_batch.json 给出",
            frozen_sha256=frozen, arm_diff=diff_report, select_metric=select_metric,
            best_select_value=best_key, n_train=len(train), n_heldout=len(held), steps=step,
            effective_max_steps=max_steps,
            assertions=dict(A1_step0_identity=a1_ok, A1_detail=a1, A5_nfe=a5, A6_no_lut_restore=a6_ok,
                            A7_heldout_guard=guard.state(), A10_anchor_contract=a10,
                            A12_edit_wiring=(a12 if edit_src is not None else "n/a"),
                            A2_teacher_spotchecks=spot),
            params=pcount, run_args=rap.name, edit=edit_meta,
            train_seconds=(history[-1]["seconds"] if history else None),
            ms_per_step_mean=(round(1000.0 * history[-1]["seconds"] / max(1, step), 1) if history else None),
            gpu_peak_gb=round(torch.cuda.max_memory_allocated() / 2 ** 30, 3),
            gpu_peak_reserved_gb=round(torch.cuda.max_memory_reserved() / 2 ** 30, 3),
            bk_arm=model.arm, train_history=history), ensure_ascii=False, indent=1))
        print(f"metrics_train -> {out_dir / 'metrics_train.json'}（final eval 跳过）", flush=True)
        return
    final = run_eval(step, "final", sabotage=a.sabotage_metric)
    print_eval(final)
    (out_dir / "metrics.json").write_text(json.dumps(dict(
        epr="EPR-051/stage0/sprf", arm=arm_name,
        generated=time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        smoke=bool(a.smoke),
        smoke_note=("冒烟数字不可读，不进任何表；只验断言全绿与损失有限值"
                    if a.smoke else None),
        frozen_sha256=frozen, arm_diff=diff_report,
        select_metric=select_metric, best_select_value=best_key,
        columns_registered=columns,
        columns_preregistered=cfg.d["eval"]["columns"],
        columns_declared_unavailable=declared_unavailable,
        n_train=len(train), n_heldout=len(held), steps=step,
        effective_max_steps=max_steps,
        assertions=dict(A1_step0_identity=a1_ok, A1_detail=a1, A5_nfe=a5,
                        A6_no_lut_restore=a6_ok,
                        A7_heldout_guard=guard.state(),
                        A10_anchor_contract=a10,
                        A12_edit_wiring=(a12 if edit_src is not None else
                                         "n/a (edit.condition = off)"),
                        A2_teacher_spotchecks=spot),
        params=pcount, run_args=rap.name, edit=edit_meta,
        wall_seconds=round(time.time() - t0, 1),
        train_seconds=(history[-1]["seconds"] if history else None),
        ms_per_step_mean=(round(1000.0 * history[-1]["seconds"] / max(1, step), 1)
                          if history else None),
        gpu_peak_gb=round(torch.cuda.max_memory_allocated() / 2 ** 30, 3),
        gpu_peak_reserved_gb=round(torch.cuda.max_memory_reserved() / 2 ** 30, 3),
        bk_arm=model.arm,
        train_history=history, heldout=final), ensure_ascii=False, indent=1))
    print(f"metrics -> {out_dir / 'metrics.json'}", flush=True)


if __name__ == "__main__":
    main()
