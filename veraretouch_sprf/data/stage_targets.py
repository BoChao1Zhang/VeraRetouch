#!/usr/bin/env python3
# 源自 experiments/prs/EPR-051_masked-restore-production/stage0/sprf/stage_targets.py @ 2026-09-06（原件未入 git；原 sha256 见 veraretouch_sprf/PROVENANCE.md），逐字复制/仅改导入与路径解析（EPR-052）。
"""EPR-051 · stage0 · arm=sprf · T2 第 1 层：阶段目标构造。

本模块是 `rebuild_forward()` 的**唯一实现**（T1 的 `preflight/sprf_preflight.py`
里那份副本按任务卡要求原地不动，两者的逐位等价由
`check_rebuild_equiv.py` 一次性断言）。除此之外本模块提供：

  * `build_intermediates`  逐样本重建 r^(0..d) 与逐阶段 h*，**在采样像素子集上**算。
    LUT 求值 (`apply_lut`) 与渲染律 (`mix_alpha`) 都是逐像素算子，所以
    「先取子集再走链」与「先走全图链再取子集」逐位相同（`check_rebuild_equiv.py`
    的 subset 断言就是这一条）。
  * `build_stage_path_sample`  m ~ U{1..d}；λ ~ ep·δ0 + (1-ep)·U(0,1)。
  * `build_lut_action_target`  h* = r^(m-1) − L_m(r^(m-1))，直接调 LUT 构造，
    **不除以 β**。
  * `path_states`  三种 path_mode（chain / linear / one_shot）下的 (z, β_m, h*_m)。
  * `assert_heldout_isolation`  目标构造 key 集与 heldout_ids 的交集断言。

数学口径：
  前向一步        r^m = mix_alpha(r^(m-1), L_m(r^(m-1)), β_m)
  一步恒等(G-A2)  r^m + β_m·h*_m = r^(m-1)，h*_m = r^(m-1) − L_m(r^(m-1))
  折线(G-A1)      A(x; β_1..ρβ_m, 0..) = (1-ρ)·r^(m-1) + ρ·r^m
  路径状态        z(m, λ) = (1-λ)·r^m + λ·r^(m-1)，λ=0 即推理时真实见到的状态
  条件速度        v = β_m·h*_m = r^(m-1) − r^m（沿该段为常量）
"""
from __future__ import annotations

import hashlib
import json
import os
import sys
from collections import OrderedDict
from pathlib import Path

import numpy as np
import torch

from veraretouch_sprf import _paths as _P  # EPR-052：替代原 sys.path 注入块
_P.ensure_sys_path()
REPO = _P.REPO

import epr050_build_degradation as B            # noqa: E402  前向链原件
from q3vl.whatb.lutdata import mix_alpha        # noqa: E402  渲染律本体

PATH_MODES = ("chain", "linear", "one_shot")
ALPHA_MODES = ("alpha_hat", "alpha_m", "alpha_full")


def die(msg: str):
    raise SystemExit(f"sprf.stage_targets: {msg}")


def _require_build_config_loaded() -> None:
    """`B.chunked` 读 `B.PIX_CHUNK`，它只有 `B.load_config()` 之后才存在。"""
    if B.PIX_CHUNK is None or B.N_STEPS is None or B.STEP_KIND is None:
        die("epr050_build_degradation 的全局量未装载：先调 "
            "train_stage0.bind_build_config()（PIX_CHUNK / N_STEPS / STEP_KIND）")


# --------------------------------------------------------------------------- #
# LUT 体：每进程一份 npz 句柄 + 小 LRU
# --------------------------------------------------------------------------- #
class LutVolumes:
    """`lut_id -> (1,3,D_b,D_g,D_r)`，与 `B.apply_lut` 的入参约定一致。

    bank 里存的是 `grid[b,g,r,c]`（q3vl/whatb/lutdata.py:94-96），所以
    `permute(0,4,1,2,3)` 这一步与 preflight / montage 的写法逐字相同。
    """

    def __init__(self, bank_dir: str | Path, cache_size: int):
        self.path = Path(bank_dir) / "luts.npz"
        if not self.path.is_file():
            die(f"LUT bank 不存在: {self.path}")
        self.cache_size = int(cache_size)
        if self.cache_size < 1:
            die(f"lut_cache_size = {self.cache_size} 必须 >= 1")
        self._npz = None
        self._pid = None
        self._lru: OrderedDict = OrderedDict()

    @property
    def npz(self):
        """每**进程**各自打开。

        `np.load` 的 NpzFile 是一个 zip 句柄，父子进程共享同一个 fd 就共享同一个
        文件偏移；DataLoader fork 出的 worker 与父进程并发读会互相踩偏移，症状是
        `BadZipFile: Bad CRC-32`（实测踩到过）。所以句柄按 pid 失效重开，
        而不是「只开一次」。
        """
        pid = os.getpid()
        if self._npz is None or self._pid != pid:
            self._npz = np.load(self.path, allow_pickle=False)
            self._pid = pid
            self._lru.clear()
        return self._npz

    def get(self, name: str, device, dtype=torch.float32) -> torch.Tensor:
        if self._pid != os.getpid():               # fork 之后先让句柄与 LRU 失效
            self.npz
        key = (name, str(device), str(dtype))
        v = self._lru.get(key)
        if v is not None:
            self._lru.move_to_end(key)
            return v
        arr = self.npz[name]
        v = (torch.from_numpy(arr[None]).to(device=device, dtype=dtype)
             .permute(0, 4, 1, 2, 3).contiguous())
        self._lru[key] = v
        while len(self._lru) > self.cache_size:
            self._lru.popitem(last=False)
        return v


# --------------------------------------------------------------------------- #
# 链重建：本仓唯一实现（preflight 的同名副本仅供逐位等价断言）
# --------------------------------------------------------------------------- #
def rebuild_forward(row: dict, npz, dev: str, max_side: int):
    """`epr050_recovery_montage.rebuild` 的同一机械，只保留前向部分。

    返回 `(x0, alphas, vols, ys, (h, w))`：`alphas[k]` 是 (1,P,1) 的 β 场
    （已乘全局强度 s），`ys[k]` 是 (1,P,3) 的 r^k。
    """
    gid = row["id"]
    src = B.open_source(row["source_path"], max_side)
    h, w = src.shape[:2]
    if [h, w] != list(row["size"]):
        die(f"{gid}: size drift {[h, w]} vs 记录 {row['size']}")
    x0 = torch.from_numpy(src.astype(np.float32) / 255.0).to(dev)

    m = row["mask"]
    bands, (t_lo, t_hi) = B.lum_bands(x0, m["q_lo"], m["q_hi"], m["width"])
    if abs(t_lo - m["t_lo"]) > 1e-6 or abs(t_hi - m["t_hi"]) > 1e-6:
        die(f"{gid}: 亮度阈漂移 ({t_lo}, {t_hi}) vs 记录 ({m['t_lo']}, {m['t_hi']})")
    hard = None
    if m["geom"] in ("subject", "semantic"):
        subj = B.load_subject(Path(row["subject_png"]), h, w, dev)
        hard = ((subj > 0.5).float().cpu().numpy() if m["geom"] == "semantic"
                else subj.cpu().numpy())
    m_geo = B.geom_alpha_from_row(row, h, w, dev, hard)

    by_kind = dict(lum_high=bands["lum_high"], lum_mid=bands["lum_mid"],
                   lum_shadow=bands["lum_shadow"],
                   **{"global": torch.ones((h, w), device=dev)}, geom=m_geo)
    if "hue" in B.STEP_KIND:
        by_kind["hue"] = B.hue_mask(x0)
    fields = [by_kind[k] for k in B.STEP_KIND]
    if [s["kind"] for s in row["steps"]] != list(B.STEP_KIND):
        die(f"{gid}: 配置步序 {list(B.STEP_KIND)} 与记录链 "
            f"{[s['kind'] for s in row['steps']]} 不符")
    s_glob = row["calib"]["s"]
    if s_glob != 1.0:
        fields = [f * s_glob for f in fields]
    alphas = [f.reshape(1, -1, 1).contiguous() for f in fields]

    vols = [torch.from_numpy(npz[n][None]).to(dev, torch.float32)
            .permute(0, 4, 1, 2, 3).contiguous() for n in row["luts"]]

    xf = x0.reshape(1, -1, 3)
    ys = [xf]
    for k in range(B.N_STEPS):
        prev = ys[-1]
        ys.append(mix_alpha(prev, B.chunked(
            lambda t, v=vols[k]: B.apply_lut(v, t), prev), alphas[k]))
    return x0, alphas, vols, ys, (h, w)


def chain_with(alphas: list[torch.Tensor], vols, x: torch.Tensor) -> torch.Tensor:
    """A(x; alphas)：完整 N_STEPS 前向，尾部 α==0 的步照样走（不做等价简化）。"""
    y = x
    for k in range(B.N_STEPS):
        y = mix_alpha(y, B.chunked(lambda t, v=vols[k]: B.apply_lut(v, t), y),
                      alphas[k])
    return y


# --------------------------------------------------------------------------- #
# 目标构造（在采样像素子集上）
# --------------------------------------------------------------------------- #
def apply_lut_px(vol: torch.Tensor, x_px: torch.Tensor) -> torch.Tensor:
    """`(P,3) -> (P,3)`，走 `B.chunked(B.apply_lut)`，与全图链同一函数。"""
    _require_build_config_loaded()
    return B.chunked(lambda t, v=vol: B.apply_lut(v, t), x_px.unsqueeze(0))[0]


def build_lut_action_target(r_prev: torch.Tensor, vol: torch.Tensor) -> torch.Tensor:
    """h* = r^(m-1) − L_m(r^(m-1))。直接调 LUT 构造，**不除以 β**。"""
    return r_prev - apply_lut_px(vol, r_prev)


def build_intermediates(x_px: torch.Tensor, alphas_px: torch.Tensor,
                        lut_ids: list[str], bank: LutVolumes,
                        device=None) -> tuple[torch.Tensor, torch.Tensor]:
    """逐样本重建中间态。

    参数
      `x_px`      (P,3) = r^0（`<id>.src.png` 的采样像素）
      `alphas_px` (K,P) = 已乘 s、**已乘 depth 掩膜**的 β 场采样像素
      `lut_ids`   长度 K 的 lut_id 列表（`row["luts"]`）
    返回
      `r`     (K+1,P,3)  r^0..r^K
      `hstar` (K,P,3)    h*_m = r^(m-1) − L_m(r^(m-1))
    """
    _require_build_config_loaded()
    if alphas_px.shape[0] != len(lut_ids):
        die(f"alphas ({alphas_px.shape[0]}) 与 luts ({len(lut_ids)}) 步数不符")
    dev = x_px.device if device is None else device
    cur = x_px.to(dev)
    r = [cur]
    hs = []
    for k, name in enumerate(lut_ids):
        vol = bank.get(name, dev, cur.dtype)
        lo = apply_lut_px(vol, cur)
        hs.append(cur - lo)
        cur = mix_alpha(cur, lo, alphas_px[k].to(dev).unsqueeze(-1))
        r.append(cur)
    return torch.stack(r), torch.stack(hs)


def build_stage_path_sample(depth: torch.Tensor, endpoint_probability: float,
                            generator: torch.Generator | None = None
                            ) -> tuple[torch.Tensor, torch.Tensor]:
    """m ~ U{1..d}（逐样本 d）；λ ~ ep·δ0 + (1-ep)·U(0,1)。

    `depth` (B,) long。返回 `(m (B,) long ∈ 1..d, lam (B,) float ∈ [0,1))`。
    """
    if not 0.0 <= float(endpoint_probability) <= 1.0:
        die(f"endpoint_probability = {endpoint_probability} 不在 [0,1]")
    dev = depth.device
    n = depth.shape[0]
    u = torch.rand(n, generator=generator, device=dev)
    m = (u * depth.to(u.dtype)).floor().long() + 1
    m = torch.minimum(m, depth)                    # u==1-eps 的数值兜底
    e = torch.rand(n, generator=generator, device=dev) < float(endpoint_probability)
    lam = torch.where(e, torch.zeros(n, device=dev),
                      torch.rand(n, generator=generator, device=dev))
    return m, lam


def _gather_stage(t: torch.Tensor, idx: torch.Tensor) -> torch.Tensor:
    """`t` (B,S,P,C) 沿 S 轴按逐样本 `idx` (B,) 取一层 -> (B,P,C)。"""
    b, s = t.shape[0], t.shape[1]
    if int(idx.min()) < 0 or int(idx.max()) >= s:
        die(f"stage index {int(idx.min())}..{int(idx.max())} 越出 0..{s-1}")
    return t[torch.arange(b, device=t.device), idx]


def path_states(path_mode: str, r: torch.Tensor, hstar: torch.Tensor,
                betas: torch.Tensor, union: torch.Tensor, depth: torch.Tensor,
                m: torch.Tensor, lam: torch.Tensor):
    """三种预注册路径下的 `(z, beta_m, hstar_m, r_prev)`。

      chain      z = (1-λ)·r^m + λ·r^(m-1)；β_m = 真实 β 场；h* = LUT 目标
      linear     P_k = y + ((d-k)/d)(x-y)；z = (1-λ)P_m + λP_(m-1)；
                 β_m = union/d；h* = x − y
      one_shot   z = (1-λ)·y + λ·x；β_m = union；h* = x − y（单步）

    形状：`r` (B,K+1,P,3)、`hstar` (B,K,P,3)、`betas` (B,K,P)、`union` (B,P)、
    `depth`/`m` (B,)、`lam` (B,)。返回四元组
    `(z, beta_m, h, target_prev)`：

      `z`            λ 采样点上的状态
      `beta_m`       该阶段更新用的 β 场（oracle）
      `h`            该阶段的 action 目标 h*
      `target_prev`  **该段的真实上游顶点 r^(m-1)**（chain）/ P_(m-1)（linear）/
                     x（one_shot）。它是 L_prev 的回归目标，**不是**
                     `z + β·h*` 那个外推点 —— 后者只在 λ=0 时才等于 r^(m-1)。

    预注册的一步预测式（B1 修复）：`pred = z + (1-λ)·β_m·ĥ`，因为
        z + (1-λ)·β·h* = (1-λ)(r^m + β·h*) + λ·r^(m-1) = r^(m-1)
    对任意 λ 都成立（β·h* = r^(m-1) − r^m 是该段的常量条件速度）。
    """
    if path_mode not in PATH_MODES:
        die(f"path_mode {path_mode!r} 不在 {PATH_MODES}")
    lam3 = lam.view(-1, 1, 1)
    if path_mode == "chain":
        r_m = _gather_stage(r, m)
        r_pm = _gather_stage(r, m - 1)
        z = (1.0 - lam3) * r_m + lam3 * r_pm
        beta_m = _gather_stage(betas.unsqueeze(-1), m - 1).squeeze(-1)
        h = _gather_stage(hstar, m - 1)
        return z, beta_m, h, r_pm
    x = r[:, 0]
    y = _gather_stage(r, depth)
    h = x - y
    if path_mode == "one_shot":
        z = (1.0 - lam3) * y + lam3 * x
        beta_m = union
        return z, beta_m, h, x
    d = depth.to(r.dtype).view(-1, 1, 1)
    t_m = (d - m.to(r.dtype).view(-1, 1, 1)) / d
    t_pm = (d - (m - 1).to(r.dtype).view(-1, 1, 1)) / d
    p_m = y + t_m * h
    p_pm = y + t_pm * h
    z = (1.0 - lam3) * p_m + lam3 * p_pm
    beta_m = union / d.view(-1, 1)
    return z, beta_m, h, p_pm


def union_mask_of(betas: torch.Tensor) -> torch.Tensor:
    """任一活跃阶段 β>0 的像素 -> 1.0；(B,K,P) -> (B,P)。"""
    return (betas.amax(dim=1) > 0).to(betas.dtype)


def compose_alpha_hat(betas: torch.Tensor) -> torch.Tensor:
    """α̂ = 1 − Π_k (1 − β_k)，即 train_stage0 的 ``prod`` 口径（β 已含 depth 掩膜）。"""
    return 1.0 - torch.prod(1.0 - betas, dim=1)


# --------------------------------------------------------------------------- #
# 损失原语（B3：形式为预注册项，参数全部来自 TOML）
# --------------------------------------------------------------------------- #
def charbonnier(d: torch.Tensor, eps: float) -> torch.Tensor:
    """`sqrt(d² + ε²)`。ε 由 `[loss] charbonnier_eps` 给，代码里无默认值。"""
    return torch.sqrt(d * d + float(eps) * float(eps))


def masked_mean(v: torch.Tensor, mask3: torch.Tensor) -> torch.Tensor:
    """`v` 与 `mask3` 同形（或可广播）；按 mask 内元素数取均值，空集给 0。"""
    m = mask3.to(v.dtype)
    return (v * m).sum() / m.sum().clamp_min(1.0)


def shuffle_partner_index(ids: list[str], salt: str, source_of) -> list[int]:
    """B5：**跨 source_id** 的确定性错位置换。

    Δ_shuffle 的负控制必须保证「拿到的条件不是同一张源图（任何 depth）的」，
    否则同源不同 depth 的条件几乎等价，Δ 会被系统性低估。做法：
    按 `sha1(salt + id)` 定序，再为每个样本取该序上**下一个 source_id 不同**的样本。
    全评测子集只有一个 source_id 时无法构造，直接报错而不是退化成自配。
    """
    n = len(ids)
    order = sorted(range(n), key=lambda i: hashlib.sha1(
        f"{salt}{ids[i]}".encode()).hexdigest())
    src = [source_of(x) for x in ids]
    if len(set(src)) < 2:
        die(f"Δ_shuffle 需要至少两个不同 source_id，当前只有 {set(src)}")
    partner = [None] * n
    for pos, i in enumerate(order):
        for step in range(1, n + 1):
            j = order[(pos + step) % n]
            if src[j] != src[i]:
                partner[i] = j
                break
        if partner[i] is None:
            die(f"{ids[i]}: 找不到跨 source 的错位对象")
    return partner


# --------------------------------------------------------------------------- #
# 诊断：FD 雅可比与 blend 谱（口径抄 preflight，= epr050_build_degradation::_gn）
# --------------------------------------------------------------------------- #
def fd_jacobian(vol: torch.Tensor, u: torch.Tensor, n_grid: int, lo: float,
                hi: float, min_h: float) -> torch.Tensor:
    """`u` (1,M,3) -> (M,3,3)。步长 d = 0.5/(n_grid-1)，正负扰动各自 clamp 到
    [lo,hi]，再按**实现距离**（不是 2d）相除，在色域边界自动退化成单边差分。
    与 `preflight/sprf_preflight.py::fd_jacobian` 同一口径。"""
    _require_build_config_loaded()
    d = 0.5 / (n_grid - 1)
    cols = []
    for ax in range(3):
        e = torch.zeros(3, device=u.device, dtype=u.dtype)
        e[ax] = d
        up, um = (u + e).clamp(lo, hi), (u - e).clamp(lo, hi)
        hh = (up[..., ax] - um[..., ax]).unsqueeze(-1).clamp_min(min_h)
        cols.append((B.chunked(lambda t, v=vol: B.apply_lut(v, t), up)
                     - B.chunked(lambda t, v=vol: B.apply_lut(v, t), um)) / hh)
    return torch.stack(cols, dim=-1)[0]


def blend_spectrum(J: torch.Tensor, beta: torch.Tensor, chunk: int):
    """`(1-β)I + βJ` 的逐像素 σ_min / σ_max。"""
    I3 = torch.eye(3, device=J.device, dtype=J.dtype)
    b = beta.reshape(-1, 1, 1)
    M = (1.0 - b) * I3 + b * J
    smin = torch.empty(M.shape[0], device=J.device, dtype=J.dtype)
    smax = torch.empty_like(smin)
    for i in range(0, M.shape[0], int(chunk)):
        sv = torch.linalg.svdvals(M[i:i + int(chunk)])
        smax[i:i + int(chunk)] = sv[:, 0]
        smin[i:i + int(chunk)] = sv[:, -1]
    return smin, smax


def condition_number(J: torch.Tensor, beta: torch.Tensor, chunk: int,
                     sigma_floor: float):
    """κ₂ 与 valid 掩膜（B1：σ_min < floor 的像素不做 clamp 兜底，显式排除）。"""
    smin, smax = blend_spectrum(J, beta, chunk)
    valid = smin >= float(sigma_floor)
    kappa = torch.where(valid, smax / torch.where(valid, smin,
                                                  torch.ones_like(smin)),
                        torch.full_like(smin, float("nan")))
    return kappa, valid, smin


# --------------------------------------------------------------------------- #
# LUT 身份 token（仅诊断臂 lut_id_condition="embed" 用；生产 restore() 禁入）
# --------------------------------------------------------------------------- #
def lut_id_token(name: str, vocab: int) -> int:
    """`lut_id -> 1..vocab-1`；0 保留给 null/unknown token（no-ID 对照）。"""
    if vocab < 2:
        die(f"lut_id_vocab = {vocab} 必须 >= 2（0 号是 null token）")
    import hashlib as _h
    return int(_h.sha1(name.encode()).hexdigest()[:8], 16) % (int(vocab) - 1) + 1


# --------------------------------------------------------------------------- #
# held-out 隔离（逐步断言）
# --------------------------------------------------------------------------- #
class HeldoutGuard:
    """目标构造只允许发生在 train 侧：逐步断言 key 集与 heldout_ids 交集为空。"""

    def __init__(self, heldout_ids: set[str]):
        self.heldout = set(heldout_ids)
        self.checked_batches = 0
        self.checked_ids = 0

    def assert_batch(self, ids, where: str) -> None:
        bad = sorted(set(ids) & self.heldout)
        if bad:
            die(f"{where}: held-out id 进入目标构造 {bad[:5]}（共 {len(bad)}）—— "
                "V_where/V_what/T_final 永不进训练")
        self.checked_batches += 1
        self.checked_ids += len(list(ids))

    def state(self) -> dict:
        return dict(n_heldout_ids=len(self.heldout),
                    checked_batches=self.checked_batches,
                    checked_ids=self.checked_ids)


def assert_heldout_isolation(ids, heldout_ids, where: str = "target build") -> None:
    HeldoutGuard(heldout_ids).assert_batch(ids, where)


# --------------------------------------------------------------------------- #
# compact row：train_stage0.compact_row + luts/grids（目标构造需要 LUT 身份）
# --------------------------------------------------------------------------- #
def install_compact_row(t0_module) -> None:
    """把 `train_stage0.compact_row` 换成带 `luts`/`grids` 的版本。

    `train_stage0.compact_row` 只留 α 重建要读的字段，SPRF 的目标构造还要
    LUT 身份。这里在 `load_shards()` 之前替换，并对同一行断言新版是旧版的
    严格超集（旧字段逐键相同），避免「悄悄换了数据口径」。
    """
    if getattr(t0_module, "_SPRF_COMPACT_INSTALLED", False):
        return
    orig = t0_module.compact_row

    def compact_row_sprf(r: dict) -> bytes:
        rec = json.loads(orig(r))
        for k in ("luts", "grids"):
            if k not in r:
                die(f"{r.get('id')}: journal 行缺 {k!r}，SPRF 目标构造无法定位 LUT")
        rec["luts"] = list(r["luts"])
        rec["grids"] = [int(g) for g in r["grids"]]
        rec["depths"] = [int(e["depth"]) for e in r.get("emits", [])]
        base = json.loads(orig(r))
        if any(rec[k] != v for k, v in base.items()):
            die(f"{r.get('id')}: compact_row 超集断言失败（旧字段被改写）")
        return json.dumps(rec, ensure_ascii=False).encode()

    t0_module.compact_row = compact_row_sprf
    t0_module._SPRF_COMPACT_INSTALLED = True
