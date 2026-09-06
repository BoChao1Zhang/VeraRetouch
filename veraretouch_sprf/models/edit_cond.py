#!/usr/bin/env python3
# 源自 experiments/prs/EPR-051_masked-restore-production/stage0/sprf/edit_cond.py @ 2026-09-06（原件未入 git；原 sha256 见 veraretouch_sprf/PROVENANCE.md），逐字复制/仅改导入与路径解析（EPR-052）。
"""EPR-051 · stage0 · sprf · X-COND：显式编辑信息的两个 oracle 来源。

假设：模型只看退化图不知道「编辑是什么」，信息瓶颈在编辑操作不可见。本模块把
编辑信息用 oracle 方式显式喂进去，两档：

  `inv_lut`  (契约 `oracle_lut`)  该阶段真实 LUT 的**逆表**。L_m 是池属性（与
             图像 / β / s 无关），所以逆表按 lut_id 一次性预计算
             （`precompute_inv_lut.py`），训练时按 journal 的 lut_id 查表，
             **不逐图求逆**。
  `ref_pair` (契约 `oracle_ref`)  该阶段真实 LUT 的**编辑演示对**。从 train 侧
             另抽一张自然图（sha1 选，禁同源），直方图匹配到本样本原图得
             ref_before，对它施加该阶段真实 LUT（β≡1 全图）得 ref_after。
             模型见演示对，不见 LUT 本体 —— 比 `inv_lut` 弱一档的 oracle。

两档的描述子**逐字同形**：`(grid³, 4)`，通道 0..2 是位移、通道 3 是有效/占用位。
只有来源不同，所以 J-C3「从演示对提取编辑 vs 直接给编辑」的差是可读的
（编码器结构与参数量在两臂上完全一致）。

存储序与 `precompute_inv_lut.output_grid` 一致：flat = (i_b·G + i_g)·G + i_r，
通道三元组是 RGB —— 与 bank 体的 `[b, g, r, c]` 同一约定。
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import torch

from veraretouch_sprf.data import stage_targets as ST
from veraretouch_sprf.data.stage_targets import die
from q3vl.whatb.lutdata import mix_alpha

CHANNELS = 4


# --------------------------------------------------------------------------- #
# 公共：把 (before, after) 的逐像素位移聚到 grid³ 个格子
# --------------------------------------------------------------------------- #
def bin_index(c: torch.Tensor, grid: int) -> torch.Tensor:
    """`c` (P,3) RGB in [0,1] -> (P,) 扁平格号，序与逆表逐字相同。"""
    q = (c.clamp(0.0, 1.0) * int(grid)).floor().long().clamp_(0, int(grid) - 1)
    r, g, b = q[:, 0], q[:, 1], q[:, 2]
    return (b * int(grid) + g) * int(grid) + r


def bin_displacement(before: torch.Tensor, after: torch.Tensor,
                     grid: int) -> torch.Tensor:
    """演示对 -> `(grid³, 4)` 描述子：格内平均位移 + 占用位。

    占用位取 {0,1}（与逆表的 `valid` 通道同语义、同取值域），格内**逐像素计数**
    另行统计后落盘，不藏在这个通道里。
    """
    n = int(grid) ** 3
    idx = bin_index(before, grid)
    acc = torch.zeros((n, 3), dtype=torch.float32)
    cnt = torch.zeros((n,), dtype=torch.float32)
    acc.index_add_(0, idx, (after - before).to(torch.float32))
    cnt.index_add_(0, idx, torch.ones_like(idx, dtype=torch.float32))
    occ = (cnt > 0)
    mean = torch.where(occ.unsqueeze(-1), acc / cnt.clamp_min(1.0).unsqueeze(-1),
                       torch.zeros_like(acc))
    return torch.cat([mean, occ.to(torch.float32).unsqueeze(-1)], dim=-1)


# --------------------------------------------------------------------------- #
# inv_lut：预计算逆表 + 指纹校验
# --------------------------------------------------------------------------- #
class InvLutSource:
    """按 lut_id 查预计算逆表。DataLoader 侧只递**行号**（(K,) long），
    真正的 (K, grid³·4) 描述子在 GPU 上由 `SprfModel.edit_descriptor` 查表得到 ——
    否则每个 batch 要在 worker↔主进程之间搬 64×6×19652×4B ≈ 30 MiB。
    """

    def __init__(self, cache_dir: str | Path, grid: int):
        d = Path(cache_dir)
        ix = d / "index.json"
        tb = d / "table.npy"
        for p in (ix, tb):
            if not p.is_file():
                die(f"逆表缓存缺文件 {p}（先跑 precompute_inv_lut.py）")
        self.dir = d
        self.index = json.loads(ix.read_text())
        self.fingerprint = self.index["fingerprint"]
        self.totals = self.index["totals"]
        if int(self.fingerprint["grid"]) != int(grid):
            die(f"逆表缓存的 grid {self.fingerprint['grid']} != [edit] grid {grid}")
        if int(self.fingerprint["channels"]) != CHANNELS:
            die(f"逆表通道数 {self.fingerprint['channels']} != {CHANNELS}")
        self.names = list(self.index["names"])
        self.row_of = {n: i for i, n in enumerate(self.names)}
        self.table_path = tb

    def load_table(self) -> torch.Tensor:
        arr = np.load(self.table_path)
        got = hashlib.sha256(np.ascontiguousarray(arr).tobytes()).hexdigest()
        if got != self.fingerprint["table_sha256"]:
            die(f"逆表 table.npy 的 sha256 {got[:12]} != index.json 记录的 "
                f"{self.fingerprint['table_sha256'][:12]} —— 缓存被改过")
        return torch.from_numpy(arr)

    def assert_fingerprint(self, build_tool_sha: str, build_cfg_sha: str,
                           bank_dir: str) -> dict:
        """缓存不是「算过就能用」：求逆机械 / 数据律 / bank 任一漂了即停机。"""
        fp = self.fingerprint
        if fp["build_tool_sha256"] != build_tool_sha:
            die(f"逆表是用 build 工具 {fp['build_tool_sha256'][:12]} 求的，本次是 "
                f"{build_tool_sha[:12]} —— 求逆机械漂了")
        if fp["build_config_sha256"] != build_cfg_sha:
            die(f"逆表是在 build config {fp['build_config_sha256'][:12]} 下求的，"
                f"本次是 {build_cfg_sha[:12]} —— INV_ITERS/INV_EARLY 口径可能漂了")
        if Path(fp["bank_dir"]) != Path(bank_dir):
            die(f"逆表的 bank {fp['bank_dir']} != [data] lut_bank_dir {bank_dir}")
        meta = Path(bank_dir) / "luts_meta.json"
        got = hashlib.sha256(meta.read_bytes()).hexdigest()
        if got != fp["luts_meta_sha256"]:
            die(f"bank 的 luts_meta.json sha {got[:12]} != 逆表记录的 "
                f"{fp['luts_meta_sha256'][:12]} —— LUT 池变了")
        return dict(cache_dir=str(self.dir), fingerprint=fp, totals=self.totals)

    def for_row(self, row: dict, x0: torch.Tensor, mask: torch.Tensor
                ) -> torch.Tensor:
        """journal 行 -> (K,) long 的逆表行号（`x0`/`mask` 本档用不到）。"""
        out = []
        for n in row["luts"]:
            i = self.row_of.get(n)
            if i is None:
                die(f"lut_id {n!r} 不在逆表缓存里（缓存覆盖 {len(self.names)} 支）")
            out.append(i)
        return torch.tensor(out, dtype=torch.long)


# --------------------------------------------------------------------------- #
# ref_pair：编辑演示对
# --------------------------------------------------------------------------- #
def hist_match_u8(ref_u8: np.ndarray, tgt01: np.ndarray) -> np.ndarray:
    """通道独立的直方图匹配：把 `ref_u8` (H,W,3) uint8 的每通道 CDF 匹配到
    `tgt01` (·,3) float [0,1] 的同通道 CDF。返回 (H,W,3) float32 [0,1]。

    两侧都在 256 个 8bit 级上建 CDF（资产本来就是 uint8），映射表由
    `np.interp(src_cdf, tgt_cdf, levels)` 给出 —— 纯函数，逐次调用逐位可复现。
    """
    out = np.empty(ref_u8.shape, dtype=np.float32)
    lv = np.arange(256, dtype=np.float64)
    tq = np.clip(np.rint(tgt01.reshape(-1, 3) * 255.0), 0, 255).astype(np.uint8)
    for c in range(3):
        sc = np.bincount(ref_u8[..., c].ravel(), minlength=256).cumsum()
        sc = sc / float(sc[-1])
        tc = np.bincount(tq[:, c], minlength=256).cumsum()
        tc = tc / float(tc[-1])
        lut = np.interp(sc, tc, lv)
        out[..., c] = lut[ref_u8[..., c]].astype(np.float32) / 255.0
    return out


class RefPairSource:
    """从固定参考池抽一张自然图，直方图匹配后逐阶段施加真实 LUT，出演示对描述子。

    参考池在**主进程**建好（uint8，(N, S, S, 3)），DataLoader `fork` 出的 worker
    共享同一份内存，不重复读盘。选谁是 `sha1(salt + sample_id)` 定序后向前找第一个
    **跨 source_id** 的池成员 —— 与 Δ_shuffle 的 `cross_source_sha1_next` 同一条
    规矩（同源的参考图与本样本几乎等价，会把演示对的信息量系统性高估）。
    """

    def __init__(self, pool_img: np.ndarray, pool_src: list[str], salt: str,
                 grid: int, bank_dir: str, lut_cache: int):
        if pool_img.dtype != np.uint8 or pool_img.ndim != 4:
            die(f"参考池必须是 (N,S,S,3) uint8，收到 {pool_img.dtype} {pool_img.shape}")
        if len(set(pool_src)) < 2:
            die(f"参考池只有 {len(set(pool_src))} 个 source_id，无法保证跨源选取")
        self.pool = pool_img
        self.src = list(pool_src)
        self.salt = str(salt)
        self.grid = int(grid)
        self.bank_dir = bank_dir
        self.lut_cache = int(lut_cache)
        self._bank = None

    @property
    def bank(self) -> ST.LutVolumes:
        if self._bank is None:                    # 每 worker 进程一份（pid 失效）
            self._bank = ST.LutVolumes(self.bank_dir, self.lut_cache)
        return self._bank

    def pick(self, sample_id: str, source_id: str) -> int:
        n = len(self.src)
        h = int(hashlib.sha1(f"{self.salt}{sample_id}".encode()).hexdigest()[:16], 16)
        for k in range(n):
            j = (h + k) % n
            if self.src[j] != source_id:
                return j
        die(f"{sample_id}: 参考池里找不到跨 source 的参考图")

    def for_row(self, row: dict, x0: torch.Tensor, mask: torch.Tensor
                ) -> torch.Tensor:
        """journal 行 -> (K, grid³·4) float32 描述子。"""
        from veraretouch_sprf.data import train_stage0 as T0
        j = self.pick(str(row["id"]), T0.source_id_of(str(row["id"])))
        before = torch.from_numpy(
            hist_match_u8(self.pool[j], x0.reshape(-1, 3).numpy())).reshape(-1, 3)
        one = torch.ones((before.shape[0], 1), dtype=before.dtype)
        out = []
        for name in row["luts"]:
            vol = self.bank.get(name, before.device, before.dtype)
            # β ≡ 1 全图：演示的是**编辑本身**，不带蒙版。mix_alpha 在 alpha==1
            # 上是 snap 分支（self_check 断言过），所以这就是 L(before)。
            after = mix_alpha(before, ST.apply_lut_px(vol, before), one)
            out.append(bin_displacement(before, after, self.grid).reshape(-1))
        return torch.stack(out)


def build_ref_pool(train_samples, blobs, n_ref: int, size: int
                   ) -> tuple[np.ndarray, list[str], dict]:
    """从 **train 侧** 按 source_id 去重、`id` 定序等距抽 `n_ref` 张，缩到 size×size。

    读的是分片资产 `<id>.src.png`（`load_pair` 的原件；它与
    `open_source(source_path, max_side)` 逐位相同，见 train_stage0:525），
    所以参考图与训练看到的源图是同一份像素，不引入第二条解码路径。
    held-out 样本永不进池（A7 的口径：train 侧才是可用信息）。
    """
    import torch.nn.functional as F
    from veraretouch_sprf.data import train_stage0 as T0
    seen, picks = set(), []
    for s in train_samples:                      # 已按 (id, depth) 排好序
        sid = T0.source_id_of(s["id"])
        if sid in seen:
            continue
        seen.add(sid)
        picks.append((sid, s))
    if len(picks) < 2:
        die(f"train 侧只有 {len(picks)} 个 source，参考池建不起来")
    n = min(int(n_ref), len(picks))
    step = max(1, len(picks) // n)
    picks = picks[::step][:n]
    imgs, srcs = [], []
    for sid, s in picks:
        row = json.loads(blobs[s["id"]])
        x0, _ = T0.load_pair(s["shard"], row, s["after_asset"])   # (H,W,3) [0,1]
        t = x0.permute(2, 0, 1)[None] * 255.0
        rs = F.interpolate(t, size=(int(size), int(size)), mode="bilinear",
                           align_corners=False, antialias=True)
        imgs.append(rs[0].permute(1, 2, 0).clamp(0, 255).round()
                    .to(torch.uint8).numpy())
        srcs.append(sid)
    pool = np.stack(imgs)
    return pool, srcs, dict(n_ref=len(srcs), size=int(size),
                            n_source_candidates=len(seen),
                            pool_bytes=int(pool.nbytes),
                            pool_sha256=hashlib.sha256(pool.tobytes()).hexdigest())


# --------------------------------------------------------------------------- #
# ΔE00（构造工具的原件，只在这里包一层子采样）
# --------------------------------------------------------------------------- #
def de00_pixels(pred: torch.Tensor, ref: torch.Tensor, max_px: int,
                de00_fn) -> np.ndarray:
    """`pred`/`ref` (P,3) RGB[0,1] -> (P',) CIEDE2000。

    `de00_fn` 是 `epr050_build_degradation.de00` 的**原件**（skimage
    `deltaE_ciede2000`，与构造侧标定 s 时用的是同一个函数）。评测像素动辄
    10 万/样本 × 4560 样本，全量走 skimage 不现实，所以按**确定性等距**子采样到
    `max_px`（不是随机抽），子采样数落盘。
    """
    p = pred.shape[0]
    if max_px > 0 and p > max_px:
        step = max(1, p // int(max_px))
        sel = torch.arange(0, p, step, device=pred.device)[:int(max_px)]
        pred, ref = pred[sel], ref[sel]
    return de00_fn(pred.reshape(-1, 3), ref.reshape(-1, 3))
