#!/usr/bin/env python3
# 源自 experiments/prs/EPR-051_masked-restore-production/stage0/sprf/stage_solver.py @ 2026-09-06（原件未入 git；原 sha256 见 veraretouch_sprf/PROVENANCE.md），逐字复制/仅改导入与路径解析（EPR-052）。
"""EPR-051 · stage0 · arm=sprf · T2 第 3 层：逆序阶段求解器。

  euler_one_per_stage      每阶段 1 次前向（NFE = d × 1）
  midpoint_two_per_stage   每阶段 2 次前向（NFE = d × 2，NFE sweep 用）
  rollout_with_metrics     附带 rollout_drift / stage_oob_fraction / stage_oob_max
  restore                  生产接口：只读白名单键，禁入 lut/lut_id/before/
                           intermediate_gt/forward_operator

更新式（PROPOSAL §2）：
    z ← where(β_m == 0, z, z + β_m·ĥ)          逆序 m = S..1
    out = where(union_mask, clamp(z, 0, 1), y)  最终

β 的来源永远是 oracle 场（journal 重建 × 标定 s），与网络看到多少 α 无关。
action 头零初始化 ⇒ ĥ≡0 ⇒ 上式逐位是恒等映射，跨越整条链（断言 1）。
"""
from __future__ import annotations

import torch

from veraretouch_sprf.data.stage_targets import PATH_MODES, die

# 生产 restore() 允许读的键；除此之外一律不碰。
PRODUCTION_KEYS = ("cond", "y", "alphas", "union", "depth", "s")
# 训练/诊断专属，生产接口禁入（有 LUT 身份或有真值）。
FORBIDDEN_KEYS = ("lut", "lut_id", "before", "intermediate_gt", "forward_operator",
                  "edit", "inv_lut", "ref_pair")


class ProductionBatch(dict):
    """读到禁入键即死的 batch 视图。

    N6：不止 `__getitem__`/`get` —— 遍历类接口（`__iter__` / `items` /
    `values` / `pop` / `popitem`）同样能把禁入键的**值**递出去，所以一并堵上。
    键名本身可见（`keys()` 不拦），拦的是值。
    """

    @staticmethod
    def _check(k):
        if k in FORBIDDEN_KEYS:
            die(f"生产 restore() 读了禁入键 {k!r}（推理端无 LUT 身份/无真值）")

    def __getitem__(self, k):
        self._check(k)
        return super().__getitem__(k)

    def get(self, k, default=None):
        self._check(k)
        return super().get(k, default)

    def pop(self, k, *a):
        self._check(k)
        return super().pop(k, *a)

    def popitem(self):
        k, v = super().popitem()
        self._check(k)
        return k, v

    def __iter__(self):
        for k in super().__iter__():
            self._check(k)
            yield k

    def items(self):
        for k, v in super().items():
            self._check(k)
            yield k, v

    def values(self):
        for k, v in super().items():
            self._check(k)
            yield v


def stages_for(path_mode: str, depth: int) -> int:
    """一次 restore 走多少个阶段（NFE 计数的 d）。"""
    if path_mode not in PATH_MODES:
        die(f"path_mode {path_mode!r} 不在 {PATH_MODES}")
    return 1 if path_mode == "one_shot" else int(depth)


def stage_beta(path_mode: str, alphas: torch.Tensor, union: torch.Tensor,
               depth: torch.Tensor, m: torch.Tensor) -> torch.Tensor:
    """第 m 阶段实际用于更新的 β 场 (B,P)；越过该样本深度的阶段恒为 0。"""
    act = (m <= depth).to(alphas.dtype).unsqueeze(-1)
    if path_mode == "chain":
        b = alphas[torch.arange(alphas.shape[0], device=alphas.device), m - 1]
    elif path_mode == "linear":
        b = union / depth.to(alphas.dtype).unsqueeze(-1)
    else:                                        # one_shot
        b = union * (m == 1).to(alphas.dtype).unsqueeze(-1)
    return b * act


def _call(model, base, z, y, alpha_hat, alphas, union, depth, s, path_mode,
          m_idx: int, lam_val: float, lut_ids=None, checkpoint: bool = False,
          edits=None):
    """一个阶段的一次前向。

    `checkpoint=True` 时用 `torch.utils.checkpoint` 只存阶段边界、反传时重算该
    阶段的层内激活。**数学与梯度不变**（同样的算子、同样的输入，重算得到逐位相同
    的前向值），换的是显存：rollout 要把 d 个阶段 × n_layers 层的激活全部留到
    反传，batch 64 × 8192 px × w512 × 8 层 × 6 阶段 实测直接 OOM（68.5 GiB）。
    """
    n = z.shape[0]
    dev = z.device
    m = torch.full((n,), int(m_idx), dtype=torch.long, device=dev)
    lam = torch.full((n,), float(lam_val), dtype=z.dtype, device=dev)
    beta = stage_beta(path_mode, alphas, union, depth, m)
    lid = None if lut_ids is None else lut_ids[:, int(m_idx) - 1]
    # X-COND：逐阶段的编辑来源（inv_lut = bank 行号 long；ref_pair = 描述子 float）
    ed = None if edits is None else edits[:, int(m_idx) - 1]
    if checkpoint and torch.is_grad_enabled():
        def _fwd(z_, base_, alphas_, ahat_, beta_):
            return model(base_, z_, y, ahat_, beta_, alphas_, lam, m, depth, s,
                         lid, ed)
        h, x0 = torch.utils.checkpoint.checkpoint(
            _fwd, z, base, alphas, alpha_hat, beta, use_reentrant=False)
    else:
        h, x0 = model(base, z, y, alpha_hat, beta, alphas, lam, m, depth, s, lid,
                      ed)
    return h, x0, beta


def _update(z: torch.Tensor, beta: torch.Tensor, h: torch.Tensor) -> torch.Tensor:
    b = beta.unsqueeze(-1)
    return torch.where(b == 0, z, z + b * h)


def _finish(z: torch.Tensor, y: torch.Tensor, union: torch.Tensor,
            lo: float, hi: float) -> torch.Tensor:
    return torch.where(union.unsqueeze(-1) > 0, z.clamp(lo, hi), y)


def euler_one_per_stage(model, base, y, alpha_hat, alphas, union, depth, s,
                        path_mode: str, n_stages: int, clamp_lo: float = 0.0,
                        clamp_hi: float = 1.0, trace=None, lut_ids=None,
                        checkpoint: bool = False, edits=None):
    """逆序 m = n_stages..1，每阶段 1 次前向。返回 `(out, nfe)`。"""
    z = y
    nfe = 0
    for m_idx in range(int(n_stages), 0, -1):
        h, _, beta = _call(model, base, z, y, alpha_hat, alphas, union, depth, s,
                           path_mode, m_idx, 0.0, lut_ids, checkpoint, edits)
        nfe += 1
        z = _update(z, beta, h)
        if trace is not None:
            trace.append(z)
    return _finish(z, y, union, clamp_lo, clamp_hi), nfe


def midpoint_two_per_stage(model, base, y, alpha_hat, alphas, union, depth, s,
                           path_mode: str, n_stages: int, clamp_lo: float = 0.0,
                           clamp_hi: float = 1.0, trace=None, lut_ids=None,
                           checkpoint: bool = False, edits=None):
    """逆序每阶段 2 次前向（中点法）。返回 `(out, nfe)`。

    段内参数化沿 `stage_targets.path_states` 的 λ：λ=0 是该阶段入口状态，
    中点取 λ=0.5，用中点处的 ĥ 走完整步。
    """
    z = y
    nfe = 0
    for m_idx in range(int(n_stages), 0, -1):
        h1, _, beta = _call(model, base, z, y, alpha_hat, alphas, union, depth, s,
                            path_mode, m_idx, 0.0, lut_ids, checkpoint, edits)
        nfe += 1
        z_mid = _update(z, 0.5 * beta, h1)
        h2, _, _ = _call(model, base, z_mid, y, alpha_hat, alphas, union, depth, s,
                         path_mode, m_idx, 0.5, lut_ids, checkpoint, edits)
        nfe += 1
        z = _update(z, beta, h2)
        if trace is not None:
            trace.append(z)
    return _finish(z, y, union, clamp_lo, clamp_hi), nfe


SOLVERS = {1: euler_one_per_stage, 2: midpoint_two_per_stage}


def rollout_with_metrics(model, base, y, alpha_hat, alphas, union, depth, s,
                         path_mode: str, n_stages: int, nfe_per_stage: int,
                         ref_states: torch.Tensor | None = None,
                         clamp_lo: float = 0.0, clamp_hi: float = 1.0,
                         lut_ids=None, return_trace: bool = False,
                         stage_mask_check: bool = False,
                         checkpoint: bool = False, edits=None):
    """整条逆序链 + 预注册守卫列。

    `ref_states` (B, n_stages+1, P, 3)：逆序参考折线，`ref_states[:, j]` 是
    「已走 j 个逆序阶段」的真实状态（j=0 即 y）。给了就算 rollout_drift。

    返回 `(out, metrics)`，metrics 列：
      `nfe`                     实际前向次数
      `rollout_drift`           Σ_j |z_j − ref_j| 在 union 像素上的均值，再对 j 取均值
      `rollout_drift_by_stage`  同上但逐 j 保留（提案的 `rollout_drift[m]`）
      `stage_oob_fraction`      逐阶段 z 越出 [lo,hi] 的像素比例，对阶段取均值
      `stage_oob_fraction_by_stage` / `stage_oob_max_by_stage`  逐阶段保留
      `stage_oob_max`           max_j max |z_j − clamp(z_j)|
    逆序阶段 j = 1..n_stages 对应正向阶段 m = n_stages+1-j（键名用 m）。
    """
    if int(nfe_per_stage) not in SOLVERS:
        die(f"nfe_per_stage = {nfe_per_stage} 未实现（可选 {sorted(SOLVERS)}）")
    trace: list[torch.Tensor] = []
    out, nfe = SOLVERS[int(nfe_per_stage)](
        model, base, y, alpha_hat, alphas, union, depth, s, path_mode,
        n_stages, clamp_lo, clamp_hi, trace=trace, lut_ids=lut_ids,
        checkpoint=checkpoint, edits=edits)
    u = union.unsqueeze(-1)
    denom = u.sum().clamp_min(1.0) * 3.0
    oob_frac, oob_max_by, drift_by = {}, {}, {}
    stage_mask_bitexact = {}
    oob_max = 0.0
    prev = y
    for j, z in enumerate(trace, start=1):
        m = int(n_stages) + 1 - j
        ex = (z - z.clamp(clamp_lo, clamp_hi)).abs()
        oob_frac[m] = float(((ex > 0) & (u > 0)).sum()) / float(denom)
        oob_max_by[m] = float(ex.max())
        oob_max = max(oob_max, oob_max_by[m])
        if ref_states is not None:
            drift_by[m] = float(((z - ref_states[:, j]).abs() * u).sum() / denom)
        if stage_mask_check:
            # N4：逐阶段掩膜守卫 —— 该阶段 β_m==0 的像素必须逐位不动（G-A3 口径）。
            mm = torch.full((z.shape[0],), m, dtype=torch.long, device=z.device)
            b0 = stage_beta(path_mode, alphas, union, depth, mm) == 0
            n0 = int(b0.sum())
            same = int((z[b0] == prev[b0]).all(dim=-1).sum()) if n0 else 0
            stage_mask_bitexact[m] = dict(n_beta_zero=n0, n_bit_exact=same,
                                          all_bit_exact=(n0 == same))
        prev = z
    return out, dict(nfe=int(nfe),
                     trace=(trace if return_trace else None),
                     stage_mask_bitexact=stage_mask_bitexact,
                     rollout_drift=(sum(drift_by.values()) / len(drift_by)
                                    if drift_by else None),
                     rollout_drift_by_stage=drift_by,
                     stage_oob_fraction=(sum(oob_frac.values()) / len(oob_frac)
                                         if oob_frac else 0.0),
                     stage_oob_fraction_by_stage=oob_frac,
                     stage_oob_max_by_stage=oob_max_by,
                     stage_oob_max=oob_max)


@torch.no_grad()
def restore(model, batch, path_mode: str, nfe_per_stage: int, depth: int,
            clamp_lo: float = 0.0, clamp_hi: float = 1.0,
            edits=None, oracle_ok: bool = False) -> torch.Tensor:
    """生产推理接口：只读 `PRODUCTION_KEYS`，永不碰 LUT / 真值。

    `batch` 必须是 `ProductionBatch`（读禁入键即死）。`depth` 是该批的阶段数，
    来自条件契约（journal 重建的 β 场步数），不是任何真值。

    X-COND 的两个臂（`edit.condition != off`）是 **oracle 诊断行**：编辑信息
    （逆 LUT / 演示对）在推理端拿不到，所以默认拒绝当生产接口用，和
    `lut_id_condition = "embed"` 同一条规矩。只有启动断言（A1/A6）会带
    `oracle_ok=True` 显式调用，此时 `edits` 必须从**参数**递进来 ——
    它永远不进 `batch`，A6「删掉禁入键输出不变」因此仍然是一条有效断言。
    """
    if not isinstance(batch, ProductionBatch):
        die("restore() 只接受 ProductionBatch（禁入键的读取守卫在它身上）")
    if getattr(model, "lut_id_condition", "off") != "off":
        die("lut_id_condition != off：该臂是 K5 的 oracle 诊断行，推理端没有 LUT "
            "身份，不能当生产 restore() 用")
    edit_cond = getattr(model, "edit_condition", "off")
    if edit_cond != "off" and not oracle_ok:
        die(f"edit.condition = {edit_cond!r}（契约 "
            f"{getattr(model, 'edit_contract', '?')!r}）：该臂是 X-COND 的 oracle "
            "诊断行，推理端没有真实编辑信息，不能当生产 restore() 用")
    if edit_cond == "off" and edits is not None:
        die("edit.condition = off 却给 restore() 传了 edits")
    if edit_cond != "off" and edits is None:
        die(f"edit.condition = {edit_cond!r} 但 restore() 没收到 edits")
    missing = [k for k in PRODUCTION_KEYS if k not in batch]
    if missing:
        die(f"restore() 缺少生产键 {missing}")
    c = batch["cond"]
    y = batch["y"]
    alphas = batch["alphas"]
    union = batch["union"]
    d = batch["depth"]
    s = batch["s"]
    alpha_hat = 1.0 - torch.prod(1.0 - alphas, dim=1)
    base = model.cond.base(c)
    n_stages = stages_for(path_mode, depth)
    out, nfe = SOLVERS[int(nfe_per_stage)](
        model, base, y, alpha_hat, alphas, union, d, s, path_mode, n_stages,
        clamp_lo, clamp_hi, edits=edits)
    want = n_stages * int(nfe_per_stage)
    if nfe != want:
        die(f"NFE 计数 {nfe} != depth×nfe_per_stage = {want}")
    return out
