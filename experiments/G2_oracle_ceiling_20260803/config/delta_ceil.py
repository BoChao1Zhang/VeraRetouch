"""G2 解析法 Δ_ceil：条件天花板（PLAN §4.1 差分分层法 Step 2–4/6）。

对一张图的像素集 {(x_i, y_i, s_i)}，问「把 s 加进逐像素算子的输入，最多还能多解释多少」。

分区
----
颜色按等宽直方图分箱，由细到粗 33³→17³→9³，某箱 n<n_min 时该箱像素并到更粗一层
（PLAN 的 n<20 并层规则），都不支持则进兜底单元。s 按 8 个**全局分位桶**切
（并列值不拆，退化 s 自然只有 1 桶；禁逐图 min-max/softmax 归一化——红线）。

单元内估计器（两套，都报）
------------------------
* `const`  —— 条件均值。PLAN §4.1 的字面口径，但有硬性量化地板：33³ 箱宽 ≈ 7.7 个
  8bit 级 → 箱内色差自身贡献 MSE≈4.9 → PSNR 封顶 ≈41 dB（实测全局档 41.41 dB）。
  编辑幅度弱时（D-CONSTRUCT 恒等映射本身就有 37–41 dB）整个被地板吞掉，不可用。
* `affine` —— 单元内最小二乘仿射 ŷ = A x + b。局部线性正是三线性插值 3D LUT 在
  单个格子里的行为，因此更贴近真实 3D LUT 的能力；地板抬到实测 63.4 dB。**主口径**。

两种 4D 分区形态（都报）
----------------------
* `delta`     嵌套式：在 3D 单元**内部**按 s 桶细分（PLAN 字面），受父单元颜色分辨率牵制。
* `delta_arm` 臂式：每个 s 桶**各自独立**建颜色分层 = N 张 3D LUT 按 s 查表，
              正是 4D LUT 的实现形态，也正是实验法互证那条臂。**headline**。

细分采三档支持度（`_refine`），保证 SSE 单调不增 ⇒ Δ_ceil ≥ 0 由构造成立。

有限样本膨胀的控制
----------------
细分越多越会拟合噪声 → Δ_ceil 天然有正偏。四道控制，全部跑在同一估计器上：
  Δ_const   s≡1 ⇒ 4D 分区严格等于 3D 分区 ⇒ **恒等于 0**（实现自检，见 selfcheck）
  Δ_donor   把**另一张图**的掩膜贴过来当 s（Δ_shuffle 的解析法版本）
  对照档     D-SFT-G 真·全局编辑 + 移植掩膜，任何非零 Δ 全是膨胀
  `*_cv`    二折交叉拟合列（分区结构用全量定，单元系数只用另一折）。
            注意：小掩膜下每折样本减半会同时踩支持度门，该列**系统性偏保守**，
            主控请看前三道。
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np
import torch

N_MIN = 20                       # PLAN: bin n<20 并粗层级（常数估计器）
N_MIN_AFFINE = 60                # 仿射估计器每单元 12 自由度，支持度要求相应提高
N_MIN_OFFSET = 20                # 「父仿射 + 常数偏置」档（3 自由度）的支持度门
COLOR_LEVELS = (33, 17, 9)       # 由细到粗
S_BUCKETS = 8
CAP_DB = 100.0


# --------------------------------------------------------------------------- 工具

def _color_code(x: torch.Tensor, k: int) -> torch.Tensor:
    """等宽直方图分箱码：(P,3) [0,1] -> (P,) int64，值域 [0, k³)。"""
    q = torch.clamp((x * k).floor().to(torch.int64), 0, k - 1)
    return (q[:, 0] * k + q[:, 1]) * k + q[:, 2]


def s_quantile_buckets(s: torch.Tensor, n_buckets: int = S_BUCKETS) -> torch.Tensor:
    """s 的分位桶码 (P,) int64 ∈ [0, n_buckets)。

    分位边界用 torch.quantile（大数组走抽样，>1e6 时对 1e6 个随机点估分位）。
    并列值（掩膜里大量的 0/1）自然落进同一桶——有效桶数因此可能 < n_buckets，
    这是正确行为（不做逐图 min-max/softmax 归一化，红线）。
    """
    p = s.numel()
    ref = s
    if p > 1_000_000:
        g = torch.Generator(device=s.device).manual_seed(0)
        ref = s[torch.randint(0, p, (1_000_000,), device=s.device, generator=g)]
    qs = torch.linspace(0, 1, n_buckets + 1, device=s.device, dtype=s.dtype)[1:-1]
    edges = torch.quantile(ref.float(), qs.float())
    edges = torch.unique(edges)
    if edges.numel() == 0:
        return torch.zeros_like(s, dtype=torch.int64)
    return torch.bucketize(s, edges, right=False)


def _group_mean(code: torch.Tensor, y: torch.Tensor
                ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """返回 (inv (P,), means (U,3), counts (U,))。"""
    uniq, inv = torch.unique(code, return_inverse=True)
    u = uniq.numel()
    counts = torch.zeros(u, device=y.device, dtype=y.dtype)
    counts.index_add_(0, inv, torch.ones_like(inv, dtype=y.dtype))
    sums = torch.zeros(u, 3, device=y.device, dtype=y.dtype)
    sums.index_add_(0, inv, y)
    return inv, sums / counts.unsqueeze(1), counts


def _psnr(pred: torch.Tensor, gt: torch.Tensor) -> float:
    qp = torch.round(pred.clamp(0, 1) * 255.0)
    qg = torch.round(gt.clamp(0, 1) * 255.0)
    mse = float(torch.mean((qp - qg) ** 2))
    if mse <= 0:
        return CAP_DB
    return min(10.0 * math.log10(255.0 ** 2 / mse), CAP_DB)


# --------------------------------------------------------------------------- 逐箱仿射

def _group_affine(code: torch.Tensor, x: torch.Tensor, y: torch.Tensor,
                  ridge: float = 1e-6
                  ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """逐单元最小二乘仿射 ŷ = A x + b（齐次 4 维），返回 (inv, pred_all, counts)。

    为什么需要它：逐箱**常数**（条件均值）估计器有一个硬性量化地板——33³ 等宽箱
    宽 1/33 ≈ 7.7 个 8bit 级，箱内色差本身就贡献 MSE≈4.9 → PSNR 上限 ≈41 dB。
    D-CONSTRUCT 的编辑幅度使「什么都不做」就有 37–41 dB，常数估计器整个被地板吞掉
    （实测 L3/L4/L5 的 PSNR_3D 甚至低于恒等），Δ_ceil 被压成不可分辨。
    逐箱仿射 = 局部线性模型，正是三线性插值 3D LUT 在单个格子内的行为，
    地板被抬到 55 dB 以上，Δ_ceil 才有量程。
    """
    uniq, inv = torch.unique(code, return_inverse=True)
    u = uniq.numel()
    xh = torch.cat([x, torch.ones_like(x[:, :1])], 1).double()      # (P,4)
    yd = y.double()
    counts = torch.zeros(u, device=x.device, dtype=torch.float64)
    counts.index_add_(0, inv, torch.ones(x.shape[0], device=x.device, dtype=torch.float64))
    gram = torch.zeros(u, 16, device=x.device, dtype=torch.float64)
    gram.index_add_(0, inv, (xh.unsqueeze(2) * xh.unsqueeze(1)).reshape(-1, 16))
    rhs = torch.zeros(u, 12, device=x.device, dtype=torch.float64)
    rhs.index_add_(0, inv, (xh.unsqueeze(2) * yd.unsqueeze(1)).reshape(-1, 12))
    g = gram.reshape(u, 4, 4) + ridge * torch.eye(4, device=x.device,
                                                 dtype=torch.float64).unsqueeze(0)
    theta = torch.linalg.solve(g, rhs.reshape(u, 4, 3))             # (U,4,3)
    pred = torch.einsum("pi,pij->pj", xh, theta[inv]).float()
    return inv, pred, counts.float()


# --------------------------------------------------------------------------- 核心

@dataclass
class CeilResult:
    psnr_3d: float
    psnr_4d: float
    delta: float
    psnr_3d_cv: float
    psnr_4d_cv: float
    delta_cv: float
    psnr_id: float                     # 恒等映射 PSNR（编辑强度参照）
    n_pixels: int
    n_cells_3d: int
    n_cells_4d: int
    frac_refined: float                # 被 s 细分（子箱 n>=N_MIN）的像素占比
    n_s_buckets: int
    extra: dict = field(default_factory=dict)

    def to_json(self) -> dict:
        d = dict(self.__dict__)
        d.pop("extra")
        d.update(self.extra)
        return d


def _hier_3d_affine(x: torch.Tensor, y: torch.Tensor, levels=COLOR_LEVELS,
                    n_min: int = N_MIN_AFFINE):
    """逐箱仿射版分层：单元划分同 `_hier_3d`，单元内预测换成最小二乘仿射。"""
    p = x.shape[0]
    cell = torch.full((p,), -1, dtype=torch.int64, device=x.device)
    assigned = torch.zeros(p, dtype=torch.bool, device=x.device)
    for li, k in enumerate(levels):
        inv, _, counts = _group_mean(_color_code(x, k), y)
        ok = (counts[inv] >= n_min) & (~assigned)
        if bool(ok.any()):
            cell[ok] = inv[ok] + li * (1 << 40)
            assigned |= ok
    cell[~assigned] = -1
    _, pred, _ = _group_affine(cell, x, y)
    return cell, pred


def _hier_3d(x: torch.Tensor, y: torch.Tensor, levels=COLOR_LEVELS, n_min: int = N_MIN):
    """层级 3D 分箱：返回 (cell_id (P,), pred3 (P,3))。

    cell_id 唯一标识「该像素最终落在的 3D 条件均值单元」，供 4D 细分做父单元。
    """
    p = x.shape[0]
    cell = torch.full((p,), -1, dtype=torch.int64, device=x.device)
    assigned = torch.zeros(p, dtype=torch.bool, device=x.device)
    for li, k in enumerate(levels):
        inv, _, counts = _group_mean(_color_code(x, k), y)
        ok = (counts[inv] >= n_min) & (~assigned)
        if bool(ok.any()):
            cell[ok] = inv[ok] + li * (1 << 40)
            assigned |= ok
    cell[~assigned] = -1                        # 兜底单元（任何层级都不支持的稀有色）
    # 预测 = **本分区每个单元自身成员**的均值。注意不能直接用第 k 层的整箱均值：
    # 整箱里已被更细层级领走的像素不属于本单元，混进来会让 4D 侧在同一单元内
    # 再细分时白捡差额，导致 s≡const 的 Δ_const 不为 0（见 selfcheck 实现自检项）。
    inv_c, means_c, _ = _group_mean(cell, y)
    return cell, means_c[inv_c]


def _cv_pred(code: torch.Tensor, y: torch.Tensor, fold: torch.Tensor,
             fallback: torch.Tensor, n_min: int) -> torch.Tensor:
    """二折交叉拟合预测：fold==0 的像素用 fold==1 的单元均值，反之亦然。

    另一折里该单元样本数 < n_min 时退回 fallback（父层预测）。
    """
    pred = fallback.clone()
    uniq, inv = torch.unique(code, return_inverse=True)
    u = uniq.numel()
    for f in (0, 1):
        src = fold == (1 - f)
        counts = torch.zeros(u, device=y.device, dtype=y.dtype)
        counts.index_add_(0, inv[src], torch.ones(int(src.sum()), device=y.device,
                                                  dtype=y.dtype))
        sums = torch.zeros(u, 3, device=y.device, dtype=y.dtype)
        sums.index_add_(0, inv[src], y[src])
        means = sums / counts.clamp(min=1).unsqueeze(1)
        tgt = (fold == f) & (counts[inv] >= n_min)
        pred[tgt] = means[inv[tgt]]
    return pred


def _refine(code: torch.Tensor, x: torch.Tensor, y: torch.Tensor,
            parent: torch.Tensor, n_min_aff: int, n_min_off: int,
            estimator: str) -> tuple[torch.Tensor, torch.Tensor]:
    """在父分区预测之上按 `code` 细分，三档支持度（保证 SSE 单调不增）：

      n ≥ n_min_aff : 本单元自己的最小二乘仿射（12 自由度）
      n ≥ n_min_off : 父仿射 + 本单元的常数残差偏置（3 自由度）
      否则          : 沿用父预测

    为什么要中间档：小掩膜（实测 D-CONSTRUCT L1 掩膜只占 2.4% 像素）在
    「颜色单元 × s 桶」上几乎每个子单元都不到 60 像素，只有 n_min_aff 一档时
    整个 s 细分会被支持度门全部否掉，4D 天花板被人为压低（同一样本硬两臂
    对照 84.2 dB vs 单档门 54.1 dB）。常数偏置档只要 3 个自由度，正好接住这一类。
    `estimator=="const"` 时两档等价（父常数 + 残差均值 = 本单元均值）。
    """
    inv, _, counts = _group_mean(code, y)
    n = counts[inv]
    pred = parent.clone()
    ok_off = n >= n_min_off
    if bool(ok_off.any()):
        _, off, _ = _group_mean(code, y - parent)
        pred[ok_off] = parent[ok_off] + off[inv[ok_off]]
    ok = ok_off
    if estimator == "affine":
        ok_aff = n >= n_min_aff
        if bool(ok_aff.any()):
            _, pa, _ = _group_affine(code, x, y)
            pred[ok_aff] = pa[ok_aff]
        ok = ok_off | ok_aff
    return pred, ok


def _cv_pred_offset(code: torch.Tensor, y: torch.Tensor, fold: torch.Tensor,
                    parent: torch.Tensor, current: torch.Tensor,
                    n_min: int, only: torch.Tensor | None = None) -> torch.Tensor:
    """常数偏置档的交叉拟合：在另一折上估 `mean(y − parent)` 再加回本折。

    只在 `current` 还等于 `parent`（即仿射档没被采纳）的像素上生效。
    """
    pred = current.clone()
    untouched = (current == parent).all(1)
    if only is not None:
        untouched &= only
    uniq, inv = torch.unique(code, return_inverse=True)
    u = uniq.numel()
    resid = y - parent
    for f in (0, 1):
        src = fold == (1 - f)
        counts = torch.zeros(u, device=y.device, dtype=y.dtype)
        counts.index_add_(0, inv[src], torch.ones(int(src.sum()), device=y.device,
                                                  dtype=y.dtype))
        sums = torch.zeros(u, 3, device=y.device, dtype=y.dtype)
        sums.index_add_(0, inv[src], resid[src])
        off = sums / counts.clamp(min=1).unsqueeze(1)
        tgt = (fold == f) & (counts[inv] >= n_min) & untouched
        pred[tgt] = parent[tgt] + off[inv[tgt]]
    return pred


def _cv_pred_affine(code: torch.Tensor, x: torch.Tensor, y: torch.Tensor,
                    fold: torch.Tensor, fallback: torch.Tensor, n_min: int,
                    ridge: float = 1e-6,
                    only: torch.Tensor | None = None) -> torch.Tensor:
    """`_cv_pred` 的仿射版：单元的仿射系数在另一折上拟合。"""
    pred = fallback.clone()
    uniq, inv = torch.unique(code, return_inverse=True)
    u = uniq.numel()
    xh = torch.cat([x, torch.ones_like(x[:, :1])], 1).double()
    yd = y.double()
    eye = torch.eye(4, device=x.device, dtype=torch.float64).unsqueeze(0)
    for f in (0, 1):
        src = fold == (1 - f)
        inv_s = inv[src]
        counts = torch.zeros(u, device=x.device, dtype=torch.float64)
        counts.index_add_(0, inv_s, torch.ones(int(src.sum()), device=x.device,
                                               dtype=torch.float64))
        gram = torch.zeros(u, 16, device=x.device, dtype=torch.float64)
        gram.index_add_(0, inv_s, (xh[src].unsqueeze(2) * xh[src].unsqueeze(1)).reshape(-1, 16))
        rhs = torch.zeros(u, 12, device=x.device, dtype=torch.float64)
        rhs.index_add_(0, inv_s, (xh[src].unsqueeze(2) * yd[src].unsqueeze(1)).reshape(-1, 12))
        theta = torch.linalg.solve(gram.reshape(u, 4, 4) + ridge * eye, rhs.reshape(u, 4, 3))
        tgt = (fold == f) & (counts[inv] >= n_min)
        if only is not None:
            tgt &= only
        if bool(tgt.any()):
            pred[tgt] = torch.einsum("pi,pij->pj", xh[tgt], theta[inv[tgt]]).float()
    return pred


def ceiling(x: np.ndarray | torch.Tensor, y: np.ndarray | torch.Tensor,
            s: np.ndarray | torch.Tensor, device: str = "cuda",
            levels=COLOR_LEVELS, n_min: int | None = None,
            n_s_buckets: int = S_BUCKETS, seed: int = 0,
            with_cv: bool = True, estimator: str = "const") -> CeilResult:
    """单图 Δ_ceil。x,y: (P,3) [0,1]；s: (P,) [0,1]。

    estimator: "const"  逐箱条件均值（PLAN §4.1 字面口径，有 ~41 dB 量化地板）
               "affine" 逐箱最小二乘仿射（局部线性 = 三线性 3D LUT 的格内行为，
                        地板抬到 55 dB 以上；编辑幅度弱时必须用这个）
    """
    if n_min is None:
        n_min = N_MIN if estimator == "const" else N_MIN_AFFINE
    xt = torch.as_tensor(np.asarray(x), dtype=torch.float32, device=device)
    yt = torch.as_tensor(np.asarray(y), dtype=torch.float32, device=device)
    st = torch.as_tensor(np.asarray(s), dtype=torch.float32, device=device)
    p = xt.shape[0]

    sb = s_quantile_buckets(st, n_s_buckets)
    n_sb = int(sb.max().item()) + 1

    if estimator == "affine":
        cell3, pred3 = _hier_3d_affine(xt, yt, levels, n_min)
    else:
        cell3, pred3 = _hier_3d(xt, yt, levels, n_min)
    code4 = cell3 * n_s_buckets + sb
    pred4, ok4 = _refine(code4, xt, yt, pred3, n_min, N_MIN_OFFSET, estimator)

    # 臂式 4D：每个 s 桶**各自独立**建颜色分层（= N 张 3D LUT 按 s 查表的实现形态，
    # 也正是实验法互证那条臂）。小掩膜下颜色分层会自动退到更粗的层级，
    # 不像嵌套细分那样受父分区分辨率牵制。
    cell_arm = cell3.clone()
    for b in range(n_sb):
        sel = sb == b
        if int(sel.sum()) < n_min:
            continue
        xb, yb = xt[sel], yt[sel]
        cb = (_hier_3d_affine(xb, yb, levels, n_min)[0] if estimator == "affine"
              else _hier_3d(xb, yb, levels, n_min)[0])
        cell_arm[sel] = cb * n_s_buckets + b + (1 << 52)
    if estimator == "affine":
        _, pred_arm, _ = _group_affine(cell_arm, xt, yt)
    else:
        inv_a, mean_a, _ = _group_mean(cell_arm, yt)
        pred_arm = mean_a[inv_a]

    res = CeilResult(
        psnr_3d=_psnr(pred3, yt), psnr_4d=_psnr(pred4, yt), delta=0.0,
        psnr_3d_cv=float("nan"), psnr_4d_cv=float("nan"), delta_cv=float("nan"),
        psnr_id=_psnr(xt, yt), n_pixels=p,
        n_cells_3d=int(torch.unique(cell3).numel()),
        n_cells_4d=int(torch.unique(code4[ok4]).numel()) if bool(ok4.any()) else 0,
        frac_refined=float(ok4.float().mean()), n_s_buckets=n_sb)
    res.delta = res.psnr_4d - res.psnr_3d
    res.extra["psnr_4d_arm"] = _psnr(pred_arm, yt)
    res.extra["delta_arm"] = res.extra["psnr_4d_arm"] - res.psnr_3d

    if with_cv:
        # 分层结构（哪个像素落哪个 3D 单元）沿用全量数据的判定，只有**单元均值**
        # 交叉拟合：一半像素估均值、另一半算误差，互换取平均。
        g = torch.Generator(device=device).manual_seed(seed)
        fold = (torch.rand(p, device=device, generator=g) < 0.5).to(torch.int64)
        gmean = yt.mean(0, keepdim=True).expand(p, 3)
        n_min_cv = max(1, n_min // 2)
        if estimator == "affine":
            n_aff_cv, n_off_cv = max(n_min_cv, 12), max(1, N_MIN_OFFSET // 2)

            def _two_tier(code: torch.Tensor, parent: torch.Tensor,
                          parent_code: torch.Tensor | None = None) -> torch.Tensor:
                # 仿射档 + 常数偏置档。两档必须**同时**用在 3D 与 4D 两侧：
                # 只给 4D 加偏置档会把「支持度门更松」误算成 s 的增益
                # （实测 L0 全局档因此虚高 2.2 dB）。
                # `only`：只在 code 相对 parent_code **真正变细**的像素上细化，
                # 否则同一分区被重复拟合一次截距，纯拟合噪声（L0 因此虚低 0.34 dB）。
                only = None
                if parent_code is not None:
                    ic, _, cc = _group_mean(code, yt)
                    ip, _, cp = _group_mean(parent_code, yt)
                    only = cc[ic] < cp[ip]
                q = _cv_pred_affine(code, xt, yt, fold, parent, n_aff_cv, only=only)
                return _cv_pred_offset(code, yt, fold, parent, q, n_off_cv, only=only)

            pred3cv = _two_tier(cell3, gmean)
            pred4cv = _two_tier(code4, pred3cv, cell3)
            predarmcv = _two_tier(cell_arm, pred3cv, cell3)
        else:
            pred3cv = _cv_pred(cell3, yt, fold, gmean, n_min_cv)
            pred4cv = _cv_pred(code4, yt, fold, pred3cv, n_min_cv)
            predarmcv = _cv_pred(cell_arm, yt, fold, pred3cv, n_min_cv)
        res.psnr_3d_cv = _psnr(pred3cv, yt)
        res.psnr_4d_cv = _psnr(pred4cv, yt)
        res.delta_cv = res.psnr_4d_cv - res.psnr_3d_cv
        res.extra["psnr_4d_arm_cv"] = _psnr(predarmcv, yt)
        res.extra["delta_arm_cv"] = res.extra["psnr_4d_arm_cv"] - res.psnr_3d_cv
    return res


# --------------------------------------------------------------------------- 混淆诊断

def confound_stats(x: np.ndarray, y: np.ndarray, hw: tuple[int, int]) -> dict:
    """PLAN §4.1 Step 5 的两个混淆统计（只报不筛，见 NOTES 决策 #3）。

    moran_i : 残差幅值场的 Moran's I（4 邻域），真局部编辑空间成簇 → 高。
    blur_drop: 残差 3x3 均值滤波后 MSE 的下降比例，>0.6 提示高频/非色彩编辑。
    """
    h, w = hw
    r = (y - x).reshape(h, w, 3)
    a = np.abs(r).mean(-1)
    am = a - a.mean()
    denom = float((am ** 2).sum()) + 1e-12
    num = float((am[:-1, :] * am[1:, :]).sum() + (am[:, :-1] * am[:, 1:]).sum()) * 2.0
    n_pairs = 2 * ((h - 1) * w + h * (w - 1))
    moran = (a.size / max(n_pairs, 1)) * num / denom

    k = np.ones((3, 3), dtype=np.float32) / 9.0
    rb = np.empty_like(r)
    for c in range(3):
        rb[..., c] = _conv3(r[..., c], k)
    mse0 = float((r ** 2).mean())
    mse1 = float((rb ** 2).mean())          # 残差 3x3 均值滤波后的能量
    return {"moran_i": moran, "blur_drop": 1.0 - (mse1 / (mse0 + 1e-12))}


def _conv3(a: np.ndarray, k: np.ndarray) -> np.ndarray:
    pad = np.pad(a, 1, mode="edge")
    out = np.zeros_like(a)
    for i in range(3):
        for j in range(3):
            out += k[i, j] * pad[i:i + a.shape[0], j:j + a.shape[1]]
    return out
