"""PR-1/PR-3 · 探针拟合与指标（INF-7「探针+MDL 自实现」）。

方法学出处（均已打开原始来源核实，见 PR13 NOTES.md §一）：
- **selectivity**：Hewitt & Liang, EMNLP 2019 (D19-1275) —— 原文 Fig.2 题注
  "Selectivity is defined as the difference between linguistic task accuracy and
  control task accuracy"。control task = 「associate word types with random
  outputs，each word token is assigned its type's output, regardless of context」。
  本项目回归任务的忠实对应见 `selectivity_pair()` 文档字符串。
- **MDL online (prequential) code**：Voita & Titov, EMNLP 2020 (arXiv 2003.12298)
  式 (4)：`L_online = t1*log2(K) − Σ_{i=1}^{S−1} log2 p_θi(y_{ti+1:ti+1}|x)`；
  compression = 均匀码长 / online 码长（原文 Table 6 题注 "compression – with
  respect to the corresponding uniform code"）；timesteps = 数据集的
  0.1, 0.2, 0.4, 0.8, 1.6, 3.2, 6.25, 12.5, 25, 50, 100 %。

纪律：
- 特征标准化只用**训练折**统计量（禁逐图归一化——那是在制造 R2）。
- 折切分按 **source_id 分组**（DATA_ASSIGNMENT §3.4：「按源再切折，杜绝同源跨折」）。
- 正则用 weight decay / ridge α，不用 dropout（Hewitt&Liang §5：dropout 对
  selectivity 无效）。
"""
from __future__ import annotations

import numpy as np

MDL_MIX = 0.05      # online code 与均匀码的混合系数（见 mdl_online）
TIMESTEP_FRACS = (0.001, 0.002, 0.004, 0.008, 0.016, 0.032,
                  0.0625, 0.125, 0.25, 0.5, 1.0)


# ---------------------------------------------------------------- 基础工具
def _standardizer(Xtr) -> tuple:
    """训练折统计量 + 退化列掩膜（**只用训练折**，禁逐图归一化）。

    近似常数的列（std 相对最大 std < 1e-8）在标准化后会变成数值巨大的纯噪声方向，
    把 Gram 矩阵的主方向全部占走（实测让 C5 像素基线的 R² 掉到 −20）。直接置零。
    """
    mu = Xtr.mean(0)
    sd = Xtr.std(0)
    keep = sd > max(1e-12, 1e-8 * float(sd.max() if sd.size else 0.0))
    return mu, np.where(keep, sd, 1.0), keep


def _apply_std(X, mu, sd, keep) -> np.ndarray:
    Z = (X - mu) / sd
    Z[:, ~keep] = 0.0
    return Z


def _sanitize(X) -> np.ndarray:
    """特征去非有限值（C2 随机骨干可能产生 inf/NaN）。**不做任何逐图归一化**。"""
    X = np.asarray(X, np.float32)
    if not np.isfinite(X).all():
        X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)
    return X


def grouped_folds(groups: np.ndarray, n_folds: int = 5, seed: int = 0) -> np.ndarray:
    """按组（source_id）切 K 折，返回逐样本 fold 号。"""
    uniq = np.unique(groups)
    rng = np.random.default_rng(seed)
    order = rng.permutation(len(uniq))
    g2f = {uniq[g]: i % n_folds for i, g in enumerate(order)}
    return np.array([g2f[g] for g in groups], dtype=np.int64)


def sample_folds(n: int, n_folds: int = 5, seed: int = 0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return rng.permutation(n) % n_folds


def r2_score(y: np.ndarray, p: np.ndarray) -> float:
    y = np.asarray(y, np.float64); p = np.asarray(p, np.float64)
    ss_res = float(((y - p) ** 2).sum())
    ss_tot = float(((y - y.mean()) ** 2).sum())
    return float("nan") if ss_tot <= 0 else 1.0 - ss_res / ss_tot


def roc_auc(scores: np.ndarray, labels: np.ndarray) -> float:
    from scipy.stats import rankdata
    s = np.asarray(scores, np.float64).ravel()
    y = np.asarray(labels).ravel().astype(bool)
    npos, nneg = int(y.sum()), int((~y).sum())
    if npos == 0 or nneg == 0:
        return float("nan")
    r = rankdata(s)
    return float((r[y].sum() - npos * (npos + 1) / 2.0) / (npos * nneg))


def bootstrap_ci(values: np.ndarray, groups: np.ndarray | None = None,
                 stat=lambda v: float(np.mean(v)), n_boot: int = 2000,
                 seed: int = 0, alpha: float = 0.05) -> tuple[float, float]:
    """（按组的）percentile bootstrap CI。groups 给定时按组重采样。"""
    rng = np.random.default_rng(seed)
    values = np.asarray(values)
    if groups is None:
        idx = rng.integers(0, len(values), size=(n_boot, len(values)))
        s = np.array([stat(values[i]) for i in idx])
    else:
        uniq = np.unique(groups)
        pos = {g: np.flatnonzero(groups == g) for g in uniq}
        s = []
        for _ in range(n_boot):
            pick = rng.integers(0, len(uniq), size=len(uniq))
            sel = np.concatenate([pos[uniq[p]] for p in pick])
            s.append(stat(values[sel]))
        s = np.array(s)
    return float(np.quantile(s, alpha / 2)), float(np.quantile(s, 1 - alpha / 2))


# ---------------------------------------------------------------- ridge 探针
def _ridge_solve(Xtr, ytr, alphas, device):
    """一次特征分解求全部 α（比逐 α solve 快一个量级，数值等价）。"""
    import torch
    G = Xtr.T @ Xtr
    b = Xtr.T @ ytr
    s, V = torch.linalg.eigh(G)
    Vb = V.T @ b
    return [V @ (Vb / (s + a)) for a in alphas]


def _blocks(groups_tr: np.ndarray) -> list[np.ndarray]:
    return [np.flatnonzero(groups_tr == u) for u in np.unique(groups_tr)]


def ridge_cv(X: np.ndarray, y: np.ndarray, folds: np.ndarray,
             alphas=(1e0, 1e1, 1e2, 1e3, 1e4, 1e5, 1e6, 1e7, 1e8),
             device: str = "cuda:0", groups: np.ndarray | None = None,
             inner_folds: int = 3) -> dict:
    """折外预测的 ridge 线性探针（P1）。

    - 外层折：调用方给（本项目一律 **按 source_id 分组**），R² / MAE 全部是折外的。
    - 内层 α 选择：外层训练折内的 **按源分组的 3 折嵌套 CV** + 1-SE 规则。
      两条踩过的坑（都实测复现过，故写死在这里）：
        (a) 按**样本**留一/切折会因同源 4 个扰动样本互为近邻而严重低估误差，
            α 被选成 1.0，C5 像素基线的折外 R² 掉到 −18.6；
        (b) 闭式 block-LOGO 在 α 小到近似插值时 (I−H_BB) 与残差同时趋 0，
            数值上给出**虚假的低误差**，A6 因此选到 α=1、折外 R² = −6.1。
      分组嵌套 CV 没有这两个病；代价是每个外层折多 3 次特征分解。
    - 标准化：逐特征均值/方差**只用外层训练折**估计（禁逐图归一化）。
    """
    import torch
    X = _sanitize(X)
    y = np.asarray(y, np.float64)
    pred = np.full(len(y), np.nan)
    chosen = []
    dev = torch.device(device if torch.cuda.is_available() else "cpu")

    def solvers(Z, yc):
        """返回 (w_fn, yhat_on) —— 一次特征分解，全部 α 复用。"""
        n, d = Z.shape
        if d >= n:                                   # 对偶：n×n 分解
            K = Z @ Z.T
            s, V = torch.linalg.eigh(K)
            Vy = V.T @ yc
            return lambda a: Z.T @ (V @ (Vy / (s + a)))
        G = Z.T @ Z
        g, U = torch.linalg.eigh(G)
        Uy = U.T @ (Z.T @ yc)
        return lambda a: U @ (Uy / (g + a))

    for f in np.unique(folds):
        tr, te = folds != f, folds == f
        mu, sd, keep = _standardizer(X[tr])
        Z = torch.tensor(_apply_std(X[tr], mu, sd, keep), device=dev, dtype=torch.float64)
        Zt = torch.tensor(_apply_std(X[te], mu, sd, keep), device=dev, dtype=torch.float64)
        yt = torch.tensor(y[tr], device=dev, dtype=torch.float64)
        ym = yt.mean()

        # --- 内层：**分组**嵌套 CV 选 α（外层测试折全程未被看到）---
        gtr = np.asarray(groups)[tr] if groups is not None else None
        inner = (grouped_folds(gtr, inner_folds, seed=17) if gtr is not None
                 else sample_folds(int(tr.sum()), inner_folds, seed=17))
        errs_per_fold = np.zeros((inner_folds, len(alphas)))
        for j in range(inner_folds):
            itr = torch.tensor(inner != j, device=dev)
            ite = torch.tensor(inner == j, device=dev)
            Zi, Zo = Z[itr], Z[ite]
            yi, yo = yt[itr], yt[ite]
            mi = yi.mean()
            wf = solvers(Zi, yi - mi)
            for ai, a in enumerate(alphas):
                p = Zo @ wf(a) + mi
                errs_per_fold[j, ai] = float(((p - yo) ** 2).mean())
        m_err = errs_per_fold.mean(0)
        se = errs_per_fold.std(0) / max(inner_folds ** 0.5, 1.0)
        i0 = int(np.argmin(m_err))
        # 1-SE 规则：取误差在最小值 1 个标准误内的**最大** α（更保守，抗内层噪声）
        thr = m_err[i0] + se[i0]
        i_sel = max(i for i in range(len(alphas)) if m_err[i] <= thr)
        best_a = alphas[i_sel]
        chosen.append(best_a)
        best_w = solvers(Z, yt - ym)(best_a)
        pred[te] = (Zt @ best_w + ym).cpu().numpy()
    return {"pred": pred, "r2": r2_score(y, pred),
            "mae": float(np.mean(np.abs(y - pred))),
            "alpha": chosen}


def mlp_cv(X: np.ndarray, y: np.ndarray, folds: np.ndarray, hidden: int,
           device: str = "cuda:0", epochs: int = 400, wd: float = 1e-2,
           lr: float = 1e-3, seed: int = 0) -> dict:
    """MLP 探针（P2–P4）。正则用 weight decay（不用 dropout）。"""
    import torch
    import torch.nn as nn
    X = _sanitize(X); y = np.asarray(y, np.float64)
    dev = torch.device(device if torch.cuda.is_available() else "cpu")
    pred = np.full(len(y), np.nan)
    for f in np.unique(folds):
        tr, te = folds != f, folds == f
        mu, sd, keep = _standardizer(X[tr])
        Xtr = torch.tensor(_apply_std(X[tr], mu, sd, keep), device=dev)
        Xte = torch.tensor(_apply_std(X[te], mu, sd, keep), device=dev)
        ym, ys = y[tr].mean(), y[tr].std() + 1e-9
        ytr = torch.tensor((y[tr] - ym) / ys, device=dev, dtype=torch.float32)
        torch.manual_seed(seed)
        net = nn.Sequential(nn.Linear(X.shape[1], hidden), nn.ReLU(),
                            nn.Linear(hidden, 1)).to(dev)
        opt = torch.optim.AdamW(net.parameters(), lr=lr, weight_decay=wd)
        for _ in range(epochs):
            opt.zero_grad()
            loss = ((net(Xtr).squeeze(-1) - ytr) ** 2).mean()
            loss.backward(); opt.step()
        with torch.no_grad():
            pred[te] = net(Xte).squeeze(-1).cpu().numpy() * ys + ym
    return {"pred": pred, "r2": r2_score(y, pred),
            "mae": float(np.mean(np.abs(y - pred)))}


def logistic_cv(X: np.ndarray, y: np.ndarray, folds: np.ndarray,
                device: str = "cuda:0", epochs: int = 300, wd: float = 1e-3,
                lr: float = 3e-2, seed: int = 0) -> dict:
    """token 级线性逻辑回归探针（空间探针 P1）。返回折外 logit。"""
    import torch
    import torch.nn as nn
    X = _sanitize(X); y = np.asarray(y, np.float32)
    dev = torch.device(device if torch.cuda.is_available() else "cpu")
    out = np.full(len(y), np.nan, dtype=np.float64)
    for f in np.unique(folds):
        tr, te = folds != f, folds == f
        mu, sd, keep = _standardizer(X[tr])
        Xtr = torch.tensor(_apply_std(X[tr], mu, sd, keep), device=dev)
        Xte = torch.tensor(_apply_std(X[te], mu, sd, keep), device=dev)
        ytr = torch.tensor(y[tr], device=dev)
        torch.manual_seed(seed)
        lin = nn.Linear(X.shape[1], 1).to(dev)
        opt = torch.optim.AdamW(lin.parameters(), lr=lr, weight_decay=wd)
        lossf = nn.BCEWithLogitsLoss()
        for _ in range(epochs):
            opt.zero_grad()
            lossf(lin(Xtr).squeeze(-1), ytr).backward()
            opt.step()
        with torch.no_grad():
            out[te] = lin(Xte).squeeze(-1).cpu().numpy()
    return {"logit": out, "auc": roc_auc(out, y > 0.5)}


# ---------------------------------------------------------------- selectivity
def selectivity_pair(X: np.ndarray, y: np.ndarray, groups: np.ndarray,
                     seed: int = 0, device: str = "cuda:0") -> dict:
    """Hewitt&Liang 口径的 selectivity（回归版），同时报朴素置换口径。

    两个数各自的含义（D-31：必须分清）：

    1. `selectivity_hl`（**预注册判据用这个**）—— H&L 的忠实对应。
       H&L 的 control task「按 word type 分配随机输出，同 type 的 token 共享该输出」。
       本任务的 type = **source_id**（同一源的全部扰动样本共享一个随机目标）。
       为了让 control task 像 H&L 里那样**原则上可被记忆**（train/test 共享 type），
       这一项在**样本级折**（同源可跨折）上算；real 与 control 用同一套折、同一探针族。
       它测的是「探针有多大能力靠记忆源身份拟合任意标签」。
    2. `selectivity_perm`（附报）—— C1 标签置换对照，在**分组折**（同源不跨折）上算。
       分组折下 control 必然 ≈0，故此数 ≈ R²_grouped，**不是独立证据**。
    """
    rng = np.random.default_rng(seed)
    uniq = np.unique(groups)
    ctrl_val = {g: float(v) for g, v in zip(uniq, rng.normal(size=len(uniq)))}
    y_ctrl = np.array([ctrl_val[g] for g in groups])

    sf = sample_folds(len(y), 5, seed=seed + 1)
    real_ws = ridge_cv(X, y, sf, device=device, groups=groups)
    ctrl_ws = ridge_cv(X, y_ctrl, sf, device=device, groups=groups)

    gf = grouped_folds(groups, 5, seed=seed + 2)
    real_g = ridge_cv(X, y, gf, device=device, groups=groups)
    yperm = rng.permutation(y)
    perm_g = ridge_cv(X, yperm, gf, device=device, groups=groups)

    # control task 的「记忆能力」下界是常数预测器（R²=0）；负 R² 只说明外推更差，
    # 不代表探针记忆得更少。故对 control 取 max(·,0) 再作差，否则 selectivity>R² 无意义。
    return {
        "selectivity_hl": real_ws["r2"] - max(ctrl_ws["r2"], 0.0),
        "r2_within_source": real_ws["r2"], "r2_controltask": ctrl_ws["r2"],
        "selectivity_perm": real_g["r2"] - max(perm_g["r2"], 0.0),
        "r2_grouped": real_g["r2"], "r2_labelperm": perm_g["r2"],
        "mae_grouped": real_g["mae"],
    }


# ---------------------------------------------------------------- MDL
def mdl_online(X: np.ndarray, y_cls: np.ndarray, groups: np.ndarray, K: int,
               device: str = "cuda:0", seed: int = 0, epochs: int = 300,
               wd: float = 1e-2) -> dict:
    """Voita&Titov 式 (4) 的 online/prequential 码长（bits）与 compression。

    样本顺序**按源打乱后整源入块**（保持 no-leak 纪律；原文不分组，此处更严）。
    每块用多类 logistic 回归（线性探针）拟合，块外 −log2 p 求和。
    """
    import torch
    import torch.nn as nn
    X = _sanitize(X); y_cls = np.asarray(y_cls, np.int64)
    n = len(y_cls)
    rng = np.random.default_rng(seed)
    uniq = rng.permutation(np.unique(groups))
    order = np.concatenate([np.flatnonzero(groups == g) for g in uniq])
    Xo, yo = X[order], y_cls[order]
    ts = sorted({max(K, int(round(f * n))) for f in TIMESTEP_FRACS})
    ts = [t for t in ts if t < n] + [n]
    dev = torch.device(device if torch.cuda.is_available() else "cpu")
    total = ts[0] * np.log2(K)             # 第一块用均匀码
    for i in range(len(ts) - 1):
        a, b = ts[i], ts[i + 1]
        mu, sd, keep = _standardizer(Xo[:a])
        Xtr = torch.tensor(_apply_std(Xo[:a], mu, sd, keep), device=dev)
        Xte = torch.tensor(_apply_std(Xo[a:b], mu, sd, keep), device=dev)
        ytr = torch.tensor(yo[:a], device=dev)
        torch.manual_seed(seed)
        lin = nn.Linear(X.shape[1], K).to(dev)
        opt = torch.optim.AdamW(lin.parameters(), lr=3e-2, weight_decay=wd)
        for _ in range(epochs):
            opt.zero_grad()
            nn.functional.cross_entropy(lin(Xtr), ytr).backward()
            opt.step()
        with torch.no_grad():
            p = torch.softmax(lin(Xte), dim=-1)
            # **与均匀码混合**（Alice/Bob 事先约定的编码方案）：早期块只用几个样本
            # 训练，未混合时会对留出样本给出接近 0 的概率、单符号码长爆到几十 bit，
            # 使 compression < 1（实测 0.23–0.30）。混合把单符号码长上界钉在
            # log2(K/eps)，是 online code 的标准工程做法。
            p = (1.0 - MDL_MIX) * p + MDL_MIX / K
            logp = torch.log2(p).cpu().numpy()
        total += float(-logp[np.arange(b - a), yo[a:b]].sum())
    uniform = n * np.log2(K)
    return {"mdl_bits": float(total), "uniform_bits": float(uniform),
            "compression": float(uniform / total) if total > 0 else float("nan"),
            "n": int(n), "K": int(K), "timesteps": ts}


def discretize(y: np.ndarray, K: int = 8) -> np.ndarray:
    """等频分箱（MDL 用）。"""
    q = np.quantile(y, np.linspace(0, 1, K + 1)[1:-1])
    return np.digitize(y, q).astype(np.int64)
