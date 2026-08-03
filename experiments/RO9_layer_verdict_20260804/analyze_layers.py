"""RO9-L：早期层 AUC 判别 —— 决定 RO-9 生死（D-32）。

问题：canonical 读出（L8–15 均值）下 ρ_region_opp 中位 0.884 判 FAIL，但分层曲线显示
早期层 ρ_region_opp 显著更低（L0=0.045 / L4=0.270 / L3=0.426）。低相关有两解：
(i) 那些层的 s 真随指令变（RO-9 只是层选错，可救）；(ii) 那些层的 s 是白噪声（低 ρ 平凡，
RO-9 作为 where 读出判死）。判别只需一个数：那些层 s 对**指令所指区域**的 AUC。

预注册判别规则（本文件 VERDICT_RULE 常量，写在跑数之前）：
  救回 = 存在层 L 使得  AUC_target(L) ≥ 0.65  且  ρ_region_opp(L) < 0.30  且  非白噪声
  判死 = 不存在这样的层（AUC≈0.5 或场是白噪声）

三组数（每层 × 三 token，light 为主叙事，另两为 D-22 附录）：
 (a) AUC_target：reg_a 指令点名主体 → 标签 = SAM3 主体掩膜；reg_b 指令点名补集 →
     标签 = 主体掩膜的补集。注意 AUC(s_b, ~M) = 1 − AUC(s_b, M)，因此
     AUC_target 高要求「reg_a 的 s 压主体、reg_b 的 s 压背景」，是真正的区域条件性读数。
     同时报配对量 ΔAUC(L) = AUC(s_a, M) − AUC(s_b, M)（逐图配对，免疫逐图主体先验）。
     **分层**：region_b_kind=background（165 源，reg_b 字面就是「背景」＝主体补集）为
     主判据口径；region_b_kind=spatial（49 源，reg_b 是「左半/右半」，与主体补集只是
     近似）单列，只报不判。两个口径都算，避免事后挑口径。
 (b) ρ_region_opp(L)：corr(s_a(L), s_b(L))，valid-mask 口径（与 G1 一致）。
 (c) 结构性：Moran's I（rook 邻接，valid 格）+ 有效秩 erank = exp(H(σ/Σσ))，
     各配一条**同支撑空间置换零模型**（把 valid 格内的值随机重排）。白噪声的签名 =
     Moran's I ≈ 零模型水平 且 erank ≈ 零模型水平。
     另附**跨指令改述可靠性** ρ_syn(L)（主批 syn_a vs syn_b）与错位对照 ρ_shuf(L)：
     若某层连「同义改述」都相关为 0，那层的低 ρ_region_opp 就不是指令条件性。
 (d) 提示词长度假说的证伪性诊断（预注册为**探索项，不进判据**）：reg_a（长的主体描述）
     与 reg_b（短的「背景」）token 数差很大，GL token 的绝对位置随之变化，早期层
     attention logit 可能被 RoPE 位置项主导。逐层报 Spearman(ρ_region_opp_i,
     |Δtoken_len|_i)。**实测该假说不成立**（L0 −0.036 / L4 −0.028，≈0），如实登记为
     null 结果；早期层低 ρ 的成因本实验只排除了长度解释，未给出正面机制。

D-22：跨 token 两两空间 ρ（canonical + 逐层）+ 三 token 各自 AUC（逐层）。

用法：.venv-lens/bin/python analyze_layers.py [--out DIR] [--n-perm 32]
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from itertools import combinations
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
G1 = REPO / "experiments" / "G1_s_identifiability_20260803"
sys.path.insert(0, str(REPO / "tools" / "scache"))
sys.path.insert(0, str(REPO / "tools" / "readout"))
sys.path.insert(0, str(G1))

GRID = 16
N_LAYERS = 24
TOKEN_TAGS = ("light", "colortemp", "colormixer")
CANON = slice(8, 16)
MASK_BIN_THR = 0.5
EARLY_BAND = (0, 5)          # D-32 点名的「早期层」

# ---- 预注册判别规则（跑数前写死；REPORT 逐条对表）----
VERDICT_RULE = {
    "auc_target_min": 0.65,
    "rho_region_opp_max": 0.30,
    "white_noise_moran_max": 0.10,        # Moran's I 中位 ≤ 此值 ⇒ 空间无结构
    "white_noise_erank_ratio_min": 0.80,  # erank/erank_null ≥ 此值 ⇒ 谱形同白噪声
    "note": ("救回 = ∃L: AUC_target(L) ≥ 0.65 ∧ ρ_region_opp(L) < 0.30 ∧ ¬白噪声(L)；"
             "白噪声(L) = Moran's I 中位 ≤ 0.10 且 erank/erank_null ≥ 0.80"),
}


# ---------------- 基础统计 ----------------
def pearson(a, b, mask=None) -> float:
    a = np.asarray(a).ravel().astype(np.float64)
    b = np.asarray(b).ravel().astype(np.float64)
    if mask is not None:
        m = np.asarray(mask).ravel().astype(bool)
        if m.sum() >= 8:
            a, b = a[m], b[m]
    if a.std() == 0 or b.std() == 0:
        return float("nan")
    return float(np.corrcoef(a, b)[0, 1])


def roc_auc(scores, labels) -> float:
    from scipy.stats import rankdata
    s = np.asarray(scores, dtype=np.float64).ravel()
    y = np.asarray(labels).ravel().astype(bool)
    n_pos, n_neg = int(y.sum()), int((~y).sum())
    if n_pos == 0 or n_neg == 0:
        return float("nan")
    r = rankdata(s)
    return float((r[y].sum() - n_pos * (n_pos + 1) / 2.0) / (n_pos * n_neg))


def morans_i(field: np.ndarray, valid: np.ndarray) -> float:
    """Moran's I，rook（4-邻）邻接，只在 valid 格之间连边。

    I = (n/S0) · Σ_ij w_ij z_i z_j / Σ_i z_i²，z = x − mean(x[valid])。
    白噪声期望 E[I] = −1/(n−1) ≈ 0（n≈160 时 ≈ −0.006）。
    """
    v = valid.astype(bool)
    n = int(v.sum())
    if n < 16:
        return float("nan")
    z = np.where(v, field - field[v].mean(), 0.0)
    denom = float((z[v] ** 2).sum())
    if denom == 0:
        return float("nan")
    # 水平邻接
    hv = v[:, :-1] & v[:, 1:]
    num = 2.0 * float((z[:, :-1] * z[:, 1:] * hv).sum())
    s0 = 2.0 * float(hv.sum())
    # 垂直邻接
    vv = v[:-1, :] & v[1:, :]
    num += 2.0 * float((z[:-1, :] * z[1:, :] * vv).sum())
    s0 += 2.0 * float(vv.sum())
    if s0 == 0:
        return float("nan")
    return float((n / s0) * num / denom)


def effective_rank(field: np.ndarray, valid: np.ndarray) -> float:
    """有效秩 erank = exp(−Σ p_k log p_k)，p = σ_k / Σσ（16×16 场的奇异值谱）。

    invalid 格置 0（先减 valid 均值），使 pad 黑边不贡献能量。
    白噪声 16×16 的 erank ≈ 11–13；平滑低秩场 ≈ 1–4。
    """
    v = valid.astype(bool)
    if v.sum() < 16:
        return float("nan")
    z = np.where(v, field - field[v].mean(), 0.0)
    sv = np.linalg.svd(z, compute_uv=False)
    tot = sv.sum()
    if tot <= 0:
        return float("nan")
    p = sv / tot
    p = p[p > 0]
    return float(np.exp(-(p * np.log(p)).sum()))


def structure_with_null(field, valid, rng, n_perm: int):
    """(Moran's I, erank, I_null_mean, erank_null_mean) —— 零模型 = valid 格内空间置换。"""
    i_obs = morans_i(field, valid)
    e_obs = effective_rank(field, valid)
    v = valid.astype(bool)
    vals = field[v]
    i_null, e_null = [], []
    for _ in range(n_perm):
        f = np.zeros_like(field)
        f[v] = rng.permutation(vals)
        i_null.append(morans_i(f, v))
        e_null.append(effective_rank(f, v))
    return i_obs, e_obs, float(np.nanmean(i_null)), float(np.nanmean(e_null))


def boot_median_ci(vals, rng, n_boot: int = 2000, lo: float = 2.5, hi: float = 97.5):
    a = np.asarray([x for x in vals if np.isfinite(x)], dtype=np.float64)
    if a.size < 5:
        return [float("nan"), float("nan")]
    idx = rng.integers(0, a.size, size=(n_boot, a.size))
    meds = np.median(a[idx], axis=1)
    return [float(np.percentile(meds, lo)), float(np.percentile(meds, hi))]


def med(vals) -> float:
    a = [x for x in vals if np.isfinite(x)]
    return float(np.median(a)) if a else float("nan")


# ---------------- 主流程 ----------------
def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=str(HERE))
    ap.add_argument("--n-perm", type=int, default=32, help="空间置换零模型次数（每场每层）")
    ap.add_argument("--viz-n", type=int, default=4)
    ap.add_argument("--seed", type=int, default=20260804)
    ap.add_argument("--limit", type=int, default=0, help="冒烟用：只取前 N 个区域源")
    args = ap.parse_args()

    from analyze_g1 import SubjectMaskBank, luma_valid_from_image
    from api import instr_hash

    # (d) 长度伪影诊断用的分词器：优先真 tokenizer（CPU，不加载权重），失败退白空格切分
    tok_enc = None
    try:
        from transformers import AutoTokenizer
        _tk = AutoTokenizer.from_pretrained("/home/bc/data/models/VeraRetouch",
                                            use_fast=True)
        tok_enc = lambda s: _tk(s, add_special_tokens=False).input_ids
        tok_src = "AutoTokenizer(/home/bc/data/models/VeraRetouch)"
    except Exception as exc:                       # 兜底：不因分词器缺席阻断主判据
        print(f"[warn] tokenizer 不可用，长度诊断退白空格切分: {exc}")
        tok_enc = lambda s: s.split()
        tok_src = "whitespace split (fallback)"
    print(f"[tok] {tok_src}")

    rng = np.random.default_rng(args.seed)
    out = Path(args.out)
    (out / "viz").mkdir(parents=True, exist_ok=True)

    stacks: dict[str, Path] = {}
    for rd in ("run_regfull", "run_regsmoke20", "run_full", "run_smoke30", "run_shufctrl"):
        d = G1 / rd / "stacks"
        if d.is_dir():
            for p in d.glob("*.npz"):
                stacks[p.stem] = p
    print(f"[data] stacks={len(stacks)}")

    bank = SubjectMaskBank()
    region = json.loads((G1 / "config" / "g1_region_opp.json").read_text())
    if args.limit:
        region = region[: args.limit]

    # 逐层 × 逐 token 累积器（*_kind 并行记录 region_b_kind 以便分层）
    auc_a_subj = {t: [[] for _ in range(N_LAYERS)] for t in TOKEN_TAGS}   # AUC(s_a, M)
    auc_b_subj = {t: [[] for _ in range(N_LAYERS)] for t in TOKEN_TAGS}   # AUC(s_b, M)
    rho_reg = [[] for _ in range(N_LAYERS)]
    case_kind: list[str] = []            # 与 auc_*/rho_reg 逐层列表同序
    dlen_tok: list[float] = []           # |token 数差|（reg_a vs reg_b）
    cross_tok_layer = {f"{a}~{b}": [[] for _ in range(N_LAYERS)]
                       for a, b in combinations(TOKEN_TAGS, 2)}
    moran, erank, moran_null, erank_null = ([[] for _ in range(N_LAYERS)] for _ in range(4))
    auc_luma_all, subj_area = [], []
    per_case: list[dict] = []
    keep_fields: dict[str, dict] = {}

    n_used = 0
    for rs in region:
        pa = stacks.get(f"{rs['img_id']}__{instr_hash(rs['instructions']['reg_a'])}")
        pb = stacks.get(f"{rs['img_id']}__{instr_hash(rs['instructions']['reg_b'])}")
        if pa is None or pb is None:
            continue
        got = bank.mask_grid(rs["img_id"])
        if got is None:
            continue
        m16, _ = got
        with np.load(pa) as z:
            stk_a = z["stack"].astype(np.float32)
        with np.load(pb) as z:
            stk_b = z["stack"].astype(np.float32)
        luma, valid = luma_valid_from_image(rs["img_path"])
        lab = (m16 >= MASK_BIN_THR)[valid]
        if lab.all() or not lab.any():
            continue          # 主体覆盖全部/零个 valid 格 ⇒ AUC 无定义
        n_used += 1
        subj_area.append(float(lab.mean()))
        auc_luma_all.append(roc_auc(luma[valid], lab))
        case_kind.append(rs["region_b_kind"])
        na = len(tok_enc(rs["instructions"]["reg_a"]))
        nb = len(tok_enc(rs["instructions"]["reg_b"]))
        dlen_tok.append(abs(na - nb))

        case = {"img_id": rs["img_id"], "pool": rs["pool"],
                "direction": rs["direction"], "region_b_kind": rs["region_b_kind"],
                "subject_region": rs["subject_region"], "conf": rs["winner_confidence"],
                "subj_area16": float(lab.mean()),
                "ntok_a": na, "ntok_b": nb, "dlen_tok": abs(na - nb)}
        for li in range(N_LAYERS):
            fa0 = stk_a[0][li]
            rho_reg[li].append(pearson(fa0, stk_b[0][li], valid))
            for ti, tag in enumerate(TOKEN_TAGS):
                fa, fb = stk_a[ti][li], stk_b[ti][li]
                auc_a_subj[tag][li].append(roc_auc(fa[valid], lab))   # reg_a 目标 = 主体
                auc_b_subj[tag][li].append(roc_auc(fb[valid], lab))   # reg_b 目标 = 补集
            for ta, tb in combinations(TOKEN_TAGS, 2):
                cross_tok_layer[f"{ta}~{tb}"][li].append(
                    pearson(stk_a[TOKEN_TAGS.index(ta)][li],
                            stk_a[TOKEN_TAGS.index(tb)][li], valid))
            io_, eo, inull, enull = structure_with_null(fa0, valid, rng, args.n_perm)
            moran[li].append(io_)
            erank[li].append(eo)
            moran_null[li].append(inull)
            erank_null[li].append(enull)
        for li in (0, 3, 4, 11, 22):
            case[f"auc_a_L{li}"] = auc_a_subj["light"][li][-1]
            case[f"auc_b_L{li}"] = auc_b_subj["light"][li][-1]
            case[f"rho_L{li}"] = rho_reg[li][-1]
        case["auc_canon_a"] = roc_auc(stk_a[0][CANON].mean(axis=0)[valid], lab)
        case["auc_canon_b"] = roc_auc(stk_b[0][CANON].mean(axis=0)[valid], lab)
        case["rho_canon"] = pearson(stk_a[0][CANON].mean(axis=0),
                                    stk_b[0][CANON].mean(axis=0), valid)
        per_case.append(case)
        keep_fields[rs["img_id"]] = {
            "sample": rs, "valid": valid, "luma": luma, "subj": m16,
            "a": stk_a[0], "b": stk_b[0],
        }
        if n_used % 25 == 0:
            print(f"[region] {n_used}/{len(region)}")

    print(f"[region] used {n_used} / {len(region)}")

    # ---- 主批（改述可靠性 ρ_syn / 方向对立 ρ_opp）与错位对照 ρ_shuf 的逐层曲线 ----
    samples = json.loads((G1 / "config" / "g1_samples.json").read_text())
    if args.limit:
        samples = samples[: args.limit]
    lay_syn = [[] for _ in range(N_LAYERS)]
    lay_opp = [[] for _ in range(N_LAYERS)]
    main_valid: dict[str, tuple] = {}
    main_stack: dict[str, np.ndarray] = {}
    for s in samples:
        ih = {k: instr_hash(v) for k, v in s["instructions"].items()}
        ps = {k: stacks.get(f"{s['img_id']}__{h}") for k, h in ih.items()}
        if any(v is None for v in ps.values()):
            continue
        arr = {}
        for k, p in ps.items():
            with np.load(p) as z:
                arr[k] = z["stack"].astype(np.float32)[0]
        _luma, valid = luma_valid_from_image(s["img_path"])
        main_valid[s["img_id"]] = (valid,)
        main_stack[s["img_id"]] = arr["syn_a"]
        for li in range(N_LAYERS):
            lay_syn[li].append(pearson(arr["syn_a"][li], arr["syn_b"][li], valid))
            lay_opp[li].append(pearson(arr["syn_a"][li], arr["opp"][li], valid))
    lay_shuf = [[] for _ in range(N_LAYERS)]
    shuf_path = G1 / "config" / "g1_shuffle_ctrl.json"
    if shuf_path.is_file():
        for sc in json.loads(shuf_path.read_text()):
            p = stacks.get(f"{sc['img_id']}__{instr_hash(sc['instructions']['shuf'])}")
            if p is None or sc["img_id"] not in main_stack:
                continue
            with np.load(p) as z:
                sh = z["stack"].astype(np.float32)[0]
            valid = main_valid[sc["img_id"]][0]
            base = main_stack[sc["img_id"]]
            for li in range(N_LAYERS):
                lay_shuf[li].append(pearson(base[li], sh[li], valid))

    # ---- 逐层汇总 ----
    from scipy.stats import spearmanr
    kind_arr = np.array(case_kind)
    dlen_arr = np.array(dlen_tok, dtype=float)
    sel = {"all": np.ones(len(kind_arr), bool),
           "background": kind_arr == "background",
           "spatial": kind_arr == "spatial"}

    def auc_target_list(tag: str, li: int, m: np.ndarray) -> list[float]:
        """目标区域 AUC 混池：reg_a 条件取 AUC(s_a, M)，reg_b 条件取 1 − AUC(s_b, M)。"""
        a = np.array(auc_a_subj[tag][li])[m]
        b = np.array(auc_b_subj[tag][li])[m]
        return list(a) + list(1.0 - b)

    layers = []
    for li in range(N_LAYERS):
        m_all, m_bg = sel["all"], sel["background"]
        da = (np.array(auc_a_subj["light"][li]) - np.array(auc_b_subj["light"][li]))
        rr = np.array(rho_reg[li])
        r = {
            "layer": li,
            # 主判据口径：background 子集（reg_b 字面 = 主体补集）
            "auc_target_light_bg": med(auc_target_list("light", li, m_bg)),
            "auc_target_light_bg_ci95": boot_median_ci(auc_target_list("light", li, m_bg), rng),
            "auc_target_light_all": med(auc_target_list("light", li, m_all)),
            "auc_target_light_spatial": med(auc_target_list("light", li, sel["spatial"])),
            "auc_target_percase_mean_bg": med(
                list((np.array(auc_a_subj["light"][li])[m_bg]
                      + 1.0 - np.array(auc_b_subj["light"][li])[m_bg]) / 2.0)),
            "auc_rega_subject_light": med(np.array(auc_a_subj["light"][li])[m_all]),
            "auc_regb_subject_light": med(np.array(auc_b_subj["light"][li])[m_all]),
            "auc_rega_subject_light_bg": med(np.array(auc_a_subj["light"][li])[m_bg]),
            "auc_regb_subject_light_bg": med(np.array(auc_b_subj["light"][li])[m_bg]),
            "delta_auc_paired_light": med(da[m_all]),
            "delta_auc_paired_ci95": boot_median_ci(list(da[m_all]), rng),
            "delta_auc_paired_light_bg": med(da[m_bg]),
            "rho_region_opp": med(rr[m_all]),
            "rho_region_opp_ci95": boot_median_ci(list(rr[m_all]), rng),
            "rho_region_opp_bg": med(rr[m_bg]),
            "rho_syn": med(lay_syn[li]),
            "rho_opp": med(lay_opp[li]),
            "rho_shuf": med(lay_shuf[li]) if lay_shuf[li] else None,
            "morans_i": med(moran[li]),
            "morans_i_null": med(moran_null[li]),
            "erank": med(erank[li]),
            "erank_null": med(erank_null[li]),
        }
        r["sep_syn_minus_region"] = r["rho_syn"] - r["rho_region_opp"]
        r["erank_ratio"] = (r["erank"] / r["erank_null"]
                            if np.isfinite(r["erank"]) and r["erank_null"] else float("nan"))
        r["is_white_noise"] = bool(
            r["morans_i"] <= VERDICT_RULE["white_noise_moran_max"]
            and r["erank_ratio"] >= VERDICT_RULE["white_noise_erank_ratio_min"])
        # (d) 长度伪影：ρ_region_opp 与 |Δtoken 数| 的 Spearman（负 = 长度差越大 ρ 越低）
        ok = np.isfinite(rr) & np.isfinite(dlen_arr)
        r["spearman_rho_vs_dlen"] = (float(getattr(spearmanr(rr[ok], dlen_arr[ok]),
                                                   "statistic")) if ok.sum() >= 10
                                     else float("nan"))
        for tag in TOKEN_TAGS:
            r[f"auc_target_{tag}_bg"] = med(auc_target_list(tag, li, m_bg))
            r[f"delta_auc_{tag}"] = med(np.array(auc_a_subj[tag][li])
                                        - np.array(auc_b_subj[tag][li]))
        for pair, v in cross_tok_layer.items():
            r[f"rho_tok_{pair}"] = med(v[li])
        # 预注册救回判据：主口径（background）与全口径任一达标即视为「有救」
        r["rescue"] = bool(
            max(r["auc_target_light_bg"], r["auc_target_light_all"])
            >= VERDICT_RULE["auc_target_min"]
            and r["rho_region_opp"] < VERDICT_RULE["rho_region_opp_max"]
            and not r["is_white_noise"])
        layers.append(r)

    # canonical 区间（L8–15 均值场）作对照行
    bgm = [c["region_b_kind"] == "background" for c in per_case]
    canon_rows = {
        "auc_target_light_all": med([c["auc_canon_a"] for c in per_case]
                                    + [1.0 - c["auc_canon_b"] for c in per_case]),
        "auc_target_light_bg": med([c["auc_canon_a"] for c, k in zip(per_case, bgm) if k]
                                   + [1.0 - c["auc_canon_b"]
                                      for c, k in zip(per_case, bgm) if k]),
        "auc_rega_subject_light": med([c["auc_canon_a"] for c in per_case]),
        "auc_regb_subject_light": med([c["auc_canon_b"] for c in per_case]),
        "delta_auc_paired_light": med([c["auc_canon_a"] - c["auc_canon_b"]
                                       for c in per_case]),
        "rho_region_opp": med([c["rho_canon"] for c in per_case]),
    }

    rescue_layers = [r["layer"] for r in layers if r["rescue"]]
    early = [r for r in layers if EARLY_BAND[0] <= r["layer"] <= EARLY_BAND[1]]
    low_rho = [r for r in layers if r["rho_region_opp"] < VERDICT_RULE["rho_region_opp_max"]]
    best_i = int(max(range(N_LAYERS), key=lambda i: layers[i]["auc_target_light_bg"]))
    verdict = {
        "rule": VERDICT_RULE,
        "rescue_layers": rescue_layers,
        "n_layers_rho_below_0.3": len(low_rho),
        "layers_rho_below_0.3": [r["layer"] for r in low_rho],
        "auc_target_bg_at_low_rho_layers": {r["layer"]: round(r["auc_target_light_bg"], 3)
                                            for r in low_rho},
        "auc_target_all_at_low_rho_layers": {r["layer"]: round(r["auc_target_light_all"], 3)
                                             for r in low_rho},
        "white_noise_at_low_rho_layers": {r["layer"]: r["is_white_noise"] for r in low_rho},
        "best_auc_target_layer": best_i,
        "best_auc_target_value_bg": layers[best_i]["auc_target_light_bg"],
        "best_auc_target_ci95_bg": layers[best_i]["auc_target_light_bg_ci95"],
        "max_abs_delta_auc_paired_light": max(abs(r["delta_auc_paired_light"])
                                              for r in layers),
        "max_sep_syn_minus_region": max(r["sep_syn_minus_region"] for r in layers),
        "verdict": "RESCUE" if rescue_layers else "DEAD",
        "recommended_layers": rescue_layers or None,
        "early_band_summary": [{k: r[k] for k in
                                ("layer", "auc_target_light_bg", "auc_target_light_all",
                                 "delta_auc_paired_light", "rho_region_opp",
                                 "rho_syn", "sep_syn_minus_region", "morans_i",
                                 "morans_i_null", "erank", "erank_null", "erank_ratio",
                                 "is_white_noise", "spearman_rho_vs_dlen", "rescue")}
                               for r in early],
    }

    git = subprocess.run(["git", "rev-parse", "--short", "HEAD"], cwd=REPO,
                         capture_output=True, text=True).stdout.strip()
    metrics = {
        "experiment": "RO9_layer_verdict", "date": "2026-08-04",
        "git_commit": git, "seed": args.seed, "n_perm": args.n_perm,
        "source": "experiments/G1_s_identifiability_20260803 stacks（零额外前向）",
        "n_region_sources": n_used, "n_main_sources": len(main_stack),
        "n_shuffle_sources": len(lay_shuf[0]) if lay_shuf[0] else 0,
        "n_region_by_kind": {k: int(v.sum()) for k, v in sel.items()},
        "tokenizer": tok_src,
        "dlen_tok_median": float(np.median(dlen_arr)) if len(dlen_arr) else None,
        "subject_area16_median": med(subj_area),
        "auc_luma_baseline_median": med(auc_luma_all),
        "verdict": verdict,
        "canonical_L8_15": canon_rows,
        "layers": layers,
        "per_case": per_case,
    }
    # D-22(a) canonical 跨 token ρ（用 L8–15 均值场重算，口径与 G1 一致）
    ct_canon = {f"{a}~{b}": [] for a, b in combinations(TOKEN_TAGS, 2)}
    d22_auc_canon = {t: {"rega": [], "regb": []} for t in TOKEN_TAGS}
    for rs in region:
        pa = stacks.get(f"{rs['img_id']}__{instr_hash(rs['instructions']['reg_a'])}")
        pb = stacks.get(f"{rs['img_id']}__{instr_hash(rs['instructions']['reg_b'])}")
        got = bank.mask_grid(rs["img_id"])
        if pa is None or pb is None or got is None:
            continue
        with np.load(pa) as z:
            sa = z["stack"].astype(np.float32)
        with np.load(pb) as z:
            sb = z["stack"].astype(np.float32)
        _l, valid = luma_valid_from_image(rs["img_path"])
        lab = (got[0] >= MASK_BIN_THR)[valid]
        if lab.all() or not lab.any():
            continue
        ca = {t: sa[i][CANON].mean(axis=0) for i, t in enumerate(TOKEN_TAGS)}
        cb = {t: sb[i][CANON].mean(axis=0) for i, t in enumerate(TOKEN_TAGS)}
        for ta, tb in combinations(TOKEN_TAGS, 2):
            ct_canon[f"{ta}~{tb}"].append(pearson(ca[ta], ca[tb], valid))
        for t in TOKEN_TAGS:
            d22_auc_canon[t]["rega"].append(roc_auc(ca[t][valid], lab))
            d22_auc_canon[t]["regb"].append(roc_auc(cb[t][valid], lab))
    metrics["d22_cross_token_rho_canonical"] = {k: med(v) for k, v in ct_canon.items()}
    metrics["d22_token_auc_canonical"] = {
        t: {"auc_rega_subject": med(d["rega"]), "auc_regb_subject": med(d["regb"]),
            "auc_target": med(d["rega"] + [1.0 - x for x in d["regb"]]),
            "delta_auc_paired": med([a - b for a, b in zip(d["rega"], d["regb"])])}
        for t, d in d22_auc_canon.items()}
    metrics["d22_cross_token_rho_layer"] = {
        pair: [med(v[li]) for li in range(N_LAYERS)] for pair, v in cross_tok_layer.items()}

    (out / "metrics.json").write_text(json.dumps(metrics, indent=1, ensure_ascii=False))
    print(json.dumps({"verdict": verdict, "canonical": canon_rows,
                      "d22_token_auc": metrics["d22_token_auc_canonical"],
                      "d22_cross_token": metrics["d22_cross_token_rho_canonical"]},
                     indent=1, ensure_ascii=False, default=float))

    # ---------------- viz ----------------
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from PIL import Image

    L = np.arange(N_LAYERS)
    g = lambda k: np.array([r[k] for r in layers], dtype=float)

    fig, ax = plt.subplots(3, 1, figsize=(11.5, 12.0), sharex=True,
                           gridspec_kw={"height_ratios": [1, 1.15, 1.15]})

    # 面板 A：s 对主体掩膜的 AUC，两条指令各一条 —— 两条几乎重合 = s 与指令无关
    aA = ax[0]
    aA.plot(L, g("auc_rega_subject_light"), "o-", color="#1f77b4", lw=2,
            label="AUC(s | reg_a「点名主体」, 标签=主体掩膜)")
    aA.plot(L, g("auc_regb_subject_light"), "s-", color="#ff7f0e", lw=2,
            label="AUC(s | reg_b「点名补集」, 标签=主体掩膜)")
    aA.axhline(0.5, color="k", ls=":", lw=1)
    aA.axhline(metrics["auc_luma_baseline_median"], color="#999", ls="-.", lw=1,
               label=f"luma 基线 {metrics['auc_luma_baseline_median']:.3f}")
    aA.set_ylabel("AUC vs 主体掩膜")
    aA.set_ylim(0.40, 0.90)
    aA.set_title("A｜两条互斥指令下 s 对主体掩膜的 AUC 逐层重合 ⇒ "
                 "s 是图像驱动的主体显著场，不随指令改指区域", fontsize=11)
    aA.legend(fontsize=8, loc="upper left")
    aA.grid(alpha=0.25)

    a0 = ax[1]
    a0.plot(L, g("auc_target_light_bg"), "o-", color="#1f77b4", lw=2,
            label="AUC_target (light, background 子集=主判据)")
    ci = np.array([r["auc_target_light_bg_ci95"] for r in layers], dtype=float)
    a0.fill_between(L, ci[:, 0], ci[:, 1], color="#1f77b4", alpha=0.18, lw=0)
    a0.plot(L, g("auc_target_light_all"), "o--", color="#4c9fd0", lw=1.2, ms=4,
            label=f"AUC_target (light, 全 {n_used} 源)")
    a0.plot(L, g("auc_target_colortemp_bg"), "s--", color="#8fbcd4", ms=4, lw=1,
            label="AUC_target (colortemp, D-22 附录)")
    a0.plot(L, g("auc_target_colormixer_bg"), "^--", color="#c7dcea", ms=4, lw=1,
            label="AUC_target (colormixer, D-22 附录)")
    a0.axhline(0.5, color="k", ls=":", lw=1)
    a0.axhline(VERDICT_RULE["auc_target_min"], color="#2ca02c", ls="--", lw=1.6,
               label=f"预注册救回线 AUC ≥ {VERDICT_RULE['auc_target_min']}")
    a0b = a0.twinx()
    a0b.plot(L, g("delta_auc_paired_light"), "d-", color="#d62728", lw=1.6, ms=5,
             label="ΔAUC 配对 = AUC(s_a,M) − AUC(s_b,M)")
    a0b.axhline(0.0, color="#d62728", ls=":", lw=0.8)
    a0b.set_ylabel("ΔAUC（配对，红）", color="#d62728")
    a0b.set_ylim(-0.28, 0.12)      # 刻意错开：不让 ΔAUC=0 与左轴 0.65 判据线重合
    a0.set_ylabel("AUC_target（蓝）")
    a0.set_ylim(0.35, 0.95)
    a0.set_title("B｜判据量：目标区域 AUC（reg_a→主体 / reg_b→补集）—— "
                 "24/24 层低于救回线 0.65，配对 ΔAUC ≈ 0", fontsize=11)
    h1, l1 = a0.get_legend_handles_labels()
    h2, l2 = a0b.get_legend_handles_labels()
    a0.legend(h1 + h2, l1 + l2, fontsize=7.5, loc="upper left", ncol=2)
    a0.grid(alpha=0.25)

    a1 = ax[2]
    a1.plot(L, g("rho_region_opp"), "o-", color="#9467bd", lw=2, label="ρ_region_opp (reg_a vs reg_b)")
    ci = np.array([r["rho_region_opp_ci95"] for r in layers], dtype=float)
    a1.fill_between(L, ci[:, 0], ci[:, 1], color="#9467bd", alpha=0.18, lw=0)
    a1.plot(L, g("rho_syn"), "s-", color="#ff7f0e", lw=1.6, label="ρ_syn（同义改述可靠性）")
    if layers[0]["rho_shuf"] is not None:
        a1.plot(L, g("rho_shuf"), "^-", color="#7f7f7f", lw=1.2, label="ρ_shuf（跨图错位指令）")
    a1.axhline(VERDICT_RULE["rho_region_opp_max"], color="#2ca02c", ls="--", lw=1.4,
               label=f"预注册判据 ρ < {VERDICT_RULE['rho_region_opp_max']}")
    a1.axhline(0.0, color="k", ls=":", lw=0.8)
    a1b = a1.twinx()
    a1b.plot(L, g("morans_i"), "v-", color="#17becf", lw=1.4, label="Moran's I（空间自相关）")
    a1b.plot(L, g("erank") / g("erank_null"), "x-", color="#e377c2", lw=1.2,
             label="erank / erank_null（1.0 = 白噪声谱）")
    a1b.axhline(1.0, color="#e377c2", ls=":", lw=0.8)
    a1b.set_ylabel("Moran's I / erank比", color="#17becf")
    a1b.set_ylim(-0.1, 1.15)
    a1.set_ylabel("Pearson ρ")
    a1.set_ylim(-0.1, 1.05)
    a1.set_xlabel("layer")
    a1.set_xticks(L)
    a1.set_title("C｜ρ_region_opp 低的层（L0/L3/L4），同义改述 ρ_syn 同样塌到 0 "
                 "⇒ 低相关是「场不可复现」，不是「随指令改指」", fontsize=11)
    h1, l1 = a1.get_legend_handles_labels()
    h2, l2 = a1b.get_legend_handles_labels()
    a1.legend(h1 + h2, l1 + l2, fontsize=7.5, loc="center left", ncol=2)
    a1.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(out / "viz" / "layer_curves_auc_rho.png", dpi=140, bbox_inches="tight")
    plt.close(fig)

    # 图 2：早期层低 ρ 的一个候选成因（提示词长度/位置）——**该假说未被数据支持**，
    # 图与文件名如实登记为 null 结果，不得写成「已证实是长度伪影」。
    fig, axd = plt.subplots(1, 3, figsize=(14, 4.4))
    for k, li in enumerate((0, 4, 11)):
        rr = np.array(rho_reg[li], dtype=float)
        axd[k].scatter(dlen_arr, rr, s=14, alpha=0.55,
                       c=["#1f77b4" if x == "background" else "#d62728" for x in case_kind])
        axd[k].set_xlabel("|Δtoken 数|  (reg_a − reg_b)")
        axd[k].set_ylabel("ρ_region_opp（逐源）")
        axd[k].set_title(f"L{li}: Spearman(ρ, |Δlen|) = "
                         f"{layers[li]['spearman_rho_vs_dlen']:+.3f}", fontsize=10)
        axd[k].axhline(0.3, color="#2ca02c", ls="--", lw=1)
        axd[k].grid(alpha=0.25)
    fig.suptitle(
        "【null 结果】「早期层低 ρ = 提示词长度/RoPE 位置伪影」假说**不成立**：\n"
        f"L0 Spearman(ρ,|Δlen|)={layers[0]['spearman_rho_vs_dlen']:+.3f}、"
        f"L4 {layers[4]['spearman_rho_vs_dlen']:+.3f}（≈0，散点无趋势）；"
        f"canonical L11 反而略强（{layers[11]['spearman_rho_vs_dlen']:+.3f}）但 ρ 已饱和在 1.0。"
        "⇒ 早期层低 ρ 另有来源，本实验只能排除长度解释（蓝=background, 红=spatial）",
        fontsize=9.5)
    fig.tight_layout()
    fig.savefig(out / "viz" / "diag_prompt_length_NULL.png", dpi=130, bbox_inches="tight")
    plt.close(fig)

    # ---- 案例图：早期层 vs canonical/后期层 s 场并排（reg_a 上行 / reg_b 下行）----
    SHOW_L = [0, 3, 4, 11, 22]
    AREA_LO, AREA_HI = 0.04, 0.60   # 主体过小/过大时 16×16 上的 AUC 方差爆炸，两端同规则筛

    def early_rescue_score(c: dict) -> float:
        """逐案例的「救回证据强度」：早期层里 ρ<0.3 且两条指令都指对目标的最好成绩。"""
        best = -1.0
        for li in (0, 3, 4):
            if c[f"rho_L{li}"] >= VERDICT_RULE["rho_region_opp_max"]:
                continue
            best = max(best, min(c[f"auc_a_L{li}"], 1.0 - c[f"auc_b_L{li}"]))
        return best

    elig = [c for c in per_case if AREA_LO <= c["subj_area16"] <= AREA_HI]
    e_rank = sorted(elig, key=early_rescue_score)
    s_rank = sorted(elig, key=lambda c: c["auc_canon_a"])
    picks = []
    for c in e_rank[-args.viz_n:]:
        ok = early_rescue_score(c) >= VERDICT_RULE["auc_target_min"]
        picks.append(("success_early" if ok else "bestcase_early", c["img_id"]))
    picks += [("failure_early", c["img_id"]) for c in e_rank[: args.viz_n]]
    picks += [("success_saliency", c["img_id"]) for c in s_rank[-3:]]
    picks += [("failure_saliency", c["img_id"]) for c in s_rank[:3]]

    for kind, img_id in picks:
        f = keep_fields[img_id]
        rs, valid, subj = f["sample"], f["valid"], f["subj"]
        c = next(x for x in per_case if x["img_id"] == img_id)
        ncol = 1 + len(SHOW_L)
        fig, axes = plt.subplots(2, ncol, figsize=(2.35 * ncol, 5.6))
        try:
            axes[0][0].imshow(Image.open(rs["img_path"]).convert("RGB"))
        except Exception:
            pass
        axes[0][0].set_title(f"{img_id}\n{rs['pool']} / {rs['direction']} / "
                             f"regB={rs['region_b_kind']}", fontsize=7)
        axes[1][0].imshow(np.where(valid, subj, np.nan), cmap="magma", vmin=0, vmax=1)
        axes[1][0].set_title(f"SAM3 subject mask\narea16={c['subj_area16']:.2f}", fontsize=7)
        for k, li in enumerate(SHOW_L, start=1):
            for row, which in ((0, "a"), (1, "b")):
                arr = f["a" if which == "a" else "b"][li]
                axes[row][k].imshow(np.where(valid, arr, np.nan), cmap="viridis")
                auc = c.get(f"auc_{which}_L{li}", float("nan"))
                tgt = auc if which == "a" else 1.0 - auc
                lbl = ("reg_a → 目标=主体" if which == "a"
                       else "reg_b → 目标=补集")
                band = "canonical 带内" if 8 <= li <= 15 else ""
                axes[row][k].set_title(f"L{li} {lbl} {band}\nAUC_target={tgt:.2f}",
                                       fontsize=7)
        for r_ in axes:
            for ax_ in r_:
                ax_.axis("off")
        fig.tight_layout()
        fig.text(0.5, -0.01,
                 f"[{kind}] 早期层救回分={early_rescue_score(c):+.2f}（判据 ≥0.65）｜"
                 f"ρ(L0)={c['rho_L0']:.2f} ρ(L3)={c['rho_L3']:.2f} ρ(L4)={c['rho_L4']:.2f} "
                 f"ρ(canonical L8-15)={c['rho_canon']:.2f}｜"
                 f"canonical AUC(a)={c['auc_canon_a']:.2f} AUC(b)={c['auc_canon_b']:.2f}\n"
                 f"A: {rs['instructions']['reg_a'][:120]}\n"
                 f"B: {rs['instructions']['reg_b'][:120]}",
                 ha="center", va="top", fontsize=8)
        fig.savefig(out / "viz" / f"{kind}_sfield_{img_id}.png", dpi=120,
                    bbox_inches="tight")
        plt.close(fig)
    # 逐案例口径：单个源上能不能救回（判据同预注册，两侧 AUC_target 都 ≥0.65）
    def case_pass(c, li) -> bool:
        return bool(min(c[f"auc_a_L{li}"], 1.0 - c[f"auc_b_L{li}"])
                    >= VERDICT_RULE["auc_target_min"])

    metrics["per_case_rule_pass"] = {
        "n_cases": len(per_case),
        "n_with_any_early_layer_rho_below_0.3": int(sum(
            any(c[f"rho_L{li}"] < VERDICT_RULE["rho_region_opp_max"] for li in (0, 3, 4))
            for c in per_case)),
        "n_pass_at_some_early_layer_with_rho_below_0.3": int(sum(
            early_rescue_score(c) >= VERDICT_RULE["auc_target_min"] for c in per_case)),
        "by_layer_n_pass_both_sides": {li: int(sum(case_pass(c, li) for c in per_case))
                                       for li in (0, 3, 4, 11, 22)},
        "by_layer_median_min_side": {li: med([min(c[f"auc_a_L{li}"],
                                                  1.0 - c[f"auc_b_L{li}"])
                                              for c in per_case])
                                     for li in (0, 3, 4, 11, 22)},
        "note": ("逐案例通过率 ≈1% 且集中在早期层（那里 AUC 本身方差最大），"
                 "canonical 与后期层 0/212 —— 与「纯噪声下的偶然命中」一致"),
    }
    metrics["viz_case_selection"] = {
        "eligible_subject_area16_range": [AREA_LO, AREA_HI],
        "n_eligible": len(elig), "n_total": len(per_case),
        "note": ("成功/失败两端用同一条主体面积筛（16×16 上主体 <4% 或 >60% 时 AUC 方差"
                 "过大），不是只对成功端筛；early 组按「早期层救回分」排序，saliency 组按"
                 "canonical AUC(reg_a) 排序"),
        "picks": [{"kind": k, "img_id": i} for k, i in picks],
    }
    (out / "metrics.json").write_text(json.dumps(metrics, indent=1, ensure_ascii=False))
    print(f"viz -> {out/'viz'} ({len(picks) + 2} figs)")


if __name__ == "__main__":
    main()
