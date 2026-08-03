"""G1 s 可辨识性分析（EXPERIMENTS_v3 G1 行 + PLAN_v2 Gate D1）。

输入：ro9 读出产物（run_dir/stacks/*.npz，含 canonical s / 逐层 stack / luma16）。
每源三指令：syn_a（instruction）/ syn_b（instruction_short）/ opp（方向词取反）。

指标（预注册判据；D-17 修订 = DATA_ASSIGNMENT §3.1 G1 行 2026-08-03 修订）：
  ρ_syn        = corr(s_syn_a, s_syn_b)   过: 中位数 > 0.7   死: < 0.5
  ρ_region_opp = corr(s_reg_a, s_reg_b)   过: 中位数 < 0.3   （主判据；同方向词、不同区域）
  ρ_opp        = corr(s_syn_a, s_opp)     对照组：只报不判，不触发 FAIL（方向对立高
                                          是「s=where 寻址场」的预期行为，D-17 前旧主判据）
  ρ_shuf       = corr(s_syn_a, s_shuf)    对照组：跨图错位指令；≈ρ_syn ⇒ s 无任何指令条件性
  ρ_Y          = corr(s, luma16)          过: |中位数| < 0.5  死: |中位数| > 0.8
gate = syn ∧ region_opp ∧ Y；ρ_region_opp 缺席（区域对立批未落地）时 gate = "PENDING"
（仅 syn/Y 触发死刑判据时输出 "DEAD"，任何情况下方向对立不进 gate）。
判据口径：canonical s（GL=<retouch_light>，L8-15 head-mean 平均，D-0 修复后），
Pearson（判定）+ Spearman（附报）；逐层 ρ 曲线 / 三 token 对照 / 跨图 s 分布
统计（RO-X2 备料）均为附报。

D-22（零成本附加度量，npz 已存三 token 逐层 stack；⚑U5 已定案 GL=<retouch_light>，
另两 token 仅作附录对照）：
  (a) cross_token_rho —— 同一 (img, instr) 下 light/colortemp/colormixer 三张 canonical s
      两两 Pearson（valid-mask 口径），回答「三 token 图是否高度同质」。
  (b) subject_auc —— 三 token 各自的 s 对 **SAM3 主体掩膜**（D-MASKBANK：veradata 银行
      `cache/subject` 的 `.subject.png`，逐源单主体软掩膜；DATA_ASSIGNMENT §3.3 L82 统一
      评分集口径）的 ROC-AUC，回答「哪个 token 最像主体感知」。附 luma16 基线 AUC
      （s 若不显著高于亮度基线 ⇒ 主体性只是亮度的马甲）与 C_GT（.cgt.png 逐候选掩膜）
      交叉核对子集。
层选择（任务 3）：`layer_scan` 在**主批 ∩ 区域批的公共源**上逐层重算
ρ_syn / ρ_region_opp / |ρ_Y| / AUC，判别力 = ρ_syn − ρ_region_opp（越大越好）。

用法：
  python analyze_g1.py --run-dirs run_smoke30 run_full run_regsmoke20 run_regfull run_shufctrl \
      --out . [--viz-n 6] [--no-subject]
"""
from __future__ import annotations

import argparse
import io
import json
import sqlite3
import subprocess
from itertools import combinations
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
GRID = 16
N_LAYERS = 24
TOKEN_TAGS = ("light", "colortemp", "colormixer")
CANON_LAYERS = slice(8, 16)      # canonical s 的层聚合口径（D2）
# D-22(b) SAM3 主体掩膜银行（DATA_ASSIGNMENT §2 D-MASKBANK：「SAM3 subject cache」）
SUBJECT_DB = "/var/cache/veradata/global.sqlite3"
SUBJECT_GROUP = "cache/subject"
SUBJECT_ROOT = "/mnt/nfs/bc/data/datasets/cache/subject"
MASK_BIN_THR = 0.5               # 16×16 格覆盖 ≥50% 判为主体格
# 层选择候选区间（闭区间，任务 3）：canonical + 逐层曲线上的峰簇 + 全层兜底
CANDIDATE_BANDS = [(8, 15), (0, 23), (2, 9), (6, 11), (11, 11), (16, 23),
                   (19, 23), (20, 23), (20, 22), (22, 22)]
CRITERIA = {
    "rho_syn_pass": 0.7, "rho_syn_dead": 0.5,
    "rho_region_opp_pass": 0.3,   # D-17 主判据：区域对立分离
    "rho_y_pass": 0.5, "rho_y_dead": 0.8,
    # D-17：方向对立（rho_opp）为对照组，只报不判——无阈值、不触发 FAIL、不进 gate
    "rho_opp_note": "control-only since D-17 (2026-08-03); no pass/fail threshold",
}


def pearson(a: np.ndarray, b: np.ndarray, mask: np.ndarray | None = None) -> float:
    """Pearson ρ；mask 给定时只在有效格（非 pad 黑边）上算——
    expand2square 的黑边是全指令共享的常量带，混入会同时抬高 ρ_syn/ρ_opp/ρ_Y。"""
    a = a.ravel().astype(np.float64)
    b = b.ravel().astype(np.float64)
    if mask is not None:
        m = mask.ravel().astype(bool)
        if m.sum() >= 8:
            a, b = a[m], b[m]
    if a.std() == 0 or b.std() == 0:
        return float("nan")
    return float(np.corrcoef(a, b)[0, 1])


def spearman(a: np.ndarray, b: np.ndarray, mask: np.ndarray | None = None) -> float:
    from scipy.stats import spearmanr

    a = a.ravel()
    b = b.ravel()
    if mask is not None:
        m = mask.ravel().astype(bool)
        if m.sum() >= 8:
            a, b = a[m], b[m]
    # getattr：scipy 存根把返回类型标成私有类，直取 .statistic 会报 attr-unknown
    r = getattr(spearmanr(a, b), "statistic")
    return float(r)


def roc_auc(scores: np.ndarray, labels: np.ndarray) -> float:
    """ROC-AUC = 归一化 Mann-Whitney U（并列取平均秩）。两类任一为空返回 nan。"""
    from scipy.stats import rankdata

    s = np.asarray(scores, dtype=np.float64).ravel()
    y = np.asarray(labels).ravel().astype(bool)
    n_pos = int(y.sum())
    n_neg = int((~y).sum())
    if n_pos == 0 or n_neg == 0:
        return float("nan")
    r = rankdata(s)
    return float((r[y].sum() - n_pos * (n_pos + 1) / 2.0) / (n_pos * n_neg))


class SubjectMaskBank:
    """veradata 银行 `cache/subject` 的逐源 SAM3 主体软掩膜（`.subject.png`）。

    键：sample meta 的 `asset_id`（= G1 的 img_id，实测 300/300 命中）。
    读取 = 索引 sqlite 拿 (shard, offset, size) 后 seek 直读 tar 成员（无解包）。
    """

    def __init__(self, db: str = SUBJECT_DB, root: str = SUBJECT_ROOT,
                 group: str = SUBJECT_GROUP):
        self.root = Path(root)
        self.group = group
        self.con = sqlite3.connect(f"file:{db}?mode=ro", uri=True)
        self.by_asset: dict[str, str] = {}
        self.subject_meta: dict[str, dict] = {}
        self._cache: dict[str, tuple[np.ndarray, dict] | None] = {}
        for sid, meta in self.con.execute(
                'select sample_id, meta from samples where "group"=?', (group,)):
            d = json.loads(meta)
            aid = d.get("asset_id")
            if aid and d.get("eligible") and d.get("subject"):
                self.by_asset[aid] = sid
                self.subject_meta[aid] = d["subject"]

    def mask_grid(self, asset_id: str, grid: int = GRID):
        """→ (mask16 float32 [0,1], meta) 或 None。

        对齐口径与 luma 完全一致：原掩膜 → 短边 512 bilinear → expand2square 黑边 pad
        → 面积均值下采样到 16×16（`ro9_gl_attention.luma_to_grid`）。
        """
        from PIL import Image

        from ro9_gl_attention import luma_to_grid

        if asset_id in self._cache:
            return self._cache[asset_id]
        sid = self.by_asset.get(asset_id)
        if sid is None:
            self._cache[asset_id] = None
            return None
        row = self.con.execute(
            'select shard, offset_data, size from members '
            'where "group"=? and sample_id=? and suffix=?',
            (self.group, sid, ".subject.png")).fetchone()
        if row is None:
            self._cache[asset_id] = None
            return None
        shard, off, size = row
        with open(self.root / "shards" / f"{shard}.tar", "rb") as fh:
            fh.seek(off)
            raw = fh.read(size)
        if len(raw) != size:
            self._cache[asset_id] = None
            return None
        im = Image.open(io.BytesIO(raw)).convert("L")
        w, h = im.size
        scale = 512 / min(w, h)
        im = im.resize((int(round(w * scale)), int(round(h * scale))),
                       Image.Resampling.BILINEAR)
        m16, _ = luma_to_grid(np.asarray(im, dtype=np.float32) / 255.0, grid)
        self._cache[asset_id] = (m16, self.subject_meta.get(asset_id, {}))
        return self._cache[asset_id]


def luma_valid_from_image(img_path: str):
    """从暂存源图重算 luma16 + valid16（对齐 expand2square pad 语义）。
    旧批次 stacks 里的 luma16 是 center-crop 口径（错），一律以此为准。"""
    from PIL import Image

    from ro9_gl_attention import luma_to_grid

    image = Image.open(img_path).convert("RGB")
    w, h = image.size
    scale = 512 / min(w, h)
    im = image.resize((int(round(w * scale)), int(round(h * scale))),
                      Image.Resampling.LANCZOS)
    luma = np.asarray(im.convert("L"), dtype=np.float32) / 255.0
    return luma_to_grid(luma)


def load_triplet(stacks: dict[str, Path], img_id: str, ihashes: dict[str, str]):
    out = {}
    for itag, ih in ihashes.items():
        p = stacks.get(f"{img_id}__{ih}")
        if p is None:
            return None
        with np.load(p) as z:
            out[itag] = {
                "s_canon": z["s_canon"].astype(np.float32),
                "stack": z["stack"].astype(np.float32),      # (3,24,16,16)
                "luma16": z["luma16"].astype(np.float32),
                "outlier_frac": z["outlier_frac"].astype(np.float32),
                "meta": json.loads(str(z["meta"])),
            }
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-dirs", nargs="+", required=True)
    ap.add_argument("--samples-json", default=str(HERE / "config" / "g1_samples.json"))
    ap.add_argument("--shuffle-json", default=str(HERE / "config" / "g1_shuffle_ctrl.json"),
                    help="指令条件性对照（NOTES §六）；文件或其 stacks 缺失则跳过")
    ap.add_argument("--region-json", default=str(HERE / "config" / "g1_region_opp.json"),
                    help="区域对立批（D-17 主判据）；文件或其 stacks 缺失则跳过")
    ap.add_argument("--out", default=str(HERE))
    ap.add_argument("--viz-n", type=int, default=6)
    ap.add_argument("--subject-db", default=SUBJECT_DB)
    ap.add_argument("--subject-root", default=SUBJECT_ROOT)
    ap.add_argument("--no-subject", action="store_true",
                    help="跳过 D-22(b) SAM3 主体掩膜 AUC（银行不可达时）")
    args = ap.parse_args()

    import sys
    sys.path.insert(0, str(HERE.parents[1] / "tools" / "scache"))
    sys.path.insert(0, str(HERE.parents[1] / "tools" / "readout"))
    from api import instr_hash

    samples = json.loads(Path(args.samples_json).read_text())
    stacks: dict[str, Path] = {}
    for rd in args.run_dirs:
        for p in (HERE / rd / "stacks").glob("*.npz"):
            stacks[p.stem] = p

    bank = None
    if not args.no_subject:
        try:
            bank = SubjectMaskBank(args.subject_db, args.subject_root)
        except Exception as exc:      # 银行不可达 ⇒ 降级跳过 D-22(b)，其余照算
            print(f"[warn] SubjectMaskBank 不可用，跳过 D-22(b): {exc}")
    subj_grids: dict[str, np.ndarray] = {}
    subj_meta: dict[str, dict] = {}

    per_source, layer_syn, layer_opp, layer_y = [], [], [], []
    layer_by_img: dict[str, dict[str, list[float]]] = {}
    token_rho = {t: {"syn": [], "opp": [], "y": []} for t in TOKEN_TAGS}
    cross_token = {f"{a}~{b}": {"syn_a": []} for a, b in combinations(TOKEN_TAGS, 2)}
    cross_img_stats = []
    triplets = {}
    for s in samples:
        ihashes = {k: instr_hash(v) for k, v in s["instructions"].items()}
        t = load_triplet(stacks, s["img_id"], ihashes)
        if t is None:
            continue
        # luma/valid 一律从源图重算（旧批 stacks 的 luma16 是 crop 口径，弃用）
        luma, valid = luma_valid_from_image(s["img_path"])
        t["_luma"], t["_valid"] = luma, valid
        triplets[s["img_id"]] = (s, t)
        a, b, o = (t[k]["s_canon"] for k in ("syn_a", "syn_b", "opp"))
        row = {
            "img_id": s["img_id"], "pool": s["pool"],
            "task_type": s["task_type"], "conf": s["winner_confidence"],
            "rho_syn": pearson(a, b, valid), "rho_opp": pearson(a, o, valid),
            "rho_y": pearson(a, luma, valid),
            "rho_syn_sp": spearman(a, b, valid), "rho_opp_sp": spearman(a, o, valid),
            "rho_y_sp": spearman(a, luma, valid),
            "rho_syn_allcells": pearson(a, b), "rho_opp_allcells": pearson(a, o),
            "valid_frac": float(valid.mean()),
            "outlier_frac": float(t["syn_a"]["outlier_frac"].mean()),
            "fallback": t["syn_a"]["meta"]["fallback"]["light"],
        }
        # 逐层（GL token=stack[0]）
        sa, sb, so = (t[k]["stack"][0] for k in ("syn_a", "syn_b", "opp"))
        l_syn = [pearson(sa[li], sb[li], valid) for li in range(N_LAYERS)]
        l_opp = [pearson(sa[li], so[li], valid) for li in range(N_LAYERS)]
        l_y = [pearson(sa[li], luma, valid) for li in range(N_LAYERS)]
        layer_syn.append(l_syn)
        layer_opp.append(l_opp)
        layer_y.append(l_y)
        layer_by_img[s["img_id"]] = {"syn": l_syn, "opp": l_opp, "y": l_y}
        # 三 token 对照（canonical 层聚合口径一致：L8-15 平均）
        canon = {}
        for ti, tag in enumerate(TOKEN_TAGS):
            ca = t["syn_a"]["stack"][ti][CANON_LAYERS].mean(axis=0)
            cb = t["syn_b"]["stack"][ti][CANON_LAYERS].mean(axis=0)
            co = t["opp"]["stack"][ti][CANON_LAYERS].mean(axis=0)
            canon[tag] = ca
            token_rho[tag]["syn"].append(pearson(ca, cb, valid))
            token_rho[tag]["opp"].append(pearson(ca, co, valid))
            token_rho[tag]["y"].append(pearson(ca, luma, valid))
        # D-22(a) 跨 token 两两空间 ρ（同一 img+instr，syn_a 条件）
        for ta, tb in combinations(TOKEN_TAGS, 2):
            rho_tt = pearson(canon[ta], canon[tb], valid)
            cross_token[f"{ta}~{tb}"]["syn_a"].append(rho_tt)
            row[f"rho_tok_{ta}~{tb}"] = rho_tt
        # D-22(b) SAM3 主体掩膜 AUC（三 token + luma 基线；syn_a 条件）
        if bank is not None:
            got = bank.mask_grid(s["img_id"])
            if got is not None:
                m16, smeta = got
                subj_grids[s["img_id"]] = m16
                subj_meta[s["img_id"]] = smeta
                lab = (m16 >= MASK_BIN_THR)[valid]
                row["subj_area16"] = float((m16 >= MASK_BIN_THR)[valid].mean())
                for tag in TOKEN_TAGS:
                    row[f"auc_{tag}"] = roc_auc(canon[tag][valid], lab)
                row["auc_luma"] = roc_auc(luma[valid], lab)
        per_source.append(row)
        # 跨图 s 分布（RO-X2 归一化实验备料；只统计有效格）
        av = a[valid]
        cross_img_stats.append({
            "img_id": s["img_id"], "pool": s["pool"],
            "mean": float(av.mean()), "std": float(av.std()),
            "min": float(av.min()), "max": float(av.max()),
            "p10": float(np.percentile(av, 10)), "p90": float(np.percentile(av, 90)),
        })

    n = len(per_source)
    med = lambda k: float(np.nanmedian([r[k] for r in per_source])) if n else float("nan")
    agg = {
        "n_sources": n,
        "rho_syn_median": med("rho_syn"), "rho_opp_median": med("rho_opp"),
        "rho_y_median": med("rho_y"),
        "rho_y_abs_median": float(np.nanmedian([abs(r["rho_y"]) for r in per_source])) if n else float("nan"),
        "rho_syn_sp_median": med("rho_syn_sp"), "rho_opp_sp_median": med("rho_opp_sp"),
        "rho_y_sp_median": med("rho_y_sp"),
        "outlier_frac_mean": float(np.mean([r["outlier_frac"] for r in per_source])) if n else 0.0,
        "fallback_rate": float(np.mean([r["fallback"] for r in per_source])) if n else 0.0,
        "valid_frac_mean": float(np.mean([r["valid_frac"] for r in per_source])) if n else 0.0,
        "rho_syn_allcells_median": med("rho_syn_allcells"),
        "rho_opp_allcells_median": med("rho_opp_allcells"),
    }
    # ---- 区域对立批（D-17 主判据）：同方向词、不同区域 → 预期 s 场分离 ----
    region_opp = None
    region_rows: list[dict] = []
    region_cases: dict[str, dict] = {}
    reg_path = Path(args.region_json)
    reg_layer_by_img: dict[str, list[float]] = {}
    reg_token_layer: dict[str, list[np.ndarray]] = {t: [] for t in TOKEN_TAGS}
    reg_keep: list[tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray | None]] = []
    if reg_path.is_file():
        reg_layer: list[list[float]] = []
        for rs in json.loads(reg_path.read_text()):
            ih_a = instr_hash(rs["instructions"]["reg_a"])
            ih_b = instr_hash(rs["instructions"]["reg_b"])
            pa = stacks.get(f"{rs['img_id']}__{ih_a}")
            pb = stacks.get(f"{rs['img_id']}__{ih_b}")
            if pa is None or pb is None:
                continue
            with np.load(pa) as z:
                s_a = z["s_canon"].astype(np.float32)
                stk_a = z["stack"].astype(np.float32)
                meta_a = json.loads(str(z["meta"]))
            with np.load(pb) as z:
                s_b = z["s_canon"].astype(np.float32)
                stk_b = z["stack"].astype(np.float32)
            luma, valid = luma_valid_from_image(rs["img_path"])
            row = {
                "img_id": rs["img_id"], "pool": rs["pool"],
                "direction": rs["direction"], "region_b_kind": rs["region_b_kind"],
                "subject_region": rs["subject_region"], "conf": rs["winner_confidence"],
                "rho_region_opp": pearson(s_a, s_b, valid),
                "rho_region_opp_sp": spearman(s_a, s_b, valid),
                "rho_region_opp_allcells": pearson(s_a, s_b),
                "valid_frac": float(valid.mean()),
                "fallback": meta_a["fallback"]["light"],
            }
            # D-22(a) 跨 token 两两 ρ（reg_a 条件）
            canon_a = {tag: stk_a[ti][CANON_LAYERS].mean(axis=0)
                       for ti, tag in enumerate(TOKEN_TAGS)}
            canon_b = {tag: stk_b[ti][CANON_LAYERS].mean(axis=0)
                       for ti, tag in enumerate(TOKEN_TAGS)}
            for ta, tb in combinations(TOKEN_TAGS, 2):
                cross_token[f"{ta}~{tb}"].setdefault("reg_a", []).append(
                    pearson(canon_a[ta], canon_a[tb], valid))
            # D-22(b) 主体掩膜 AUC：reg_a（指令点名主体）vs reg_b（指令点名补集）
            if bank is not None:
                got = bank.mask_grid(rs["img_id"])
                if got is not None:
                    m16, smeta = got
                    subj_grids.setdefault(rs["img_id"], m16)
                    subj_meta.setdefault(rs["img_id"], smeta)
                    lab = (m16 >= MASK_BIN_THR)[valid]
                    row["subj_area16"] = float(lab.mean())
                    for tag in TOKEN_TAGS:
                        row[f"auc_rega_{tag}"] = roc_auc(canon_a[tag][valid], lab)
                        row[f"auc_regb_{tag}"] = roc_auc(canon_b[tag][valid], lab)
                    row["auc_reg_luma"] = roc_auc(luma[valid], lab)
                    # 逐层 AUC（三 token × 24 层，reg_a 条件）——层选择用
                    for ti, tag in enumerate(TOKEN_TAGS):
                        reg_token_layer[tag].append(np.array(
                            [roc_auc(stk_a[ti][li][valid], lab) for li in range(N_LAYERS)]))
            region_rows.append(row)
            region_cases[rs["img_id"]] = {
                "sample": rs, "s_a": s_a, "s_b": s_b, "luma": luma, "valid": valid,
                "rho": row["rho_region_opp"],
                "subj": subj_grids.get(rs["img_id"]),
            }
            lyr = [pearson(stk_a[0][li], stk_b[0][li], valid) for li in range(N_LAYERS)]
            reg_layer.append(lyr)
            reg_layer_by_img[rs["img_id"]] = lyr
            reg_keep.append((stk_a, stk_b, valid, subj_grids.get(rs["img_id"])))
        if region_rows:
            rmed = float(np.nanmedian([r["rho_region_opp"] for r in region_rows]))
            region_opp = {
                "n": len(region_rows),
                "rho_region_opp_median": rmed,
                "rho_region_opp_sp_median": float(np.nanmedian(
                    [r["rho_region_opp_sp"] for r in region_rows])),
                "rho_region_opp_allcells_median": float(np.nanmedian(
                    [r["rho_region_opp_allcells"] for r in region_rows])),
                "fallback_rate": float(np.mean([r["fallback"] for r in region_rows])),
                "by_pool": {p: float(np.nanmedian(
                    [r["rho_region_opp"] for r in region_rows if r["pool"] == p]))
                    for p in ("unsplash", "awards", "ppr10k")
                    if any(r["pool"] == p for r in region_rows)},
                "by_region_b_kind": {k: {"n": sum(r["region_b_kind"] == k for r in region_rows),
                                         "median": float(np.nanmedian(
                                             [r["rho_region_opp"] for r in region_rows
                                              if r["region_b_kind"] == k]))}
                                     for k in ("background", "spatial")
                                     if any(r["region_b_kind"] == k for r in region_rows)},
                "by_direction": {d: float(np.nanmedian(
                    [r["rho_region_opp"] for r in region_rows if r["direction"] == d]))
                    for d in ("brighten", "darken")
                    if any(r["direction"] == d for r in region_rows)},
                "by_conf": {c: {"n": sum(r["conf"] == c for r in region_rows),
                                "median": float(np.nanmedian(
                                    [r["rho_region_opp"] for r in region_rows
                                     if r["conf"] == c]))}
                            for c in ("normal", "low")
                            if any(r["conf"] == c for r in region_rows)},
                "rho_region_opp_q10_q90": [
                    float(np.nanpercentile([r["rho_region_opp"] for r in region_rows], 10)),
                    float(np.nanpercentile([r["rho_region_opp"] for r in region_rows], 90))],
                "n_below_0.3": int(sum(r["rho_region_opp"] < 0.3 for r in region_rows)),
                "layer_median_curve": np.nanmedian(np.array(reg_layer), axis=0).tolist(),
                "per_source": region_rows,
            }

    # D-17（DATA_ASSIGNMENT §3.1 G1 行 2026-08-03 修订）：
    # 主判据 = syn / region_opp / y；方向对立（rho_opp）为对照组只报不判、不触发 FAIL。
    verdict = {
        "rho_syn": "PASS" if agg["rho_syn_median"] > 0.7 else ("DEAD" if agg["rho_syn_median"] < 0.5 else "MARGINAL"),
        "rho_opp": "CONTROL",   # 对照组：数值照报（aggregate.rho_opp_median），不判 PASS/FAIL
        "rho_y": ("DEAD" if agg["rho_y_abs_median"] > 0.8 else
                  ("PASS" if agg["rho_y_abs_median"] < 0.5 else "MARGINAL")),
        "rho_region_opp": "PENDING",  # 主判据；区域对立批未落地时缺席
    }
    if region_opp is not None:
        verdict["rho_region_opp"] = ("PASS" if region_opp["rho_region_opp_median"] < 0.3
                                     else "FAIL")
    gate_keys = ["rho_syn", "rho_region_opp", "rho_y"]
    gate_verdicts = [verdict[k] for k in gate_keys]
    if "DEAD" in gate_verdicts:
        gate = "DEAD"        # 死刑判据（syn<0.5 / |ρ_Y|>0.8）不因主判据缺席豁免
    elif "PENDING" in gate_verdicts:
        gate = "PENDING"     # 主判据 ρ_region_opp 缺席 ⇒ 不可判（不是 FAIL）
    elif all(v == "PASS" for v in gate_verdicts):
        gate = "PASS"
    else:
        gate = "FAIL"

    layer_curves = {
        "rho_syn": np.nanmedian(np.array(layer_syn), axis=0).tolist() if n else [],
        "rho_opp": np.nanmedian(np.array(layer_opp), axis=0).tolist() if n else [],
        "rho_y": np.nanmedian(np.array(layer_y), axis=0).tolist() if n else [],
    }
    by_pool = {}
    for pool in ("unsplash", "awards", "ppr10k"):
        rs = [r for r in per_source if r["pool"] == pool]
        if rs:
            by_pool[pool] = {
                "n": len(rs),
                "rho_syn_median": float(np.nanmedian([r["rho_syn"] for r in rs])),
                "rho_opp_median": float(np.nanmedian([r["rho_opp"] for r in rs])),
                "rho_y_abs_median": float(np.nanmedian([abs(r["rho_y"]) for r in rs])),
            }
    token_agg = {tag: {k: float(np.nanmedian(v)) for k, v in d.items()}
                 for tag, d in token_rho.items()}
    # 分层报告：task_type × winner_confidence（REPORT 分层表）
    by_task_conf = {}
    for tt in ("local", "style"):
        for cf in ("normal", "low"):
            rs = [r for r in per_source if r["task_type"] == tt and r["conf"] == cf]
            if not rs:
                continue
            cell = {
                "n": len(rs),
                "rho_syn_median": float(np.nanmedian([r["rho_syn"] for r in rs])),
                "rho_opp_median": float(np.nanmedian([r["rho_opp"] for r in rs])),
                "rho_y_abs_median": float(np.nanmedian([abs(r["rho_y"]) for r in rs])),
                "fallback_rate": float(np.mean([r["fallback"] for r in rs])),
            }
            aucs = [r["auc_light"] for r in rs if "auc_light" in r]
            if aucs:
                cell["auc_light_median"] = float(np.nanmedian(aucs))
                cell["auc_luma_median"] = float(np.nanmedian(
                    [r["auc_luma"] for r in rs if "auc_luma" in r]))
            by_task_conf[f"{tt}/{cf}"] = cell
    # 指令条件性对照（错位指令）：corr(s(img,syn_a), s(img,shuf))
    shuffle_ctrl = None
    shuf_layer_by_img: dict[str, list[float]] = {}
    shuf_path = Path(args.shuffle_json)
    if shuf_path.is_file():
        rho_shuf, shuf_ids = [], []
        for sc in json.loads(shuf_path.read_text()):
            ih_shuf = instr_hash(sc["instructions"]["shuf"])
            p = stacks.get(f"{sc['img_id']}__{ih_shuf}")
            base = triplets.get(sc["img_id"])
            if p is None or base is None:
                continue
            with np.load(p) as z:
                s_shuf = z["s_canon"].astype(np.float32)
                stk_shuf = z["stack"].astype(np.float32)
            valid_b = base[1]["_valid"]
            rho_shuf.append(pearson(base[1]["syn_a"]["s_canon"], s_shuf, valid_b))
            shuf_ids.append(sc["img_id"])
            shuf_layer_by_img[sc["img_id"]] = [
                pearson(base[1]["syn_a"]["stack"][0][li], stk_shuf[0][li], valid_b)
                for li in range(N_LAYERS)]
        if rho_shuf:
            # 配对读法：同一批 60 源上的 ρ_syn / ρ_opp（跨集合比中位数会失真）
            by_id = {r["img_id"]: r for r in per_source}
            pairs = [(by_id[i]["rho_syn"], rs) for i, rs in zip(shuf_ids, rho_shuf)
                     if i in by_id]
            paired = [by_id[i] for i in shuf_ids if i in by_id]
            wins = [float(a) > float(b) for a, b in pairs
                    if np.isfinite(a) and np.isfinite(b)]
            shuffle_ctrl = {
                "n": len(rho_shuf),
                "rho_shuf_median": float(np.nanmedian(rho_shuf)),
                "paired_rho_syn_median": float(np.nanmedian(
                    [r["rho_syn"] for r in paired])) if paired else None,
                "paired_rho_opp_median": float(np.nanmedian(
                    [r["rho_opp"] for r in paired])) if paired else None,
                "paired_delta_syn_minus_shuf": float(np.nanmedian(
                    [a - b for a, b in pairs])) if pairs else None,
                "paired_win_rate_syn_gt_shuf": float(np.mean(wins)) if wins else None,
                "paired_n": len(pairs),
                "layer_median_curve": np.nanmedian(
                    np.array([shuf_layer_by_img[i] for i in shuf_ids]), axis=0).tolist(),
                "note": "≈rho_syn ⇒ s 无指令条件性（NOTES §六）",
            }

    # ---- D-22(a) 跨 token 两两空间 ρ ----
    cross_token_rho: dict[str, object] = {
        pair: {cond: float(np.nanmedian(vals)) for cond, vals in conds.items() if vals}
        for pair, conds in cross_token.items()}
    cross_token_rho["_note"] = ("同一 (img, instr) 下三 token canonical s 两两 Pearson"
                               "（valid-mask）；接近 1 = 三个 special token 的 attention "
                               "图高度同质，U5 选 light 不损失信息")

    # ---- D-22(b) 三 token 对 SAM3 主体掩膜 AUC ----
    subject_auc = None
    if bank is not None and subj_grids:
        def _med(rows, key):
            v = [r[key] for r in rows if key in r and np.isfinite(r[key])]
            return float(np.nanmedian(v)) if v else None

        def _blk(rows, prefix, luma_key=None) -> dict[str, object]:
            out: dict[str, object] = {"n": sum(1 for r in rows if f"{prefix}light" in r)}
            for tag in TOKEN_TAGS:
                out[tag] = _med(rows, f"{prefix}{tag}")
            out["luma_baseline"] = _med(rows, luma_key or f"{prefix}luma")
            best = [(out[t], t) for t in TOKEN_TAGS if out[t] is not None]
            out["best_token"] = max(best)[1] if best else None
            return out

        subject_auc = {
            "mask": "veradata bank cache/subject/.subject.png（SAM3 主体软掩膜，逐源单主体）",
            "label_rule": f"16×16 格主体覆盖 ≥{MASK_BIN_THR} 判正，仅取 valid（非 pad）格",
            "main_syn_a": _blk(per_source, "auc_"),
            "region_reg_a": _blk(region_rows, "auc_rega_", "auc_reg_luma"),
            "region_reg_b": _blk(region_rows, "auc_regb_", "auc_reg_luma"),
            "subject_area16_median": float(np.nanmedian(
                [r["subj_area16"] for r in per_source if "subj_area16" in r])),
            "by_task_type": {
                tt: _blk([r for r in per_source if r["task_type"] == tt], "auc_")
                for tt in ("local", "style")
                if any(r["task_type"] == tt for r in per_source)},
            "note": ("reg_a（指令点名主体）与 reg_b（指令点名补集）的 AUC 之差 = s 的区域"
                     "条件性直接读数；≈0 ⇒ s 与指令所指区域无关"),
        }
        if subject_auc["region_reg_a"]["light"] is not None:
            subject_auc["rega_minus_regb_light"] = (
                subject_auc["region_reg_a"]["light"] - subject_auc["region_reg_b"]["light"])
        # 配对读法（终判用）：同一张图上 AUC(reg_a) vs AUC(reg_b)。中位差 ≈ 0 且 win rate
        # ≈ 0.5 ⇒ 指令说主体还是说背景，s 对主体掩膜的判别力一样 ⇒ s 与指令所指区域无关。
        pa = np.array([r["auc_rega_light"] for r in region_rows
                       if np.isfinite(r.get("auc_rega_light", np.nan))
                       and np.isfinite(r.get("auc_regb_light", np.nan))])
        pb = np.array([r["auc_regb_light"] for r in region_rows
                       if np.isfinite(r.get("auc_rega_light", np.nan))
                       and np.isfinite(r.get("auc_regb_light", np.nan))])
        if pa.size:
            from scipy.stats import wilcoxon
            w = wilcoxon(pa, pb)
            subject_auc["paired_rega_vs_regb_light"] = {
                "paired_n": int(pa.size),
                "median_auc_rega": float(np.median(pa)),
                "median_auc_regb": float(np.median(pb)),
                "median_delta": float(np.median(pa - pb)),
                "mean_delta": float(np.mean(pa - pb)),
                "win_rate_rega_gt_regb": float(np.mean(pa > pb)),
                "frac_regb_gt_rega": float(np.mean(pb > pa)),
                "wilcoxon_p": float(getattr(w, "pvalue")),
                "by_region_b_kind": {
                    k: {
                        "n": int(sum(1 for r in region_rows if r["region_b_kind"] == k
                                     and np.isfinite(r.get("auc_regb_light", np.nan)))),
                        "frac_regb_gt_rega": float(np.mean(
                            [r["auc_regb_light"] > r["auc_rega_light"] for r in region_rows
                             if r["region_b_kind"] == k
                             and np.isfinite(r.get("auc_rega_light", np.nan))
                             and np.isfinite(r.get("auc_regb_light", np.nan))])),
                    }
                    for k in ("background", "spatial")
                    if any(r["region_b_kind"] == k for r in region_rows)},
                "note": ("s 是否随指令改变所指区域的**直接**配对检验；median_delta≈0 + "
                         "win rate≈0.5 + wilcoxon 不显著 ⇒ 无区域条件性"),
            }
        # 逐层 AUC 曲线（三 token，reg_a 条件）
        if any(reg_token_layer[t] for t in TOKEN_TAGS):
            subject_auc["layer_auc_median_curve"] = {
                tag: np.nanmedian(np.array(reg_token_layer[tag]), axis=0).tolist()
                for tag in TOKEN_TAGS if reg_token_layer[tag]}

    # ---- 层选择重扫（任务 3）：主批 ∩ 区域批公共源上逐层对齐口径 ----
    layer_scan = None
    common = sorted(set(layer_by_img) & set(reg_layer_by_img))
    if common:
        syn_c = np.array([layer_by_img[i]["syn"] for i in common])
        opp_c = np.array([layer_by_img[i]["opp"] for i in common])
        y_c = np.abs(np.array([layer_by_img[i]["y"] for i in common]))
        reg_c = np.array([reg_layer_by_img[i] for i in common])
        m_syn = np.nanmedian(syn_c, axis=0)
        m_reg = np.nanmedian(reg_c, axis=0)
        sep = m_syn - m_reg
        shuf_c = ([shuf_layer_by_img[i] for i in common if i in shuf_layer_by_img])
        m_shuf = np.nanmedian(np.array(shuf_c), axis=0) if shuf_c else None
        auc_curve = None
        if subject_auc and "layer_auc_median_curve" in subject_auc:
            auc_curve = subject_auc["layer_auc_median_curve"].get("light")
        layer_scan = {
            "n_common_sources": len(common),
            "n_common_shuffle": len(shuf_c),
            "criterion": ("判别力 sep(L) = ρ_syn(L) − ρ_region_opp(L)（同一批源、同一 "
                          "valid-mask 口径）；可用读出层需 sep 显著 >0 且 ρ_syn 高、|ρ_Y| 低"),
            "rho_syn": m_syn.tolist(),
            "rho_opp": np.nanmedian(opp_c, axis=0).tolist(),
            "rho_region_opp": m_reg.tolist(),
            "rho_y_abs": np.nanmedian(y_c, axis=0).tolist(),
            "rho_shuf": m_shuf.tolist() if m_shuf is not None else None,
            "sep_syn_minus_region": sep.tolist(),
            "sep_syn_minus_shuf": (m_syn - m_shuf).tolist() if m_shuf is not None else None,
            "subject_auc_light": auc_curve,
            "best_sep_layer": int(np.nanargmax(sep)),
            "best_sep_value": float(np.nanmax(sep)),
            "best_auc_layer": (int(np.nanargmax(np.abs(np.array(auc_curve) - 0.5)))
                               if auc_curve else None),
            "canonical_band": [8, 15],
            "canonical_sep": float(np.nanmedian(sep[CANON_LAYERS])),
        }
        # 推荐区间：sep 达不到可用阈值时给「不可用」结论 + 退而求其次的 AUC 区间
        SEP_MIN = 0.10
        if layer_scan["best_sep_value"] >= SEP_MIN:
            good = [int(li) for li in range(N_LAYERS) if sep[li] >= SEP_MIN]
            layer_scan["recommended_band"] = [min(good), max(good)]
            layer_scan["recommendation"] = (
                f"按判别力选层：sep≥{SEP_MIN} 的层 = {good}")
        else:
            layer_scan["recommended_band"] = None
            if auc_curve:
                a = np.array(auc_curve)
                top = np.argsort(-np.abs(a - 0.5))[:6]
                band = sorted(int(x) for x in top)
                layer_scan["recommendation"] = (
                    f"**无层可用于指令条件读出**（sep 最大仅 {layer_scan['best_sep_value']:.3f}"
                    f" < {SEP_MIN}，全层 ρ_syn ≤ ρ_region_opp）。若 RO-9 降级为「图像驱动"
                    f"主体显著性」用途，按主体掩膜 AUC 偏离 0.5 最大的层选：{band}"
                    f"（AUC 中位 {[round(float(a[i]), 3) for i in band]}）")
                layer_scan["auc_recommended_layers"] = band
            else:
                layer_scan["recommendation"] = (
                    f"**无层可用**（sep 最大 {layer_scan['best_sep_value']:.3f} < {SEP_MIN}）")

    # ---- 层区间扫描（任务 3 的实际交付：给推荐读出层区间，而非单层）----
    band_scan = None
    if triplets and reg_keep:
        band_rows = []
        for lo, hi in CANDIDATE_BANDS:
            sl = slice(lo, hi + 1)
            syn, opp, yab = [], [], []
            for _s, t in triplets.values():
                v = t["_valid"]
                ba = t["syn_a"]["stack"][0][sl].mean(axis=0)
                bb = t["syn_b"]["stack"][0][sl].mean(axis=0)
                bo = t["opp"]["stack"][0][sl].mean(axis=0)
                syn.append(pearson(ba, bb, v))
                opp.append(pearson(ba, bo, v))
                yab.append(abs(pearson(ba, t["_luma"], v)))
            reg, aucs = [], {tag: [] for tag in TOKEN_TAGS}
            for stk_a, stk_b, v, m16 in reg_keep:
                reg.append(pearson(stk_a[0][sl].mean(axis=0), stk_b[0][sl].mean(axis=0), v))
                if m16 is None:
                    continue
                lab = (m16 >= MASK_BIN_THR)[v]
                for ti, tag in enumerate(TOKEN_TAGS):
                    aucs[tag].append(roc_auc(stk_a[ti][sl].mean(axis=0)[v], lab))
            row = {
                "band": [lo, hi],
                "rho_syn_median": float(np.nanmedian(syn)),
                "rho_opp_median": float(np.nanmedian(opp)),
                "rho_y_abs_median": float(np.nanmedian(yab)),
                "rho_region_opp_median": float(np.nanmedian(reg)),
            }
            row["sep_syn_minus_region"] = row["rho_syn_median"] - row["rho_region_opp_median"]
            for tag in TOKEN_TAGS:
                row[f"auc_{tag}_median"] = (float(np.nanmedian(aucs[tag]))
                                            if aucs[tag] else None)
            band_rows.append(row)
        best_auc = max(band_rows, key=lambda r: r["auc_light_median"] or 0.0)
        best_sep = max(band_rows, key=lambda r: r["sep_syn_minus_region"])
        canon_row = next(r for r in band_rows if r["band"] == [8, 15])
        band_scan = {
            "note": ("候选区间 = canonical(L8-15) + 逐层曲线峰簇 + 全层兜底；"
                     "区间内 head-mean logit 先按层平均再算指标（与 canonical 同口径）"),
            "bands": band_rows,
            "canonical_band_row": canon_row,
            "best_by_sep": best_sep,
            "best_by_subject_auc": best_auc,
            "recommended_band": best_auc["band"],
            "recommendation": (
                f"判别力口径全线失效（各区间 sep ≤ {best_sep['sep_syn_minus_region']:.3f}，"
                f"canonical L8-15 sep = {canon_row['sep_syn_minus_region']:.3f}）⇒ "
                f"**没有任何区间支持指令条件读出**。若 RO-9 仅保留「图像驱动主体显著性」"
                f"用途，推荐区间 = L{best_auc['band'][0]}–{best_auc['band'][1]}"
                f"（AUC_light 中位 {best_auc['auc_light_median']:.3f}，"
                f"对比 canonical L8-15 的 {canon_row['auc_light_median']:.3f}）"),
        }

    s_all_means = [c["mean"] for c in cross_img_stats]
    s_all_stds = [c["std"] for c in cross_img_stats]
    cross = {
        "per_image": cross_img_stats,
        "across_images": {
            "mean_of_means": float(np.mean(s_all_means)) if n else None,
            "std_of_means": float(np.std(s_all_means)) if n else None,
            "mean_of_stds": float(np.mean(s_all_stds)) if n else None,
            "std_of_stds": float(np.std(s_all_stds)) if n else None,
        },
    }

    git = subprocess.run(["git", "rev-parse", "--short", "HEAD"], cwd=HERE,
                         capture_output=True, text=True).stdout.strip()
    metrics = {
        "experiment": "G1_s_identifiability", "date": "2026-08-03",
        "git_commit": git, "criteria": CRITERIA, "aggregate": agg,
        "verdict": verdict, "gate": gate,
        "gate_keys": gate_keys,   # D-17：主判据恒为 syn/region_opp/y，rho_opp 对照不进 gate
        "by_pool": by_pool,
        "by_task_conf": by_task_conf,
        "token_comparison_median": token_agg,
        "cross_token_rho": cross_token_rho,          # D-22(a)
        "subject_auc": subject_auc,                  # D-22(b)
        "layer_scan": layer_scan,                    # 层选择重扫（任务 3，逐层）
        "band_scan": band_scan,                      # 层选择重扫（任务 3，区间）
        "shuffle_control": shuffle_ctrl,
        "region_opposition": region_opp,
        "layer_median_curves": layer_curves,
        "cross_image_s_stats": cross,
        "per_source": per_source,
    }
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    (out / "metrics.json").write_text(json.dumps(metrics, indent=1))
    print(json.dumps({"n": n, "aggregate": {k: round(v, 4) if isinstance(v, float) else v
                                            for k, v in agg.items()},
                      "verdict": verdict, "gate": gate, "gate_keys": gate_keys,
                      "by_pool": by_pool, "by_task_conf": by_task_conf,
                      "token_median": token_agg,
                      "cross_token_rho": cross_token_rho,
                      "subject_auc": ({k: v for k, v in subject_auc.items()
                                       if k != "layer_auc_median_curve"}
                                      if subject_auc else None),
                      "layer_scan": ({k: v for k, v in layer_scan.items()
                                      if not isinstance(v, list)}
                                     if layer_scan else None),
                      "band_scan": band_scan,
                      "shuffle_control": shuffle_ctrl,
                      "region_opposition": (
                          {k: v for k, v in region_opp.items() if k != "per_source"}
                          if region_opp else None)}, indent=1, ensure_ascii=False))

    # ---- viz：主批成功/失败（D-17 主批判据 = ρ_syn>0.7 且 |ρ_Y|<0.5）----
    if n and args.viz_n:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        from PIL import Image

        viz = out / "viz"
        viz.mkdir(exist_ok=True)
        # 排序按主批判据的裕度：ρ_syn 越高、|ρ_Y| 越低越好（D-17 起 ρ_opp 是对照不参与判定）
        score = {r["img_id"]: (r["rho_syn"] - abs(r["rho_y"])) for r in per_source}
        ranked = sorted(score, key=score.get)  # type: ignore[arg-type]
        cases = ([("failure", i) for i in ranked[: args.viz_n]] +
                 [("success", i) for i in ranked[-args.viz_n:]])
        for kind, img_id in cases:
            s, t = triplets[img_id]
            r = next(x for x in per_source if x["img_id"] == img_id)
            fig, axes = plt.subplots(1, 5, figsize=(20, 4.2))
            try:
                axes[0].imshow(Image.open(s["img_path"]).convert("RGB"))
            except Exception:
                pass
            axes[0].set_title(f"{img_id}\npool={s['pool']} task={r['task_type']} "
                              f"conf={r['conf']}", fontsize=8)
            vmin = min(t[k]["s_canon"].min() for k in ("syn_a", "syn_b", "opp"))
            vmax = max(t[k]["s_canon"].max() for k in ("syn_a", "syn_b", "opp"))
            im = None
            for ax, key, ttl in ((axes[1], "syn_a", "s(instruction)"),
                                 (axes[2], "syn_b", "s(instruction_short)"),
                                 (axes[3], "opp", "s(antonym)")):
                im = ax.imshow(t[key]["s_canon"], vmin=vmin, vmax=vmax, cmap="viridis")
                ax.set_title(ttl, fontsize=9)
            if im is not None:
                fig.colorbar(im, ax=axes[3], fraction=0.046)
            luma_show = t["_luma"].copy()
            luma_show[~t["_valid"]] = np.nan  # pad 黑边区域画白
            axes[4].imshow(luma_show, cmap="gray")
            axes[4].set_title(f"luma16\nρ_syn={r['rho_syn']:.2f} ρ_opp={r['rho_opp']:.2f} "
                              f"ρ_Y={r['rho_y']:.2f}", fontsize=9)
            for ax in axes:
                ax.axis("off")
            instr = s["instructions"]["syn_a"]
            fig.suptitle(instr[:150], fontsize=8, y=0.02, va="bottom")
            fig.tight_layout()
            fig.savefig(viz / f"{kind}_{img_id}.png", dpi=110,
                        bbox_inches="tight")
            plt.close(fig)
        print(f"viz: {2 * args.viz_n} cases -> {viz}")

    # ---- viz：区域对立案例（success = 分离 ρ 低；failure = 不分离 ρ 高）----
    if region_cases and args.viz_n:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        from PIL import Image

        viz = out / "viz"
        viz.mkdir(exist_ok=True)
        ranked = sorted(region_cases, key=lambda k: region_cases[k]["rho"])
        # 命名纪律（D-30/终版）：文件名**只陈述实测判据值与实判结果**，不出现 success/
        # bestcase 这类会对外声称实验没达到的结论的词。格式：
        #   region_<PASS|FAIL>_rho<实测 ρ_region_opp>_<minrho|maxrho>_<img_id>.png
        # minrho = 全批最分离的一端（ρ 最低），maxrho = 最不分离的一端（ρ 最高）。
        def _region_kind(img_id: str, end: str) -> str:
            rho = region_cases[img_id]["rho"]
            return f"region_{'PASS' if rho < 0.3 else 'FAIL'}_rho{rho:.2f}_{end}"

        cases = ([(_region_kind(i, "minrho"), i) for i in ranked[: args.viz_n]] +
                 [(_region_kind(i, "maxrho"), i) for i in ranked[-args.viz_n:]])
        for kind, img_id in cases:
            c = region_cases[img_id]
            rs = c["sample"]
            rrow = next((x for x in region_rows if x["img_id"] == img_id), {})
            fig, axes = plt.subplots(1, 6, figsize=(24, 4.2))
            try:
                axes[0].imshow(Image.open(rs["img_path"]).convert("RGB"))
            except Exception:
                pass
            axes[0].set_title(f"{img_id}\npool={rs['pool']} dir={rs['direction']} "
                              f"regB={rs['region_b_kind']} conf={rs['winner_confidence']}",
                              fontsize=8)
            vmin = min(c["s_a"].min(), c["s_b"].min())
            vmax = max(c["s_a"].max(), c["s_b"].max())
            im = None
            for ax, s_map, ttl in ((axes[1], c["s_a"], "s(reg_a: subject)"),
                                   (axes[2], c["s_b"], "s(reg_b: complement)")):
                im = ax.imshow(s_map, vmin=vmin, vmax=vmax, cmap="viridis")
                ax.set_title(ttl, fontsize=9)
            if im is not None:
                fig.colorbar(im, ax=axes[2], fraction=0.046)
            d = c["s_a"] - c["s_b"]
            lim = float(np.abs(d[c["valid"]]).max()) or 1.0
            axes[3].imshow(np.where(c["valid"], d, np.nan), cmap="coolwarm",
                           vmin=-lim, vmax=lim)
            axes[3].set_title(f"s(reg_a) − s(reg_b)\nρ_region_opp={c['rho']:.2f} "
                              f"(判据 <0.30 → {'PASS' if c['rho'] < 0.3 else 'FAIL'})",
                              fontsize=9)
            if c.get("subj") is not None:
                axes[4].imshow(np.where(c["valid"], c["subj"], np.nan),
                               cmap="magma", vmin=0, vmax=1)
                axes[4].set_title(
                    "SAM3 subject mask\n"
                    f"AUC(a)={rrow.get('auc_rega_light', float('nan')):.2f} "
                    f"AUC(b)={rrow.get('auc_regb_light', float('nan')):.2f}", fontsize=9)
            else:
                axes[4].set_title("SAM3 mask: n/a", fontsize=9)
            luma_show = c["luma"].copy()
            luma_show[~c["valid"]] = np.nan
            axes[5].imshow(luma_show, cmap="gray")
            axes[5].set_title("luma16", fontsize=9)
            for ax in axes:
                ax.axis("off")
            fig.suptitle(f"A: {rs['instructions']['reg_a'][:110]}\n"
                         f"B: {rs['instructions']['reg_b'][:110]}",
                         fontsize=7, y=0.02, va="bottom")
            fig.tight_layout()
            fig.savefig(viz / f"{kind}_{img_id}.png", dpi=110, bbox_inches="tight")
            plt.close(fig)
        print(f"region viz: {len(cases)} cases -> {viz}")

        # ---- viz：指令无关性最有说服力的失败图（D-30 终判的主图）----
        # 选样判据（预先写死，非事后挑图）：region_b_kind == "background"（指令 B 明说
        # "调背景，主体保持不变"）且 AUC(b) > AUC(a) —— 即 s 在"说背景"时对主体掩膜的
        # 判别力**反而更强**。排序分 = (AUC(b) − 0.5) + (AUC(b) − AUC(a))：既要绝对上
        # 压在主体上（远离随机 0.5），又要相对上比"说主体"时压得更死。
        # 选样规则（单条，预先写死；候选池大小随图一起报，杜绝「挑图」质疑）：
        #   ① region_b_kind == "background"（指令 B 明说「调背景，主体保持不变」）
        #   ② AUC(b) > AUC(a)                （说背景时反而更压主体 —— 要展示的现象）
        #   ③ AUC(b) >= 0.70                 （绝对意义上确实压在主体上，不只是相对更高）
        #   ④ subj_area16 >= 0.06            （16×16 上主体至少 ~15 格，图才看得出来）
        # 排序 = Δ = AUC(b) − AUC(a) 降序，取前 N。
        BLIND_AUC_MIN, BLIND_AREA_MIN = 0.70, 0.06
        blind = [r for r in region_rows
                 if r["region_b_kind"] == "background"
                 and np.isfinite(r.get("auc_rega_light", np.nan))
                 and np.isfinite(r.get("auc_regb_light", np.nan))
                 and r["auc_regb_light"] > r["auc_rega_light"]
                 and r["auc_regb_light"] >= BLIND_AUC_MIN
                 and r.get("subj_area16", 0.0) >= BLIND_AREA_MIN
                 and r["img_id"] in region_cases]
        blind.sort(key=lambda r: -(r["auc_regb_light"] - r["auc_rega_light"]))
        picked = blind[: max(2, args.viz_n // 2)]
        for rank, rrow in enumerate(picked):
            img_id = rrow["img_id"]
            c = region_cases[img_id]
            rs = c["sample"]
            v, subj = c["valid"], c.get("subj")
            fig, axes = plt.subplots(1, 4, figsize=(17.5, 5.0))
            fig.subplots_adjust(top=0.80, bottom=0.20, wspace=0.06)
            try:
                axes[0].imshow(Image.open(rs["img_path"]).convert("RGB"))
            except Exception:
                pass
            axes[0].set_title(f"源图 {img_id} ({rs['pool']})\n"
                              f"主体占 valid 格 {rrow.get('subj_area16', float('nan')):.1%}",
                              fontsize=9.5)
            if subj is not None:
                axes[1].imshow(np.where(v, subj, np.nan), cmap="magma", vmin=0, vmax=1)
            axes[1].set_title("SAM3 主体掩膜（16×16 标签）", fontsize=9.5)
            vmin = min(c["s_a"].min(), c["s_b"].min())
            vmax = max(c["s_a"].max(), c["s_b"].max())
            lab = (subj >= MASK_BIN_THR).astype(float) if subj is not None else None
            im = None
            for ax, s_map, tag, auc in (
                    (axes[2], c["s_a"], f"A：指令说「{rs['direction']} 主体」",
                     rrow["auc_rega_light"]),
                    (axes[3], c["s_b"], f"B：指令说「{rs['direction']} 背景」",
                     rrow["auc_regb_light"])):
                im = ax.imshow(s_map, vmin=vmin, vmax=vmax, cmap="viridis")
                if lab is not None:   # 主体掩膜轮廓叠在 s 图上，肉眼可判 s 压在哪
                    ax.contour(lab, levels=[0.5], colors="red", linewidths=2.0)
                ax.set_title(f"s | {tag}\nAUC(主体掩膜) = {auc:.3f}", fontsize=9.5)
            if im is not None:
                fig.colorbar(im, ax=axes[3], fraction=0.046)
            for ax in axes:
                ax.axis("off")
            d_auc = rrow["auc_regb_light"] - rrow["auc_rega_light"]
            fig.suptitle(
                "指令无关性（D-30）：B 明说「调背景、主体保持不变」，s 仍压在红圈主体上\n"
                f"AUC(b) = {rrow['auc_regb_light']:.3f}  >  AUC(a) = {rrow['auc_rega_light']:.3f}"
                f"   (Δ = +{d_auc:.3f})      ρ_region_opp = {c['rho']:.3f}（判据 <0.30 → FAIL）",
                fontsize=11, y=0.985, va="top")
            fig.text(0.5, 0.015,
                     f"A: {rs['instructions']['reg_a'][:130]}\n"
                     f"B: {rs['instructions']['reg_b'][:130]}",
                     fontsize=8.5, ha="center", va="bottom", family="monospace")
            fig.savefig(
                viz / (f"failure_instrblind_{rank + 1}_aucB{rrow['auc_regb_light']:.2f}"
                       f"_gt_aucA{rrow['auc_rega_light']:.2f}_{img_id}.png"), dpi=120)
            plt.close(fig)
        print(f"instr-blind viz: {len(picked)} cases "
              f"(candidate pool={len(blind)}, region batch={len(region_rows)}) -> {viz}")


if __name__ == "__main__":
    main()
