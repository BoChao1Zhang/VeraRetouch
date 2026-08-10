"""重算 RO-9b **受限档 × AUC 准则**（REPORT §2a，AUC 0.9200）的 out-of-fold s 场。

## 为什么需要这一步（不是 gold-plating）

汇报 §5.2 的图 5(a)(b)(c) 现用 `RO9b/config/fields_final.npz` 的 `s_fin_*`，
而那份场对应的是 **AUC_target 准则**的终配置
`{"interaction":"pre","head":"top3","layer":"wsum","post":"a1","zlayer":true}`，
其定位质量 **AUC 中位 0.7278**——**不是**报告正文里说的 0.93。
报告正文的 0.93 有两条来源，都没有落盘逐源场：
  §2a 受限档（禁学习式层加权）0.9200：`{"qq","top3","best1","aff",zlayer=False}`
  §2b 全档（linfit，用 GT 拟合 24 个层权重）0.9349
主叙事 = §2a（REPORT 原话「主叙事请引用 §2a 的 0.9200」）。

本脚本**只做读出**：复用已落盘的逐头栈 `/var/cache/veradata/ro9b_stacks_20260803`，
逐字复用 `analyze_ro9b` 的 `Data / cv_folds / run_cfg`（同 SEED=20260803、同 5 折
source-level CV、同 out-of-fold 纪律），把 §2a 终配置的 a / b / shuf 三份场存下来。
**零 GPU，不改任何实验目录。**
"""
from __future__ import annotations

import sqlite3  # noqa: F401  isort:skip  ⚑ 必须早于科学栈（战役 bug R6）
import json
import sys
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[2]
RO9B = REPO / "experiments" / "RO9b_readout_fix_20260803"
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "tools" / "readout"))
sys.path.insert(0, str(REPO / "tools" / "scache"))
sys.path.insert(0, str(REPO / "experiments" / "G1_s_identifiability_20260803"))
sys.path.insert(0, str(RO9B))

from analyze_ro9b import (  # noqa: E402
    GRID, SEED, Agg, Data, cv_folds, headline, per_case, run_cfg,
)

# REPORT §2a 终配置（`metrics_supp_restricted.json::final_auc.cfg` 逐字）
CFG_RESTRICTED = {"interaction": "qq", "head": "top3", "layer": "best1",
                  "post": "aff", "zlayer": False, "crit": "auc"}
# 对照：RO-9 原样基线（S0），与 fields_final.npz 的 s_base_* 同口径
CFG_BASE = {"interaction": "pre", "head": "mean", "layer": "canon", "post": "a1"}


def main() -> None:
    t0 = time.time()
    rng = np.random.default_rng(SEED)
    d = Data("/var/cache/veradata/ro9b_stacks_20260803", 0)
    folds = cv_folds(d.n, 5, SEED)
    print(f"[data] n={d.n} folds={[len(f) for f in folds]} ({time.time()-t0:.0f}s)",
          flush=True)

    # ---- S0 基线（无拟合，全集；与 analyze_ro9b.main 的 S0 逐字一致）----
    base = Agg(d, **CFG_BASE)
    base.mu2 = None
    base.lsel, base.lw = list(range(8, 16)), None
    bstore = {k: np.zeros((d.n, GRID, GRID), np.float32) for k in ("a", "b", "sh")}
    pc0 = per_case(base, np.arange(d.n), bstore)
    h0 = headline(d, pc0, rng)
    print(f"[S0]  AUC={h0['auc']:.4f} AUC_t={h0['auc_target_bg']:.4f} "
          f"({time.time()-t0:.0f}s)", flush=True)

    # ---- §2a 受限档终配置（5 折 out-of-fold）----
    store = {k: np.zeros((d.n, GRID, GRID), np.float32) for k in ("a", "b", "sh")}
    h, pc, info = run_cfg(d, folds, rng, store_fields=store, **CFG_RESTRICTED)
    print(f"[S4r] AUC={h['auc']:.4f} (报告 0.9200) AUC_t={h['auc_target_bg']:.4f} "
          f"(报告 0.5042) ({time.time()-t0:.0f}s)", flush=True)
    print(f"[S4r] 每折选中层 = {[i['layers'] for i in info]}", flush=True)

    np.savez_compressed(
        HERE / "fields_restricted_auc.npz",
        img_id=np.array([r["img_id"] for r in d.src]),
        img_path=np.array([r["img_path"] for r in d.src]),
        region_b_kind=d.kind,
        subject_region=np.array([r["subject_region"] for r in d.src]),
        pool=np.array([r["pool"] for r in d.src]),
        instr_a=np.array([r["instructions"]["reg_a"] for r in d.src]),
        instr_b=np.array([r["instructions"]["reg_b"] for r in d.src]),
        valid16=np.stack(d.valid), mask16=np.stack(d.mask16), luma16=np.stack(d.luma),
        s_fix_a=store["a"], s_fix_b=store["b"], s_fix_sh=store["sh"],
        s_base_a=bstore["a"], s_base_b=bstore["b"], s_base_sh=bstore["sh"],
        auc_fix_a=pc["a"], auc_fix_b=pc["b"], auc_fix_sh=pc["sh"],
        auc_base_a=pc0["a"], auc_base_b=pc0["b"], auc_base_sh=pc0["sh"],
        cfg_fix=json.dumps(CFG_RESTRICTED), cfg_base=json.dumps(CFG_BASE),
        headline_fix=json.dumps(h, default=float),
        headline_base=json.dumps(h0, default=float),
        layers_per_fold=json.dumps([i["layers"] for i in info]))
    print(f"wrote {HERE/'fields_restricted_auc.npz'} ({time.time()-t0:.0f}s)", flush=True)


if __name__ == "__main__":
    main()
