"""⚑ 出图前的**否决性检查**：换成正确着色后，两条对立指令的场是否出现肉眼可见差异？

任务卡原话：「如果两条对立指令下的场出现了肉眼可见的差异，请立即停下并报告，不要继续
出图」——因为报告 §5.1/§5.2 的核心结论正是「两条相反指令的场几乎重合」。若正确着色后
差异变得可见，说明此前的结论建立在一个被压平的可视化上。

三个口径（全部在**未归一化的原始场**上算，不经过任何着色路径）：
  ρ_valid      有效格 Pearson（G1 主判据同款；预注册 <0.30 才算分离）
  Δrank_p50    有效格秩次百分位的中位绝对差（着色无关；0 = 完全同序）
  topk_IoU     两场各取 top-k（k = GT 主体格数）后的 IoU（1 = 选中同一批格）
另加**着色可见性**口径 Δnorm_max：走本次实际出图的那条链路
（masked-smooth → 有效格联合 min-max → [0,1]），两场归一化后的最大逐格差。
经验判读：Δnorm_max < 0.15 ⇒ 同一把色标下人眼分辨不出；> 0.35 ⇒ 明显可见。
"""
from __future__ import annotations

import sqlite3  # noqa: F401  isort:skip
import hashlib
import json
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[2]
sys.path.insert(0, str(HERE))
G1 = REPO / "experiments" / "G1_s_identifiability_20260803"

import metrics as MT  # noqa: E402
import redrawlib as RL  # noqa: E402


def instr_hash(s: str, n: int = 12) -> str:
    return hashlib.md5(s.encode("utf-8")).hexdigest()[:n]


def pair_stats(fa, fb, valid, m_soft) -> dict:
    ra, rb = MT.rank_pct(fa, valid), MT.rank_pct(fb, valid)
    k = int(((m_soft >= 0.5) & valid).sum())
    ta, tb = MT.topk_mask(fa, valid, k), MT.topk_mask(fb, valid, k)
    inter = int((ta & tb).sum())
    union = int((ta | tb).sum())
    (sa, sb), lo, hi = RL.valid_norm([fa, fb], valid)
    na = np.clip((sa - lo) / (hi - lo), 0, 1)
    nb = np.clip((sb - lo) / (hi - lo), 0, 1)
    dn = np.abs(na - nb)[valid]
    return {"rho": MT.pearson_valid(fa, fb, valid),
            "d_rank_p50": float(np.nanmedian(np.abs(ra - rb)[valid])),
            "topk_iou": float(inter / union) if union else float("nan"),
            "d_norm_max": float(np.nanmax(dn)), "d_norm_p50": float(np.nanmedian(dn))}


def summarize(name: str, rows: list[dict]) -> dict:
    out = {"n": len(rows)}
    for k in ("rho", "d_rank_p50", "topk_iou", "d_norm_max", "d_norm_p50"):
        v = np.array([r[k] for r in rows], dtype=float)
        out[k] = {"p10": float(np.nanpercentile(v, 10)),
                  "median": float(np.nanmedian(v)),
                  "p90": float(np.nanpercentile(v, 90))}
    out["n_rho_below_0.30"] = int(np.nansum(np.array([r["rho"] for r in rows]) < 0.30))
    out["n_d_norm_max_above_0.35"] = int(
        np.nansum(np.array([r["d_norm_max"] for r in rows]) > 0.35))
    print(f"\n===== {name} (n={out['n']}) =====")
    for k in ("rho", "d_rank_p50", "topk_iou", "d_norm_max", "d_norm_p50"):
        s = out[k]
        print(f"  {k:12s} p10={s['p10']:+.4f}  median={s['median']:+.4f}  p90={s['p90']:+.4f}")
    print(f"  ρ<0.30 的源：{out['n_rho_below_0.30']}/{out['n']}"
          f"   Δnorm_max>0.35 的源：{out['n_d_norm_max_above_0.35']}/{out['n']}")
    return out


def main() -> None:
    res = {}

    # ---------------- A 组：G1 区域对立批（canonical s，GL=<retouch_light>）----------
    reg = json.loads((G1 / "config" / "g1_region_opp.json").read_text())
    sdir = G1 / "run_regfull" / "stacks"
    g1b = REPO / "experiments" / "G1b_difflmm_20260803" / "config"
    with np.load(g1b / "subject_masks16.npz") as z:
        masks = {k: z[k] for k in z.files}
    rows, per = [], {}
    for r in reg:
        pa = sdir / f"{r['img_id']}__{instr_hash(r['instructions']['reg_a'])}.npz"
        pb = sdir / f"{r['img_id']}__{instr_hash(r['instructions']['reg_b'])}.npz"
        if not (pa.is_file() and pb.is_file()) or r["img_id"] not in masks:
            continue
        sa = np.load(pa)["s_canon"].astype(np.float32)
        sb = np.load(pb)["s_canon"].astype(np.float32)
        _, valid = RL.luma_valid(r["img_path"])
        st = pair_stats(sa, sb, valid, masks[r["img_id"]])
        st["img_id"] = r["img_id"]
        rows.append(st)
        per[r["img_id"]] = st
    res["A_G1_region_opposition"] = summarize("A 组 · G1 区域对立（s_canon）", rows)
    res["A_per_source"] = per

    # ---------------- B 组：RO-9b 修正后（受限档 §2a，AUC 0.9200）------------------
    z = np.load(HERE / "fields_restricted_auc.npz", allow_pickle=True)
    ids = [str(x) for x in z["img_id"]]
    for tag, ka, kb in (("修正后 §2a", "s_fix_a", "s_fix_b"),
                        ("修正前 S0", "s_base_a", "s_base_b")):
        rows, per = [], {}
        for i, iid in enumerate(ids):
            st = pair_stats(z[ka][i], z[kb][i], z["valid16"][i], z["mask16"][i])
            st["img_id"] = iid
            rows.append(st)
            per[iid] = st
        res[f"B_RO9b_{ka}_vs_{kb}"] = summarize(f"B 组 · {tag}：指令A vs 指令B", rows)
        res[f"B_per_source_{ka}"] = per

    (HERE / "check_opposite.json").write_text(
        json.dumps(res, indent=1, ensure_ascii=False, default=float))
    print(f"\nwrote {HERE/'check_opposite.json'}")


if __name__ == "__main__":
    main()
