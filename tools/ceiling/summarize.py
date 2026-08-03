"""G2 汇总：把三档解析结果 + MLP 探针 + GLUT 互证收成一份 metrics.json（判据并排）。"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

# L0 = 全局编辑（掩膜恒 1），L5 = mismatch_semantic_gt（掩膜与编辑区**故意**错配）。
# 这两级按构造就该给 Δ_ceil≈0，正是本实验的两个零点对照，不能算进
# 「4D 相对 3D 有增益」的判据分母 —— 判据①因此在**局部级**上判，全 8 级中位数并列报出。
LOCAL_LEVELS = ("L1", "L2", "L3", "L4", "L6", "L7")
NULL_LEVELS = ("L0", "L5")

CRITERIA = {
    "construct_delta_ge_8db": {"desc": "① D-CONSTRUCT(S-val) **局部级(L1-L4,L6,L7)** Δ_ceil 中位数 ≥ 8 dB",
                               "threshold": 8.0, "op": ">="},
    "real_delta_ge_1db": {"desc": "② 真实档 D-SFT-L(S-val,normal) Δ_ceil 中位数 ≥ 1.0 dB",
                          "threshold": 1.0, "op": ">="},
    "real_delta_death_lt_04db": {"desc": "② 死刑线：真实档 Δ_ceil < 0.4 dB → 渲染器线停工",
                                 "threshold": 0.4, "op": "<", "kind": "death_line"},
    "control_delta_near_zero": {"desc": "③ 对照档 D-SFT-G(S-val) Δ_ceil ≈ 0（方法不虚高）",
                                "threshold": 0.4, "op": "<"},
    "mlp_probe_ge_45db": {"desc": "D0-4 MLP 容量探针（4→256×4→3）单样本 overfit ≥ 45 dB",
                          "threshold": 45.0, "op": ">="},
    "mlp_probe_death_lt_40db": {"desc": "MLP 探针 < 40 dB → 问题不在渲染器，全线转修数据",
                                "threshold": 40.0, "op": "<", "kind": "death_line"},
    "bandwidth_spearman_gt_08": {"desc": "PLAN §4.1 Step 6 带宽稳定性 Spearman(33³,17³) > 0.8",
                                 "threshold": 0.8, "op": ">"},
    "l0_degenerate_lt_01db": {"desc": "PLAN §4.1 Step 4 退化验收：全局档 L0 Δ_ceil < 0.1 dB",
                              "threshold": 0.1, "op": "<"},
}


def _load(p: Path):
    return json.load(open(p, encoding="utf-8")) if p.exists() else None


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--exp-dir", type=Path, required=True)
    args = ap.parse_args()
    m = args.exp_dir / "metrics"

    out: dict = {"exp": "G2_oracle_ceiling", "date": "2026-08-03",
                 "primary_estimator": "affine (逐箱最小二乘仿射)",
                 "headline_key": "delta_arm（臂式 4D：每个 s 桶一张独立 3D 表）",
                 "tracks": {}, "criteria": {}}

    for track, label in (("construct", "① D-CONSTRUCT(S-val)"),
                         ("real", "② D-SFT-L(S-val,normal) 真实档"),
                         ("control", "③ D-SFT-G(S-val) 对照档")):
        a = _load(m / f"agg_{track}.json")
        if a:
            out["tracks"][track] = {"label": label, **a}

    mlp = _load(m / "mlp_probe.json")
    if mlp:
        out["mlp_probe"] = {k: v for k, v in mlp.items() if k != "per_sample"}
        out["mlp_probe"]["per_sample"] = [
            {k: v for k, v in r.items() if not k.startswith("curve")}
            for r in mlp.get("per_sample", [])]
    xc = _load(m / "glut_crosscheck.json")
    if xc:
        out["glut_crosscheck"] = xc["agg"]

    def med(track: str, key: str):
        t = out["tracks"].get(track)
        return None if not t or key not in t else t[key]["median"]

    c_all, r_arm, g_arm = (med(t, "delta_arm") for t in ("construct", "real", "control"))
    # 判据① 只在局部级上判（见 LOCAL_LEVELS 注释）
    cp = m / "per_image_construct.jsonl"
    c_arm = c_all
    if cp.exists():
        rows_c = [json.loads(x) for x in open(cp, encoding="utf-8")]
        loc = [r["delta_arm"] for r in rows_c if r["level"] in LOCAL_LEVELS]
        nul = [r["delta_arm"] for r in rows_c if r["level"] in NULL_LEVELS]
        if loc:
            c_arm = float(np.median(loc))
        out["construct_split"] = {
            "local_levels": list(LOCAL_LEVELS), "null_levels": list(NULL_LEVELS),
            "delta_arm_median_local": c_arm,
            "delta_arm_median_null": float(np.median(nul)) if nul else None,
            "delta_arm_median_all8": c_all,
            "n_local": len(loc), "n_null": len(nul)}
    r_nest = med("real", "delta")
    sp = (out["tracks"].get("real") or {}).get("spearman_delta_arm_vs_alt")
    l0 = None
    cby = (out["tracks"].get("construct") or {}).get("by_level", {})
    if "L0" in cby:
        l0 = cby["L0"]["delta_arm"]["median"]
    mlp_min = (mlp or {}).get("psnr_4d_min")

    measured = {
        "construct_delta_ge_8db": c_arm,
        "real_delta_ge_1db": r_arm,
        "real_delta_death_lt_04db": r_arm,
        "control_delta_near_zero": g_arm,
        "mlp_probe_ge_45db": mlp_min,
        "mlp_probe_death_lt_40db": mlp_min,
        "bandwidth_spearman_gt_08": sp,
        "l0_degenerate_lt_01db": l0,
    }
    for k, spec in CRITERIA.items():
        v = measured.get(k)
        ok = None
        if v is not None:
            ok = (v >= spec["threshold"] if spec["op"] == ">=" else
                  v > spec["threshold"] if spec["op"] == ">" else
                  v < spec["threshold"])
        rec = {**spec, "measured": v}
        if spec.get("kind") == "death_line":
            # 死刑线：`op` 描述的是**触发**条件，触发 = 坏消息
            rec["triggered"] = ok
            rec["ok"] = None if ok is None else (not ok)
        else:
            rec["pass"] = ok
            rec["ok"] = ok
        rec.setdefault("kind", "gate")
        out["criteria"][k] = rec

    # 「自家数据局部信号强度」= ② − ③（预登记数字）
    if r_arm is not None and g_arm is not None:
        out["local_signal_strength_db"] = {
            "desc": "②−③：真实档 Δ_ceil 中位数 − 对照档 Δ_ceil 中位数",
            "value": r_arm - g_arm, "real": r_arm, "control": g_arm}
    if r_nest is not None:
        out["real_nested_delta_median"] = r_nest

    (args.exp_dir / "metrics.json").write_text(
        json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({k: {"kind": v["kind"], "measured": v["measured"], "ok": v["ok"]}
                      for k, v in out["criteria"].items()}, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
