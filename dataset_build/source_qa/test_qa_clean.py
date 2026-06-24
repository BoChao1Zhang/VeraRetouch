"""离线纯规则单测（无 GPU / 无 DB），三套清洗器 + 反作弊 POS_MAP 间隔约束。

设计文档 §4.1 第 1 步：实现前三套清洗器必须先过此单测。
运行：  python -m dataset_build.source_qa.test_qa_clean
或：    pytest dataset_build/source_qa/test_qa_clean.py
"""
from __future__ import annotations

import math

from . import config as C
from . import qa_clean as Q


# --------------------------------------------------------------------------- #
# 构造 "好图" 应答串的工具
# --------------------------------------------------------------------------- #
def _imq_good(pos_map, asset, overrides=None):
    overrides = overrides or {}
    parts = []
    for p in sorted(pos_map):
        claim, pol, aid = pos_map[p]
        if aid in overrides:
            b = overrides[aid]
        elif pol == "F":
            b = 1
        elif pol == "R":
            b = 0
        elif claim == "ANCHOR":
            b = 1
        elif claim == "LANDSCAPE":
            w, h = asset["width"], asset["height"]
            b = 1 if (w is not None and h is not None and w > h) else 0
        elif claim == "FACE_GT":
            mff = asset.get("max_face_frac")
            b = 1 if (mff is not None and mff > C.IMQ_FACE_MIN) else 0
        elif claim == "COLOR":
            b = 0 if asset.get("is_bw_img") else 1
        else:
            b = 0
        parts.append(f"{p}{b}")
    return " ".join(parts)


def _aes_good(pos_map, asset, overrides=None):
    overrides = overrides or {}
    parts = []
    for p in sorted(pos_map):
        claim, pol, aid = pos_map[p]
        if aid in overrides:
            b = overrides[aid]
        elif pol == "F":
            b = 1
        elif pol == "R":
            b = 0
        elif aid == "T1":
            b = 1
        elif aid == "H1":
            b = 1
        elif aid == "H2":
            b = 0
        elif aid == "T4":
            b = 1
        elif aid == "T2":
            w, h = asset["width"], asset["height"]
            b = 1 if (w is not None and h is not None and w > h) else 0
        elif aid == "T3":
            b = 0 if asset.get("is_bw_img") else 1
        else:
            b = 0
        parts.append(f"{p}{b}")
    return " ".join(parts)


def _preset_good(pm, t2_truth, t3_truth, overrides=None):
    overrides = overrides or {}
    good = {"P1": 1, "P2": 0, "I1": 1, "I2": 0, "H1": 1, "H2": 0,
            "T1": 1, "T1b": 0, "T2": t2_truth, "T3": t3_truth}
    good.update(overrides)
    parts = []
    for p in sorted(pm):
        aid = pm[p][2]
        parts.append(f"{p}{good[aid]}")
    return " ".join(parts)


LAND = {"is_portrait_pool": 0, "width": 1200, "height": 800,
        "max_face_frac": None, "is_bw_img": 0}
PORTRAIT = {"is_portrait_pool": 1, "width": 800, "height": 1200,
            "max_face_frac": 0.20, "is_bw_img": 0}


# --------------------------------------------------------------------------- #
# 反作弊: POS_MAP 间隔约束（每对 F/R 间隔 ≥⌈N/3⌉ 且不相邻）
# --------------------------------------------------------------------------- #
def _pair_positions(pos_map):
    """返回 {pair_key: [int_pos,...]}，pair_key=(claim) 仅取恰有 2 个位置的对。"""
    from collections import defaultdict
    g = defaultdict(list)
    for p, (claim, pol, aid) in pos_map.items():
        g[claim].append(int(p))
    return {k: sorted(v) for k, v in g.items() if len(v) == 2}


def test_scramble_intervals():
    for name, pm in [("IMQ", C.IMQ_POS_MAP), ("IMQ_P", C.IMQ_POS_MAP_PORTRAIT),
                     ("AES", C.AES_POS_MAP), ("AES_P", C.AES_POS_MAP_PORTRAIT),
                     ("PRESET", C.PRESET_POS_MAP)]:
        N = len(pm)
        need = math.ceil(N / 3)
        # 位置码连续 01..N
        assert sorted(pm) == [f"{i:02d}" for i in range(1, N + 1)], f"{name} codes not contiguous"
        for claim, (a, b) in _pair_positions(pm).items():
            gap = b - a
            assert gap >= need, f"{name} pair {claim} gap {gap} < ceil(N/3)={need}"
            assert gap >= 2, f"{name} pair {claim} adjacent"
        # 审计 ID 唯一
        aids = [v[2] for v in pm.values()]
        assert len(aids) == len(set(aids)), f"{name} dup audit ids"


# --------------------------------------------------------------------------- #
# IMQ 清洗器
# --------------------------------------------------------------------------- #
def test_imq_good_nonportrait():
    r = Q.clean_imq(_imq_good(C.IMQ_POS_MAP, LAND), LAND)
    assert r["reliable"], r["reason"]
    assert r["defect_count"] == 0
    assert r["contradiction_count"] == 0


def test_imq_good_portrait():
    r = Q.clean_imq(_imq_good(C.IMQ_POS_MAP_PORTRAIT, PORTRAIT), PORTRAIT)
    assert r["reliable"], r["reason"]
    assert r["defect_count"] == 0


def test_imq_defect_count():
    # 3 道缺陷干净成立(F=0,R=1) → defect_count=3, 无矛盾
    s = _imq_good(C.IMQ_POS_MAP, LAND,
                  {"SHARP_F": 0, "SHARP_R": 1, "NOISE_F": 0, "NOISE_R": 1,
                   "COMP_F": 0, "COMP_R": 1})
    r = Q.clean_imq(s, LAND)
    assert r["reliable"], r["reason"]
    assert r["contradiction_count"] == 0
    assert r["defect_count"] == 3
    v, reason = Q.verdict_imq(r["A"], False, None, r["defect_count"])
    assert v == "review"          # DEF_REVIEW=2, 未开 DEFECT_HARD_DROP


def test_imq_parse_empty():
    assert Q.clean_imq("", LAND)["reason"] == "parse:empty"


def test_imq_parse_missing():
    s = _imq_good(C.IMQ_POS_MAP, LAND)
    s = " ".join(s.split()[:-1])   # 丢最后一题
    assert Q.clean_imq(s, LAND)["reason"] == "parse:missing"


def test_imq_parse_dup_conflict():
    # F2: 同码冲突比特(01 既 1 又 0) → parse:dup
    s = _imq_good(C.IMQ_POS_MAP, LAND) + " 010"
    assert Q.clean_imq(s, LAND)["reason"] == "parse:dup"


def test_imq_parse_dup_benign_salvaged():
    # F2: 同码重复同一比特(良性, round1 真实根因) → 折叠, 不判 dup, 仍可信
    s = _imq_good(C.IMQ_POS_MAP, LAND)
    first = s.split()[0]           # '011' (REAL_F=1)
    r = Q.clean_imq(s + " " + first, LAND)
    assert r["reliable"], r["reason"]


def test_imq_parse_unknown():
    s = _imq_good(C.IMQ_POS_MAP, LAND) + " 991"
    assert Q.clean_imq(s, LAND)["reason"] == "parse:unknown"


def test_imq_anchor_fail():
    r = Q.clean_imq(_imq_good(C.IMQ_POS_MAP, LAND, {"ANCHOR": 0}), LAND)
    assert r["reason"] == "anchor"


def test_imq_trap_face_fail():
    # Haar 正检(mff=0.20>FACE_MIN) 但 FACE_GT 答 0 → trap:face(强制真值=1)
    r = Q.clean_imq(_imq_good(C.IMQ_POS_MAP_PORTRAIT, PORTRAIT, {"FACE_GT": 0}), PORTRAIT)
    assert r["reason"] == "trap:face"
    assert r["trap_fail"] == "face"


def test_imq_face_skip_when_haar_miss():
    # F3: Haar 完全漏检(mff=0.0) 时 FACE_GT 不入硬门, 即便答 1(真有脸 Haar 漏) 也不剔
    portrait_noface_signal = {**PORTRAIT, "max_face_frac": 0.0}
    r1 = Q.clean_imq(_imq_good(C.IMQ_POS_MAP_PORTRAIT, portrait_noface_signal, {"FACE_GT": 1}),
                     portrait_noface_signal)
    assert r1["reliable"], r1["reason"]
    assert r1["trap_skipped"] >= 1
    # mff 略低于 FACE_MIN 同样 skip
    pp2 = {**PORTRAIT, "max_face_frac": 0.008}
    r2 = Q.clean_imq(_imq_good(C.IMQ_POS_MAP_PORTRAIT, pp2, {"FACE_GT": 1}), pp2)
    assert r2["reliable"], r2["reason"]


def test_imq_landscape_soft_when_no_exif():
    # HAS_EXIF_FIXED=False → LANDSCAPE 答错只软监控, 不置 reliable=False
    assert not C.HAS_EXIF_FIXED
    r = Q.clean_imq(_imq_good(C.IMQ_POS_MAP, LAND, {"LANDSCAPE": 0}), LAND)
    assert r["reliable"], "LANDSCAPE 在 EXIF 未修复时不应硬剔"
    assert r["landscape_soft_fail"] == 1


def test_imq_contradiction_hard():
    # REAL_F=1 且 REAL_R=1 → 硬对矛盾(REAL 仍硬对)
    r = Q.clean_imq(_imq_good(C.IMQ_POS_MAP, LAND, {"REAL_R": 1}), LAND)
    assert r["reason"] == "contradiction_hard"


def test_imq_clean_demoted_to_soft():
    # G3: CLEAN both-0(模型欠检叠加物) 不再硬矛盾, 转 reliable→verdict review(invalid:CLEAN)
    r = Q.clean_imq(_imq_good(C.IMQ_POS_MAP, LAND, {"CLEAN_F": 0, "CLEAN_R": 0}), LAND)
    assert r["reliable"], r["reason"]
    v, reason = Q.verdict_imq(r["A"], False, None, r["defect_count"])
    assert v == "review" and reason.startswith("invalid:CLEAN")


def test_imq_contradiction_soft_tol():
    # G1: 软对矛盾仅计 F=1∧R=1。1 个(SHARP both-1)在 IMQ_CONTRA_TOL=1 内 → 仍 reliable
    r = Q.clean_imq(_imq_good(C.IMQ_POS_MAP, LAND, {"SHARP_R": 1, "SHARP_F": 1}), LAND)
    assert r["reliable"], r["reason"]
    assert r["contradiction_count"] == 1
    # 2 个 both-1 软对 → 超 tol
    r2 = Q.clean_imq(_imq_good(C.IMQ_POS_MAP, LAND,
                               {"SHARP_F": 1, "SHARP_R": 1, "NOISE_F": 1, "NOISE_R": 1}), LAND)
    assert r2["reason"] == "contradiction_soft"


def test_imq_both0_not_contradiction():
    # G1: 软对 both-0(中庸/缺陷不成立) 不计矛盾。把多对设 both-0 → 仍 reliable
    ov = {"SHARP_F": 0, "SHARP_R": 0, "NOISE_F": 0, "NOISE_R": 0,
          "COMP_F": 0, "COMP_R": 0, "INTACT_F": 0, "INTACT_R": 0}
    r = Q.clean_imq(_imq_good(C.IMQ_POS_MAP, LAND, ov), LAND)
    assert r["reliable"], r["reason"]
    assert r["contradiction_count"] == 0
    assert r["defect_count"] == 0     # R 全 0 → 无缺陷


def test_imq_reconcile_iqa_conflict():
    r = Q.clean_imq(_imq_good(C.IMQ_POS_MAP, LAND), LAND)
    v, reason = Q.verdict_imq(r["A"], False, None, r["defect_count"])
    assert v == "keep"
    bad_iqa = {"musiq": 20.0, "niqe": None, "noise_sigma": None}
    # IQA 关(默认): 纯 LLM QA, reconcile 放行不降级
    assert not C.IQA_IN_CLEAN
    assert Q.reconcile_imq(v, reason, bad_iqa) == (v, reason)
    # IQA 开: 软尾降级 review (临时开启验证旧行为仍在)
    orig = C.IQA_IN_CLEAN
    try:
        C.IQA_IN_CLEAN = True
        v2, reason2 = Q.reconcile_imq(v, reason, bad_iqa)
        assert v2 == "review" and reason2 == "iqa_conflict"
    finally:
        C.IQA_IN_CLEAN = orig


# --------------------------------------------------------------------------- #
# AES 清洗器
# --------------------------------------------------------------------------- #
def test_aes_good_nonportrait():
    r = Q.clean_aes(_aes_good(C.AES_POS_MAP, LAND), LAND)
    assert r["reliable"], r["reason"]
    assert r["merit_count"] == 9 and r["merit_n"] == 9
    assert r["merit_frac"] == 1.0


def test_aes_good_portrait():
    r = Q.clean_aes(_aes_good(C.AES_POS_MAP_PORTRAIT, PORTRAIT), PORTRAIT)
    assert r["reliable"], r["reason"]
    assert r["merit_n"] == 10           # 有脸追加 M_moment


def test_aes_merit_partial():
    # 6 对设 F=0,R=1(干净 XOR, 非 merit 命中); 其余 3 对仍 1/0(merit 命中) → merit=3
    ov = {}
    for fa, ra in (("K1", "K2"), ("L1", "L2"), ("C1", "C2"),
                   ("L3", "L4"), ("D1", "D2"), ("M1", "M2")):
        ov[fa] = 0
        ov[ra] = 1
    r = Q.clean_aes(_aes_good(C.AES_POS_MAP, LAND, ov), LAND)
    assert r["reliable"], r["reason"]
    assert r["contradiction_count"] == 0
    assert r["merit_count"] == 3        # 9 对中 3 对(K_frame/D_sep/N_clean)仍 1/0
    assert abs(r["merit_frac"] - 3 / 9) < 1e-9


def test_aes_anchor_fail():
    assert Q.clean_aes(_aes_good(C.AES_POS_MAP, LAND, {"T1": 0}), LAND)["reason"] == "anchor"


def test_aes_honesty_fail_all_ones():
    # 全 1 敷衍: H2 应为 0 却答 1 → honesty
    assert Q.clean_aes(_aes_good(C.AES_POS_MAP, LAND, {"H2": 1}), LAND)["reason"] == "honesty"


def test_aes_trap_t4_fail():
    r = Q.clean_aes(_aes_good(C.AES_POS_MAP_PORTRAIT, PORTRAIT, {"T4": 0}), PORTRAIT)
    assert r["reason"] == "trap:T4"


def test_aes_contradiction_soft():
    # G1: 仅 F=1∧R=1 计矛盾。2 对 both-1(K1=1,K2=1;L1=1,L2=1) → viol=2 > AES_CONTRA_TOL=1
    r = Q.clean_aes(_aes_good(C.AES_POS_MAP, LAND, {"K2": 1, "L2": 1}), LAND)
    assert r["reason"] == "contradiction_soft"
    assert r["contradiction_count"] == 2


def test_aes_both0_not_contradiction():
    # G1: both-0(中庸维度) 不计矛盾, 也不命中 merit。6 对设 both-0 → reliable, merit=3
    ov = {}
    for fa, ra in (("K1", "K2"), ("L1", "L2"), ("C1", "C2"),
                   ("L3", "L4"), ("D1", "D2"), ("M1", "M2")):
        ov[fa] = 0
        ov[ra] = 0           # both-0 中庸
    r = Q.clean_aes(_aes_good(C.AES_POS_MAP, LAND, ov), LAND)
    assert r["reliable"], r["reason"]
    assert r["contradiction_count"] == 0
    assert r["merit_count"] == 3     # 剩 3 对仍 1/0


def test_aes_never_drops():
    # 不可信 → merit_frac None, 但绝不产生 drop verdict（软信号）
    r = Q.clean_aes("", LAND)
    assert not r["reliable"] and r["merit_frac"] is None


def test_aes_soft_trap_landscape():
    assert not C.HAS_EXIF_FIXED
    r = Q.clean_aes(_aes_good(C.AES_POS_MAP, LAND, {"T2": 0}), LAND)   # 横图答竖
    assert r["reliable"]                # T2 软监控不硬剔
    assert r["soft_trap_fail"] == 1


# --------------------------------------------------------------------------- #
# preset 清洗器
# --------------------------------------------------------------------------- #
PM_CHANGE = {"delta_e2000_mean": 8.0, "ssim": 0.97, "hist_emd_ab": 0.5, "noop_score": 0}
PM_NOOP = {"delta_e2000_mean": 1.0, "ssim": 0.999, "hist_emd_ab": 0.02, "noop_score": 1}
PM_DESTRUCT = {"delta_e2000_mean": 12.0, "ssim": 0.80, "hist_emd_ab": 0.9, "noop_score": 0}


def test_preset_good_probe():
    # 明显变化探针: t2_truth=0(ΔE>2.5), t3_truth=0(ssim>0.9)
    s = _preset_good(C.PRESET_POS_MAP, 0, 0)
    r = Q.clean_preset_probe(s, PM_CHANGE, "param")
    assert r["reliable"], r["reason"]


def test_preset_anchor_neg():
    # 纯 acquiescence: T1=1,T1b=1 → anchor(T1b!=0)
    s = _preset_good(C.PRESET_POS_MAP, 0, 0, {"T1b": 1})
    assert Q.clean_preset_probe(s, PM_CHANGE, "param")["reason"] == "anchor"


def test_preset_contradiction_pro():
    s = _preset_good(C.PRESET_POS_MAP, 0, 0, {"P2": 1})   # P1=1,P2=1
    assert Q.clean_preset_probe(s, PM_CHANGE, "param")["reason"] == "contradiction_hard:PRO"


def test_preset_contradiction_intent():
    s = _preset_good(C.PRESET_POS_MAP, 0, 0, {"I2": 1})
    assert Q.clean_preset_probe(s, PM_CHANGE, "param")["reason"] == "contradiction_hard:INTENT"


def test_preset_coh_soft():
    s = _preset_good(C.PRESET_POS_MAP, 0, 0, {"H2": 1})   # H1=1,H2=1, CONTRA_TOL=0
    assert Q.clean_preset_probe(s, PM_CHANGE, "param")["reason"] == "contradiction_soft:COH"


def test_preset_trap_soft_when_not_hard():
    # T2 答错(说"几乎一样"但 ΔE=8): 未开 PRESET_T2_HARD → 只软标记, 仍 reliable
    assert not C.PRESET_T2_HARD
    s = _preset_good(C.PRESET_POS_MAP, 1, 0)   # T2=1 但真值=0
    r = Q.clean_preset_probe(s, PM_CHANGE, "param")
    assert r["reliable"]
    assert "T2" in (r["trap_fail"] or "")


def test_preset_parse_missing_probe_independent():
    # 多探针: 一个探针解析失败不污染其它（probe_id 独立, 不覆盖）
    good = [Q.clean_preset_probe(_preset_good(C.PRESET_POS_MAP, 0, 0), PM_CHANGE, "param")
            for _ in range(5)]
    bad = Q.clean_preset_probe("011 021", PM_CHANGE, "param")
    assert not bad["reliable"] and bad["reason"].startswith("parse:")
    probes = good + [bad]
    agg = Q.aggregate_preset(probes)
    assert agg["reliable_probe_count"] == 5      # 坏探针不计入


def test_preset_order0_noop_drop():
    previews = [PM_NOOP] * 6
    probes = [Q.clean_preset_probe(_preset_good(C.PRESET_POS_MAP, 1, 0), PM_NOOP, "param")
              for _ in range(6)]
    out = Q.map_verdict_preset(previews, probes, "param")
    assert out["auto_verdict"] == "drop" and out["verdict_reason"] == "near_noop"
    assert out["pass_c"] == 0


def test_preset_all_pass_review_never_keep():
    previews = [PM_CHANGE] * 6
    probes = [Q.clean_preset_probe(_preset_good(C.PRESET_POS_MAP, 0, 0), PM_CHANGE, "param")
              for _ in range(6)]
    out = Q.map_verdict_preset(previews, probes, "param")
    assert out["pass_c"] == 1
    assert out["auto_verdict"] == "review"       # preset 永不自动 keep
    assert out["verdict_reason"] == "all_pass"


def test_preset_insufficient_probes():
    previews = [PM_CHANGE] * 6
    probes = [Q.clean_preset_probe("011 021", PM_CHANGE, "param") for _ in range(6)]  # 全坏
    out = Q.map_verdict_preset(previews, probes, "param")
    assert out["verdict_reason"] == "insufficient_reliable_probes"
    assert out["pass_c"] == 0 and out["auto_verdict"] == "review"


def test_preset_no_auto_keep_ever():
    # 任意组合都不得产出 auto_verdict='keep'
    for t2 in (0, 1):
        for t3 in (0, 1):
            previews = [PM_CHANGE] * 6
            probes = [Q.clean_preset_probe(_preset_good(C.PRESET_POS_MAP, t2, t3), PM_CHANGE, "param")
                      for _ in range(6)]
            out = Q.map_verdict_preset(previews, probes, "param")
            assert out["auto_verdict"] != "keep"


# --------------------------------------------------------------------------- #
# tag_preset_function
# --------------------------------------------------------------------------- #
def _sig(name, dL=0, da=0, db=0, after_C=40, before_C=40, contrast_ratio=1.0, shadow_dL=0,
         a_before=0.0, b_before=0.0):
    # a/b_after derived from before + delta so hue rotation is computable
    return {"name": name, "dL": dL, "da": da, "db": db, "dC": after_C - before_C,
            "after_C": after_C, "before_C": before_C, "contrast_ratio": contrast_ratio,
            "shadow_dL": shadow_dL, "a_before": a_before, "b_before": b_before,
            "a_after": a_before + da, "b_after": b_before + db}


def test_tag_temperature_from_neutral():
    # 中性探针读色温: 中性 Δb*>0 → warm, 即便色探针被压缩
    sigs = [_sig(n, db=-20, after_C=20, before_C=50) for n in ("red", "yellow", "green", "blue")]
    sigs += [_sig("skin", db=2), _sig("neutral", db=8, da=-5)]  # 中性偏暖+绿罩
    out = Q.tag_preset_function(sigs)
    assert out["temperature"] == "warm" and out["tint"] == "green", out


def test_tag_bw():
    sigs = [_sig(n, after_C=2, before_C=45) for n in ("red", "yellow", "green", "blue", "skin", "neutral")]
    out = Q.tag_preset_function(sigs)
    assert out["grade_family"] == "bw" and out["saturation"] == "bw", out


def test_tag_muted_relative():
    # 色探针相对 chroma 大幅下降 → muted
    sigs = [_sig(n, after_C=30, before_C=50) for n in ("red", "yellow", "green", "blue")]
    sigs += [_sig("skin"), _sig("neutral")]
    out = Q.tag_preset_function(sigs)
    assert out["saturation"] == "muted", out


def test_tag_vintage_film():
    sigs = [_sig(n, shadow_dL=12, contrast_ratio=0.82, after_C=30, before_C=45)
            for n in ("red", "yellow", "green", "blue", "skin", "neutral")]
    out = Q.tag_preset_function(sigs)
    assert out["grade_family"] == "vintage_film", out
    assert out["tone"] == "lifted" and out["contrast"] == "flat"


def test_tag_teal_orange():
    # 蓝探针色相 290°→200° (向青旋转, Δ负) + 暖探针 chroma 保留
    sigs = [_sig("blue", a_before=10, b_before=-27, da=-25, db=20, after_C=22, before_C=29)]  # 旋向 teal
    sigs += [_sig("red", after_C=40, before_C=42), _sig("skin", after_C=30, before_C=32),
             _sig("yellow"), _sig("green"), _sig("neutral")]
    out = Q.tag_preset_function(sigs)
    assert out["metrics"]["teal_rot"] < -12, out
    assert out["grade_family"] == "teal_orange", out


def test_tag_clean_natural():
    sigs = [_sig(n) for n in ("red", "yellow", "green", "blue", "skin", "neutral")]
    out = Q.tag_preset_function(sigs)
    assert out["grade_family"] == "clean_natural", out
    assert out["tags"] == ["family:clean_natural"], out


def test_tag_empty():
    assert Q.tag_preset_function([])["grade_family"] == "unknown"


# --------------------------------------------------------------------------- #
def _run_all():
    fns = [(n, f) for n, f in sorted(globals().items())
           if n.startswith("test_") and callable(f)]
    passed = 0
    for n, f in fns:
        f()
        passed += 1
        print(f"  ok  {n}")
    print(f"\n{passed}/{len(fns)} tests passed")


if __name__ == "__main__":
    _run_all()
