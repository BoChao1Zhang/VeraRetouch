r"""纯规则确定性清洗器（三套问卷共享，无 GPU / 无 DB 依赖）。

实现设计文档 LLM_QA_QUESTIONNAIRE_DESIGN_2026-06-17 的：
  * 流程 1 IMQ  §1.6 5 门清洗器 + §1.7 verdict / reconcile
  * 流程 2 AES  §2.6 6 门清洗器 + §2.7 merit
  * 流程 3 preset §3.6 单探针清洗 + §3.7 聚合 + §3.8 判级

机制（§1.1-1.3）：模型只输出"两位位置码紧跟答案"的串；清洗器解析
`(\d{2})([01])` → {pos:bit}，经 POS_MAP 还原成 claim+极性，再按 claim 判 F⊕R、
比对确定性陷阱真值。三套差异仅在 POS_MAP / 硬软对划分 / 陷阱真值源 / 可区分信号。

这些函数接受普通 dict（asset / probe / preset），便于离线纯规则单测（见
tests/test_qa_clean.py），运行时 runner 把 DB 行喂进来即可。
"""
from __future__ import annotations

import re
from statistics import mean, pstdev
from typing import Dict, List, Optional

from . import config as C

_PAIR_RE = re.compile(r"(\d{2})([01])")


# --------------------------------------------------------------------------- #
# 共享: 解析门 + POS_MAP 还原
# --------------------------------------------------------------------------- #
def parse_codes(raw: str, expected_pos: set) -> tuple:
    """位置码解析门。返回 (bit_dict, ok, reason)；reason∈{empty,dup,unknown,missing,None}。

    R2 修订(F2, parsing): 模型常把某位置码**重复输出同一比特**(良性重复, 如 `091 091`)。
    只在同码出现**冲突比特**时才判 parse:dup; 相同比特折叠, 救回诚实样本(round1 占 11%)。
    """
    pairs = _PAIR_RE.findall((raw or "").strip())
    if not pairs:
        return {}, False, "empty"
    bit: dict = {}
    for p, b in pairs:
        b = int(b)
        if p in bit and bit[p] != b:        # 同码不同比特 = 真冲突
            return bit, False, "dup"
        bit[p] = b
    if set(bit) - expected_pos:
        return bit, False, "unknown"
    if expected_pos - set(bit):
        return bit, False, "missing"
    return bit, True, None


def restore_by_claim(bit: Dict[str, int], pos_map: dict) -> Dict[str, object]:
    """还原成 {claim: {F:bit,R:bit}} 与 {trapclaim: bit}（IMQ 用，按 base_claim 分组）。"""
    A: Dict[str, object] = {}
    for p, b in bit.items():
        claim, pol, _aid = pos_map[p]
        if pol in ("trap", "anchor", "anchor_neg", "honesty_pos", "honesty_neg"):
            A[claim] = b
        else:
            A.setdefault(claim, {})[pol] = b
    return A


def restore_by_audit(bit: Dict[str, int], pos_map: dict) -> Dict[str, int]:
    """还原成 {audit_id: bit}（AES / preset 用，按审计 ID 取）。"""
    return {pos_map[p][2]: b for p, b in bit.items()}


# --------------------------------------------------------------------------- #
# 流程 1: IMQ 清洗器（§1.6）
# --------------------------------------------------------------------------- #
# R3 修订(G3, 确认 by Review-40): CLEAN 由硬对降软对。round3 hard_contra 23 中 20 为 CLEAN both-0
# (模型 768px 下漏检淡水印/版权文字/拼贴线, CLEAN_F=0 但 CLEAN_R=0)——这是模型对叠加物的"欠检诚实",
# 非真不一致; CLEAN_F=0 仍由 verdict_imq 路由 review(invalid:CLEAN), 安全网不变。REAL 保留硬对(核心有效性锚)。
IMQ_HARD = ["REAL"]                                # F⊕R 硬对(仅 REAL); CLEAN/INTACT 已降 soft
IMQ_SOFT = ["CLEAN", "INTACT", "SHARP", "NOISE", "COMP", "UPSC", "EXPO", "OVERCOOK", "SUBJ"]
IMQ_DEFECTS = ["SHARP", "NOISE", "COMP", "UPSC", "EXPO", "OVERCOOK"]


def _imq_fail(out, reason, A=None):
    out["reliable"] = False
    out["reason"] = reason
    if reason.startswith("trap:"):
        out["trap_fail"] = reason.split(":", 1)[1]
    if A is not None:
        out["A"] = A
    out["defect_count"] = None
    return out


def clean_imq(raw: str, asset: dict, cfg=C) -> dict:
    """asset 需含: is_portrait_pool, width, height, max_face_frac, is_bw_img。"""
    is_portrait = bool(asset.get("is_portrait_pool"))
    pos_map = cfg.imq_pos_map(is_portrait)
    expected = set(pos_map)
    out = {"reliable": True, "reason": None, "contradiction_count": 0,
           "trap_fail": None, "trap_skipped": 0, "reduced_anchor": False,
           "landscape_soft_fail": None, "defect_count": None, "A": None, "raw": raw}

    bit, ok, reason = parse_codes(raw, expected)
    if not ok:
        return _imq_fail(out, f"parse:{reason}")

    A = restore_by_claim(bit, pos_map)
    out["A"] = A

    # 门2 锚点
    if A.get("ANCHOR") != 1:
        return _imq_fail(out, "anchor", A)

    # 门3 确定性陷阱
    w, h = asset.get("width"), asset.get("height")
    claims = {c for c, _, _ in pos_map.values()}
    live_traps = 0
    if "LANDSCAPE" in claims:
        if w is None or h is None:
            out["trap_skipped"] += 1
        elif w != h:
            truth = 1 if w > h else 0
            if cfg.HAS_EXIF_FIXED:                       # 仅 EXIF 旋正落地后入硬门(§1.5 P-1)
                live_traps += 1
                if A["LANDSCAPE"] != truth:
                    return _imq_fail(out, "trap:landscape", A)
            else:                                        # 否则软监控不入硬门
                out["trap_skipped"] += 1
                out["landscape_soft_fail"] = 1 if A["LANDSCAPE"] != truth else 0
        # w==h(正方形)无横竖真值 → 不判
    if is_portrait and "FACE_GT" in claims:
        mff = asset.get("max_face_frac")
        # R2 修订(F3, trap_misanchored): cv2 Haar 漏检率高(round1 trap:face 10 例中 8 例 mff=0
        # 实为侧脸/小脸/遮挡真脸)。只信 Haar 正检: mff>FACE_MIN 才强制 FACE_GT=1;
        # mff<=FACE_MIN(漏检 FN 高发, "无脸"真值不可靠)→ skip, 不硬剔诚实回答。
        if mff is not None and mff > cfg.IMQ_FACE_MIN:
            live_traps += 1
            if A["FACE_GT"] != 1:
                return _imq_fail(out, "trap:face", A)
        else:
            out["trap_skipped"] += 1
    if "COLOR" in claims and cfg.ENABLE_T_COLOR:
        isbw = asset.get("is_bw_img")
        if isbw is None:
            out["trap_skipped"] += 1
        else:
            live_traps += 1
            if A["COLOR"] != (0 if isbw else 1):
                return _imq_fail(out, "trap:color", A)
    if out["trap_skipped"] and live_traps == 0:
        out["reduced_anchor"] = True

    # 门4 正反矛盾
    # R2 修订(G1, 确认 by Review-40): 软对的 both-0 = 该维度"诚实中庸"(既不明显好也不明显坏),
    # 非逻辑矛盾; 真矛盾只应是 F=1∧R=1(同时声称又好又坏=未独立看图)。故软对仅计 F==1∧R==1。
    # 硬对(REAL/CLEAN, 严格互补)仍用全 XOR(F+R!=1): both-0/both-1 都是真不一致 → 路由 review。
    # acquiescence 仍被兜住: 全1→REAL/CLEAN both-1 硬矛盾; 全0→REAL/CLEAN both-0 硬矛盾 + ANCHOR=0。
    soft = list(IMQ_SOFT)
    if is_portrait:
        soft.append("FACE")
    hc = sum(1 for c in IMQ_HARD if A[c]["F"] + A[c]["R"] != 1)
    sc = sum(1 for c in soft if A[c]["F"] == 1 and A[c]["R"] == 1)
    out["contradiction_count"] = hc + sc
    if hc > 0:
        return _imq_fail(out, "contradiction_hard", A)
    if sc > cfg.IMQ_CONTRA_TOL:
        return _imq_fail(out, "contradiction_soft", A)

    # 门5 裁定
    out["defect_count"] = sum(A[c]["R"] for c in IMQ_DEFECTS)   # 0..6
    return out


def verdict_imq(A: dict, is_portrait: bool, max_face_frac, defect_count, cfg=C) -> tuple:
    """仅在 reliable=True 时调用（§1.7）。返回 (verdict, reason)。"""
    for claim in ("REAL", "CLEAN"):
        if A[claim]["F"] == 0:
            if cfg.KAPPA_PASS[claim]:
                return ("drop", f"invalid:{claim}")
            return ("review", f"invalid:{claim}:unkappa")
    if is_portrait and A.get("FACE", {}).get("F", 1) == 0:
        if cfg.KAPPA_PASS["FACE"] and max_face_frac is not None and max_face_frac > cfg.IMQ_FACE_MIN:
            return ("drop", "no_usable_face")
        return ("review", "face_uncertain")
    weak_subject = (A["SUBJ"]["F"] == 0)
    if defect_count >= cfg.DEF_DROP and cfg.DEFECT_HARD_DROP:
        return ("drop", "quality_defects")
    if defect_count >= cfg.DEF_REVIEW or weak_subject or defect_count >= cfg.DEF_DROP:
        return ("review", "quality_or_subject")
    return ("keep", None)


def reconcile_imq(verdict: str, reason, iqa: dict, cfg=C) -> tuple:
    """keep→与 NR-IQA 软尾取交集降级（§1.7）。IQA_IN_CLEAN=False 时直接放行(纯 LLM QA)。"""
    if not cfg.IQA_IN_CLEAN:
        return (verdict, reason)
    if verdict == "keep":
        musiq = iqa.get("musiq")
        niqe = iqa.get("niqe")
        nsig = iqa.get("noise_sigma")
        if (musiq is not None and musiq < cfg.MUSIQ_DROP_BELOW) \
           or (niqe is not None and niqe > cfg.NIQE_CONFLICT_ABOVE) \
           or (nsig is not None and nsig > cfg.NOISE_SIGMA_DROP_ABOVE):
            return ("review", "iqa_conflict")
    return (verdict, reason)


# --------------------------------------------------------------------------- #
# 流程 2: AES 清洗器（§2.6）
# --------------------------------------------------------------------------- #
AES_SOFT_PAIRS = [("K1", "K2"), ("K3", "K4"), ("L1", "L2"), ("L3", "L4"),
                  ("C1", "C2"), ("D1", "D2"), ("D3", "D4"), ("M1", "M2"),
                  ("N1", "N2")]
AES_MOMENT_PAIR = ("M3", "M4")


def _aes_fail(out, reason):
    out["reliable"] = False
    out["reason"] = reason
    if reason.startswith("trap:"):
        out["trap_fail"] = reason.split(":", 1)[1]
    out["merit_count"] = None
    out["merit_frac"] = None
    return out


def clean_aes(raw: str, asset: dict, cfg=C) -> dict:
    """asset 需含: max_face_frac, width, height, is_bw_img。软信号永不 drop。"""
    mff = asset.get("max_face_frac")
    has_face = (mff is not None and mff > cfg.AES_FACE_MIN)
    pos_map = cfg.aes_pos_map(has_face)
    # 动态应发集（COLOR/T3 仅 HAS_ISBW_IMG 时发；当前默认不在 pos_map 中）
    expected = set(pos_map)
    out = {"reliable": True, "reason": None, "contradiction_count": 0,
           "trap_fail": None, "soft_trap_fail": None,
           "merit_count": None, "merit_n": None, "merit_frac": None,
           "has_face": has_face, "ans": None, "raw": raw}

    bit, ok, reason = parse_codes(raw, expected)
    if not ok:
        return _aes_fail(out, f"parse:{reason}")
    ans = restore_by_audit(bit, pos_map)
    out["ans"] = ans

    # 门2 锚点
    if ans.get("T1") != 1:
        return _aes_fail(out, "anchor")
    # 门3 诚实性软陷阱（硬门，构造真值，抓全1/全0敷衍）
    if ans.get("H1") != 1 or ans.get("H2") != 0:
        return _aes_fail(out, "honesty")
    # 门4 确定性陷阱
    if has_face and ans.get("T4") != 1:
        return _aes_fail(out, "trap:T4")
    if cfg.HAS_ISBW_IMG and "T3" in ans:
        expect_t3 = 0 if asset.get("is_bw_img") else 1
        if ans["T3"] != expect_t3:
            return _aes_fail(out, "trap:T3")
    w, h = asset.get("width"), asset.get("height")
    if "T2" in ans:
        if cfg.HAS_EXIF_FIXED and w is not None and h is not None:
            if ans["T2"] != (1 if w > h else 0):
                return _aes_fail(out, "trap:T2")
        elif w is not None and h is not None:                # 软监控
            out["soft_trap_fail"] = 1 if ans["T2"] != (1 if w > h else 0) else 0

    # 门5 正反矛盾（软对）
    # R2 修订(G1, 确认 by Review-40, round1 全部 78 条违反皆 both-0=诚实中庸): 软对仅计
    # F==1∧R==1(同时声称又好又坏=未独立看图)为矛盾; both-0=该审美维度中庸(不命中 merit, 非矛盾)。
    # acquiescence 由 H_honesty 硬门(H1=1/H2=0)兜住, 不再靠软对 both-0/both-1 误判中庸。
    pairs = list(AES_SOFT_PAIRS)
    if has_face:
        pairs.append(AES_MOMENT_PAIR)
    viol = sum(1 for f, r in pairs if ans[f] == 1 and ans[r] == 1)
    out["contradiction_count"] = viol
    if viol > cfg.AES_CONTRA_TOL:
        return _aes_fail(out, "contradiction_soft")

    # 门6 裁定 + merit
    merit = sum(1 for f, r in pairs if ans[f] == 1 and ans[r] == 0)
    out["merit_count"] = merit
    out["merit_n"] = len(pairs)
    out["merit_frac"] = merit / len(pairs)
    out["aes_sort_key"] = out["merit_frac"]
    out["aes_keep_vote"] = 1 if out["merit_frac"] >= cfg.MERIT_KEEP_FRAC else 0
    return out


# --------------------------------------------------------------------------- #
# 流程 3: preset 清洗器（§3.6-3.8）
# --------------------------------------------------------------------------- #
def clean_preset_probe(raw: str, paired_metrics: dict, engine: str, cfg=C) -> dict:
    """单探针清洗。engine∈{'param'(LR/darktable),'lut'}。返回含 reliable/reason/d/trap_fail。"""
    pos_map = cfg.PRESET_POS_MAP
    expected = set(pos_map)
    out = {"reliable": True, "reason": None, "trap_fail": None, "d": None, "raw": raw}

    eng_lut = (engine == "lut")
    tau_noop_low = cfg.TAU_NOOP_LOW_LUT if eng_lut else cfg.TAU_NOOP_LOW
    tau_ssim = cfg.TAU_SSIM

    bit, ok, reason = parse_codes(raw, expected)
    if not ok:
        out["reliable"] = False
        out["reason"] = f"parse:{reason}"
        return out
    d = restore_by_audit(bit, pos_map)
    out["d"] = d

    # 门2 锚点（F⊕R 互斥）
    if d["T1"] != 1 or d["T1b"] != 0:
        out["reliable"] = False
        out["reason"] = "anchor"
        return out
    if d["T1"] == d["T1b"]:
        out["reliable"] = False
        out["reason"] = "contradiction_anchor"
        return out

    # 门3 确定性陷阱（硬剔除受 PRESET_T2_HARD/T3_HARD 开关；未标定前软标记）
    de = paired_metrics.get("delta_e2000_mean")
    ssim = paired_metrics.get("ssim")
    if de is not None:
        t2_truth = 1 if de < tau_noop_low else 0
        if d["T2"] != t2_truth:
            if cfg.PRESET_T2_HARD:
                out["reliable"] = False
                out["reason"] = "trap:T2"
                out["trap_fail"] = "T2"
                return out
            out["trap_fail"] = (out["trap_fail"] + ",T2") if out["trap_fail"] else "T2"
    if ssim is not None:
        t3_truth = 1 if ssim < tau_ssim else 0
        if d["T3"] != t3_truth:
            if cfg.PRESET_T3_HARD:
                out["reliable"] = False
                out["reason"] = "trap:T3"
                out["trap_fail"] = (out["trap_fail"] + ",T3") if out["trap_fail"] else "T3"
                return out
            out["trap_fail"] = (out["trap_fail"] + ",T3") if out["trap_fail"] else "T3"

    # 门4 正反矛盾
    # R2: PRO 只在 F=1∧R=1(同时"干净专业"且"有破坏")才是真矛盾; both-0(既非明显破坏、
    # 也未被判为可用专业成品)=诚实中庸(典型: 强风格化但不破坏的 look), 放行不计矛盾,
    # 由 aggregate 的 pro_pass(P1==1∧P2==0)自然计 0。沿用 image-pilot R2 软对模式。
    if d["P1"] == 1 and d["P2"] == 1:
        out["reliable"] = False
        out["reason"] = "contradiction_hard:PRO"
        return out
    if d["I1"] == d["I2"]:
        out["reliable"] = False
        out["reason"] = "contradiction_hard:INTENT"
        return out
    coh_viol = 1 if d["H1"] == d["H2"] else 0
    if coh_viol > cfg.PRESET_CONTRA_TOL:
        out["reliable"] = False
        out["reason"] = "contradiction_soft:COH"
        return out

    return out


def aggregate_preset(probes: List[dict], cfg=C) -> Optional[dict]:
    """probes: 单探针 clean 结果列表（含 reliable / d / paired_metrics）。§3.7。"""
    R = [p for p in probes if p.get("reliable")]
    if len(R) < cfg.MIN_RELIABLE_PROBES:
        return None
    pro = mean(1 if (p["d"]["P1"] == 1 and p["d"]["P2"] == 0) else 0 for p in R)
    intent = mean(1 if (p["d"]["I1"] == 1 and p["d"]["I2"] == 0) else 0 for p in R)
    coh = mean(1 if (p["d"]["H1"] == 1 and p["d"]["H2"] == 0) else 0 for p in R)
    emd = [p.get("paired_metrics", {}).get("hist_emd_ab") for p in R]
    emd = [e for e in emd if e is not None]
    disp = (pstdev(emd) / (mean(emd) + 1e-6)) if len(emd) >= 2 else 0.0
    return {
        "R": R, "pro_pass_rate": pro, "intent_pass_rate": intent, "coh_pass_rate": coh,
        "vote_PRO": 1 if pro >= cfg.VOTE_THRESH else 0,
        "vote_INTENT": 1 if intent >= cfg.VOTE_THRESH else 0,
        "vote_COH": 1 if coh >= cfg.COH_VOTE_THRESH else 0,
        "coherence_score": cfg.W_PRO * pro + cfg.W_COH * coh,
        "edit_direction_dispersion": disp,
        "reliable_probe_count": len(R),
    }


def map_verdict_preset(previews: List[dict], probes: List[dict], engine: str, cfg=C) -> dict:
    """previews: 每探针 paired_metrics 列表（order=0 用）。probes: clean 结果。§3.8。
    返回 set_fields dict（pass_c 恒 0/1，绝不 None）。"""
    eng_lut = (engine == "lut")
    tau_noop = cfg.TAU_NOOP_MEAN_LUT if eng_lut else cfg.TAU_NOOP_MEAN
    des = [pm.get("delta_e2000_mean") for pm in previews if pm.get("delta_e2000_mean") is not None]
    mean_de = mean(des) if des else 0.0
    all_noop = bool(previews) and all(pm.get("noop_score") == 1 for pm in previews)
    if all_noop or mean_de < tau_noop:
        return {"pass_c": 0, "auto_verdict": "drop", "verdict_reason": "near_noop",
                "near_noop": 1}

    agg = aggregate_preset(probes, cfg)
    if agg is None:
        return {"pass_c": 0, "auto_verdict": "review",
                "verdict_reason": "insufficient_reliable_probes", "near_noop": 0,
                "reliable_probe_count": sum(1 for p in probes if p.get("reliable"))}
    base = {"near_noop": 0, "pro_pass_rate": agg["pro_pass_rate"],
            "intent_pass_rate": agg["intent_pass_rate"], "coh_pass_rate": agg["coh_pass_rate"],
            "coherence_score": agg["coherence_score"], "vote_pro": agg["vote_PRO"],
            "vote_intent": agg["vote_INTENT"], "vote_coh": agg["vote_COH"],
            "edit_direction_dispersion": agg["edit_direction_dispersion"],
            "reliable_probe_count": agg["reliable_probe_count"]}
    if not agg["vote_PRO"]:
        v = "drop" if cfg.PRO_DROP_ENABLED else "review"
        r = "not_professional" if cfg.PRO_DROP_ENABLED else "not_professional:pending_kappa"
        return {**base, "pass_c": 0, "auto_verdict": v, "verdict_reason": r}
    if not agg["vote_INTENT"]:
        v = "drop" if cfg.INTENT_DROP_ENABLED else "review"
        r = "no_intent" if cfg.INTENT_DROP_ENABLED else "no_intent:pending_kappa"
        return {**base, "pass_c": 0, "auto_verdict": v, "verdict_reason": r}
    if not agg["vote_COH"]:
        return {**base, "pass_c": 0, "auto_verdict": "review", "verdict_reason": "incoherent_look"}
    return {**base, "pass_c": 1, "auto_verdict": "review", "verdict_reason": "all_pass"}


# --------------------------------------------------------------------------- #
# preset 功能打 tag (确定性 LAB 变换 + QA 结果; 2026-06-22 用户指令)
# --------------------------------------------------------------------------- #
def _hue_deg(a, b):
    import math
    return math.degrees(math.atan2(b, a)) % 360.0


def tag_preset_function(sigs: List[dict], qa: Optional[dict] = None, cfg=C) -> dict:
    """据 6 主色探针(4 色相+肤色+中性)的 before/after LAB 签名 + QA, 客观判定 preset 功能。

    sigs[i]: {name, dL, da, db, dC, after_C, before_C, a_before/b_before/a_after/b_after,
              contrast_ratio, shadow_dL}
    色温/色罩在【中性探针】读(避开极饱和色探针的 chroma 压缩失真); 饱和用色探针【相对】
    chroma 变化; 对比/影调用全探针; teal-orange 用蓝探针色相旋转。
    返回 4 轴 tag + tags 列表 + 聚合 metrics。
    """
    t = cfg.PRESET_TAG_THRESH
    if not sigs:
        return {"tags": ["no_signal"], "grade_family": "unknown"}
    by = {s["name"]: s for s in sigs}
    hue_sigs = [by[n] for n in ("red", "yellow", "green", "blue") if n in by] or sigs

    # 轴1 色温/色罩: 中性探针(无极端 chroma)→ 退 skin → 退全局均值
    cast = by.get("neutral") or by.get("skin")
    db_cast = cast["db"] if cast else mean(s["db"] for s in sigs)
    da_cast = cast["da"] if cast else mean(s["da"] for s in sigs)
    temperature = "warm" if db_cast > t["warm_db"] else "cool" if db_cast < -t["warm_db"] else "neutral"
    tint = "magenta" if da_cast > t["tint_da"] else "green" if da_cast < -t["tint_da"] else "neutral"

    # 轴2 饱和: 色探针相对 chroma 变化
    relC = mean((s["after_C"] - s["before_C"]) / (s["before_C"] + 1e-6) for s in hue_sigs)
    afterC_hue = mean(s["after_C"] for s in hue_sigs)
    is_bw = afterC_hue < t["bw_after_C"]
    saturation = ("bw" if is_bw else "vibrant" if relC > t["sat_rel"]
                  else "muted" if relC < -t["sat_rel"] else "neutral")

    # 轴3 对比/影调/曝光 (全探针)
    contrast_m = mean(s["contrast_ratio"] for s in sigs)
    shadow_dL_m = mean(s["shadow_dL"] for s in sigs)
    dL_m = mean(s["dL"] for s in sigs)
    contrast = ("punchy" if contrast_m > t["contrast_ratio_hi"]
                else "flat" if contrast_m < t["contrast_ratio_lo"] else "neutral")
    tone = ("lifted" if shadow_dL_m > t["lift_dLmin_blacks"]
            else "crushed" if shadow_dL_m < t["crush_dLmin_blacks"] else "neutral")
    exposure = ("high_key" if dL_m > t["lift_dLmin_blacks"]
                else "low_key" if dL_m < t["crush_dLmin_blacks"] else "neutral")

    # teal-orange: 蓝探针色相向青/teal 旋转 + 暖探针(red/skin) chroma 保留(不被压成灰)
    def _dhue(s):
        h0 = _hue_deg(s["a_before"], s["b_before"]); h1 = _hue_deg(s["a_after"], s["b_after"])
        return (h1 - h0 + 180) % 360 - 180
    # 蓝(LAB hue≈290°)→青/teal(≈195°)是色相【减小】, 故 teal_rot 应为负
    blue = by.get("blue"); warm = by.get("red") or by.get("skin")
    teal_rot = _dhue(blue) if (blue and "a_before" in blue) else 0.0
    teal_orange = bool(blue and warm and "a_before" in blue
                       and teal_rot < -t["teal_hue_rot"] and warm["after_C"] > t["warm_keep_C"])

    # 轴4 调色家族 (优先级)
    if is_bw:
        family = "bw"
    elif teal_orange:
        family = "teal_orange"
    elif tone == "lifted" and contrast == "flat" and relC < 0:
        family = "vintage_film"          # 褪色胶片: 提暗部 + 降对比 + 去饱和
    elif (temperature == "neutral" and tint == "neutral" and saturation == "neutral"
          and contrast == "neutral" and tone == "neutral"):
        family = "clean_natural"
    else:
        family = "stylized"

    tags = [f"temp:{temperature}", f"tint:{tint}", f"sat:{saturation}",
            f"contrast:{contrast}", f"tone:{tone}", f"exposure:{exposure}", f"family:{family}"]
    tags = [x for x in tags if not x.endswith(":neutral")]
    return {
        "temperature": temperature, "tint": tint, "saturation": saturation,
        "contrast": contrast, "tone": tone, "exposure": exposure,
        "grade_family": family, "tags": tags,
        "qa_pass_c": (qa or {}).get("pass_c"), "qa_verdict": (qa or {}).get("verdict_reason"),
        "metrics": {"db_cast": round(db_cast, 2), "da_cast": round(da_cast, 2),
                    "rel_chroma": round(relC, 3), "after_C_hue": round(afterC_hue, 2),
                    "contrast_ratio": round(contrast_m, 3), "shadow_dL": round(shadow_dL_m, 2),
                    "dL_mean": round(dL_m, 2), "teal_rot": round(teal_rot, 1)},
    }
