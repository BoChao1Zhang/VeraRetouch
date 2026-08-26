#!/usr/bin/env python3
"""EPR-050 v3 summary tables from pairs.jsonl.  Numbers only, no prose.

Aggregation convention (stated verbatim in RESULT.md):
  Every per-sample number in pairs.jsonl is already a *pixel-level* statistic over
  that sample's pixels.  Cross-sample aggregation therefore reports

    mean   = pixel-count-weighted mean of the per-sample means  (exact pooled mean)
    p50*   = median over units of the per-sample p50            (NOT a pooled pixel quantile)
    p95*   = median over units of the per-sample p95            (idem)
    max    = max over units of the per-sample max               (exact)
    n_u    = number of (sample[, step]) units;  px = total pixels

  A trailing * marks a median-of-per-sample-quantile, which is not the same object
  as the quantile of the pooled pixel population and is never mixed with one.

Tables
  T1  rec_band x alpha-band x gamut_cover tertile   (final restore-E error)
  T2  geom type (== uses_subject) x unwind depth    (depth_err_all)
  T3  per-step saturation-fraction increase over `before`
  T4  per-step convergence failure, split saturated / interior
  plus: guard counters, mask/geom/major/pool census, path-N control column.
"""
from __future__ import annotations

import argparse
import json
import statistics as st
from pathlib import Path

DE00_TOL_DOC = 0.05
BANDS = ["a0", "mid", "a1"]
BAND_LABEL = {"a0": "alpha=0", "mid": "0<alpha<1", "a1": "alpha=1"}
# Chain order is a config value ([run] step_order, v3.3).  Rebound in main() from
# the run manifest; this literal is only the pre-v3.3 order used for old journals
# whose manifest predates the key.
STEP_KIND = ["lum_high", "lum_mid", "lum_shadow", "global", "geom"]
# The subject-shaped geometry family: "subject" is the v3.1-v3.4 name, "semantic"
# the v3.5+ main-chain one.  Both are the subject's own outline, which is what the
# `uses_subject` column and the coverage/span exemption are about.
SUBJECT_SHAPED = ("subject", "semantic")


def agg(units: list[dict]) -> dict:
    """units: list of band_stats dicts (may contain n==0 entries)."""
    u = [d for d in units if d and d.get("n", 0) > 0]
    if not u:
        return dict(n_u=0, px=0)
    px = sum(d["n"] for d in u)
    return dict(
        n_u=len(u), px=px,
        mean=sum(d["mean"] * d["n"] for d in u) / px,
        p50=st.median(d["p50"] for d in u),
        p95=st.median(d["p95"] for d in u),
        max=max(d["max"] for d in u),
    )


def fmt(a: dict) -> str:
    if a["n_u"] == 0:
        return "| 0 | 0 | - | - | - | - "
    return (f"| {a['n_u']} | {a['px']:,} | {a['mean']:.4g} | {a['p50']:.4g} "
            f"| {a['p95']:.4g} | {a['max']:.4g} ")


def qbins(vals: list[float], k: int = 3) -> list[float]:
    s = sorted(vals)
    return [s[round(i * len(s) / k)] for i in range(1, k)]


def binof(v: float, cuts: list[float]) -> int:
    return sum(v >= c for c in cuts)


def num(vals: list[float]) -> dict:
    if not vals:
        return {}
    s = sorted(vals)
    return dict(n=len(s), mean=sum(s) / len(s), p50=st.median(s),
                p95=s[min(len(s) - 1, round(0.95 * (len(s) - 1)))], max=s[-1], min=s[0])


def nfmt(d: dict) -> str:
    if not d:
        return "| 0 | - | - | - | - | - "
    return (f"| {d['n']} | {d['mean']:+.5f} | {d['p50']:+.5f} | {d['p95']:+.5f} "
            f"| {d['max']:+.5f} | {d['min']:+.5f} ")


def main():
    global STEP_KIND
    ap = argparse.ArgumentParser()
    ap.add_argument("--pairs",
                    default="/home/bc/data/builds/epr050-degrade-20260825/v3/pairs.jsonl")
    ap.add_argument("--out-json", default=None)
    ap.add_argument("--skipped", default=None,
                    help="skipped.json: sources excluded by the mask band criterion")
    a = ap.parse_args()
    rows = [json.loads(l) for l in open(a.pairs)]
    # thresholds quoted in the tables come from the frozen manifest, never from a
    # literal in this file (v3.1 numbers would otherwise be printed under v3.2)
    manp = Path(a.pairs).parent / "run_args.json"
    CFG = json.loads(manp.read_text()).get("config", {}) if manp.exists() else {}
    MK = CFG.get("masks", {})
    RUN = CFG.get("run", {})
    # v3.3: chain order from the manifest; fall back to the recorded per-step
    # `kind` values, and only then to the pre-v3.3 literal.
    if RUN.get("step_order"):
        STEP_KIND = list(RUN["step_order"])
    elif rows and all("kind" in s for s in rows[0]["steps"]):
        STEP_KIND = [s["kind"] for s in rows[0]["steps"]]
    NST = len(STEP_KIND)
    GEOM_STEP = STEP_KIND.index("geom")
    GEOM_KEY = f"step{GEOM_STEP + 1}_geom"
    sk = []
    skp = Path(a.skipped) if a.skipped else Path(a.pairs).parent / "skipped.json"
    if skp.exists():
        sk = json.loads(skp.read_text())
    n = len(rows)
    O: list[str] = []
    P = O.append
    J: dict = {}

    # ---------------- census -------------------------------------------------
    pools = {}
    geoms = {}
    majors = {}
    recb = {}
    for r in rows:
        pools[r["pool"]] = pools.get(r["pool"], 0) + 1
        geoms[r["mask"]["geom"]] = geoms.get(r["mask"]["geom"], 0) + 1
        majors[r["major"]] = majors.get(r["major"], 0) + 1
        recb[r["rec_band"]] = recb.get(r["rec_band"], 0) + 1
    sizes = [r["size"] for r in rows]
    J["census"] = dict(n=n, pool=pools, geom=geoms, rec_band=recb, major=majors,
                       max_side=max(max(s) for s in sizes),
                       px_total=sum(s[0] * s[1] for s in sizes),
                       uses_subject=sum(r["mask"]["uses_subject"] for r in rows),
                       attempts=num([r["mask"]["attempts"] for r in rows]),
                       split_rule={k: sum(r["split_rule"] == k for r in rows)
                                   for k in {r["split_rule"] for r in rows}},
                       n_naive=sum(r["err_N"] is not None for r in rows))
    P("## 0. 池构成 / 口径\n")
    P(f"- n = {n}；pool {pools}；rec_band {recb}")
    P(f"- rec_band = 该样本 {NST} 个 LUT 的 `recovered` 最小值落档；"
      f"主池门槛 recovered_min = {CFG.get('pool', {}).get('recovered_min')}、"
      f"clip_max = {CFG.get('pool', {}).get('clip_max')}")
    P("")
    P("| pool \\ rec_band | " + " | ".join(sorted(recb)) + " | 合计 |")
    P("|---" * (len(recb) + 2) + "|")
    for p in sorted(pools):
        cells = [sum(1 for r in rows if r["pool"] == p and r["rec_band"] == rb)
                 for rb in sorted(recb)]
        P(f"| {p} | " + " | ".join(str(c) for c in cells) + f" | {sum(cells)} |")
    J["census"]["pool_x_recband"] = {
        p: {rb: sum(1 for r in rows if r["pool"] == p and r["rec_band"] == rb)
            for rb in sorted(recb)} for p in sorted(pools)}
    P("")
    P(f"- geom {geoms}；uses_subject = {J['census']['uses_subject']}"
      f"（uses_subject 恒等于 geom==subject）")
    P(f"- major {majors}")
    P(f"- split_rule {J['census']['split_rule']}")
    P(f"- 分辨率上限 max_side = {J['census']['max_side']}；总像素 "
      f"{J['census']['px_total']:,}")
    P(f"- mask.attempts: mean {J['census']['attempts']['mean']:.3f} / max "
      f"{J['census']['attempts']['max']:.0f}")
    P(f"- 路径 N 对照列样本数 = {J['census']['n_naive']}")
    J["step_order"] = list(STEP_KIND)
    P(f"- 退化步序 step_order = {STEP_KIND}（恢复方向为其逆序）；"
      f"几何步在第 {GEOM_STEP + 1} 位")
    iters = [s["inv_iters"] for r in rows for s in r["steps"]]
    hist = {}
    for v in iters:
        hist[v] = hist.get(v, 0) + 1
    J["inv_iters"] = dict(hist={str(k): v for k, v in sorted(hist.items())}, **num(iters))
    P(f"- inv_iters（{n}x{NST} 步）: mean {J['inv_iters']['mean']:.2f} / p50 "
      f"{J['inv_iters']['p50']:.0f} / p95 {J['inv_iters']['p95']:.0f} / max "
      f"{J['inv_iters']['max']:.0f}；打满 200 的步数 = {hist.get(200, 0)}")
    P(f"- inv_iters 直方图: {json.dumps(J['inv_iters']['hist'])}\n")

    # ---------------- T1  rec_band x alpha x gamut ---------------------------
    gc = {r["id"]: sum(s["gamut_cover"] for s in r["steps"]) / NST for r in rows}
    cuts = qbins(list(gc.values()), 3)
    glabel = [f"gc<{cuts[0]:.4f}", f"{cuts[0]:.4f}<=gc<{cuts[1]:.4f}",
              f"gc>={cuts[1]:.4f}"]
    P("## 1. 主表 rec_band x alpha 档 x gamut_cover（恢复误差 E，8bit 级，"
      "per-pixel RGB L-inf）\n")
    P(f"gamut_cover = 每样本 {NST} 步 gamut_cover 均值；三分位切点 "
      f"{cuts[0]:.6f} / {cuts[1]:.6f}\n")
    P("单元 = (样本 x 步)，即每步的 alpha 场对**最终**恢复误差的分层"
      "（`err_E_by_step_alpha`）。\n")
    P("| rec_band | gamut_cover | alpha 档 | n_u | px | mean | p50* | p95* | max |")
    P("|---|---|---|---|---|---|---|---|---|")
    t1 = {}
    for rb in sorted(recb):
        for gi, gl in enumerate(glabel):
            sub = [r for r in rows if r["rec_band"] == rb and binof(gc[r["id"]], cuts) == gi]
            for b in BANDS:
                u = [r["err_E_by_step_alpha"][f"step{k+1}_{STEP_KIND[k]}"][b]
                     for r in sub for k in range(NST)]
                A = agg(u)
                t1[f"{rb}|{gl}|{b}"] = A
                P(f"| {rb} | {gl} | {BAND_LABEL[b]} " + fmt(A) + "|")
    J["T1"] = t1
    P("")
    # T1b: rec_band 取**该步自己那支 LUT** 的 recovered，而不是样本 chain_len 支的最小值。
    # 样本级 rec_band 是 min-over-5，[0.99,1] 档在 n=100 上可能为空；步级口径把
    # 同一批像素按每步的 LUT 良定度重新分层，两档并排。
    def sband(v: float) -> str:
        # v3.4: three main-pool bands (recovered_min dropped to 0.85);
        # "0.50-0.85" only names the gap that no pool covers.
        if v >= 0.99:
            return "ge0.99"
        if v >= 0.95:
            return "0.95-0.99"
        if v >= 0.85:
            return "0.85-0.95"
        return "lt0.50" if v < 0.5 else "0.50-0.85"
    sbands = sorted({sband(s["recovered"]) for r in rows for s in r["steps"]})
    P("### 1b. 步级 rec_band（该步 LUT 自身的 recovered）x alpha 档\n")
    P(f"样本级 rec_band = 该样本 {NST} 支 LUT 的 min；步级 rec_band = 该步那一支的 recovered。"
      "同一批像素，两种分层口径。\n")
    P("| pool | 步级 rec_band | LUT 步数 | alpha 档 | n_u | px | mean | p50* | p95* | max |")
    P("|---|---|---|---|---|---|---|---|---|---|")
    t1b = {}
    for p in ["main", "ctrl", "all"]:
        sub = [r for r in rows if p == "all" or r["pool"] == p]
        for sb in sbands:
            units = [(r, k) for r in sub for k in range(NST)
                     if sband(r["steps"][k]["recovered"]) == sb]
            if not units:
                continue
            for b in BANDS:
                A = agg([r["err_E_by_step_alpha"][f"step{k+1}_{STEP_KIND[k]}"][b]
                         for r, k in units])
                t1b[f"{p}|{sb}|{b}"] = A
                P(f"| {p} | {sb} | {len(units)} | {BAND_LABEL[b]} " + fmt(A) + "|")
    J["T1b"] = t1b
    P("")
    P(f"步级 recovered 分布（每样本 {NST} 支 LUT，共 n x {NST} 支）：\n")
    P("| pool | 步级 rec_band | LUT 步数 | recovered mean | p50 | min | max |")
    P("|---|---|---|---|---|---|---|")
    for p in ["main", "ctrl", "all"]:
        sub = [r for r in rows if p == "all" or r["pool"] == p]
        for sb in sbands:
            v = [s["recovered"] for r in sub for s in r["steps"]
                 if sband(s["recovered"]) == sb]
            d = num(v)
            if not d:
                continue
            J.setdefault("T1b_rec", {})[f"{p}|{sb}"] = d
            P(f"| {p} | {sb} | {d['n']} | {d['mean']:.6f} | {d['p50']:.6f} "
              f"| {d['min']:.6f} | {d['max']:.6f} |")
    allmin = {p: sum(1 for r in rows if (p == "all" or r["pool"] == p)
                     and all(s["recovered"] >= 0.99 for s in r["steps"]))
              for p in ["main", "ctrl", "all"]}
    J["rows_all5_ge099"] = allmin
    P("")
    P(f"- {NST} 支 LUT 全部 >= 0.99 的样本数：main {allmin['main']} / ctrl "
      f"{allmin['ctrl']} / all {allmin['all']}（样本级 rec_band = ge0.99 的充要条件）\n")
    P("整体（不分层，final error 全像素 `err_E_all`）：\n")
    P("| pool | n_u | px | mean | p50* | p95* | max |")
    P("|---|---|---|---|---|---|---|")
    for p in ["main", "ctrl"]:
        A = agg([r["err_E_all"] for r in rows if r["pool"] == p])
        J.setdefault("overall", {})[p] = A
        P(f"| {p} " + fmt(A) + "|")
    A = agg([r["err_E_all"] for r in rows])
    J["overall"]["all"] = A
    P("| all " + fmt(A) + "|")
    P("")
    nn = [r for r in rows if r["err_N"]]
    if nn:
        P(f"路径 N（朴素）对照列，前 {len(nn)} 条，全像素：\n")
        P("| path | n_u | px | mean | p50* | p95* | max |")
        P("|---|---|---|---|---|---|---|")
        AN = agg([r["err_N"]["all"] for r in nn])
        AE = agg([r["err_E_all"] for r in nn])
        J["path_N"] = dict(N=AN, E_same_subset=AE)
        P("| N " + fmt(AN) + "|")
        P("| E (同 " + f"{len(nn)}" + " 条) " + fmt(AE) + "|")
        P("")
        P(f"| path | alpha 档({GEOM_KEY}) | n_u | px | mean | p50* | p95* | max |")
        P("|---|---|---|---|---|---|---|---|")
        for b in BANDS:
            P(f"| N | {BAND_LABEL[b]} " + fmt(agg([r["err_N"][b] for r in nn])) + "|")
        P("")

    # ---------------- T2  geom x depth --------------------------------------
    P("## 2. 几何类型 x 是否主体掩膜 x 步深（depth_err_all，8bit 级）\n")
    P("步深 = 已反解步数 d：`x_hat_{5-d}` 对 `y_{5-d}` 的误差；d=5 即最终恢复误差。\n")
    P("| geom | uses_subject | 已反解步数 d | n_u | px | mean | p50* | p95* | max |")
    P("|---|---|---|---|---|---|---|---|---|")
    t2 = {}
    for g in sorted(geoms):
        # v3.5 renamed the subject-shaped family "subject" -> "semantic" (it is
        # the main chain's `_semantic_alpha`).  Both names mean "this mask is the
        # subject's own shape", which is what the column reports and what the
        # coverage/span exemption keys off in the build tool
        # (`uses_subject = kind in ("subject", "semantic")`).
        us = (g in SUBJECT_SHAPED)
        sub = [r for r in rows if r["mask"]["geom"] == g]
        for d in range(1, NST + 1):
            k = NST - d                      # steps[k].depth_err_all
            A = agg([r["steps"][k]["depth_err_all"] for r in sub])
            t2[f"{g}|d{d}"] = A
            P(f"| {g} | {us} | {d} " + fmt(A) + "|")
    J["T2"] = t2
    P("")
    P(f"同分层的最终误差按 alpha 档（几何步 {GEOM_KEY} 的 alpha）：\n")
    P("| geom | alpha 档 | n_u | px | mean | p50* | p95* | max |")
    P("|---|---|---|---|---|---|---|---|")
    for g in sorted(geoms):
        sub = [r for r in rows if r["mask"]["geom"] == g]
        for b in BANDS:
            A = agg([r["err_E_by_step_alpha"][GEOM_KEY][b] for r in sub])
            J.setdefault("T2_alpha", {})[f"{g}|{b}"] = A
            P(f"| {g} | {BAND_LABEL[b]} " + fmt(A) + "|")
    P("")

    # ---------------- T3  saturation ----------------------------------------
    P("## 3. 每步饱和率增量（sat_delta = 该步后饱和率 - before 饱和率）\n")
    P(f"饱和 = 任一通道 >= {254}/255 或 <= 1/255；warning 阈 +0.005。\n")
    P("| pool | step | kind | n | mean | p50 | p95 | max | min | sat_warn 数 |")
    P("|---|---|---|---|---|---|---|---|---|---|")
    t3 = {}
    for p in ["main", "ctrl", "all"]:
        sub = rows if p == "all" else [r for r in rows if r["pool"] == p]
        if not sub:                       # an empty pool (small smoke) has no row
            continue
        for k in range(NST):
            v = [r["steps"][k]["sat_delta"] for r in sub]
            w = sum(r["steps"][k]["sat_warn"] for r in sub)
            D = num(v)
            t3[f"{p}|{k+1}"] = dict(D, warn=w)
            P(f"| {p} | {k+1} | {STEP_KIND[k]} " + nfmt(D) + f"| {w} |")
    J["T3"] = t3
    bs = num([r["base_sat_frac"] for r in rows])
    J["base_sat"] = bs
    P("")
    P(f"before 饱和率 base_sat_frac: mean {bs['mean']:.5f} / p50 {bs['p50']:.5f} "
      f"/ p95 {bs['p95']:.5f} / max {bs['max']:.5f}")
    warn_rows = [r for r in rows if r["sat_warn_steps"]]
    J["sat_warn"] = dict(rows=len(warn_rows),
                         events=sum(len(r["sat_warn_steps"]) for r in rows),
                         by_pool={p: sum(len(r["sat_warn_steps"]) for r in rows
                                         if r["pool"] == p) for p in pools},
                         ids={r["id"]: r["sat_warn_steps"] for r in warn_rows})
    P(f"- sat_warn 触发样本数 {len(warn_rows)} / {n}；事件数 "
      f"{J['sat_warn']['events']}（按 pool {J['sat_warn']['by_pool']}）\n")

    # v3.4: clip_max was raised from 0 to 0.005, so the static screen no longer
    # guarantees "no grid node on the [0,1] rail".  The dynamic guard is what
    # takes over, and its trigger counts are therefore reported per rec_band and
    # per clip value of the LUT used at that step.
    P("### 3b. sat_warn 事件按步级 rec_band x 该步 LUT 的 clip 分层（v3.4）\n")
    P("单元 = (样本 x 步)。`clip` 是该步那一支 LUT 的网格节点被钉在 0/1 的比例；"
      "`clip == 0` 是 v3.3 及以前就能入池的那批，`0 < clip <= clip_max` 是 v3.4 新入场的。\n")
    P("| 步级 rec_band | clip 档 | LUT 步数 | sat_warn 事件 | 事件率 | sat_delta mean "
      "| sat_delta p95 | sat_delta max |")
    P("|---|---|---|---|---|---|---|---|")
    t3b = {}
    units = [(r, s) for r in rows for s in r["steps"]]
    cbands = [("clip == 0", lambda c: c == 0),
              ("0 < clip <= 0.005", lambda c: 0 < c <= 0.005),
              ("clip > 0.005", lambda c: c > 0.005)]
    sbs = sorted({sband(s["recovered"]) for _, s in units})
    for sb in sbs + ["all"]:
        for cl, fn in cbands + [("all", lambda c: True)]:
            u = [s for _, s in units
                 if (sb == "all" or sband(s["recovered"]) == sb) and fn(s["clip"])]
            if not u:
                continue
            w = sum(s["sat_warn"] for s in u)
            d = num([s["sat_delta"] for s in u])
            t3b[f"{sb}|{cl}"] = dict(n=len(u), warn=w, rate=w / len(u), sat=d)
            P(f"| {sb} | {cl} | {len(u)} | {w} | {w/len(u):.4f} | {d['mean']:+.5f} "
              f"| {d['p95']:+.5f} | {d['max']:+.5f} |")
    J["T3b"] = t3b
    cl_census = {lab: sum(1 for _, s in units if fn(s["clip"]))
                 for lab, fn in cbands}
    J["clip_census"] = cl_census
    P(f"\n- 全部 {len(units)} 个 LUT 步按 clip 档计数：{json.dumps(cl_census, ensure_ascii=False)}")
    P(f"- 池门槛（config）：recovered_min = {CFG.get('pool', {}).get('recovered_min')}、"
      f"clip_max = {CFG.get('pool', {}).get('clip_max')}\n")

    # ---------------- T4  convergence failure -------------------------------
    P("## 4. 求逆收敛失败率拆分（残差 > 1 个 8bit 级的像素占比）\n")
    P("sat = 解落在 [0,1]^3 边界（域内无原像）；interior = 内点，解算器放弃。\n")
    P("| pool | step | kind | n | fail mean | fail p50 | fail p95 | fail max "
      "| sat mean | interior mean | interior max | 有失败样本数 |")
    P("|---|---|---|---|---|---|---|---|---|---|---|---|")
    t4 = {}
    for p in ["main", "ctrl", "all"]:
        sub = rows if p == "all" else [r for r in rows if r["pool"] == p]
        if not sub:
            continue
        for k in range(NST):
            f = [r["steps"][k]["conv_fail_frac"] for r in sub]
            s = [r["steps"][k]["conv_fail_sat_frac"] for r in sub]
            i = [r["steps"][k]["conv_fail_interior_frac"] for r in sub]
            F, S, I = num(f), num(s), num(i)
            nz = sum(x > 0 for x in f)
            t4[f"{p}|{k+1}"] = dict(fail=F, sat=S, interior=I, nonzero=nz)
            P(f"| {p} | {k+1} | {STEP_KIND[k]} | {F['n']} | {F['mean']:.3e} "
              f"| {F['p50']:.3e} | {F['p95']:.3e} | {F['max']:.3e} | {S['mean']:.3e} "
              f"| {I['mean']:.3e} | {I['max']:.3e} | {nz} |")
    J["T4"] = t4
    P("")
    ch = num([r["conv_fail_frac_chain"] for r in rows])
    J["conv_fail_chain"] = dict(ch, nonzero=sum(r["conv_fail_frac_chain"] > 0
                                                for r in rows))
    P(f"整链 conv_fail_frac_chain（{NST} 步取 max）: mean {ch['mean']:.3e} / p50 "
      f"{ch['p50']:.3e} / p95 {ch['p95']:.3e} / max {ch['max']:.3e}；"
      f">0 的样本数 {J['conv_fail_chain']['nonzero']} / {n}")
    resid = num([s["resid_p95"] for r in rows for s in r["steps"]])
    J["resid_p95"] = resid
    P(f"每步残差 resid_p95（8bit 级，{n}x{NST}）: mean {resid['mean']:.3e} / p50 "
      f"{resid['p50']:.3e} / p95 {resid['p95']:.3e} / max {resid['max']:.3e}\n")

    # ---------------- guards -------------------------------------------------
    a0px = [r["composite_alpha0_px"] for r in rows]
    J["guards"] = dict(
        stepwise_bitexact_checks=n * NST * 2,
        stepwise_bitexact_raised=0,
        composite_alpha0_px_total=sum(a0px),
        composite_alpha0_px_max=max(a0px),
        rows_with_composite_alpha0=sum(x > 0 for x in a0px),
        mask_reject_max_attempts=int(max(r["mask"]["attempts"] for r in rows)),
        sat_warn_events=J["sat_warn"]["events"],
    )
    P("## 5. 守卫计数\n")
    P("| 守卫 | 次数 | 触发 raise |")
    P("|---|---|---|")
    P(f"| 每步 alpha==0 正向逐比特 | {n*NST} | 0 |")
    P(f"| 每步 alpha==0 反向逐比特 | {n*NST} | 0 |")
    P(f"| 复合 alpha==0 逐比特 | {n} | 0 |")
    P(f"| 蒙版各档 >=1% 拒绝采样（最多 40 次） | {n + len(sk)} | {len(sk)}（计数排除，顺位替补） |")
    P(f"| 自检（恒等 LUT / mix_alpha 吸附 / gamma 逆） | 1 | 0 |")
    P(f"| 续跑身份一致 | 1 | 0 |")
    P("")
    P(f"- composite_alpha0_px: 总计 {sum(a0px)}，max {max(a0px)}，>0 的样本数 "
      f"{sum(x > 0 for x in a0px)} / {n}")
    P(f"- mask.attempts max = {J['guards']['mask_reject_max_attempts']}（上限 40）")
    P(f"- sat_warn 事件 {J['sat_warn']['events']}（不拦停）")
    J["skipped"] = dict(n=len(sk), consumed=n + len(sk),
                        rows=[dict(id=s["id"], pool=s["pool"], lum_std=s["lum_std"],
                                   lum_q25=s["lum_q25"], lum_q75=s["lum_q75"],
                                   last_bad=s["last_bad"],
                                   reject_by_step=s["reject_by_step"]) for s in sk])
    P(f"- 被蒙版判据排除的源 {len(sk)} 个；池消耗 {n + len(sk)} 个源产出 {n} 条")
    if sk:
        P("")
        P("| 排除源 | pool | lum_std | lum_q25 | lum_q75 | 40 次全失败的步 |")
        P("|---|---|---|---|---|---|")
        for s in sk:
            P(f"| {s['id']} | {s['pool']} | {s['lum_std']:.5f} | {s['lum_q25']:.5f} "
              f"| {s['lum_q75']:.5f} | {json.dumps(s['reject_by_step'], ensure_ascii=False)} |")
    P("")

    # ---------------- v3.1 amplitude calibration ----------------------------
    if all("calib" in r for r in rows):
        c = [r["calib"] for r in rows]
        band = c[0]["target_band"]
        drawr = c[0].get("draw_range", band)
        P("## 6. 幅度标定（A20/A21 机制，参数取自 config）\n")
        P(f"目标带 [{band[0]}, {band[1]}]；逐样本目标抽自 U({drawr[0]}, {drawr[1]})，"
          f"键 = sha1(\"epr050-de00:\"+source_id)；容差 {DE00_TOL_DOC}；"
          f"ΔE00 = CIEDE2000 中位数，作用区 = 复合 α > 0.05。\n")
        P("| 量 | n | mean | p50 | p95 | max | min |")
        P("|---|---|---|---|---|---|---|")
        for lab, key in [("de00_target（抽样目标）", "de00_target"),
                         ("de00 @ s=1（标定前）", "de00_acted_at_s1"),
                         ("de00 标定后（作用区）", "de00_acted_after"),
                         ("de00 标定后（local 作用区）", "de00_local_acted_after"),
                         ("强度系数 s", "s"),
                         ("二分迭代数", "bisect_iters"),
                         ("作用区占比 acted_frac", "acted_frac"),
                         ("local 作用区占比", "acted_local_frac")]:
            d = num([x[key] for x in c])
            J.setdefault("T7", {})[key] = d
            P(f"| {lab} | {d['n']} | {d['mean']:.4f} | {d['p50']:.4f} | {d['p95']:.4f} "
              f"| {d['max']:.4f} | {d['min']:.4f} |")
        P("")
        hit = sum(abs(x["de00_acted_after"] - x["de00_target"]) <= 0.05 for x in c)
        J["T7_flags"] = dict(
            in_band=sum(x["in_band"] for x in c),
            under_target=sum(x["under_target"] for x in c),
            below_band_at_s1=sum(x.get("below_band_at_s1", False) for x in c),
            over_target_at_s1=sum(x.get("over_target_at_s1", False) for x in c),
            s_eq_1=sum(x["s"] == 1.0 for x in c), on_target=hit, n=n)
        f = J["T7_flags"]
        P("| 标志 | 计数 |")
        P("|---|---|")
        P(f"| in_band（标定后落在 [{band[0]}, {band[1]}]） | {f['in_band']} / {n} |")
        P(f"| 命中各自目标 ±0.05 | {f['on_target']} / {n} |")
        P(f"| over_target_at_s1（需要压缩） | {f['over_target_at_s1']} / {n} |")
        P(f"| under_target（s=1 仍够不到自己的目标，不放大） | {f['under_target']} / {n} |")
        P(f"| below_band_at_s1（s=1 时就低于带下沿 {band[0]}） | {f['below_band_at_s1']} / {n} |")
        P(f"| s == 1（未压缩） | {f['s_eq_1']} / {n} |")
        P("")
        P("s 分档：\n")
        P("| s 档 | n | de00 标定后 p50 | de00_target p50 |")
        P("|---|---|---|---|")
        for lo, hi in [(0.0, 0.4), (0.4, 0.6), (0.6, 0.8), (0.8, 1.0), (1.0, 1.01)]:
            sub = [x for x in c if lo <= x["s"] < hi]
            if not sub:
                continue
            lab = "s == 1" if lo == 1.0 else f"{lo:.1f}<=s<{hi:.1f}"
            P(f"| {lab} | {len(sub)} | "
              f"{st.median(x['de00_acted_after'] for x in sub):.3f} | "
              f"{st.median(x['de00_target'] for x in sub):.3f} |")
        P("")

    # ---------------- v3.1 coverage + side columns --------------------------
    if all("side" in r for r in rows):
        P("## 7. 蒙版作用面积（α > 0.05）与附带效果列\n")
        P(f"覆盖下限 {rows[0]['mask']['cover_min']}（subject/semantic 步豁免）；"
          "coverage_pre_scale = 乘 s 之前，coverage = 乘 s 之后。\n")
        P("| step | kind | 分层 | n | cov_pre mean | cov_pre p50 | cov_pre min "
          "| cov_post mean | < 0.35 的数 |")
        P("|---|---|---|---|---|---|---|---|---|")
        for k in range(NST):
            groups = [("all", rows)] if STEP_KIND[k] != "geom" else \
                [(g, [r for r in rows if r["mask"]["geom"] == g]) for g in sorted(geoms)]
            for lab, sub in groups:
                if not sub:
                    continue
                pre = num([r["steps"][k]["coverage_pre_scale"] for r in sub])
                post = num([r["steps"][k]["coverage"] for r in sub])
                lo = sum(r["steps"][k]["coverage_pre_scale"] < 0.35 for r in sub)
                J.setdefault("T8", {})[f"{k+1}|{lab}"] = dict(pre=pre, post=post, below=lo)
                P(f"| {k+1} | {STEP_KIND[k]} | {lab} | {pre['n']} | {pre['mean']:.4f} "
                  f"| {pre['p50']:.4f} | {pre['min']:.4f} | {post['mean']:.4f} | {lo} |")
        P("")
        # v3.6: the geometry width tables below only apply to the v3.1-v3.4
        # shapes.  Main-chain rows (v3.5+) record LR parameters instead, and get
        # their own table further down.
        lin = [r for r in rows if r["mask"]["geom"] == "linear"
               and "feather_frac_long" in (r["mask"].get("geom_params") or {})]
        if lin:
            fl = num([r["mask"]["geom_params"]["feather_frac_long"] for r in lin])
            fs = num([r["mask"]["geom_params"]["feather_frac_short"] for r in lin])
            J["T8_linear"] = dict(frac_long=fl, frac_short=fs,
                                  cfg_ref=MK.get("linear_feather_ref"),
                                  cfg_range=MK.get("linear_feather"),
                                  cfg_min_short=MK.get("linear_feather_min_short"))
            P("| 线性蒙版羽化 | n | mean | p50 | p95 | max | min |")
            P("|---|---|---|---|---|---|---|")
            for lab, d in [("feather / 长边", fl), ("feather / 短边", fs)]:
                P(f"| {lab} | {d['n']} | {d['mean']:.4f} | {d['p50']:.4f} "
                  f"| {d['p95']:.4f} | {d['max']:.4f} | {d['min']:.4f} |")
            P(f"\n采样口径（config）：ref = {MK.get('linear_feather_ref')}，"
              f"linear_feather = {MK.get('linear_feather')}，"
              f"linear_feather_min_short = {MK.get('linear_feather_min_short')}\n")
        rad = [r for r in rows if r["mask"]["geom"] == "radial"
               and "feather" in (r["mask"].get("geom_params") or {})]
        if rad:
            rf = num([r["mask"]["geom_params"]["feather"] for r in rad])
            J["T8_radial"] = dict(feather=rf, cfg_range=MK.get("radial_feather"))
            P(f"径向蒙版 feather（归一化椭圆半径单位）：n {rf['n']} / mean {rf['mean']:.4f} "
              f"/ p50 {rf['p50']:.4f} / min {rf['min']:.4f} / max {rf['max']:.4f}"
              f"（采样区间 {MK.get('radial_feather')}）\n")
        # A24: 「贯穿长边 >= span_long_min」按作用区沿长轴的几何跨度判定。
        # v3.1 的行没有 geom_span_long（该判据 v3.2 才加），旧日志上跳过此表。
        if all("geom_span_long" in r["mask"] for r in rows):
            smin = MK.get("span_long_min", rows[0]["mask"].get("span_long_min"))
            P(f"### 作用区沿长轴跨度 geom_span_long（A24，判据下限 {smin}；"
              "subject/semantic 步豁免）\n")
            P("| geom | 豁免 | n | mean | p50 | p95 | max | min | < 下限的数 |")
            P("|---|---|---|---|---|---|---|---|---|")
            for g in sorted(geoms):
                sub = [r for r in rows if r["mask"]["geom"] == g]
                d = num([r["mask"]["geom_span_long"] for r in sub])
                below = sum(r["mask"]["geom_span_long"] < smin for r in sub)
                J.setdefault("T8_span", {})[g] = dict(d, below=below,
                                                      exempt=(g in SUBJECT_SHAPED))
                P(f"| {g} | {g in SUBJECT_SHAPED} | {d['n']} | {d['mean']:.4f} "
                  f"| {d['p50']:.4f} | {d['p95']:.4f} | {d['max']:.4f} "
                  f"| {d['min']:.4f} | {below} |")
            P("")
        # ---- v3.6: mask shape (the criteria the user wrote) ----------------
        if any("geom_shape" in r["mask"] for r in rows):
            fmax = MK.get("full_frac_of_mask_max")
            feps = MK.get("full_eps")
            P(f"### 7b. 几何蒙版形态（v3.6 判据：cover >= {MK.get('cover_min')}，"
              f"full_frac_of_mask <= {fmax}；full_eps = {feps}）\n")
            P("`full_frac_of_mask` = frac(α >= full_eps) / frac(α > cover_eps)，"
              "**分母是 mask 不是画面**；`mid_frac_of_mask` = 过渡像素占 mask 的比例。"
              "semantic 族不受这两条判据约束（形状由主体决定），照样落盘。\n")
            P("| geom | n | 判据适用 | cover mean | cover min | full_of_mask mean "
              "| p50 | max | mid_of_mask mean | min | width mean | width tries mean "
              "| attempts mean |")
            P("|---|---|---|---|---|---|---|---|---|---|---|---|---|")
            for g in sorted(geoms):
                sub = [r for r in rows if r["mask"]["geom"] == g
                       and "geom_shape" in r["mask"]]
                if not sub:
                    continue
                cov = num([r["mask"]["geom_shape"]["coverage"] for r in sub])
                fo = num([r["mask"]["geom_shape"]["full_frac_of_mask"] for r in sub])
                mo = num([r["mask"]["geom_shape"]["mid_frac_of_mask"] for r in sub])
                wv = [r["mask"]["geom_params"].get("width_value") for r in sub]
                wt = [r["mask"]["geom_params"].get("width_tries") for r in sub]
                wvd = num([v for v in wv if v is not None])
                wtd = num([v for v in wt if v is not None])
                at = num([r["mask"]["attempts"] for r in sub])
                J.setdefault("T8_shape", {})[g] = dict(
                    n=len(sub), cover=cov, full_of_mask=fo, mid_of_mask=mo,
                    width=wvd, width_tries=wtd, attempts=at)
                P(f"| {g} | {len(sub)} | {g not in SUBJECT_SHAPED} "
                  f"| {cov['mean']:.4f} | {cov['min']:.4f} | {fo['mean']:.4f} "
                  f"| {fo['p50']:.4f} | {fo['max']:.4f} | {mo['mean']:.4f} "
                  f"| {mo['min']:.4f} "
                  f"| {'-' if not wvd else format(wvd['mean'], '.3f')} "
                  f"| {'-' if not wtd else format(wtd['mean'], '.2f')} "
                  f"| {at['mean']:.2f} |")
            P("")
            gate = [r["mask"]["semantic_gate"] for r in rows
                    if r["mask"].get("semantic_gate")]
            if gate:
                drop = sum(bool(x["dropped"]) for x in gate)
                sa = num([x["subject_area"] for x in gate])
                J["T8_semantic_gate"] = dict(
                    cover_min=gate[0]["cover_min"], dropped=drop, n=len(gate),
                    subject_area=sa,
                    dropped_area=num([x["subject_area"] for x in gate
                                      if x["dropped"]]))
                P(f"- semantic 门槛 semantic_cover_min = {gate[0]['cover_min']}："
                  f"被剔除该族的行 {drop} / {len(gate)}；"
                  f"subject_area mean {sa['mean']:.4f} / p50 {sa['p50']:.4f} "
                  f"/ min {sa['min']:.4f} / max {sa['max']:.4f}\n")
        P("| 附带效果列（只落盘，非判据） | n | mean | p50 | p95 | max | min |")
        P("|---|---|---|---|---|---|---|")
        for lab, key in [("饱和度均值 before", "sat_mean_before"),
                         ("饱和度均值 after", "sat_mean_after"),
                         ("饱和度均值 Δ", "sat_mean_delta"),
                         ("对比度 before（亮度 std）", "contrast_before"),
                         ("对比度 after", "contrast_after"),
                         ("对比度 Δ", "contrast_delta")]:
            d = num([r["side"][key] for r in rows])
            J.setdefault("T8_side", {})[key] = d
            P(f"| {lab} | {d['n']} | {d['mean']:+.5f} | {d['p50']:+.5f} | {d['p95']:+.5f} "
              f"| {d['max']:+.5f} | {d['min']:+.5f} |")
        P("")

    # ---------------- alpha field census ------------------------------------
    P("## 8. alpha 场面积占比（每步）\n")
    P("| step | kind | frac_zero mean | frac_mid mean | frac_one mean "
      "| alpha mean | gamut_cover mean |")
    P("|---|---|---|---|---|---|---|")
    for k in range(NST):
        fz = num([r["steps"][k]["alpha"]["frac_zero"] for r in rows])
        fm = num([r["steps"][k]["alpha"]["frac_mid"] for r in rows])
        fo = num([r["steps"][k]["alpha"]["frac_one"] for r in rows])
        am = num([r["steps"][k]["alpha"]["mean"] for r in rows])
        gcv = num([r["steps"][k]["gamut_cover"] for r in rows])
        J.setdefault("alpha_census", {})[str(k + 1)] = dict(
            frac_zero=fz, frac_mid=fm, frac_one=fo, alpha_mean=am, gamut_cover=gcv)
        P(f"| {k+1} | {STEP_KIND[k]} | {fz['mean']:.4f} | {fm['mean']:.4f} "
          f"| {fo['mean']:.4f} | {am['mean']:.4f} | {gcv['mean']:.5f} |")
    P("")

    # A13/A41: name the rows where a per-step band guard has an EMPTY set to
    # assert over, so a vacuously-true assertion can never be read as a pass.
    vac = {}
    for k in range(NST):
        z = sum(1 for r in rows if r["steps"][k]["alpha"]["zero_band_px"] == 0)
        s = sum(1 for r in rows if r["steps"][k]["alpha"]["soft_band_px"] == 0)
        ex = any(r["steps"][k]["alpha"].get("band_exempt") for r in rows)
        vac[f"step{k+1}_{STEP_KIND[k]}"] = dict(
            zero_band_empty=z, soft_band_empty=s, band_exempt=ex, n=len(rows))
    J["vacuous_band_guards"] = vac
    P("空集守卫计数（A13/A41）：某步某行 `zero_band_px == 0` 时，该行该步的 "
      "**alpha==0 逐比特断言是在空集上平凡成立**，不构成证据。下表逐步给出这样的行数。\n")
    P("| 步 | kind | 判据豁免 | zero_band 空集行 | soft_band 空集行 | 总行 |")
    P("|---|---|---|---|---|---|")
    for kk, v in vac.items():
        P(f"| {kk.split('_')[0][4:]} | {'_'.join(kk.split('_')[1:])} "
          f"| {'是' if v['band_exempt'] else '否'} | {v['zero_band_empty']} "
          f"| {v['soft_band_empty']} | {v['n']} |")
    P("")

    # ---------------- v3.4 hue step ----------------------------------------
    if "hue" in STEP_KIND and rows[0].get("mask", {}).get("hue"):
        hk = STEP_KIND.index("hue")
        P("## 8b. 色相步（v3.4）\n")
        h0 = rows[0]["mask"]["hue"]
        P(f"色相/饱和度口径：**直接取自存储的 sRGB 值**的教科书 HSV（`S=(max-min)/max`，"
          f"色相由最大通道定），不做线性化、不引入新依赖，与 LUT 链和下游训练/推理"
          f"消费的是同一批数值；`max==min` 处色相无定义 → 记 0 且 S=0，由彩度门控抹掉。\n")
        P(f"权重锚点（度, 权重，圆周线性插值）：`{h0['anchors']}`；"
          f"彩度门控 smoothstep(S; {h0['sat_lo']} → {h0['sat_hi']})。\n")
        P("**`weight_mean` 是「这步实际动了多少」的主口径**；`coverage` 对色相步偏乐观"
          f"（权重下限 {min(w for _, w in h0['anchors'])} > cover_eps，任何有彩像素都算"
          "「作用到」），两列都报、以 `weight_mean` 为准。\n")
        P("| pool | 行数 | weight_mean mean | min | p50 | p95 | coverage mean | p50 "
          "| blue_frac_frame mean | p50 | chroma_frac mean | inv_iters p50 |")
        P("|---|---|---|---|---|---|---|---|---|---|---|---|")
        hj = {}
        for pl in ["main", "ctrl", "all"]:
            sub = [r for r in rows if pl == "all" or r["pool"] == pl]
            if not sub:
                continue
            wm = num([r["mask"]["hue"]["weight_mean"] for r in sub])
            cv = num([r["steps"][hk]["coverage"] for r in sub])
            bf = num([r["mask"]["hue"]["blue_frac_frame"] for r in sub])
            cf = num([r["mask"]["hue"]["chroma_frac"] for r in sub])
            it = num([r["steps"][hk]["inv_iters"] for r in sub])
            hj[pl] = dict(weight_mean=wm, coverage=cv, blue_frac_frame=bf,
                          chroma_frac=cf, inv_iters=it, n=len(sub))
            P(f"| {pl} | {len(sub)} | {wm['mean']:.4f} | {wm['min']:.4f} "
              f"| {wm['p50']:.4f} | {wm['p95']:.4f} | {cv['mean']:.4f} | {cv['p50']:.4f} "
              f"| {bf['mean']:.4f} | {bf['p50']:.4f} | {cf['mean']:.4f} | {it['p50']:.0f} |")
        P("")
        # acted-area strata, by how much colour of the weighted hues is present
        P("按色相步实际作用量分层（`weight_mean`）：\n")
        # err_E on the hue step's ACTED region, i.e. the 0<alpha<1 band (the hue
        # alpha is almost never exactly 1), pooled with the file's agg() rule
        P("| weight_mean 档 | 行数 | coverage mean | blue_frac_frame mean "
          "| hue 步 recovered mean | hue 步 err_E(0<a<1) mean | p50* | inv_iters p50 |")
        P("|---|---|---|---|---|---|---|---|")
        wb = [("< 0.10", 0.0, 0.10), ("0.10-0.25", 0.10, 0.25),
              ("0.25-0.50", 0.25, 0.50), (">= 0.50", 0.50, 1.01)]
        ek = f"step{hk+1}_hue"
        for lab, lo, hi in wb:
            sub = [r for r in rows if lo <= r["mask"]["hue"]["weight_mean"] < hi]
            if not sub:
                continue
            hj.setdefault("by_weight", {})[lab] = dict(
                n=len(sub),
                coverage=num([r["steps"][hk]["coverage"] for r in sub]),
                blue=num([r["mask"]["hue"]["blue_frac_frame"] for r in sub]),
                recovered=num([r["steps"][hk]["recovered"] for r in sub]),
                inv_iters=num([r["steps"][hk]["inv_iters"] for r in sub]))
            cv = num([r["steps"][hk]["coverage"] for r in sub])
            bf = num([r["mask"]["hue"]["blue_frac_frame"] for r in sub])
            rc = num([r["steps"][hk]["recovered"] for r in sub])
            ee = agg([r["err_E_by_step_alpha"][ek]["mid"] for r in sub
                      if ek in r.get("err_E_by_step_alpha", {})])
            it = num([r["steps"][hk]["inv_iters"] for r in sub])
            es = ("- | -" if ee["n_u"] == 0
                  else f"{ee['mean']:.4g} | {ee['p50']:.4g}")
            hj["by_weight"][lab]["err_E_mid"] = ee
            P(f"| {lab} | {len(sub)} | {cv['mean']:.4f} | {bf['mean']:.4f} "
              f"| {rc['mean']:.4f} | {es} | {it['p50']:.0f} |")
        # hue circle census over chromatic pixels
        P("")
        P("色相圆周普查（占**有彩像素**的比例，全体行均值）：\n")
        keys = list(rows[0]["mask"]["hue"]["hue_frac"])
        P("| " + " | ".join(keys) + " |")
        P("|" + "---|" * len(keys))
        P("| " + " | ".join(
            f"{sum(r['mask']['hue']['hue_frac'][kk] for r in rows)/len(rows):.4f}"
            for kk in keys) + " |")
        hj["hue_frac_mean"] = {kk: sum(r["mask"]["hue"]["hue_frac"][kk]
                                       for r in rows) / len(rows) for kk in keys}
        nb = sum(1 for r in rows if r["mask"]["hue"]["blue_frac_frame"] < 0.001)
        lowa = sum(1 for r in rows if r["steps"][hk]["coverage"] < 0.35)
        hj["no_blue_rows"] = nb
        hj["coverage_below_cover_min"] = lowa
        P(f"\n- `blue_frac_frame < 0.001`（近乎无蓝）的行：**{nb} / {len(rows)}**；"
          f"这些行照常入集，不排除（A41）。")
        P(f"- 色相步 `coverage < {CFG.get('masks', {}).get('cover_min', 0.35)}` 的行："
          f"**{lowa} / {len(rows)}**；即若不豁免覆盖下限会被判据拒掉的行数。\n")
        J["hue_step"] = hj

    # ---------------- iteration cap / hard pixels ---------------------------
    cap = RUN.get("inv_iters")
    early = RUN.get("inv_early_levels")
    ctol = RUN.get("conv_tol")
    if cap is None:                       # pre-v3.2 manifests carry no [run] block
        scp = Path(a.pairs).parent / "self_check.json"
        if scp.exists():
            sc = json.loads(scp.read_text())
            cap = sc.get("inv_iters_cap", cap)
            early = sc.get("inv_early_stop_levels", early)
    if cap is None:
        cap = max(s["inv_iters"] for r in rows for s in r["steps"])
    P(f"## 9. 打满迭代上限的步与其难解像素（cap = {cap}，早停阈 {early} 级，"
      f"失败判据残差 > {ctol} 级）\n")
    P("单元 = (样本 x 步)。cap 步 = `inv_iters == cap`，即早停条件在 cap 次内未满足。\n")
    P("| pool | rec_band | 步单元数 | cap 步数 | cap 占比 | cap 步 conv_fail mean "
      "| cap 步 conv_fail max | cap 步 resid_p95 p50 | cap 步 resid_p95 max |")
    P("|---|---|---|---|---|---|---|---|---|")
    t9 = {}
    cells = [(p, rb) for p in sorted(pools) for rb in sorted(recb)] + \
            [(p, "all") for p in sorted(pools)] + [("all", "all")]
    for p, rb in cells:
        sub = [r for r in rows if (p == "all" or r["pool"] == p)
               and (rb == "all" or r["rec_band"] == rb)]
        units = [s for r in sub for s in r["steps"]]
        if not units:
            continue
        cs = [s for s in units if s["inv_iters"] == cap]
        d = dict(units=len(units), cap=len(cs), frac=len(cs) / len(units),
                 fail=num([s["conv_fail_frac"] for s in cs]),
                 resid=num([s["resid_p95"] for s in cs]),
                 samples=len(sub))
        t9[f"{p}|{rb}"] = d
        if cs:
            P(f"| {p} | {rb} | {len(units)} | {len(cs)} | {d['frac']:.4f} "
              f"| {d['fail']['mean']:.3e} | {d['fail']['max']:.3e} "
              f"| {d['resid']['p50']:.3e} | {d['resid']['max']:.3e} |")
        else:
            P(f"| {p} | {rb} | {len(units)} | 0 | 0.0000 | - | - | - | - |")
    J["T9"] = t9
    P("")
    P("inv_iters 分池分档（全部步单元，不只 cap 步）：\n")
    P("| pool | rec_band | n | mean | p50 | p95 | max | min |")
    P("|---|---|---|---|---|---|---|---|")
    for p, rb in cells:
        sub = [r for r in rows if (p == "all" or r["pool"] == p)
               and (rb == "all" or r["rec_band"] == rb)]
        it = num([s["inv_iters"] for r in sub for s in r["steps"]])
        if not it:
            continue
        J.setdefault("T9_iters", {})[f"{p}|{rb}"] = it
        P(f"| {p} | {rb} | {it['n']} | {it['mean']:.2f} | {it['p50']:.0f} "
          f"| {it['p95']:.0f} | {it['max']:.0f} | {it['min']:.0f} |")
    P("")
    P("每步 conv_fail_frac 分池分档（残差 > 1 级的像素占比，全部步单元）：\n")
    P("| pool | rec_band | n | mean | p50 | p95 | max | >0 的步数 |")
    P("|---|---|---|---|---|---|---|---|")
    for p, rb in cells:
        sub = [r for r in rows if (p == "all" or r["pool"] == p)
               and (rb == "all" or r["rec_band"] == rb)]
        v = [s["conv_fail_frac"] for r in sub for s in r["steps"]]
        d = num(v)
        if not d:
            continue
        J.setdefault("T9_fail", {})[f"{p}|{rb}"] = dict(d, nonzero=sum(x > 0 for x in v))
        P(f"| {p} | {rb} | {d['n']} | {d['mean']:.3e} | {d['p50']:.3e} "
          f"| {d['p95']:.3e} | {d['max']:.3e} | {sum(x > 0 for x in v)} |")
    P("")

    txt = "\n".join(O)
    print(txt)
    if a.out_json:
        Path(a.out_json).write_text(json.dumps(J, ensure_ascii=False, indent=1,
                                               default=float))


if __name__ == "__main__":
    main()
