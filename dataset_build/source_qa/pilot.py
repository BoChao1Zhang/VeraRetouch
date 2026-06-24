"""固定 pilot 样本的选锁 + 每轮真实推理 + §7 指标测量驱动（三套问卷迭代）。

设计文档 LLM_QA_QUESTIONNAIRE_DESIGN_2026-06-17 §4.1 第 3 步（pilot 硬 gate）。

子命令:
  select   选 200 图(portrait100+非100, 混横/竖, 跨 corpora)+200 preset 锁 pilot/manifest.json
  run-img  对锁定 200 图各跑 IMQ+aes 真实 vLLM, 原始位置码串+裁定+信号落 round_<r>/raw.jsonl
  measure  据 raw.jsonl 算 §7 指标 → round_<r>/metrics.json
  status   打印 state.json

样本锁定不变量: select 仅在 round 0 写一次, 后续只读复用(指标轮间可比)。
硬 drop 全程关闭(DEFECT_HARD_DROP/PRO_DROP_ENABLED/INTENT_DROP_ENABLED/KAPPA_PASS=False)。
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import threading
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed

from . import config as C
from . import db
from . import qa_clean as Q
from . import qa_runner as R

PILOT_DIR = os.path.join(os.path.dirname(__file__), "pilot")
MANIFEST = os.path.join(PILOT_DIR, "manifest.json")
STATE = os.path.join(PILOT_DIR, "state.json")

SEL_SEED = "pilot-2026-06-18"          # 选样确定性种子(记 manifest)
IMG_SIGNAL_COLS = ["asset_id", "corpus", "is_portrait_pool", "width", "height",
                   "max_face_frac", "is_bw_img", "musiq", "niqe", "brisque",
                   "noise_sigma", "sharpness", "clipiqa", "aesthetic", "aesthetic_vlm"]


# --------------------------------------------------------------------------- #
def _read_json(p, default=None):
    if os.path.exists(p):
        with open(p) as f:
            return json.load(f)
    return default


def _write_json(p, obj):
    os.makedirs(os.path.dirname(p), exist_ok=True)
    with open(p, "w") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)


# --------------------------------------------------------------------------- #
# 选样 + 锁定
# --------------------------------------------------------------------------- #
def _pick_images(conn):
    cols = ",".join(IMG_SIGNAL_COLS)
    picks = []
    # 4 层 × 50 = 200: pp∈{0,1} × ori∈{land(w>h), port(w<h)}; 跨 corpora 由 md5 序自然混
    for pp in (1, 0):
        for ori, cmp in (("land", ">"), ("port", "<")):
            rows = conn.execute(
                f"SELECT {cols} FROM assets WHERE asset_type='image' AND musiq IS NOT NULL "
                f"AND dup_of IS NULL AND width IS NOT NULL AND is_portrait_pool=%s "
                f"AND width {cmp} height "
                f"ORDER BY md5(asset_id || %s) LIMIT 50",
                (pp, SEL_SEED)).fetchall()
            for r in rows:
                d = {k: r[k] for k in IMG_SIGNAL_COLS}
                d["orientation"] = ori
                picks.append(d)
    return picks


def _pick_presets(conn):
    picks = []
    # global(has_local_mask=0) 100 + local 100; status IN(preset_meta_pass, preset_meta_local)
    for local, n in ((0, 100), (1, 100)):
        rows = conn.execute(
            "SELECT asset_id, corpus, status, has_local_mask, kind, preset_content_hash "
            "FROM assets WHERE asset_type='preset' AND has_local_mask=%s "
            "AND status IN ('preset_meta_pass','preset_meta_local') "
            "ORDER BY md5(asset_id || %s) LIMIT %s",
            (local, SEL_SEED, n)).fetchall()
        for r in rows:
            picks.append({k: r[k] for k in
                          ("asset_id", "corpus", "status", "has_local_mask", "kind",
                           "preset_content_hash")})
    return picks


def select_and_lock(force=False):
    state = _read_json(STATE, {"round": 0, "converged": False})
    if os.path.exists(MANIFEST) and not force:
        print("[pilot] manifest 已存在; 样本锁定不变量 → 拒绝重选(用 --force 覆盖, 仅 round 0)",
              file=sys.stderr)
        return _read_json(MANIFEST)
    if state.get("round", 0) != 0 and not force:
        print(f"[pilot] round={state['round']} != 0, 拒绝重选样本", file=sys.stderr)
        return _read_json(MANIFEST)
    conn = db.connect()
    images = _pick_images(conn)
    presets = _pick_presets(conn)
    manifest = {
        "sel_seed": SEL_SEED,
        "scramble_seeds": {"IMQ": C.IMQ_SCRAMBLE_SEED, "aes": C.AES_SCRAMBLE_SEED,
                           "preset": C.PRESET_SCRAMBLE_SEED},
        "temperature": {"IMQ": 0.1, "aes": C.AES_TEMPERATURE},
        "flags": {"HAS_EXIF_FIXED": C.HAS_EXIF_FIXED, "HAS_ISBW_IMG": C.HAS_ISBW_IMG,
                  "ENABLE_T_COLOR": C.ENABLE_T_COLOR,
                  "DEFECT_HARD_DROP": C.DEFECT_HARD_DROP,
                  "PRO_DROP_ENABLED": C.PRO_DROP_ENABLED,
                  "INTENT_DROP_ENABLED": C.INTENT_DROP_ENABLED,
                  "PRESET_T2_HARD": C.PRESET_T2_HARD, "PRESET_T3_HARD": C.PRESET_T3_HARD},
        "n_images": len(images), "n_presets": len(presets),
        "image_strata": dict(Counter((i["is_portrait_pool"], i["orientation"]) for i in images).most_common()),
        "image_corpora": dict(Counter(i["corpus"] for i in images).most_common()),
        "preset_strata": dict(Counter(p["has_local_mask"] for p in presets).most_common()),
        "images": images, "presets": presets,
        "notes": "is_bw_img 未计算(HAS_ISBW_IMG=False); color/bw 分层与 COLOR/LANDSCAPE 陷阱本轮降级; "
                 "preset 真渲染需 mean_luma 探针列 + LR farm 在线, 本轮 image-only。",
    }
    # 把元组键转成字符串便于 JSON
    manifest["image_strata"] = {f"pp{k[0]}_{k[1]}": v for k, v in
                                Counter((i["is_portrait_pool"], i["orientation"]) for i in images).items()}
    _write_json(MANIFEST, manifest)
    _write_json(STATE, {"round": 0, "converged": False,
                        "locked": True, "n_images": len(images), "n_presets": len(presets)})
    print(f"[pilot] locked: {len(images)} images, {len(presets)} presets → {MANIFEST}")
    print(f"  image strata: {manifest['image_strata']}")
    print(f"  image corpora: {manifest['image_corpora']}")
    print(f"  preset strata(has_local_mask): {manifest['preset_strata']}")
    return manifest


# --------------------------------------------------------------------------- #
# 每轮真实推理(图像)
# --------------------------------------------------------------------------- #
def _ans_by_audit(raw, pos_map):
    """best-effort 还原 {audit_id: bit}(供 per-题 yes 率测量, 容忍解析不全)。"""
    bit, ok, _ = Q.parse_codes(raw or "", set(pos_map))
    return {pos_map[p][2]: b for p, b in bit.items() if p in pos_map}


def _process_image(asset):
    rec = {"asset_id": asset["asset_id"], "corpus": asset["corpus"],
           "is_portrait_pool": asset["is_portrait_pool"], "orientation": asset.get("orientation"),
           "signals": {k: asset.get(k) for k in
                       ("width", "height", "max_face_frac", "is_bw_img", "musiq", "niqe",
                        "brisque", "noise_sigma", "aesthetic", "aesthetic_vlm")}}
    try:
        uri = R.img_data_uri(asset["path"])
    except Exception as e:  # noqa: BLE001
        rec["error"] = f"decode:{e}"
        return rec
    # IMQ
    try:
        imq = R.run_imq(asset, uri)
        pm = C.imq_pos_map(bool(asset.get("is_portrait_pool")))
        v, vr = (None, None)
        if imq["reliable"]:
            v, vr = Q.verdict_imq(imq["A"], bool(asset.get("is_portrait_pool")),
                                  asset.get("max_face_frac"), imq["defect_count"])
            v, vr = Q.reconcile_imq(v, vr, asset)
        rec["imq"] = {"reliable": imq["reliable"], "reason": imq["reason"],
                      "defect_count": imq["defect_count"], "contra": imq["contradiction_count"],
                      "trap_fail": imq["trap_fail"], "trap_skipped": imq["trap_skipped"],
                      "reduced_anchor": imq["reduced_anchor"],
                      "landscape_soft_fail": imq.get("landscape_soft_fail"),
                      "reask": imq.get("reask_count", 0), "verdict": v, "verdict_reason": vr,
                      "raw": imq.get("raw"), "ans": _ans_by_audit(imq.get("raw"), pm)}
    except Exception as e:  # noqa: BLE001
        rec["imq_error"] = str(e)
    # aes
    try:
        aes = R.run_aes(asset, uri)
        mff = asset.get("max_face_frac")
        has_face = (mff is not None and mff > C.AES_FACE_MIN)
        pm = C.aes_pos_map(has_face)
        rec["aes"] = {"reliable": aes["reliable"], "reason": aes["reason"],
                      "merit_count": aes["merit_count"], "merit_n": aes["merit_n"],
                      "merit_frac": aes["merit_frac"], "contra": aes["contradiction_count"],
                      "trap_fail": aes["trap_fail"], "soft_trap_fail": aes.get("soft_trap_fail"),
                      "aes_keep_vote": aes.get("aes_keep_vote"),
                      "reask": aes.get("reask_count", 0), "has_face": has_face,
                      "raw": aes.get("raw"), "ans": _ans_by_audit(aes.get("raw"), pm)}
    except Exception as e:  # noqa: BLE001
        rec["aes_error"] = str(e)
    return rec


def run_images_round(concurrency=None, limit=None):
    manifest = _read_json(MANIFEST)
    if not manifest:
        print("[pilot] 无 manifest, 先 select", file=sys.stderr)
        return
    state = _read_json(STATE, {"round": 0, "converged": False})
    rnd = state.get("round", 0)
    if rnd == 0:
        rnd = 1            # 进入第 1 轮
    concurrency = concurrency or C.VLLM_CONCURRENCY
    conn = db.connect()
    # 拼回 path(manifest 不存 path, 防泄漏 + 路径可能变)
    ids = [i["asset_id"] for i in manifest["images"]]
    if limit:
        ids = ids[:limit]
    paths = dict(conn.execute(
        "SELECT asset_id, path FROM assets WHERE asset_id = ANY(%s)", (ids,)).fetchall())
    work = []
    by_id = {i["asset_id"]: i for i in manifest["images"]}
    for aid in ids:
        a = dict(by_id[aid])
        a["path"] = paths.get(aid)
        work.append(a)

    rdir = os.path.join(PILOT_DIR, f"round_{rnd}")
    os.makedirs(rdir, exist_ok=True)
    outp = os.path.join(rdir, "raw.jsonl")
    lock = threading.Lock()
    n_ok = n_err = 0
    with open(outp, "w") as fh, ThreadPoolExecutor(max_workers=concurrency) as ex:
        futs = {ex.submit(_process_image, a): a["asset_id"] for a in work}
        for j, fut in enumerate(as_completed(futs)):
            rec = fut.result()
            with lock:
                fh.write(json.dumps(rec, ensure_ascii=False) + "\n")
                fh.flush()
            if rec.get("error") or rec.get("imq_error") or rec.get("aes_error"):
                n_err += 1
            else:
                n_ok += 1
            if (j + 1) % 25 == 0:
                print(f"[pilot] round {rnd}: {j+1}/{len(work)} ok={n_ok} err={n_err}", file=sys.stderr)
    _write_json(STATE, {**state, "round": rnd, "converged": state.get("converged", False),
                        f"round_{rnd}_ran": True})
    print(f"[pilot] round {rnd} images done: {n_ok} ok, {n_err} err → {outp}")
    return outp


_IMG_COLS = ("asset_id", "corpus", "path", "is_portrait_pool", "max_face_frac",
             "is_bw_img", "musiq", "niqe", "brisque", "noise_sigma", "aesthetic", "aesthetic_vlm")


def run_images_full(concurrency=None, limit=None):
    """全量 source img 两阶段 LLM QA(IMQ+审美), 落库 verdict/merit + 续跑。
    IQA 不参与判定(config.IQA_IN_CLEAN=False)。结果同时落 pilot/full/img_qa.jsonl。"""
    concurrency = concurrency or C.VLLM_CONCURRENCY
    conn = db.connect()
    run_id = db.start_run(conn, "img_qa_full", {"limit": limit})
    rows = conn.execute(
        f"SELECT {','.join(_IMG_COLS)} FROM assets WHERE asset_type='image' AND dup_of IS NULL "
        "AND path IS NOT NULL"
        + (f" LIMIT {int(limit)}" if limit else "")).fetchall()
    outdir = os.path.join(PILOT_DIR, "full"); os.makedirs(outdir, exist_ok=True)
    outp = os.path.join(outdir, "img_qa.jsonl")
    done = set()
    if os.path.exists(outp):
        for l in open(outp):
            try:
                done.add(json.loads(l)["asset_id"])
            except Exception:  # noqa: BLE001
                pass
    work = [dict(r) for r in rows if r["asset_id"] not in done]
    print(f"[img-full] {len(rows)} imgs, resume skip {len(done)}, {len(work)} remain", file=sys.stderr)
    lock = threading.Lock()
    counts = {"keep": 0, "review": 0, "drop": 0, "unreliable": 0, "err": 0}

    def _persist(rec):
        aid = rec["asset_id"]
        imq = rec.get("imq") or {}
        v = imq.get("verdict")
        aes = rec.get("aes") or {}
        with lock:
            if v in ("keep", "review", "drop"):
                db.update_asset_fields(conn, aid, auto_verdict=v,
                                       merit_frac=aes.get("merit_frac"))
                _i = lambda d: {k: (int(x) if isinstance(x, bool) else x) for k, x in d.items()}
                db.add_llm_qa_run(conn, aid, "IMQ", _i(imq), model=C.VLLM_MODEL, run_id=run_id)
                if aes:
                    db.add_llm_qa_run(conn, aid, "AES", _i(aes), model=C.VLLM_MODEL, run_id=run_id)
                counts[v] += 1
            else:
                counts["unreliable"] += 1
            conn.commit()

    with open(outp, "a") as fh, ThreadPoolExecutor(max_workers=concurrency) as ex:
        futs = {ex.submit(_process_image, a): a["asset_id"] for a in work}
        for j, fut in enumerate(as_completed(futs)):
            try:
                rec = fut.result()
            except Exception as e:  # noqa: BLE001
                counts["err"] += 1; continue
            with lock:
                fh.write(json.dumps(rec, ensure_ascii=False) + "\n"); fh.flush()
            if rec.get("error") or rec.get("imq_error"):
                counts["err"] += 1
            else:
                _persist(rec)
            if (j + 1) % 200 == 0:
                print(f"[img-full] {j+1}/{len(work)} {counts}", file=sys.stderr)
    db.finish_run(conn, run_id, counts)
    conn.close()
    print(json.dumps(counts))
    return counts


# --------------------------------------------------------------------------- #
# 测量 §7
# --------------------------------------------------------------------------- #
def _band(x, lo=0.03, hi=0.97):
    return lo <= x <= hi


def measure(rnd=None):
    state = _read_json(STATE, {"round": 0})
    rnd = rnd or state.get("round", 1)
    rdir = os.path.join(PILOT_DIR, f"round_{rnd}")
    recs = [json.loads(l) for l in open(os.path.join(rdir, "raw.jsonl"))]
    n = len(recs)

    # ---- IMQ ----
    imq = [r["imq"] for r in recs if "imq" in r]
    imq_rel = [x for x in imq if x["reliable"]]
    yes = defaultdict(lambda: [0, 0])      # audit_id -> [yes, total]
    for x in imq:
        for aid, b in (x.get("ans") or {}).items():
            yes[aid][0] += b
            yes[aid][1] += 1
    imq_yes = {k: round(v[0] / v[1], 4) for k, v in sorted(yes.items()) if v[1]}
    # 缺陷题(F&R)出带检查
    DEF_Q = [f"{d}_{p}" for d in ("SHARP", "NOISE", "COMP", "UPSC", "EXPO", "OVERCOOK") for p in ("F", "R")]
    imq_out_of_band = {k: imq_yes[k] for k in imq_yes
                       if k in DEF_Q and not _band(imq_yes[k])}
    defect_dist = Counter(x["defect_count"] for x in imq_rel if x["defect_count"] is not None)
    imq_reason = Counter(x["reason"] for x in imq if not x["reliable"])
    hard_contra = sum(1 for x in imq if x["reason"] == "contradiction_hard")
    landscape_softfail = [x.get("landscape_soft_fail") for x in imq if x.get("landscape_soft_fail") is not None]
    imq_verdict = Counter(x.get("verdict") for x in imq_rel)

    # ---- aes ----
    aes = [r["aes"] for r in recs if "aes" in r]
    aes_rel = [x for x in aes if x["reliable"]]
    ayes = defaultdict(lambda: [0, 0])
    for x in aes:
        for aid, b in (x.get("ans") or {}).items():
            ayes[aid][0] += b
            ayes[aid][1] += 1
    aes_yes = {k: round(v[0] / v[1], 4) for k, v in sorted(ayes.items()) if v[1]}
    AES_FR = [a for a in aes_yes if a[0] in "KLCDMN" and a[-1].isdigit()]
    aes_out_of_band = {k: aes_yes[k] for k in AES_FR if not _band(aes_yes[k])}
    mf = [x["merit_frac"] for x in aes_rel if x["merit_frac"] is not None]
    mf_hist = Counter(round(v, 2) for v in mf)
    mf_mode_frac = (max(mf_hist.values()) / len(mf)) if mf else None
    aes_reason = Counter(x["reason"] for x in aes if not x["reliable"])
    aes_softfail = [x.get("soft_trap_fail") for x in aes if x.get("soft_trap_fail") is not None]

    # merit_frac vs aesthetic_vlm Spearman
    spear = None
    pairs = [(r["aes"]["merit_frac"], r["signals"].get("aesthetic_vlm"))
             for r in recs if r.get("aes", {}).get("reliable") and r["aes"]["merit_frac"] is not None
             and r["signals"].get("aesthetic_vlm") is not None]
    if len(pairs) >= 10:
        spear = _spearman([p[0] for p in pairs], [p[1] for p in pairs])

    metrics = {
        "round": rnd, "n": n,
        "IMQ": {
            "first_unreliable_rate": round(1 - len(imq_rel) / len(imq), 4) if imq else None,
            "unreliable_reasons": dict(imq_reason),
            "hard_contradiction_count": hard_contra,
            "per_item_yes": imq_yes,
            "defect_questions_out_of_band": imq_out_of_band,
            "defect_count_dist": {str(k): v for k, v in sorted(defect_dist.items())},
            "verdict_dist": {str(k): v for k, v in imq_verdict.items()},
            "landscape_soft_fail_rate": (round(sum(landscape_softfail) / len(landscape_softfail), 4)
                                         if landscape_softfail else None),
        },
        "aes": {
            "first_unreliable_rate": round(1 - len(aes_rel) / len(aes), 4) if aes else None,
            "unreliable_reasons": dict(aes_reason),
            "per_item_yes": aes_yes,
            "fr_out_of_band": aes_out_of_band,
            "merit_frac_mode_concentration": round(mf_mode_frac, 4) if mf_mode_frac else None,
            "merit_frac_hist": {str(k): v for k, v in sorted(mf_hist.items())},
            "merit_frac_mean": round(sum(mf) / len(mf), 4) if mf else None,
            "spearman_merit_vs_aesthetic_vlm": round(spear, 4) if spear is not None else None,
            "soft_trap_fail_rate": (round(sum(aes_softfail) / len(aes_softfail), 4)
                                    if aes_softfail else None),
            "drop_count_assert": 0,   # 审美永不 drop
        },
    }
    _write_json(os.path.join(rdir, "metrics.json"), metrics)
    print(json.dumps(metrics, ensure_ascii=False, indent=2))
    return metrics


def _spearman(a, b):
    def rank(xs):
        order = sorted(range(len(xs)), key=lambda i: xs[i])
        r = [0.0] * len(xs)
        i = 0
        while i < len(xs):
            j = i
            while j + 1 < len(xs) and xs[order[j + 1]] == xs[order[i]]:
                j += 1
            avg = (i + j) / 2.0
            for k in range(i, j + 1):
                r[order[k]] = avg
            i = j + 1
        return r
    ra, rb = rank(a), rank(b)
    ma, mb = sum(ra) / len(ra), sum(rb) / len(rb)
    num = sum((x - ma) * (y - mb) for x, y in zip(ra, rb))
    da = sum((x - ma) ** 2 for x in ra) ** 0.5
    dbb = sum((y - mb) ** 2 for y in rb) ** 0.5
    return num / (da * dbb) if da and dbb else 0.0


# --------------------------------------------------------------------------- #
def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("select").add_argument("--force", action="store_true")
    rp = sub.add_parser("run-img")
    rp.add_argument("--concurrency", type=int, default=None)
    rp.add_argument("--limit", type=int, default=None)
    rp.add_argument("--all", action="store_true", help="全量: 所有 source img 落库 verdict(非 manifest 200)")
    mp = sub.add_parser("measure")
    mp.add_argument("--round", type=int, default=None)
    sub.add_parser("status")
    args = ap.parse_args()
    if args.cmd == "select":
        select_and_lock(force=args.force)
    elif args.cmd == "run-img":
        if args.all:
            run_images_full(concurrency=args.concurrency, limit=args.limit)
        else:
            run_images_round(concurrency=args.concurrency, limit=args.limit)
    elif args.cmd == "measure":
        measure(rnd=args.round)
    elif args.cmd == "status":
        print(json.dumps(_read_json(STATE, {}), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
