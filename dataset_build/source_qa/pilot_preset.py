"""流程 3 preset QA pilot 驱动（新设计：确定性 6 探针 + 10 题位置码问卷 + qa_clean 投票裁定）。

复用 preset_qa 的真实 LR 渲染 + render_jobs 缓存 + paired_metrics；替换旧 _questionnaire_c(JSON/style)
为 config.PRESET_SYSTEM_PROMPT/PRESET_POS_MAP 的二元位置码问卷，交 qa_clean.clean_preset_probe/
aggregate_preset/map_verdict_preset 裁定。轮 1 真渲染最贵(presetN×6, local 需 LrC client 在线),
渲染命中 render_jobs 缓存 → 轮 2+ 改题只重判不重渲。

子命令:
  signals   对候选池算 6 探针确定性图像信号(mean_luma/highlight_frac/shadow_frac/saturation_mean)
  probes    resolve 6 固定探针(确定性信号选取), 落 manifest
  run       对锁定 preset 渲染 6 探针 + 新问卷 + 清洗 + 投票, 落 round_<r>/preset_raw.jsonl + DB
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed

from . import config as C
from . import db
from . import qa_clean as Q
from . import preset_qa as P

PILOT_DIR = os.path.join(os.path.dirname(__file__), "pilot")
MANIFEST = os.path.join(PILOT_DIR, "manifest.json")
STATE = os.path.join(PILOT_DIR, "state.json")

_LOCAL = threading.local()


def _session():
    if not hasattr(_LOCAL, "s"):
        import requests
        _LOCAL.s = requests.Session()
    return _LOCAL.s


def _read_json(p, d=None):
    return json.load(open(p)) if os.path.exists(p) else d


def _write_json(p, o):
    os.makedirs(os.path.dirname(p), exist_ok=True)
    json.dump(o, open(p, "w"), ensure_ascii=False, indent=2)


# --------------------------------------------------------------------------- #
# 6 探针确定性图像信号 (§3.11)
# --------------------------------------------------------------------------- #
def compute_probe_signals(pool_limit=800, longedge=1024):
    """对高美学候选池算 mean_luma/highlight_frac/shadow_frac/saturation_mean 落 assets。"""
    import numpy as np
    from PIL import Image, ImageOps
    Image.MAX_IMAGE_PIXELS = None
    conn = db.connect()
    rows = conn.execute(
        "SELECT asset_id, path FROM assets WHERE asset_type='image' AND aesthetic IS NOT NULL "
        "AND dup_of IS NULL AND width IS NOT NULL AND mean_luma IS NULL "
        "ORDER BY aesthetic DESC, asset_id LIMIT %s", (pool_limit,)).fetchall()
    print(f"[probe-signals] {len(rows)} images to compute", file=sys.stderr)
    n = 0
    for r in rows:
        try:
            im = Image.open(r["path"]); im.load()
            im = ImageOps.exif_transpose(im) or im
            im = im.convert("RGB")
            im.thumbnail((longedge, longedge))
            arr = np.asarray(im, dtype="float32") / 255.0
            luma = 0.299 * arr[..., 0] + 0.587 * arr[..., 1] + 0.114 * arr[..., 2]
            mx = arr.max(-1); mn = arr.min(-1)
            sat = np.where(mx > 0, (mx - mn) / (mx + 1e-6), 0.0)
            fields = {"mean_luma": float(luma.mean()),
                      "highlight_frac": float((luma >= 0.95).mean()),
                      "shadow_frac": float((luma <= 0.05).mean()),
                      "saturation_mean": float(sat.mean())}
            db.update_asset_fields(conn, r["asset_id"], **fields)
            n += 1
            if n % 100 == 0:
                conn.commit()
                print(f"[probe-signals] {n}/{len(rows)}", file=sys.stderr)
        except Exception as e:  # noqa: BLE001
            print(f"[probe-signals] {r['asset_id']} failed: {e}", file=sys.stderr)
    conn.commit()
    conn.close()
    print(f"[probe-signals] computed {n}")
    return n


def resolve_probes_v2(conn, k=None):
    """据 PRESET_PROBE_SLOTS 确定性信号选 6 探针(去重), 落 manifest['probes']。"""
    k = k or C.PRESET_PROBE_COUNT
    picked, seen = [], set()
    for slot in C.PRESET_PROBE_SLOTS:
        row = conn.execute(
            f"SELECT asset_id, path, is_portrait_pool, max_face_frac, mean_luma, highlight_frac, "
            f"shadow_frac, saturation_mean FROM assets WHERE asset_type='image' AND dup_of IS NULL "
            f"AND mean_luma IS NOT NULL AND ({slot['where']}) "
            f"AND asset_id <> ALL(%s) ORDER BY {slot['order']} LIMIT 1",
            (list(seen) or [""],)).fetchone()
        if row:
            seen.add(row["asset_id"])
            picked.append({"slot": slot["slot"], "desc": slot["desc"], "asset_id": row["asset_id"],
                           "path": row["path"], "is_portrait_pool": row["is_portrait_pool"],
                           "max_face_frac": row["max_face_frac"], "mean_luma": row["mean_luma"],
                           "highlight_frac": row["highlight_frac"], "shadow_frac": row["shadow_frac"],
                           "saturation_mean": row["saturation_mean"]})
    return picked


def _img_color_stats(path, longedge=256):
    """LAB 色彩统计: 4 主色相 chroma 加权 mass + 全局 mean chroma + 低饱和占比。"""
    import numpy as np
    from PIL import Image, ImageOps
    from skimage.color import rgb2lab
    Image.MAX_IMAGE_PIXELS = None
    im = Image.open(path); im.load()
    im = (ImageOps.exif_transpose(im) or im).convert("RGB")
    im.thumbnail((longedge, longedge))
    lab = rgb2lab(np.asarray(im, dtype="float32") / 255.0)
    a, b = lab[..., 1], lab[..., 2]
    Cr = np.sqrt(a * a + b * b)               # chroma (不与 config 别名 C 冲突)
    h = (np.degrees(np.arctan2(b, a)) % 360.0)
    masses = {}
    for slot in C.PRESET_PROBE_SLOTS_LAB:
        if slot["kind"] != "hue":
            continue
        d = np.abs((h - slot["hue_center"] + 180) % 360 - 180)  # 环形角差
        sel = d <= C.PRESET_HUE_HALF
        masses[slot["name"]] = float((Cr * sel).sum() / Cr.size)  # chroma 加权 mass
    return {"mass": masses, "mean_C": float(Cr.mean()),
            "lowsat_frac": float((Cr < 8).mean())}


def _lab_sig(before_path, after_path, longedge=384):
    """单探针 before→after LAB 变换签名(供 tag_preset_function)。"""
    import numpy as np
    from PIL import Image
    from skimage.color import rgb2lab
    Image.MAX_IMAGE_PIXELS = None
    b = Image.open(before_path).convert("RGB"); b.load()
    a = Image.open(after_path).convert("RGB"); a.load()
    if max(b.size) > longedge:
        s = longedge / max(b.size)
        b = b.resize((max(1, int(b.size[0] * s)), max(1, int(b.size[1] * s))))
    a = a.resize(b.size)
    lb = rgb2lab(np.asarray(b, dtype="float32") / 255.0)
    la = rgb2lab(np.asarray(a, dtype="float32") / 255.0)
    Lb, ab, bb = lb[..., 0], lb[..., 1], lb[..., 2]
    La, aa, ba = la[..., 0], la[..., 1], la[..., 2]
    Cb = np.sqrt(ab * ab + bb * bb); Ca = np.sqrt(aa * aa + ba * ba)
    dark = Lb < 25
    shadow_dL = float(La[dark].mean() - Lb[dark].mean()) if dark.any() else float(La.mean() - Lb.mean())
    return {"dL": float(La.mean() - Lb.mean()), "da": float(aa.mean() - ab.mean()),
            "db": float(ba.mean() - bb.mean()), "dC": float(Ca.mean() - Cb.mean()),
            "after_C": float(Ca.mean()), "before_C": float(Cb.mean()),
            "a_before": float(ab.mean()), "b_before": float(bb.mean()),
            "a_after": float(aa.mean()), "b_after": float(ba.mean()),
            "contrast_ratio": float(La.std() / (Lb.std() + 1e-6)), "shadow_dL": shadow_dL}


_PROBE_ORDER = ("red", "yellow", "green", "blue", "skin", "neutral")


_PROBE_CN = {"red": "红", "yellow": "黄", "green": "绿", "blue": "蓝", "skin": "肤色", "neutral": "中性"}


def _probe_table(sigs: list) -> str:
    """逐探针 LAB 客观指标表(纯文本, 替代不准确的 render 拼图)。"""
    import math
    by = {s["name"]: s for s in sigs}
    lines = ["探针 ΔL    Δa*(绿-/品红+) Δb*(冷-/暖+) chroma前→后(相对)  色相旋转"]
    for nm in _PROBE_ORDER:
        s = by.get(nm)
        if not s:
            continue
        h0 = math.degrees(math.atan2(s["b_before"], s["a_before"])) % 360
        h1 = math.degrees(math.atan2(s["b_after"], s["a_after"])) % 360
        dh = (h1 - h0 + 180) % 360 - 180
        rel = (s["after_C"] - s["before_C"]) / (s["before_C"] + 1e-6)
        lines.append(f"{_PROBE_CN[nm]:<3}{s['dL']:+5.1f} {s['da']:+6.1f}       "
                     f"{s['db']:+6.1f}      {s['before_C']:4.0f}→{s['after_C']:<4.0f}({rel:+.0%})  {dh:+5.0f}°")
    return "\n".join(lines)


def _vlm_tag_grounding(det: dict, sigs: list) -> str:
    """逐探针 LAB 指标表 + 聚合判读(全文本 grounding, 不含图)。"""
    m = det["metrics"]
    return (
        "【逐探针 LAB 客观指标】(程序在真实 before/after 上算出, 不可推翻; 不要凭想象)\n"
        + _probe_table(sigs) + "\n\n"
        "【聚合判读】(已据上表算出; 色温/色罩只看『中性』『肤色』行)\n"
        f"- 色温(中性灰 Δb*): {m['db_cast']:+.1f} → {det['temperature']}  (>0暖 <0冷)\n"
        f"- 色罩(中性灰 Δa*): {m['da_cast']:+.1f} → {det['tint']}  (>0品红 <0绿)\n"
        f"- 饱和(色探针相对chroma): {m['rel_chroma']:+.0%} → {det['saturation']}\n"
        f"- 对比(L标准差比): {m['contrast_ratio']:.2f} → {det['contrast']}\n"
        f"- 暗部ΔL: {m['shadow_dL']:+.1f} → {det['tone']}   整体明度ΔL: {m['dL_mean']:+.1f} → {det['exposure']}\n"
        f"- 程序初判 family: {det['grade_family']}   QA: pass_c={det.get('qa_pass_c')} ({det.get('qa_verdict')})\n\n"
        '请输出 JSON: {"name":"≤10字中文look名",'
        '"per_probe":{"红":"该探针走向(用ΔL/Δa/Δb/Δhue/相对chroma说话,如 保饱/降饱-40%/转青+18°/压暗-12)",'
        '"黄":"...","绿":"...","蓝":"...","肤色":"...","中性":"..."},'
        '"caption":"1-2句:这个预设最显著、区别于同类的处理(逐探针细节,不得与其它预设雷同)",'
        '"function":"一句话:适合什么场景",'
        '"axes":{"temperature":"warm|cool|neutral","tint":"magenta|green|neutral",'
        '"saturation":"vibrant|muted|neutral|bw","contrast":"punchy|flat|neutral",'
        '"tone":"lifted|crushed|neutral","family":"..."},'
        '"consistent_with_metrics":true}'
    )


def _vlm_tag_call(det: dict, sigs: list, retries=3):
    """逐探针 LAB 指标表(纯文本) → vLLM 命名。返回 parsed JSON 或 None。"""
    import re, json_repair
    payload = {"model": C.VLLM_MODEL, "max_tokens": 640, "temperature": 0.2,
               "messages": [{"role": "system", "content": C.PRESET_TAG_SYSTEM_PROMPT},
                            {"role": "user", "content": _vlm_tag_grounding(det, sigs)}]}
    if not C.VLLM_ENABLE_THINKING:
        payload["chat_template_kwargs"] = {"enable_thinking": False}
    headers = {"Authorization": f"Bearer {C.VLLM_API_KEY}", "X-vgate-class": "qa-judge"}
    for _ in range(retries):
        try:
            r = _session().post(C.VLLM_BASE_URL + "/chat/completions", json=payload,
                                headers=headers, timeout=120)
            r.raise_for_status()
            txt = r.json()["choices"][0]["message"]["content"]
            txt = re.sub(r"<think>.*?</think>", "", txt, flags=re.S)
            obj = json_repair.loads(txt[txt.find("{"):] if "{" in txt else txt)
            return obj if isinstance(obj, dict) else None
        except Exception:  # noqa: BLE001
            continue
    return None


def tag_round(round_dir=None, vlm=False, limit=None, workers=6):
    """打 tag 轮: 读 LAB 探针 before/after 缓存 + 本轮 QA → 客观功能 tag。
    确定性 axes(测量) 恒算; --vlm 时再让 35B 看 6 拼图+测量做最终命名(name/function)。
    落 round_<r>/preset_tags.jsonl。需先以 LAB 探针跑过 run。"""
    import os
    from collections import defaultdict, Counter
    manifest = _read_json(MANIFEST)
    probes = manifest.get("probes") or []
    pid2name = {p["asset_id"]: p.get("name") for p in probes}
    lab_ids = list(pid2name)
    state = _read_json(STATE, {"round": 1})
    rnd = state.get("round", 1) or 1
    rdir = round_dir or os.path.join(PILOT_DIR, f"round_{rnd}")
    conn = db.connect()
    rows = conn.execute(
        "SELECT DISTINCT ON (asset_id, probe_image) asset_id, probe_image, before_path, after_path "
        "FROM preset_previews WHERE probe_image = ANY(%s) AND after_path IS NOT NULL "
        "ORDER BY asset_id, probe_image, id DESC", (lab_ids,)).fetchall()
    by_asset = defaultdict(list)
    for r in rows:
        by_asset[r["asset_id"]].append(r)
    qa_by = {}
    raw_p = os.path.join(rdir, "preset_raw.jsonl")
    if os.path.exists(raw_p):
        for l in open(raw_p):
            if l.strip():
                rec = json.loads(l)
                qa_by[rec["asset_id"]] = rec.get("verdict")

    # 先算每个 preset 的确定性 tag + 逐探针 sigs(供 vLLM 命名)。
    # _lab_sig 要读 5817×6≈35k 缓存 JPG(CPU), 并行化避免串行预计算拖死(否则 GPU 久等)。
    def _build_item(kv):
        aid, prevs = kv
        sigs = []
        for pr in prevs:
            nm = pid2name.get(pr["probe_image"])
            if not nm or not (pr["before_path"] and pr["after_path"]
                              and os.path.exists(pr["before_path"]) and os.path.exists(pr["after_path"])):
                continue
            try:
                sig = _lab_sig(pr["before_path"], pr["after_path"]); sig["name"] = nm
                sigs.append(sig)
            except Exception:  # noqa: BLE001
                continue
        if len(sigs) < 4:
            return None
        det = Q.tag_preset_function(sigs, qa_by.get(aid))
        det["asset_id"] = aid; det["n_probe_sigs"] = len(sigs)
        return (det, sigs)

    kvs = list(by_asset.items())
    if limit:
        kvs = kvs[:limit]
    from concurrent.futures import ThreadPoolExecutor
    print(f"[tag] building sigs for {len(kvs)} presets (parallel JPG read)...", flush=True)
    with ThreadPoolExecutor(max_workers=max(8, workers)) as ex:
        items = [r for r in ex.map(_build_item, kvs) if r]
    print(f"[tag] sigs built for {len(items)} presets → vLLM naming", flush=True)
    conn.close()

    if vlm:
        from concurrent.futures import ThreadPoolExecutor
        with ThreadPoolExecutor(max_workers=workers) as ex:
            futs = {ex.submit(_vlm_tag_call, det, sigs): det for det, sigs in items}
            for fu in futs:
                det = futs[fu]
                try:
                    res = fu.result()
                except Exception:  # noqa: BLE001
                    res = None
                if res:
                    det["vlm_name"] = res.get("name")
                    det["vlm_per_probe"] = res.get("per_probe")
                    det["vlm_caption"] = res.get("caption")
                    det["vlm_function"] = res.get("function")
                    det["vlm_axes"] = res.get("axes")
                    det["vlm_consistent"] = res.get("consistent_with_metrics")

    out_p = os.path.join(rdir, "preset_tags.jsonl")
    fam_c, name_c = Counter(), Counter()
    with open(out_p, "w") as fh:
        for det, _ in items:
            fh.write(json.dumps(det, ensure_ascii=False) + "\n")
            fam_c[det["grade_family"]] += 1
            if det.get("vlm_name"):
                name_c[det["vlm_name"]] += 1
    print(f"[tag] {len(items)} presets tagged → {out_p}")
    print("[tag] det grade_family:", dict(fam_c.most_common()))
    if vlm:
        print(f"[tag] vLLM named: {sum(name_c.values())}/{len(items)}; top names:",
              dict(name_c.most_common(12)))
    return len(items)


def resolve_probes_lab(conn, pool_limit=1200):
    """4 主色相(chroma 加权)+ 肤色 + 中性 = 6 探针。扫高美学池在线算 LAB 信号选取。"""
    import os
    rows = conn.execute(
        "SELECT asset_id, path, is_portrait_pool, max_face_frac, mean_luma, saturation_mean "
        "FROM assets WHERE asset_type='image' AND aesthetic IS NOT NULL AND dup_of IS NULL "
        "AND mean_luma IS NOT NULL AND mean_luma BETWEEN 0.18 AND 0.82 "
        "ORDER BY aesthetic DESC, asset_id LIMIT %s", (pool_limit,)).fetchall()
    cand = []
    for r in rows:
        if not os.path.exists(r["path"]):
            continue
        try:
            st = _img_color_stats(r["path"])
        except Exception:  # noqa: BLE001
            continue
        cand.append((r, st))
    print(f"[probes-lab] {len(cand)} candidates with color stats", file=sys.stderr)
    picked, seen = [], set()

    def _emit(slot, r, st, extra):
        seen.add(r["asset_id"])
        picked.append({"slot": slot["slot"], "name": slot["name"], "desc": slot["desc"],
                       "asset_id": r["asset_id"], "path": r["path"],
                       "is_portrait_pool": r["is_portrait_pool"], "max_face_frac": r["max_face_frac"],
                       "mean_luma": r["mean_luma"], "mean_C": round(st["mean_C"], 2), **extra})

    # 4 主色相: 该色相 mass 最高且为该图主色相(避免同图占多槽)
    for slot in [s for s in C.PRESET_PROBE_SLOTS_LAB if s["kind"] == "hue"]:
        nm = slot["name"]
        ranked = sorted((c for c in cand if c[0]["asset_id"] not in seen),
                        key=lambda c: c[1]["mass"][nm], reverse=True)
        for r, st in ranked:
            # 要求该色相是这张图的主导色相(mass 最大), 才算"X 主导"
            if max(st["mass"], key=st["mass"].get) == nm and st["mass"][nm] > 0:
                _emit(slot, r, st, {"hue_mass": round(st["mass"][nm], 2),
                                    "hue_center": slot["hue_center"]})
                break
    # 肤色: 人像池 + 中等人脸占比
    skin_slot = next(s for s in C.PRESET_PROBE_SLOTS_LAB if s["name"] == "skin")
    skin = sorted((c for c in cand if c[0]["asset_id"] not in seen and c[0]["is_portrait_pool"]
                   and (c[0]["max_face_frac"] or 0) > 0.04),
                  key=lambda c: -(c[0]["max_face_frac"] or 0))
    if skin:
        _emit(skin_slot, skin[0][0], skin[0][1], {"hue_mass": None})
    # 中性: mean_C 最低(但非纯黑白)
    neu_slot = next(s for s in C.PRESET_PROBE_SLOTS_LAB if s["name"] == "neutral")
    neutral = sorted((c for c in cand if c[0]["asset_id"] not in seen and c[1]["mean_C"] > 3),
                     key=lambda c: c[1]["mean_C"])
    if neutral:
        _emit(neu_slot, neutral[0][0], neutral[0][1], {"hue_mass": None})
    return picked


# --------------------------------------------------------------------------- #
# 新 preset 问卷 (10 题位置码, before+after 双图)
# --------------------------------------------------------------------------- #
def _preset_prompt():
    items = C.render_items_block(C.PRESET_POS_MAP, C.PRESET_QTEXT)
    return C.PRESET_SYSTEM_PROMPT.format(items=items)


def _call_preset(before_uri, after_uri, prompt, temperature=0.1, max_tokens=120, retries=3):
    payload = {"model": C.VLLM_MODEL, "max_tokens": max_tokens, "temperature": temperature,
               "messages": [{"role": "user", "content": [
                   {"type": "image_url", "image_url": {"url": before_uri}},
                   {"type": "image_url", "image_url": {"url": after_uri}},
                   {"type": "text", "text": prompt}]}]}
    if not C.VLLM_ENABLE_THINKING:
        payload["chat_template_kwargs"] = {"enable_thinking": False}
    headers = {"Authorization": f"Bearer {C.VLLM_API_KEY}", "X-vgate-class": "qa-judge"}
    last = ""
    for _ in range(retries):
        try:
            r = _session().post(C.VLLM_BASE_URL + "/chat/completions", json=payload,
                                headers=headers, timeout=120)
            r.raise_for_status()
            return r.json()["choices"][0]["message"]["content"]
        except Exception as e:  # noqa: BLE001
            last = str(e)
    raise RuntimeError(f"preset vllm failed: {last}")


def _settle_probe(before_uri, after_uri, prompt, pm, engine):
    """单探针: 问卷→clean; 按 reason 分流重问(§3.6 settle_probe)。"""
    raw = _call_preset(before_uri, after_uri, prompt, temperature=0.1)
    out = Q.clean_preset_probe(raw, pm, engine)
    out["reask"] = 0
    if not out["reliable"] and C.PRESET_REASK_MAX >= 1:
        base = (out["reason"] or "").split(":")[0]
        if base in ("parse", "anchor"):
            raw2 = _call_preset(before_uri, after_uri, prompt, temperature=0.0)
        else:                              # trap/contradiction → 轻提温重问
            raw2 = _call_preset(before_uri, after_uri, prompt, temperature=0.3)
        out2 = Q.clean_preset_probe(raw2, pm, engine)
        out2["reask"] = 1
        out2["raw_first"] = raw
        out = out2
    return out


# --------------------------------------------------------------------------- #
# run
# --------------------------------------------------------------------------- #
def run(limit=None, workers=3, only_engine=None, all_presets=False):
    from PIL import Image
    Image.MAX_IMAGE_PIXELS = None
    from . import paired_metrics
    from dataset_build.core import build_lr_client
    manifest = _read_json(MANIFEST)
    state = _read_json(STATE, {"round": 1})
    rnd = state.get("round", 1) or 1
    probes = manifest.get("probes")
    if not probes:
        print("[preset] manifest 无 probes, 先 probes", file=sys.stderr); return
    lr = build_lr_client()
    P._parser()
    conn = db.connect()
    run_id = db.start_run(conn, "pilot_preset", {"limit": limit, "round": rnd})
    os.makedirs(C.RENDER_STAGE, exist_ok=True)

    # stage probe images (1600px before)
    probe_pils = {}
    for p in probes:
        try:
            sp = P._stage_probe(p)
            im = Image.open(sp); im.load(); im = im.convert("RGB")
            probe_pils[p["asset_id"]] = (p, sp, im)
        except Exception as e:  # noqa: BLE001
            print(f"[preset] probe {p['asset_id']} stage failed: {e}", file=sys.stderr)
    print(f"[preset] {len(probe_pils)} probes staged; LR health={lr.health()}", file=sys.stderr)

    if all_presets:
        # 全量: 所有 stage1-pass 非局部 preset + 早期(错端口/路径失效/farm 离线)渲染失败的
        # preset_render_failed(farm+路径现已正常, 重渲会成功)。local-mask 需带 mask 路径另算。
        presets = [{"asset_id": r["asset_id"], "kind": r["kind"]} for r in conn.execute(
            "SELECT asset_id, kind FROM assets WHERE asset_type='preset' "
            "AND status IN ('preset_meta_pass','preset_render_failed') "
            "AND dup_of IS NULL ORDER BY asset_id").fetchall()]
    else:
        presets = manifest["presets"]
    if only_engine == "param":
        presets = [p for p in presets if p["kind"] == "param"]
    elif only_engine == "lut":
        presets = [p for p in presets if p["kind"] == "lut"]
    if limit:
        presets = presets[:limit]

    prompt = _preset_prompt()
    rdir = os.path.join(PILOT_DIR, f"round_{rnd}")
    os.makedirs(rdir, exist_ok=True)
    outp = os.path.join(rdir, "preset_raw.jsonl")
    # resume: 跳过本轮已写出的 preset(全量长跑可断点续)
    done_ids = set()
    if all_presets and os.path.exists(outp):
        for l in open(outp):
            try:
                done_ids.add(json.loads(l)["asset_id"])
            except Exception:  # noqa: BLE001
                pass
        presets = [p for p in presets if p["asset_id"] not in done_ids]
        print(f"[preset] resume: skip {len(done_ids)} done, {len(presets)} remain", file=sys.stderr)
    # fetch full preset rows (path/fmt/style/...)
    ids = [p["asset_id"] for p in presets]
    prow = {r["asset_id"]: r for r in conn.execute(
        "SELECT asset_id, path, fmt, kind, style, scene_affinity, has_local_mask, has_ai_mask, "
        "preset_content_hash FROM assets WHERE asset_id = ANY(%s)", (ids,)).fetchall()}
    db_lock = threading.Lock()
    gpu_lock = threading.Lock()
    fh = open(outp, "a")
    counts = {"ok": 0, "near_noop": 0, "insufficient": 0, "skip": 0, "all_pass": 0}
    cl = threading.Lock()

    def _bump(k):
        with cl:
            counts[k] = counts.get(k, 0) + 1

    def _process(pmeta):
        aid = pmeta["asset_id"]
        r = prow.get(aid)
        if not r:
            _bump("skip"); return
        engine = "lrc" if r["kind"] == "param" else "lut_trilinear"
        eng_thr = "param" if engine == "lrc" else "lut"      # qa_clean engine 口径
        region_local = bool(r["has_local_mask"])
        content_hash = r["preset_content_hash"] or aid
        cube = None
        if engine == "lut_trilinear":
            try:
                cube = P._parser().load_cube(r["path"])
            except Exception as e:  # noqa: BLE001
                _bump("skip"); return
        outdir = os.path.join(C.PREVIEW_DIR, aid)
        os.makedirs(outdir, exist_ok=True)

        previews, probe_results = [], []
        for pid, (p, sp, before_im) in probe_pils.items():
            with db_lock:
                cached = db.cached_render(conn, content_hash, pid, engine)
            after_path = job_id = None
            if cached and cached["after_jpg_path"] and os.path.exists(cached["after_jpg_path"]):
                after_path, job_id = cached["after_jpg_path"], cached["job_id"]
            else:
                with db_lock:
                    job_id = P._job_id_for(conn, content_hash, pid, engine, aid, region_local,
                                           r["fmt"], r["path"], sp, run_id,
                                           ai_mask=bool(r["has_ai_mask"]))
                    conn.commit()
                if engine == "lrc":
                    res = lr.render(r["path"], r["fmt"], sp)
                    ok = bool(res and res.get("ok"))
                    after_path = res["after_path"] if ok else None
                    with db_lock:
                        db.mark_job(conn, job_id, "done" if ok else "error",
                                    after_jpg_path=after_path,
                                    engine_version=(res or {}).get("engine_version"), node="lrc",
                                    error=None if ok else json.dumps(res or {}, ensure_ascii=False)[:800])
                        conn.commit()
                else:
                    try:
                        after_im0 = P._apply_cube(before_im, cube)
                        after_path = os.path.join(outdir, f"{pid}_after.jpg")
                        after_im0.save(after_path, "JPEG", quality=95)
                        with db_lock:
                            db.mark_job(conn, job_id, "done", after_jpg_path=after_path, node="numpy")
                            conn.commit()
                    except Exception as e:  # noqa: BLE001
                        with db_lock:
                            db.mark_job(conn, job_id, "error", error=str(e)[:200]); conn.commit()
            if not after_path or not os.path.exists(after_path):
                continue
            try:
                after_im = Image.open(after_path); after_im.load(); after_im = after_im.convert("RGB")
            except Exception:
                continue
            bpath = os.path.join(outdir, f"{pid}_before.jpg")
            if not os.path.exists(bpath):
                before_im.save(bpath, "JPEG", quality=92)
            pm = paired_metrics.paired(before_im, after_im)
            # 新问卷判级
            cln = _settle_probe(P._img_uri(before_im), P._img_uri(after_im), prompt, pm, eng_thr)
            cln["paired_metrics"] = pm
            probe_results.append(cln)
            previews.append(pm)
            with db_lock:
                db.add_preset_preview(conn, aid, pid, bpath, after_path, {}, pm, engine,
                                      int(region_local), job_id, run_id, ai_mask=int(bool(r["has_ai_mask"])))
                # 逐探针清洗结果 + 逐题落 llm_qa(probe_id 区分)
                P_qid = {pid_: bit for pid_, bit in (cln.get("d") or {}).items()}
                if cln.get("d"):
                    items = {aid_: {"answer": b, "raw": cln.get("raw")} for aid_, b in cln["d"].items()}
                    db.add_qa(conn, aid, "preset", items, model=C.VLLM_MODEL, run_id=run_id, probe_id=pid)
                conn.execute("UPDATE preset_previews SET probe_reliable=%s, probe_reason=%s, trap_fail=%s "
                             "WHERE asset_id=%s AND probe_image=%s AND job_id=%s",
                             (1 if cln["reliable"] else 0, cln["reason"], cln.get("trap_fail"),
                              aid, pid, job_id))
                conn.commit()

        if not previews:
            with db_lock:
                new_status = "preset_needs_local_render" if region_local else "preset_render_failed"
                db.update_asset_fields(conn, aid, status=new_status,
                                       auto_verdict="needs_local_render" if region_local else "review")
                conn.commit()
            _bump("skip"); return

        verdict = Q.map_verdict_preset(previews, probe_results, eng_thr)
        verdict["total_probe_count"] = len(probe_results)
        with db_lock:
            db.add_preset_qa_run(conn, aid, verdict, run_id=run_id)
            db.update_asset_fields(conn, aid, pass_c=verdict["pass_c"],
                                   auto_verdict=verdict["auto_verdict"])
            conn.commit()
        rec = {"asset_id": aid, "kind": r["kind"], "engine": eng_thr,
               "has_local_mask": region_local, "verdict": verdict,
               "probes": [{"pid": pr_pid, "reliable": pr["reliable"], "reason": pr["reason"],
                           "trap_fail": pr.get("trap_fail"), "d": pr.get("d"),
                           "pm": {k: pr["paired_metrics"].get(k) for k in
                                  ("delta_e2000_mean", "ssim", "noop_score", "hist_emd_ab")},
                           "reask": pr.get("reask", 0), "raw": pr.get("raw")}
                          for pr_pid, pr in zip(probe_pils.keys(), probe_results)]}
        with cl:
            fh.write(json.dumps(rec, ensure_ascii=False) + "\n"); fh.flush()
        vr = verdict["verdict_reason"]
        _bump("near_noop" if vr == "near_noop" else
              "insufficient" if vr == "insufficient_reliable_probes" else
              "all_pass" if vr == "all_pass" else "ok")

    with ThreadPoolExecutor(max_workers=workers) as ex:
        futs = [ex.submit(_process, p) for p in presets]
        for j, f in enumerate(as_completed(futs)):
            try:
                f.result()
            except Exception as e:  # noqa: BLE001
                print(f"[preset] worker error: {e}", file=sys.stderr)
            if (j + 1) % 10 == 0:
                print(f"[preset] {j+1}/{len(presets)} {counts}", file=sys.stderr)
    fh.close()
    db.finish_run(conn, run_id, counts)
    conn.close()
    print(json.dumps(counts))
    return counts


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    sp = sub.add_parser("signals"); sp.add_argument("--pool", type=int, default=800)
    sub.add_parser("probes")
    sub.add_parser("probes-lab")
    tp = sub.add_parser("tag")
    tp.add_argument("--vlm", action="store_true", help="vLLM 看 6 拼图+测量做最终命名")
    tp.add_argument("--limit", type=int, default=None)
    tp.add_argument("--workers", type=int, default=6)
    rp = sub.add_parser("run")
    rp.add_argument("--limit", type=int, default=None)
    rp.add_argument("--workers", type=int, default=3)
    rp.add_argument("--engine", choices=["param", "lut"], default=None)
    rp.add_argument("--all", action="store_true", help="全量: 所有 stage1-pass 非局部 preset(非 manifest)")
    args = ap.parse_args()
    if args.cmd == "signals":
        compute_probe_signals(pool_limit=args.pool)
    elif args.cmd == "probes":
        conn = db.connect()
        probes = resolve_probes_v2(conn)
        man = _read_json(MANIFEST)
        man["probes"] = probes
        _write_json(MANIFEST, man)
        for p in probes:
            print(f"slot{p['slot']} {p['desc']}: {p['asset_id']} "
                  f"luma={p['mean_luma']:.2f} hl={p['highlight_frac']:.3f} "
                  f"sh={p['shadow_frac']:.3f} sat={p['saturation_mean']:.2f} "
                  f"pp={p['is_portrait_pool']} face={p['max_face_frac']}")
    elif args.cmd == "probes-lab":
        conn = db.connect()
        probes = resolve_probes_lab(conn)
        man = _read_json(MANIFEST)
        if "probes_luma_v1" not in man and man.get("probes"):
            man["probes_luma_v1"] = man["probes"]   # 备份旧 luma 探针集
        man["probes"] = probes
        man["probe_set"] = "lab_v2"
        _write_json(MANIFEST, man)
        for p in probes:
            print(f"slot{p['slot']} {p['name']:7s} {p['desc']}: {p['asset_id']} "
                  f"luma={p['mean_luma']:.2f} meanC={p['mean_C']:.1f} "
                  f"hue_mass={p.get('hue_mass')} pp={p['is_portrait_pool']} face={p['max_face_frac']}")
    elif args.cmd == "tag":
        tag_round(vlm=args.vlm, limit=args.limit, workers=args.workers)
    elif args.cmd == "run":
        run(limit=args.limit, workers=args.workers, only_engine=args.engine, all_presets=args.all)


if __name__ == "__main__":
    main()
