"""LLM QA runner for input images: redesigned questionnaire A (validity, binary
hard-fail) + B (suitability: binary safety + yes/no/uncertain scene + graded
0-3 quality/compression/subject/face) + a one-line caption, in a SINGLE
multimodal call to the running vLLM Qwen3.5-35B (:8002), thinking OFF.

Design fixes (vs the audited version):
  * graded B so the gate gets magnitude, not a brittle all-yes binary
  * NULL is NOT collapsed to fail: a parse-miss -> pass=NULL -> gate routes to
    review (fail-closed), distinct from an explicit 'no'
  * yes/no/uncertain scene: a label mismatch the model is unsure about -> review,
    not an auto-drop
  * json_repair instead of a brittle regex, max_tokens raised, rationale only on
    no/uncertain (saves tokens, focuses signal)
  * cheap deterministic TECH-GATE before the 35B: sub-resolution / sub-musiq /
    sub-sharpness images are auto-dropped without spending a multimodal call

Run: python -m dataset_build.source_qa.llm_qa [--limit N] [--corpus C] [--concurrency K]
"""
from __future__ import annotations

import argparse
import base64
import io
import json
import sys
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Dict, Optional

import json_repair

from . import config, db

_A = config.QUESTIONNAIRE_A
_B = config.QUESTIONNAIRE_B
_G = config.GATE
_QP = config.QA_PASS


# --------------------------------------------------------------------------- #
# image -> data URI
# --------------------------------------------------------------------------- #
def _img_data_uri(path: str, longedge: int = None) -> str:
    from PIL import Image
    Image.MAX_IMAGE_PIXELS = None
    longedge = longedge or config.VLLM_IMAGE_LONGEDGE
    im = Image.open(path); im.load(); im = im.convert("RGB")
    w, h = im.size
    if max(w, h) > longedge:
        s = longedge / max(w, h)
        im = im.resize((max(1, int(w * s)), max(1, int(h * s))))
    buf = io.BytesIO(); im.save(buf, "JPEG", quality=88)
    return "data:image/jpeg;base64," + base64.b64encode(buf.getvalue()).decode()


# --------------------------------------------------------------------------- #
# prompt
# --------------------------------------------------------------------------- #
_TYPE_HINT = {
    "yn": "回答 true/false",
    "yn3": "回答 \"yes\"/\"no\"/\"uncertain\"（不确定时答 uncertain，不要乱猜）",
    "grade": "回答整数 0/1/2/3",
}


def _items_block(items: dict, scene: str, is_portrait: bool) -> str:
    lines = []
    for k, spec in items.items():
        if spec.get("portrait_only") and not is_portrait:
            continue
        q = spec["q"].replace("{scene}", scene or "any")
        lines.append(f'  {k} ({_TYPE_HINT[spec["type"]]}): {q}')
    return "\n".join(lines)


def _build_prompt(asset) -> str:
    scene = asset["scene"] or "any"
    is_portrait = bool(asset["is_portrait_pool"])
    pool = "（portrait-pool）" if is_portrait else ""
    a_block = _items_block(_A["items"], scene, is_portrait)
    b_block = _items_block(_B["items"], scene, is_portrait)
    return (
        "你是修图训练数据的严格质检员。本图片是某『修图 before 源』候选，"
        f"被分配的场景标签 = `{scene}` {pool}。\n"
        "重要前提：该数据集会在【高质量图】上做退化再让模型还原，因此【已精修的成品也是优质源】，"
        "不要因为图片『看起来已编辑/已调色』而扣画质分。\n\n"
        f"问卷A（{_A['title']}，全部为 true 才通过）：\n{a_block}\n\n"
        f"问卷B（{_B['title']}）：\n{b_block}\n\n"
        "再给一句中文 caption 描述画面内容。\n"
        "理由 why 只在答案为 false / no / uncertain / 低分(0或1) 时给出一句话；其余可留空。\n"
        "只输出如下 JSON（不要任何多余文字/markdown）：\n"
        '{"caption":"...",'
        '"A":{"A1":{"answer":true,"why":""},"A2":{...},"A3":{...}},'
        '"B":{"B_safe":{"answer":true,"why":""},"B_scene":{"answer":"yes","why":""},'
        '"B_quality":{"answer":3,"why":""},"B_comp":{"answer":3,"why":""},'
        '"B_subject":{"answer":2,"why":""}' + (',"B_face":{"answer":3,"why":""}' if is_portrait else "") + "}}"
    )


# --------------------------------------------------------------------------- #
# vLLM call
# --------------------------------------------------------------------------- #
_LOCAL = threading.local()


def _session():
    if not hasattr(_LOCAL, "s"):
        import requests
        _LOCAL.s = requests.Session()
    return _LOCAL.s


def _call_vllm(data_uri: str, prompt: str, max_retries: int = 3) -> str:
    payload = {
        "model": config.VLLM_MODEL,
        "messages": [{"role": "user", "content": [
            {"type": "image_url", "image_url": {"url": data_uri}},
            {"type": "text", "text": prompt},
        ]}],
        "max_tokens": config.VLLM_MAX_TOKENS,
        "temperature": 0.1,
    }
    if not config.VLLM_ENABLE_THINKING:
        payload["chat_template_kwargs"] = {"enable_thinking": False}
    headers = {"Authorization": f"Bearer {config.VLLM_API_KEY}",
               "X-vgate-class": "qa-judge"}  # vGate priority (P1); ignored by plain vLLM
    last = ""
    for _ in range(max_retries):
        try:
            r = _session().post(config.VLLM_BASE_URL + "/chat/completions",
                                json=payload, headers=headers, timeout=120)
            r.raise_for_status()
            return r.json()["choices"][0]["message"]["content"]
        except Exception as e:
            last = str(e)
    raise RuntimeError(f"vllm call failed: {last}")


# --------------------------------------------------------------------------- #
# parsing + coercion
# --------------------------------------------------------------------------- #
def _parse(text: str) -> Optional[dict]:
    """Robust JSON extraction with json_repair (handles trailing CoT, partial
    objects, smart quotes, missing commas)."""
    t = text.strip()
    if t.startswith("```"):
        t = t.strip("`")
        t = t[t.find("{"):] if "{" in t else t
    try:
        obj = json_repair.loads(t)
        return obj if isinstance(obj, dict) else None
    except Exception:
        return None


def _yn(v) -> Optional[int]:
    if isinstance(v, bool):
        return 1 if v else 0
    if isinstance(v, (int, float)):
        return 1 if v else 0
    if isinstance(v, str):
        s = v.strip().lower()
        if s in ("true", "yes", "y", "1", "pass", "是"):
            return 1
        if s in ("false", "no", "n", "0", "fail", "否"):
            return 0
    return None


def _yn3(v) -> Optional[str]:
    """-> 'yes' | 'no' | 'uncertain' | None(missing/unparsed)."""
    if isinstance(v, bool):
        return "yes" if v else "no"
    if isinstance(v, str):
        s = v.strip().lower()
        if s in ("yes", "true", "y", "是", "一致"):
            return "yes"
        if s in ("no", "false", "n", "否", "不一致"):
            return "no"
        if s in ("uncertain", "unsure", "maybe", "不确定", "不知道"):
            return "uncertain"
    return None


def _grade(v) -> Optional[int]:
    try:
        g = int(round(float(v)))
        return max(0, min(3, g))
    except (TypeError, ValueError):
        return None


def _coerce(item_key: str, spec: dict, parsed_block: dict):
    """Returns (stored_answer, good_bool, present_bool, raw_value)."""
    d = (parsed_block or {}).get(item_key) or {}
    raw = d.get("answer")
    typ = spec["type"]
    if typ == "yn":
        a = _yn(raw)
        return a, (a == 1), (a is not None), raw
    if typ == "yn3":
        a = _yn3(raw)
        present = a is not None
        stored = 1 if a == "yes" else (0 if a == "no" else None)  # uncertain -> NULL
        return a, (a == "yes"), present, raw          # note: a is the str here
    g = _grade(raw)
    return g, (g is not None and g >= _grade_min(item_key)), (g is not None), raw


_GRADE_MIN = {"B_quality": "b_quality_min", "B_comp": "b_comp_min",
              "B_subject": "b_subject_min", "B_face": "b_face_min"}


def _grade_min(item_key: str) -> int:
    return _QP.get(_GRADE_MIN.get(item_key, ""), 0)


# --------------------------------------------------------------------------- #
# one image
# --------------------------------------------------------------------------- #
def _process_one(asset) -> dict:
    aid = asset["asset_id"]
    is_portrait = bool(asset["is_portrait_pool"])
    try:
        uri = _img_data_uri(asset["path"])
    except Exception as e:
        return {"asset_id": aid, "error": f"decode: {e}"}
    try:
        raw = _call_vllm(uri, _build_prompt(asset))
    except Exception as e:
        return {"asset_id": aid, "error": f"llm: {e}"}
    parsed = _parse(raw)
    if not parsed:
        return {"asset_id": aid, "error": "parse", "raw": raw[:500]}

    A, B = parsed.get("A") or {}, parsed.get("B") or {}
    items_a, items_b = {}, {}
    n_answered = n_yes = 0

    # --- questionnaire A (validity, all yn) ---
    a_vals = []
    for k, spec in _A["items"].items():
        ans, good, present, _ = _coerce(k, spec, A)
        items_a[k] = {"answer": ans, "rationale": str((A.get(k) or {}).get("why", ""))[:300]}
        a_vals.append((ans, present))
        n_answered += int(present); n_yes += int(good)
    if any(not p for _, p in a_vals):
        pass_a = None
    elif all(v == 1 for v, _ in a_vals):
        pass_a = 1
    else:
        pass_a = 0

    # --- questionnaire B (mixed) ---
    grades = {}
    safe = scene = None
    for k, spec in _B["items"].items():
        if spec.get("portrait_only") and not is_portrait:
            continue
        ans, good, present, _ = _coerce(k, spec, B)
        n_answered += int(present); n_yes += int(good)
        why = str((B.get(k) or {}).get("why", ""))[:300]
        if spec["type"] == "yn3":
            stored = 1 if ans == "yes" else (0 if ans == "no" else None)
            items_b[k] = {"answer": stored, "rationale": (why or (ans or ""))}
            scene = ans
        elif spec["type"] == "yn":
            items_b[k] = {"answer": ans, "rationale": why}
            safe = ans
        else:
            items_b[k] = {"answer": ans, "rationale": why}
            grades[k] = ans

    bq, bc, bs = grades.get("B_quality"), grades.get("B_comp"), grades.get("B_subject")
    bf = grades.get("B_face")
    hard_fail = (
        safe == 0 or scene == "no"
        or (bq is not None and bq < _QP["b_quality_min"])
        or (bc is not None and bc < _QP["b_comp_min"])
        or (bs is not None and bs < _QP["b_subject_min"])
        or (is_portrait and bf is not None and bf < _QP["b_face_min"])
    )
    required_present = (safe is not None and scene is not None
                        and bq is not None and bc is not None and bs is not None
                        and (bf is not None or not is_portrait))
    if hard_fail:
        pass_b = 0
    elif scene == "uncertain" or not required_present:
        pass_b = None          # -> review (missing signal or unsure scene)
    else:
        pass_b = 1

    caption = str(parsed.get("caption", ""))[:500]
    return {
        "asset_id": aid, "items_a": items_a, "items_b": items_b, "caption": caption,
        "pass_a": pass_a, "pass_b": pass_b,
        "b_quality": bq, "b_comp": bc, "b_subject": bs, "b_face": bf,
        "n_answered": n_answered, "n_yes": n_yes, "raw": raw,
    }


# --------------------------------------------------------------------------- #
# cheap deterministic tech-gate (runs BEFORE the 35B; closes FLOW-3 + QA-1)
# --------------------------------------------------------------------------- #
def _tech_gate(conn, corpus: Optional[str], run_id: str) -> int:
    """Auto-drop images that already failed a hard technical floor in IQA, so the
    expensive multimodal judge never sees them. Non-destructive (auto_verdict)."""
    where = ["asset_type='image'", "musiq IS NOT NULL", "auto_verdict IS DISTINCT FROM 'drop'",
             "(megapixels < %s OR musiq < %s OR sharpness < %s)"]
    params = [_G["min_megapixels"], _G["tech_musiq_drop_below"], _G["tech_sharp_drop_below"]]
    if corpus:
        where.append("corpus=%s"); params.append(corpus)
    rows = conn.execute(
        f"SELECT asset_id, megapixels, musiq, sharpness FROM assets WHERE {' AND '.join(where)}",
        params).fetchall()
    for r in rows:
        reason = ("megapixels<%.2f" % _G["min_megapixels"] if (r["megapixels"] or 1e9) < _G["min_megapixels"]
                  else "musiq<%.0f" % _G["tech_musiq_drop_below"] if (r["musiq"] or 1e9) < _G["tech_musiq_drop_below"]
                  else "sharpness<%.0f" % _G["tech_sharp_drop_below"])
        db.update_asset_fields(conn, r["asset_id"], auto_verdict="drop", status="auto_fail")
        db.log_event(conn, r["asset_id"], "tech_gate", "ok", {"verdict": "drop", "reason": reason}, run_id)
    conn.commit()
    return len(rows)


# --------------------------------------------------------------------------- #
# run
# --------------------------------------------------------------------------- #
def run(limit: Optional[int] = None, corpus: Optional[str] = None,
        concurrency: Optional[int] = None) -> dict:
    concurrency = concurrency or config.VLLM_CONCURRENCY
    conn = db.connect()
    run_id = db.start_run(conn, "llm_qa", {"corpus": corpus, "concurrency": concurrency})
    n_tech = _tech_gate(conn, corpus, run_id)
    print(f"[llm_qa] tech-gate auto-dropped {n_tech} images before the judge", file=sys.stderr)

    # Only judge images that passed IQA's hard floors and have no A row yet.
    where = ["asset_type='image'",
             "dup_of IS NULL",                      # heads only; siblings inherit at apply
             "musiq IS NOT NULL",
             "auto_verdict IS DISTINCT FROM 'drop'",
             "NOT EXISTS (SELECT 1 FROM llm_qa q WHERE q.asset_id=assets.asset_id AND q.questionnaire='A')"]
    params: list = []
    if corpus:
        where.append("corpus=?"); params.append(corpus)
    sql = f"SELECT asset_id, path, scene, is_portrait_pool FROM assets WHERE {' AND '.join(where)}"
    if limit:
        sql += f" LIMIT {int(limit)}"
    todo = conn.execute(sql, params).fetchall()
    print(f"[llm_qa] {len(todo)} images to QA, concurrency={concurrency}", file=sys.stderr)

    n_ok = n_err = n_drop = 0
    write_lock = threading.Lock()

    def _persist(res, aid):
        if res.get("error"):
            db.log_event(conn, aid, "llm_qa", "error", {"err": res["error"]}, run_id)
        else:
            db.add_qa(conn, aid, "A", res["items_a"], model=config.VLLM_MODEL, run_id=run_id)
            db.add_qa(conn, aid, "B", res["items_b"], model=config.VLLM_MODEL, run_id=run_id)
            db.add_qa(conn, aid, "caption", {"caption": {"answer": None, "raw": res["caption"]}},
                      model=config.VLLM_MODEL, run_id=run_id)
            db.update_asset_fields(
                conn, aid, pass_a=res["pass_a"], pass_b=res["pass_b"],
                b_quality=res["b_quality"], b_comp=res["b_comp"], b_subject=res["b_subject"],
                b_face=res["b_face"], n_answered=res["n_answered"], n_yes=res["n_yes"],
                status="qa_done")
            db.log_event(conn, aid, "llm_qa", "ok",
                         {"pass_a": res["pass_a"], "pass_b": res["pass_b"]}, run_id)

    with ThreadPoolExecutor(max_workers=concurrency) as ex:
        futs = {ex.submit(_process_one, a): a["asset_id"] for a in todo}
        for j, fut in enumerate(as_completed(futs)):
            res = fut.result()
            aid = res["asset_id"]
            try:
                with write_lock:
                    db.write_retry(conn, lambda r=res, a=aid: _persist(r, a))
                if res.get("error"):
                    n_err += 1
                else:
                    n_ok += 1
            except Exception as e:
                n_drop += 1
                print(f"[llm_qa] persist failed for {aid}, skipping (resume redoes): {e}", file=sys.stderr)
            if (j + 1) % 200 == 0:
                print(f"[llm_qa] {j+1}/{len(todo)} ok={n_ok} err={n_err} dropped={n_drop}", file=sys.stderr)
    try:
        conn.commit()
    except Exception:
        pass
    db.finish_run(conn, run_id, {"ok": n_ok, "err": n_err, "dropped": n_drop, "tech_dropped": n_tech})
    conn.close()
    return {"ok": n_ok, "err": n_err, "dropped": n_drop, "tech_dropped": n_tech, "run_id": run_id}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--corpus", default=None)
    ap.add_argument("--concurrency", type=int, default=None)
    args = ap.parse_args()
    print(json.dumps(run(limit=args.limit, corpus=args.corpus, concurrency=args.concurrency)))


if __name__ == "__main__":
    main()
