"""Source 图 caption + 主体标注：对 keep 池每张源图做一次多模态调用（Qwen3.5-35B,
thinking OFF），产出修图导向中文 caption + 可分割主体列表，入库 source_captions。

主体 en 名直接作为 SAM3 text-prompt（下游 sam3_subjects.py 消费），因此提示词强约束：
小写具体名词短语、可像素级分割、显著度降序、恰好一个 main。

Run: python -m dataset_build.source_qa.caption_subjects [--limit N] [--concurrency K]
     [--verdict keep,review] [--redo]
"""
from __future__ import annotations

import argparse
import json
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed

import json_repair

from . import config, db
from .llm_qa import _call_vllm, _img_data_uri

_PROMPT = (
    "你是修图训练数据的图像标注员。请对这张『修图 before 源图』做两件事：\n\n"
    "1. caption：一到两句中文描述，依次覆盖：主体是什么、所处场景、光线与色彩氛围"
    "（如逆光/柔光/冷调/暖调/高对比）。写给修图师看，不要文学化修辞。\n\n"
    "2. subjects：图中值得单独局部调整的主体/区域，1-4 个，按视觉显著度从高到低排列。每个主体给出：\n"
    "  - en: 英文名，将直接用作分割模型的文本提示，必须是小写的具体名词短语（1-3 个词），"
    "指代画面中一块可框选的像素区域。例：\"woman\", \"sky\", \"wooden boat\", \"plate of food\"。\n"
    "    禁止：抽象概念（lighting/mood/composition/colors）、方位描述（left side/background）、"
    "整幅画面（image/photo/scene/foreground）。\n"
    "  - cn: 对应中文名。\n"
    "  - main: 是否画面第一主体，恰好一个主体为 true。\n"
    "  - area: 该主体约占画面面积的比例，0 到 1 的小数。\n"
    "无明确主体的纯风景图，用最主要的语义区域作主体（如 sky/mountain/water/forest）。\n"
    "同类多个个体合并为一个复数主体（如三只海鸥 -> \"seagulls\"），不要逐个列。\n\n"
    "只输出如下 JSON（不要任何多余文字/markdown）：\n"
    '{"caption":"...","subjects":[{"en":"...","cn":"...","main":true,"area":0.3}]}'
)

_BAD_EN = {"image", "photo", "picture", "scene", "background", "foreground",
           "lighting", "mood", "composition", "colors", "color", "atmosphere"}


def _parse(raw: str) -> dict | None:
    d = json_repair.loads(raw)
    if not isinstance(d, dict):
        return None
    caption = str(d.get("caption", "")).strip()[:500]
    subs = []
    for s in (d.get("subjects") or [])[:4]:
        if not isinstance(s, dict):
            continue
        en = str(s.get("en", "")).strip().lower()
        if not en or en in _BAD_EN or len(en.split()) > 4:
            continue
        try:
            area = max(0.0, min(1.0, float(s.get("area", 0.0))))
        except (TypeError, ValueError):
            area = 0.0
        subs.append({"en": en, "cn": str(s.get("cn", "")).strip()[:40],
                     "main": bool(s.get("main")), "area": area})
    if not caption or not subs:
        return None
    if sum(s["main"] for s in subs) != 1:      # 恰好一个 main：修为首个
        for i, s in enumerate(subs):
            s["main"] = i == 0
    return {"caption": caption, "subjects": subs}


def _one(asset) -> dict | None:
    raw = _call_vllm(_img_data_uri(asset["path"]), _PROMPT)
    return _parse(raw)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--concurrency", type=int, default=config.VLLM_CONCURRENCY)
    ap.add_argument("--verdict", default="keep", help="comma list of auto_verdict pools")
    ap.add_argument("--redo", action="store_true", help="ignore existing source_captions rows")
    a = ap.parse_args()

    db.init_db()
    conn = db.connect()
    verd = tuple(v.strip() for v in a.verdict.split(",") if v.strip())
    rows = conn.execute(
        "SELECT asset_id, path FROM assets WHERE asset_type='image' AND dup_of IS NULL "
        f"AND auto_verdict IN ({','.join(['%s'] * len(verd))}) ORDER BY asset_id", verd).fetchall()
    if not a.redo:
        done = {r[0] for r in conn.execute("SELECT asset_id FROM source_captions").fetchall()}
        rows = [r for r in rows if r["asset_id"] not in done]
    import os
    rows = [r for r in rows if os.path.exists(r["path"])]
    if a.limit:
        rows = rows[:a.limit]
    run_id = db.start_run(conn, "caption_subjects", {"verdict": verd, "n": len(rows)})
    print(f"[caption] {len(rows)} images to caption (run {run_id})", flush=True)

    ok = fail = 0
    with ThreadPoolExecutor(max_workers=a.concurrency) as ex:
        futs = {ex.submit(_one, r): r for r in rows}
        for fut in as_completed(futs):
            r = futs[fut]
            try:
                res = fut.result()
            except Exception as e:  # noqa: BLE001
                res = None
                print(f"[caption] {r['asset_id']}: {e}", file=sys.stderr)
            if not res:
                fail += 1
                db.log_event(conn, r["asset_id"], "caption_subjects", "error", run_id=run_id)
                conn.commit()
                continue
            main_en = next(s["en"] for s in res["subjects"] if s["main"])
            conn.execute(
                "INSERT INTO source_captions(asset_id, caption, subjects, main_subject, model, run_id, updated_at) "
                "VALUES(%s,%s,%s,%s,%s,%s,now()) ON CONFLICT(asset_id) DO UPDATE SET "
                "caption=EXCLUDED.caption, subjects=EXCLUDED.subjects, main_subject=EXCLUDED.main_subject, "
                "model=EXCLUDED.model, run_id=EXCLUDED.run_id, updated_at=now()",
                (r["asset_id"], res["caption"], json.dumps(res["subjects"], ensure_ascii=False),
                 main_en, config.VLLM_MODEL, run_id))
            ok += 1
            if ok % 50 == 0:
                conn.commit()
                print(f"[caption] ok={ok} fail={fail} / {len(rows)}", flush=True)
            else:
                conn.commit()
    db.finish_run(conn, run_id, {"ok": ok, "fail": fail})
    conn.close()
    print(f"[caption] DONE ok={ok} fail={fail}", flush=True)


if __name__ == "__main__":
    main()
