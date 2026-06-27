"""Databuild viewer — FastAPI backend (浏览 / 构建过程 / 人工 review).

只读浏览 Postgres vera_source_qa + 磁盘渲染图，复用 dataset_build.source_qa.db 的 helper。
唯一写操作：human_review 表（建表 + 投票/打分 upsert）。

跑：  python -m databuild_viewer.backend.app          (默认 127.0.0.1:8077)
自检：python -m databuild_viewer.backend.app --selfcheck
前端 dev：cd databuild_viewer/frontend && npm install && npm run dev  (Vite 代理 /api,/img → :8077)
前端 prod：npm run build → 后端自动从 frontend/dist 提供 SPA。

ponytail: 单人本地工具 — 每请求新开 db 连接（psycopg 连接非线程安全），不做连接池/鉴权。
"""
from __future__ import annotations

import io
import json
import os
from datetime import date, datetime
from decimal import Decimal
from functools import lru_cache
from typing import Optional

from fastapi import FastAPI, HTTPException, Query
from fastapi.responses import FileResponse, HTMLResponse, Response
from fastapi.staticfiles import StaticFiles

from dataset_build.source_qa import db

# --- 路径常量 --------------------------------------------------------------- #
DATA_ROOT = "/home/bc/data/datasets/vera_directionA_1M"
BANK_DIR = os.path.join(DATA_ROOT, "preset_bank_v2")
REVIEW_DIR = os.path.join(BANK_DIR, "review")
SOURCE_INDEX = os.path.join(DATA_ROOT, "source_index.jsonl")
DIST_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "frontend", "dist")
ALLOWED_ROOTS = ("/home/bc/data/datasets/",)  # /img 只放行这些前缀
MAX_PAGE = 200
# preset 详情里需要 JSON.parse 的 TEXT 列（服务端先解析好）
JSON_COLS = ("preset_per_probe", "preset_axes", "preset_tag_metrics", "meta_json")

app = FastAPI(title="databuild-viewer")


# --- 工具 ------------------------------------------------------------------- #
def _conv(v):
    if isinstance(v, (datetime, date)):
        return v.isoformat()
    if isinstance(v, Decimal):
        return float(v)
    return v


def rowdict(r) -> dict:
    return {k: _conv(r[k]) for k in r.keys()}


def _maybe_json(s):
    if isinstance(s, str) and s and s[0] in "{[":
        try:
            return json.loads(s)
        except Exception:
            return s
    return s


@lru_cache(maxsize=1)
def _emb_ids() -> frozenset:
    """已算 embedding 的 preset id（features.jsonl 里出现即算）。"""
    ids = set()
    feats = os.path.join(BANK_DIR, "features.jsonl")
    if os.path.exists(feats):
        with open(feats) as f:
            for line in f:
                try:
                    ids.add(json.loads(line)["preset_id"])
                except Exception:
                    pass
    return frozenset(ids)


@lru_cache(maxsize=1)
def _source_index() -> dict:
    """{source_id: {path, corpus, scene}} — construct/shard 的 source_path 指向已清的 _scratch，用它兜底。"""
    idx = {}
    if os.path.exists(SOURCE_INDEX):
        with open(SOURCE_INDEX) as f:
            for line in f:
                try:
                    d = json.loads(line)
                    idx[d["source_id"]] = {"path": d.get("path"), "corpus": d.get("corpus"),
                                           "scene": d.get("scene")}
                except Exception:
                    pass
    return idx


def _safe_path(p: str) -> str:
    rp = os.path.realpath(p)
    if not any(rp.startswith(os.path.realpath(r)) for r in ALLOWED_ROOTS):
        raise HTTPException(403, "path not allowed")
    if not os.path.exists(rp):
        raise HTTPException(404, "not found")
    return rp


# --- 建表（唯一的写 DDL） --------------------------------------------------- #
DDL = """
CREATE TABLE IF NOT EXISTS human_review (
  id          BIGSERIAL PRIMARY KEY,
  task_kind   TEXT NOT NULL,
  task_id     TEXT NOT NULL,
  subject_id  TEXT,
  choice      TEXT,
  score       DOUBLE PRECISION,
  reviewer    TEXT,
  meta        JSONB,
  created_at  TIMESTAMPTZ DEFAULT now(),
  UNIQUE(task_kind, task_id)
);
"""


def ensure_review_table():
    conn = db.connect()
    try:
        conn.execute(DDL)
        conn.commit()
    finally:
        conn.close()


# --- 浏览：列表（source 图 / preset） --------------------------------------- #
@app.get("/api/list")
def api_list(type: str = Query("image"), corpus: Optional[str] = None,
             sort: str = "aes", order: str = "desc",
             page: int = 1, page_size: int = MAX_PAGE):
    page_size = max(1, min(page_size, MAX_PAGE))
    page = max(1, page)
    off = (page - 1) * page_size
    od = "DESC" if order.lower() == "desc" else "ASC"
    conn = db.connect()
    try:
        if type == "image":
            where = "asset_type='image' AND dup_of IS NULL"
            args = []
            if corpus:
                where += " AND corpus=?"
                args.append(corpus)
            # merit_frac = AES 轮审美 merit 占比(0–1)，覆盖最广，是 source 图排序键
            sort_sql = "merit_frac" if sort == "aes" else "created_at"
            total = conn.execute(f"SELECT count(*) FROM assets WHERE {where}", args).fetchone()[0]
            rows = conn.execute(
                f"SELECT asset_id, corpus, path, status, final_decision, pass_a, pass_b, "
                f"aesthetic_vlm, merit_frac, width, height "
                f"FROM assets WHERE {where} "
                f"ORDER BY {sort_sql} {od} NULLS LAST, asset_id LIMIT ? OFFSET ?",
                args + [page_size, off]).fetchall()
            items = [rowdict(r) for r in rows]
        else:  # preset
            where = "asset_type='preset'"
            args = []
            if corpus:  # 对 preset 复用 corpus 槽当 pack_id 过滤
                where += " AND pack_id=?"
                args.append(corpus)
            sort_sql = "preset_pro_rate" if sort == "aes" else "created_at"
            total = conn.execute(f"SELECT count(*) FROM assets WHERE {where}", args).fetchone()[0]
            rows = conn.execute(
                f"SELECT asset_id, pack_id, kind, fmt, status, pass_c, preset_clean_verdict, "
                f"preset_look_name, preset_pro_rate "
                f"FROM assets WHERE {where} "
                f"ORDER BY {sort_sql} {od} NULLS LAST, asset_id LIMIT ? OFFSET ?",
                args + [page_size, off]).fetchall()
            emb = _emb_ids()
            items = []
            for r in rows:
                d = rowdict(r)
                d["has_embedding"] = d["asset_id"] in emb
                items.append(d)
        return {"total": total, "page": page, "page_size": page_size, "items": items}
    finally:
        conn.close()


@app.get("/api/corpora")
def api_corpora(type: str = "image"):
    conn = db.connect()
    try:
        if type == "image":
            rows = conn.execute(
                "SELECT corpus, count(*) n FROM assets WHERE asset_type='image' "
                "AND dup_of IS NULL GROUP BY corpus ORDER BY n DESC").fetchall()
        else:
            rows = conn.execute(
                "SELECT pack_id corpus, count(*) n FROM assets WHERE asset_type='preset' "
                "GROUP BY pack_id ORDER BY n DESC").fetchall()
        return [{"corpus": r["corpus"], "n": r["n"]} for r in rows]
    finally:
        conn.close()


# --- 浏览：资产详情 --------------------------------------------------------- #
@app.get("/api/asset/{asset_id}")
def api_asset(asset_id: str):
    conn = db.connect()
    try:
        a = db.get_asset(conn, asset_id)
        if not a:
            raise HTTPException(404, "asset not found")
        asset = rowdict(a)
        for c in JSON_COLS:
            if c in asset:
                asset[c] = _maybe_json(asset[c])
        qa = [rowdict(r) for r in db.asset_qa(conn, asset_id)]
        # 2 轮 QA 拆分：round1 = A/B，round2(AES) = aes 项，caption 单列
        round1 = [q for q in qa if q["questionnaire"] in ("A", "B")]
        caption = next((q["raw"] for q in qa if q["questionnaire"] == "caption"), None)
        round2 = [q for q in qa if q["questionnaire"] == "aes"]
        out = {
            "asset": asset,
            "round1": round1,
            "caption": caption,
            "round2": round2,
            "scores": [rowdict(r) for r in db.asset_scores(conn, asset_id)],
            "decisions": [rowdict(r) for r in db.asset_decisions(conn, asset_id)],
            "events": [rowdict(r) for r in db.asset_events(conn, asset_id)],
        }
        if asset.get("asset_type") == "preset":
            prev = []
            for r in db.asset_previews(conn, asset_id):
                d = rowdict(r)
                d["after_iqa"] = _maybe_json(d.get("after_iqa"))
                d["paired_metrics"] = _maybe_json(d.get("paired_metrics"))
                prev.append(d)
            out["previews"] = prev
            out["has_embedding"] = asset_id in _emb_ids()
        return out
    finally:
        conn.close()


# --- 构建过程：construct 流水线已全部入库 ----------------------------------- #
# 每个 group = 一张源图的 databuild 单元；candidates = N 个 preset 渲染(after 图持久化在
# render_stage) + QA + 排名 + role；group 产出 sft / dpo。全部读 construct_* 库表。
@app.get("/api/runs")
def api_runs():
    conn = db.connect()
    try:
        rows = conn.execute(
            "SELECT run_id, route, count(*) n FROM construct_groups "
            "GROUP BY run_id, route ORDER BY max(created_at) DESC").fetchall()
        return [rowdict(r) for r in rows]
    finally:
        conn.close()


@app.get("/api/group/list")
def api_group_list(run: Optional[str] = None, route: Optional[str] = None,
                   page: int = 1, page_size: int = MAX_PAGE):
    page_size = max(1, min(page_size, MAX_PAGE))
    page = max(1, page)
    off = (page - 1) * page_size
    where, args = "1=1", []
    if run:
        where += " AND run_id=?"; args.append(run)
    if route:
        where += " AND route=?"; args.append(route)
    conn = db.connect()
    try:
        total = conn.execute(f"SELECT count(*) FROM construct_groups WHERE {where}", args).fetchone()[0]
        rows = conn.execute(
            f"SELECT group_id, run_id, route, source_path, source_asset_id, is_portrait, "
            f"n_candidates, created_at FROM construct_groups WHERE {where} "
            f"ORDER BY created_at DESC LIMIT ? OFFSET ?", args + [page_size, off]).fetchall()
        items = []
        for r in rows:
            d = rowdict(r)
            src = _source_index().get(d.get("source_asset_id")) or {}
            d["_src_path"] = src.get("path") or d.get("source_path")
            items.append(d)
        return {"total": total, "page": page, "page_size": page_size, "items": items}
    finally:
        conn.close()


@app.get("/api/group/{group_id}")
def api_group(group_id: str):
    conn = db.connect()
    try:
        g = conn.execute("SELECT * FROM construct_groups WHERE group_id=?", [group_id]).fetchone()
        if not g:
            raise HTTPException(404, "group not found")
        group = rowdict(g)
        src = _source_index().get(group.get("source_asset_id")) or {}
        group["_src_path"] = src.get("path") or group.get("source_path")
        cands = [rowdict(r) for r in conn.execute(
            "SELECT * FROM construct_candidates WHERE group_id=? "
            "ORDER BY rank NULLS LAST, merit_score DESC NULLS LAST", [group_id]).fetchall()]
        sft = [rowdict(r) for r in conn.execute(
            "SELECT * FROM construct_sft WHERE group_id=? ORDER BY created_at", [group_id]).fetchall()]
        dpo = [rowdict(r) for r in conn.execute(
            "SELECT * FROM construct_dpo WHERE group_id=? ORDER BY created_at", [group_id]).fetchall()]
        return {"group": group, "candidates": cands, "sft": sft, "dpo": dpo}
    finally:
        conn.close()


# --- 人工 review ------------------------------------------------------------ #
def _load_tasks(kind: str) -> list:
    fn = {"pairwise": "pairwise_tasks.jsonl", "scalar": "scalar_tasks.jsonl"}.get(kind)
    if not fn:
        raise HTTPException(400, "bad kind")
    p = os.path.join(REVIEW_DIR, fn)
    if not os.path.exists(p):
        return []
    out = []
    with open(p) as f:
        for line in f:
            line = line.strip()
            if line:
                try:
                    out.append(json.loads(line))
                except Exception:
                    pass
    return out


@app.get("/api/review/tasks")
def api_review_tasks(kind: str = "pairwise"):
    tasks = _load_tasks(kind)
    conn = db.connect()
    try:
        done = {r["task_id"]: rowdict(r) for r in conn.execute(
            "SELECT task_id, choice, score, reviewer FROM human_review WHERE task_kind=?",
            [kind]).fetchall()}
    finally:
        conn.close()
    for t in tasks:
        t["_review"] = done.get(t.get("task_id"))
    return {"kind": kind, "available": bool(tasks), "tasks": tasks}


@app.post("/api/review")
async def api_review_submit(payload: dict):
    kind = payload.get("task_kind")
    tid = payload.get("task_id")
    if kind not in ("pairwise", "scalar") or not tid:
        raise HTTPException(400, "task_kind+task_id required")
    conn = db.connect()
    try:
        conn.execute(
            "INSERT INTO human_review (task_kind, task_id, subject_id, choice, score, reviewer, meta) "
            "VALUES (?,?,?,?,?,?,?::jsonb) "
            "ON CONFLICT (task_kind, task_id) DO UPDATE SET "
            "choice=EXCLUDED.choice, score=EXCLUDED.score, reviewer=EXCLUDED.reviewer, "
            "meta=EXCLUDED.meta, created_at=now()",
            [kind, tid, payload.get("subject_id"), payload.get("choice"),
             payload.get("score"), payload.get("reviewer") or "local",
             json.dumps(payload.get("meta") or {})])
        conn.commit()
    finally:
        conn.close()
    return {"ok": True}


# --- 图片服务 --------------------------------------------------------------- #
@lru_cache(maxsize=2048)
def _thumb(path: str, mtime: float, w: int) -> bytes:
    from PIL import Image
    Image.MAX_IMAGE_PIXELS = None
    im = Image.open(path).convert("RGB")
    if w and im.width > w:
        im.thumbnail((w, w * 4))
    buf = io.BytesIO()
    im.save(buf, "JPEG", quality=88)
    return buf.getvalue()


@app.get("/img")
def img(path: str, w: int = 512, full: int = 0):
    rp = _safe_path(path)
    if full:
        return FileResponse(rp)
    try:
        data = _thumb(rp, os.path.getmtime(rp), int(w))
    except Exception as e:
        raise HTTPException(500, f"thumb failed: {e}")
    return Response(data, media_type="image/jpeg")


# --- SPA（生产用 dist；dev 直接用 vite） ------------------------------------ #
if os.path.isdir(DIST_DIR):
    app.mount("/", StaticFiles(directory=DIST_DIR, html=True), name="spa")
else:
    @app.get("/", response_class=HTMLResponse)
    def _devhint():
        return ("<body style='font-family:monospace;background:#14161a;color:#e6e3dc;padding:40px'>"
                "<h2>databuild-viewer · 前端未构建</h2>"
                "<p>开发：<code>cd databuild_viewer/frontend &amp;&amp; npm install &amp;&amp; npm run dev</code> "
                "（Vite 在 5173，自动代理 /api,/img 到本后端）</p>"
                "<p>生产：<code>npm run build</code> 后刷新本页，后端将从 frontend/dist 提供 SPA。</p></body>")


# --- 自检 / 启动 ------------------------------------------------------------ #
def selfcheck():
    assert _source_index(), "source_index empty"
    assert _emb_ids(), "embedding ids empty"
    ensure_review_table()
    r = api_list(type="image", sort="aes", page=1, page_size=5)
    assert r["total"] > 0 and r["items"], "image list empty"
    a = api_asset(r["items"][0]["asset_id"])
    assert a["asset"], "asset detail empty"
    p = api_list(type="preset", page=1, page_size=5)
    assert p["items"], "preset list empty"
    runs = api_runs()
    gl = api_group_list(page=1, page_size=5)
    extra = ""
    if gl["items"]:
        g = api_group(gl["items"][0]["group_id"])
        extra = f" cands={len(g['candidates'])} sft={len(g['sft'])} dpo={len(g['dpo'])}"
    print(f"selfcheck OK: images={r['total']} presets={p['total']} "
          f"emb_ids={len(_emb_ids())} runs={len(runs)} groups={gl['total']}{extra}")


if __name__ == "__main__":
    import sys
    if "--selfcheck" in sys.argv:
        selfcheck()
    else:
        import socket
        import uvicorn
        port = int(os.environ.get("DBV_PORT", "8077"))
        host = os.environ.get("DBV_HOST", "0.0.0.0")  # 默认对局域网开放
        ensure_review_table()
        lan = socket.gethostbyname(socket.gethostname())
        print(f"databuild-viewer: http://{lan}:{port}  (局域网) · http://127.0.0.1:{port} (本机)")
        uvicorn.run(app, host=host, port=port, log_level="info")
