"""FastAPI review UI for source QA.

  /                dashboard (counts, pass-rates, score summaries)
  /assets          filterable, paginated thumbnail gallery
  /asset/{id}      detail: image, IQA scores, questionnaire answers, event
                   timeline, decisions, preset before/after previews
  /thumb/{id}      lazily-generated cached thumbnail
  /image/{id}      downscaled full image
  /preview?path=   serve a preset render preview
  POST /api/decision      record keep/drop/hold for one asset
  POST /api/bulk_decision  apply a decision to an explicit id list

Run: python -m dataset_build.source_qa.webapp.app  (uvicorn on :8077)
"""
from __future__ import annotations

import io
import json
import os
from typing import List, Optional

from fastapi import FastAPI, Request, Query
from fastapi.responses import HTMLResponse, JSONResponse, Response, FileResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel

from .. import config, db

HERE = os.path.dirname(__file__)
templates = Jinja2Templates(directory=os.path.join(HERE, "templates"))
app = FastAPI(title="VeraRetouch Source QA")
app.mount("/static", StaticFiles(directory=os.path.join(HERE, "static")), name="static")

# columns offered for sorting in the gallery
SORTABLE = {"musiq", "niqe", "brisque", "clipiqa", "sharpness", "aesthetic",
            "aesthetic_vlm", "artimuse_score", "charm_score", "iaa_mixed",
            "bytes_size", "created_at", "asset_id"}


def _conn():
    return db.connect()


# --------------------------------------------------------------------------- #
# Dashboard
# --------------------------------------------------------------------------- #
@app.get("/", response_class=HTMLResponse)
def dashboard(request: Request):
    conn = _conn()
    g = lambda q, *a: conn.execute(q, a).fetchall()
    totals = dict(g("SELECT asset_type, COUNT(*) FROM assets GROUP BY asset_type"))
    by_status = g("SELECT asset_type, status, COUNT(*) c FROM assets GROUP BY asset_type, status ORDER BY asset_type, c DESC")
    by_verdict = g("SELECT asset_type, COALESCE(auto_verdict,'(none)') v, COUNT(*) c FROM assets GROUP BY asset_type, v")
    by_decision = g("SELECT COALESCE(final_decision,'(undecided)') d, COUNT(*) c FROM assets GROUP BY d")
    by_corpus = g("SELECT corpus, asset_type, COUNT(*) c, "
                  "SUM(CASE WHEN pass_a=1 THEN 1 ELSE 0 END) pa, "
                  "SUM(CASE WHEN pass_b=1 THEN 1 ELSE 0 END) pb, "
                  "SUM(CASE WHEN auto_verdict='drop' THEN 1 ELSE 0 END) dropn, "
                  "AVG(musiq) amusiq, AVG(aesthetic) aaes "
                  "FROM assets GROUP BY corpus, asset_type ORDER BY c DESC")
    runs = g("SELECT run_id, kind, started_at, finished_at, stats FROM runs ORDER BY started_at DESC LIMIT 12")
    conn.close()
    return templates.TemplateResponse("dashboard.html", {
        "request": request, "totals": totals, "by_status": by_status,
        "by_verdict": by_verdict, "by_decision": by_decision,
        "by_corpus": by_corpus, "runs": runs,
    })


# --------------------------------------------------------------------------- #
# Gallery
# --------------------------------------------------------------------------- #
@app.get("/assets", response_class=HTMLResponse)
def assets(
    request: Request,
    type: str = "image",
    corpus: str = "",
    status: str = "",
    verdict: str = "",
    decision: str = "",
    pass_a: str = "",
    pass_b: str = "",
    local: str = "",
    metric: str = "",
    op: str = ">=",
    val: Optional[float] = None,
    q: str = "",
    sort: str = "created_at",
    order: str = "desc",
    page: int = 1,
    page_size: int = 60,
):
    conn = _conn()
    where, params = [], []
    if type and type != "all":
        where.append("asset_type=?"); params.append(type)
    if corpus:
        where.append("corpus=?"); params.append(corpus)
    if status:
        where.append("status=?"); params.append(status)
    if verdict:
        where.append("auto_verdict=?"); params.append(verdict)
    if decision == "none":
        where.append("final_decision IS NULL")
    elif decision:
        where.append("final_decision=?"); params.append(decision)
    if pass_a in ("0", "1"):
        where.append("pass_a=?"); params.append(int(pass_a))
    if pass_b in ("0", "1"):
        where.append("pass_b=?"); params.append(int(pass_b))
    if local in ("0", "1"):
        where.append("has_local_mask=?"); params.append(int(local))
    if metric in SORTABLE and val is not None and op in (">=", "<=", ">", "<", "="):
        where.append(f"{metric} {op} ?"); params.append(val)
    if q:
        where.append("(asset_id LIKE ? OR path LIKE ?)"); params += [f"%{q}%", f"%{q}%"]

    wsql = (" WHERE " + " AND ".join(where)) if where else ""
    total = conn.execute(f"SELECT COUNT(*) FROM assets{wsql}", params).fetchone()[0]
    sort_col = sort if sort in SORTABLE else "created_at"
    order = "ASC" if order.lower() == "asc" else "DESC"
    # NULLs last for score sorts
    null_clause = f"{sort_col} IS NULL, " if sort_col not in ("created_at", "asset_id") else ""
    page = max(1, page)
    offset = (page - 1) * page_size
    rows = conn.execute(
        f"SELECT * FROM assets{wsql} ORDER BY {null_clause}{sort_col} {order} LIMIT ? OFFSET ?",
        params + [page_size, offset],
    ).fetchall()
    corpora = [r[0] for r in conn.execute(
        "SELECT DISTINCT corpus FROM assets WHERE corpus IS NOT NULL ORDER BY corpus")]
    statuses = [r[0] for r in conn.execute(
        "SELECT DISTINCT status FROM assets ORDER BY status")]
    conn.close()
    return templates.TemplateResponse("gallery.html", {
        "request": request, "rows": rows, "total": total, "page": page,
        "page_size": page_size, "pages": (total + page_size - 1) // page_size,
        "corpora": corpora, "statuses": statuses, "sortable": sorted(SORTABLE),
        "f": {"type": type, "corpus": corpus, "status": status, "verdict": verdict,
              "decision": decision, "pass_a": pass_a, "pass_b": pass_b, "local": local,
              "metric": metric, "op": op, "val": val, "q": q, "sort": sort, "order": order.lower()},
    })


# --------------------------------------------------------------------------- #
# Asset detail
# --------------------------------------------------------------------------- #
@app.get("/asset/{asset_id}", response_class=HTMLResponse)
def asset_detail(request: Request, asset_id: str):
    conn = _conn()
    a = db.get_asset(conn, asset_id)
    if a is None:
        conn.close()
        return HTMLResponse("not found", status_code=404)
    scores = db.asset_scores(conn, asset_id)
    qa = db.asset_qa(conn, asset_id)
    events = db.asset_events(conn, asset_id)
    decisions = db.asset_decisions(conn, asset_id)
    previews = db.asset_previews(conn, asset_id)
    # next/prev within same corpus+type for review flow
    nxt = conn.execute(
        "SELECT asset_id FROM assets WHERE asset_type=? AND corpus IS ? AND created_at>? "
        "ORDER BY created_at ASC LIMIT 1", (a["asset_type"], a["corpus"], a["created_at"])).fetchone()
    conn.close()
    # group qa by questionnaire
    qa_groups: dict = {}
    for r in qa:
        qa_groups.setdefault(r["questionnaire"], []).append(r)
    meta = {}
    try:
        meta = json.loads(a["meta_json"]) if a["meta_json"] else {}
    except Exception:
        meta = {}
    defs = {"A": config.QUESTIONNAIRE_A, "B": config.QUESTIONNAIRE_B, "C": config.QUESTIONNAIRE_C}
    return templates.TemplateResponse("asset.html", {
        "request": request, "a": a, "scores": scores, "qa_groups": qa_groups,
        "events": events, "decisions": decisions, "previews": previews,
        "meta": meta, "defs": defs, "next_id": nxt[0] if nxt else None,
    })


# --------------------------------------------------------------------------- #
# Image serving (thumbnail + full), with on-disk thumb cache
# --------------------------------------------------------------------------- #
def _load_pil(path: str):
    from PIL import Image
    Image.MAX_IMAGE_PIXELS = None
    try:
        im = Image.open(path)
        im.load()
        return im.convert("RGB")
    except Exception:
        # raw fallback (DNG): try rawpy if available
        try:
            import rawpy
            with rawpy.imread(path) as raw:
                from PIL import Image as I
                return I.fromarray(raw.postprocess())
        except Exception:
            return None


def _placeholder(text: str = "no preview") -> Response:
    svg = (f'<svg xmlns="http://www.w3.org/2000/svg" width="256" height="256">'
           f'<rect width="100%" height="100%" fill="#222"/>'
           f'<text x="50%" y="50%" fill="#888" font-size="14" text-anchor="middle">{text}</text></svg>')
    return Response(svg, media_type="image/svg+xml")


@app.get("/thumb/{asset_id}")
def thumb(asset_id: str, size: int = 256):
    conn = _conn()
    a = db.get_asset(conn, asset_id)
    conn.close()
    if a is None:
        return _placeholder("404")
    if a["asset_type"] != "image":
        # presets: show the rendered "after" of the first stage-2 preview if present
        prev = db.connect()
        pr = prev.execute("SELECT after_path FROM preset_previews WHERE asset_id=? ORDER BY id LIMIT 1",
                          (asset_id,)).fetchone()
        prev.close()
        if pr and pr["after_path"] and os.path.exists(pr["after_path"]):
            return FileResponse(pr["after_path"])
        return _placeholder(f"{a['fmt'] or 'preset'}")
    cache = os.path.join(config.THUMB_DIR, f"{asset_id}_{size}.jpg")
    if os.path.exists(cache):
        return FileResponse(cache)
    im = _load_pil(a["path"])
    if im is None:
        return _placeholder("decode fail")
    im.thumbnail((size, size))
    os.makedirs(config.THUMB_DIR, exist_ok=True)
    im.save(cache, "JPEG", quality=82)
    return FileResponse(cache)


@app.get("/image/{asset_id}")
def image(asset_id: str, longedge: int = 1400):
    conn = _conn()
    a = db.get_asset(conn, asset_id)
    conn.close()
    if a is None or a["asset_type"] != "image":
        return _placeholder("n/a")
    im = _load_pil(a["path"])
    if im is None:
        return _placeholder("decode fail")
    w, h = im.size
    if max(w, h) > longedge:
        s = longedge / max(w, h)
        im = im.resize((int(w * s), int(h * s)))
    buf = io.BytesIO()
    im.save(buf, "JPEG", quality=88)
    return Response(buf.getvalue(), media_type="image/jpeg")


@app.get("/preview")
def preview(path: str):
    if not os.path.exists(path) or not path.startswith(config.PREVIEW_DIR):
        return _placeholder("no preview")
    return FileResponse(path)


# --------------------------------------------------------------------------- #
# Decisions
# --------------------------------------------------------------------------- #
class Decision(BaseModel):
    asset_id: str
    decision: str          # keep | drop | hold
    reason: str = ""
    reviewer: str = "human"


class BulkDecision(BaseModel):
    asset_ids: List[str]
    decision: str
    reason: str = ""
    reviewer: str = "human"


@app.post("/api/decision")
def api_decision(d: Decision):
    if d.decision not in ("keep", "drop", "hold"):
        return JSONResponse({"error": "bad decision"}, status_code=400)
    conn = _conn()
    db.add_decision(conn, d.asset_id, d.decision, f"human:{d.reviewer}", d.reason)
    db.log_event(conn, d.asset_id, "decision", "ok",
                 {"decision": d.decision, "by": d.reviewer, "reason": d.reason})
    conn.commit(); conn.close()
    return {"ok": True}


@app.post("/api/bulk_decision")
def api_bulk(d: BulkDecision):
    if d.decision not in ("keep", "drop", "hold"):
        return JSONResponse({"error": "bad decision"}, status_code=400)
    conn = _conn()
    for aid in d.asset_ids:
        db.add_decision(conn, aid, d.decision, f"human:{d.reviewer}", d.reason)
        db.log_event(conn, aid, "decision", "ok", {"decision": d.decision, "bulk": True})
    conn.commit(); conn.close()
    return {"ok": True, "n": len(d.asset_ids)}


# ---- bulk decision over an ENTIRE filtered set (across all pages) ------------
_FILTER_COLS = {"asset_type": "asset_type", "corpus": "corpus", "status": "status",
                "verdict": "auto_verdict", "decision": "final_decision",
                "pass_a": "pass_a", "pass_b": "pass_b", "local": "has_local_mask"}


def _filter_where(f: dict):
    where, params = [], []
    for key, col in _FILTER_COLS.items():
        v = f.get(key)
        if v in (None, "", "all"):
            continue
        if key == "decision" and v == "none":
            where.append(f"{col} IS NULL"); continue
        if key in ("pass_a", "pass_b", "local"):
            if str(v) not in ("0", "1"):
                continue
            where.append(f"{col}=?"); params.append(int(v)); continue
        where.append(f"{col}=?"); params.append(v)
    if f.get("metric") in SORTABLE and f.get("val") not in (None, "") and f.get("op") in (">=", "<=", ">", "<", "="):
        where.append(f"{f['metric']} {f['op']} ?"); params.append(float(f["val"]))
    if f.get("q"):
        where.append("(asset_id LIKE ? OR path LIKE ?)"); params += [f"%{f['q']}%", f"%{f['q']}%"]
    return (" WHERE " + " AND ".join(where)) if where else "", params


class FilterBulk(BaseModel):
    filters: dict
    decision: str
    reason: str = ""
    reviewer: str = "human"
    limit: int = 100000


@app.post("/api/bulk_by_filter")
def api_bulk_by_filter(d: FilterBulk):
    if d.decision not in ("keep", "drop", "hold"):
        return JSONResponse({"error": "bad decision"}, status_code=400)
    conn = _conn()
    wsql, params = _filter_where(d.filters)
    ids = [r[0] for r in conn.execute(
        f"SELECT asset_id FROM assets{wsql} LIMIT ?", params + [d.limit])]
    for aid in ids:
        db.add_decision(conn, aid, d.decision, f"human:{d.reviewer}", d.reason or "bulk-by-filter")
        db.log_event(conn, aid, "decision", "ok", {"decision": d.decision, "bulk_filter": True})
    conn.commit(); conn.close()
    return {"ok": True, "n": len(ids)}


# ---- human override of a single LLM questionnaire item ----------------------
class QAOverride(BaseModel):
    asset_id: str
    questionnaire: str       # A | B | C
    item: str                # A1.. B1.. C1..
    answer: int              # 0 | 1
    reviewer: str = "human"


@app.post("/api/qa_override")
def api_qa_override(d: QAOverride):
    conn = _conn()
    db.add_qa(conn, d.asset_id, d.questionnaire,
              {d.item: {"answer": int(d.answer), "rationale": f"human override by {d.reviewer}"}},
              model=f"human:{d.reviewer}")
    # recompute pass_a / pass_b from the LATEST answer per item
    for qk, col in (("A", "pass_a"), ("B", "pass_b")):
        latest = {}
        for r in conn.execute(
            "SELECT item, answer FROM llm_qa WHERE asset_id=? AND questionnaire=? ORDER BY id",
            (d.asset_id, qk)):
            latest[r["item"]] = r["answer"]
        items = [v for k, v in latest.items() if k != "overall"]
        if items and all(v is not None for v in items):
            db.update_asset_fields(conn, d.asset_id, **{col: (1 if all(v == 1 for v in items) else 0)})
    db.log_event(conn, d.asset_id, "qa_override", "ok",
                 {"q": d.questionnaire, "item": d.item, "answer": d.answer})
    conn.commit(); conn.close()
    return {"ok": True}


# --------------------------------------------------------------------------- #
# Stats / distribution + calibrated thresholds
# --------------------------------------------------------------------------- #
@app.get("/stats", response_class=HTMLResponse)
def stats(request: Request, metric: str = "musiq", corpus: str = ""):
    conn = _conn()
    # per-corpus questionnaire pass-rates
    qa_rates = conn.execute(
        "SELECT corpus, COUNT(*) n, "
        "SUM(CASE WHEN pass_a=1 THEN 1 ELSE 0 END) pa, SUM(CASE WHEN pass_a IS NOT NULL THEN 1 ELSE 0 END) na, "
        "SUM(CASE WHEN pass_b=1 THEN 1 ELSE 0 END) pb, SUM(CASE WHEN pass_b IS NOT NULL THEN 1 ELSE 0 END) nb "
        "FROM assets WHERE asset_type='image' GROUP BY corpus ORDER BY n DESC").fetchall()
    # histogram of the chosen metric (optionally per corpus)
    col = metric if metric in SORTABLE else "musiq"
    where = [f"{col} IS NOT NULL", "asset_type='image'"]
    params = []
    if corpus:
        where.append("corpus=?"); params.append(corpus)
    vals = [r[0] for r in conn.execute(
        f"SELECT {col} FROM assets WHERE {' AND '.join(where)}", params)]
    hist = []
    lo = hi = None
    if vals:
        lo, hi = min(vals), max(vals)
        nb = 24
        span = (hi - lo) or 1.0
        buckets = [0] * nb
        for v in vals:
            bi = min(nb - 1, int((v - lo) / span * nb))
            buckets[bi] += 1
        mx = max(buckets) or 1
        hist = [{"x0": lo + i * span / nb, "c": c, "h": int(100 * c / mx)} for i, c in enumerate(buckets)]
    thresholds = conn.execute(
        "SELECT * FROM gate_thresholds ORDER BY (corpus='*') DESC, corpus, metric").fetchall()
    corpora = [r[0] for r in conn.execute(
        "SELECT DISTINCT corpus FROM assets WHERE asset_type='image' ORDER BY corpus")]
    conn.close()
    return templates.TemplateResponse("stats.html", {
        "request": request, "qa_rates": qa_rates, "metric": col, "corpus": corpus,
        "corpora": corpora, "hist": hist, "lo": lo, "hi": hi, "n": len(vals),
        "thresholds": thresholds, "sortable": sorted(SORTABLE),
    })


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=int(os.environ.get("SOURCE_QA_PORT", "8077")))
