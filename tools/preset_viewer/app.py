"""TOOL-PresetView-1 — 独立只读 preset 浏览界面（默认 127.0.0.1:8078）。

浏览 `vera_source_qa.assets` 里的 12,192 个 preset：元数据 / 逐通道 per_probe /
llm_qa 记录 / preset 源文件文本 / 该 preset 在 canonical build 里的真实渲染例图
（例图元数据来自 `research.canonical_candidates`）。

跑：    python -m tools.preset_viewer.app --config databuild.prod-g4-global25k-20260801.toml
自检：  python -m tools.preset_viewer.app --config <toml> --selfcheck

after 例图（TOOL-PresetView-2）：本机 `/mnt/ramstage` 已清空，candidate 的 after 图
只在 NFS 归档里。本工具**不碰 NFS**，改为借道主 databuild viewer（`--dbv`，默认
`http://127.0.0.1:8077`）的 prepare 协议：逐 group POST 让 8077 把成员物化进它自己的
缓存，ready 之后再通过本进程的 `/api/dbvimg` 反代 8077 的 `/img`。8077 是单 worker
FIFO 且主人的浏览器也在用同一条队列，所以只在用户点「加载例图」时才发，且逐个串行。

只读纪律：两个业务连接都以 `-c default_transaction_read_only=on` 建连（会话级，
事务无法翻回可写）。唯一的写操作是启动时那条 `CREATE INDEX IF NOT EXISTS`，走单
独的、没有只读选项的连接（见 `ensure_index`，已获主 agent 授权，理由见 NOTES.md）。

ponytail: 单人本地工具 —— 每请求新开连接（psycopg 连接非线程安全），不做连接池、
不做鉴权、前端是一段内嵌字符串，无构建步骤。
"""
from __future__ import annotations

import argparse
import io
import json
import os
import sys
import tomllib
from datetime import date, datetime
from decimal import Decimal
from typing import Any, Optional

import httpx
import psycopg
from fastapi import FastAPI, HTTPException, Query
from fastapi.responses import HTMLResponse, JSONResponse, Response, StreamingResponse
from psycopg.conninfo import conninfo_to_dict, make_conninfo
from psycopg.rows import dict_row

# --- 常量 ------------------------------------------------------------------- #
QA_DB = "vera_source_qa"          # assets / llm_qa
RESEARCH_DB = "research"          # canonical_candidates / canonical_groups

# /img 放行的根。realpath 归一化后做前缀匹配，其余一律 403。
# 第三个根是本次实施新增（MMArt-PPR10k 是 canonical_groups.source_path 唯一还在本地
# 的一批源图；/mnt/ramstage 已空）——见 NOTES.md「待主 agent 决策」D1。
ALLOWED_ROOTS = (
    "/mnt/ramstage/",
    "/home/bc/data/datasets/",
    "/home/bc/datasets/MMArt-PPR10k/",
)

# after 例图物化只可能落在这个根下（8077 的 canonical 逻辑路径）。这里只做一道廉价
# 前置过滤，权威校验由 8077 的 `_authorized_image_path` 负责——见 NOTES.md D5。
DBV_IMG_ROOT = "/mnt/ramstage/"
DBV_TIMEOUT = httpx.Timeout(30.0, connect=2.0)    # 连接 2s / 读 30s

MAX_PAGE_SIZE = 200
SRC_MAX_LINES = 200               # 源文件最多展示行数
SRC_READ_BYTES = 512 * 1024       # 只读文件头这么多字节来取那 200 行
CANDIDATE_LIMIT = 12              # 详情页例图条数
# TEXT 存 JSON 的列，服务端先 json.loads
JSON_COLS = ("preset_per_probe", "preset_axes", "preset_tag_metrics", "meta_json")
LIST_COLS = (
    "asset_id", "fmt", "kind", "pack_id", "status", "preset_look_name",
    "preset_clean_verdict", "preset_pro_rate", "preset_grade_family",
)
SORTS = {
    "pro_rate": "preset_pro_rate DESC NULLS LAST, asset_id ASC",
    "asset_id": "asset_id ASC",
}

app = FastAPI(title="preset-viewer")
QA_DSN = ""
RESEARCH_DSN = ""
DBV_BASE = "http://127.0.0.1:8077"    # 主 databuild viewer，--dbv 覆盖


# --- 连接 ------------------------------------------------------------------- #
def _dsn_for(base_dsn: str, dbname: str, read_only: bool = True) -> str:
    d = conninfo_to_dict(base_dsn)
    d["dbname"] = dbname
    opts = ["-c statement_timeout=120000"]
    if read_only:
        opts.insert(0, "-c default_transaction_read_only=on")
    d["options"] = " ".join(opts)
    return make_conninfo(**d)


def load_dsns(config_path: str) -> tuple[str, str]:
    """从 databuild TOML 的 [viewer].postgres_dsn 派生两个只读连接串。"""
    with open(config_path, "rb") as f:
        cfg = tomllib.load(f)
    base = (cfg.get("viewer") or {}).get("postgres_dsn")
    if not base:
        raise SystemExit(f"{config_path}: 缺少 [viewer].postgres_dsn")
    return _dsn_for(base, QA_DB), _dsn_for(base, RESEARCH_DB)


def qa_conn():
    return psycopg.connect(QA_DSN, row_factory=dict_row)


def research_conn():
    return psycopg.connect(RESEARCH_DSN, row_factory=dict_row)


def ensure_index(base_dsn_readonly: str) -> str:
    """唯一的写操作：按 preset_id 查例图所需的索引。缺它就是 4.2GB 全表扫。

    注意用的是真实列 `canonical_candidates.preset_id`（普通 text 列，已填充），
    不是任务卡里写的表达式 `(payload->>'preset_id')` —— 见 NOTES.md D2。
    """
    d = conninfo_to_dict(base_dsn_readonly)
    d["options"] = "-c statement_timeout=600000"          # 建索引不能只读
    try:
        with psycopg.connect(make_conninfo(**d), autocommit=True) as c:
            c.execute(
                "CREATE INDEX IF NOT EXISTS canonical_candidates_preset_idx "
                "ON canonical_candidates (preset_id)")
        return "ok"
    except Exception as e:                                 # 索引缺失只是慢，不该拦启动
        return f"skipped: {type(e).__name__}: {e}"


# --- 小工具 ----------------------------------------------------------------- #
def _conv(v: Any) -> Any:
    if isinstance(v, (datetime, date)):
        return v.isoformat()
    if isinstance(v, Decimal):
        return float(v)
    return v


def rowdict(r: dict) -> dict:
    return {k: _conv(v) for k, v in r.items()}


def _maybe_json(s: Any) -> Any:
    if isinstance(s, str) and s[:1] in ("{", "["):
        try:
            return json.loads(s)
        except Exception:
            return s
    return s


def safe_path(path: str) -> str:
    """realpath 归一化 + 白名单前缀校验。这是信任边界，别省。"""
    real = os.path.realpath(path)
    if not any(real.startswith(root) for root in ALLOWED_ROOTS):
        raise HTTPException(403, "path outside allowed roots")
    return real


def read_source_text(path: Optional[str]) -> dict:
    """preset 源文件：前 200 行文本；二进制 / 读不动只给大小。"""
    if not path:
        return {"path": None, "exists": False}
    out: dict[str, Any] = {"path": path, "exists": os.path.exists(path)}
    if not out["exists"]:
        return out
    try:
        out["bytes"] = os.path.getsize(path)
        with open(path, "rb") as f:
            head = f.read(SRC_READ_BYTES)
    except OSError as e:
        out["error"] = str(e)
        return out
    if b"\x00" in head:
        out["binary"] = True
        return out
    lines = head.decode("utf-8", errors="replace").splitlines()
    out["binary"] = False
    out["text"] = "\n".join(lines[:SRC_MAX_LINES])
    out["truncated"] = len(lines) > SRC_MAX_LINES or out["bytes"] > len(head)
    out["shown_lines"] = min(len(lines), SRC_MAX_LINES)
    return out


def fetch_candidates(preset_id: str, limit: int = CANDIDATE_LIMIT) -> list[dict]:
    """该 preset 在各 canonical build 里的渲染例图（winner 优先，rank 升序）。"""
    with research_conn() as c:
        rows = c.execute(
            "SELECT candidate_id, group_id, build_id, slot_index, preset_format, "
            "       major, minor, after_path, cgt_path, region, winner, rank, "
            "       payload->>'style_name' AS style_name, "
            "       payload->>'preset_path' AS preset_path, qa "
            "FROM canonical_candidates WHERE preset_id = %s "
            "ORDER BY winner DESC, rank ASC NULLS LAST LIMIT %s",
            (preset_id, limit)).fetchall()
        items = [rowdict(r) for r in rows]
        gids = sorted({i["group_id"] for i in items if i.get("group_id")})
        gmap = {}
        if gids:
            gmap = {g["group_id"]: g for g in c.execute(
                "SELECT group_id, source_id, source_path, render_mode, scene "
                "FROM canonical_groups WHERE group_id = ANY(%s)", (gids,)).fetchall()}
    for it in items:
        g = gmap.get(it.get("group_id")) or {}
        it["source_path"] = g.get("source_path")            # 查不到就留空
        it["source_id"] = g.get("source_id")
        it["render_mode"] = g.get("render_mode")
        it["after_exists"] = bool(it.get("after_path")) and os.path.exists(it["after_path"])
        it["source_exists"] = bool(it.get("source_path")) and os.path.exists(it["source_path"])
    return items


def candidate_group_keys(asset_id: str, limit: int = CANDIDATE_LIMIT
                         ) -> list[tuple[str, str]]:
    """详情页那 ≤12 张例图涉及的 (group_id, build_id)，去重、保持展示顺序。"""
    keys: list[tuple[str, str]] = []
    for c in fetch_candidates(asset_id):
        k = (c.get("group_id"), c.get("build_id"))
        if k[0] and k[1] and k not in keys:
            keys.append(k)
    return keys[:limit]


# --- 8077 prepare 通道（本进程只当 HTTP 客户端，不碰 NFS、不自建缓存） -------- #
def _dbv_call(method: str, path: str, params: dict) -> dict:
    """对 8077 发一次调用。8077 不可达一律返回 {'error': ...} 而不是抛异常。"""
    try:
        with httpx.Client(timeout=DBV_TIMEOUT) as cli:
            r = cli.request(method, DBV_BASE + path, params=params)
    except httpx.RequestError as e:
        return {"error": f"{type(e).__name__}: {e}", "http": None}
    out: dict[str, Any]
    try:
        body = r.json()
        out = body if isinstance(body, dict) else {"body": body}
    except ValueError:
        out = {"body": r.text[:200]}
    out["http"] = r.status_code
    if r.status_code >= 400:
        out.setdefault("error", out.pop("detail", f"HTTP {r.status_code}"))
    return out


def _group_row(key: tuple[str, str], res: dict, on_404: str = "not_started") -> dict:
    gid, bid = key
    # GET 侧 404 = "还没起过任务"，是正常初始态；POST 侧 404 = 8077 不认识这个
    # group（它加载的 build 集合和我们查的库不一致），那是真错误。
    state = res.get("state")
    if state is None:
        state = on_404 if res.get("http") == 404 else "error"
    return {
        "group_id": gid, "build_id": bid, "state": state,
        "files": res.get("files"), "bytes": res.get("bytes"),
        "message": res.get("message"), "http": res.get("http"),
        "error": res.get("error") if state in ("error", "not_found") else None,
    }


def dbv_prepare(keys: list[tuple[str, str]], retry: bool = False) -> dict:
    """逐 group **串行** POST。8077 自己会对已 ready 的 group 去重。"""
    rows = [_group_row(k, _dbv_call(
        "POST", f"/api/groups/{k[0]}/prepare",
        {"build_id": k[1], **({"retry": "true"} if retry else {})}),
        on_404="not_found") for k in keys]
    return _summary(rows)


def dbv_status(keys: list[tuple[str, str]]) -> dict:
    rows = [_group_row(k, _dbv_call(
        "GET", f"/api/groups/{k[0]}/prepare", {"build_id": k[1]})) for k in keys]
    return _summary(rows)


def _summary(rows: list[dict]) -> dict:
    # `not_started` 在轮询侧也算终态：8077 的 POST 是同步登记任务的（POST 立刻就能被
    # GET 到），所以 POST 之后还 404 只可能是那次 POST 本身没登记上，等下去没有意义。
    done = {"ready", "failed", "error", "not_found", "not_started"}
    return {
        "dbv": DBV_BASE,
        "reachable": any(r["http"] is not None for r in rows) if rows else None,
        "n_total": len(rows),
        "n_ready": sum(r["state"] == "ready" for r in rows),
        "all_done": all(r["state"] in done for r in rows) if rows else True,
        "groups": rows,
    }


# --- API -------------------------------------------------------------------- #
@app.get("/api/presets")
def api_presets(
    verdict: Optional[str] = None,
    fmt: Optional[str] = None,
    pack: Optional[str] = None,
    q: Optional[str] = None,
    sort: str = "pro_rate",
    page: int = 1,
    page_size: int = 50,
):
    page = max(1, page)
    page_size = max(1, min(page_size, MAX_PAGE_SIZE))
    order = SORTS.get(sort)
    if order is None:
        raise HTTPException(400, f"sort must be one of {sorted(SORTS)}")

    where = ["asset_type = 'preset'"]
    args: list[Any] = []
    if verdict:
        if verdict == "__null__":
            where.append("preset_clean_verdict IS NULL")
        else:
            where.append("preset_clean_verdict = %s")
            args.append(verdict)
    if fmt:
        where.append("fmt = %s")
        args.append(fmt)
    if pack:
        where.append("pack_id = %s")
        args.append(pack)
    if q:
        where.append("(asset_id ILIKE %s OR preset_look_name ILIKE %s "
                     "OR preset_caption ILIKE %s)")
        args += [f"%{q}%"] * 3
    wsql = " AND ".join(where)
    cols = ", ".join(LIST_COLS)

    with qa_conn() as c:
        total = c.execute(f"SELECT count(*) AS n FROM assets WHERE {wsql}",
                          args).fetchone()["n"]
        rows = c.execute(
            f"SELECT {cols}, left(coalesce(preset_caption, ''), 120) AS caption "
            f"FROM assets WHERE {wsql} ORDER BY {order} LIMIT %s OFFSET %s",
            args + [page_size, (page - 1) * page_size]).fetchall()
        facets = {
            k: [rowdict(r) for r in c.execute(
                f"SELECT {col} AS v, count(*) AS n FROM assets "
                f"WHERE asset_type='preset' GROUP BY 1 ORDER BY n DESC").fetchall()]
            for k, col in (("verdict", "preset_clean_verdict"),
                           ("fmt", "fmt"), ("pack", "pack_id"))
        }
    return {"total": total, "page": page, "page_size": page_size,
            "sort": sort, "items": [rowdict(r) for r in rows], "facets": facets}


@app.get("/api/preset/{asset_id}")
def api_preset(asset_id: str):
    with qa_conn() as c:
        row = c.execute("SELECT * FROM assets WHERE asset_id = %s AND "
                        "asset_type = 'preset'", (asset_id,)).fetchone()
        if not row:
            raise HTTPException(404, "preset not found")
        asset = rowdict(row)
        for col in JSON_COLS:
            if col in asset:
                asset[col] = _maybe_json(asset[col])
        qa = [rowdict(r) for r in c.execute(
            "SELECT id, questionnaire, item, answer, rationale, raw, model, run_id, "
            "probe_id, created_at FROM llm_qa WHERE asset_id = %s "
            "ORDER BY run_id, probe_id, id", (asset_id,)).fetchall()]
    return {
        "asset": asset,
        "qa": qa,
        "source_file": read_source_text(asset.get("path")),
        "candidates": fetch_candidates(asset_id),
    }


@app.post("/api/preset/{asset_id}/prepare")
def api_prepare_post(asset_id: str, retry: bool = False,
                     limit: int = Query(CANDIDATE_LIMIT, ge=1, le=CANDIDATE_LIMIT)):
    """让 8077 物化该 preset 例图所在的 group。立即返回受理列表，不等 ready。

    `limit` 只给 selfcheck 用（把对 8077 队列的打扰压到 1 个 group）；前端用默认值。
    """
    return dbv_prepare(candidate_group_keys(asset_id, limit), retry=retry)


@app.get("/api/preset/{asset_id}/prepare")
def api_prepare_get(asset_id: str,
                    limit: int = Query(CANDIDATE_LIMIT, ge=1, le=CANDIDATE_LIMIT)):
    return dbv_status(candidate_group_keys(asset_id, limit))


@app.get("/api/dbvimg")
def dbvimg(path: str, w: Optional[int] = Query(None, ge=16, le=4096)):
    """流式反代 8077 的 /img（避免浏览器跨端口直连）。

    这里只挡住明显不属于 canonical 物化域的路径；真正的权威授权在 8077 那边。
    8077 对"group 还没 ready"的读返回 409，原样透传给前端。
    """
    if not os.path.realpath(path).startswith(DBV_IMG_ROOT):
        raise HTTPException(403, f"path outside {DBV_IMG_ROOT}")
    params: dict[str, Any] = {"path": path}
    if w:
        params["w"] = w
    else:
        params["full"] = "true"
    cli = httpx.Client(timeout=DBV_TIMEOUT)
    try:
        resp = cli.send(cli.build_request("GET", DBV_BASE + "/img", params=params),
                        stream=True)
    except httpx.RequestError as e:
        cli.close()
        raise HTTPException(502, f"databuild viewer unreachable: "
                                 f"{type(e).__name__}: {e}") from None
    if resp.status_code != 200:
        code = resp.status_code
        resp.close()
        cli.close()
        raise HTTPException(code if code in (403, 404, 409, 422) else 502,
                            f"databuild viewer /img -> {code}")

    def body():
        try:
            yield from resp.iter_bytes()
        finally:
            resp.close()
            cli.close()

    return StreamingResponse(
        body(), media_type=resp.headers.get("content-type", "image/jpeg"))


@app.get("/img")
def img(path: str, w: Optional[int] = Query(None, ge=16, le=4096)):
    real = safe_path(path)
    if not os.path.isfile(real):
        raise HTTPException(404, "file not found")   # ramstage 轮换后失效属预期
    if not w:
        with open(real, "rb") as f:
            data = f.read()
        ext = os.path.splitext(real)[1].lower()
        mime = {".png": "image/png", ".webp": "image/webp",
                ".jpg": "image/jpeg", ".jpeg": "image/jpeg"}.get(ext, "image/jpeg")
        return Response(data, media_type=mime)
    from PIL import Image
    with Image.open(real) as im:
        im = im.convert("RGB")
        im.thumbnail((w, w * 4))
        buf = io.BytesIO()
        im.save(buf, "JPEG", quality=88)
    return Response(buf.getvalue(), media_type="image/jpeg")


@app.get("/healthz")
def healthz():
    with qa_conn() as c:
        n = c.execute("SELECT count(*) AS n FROM assets "
                      "WHERE asset_type='preset'").fetchone()["n"]
    return JSONResponse({"ok": True, "presets": n})


# --- 前端（单页，内嵌，无构建步骤） ------------------------------------------ #
INDEX_HTML = r"""<!doctype html>
<meta charset="utf-8"><title>preset viewer</title>
<meta name="viewport" content="width=device-width,initial-scale=1">
<style>
 *{box-sizing:border-box} body{margin:0;font:13px/1.5 system-ui,sans-serif;color:#222}
 #wrap{display:flex;height:100vh}
 #left{width:380px;min-width:380px;border-right:1px solid #ddd;display:flex;flex-direction:column}
 #right{flex:1;overflow:auto;padding:14px}
 #filters{padding:8px;border-bottom:1px solid #ddd;display:grid;grid-template-columns:1fr 1fr;gap:5px}
 #filters input,#filters select{width:100%;padding:3px;font:inherit}
 #list{flex:1;overflow:auto}
 .it{padding:6px 8px;border-bottom:1px solid #eee;cursor:pointer}
 .it:hover{background:#f2f6ff} .it.sel{background:#dbe8ff}
 .it b{font-weight:600} .mut{color:#777;font-size:11px}
 #pager{padding:6px 8px;border-top:1px solid #ddd;display:flex;gap:6px;align-items:center}
 table{border-collapse:collapse;margin:6px 0} td,th{border:1px solid #ddd;padding:2px 6px;text-align:left;vertical-align:top}
 th{background:#f6f6f6;font-weight:600}
 pre{background:#f7f7f7;border:1px solid #ddd;padding:8px;overflow:auto;max-height:420px;white-space:pre-wrap;word-break:break-all}
 h2{margin:4px 0 8px;font-size:17px} h3{margin:14px 0 4px;font-size:14px}
 .grid{display:grid;grid-template-columns:repeat(auto-fill,minmax(300px,1fr));gap:10px}
 .card{border:1px solid #ddd;padding:6px} .card img{width:100%;background:#eee;min-height:40px}
 .pair{display:grid;grid-template-columns:1fr 1fr;gap:4px}
 .kv{display:grid;grid-template-columns:max-content 1fr;gap:2px 10px;font-size:12px}
 .kv div:nth-child(odd){color:#666} .tag{background:#eef;padding:0 5px;border-radius:3px;margin-right:4px}
 details{margin:6px 0} summary{cursor:pointer;user-select:none}
 .warn{background:#fff6e0;border:1px solid #e8c97a;padding:6px;margin:6px 0}
 .slot{display:flex;align-items:center;justify-content:center;min-height:90px;
       background:#f0f0f0;border:1px dashed #ccc;color:#888;font-size:11px;text-align:center}
 button{font:inherit;padding:2px 10px;cursor:pointer}
</style>
<div id=wrap>
 <div id=left>
  <div id=filters>
   <select id=f_verdict></select>
   <select id=f_fmt></select>
   <select id=f_pack></select>
   <select id=f_sort><option value=pro_rate>sort: pro_rate</option><option value=asset_id>sort: asset_id</option></select>
   <input id=f_q placeholder="q: id / look_name / caption" style="grid-column:1/3">
  </div>
  <div id=list></div>
  <div id=pager><button id=prev>◀</button><span id=pginfo></span><button id=next>▶</button></div>
 </div>
 <div id=right><p class=mut>← 左侧选一个 preset</p></div>
</div>
<script>
const $=s=>document.querySelector(s);
let page=1,total=0,cur=null,facetsDone=false,prepTimer=null;
const esc=s=>String(s??'').replace(/[&<>"]/g,c=>({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;'}[c]));
const num=v=>v==null?'':(typeof v==='number'?(Number.isInteger(v)?v:v.toFixed(3)):v);
const imgurl=(p,w)=>'/img?path='+encodeURIComponent(p)+(w?'&w='+w:'');

function qs(){
 const p=new URLSearchParams({page,page_size:50,sort:$('#f_sort').value});
 for(const [k,el] of [['verdict','#f_verdict'],['fmt','#f_fmt'],['pack','#f_pack'],['q','#f_q']]){
   const v=$(el).value; if(v) p.set(k,v);
 } return p.toString();
}
function fillFacet(el,facet,label){
 if(facetsDone) return;
 el.innerHTML='<option value="">'+label+': all</option>'+facet.map(f=>
   '<option value="'+esc(f.v===null?'__null__':f.v)+'">'+esc(f.v===null?'(null)':f.v)+' ('+f.n+')</option>').join('');
}
async function load(){
 const d=await (await fetch('/api/presets?'+qs())).json();
 total=d.total;
 fillFacet($('#f_verdict'),d.facets.verdict,'verdict');
 fillFacet($('#f_fmt'),d.facets.fmt,'fmt');
 fillFacet($('#f_pack'),d.facets.pack,'pack');
 facetsDone=true;
 $('#list').innerHTML=d.items.map(i=>
  '<div class=it data-id="'+esc(i.asset_id)+'"><b>'+esc(i.preset_look_name||'(no name)')+'</b> '+
  '<span class=tag>'+esc(i.fmt)+'</span><span class=tag>'+esc(i.pack_id)+'</span>'+
  (i.preset_pro_rate!=null?'<span class=tag>pro '+num(i.preset_pro_rate)+'</span>':'')+
  '<div class=mut>'+esc(i.asset_id)+' · '+esc(i.preset_clean_verdict||'-')+' · '+esc(i.status||'')+'</div>'+
  '<div class=mut>'+esc(i.caption||'')+'</div></div>').join('')||'<p class=mut style=padding:8px>无结果</p>';
 const pages=Math.max(1,Math.ceil(total/50));
 $('#pginfo').textContent='第 '+page+' / '+pages+' 页 · total '+total;
 document.querySelectorAll('.it').forEach(e=>e.onclick=()=>open_(e.dataset.id));
 if(cur) document.querySelectorAll('.it').forEach(e=>e.classList.toggle('sel',e.dataset.id===cur));
}
function kvTable(o,keys){
 return '<div class=kv>'+keys.filter(k=>o[k]!=null&&o[k]!=='').map(k=>
   '<div>'+esc(k)+'</div><div>'+esc(num(o[k]))+'</div>').join('')+'</div>';
}
function probeTable(pp){
 if(!pp||typeof pp!=='object') return '<p class=mut>无 per_probe</p>';
 return '<table><tr><th>通道</th><th>描述</th></tr>'+Object.entries(pp).map(([k,v])=>
  '<tr><td>'+esc(k)+'</td><td>'+esc(typeof v==='object'?JSON.stringify(v):v)+'</td></tr>').join('')+'</table>';
}
async function open_(id){
 cur=id; location.hash='#/preset/'+id;
 document.querySelectorAll('.it').forEach(e=>e.classList.toggle('sel',e.dataset.id===id));
 $('#right').innerHTML='<p class=mut>loading…</p>';
 const r=await fetch('/api/preset/'+encodeURIComponent(id));
 if(!r.ok){$('#right').innerHTML='<p>加载失败 '+r.status+'</p>';return;}
 const d=await r.json(),a=d.asset,sf=d.source_file;
 const meta=['asset_id','kind','fmt','pack_id','status','preset_clean_verdict','preset_grade_family',
   'preset_look_name','preset_caption','preset_pro_rate','preset_intent_rate','preset_coh_rate',
   'preset_vote_pro','preset_vote_intent','preset_vote_coh','preset_reliable_probes',
   'preset_edit_dispersion','preset_near_noop','lut_size','preset_content_hash','scene_affinity',
   'is_bw','is_technical','has_local_mask','split','path'];
 const cands=d.candidates||[];
 const noAfter=cands.length&&!cands.some(c=>c.after_exists);
 $('#right').innerHTML=
  '<h2>'+esc(a.preset_look_name||a.asset_id)+' <span class=mut>'+esc(a.asset_id)+'</span></h2>'+
  kvTable(a,meta)+
  (a.preset_axes?'<h3>axes</h3><pre>'+esc(JSON.stringify(a.preset_axes,null,1))+'</pre>':'')+
  '<h3>per_probe 逐通道</h3>'+probeTable(a.preset_per_probe)+
  (a.preset_tag_metrics?'<h3>tag_metrics</h3><pre>'+esc(JSON.stringify(a.preset_tag_metrics,null,1))+'</pre>':'')+
  '<details><summary>llm_qa 记录 ('+d.qa.length+')</summary>'+
    (d.qa.length?'<table><tr><th>run_id</th><th>probe_id</th><th>questionnaire</th><th>item</th><th>answer</th><th>model</th><th>raw/rationale</th></tr>'+
     d.qa.map(q=>'<tr><td>'+esc(q.run_id)+'</td><td>'+esc(q.probe_id)+'</td><td>'+esc(q.questionnaire)+'</td><td>'+esc(q.item)+
     '</td><td>'+esc(q.answer)+'</td><td>'+esc(q.model)+'</td><td>'+esc(q.rationale||q.raw||'')+'</td></tr>').join('')+'</table>'
     :'<p class=mut>无 QA 记录</p>')+'</details>'+
  '<details open><summary>preset 源文件 '+esc(sf.path||'')+
    (sf.bytes!=null?' ('+sf.bytes+' bytes)':'')+(sf.truncated?' — 只显示前 '+sf.shown_lines+' 行，已截断':'')+'</summary>'+
    (!sf.exists?'<p class=warn>源文件不存在</p>':sf.binary?'<p class=warn>二进制文件，只给大小：'+sf.bytes+' bytes</p>'
      :'<pre>'+esc(sf.text||'')+'</pre>')+'</details>'+
  '<h3>canonical 渲染例图 ('+cands.length+')</h3>'+
  (noAfter?'<div class=warn>本机 <code>/mnt/ramstage</code> 已清空，after 图不在本地；'+
    '点「加载例图」让主 viewer (8077) 从归档物化后再出图。before 图只有 MMArt-PPR10k 那批还在本地。</div>':'')+
  (cands.length?'<p><button id=btnprep>加载例图</button> <span id=prepmsg class=mut></span></p>':'')+
  (cands.length?'<div class=grid>'+cands.map(c=>
    '<div class=card><div class=pair>'+
    '<div><div class=mut>before</div>'+(c.source_path?'<img loading=lazy src="'+imgurl(c.source_path,420)+'">':'<div class=mut>（无源图路径）</div>')+'</div>'+
    '<div><div class=mut>after</div><div class=aslot data-gid="'+esc(c.group_id)+'" data-after="'+esc(c.after_path||'')+'">'+
      (!c.after_path?'<div class=slot>（无 after 路径）</div>'
       :c.after_exists?'<img loading=lazy src="'+imgurl(c.after_path,420)+'">'
       :'<div class=slot>未物化<br>点「加载例图」</div>')+'</div></div>'+
    '</div><div class=mut>'+esc(c.build_id)+' · rank '+esc(c.rank)+(c.winner?' · <b>winner</b>':'')+
    ' · '+esc(c.major||'')+'/'+esc(c.minor||'')+' · '+esc(c.style_name||'')+'</div>'+
    '<div class=mut>after: '+esc(c.after_path||'')+'</div>'+
    '<div class=mut>src: '+esc(c.source_path||'（group 未查到源图路径）')+'</div></div>').join('')+'</div>'
   :'<p class=mut>该 preset 在 canonical build 里没有候选记录</p>');
 if($('#btnprep')) $('#btnprep').onclick=()=>prepare(id);
}

// --- 例图物化：借道 8077 的 prepare 队列（单 worker FIFO，主人也在用，别狂轰） ---
function fillReady(gids){
 document.querySelectorAll('.aslot').forEach(el=>{
  if(!gids.has(el.dataset.gid)||!el.dataset.after||el.querySelector('img')) return;
  el.innerHTML='<img loading=lazy src="/api/dbvimg?path='+encodeURIComponent(el.dataset.after)+'&w=420">';
 });
}
function markFailed(gids,label){
 document.querySelectorAll('.aslot').forEach(el=>{
  if(gids.has(el.dataset.gid)&&!el.querySelector('img'))
    el.innerHTML='<div class=slot>'+esc(label)+'</div>';
 });
}
async function prepare(id){
 if(prepTimer){clearTimeout(prepTimer);prepTimer=null}
 const btn=$('#btnprep'),msg=$('#prepmsg');
 btn.disabled=true; msg.className='mut'; msg.textContent='提交 prepare…';
 const bad=t=>{msg.className='warn';msg.textContent=t;btn.disabled=false};
 let s;
 try{ s=await (await fetch('/api/preset/'+encodeURIComponent(id)+'/prepare',{method:'POST'})).json(); }
 catch(e){ bad('提交失败：'+e); return; }
 if(s.reachable===false){
   bad('主 viewer '+s.dbv+' 不可达（'+((s.groups[0]||{}).error||'')+'），例图无法物化');
   markFailed(new Set(s.groups.map(g=>g.group_id)),'8077 不可达'); return;
 }
 let polls=0;
 const tick=async()=>{
  if(cur!==id) return;                       // 用户已切走，停轮询
  let st;
  try{ st=await (await fetch('/api/preset/'+encodeURIComponent(id)+'/prepare')).json(); }
  catch(e){ bad('轮询失败：'+e); return; }
  fillReady(new Set(st.groups.filter(g=>g.state==='ready').map(g=>g.group_id)));
  markFailed(new Set(st.groups.filter(g=>g.state==='failed'||g.state==='error').map(g=>g.group_id)),'物化失败');
  const byState={}; st.groups.forEach(g=>byState[g.state]=(byState[g.state]||0)+1);
  msg.textContent='prepare '+st.n_ready+'/'+st.n_total+' ready · '+
    Object.entries(byState).map(([k,v])=>k+':'+v).join(' ')+(st.all_done?'':' · 轮询中…');
  if(st.all_done){ btn.disabled=false; btn.textContent='重新加载例图'; return; }
  if(++polls>=60){ msg.textContent+=' · 已轮询 3 分钟，停止（可再点按钮）'; btn.disabled=false; return; }
  prepTimer=setTimeout(tick,3000);
 };
 msg.textContent='已受理 '+s.n_total+' 个 group，轮询中…';
 prepTimer=setTimeout(tick,300);
}
for(const s of ['#f_verdict','#f_fmt','#f_pack','#f_sort']) $(s).onchange=()=>{page=1;load()};
let t; $('#f_q').oninput=()=>{clearTimeout(t);t=setTimeout(()=>{page=1;load()},300)};
$('#prev').onclick=()=>{if(page>1){page--;load()}};
$('#next').onclick=()=>{if(page*50<total){page++;load()}};
load().then(()=>{const m=location.hash.match(/^#\/preset\/(.+)$/); if(m) open_(decodeURIComponent(m[1]));});
</script>
"""


@app.get("/", response_class=HTMLResponse)
def index():
    return HTMLResponse(INDEX_HTML)


# --- selfcheck -------------------------------------------------------------- #
def _selfcheck_prepare(client, pid: str, budget_s: int = 120) -> None:
    """8077 prepare 链路。8077 不在 / 超时 → 打 WARN 跳过，不判失败。

    `limit=1`：只让 8077 物化一个 group，别占它的单 worker 队列。
    """
    import time

    r = client.post(f"/api/preset/{pid}/prepare", params={"limit": 1})
    if r.status_code != 200:
        print(f"[WARN] prepare 段跳过：POST -> {r.status_code}")
        return
    s = r.json()
    if not s["groups"]:
        print("[WARN] prepare 段跳过：该 preset 无 group")
        return
    if not s["reachable"]:
        print(f"[WARN] prepare 段跳过：主 viewer {s['dbv']} 不可达 "
              f"({s['groups'][0].get('error')})")
        return
    g0 = s["groups"][0]
    print(f"[ok] POST /api/preset/{pid}/prepare (limit=1) -> "
          f"{g0['group_id'][:22]}… state={g0['state']}")

    t0, st = time.time(), None
    while time.time() - t0 < budget_s:
        st = client.get(f"/api/preset/{pid}/prepare", params={"limit": 1}).json()
        if st["all_done"]:
            break
        time.sleep(3)
    if not st or not st["all_done"]:
        print(f"[WARN] prepare 段跳过：{budget_s}s 内未 ready "
              f"(state={(st or {}).get('groups', [{}])[0].get('state')})；"
              "8077 是单 worker FIFO，可能排在主人的任务后面")
        return
    g = st["groups"][0]
    if g["state"] != "ready":
        print(f"[WARN] prepare 段跳过：group 终态 {g['state']} "
              f"({g.get('message') or g.get('error')})")
        return
    print(f"[ok] 轮询至 ready ({round(time.time() - t0, 1)}s, "
          f"bytes={(g.get('bytes') or {}).get('total')})")

    after = next((c["after_path"] for c in fetch_candidates(pid)
                  if c.get("group_id") == g["group_id"] and c.get("after_path")), None)
    assert after, "ready 的 group 里找不到 after_path"
    for params in ({"path": after}, {"path": after, "w": 420}):
        resp = client.get("/api/dbvimg", params=params)
        assert resp.status_code == 200, \
            f"/api/dbvimg {params} -> {resp.status_code} {resp.text[:120]}"
        assert len(resp.content) > 1000, "/api/dbvimg 返回体过小"
    assert client.get("/api/dbvimg",
                      params={"path": "/etc/passwd"}).status_code == 403
    print(f"[ok] /api/dbvimg 200（原图 + w=420）+ 白名单外 403: {after}")


def selfcheck() -> int:
    from fastapi.testclient import TestClient

    client = TestClient(app)

    n = client.get("/healthz").json()["presets"]
    assert n > 10000, f"preset 总数 {n} <= 10000"
    print(f"[ok] vera_source_qa.assets asset_type='preset' = {n}")

    r = client.get("/api/presets?page=1&page_size=5").json()
    assert r["items"], "列表为空"
    print(f"[ok] /api/presets total={r['total']} "
          f"facets: verdict={len(r['facets']['verdict'])} "
          f"fmt={len(r['facets']['fmt'])} pack={len(r['facets']['pack'])}")

    # ORDER BY 让重复自检永远打同一个 group，避免反复占用 8077 的单 worker 队列。
    with research_conn() as c:
        pid = c.execute("SELECT preset_id FROM canonical_candidates "
                        "WHERE preset_id IS NOT NULL ORDER BY preset_id "
                        "LIMIT 1").fetchone()["preset_id"]
    d = client.get(f"/api/preset/{pid}")
    assert d.status_code == 200, f"/api/preset/{pid} -> {d.status_code}"
    d = d.json()
    cands = d["candidates"]
    assert len(cands) >= 1, f"{pid} 无 candidates"
    print(f"[ok] /api/preset/{pid}: candidates={len(cands)} qa={len(d['qa'])} "
          f"source_file_exists={d['source_file'].get('exists')}")

    # /img：优先拿一张真实存在的 after 图；ramstage 已清空时退到 group 源图。
    probe, kind = None, None
    with research_conn() as c:
        rows = c.execute("SELECT after_path FROM canonical_candidates "
                         "WHERE after_path IS NOT NULL LIMIT 200").fetchall()
        for r_ in rows:
            if os.path.exists(r_["after_path"]):
                probe, kind = r_["after_path"], "after_path"
                break
        if probe is None:
            for r_ in c.execute("SELECT source_path FROM canonical_groups "
                                "WHERE source_path IS NOT NULL LIMIT 3000").fetchall():
                if os.path.exists(r_["source_path"]):
                    probe, kind = r_["source_path"], "group source_path"
                    break
    assert probe, "找不到任何仍在本地的 canonical 图片，/img 无法验证"
    if kind != "after_path":
        print("[WARN] /mnt/ramstage 已清空，没有任何 after_path 落在本地；"
              "改用 group source_path 验证 /img（详见 NOTES.md D1）")
    assert client.get("/img", params={"path": probe}).status_code == 200, "/img 原图非 200"
    assert client.get("/img", params={"path": probe, "w": 256}).status_code == 200, "/img 缩略非 200"
    print(f"[ok] /img 200 on existing {kind}: {probe}")

    assert client.get("/img", params={"path": "/etc/passwd"}).status_code == 403
    assert client.get("/img", params={
        "path": "/home/bc/data/datasets/../../../etc/passwd"}).status_code == 403
    print("[ok] /img 白名单外路径 + ../ 穿越均 403")

    assert "preset viewer" in client.get("/").text
    print("[ok] / 返回内嵌 HTML")

    _selfcheck_prepare(client, pid)

    # 只读断言
    for name, dsn in (("vera_source_qa", QA_DSN), ("research", RESEARCH_DSN)):
        with psycopg.connect(dsn) as c:
            ro = c.execute("SHOW default_transaction_read_only").fetchone()[0]
            assert ro == "on", f"{name} 连接不是只读: {ro}"
    print("[ok] 两库连接 default_transaction_read_only=on")

    print("OK")
    return 0


# --- main ------------------------------------------------------------------- #
def main(argv: Optional[list[str]] = None) -> int:
    global QA_DSN, RESEARCH_DSN, DBV_BASE
    ap = argparse.ArgumentParser(description="独立只读 preset 浏览界面")
    ap.add_argument("--config", required=True,
                    help="databuild TOML，读其 [viewer].postgres_dsn（0600，路径不写死）")
    ap.add_argument("--dbv", default="http://127.0.0.1:8077",
                    help="主 databuild viewer 基址；after 例图借它的 prepare 通道物化")
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=8078)
    ap.add_argument("--selfcheck", action="store_true")
    ap.add_argument("--no-index", action="store_true",
                    help="跳过启动时的 CREATE INDEX IF NOT EXISTS 兜底")
    a = ap.parse_args(argv)

    with open(a.config, "rb") as f:
        base = (tomllib.load(f).get("viewer") or {}).get("postgres_dsn")
    if not base:
        raise SystemExit(f"{a.config}: 缺少 [viewer].postgres_dsn")
    QA_DSN = _dsn_for(base, QA_DB)
    RESEARCH_DSN = _dsn_for(base, RESEARCH_DB)
    DBV_BASE = a.dbv.rstrip("/")

    if not a.no_index:
        print(f"[index] canonical_candidates_preset_idx: "
              f"{ensure_index(_dsn_for(base, RESEARCH_DB))}", flush=True)

    if a.selfcheck:
        return selfcheck()

    import uvicorn
    print(f"[serve] http://{a.host}:{a.port}  (dbv={DBV_BASE})", flush=True)
    uvicorn.run(app, host=a.host, port=a.port, log_level="info")
    return 0


if __name__ == "__main__":
    sys.exit(main())
