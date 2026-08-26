"""EPR-049 questionnaire v3: the human replays the v3 tournament, source image included.

``build`` reads **Phase A only** (``stage_render_qa.jsonl``), so one human schedule serves
gemini, terra and OneAlign alike and can be built before any arm finishes.

Per group, five items:

* **prelim** ``<gid>_p0..p3`` — the source shown on the left as an unlabelled reference,
  then FIVE numbered options = that quadruple's 4 candidates **plus the source itself**,
  blind-shuffled.  The rater picks one.  Picking the source means "no edit here beats
  leaving it alone".
* **final** ``<gid>_f`` — only the candidates that beat the source in their own quadruple
  (0..4 of them).  With >=2 the rater picks top1 and top2; with <=1 the page skips the
  item automatically and records why, mirroring the model arms exactly.

The quadruple split is ``epr049_build_groups.quadruples`` itself (imported, not
re-derived).  The in-item option order is an independent ``sha1(item_key, item)`` shuffle,
and the final's order is seeded on the *set* of surviving candidate ids so it cannot encode
the order they were won in.

Blinding: thumbnails are named ``sha1(group_key, item)``; **the source gets two different
filenames** — one for the reference slot, one for the option slot — so a rater cannot
identify the source option by comparing URLs.  No preset id, model name, arm name, score
or rank appears in any filename, caption or DOM node.

Usage::

    python -m dataset_build.tools.epr049_questionnaire_v3 build \
        --build-root /home/bc/data/builds/epr049-aesth-20260825 \
        --out-dir docs/assets/epr049_aesthq_v3_20260825
    python -m dataset_build.tools.epr049_questionnaire_v3 analyze \
        --csv .../aesthq.csv --item-key .../item_key.json \
        --arm gemini=.../groups.v3.gemini.jsonl --arm terra=.../groups.v3.terra.jsonl
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
from collections import Counter
from fractions import Fraction
from itertools import combinations
from math import comb
from pathlib import Path
from typing import Any, Mapping, Sequence

SCHEMA = "epr049-aesthetic-questionnaire-v3"
KEY_PREFIX = "v3"
THUMB_SHORT_EDGE = 512
THUMB_QUALITY = 90
CSV_HEADER = "item_id,kind,pick1,pick2,notes"
QUAD_SIZE = 4
QUADS_PER_GROUP = 4
CANDIDATES_PER_GROUP = 16
PRELIM_OPTIONS = 5          # 4 candidates + the source
ITEMS_PER_GROUP = QUADS_PER_GROUP + 1
SOURCE_ITEM = "src"

FLOOR_M1 = 1.0 / PRELIM_OPTIONS      # rank1 of 5
FLOOR_M8 = 1.0 / PRELIM_OPTIONS      # source is 1 of 5 options
FLOOR_M3 = 2.0 / CANDIDATES_PER_GROUP
FLOOR_M5_PAIR_ORDER = 0.5
FLOOR_TAU = 0.0


def sha1_hex(*parts: object) -> str:
    return hashlib.sha1("\x1f".join(str(p) for p in parts).encode("utf-8")).hexdigest()


def _pad(text: str, width: int) -> str:
    import unicodedata
    shown = sum(2 if unicodedata.east_asian_width(c) in "WF" else 1 for c in text)
    return text + " " * max(1, width - shown)


def _js_json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True).replace("</", "<\\/")


def blind_order(item_key: str, items: Sequence[str]) -> list[str]:
    return sorted(items, key=lambda i: (sha1_hex(item_key, i), i))


def final_order(candidate_ids: Sequence[str]) -> list[int]:
    """Display order of the survivors, keyed by the *set* of their candidate ids."""
    seed = sha1_hex("epr049-final-v3", *sorted(candidate_ids))
    return sorted(range(len(candidate_ids)),
                  key=lambda i: (sha1_hex(seed, candidate_ids[i]), candidate_ids[i]))


# ------------------------------------------------------------------ chance floors


def hypergeometric_jaccard(pool: int, size_a: int, size_b: int) -> dict[str, Any]:
    total = comb(pool, size_b)
    terms, expectation = [], 0.0
    for k in range(0, min(size_a, size_b) + 1):
        ways = comb(size_a, k) * comb(pool - size_a, size_b - k)
        if not ways:
            continue
        p = ways / total
        j = k / (size_a + size_b - k) if (size_a + size_b - k) else 0.0
        expectation += p * j
        terms.append({"k": k, "p": round(p, 8), "jaccard": round(j, 8)})
    return {"formula": (f"P(k)=C({size_a},k)*C({pool-size_a},{size_b}-k)/C({pool},{size_b}); "
                        f"J=k/({size_a}+{size_b}-k)"),
            "terms": terms, "expectation": round(expectation, 8)}


FLOOR_M4 = hypergeometric_jaccard(CANDIDATES_PER_GROUP, 2, 2)


def m2_floor(a: int, b: int) -> float:
    """Exact E[Jaccard] of two finalist sets under the tournament's own null.

    A finalist exists for a quadruple only if the chooser picked a candidate rather than
    the source, so sizes are 0..4 and each finalist comes from a *distinct* quadruple.
    Conditioning on the observed sizes ``a`` and ``b``:
        m = #quadruples where both produced a finalist ~ Hypergeometric(4, a, b)
        k | m ~ Binomial(m, 1/4)     (same candidate out of that quadruple's four)
        J = k / (a + b - k)
    This is why a plain 4-of-16 hypergeometric would be the wrong floor here.
    """
    if a == 0 or b == 0:
        return 0.0
    total = Fraction(0)
    for m in range(0, min(a, b) + 1):
        if b - m > 4 - a:
            continue
        pm = Fraction(comb(a, m) * comb(4 - a, b - m), comb(4, b))
        for k in range(0, m + 1):
            pk = Fraction(comb(m, k)) * Fraction(1, 4) ** k * Fraction(3, 4) ** (m - k)
            j = Fraction(k, a + b - k) if (a + b - k) else Fraction(0)
            total += pm * pk * j
    return float(total)


# ---------------------------------------------------------------------------- thumbnails


class Thumbnails:
    def __init__(self, out_dir: Path, short_edge: int):
        self.dir = out_dir / "imgs"
        self.dir.mkdir(parents=True, exist_ok=True)
        self.short_edge = short_edge
        self.written: set[str] = set()
        self.failed: list[str] = []

    def rel(self, source: Path, blind_name: str) -> str | None:
        from PIL import Image, ImageOps
        target = self.dir / f"{blind_name}.jpg"
        if target.is_file() and target.stat().st_size > 0:
            self.written.add(blind_name)
            return f"imgs/{target.name}"
        try:
            with Image.open(source) as handle:
                image = ImageOps.exif_transpose(handle).convert("RGB")
                scale = self.short_edge / min(image.size)
                if scale < 1.0:
                    size = tuple(max(1, int(round(v * scale))) for v in image.size)
                    r = getattr(Image, "Resampling", Image)
                    image = image.resize(size, r.LANCZOS)
                image.save(target, "JPEG", quality=THUMB_QUALITY)
        except Exception as exc:  # noqa: BLE001
            self.failed.append(f"{blind_name}:{type(exc).__name__}")
            return None
        self.written.add(blind_name)
        return f"imgs/{target.name}"


# --------------------------------------------------------------------------------- page

_STYLE = """
:root{color-scheme:light}
*{box-sizing:border-box}
body{font-family:system-ui,'Noto Sans CJK SC',sans-serif;margin:0;padding:14px 18px;
background:#fff;color:#111}
h1{font-size:18px;margin:0 0 4px}
.hint{font-size:13px;color:#444;margin:0 0 8px}
.topbar{display:flex;align-items:center;gap:14px;flex-wrap:wrap;margin-bottom:6px}
#progress{font-size:20px;font-weight:700;font-variant-numeric:tabular-nums}
#done,#unrated,#task,#stage{font-size:13px;color:#444}
#task{font-weight:600;color:#111}
.track{height:6px;background:#e6e6e6;border-radius:3px;overflow:hidden;margin:0 0 10px}
#bar{height:100%;background:#2b6cb0;width:0}
.wrap{display:grid;grid-template-columns:minmax(180px,22%) 1fr;gap:16px;align-items:start}
.ref figure,.opts figure{margin:0}
.ref img,.opts img{display:block;width:100%;height:auto;border:1px solid #ccc;background:#202020}
.ref figcaption{font-size:13px;font-weight:600;color:#333;margin-top:4px;text-align:center}
.opts{display:grid;gap:10px}
.opts figcaption{font-size:13px;color:#333;margin-top:4px;text-align:center}
.picks{display:flex;gap:8px;flex-wrap:wrap;margin:12px 0 4px}
.pick{flex:1 1 110px;padding:12px 10px;font-size:15px;cursor:pointer;border:1px solid #bbb;
border-radius:8px;background:#fafafa;color:#111;text-align:center}
.pick:hover{background:#f0f4f8}
.pick.one{background:#2b6cb0;border-color:#2b6cb0;color:#fff}
.pick.two{background:#7fb0d8;border-color:#7fb0d8;color:#111}
.legend{font-size:12px;color:#666;margin:0 0 8px}
.skip{padding:14px;border:1px dashed #888;border-radius:8px;color:#333;font-size:14px;
background:#f7f7f7}
.nav{display:flex;gap:10px;align-items:center;flex-wrap:wrap;margin:10px 0}
button{font-family:inherit}
.nav button,#export{padding:10px 16px;font-size:14px;cursor:pointer;border:1px solid #bbb;
border-radius:6px;background:#fafafa}
.nav button:disabled{opacity:.4;cursor:default}
#notes{width:100%;padding:6px;font-size:13px;font-family:inherit;border:1px solid #ccc;
border-radius:6px}
.keys{font-size:12px;color:#666}
"""

_SCRIPT = """
const ITEMS = __ITEMS__;
const VIEW = __VIEW__;
const GROUPS = __GROUPS__;
const STORE_KEY = __STORE_KEY__;
const total = ITEMS.length;
let cursor = 0;
let state = {picks: {}, notes: {}, cursor: 0};

function load() {
  try {
    const raw = window.localStorage.getItem(STORE_KEY);
    if (raw) {
      const p = JSON.parse(raw);
      state.picks = (p && p.picks) || {};
      state.notes = (p && p.notes) || {};
      cursor = Math.min(Math.max(parseInt(p && p.cursor, 10) || 0, 0), Math.max(total-1, 0));
    }
  } catch (err) { console.warn('localStorage load failed', err); }
}
function save() {
  state.cursor = cursor;
  try { window.localStorage.setItem(STORE_KEY, JSON.stringify(state)); }
  catch (err) { console.warn('localStorage save failed', err); }
}

function prelimPicks(group) {
  const out = [];
  for (let qi = 0; qi < 4; qi += 1) {
    const p = (state.picks[group + '_p' + qi] || {}).p1;
    if (!p) { return null; }
    out.push(p);
  }
  return out;
}

/* Surviving candidates in final display order; [] when every quadruple went to the
   source. Returns null while the group's prelims are unfinished. */
function finalSlots(group) {
  const picks = prelimPicks(group);
  if (!picks) { return null; }
  const g = GROUPS[group];
  const ord = g.forder[picks.join('')];
  if (ord === undefined) { return null; }
  const out = [];
  for (let i = 0; i < ord.length; i += 1) {
    const qi = parseInt(ord.charAt(i), 10);
    out.push(g.quads[qi][picks[qi] - 1]);
  }
  return out;
}

function complete(iid) {
  const v = VIEW[iid];
  const pick = state.picks[iid] || {};
  if (v.kind === 'final') {
    const s = finalSlots(v.group);
    if (s === null) { return false; }
    if (s.length < 2) { return true; }   /* auto-skipped, counts as done */
    return !!pick.p1 && !!pick.p2;
  }
  return !!pick.p1;
}
function doneCount() { let n = 0; for (const i of ITEMS) { if (complete(i)) n += 1; } return n; }

function render() {
  const iid = ITEMS[cursor];
  const v = VIEW[iid];
  const g = GROUPS[v.group];
  const ref = document.getElementById('ref');
  const opts = document.getElementById('opts');
  const picksBox = document.getElementById('picks');
  const skip = document.getElementById('skip');
  ref.innerHTML = ''; opts.innerHTML = ''; picksBox.innerHTML = ''; skip.style.display = 'none';

  const rf = document.createElement('figure');
  const ri = document.createElement('img');
  ri.src = g.srcref; ri.alt = '原图';
  const rc = document.createElement('figcaption'); rc.textContent = '原图（参考）';
  rf.appendChild(ri); rf.appendChild(rc); ref.appendChild(rf);

  let shown = null;
  if (v.kind === 'final') {
    const s = finalSlots(v.group);
    if (s === null) {
      skip.style.display = 'block';
      skip.textContent = '本组前 4 道题还没做完，做完后这道决胜题会自动出现。';
    } else if (s.length === 0) {
      skip.style.display = 'block';
      skip.textContent = '本组四道初赛你都选了原图，没有候选进入决胜轮 —— 本题自动跳过。';
    } else if (s.length === 1) {
      skip.style.display = 'block';
      skip.textContent = '本组只有 1 张候选进入决胜轮，它自动成为第一名 —— 本题自动跳过。';
    } else { shown = s; }
  } else { shown = v.opts; }

  if (shown) {
    opts.style.gridTemplateColumns = 'repeat(' + Math.min(shown.length, 3) + ',1fr)';
    for (let i = 0; i < shown.length; i += 1) {
      const fig = document.createElement('figure');
      const img = document.createElement('img');
      img.src = g.thumb[shown[i]];
      const cap = '候选 ' + (i + 1);
      img.alt = cap;
      const fc = document.createElement('figcaption'); fc.textContent = cap;
      fig.appendChild(img); fig.appendChild(fc); opts.appendChild(fig);
    }
    const cur = state.picks[iid] || {};
    for (let i = 1; i <= shown.length; i += 1) {
      const b = document.createElement('button');
      b.type = 'button'; b.className = 'pick'; b.textContent = '候选 ' + i;
      if (cur.p1 === i) { b.classList.add('one'); } else if (cur.p2 === i) { b.classList.add('two'); }
      b.addEventListener('click', function () { choose(i); });
      picksBox.appendChild(b);
    }
  }

  document.getElementById('task').textContent = v.kind === 'final'
    ? '决胜题：在你自己选出的候选里，先点最好的（深蓝），再点第二好的（浅蓝）'
    : '初赛题：五张里点你认为最好的一张。左边是原图参考；如果你觉得五张里最好的就是没修过的那张，就选它。';
  document.getElementById('stage').textContent = v.kind === 'final'
    ? '本组第 5 / 5 题' : ('本组第 ' + (v.quad + 1) + ' / 5 题');
  document.getElementById('legend').textContent = v.kind === 'final'
    ? '深蓝 = 第一名，浅蓝 = 第二名；再点一次可取消。' : '再点一次可取消。';
  document.getElementById('progress').textContent = (cursor + 1) + ' / ' + total;
  const done = doneCount(); const left = total - done;
  document.getElementById('done').textContent = '已完成 ' + done + ' / ' + total;
  document.getElementById('bar').style.width = (total ? (done*100/total) : 0) + '%';
  document.getElementById('notes').value = state.notes[iid] || '';
  document.getElementById('prev').disabled = cursor <= 0;
  document.getElementById('next').disabled = cursor >= total - 1;
  document.getElementById('export').textContent =
    left > 0 ? ('导出 CSV（还剩 ' + left + ' 条未完成）') : '导出 CSV';
  document.getElementById('unrated').textContent =
    left > 0 ? ('未完成 ' + left + ' 条，仍可导出') : '全部已完成';
}

function choose(value) {
  const iid = ITEMS[cursor];
  const v = VIEW[iid];
  const pick = Object.assign({}, state.picks[iid] || {});
  if (v.kind === 'final') {
    if (pick.p1 === value) { delete pick.p1; }
    else if (pick.p2 === value) { delete pick.p2; }
    else if (!pick.p1) { pick.p1 = value; }
    else if (!pick.p2) { pick.p2 = value; }
    else { pick.p1 = value; delete pick.p2; }
  } else {
    if (pick.p1 === value) { delete pick.p1; } else { pick.p1 = value; }
    const fid = v.group + '_f';
    if (state.picks[fid]) { delete state.picks[fid]; }  /* survivors changed */
  }
  state.picks[iid] = pick;
  save();
  if (complete(iid) && cursor < total - 1) { cursor += 1; save(); }
  render();
}
function go(d) { const n = cursor + d; if (n < 0 || n >= total) return; cursor = n; save(); render(); }
function csvField(t) {
  const v = String(t == null ? '' : t);
  if (/[",\\r\\n]/.test(v)) { return '"' + v.replace(/"/g, '""') + '"'; }
  return v;
}
function buildCsv() {
  const lines = [__HEADER__];
  for (const iid of ITEMS) {
    const v = VIEW[iid]; const p = state.picks[iid] || {};
    let p1 = p.p1 ? String(p.p1) : '', p2 = p.p2 ? String(p.p2) : '';
    if (v.kind === 'final') {
      const s = finalSlots(v.group);
      if (s !== null && s.length < 2) { p1 = ''; p2 = ''; }  /* auto-skipped */
    }
    lines.push([iid, v.kind, p1, p2, csvField(state.notes[iid] || '')].join(','));
  }
  return lines.join('\\n') + '\\n';
}
function exportCsv() {
  const blob = new Blob([buildCsv()], {type: 'text/csv;charset=utf-8'});
  const url = URL.createObjectURL(blob);
  const a = document.createElement('a');
  a.href = url; a.download = 'aesthq.csv';
  document.body.appendChild(a); a.click(); document.body.removeChild(a);
  setTimeout(function () { URL.revokeObjectURL(url); }, 2000);
}
function inField(t) {
  if (!t || !t.tagName) return false;
  const g = t.tagName.toLowerCase();
  return g === 'input' || g === 'textarea';
}
window.addEventListener('DOMContentLoaded', function () {
  load();
  document.getElementById('prev').addEventListener('click', function () { go(-1); });
  document.getElementById('next').addEventListener('click', function () { go(1); });
  document.getElementById('export').addEventListener('click', exportCsv);
  document.getElementById('notes').addEventListener('input', function (e) {
    state.notes[ITEMS[cursor]] = e.target.value; save();
  });
  document.addEventListener('keydown', function (e) {
    if (e.ctrlKey || e.metaKey || e.altKey || inField(e.target)) return;
    if (e.key >= '1' && e.key <= '5') { choose(parseInt(e.key, 10)); e.preventDefault(); }
    else if (e.key === 'ArrowLeft') { go(-1); e.preventDefault(); }
    else if (e.key === 'ArrowRight') { go(1); e.preventDefault(); }
  });
  render();
  window.__READY__ = true;
});
"""


def write_page(out_dir: Path, order, view, groups, store_key: str) -> Path:
    script = (_SCRIPT.replace("__ITEMS__", _js_json(list(order)))
              .replace("__VIEW__", _js_json(dict(view)))
              .replace("__GROUPS__", _js_json(dict(groups)))
              .replace("__STORE_KEY__", _js_json(store_key))
              .replace("__HEADER__", _js_json(CSV_HEADER)))
    parts = [
        '<!doctype html><html lang="zh"><head><meta charset="utf-8">',
        '<meta name="viewport" content="width=device-width,initial-scale=1">',
        "<title>修图效果盲评问卷</title>",
        f"<style>{_STYLE}</style></head><body>",
        "<h1>修图效果盲评问卷</h1>",
        '<p class="hint">左边是<b>原图（参考）</b>。右边是同一张原图的几个版本——'
        '<b>其中可能就有没修过的原图本身</b>。每组 5 题：前 4 题各选一张最好的，'
        "第 5 题在你自己选出的候选里排前二。选择自动保存在本机浏览器，"
        "做完点「导出 CSV」下载 aesthq.csv。</p>",
        '<div class="topbar"><span id="progress">- / -</span>'
        '<span id="stage"></span><span id="done"></span><span id="unrated"></span>'
        '<span class="keys">也可用键盘：1–5 选择，← / → 翻页</span></div>',
        '<div class="track"><div id="bar"></div></div>',
        '<p id="task" class="hint"></p>',
        '<div class="skip" id="skip" style="display:none"></div>',
        '<div class="wrap"><div class="ref" id="ref"></div><div class="opts" id="opts"></div></div>',
        '<div class="picks" id="picks"></div>',
        '<p class="legend" id="legend"></p>',
        '<div class="nav"><button type="button" id="prev">← 上一条</button>'
        '<button type="button" id="next">下一条 →</button>'
        '<button type="button" id="export">导出 CSV</button></div>',
        '<textarea id="notes" rows="2" placeholder="备注（可留空）"></textarea>',
        f"<script>{script}</script>",
        "</body></html>",
    ]
    path = out_dir / "aesthq.html"
    path.write_text("\n".join(parts) + "\n", encoding="utf-8", newline="\n")
    return path


def write_csv(out_dir: Path, order, view) -> Path:
    lines = [CSV_HEADER]
    lines.extend(f"{i},{view[i]['kind']},,," for i in order)
    path = out_dir / "aesthq.csv"
    path.write_text("\n".join(lines) + "\n", encoding="utf-8", newline="\n")
    return path


# -------------------------------------------------------------------------------- build


def read_stage(build_root: Path) -> list[dict[str, Any]]:
    path = build_root / "stage_render_qa.jsonl"
    if not path.is_file():
        raise SystemExit(f"missing {path}")
    latest: dict[str, dict[str, Any]] = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            row = json.loads(line)
            latest[str(row["group_id"])] = row
    return [latest[g] for g in sorted(latest)]


def cmd_build(args: argparse.Namespace) -> int:
    from dataset_build.tools.epr049_build_groups import quadruples

    build_root, out_dir = Path(args.build_root), Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    rows = read_stage(build_root)
    dropped: Counter[str] = Counter()
    thumbs = Thumbnails(out_dir, args.thumb_short_edge)

    group_order: list[str] = []
    view: dict[str, Any] = {}
    groups_js: dict[str, Any] = {}
    items: dict[str, Any] = {}
    group_meta: dict[str, Any] = {}

    for row in rows:
        gid = str(row["group_id"])
        cands = row["candidates"]
        if len(cands) != CANDIDATES_PER_GROUP:
            dropped["candidate_count"] += 1
            continue
        by_slot = {str(c["slot_id"]): c for c in cands}
        gkey = f"{KEY_PREFIX}:{gid}:q"

        # Two different filenames for the same source pixels: a rater must not be able to
        # spot the source option by matching it against the reference image's URL.
        srcref = thumbs.rel(build_root / row["source_asset"], sha1_hex(gkey, "srcref"))
        srcopt = thumbs.rel(build_root / row["source_asset"], sha1_hex(gkey, "srcopt"))
        if srcref is None or srcopt is None:
            dropped["source_thumb_failed"] += 1
            continue
        thumb: dict[str, str] = {SOURCE_ITEM: srcopt}
        broken = False
        for slot, entry in by_slot.items():
            rel = thumbs.rel(build_root / entry["asset"], sha1_hex(gkey, slot))
            if rel is None:
                broken = True
                break
            thumb[slot] = rel
        if broken:
            dropped["candidate_thumb_failed"] += 1
            continue

        quads = quadruples(gid, list(by_slot))
        if len(quads) != QUADS_PER_GROUP or any(len(q) != QUAD_SIZE for q in quads):
            dropped["quad_shape"] += 1
            continue

        quads_display: list[list[str]] = []
        for qi, quad in enumerate(quads):
            item_id = f"{gid}_p{qi}"
            item_key = f"{KEY_PREFIX}:{gid}:p{qi}"
            shown = blind_order(item_key, list(quad) + [SOURCE_ITEM])
            quads_display.append(shown)
            view[item_id] = {"kind": "prelim", "group": gid, "quad": qi, "opts": shown}
            items[item_id] = {
                "item_key": item_key, "group_id": gid, "kind": "prelim",
                "major": row["major"], "quad_index": qi, "quad_slots": list(quad),
                "options": shown,
                "position_to_item": {str(i + 1): s for i, s in enumerate(shown)},
                "source_label": shown.index(SOURCE_ITEM) + 1,
            }

        # All 5^4 = 625 prelim answer combinations pre-resolved into the survivor order.
        forder: dict[str, str] = {}
        for p0 in range(PRELIM_OPTIONS):
            for p1 in range(PRELIM_OPTIONS):
                for p2 in range(PRELIM_OPTIONS):
                    for p3 in range(PRELIM_OPTIONS):
                        picks = (p0, p1, p2, p3)
                        surv = [qi for qi in range(QUADS_PER_GROUP)
                                if quads_display[qi][picks[qi]] != SOURCE_ITEM]
                        key = "".join(str(p + 1) for p in picks)
                        if not surv:
                            forder[key] = ""
                            continue
                        ids = [by_slot[quads_display[qi][picks[qi]]]["candidate_id"]
                               for qi in surv]
                        forder[key] = "".join(str(surv[i]) for i in final_order(ids))

        fid = f"{gid}_f"
        view[fid] = {"kind": "final", "group": gid, "quad": QUADS_PER_GROUP}
        items[fid] = {
            "item_key": f"{KEY_PREFIX}:{gid}:f", "group_id": gid, "kind": "final",
            "major": row["major"],
            "prelim_items": [f"{gid}_p{i}" for i in range(QUADS_PER_GROUP)],
        }
        groups_js[gid] = {"srcref": srcref, "thumb": thumb,
                          "quads": quads_display, "forder": forder}
        group_meta[gid] = {
            "major": row["major"], "source_id": row["source"]["source_id"],
            "scene": row["source"].get("scene"),
            "quads_display": quads_display, "final_order_table": forder,
            "slot_to_candidate_id": {s: c["candidate_id"] for s, c in by_slot.items()},
            "onealign_scores": row["onealign"]["scores"],
            "onealign_source_score": row["onealign"].get("source_score"),
        }
        group_order.append(gid)

    group_order.sort(key=lambda g: (sha1_hex("epr049-groupshuffle-v3", g), g))
    display_order = [i for g in group_order
                     for i in ([f"{g}_p{k}" for k in range(QUADS_PER_GROUP)] + [f"{g}_f"])]

    key_payload = {
        "schema": SCHEMA, "key_prefix": KEY_PREFIX, "build_root": str(build_root),
        "thumb_short_edge": args.thumb_short_edge,
        "prelim_options": PRELIM_OPTIONS,
        "groups_total": len(rows), "groups_used": len(group_order),
        "groups_dropped": dict(sorted(dropped.items())),
        "items_per_group": ITEMS_PER_GROUP,
        "display_order": display_order, "group_order": group_order,
        "items": items, "groups": group_meta,
    }
    key_path = out_dir / "item_key.json"
    key_path.write_text(json.dumps(key_payload, ensure_ascii=False, indent=2,
                                   sort_keys=True) + "\n", encoding="utf-8", newline="\n")
    csv_path = write_csv(out_dir, display_order, view)
    page = write_page(out_dir, display_order, view, groups_js, f"{SCHEMA}:{build_root.name}")

    src_labels = Counter(items[i]["source_label"] for i in items
                         if items[i]["kind"] == "prelim")
    stats = {
        "schema": SCHEMA, "page": str(page), "csv": str(csv_path), "item_key": str(key_path),
        "groups_total": len(rows), "groups_used": len(group_order),
        "groups_dropped": dict(sorted(dropped.items())),
        "items": len(display_order),
        "items_prelim": sum(1 for i in items.values() if i["kind"] == "prelim"),
        "items_final": sum(1 for i in items.values() if i["kind"] == "final"),
        "decisions_expected": len(group_order) * ITEMS_PER_GROUP,
        "source_label_distribution": dict(sorted(src_labels.items())),
        "thumbnails": len(thumbs.written), "thumbnail_failures": thumbs.failed,
    }
    print(json.dumps(stats, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


# ------------------------------------------------------------------------------ metrics


class MetricRegistry:
    def __init__(self, expected: Sequence[str]):
        self.expected = tuple(expected)
        self.calls: dict[str, dict[str, Any]] = {}

    def record(self, name: str, kind: str, payload: Mapping[str, Any]) -> dict[str, Any]:
        self.calls[name] = {"metric": name, "kind": kind, **payload}
        return self.calls[name]

    def assert_wired(self, expect_n: Mapping[str, int]) -> None:
        missing = [n for n in self.expected if n not in self.calls]
        if missing:
            raise RuntimeError(f"pre-registered metrics never ran: {missing}")
        bad = [f"{n}:{a}" for n, row in sorted(self.calls.items())
               for a in (row.get("arms_expected") or ())
               if a not in (row.get("per_arm") or {})]
        if bad:
            raise RuntimeError(f"metric/arm cells never ran: {bad}")
        mism = []
        for name, row in sorted(self.calls.items()):
            want = expect_n.get(name)
            if want is not None and int(row.get("n_human", -1)) != int(want):
                mism.append(f"{name}(n_human={row.get('n_human')} != {want})")
        if mism:
            raise RuntimeError("metric denominators do not reconcile: " + ", ".join(mism))


def _rate(hits):
    n = len(hits)
    return {"n": n, "value": round(sum(hits)/n, 6) if n else None}


def _mean(vals):
    n = len(vals)
    return {"n": n, "value": round(sum(vals)/n, 6) if n else None}


def jaccard(a, b):
    sa, sb = set(a), set(b)
    u = sa | sb
    return len(sa & sb)/len(u) if u else 0.0


def kendall_tau(a, b):
    items = list(a)
    if sorted(items) != sorted(b) or len(items) < 2:
        return None
    ra = {x: i for i, x in enumerate(a)}
    rb = {x: i for i, x in enumerate(b)}
    c = d = 0
    for l, r in combinations(items, 2):
        s = (ra[l]-ra[r]) * (rb[l]-rb[r])
        if s > 0: c += 1
        elif s < 0: d += 1
    return (c-d)/(c+d) if (c+d) else None


def _quantiles(vals):
    if not vals:
        return {"n_values": 0, "mean": None, "median": None, "min": None, "max": None,
                "histogram": {}}
    o = sorted(vals)
    m = len(o)//2
    med = o[m] if len(o) % 2 else 0.5*(o[m-1]+o[m])
    return {"n_values": len(o), "mean": round(sum(o)/len(o), 6), "median": round(med, 6),
            "min": round(o[0], 6), "max": round(o[-1], 6),
            "histogram": dict(sorted(Counter(round(v, 4) for v in o).items()))}


def restrict(scores: Mapping[str, float], items: Sequence[str]) -> list[str]:
    known = [i for i in items if i in scores]
    return sorted(known, key=lambda i: (-float(scores[i]), i))


# -------------------------------------------------------------------------------- arms


def load_arm(path: Path) -> dict[str, dict[str, Any]]:
    latest: dict[str, dict[str, Any]] = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            r = json.loads(line)
            latest[str(r["group_id"])] = r
    return latest


def arm_view(row: Mapping[str, Any]) -> dict[str, Any] | None:
    """One arm's v3 result for one group."""
    if row.get("status") != "complete":
        return None
    b = row.get("tournament") or row.get("gemini") or {}
    r2 = b.get("round2") or {}
    rank1_by_quad, source_won = {}, {}
    for e in b.get("round1") or []:
        if e.get("ok"):
            rank1_by_quad[int(e["quad_index"])] = str(e["rank1"])
            source_won[int(e["quad_index"])] = bool(e.get("source_won"))
    return {
        "finalists": list(b.get("finalists") or []),
        "ranking": list(r2.get("ranking_slots") or []),
        "top2": list(r2.get("top2") or []),
        "rank1_by_quad": rank1_by_quad,
        "source_won": source_won,
        "n_source_won": int(b.get("n_source_won", 0)),
        "final_note": b.get("final_note"),
    }


def onealign_view(meta: Mapping[str, Any]) -> dict[str, Any]:
    """OneAlign as an arm, v3: the source competes inside each quadruple."""
    scores = dict(meta["onealign_scores"])
    src = meta.get("onealign_source_score")
    if src is not None:
        scores[SOURCE_ITEM] = src
    return {"scores": scores}


# ------------------------------------------------------------------------------ analyze


def read_answers(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as h:
        return [{k: str(r.get(k) or "").strip()
                 for k in ("item_id", "kind", "pick1", "pick2", "notes")}
                for r in csv.DictReader(h)]


def _pos(raw: str, options: int) -> int | None:
    try:
        v = int(raw)
    except (TypeError, ValueError):
        return None
    return v if 1 <= v <= options else None


def cmd_analyze(args: argparse.Namespace) -> int:
    csv_path = Path(args.csv)
    key_path = Path(args.item_key or (csv_path.parent / "item_key.json"))
    key = json.loads(key_path.read_text(encoding="utf-8"))
    items, gmeta = key["items"], key["groups"]
    rows = read_answers(csv_path)

    arms: dict[str, dict[str, dict[str, Any]]] = {}
    for spec in args.arm or []:
        name, _, path = spec.partition("=")
        if not path:
            raise SystemExit(f"--arm expects name=path, got {spec!r}")
        arms[name] = load_arm(Path(path))
    model_arms = sorted(arms)
    arm_names = model_arms + ["onealign"]

    prelim_pick: dict[str, dict[int, str]] = {}
    prelim_pos: dict[str, dict[int, int]] = {}
    final_raw: dict[str, tuple[int, int]] = {}
    rejected: Counter[str] = Counter()
    unknown: list[str] = []
    for row in rows:
        entry = items.get(row["item_id"])
        if entry is None:
            unknown.append(row["item_id"]); continue
        kind = entry["kind"]
        if row["kind"] and row["kind"] != kind:
            rejected["kind_mismatch"] += 1; continue
        gid = entry["group_id"]
        if kind == "prelim":
            p = _pos(row["pick1"], PRELIM_OPTIONS)
            if p is None:
                rejected["prelim_missing_pick1"] += 1; continue
            prelim_pick.setdefault(gid, {})[entry["quad_index"]] = \
                entry["position_to_item"][str(p)]
            prelim_pos.setdefault(gid, {})[entry["quad_index"]] = p
        else:
            p1 = _pos(row["pick1"], QUAD_SIZE)
            p2 = _pos(row["pick2"], QUAD_SIZE)
            final_raw[gid] = (p1 or 0, p2 or 0)

    # ---- human prelim records (one per answered quadruple)
    prelim_records = []
    for gid, per in sorted(prelim_pick.items()):
        for qi, item in sorted(per.items()):
            prelim_records.append({
                "group_id": gid, "major": gmeta[gid]["major"], "quad_index": qi,
                "human_top1": item, "human_source_won": item == SOURCE_ITEM,
                "options": gmeta[gid]["quads_display"][qi],
            })

    # ---- human final records, only for groups with all 4 prelims answered
    final_records, final_skipped = [], Counter()
    for gid, per in sorted(prelim_pick.items()):
        if len(per) != QUADS_PER_GROUP:
            final_skipped["prelims_incomplete"] += 1; continue
        meta = gmeta[gid]
        pos_key = "".join(str(prelim_pos[gid][q]) for q in range(QUADS_PER_GROUP))
        ordr = meta["final_order_table"].get(pos_key)
        if ordr is None:
            rejected["final_order_key_missing"] += 1; continue
        surv = [per[int(c)] for c in ordr]
        rec = {"group_id": gid, "major": meta["major"], "human_finalists": surv,
               "n_finalists": len(surv)}
        if len(surv) == 0:
            final_skipped["all_source_won"] += 1
        elif len(surv) == 1:
            final_skipped["single_finalist"] += 1
            rec["human_top1"] = surv[0]; rec["human_top2"] = None
        else:
            p1, p2 = final_raw.get(gid, (0, 0))
            if not (1 <= p1 <= len(surv)) or not (1 <= p2 <= len(surv)) or p1 == p2:
                rejected["final_missing_or_duplicate_pick"] += 1; continue
            rec["human_top1"] = surv[p1-1]
            rec["human_top2"] = [surv[p1-1], surv[p2-1]]
        final_records.append(rec)

    ranked_finals = [r for r in final_records if r.get("human_top2")]
    accepted = {"prelim": len(prelim_records), "final_rows": len(final_records),
                "final_ranked": len(ranked_finals)}

    def av(gid, arm):
        if arm == "onealign":
            return onealign_view(gmeta[gid])
        r = arms[arm].get(gid)
        return arm_view(r) if r else None

    def arm_rank1(gid, arm, qi, options):
        v = av(gid, arm)
        if v is None:
            return None, None
        if arm == "onealign":
            o = restrict(v["scores"], options)
            return (o[0] if o else None), v
        return v["rank1_by_quad"].get(qi), v

    def arm_finalists(gid, arm):
        v = av(gid, arm)
        if v is None:
            return None
        if arm == "onealign":
            out = []
            for qi, opts in enumerate(gmeta[gid]["quads_display"]):
                o = restrict(v["scores"], opts)
                if o and o[0] != SOURCE_ITEM:
                    out.append(o[0])
            return out
        return v["finalists"]

    def arm_top2(gid, arm):
        if arm == "onealign":
            f = arm_finalists(gid, arm)
            if f is None or len(f) < 2:
                return None
            return restrict(onealign_view(gmeta[gid])["scores"], f)[:2]
        v = av(gid, arm)
        if v is None or len(v["top2"]) < 2:
            return None
        return v["top2"]

    reg = MetricRegistry(("M1'", "M2'", "M3'", "M4'", "M5'", "M7'", "M8'"))
    dropped: dict[str, Counter] = {a: Counter() for a in arm_names}

    # M1' prelim top1 agreement (5 options)
    m1 = {}
    for arm in arm_names:
        hits = []
        for r in prelim_records:
            rank1, v = arm_rank1(r["group_id"], arm, r["quad_index"], r["options"])
            if rank1 is None:
                dropped[arm]["M1'_no_arm_result"] += 1; continue
            hits.append(int(r["human_top1"] == rank1))
        m1[arm] = _rate(hits)
    reg.record("M1'", "prelim", {"n_human": len(prelim_records), "per_arm": m1,
                                 "arms_expected": tuple(arm_names),
                                 "floor": round(FLOOR_M1, 6),
                                 "floor_note": "1/5: rank1 of 4 candidates + the source"})

    # M8' source-win rate (per arm, and the human), plus finalist-count distributions
    m8, fdist = {}, {}
    human_src = [int(r["human_source_won"]) for r in prelim_records]
    m8["human"] = _rate(human_src)
    for arm in arm_names:
        hits = []
        for r in prelim_records:
            rank1, _ = arm_rank1(r["group_id"], arm, r["quad_index"], r["options"])
            if rank1 is None:
                dropped[arm]["M8'_no_arm_result"] += 1; continue
            hits.append(int(rank1 == SOURCE_ITEM))
        m8[arm] = _rate(hits)
    fdist["human"] = dict(sorted(Counter(r["n_finalists"] for r in final_records).items()))
    for arm in arm_names:
        c = Counter()
        for gid in sorted(gmeta):
            f = arm_finalists(gid, arm)
            if f is not None:
                c[len(f)] += 1
        fdist[arm] = dict(sorted(c.items()))
    reg.record("M8'", "prelim", {"n_human": len(prelim_records), "per_arm": m8,
                                 "arms_expected": tuple(arm_names),
                                 "finalist_count_distribution": fdist,
                                 "floor": round(FLOOR_M8, 6),
                                 "floor_note": "1/5: the source is one of five options"})

    # M2' finalist-set Jaccard, per-group exact floor under the tournament's own null
    m2, m2_floors = {}, {}
    for arm in arm_names:
        vals, floors, strat = [], [], {}
        for r in final_records:
            f = arm_finalists(r["group_id"], arm)
            if f is None:
                dropped[arm]["M2'_no_arm_result"] += 1; continue
            j = jaccard(r["human_finalists"], f)
            fl = m2_floor(len(r["human_finalists"]), len(f))
            vals.append(j); floors.append(fl)
            k = f"human{len(r['human_finalists'])}_arm{len(f)}"
            strat.setdefault(k, []).append(j)
        m2[arm] = {**_mean(vals),
                   "floor": round(sum(floors)/len(floors), 6) if floors else None,
                   "by_size": {k: {"n": len(v), "value": round(sum(v)/len(v), 6)}
                               for k, v in sorted(strat.items())}}
    reg.record("M2'", "final", {"n_human": len(final_records), "per_arm": m2,
                                "arms_expected": tuple(arm_names),
                                "floor": None,
                                "floor_note": "per-group exact: m~Hypergeom(4,a,b), "
                                              "k|m~Binom(m,1/4), J=k/(a+b-k)"})

    # M3' human final top1 inside arm final top2 (needs human top1 and arm top2)
    m3 = {}
    for arm in arm_names:
        hits = []
        for r in final_records:
            if not r.get("human_top1"):
                continue
            t2 = arm_top2(r["group_id"], arm)
            if t2 is None:
                dropped[arm]["M3'_arm_top2_unavailable"] += 1; continue
            hits.append(int(r["human_top1"] in set(t2)))
        m3[arm] = _rate(hits)
    n_m3 = sum(1 for r in final_records if r.get("human_top1"))
    reg.record("M3'", "final", {"n_human": n_m3, "per_arm": m3,
                                "arms_expected": tuple(arm_names),
                                "floor": round(FLOOR_M3, 6),
                                "floor_note": "2/16, independent of finalist counts"})

    # M4' top2 vs top2 Jaccard (both sides need >=2)
    m4 = {}
    for arm in arm_names:
        vals = []
        for r in ranked_finals:
            t2 = arm_top2(r["group_id"], arm)
            if t2 is None:
                dropped[arm]["M4'_arm_top2_unavailable"] += 1; continue
            vals.append(jaccard(r["human_top2"], t2))
        m4[arm] = _mean(vals)
    reg.record("M4'", "final", {"n_human": len(ranked_finals), "per_arm": m4,
                                "arms_expected": tuple(arm_names),
                                "floor": FLOOR_M4["expectation"],
                                "floor_formula": FLOOR_M4["formula"]})

    # M5' OneAlign restricted to the human's OWN finalists
    t1, ord_hits, t1_floors = [], [], []
    for r in final_records:
        f = r["human_finalists"]
        if len(f) < 2:
            continue
        sc = onealign_view(gmeta[r["group_id"]])["scores"]
        ranked = restrict(sc, f)
        if len(ranked) != len(f):
            dropped["onealign"]["M5'_incomplete_scores"] += 1; continue
        t1.append(int(r["human_top1"] == ranked[0]))
        t1_floors.append(1.0/len(f))
        if r.get("human_top2"):
            rank = {s: i for i, s in enumerate(ranked)}
            a, b = r["human_top2"]
            ord_hits.append(int(rank[a] < rank[b]))
    reg.record("M5'", "final", {
        "n_human": len(ranked_finals), "arms_expected": ("onealign",),
        "per_arm": {"onealign": {
            "top1": {**_rate(t1),
                     "floor": round(sum(t1_floors)/len(t1_floors), 6) if t1_floors else None},
            "pair_order": {**_rate(ord_hits), "floor": FLOOR_M5_PAIR_ORDER}}},
        "floor": None,
        "floor_note": "top1 floor = mean(1/|human finalists|); pair_order = 0.5"})

    # M7' model-vs-model Kendall tau over shared finalists
    m7, m7_drop = {}, Counter()
    for left, right in combinations(arm_names, 2):
        taus = []
        for gid in sorted(gmeta):
            vl, vr = av(gid, left), av(gid, right)
            if vl is None or vr is None:
                m7_drop[f"{left}|{right}:missing_arm"] += 1; continue
            sc = onealign_view(gmeta[gid])["scores"]
            fl = arm_finalists(gid, left); fr = arm_finalists(gid, right)
            if fl is None or fr is None:
                m7_drop[f"{left}|{right}:missing_arm"] += 1; continue
            if right == "onealign":
                shared = sorted(fl)
                ol = [s for s in vl["ranking"] if s in shared]; orr = restrict(sc, shared)
            elif left == "onealign":
                shared = sorted(fr)
                orr = [s for s in vr["ranking"] if s in shared]; ol = restrict(sc, shared)
            else:
                shared = sorted(set(fl) & set(fr))
                ol = [s for s in vl["ranking"] if s in shared]
                orr = [s for s in vr["ranking"] if s in shared]
            if len(shared) < 2 or len(ol) != len(shared) or len(orr) != len(shared):
                m7_drop[f"{left}|{right}:overlap_lt_2"] += 1; continue
            tau = kendall_tau(ol, orr)
            if tau is None:
                m7_drop[f"{left}|{right}:tau_undefined"] += 1; continue
            taus.append(tau)
        m7[f"{left}|{right}"] = {**_quantiles(taus), "floor": FLOOR_TAU}
    reg.record("M7'", "model_vs_model", {"n_human": 0, "per_arm": None,
                                         "arms_expected": (), "pairs": m7,
                                         "dropped": dict(sorted(m7_drop.items())),
                                         "floor": FLOOR_TAU})

    expect_n = {"M1'": len(prelim_records), "M8'": len(prelim_records),
                "M2'": len(final_records), "M3'": n_m3,
                "M4'": len(ranked_finals), "M5'": len(ranked_finals), "M7'": 0}
    reg.assert_wired(expect_n)

    expected_decisions = int(key["groups_used"]) * ITEMS_PER_GROUP
    reconciliation = {
        "decisions_expected": expected_decisions, "csv_rows": len(rows),
        "accepted": accepted, "final_skipped": dict(sorted(final_skipped.items())),
        "rejected": dict(sorted(rejected.items())), "unknown": len(unknown),
        "coverage_prelim": round(len(prelim_records) /
                                 (int(key["groups_used"]) * QUADS_PER_GROUP), 4),
    }
    if not args.allow_partial:
        problems = []
        if len(rows) != expected_decisions:
            problems.append(f"csv_rows={len(rows)} != {expected_decisions}")
        if accepted["prelim"] != int(key["groups_used"]) * QUADS_PER_GROUP:
            problems.append(f"accepted_prelim={accepted['prelim']} != "
                            f"{int(key['groups_used'])*QUADS_PER_GROUP}")
        if problems:
            raise RuntimeError("returned CSV does not reconcile with item_key.json: "
                               + "; ".join(problems)
                               + "  (use --allow-partial for an interim look)")

    by_major = {}
    for major in sorted({r["major"] for r in prelim_records}):
        sub_p = [r for r in prelim_records if r["major"] == major]
        cell = {"n_prelim": len(sub_p),
                "human_source_rate": _rate([int(r["human_source_won"]) for r in sub_p])["value"]}
        for arm in arm_names:
            hits = []
            for r in sub_p:
                rank1, _ = arm_rank1(r["group_id"], arm, r["quad_index"], r["options"])
                if rank1 is not None:
                    hits.append(int(r["human_top1"] == rank1))
            cell["M1'_" + arm] = _rate(hits)["value"]
        by_major[major] = cell

    analysis = {"schema": SCHEMA, "csv": str(csv_path), "item_key": str(key_path),
                "arms": arm_names, "reconciliation": reconciliation,
                "dropped_arm_cells": {a: dict(sorted(c.items())) for a, c in dropped.items()},
                "floors": {"M1'": FLOOR_M1, "M3'": FLOOR_M3, "M4'": FLOOR_M4,
                           "M8'": FLOOR_M8, "M7'": FLOOR_TAU,
                           "M2'": "per-group exact (see floor_note)",
                           "M5'": "per-group 1/|finalists|, pair_order 0.5"},
                "metrics_wired": sorted(reg.calls), "metrics": reg.calls,
                "by_major": by_major}
    out_path = Path(args.out) if args.out else csv_path.parent / "analysis.json"
    out_path.write_text(json.dumps(analysis, ensure_ascii=False, indent=2,
                                   sort_keys=True) + "\n", encoding="utf-8", newline="\n")

    lines = [
        f"csv_rows={len(rows)}  decisions_expected={expected_decisions}  "
        f"coverage_prelim={reconciliation['coverage_prelim']}",
        f"accepted: prelim={accepted['prelim']} final_rows={accepted['final_rows']} "
        f"final_ranked={accepted['final_ranked']}  "
        f"final_skipped={dict(final_skipped)}  rejected={sum(rejected.values())}  "
        f"unknown={len(unknown)}",
        f"arms = {', '.join(arm_names)}",
        "",
        _pad("metric", 8) + "".join(f"{a:>13}" for a in arm_names) + f"{'floor':>11}",
    ]
    for name in ("M1'", "M3'", "M4'"):
        row = reg.calls[name]
        lines.append(_pad(name, 8)
                     + "".join(f"{str(row['per_arm'][a]['value']):>13}" for a in arm_names)
                     + f"{str(row['floor']):>11}")
    r2row = reg.calls["M2'"]
    lines.append(_pad("M2'", 8)
                 + "".join(f"{str(r2row['per_arm'][a]['value']):>13}" for a in arm_names)
                 + f"{'per-group':>11}")
    lines.append(_pad("M2'flr", 8)
                 + "".join(f"{str(r2row['per_arm'][a]['floor']):>13}" for a in arm_names))
    m8row = reg.calls["M8'"]
    lines.extend(["",
                  "M8' source-win rate (floor 0.2):  human = "
                  + str(m8row["per_arm"]["human"]["value"])
                  + "  " + "  ".join(f"{a} = {m8row['per_arm'][a]['value']}"
                                     for a in arm_names),
                  "finalist-count distribution: "
                  + json.dumps(m8row["finalist_count_distribution"], ensure_ascii=False)])
    m5 = reg.calls["M5'"]["per_arm"]["onealign"]
    lines.extend(["",
                  f"M5' OneAlign on the human's own finalists: top1={m5['top1']['value']} "
                  f"(n={m5['top1']['n']}, floor {m5['top1']['floor']})  "
                  f"pair_order={m5['pair_order']['value']} "
                  f"(n={m5['pair_order']['n']}, floor {m5['pair_order']['floor']})",
                  "", "M7' pairwise Kendall tau (model vs model):"])
    for pair, blk in reg.calls["M7'"]["pairs"].items():
        lines.append(f"  {_pad(pair, 22)} n={blk['n_values']:>4} mean={blk['mean']} "
                     f"median={blk['median']} min={blk['min']} max={blk['max']}")
    m7_drop_summary = reg.calls["M7'"]["dropped"]
    lines.append(f"M7' dropped: {m7_drop_summary}")
    lines.extend(["", _pad("major", 14) + f"{'n_prelim':>9}{'human_src':>11}"
                  + "".join("{:>14}".format("M1'_" + a) for a in arm_names)])
    for major, blk in by_major.items():
        lines.append(_pad(major, 14) + f"{blk['n_prelim']:>9}{str(blk['human_source_rate']):>11}"
                     + "".join("{:>14}".format(str(blk["M1'_" + a])) for a in arm_names))
    lines.extend(["", f"written {out_path}"])
    print("\n".join(lines))
    return 0


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    b = sub.add_parser("build")
    b.add_argument("--build-root", type=Path,
                   default=Path("/home/bc/data/builds/epr049-aesth-20260825"))
    b.add_argument("--out-dir", type=Path,
                   default=Path("docs/assets/epr049_aesthq_v3_20260825"))
    b.add_argument("--thumb-short-edge", type=int, default=THUMB_SHORT_EDGE)
    b.set_defaults(func=cmd_build)
    a = sub.add_parser("analyze")
    a.add_argument("--csv", type=Path, required=True)
    a.add_argument("--item-key", type=Path, default=None)
    a.add_argument("--arm", action="append", default=None, help="name=path, repeatable")
    a.add_argument("--allow-partial", action="store_true")
    a.add_argument("--out", type=Path, default=None)
    a.set_defaults(func=cmd_analyze)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    return int(args.func(args))


if __name__ == "__main__":
    raise SystemExit(main())
