"""EPR-049 questionnaire v2: the human replays the model's own tournament.

``build`` reads **Phase A only** (``stage_render_qa.jsonl``: the 16 candidates and their
quadruple split).  It never opens a model arm's ``groups.jsonl``, so one human schedule
serves gemini, terra and OneAlign alike and can be built before any arm has finished.

Per group, five items, in this order and adjacent on the page:

* **prelim** ``<gid>_p0..p3`` — source + the four candidates of quadruple ``qi``.  The
  quadruple is ``epr049_build_groups.quadruples`` itself (imported, not re-derived, so the
  split is provably the one the models saw); the in-item display order is an independent
  ``sha1(item_key, slot)`` shuffle.  The rater picks a top1.
* **final** ``<gid>_f`` — the rater's own four prelim winners, ordered by
  ``sha1`` over the *set* of winning candidate ids, so the order cannot encode the order
  they were won in.  The rater picks a top1 and a top2.

The final item is assembled in the page from ``localStorage``: all 16 thumbnails of the
group are already on the page, and the 256 possible winner combinations are pre-resolved
into ``GROUPS[gid].order`` at build time, so no hashing (and no server) is needed at
answer time.

Blinding: thumbnails are named ``sha1(group_key, slot)``, captions are neutral ordinals,
and no preset id, model name, arm name, score or rank appears in any filename, caption or
DOM node.  The position -> slot mapping and the winner-order table live only in
``item_key.json``, which is written *before* the page is rendered.

``analyze`` joins the returned CSV to that key file plus one ``--arm`` journal per model
arm and prints M1'..M5' and M7' next to their chance floors.  Groups where an arm has no
usable result are dropped from that arm's column and counted; the human rows are kept.

Usage::

    python -m dataset_build.tools.epr049_questionnaire build \
        --build-root /home/bc/data/builds/epr049-aesth-20260825 \
        --out-dir docs/assets/epr049_aesthq_20260825
    python -m dataset_build.tools.epr049_questionnaire analyze \
        --csv docs/assets/epr049_aesthq_20260825/aesthq.csv \
        --arm gemini=/home/bc/data/builds/epr049-aesth-20260825/groups.jsonl
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import sys
from collections import Counter
from itertools import combinations
from math import comb
from pathlib import Path
from typing import Any, Mapping, Sequence

SCHEMA = "epr049-aesthetic-questionnaire-v2"
THUMB_SHORT_EDGE = 512
THUMB_QUALITY = 90
CSV_HEADER = "item_id,kind,pick1,pick2,notes"
QUAD_SIZE = 4
QUADS_PER_GROUP = 4
CANDIDATES_PER_GROUP = 16
PRELIM_OPTIONS = 4
FINAL_OPTIONS = 4
ITEMS_PER_GROUP = QUADS_PER_GROUP + 1

# ------------------------------------------------------------------ chance floors
# Written down here, derived once, rather than asserted in prose in the report.


def hypergeometric_jaccard(pool: int, size_a: int, size_b: int) -> dict[str, Any]:
    """E[Jaccard] for two independent uniform subsets of ``pool``.

    P(|A n B| = k) = C(size_a, k) * C(pool - size_a, size_b - k) / C(pool, size_b)
    Jaccard = k / (size_a + size_b - k).
    """
    total = comb(pool, size_b)
    terms = []
    expectation = 0.0
    for k in range(0, min(size_a, size_b) + 1):
        ways = comb(size_a, k) * comb(pool - size_a, size_b - k)
        if ways == 0:
            continue
        prob = ways / total
        jaccard = k / (size_a + size_b - k) if (size_a + size_b - k) else 0.0
        expectation += prob * jaccard
        terms.append({"k": k, "p": round(prob, 8), "jaccard": round(jaccard, 8)})
    return {
        "formula": (f"P(k)=C({size_a},k)*C({pool - size_a},{size_b}-k)/C({pool},{size_b}); "
                    f"J=k/({size_a}+{size_b}-k)"),
        "terms": terms,
        "expectation": round(expectation, 8),
    }


FLOOR_TOP1_OF_4 = 1.0 / PRELIM_OPTIONS               # M1'
FLOOR_M2 = hypergeometric_jaccard(16, 4, 4)          # M2': 4-of-16 vs 4-of-16
FLOOR_M3 = 2.0 / 16.0                                # M3': one of 16 lands in a fixed 2
FLOOR_M4 = hypergeometric_jaccard(16, 2, 2)          # M4': 2-of-16 vs 2-of-16
FLOOR_M5_TOP1 = 1.0 / 4.0                            # M5': rank1 of 4
FLOOR_M5_ORDER = 0.5                                 # M5': orientation of one pair
FLOOR_TAU = 0.0                                      # M7'


def sha1_hex(*parts: object) -> str:
    return hashlib.sha1("\x1f".join(str(part) for part in parts).encode("utf-8")).hexdigest()


def _pad(text: str, width: int) -> str:
    """Left-align in terminal columns: taxonomy majors are CJK and render double-width."""
    import unicodedata

    shown = sum(2 if unicodedata.east_asian_width(ch) in "WF" else 1 for ch in text)
    return text + " " * max(1, width - shown)


def _js_json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True).replace("</", "<\\/")


def blind_order(item_key: str, slots: Sequence[str]) -> list[str]:
    return sorted(slots, key=lambda slot: (sha1_hex(item_key, slot), slot))


def final_order(candidate_ids: Sequence[str]) -> list[int]:
    """Display order of the four winners, keyed by the *set* of their candidate ids.

    Returns quad indices.  Seeding on the sorted id set is what makes the order
    independent of the order the winners were produced in.
    """
    seed = sha1_hex("epr049-final", *sorted(candidate_ids))
    return sorted(range(len(candidate_ids)),
                  key=lambda index: (sha1_hex(seed, candidate_ids[index]),
                                     candidate_ids[index]))


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
                    size = tuple(max(1, int(round(value * scale))) for value in image.size)
                    resampling = getattr(Image, "Resampling", Image)
                    image = image.resize(size, resampling.LANCZOS)
                image.save(target, "JPEG", quality=THUMB_QUALITY)
        except Exception as exc:  # noqa: BLE001 - one unreadable asset drops one group
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
#stage{font-variant-numeric:tabular-nums}
.track{height:6px;background:#e6e6e6;border-radius:3px;overflow:hidden;margin:0 0 10px}
#bar{height:100%;background:#2b6cb0;width:0}
.strip{display:grid;gap:10px;align-items:start}
.strip figure{margin:0}
.strip img{display:block;width:100%;height:auto;border:1px solid #ccc;background:#202020}
.strip figcaption{font-size:13px;color:#333;margin-top:4px;text-align:center}
.strip figure.src figcaption{font-weight:600}
.picks{display:flex;gap:8px;flex-wrap:wrap;margin:12px 0 4px}
.pick{flex:1 1 120px;padding:12px 10px;font-size:15px;cursor:pointer;border:1px solid #bbb;
border-radius:8px;background:#fafafa;color:#111;text-align:center}
.pick:hover{background:#f0f4f8}
.pick.one{background:#2b6cb0;border-color:#2b6cb0;color:#fff}
.pick.two{background:#7fb0d8;border-color:#7fb0d8;color:#111}
.legend{font-size:12px;color:#666;margin:0 0 8px}
.blocked{padding:14px;border:1px dashed #c00;border-radius:8px;color:#900;font-size:14px;
background:#fff6f6}
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
      const parsed = JSON.parse(raw);
      state.picks = (parsed && parsed.picks) || {};
      state.notes = (parsed && parsed.notes) || {};
      cursor = Math.min(Math.max(parseInt(parsed && parsed.cursor, 10) || 0, 0),
                        Math.max(total - 1, 0));
    }
  } catch (err) { console.warn('localStorage load failed', err); }
}

function save() {
  state.cursor = cursor;
  try { window.localStorage.setItem(STORE_KEY, JSON.stringify(state)); }
  catch (err) { console.warn('localStorage save failed', err); }
}

/* The four prelim picks of a group, or null if any is still missing. */
function prelimPicks(group) {
  const picks = [];
  for (let qi = 0; qi < 4; qi += 1) {
    const iid = group + '_p' + qi;
    const pick = state.picks[iid] || {};
    if (!pick.p1) { return null; }
    picks.push(pick.p1);
  }
  return picks;
}

/* Winner slots of a group in final display order, or null if the group is unfinished. */
function finalSlots(group) {
  const picks = prelimPicks(group);
  if (!picks) { return null; }
  const g = GROUPS[group];
  const order = g.order[picks.join('')];
  if (!order) { return null; }
  const slots = [];
  for (let i = 0; i < order.length; i += 1) {
    const qi = parseInt(order.charAt(i), 10);
    slots.push(g.quads[qi][picks[qi] - 1]);
  }
  return slots;
}

function complete(iid) {
  const view = VIEW[iid];
  const pick = state.picks[iid] || {};
  if (view.kind === 'final') {
    if (!finalSlots(view.group)) { return false; }
    return !!pick.p1 && !!pick.p2;
  }
  return !!pick.p1;
}

function doneCount() {
  let n = 0;
  for (const iid of ITEMS) { if (complete(iid)) { n += 1; } }
  return n;
}

function imagesFor(iid) {
  const view = VIEW[iid];
  const g = GROUPS[view.group];
  if (view.kind !== 'final') { return view.imgs.map(function (s) { return g.thumb[s]; }); }
  const slots = finalSlots(view.group);
  if (!slots) { return null; }
  return slots.map(function (s) { return g.thumb[s]; });
}

function render() {
  const iid = ITEMS[cursor];
  const view = VIEW[iid];
  const g = GROUPS[view.group];
  const strip = document.getElementById('strip');
  const picksBox = document.getElementById('picks');
  const blocked = document.getElementById('blocked');
  strip.innerHTML = '';
  picksBox.innerHTML = '';
  const imgs = imagesFor(iid);

  if (imgs === null) {
    blocked.style.display = 'block';
    blocked.textContent = '本组前 4 道题还没做完，做完后这道决胜题会自动出现。';
    strip.style.gridTemplateColumns = '1fr';
  } else {
    blocked.style.display = 'none';
    strip.style.gridTemplateColumns = 'repeat(' + (imgs.length + 1) + ',1fr)';
    const all = [g.src].concat(imgs);
    for (let i = 0; i < all.length; i += 1) {
      const fig = document.createElement('figure');
      if (i === 0) { fig.className = 'src'; }
      const img = document.createElement('img');
      img.src = all[i];
      const caption = i === 0 ? '原图' : ('候选 ' + i);
      img.alt = caption;
      const cap = document.createElement('figcaption');
      cap.textContent = caption;
      fig.appendChild(img);
      fig.appendChild(cap);
      strip.appendChild(fig);
    }
    const current = state.picks[iid] || {};
    for (let i = 1; i <= imgs.length; i += 1) {
      const btn = document.createElement('button');
      btn.type = 'button';
      btn.className = 'pick';
      btn.textContent = '候选 ' + i;
      if (current.p1 === i) { btn.classList.add('one'); }
      else if (current.p2 === i) { btn.classList.add('two'); }
      btn.addEventListener('click', function () { choose(i); });
      picksBox.appendChild(btn);
    }
  }

  document.getElementById('task').textContent = view.kind === 'final'
    ? '决胜题：从你自己选出的 4 张里，先点最好的（深蓝），再点第二好的（浅蓝）'
    : '初赛题：点你认为最好的一张';
  document.getElementById('stage').textContent = view.kind === 'final'
    ? '本组第 5 / 5 题'
    : ('本组第 ' + (view.quad + 1) + ' / 5 题');
  document.getElementById('legend').textContent = view.kind === 'final'
    ? '深蓝 = 第一名，浅蓝 = 第二名；再点一次可取消。'
    : '再点一次可取消。';
  document.getElementById('progress').textContent = (cursor + 1) + ' / ' + total;
  const done = doneCount();
  const left = total - done;
  document.getElementById('done').textContent = '已完成 ' + done + ' / ' + total;
  document.getElementById('bar').style.width = (total ? (done * 100 / total) : 0) + '%';
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
  const view = VIEW[iid];
  const pick = Object.assign({}, state.picks[iid] || {});
  if (view.kind === 'final') {
    if (pick.p1 === value) { delete pick.p1; }
    else if (pick.p2 === value) { delete pick.p2; }
    else if (!pick.p1) { pick.p1 = value; }
    else if (!pick.p2) { pick.p2 = value; }
    else { pick.p1 = value; delete pick.p2; }
  } else {
    if (pick.p1 === value) { delete pick.p1; }
    else { pick.p1 = value; }
    /* Changing a prelim winner invalidates the final answer of the same group. */
    const fid = view.group + '_f';
    if (state.picks[fid]) { delete state.picks[fid]; }
  }
  state.picks[iid] = pick;
  save();
  if (complete(iid) && cursor < total - 1) { cursor += 1; save(); }
  render();
}

function go(delta) {
  const next = cursor + delta;
  if (next < 0 || next >= total) { return; }
  cursor = next;
  save();
  render();
}

function csvField(text) {
  const value = String(text == null ? '' : text);
  if (/[",\\r\\n]/.test(value)) { return '"' + value.replace(/"/g, '""') + '"'; }
  return value;
}

function buildCsv() {
  const lines = [__HEADER__];
  for (const iid of ITEMS) {
    const view = VIEW[iid];
    const pick = state.picks[iid] || {};
    lines.push([iid, view.kind, pick.p1 ? String(pick.p1) : '',
                pick.p2 ? String(pick.p2) : '',
                csvField(state.notes[iid] || '')].join(','));
  }
  return lines.join('\\n') + '\\n';
}

function exportCsv() {
  const blob = new Blob([buildCsv()], {type: 'text/csv;charset=utf-8'});
  const url = URL.createObjectURL(blob);
  const link = document.createElement('a');
  link.href = url;
  link.download = 'aesthq.csv';
  document.body.appendChild(link);
  link.click();
  document.body.removeChild(link);
  setTimeout(function () { URL.revokeObjectURL(url); }, 2000);
}

function inField(target) {
  if (!target || !target.tagName) { return false; }
  const tag = target.tagName.toLowerCase();
  return tag === 'input' || tag === 'textarea';
}

window.addEventListener('DOMContentLoaded', function () {
  load();
  document.getElementById('prev').addEventListener('click', function () { go(-1); });
  document.getElementById('next').addEventListener('click', function () { go(1); });
  document.getElementById('export').addEventListener('click', exportCsv);
  document.getElementById('notes').addEventListener('input', function (event) {
    state.notes[ITEMS[cursor]] = event.target.value;
    save();
  });
  document.addEventListener('keydown', function (event) {
    if (event.ctrlKey || event.metaKey || event.altKey || inField(event.target)) { return; }
    if (event.key >= '1' && event.key <= '4') {
      choose(parseInt(event.key, 10));
      event.preventDefault();
    } else if (event.key === 'ArrowLeft') { go(-1); event.preventDefault(); }
    else if (event.key === 'ArrowRight') { go(1); event.preventDefault(); }
  });
  render();
  window.__READY__ = true;
});
"""


def write_page(out_dir: Path, order: Sequence[str], view: Mapping[str, Any],
               groups: Mapping[str, Any], store_key: str) -> Path:
    script = (
        _SCRIPT
        .replace("__ITEMS__", _js_json(list(order)))
        .replace("__VIEW__", _js_json(dict(view)))
        .replace("__GROUPS__", _js_json(dict(groups)))
        .replace("__STORE_KEY__", _js_json(store_key))
        .replace("__HEADER__", _js_json(CSV_HEADER))
    )
    parts = [
        '<!doctype html><html lang="zh"><head><meta charset="utf-8">',
        '<meta name="viewport" content="width=device-width,initial-scale=1">',
        "<title>修图效果盲评问卷</title>",
        f"<style>{_STYLE}</style></head><body>",
        "<h1>修图效果盲评问卷</h1>",
        '<p class="hint">最左边一张是<b>原图</b>，右边几张是同一张原图的不同修图结果。'
        "每组 5 题：前 4 题各选出一张最好的，第 5 题在你自己选出的 4 张里排出前二。"
        "选择自动保存在本机浏览器，做完点「导出 CSV」下载 aesthq.csv。</p>",
        '<div class="topbar"><span id="progress">- / -</span>'
        '<span id="stage"></span><span id="done"></span><span id="unrated"></span>'
        '<span class="keys">也可用键盘：1–4 选择，← / → 翻页</span></div>',
        '<div class="track"><div id="bar"></div></div>',
        '<p id="task" class="hint"></p>',
        '<div class="blocked" id="blocked" style="display:none"></div>',
        '<div class="strip" id="strip"></div>',
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


def write_csv(out_dir: Path, order: Sequence[str], view: Mapping[str, Any]) -> Path:
    lines = [CSV_HEADER]
    lines.extend(f"{item_id},{view[item_id]['kind']},,," for item_id in order)
    path = out_dir / "aesthq.csv"
    path.write_text("\n".join(lines) + "\n", encoding="utf-8", newline="\n")
    return path


# -------------------------------------------------------------------------------- build


def read_stage(build_root: Path) -> list[dict[str, Any]]:
    """Phase A journal, deduplicated by group (append-only, last line wins)."""
    path = build_root / "stage_render_qa.jsonl"
    if not path.is_file():
        raise SystemExit(f"missing {path}")
    latest: dict[str, dict[str, Any]] = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            row = json.loads(line)
            latest[str(row["group_id"])] = row
    return [latest[group_id] for group_id in sorted(latest)]


def cmd_build(args: argparse.Namespace) -> int:
    from dataset_build.tools.epr049_build_groups import quadruples

    build_root = Path(args.build_root)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    rows = read_stage(build_root)
    dropped: Counter[str] = Counter()
    thumbs = Thumbnails(out_dir, args.thumb_short_edge)

    order: list[str] = []
    view: dict[str, Any] = {}
    groups_js: dict[str, Any] = {}
    items: dict[str, Any] = {}
    group_meta: dict[str, Any] = {}

    for row in rows:
        group_id = str(row["group_id"])
        candidates = row["candidates"]
        if len(candidates) != CANDIDATES_PER_GROUP:
            dropped["candidate_count"] += 1
            continue
        by_slot = {str(entry["slot_id"]): entry for entry in candidates}
        group_key = f"{group_id}:q"

        source_thumb = thumbs.rel(build_root / row["source_asset"],
                                  sha1_hex(group_key, "src"))
        if source_thumb is None:
            dropped["source_thumb_failed"] += 1
            continue
        thumb: dict[str, str] = {}
        broken = False
        for slot, entry in by_slot.items():
            rel = thumbs.rel(build_root / entry["asset"], sha1_hex(group_key, slot))
            if rel is None:
                broken = True
                break
            thumb[slot] = rel
        if broken:
            dropped["candidate_thumb_failed"] += 1
            continue

        # The models' own split, imported rather than re-derived.
        quads = quadruples(group_id, list(by_slot))
        if len(quads) != QUADS_PER_GROUP or any(len(q) != QUAD_SIZE for q in quads):
            dropped["quad_shape"] += 1
            continue

        quads_display: list[list[str]] = []
        group_items: list[str] = []
        for quad_index, quad in enumerate(quads):
            item_id = f"{group_id}_p{quad_index}"
            item_key = f"{group_id}:p{quad_index}"
            shown = blind_order(item_key, quad)
            quads_display.append(shown)
            group_items.append(item_id)
            view[item_id] = {
                "kind": "prelim",
                "group": group_id,
                "quad": quad_index,
                "imgs": shown,
            }
            items[item_id] = {
                "group_id": group_id,
                "kind": "prelim",
                "major": row["major"],
                "quad_index": quad_index,
                "quad_slots": list(quad),
                "position_to_slot": {str(i + 1): slot for i, slot in enumerate(shown)},
            }

        # All 256 winner combinations pre-resolved: key = the four prelim pick positions
        # (1..4) concatenated, value = quad indices in final display order.
        order_table: dict[str, str] = {}
        for p0 in range(QUAD_SIZE):
            for p1 in range(QUAD_SIZE):
                for p2 in range(QUAD_SIZE):
                    for p3 in range(QUAD_SIZE):
                        picks = (p0, p1, p2, p3)
                        winners = [quads_display[qi][picks[qi]] for qi in range(4)]
                        ids = [by_slot[slot]["candidate_id"] for slot in winners]
                        key = "".join(str(p + 1) for p in picks)
                        order_table[key] = "".join(str(q) for q in final_order(ids))

        final_id = f"{group_id}_f"
        group_items.append(final_id)
        view[final_id] = {"kind": "final", "group": group_id, "quad": QUADS_PER_GROUP}
        items[final_id] = {
            "group_id": group_id,
            "kind": "final",
            "major": row["major"],
            "prelim_items": [f"{group_id}_p{i}" for i in range(QUADS_PER_GROUP)],
        }
        groups_js[group_id] = {
            "src": source_thumb,
            "thumb": thumb,
            "quads": quads_display,
            "order": order_table,
        }
        group_meta[group_id] = {
            "major": row["major"],
            "source_id": row["source"]["source_id"],
            "scene": row["source"].get("scene"),
            "quads_display": quads_display,
            "final_order_table": order_table,
            "slot_to_candidate_id": {s: e["candidate_id"] for s, e in by_slot.items()},
            "onealign_scores": row["onealign"]["scores"],
            "onealign_ranking_full": row["onealign"]["ranking"],
        }
        order.append(group_id)

    # Groups shuffled globally; the five items of a group stay adjacent (the tournament
    # needs its own prelims answered first).  Recorded as a known limitation.
    order.sort(key=lambda group_id: (sha1_hex("epr049-groupshuffle", group_id), group_id))
    display_order = [item for group_id in order
                     for item in ([f"{group_id}_p{i}" for i in range(QUADS_PER_GROUP)]
                                  + [f"{group_id}_f"])]

    key_payload = {
        "schema": SCHEMA,
        "build_root": str(build_root),
        "thumb_short_edge": args.thumb_short_edge,
        "groups_total": len(rows),
        "groups_used": len(order),
        "groups_dropped": dict(sorted(dropped.items())),
        "items_per_group": ITEMS_PER_GROUP,
        "display_order": display_order,
        "group_order": order,
        "items": items,
        "groups": group_meta,
    }
    # Results before optional stages: the key file (quadruple membership, blind order
    # maps, winner-order table) is what `analyze` needs, so it lands before the render.
    key_path = out_dir / "item_key.json"
    key_path.write_text(
        json.dumps(key_payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8", newline="\n",
    )
    csv_path = write_csv(out_dir, display_order, view)
    page = write_page(out_dir, display_order, view, groups_js,
                      f"{SCHEMA}:{build_root.name}")
    stats = {
        "schema": SCHEMA,
        "page": str(page),
        "csv": str(csv_path),
        "item_key": str(key_path),
        "groups_total": len(rows),
        "groups_used": len(order),
        "groups_dropped": dict(sorted(dropped.items())),
        "items": len(display_order),
        "items_prelim": sum(1 for i in items.values() if i["kind"] == "prelim"),
        "items_final": sum(1 for i in items.values() if i["kind"] == "final"),
        "decisions_expected": len(order) * ITEMS_PER_GROUP,
        "thumbnails": len(thumbs.written),
        "thumbnail_failures": thumbs.failed,
    }
    print(json.dumps(stats, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


# ------------------------------------------------------------------------------ metrics


class MetricRegistry:
    """Pre-registered metrics with a runtime wiring assertion.

    Defining a criterion and never calling it has silently emptied three evaluations in
    this campaign, so every metric records its own invocation and sample count here and
    ``assert_wired`` refuses to print a table where any of them is missing, or where the
    human-side denominators do not reconcile with the accepted CSV decisions.
    """

    def __init__(self, expected: Sequence[str]):
        self.expected = tuple(expected)
        self.calls: dict[str, dict[str, Any]] = {}

    def record(self, name: str, kind: str, payload: Mapping[str, Any]) -> dict[str, Any]:
        row = {"metric": name, "kind": kind, **payload}
        self.calls[name] = row
        return row

    def assert_wired(self, accepted: Mapping[str, int], arms: Sequence[str]) -> None:
        missing = [name for name in self.expected if name not in self.calls]
        if missing:
            raise RuntimeError(f"pre-registered metrics never ran: {missing}")
        # Each metric declares which arms it is *supposed* to cover: M1'..M4' run on every
        # arm, M5' is OneAlign-only by definition (it re-ranks the human's own finalists
        # by OneAlign score), M7' is pairwise and carries no per-arm cell.  Asserting
        # against the declared scope keeps this strict without demanding a gemini cell for
        # a metric that has none.
        missing_arms = [
            f"{name}:{arm}" for name, row in sorted(self.calls.items())
            for arm in (row.get("arms_expected") or ())
            if arm not in (row.get("per_arm") or {})
        ]
        if missing_arms:
            raise RuntimeError(f"metric/arm cells never ran: {missing_arms}")
        # Human denominators: M1' is one decision per quadruple, M2'..M5' one per group.
        checks = {
            "M1'": ("prelim", accepted.get("prelim", -1)),
            "M2'": ("final", accepted.get("final", -1)),
            "M3'": ("final", accepted.get("final", -1)),
            "M4'": ("final", accepted.get("final", -1)),
            "M5'": ("final", accepted.get("final", -1)),
        }
        bad = []
        for name, (kind, expected_n) in checks.items():
            row = self.calls.get(name)
            if row is None:
                continue
            if int(row["n_human"]) != int(expected_n):
                bad.append(f"{name}(n_human={row['n_human']} != accepted[{kind}]={expected_n})")
        if bad:
            raise RuntimeError(
                "metric sample counts do not reconcile with the accepted CSV decisions: "
                + ", ".join(bad)
            )


def _rate(hits: Sequence[int]) -> dict[str, Any]:
    n = len(hits)
    return {"n": n, "value": round(sum(hits) / n, 6) if n else None}


def _mean(values: Sequence[float]) -> dict[str, Any]:
    n = len(values)
    return {"n": n, "value": round(sum(values) / n, 6) if n else None}


def jaccard(left: Sequence[str], right: Sequence[str]) -> float:
    a, b = set(left), set(right)
    union = a | b
    return len(a & b) / len(union) if union else 0.0


def kendall_tau(order_a: Sequence[str], order_b: Sequence[str]) -> float | None:
    """Kendall tau-a over two orderings of the same item set (no ties by construction)."""
    items = list(order_a)
    if sorted(items) != sorted(order_b) or len(items) < 2:
        return None
    rank_a = {item: index for index, item in enumerate(order_a)}
    rank_b = {item: index for index, item in enumerate(order_b)}
    concordant = discordant = 0
    for left, right in combinations(items, 2):
        sign = (rank_a[left] - rank_a[right]) * (rank_b[left] - rank_b[right])
        if sign > 0:
            concordant += 1
        elif sign < 0:
            discordant += 1
    total = concordant + discordant
    return (concordant - discordant) / total if total else None


def _quantiles(values: Sequence[float]) -> dict[str, Any]:
    """Summary of a value list; the count key is ``n_values``, never ``n``.

    A plain ``n`` here used to overwrite the ``n`` of the row this dict is merged into.
    """
    if not values:
        return {"n_values": 0, "mean": None, "median": None, "min": None, "max": None,
                "histogram": {}}
    ordered = sorted(values)
    mid = len(ordered) // 2
    median = ordered[mid] if len(ordered) % 2 else 0.5 * (ordered[mid - 1] + ordered[mid])
    return {
        "n_values": len(ordered),
        "mean": round(sum(ordered) / len(ordered), 6),
        "median": round(median, 6),
        "min": round(ordered[0], 6),
        "max": round(ordered[-1], 6),
        "histogram": dict(sorted(Counter(round(value, 4) for value in ordered).items())),
    }


def restrict(scores: Mapping[str, float], slots: Sequence[str]) -> list[str]:
    known = [slot for slot in slots if slot in scores]
    return sorted(known, key=lambda slot: (-float(scores[slot]), slot))


# ------------------------------------------------------------------------------ arms


def load_arm(path: Path) -> dict[str, dict[str, Any]]:
    """One arm's ``groups.jsonl``, deduplicated by group, complete rows only."""
    latest: dict[str, dict[str, Any]] = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            row = json.loads(line)
            latest[str(row["group_id"])] = row
    return latest


def arm_view(row: Mapping[str, Any]) -> dict[str, Any] | None:
    """The four finalists, the round-2 ranking and top2 of one arm for one group."""
    if row.get("status") != "complete":
        return None
    # Rows written before the multi-arm split carry the block under "gemini"; every arm
    # written after it uses "tournament".  Read either.
    gemini = row.get("tournament") or row.get("gemini") or {}
    round2 = gemini.get("round2") or {}
    finalists = list(round2.get("finalists") or [])
    ranking = list(round2.get("ranking_slots") or [])
    top2 = list(round2.get("top2") or [])
    if len(finalists) != QUAD_SIZE or len(ranking) != QUAD_SIZE or len(top2) != 2:
        return None
    rank1_by_quad: dict[int, str] = {}
    for entry in gemini.get("round1") or []:
        if entry.get("ok") and entry.get("rank1"):
            rank1_by_quad[int(entry["quad_index"])] = str(entry["rank1"])
    return {"finalists": finalists, "ranking": ranking, "top2": top2,
            "rank1_by_quad": rank1_by_quad}


def onealign_view(meta: Mapping[str, Any]) -> dict[str, Any]:
    scores = meta["onealign_scores"]
    full = restrict(scores, list(scores))
    return {"scores": scores, "finalists": full[:4], "ranking": full[:4],
            "top2": full[:2], "full_ranking": full}


# ------------------------------------------------------------------------------ analyze


def read_answers(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return [
            {key: str(row.get(key) or "").strip()
             for key in ("item_id", "kind", "pick1", "pick2", "notes")}
            for row in csv.DictReader(handle)
        ]


def _position(raw: str, options: int) -> int | None:
    try:
        value = int(raw)
    except (TypeError, ValueError):
        return None
    return value if 1 <= value <= options else None


def cmd_analyze(args: argparse.Namespace) -> int:
    csv_path = Path(args.csv)
    key_path = Path(args.item_key or (csv_path.parent / "item_key.json"))
    key = json.loads(key_path.read_text(encoding="utf-8"))
    items: dict[str, Any] = key["items"]
    group_meta: dict[str, Any] = key["groups"]
    rows = read_answers(csv_path)

    arms: dict[str, dict[str, dict[str, Any]]] = {}
    for spec in args.arm or []:
        if "=" not in spec:
            raise SystemExit(f"--arm expects name=path, got {spec!r}")
        name, _, path = spec.partition("=")
        arms[name] = load_arm(Path(path))
    arm_names = sorted(arms) + ["onealign"]

    # ---- decode the human decisions
    prelim_pick: dict[str, dict[int, str]] = {}   # group -> quad -> slot
    prelim_pos: dict[str, dict[int, int]] = {}    # group -> quad -> pick position
    final_raw: dict[str, tuple[int, int]] = {}
    rejected: Counter[str] = Counter()
    unknown: list[str] = []
    for row in rows:
        item_id = row["item_id"]
        entry = items.get(item_id)
        if entry is None:
            unknown.append(item_id)
            continue
        kind = entry["kind"]
        if row["kind"] and row["kind"] != kind:
            rejected["kind_mismatch"] += 1
            continue
        group_id = entry["group_id"]
        if kind == "prelim":
            pick = _position(row["pick1"], PRELIM_OPTIONS)
            if pick is None:
                rejected["prelim_missing_pick1"] += 1
                continue
            prelim_pick.setdefault(group_id, {})[entry["quad_index"]] = \
                entry["position_to_slot"][str(pick)]
            prelim_pos.setdefault(group_id, {})[entry["quad_index"]] = pick
        else:
            p1 = _position(row["pick1"], FINAL_OPTIONS)
            p2 = _position(row["pick2"], FINAL_OPTIONS)
            if p1 is None or p2 is None or p1 == p2:
                rejected["final_missing_or_duplicate_pick"] += 1
                continue
            final_raw[group_id] = (p1, p2)

    prelim_records: list[dict[str, Any]] = []
    for group_id, per_quad in sorted(prelim_pick.items()):
        meta = group_meta[group_id]
        for quad_index, slot in sorted(per_quad.items()):
            prelim_records.append({
                "group_id": group_id, "major": meta["major"],
                "quad_index": quad_index, "human_top1": slot,
                "quad_slots": meta["quads_display"][quad_index],
            })

    final_records: list[dict[str, Any]] = []
    for group_id, (p1, p2) in sorted(final_raw.items()):
        per_quad = prelim_pick.get(group_id, {})
        if len(per_quad) != QUADS_PER_GROUP:
            rejected["final_without_all_prelims"] += 1
            continue
        meta = group_meta[group_id]
        pos_key = "".join(str(prelim_pos[group_id][qi]) for qi in range(QUADS_PER_GROUP))
        order_str = meta["final_order_table"].get(pos_key)
        if order_str is None:
            rejected["final_order_key_missing"] += 1
            continue
        shown = [per_quad[int(ch)] for ch in order_str]
        final_records.append({
            "group_id": group_id, "major": meta["major"],
            "human_finalists": [per_quad[qi] for qi in range(QUADS_PER_GROUP)],
            "human_shown": shown,
            "human_top1": shown[p1 - 1],
            "human_top2": [shown[p1 - 1], shown[p2 - 1]],
        })

    accepted = {"prelim": len(prelim_records), "final": len(final_records)}

    # Reconcile against the *key file*, not against the CSV itself.  Deriving the
    # denominator from the same CSV makes the check self-consistent and unable to notice
    # a truncated or partially-answered return -- caught by a truncation negative control
    # on 2026-08-25, when a 200-row CSV sailed through a "500 decision" assertion.
    expected_decisions = int(key["groups_used"]) * ITEMS_PER_GROUP
    seen = len(rows)
    reconciliation = {
        "decisions_expected": expected_decisions,
        "csv_rows": seen,
        "accepted_total": accepted["prelim"] + accepted["final"],
        "rejected_total": sum(rejected.values()),
        "unknown_total": len(unknown),
    }
    if not args.allow_partial:
        problems = []
        if seen != expected_decisions:
            problems.append(f"csv_rows={seen} != decisions_expected={expected_decisions}")
        covered = accepted["prelim"] + accepted["final"] + sum(rejected.values()) + len(unknown)
        if covered != seen:
            problems.append(f"accepted+rejected+unknown={covered} != csv_rows={seen}")
        if accepted["prelim"] != int(key["groups_used"]) * QUADS_PER_GROUP:
            problems.append(
                f"accepted_prelim={accepted['prelim']} != "
                f"{int(key['groups_used']) * QUADS_PER_GROUP}")
        if accepted["final"] != int(key["groups_used"]):
            problems.append(f"accepted_final={accepted['final']} != {key['groups_used']}")
        if problems:
            raise RuntimeError(
                "returned CSV does not reconcile with item_key.json: "
                + "; ".join(problems)
                + "  (pass --allow-partial for an interim look at an unfinished return)"
            )

    def arm_for(group_id: str, arm: str) -> dict[str, Any] | None:
        if arm == "onealign":
            return onealign_view(group_meta[group_id])
        row = arms[arm].get(group_id)
        return arm_view(row) if row else None

    registry = MetricRegistry(("M1'", "M2'", "M3'", "M4'", "M5'", "M7'"))
    dropped_cells: dict[str, Counter] = {arm: Counter() for arm in arm_names}

    # ---- M1' prelim top1 agreement, per arm, over quadruples
    m1: dict[str, Any] = {}
    for arm in arm_names:
        hits: list[int] = []
        for record in prelim_records:
            view = arm_for(record["group_id"], arm)
            if view is None:
                dropped_cells[arm]["M1'_no_arm_result"] += 1
                continue
            if arm == "onealign":
                pick = restrict(view["scores"], record["quad_slots"])
                rank1 = pick[0] if pick else None
            else:
                rank1 = view["rank1_by_quad"].get(record["quad_index"])
            if rank1 is None:
                dropped_cells[arm]["M1'_no_quad_rank1"] += 1
                continue
            hits.append(int(record["human_top1"] == rank1))
        m1[arm] = _rate(hits)
    registry.record("M1'", "prelim", {
        "n_human": len(prelim_records), "per_arm": m1, "arms_expected": tuple(arm_names),
        "floor": round(FLOOR_TOP1_OF_4, 6),
        "floor_note": "1/4: rank1 of a 4-way quadruple",
    })

    # ---- M2' finalist-set Jaccard (4-of-16 vs 4-of-16)
    m2: dict[str, Any] = {}
    for arm in arm_names:
        values: list[float] = []
        for record in final_records:
            view = arm_for(record["group_id"], arm)
            if view is None:
                dropped_cells[arm]["M2'_no_arm_result"] += 1
                continue
            values.append(jaccard(record["human_finalists"], view["finalists"]))
        m2[arm] = _mean(values)
    registry.record("M2'", "final", {
        "n_human": len(final_records), "per_arm": m2, "arms_expected": tuple(arm_names),
        "floor": FLOOR_M2["expectation"], "floor_formula": FLOOR_M2["formula"],
        "floor_terms": FLOOR_M2["terms"],
    })

    # ---- M3' human final top1 inside the arm's final top2
    m3: dict[str, Any] = {}
    for arm in arm_names:
        hits: list[int] = []
        for record in final_records:
            view = arm_for(record["group_id"], arm)
            if view is None:
                dropped_cells[arm]["M3'_no_arm_result"] += 1
                continue
            hits.append(int(record["human_top1"] in set(view["top2"])))
        m3[arm] = _rate(hits)
    registry.record("M3'", "final", {
        "n_human": len(final_records), "per_arm": m3, "arms_expected": tuple(arm_names),
        "floor": round(FLOOR_M3, 6),
        "floor_note": "2/16: one of 16 candidates landing in a fixed pair",
    })

    # ---- M4' top2 set Jaccard (2-of-16 vs 2-of-16)
    m4: dict[str, Any] = {}
    for arm in arm_names:
        values = []
        for record in final_records:
            view = arm_for(record["group_id"], arm)
            if view is None:
                dropped_cells[arm]["M4'_no_arm_result"] += 1
                continue
            values.append(jaccard(record["human_top2"], view["top2"]))
        m4[arm] = _mean(values)
    registry.record("M4'", "final", {
        "n_human": len(final_records), "per_arm": m4, "arms_expected": tuple(arm_names),
        "floor": FLOOR_M4["expectation"], "floor_formula": FLOOR_M4["formula"],
        "floor_terms": FLOOR_M4["terms"],
    })

    # ---- M5' OneAlign restricted to the human's own four finalists
    top1_hits: list[int] = []
    order_hits: list[int] = []
    for record in final_records:
        scores = group_meta[record["group_id"]]["onealign_scores"]
        ranked = restrict(scores, record["human_finalists"])
        if len(ranked) != QUAD_SIZE:
            dropped_cells["onealign"]["M5'_incomplete_scores"] += 1
            continue
        top1_hits.append(int(record["human_top1"] == ranked[0]))
        rank = {slot: index for index, slot in enumerate(ranked)}
        first, second = record["human_top2"]
        order_hits.append(int(rank[first] < rank[second]))
    registry.record("M5'", "final", {
        "n_human": len(final_records),
        "arms_expected": ("onealign",),
        "per_arm": {"onealign": {
            "top1": {**_rate(top1_hits), "floor": round(FLOOR_M5_TOP1, 6)},
            "pair_order": {**_rate(order_hits), "floor": round(FLOOR_M5_ORDER, 6)},
        }},
        "floor": round(FLOOR_M5_TOP1, 6),
        "floor_note": "0.25 for rank1 of the human's 4; 0.50 for the orientation of "
                      "the human's top1/top2 pair",
    })

    # ---- M7' model-vs-model Kendall tau, pairwise, no human involved
    m7: dict[str, Any] = {}
    m7_dropped: Counter[str] = Counter()
    for left, right in combinations(arm_names, 2):
        taus: list[float] = []
        for group_id in sorted(group_meta):
            view_l = arm_for(group_id, left)
            view_r = arm_for(group_id, right)
            if view_l is None or view_r is None:
                m7_dropped[f"{left}|{right}:missing_arm"] += 1
                continue
            scores = group_meta[group_id]["onealign_scores"]
            # Model-arm vs OneAlign is the task card's "OneAlign 限 4 finalist" reading:
            # OneAlign is re-ranked on *that arm's* four finalists, so the two orderings
            # always cover the same 4 slots.  Only model-arm vs model-arm needs the
            # intersection, because two arms can promote different finalists.
            if right == "onealign":
                shared = sorted(view_l["finalists"])
                order_l = [s for s in view_l["ranking"] if s in shared]
                order_r = restrict(scores, shared)
            elif left == "onealign":
                shared = sorted(view_r["finalists"])
                order_r = [s for s in view_r["ranking"] if s in shared]
                order_l = restrict(scores, shared)
            else:
                shared = sorted(set(view_l["finalists"]) & set(view_r["finalists"]))
                order_l = [s for s in view_l["ranking"] if s in shared]
                order_r = [s for s in view_r["ranking"] if s in shared]
            if len(shared) < 2 or len(order_l) != len(shared) or len(order_r) != len(shared):
                m7_dropped[f"{left}|{right}:overlap_lt_2"] += 1
                continue
            tau = kendall_tau(order_l, order_r)
            if tau is None:
                m7_dropped[f"{left}|{right}:tau_undefined"] += 1
                continue
            taus.append(tau)
        m7[f"{left}|{right}"] = {**_quantiles(taus), "floor": FLOOR_TAU}
    registry.record("M7'", "model_vs_model", {
        "n_human": 0, "per_arm": None, "arms_expected": (), "pairs": m7,
        "dropped": dict(sorted(m7_dropped.items())),
        "floor": FLOOR_TAU, "floor_note": "tau = 0 under independent orderings",
    })

    registry.assert_wired(accepted, arm_names)

    by_major: dict[str, Any] = {}
    for major in sorted({r["major"] for r in final_records}):
        subset_final = [r for r in final_records if r["major"] == major]
        subset_prelim = [r for r in prelim_records if r["major"] == major]
        cell: dict[str, Any] = {"n_prelim": len(subset_prelim), "n_final": len(subset_final)}
        for arm in arm_names:
            hits = []
            for record in subset_prelim:
                view = arm_for(record["group_id"], arm)
                if view is None:
                    continue
                if arm == "onealign":
                    picked = restrict(view["scores"], record["quad_slots"])
                    rank1 = picked[0] if picked else None
                else:
                    rank1 = view["rank1_by_quad"].get(record["quad_index"])
                if rank1 is not None:
                    hits.append(int(record["human_top1"] == rank1))
            cell[f"M1'_{arm}"] = _rate(hits)["value"]
        by_major[major] = cell

    analysis = {
        "schema": SCHEMA,
        "csv": str(csv_path),
        "item_key": str(key_path),
        "arms": arm_names,
        "arm_sources": {name: str(path) for name, _, path in
                        ((s.partition("=")[0], "=", s.partition("=")[2])
                         for s in (args.arm or []))},
        "n_csv_rows": len(rows),
        "n_items_in_key": len(items),
        "decisions_expected": key["groups_used"] * ITEMS_PER_GROUP,
        "reconciliation": reconciliation,
        "n_accepted": accepted,
        "n_rejected": dict(sorted(rejected.items())),
        "n_unknown_item_id": len(unknown),
        "unknown_item_id": sorted(unknown)[:50],
        "dropped_arm_cells": {a: dict(sorted(c.items())) for a, c in dropped_cells.items()},
        "floors": {
            "M1'": round(FLOOR_TOP1_OF_4, 6),
            "M2'": FLOOR_M2,
            "M3'": round(FLOOR_M3, 6),
            "M4'": FLOOR_M4,
            "M5'": {"top1": round(FLOOR_M5_TOP1, 6), "pair_order": FLOOR_M5_ORDER},
            "M7'": FLOOR_TAU,
        },
        "metrics_wired": sorted(registry.calls),
        "metrics": registry.calls,
        "by_major": by_major,
    }
    out_path = Path(args.out) if args.out else csv_path.parent / "analysis.json"
    out_path.write_text(
        json.dumps(analysis, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8", newline="\n",
    )

    lines = [
        f"csv_rows = {len(rows)}  key_items = {len(items)}  "
        f"decisions_expected = {analysis['decisions_expected']}  "
        f"accepted_prelim = {accepted['prelim']}  accepted_final = {accepted['final']}  "
        f"rejected = {sum(rejected.values())}  unknown = {len(unknown)}",
        f"arms = {', '.join(arm_names)}",
        "",
        _pad("metric", 10) + "".join(f"{arm:>14}" for arm in arm_names) + f"{'floor':>12}",
    ]
    for name in ("M1'", "M2'", "M3'", "M4'"):
        row = registry.calls[name]
        cells = "".join(f"{str(row['per_arm'][arm]['value']):>14}" for arm in arm_names)
        lines.append(_pad(name, 10) + cells + f"{str(row['floor']):>12}")
    m5row = registry.calls["M5'"]["per_arm"]["onealign"]
    lines.extend([
        "",
        f"M5' (OneAlign restricted to the human's own 4 finalists): "
        f"top1 = {m5row['top1']['value']} (n={m5row['top1']['n']}, floor "
        f"{m5row['top1']['floor']})  "
        f"pair_order = {m5row['pair_order']['value']} (n={m5row['pair_order']['n']}, "
        f"floor {m5row['pair_order']['floor']})",
        "",
        "M7' pairwise Kendall tau (model vs model, no human):",
    ])
    for pair, block in registry.calls["M7'"]["pairs"].items():
        lines.append(
            f"  {_pad(pair, 24)} n = {block['n_values']:>4}  mean = {block['mean']}  "
            f"median = {block['median']}  min = {block['min']}  max = {block['max']}"
        )
    m7_dropped_summary = registry.calls["M7'"]["dropped"]
    lines.extend([
        f"M7' dropped: {m7_dropped_summary}",
        "",
        f"floors: M2' E[J] = {FLOOR_M2['expectation']}  ({FLOOR_M2['formula']})",
        f"        M4' E[J] = {FLOOR_M4['expectation']}  ({FLOOR_M4['formula']})",
        "",
        _pad("major", 14) + f"{'n_prelim':>10}{'n_final':>9}"
        + "".join("{:>16}".format("M1'_" + arm) for arm in arm_names),
    ])
    for major, block in by_major.items():
        lines.append(
            _pad(major, 14) + f"{block['n_prelim']:>10}{block['n_final']:>9}"
            + "".join("{:>16}".format(str(block["M1'_" + arm])) for arm in arm_names)
        )
    lines.extend(["", f"written {out_path}"])
    print("\n".join(lines))
    return 0


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)

    build = sub.add_parser("build", help="render the blind tournament page from Phase A")
    build.add_argument("--build-root", type=Path,
                       default=Path("/home/bc/data/builds/epr049-aesth-20260825"))
    build.add_argument("--out-dir", type=Path,
                       default=Path("docs/assets/epr049_aesthq_20260825"))
    build.add_argument("--thumb-short-edge", type=int, default=THUMB_SHORT_EDGE)
    build.set_defaults(func=cmd_build)

    analyze = sub.add_parser("analyze", help="join the returned CSV to item_key.json")
    analyze.add_argument("--csv", type=Path, required=True)
    analyze.add_argument("--item-key", type=Path, default=None)
    analyze.add_argument("--arm", action="append", default=None,
                         help="name=path/to/groups.jsonl, repeatable")
    analyze.add_argument("--allow-partial", action="store_true",
                         help="skip the 500-decision reconciliation for an interim look "
                              "at an unfinished return")
    analyze.add_argument("--out", type=Path, default=None)
    analyze.set_defaults(func=cmd_analyze)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    return int(args.func(args))


if __name__ == "__main__":
    raise SystemExit(main())
