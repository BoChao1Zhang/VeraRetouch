"""Manual QC questionnaire for the offline source diagnoses (task card A5a).

``build``   deterministically samples 30 annotated sources out of a pinned manifest
            snapshot, copies a short-edge-640 display JPEG per item and emits a
            one-item-per-screen rating page (5 big buttons, localStorage, in-page CSV
            export) plus an ``item_key.json`` mapping and an empty ``diag_check.csv``.
``analyze`` reads the filled-in CSV (``item_id,rating,notes``), joins it with
            ``item_key.json`` and prints the rating counts / per-scene table.

Sampling rule (A5a, verbatim): order the snapshot ``source_id`` values by ``sha1(source_id)``
ascending, take the first 60, and keep the even indices (0, 2, ..., 58) -> 30 items.
The odd indices belong to a different task card and are never touched here.

Usage:
    python -m dataset_build.tools.diag_check_questionnaire build
    python -m dataset_build.tools.diag_check_questionnaire analyze \
        --csv docs/assets/lut_cluster_pilot_20260819/diag_check/diag_check.csv
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import html
import json
import sys
from pathlib import Path
from typing import Any, Mapping, Sequence

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

SCHEMA = "diag-check-questionnaire-v1"
DEFAULT_MANIFEST = Path(
    "/home/bc/data/agent_loop/local-v1/sources5k.annotated.jsonl"
)
# manifest snapshot pinned at read time (A5a): first 3954 lines,
# sha256 = 23848c53518f9fbf6683a25295128191061e9c4d31a6a50cb9091cc794e1b86a
DEFAULT_MAX_LINES = 3954
HEAD_POOL = 60  # first N of the sha1 order
STRIDE = 2      # even indices of that pool
PARITY = 0      # 0 = this card; 1 = the sibling card
DEFAULT_SHORT_EDGE = 640
DEFAULT_JPEG_QUALITY = 92

CSV_HEADER = "item_id,rating,notes"
RATING_MIN, RATING_MAX = 1, 5
RATING_LABELS = [
    (1, "严重不符或编造缺陷"),
    (2, "多处不符"),
    (3, "基本可用但有遗漏或夸大"),
    (4, "准确、小瑕疵"),
    (5, "准确且完整"),
]
# diagnosis field -> Chinese label, in display order
DIAG_FIELDS: list[tuple[str, str]] = [
    ("correction_needs", "需要修正 correction_needs"),
    ("preserve_intent", "需保留意图 preserve_intent"),
    ("enhancement_opportunities", "可增强机会 enhancement_opportunities"),
    ("forbidden_directions", "禁止方向 forbidden_directions"),
    ("evidence", "图像证据 evidence"),
]


# --------------------------------------------------------------------------- sampling


def sha1_hex(text: str) -> str:
    return hashlib.sha1(text.encode("utf-8")).hexdigest()


def read_manifest(path: Path, max_lines: int | None) -> tuple[list[dict[str, Any]], str, int]:
    """Return (rows, sha256 of the consumed byte span, n_lines)."""
    digest = hashlib.sha256()
    rows: list[dict[str, Any]] = []
    with path.open("rb") as handle:
        for index, raw in enumerate(handle):
            if max_lines is not None and index >= max_lines:
                break
            digest.update(raw)
            text = raw.decode("utf-8").strip()
            if not text:
                continue
            rows.append(json.loads(text))
    return rows, digest.hexdigest(), len(rows)


def select_items(rows: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    """sha1(source_id) ascending -> head 60 -> even indices -> 30 items."""
    by_id: dict[str, Mapping[str, Any]] = {}
    duplicates = 0
    for row in rows:
        source_id = str(row["source_id"])
        if source_id in by_id:
            duplicates += 1
            continue
        by_id[source_id] = row
    if duplicates:
        print(f"[warn] {duplicates} duplicate source_id rows ignored (first wins)",
              file=sys.stderr)
    order = sorted(by_id, key=lambda sid: (sha1_hex(sid), sid))
    pool = order[:HEAD_POOL]
    picked = [pool[index] for index in range(PARITY, len(pool), STRIDE)]
    items: list[dict[str, Any]] = []
    for slot, source_id in enumerate(picked, start=1):
        row = by_id[source_id]
        items.append({
            "item_id": f"diag_{slot:03d}",
            "source_id": source_id,
            "sha1": sha1_hex(source_id),
            "pool_index": order.index(source_id),
            "scene": row.get("scene"),
            "source_path": row["source_path"],
            "annotation_path": row["source_annotation_path"],
            "subject_description": (row.get("subject") or {}).get("description"),
            "subject_mask_area": (row.get("subject") or {}).get("mask_area"),
        })
    return items


def load_diagnosis(annotation_path: str) -> dict[str, Any]:
    payload = json.loads(Path(annotation_path).read_text(encoding="utf-8"))
    diagnosis = payload.get("diagnosis") or {}
    if not isinstance(diagnosis, dict):
        raise ValueError(f"{annotation_path}: diagnosis is not an object")
    return diagnosis


# --------------------------------------------------------------------------- images


def copy_display_image(src: Path, dst: Path, short_edge: int, quality: int) -> tuple[int, int]:
    from PIL import Image

    with Image.open(src) as image:
        image = image.convert("RGB")
        scale = short_edge / min(image.size)
        if scale < 1.0:
            image = image.resize(
                (max(1, round(image.width * scale)), max(1, round(image.height * scale))),
                Image.LANCZOS,
            )
        image.save(dst, format="JPEG", quality=quality, optimize=True)
        return image.width, image.height


# --------------------------------------------------------------------------- html


_HTML_STYLE = """
:root{color-scheme:light}
body{font-family:system-ui,'Noto Sans CJK SC',sans-serif;margin:16px auto;max-width:1080px;
background:#fff;color:#111;padding:0 12px}
h1{font-size:18px;margin:0 0 6px}
.hint{font-size:13px;color:#444;margin:0 0 10px;line-height:1.5}
.topbar{display:flex;align-items:center;gap:14px;flex-wrap:wrap;margin-bottom:8px}
#progress{font-size:20px;font-weight:700;font-variant-numeric:tabular-nums}
#done,#unrated,#meta{font-size:13px;color:#444}
.track{height:6px;background:#e6e6e6;border-radius:3px;overflow:hidden;margin:0 0 12px}
#bar{height:100%;background:#2b6cb0;width:0}
.pane{display:flex;gap:16px;flex-wrap:wrap;align-items:flex-start}
.pane>div{flex:1 1 420px;min-width:320px}
#src-img{display:block;width:100%;height:auto;border:1px solid #ccc;background:#202020}
#imgcap{font-size:12px;color:#666;margin-top:4px;word-break:break-all}
.diag{border:1px solid #ddd;border-radius:8px;padding:10px 12px;background:#fbfbfb;
max-height:none}
.diag h3{font-size:13px;margin:10px 0 4px;color:#2b6cb0;letter-spacing:.02em}
.diag h3:first-child{margin-top:0}
.diag ul{margin:0;padding-left:20px}
.diag li{font-size:13px;line-height:1.55;margin:2px 0}
.diag .flat{font-size:13px;line-height:1.55;margin:2px 0}
.chips{display:flex;gap:8px;flex-wrap:wrap;margin-bottom:8px}
.chip{font-size:12px;padding:3px 9px;border-radius:999px;background:#eef2f7;color:#234;
border:1px solid #cdd8e5}
.rates{display:flex;gap:8px;flex-wrap:wrap;margin:14px 0 8px}
.rate{flex:1 1 180px;padding:16px 10px;font-size:15px;line-height:1.35;cursor:pointer;
border:1px solid #bbb;border-radius:8px;background:#fafafa;color:#111;text-align:left}
.rate:hover{background:#f0f4f8}
.rate.on{background:#2b6cb0;border-color:#2b6cb0;color:#fff}
.rate b{font-size:20px;margin-right:8px}
.nav{display:flex;gap:10px;align-items:center;flex-wrap:wrap;margin:10px 0}
button{font-family:inherit}
.nav button,#export{padding:12px 20px;font-size:15px;cursor:pointer;border:1px solid #bbb;
border-radius:8px;background:#fafafa}
.nav button:disabled{opacity:.4;cursor:default}
#notes{width:100%;box-sizing:border-box;padding:8px;font-size:13px;font-family:inherit;
border:1px solid #ccc;border-radius:6px}
"""

_HTML_SCRIPT = """
const ITEMS = __ITEMS__;
const FIELDS = __FIELDS__;
const STORE_KEY = __STORE_KEY__;
const total = ITEMS.length;
let cursor = 0;
let state = {ratings: {}, notes: {}, cursor: 0};

function load() {
  try {
    const raw = window.localStorage.getItem(STORE_KEY);
    if (raw) {
      const parsed = JSON.parse(raw);
      state.ratings = (parsed && parsed.ratings) || {};
      state.notes = (parsed && parsed.notes) || {};
      cursor = Math.min(Math.max(parseInt(parsed && parsed.cursor, 10) || 0, 0),
                        Math.max(total - 1, 0));
    }
  } catch (err) {
    console.warn('localStorage load failed', err);
  }
}

function save() {
  state.cursor = cursor;
  try {
    window.localStorage.setItem(STORE_KEY, JSON.stringify(state));
  } catch (err) {
    console.warn('localStorage save failed', err);
  }
}

function ratedCount() {
  let n = 0;
  for (const item of ITEMS) { if (state.ratings[item.item_id]) { n += 1; } }
  return n;
}

function textNode(tag, text, cls) {
  const node = document.createElement(tag);
  node.textContent = text;
  if (cls) { node.className = cls; }
  return node;
}

function renderDiag(item) {
  const box = document.getElementById('diag');
  box.textContent = '';
  const chips = document.createElement('div');
  chips.className = 'chips';
  chips.appendChild(textNode('span', '场景 ' + (item.scene || '-'), 'chip'));
  chips.appendChild(textNode('span', '意图模式 intent_mode: ' +
    (item.intent_mode == null ? '-' : item.intent_mode), 'chip'));
  chips.appendChild(textNode('span', '置信度 confidence: ' +
    (item.confidence == null ? '-' : item.confidence), 'chip'));
  box.appendChild(chips);
  if (item.subject_description) {
    box.appendChild(textNode('h3', '主体 subject'));
    box.appendChild(textNode('p', item.subject_description, 'flat'));
  }
  for (const field of FIELDS) {
    const value = item.diagnosis[field.key];
    box.appendChild(textNode('h3', field.label));
    if (Array.isArray(value)) {
      if (!value.length) {
        box.appendChild(textNode('p', '（空）', 'flat'));
        continue;
      }
      const list = document.createElement('ul');
      for (const entry of value) {
        list.appendChild(textNode('li', String(entry)));
      }
      box.appendChild(list);
    } else {
      box.appendChild(textNode('p', value == null ? '（空）' : String(value), 'flat'));
    }
  }
}

function render() {
  const item = ITEMS[cursor];
  const img = document.getElementById('src-img');
  img.src = 'images/' + item.item_id + '.jpg';
  img.alt = item.item_id;
  document.getElementById('imgcap').textContent = item.item_id;
  document.getElementById('progress').textContent = (cursor + 1) + ' / ' + total;
  renderDiag(item);
  const done = ratedCount();
  const left = total - done;
  document.getElementById('done').textContent = '已评 ' + done + ' / ' + total;
  document.getElementById('bar').style.width = (total ? (done * 100 / total) : 0) + '%';
  const current = state.ratings[item.item_id] || 0;
  for (const btn of document.querySelectorAll('.rate')) {
    btn.classList.toggle('on', parseInt(btn.dataset.value, 10) === current);
  }
  document.getElementById('notes').value = state.notes[item.item_id] || '';
  document.getElementById('prev').disabled = cursor <= 0;
  document.getElementById('next').disabled = cursor >= total - 1;
  document.getElementById('export').textContent =
    left > 0 ? ('导出 CSV（还剩 ' + left + ' 条未评）') : '导出 CSV';
  document.getElementById('unrated').textContent =
    left > 0 ? ('未评 ' + left + ' 条，仍可导出') : '全部已评';
  window.scrollTo(0, 0);
}

function go(delta) {
  const next = cursor + delta;
  if (next < 0 || next >= total) { return; }
  cursor = next;
  save();
  render();
}

function rate(value) {
  if (value < __RMIN__ || value > __RMAX__) { return; }
  state.ratings[ITEMS[cursor].item_id] = value;
  save();
  if (cursor < total - 1) { cursor += 1; save(); }
  render();
}

function csvField(text) {
  const value = String(text == null ? '' : text);
  if (/[",\\r\\n]/.test(value)) { return '"' + value.replace(/"/g, '""') + '"'; }
  return value;
}

function buildCsv() {
  const lines = [__HEADER__];
  for (const item of ITEMS) {
    const rating = state.ratings[item.item_id] ? String(state.ratings[item.item_id]) : '';
    lines.push(item.item_id + ',' + rating + ',' + csvField(state.notes[item.item_id] || ''));
  }
  return lines.join('\\n') + '\\n';
}

function exportCsv() {
  const blob = new Blob([buildCsv()], {type: 'text/csv;charset=utf-8'});
  const url = URL.createObjectURL(blob);
  const link = document.createElement('a');
  link.href = url;
  link.download = 'diag_check.csv';
  document.body.appendChild(link);
  link.click();
  document.body.removeChild(link);
  setTimeout(function () { URL.revokeObjectURL(url); }, 2000);
}

window.addEventListener('DOMContentLoaded', function () {
  load();
  for (const btn of document.querySelectorAll('.rate')) {
    btn.addEventListener('click', function () { rate(parseInt(btn.dataset.value, 10)); });
  }
  document.getElementById('prev').addEventListener('click', function () { go(-1); });
  document.getElementById('next').addEventListener('click', function () { go(1); });
  document.getElementById('export').addEventListener('click', exportCsv);
  document.getElementById('notes').addEventListener('input', function (event) {
    state.notes[ITEMS[cursor].item_id] = event.target.value;
    save();
  });
  render();
  window.__READY__ = true;
});
"""


def _js_json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True).replace("</", "<\\/")


def storage_key(manifest_sha256: str) -> str:
    return f"{SCHEMA}:snapshot:{manifest_sha256[:16]}"


def write_html(out_dir: Path, items: Sequence[Mapping[str, Any]], manifest_sha256: str,
               n_lines: int) -> None:
    payload = [
        {
            "item_id": item["item_id"],
            "scene": item["scene"],
            "intent_mode": item["diagnosis"].get("intent_mode"),
            "confidence": item["diagnosis"].get("confidence"),
            "subject_description": item.get("subject_description"),
            "diagnosis": {key: item["diagnosis"].get(key) for key, _ in DIAG_FIELDS},
        }
        for item in items
    ]
    script = (
        _HTML_SCRIPT
        .replace("__ITEMS__", _js_json(payload))
        .replace("__FIELDS__", _js_json([{"key": key, "label": label}
                                         for key, label in DIAG_FIELDS]))
        .replace("__STORE_KEY__", _js_json(storage_key(manifest_sha256)))
        .replace("__HEADER__", _js_json(CSV_HEADER))
        .replace("__RMIN__", str(RATING_MIN))
        .replace("__RMAX__", str(RATING_MAX))
    )
    rate_buttons = "".join(
        f'<button type="button" class="rate" data-value="{value}">'
        f"<b>{value}</b>{html.escape(label)}</button>"
        for value, label in RATING_LABELS
    )
    parts = [
        '<!doctype html><html lang="zh"><head><meta charset="utf-8">',
        '<meta name="viewport" content="width=device-width,initial-scale=1">',
        "<title>离线诊断人工质检问卷 2026-08-19</title>",
        f"<style>{_HTML_STYLE}</style></head><body>",
        "<h1>离线诊断人工质检问卷</h1>",
        '<p class="hint">每屏一张 source 图 + 该图的离线诊断内容。请判断'
        "<b>该诊断与图片的相符与完整程度</b>，点下面 5 个按钮之一；"
        "评分自动保存在本机浏览器，标完点「导出 CSV」下载 diag_check.csv。</p>",
        '<div class="topbar"><span id="progress">- / -</span>'
        '<span id="done"></span><span id="unrated"></span>'
        f'<span id="meta">快照 {html.escape(manifest_sha256[:16])} · '
        f"{n_lines} 行 · 抽 {len(items)} 条</span></div>",
        '<div class="track"><div id="bar"></div></div>',
        '<div class="pane"><div><img id="src-img" alt="source">'
        '<div id="imgcap"></div></div>'
        '<div><div class="diag" id="diag"></div></div></div>',
        f'<div class="rates">{rate_buttons}</div>',
        '<div class="nav"><button type="button" id="prev">← 上一条</button>'
        '<button type="button" id="next">下一条 →</button>'
        '<button type="button" id="export">导出 CSV</button></div>',
        '<textarea id="notes" rows="3" '
        'placeholder="备注（可留空）：发现编造缺陷 / 漏诊 / 方向错误请注明"></textarea>',
        f"<script>{script}</script>",
        "</body></html>",
    ]
    (out_dir / "diag_check.html").write_text(
        "\n".join(parts), encoding="utf-8", newline="\n"
    )


def write_csv(out_dir: Path, items: Sequence[Mapping[str, Any]]) -> None:
    lines = [CSV_HEADER]
    lines.extend(f"{item['item_id']},," for item in items)
    (out_dir / "diag_check.csv").write_text(
        "\n".join(lines) + "\n", encoding="utf-8", newline="\n"
    )


def write_item_key(out_dir: Path, items: Sequence[Mapping[str, Any]], manifest: Path,
                   manifest_sha256: str, n_lines: int) -> None:
    payload = {
        "schema": SCHEMA,
        "manifest": str(manifest),
        "manifest_lines_used": n_lines,
        "manifest_sha256": manifest_sha256,
        "sampling": {
            "order_key": "sha1(source_id) ascending",
            "head_pool": HEAD_POOL,
            "stride": STRIDE,
            "parity": PARITY,
            "n_items": len(items),
        },
        "rating_scale": {str(value): label for value, label in RATING_LABELS},
        "display_order": [item["item_id"] for item in items],
        "items": {
            item["item_id"]: {
                "source_id": item["source_id"],
                "sha1": item["sha1"],
                "pool_index": item["pool_index"],
                "annotation_path": item["annotation_path"],
                "source_path": item["source_path"],
                "scene": item["scene"],
                "intent_mode": item["diagnosis"].get("intent_mode"),
                "confidence": item["diagnosis"].get("confidence"),
            }
            for item in items
        },
    }
    (out_dir / "item_key.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8", newline="\n",
    )


# --------------------------------------------------------------------------- commands


def cmd_build(args: argparse.Namespace) -> int:
    rows, manifest_sha256, n_lines = read_manifest(args.manifest, args.max_lines)
    items = select_items(rows)
    if len(items) != HEAD_POOL // STRIDE:
        raise SystemExit(
            f"expected {HEAD_POOL // STRIDE} items, got {len(items)} "
            f"(manifest has {len(rows)} rows)"
        )
    for item in items:
        item["diagnosis"] = load_diagnosis(item["annotation_path"])

    out_dir = args.out_dir
    (out_dir / "images").mkdir(parents=True, exist_ok=True)

    rendered = 0
    if not args.skip_images:
        for item in items:
            width, height = copy_display_image(
                Path(item["source_path"]),
                out_dir / "images" / f"{item['item_id']}.jpg",
                args.short_edge, args.jpeg_quality,
            )
            item["display_size"] = [width, height]
            rendered += 1

    write_item_key(out_dir, items, args.manifest, manifest_sha256, n_lines)
    write_csv(out_dir, items)
    write_html(out_dir, items, manifest_sha256, n_lines)

    scenes: dict[str, int] = {}
    modes: dict[str, int] = {}
    for item in items:
        scenes[str(item["scene"])] = scenes.get(str(item["scene"]), 0) + 1
        mode = str(item["diagnosis"].get("intent_mode"))
        modes[mode] = modes.get(mode, 0) + 1
    print(json.dumps({
        "out_dir": str(out_dir),
        "manifest": str(args.manifest),
        "manifest_lines_used": n_lines,
        "manifest_sha256": manifest_sha256,
        "n_items": len(items),
        "item_ids": [item["item_id"] for item in items],
        "source_ids": [item["source_id"] for item in items],
        "scene_counts": dict(sorted(scenes.items())),
        "intent_mode_counts": dict(sorted(modes.items())),
        "images_written": rendered,
    }, ensure_ascii=False, indent=2))
    return 0


def read_ratings(path: Path) -> list[tuple[str, str, str]]:
    rows: list[tuple[str, str, str]] = []
    with path.open("r", encoding="utf-8", newline="") as handle:
        for row in csv.DictReader(handle):
            key = row.get("item_id")
            if key is None:
                raise ValueError(f"{path}: no item_id column")
            rows.append((str(key).strip(), str(row.get("rating") or "").strip(),
                         str(row.get("notes") or "")))
    return rows


def cmd_analyze(args: argparse.Namespace) -> int:
    key = json.loads(args.item_key.read_text(encoding="utf-8"))
    items = key["items"]
    rows = read_ratings(args.csv)
    valid = {str(value) for value in range(RATING_MIN, RATING_MAX + 1)}

    unknown = sorted({item_id for item_id, _, _ in rows if item_id not in items})
    unfilled = sorted({item_id for item_id, value, _ in rows if value == ""})
    invalid = sorted({(item_id, value) for item_id, value, _ in rows
                      if value != "" and value not in valid})
    missing = sorted(set(items) - {item_id for item_id, _, _ in rows})

    counts = {str(value): 0 for value in range(RATING_MIN, RATING_MAX + 1)}
    per_scene: dict[str, dict[str, Any]] = {}
    per_mode: dict[str, dict[str, Any]] = {}
    total, rated = 0, 0
    for item_id, value, _ in rows:
        if item_id not in items or value not in valid:
            continue
        counts[value] += 1
        rated += 1
        total += int(value)
        for bucket, field in ((per_scene, "scene"), (per_mode, "intent_mode")):
            name = str(items[item_id].get(field))
            block = bucket.setdefault(name, {"n_rated": 0, "rating_sum": 0})
            block["n_rated"] += 1
            block["rating_sum"] += int(value)

    def table(bucket: Mapping[str, Mapping[str, Any]]) -> list[dict[str, Any]]:
        return [
            {"name": name, "n_rated": block["n_rated"],
             "mean_rating": round(block["rating_sum"] / block["n_rated"], 4)}
            for name, block in sorted(bucket.items())
        ]

    analysis = {
        "schema": "diag-check-questionnaire-analysis-v1",
        "csv": str(args.csv),
        "item_key": str(args.item_key),
        "rating_scale": {str(value): label for value, label in RATING_LABELS},
        "n_rows": len(rows),
        "n_items_in_key": len(items),
        "n_rated": rated,
        "mean_rating": round(total / rated, 4) if rated else None,
        "rating_counts": counts,
        "per_scene": table(per_scene),
        "per_intent_mode": table(per_mode),
        "n_unrated": len(unfilled),
        "unrated_item_ids": unfilled,
        "n_invalid": len(invalid),
        "invalid_rows": [list(entry) for entry in invalid],
        "n_unknown_item_ids": len(unknown),
        "unknown_item_ids": unknown,
        "n_missing_from_csv": len(missing),
        "missing_item_ids": missing,
    }
    out_path = args.out or (args.csv.parent / "analysis.json")
    out_path.write_text(
        json.dumps(analysis, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8", newline="\n",
    )
    lines = [
        f"n_rows = {len(rows)}  n_rated = {rated}  mean_rating = {analysis['mean_rating']}",
        "rating_counts = " + "/".join(
            f"{value}:{counts[str(value)]}" for value in range(RATING_MIN, RATING_MAX + 1)
        ),
    ]
    for title, block in (("[scene]", analysis["per_scene"]),
                         ("[intent_mode]", analysis["per_intent_mode"])):
        lines.append(f"{title} name  n_rated  mean_rating")
        for row in block:
            lines.append(f"{row['name']:<16} {row['n_rated']:<8} {row['mean_rating']:.4f}")
    lines.append(f"n_unrated = {len(unfilled)}  n_invalid = {len(invalid)}  "
                 f"n_unknown = {len(unknown)}  n_missing_from_csv = {len(missing)}")
    lines.append(f"analysis = {out_path}")
    print("\n".join(lines))
    return 0


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    default_out = REPO_ROOT / "docs/assets/lut_cluster_pilot_20260819/diag_check"
    sub = parser.add_subparsers(dest="command", required=True)

    build = sub.add_parser("build", help="sample 30 diagnoses and emit the QC questionnaire")
    build.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    build.add_argument("--max-lines", type=int, default=DEFAULT_MAX_LINES,
                       help="pin the growing manifest to its snapshot line count")
    build.add_argument("--out-dir", type=Path, default=default_out)
    build.add_argument("--short-edge", type=int, default=DEFAULT_SHORT_EDGE)
    build.add_argument("--jpeg-quality", type=int, default=DEFAULT_JPEG_QUALITY)
    build.add_argument("--skip-images", action="store_true",
                       help="emit html/csv/key only (determinism re-run)")
    build.set_defaults(func=cmd_build)

    analyze = sub.add_parser("analyze", help="read the filled-in CSV, print rating tables")
    analyze.add_argument("--csv", type=Path, default=default_out / "diag_check.csv")
    analyze.add_argument("--item-key", type=Path, default=default_out / "item_key.json")
    analyze.add_argument("--out", type=Path, default=None)
    analyze.set_defaults(func=cmd_analyze)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    return int(args.func(args))


if __name__ == "__main__":
    raise SystemExit(main())
