"""Blind LUT-pair questionnaire for cluster threshold acceptance (A1b, §2.5).

``build``   stratified-samples LUT pairs inside every ``style_major`` by normalised
            feature distance, renders one side-by-side probe collage per pair, and emits
            a blind questionnaire plus a separate ``pair_key.json``.  The HTML is a
            single-pair-at-a-time rating UI (5-point scale, localStorage, in-page CSV export).
``analyze`` reads the filled-in CSV (``pair_id,rating,notes``), joins it with
            ``pair_key.json`` and prints the per-bin mean rating / distinguishable rate
            (rating >= 3) table plus the derived threshold number.

Usage:
    python -m dataset_build.tools.lut_pair_questionnaire build --workers 8
    python -m dataset_build.tools.lut_pair_questionnaire analyze \
        --csv docs/assets/lut_cluster_pilot_20260819/questionnaire/questionnaire.csv
"""
from __future__ import annotations

import argparse
import concurrent.futures as futures
import csv
import html
import json
import sys
import time
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from dataset_build.tools.cluster_lut_effects import (  # noqa: E402
    FEATURE_SPEC,
    build_features,
    load_catalog,
    load_probe,
)

SCHEMA = "lut-pair-questionnaire-v1"
DEFAULT_SEED = 20260819
DEFAULT_BINS = 6
DEFAULT_BIN_MAX = 13.0  # 2 x t30 (6.504572), per task card
DEFAULT_PAIRS_PER_BIN = 10
CANDIDATE_CAP = 60  # per (bin, major) candidate pool size before round-robin


# --------------------------------------------------------------------------- sampling


def bin_edges(bin_count: int, upper: float) -> list[float]:
    return [upper * index / bin_count for index in range(bin_count + 1)]


def candidate_pools(preset_ids: Sequence[str], matrix: np.ndarray, edges: Sequence[float],
                    rng: np.random.Generator, cap: int) -> dict[int, list[tuple[float, str, str]]]:
    """Per-bin deterministic candidate sample of pairs drawn from one style_major."""
    from scipy.spatial.distance import pdist

    pools: dict[int, list[tuple[float, str, str]]] = {}
    if len(preset_ids) < 2:
        return pools
    distances = pdist(matrix, metric="euclidean")
    rows, cols = np.triu_indices(len(preset_ids), k=1)
    for index in range(len(edges) - 1):
        low, high = edges[index], edges[index + 1]
        if index == len(edges) - 2:
            mask = (distances >= low) & (distances <= high)
        else:
            mask = (distances >= low) & (distances < high)
        hits = np.nonzero(mask)[0]
        if hits.size == 0:
            continue
        order = rng.permutation(hits.size)[:cap]
        pool = []
        for slot in order.tolist():
            flat = int(hits[slot])
            a, b = preset_ids[int(rows[flat])], preset_ids[int(cols[flat])]
            if a > b:
                a, b = b, a
            pool.append((float(distances[flat]), a, b))
        pools[index] = pool
    return pools


def sample_pairs(records: Sequence[Any], matrix: np.ndarray, edges: Sequence[float],
                 seed: int, per_bin: int, cap: int,
                 exclude: Sequence[tuple[str, str]] | None = None,
                 per_bin_counts: Sequence[int] | None = None,
                 pair_prefix: str = "pair") -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Stratified pair sample.

    ``exclude`` drops already-used ``(preset_a, preset_b)`` combinations (sorted tuple);
    ``per_bin_counts`` overrides the flat ``per_bin`` quota bin by bin.  Both default to
    the A1b behaviour so the first questionnaire re-renders byte-identically.
    """
    excluded = {tuple(sorted(pair)) for pair in (exclude or ())}
    by_major: dict[str, list[int]] = {}
    for index, row in enumerate(records):
        by_major.setdefault(row.style_major, []).append(index)
    majors = sorted(by_major)

    pools: dict[int, dict[str, list[tuple[float, str, str]]]] = {
        index: {} for index in range(len(edges) - 1)
    }
    for major in majors:
        indices = by_major[major]
        rng = np.random.default_rng([seed, len(major), *(ord(ch) % 251 for ch in major[:16])])
        major_pools = candidate_pools(
            [records[i].preset_id for i in indices], matrix[indices], edges, rng, cap
        )
        for bin_index, pool in major_pools.items():
            pools[bin_index][major] = pool

    order_rng = np.random.default_rng(seed + 1)
    selected: list[dict[str, Any]] = []
    shortfall: dict[str, int] = {}
    for bin_index in range(len(edges) - 1):
        want = per_bin if per_bin_counts is None else int(per_bin_counts[bin_index])
        available = sorted(pools[bin_index])
        if not available:
            shortfall[str(bin_index)] = want
            continue
        rotation = [available[i] for i in order_rng.permutation(len(available)).tolist()]
        cursors = {major: 0 for major in rotation}
        picked: list[tuple[float, str, str, str]] = []
        seen: set[tuple[str, str]] = set()
        while len(picked) < want:
            progressed = False
            for major in rotation:
                if len(picked) >= want:
                    break
                pool = pools[bin_index][major]
                while cursors[major] < len(pool):
                    distance, a, b = pool[cursors[major]]
                    cursors[major] += 1
                    if (a, b) in seen or (a, b) in excluded:
                        continue
                    seen.add((a, b))
                    picked.append((distance, a, b, major))
                    progressed = True
                    break
            if not progressed:
                break
        if len(picked) < want:
            shortfall[str(bin_index)] = want - len(picked)
        for distance, a, b, major in picked:
            selected.append({
                "preset_a": a, "preset_b": b, "style_major": major,
                "distance": round(distance, 6), "bin": bin_index,
                "bin_low": round(edges[bin_index], 6), "bin_high": round(edges[bin_index + 1], 6),
            })

    for slot, entry in enumerate(selected, start=1):
        entry["pair_id"] = f"{pair_prefix}_{slot:03d}"

    display_rng = np.random.default_rng(seed + 2)
    display_order = [
        selected[i]["pair_id"] for i in display_rng.permutation(len(selected)).tolist()
    ]
    meta = {
        "shortfall_per_bin": shortfall,
        "majors_used": sorted({entry["style_major"] for entry in selected}),
        "display_order": display_order,
    }
    return selected, meta


# --------------------------------------------------------------------------- rendering


_PROBE: np.ndarray | None = None
_LOADER = None


def _init_worker(probe: np.ndarray, databuild: str) -> None:
    global _PROBE, _LOADER
    from dataset_build.agent_loop.source_reach import configured_lut_loader

    _PROBE = probe
    _LOADER = configured_lut_loader(Path(databuild))


def _render_side(lut_path: str, short_edge: int):
    from PIL import Image

    from dataset_build.src.construct.rendering import apply_lut_cpu_oracle

    grid, dmin, dmax = _LOADER.load(Path(lut_path))
    rendered = apply_lut_cpu_oracle(_PROBE, grid, domain_min=dmin, domain_max=dmax)
    image = Image.fromarray(np.clip(rendered * 255.0 + 0.5, 0, 255).astype(np.uint8))
    scale = short_edge / min(image.size)
    if scale < 1.0:
        image = image.resize(
            (max(1, round(image.width * scale)), max(1, round(image.height * scale))),
            Image.LANCZOS,
        )
    return image


def _render_pair(job: tuple[str, str, str, str, int, int]) -> tuple[str, bool, str]:
    from PIL import Image

    pair_id, lut_a, lut_b, out_path, short_edge, gap = job
    try:
        left = _render_side(lut_a, short_edge)
        right = _render_side(lut_b, short_edge)
        height = max(left.height, right.height)
        canvas = Image.new("RGB", (left.width + gap + right.width, height), (32, 32, 32))
        canvas.paste(left, (0, (height - left.height) // 2))
        canvas.paste(right, (left.width + gap, (height - right.height) // 2))
        canvas.save(out_path, format="PNG", optimize=True)
        return (pair_id, True, "")
    except Exception as exc:  # pragma: no cover - reported, never silent
        return (pair_id, False, f"{type(exc).__name__}: {exc}")


# --------------------------------------------------------------------------- outputs


CSV_HEADER = "pair_id,rating,notes"
RATING_LABELS = [
    (1, "看不出区别"),
    (2, "仔细看有极轻微差别"),
    (3, "可察觉差别"),
    (4, "差别较明显"),
    (5, "明显区别"),
]
RATING_MIN, RATING_MAX = 1, 5
DISTINGUISHABLE_MIN = 3  # rating >= 3 counts as distinguishable, per DECISIONS §2.5
# strength-perception section (A1d, §2.5 "强度档位感知校准"); used by the ext builder + analyze
STRENGTH_RATING_LABELS = [
    (1, "看不出变化"),
    (2, "轻微"),
    (3, "自然明显"),
    (4, "较强"),
    (5, "过度"),
]

_HTML_SCRIPT = """
const PAIRS = __PAIRS__;
const STORE_KEY = __STORE_KEY__;
const total = PAIRS.length;
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
  for (const pid of PAIRS) { if (state.ratings[pid]) { n += 1; } }
  return n;
}

function render() {
  const pid = PAIRS[cursor];
  const img = document.getElementById('pair-img');
  img.src = 'pairs/' + pid + '.png';
  img.alt = 'pair ' + (cursor + 1);
  document.getElementById('progress').textContent = (cursor + 1) + ' / ' + total;
  const done = ratedCount();
  const left = total - done;
  document.getElementById('done').textContent = '已评 ' + done + ' / ' + total;
  document.getElementById('bar').style.width = (total ? (done * 100 / total) : 0) + '%';
  const current = state.ratings[pid] || 0;
  for (const btn of document.querySelectorAll('.rate')) {
    btn.classList.toggle('on', parseInt(btn.dataset.value, 10) === current);
  }
  document.getElementById('notes').value = state.notes[pid] || '';
  document.getElementById('prev').disabled = cursor <= 0;
  document.getElementById('next').disabled = cursor >= total - 1;
  document.getElementById('export').textContent =
    left > 0 ? ('导出 CSV（还剩 ' + left + ' 对未评）') : '导出 CSV';
  document.getElementById('unrated').textContent =
    left > 0 ? ('未评 ' + left + ' 对，仍可导出') : '全部已评';
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
  state.ratings[PAIRS[cursor]] = value;
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
  for (const pid of PAIRS) {
    const rating = state.ratings[pid] ? String(state.ratings[pid]) : '';
    lines.push(pid + ',' + rating + ',' + csvField(state.notes[pid] || ''));
  }
  return lines.join('\\n') + '\\n';
}

function exportCsv() {
  const blob = new Blob([buildCsv()], {type: 'text/csv;charset=utf-8'});
  const url = URL.createObjectURL(blob);
  const link = document.createElement('a');
  link.href = url;
  link.download = 'questionnaire.csv';
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
  for (const btn of document.querySelectorAll('.rate')) {
    btn.addEventListener('click', function () { rate(parseInt(btn.dataset.value, 10)); });
  }
  document.getElementById('prev').addEventListener('click', function () { go(-1); });
  document.getElementById('next').addEventListener('click', function () { go(1); });
  document.getElementById('export').addEventListener('click', exportCsv);
  document.getElementById('notes').addEventListener('input', function (event) {
    state.notes[PAIRS[cursor]] = event.target.value;
    save();
  });
  document.addEventListener('keydown', function (event) {
    if (event.ctrlKey || event.metaKey || event.altKey || inField(event.target)) { return; }
    if (event.key >= '1' && event.key <= '5') {
      rate(parseInt(event.key, 10));
      event.preventDefault();
    } else if (event.key === 'ArrowLeft') {
      go(-1);
      event.preventDefault();
    } else if (event.key === 'ArrowRight') {
      go(1);
      event.preventDefault();
    }
  });
  render();
  window.__READY__ = true;
});
"""

_HTML_STYLE = """
:root{color-scheme:light}
body{font-family:system-ui,'Noto Sans CJK SC',sans-serif;margin:16px auto;max-width:1000px;
background:#fff;color:#111}
h1{font-size:18px;margin:0 0 6px}
.hint{font-size:13px;color:#444;margin:0 0 10px}
.topbar{display:flex;align-items:center;gap:14px;flex-wrap:wrap;margin-bottom:8px}
#progress{font-size:20px;font-weight:700;font-variant-numeric:tabular-nums}
#done,#unrated{font-size:13px;color:#444}
.track{height:6px;background:#e6e6e6;border-radius:3px;overflow:hidden;margin:0 0 12px}
#bar{height:100%;background:#2b6cb0;width:0}
#pair-img{display:block;width:100%;height:auto;border:1px solid #ccc;background:#202020}
.rates{display:flex;gap:8px;flex-wrap:wrap;margin:12px 0 8px}
.rate{flex:1 1 160px;padding:10px 8px;font-size:14px;line-height:1.35;cursor:pointer;
border:1px solid #bbb;border-radius:6px;background:#fafafa;color:#111;text-align:left}
.rate:hover{background:#f0f4f8}
.rate.on{background:#2b6cb0;border-color:#2b6cb0;color:#fff}
.rate b{font-size:16px;margin-right:6px}
.nav{display:flex;gap:10px;align-items:center;flex-wrap:wrap;margin:10px 0}
button{font-family:inherit}
.nav button,#export{padding:8px 14px;font-size:14px;cursor:pointer;border:1px solid #bbb;
border-radius:6px;background:#fafafa}
.nav button:disabled{opacity:.4;cursor:default}
#notes{width:100%;box-sizing:border-box;padding:6px;font-size:13px;font-family:inherit;
border:1px solid #ccc;border-radius:6px}
.keys{font-size:12px;color:#666}
.fail{color:#a00;font-size:13px}
"""


def _js_json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False).replace("</", "<\\/")


def storage_key(seed: int) -> str:
    return f"{SCHEMA}:seed:{seed}"


def write_questionnaire_html(out_dir: Path, display_order: Sequence[str],
                             failures: Sequence[tuple[str, str]], seed: int) -> None:
    script = (
        _HTML_SCRIPT
        .replace("__PAIRS__", _js_json(list(display_order)))
        .replace("__STORE_KEY__", _js_json(storage_key(seed)))
        .replace("__HEADER__", _js_json(CSV_HEADER))
        .replace("__RMIN__", str(RATING_MIN))
        .replace("__RMAX__", str(RATING_MAX))
    )
    rate_buttons = "".join(
        f'<button type="button" class="rate" data-value="{value}">'
        f"<b>{value}</b>{html.escape(label)}</button>"
        for value, label in RATING_LABELS
    )
    fail_block = ""
    if failures:
        items = "".join(
            f"<li>{html.escape(pair_id)}: {html.escape(detail)}</li>"
            for pair_id, detail in failures
        )
        fail_block = f'<div class="fail"><b>渲染失败</b><ul>{items}</ul></div>'
    parts = [
        '<!doctype html><html lang="zh"><head><meta charset="utf-8">',
        '<meta name="viewport" content="width=device-width,initial-scale=1">',
        "<title>LUT 对盲标问卷 2026-08-19</title>",
        f"<style>{_HTML_STYLE}</style></head><body>",
        "<h1>LUT 对盲标问卷</h1>",
        '<p class="hint">左右两张是同一张探针图套两个不同 LUT 的结果。给这一对的差别打分，'
        "评分自动保存在本机浏览器；标完点「导出 CSV」下载填好的 questionnaire.csv。</p>",
        '<div class="topbar"><span id="progress">- / -</span>'
        '<span id="done"></span><span id="unrated"></span>'
        '<span class="keys">键盘：1–5 评分并跳下一对，← / → 翻页</span></div>',
        '<div class="track"><div id="bar"></div></div>',
        '<img id="pair-img" alt="pair">',
        f'<div class="rates">{rate_buttons}</div>',
        '<div class="nav"><button type="button" id="prev">← 上一对</button>'
        '<button type="button" id="next">下一对 →</button>'
        '<button type="button" id="export">导出 CSV</button></div>',
        '<textarea id="notes" rows="2" placeholder="备注（可留空）"></textarea>',
        fail_block,
        f"<script>{script}</script>",
        "</body></html>",
    ]
    (out_dir / "questionnaire.html").write_text(
        "\n".join(part for part in parts if part), encoding="utf-8", newline="\n"
    )


def write_questionnaire_csv(out_dir: Path, display_order: Sequence[str]) -> None:
    lines = [CSV_HEADER]
    lines.extend(f"{pair_id},," for pair_id in display_order)
    (out_dir / "questionnaire.csv").write_text(
        "\n".join(lines) + "\n", encoding="utf-8", newline="\n"
    )


def write_pair_key(out_dir: Path, pairs: Sequence[dict[str, Any]], meta: dict[str, Any],
                   edges: Sequence[float], seed: int, probe_source: str) -> None:
    payload = {
        "schema": SCHEMA,
        "feature_spec": FEATURE_SPEC,
        "seed": seed,
        "probe_source": probe_source,
        "bin_edges": [round(edge, 6) for edge in edges],
        "shortfall_per_bin": meta["shortfall_per_bin"],
        "majors_used": meta["majors_used"],
        "display_order": list(meta["display_order"]),
        "pairs": {
            entry["pair_id"]: {
                "preset_a": entry["preset_a"],
                "preset_b": entry["preset_b"],
                "style_major": entry["style_major"],
                "distance": entry["distance"],
                "bin": entry["bin"],
                "bin_low": entry["bin_low"],
                "bin_high": entry["bin_high"],
            }
            for entry in pairs
        },
    }
    (out_dir / "pair_key.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8", newline="\n",
    )


# --------------------------------------------------------------------------- commands


def cmd_build(args: argparse.Namespace) -> int:
    catalog, databuild, _ = load_catalog(args.config)
    records = list(catalog.records)
    matrix, _, _ = build_features(records)
    edges = bin_edges(args.bins, args.bin_max)
    pairs, meta = sample_pairs(records, matrix, edges, args.seed, args.pairs_per_bin,
                               args.candidate_cap)

    out_dir = args.out_dir
    (out_dir / "pairs").mkdir(parents=True, exist_ok=True)

    failures: list[tuple[str, str]] = []
    elapsed = 0.0
    probe_source = "(skipped)"
    if not args.skip_render:
        probe, probe_source = load_probe(args.probe_jsonl, args.probe_short_edge)
        jobs = [
            (entry["pair_id"], catalog.by_id[entry["preset_a"]].path,
             catalog.by_id[entry["preset_b"]].path,
             str(out_dir / "pairs" / f"{entry['pair_id']}.png"), args.side_edge, args.gap)
            for entry in sorted(pairs, key=lambda item: item["pair_id"])
        ]
        started = time.time()
        with futures.ProcessPoolExecutor(
            max_workers=args.workers, initializer=_init_worker,
            initargs=(probe, str(databuild)),
        ) as pool:
            for pair_id, ok, detail in pool.map(_render_pair, jobs, chunksize=2):
                if not ok:
                    failures.append((pair_id, detail))
        elapsed = time.time() - started

    write_pair_key(out_dir, pairs, meta, edges, args.seed, probe_source)
    write_questionnaire_csv(out_dir, meta["display_order"])
    write_questionnaire_html(out_dir, meta["display_order"], failures, args.seed)

    per_bin: dict[str, dict[str, int]] = {}
    for entry in pairs:
        block = per_bin.setdefault(str(entry["bin"]), {})
        block[entry["style_major"]] = block.get(entry["style_major"], 0) + 1
    print(json.dumps({
        "out_dir": str(out_dir),
        "pairs": len(pairs),
        "bin_edges": [round(edge, 6) for edge in edges],
        "per_bin_major_counts": per_bin,
        "majors_used": meta["majors_used"],
        "shortfall_per_bin": meta["shortfall_per_bin"],
        "rendered": 0 if args.skip_render else len(pairs),
        "render_failures": failures,
        "render_seconds": round(elapsed, 2),
        "probe_source": probe_source,
    }, ensure_ascii=False, indent=2))
    return 0


def read_ratings(path: Path) -> list[tuple[str, str]]:
    """Read ``<id>,rating,notes``; the id column is ``pair_id`` or ``item_id``."""
    rows: list[tuple[str, str]] = []
    with path.open("r", encoding="utf-8", newline="") as handle:
        for row in csv.DictReader(handle):
            key = row.get("pair_id")
            if key is None:
                key = row.get("item_id")
            if key is None:
                raise ValueError(f"{path}: no pair_id/item_id column")
            rows.append((str(key).strip(), str(row.get("rating") or "").strip()))
    return rows


def bin_table(pairs: Mapping[str, Any], rows: Sequence[tuple[str, str]],
              edges: Sequence[float]) -> list[dict[str, Any]]:
    valid = {str(value) for value in range(RATING_MIN, RATING_MAX + 1)}
    buckets: dict[int, dict[str, int]] = {
        index: {"n_pairs": 0, "n_rated": 0, "n_distinguishable": 0, "rating_sum": 0}
        for index in range(len(edges) - 1)
    }
    for entry in pairs.values():
        buckets[int(entry["bin"])]["n_pairs"] += 1
    for pair_id, value in rows:
        if pair_id not in pairs or value not in valid:
            continue
        rating = int(value)
        block = buckets[int(pairs[pair_id]["bin"])]
        block["n_rated"] += 1
        block["rating_sum"] += rating
        block["n_distinguishable"] += int(rating >= DISTINGUISHABLE_MIN)
    table: list[dict[str, Any]] = []
    for index in sorted(buckets):
        block = buckets[index]
        n_rated = block["n_rated"]
        rate = (block["n_distinguishable"] / n_rated) if n_rated else None
        mean = (block["rating_sum"] / n_rated) if n_rated else None
        table.append({
            "bin": index,
            "bin_low": edges[index],
            "bin_high": edges[index + 1],
            "n_pairs": block["n_pairs"],
            "n_rated": n_rated,
            "n_unrated": block["n_pairs"] - n_rated,
            "mean_rating": None if mean is None else round(mean, 4),
            "n_distinguishable": block["n_distinguishable"],
            "distinguishable_rate": None if rate is None else round(rate, 4),
        })
    return table


def strength_table(items: Mapping[str, Any],
                   rows: Sequence[tuple[str, str]]) -> list[dict[str, Any]]:
    """Per whole-image ΔE target: mean rating + per-rating counts."""
    valid = {str(value) for value in range(RATING_MIN, RATING_MAX + 1)}
    groups: dict[float, dict[str, Any]] = {}
    for item_id, entry in items.items():
        block = groups.setdefault(float(entry["de_target"]), {
            "n_items": 0, "n_rated": 0, "rating_sum": 0,
            "counts": {str(v): 0 for v in range(RATING_MIN, RATING_MAX + 1)},
            "de_measured": [], "ids": set(),
        })
        block["n_items"] += 1
        block["ids"].add(item_id)
        measured = entry.get("de_measured_full")
        if measured is not None:
            block["de_measured"].append(float(measured))
    for item_id, value in rows:
        if item_id not in items or value not in valid:
            continue
        block = groups[float(items[item_id]["de_target"])]
        block["n_rated"] += 1
        block["rating_sum"] += int(value)
        block["counts"][value] += 1
    table: list[dict[str, Any]] = []
    for target in sorted(groups):
        block = groups[target]
        measured = block["de_measured"]
        table.append({
            "de_target": target,
            "n_items": block["n_items"],
            "n_rated": block["n_rated"],
            "n_unrated": block["n_items"] - block["n_rated"],
            "mean_rating": (round(block["rating_sum"] / block["n_rated"], 4)
                            if block["n_rated"] else None),
            "rating_counts": block["counts"],
            "de_measured_min": round(min(measured), 4) if measured else None,
            "de_measured_max": round(max(measured), 4) if measured else None,
            "de_measured_mean": (round(sum(measured) / len(measured), 4)
                                 if measured else None),
        })
    return table


def cmd_analyze(args: argparse.Namespace) -> int:
    key = json.loads(args.pair_key.read_text(encoding="utf-8"))
    pairs = key["pairs"]
    edges = key["bin_edges"]

    valid_ratings = {str(value) for value in range(RATING_MIN, RATING_MAX + 1)}

    rows = read_ratings(args.csv)

    unknown = [pair_id for pair_id, _ in rows if pair_id not in pairs]
    unfilled = [pair_id for pair_id, value in rows if value == ""]
    invalid = [
        (pair_id, value) for pair_id, value in rows
        if value != "" and value not in valid_ratings
    ]
    missing = sorted(set(pairs) - {pair_id for pair_id, _ in rows})

    table = bin_table(pairs, rows, edges)

    def threshold(rows_table: Sequence[dict[str, Any]]) -> tuple[list[int], float | None]:
        hits = [
            row["bin"] for row in rows_table
            if row["distinguishable_rate"] is not None
            and row["distinguishable_rate"] <= args.rate_cap
        ]
        return sorted(hits), (edges[max(hits) + 1] if hits else None)

    qualifying, recommended = threshold(table)

    ext_block: dict[str, Any] | None = None
    if args.ext_csv is not None:
        ext_key_path = args.ext_pair_key or (args.ext_csv.parent / "pair_key_ext.json")
        ext_key = json.loads(ext_key_path.read_text(encoding="utf-8"))
        if [round(float(edge), 6) for edge in ext_key["bin_edges"]] != \
                [round(float(edge), 6) for edge in edges]:
            raise ValueError(
                f"bin_edges mismatch: {args.pair_key} vs {ext_key_path}; refusing to merge")
        ext_pairs = ext_key.get("pairs") or {}
        ext_items = ext_key.get("strength_items") or {}
        ext_rows = read_ratings(args.ext_csv)
        overlap = sorted(set(ext_pairs) & set(pairs))
        merged_pairs = {**pairs, **ext_pairs}
        merged_rows = rows + [
            (item_id, value) for item_id, value in ext_rows if item_id not in pairs
        ]
        merged_table = bin_table(merged_pairs, merged_rows, edges)
        merged_qualifying, merged_recommended = threshold(merged_table)
        ext_unknown = [
            item_id for item_id, _ in ext_rows
            if item_id not in ext_pairs and item_id not in ext_items
        ]
        ext_block = {
            "ext_csv": str(args.ext_csv),
            "ext_pair_key": str(ext_key_path),
            "n_ext_pairs_in_key": len(ext_pairs),
            "n_strength_items_in_key": len(ext_items),
            "n_ext_rows": len(ext_rows),
            "n_ext_unknown_ids": len(ext_unknown),
            "ext_unknown_ids": sorted(ext_unknown),
            "n_pair_id_overlap": len(overlap),
            "pair_id_overlap": overlap,
            "per_bin_ext": bin_table(ext_pairs, ext_rows, edges),
            "per_bin_merged": merged_table,
            "merged_qualifying_bins": merged_qualifying,
            "merged_recommended_threshold": (None if merged_recommended is None
                                             else round(merged_recommended, 6)),
            "strength": strength_table(ext_items, ext_rows),
        }

    analysis = {
        "schema": "lut-pair-questionnaire-analysis-v2",
        "csv": str(args.csv),
        "pair_key": str(args.pair_key),
        "rating_scale": {str(value): label for value, label in RATING_LABELS},
        "distinguishable_min_rating": DISTINGUISHABLE_MIN,
        "rate_cap": args.rate_cap,
        "n_rows": len(rows),
        "n_pairs_in_key": len(pairs),
        "n_unrated": len(unfilled),
        "unrated_pair_ids": sorted(unfilled),
        "n_invalid": len(invalid),
        "invalid_rows": sorted(invalid),
        "n_unknown_pair_ids": len(unknown),
        "unknown_pair_ids": sorted(unknown),
        "n_missing_from_csv": len(missing),
        "missing_pair_ids": missing,
        "per_bin": table,
        "qualifying_bins": sorted(qualifying),
        "recommended_threshold": None if recommended is None else round(recommended, 6),
    }
    if ext_block is not None:
        analysis["ext"] = ext_block
        analysis["strength_rating_scale"] = {
            str(value): label for value, label in STRENGTH_RATING_LABELS
        }
    out_path = args.out or (args.csv.parent / "analysis.json")
    out_path.write_text(
        json.dumps(analysis, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8", newline="\n",
    )
    header = "bin  low        high       n_pairs  n_rated  n_unrated  mean_rating  n_dist  rate"

    def render_table(title: str, rows_table: Sequence[dict[str, Any]]) -> list[str]:
        out = [title, header]
        for row in rows_table:
            rate = ("-" if row["distinguishable_rate"] is None
                    else f"{row['distinguishable_rate']:.4f}")
            mean = "-" if row["mean_rating"] is None else f"{row['mean_rating']:.4f}"
            out.append(
                f"{row['bin']:<4} {row['bin_low']:<10.6f} {row['bin_high']:<10.6f} "
                f"{row['n_pairs']:<8} {row['n_rated']:<8} {row['n_unrated']:<10} "
                f"{mean:<12} {row['n_distinguishable']:<7} {rate}"
            )
        return out

    lines = render_table("[main]", table)
    if ext_block is not None:
        lines.extend(render_table("[ext]", ext_block["per_bin_ext"]))
        lines.extend(render_table("[merged]", ext_block["per_bin_merged"]))
        lines.append("[strength] de_target  n_items  n_rated  mean_rating  counts_1..5  "
                     "de_measured_min..max")
        for row in ext_block["strength"]:
            mean = "-" if row["mean_rating"] is None else f"{row['mean_rating']:.4f}"
            counts = "/".join(str(row["rating_counts"][str(v)])
                              for v in range(RATING_MIN, RATING_MAX + 1))
            span = ("-" if row["de_measured_min"] is None
                    else f"{row['de_measured_min']:.4f}..{row['de_measured_max']:.4f}")
            lines.append(
                f"{row['de_target']:<10.2f} {row['n_items']:<8} {row['n_rated']:<8} "
                f"{mean:<12} {counts:<12} {span}"
            )
        lines.append(f"merged_qualifying_bins = {ext_block['merged_qualifying_bins']}")
        lines.append("merged_recommended_threshold = "
                     f"{ext_block['merged_recommended_threshold']}")
    lines.append(f"rate_cap = {args.rate_cap}  distinguishable = rating >= {DISTINGUISHABLE_MIN}")
    lines.append(f"qualifying_bins = {sorted(qualifying)}")
    lines.append(f"recommended_threshold = {analysis['recommended_threshold']}")
    lines.append(f"n_unrated = {len(unfilled)}  unrated_pair_ids = {sorted(unfilled)}")
    lines.append(f"n_invalid = {len(invalid)}  n_missing_from_csv = {len(missing)}")
    lines.append(f"analysis = {out_path}")
    print("\n".join(lines))
    return 0


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    default_out = REPO_ROOT / "docs/assets/lut_cluster_pilot_20260819/questionnaire"
    sub = parser.add_subparsers(dest="command", required=True)

    build = sub.add_parser("build", help="sample pairs, render collages, emit blind questionnaire")
    build.add_argument("--config", type=Path,
                       default=REPO_ROOT / "configs/agent_loop.terra-smoke.toml")
    build.add_argument("--out-dir", type=Path, default=default_out)
    build.add_argument("--probe-jsonl", type=Path,
                       default=REPO_ROOT / "configs/agent_loop.smoke5.jsonl")
    build.add_argument("--seed", type=int, default=DEFAULT_SEED)
    build.add_argument("--bins", type=int, default=DEFAULT_BINS)
    build.add_argument("--bin-max", type=float, default=DEFAULT_BIN_MAX)
    build.add_argument("--pairs-per-bin", type=int, default=DEFAULT_PAIRS_PER_BIN)
    build.add_argument("--candidate-cap", type=int, default=CANDIDATE_CAP)
    build.add_argument("--workers", type=int, default=8)
    build.add_argument("--probe-short-edge", type=int, default=512)
    build.add_argument("--side-edge", type=int, default=448)
    build.add_argument("--gap", type=int, default=8)
    build.add_argument("--skip-render", action="store_true",
                       help="emit csv/key/html only (determinism re-run)")
    build.set_defaults(func=cmd_build)

    analyze = sub.add_parser("analyze", help="read the filled-in CSV, print per-bin rates")
    analyze.add_argument("--csv", type=Path, default=default_out / "questionnaire.csv")
    analyze.add_argument("--pair-key", type=Path, default=default_out / "pair_key.json")
    analyze.add_argument("--ext-csv", type=Path, default=None,
                         help="extended questionnaire CSV (A1d): merged pair curve + strength")
    analyze.add_argument("--ext-pair-key", type=Path, default=None,
                         help="defaults to <ext-csv dir>/pair_key_ext.json")
    analyze.add_argument("--rate-cap", type=float, default=0.20)
    analyze.add_argument("--out", type=Path, default=None)
    analyze.set_defaults(func=cmd_analyze)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    return int(args.func(args))


if __name__ == "__main__":
    raise SystemExit(main())
