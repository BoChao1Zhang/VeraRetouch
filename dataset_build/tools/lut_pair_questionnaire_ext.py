"""Extended blind LUT questionnaire (A1d, DECISIONS_agent_loop_annotation_optA_20260819 §2.5).

Two sections, one page, its own output directory / localStorage key, fully separate from the
first 60-pair questionnaire (which this module never reads for anything but the exclusion set):

* section 1 -- 140 **new** LUT pairs (same stratified sampling as A1b, 6 distance bins,
  ``(preset_a, preset_b)`` combinations already used by the first questionnaire excluded),
  4 different scene probes rotated deterministically by pair order.
* section 2 -- 40 strength-perception calibration items: 8 LUTs (8 distinct ``style_major``,
  dispersed ``de_med``) x 5 whole-image mean-CIEDE2000 targets {2.5, 4.0, 5.5, 7.0, 8.5};
  ``rendered = source + s * (lut(source) - source)`` with ``s`` solved by 5-step bisection on a
  4096-pixel sample; shown as ``source | rendered``.

Usage:
    python -m dataset_build.tools.lut_pair_questionnaire_ext build --workers 8
    python -m dataset_build.tools.lut_pair_questionnaire analyze \
        --csv  .../questionnaire/questionnaire.csv \
        --ext-csv .../questionnaire_ext/questionnaire_ext.csv
"""
from __future__ import annotations

import argparse
import concurrent.futures as futures
import html
import json
import sys
import time
from pathlib import Path
from typing import Any, Sequence

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from dataset_build.tools.cluster_lut_effects import (  # noqa: E402
    FEATURE_SPEC,
    build_features,
    load_catalog,
)
from dataset_build.tools.lut_pair_questionnaire import (  # noqa: E402
    CANDIDATE_CAP,
    DEFAULT_BIN_MAX,
    DEFAULT_BINS,
    RATING_LABELS,
    STRENGTH_RATING_LABELS,
    _HTML_STYLE,
    _js_json,
    bin_edges,
    sample_pairs,
)

EXT_SCHEMA = "lut-pair-questionnaire-ext-v1"
DEFAULT_EXT_SEED = 20260820
DEFAULT_EXT_PAIRS = 140
DEFAULT_PROBES = 4
DE_TARGETS = (2.5, 4.0, 5.5, 7.0, 8.5)
STRENGTH_LUTS = 8
BISECT_STEPS = 5
BISECT_SAMPLE = 4096

_SCENE_SAMPLES = REPO_ROOT / "docs/assets/local_retouch_agent_loop_20260818/scene_samples_low_512"
DEFAULT_PROBE_MANIFESTS = (
    REPO_ROOT / "configs/agent_loop.smoke5.jsonl",
    _SCENE_SAMPLES / "source_manifest.jsonl",
)
DEFAULT_PROBE_FALLBACK_DIR = _SCENE_SAMPLES / "rendered"

PAIR_PREFIX = "ext_pair"
STRENGTH_PREFIX = "str_item"

CSV_HEADER = "item_id,rating,notes"
RATING_MIN, RATING_MAX = 1, 5
PAIR_RATING_LABELS = RATING_LABELS


def storage_key(seed: int) -> str:
    return f"{EXT_SCHEMA}:seed:{seed}"


# --------------------------------------------------------------------------- probes


def resolve_source(row: dict[str, Any], fallback_dir: Path | None) -> Path | None:
    """``source_path`` when present on disk, else ``<fallback_dir>/<scene>/source.jpg``."""
    source = Path(str(row.get("source_path") or ""))
    if source.is_file():
        return source
    if fallback_dir is not None:
        alternate = fallback_dir / str(row.get("scene") or "") / "source.jpg"
        if alternate.is_file():
            return alternate
    return None


def load_probes(manifests: Sequence[Path], short_edge: int, count: int,
                fallback_dir: Path | None) -> list[dict[str, Any]]:
    """First readable row per unseen ``scene``, manifest order, until ``count`` scenes."""
    from PIL import Image

    picked: list[dict[str, Any]] = []
    seen: set[str] = set()
    skipped: list[str] = []
    for jsonl in manifests:
        if not jsonl.is_file():
            continue
        with jsonl.open("r", encoding="utf-8") as handle:
            for line in handle:
                if not line.strip():
                    continue
                row = json.loads(line)
                scene = str(row.get("scene") or "")
                if scene in seen:
                    continue
                source = resolve_source(row, fallback_dir)
                if source is None:
                    skipped.append(f"{jsonl.name}:{scene}:{row.get('source_path')}")
                    continue
                seen.add(scene)
                with Image.open(source) as image:
                    image = image.convert("RGB")
                    scale = short_edge / min(image.size)
                    if scale < 1.0:
                        image = image.resize(
                            (max(1, round(image.width * scale)),
                             max(1, round(image.height * scale))),
                            Image.LANCZOS,
                        )
                    array = np.asarray(image, dtype=np.float32) / 255.0
                picked.append({"scene": scene, "source": str(source), "array": array,
                               "manifest": str(jsonl)})
                if len(picked) >= count:
                    return picked
    raise ValueError(
        f"only {len(picked)} readable distinct scenes, need {count}; missing: {skipped}")


# --------------------------------------------------------------------------- selection


def existing_combinations(pair_key: Path) -> list[tuple[str, str]]:
    if not pair_key.is_file():
        return []
    payload = json.loads(pair_key.read_text(encoding="utf-8"))
    return [
        (str(entry["preset_a"]), str(entry["preset_b"]))
        for entry in (payload.get("pairs") or {}).values()
    ]


def bin_quota(total: int, bins: int) -> list[int]:
    base, extra = divmod(total, bins)
    return [base + (1 if index < extra else 0) for index in range(bins)]


def pick_strength_luts(records: Sequence[Any], count: int) -> list[Any]:
    """``count`` distinct style_major, one LUT each, at dispersed de_med quantiles.

    Majors ranked by (-record count, name); the k-th major contributes the record at
    de_med quantile ``(k + 0.5) / count`` of that major.  No RNG.
    """
    by_major: dict[str, list[Any]] = {}
    for row in records:
        by_major.setdefault(row.style_major, []).append(row)
    majors = sorted(by_major, key=lambda name: (-len(by_major[name]), name))[:count]
    if len(majors) < count:
        raise ValueError(f"catalog has {len(majors)} style_major, need {count}")
    picked: list[Any] = []
    for index, major in enumerate(sorted(majors)):
        pool = sorted(by_major[major], key=lambda row: (float(row.de_med), row.preset_id))
        slot = min(len(pool) - 1, int((index + 0.5) / count * len(pool)))
        picked.append(pool[slot])
    return picked


def strength_items(records: Sequence[Any], probes: Sequence[dict[str, Any]],
                   count: int, targets: Sequence[float]) -> list[dict[str, Any]]:
    luts = pick_strength_luts(records, count)
    items: list[dict[str, Any]] = []
    for lut_index, row in enumerate(luts):
        for target_index, target in enumerate(targets):
            ordinal = lut_index * len(targets) + target_index
            items.append({
                "item_id": f"{STRENGTH_PREFIX}_{ordinal + 1:03d}",
                "preset_id": row.preset_id,
                "style_major": row.style_major,
                "de_med": round(float(row.de_med), 6),
                "lut_path": row.path,
                "de_target": float(target),
                "probe_index": ordinal % len(probes),
            })
    return items


# --------------------------------------------------------------------------- rendering


_PROBES: list[np.ndarray] = []
_LOADER = None


def _init_worker(probes: Sequence[np.ndarray], databuild: str) -> None:
    global _PROBES, _LOADER
    from dataset_build.agent_loop.source_reach import configured_lut_loader

    _PROBES = list(probes)
    _LOADER = configured_lut_loader(Path(databuild))


def _apply(lut_path: str, probe: np.ndarray) -> np.ndarray:
    from dataset_build.src.construct.rendering import apply_lut_cpu_oracle

    grid, dmin, dmax = _LOADER.load(Path(lut_path))
    return np.asarray(apply_lut_cpu_oracle(probe, grid, domain_min=dmin, domain_max=dmax))


def _to_image(array: np.ndarray, short_edge: int):
    from PIL import Image

    image = Image.fromarray(np.clip(array * 255.0 + 0.5, 0, 255).astype(np.uint8))
    scale = short_edge / min(image.size)
    if scale < 1.0:
        image = image.resize(
            (max(1, round(image.width * scale)), max(1, round(image.height * scale))),
            Image.LANCZOS,
        )
    return image


def _collage(left, right, gap: int, out_path: str) -> None:
    from PIL import Image

    height = max(left.height, right.height)
    canvas = Image.new("RGB", (left.width + gap + right.width, height), (32, 32, 32))
    canvas.paste(left, (0, (height - left.height) // 2))
    canvas.paste(right, (left.width + gap, (height - right.height) // 2))
    canvas.save(out_path, format="PNG", optimize=True)


def _render_pair(job: tuple[str, str, str, int, str, int, int]) -> tuple[str, bool, str, dict]:
    pair_id, lut_a, lut_b, probe_index, out_path, short_edge, gap = job
    try:
        probe = _PROBES[probe_index]
        left = _to_image(_apply(lut_a, probe), short_edge)
        right = _to_image(_apply(lut_b, probe), short_edge)
        _collage(left, right, gap, out_path)
        return (pair_id, True, "", {})
    except Exception as exc:  # pragma: no cover - reported, never silent
        return (pair_id, False, f"{type(exc).__name__}: {exc}", {})


def _mean_delta_e(before: np.ndarray, after: np.ndarray) -> float:
    from dataset_build.agent_loop.render import delta_e_map

    return float(np.mean(delta_e_map(
        np.asarray(before, dtype=np.float64), np.asarray(after, dtype=np.float64))))


def solve_strength(probe: np.ndarray, rendered: np.ndarray, target: float,
                   steps: int, sample: int) -> tuple[float, float, int]:
    """Bisect ``s`` on a deterministic ``sample``-pixel stride so mean CIEDE2000 ~= target."""
    flat_src = probe.reshape(-1, 3)
    flat_dst = rendered.reshape(-1, 3)
    n = flat_src.shape[0]
    take = min(sample, n)
    index = np.linspace(0, n - 1, take).astype(np.int64)
    src = flat_src[index].reshape(-1, 1, 3)
    dst = flat_dst[index].reshape(-1, 1, 3)
    delta = dst - src

    def measure(scale: float) -> float:
        return _mean_delta_e(src, src + scale * delta)

    full = measure(1.0)
    if full <= target:
        return 1.0, full, take
    low, high = 0.0, 1.0
    scale = 1.0
    value = full
    for _ in range(steps):
        scale = 0.5 * (low + high)
        value = measure(scale)
        if value < target:
            low = scale
        else:
            high = scale
    return float(scale), float(value), take


def _render_strength(job: tuple[str, str, int, float, str, int, int, int, int]
                     ) -> tuple[str, bool, str, dict]:
    item_id, lut_path, probe_index, target, out_path, short_edge, gap, steps, sample = job
    try:
        probe = _PROBES[probe_index]
        rendered_full = _apply(lut_path, probe)
        scale, de_sampled, n_sample = solve_strength(
            probe, rendered_full, target, steps, sample)
        mixed = probe + scale * (rendered_full - probe)
        de_full = _mean_delta_e(probe, mixed)
        _collage(_to_image(probe, short_edge), _to_image(mixed, short_edge), gap, out_path)
        return (item_id, True, "", {
            "s": round(float(scale), 6),
            "de_sampled": round(float(de_sampled), 4),
            "de_measured_full": round(float(de_full), 4),
            "de_full_strength": round(float(_mean_delta_e(probe, rendered_full)), 4),
            "n_sample_pixels": int(n_sample),
        })
    except Exception as exc:  # pragma: no cover - reported, never silent
        return (item_id, False, f"{type(exc).__name__}: {exc}", {})


# --------------------------------------------------------------------------- html

_HTML_SCRIPT = """
const ITEMS = __ITEMS__;
const LABELS = __LABELS__;
const STORE_KEY = __STORE_KEY__;
const SECTIONS = __SECTIONS__;
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

function sectionOf(index) {
  return index < SECTIONS[0].count ? 0 : 1;
}

function sectionStats(sec) {
  const start = sec === 0 ? 0 : SECTIONS[0].count;
  const count = SECTIONS[sec].count;
  let done = 0;
  for (let i = start; i < start + count; i += 1) {
    if (state.ratings[ITEMS[i].id]) { done += 1; }
  }
  return {start: start, count: count, done: done};
}

function render() {
  const item = ITEMS[cursor];
  const sec = sectionOf(cursor);
  const img = document.getElementById('item-img');
  img.src = item.dir + '/' + item.id + '.png';
  img.alt = item.id;
  document.getElementById('sec-title').textContent = SECTIONS[sec].title;
  document.getElementById('sec-hint').textContent = SECTIONS[sec].hint;
  for (let s = 0; s < 2; s += 1) {
    const stats = sectionStats(s);
    const local = sec === s ? (cursor - stats.start + 1) : 0;
    document.getElementById('pos' + s).textContent =
      (s === sec ? (local + ' / ' + stats.count) : ('- / ' + stats.count));
    document.getElementById('done' + s).textContent = '已评 ' + stats.done + ' / ' + stats.count;
    document.getElementById('bar' + s).style.width =
      (stats.count ? (stats.done * 100 / stats.count) : 0) + '%';
    document.getElementById('block' + s).classList.toggle('active', s === sec);
    document.getElementById('rates' + s).style.display = (s === sec) ? 'flex' : 'none';
  }
  const current = state.ratings[item.id] || 0;
  for (const btn of document.querySelectorAll('.rate')) {
    btn.classList.toggle('on', parseInt(btn.dataset.value, 10) === current
      && parseInt(btn.dataset.section, 10) === sec);
  }
  document.getElementById('notes').value = state.notes[item.id] || '';
  document.getElementById('prev').disabled = cursor <= 0;
  document.getElementById('next').disabled = cursor >= total - 1;
  let left = 0;
  for (const it of ITEMS) { if (!state.ratings[it.id]) { left += 1; } }
  document.getElementById('export').textContent =
    left > 0 ? ('导出 CSV（还剩 ' + left + ' 题未评）') : '导出 CSV';
}

function go(delta) {
  const next = cursor + delta;
  if (next < 0 || next >= total) { return; }
  cursor = next;
  save();
  render();
}

function jump(index) {
  if (index < 0 || index >= total) { return; }
  cursor = index;
  save();
  render();
}

function rate(value) {
  if (value < __RMIN__ || value > __RMAX__) { return; }
  state.ratings[ITEMS[cursor].id] = value;
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
  for (const it of ITEMS) {
    const rating = state.ratings[it.id] ? String(state.ratings[it.id]) : '';
    lines.push(it.id + ',' + rating + ',' + csvField(state.notes[it.id] || ''));
  }
  return lines.join('\\n') + '\\n';
}

function exportCsv() {
  const blob = new Blob([buildCsv()], {type: 'text/csv;charset=utf-8'});
  const url = URL.createObjectURL(blob);
  const link = document.createElement('a');
  link.href = url;
  link.download = __CSVNAME__;
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
  document.getElementById('goto0').addEventListener('click', function () { jump(0); });
  document.getElementById('goto1').addEventListener('click',
    function () { jump(SECTIONS[0].count); });
  document.getElementById('notes').addEventListener('input', function (event) {
    state.notes[ITEMS[cursor].id] = event.target.value;
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

_EXT_STYLE = """
.blocks{display:flex;gap:12px;flex-wrap:wrap;margin:0 0 10px}
.block{flex:1 1 320px;border:1px solid #ddd;border-radius:8px;padding:8px 10px;background:#fbfbfb}
.block.active{border-color:#2b6cb0;background:#eef4fb}
.block h2{font-size:14px;margin:0 0 4px}
.block .pos{font-size:20px;font-weight:700;font-variant-numeric:tabular-nums;margin-right:10px}
.block .cnt{font-size:13px;color:#444}
.rate{min-height:64px}
.jump{font-size:13px;padding:6px 10px;border:1px solid #bbb;border-radius:6px;background:#fafafa;
cursor:pointer}
"""


def write_html(out_dir: Path, items: Sequence[dict[str, str]], n_pairs: int, n_strength: int,
               failures: Sequence[tuple[str, str]], seed: int, csv_name: str) -> None:
    sections = [
        {"title": "第 1 节 · LUT 对差异（共 %d 对）" % n_pairs,
         "hint": "左右两张是同一张探针图套两个不同 LUT 的结果，给这一对的差别打分。",
         "count": n_pairs},
        {"title": "第 2 节 · 强度感知（共 %d 题）" % n_strength,
         "hint": "左边是原图，右边是套了同一个 LUT 的结果，给右图相对左图的调整强度打分。",
         "count": n_strength},
    ]
    labels = {
        "0": [{"value": value, "text": text} for value, text in PAIR_RATING_LABELS],
        "1": [{"value": value, "text": text} for value, text in STRENGTH_RATING_LABELS],
    }
    script = (
        _HTML_SCRIPT
        .replace("__ITEMS__", _js_json(list(items)))
        .replace("__LABELS__", _js_json(labels))
        .replace("__SECTIONS__", _js_json(sections))
        .replace("__STORE_KEY__", _js_json(storage_key(seed)))
        .replace("__HEADER__", _js_json(CSV_HEADER))
        .replace("__CSVNAME__", _js_json(csv_name))
        .replace("__RMIN__", str(RATING_MIN))
        .replace("__RMAX__", str(RATING_MAX))
    )

    def buttons(section: int, table: Sequence[tuple[int, str]]) -> str:
        return "".join(
            f'<button type="button" class="rate" data-section="{section}" data-value="{value}">'
            f"<b>{value}</b>{html.escape(label)}</button>"
            for value, label in table
        )

    fail_block = ""
    if failures:
        entries = "".join(
            f"<li>{html.escape(item_id)}: {html.escape(detail)}</li>"
            for item_id, detail in failures
        )
        fail_block = f'<div class="fail"><b>渲染失败</b><ul>{entries}</ul></div>'
    parts = [
        '<!doctype html><html lang="zh"><head><meta charset="utf-8">',
        '<meta name="viewport" content="width=device-width,initial-scale=1">',
        "<title>LUT 扩展盲标问卷 2026-08-19</title>",
        f"<style>{_HTML_STYLE}{_EXT_STYLE}</style></head><body>",
        "<h1>LUT 扩展盲标问卷</h1>",
        '<p class="hint">共两节：先 %d 对 LUT 差异，再 %d 题强度感知。点大按钮评分，'
        "评分自动保存在本机浏览器；标完点「导出 CSV」下载填好的 %s。</p>"
        % (n_pairs, n_strength, html.escape(csv_name)),
        '<div class="blocks">',
        f'<div class="block" id="block0"><h2>{html.escape(sections[0]["title"])}</h2>'
        '<span class="pos" id="pos0">- / -</span><span class="cnt" id="done0"></span>'
        '<div class="track"><div id="bar0"></div></div></div>',
        f'<div class="block" id="block1"><h2>{html.escape(sections[1]["title"])}</h2>'
        '<span class="pos" id="pos1">- / -</span><span class="cnt" id="done1"></span>'
        '<div class="track"><div id="bar1"></div></div></div>',
        "</div>",
        '<p class="hint"><b id="sec-title"></b> <span id="sec-hint"></span></p>',
        '<img id="item-img" alt="item">',
        f'<div class="rates" id="rates0">{buttons(0, PAIR_RATING_LABELS)}</div>',
        f'<div class="rates" id="rates1">{buttons(1, STRENGTH_RATING_LABELS)}</div>',
        '<div class="nav"><button type="button" id="prev">← 上一题</button>'
        '<button type="button" id="next">下一题 →</button>'
        '<button type="button" id="export">导出 CSV</button>'
        '<button type="button" class="jump" id="goto0">跳到第 1 节</button>'
        '<button type="button" class="jump" id="goto1">跳到第 2 节</button>'
        '<span class="keys">键盘：1–5 评分并跳下一题，← / → 翻页</span></div>',
        '<textarea id="notes" rows="2" placeholder="备注（可留空）"></textarea>',
        fail_block,
        f"<script>{script}</script>",
        "</body></html>",
    ]
    (out_dir / "questionnaire_ext.html").write_text(
        "\n".join(part for part in parts if part), encoding="utf-8", newline="\n"
    )


# --------------------------------------------------------------------------- outputs


def write_csv(out_dir: Path, items: Sequence[dict[str, str]], name: str) -> None:
    lines = [CSV_HEADER]
    lines.extend(f"{item['id']},," for item in items)
    (out_dir / name).write_text("\n".join(lines) + "\n", encoding="utf-8", newline="\n")


def write_key(out_dir: Path, pairs: Sequence[dict[str, Any]], meta: dict[str, Any],
              strength: Sequence[dict[str, Any]], probes: Sequence[dict[str, Any]],
              edges: Sequence[float], seed: int, excluded: int, name: str) -> None:
    payload = {
        "schema": EXT_SCHEMA,
        "feature_spec": FEATURE_SPEC,
        "seed": seed,
        "bin_edges": [round(edge, 6) for edge in edges],
        "excluded_combinations": excluded,
        "shortfall_per_bin": meta["shortfall_per_bin"],
        "majors_used": meta["majors_used"],
        "display_order": list(meta["display_order"]),
        "probes": [
            {"index": index, "scene": probe["scene"], "source": probe["source"]}
            for index, probe in enumerate(probes)
        ],
        "pairs": {
            entry["pair_id"]: {
                "preset_a": entry["preset_a"],
                "preset_b": entry["preset_b"],
                "style_major": entry["style_major"],
                "distance": entry["distance"],
                "bin": entry["bin"],
                "bin_low": entry["bin_low"],
                "bin_high": entry["bin_high"],
                "probe_index": entry["probe_index"],
                "probe_scene": probes[entry["probe_index"]]["scene"],
            }
            for entry in pairs
        },
        "strength_items": {
            entry["item_id"]: {
                key: entry[key] for key in sorted(entry) if key != "item_id"
            }
            for entry in strength
        },
        "strength_rating_scale": {str(v): t for v, t in STRENGTH_RATING_LABELS},
        "pair_rating_scale": {str(v): t for v, t in PAIR_RATING_LABELS},
    }
    (out_dir / name).write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8", newline="\n",
    )


# --------------------------------------------------------------------------- command


def cmd_build(args: argparse.Namespace) -> int:
    catalog, databuild, _ = load_catalog(args.config)
    records = list(catalog.records)
    matrix, _, _ = build_features(records)
    edges = bin_edges(args.bins, args.bin_max)
    exclude = existing_combinations(args.exclude_pair_key)
    quota = bin_quota(args.pairs, args.bins)
    pairs, meta = sample_pairs(
        records, matrix, edges, args.seed, 0, args.candidate_cap,
        exclude=exclude, per_bin_counts=quota, pair_prefix=PAIR_PREFIX,
    )

    manifests = list(args.probe_jsonl or DEFAULT_PROBE_MANIFESTS)
    probes = load_probes(manifests, args.probe_short_edge, args.probes,
                         args.probe_fallback_dir)
    for index, entry in enumerate(sorted(pairs, key=lambda item: item["pair_id"])):
        entry["probe_index"] = index % len(probes)

    strength = strength_items(records, probes, args.strength_luts, DE_TARGETS)

    out_dir = args.out_dir
    (out_dir / "pairs").mkdir(parents=True, exist_ok=True)
    (out_dir / "strength").mkdir(parents=True, exist_ok=True)

    failures: list[tuple[str, str]] = []
    elapsed = 0.0
    if not args.skip_render:
        pair_jobs = [
            (entry["pair_id"], catalog.by_id[entry["preset_a"]].path,
             catalog.by_id[entry["preset_b"]].path, entry["probe_index"],
             str(out_dir / "pairs" / f"{entry['pair_id']}.png"), args.side_edge, args.gap)
            for entry in sorted(pairs, key=lambda item: item["pair_id"])
        ]
        strength_jobs = [
            (entry["item_id"], entry["lut_path"], entry["probe_index"], entry["de_target"],
             str(out_dir / "strength" / f"{entry['item_id']}.png"), args.side_edge, args.gap,
             args.bisect_steps, args.bisect_sample)
            for entry in sorted(strength, key=lambda item: item["item_id"])
        ]
        started = time.time()
        arrays = [probe["array"] for probe in probes]
        with futures.ProcessPoolExecutor(
            max_workers=args.workers, initializer=_init_worker,
            initargs=(arrays, str(databuild)),
        ) as pool:
            for pair_id, ok, detail, _ in pool.map(_render_pair, pair_jobs, chunksize=2):
                if not ok:
                    failures.append((pair_id, detail))
            measured: dict[str, dict[str, Any]] = {}
            for item_id, ok, detail, extra in pool.map(_render_strength, strength_jobs,
                                                       chunksize=1):
                if ok:
                    measured[item_id] = extra
                else:
                    failures.append((item_id, detail))
        elapsed = time.time() - started
        for entry in strength:
            entry.update(measured.get(entry["item_id"], {}))

    pair_ids = [entry for entry in meta["display_order"]]
    items = [{"id": pair_id, "dir": "pairs"} for pair_id in pair_ids]
    items.extend({"id": entry["item_id"], "dir": "strength"}
                 for entry in sorted(strength, key=lambda item: item["item_id"]))

    write_key(out_dir, pairs, meta, strength, probes, edges, args.seed, len(exclude),
              args.key_name)
    write_csv(out_dir, items, args.csv_name)
    write_html(out_dir, items, len(pairs), len(strength), failures, args.seed, args.csv_name)

    per_bin: dict[str, dict[str, int]] = {}
    probe_counts: dict[str, int] = {}
    for entry in pairs:
        block = per_bin.setdefault(str(entry["bin"]), {})
        block[str(entry["probe_index"])] = block.get(str(entry["probe_index"]), 0) + 1
        probe_counts[str(entry["probe_index"])] = probe_counts.get(
            str(entry["probe_index"]), 0) + 1
    print(json.dumps({
        "out_dir": str(out_dir),
        "pairs": len(pairs),
        "strength_items": len(strength),
        "bin_edges": [round(edge, 6) for edge in edges],
        "bin_quota": quota,
        "per_bin_probe_counts": per_bin,
        "probe_counts": probe_counts,
        "probes": [{"index": i, "scene": p["scene"], "source": p["source"]}
                   for i, p in enumerate(probes)],
        "majors_used": meta["majors_used"],
        "shortfall_per_bin": meta["shortfall_per_bin"],
        "excluded_combinations": len(exclude),
        "rendered": 0 if args.skip_render else len(pairs) + len(strength),
        "render_failures": failures,
        "render_seconds": round(elapsed, 2),
        "strength": [
            {key: entry.get(key) for key in
             ("item_id", "preset_id", "style_major", "de_med", "de_target",
              "de_measured_full", "de_sampled", "s", "probe_index")}
            for entry in sorted(strength, key=lambda item: item["item_id"])
        ],
    }, ensure_ascii=False, indent=2))
    return 0


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    default_out = REPO_ROOT / "docs/assets/lut_cluster_pilot_20260819/questionnaire_ext"
    sub = parser.add_subparsers(dest="command", required=True)

    build = sub.add_parser("build", help="sample 140 new pairs + 40 strength items")
    build.add_argument("--config", type=Path,
                       default=REPO_ROOT / "configs/agent_loop.terra-smoke.toml")
    build.add_argument("--out-dir", type=Path, default=default_out)
    build.add_argument("--exclude-pair-key", type=Path,
                       default=REPO_ROOT
                       / "docs/assets/lut_cluster_pilot_20260819/questionnaire/pair_key.json")
    build.add_argument("--probe-jsonl", type=Path, action="append", default=None,
                       help="repeatable; defaults to smoke5 then the scene-sample manifest")
    build.add_argument("--probe-fallback-dir", type=Path, default=DEFAULT_PROBE_FALLBACK_DIR)
    build.add_argument("--seed", type=int, default=DEFAULT_EXT_SEED)
    build.add_argument("--bins", type=int, default=DEFAULT_BINS)
    build.add_argument("--bin-max", type=float, default=DEFAULT_BIN_MAX)
    build.add_argument("--pairs", type=int, default=DEFAULT_EXT_PAIRS)
    build.add_argument("--probes", type=int, default=DEFAULT_PROBES)
    build.add_argument("--strength-luts", type=int, default=STRENGTH_LUTS)
    build.add_argument("--bisect-steps", type=int, default=BISECT_STEPS)
    build.add_argument("--bisect-sample", type=int, default=BISECT_SAMPLE)
    build.add_argument("--candidate-cap", type=int, default=CANDIDATE_CAP)
    build.add_argument("--workers", type=int, default=8)
    build.add_argument("--probe-short-edge", type=int, default=512)
    build.add_argument("--side-edge", type=int, default=448)
    build.add_argument("--gap", type=int, default=8)
    build.add_argument("--csv-name", type=str, default="questionnaire_ext.csv")
    build.add_argument("--key-name", type=str, default="pair_key_ext.json")
    build.add_argument("--skip-render", action="store_true",
                       help="emit csv/key/html only (determinism re-run)")
    build.set_defaults(func=cmd_build)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    return int(args.func(args))


if __name__ == "__main__":
    raise SystemExit(main())
