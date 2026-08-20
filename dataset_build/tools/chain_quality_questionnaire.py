"""Whole-chain quality questionnaire (A9): 300 committed chains, blind 5-point rating.

``build`` reads one agent-loop campaign from PostgreSQL, keeps only chains whose leaf
carries ``commit_status=committed``, draws a deterministic stratified sample of 300
(global bin x local bin nine-cell grid, per-cell quota = round(300 * share), at most two
chains per source, within-cell order = sha1(branch_id)), and writes a static one-screen
page showing ``source | global_after | final_after`` with five large rating buttons.
The page carries no bin / strength / delta-E / preset text: every parameter lives in the
separate ``item_key.json`` used by ``analyze``.

Artifact resolution and thumbnail rendering are imported unchanged from
``dataset_build.tools.export_agent_loop_review`` (A8).

Usage:
    python -m dataset_build.tools.chain_quality_questionnaire build \
        --campaign local-v1-iter1 \
        --out-dir docs/assets/lut_cluster_pilot_20260819/chain_quality_300
    python -m dataset_build.tools.chain_quality_questionnaire analyze \
        --csv docs/assets/lut_cluster_pilot_20260819/chain_quality_300/chainq.csv
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import html
import json
import math
import os
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Mapping, Sequence

from dataset_build.tools.export_agent_loop_review import (
    DEFAULT_CATALOG_DB,
    DEFAULT_ROOT_PARENTS,
    GLOBAL_BIN_ORDER,
    LOCAL_BIN_ORDER,
    ArtifactResolver,
    Thumbnails,
    _connect,
    _default_roots,
    _loads,
    _rows,
    print_winner_confidence_warning,
    winner_confidence_counts,
)

SCHEMA = "chain-quality-questionnaire-v1"
DEFAULT_CAMPAIGN = "local-v1-iter1"
DEFAULT_OUT_DIR = Path("docs/assets/lut_cluster_pilot_20260819/chain_quality_300")
TARGET_N = 300
PER_SOURCE_CAP = 2
THUMB_SHORT_EDGE = 512
CSV_HEADER = "item_id,rating,notes"
RATING_MIN, RATING_MAX = 1, 5
DETERIORATED_MAX = 2  # rating <= 2 counts as deteriorated
IMPROVED_MIN = 4  # rating >= 4 counts as improved
DE_BINS = 6

# C1b items 3-5. The committed leaf carries `winner_confidence` ("normal" when the
# validator passed the chain, "low" otherwise). Campaign discipline keeps
# `winner_confidence=low` out of main training and out of evaluation GT, so a sample
# drawn without carrying the field cannot be audited against that rule afterwards.
# The sampler does NOT filter on it: the current campaign is 100% `low`, so filtering
# would return an empty set. It is carried into `item_key.json` and counted, and a
# warning line is printed above the stats block whenever any `low` row was sampled.
WINNER_CONFIDENCE_FILTERED = False

RATING_LABELS = (
    (1, "整个链条劣化了原图"),
    (2, "global 改善但 local 劣化"),
    (3, "基本看不到变化"),
    (4, "global 有提升，但 local 没提升"),
    (5, "global 与 local 都有明显提升"),
)


# --------------------------------------------------------------------------- helpers


def _sha1(value: str) -> str:
    return hashlib.sha1(str(value).encode("utf-8")).hexdigest()


def _float(value: Any) -> float | None:
    if value is None or isinstance(value, bool):
        return None
    try:
        result = float(value)
    except (TypeError, ValueError):
        return None
    return result if math.isfinite(result) else None


def _text(value: Any) -> str | None:
    if value is None:
        return None
    result = str(value).strip()
    return result or None


def _bin_order(values: Sequence[str], base: Sequence[str]) -> list[str]:
    seen = set(values)
    ordered = [name for name in base if name in seen]
    ordered.extend(sorted(name for name in seen if name not in set(base)))
    return ordered


def _js_json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True).replace("</", "<\\/")


# ------------------------------------------------------------------------ chain build


def _collect_chains(conn, campaign: str) -> dict[str, Any]:
    sources = _rows(
        conn,
        "SELECT source_id, source_sha256, status, manifest_json FROM agent_source_run "
        "WHERE campaign_id=%s ORDER BY source_id",
        (campaign,),
    )
    branches = _rows(
        conn,
        "SELECT branch_id, source_sha256, parent_id, level, status, proposal_json, "
        "result_json FROM agent_branch WHERE campaign_id=%s ORDER BY branch_id",
        (campaign,),
    )
    audits = _rows(
        conn,
        "SELECT branch_id, direction_cosine FROM proposal_audit WHERE campaign_id=%s",
        (campaign,),
    )
    source_by_sha = {str(row["source_sha256"]): row for row in sources}
    cosine = {str(row["branch_id"]): _float(row["direction_cosine"]) for row in audits}
    globals_by_id: dict[str, dict[str, Any]] = {}
    leaves: list[dict[str, Any]] = []
    for row in branches:
        row["proposal"] = _loads(row.pop("proposal_json")) or {}
        row["result"] = _loads(row.pop("result_json")) or {}
        if row["level"] == "global":
            globals_by_id[str(row["branch_id"])] = row
        else:
            leaves.append(row)

    committed = [
        row for row in leaves
        if str((row["result"] or {}).get("commit_status") or "") == "committed"
    ]
    dropped: Counter[str] = Counter()
    chains: list[dict[str, Any]] = []
    for leaf in committed:
        source = source_by_sha.get(str(leaf["source_sha256"]))
        if source is None:
            dropped["source_row_missing"] += 1
            continue
        if _loads(source["manifest_json"]) is None:
            dropped[f"source_manifest_null({source['status']})"] += 1
            continue
        parent = globals_by_id.get(str(leaf["parent_id"] or ""))
        if parent is None:
            dropped["global_parent_missing"] += 1
            continue
        result = leaf["result"] or {}
        l_render = result.get("render") or {}
        l_parameters = l_render.get("parameters") or {}
        l_metrics = l_render.get("metrics") or {}
        l_proposal = result.get("proposal") or leaf["proposal"] or {}
        g_result = parent["result"] or {}
        g_render = g_result.get("global_render") or {}
        g_parameters = g_render.get("parameters") or {}
        g_metrics = g_render.get("metrics") or {}
        g_proposal = g_result.get("proposal") or parent["proposal"] or {}
        g_bin = _text(g_parameters.get("strength_bin") or g_proposal.get("strength_bin"))
        l_bin = _text(l_parameters.get("strength_bin") or l_proposal.get("strength_bin"))
        if g_bin is None or l_bin is None:
            dropped["strength_bin_null"] += 1
            continue
        branch_id = str(leaf["branch_id"])
        chains.append({
            "branch_id": branch_id,
            "global_branch_id": str(parent["branch_id"]),
            "source_id": str(source["source_id"]),
            "source_sha256": str(source["source_sha256"]),
            "source_status": str(source["status"]),
            # C1b item 3: carried, counted, never used to filter.
            "winner_confidence": _text(result.get("winner_confidence")),
            "cell": (g_bin, l_bin),
            "sort_key": _sha1(branch_id),
            "global_artifact": g_render.get("artifact"),
            "local_artifact": l_render.get("artifact"),
            "global": {
                "preset": _text(g_parameters.get("preset_id")
                                or g_proposal.get("preset_id")),
                "bin": g_bin,
                "strength": _float(g_parameters.get("global_strength")),
                "de_measured": _float(g_metrics.get("delta_e")),
            },
            "local": {
                "preset": _text(l_parameters.get("preset_id")
                                or l_proposal.get("preset_id")),
                "bin": l_bin,
                "strength": _float(l_parameters.get("local_strength")),
                "de_masked": _float(l_metrics.get("delta_e")),
                "mask_family": _text(result.get("mask_family")),
                "direction_cosine": cosine.get(branch_id),
            },
        })
    chains.sort(key=lambda row: (row["sort_key"], row["branch_id"]))
    return {
        "sources": sources,
        "source_by_sha": source_by_sha,
        "chains": chains,
        "snapshot": {
            "campaign": campaign,
            "sources_total": len(sources),
            "sources_by_status": dict(sorted(
                Counter(str(row["status"]) for row in sources).items())),
            "global_branches": len(globals_by_id),
            "local_branches": len(leaves),
            "committed_leaves": len(committed),
            "committed_dropped": dict(sorted(dropped.items())),
            "chain_population": len(chains),
            "chain_population_sources": len({row["source_id"] for row in chains}),
            "chain_population_winner_confidence": winner_confidence_counts(
                row["winner_confidence"] for row in chains
            ),
        },
    }


def _resolvable(chains: Sequence[Mapping[str, Any]], resolver: ArtifactResolver,
                source_refs: Mapping[str, Any]) -> tuple[list[dict[str, Any]], Counter]:
    """Keep chains whose three blobs all resolve; count the rest by reason."""
    dropped: Counter[str] = Counter()
    cache: dict[str, bool] = {}

    def ok(ref: Any) -> bool:
        if not isinstance(ref, Mapping) or not ref.get("sha256"):
            return False
        digest = str(ref["sha256"])
        if digest not in cache:
            try:
                resolver.path_for(ref)
                cache[digest] = True
            except (FileNotFoundError, ValueError):
                cache[digest] = False
        return cache[digest]

    kept: list[dict[str, Any]] = []
    for chain in chains:
        source_ref = source_refs.get(chain["source_id"])
        if not ok(source_ref):
            dropped["source_blob_missing"] += 1
            continue
        if not ok(chain["global_artifact"]):
            dropped["global_after_blob_missing"] += 1
            continue
        if not ok(chain["local_artifact"]):
            dropped["final_after_blob_missing"] += 1
            continue
        kept.append(dict(chain, source_artifact=source_ref))
    return kept, dropped


# ----------------------------------------------------------------------- stratify


def _quotas(cell_counts: Mapping[tuple[str, str], int], cells: Sequence[tuple[str, str]],
            target: int) -> dict[tuple[str, str], int]:
    total = sum(cell_counts[cell] for cell in cells)
    quota = {
        cell: min(cell_counts[cell],
                  int(math.floor(target * cell_counts[cell] / total + 0.5)))
        for cell in cells
    }
    # largest cell first (ties: canonical cell order) absorbs the rounding remainder.
    priority = sorted(cells, key=lambda cell: (-cell_counts[cell], cells.index(cell)))
    delta = target - sum(quota.values())
    while delta != 0:
        moved = False
        for cell in priority:
            if delta > 0 and quota[cell] < cell_counts[cell]:
                quota[cell] += 1
                delta -= 1
                moved = True
            elif delta < 0 and quota[cell] > 0:
                quota[cell] -= 1
                delta += 1
                moved = True
            if delta == 0:
                break
        if not moved:
            break
    return quota


def _select(chains: Sequence[Mapping[str, Any]], target: int,
            cap: int) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    by_cell: dict[tuple[str, str], list[Mapping[str, Any]]] = defaultdict(list)
    for chain in chains:
        by_cell[chain["cell"]].append(chain)
    globals_seen = _bin_order([cell[0] for cell in by_cell], GLOBAL_BIN_ORDER)
    locals_seen = _bin_order([cell[1] for cell in by_cell], LOCAL_BIN_ORDER)
    cells = [
        (gbin, lbin) for gbin in globals_seen for lbin in locals_seen
        if (gbin, lbin) in by_cell
    ]
    counts = {cell: len(by_cell[cell]) for cell in cells}
    quota = _quotas(counts, cells, target)

    used: Counter[str] = Counter()
    picked: dict[tuple[str, str], list[Mapping[str, Any]]] = {cell: [] for cell in cells}
    taken: set[str] = set()

    def fill(cell: tuple[str, str], want: int) -> None:
        for chain in by_cell[cell]:  # already sha1-sorted
            if len(picked[cell]) >= want:
                return
            if chain["branch_id"] in taken or used[chain["source_id"]] >= cap:
                continue
            picked[cell].append(chain)
            taken.add(chain["branch_id"])
            used[chain["source_id"]] += 1

    for cell in cells:
        fill(cell, quota[cell])
    quota_shortfall = {
        f"{cell[0]}|{cell[1]}": quota[cell] - len(picked[cell])
        for cell in cells if len(picked[cell]) < quota[cell]
    }
    # per-source cap shortfall is redistributed to the largest cells that still have
    # eligible candidates, keeping the total at exactly `target`.
    priority = sorted(cells, key=lambda cell: (-counts[cell], cells.index(cell)))
    backfilled: Counter[str] = Counter()
    while sum(len(rows) for rows in picked.values()) < target:
        before = sum(len(rows) for rows in picked.values())
        for cell in priority:
            if sum(len(rows) for rows in picked.values()) >= target:
                break
            want = len(picked[cell]) + 1
            fill(cell, want)
            if len(picked[cell]) == want:
                backfilled[f"{cell[0]}|{cell[1]}"] += 1
        if sum(len(rows) for rows in picked.values()) == before:
            break

    selected = [chain for cell in cells for chain in picked[cell]]
    selected.sort(key=lambda row: (row["sort_key"], row["branch_id"]))
    grid = {
        "global_bins": globals_seen,
        "local_bins": locals_seen,
        "cells": [
            {
                "global_bin": cell[0],
                "local_bin": cell[1],
                "population": counts[cell],
                "share": round(counts[cell] / sum(counts.values()), 6),
                "quota": quota[cell],
                "selected": len(picked[cell]),
            }
            for cell in cells
        ],
        "quota_shortfall_before_backfill": quota_shortfall,
        "cap_backfill": dict(sorted(backfilled.items())),
        "sources_used": len({chain["source_id"] for chain in selected}),
        "max_chains_per_source": max(
            Counter(chain["source_id"] for chain in selected).values(), default=0),
    }
    return selected, grid


# --------------------------------------------------------------------------- page


_STYLE = """
:root{color-scheme:light}
*{box-sizing:border-box}
body{font-family:system-ui,'Noto Sans CJK SC',sans-serif;margin:0;padding:14px 18px;
background:#fff;color:#111}
h1{font-size:18px;margin:0 0 4px}
.hint{font-size:13px;color:#444;margin:0 0 8px}
.topbar{display:flex;align-items:center;gap:14px;flex-wrap:wrap;margin-bottom:6px}
#progress{font-size:20px;font-weight:700;font-variant-numeric:tabular-nums}
#done,#unrated{font-size:13px;color:#444}
.track{height:6px;background:#e6e6e6;border-radius:3px;overflow:hidden;margin:0 0 10px}
#bar{height:100%;background:#2b6cb0;width:0}
.trip{display:grid;grid-template-columns:repeat(3,1fr);gap:10px;align-items:start}
.trip figure{margin:0}
.trip img{display:block;width:100%;height:auto;border:1px solid #ccc;background:#202020}
.trip figcaption{font-size:13px;color:#333;margin-top:4px;text-align:center}
.rates{display:flex;gap:8px;flex-wrap:wrap;margin:12px 0 8px}
.rate{flex:1 1 170px;padding:14px 10px;font-size:15px;line-height:1.35;cursor:pointer;
border:1px solid #bbb;border-radius:8px;background:#fafafa;color:#111;text-align:left}
.rate:hover{background:#f0f4f8}
.rate.on{background:#2b6cb0;border-color:#2b6cb0;color:#fff}
.rate b{font-size:18px;margin-right:8px}
.nav{display:flex;gap:10px;align-items:center;flex-wrap:wrap;margin:10px 0}
button{font-family:inherit}
.nav button,#export{padding:10px 16px;font-size:14px;cursor:pointer;border:1px solid #bbb;
border-radius:6px;background:#fafafa}
.nav button:disabled{opacity:.4;cursor:default}
#notes{width:100%;padding:6px;font-size:13px;font-family:inherit;border:1px solid #ccc;
border-radius:6px}
.keys{font-size:12px;color:#666}
@media (max-width:1100px){.trip{grid-template-columns:repeat(2,1fr)}
.trip figure:first-child{grid-column:1/-1}}
"""

_SCRIPT = """
const ITEMS = __ITEMS__;
const IMAGES = __IMAGES__;
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
  for (const iid of ITEMS) { if (state.ratings[iid]) { n += 1; } }
  return n;
}

function render() {
  const iid = ITEMS[cursor];
  const trio = IMAGES[iid];
  const ids = ['img-source', 'img-global', 'img-final'];
  for (let i = 0; i < ids.length; i += 1) {
    document.getElementById(ids[i]).src = trio[i];
  }
  document.getElementById('progress').textContent = (cursor + 1) + ' / ' + total;
  const done = ratedCount();
  const left = total - done;
  document.getElementById('done').textContent = '已评 ' + done + ' / ' + total;
  document.getElementById('bar').style.width = (total ? (done * 100 / total) : 0) + '%';
  const current = state.ratings[iid] || 0;
  for (const btn of document.querySelectorAll('.rate')) {
    btn.classList.toggle('on', parseInt(btn.dataset.value, 10) === current);
  }
  document.getElementById('notes').value = state.notes[iid] || '';
  document.getElementById('prev').disabled = cursor <= 0;
  document.getElementById('next').disabled = cursor >= total - 1;
  document.getElementById('export').textContent =
    left > 0 ? ('导出 CSV（还剩 ' + left + ' 条未评）') : '导出 CSV';
  document.getElementById('unrated').textContent =
    left > 0 ? ('未评 ' + left + ' 条，仍可导出') : '全部已评';
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
  state.ratings[ITEMS[cursor]] = value;
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
  for (const iid of ITEMS) {
    const rating = state.ratings[iid] ? String(state.ratings[iid]) : '';
    lines.push(iid + ',' + rating + ',' + csvField(state.notes[iid] || ''));
  }
  return lines.join('\\n') + '\\n';
}

function exportCsv() {
  const blob = new Blob([buildCsv()], {type: 'text/csv;charset=utf-8'});
  const url = URL.createObjectURL(blob);
  const link = document.createElement('a');
  link.href = url;
  link.download = 'chainq.csv';
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
    state.notes[ITEMS[cursor]] = event.target.value;
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

FIGURE_CAPTIONS = ("原图", "第一步处理后", "最终结果")


def write_page(out_dir: Path, order: Sequence[str], images: Mapping[str, list[str]],
               store_key: str) -> Path:
    script = (
        _SCRIPT
        .replace("__ITEMS__", _js_json(list(order)))
        .replace("__IMAGES__", _js_json(dict(images)))
        .replace("__STORE_KEY__", _js_json(store_key))
        .replace("__HEADER__", _js_json(CSV_HEADER))
        .replace("__RMIN__", str(RATING_MIN))
        .replace("__RMAX__", str(RATING_MAX))
    )
    buttons = "".join(
        f'<button type="button" class="rate" data-value="{value}">'
        f"<b>{value}</b>{html.escape(label)}</button>"
        for value, label in RATING_LABELS
    )
    figures = "".join(
        f'<figure><img id="{ident}" alt="{html.escape(caption)}">'
        f"<figcaption>{html.escape(caption)}</figcaption></figure>"
        for ident, caption in zip(
            ("img-source", "img-global", "img-final"), FIGURE_CAPTIONS)
    )
    parts = [
        '<!doctype html><html lang="zh"><head><meta charset="utf-8">',
        '<meta name="viewport" content="width=device-width,initial-scale=1">',
        "<title>整链质量盲标问卷</title>",
        f"<style>{_STYLE}</style></head><body>",
        "<h1>整链质量盲标问卷</h1>",
        '<p class="hint">左边是原图，中间是第一步处理后，右边是最终结果。'
        "只对「最终结果相对原图」这条链整体打分；评分自动保存在本机浏览器，"
        "标完点「导出 CSV」下载 chainq.csv。</p>",
        '<div class="topbar"><span id="progress">- / -</span>'
        '<span id="done"></span><span id="unrated"></span>'
        '<span class="keys">也可用键盘：1–5 评分并跳下一条，← / → 翻页</span></div>',
        '<div class="track"><div id="bar"></div></div>',
        f'<div class="trip">{figures}</div>',
        f'<div class="rates">{buttons}</div>',
        '<div class="nav"><button type="button" id="prev">← 上一条</button>'
        '<button type="button" id="next">下一条 →</button>'
        '<button type="button" id="export">导出 CSV</button></div>',
        '<textarea id="notes" rows="2" placeholder="备注（可留空）"></textarea>',
        f"<script>{script}</script>",
        "</body></html>",
    ]
    path = out_dir / "chainq.html"
    path.write_text("\n".join(parts) + "\n", encoding="utf-8", newline="\n")
    return path


def write_csv(out_dir: Path, order: Sequence[str]) -> Path:
    lines = [CSV_HEADER]
    lines.extend(f"{item_id},," for item_id in order)
    path = out_dir / "chainq.csv"
    path.write_text("\n".join(lines) + "\n", encoding="utf-8", newline="\n")
    return path


def write_item_key(out_dir: Path, payload: Mapping[str, Any]) -> Path:
    path = out_dir / "item_key.json"
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8", newline="\n",
    )
    return path


# ------------------------------------------------------------------------- commands


def cmd_build(args: argparse.Namespace) -> int:
    dsn = args.dsn or os.environ.get("VERARETOUCH_AGENT_POSTGRES_DSN")
    if not dsn:
        raise SystemExit(
            "no PostgreSQL DSN: pass --dsn or export VERARETOUCH_AGENT_POSTGRES_DSN")
    with _connect(dsn) as conn:
        data = _collect_chains(conn, args.campaign)

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    resolver = ArtifactResolver(_default_roots(args.artifact_root), args.catalog_db)

    # source artifact + scene come from the per-source tree manifest (A8 layout).
    wanted = {chain["source_id"] for chain in data["chains"]}
    source_refs: dict[str, Any] = {}
    scenes: dict[str, str | None] = {}
    unreadable: list[str] = []
    for source in data["sources"]:
        source_id = str(source["source_id"])
        if source_id not in wanted:
            continue
        manifest_ref = _loads(source["manifest_json"])
        if not isinstance(manifest_ref, Mapping):
            continue
        try:
            tree = resolver.read_json(manifest_ref)
        except (FileNotFoundError, ValueError, json.JSONDecodeError) as exc:
            unreadable.append(f"{source_id}:tree:{type(exc).__name__}")
            continue
        source_refs[source_id] = tree.get("source_artifact")
        scene = None
        ref = tree.get("source_annotation_artifact")
        if ref:
            try:
                scene = _text((resolver.read_json(ref) or {}).get("scene"))
            except (FileNotFoundError, ValueError, json.JSONDecodeError) as exc:
                unreadable.append(f"{source_id}:annotation:{type(exc).__name__}")
        scenes[source_id] = scene

    eligible, blob_dropped = _resolvable(data["chains"], resolver, source_refs)
    selected, grid = _select(eligible, args.target, args.per_source_cap)

    thumbs = Thumbnails(resolver, out_dir, args.thumb_short_edge)
    legacy = thumbs.dir
    thumbs.dir = out_dir / "imgs"
    thumbs.dir.mkdir(parents=True, exist_ok=True)
    if legacy != thumbs.dir and legacy.is_dir() and not any(legacy.iterdir()):
        legacy.rmdir()

    def rel(ref: Any) -> str | None:
        value = thumbs.rel(ref)
        return None if value is None else "imgs/" + value.split("/", 1)[1]

    order: list[str] = []
    images: dict[str, list[str]] = {}
    items: dict[str, Any] = {}
    failures: list[str] = []
    for index, chain in enumerate(selected, start=1):
        item_id = f"c{index:03d}"
        trio = [rel(chain["source_artifact"]), rel(chain["global_artifact"]),
                rel(chain["local_artifact"])]
        if any(value is None for value in trio):
            failures.append(f"{item_id}:{chain['branch_id']}")
            continue
        order.append(item_id)
        images[item_id] = trio
        items[item_id] = {
            "branch_id": chain["branch_id"],
            "global_branch_id": chain["global_branch_id"],
            "source_id": chain["source_id"],
            "scene": scenes.get(chain["source_id"]),
            "winner_confidence": chain["winner_confidence"],
            "global": chain["global"],
            "local": chain["local"],
        }

    sampled_confidence = winner_confidence_counts(
        entry["winner_confidence"] for entry in items.values()
    )
    key_payload = {
        "schema": SCHEMA,
        "campaign": args.campaign,
        "target_n": args.target,
        "per_source_cap": args.per_source_cap,
        "thumb_short_edge": args.thumb_short_edge,
        "snapshot": data["snapshot"],
        "population_blob_dropped": dict(sorted(blob_dropped.items())),
        "population_eligible": len(eligible),
        "unreadable_json": sorted(unreadable),
        "grid": grid,
        "winner_confidence_filtered": WINNER_CONFIDENCE_FILTERED,
        "winner_confidence_counts": sampled_confidence,
        "display_order": order,
        "items": items,
    }
    # C1b item 10: `item_key.json` is the only irreplaceable product of a build (it is
    # what `analyze` reads and what pins every sampled parameter), and the CSV is the
    # rating surface. Both land before the page render and before the optional prune,
    # so an HTML or thumbnail failure can no longer destroy a finished sample.
    key_path = write_item_key(out_dir, key_payload)
    csv_path = write_csv(out_dir, order)
    page = write_page(out_dir, order, images, f"{SCHEMA}:{args.campaign}")
    try:
        pruned = thumbs.prune()
        prune_error = None
    except OSError as exc:
        pruned, prune_error = [], f"{type(exc).__name__}: {exc}"
    warning = print_winner_confidence_warning(
        sampled_confidence, filtered=WINNER_CONFIDENCE_FILTERED
    )
    stats = {
        "winner_confidence_warning": warning,
        "winner_confidence_counts": sampled_confidence,
        "winner_confidence_filtered": WINNER_CONFIDENCE_FILTERED,
        "page": str(page),
        "csv": str(csv_path),
        "item_key": str(key_path),
        "items": len(order),
        "render_failures": failures,
        "thumbnails": len(thumbs.written),
        "thumbnails_pruned": pruned,
        "thumbnails_prune_error": prune_error,
        "missing_blobs": len(thumbs.missing),
        "thumbnail_failures": len(thumbs.failed),
        "snapshot": data["snapshot"],
        "grid": grid,
    }
    print(json.dumps(stats, ensure_ascii=False, sort_keys=True, indent=2))
    return 0


def read_ratings(path: Path) -> list[tuple[str, str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return [
            (str(row.get("item_id") or "").strip(),
             str(row.get("rating") or "").strip(),
             str(row.get("notes") or "").strip())
            for row in csv.DictReader(handle)
        ]


def _stats(values: Sequence[int]) -> dict[str, Any]:
    if not values:
        return {"n": 0, "mean_rating": None, "deteriorated_rate": None,
                "improved_rate": None, "rate_1": None, "rate_2": None,
                "rate_4": None, "rate_5": None}
    total = len(values)
    return {
        "n": total,
        "mean_rating": round(sum(values) / total, 4),
        "deteriorated_rate": round(
            sum(1 for value in values if value <= DETERIORATED_MAX) / total, 4),
        "improved_rate": round(
            sum(1 for value in values if value >= IMPROVED_MIN) / total, 4),
        "rate_1": round(sum(1 for value in values if value == 1) / total, 4),
        "rate_2": round(sum(1 for value in values if value == 2) / total, 4),
        "rate_4": round(sum(1 for value in values if value == 4) / total, 4),
        "rate_5": round(sum(1 for value in values if value == 5) / total, 4),
    }


def _equal_width_bins(values: Sequence[float], count: int) -> list[float]:
    low, high = min(values), max(values)
    if high <= low:
        high = low + 1.0
    step = (high - low) / count
    return [low + step * index for index in range(count + 1)]


def _bin_index(value: float, edges: Sequence[float]) -> int:
    for index in range(len(edges) - 1):
        if value < edges[index + 1]:
            return index
    return len(edges) - 2


def cmd_analyze(args: argparse.Namespace) -> int:
    key = json.loads(Path(args.item_key or (Path(args.csv).parent / "item_key.json"))
                     .read_text(encoding="utf-8"))
    items: dict[str, Any] = key["items"]
    rows = read_ratings(Path(args.csv))

    ratings: dict[str, int] = {}
    unknown_item: list[str] = []
    invalid: list[str] = []
    for item_id, raw, _notes in rows:
        if item_id not in items:
            unknown_item.append(item_id)
            continue
        if raw == "":
            continue
        try:
            value = int(raw)
        except ValueError:
            invalid.append(f"{item_id}={raw}")
            continue
        if not RATING_MIN <= value <= RATING_MAX:
            invalid.append(f"{item_id}={raw}")
            continue
        ratings[item_id] = value
    missing = sorted(item_id for item_id in items if item_id not in ratings)

    grid_cells: dict[tuple[str, str], list[int]] = defaultdict(list)
    for item_id, value in sorted(ratings.items()):
        entry = items[item_id]
        grid_cells[(entry["global"]["bin"], entry["local"]["bin"])].append(value)
    global_bins = _bin_order([cell[0] for cell in grid_cells], GLOBAL_BIN_ORDER)
    local_bins = _bin_order([cell[1] for cell in grid_cells], LOCAL_BIN_ORDER)
    grid = [
        dict(global_bin=gbin, local_bin=lbin, **_stats(grid_cells[(gbin, lbin)]))
        for gbin in global_bins for lbin in local_bins
        if (gbin, lbin) in grid_cells
    ]

    def de_curve(level: str, field: str) -> dict[str, Any]:
        values = [
            (float(items[item_id][level][field]), value)
            for item_id, value in sorted(ratings.items())
            if items[item_id][level].get(field) is not None
        ]
        null_n = len(ratings) - len(values)
        if not values:
            return {"field": f"{level}.{field}", "edges": [], "bins": [],
                    "n_null_value": null_n}
        edges = _equal_width_bins([value for value, _ in values], DE_BINS)
        buckets: dict[int, list[int]] = defaultdict(list)
        for measure, rating in values:
            buckets[_bin_index(measure, edges)].append(rating)
        return {
            "field": f"{level}.{field}",
            "edges": [round(edge, 4) for edge in edges],
            "bins": [
                dict(bin=index, low=round(edges[index], 4),
                     high=round(edges[index + 1], 4), **_stats(buckets[index]))
                for index in range(DE_BINS)
            ],
            "n_null_value": null_n,
        }

    scene_cells: dict[str, list[int]] = defaultdict(list)
    scene_null = 0
    for item_id, value in sorted(ratings.items()):
        scene = items[item_id].get("scene")
        if scene is None:
            scene_null += 1
            continue
        scene_cells[str(scene)].append(value)
    scenes = [dict(scene=name, **_stats(scene_cells[name]))
              for name in sorted(scene_cells)]

    rated_confidence = winner_confidence_counts(
        items[item_id].get("winner_confidence") for item_id in sorted(ratings)
    )
    confidence_warning = print_winner_confidence_warning(
        rated_confidence, filtered=WINNER_CONFIDENCE_FILTERED
    )

    analysis = {
        "schema": SCHEMA,
        "csv": str(args.csv),
        "item_key": str(args.item_key or (Path(args.csv).parent / "item_key.json")),
        "n_items": len(items),
        "n_rated": len(ratings),
        "n_missing_rating": len(missing),
        "missing_rating": missing,
        "n_invalid_rating": len(invalid),
        "invalid_rating": invalid,
        "n_unknown_item_id": len(unknown_item),
        "unknown_item_id": sorted(unknown_item),
        "overall": _stats(sorted(ratings.values())),
        "grid": grid,
        "de_curves": [de_curve("global", "de_measured"),
                      de_curve("local", "de_masked")],
        "scene_null": scene_null,
        "scenes": scenes,
        # C1b item 3: the rated population's confidence mix travels with the analysis.
        "winner_confidence_filtered": WINNER_CONFIDENCE_FILTERED,
        "winner_confidence_counts": rated_confidence,
        "winner_confidence_warning": confidence_warning,
    }
    out_path = Path(args.out) if args.out else Path(args.csv).parent / "analysis.json"
    out_path.write_text(
        json.dumps(analysis, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8", newline="\n",
    )
    lines = [
        *([confidence_warning, ""] if confidence_warning else []),
        f"n_items = {len(items)}  n_rated = {len(ratings)}  "
        f"n_missing = {len(missing)}  n_invalid = {len(invalid)}  "
        f"n_unknown_item_id = {len(unknown_item)}",
        "",
        "global_bin x local_bin",
        f"{'global':<10}{'local':<10}{'n':>6}{'mean':>8}{'<=2':>8}{'=1':>8}"
        f"{'=2':>8}{'>=4':>8}{'=4':>8}{'=5':>8}",
    ]
    for row in grid:
        lines.append(
            f"{row['global_bin']:<10}{row['local_bin']:<10}{row['n']:>6}"
            f"{row['mean_rating']:>8}{row['deteriorated_rate']:>8}"
            f"{row['rate_1']:>8}{row['rate_2']:>8}"
            f"{row['improved_rate']:>8}{row['rate_4']:>8}{row['rate_5']:>8}"
        )
    for curve in analysis["de_curves"]:
        lines.extend(["", curve["field"] + f"  (n_null_value = {curve['n_null_value']})",
                      f"{'bin':<6}{'low':>10}{'high':>10}{'n':>6}{'mean':>8}"
                      f"{'<=2':>8}{'=1':>8}{'=2':>8}{'>=4':>8}{'=4':>8}"
                      f"{'=5':>8}"])
        for row in curve["bins"]:
            lines.append(
                f"{row['bin']:<6}{row['low']:>10}{row['high']:>10}{row['n']:>6}"
                f"{str(row['mean_rating']):>8}{str(row['deteriorated_rate']):>8}"
                f"{str(row['rate_1']):>8}{str(row['rate_2']):>8}"
                f"{str(row['improved_rate']):>8}{str(row['rate_4']):>8}"
                f"{str(row['rate_5']):>8}"
            )
    lines.extend(["", f"scene  (scene_null = {scene_null})",
                  f"{'scene':<24}{'n':>6}{'mean':>8}"])
    for row in scenes:
        lines.append(f"{row['scene']:<24}{row['n']:>6}{row['mean_rating']:>8}")
    lines.extend(["", f"written {out_path}"])
    print("\n".join(lines))
    return 0


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)

    build = sub.add_parser("build", help="sample 300 committed chains and write the page")
    build.add_argument("--campaign", default=DEFAULT_CAMPAIGN)
    build.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    build.add_argument("--dsn", default=None,
                       help="default $VERARETOUCH_AGENT_POSTGRES_DSN")
    build.add_argument("--artifact-root", type=Path, action="append", default=[],
                       help="repeatable; default = every */blobs under "
                            + " and ".join(str(item) for item in DEFAULT_ROOT_PARENTS))
    build.add_argument("--catalog-db", type=Path,
                       default=DEFAULT_CATALOG_DB if DEFAULT_CATALOG_DB.is_file()
                       else None)
    build.add_argument("--thumb-short-edge", type=int, default=THUMB_SHORT_EDGE)
    build.add_argument("--target", type=int, default=TARGET_N)
    build.add_argument("--per-source-cap", type=int, default=PER_SOURCE_CAP)
    build.set_defaults(func=cmd_build)

    analyze = sub.add_parser("analyze", help="read the filled CSV back with item_key")
    analyze.add_argument("--csv", type=Path, required=True)
    analyze.add_argument("--item-key", type=Path, default=None,
                         help="default <csv dir>/item_key.json")
    analyze.add_argument("--out", type=Path, default=None,
                         help="default <csv dir>/analysis.json")
    analyze.set_defaults(func=cmd_analyze)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    return int(args.func(args))


if __name__ == "__main__":
    raise SystemExit(main())
