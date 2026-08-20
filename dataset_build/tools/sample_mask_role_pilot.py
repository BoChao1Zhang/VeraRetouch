"""Review-only pilot for the Mask v2 background role (B1).

Deterministically samples sources from the local-v1 source list, builds the
role-aware mask bank (`include_background=True`), allocates one sibling packet
per source, and writes review sheets plus machine-readable measurements.

Run from the repository root:

    PYTHONPATH=. .venv/bin/python -m dataset_build.tools.sample_mask_role_pilot
"""
from __future__ import annotations

import argparse
import hashlib
import html
import json
import tempfile
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image, ImageDraw, ImageFont, ImageOps

from dataset_build.agent_loop.artifacts import ArtifactStore
from dataset_build.agent_loop.candidates import (
    BACKGROUND_ROLE_GATE, CandidateError, allocate_role_packets, build_mask_bank,
)

BUILD_ID = "mask-role-pilot-b1"
DEFAULT_SOURCES = Path("/home/bc/data/agent_loop/local-v1/sources5k.jsonl")
DEFAULT_OUT = Path("docs/assets/mask_role_pilot_20260819")
SUBJECT_AREA_SKIP = 0.60
AREA_BUCKETS = (
    ("tiny", 0.0, 0.020),
    ("small", 0.020, 0.060),
    ("medium", 0.060, 0.150),
    ("large", 0.150, SUBJECT_AREA_SKIP + 1e-9),
)
ROLE_COLORS = {
    "subject": (49, 196, 112),
    "background": (86, 156, 255),
}
SUBJECT_OUTLINE = (238, 73, 73)
_RESAMPLING = getattr(Image, "Resampling", Image)


def _bucket(area: float) -> str:
    for name, low, high in AREA_BUCKETS:
        if low <= area < high:
            return name
    return "skipped"


def _font(size: int) -> Any:
    path = Path("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf")
    return ImageFont.truetype(str(path), size) if path.is_file() \
        else ImageFont.load_default()


def _outline(binary: np.ndarray) -> np.ndarray:
    """Inner four-neighbour boundary of a boolean field (no scipy/cv2 dependency)."""
    inner = np.asarray(binary, dtype=bool).copy()
    inner[:-1, :] &= binary[1:, :]
    inner[1:, :] &= binary[:-1, :]
    inner[:, :-1] &= binary[:, 1:]
    inner[:, 1:] &= binary[:, :-1]
    return np.asarray(binary, dtype=bool) & ~inner


def _panel(image: Image.Image, core: np.ndarray, alpha: np.ndarray | None,
           title: str, lines: list[str], color: tuple[int, int, int]) -> Image.Image:
    rgb = np.asarray(image, dtype=np.float32)
    if alpha is not None:
        weight = (0.58 * alpha)[..., None]
        rgb = rgb * (1.0 - weight) + np.asarray(color, dtype=np.float32) * weight
        edge = _outline(alpha >= 0.5)
        rgb[edge] = np.asarray(color, dtype=np.float32)
    rgb[_outline(core > 0.5)] = np.asarray(SUBJECT_OUTLINE, dtype=np.float32)
    body = Image.fromarray(np.clip(rgb, 0, 255).astype(np.uint8), "RGB")
    bar = 26 + 18 * len(lines)
    panel = Image.new("RGB", (body.width, body.height + bar), (20, 23, 27))
    panel.paste(body, (0, bar))
    draw = ImageDraw.Draw(panel)
    draw.text((9, 4), title, fill=(244, 246, 248), font=_font(14))
    for index, line in enumerate(lines):
        draw.text((9, 24 + 18 * index), line, fill=(188, 195, 204), font=_font(12))
    return panel


def _mask_lines(mask: dict[str, Any]) -> list[str]:
    if str(mask["role"]) == "background":
        return [
            f"subj_alpha {mask['subject_alpha_mean']:.3f} (<= "
            f"{BACKGROUND_ROLE_GATE['subject_alpha_mean_max']}) | subj>=.5 "
            f"{mask['subject_high_coverage']:.4f} (<= "
            f"{BACKGROUND_ROLE_GATE['subject_high_coverage_max']})",
            f"bg_alpha {mask['background_alpha_mean']:.3f} (>= "
            f"{BACKGROUND_ROLE_GATE['background_alpha_mean_min']}) | area>=.5 "
            f"{mask['half_area']:.3f} (>= {BACKGROUND_ROLE_GATE['half_area_min']})",
        ]
    return [
        f"alpha_mean {mask['effective_alpha_mean']:.3f} (> 0.45) | area>=.5 "
        f"{mask['half_area']:.3f}",
        f"subj>=.5 {mask['subject_high_coverage']:.4f} (>= 0.98) | subj>.05 "
        f"{mask['subject_support_coverage']:.4f} (= 1.0)",
    ]


def _load_alpha(store: ArtifactStore, mask: dict[str, Any],
                size: tuple[int, int]) -> np.ndarray:
    with Image.open(store.path_for(mask["alpha_artifact"])) as handle:
        alpha = handle.convert("L").resize(size, _RESAMPLING.BILINEAR)
    return np.asarray(alpha, dtype=np.float32) / 255.0


def _save_sheet(path: Path, image: Image.Image, core: np.ndarray, row: dict[str, Any],
                packet: list[dict[str, Any]], alphas: list[np.ndarray]) -> None:
    panels = [_panel(image, core, None, "original", [
        f"{row['source_id']} | {row['scene']}",
        f"subject_area {row['subject_area']:.3f} | bucket {row['bucket']}",
    ], SUBJECT_OUTLINE)]
    for mask, alpha in zip(packet, alphas):
        panels.append(_panel(
            image, core, alpha,
            f"{mask['role']} | {mask['family']} | {mask['direction']}",
            _mask_lines(mask), ROLE_COLORS[str(mask["role"])],
        ))
    header = 62
    width = max(panel.width for panel in panels)
    height = max(panel.height for panel in panels)
    sheet = Image.new("RGB", (len(panels) * width, header + height), (12, 14, 17))
    draw = ImageDraw.Draw(sheet)
    note = row["packet_note"]
    draw.text((12, 6), (
        f"{row['source_id']} | subject_area={row['subject_area']:.3f} | "
        f"roles={note['role_counts']} | fallback={note['fallback']}"
    ), fill=(245, 247, 249), font=_font(17))
    draw.text((12, 30), (
        f"bank: {row['bank_subject']} subject + {row['bank_background']} background | "
        f"background rejects: {row['reject_counts'] or '{}'}"
    ), fill=(174, 181, 191), font=_font(13))
    draw.text((12, 46), (
        "red outline = SAM subject; green fill = subject-role mask; "
        "blue fill = background-role mask"
    ), fill=(150, 158, 168), font=_font(12))
    for index, panel in enumerate(panels):
        sheet.paste(panel, (index * width, header))
    sheet.save(path, quality=92)


STORE_KEY = "mask-role-pilot-b1:20260819"
CSV_HEADER = "sheet_id,rating,notes"
RATING_MIN, RATING_MAX = 1, 5
RATING_LABELS = (
    (1, "不可用（mask 明显错）"),
    (2, "背景 mask 侵入主体，或覆盖差"),
    (3, "勉强可用"),
    (4, "良好"),
    (5, "完美贴合精修直觉"),
)

_STYLE = """
:root{color-scheme:dark}
*{box-sizing:border-box}
body{margin:0;padding:12px 16px 28px;background:#111418;color:#edf0f3;
font:14px system-ui,'Noto Sans CJK SC',sans-serif}
h1{font-size:17px;margin:0 0 4px}
.hint{font-size:13px;color:#aeb5bf;margin:0 0 8px}
.topbar{display:flex;align-items:center;gap:14px;flex-wrap:wrap;margin-bottom:6px}
#progress{font-size:20px;font-weight:700;font-variant-numeric:tabular-nums}
#done,#unrated,#meta{font-size:13px;color:#aeb5bf}
.keys{font-size:12px;color:#7f8894}
.track{height:6px;background:#252b32;border-radius:3px;overflow:hidden;margin:0 0 10px}
#bar{height:100%;background:#569cff;width:0}
#sheetlink{display:block}
#sheet{display:block;width:100%;height:auto;max-height:62vh;object-fit:contain;
background:#0c0e11;border:1px solid #303740}
.rates{display:flex;gap:8px;flex-wrap:wrap;margin:12px 0 8px}
.rate{flex:1 1 190px;padding:14px 10px;font:15px/1.35 inherit;cursor:pointer;
border:1px solid #3b434d;border-radius:8px;background:#1b2026;color:#edf0f3;
text-align:left}
.rate:hover{background:#232a32}
.rate.on{background:#2b6cb0;border-color:#569cff;color:#fff}
.rate b{font-size:18px;margin-right:8px}
.nav{display:flex;gap:10px;align-items:center;flex-wrap:wrap;margin:10px 0}
.nav button,#export{padding:10px 16px;font:14px inherit;cursor:pointer;
border:1px solid #3b434d;border-radius:6px;background:#1b2026;color:#edf0f3}
.nav button:disabled{opacity:.4;cursor:default}
#notes{width:100%;padding:6px;font:13px inherit;background:#1b2026;color:#edf0f3;
border:1px solid #3b434d;border-radius:6px}
details{margin-top:18px;border-top:1px solid #303740;padding-top:10px}
summary{cursor:pointer;color:#aeb5bf}
details pre{color:#aeb5bf;font:12px monospace;white-space:pre-wrap}
"""

_SCRIPT = """
const SHEETS = __SHEETS__;
const STORE_KEY = __STORE_KEY__;
const total = SHEETS.length;
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
  for (const item of SHEETS) { if (state.ratings[item.sheet_id]) { n += 1; } }
  return n;
}

function render() {
  const item = SHEETS[cursor];
  document.getElementById('sheet').src = item.file;
  document.getElementById('sheetlink').href = item.file;
  document.getElementById('meta').textContent = item.meta;
  document.getElementById('progress').textContent = (cursor + 1) + ' / ' + total;
  const done = ratedCount();
  const left = total - done;
  document.getElementById('done').textContent = '已评 ' + done + ' / ' + total;
  document.getElementById('bar').style.width =
    (total ? ((cursor + 1) * 100 / total) : 0) + '%';
  const current = state.ratings[item.sheet_id] || 0;
  for (const btn of document.querySelectorAll('.rate')) {
    btn.classList.toggle('on', parseInt(btn.dataset.value, 10) === current);
  }
  document.getElementById('notes').value = state.notes[item.sheet_id] || '';
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
  state.ratings[SHEETS[cursor].sheet_id] = value;
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
  for (const item of SHEETS) {
    const rating = state.ratings[item.sheet_id]
      ? String(state.ratings[item.sheet_id]) : '';
    lines.push(csvField(item.sheet_id) + ',' + rating + ','
               + csvField(state.notes[item.sheet_id] || ''));
  }
  return lines.join('\\n') + '\\n';
}

function exportCsv() {
  const blob = new Blob([buildCsv()], {type: 'text/csv;charset=utf-8'});
  const url = URL.createObjectURL(blob);
  const link = document.createElement('a');
  link.href = url;
  link.download = 'mask_role_pilot.csv';
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
    btn.addEventListener('click', function () {
      rate(parseInt(btn.dataset.value, 10));
    });
  }
  document.getElementById('prev').addEventListener('click', function () { go(-1); });
  document.getElementById('next').addEventListener('click', function () { go(1); });
  document.getElementById('export').addEventListener('click', exportCsv);
  document.getElementById('notes').addEventListener('input', function (event) {
    state.notes[SHEETS[cursor].sheet_id] = event.target.value;
    save();
  });
  document.addEventListener('keydown', function (event) {
    if (event.ctrlKey || event.metaKey || event.altKey || inField(event.target)) {
      return;
    }
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


def _js_json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False).replace("</", "<\\/")


def _write_index(out: Path, records: list[dict[str, Any]], summary: dict[str, Any]) -> None:
    """One sheet per screen, five rating buttons, CSV export (display order)."""
    sheets = []
    for record in records:
        note = record["packet_note"]
        sheets.append({
            "sheet_id": Path(str(record["file"])).stem,
            "file": record["file"],
            "meta": (
                f"{record['bucket']} | A={record['subject_area']:.3f} | "
                f"subj={note['role_counts']['subject']} "
                f"bg={note['role_counts']['background']}"
                f"{' | FALLBACK' if note['fallback'] else ''}"
            ),
        })
    script = (
        _SCRIPT
        .replace("__SHEETS__", _js_json(sheets))
        .replace("__STORE_KEY__", _js_json(STORE_KEY))
        .replace("__HEADER__", _js_json(CSV_HEADER))
        .replace("__RMIN__", str(RATING_MIN))
        .replace("__RMAX__", str(RATING_MAX))
    )
    buttons = "".join(
        f'<button type="button" class="rate" data-value="{value}">'
        f"<b>{value}</b>{html.escape(label)}</button>"
        for value, label in RATING_LABELS
    )
    stats = json.dumps({
        key: summary[key] for key in (
            "count", "background_gate", "role_feasibility", "by_bucket",
            "background_rejects", "skipped_large_subject",
        ) if key in summary
    }, ensure_ascii=False, indent=1)
    parts = [
        '<!doctype html><html lang="zh"><head><meta charset="utf-8">',
        '<meta name="viewport" content="width=device-width,initial-scale=1">',
        "<title>Mask role pilot 100 (B1)</title>",
        f"<style>{_STYLE}</style></head><body>",
        "<h1>Mask role pilot 100 (B1)</h1>",
        '<p class="hint">每屏一张 sheet：最左是原图，右边三张是同一张图的 sibling '
        "mask（红轮廓 = SAM 主体，绿填充 = subject-role，蓝填充 = background-role）。"
        "对这一组 mask 整体打分；评分自动存在本机浏览器，标完点「导出 CSV」。</p>",
        '<div class="topbar"><span id="progress">- / -</span>'
        '<span id="done"></span><span id="unrated"></span><span id="meta"></span>'
        '<span class="keys">键盘：1–5 评分并跳下一条，← / → 翻页</span></div>',
        '<div class="track"><div id="bar"></div></div>',
        '<a id="sheetlink" href="#" target="_blank" rel="noopener">'
        '<img id="sheet" alt="mask role sheet"></a>',
        f'<div class="rates">{buttons}</div>',
        '<div class="nav"><button type="button" id="prev">← 上一条</button>'
        '<button type="button" id="next">下一条 →</button>'
        '<button type="button" id="export">导出 CSV</button></div>',
        '<textarea id="notes" rows="2" placeholder="备注（可留空）"></textarea>',
        f"<details><summary>summary.json 汇总</summary><pre>{html.escape(stats)}"
        "</pre></details>",
        f"<script>{script}</script>",
        "</body></html>",
    ]
    (out / "index.html").write_text(
        "\n".join(parts) + "\n", encoding="utf-8", newline="\n"
    )


def _sources(path: Path) -> list[dict[str, Any]]:
    rows = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                rows.append(json.loads(line))
    return sorted(
        rows, key=lambda row: hashlib.sha1(
            str(row["source_id"]).encode("utf-8")
        ).hexdigest()
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--sources", type=Path, default=DEFAULT_SOURCES)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--count", type=int, default=100)
    parser.add_argument("--long-edge", type=int, default=640)
    parser.add_argument(
        "--page-only", action="store_true",
        help="rewrite index.html from the existing samples.jsonl / summary.json "
             "without touching the sheet images",
    )
    args = parser.parse_args(argv)
    args.out.mkdir(parents=True, exist_ok=True)

    if args.page_only:
        with (args.out / "samples.jsonl").open("r", encoding="utf-8") as handle:
            records = [json.loads(line) for line in handle if line.strip()]
        summary = json.loads((args.out / "summary.json").read_text(encoding="utf-8"))
        _write_index(args.out, records, summary)
        print(json.dumps({"out": str(args.out), "page_only": True,
                          "sheets": len(records)}))
        return 0

    store = ArtifactStore(Path(tempfile.mkdtemp(prefix="mask-role-pilot-")))
    accepted: list[dict[str, Any]] = []
    # C1b item 10: one accepted record costs a mask build plus a rendered contact
    # sheet, and the loop runs for hundreds of sources. Each record is appended to
    # `samples.jsonl` the moment it is accepted, so a crash (or a kill) partway through
    # leaves a readable partial run instead of only orphan PNGs. `--page-only` reads
    # exactly this file, so a partial run is still inspectable.
    samples_handle = (args.out / "samples.jsonl").open("w", encoding="utf-8")
    skipped_large = 0
    build_failures: Counter[str] = Counter()
    rejects: Counter[str] = Counter()
    attempts = 0

    for row in _sources(args.sources):
        if len(accepted) >= args.count:
            break
        if float(row["subject"]["mask_area"]) > SUBJECT_AREA_SKIP:
            skipped_large += 1
            continue
        attempts += 1
        with Image.open(row["source_path"]) as handle:
            image = ImageOps.exif_transpose(handle).convert("RGB")
        scale = min(1.0, args.long_edge / max(image.size))
        size = (max(1, round(image.width * scale)),
                max(1, round(image.height * scale)))
        image = image.resize(size, _RESAMPLING.LANCZOS)
        diagnostics: list[dict[str, Any]] = []
        try:
            masks = build_mask_bank(
                row["subject_path"], render_size=size,
                source_id=str(row["source_id"]), prompt_revision=BUILD_ID,
                artifacts=store, include_background=True, diagnostics=diagnostics,
            )
            packet, _remaining, note = allocate_role_packets(masks, str(row["source_id"]))
        except CandidateError as exc:
            build_failures[str(exc).split(":")[0][:60]] += 1
            continue
        for item in diagnostics:
            rejects[f"{item['family']}:{item['reason']}"] += 1
        with Image.open(row["subject_path"]) as handle:
            core = np.asarray(
                ImageOps.exif_transpose(handle).convert("L").resize(
                    size, _RESAMPLING.NEAREST
                ), dtype=np.float32
            ) / 255.0
        core = (core > 0.5).astype(np.float32)
        area = float(masks[0]["subject_area"])
        record = {
            "index": len(accepted) + 1,
            "file": f"{len(accepted) + 1:03d}_{row['source_id'][:20]}.jpg",
            "source_id": row["source_id"],
            "source_path": row["source_path"],
            "scene": row.get("scene", "unknown"),
            "subject_area": area,
            "bucket": _bucket(area),
            "bank_subject": sum(1 for m in masks if m["role"] == "subject"),
            "bank_background": sum(1 for m in masks if m["role"] == "background"),
            "reject_counts": dict(Counter(
                f"{item['family']}:{item['reason']}" for item in diagnostics
            )),
            "packet_note": note,
            "packet": [{
                key: mask[key] for key in (
                    "mask_id", "role", "family", "direction", "center_hint",
                    "effective_alpha_mean", "half_area", "support_area",
                    "subject_high_coverage", "subject_support_coverage",
                ) if key in mask
            } | {
                key: mask[key] for key in ("subject_alpha_mean", "background_alpha_mean")
                if key in mask
            } for mask in packet],
            "background_bank": [{
                "family": mask["family"], "direction": mask["direction"],
                "subject_alpha_mean": mask["subject_alpha_mean"],
                "subject_high_coverage": mask["subject_high_coverage"],
                "background_alpha_mean": mask["background_alpha_mean"],
                "half_area": mask["half_area"],
            } for mask in masks if mask["role"] == "background"],
        }
        _save_sheet(
            args.out / record["file"], image, core, record, packet,
            [_load_alpha(store, mask, size) for mask in packet],
        )
        accepted.append(record)
        samples_handle.write(json.dumps(record, ensure_ascii=False) + "\n")
        samples_handle.flush()
        print(json.dumps({"accepted": len(accepted), "file": record["file"]}), flush=True)

    by_bucket: dict[str, dict[str, int]] = {}
    for name, _low, _high in AREA_BUCKETS:
        rows = [record for record in accepted if record["bucket"] == name]
        by_bucket[name] = {
            "n": len(rows),
            "packet_with_background": sum(
                1 for record in rows if record["packet_note"]["role_counts"]["background"]
            ),
            "fallback_background_infeasible": sum(
                1 for record in rows if record["packet_note"]["fallback"]
            ),
            "bank_background_masks": sum(record["bank_background"] for record in rows),
            "bank_background_zero": sum(
                1 for record in rows if record["bank_background"] == 0
            ),
        }
    summary = {
        "build_id": BUILD_ID,
        "count": len(accepted),
        "attempts": attempts,
        "sources": str(args.sources),
        "sampling": "sha1(source_id) ascending; subject mask_area > 0.60 skipped",
        "skipped_large_subject": skipped_large,
        "build_failures": dict(build_failures),
        "background_gate": dict(BACKGROUND_ROLE_GATE),
        "role_feasibility": {
            "packet_with_background": sum(
                1 for record in accepted
                if record["packet_note"]["role_counts"]["background"]
            ),
            "fallback_background_infeasible": sum(
                1 for record in accepted if record["packet_note"]["fallback"]
            ),
            "bank_background_zero": sum(
                1 for record in accepted if record["bank_background"] == 0
            ),
            "bank_background_total": sum(record["bank_background"] for record in accepted),
        },
        "by_bucket": by_bucket,
        "background_rejects": dict(sorted(rejects.items())),
        "background_family_counts": dict(Counter(
            mask["family"] for record in accepted for mask in record["background_bank"]
        )),
        "packet_role_counts": dict(Counter(
            json.dumps(record["packet_note"]["role_counts"], sort_keys=True)
            for record in accepted
        )),
    }
    samples_handle.close()
    (args.out / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    # The contact-sheet index is a viewing convenience; a failure here must not lose the
    # records and the summary that are already on disk.
    index_error = None
    try:
        _write_index(args.out, accepted, summary)
    except Exception as exc:  # noqa: BLE001 - reported, never silent
        index_error = f"{type(exc).__name__}: {exc}"
    print(json.dumps({"out": str(args.out), "index_error": index_error,
                      **summary["role_feasibility"]}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
