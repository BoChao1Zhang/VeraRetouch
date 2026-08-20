"""Blind questionnaire for the 30 decision-boundary LUT pairs produced by A1f (task card A1g).

Input is the A1f backfill candidate list (30 JSONL rows carrying ``preset_a`` / ``preset_b`` /
``style_major`` plus the metric-A/B numbers).  Output is a self-contained static page with the
same interaction as the first two questionnaires (single pair per screen, 5 large rating
buttons, localStorage autosave under its own key, CSV export) in its own directory.

Blind: the page and the empty CSV carry only ``b30_pair_*`` ids -- no preset id, no
``style_major``, no probability / distance number.  Those live in ``pair_key_b30.json``.

Probes: the same 4 scene probes as A1d/A1f, rotated by pair ordinal (``ordinal % 4``).

Usage:
    python -m dataset_build.tools.lut_pair_questionnaire_b30 build --workers 8
    python -m dataset_build.tools.lut_pair_questionnaire_b30 build --skip-render  # determinism
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
    load_catalog,
)
from dataset_build.tools.lut_pair_questionnaire import (  # noqa: E402
    RATING_LABELS,
    RATING_MAX,
    RATING_MIN,
    _HTML_SCRIPT,
    _HTML_STYLE,
    _js_json,
)
from dataset_build.tools.lut_pair_questionnaire_ext import (  # noqa: E402
    DEFAULT_PROBE_FALLBACK_DIR,
    DEFAULT_PROBE_MANIFESTS,
    DEFAULT_PROBES,
    _init_worker,
    _render_pair,
    load_probes,
)

B30_SCHEMA = "lut-pair-questionnaire-b30-v1"
DEFAULT_B30_SEED = 20260821
PAIR_PREFIX = "b30_pair"
CSV_HEADER = "item_id,rating,notes"
DEFAULT_CANDIDATES = Path(
    "/home/bc/data/scratch/lut_clusters/metric_ab/backfill_candidates.jsonl")

# metric columns copied verbatim from the candidate rows into the answer key
CANDIDATE_METRICS = (
    "abs_gap_to_threshold",
    "c3_probability",
    "feat_l2",
    "render_mean_1probe",
    "render_mean_4probe",
    "render_p95_4probe",
)


def storage_key(seed: int) -> str:
    return f"{B30_SCHEMA}:seed:{seed}"


# --------------------------------------------------------------------------- selection


def load_candidates(path: Path) -> list[dict[str, Any]]:
    """Candidate rows in file order; ``preset_a``/``preset_b``/``style_major`` required."""
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for lineno, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            row = json.loads(line)
            for field in ("preset_a", "preset_b", "style_major"):
                if not row.get(field):
                    raise ValueError(f"{path}:{lineno} missing {field}")
            rows.append(row)
    if not rows:
        raise ValueError(f"{path} has no candidate rows")
    return rows


def build_pairs(rows: Sequence[dict[str, Any]], n_probes: int) -> list[dict[str, Any]]:
    """One entry per candidate row, ids by file order, probe rotated by pair ordinal."""
    pairs: list[dict[str, Any]] = []
    for ordinal, row in enumerate(rows):
        entry: dict[str, Any] = {
            "pair_id": f"{PAIR_PREFIX}_{ordinal + 1:03d}",
            "candidate_index": ordinal,
            "preset_a": str(row["preset_a"]),
            "preset_b": str(row["preset_b"]),
            "style_major": str(row["style_major"]),
            "probe_index": ordinal % n_probes,
        }
        for key in CANDIDATE_METRICS:
            if key in row:
                entry[key] = row[key]
        pairs.append(entry)
    return pairs


def display_order(pairs: Sequence[dict[str, Any]], seed: int) -> list[str]:
    """Seeded permutation so the on-screen order does not follow the candidate ranking."""
    rng = np.random.default_rng(seed + 2)
    return [pairs[i]["pair_id"] for i in rng.permutation(len(pairs)).tolist()]


# --------------------------------------------------------------------------- outputs


def write_csv(out_dir: Path, order: Sequence[str], name: str) -> None:
    lines = [CSV_HEADER]
    lines.extend(f"{item_id},," for item_id in order)
    (out_dir / name).write_text("\n".join(lines) + "\n", encoding="utf-8", newline="\n")


def write_key(out_dir: Path, pairs: Sequence[dict[str, Any]], order: Sequence[str],
              probes: Sequence[dict[str, Any]], seed: int, candidates: Path,
              name: str) -> None:
    payload = {
        "schema": B30_SCHEMA,
        "feature_spec": FEATURE_SPEC,
        "seed": seed,
        "candidates_source": str(candidates),
        "display_order": list(order),
        "majors_used": sorted({entry["style_major"] for entry in pairs}),
        "probes": [
            {"index": index, "scene": probe["scene"], "source": probe["source"]}
            for index, probe in enumerate(probes)
        ],
        "pairs": {
            entry["pair_id"]: {
                **{key: entry[key] for key in sorted(entry) if key != "pair_id"},
                "probe_scene": probes[entry["probe_index"]]["scene"],
            }
            for entry in pairs
        },
        "pair_rating_scale": {str(value): text for value, text in RATING_LABELS},
    }
    (out_dir / name).write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8", newline="\n",
    )


_CSV_NAME_TOKEN = "link.download = 'questionnaire.csv';"


def write_html(out_dir: Path, order: Sequence[str], failures: Sequence[tuple[str, str]],
               seed: int, csv_name: str, html_name: str) -> None:
    if _CSV_NAME_TOKEN not in _HTML_SCRIPT:
        raise RuntimeError("upstream _HTML_SCRIPT changed: csv download name token not found")
    script = (
        _HTML_SCRIPT
        .replace(_CSV_NAME_TOKEN, "link.download = __CSVNAME__;")
        .replace("__PAIRS__", _js_json(list(order)))
        .replace("__STORE_KEY__", _js_json(storage_key(seed)))
        .replace("__HEADER__", _js_json(CSV_HEADER))
        .replace("__CSVNAME__", _js_json(csv_name))
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
        "<title>LUT 对盲标问卷 B30 2026-08-19</title>",
        f"<style>{_HTML_STYLE}</style></head><body>",
        "<h1>LUT 对盲标问卷 · 补标 30 对</h1>",
        '<p class="hint">左右两张是同一张探针图套两个不同 LUT 的结果。给这一对的差别打分，'
        "评分自动保存在本机浏览器；标完点「导出 CSV」下载填好的 %s。</p>"
        % html.escape(csv_name),
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
    (out_dir / html_name).write_text(
        "\n".join(part for part in parts if part), encoding="utf-8", newline="\n"
    )


# --------------------------------------------------------------------------- command


def cmd_build(args: argparse.Namespace) -> int:
    catalog, databuild, _ = load_catalog(args.config)
    rows = load_candidates(args.candidates)

    manifests = list(args.probe_jsonl or DEFAULT_PROBE_MANIFESTS)
    probes = load_probes(manifests, args.probe_short_edge, args.probes,
                         args.probe_fallback_dir)
    pairs = build_pairs(rows, len(probes))

    missing = sorted({
        preset
        for entry in pairs
        for preset in (entry["preset_a"], entry["preset_b"])
        if preset not in catalog.by_id
    })
    if missing:
        raise ValueError(f"{len(missing)} preset ids absent from catalog: {missing}")

    order = display_order(pairs, args.seed)

    out_dir = args.out_dir
    (out_dir / "pairs").mkdir(parents=True, exist_ok=True)

    failures: list[tuple[str, str]] = []
    elapsed = 0.0
    rendered = 0
    if not args.skip_render:
        jobs = [
            (entry["pair_id"], catalog.by_id[entry["preset_a"]].path,
             catalog.by_id[entry["preset_b"]].path, entry["probe_index"],
             str(out_dir / "pairs" / f"{entry['pair_id']}.png"), args.side_edge, args.gap)
            for entry in pairs
        ]
        started = time.time()
        arrays = [probe["array"] for probe in probes]
        with futures.ProcessPoolExecutor(
            max_workers=args.workers, initializer=_init_worker,
            initargs=(arrays, str(databuild)),
        ) as pool:
            for pair_id, ok, detail, _ in pool.map(_render_pair, jobs, chunksize=1):
                if ok:
                    rendered += 1
                else:
                    failures.append((pair_id, detail))
        elapsed = time.time() - started

    write_key(out_dir, pairs, order, probes, args.seed, args.candidates, args.key_name)
    write_csv(out_dir, order, args.csv_name)
    write_html(out_dir, order, failures, args.seed, args.csv_name, args.html_name)

    major_probe: dict[str, dict[str, int]] = {}
    probe_counts: dict[str, int] = {}
    for entry in pairs:
        block = major_probe.setdefault(entry["style_major"], {})
        key = str(entry["probe_index"])
        block[key] = block.get(key, 0) + 1
        probe_counts[key] = probe_counts.get(key, 0) + 1
    print(json.dumps({
        "out_dir": str(out_dir),
        "pairs": len(pairs),
        "candidates_source": str(args.candidates),
        "storage_key": storage_key(args.seed),
        "probes": [{"index": i, "scene": p["scene"], "source": p["source"]}
                   for i, p in enumerate(probes)],
        "probe_counts": probe_counts,
        "major_probe_counts": {major: major_probe[major] for major in sorted(major_probe)},
        "majors_used": sorted(major_probe),
        "rendered": rendered,
        "render_failures": failures,
        "render_seconds": round(elapsed, 2),
    }, ensure_ascii=False, indent=2))
    return 0 if not failures else 1


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    default_out = REPO_ROOT / "docs/assets/lut_cluster_pilot_20260819/questionnaire_b30"
    sub = parser.add_subparsers(dest="command", required=True)

    build = sub.add_parser("build", help="render the 30 A1f backfill pairs as a blind page")
    build.add_argument("--config", type=Path,
                       default=REPO_ROOT / "configs/agent_loop.terra-smoke.toml")
    build.add_argument("--candidates", type=Path, default=DEFAULT_CANDIDATES)
    build.add_argument("--out-dir", type=Path, default=default_out)
    build.add_argument("--probe-jsonl", type=Path, action="append", default=None,
                       help="repeatable; defaults to smoke5 then the scene-sample manifest")
    build.add_argument("--probe-fallback-dir", type=Path, default=DEFAULT_PROBE_FALLBACK_DIR)
    build.add_argument("--seed", type=int, default=DEFAULT_B30_SEED)
    build.add_argument("--probes", type=int, default=DEFAULT_PROBES)
    build.add_argument("--workers", type=int, default=8)
    build.add_argument("--probe-short-edge", type=int, default=512)
    build.add_argument("--side-edge", type=int, default=448)
    build.add_argument("--gap", type=int, default=8)
    build.add_argument("--csv-name", type=str, default="questionnaire_b30.csv")
    build.add_argument("--html-name", type=str, default="questionnaire_b30.html")
    build.add_argument("--key-name", type=str, default="pair_key_b30.json")
    build.add_argument("--skip-render", action="store_true",
                       help="emit csv/key/html only (determinism re-run)")
    build.set_defaults(func=cmd_build)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    return int(args.func(args))


if __name__ == "__main__":
    raise SystemExit(main())
