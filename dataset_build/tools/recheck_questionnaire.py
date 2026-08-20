"""Recheck questionnaire (B9): re-rate every item the user scored 5 in three rounds.

``build`` reads the three filled intent-quality CSVs together with their
``item_key.json`` / ``intentq.html`` siblings, keeps every row whose ``rating`` is 5,
re-numbers the survivors to ``rc_XXX`` in a seed-shuffled order, copies the existing
triptych thumbnails (no re-render) into one ``imgs/`` directory and writes the same
one-screen blind page as ``intent_quality_questionnaire`` (identical five rating
labels, identical keyboard handling) under an independent ``localStorage`` key.
The page shows images only: round, intent, preset, bin, strength and every other
parameter live in ``item_key.json``.

``analyze`` reads the filled CSV back and reports, per ``origin_round``, the recheck
rating distribution, the share that stayed at 5, and the 5 -> {1..5} flow.

Usage:
    python -m dataset_build.tools.recheck_questionnaire build
    python -m dataset_build.tools.recheck_questionnaire analyze \
        --csv docs/assets/lut_cluster_pilot_20260819/intent_quality_recheck/recheckq.csv
"""
from __future__ import annotations

import argparse
import html
import json
import re
import shutil
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Mapping, Sequence

from dataset_build.tools.export_agent_loop_review import (
    print_winner_confidence_warning,
    winner_confidence_counts,
)
from dataset_build.tools.intent_quality_questionnaire import (
    FIGURE_CAPTIONS,
    RATING_LABELS,
    RATING_MAX,
    RATING_MIN,
    WINNER_CONFIDENCE_FILTERED,
    _js_json,
    _sha1,
    _SCRIPT,
    _STYLE,
    read_ratings,
)

SCHEMA = "intent-quality-recheck-questionnaire-v1"
SOURCE_SCHEMA = "intent-quality-questionnaire-v1"
QUESTIONNAIRE_DIR = Path("docs/assets/questionnaire")
PILOT_DIR = Path("docs/assets/lut_cluster_pilot_20260819")
DEFAULT_OUT_DIR = PILOT_DIR / "intent_quality_recheck"
DEFAULT_SEED = "20260819"
TARGET_RATING = 5
CSV_NAME = "recheckq.csv"
PAGE_NAME = "recheckq.html"
CSV_HEADER = "item_id,rating,notes"

# (round label, filled CSV, source bundle directory)
DEFAULT_ROUNDS: tuple[tuple[str, Path, Path], ...] = (
    ("v1", QUESTIONNAIRE_DIR / "intentq.csv", PILOT_DIR / "intent_quality_200"),
    ("v2", QUESTIONNAIRE_DIR / "intentq (1).csv", PILOT_DIR / "intent_quality_v2"),
    ("v3", QUESTIONNAIRE_DIR / "intentq (2).csv", PILOT_DIR / "intent_quality_v3"),
)

# item_key fields carried over verbatim from the originating round.
CARRIED_FIELDS = (
    "branch_id",
    "global_branch_id",
    "source_id",
    "scene",
    "intent",
    "intent_variant",
    "mask_role",
    "subject_headroom",
    # C1b item 5: the originating round now records the committed leaf's
    # `winner_confidence`; a recheck bundle that drops it cannot be audited against the
    # `winner_confidence=low` discipline. Rounds built before C1b carry no such key and
    # land as `None`, which the counter reports as `unknown`.
    "winner_confidence",
    "global",
    "local",
)

# strings that must never reach the rendered page (blind-labelling leak scan).
LEAK_FIELDS = ("branch_id", "global_branch_id", "source_id", "scene", "intent",
               "intent_variant", "mask_role")
LEAK_NESTED = (("global", "preset"), ("global", "bin"), ("local", "preset"),
               ("local", "bin"), ("local", "mask_family"))


# --------------------------------------------------------------------------- read


def _images_from_page(path: Path) -> dict[str, list[str]]:
    """The per-item triptych paths only exist inside the generated page."""
    prefix = "const IMAGES = "
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.startswith(prefix):
            continue
        payload = line[len(prefix):].strip().rstrip(";")
        return {
            str(key): [str(value) for value in values]
            for key, values in json.loads(payload.replace("<\\/", "</")).items()
        }
    raise ValueError(f"no IMAGES literal in {path}")


def _read_round(name: str, csv_path: Path, bundle: Path) -> dict[str, Any]:
    key = json.loads((bundle / "item_key.json").read_text(encoding="utf-8"))
    if str(key.get("schema")) != SOURCE_SCHEMA:
        raise ValueError(f"{bundle}/item_key.json: unexpected schema {key.get('schema')}")
    items = key["items"]
    images = _images_from_page(bundle / "intentq.html")
    rows = read_ratings(csv_path)

    picked: list[dict[str, Any]] = []
    dropped: Counter[str] = Counter()
    for item_id, raw, notes in rows:
        if item_id not in items:
            dropped["unknown_item_id"] += 1
            continue
        if raw == "":
            dropped["blank_rating"] += 1
            continue
        try:
            value = int(raw)
        except ValueError:
            dropped["invalid_rating"] += 1
            continue
        if value != TARGET_RATING:
            dropped[f"rating_{value}"] += 1
            continue
        if item_id not in images:
            dropped["image_missing"] += 1
            continue
        picked.append({
            "origin_round": name,
            "origin_campaign": str(key.get("campaign") or ""),
            "origin_item_id": item_id,
            "origin_rating": value,
            "origin_notes": notes,
            "origin_images": images[item_id],
            "entry": items[item_id],
        })
    return {
        "round": name,
        "csv": str(csv_path),
        "bundle": str(bundle),
        "campaign": str(key.get("campaign") or ""),
        "csv_rows": len(rows),
        "bundle_items": len(items),
        "rating_5": len(picked),
        "dropped": dict(sorted(dropped.items())),
        "picked": picked,
    }


# ------------------------------------------------------------------------- page


def _script(order: Sequence[str], images: Mapping[str, list[str]],
            store_key: str) -> str:
    return (
        _SCRIPT
        .replace("__ITEMS__", _js_json(list(order)))
        .replace("__IMAGES__", _js_json(dict(images)))
        .replace("__STORE_KEY__", _js_json(store_key))
        .replace("__HEADER__", _js_json(CSV_HEADER))
        .replace("__RMIN__", str(RATING_MIN))
        .replace("__RMAX__", str(RATING_MAX))
        .replace("'intentq.csv'", f"'{CSV_NAME}'")
    )


def write_page(out_dir: Path, order: Sequence[str], images: Mapping[str, list[str]],
               store_key: str) -> Path:
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
        "<title>整链质量复核盲标问卷</title>",
        f"<style>{_STYLE}</style></head><body>",
        "<h1>整链质量复核盲标问卷</h1>",
        '<p class="hint">左边是原图，中间是第一步处理后，右边是最终结果。'
        "只对「最终结果相对原图」这条链整体打分；评分自动保存在本机浏览器，"
        f"标完点「导出 CSV」下载 {CSV_NAME}。</p>",
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
        f"<script>{_script(order, images, store_key)}</script>",
        "</body></html>",
    ]
    path = out_dir / PAGE_NAME
    path.write_text("\n".join(parts) + "\n", encoding="utf-8", newline="\n")
    return path


def write_csv(out_dir: Path, order: Sequence[str]) -> Path:
    lines = [CSV_HEADER]
    lines.extend(f"{item_id},," for item_id in order)
    path = out_dir / CSV_NAME
    path.write_text("\n".join(lines) + "\n", encoding="utf-8", newline="\n")
    return path


def _scannable(page: Path) -> str:
    """Page text minus the two item-independent constants: the CSS block and the
    localStorage key line. Both are identical for every item, so a substring hit in
    them (`background:#fff`, the `...-v1:` schema suffix) carries no item information;
    everything else on the page is in scope."""
    text = page.read_text(encoding="utf-8")
    head, _, rest = text.partition("<style>")
    _, _, tail = rest.partition("</style>")
    body = head + tail
    return "\n".join(line for line in body.splitlines()
                     if not line.startswith("const STORE_KEY = "))


def leak_scan(page: Path, items: Mapping[str, Any]) -> list[str]:
    """Every parameter string of every item must be absent from the page, matched as a
    whole token so that a hex thumbnail name such as `c008461d...` is not a hit."""
    text = _scannable(page)
    needles: set[str] = set()
    for entry in items.values():
        needles.add(str(entry["origin_round"]))
        needles.add(str(entry["origin_campaign"]))
        needles.add(str(entry["origin_item_id"]))
        for field in LEAK_FIELDS:
            value = entry.get(field)
            if isinstance(value, str) and value:
                needles.add(value)
        for outer, inner in LEAK_NESTED:
            value = (entry.get(outer) or {}).get(inner)
            if isinstance(value, str) and value:
                needles.add(value)
    return sorted(
        needle for needle in needles
        if needle and re.search(rf"(?<![0-9A-Za-z]){re.escape(needle)}(?![0-9A-Za-z])",
                                text)
    )


# ----------------------------------------------------------------------- commands


def _parse_round(spec: str) -> tuple[str, Path, Path]:
    parts = spec.split("=")
    if len(parts) != 3:
        raise argparse.ArgumentTypeError(f"expected name=csv=bundle_dir, got {spec!r}")
    return parts[0], Path(parts[1]), Path(parts[2])


def cmd_build(args: argparse.Namespace) -> int:
    rounds = tuple(args.round) if args.round else DEFAULT_ROUNDS
    out_dir = Path(args.out_dir)
    imgs_dir = out_dir / "imgs"
    imgs_dir.mkdir(parents=True, exist_ok=True)

    summaries = [_read_round(name, csv_path, bundle) for name, csv_path, bundle in rounds]
    picked = [row for summary in summaries for row in summary["picked"]]
    # seed shuffle: deterministic, and independent of the per-round display order.
    picked.sort(key=lambda row: (
        _sha1(f"{args.seed}|{row['origin_round']}|{row['origin_item_id']}"),
        row["origin_round"], row["origin_item_id"]))

    bundles = {summary["round"]: Path(summary["bundle"]) for summary in summaries}
    order: list[str] = []
    images: dict[str, list[str]] = {}
    items: dict[str, Any] = {}
    copied: set[str] = set()
    missing: list[str] = []
    for index, row in enumerate(picked, start=1):
        item_id = f"rc_{index:03d}"
        source_dir = bundles[row["origin_round"]]
        trio: list[str] = []
        gap = False
        for rel in row["origin_images"]:
            src = source_dir / rel
            if not src.is_file():
                missing.append(f"{item_id}:{row['origin_round']}/{rel}")
                gap = True
                continue
            name = Path(rel).name
            dst = imgs_dir / name
            if name not in copied and not dst.is_file():
                shutil.copy2(src, dst)
            copied.add(name)
            trio.append("imgs/" + name)
        if gap:
            continue
        order.append(item_id)
        images[item_id] = trio
        entry = row["entry"]
        items[item_id] = {
            "origin_round": row["origin_round"],
            "origin_campaign": row["origin_campaign"],
            "origin_item_id": row["origin_item_id"],
            "origin_rating": row["origin_rating"],
            "origin_notes": row["origin_notes"],
            "origin_images": list(row["origin_images"]),
            **{field: entry.get(field) for field in CARRIED_FIELDS},
        }

    store_key = f"{SCHEMA}:{args.seed}"
    sampled_confidence = winner_confidence_counts(
        entry.get("winner_confidence") for entry in items.values()
    )

    payload = {
        "schema": SCHEMA,
        "seed": args.seed,
        "target_rating": TARGET_RATING,
        "store_key": store_key,
        "rounds": [
            {key: summary[key] for key in
             ("round", "csv", "bundle", "campaign", "csv_rows", "bundle_items",
              "rating_5", "dropped")}
            for summary in summaries
        ],
        "rating_5_total": sum(summary["rating_5"] for summary in summaries),
        "items_by_round": dict(sorted(
            Counter(entry["origin_round"] for entry in items.values()).items())),
        "image_missing": missing,
        "winner_confidence_filtered": WINNER_CONFIDENCE_FILTERED,
        "winner_confidence_counts": sampled_confidence,
        "display_order": order,
        "images": images,
        "items": items,
    }
    # C1b item 10: `item_key.json` (every carried parameter of every recheck item) and
    # the rating CSV land before the page render, the leak scan and the optional image
    # prune, so a failure in any of those cannot destroy a finished selection. The leak
    # scan result is folded into the key file afterwards, and is re-runnable from disk.
    key_path = out_dir / "item_key.json"

    def write_key() -> None:
        key_path.write_text(
            json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
            encoding="utf-8", newline="\n")

    write_key()
    csv_path = write_csv(out_dir, order)
    page = write_page(out_dir, order, images, store_key)
    leaks = leak_scan(page, items)
    payload["leak_hits"] = leaks
    write_key()

    stray = sorted(
        path.name for path in imgs_dir.iterdir()
        if path.is_file() and path.name not in copied)
    if args.prune:
        for name in stray:
            (imgs_dir / name).unlink()

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
        "rating_5_total": payload["rating_5_total"],
        "items_by_round": payload["items_by_round"],
        "rating_5_by_round": {summary["round"]: summary["rating_5"]
                              for summary in summaries},
        "images_copied": len(copied),
        "image_missing": missing,
        "images_stray": stray if not args.prune else [],
        "images_pruned": stray if args.prune else [],
        "leak_hits": leaks,
    }
    print(json.dumps(stats, ensure_ascii=False, sort_keys=True, indent=2))
    return 0


def _dist(values: Sequence[int]) -> dict[str, Any]:
    total = len(values)
    counts = Counter(values)
    row: dict[str, Any] = {
        "n": total,
        "mean_rating": round(sum(values) / total, 4) if total else None,
        "hold_5_rate": round(counts[5] / total, 4) if total else None,
    }
    for value in range(RATING_MIN, RATING_MAX + 1):
        row[f"n_{value}"] = counts[value]
        row[f"rate_{value}"] = round(counts[value] / total, 4) if total else None
    return row


def cmd_analyze(args: argparse.Namespace) -> int:
    key_path = Path(args.item_key or (Path(args.csv).parent / "item_key.json"))
    key = json.loads(key_path.read_text(encoding="utf-8"))
    items: dict[str, Any] = key["items"]
    rows = read_ratings(Path(args.csv))

    ratings: dict[str, int] = {}
    unknown: list[str] = []
    invalid: list[str] = []
    for item_id, raw, _notes in rows:
        if item_id not in items:
            unknown.append(item_id)
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

    by_round: dict[str, list[int]] = defaultdict(list)
    confusion: dict[str, Counter[int]] = defaultdict(Counter)
    confusion_by_round: dict[str, dict[str, Counter[int]]] = defaultdict(
        lambda: defaultdict(Counter))
    for item_id, value in sorted(ratings.items()):
        entry = items[item_id]
        origin_round = str(entry.get("origin_round"))
        origin = int(entry.get("origin_rating"))
        by_round[origin_round].append(value)
        confusion[str(origin)][value] += 1
        confusion_by_round[origin_round][str(origin)][value] += 1

    def matrix(table: Mapping[str, Counter[int]]) -> list[dict[str, Any]]:
        return [
            {
                "origin_rating": int(origin),
                "n": sum(table[origin].values()),
                **{f"to_{value}": table[origin][value]
                   for value in range(RATING_MIN, RATING_MAX + 1)},
                **{f"to_{value}_rate": (
                    round(table[origin][value] / sum(table[origin].values()), 4)
                    if sum(table[origin].values()) else None)
                   for value in range(RATING_MIN, RATING_MAX + 1)},
            }
            for origin in sorted(table)
        ]

    rated_confidence = winner_confidence_counts(
        items[item_id].get("winner_confidence") for item_id in sorted(ratings)
    )
    confidence_warning = print_winner_confidence_warning(
        rated_confidence, filtered=WINNER_CONFIDENCE_FILTERED
    )

    analysis = {
        "schema": SCHEMA,
        "csv": str(args.csv),
        "item_key": str(key_path),
        "seed": key.get("seed"),
        "n_items": len(items),
        "n_rated": len(ratings),
        "n_missing_rating": len(missing),
        "missing_rating": missing,
        "n_invalid_rating": len(invalid),
        "invalid_rating": invalid,
        "n_unknown_item_id": len(unknown),
        "unknown_item_id": sorted(unknown),
        "overall": _dist(sorted(ratings.values())),
        "rounds": [dict(origin_round=name, **_dist(by_round[name]))
                   for name in sorted(by_round)],
        "confusion_overall": matrix(confusion),
        "confusion_by_round": {name: matrix(confusion_by_round[name])
                               for name in sorted(confusion_by_round)},
        # C1b item 5: the rated population's confidence mix travels with the analysis.
        "winner_confidence_filtered": WINNER_CONFIDENCE_FILTERED,
        "winner_confidence_counts": rated_confidence,
        "winner_confidence_warning": confidence_warning,
    }
    out_path = Path(args.out) if args.out else Path(args.csv).parent / "analysis.json"
    out_path.write_text(
        json.dumps(analysis, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8", newline="\n")

    head = (f"{'origin_round':<16}{'n':>6}{'mean':>8}{'hold_5':>9}"
            + "".join(f"{'=' + str(value):>7}" for value in
                      range(RATING_MIN, RATING_MAX + 1)))
    lines = [
        *([confidence_warning, ""] if confidence_warning else []),
        f"n_items = {len(items)}  n_rated = {len(ratings)}  "
        f"n_missing = {len(missing)}  n_invalid = {len(invalid)}  "
        f"n_unknown_item_id = {len(unknown)}",
        "",
        "recheck rating distribution by origin_round",
        head,
    ]
    for row in [analysis["overall"] | {"origin_round": "ALL"}, *analysis["rounds"]]:
        lines.append(
            f"{row['origin_round']:<16}{row['n']:>6}{str(row['mean_rating']):>8}"
            f"{str(row['hold_5_rate']):>9}"
            + "".join(f"{row['n_' + str(value)]:>7}"
                      for value in range(RATING_MIN, RATING_MAX + 1)))
    lines.extend(["", "confusion (origin rating -> recheck rating), counts",
                  f"{'scope':<16}{'origin':>7}{'n':>6}"
                  + "".join(f"{'->' + str(value):>7}"
                            for value in range(RATING_MIN, RATING_MAX + 1))])
    scopes = [("ALL", analysis["confusion_overall"])]
    scopes.extend(sorted(analysis["confusion_by_round"].items()))
    for scope, table in scopes:
        for row in table:
            lines.append(
                f"{scope:<16}{row['origin_rating']:>7}{row['n']:>6}"
                + "".join(f"{row['to_' + str(value)]:>7}"
                          for value in range(RATING_MIN, RATING_MAX + 1)))
    lines.extend(["", f"written {out_path}"])
    print("\n".join(lines))
    return 0


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)

    build = sub.add_parser(
        "build", help="collect every rating=5 item from the three rounds")
    build.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    build.add_argument("--round", type=_parse_round, action="append", default=[],
                       help="repeatable name=csv=bundle_dir; default = v1/v2/v3")
    build.add_argument("--seed", default=DEFAULT_SEED,
                       help="display-order shuffle seed (string)")
    build.add_argument("--prune", action="store_true",
                       help="delete imgs/ files no longer referenced")
    build.set_defaults(func=cmd_build)

    analyze = sub.add_parser("analyze", help="read the filled recheck CSV back")
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
