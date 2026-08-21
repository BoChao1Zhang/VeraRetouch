"""E3: read-only match check between the frozen source histogram and the v2 LUT table.

This is a *test* tool. It reads three inputs and writes exactly one markdown file under
``docs/assets/diagnose_v2_test_20260821/``. It touches no prompt revision, no agent-loop
artifact store, no audit store, no revision record, and no LUT catalog; it imports
``dataset_build.agent_loop`` only for the frozen read-only functions
``source_histogram`` / ``histogram_match_bonus`` /
``load_segment_histograms`` and mutates none of their registered constants.

Per source it reports, as numbers only:

1. the frozen-contract ``clip_low`` / ``clip_high`` (4096 deterministic samples) next to
   the full-resolution ``clip_low`` / ``clip_high`` carried in the E1 harness ``stats``;
2. whether each of the two pre-registered gates fires
   (``clip_low >= 0.02`` / ``clip_high >= 0.02``);
3. over all rows of the v2 fingerprint table: how many carry ``bonus > 0``, plus the
   max and p50 of the bonus;
4. the 5 highest-bonus rows with ``preset_id, bonus, d_shadow, d_mid, d_high``;
5. the verbatim entries of the five list fields of the v2 diagnosis that contain any of
   shadow / black / crush / blocked / highlight / clip / blown (case-insensitive).

Determinism: ``source_histogram`` is seed-free, the table is read in file order, and
ties in the top-5 break on ``preset_id``. Two runs produce byte-identical output.

Usage::

    .venv/bin/python -m dataset_build.tools.test_diagnose_e3_retrieval_match
"""
from __future__ import annotations

import argparse
import json
import math
import re
from pathlib import Path
from typing import Any, Mapping, Sequence

from dataset_build.agent_loop.segment_fingerprints import load_segment_histograms
from dataset_build.agent_loop.source_histogram import (
    HISTOGRAM_MATCH_GATE,
    HISTOGRAM_SAMPLE_PIXELS,
    histogram_match_bonus,
    source_histogram,
)

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_RECORDS = REPO_ROOT / "docs/assets/diagnose_v2_test_20260821/records.jsonl"
DEFAULT_FINGERPRINTS = Path(
    "/home/bc/data/scratch/lut_reannotate/out/segment_fingerprints.v2.jsonl"
)
DEFAULT_OUT = REPO_ROOT / "docs/assets/diagnose_v2_test_20260821/e3_retrieval_match.md"

RUN_COMMAND = (
    ".venv/bin/python -m dataset_build.tools.test_diagnose_e3_retrieval_match"
)

# The five list fields of the frozen diagnosis schema.
DIAGNOSIS_LIST_FIELDS: tuple[str, ...] = (
    "correction_needs",
    "preserve_intent",
    "enhancement_opportunities",
    "forbidden_directions",
    "evidence",
)

KEYWORDS: tuple[str, ...] = (
    "shadow", "black", "crush", "blocked", "highlight", "clip", "blown",
)
KEYWORD_PATTERN = re.compile("|".join(KEYWORDS), re.IGNORECASE)

TOP_K = 5


# ------------------------------------------------------------------ helpers
def p50(values: Sequence[float]) -> float:
    """Linear-interpolated median, same convention as `segment_fingerprints._quantile`."""
    if not values:
        return float("nan")
    ordered = sorted(values)
    position = (len(ordered) - 1) * 0.5
    low = math.floor(position)
    high = math.ceil(position)
    if low == high:
        return float(ordered[low])
    weight = position - low
    return float(ordered[low] * (1.0 - weight) + ordered[high] * weight)


def read_records(path: Path) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                records.append(json.loads(line))
    return records


def keyword_hits(parsed: Mapping[str, Any]) -> list[tuple[str, str]]:
    """`(field, entry)` for every list entry matching one of `KEYWORDS`."""
    hits: list[tuple[str, str]] = []
    for field in DIAGNOSIS_LIST_FIELDS:
        for entry in parsed.get(field) or []:
            text = str(entry)
            if KEYWORD_PATTERN.search(text):
                hits.append((field, text))
    return hits


def score_table(
    reading: Mapping[str, Any], table: Mapping[str, Mapping[str, Any]]
) -> list[tuple[str, float, Mapping[str, Any]]]:
    """`(preset_id, bonus, response)` in table file order."""
    return [
        (preset_id, histogram_match_bonus(reading, response), response)
        for preset_id, response in table.items()
    ]


def _f(value: Any, digits: int = 6) -> str:
    return f"{float(value):.{digits}f}"


# ------------------------------------------------------------------ report
def build_report(
    records: Sequence[Mapping[str, Any]],
    table: Mapping[str, Mapping[str, Any]],
    records_path: Path,
    fingerprints_path: Path,
) -> str:
    gate_low = float(HISTOGRAM_MATCH_GATE["clip_low_min"])
    gate_high = float(HISTOGRAM_MATCH_GATE["clip_high_min"])

    lines: list[str] = []
    lines.append("# E3 — B12 histogram retrieval vs. v2 diagnosis, per source")
    lines.append("")
    lines.append("```")
    lines.append(RUN_COMMAND)
    lines.append("```")
    lines.append("")
    lines.append(f"- records: `{records_path}` (n={len(records)})")
    lines.append(f"- fingerprints: `{fingerprints_path}` (rows={len(table)})")
    lines.append(
        f"- `source_histogram` sample budget: {HISTOGRAM_SAMPLE_PIXELS} pixels; "
        "harness `stats` clip is over all pixels of the original file"
    )
    lines.append(
        "- gate: `clip_low >= "
        f"{gate_low}` / `clip_high >= {gate_high}`; "
        "`delta_scale="
        f"{HISTOGRAM_MATCH_GATE['delta_scale']}`, "
        f"`shadow_weight={HISTOGRAM_MATCH_GATE['shadow_weight']}`, "
        f"`highlight_weight={HISTOGRAM_MATCH_GATE['highlight_weight']}`, "
        f"`shadow_sign={HISTOGRAM_MATCH_GATE['shadow_sign']}`, "
        f"`highlight_sign={HISTOGRAM_MATCH_GATE['highlight_sign']}`"
    )
    lines.append("")

    summary_rows: list[str] = []

    for record in records:
        source_id = str(record["source_id"])
        source_path = str(record["source_path"])
        reading = source_histogram(source_path)
        stats = record.get("stats") or {}
        clip_low = float(reading["clip_low"])
        clip_high = float(reading["clip_high"])
        fired_low = clip_low >= gate_low
        fired_high = clip_high >= gate_high

        scored = score_table(reading, table)
        bonuses = [bonus for _, bonus, _ in scored]
        nonzero = [bonus for bonus in bonuses if bonus > 0.0]
        top = sorted(scored, key=lambda item: (-item[1], item[0]))[:TOP_K]

        lines.append(f"## {source_id}")
        lines.append("")
        lines.append(f"- `source_path`: `{source_path}`")
        lines.append("")
        lines.append("| quantity | frozen contract (4096 samples) | harness stats (all pixels) |")
        lines.append("| --- | --- | --- |")
        lines.append(
            f"| clip_low | {_f(clip_low)} | {_f(stats.get('clip_low', float('nan')))} |"
        )
        lines.append(
            f"| clip_high | {_f(clip_high)} | {_f(stats.get('clip_high', float('nan')))} |"
        )
        lines.append("")
        lines.append(
            f"- gate clip_low >= {gate_low}: **{'fired' if fired_low else 'not fired'}**"
            f"; gate clip_high >= {gate_high}: "
            f"**{'fired' if fired_high else 'not fired'}**"
        )
        lines.append(
            f"- rows with bonus > 0: {len(nonzero)} / {len(bonuses)}"
            f"; bonus max = {_f(max(bonuses))}; bonus p50 = {_f(p50(bonuses))}"
        )
        lines.append("")
        lines.append("| rank | preset_id | bonus | d_shadow | d_mid | d_high |")
        lines.append("| --- | --- | --- | --- | --- | --- |")
        for rank, (preset_id, bonus, response) in enumerate(top, 1):
            lines.append(
                f"| {rank} | `{preset_id}` | {_f(bonus)} "
                f"| {_f(response.get('d_shadow', float('nan')))} "
                f"| {_f(response.get('d_mid', float('nan')))} "
                f"| {_f(response.get('d_high', float('nan')))} |"
            )
        lines.append("")

        hits = keyword_hits(record.get("v2", {}).get("parsed", {}) or {})
        lines.append(f"v2 diagnosis entries matching {list(KEYWORDS)} — {len(hits)} entry(ies):")
        lines.append("")
        if hits:
            for field, text in hits:
                lines.append(f"- `{field}`: {text}")
        else:
            lines.append("- (none)")
        lines.append("")

        gate_cell = (
            f"low={'Y' if fired_low else 'N'} high={'Y' if fired_high else 'N'}"
        )
        summary_rows.append(
            f"| `{source_id}` | {_f(clip_low)} | {_f(clip_high)} | {gate_cell} "
            f"| {len(nonzero)} |"
        )

    lines.append("## Summary")
    lines.append("")
    lines.append("| source_id | clip_low (4096) | clip_high (4096) | gates fired | rows with bonus > 0 |")
    lines.append("| --- | --- | --- | --- | --- |")
    lines.extend(summary_rows)
    lines.append("")
    return "\n".join(lines)


# ------------------------------------------------------------------ assertion
def assert_ungated_source_scores_zero(
    records: Sequence[Mapping[str, Any]], table: Mapping[str, Mapping[str, Any]]
) -> str:
    """Pre-registered runtime assertion: a source firing neither gate scores 0 everywhere.

    Returns the `source_id` the assertion ran on. Raises if no such source exists, so a
    silently-skipped assertion cannot happen.
    """
    gate_low = float(HISTOGRAM_MATCH_GATE["clip_low_min"])
    gate_high = float(HISTOGRAM_MATCH_GATE["clip_high_min"])
    for record in records:
        reading = source_histogram(str(record["source_path"]))
        if float(reading["clip_low"]) >= gate_low:
            continue
        if float(reading["clip_high"]) >= gate_high:
            continue
        for preset_id, response in table.items():
            bonus = histogram_match_bonus(reading, response)
            if bonus != 0.0:
                raise AssertionError(
                    f"{record['source_id']} fires no gate but {preset_id} scores {bonus}"
                )
        return str(record["source_id"])
    raise AssertionError("no source in the record set fires neither gate")


# ------------------------------------------------------------------ entry point
def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--records", type=Path, default=DEFAULT_RECORDS)
    parser.add_argument("--fingerprints", type=Path, default=DEFAULT_FINGERPRINTS)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    args = parser.parse_args(argv)

    records = read_records(args.records)
    table = load_segment_histograms(args.fingerprints)

    checked = assert_ungated_source_scores_zero(records, table)
    print(f"ungated-source zero-bonus assertion ran on: {checked}")

    report = build_report(records, table, args.records, args.fingerprints)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(report, encoding="utf-8")
    print(f"wrote {args.out} ({len(report.encode('utf-8'))} bytes)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
