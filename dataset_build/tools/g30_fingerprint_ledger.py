"""G2 step 2: the LUT fingerprint "nominal vs measured" ledger of one campaign.

Numbers only. For every accepted **global** render of the campaign it puts side by
side (a) the fingerprint columns of the verbatim shortlist row the model picked from
and (b) a direct measurement of the same axes on the (source render, global render)
image pair, then counts per-column sign agreement.

Nominal side
------------
Parsed straight out of `api_request.canonical_request_json` (stage `global_propose`),
which stores the exact prompt bytes the model saw. The row layout is
`prompts.shortlist_row_text`::

    <row_index> | <achievable_bins> | <8 fingerprint numbers> | <caption> [| d_shadow d_mid d_high]

with the eight numbers in `models.FINGERPRINT_FIELDS` order.

The nominal cast is polar (`cast_hue` degrees, `cast_mag` Lab chroma), so it is put on
the measured Cartesian axes by the identity
`cast_a = cast_mag * cos(cast_hue)`, `cast_b = cast_mag * sin(cast_hue)`.

Measured side
-------------
* five axes from `direction_match.measure_direction(before, after)` -
  `cast_a`, `cast_b`, `lightness`, `saturation`, `contrast`;
* `d_shadow` / `d_mid` / `d_high` from `source_histogram.l_bin_shares` on both images,
  differenced bin by bin and summed over `segment_fingerprints.HISTOGRAM_GROUPS`
  (`delta = out - in`, exactly the derivation of the mounted v2 fingerprint).

`before` is the render's `input_json.input_image_sha256` blob, `after` is its
`artifact_json.sha256` blob, both read from the campaign artifact root.

Caliber note (not a judgement, a fact about the two sides): the nominal fingerprint is
a full-strength reading taken on the LUT's own frozen probe pixel set, while the render
applies `parameters_json.global_strength` to this particular image; the per-proposal
strength is therefore carried as its own column.

Pre-registered near-zero thresholds
-----------------------------------
`NEAR_ZERO` below. A (nominal, measured) pair is excluded from a column's sign
agreement rate when **either** side is below its threshold in absolute value. The
`contrast` column compares `nominal - 1.0` (the header calls 1.000 unchanged) against
the measured highlight-minus-shadow dL, so it carries two thresholds.

Usage::

    .venv/bin/python -m dataset_build.tools.g30_fingerprint_ledger \\
        --campaign g30 --root /mnt/ramstage/agent_loop/g30 \\
        --out docs/assets/g30_fingerprint_check_20260824/ledger.json
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any

import numpy as np
import psycopg
from skimage.color import rgb2lab

from dataset_build.agent_loop.direction_match import measure_direction
from dataset_build.agent_loop.models import FINGERPRINT_FIELDS
from dataset_build.agent_loop.segment_fingerprints import (
    HISTOGRAM_AGGREGATES, HISTOGRAM_GROUPS,
)
from dataset_build.agent_loop.source_histogram import l_bin_shares, sample_pixels

DSN = (
    "postgresql://research:research@127.0.0.1:5432/agent_loop"
    "?options=-c%20search_path%3Dagent_loop"
)

# --- pre-registered constants -------------------------------------------------------
# Absolute value below which a reading counts as near zero and its pair leaves the
# sign-agreement denominator. Units: L* for the lightness axes, percent for saturation,
# Lab chroma for the cast axes, pixel share for the histogram aggregates.
NEAR_ZERO: dict[str, float] = {
    "dL": 0.5, "lightness": 0.5,
    "dSat": 0.5, "saturation": 0.5,
    "cast_a": 0.5, "cast_b": 0.5,
    "contrast_nominal": 0.01,   # applied to |contrast - 1.0|
    "contrast_measured": 0.5,
    "d_shadow": 0.005, "d_mid": 0.005, "d_high": 0.005,
}
# (column label, nominal key, measured key, nominal threshold, measured threshold)
SIGN_COLUMNS: tuple[tuple[str, str, str, float, float], ...] = (
    ("dL / lightness", "dL", "lightness", NEAR_ZERO["dL"], NEAR_ZERO["lightness"]),
    ("dSat / saturation", "dSat", "saturation",
     NEAR_ZERO["dSat"], NEAR_ZERO["saturation"]),
    ("cast_a", "cast_a", "cast_a", NEAR_ZERO["cast_a"], NEAR_ZERO["cast_a"]),
    ("cast_b", "cast_b", "cast_b", NEAR_ZERO["cast_b"], NEAR_ZERO["cast_b"]),
    ("contrast", "contrast_offset", "contrast",
     NEAR_ZERO["contrast_nominal"], NEAR_ZERO["contrast_measured"]),
    ("d_shadow", "d_shadow", "d_shadow", NEAR_ZERO["d_shadow"], NEAR_ZERO["d_shadow"]),
    ("d_mid", "d_mid", "d_mid", NEAR_ZERO["d_mid"], NEAR_ZERO["d_mid"]),
    ("d_high", "d_high", "d_high", NEAR_ZERO["d_high"], NEAR_ZERO["d_high"]),
)


# --- shortlist row parsing ----------------------------------------------------------
def shortlist_rows(request_json: str) -> dict[str, dict[int, str]]:
    """`{major: {row_index: verbatim row line}}` from a stored canonical request."""
    payload = json.loads(request_json)
    text = ""
    for message in payload.get("input", []):
        for content in message.get("content", []):
            body = content.get("text") or ""
            if body.startswith("LUT shortlist table."):
                text = body
    out: dict[str, dict[int, str]] = {}
    major: str | None = None
    for line in text.splitlines():
        if line.startswith("[major] "):
            major = line[len("[major] "):].strip()
            out[major] = {}
        elif major is not None and " | " in line:
            head = line.split(" | ", 1)[0].strip()
            if head.isdigit():
                out[major][int(head)] = line
    return out


def shortlist_header(request_json: str) -> str:
    """The verbatim shortlist header block (everything before the first `[major]`)."""
    payload = json.loads(request_json)
    for message in payload.get("input", []):
        for content in message.get("content", []):
            body = content.get("text") or ""
            if body.startswith("LUT shortlist table."):
                return body.split("\n[major] ", 1)[0]
    return ""


def parse_row(line: str) -> dict[str, Any]:
    """The fingerprint numbers of one verbatim global shortlist row."""
    parts = [chunk.strip() for chunk in line.split(" | ")]
    if len(parts) < 4:
        raise ValueError(f"shortlist row has too few columns: {line!r}")
    numbers = parts[2].split()
    if len(numbers) != len(FINGERPRINT_FIELDS):
        raise ValueError(f"shortlist row carries {len(numbers)} fingerprint numbers")
    row: dict[str, Any] = {
        "row_index": int(parts[0]),
        "achievable_bins": parts[1],
        "row_text": line,
    }
    for name, value in zip(FINGERPRINT_FIELDS, numbers):
        row[name] = float(value)
    # The trailing histogram triple, when the catalog mounted a v2 fingerprint table.
    tail = parts[-1].split()
    if len(tail) == len(HISTOGRAM_AGGREGATES) and all(
        _is_number(value) for value in tail
    ):
        for name, value in zip(HISTOGRAM_AGGREGATES, tail):
            row[name] = float(value)
        row["caption"] = " | ".join(parts[3:-1])
    else:
        row["caption"] = " | ".join(parts[3:])
    # Polar cast -> the Cartesian axes `measure_direction` reports.
    angle = math.radians(row["cast_hue"])
    row["cast_a"] = row["cast_mag"] * math.cos(angle)
    row["cast_b"] = row["cast_mag"] * math.sin(angle)
    row["contrast_offset"] = row["contrast"] - 1.0
    return row


def _is_number(text: str) -> bool:
    try:
        float(text)
    except ValueError:
        return False
    return True


# --- measurement --------------------------------------------------------------------
def blob(root: Path, sha: str) -> Path:
    return root / "blobs" / sha[:2] / sha[2:4] / sha


def histogram_delta(before: Path, after: Path) -> dict[str, float]:
    """`d_shadow` / `d_mid` / `d_high` of the two images, `out - in` bin shares."""
    shares = []
    for path in (before, after):
        pixels = sample_pixels(path)
        lab = np.asarray(
            rgb2lab(pixels.reshape(-1, 1, 3)), dtype=np.float64
        ).reshape(-1, 3)
        shares.append(l_bin_shares(lab[:, 0]))
    delta = [after_bin - before_bin for before_bin, after_bin in zip(*shares)]
    return {
        name: float(sum(delta[index] for index in HISTOGRAM_GROUPS[name]))
        for name in HISTOGRAM_AGGREGATES
    }


# --- database -----------------------------------------------------------------------
def fetch(conn, campaign: str) -> dict[str, Any]:
    runs = [
        {"source_id": source_id, "source_sha256": sha, "status": status,
         "counts": json.loads(counts or "{}")}
        for source_id, sha, status, counts in conn.execute(
            "select source_id,source_sha256,status,counts_json from agent_source_run "
            "where campaign_id=%s order by source_id", (campaign,)
        ).fetchall()
    ]
    branches = {
        branch_id: {"branch_id": branch_id, "source_sha256": sha, "status": status,
                    "proposal": json.loads(proposal)}
        for branch_id, sha, status, proposal in conn.execute(
            "select branch_id,source_sha256,status,proposal_json from agent_branch "
            "where campaign_id=%s and level='global' order by branch_id", (campaign,)
        ).fetchall()
    }
    renders = [
        {"branch_id": branch_id, "input": json.loads(input_json),
         "params": json.loads(params or "{}"), "metrics": json.loads(metrics or "{}"),
         "artifact": json.loads(artifact or "{}")}
        for branch_id, input_json, params, metrics, artifact in conn.execute(
            "select r.branch_id,r.input_json,r.parameters_json,r.metrics_json,"
            "r.artifact_json from render_record r "
            "join agent_branch b on b.branch_id=r.branch_id "
            "where b.campaign_id=%s and r.stage='global' and r.status='accepted' "
            "order by r.branch_id", (campaign,)
        ).fetchall()
    ]
    requests = {}
    for sha, response, request in conn.execute(
        "select x.source_sha256,q.response_json,q.canonical_request_json "
        "from api_request_context x join api_request q on q.request_hash=x.request_hash "
        "where x.campaign_id=%s and x.stage='global_propose'", (campaign,)
    ).fetchall():
        parsed = (json.loads(response) or {}).get("parsed") or {}
        if parsed.get("major"):
            requests[sha] = {
                "major": parsed["major"], "rows": shortlist_rows(request),
                "header": shortlist_header(request),
            }
    return {"runs": runs, "branches": branches, "renders": renders,
            "requests": requests}


# --- ledger -------------------------------------------------------------------------
def build(pairs: list[tuple[str, Path]], dsn: str) -> dict[str, Any]:
    """One ledger over `[(campaign, artifact_root), ...]`, later pairs are backfills.

    A source that terminated `error` in an earlier campaign and was re-run in a later
    one contributes its later status to `final_status`; the per-campaign counts stay
    separate.
    """
    entries: list[dict[str, Any]] = []
    skipped: list[dict[str, Any]] = []
    runs_by_campaign: dict[str, list[dict[str, Any]]] = {}
    final_status: dict[str, str] = {}
    for campaign, root in pairs:
        part = _build_one(campaign, root, dsn)
        entries.extend(part["entries"])
        skipped.extend(part["skipped"])
        runs_by_campaign[campaign] = part["runs"]
        for row in part["runs"]:
            final_status[row["source_id"]] = row["status"]
    entries.sort(key=lambda row: (row["source_id"], row["row_index"]))
    counts: dict[str, int] = {}
    for status in final_status.values():
        counts[status] = counts.get(status, 0) + 1
    return {
        "campaigns": [campaign for campaign, _root in pairs],
        "artifact_roots": {campaign: str(root) for campaign, root in pairs},
        "near_zero": NEAR_ZERO,
        "runs_by_campaign": runs_by_campaign,
        "final_status": final_status,
        "final_counts": counts,
        "entries": entries, "skipped": skipped,
        "sign_agreement": sign_agreement(entries),
    }


def _build_one(campaign: str, root: Path, dsn: str) -> dict[str, Any]:
    with psycopg.connect(dsn) as conn:
        data = fetch(conn, campaign)
    source_id_by_sha = {row["source_sha256"]: row["source_id"] for row in data["runs"]}
    entries: list[dict[str, Any]] = []
    skipped: list[dict[str, Any]] = []
    for render in data["renders"]:
        branch = data["branches"].get(render["branch_id"])
        if branch is None:
            skipped.append({"branch_id": render["branch_id"], "why": "branch_missing"})
            continue
        sha = branch["source_sha256"]
        request = data["requests"].get(sha)
        if request is None:
            skipped.append({"branch_id": branch["branch_id"], "why": "request_missing"})
            continue
        proposal = branch["proposal"]
        row_index = int(proposal["row_index"])
        line = request["rows"].get(request["major"], {}).get(row_index)
        if line is None:
            skipped.append({"branch_id": branch["branch_id"], "why": "row_missing"})
            continue
        nominal = parse_row(line)
        before = blob(root, str(render["input"]["input_image_sha256"]))
        after = blob(root, str(render["artifact"]["sha256"]))
        if not before.is_file() or not after.is_file():
            skipped.append({"branch_id": branch["branch_id"], "why": "blob_missing"})
            continue
        vector = measure_direction(before, after).as_dict()
        measured = {
            "cast_a": float(vector["cast_a"]), "cast_b": float(vector["cast_b"]),
            "lightness": float(vector["lightness"]),
            "saturation": float(vector["saturation"]),
            "contrast": float(vector["contrast"]),
            **histogram_delta(before, after),
        }
        entries.append({
            "source_id": source_id_by_sha.get(sha, sha[:12]),
            "source_sha256": sha,
            "branch_id": branch["branch_id"],
            "branch_status": branch["status"],
            "preset_id": str(proposal["preset_id"]),
            "strength_bin": str(proposal["strength_bin"]),
            "major": request["major"],
            "row_index": row_index,
            "global_strength": render["params"].get("global_strength"),
            "delta_e": render["metrics"].get("delta_e"),
            "nominal": nominal,
            "measured": measured,
            "shortlist_header": request["header"],
            "campaign": campaign,
        })
    for row in skipped:
        row["campaign"] = campaign
    return {"campaign": campaign, "runs": data["runs"],
            "entries": entries, "skipped": skipped}


def sign_agreement(entries: list[dict[str, Any]]) -> list[dict[str, Any]]:
    out = []
    for label, nominal_key, measured_key, nominal_gate, measured_gate in SIGN_COLUMNS:
        total = agree = near_zero = 0
        for entry in entries:
            if nominal_key not in entry["nominal"]:
                continue
            nominal = float(entry["nominal"][nominal_key])
            measured = float(entry["measured"][measured_key])
            total += 1
            if abs(nominal) < nominal_gate or abs(measured) < measured_gate:
                near_zero += 1
                continue
            agree += int((nominal > 0) == (measured > 0))
        scored = total - near_zero
        out.append({
            "column": label, "n_rows": total, "n_near_zero_excluded": near_zero,
            "n_scored": scored, "n_same_sign": agree,
            "rate": round(agree / scored, 4) if scored else None,
            "nominal_gate": nominal_gate, "measured_gate": measured_gate,
        })
    return out


def _cell(text: Any) -> str:
    return str(text).replace("|", "\\|")


def markdown(ledger: dict[str, Any]) -> str:
    """The side-by-side ledger table plus the per-column sign-agreement table."""
    lines: list[str] = []
    campaigns = ledger["campaigns"]
    statuses = sorted({
        row["status"] for rows in ledger["runs_by_campaign"].values() for row in rows
    } | set(ledger["final_counts"]))
    lines.append("### run 计数\n")
    lines.append("| campaign | " + " | ".join(statuses) + " | 合计 |")
    lines.append("| " + " --- |" * (len(statuses) + 2))
    for campaign in campaigns:
        rows = ledger["runs_by_campaign"][campaign]
        counts = {status: sum(1 for row in rows if row["status"] == status)
                  for status in statuses}
        lines.append(
            f"| `{campaign}` | " + " | ".join(str(counts[s]) for s in statuses)
            + f" | {len(rows)} |"
        )
    final = ledger["final_counts"]
    lines.append(
        "| 合并(后跑批覆盖同源早前状态) | "
        + " | ".join(str(final.get(s, 0)) for s in statuses)
        + f" | {sum(final.values())} |"
    )
    lines.append("")
    lines.append(
        f"入表提案 {len(ledger['entries'])} 条；未入表 {len(ledger['skipped'])} 条"
        f"（原因计数：" + ", ".join(
            f"{why} {sum(1 for row in ledger['skipped'] if row['why'] == why)}"
            for why in sorted({row["why"] for row in ledger["skipped"]})
        ) + "）。\n" if ledger["skipped"] else
        f"入表提案 {len(ledger['entries'])} 条；未入表 0 条。\n"
    )
    lines.append("### 标称 vs 实测 对数表\n")
    lines.append(
        "| # | source_id | preset_id | bin | strength | ΔE | "
        "标称 dL | 实测 lightness | 标称 dSat | 实测 saturation | "
        "标称 cast_hue | 标称 cast_mag | 标称 cast_a | 实测 cast_a | "
        "标称 cast_b | 实测 cast_b | 标称 contrast | 实测 contrast | "
        "标称 shadow_dL | 标称 highlight_dL | 标称 hue_rot | "
        "标称 d_shadow | 实测 d_shadow | 标称 d_mid | 实测 d_mid | "
        "标称 d_high | 实测 d_high |"
    )
    lines.append("| " + " --- |" * 27)
    for index, entry in enumerate(ledger["entries"], 1):
        nominal, measured = entry["nominal"], entry["measured"]
        strength = entry["global_strength"]
        lines.append("| " + " | ".join([
            str(index), _cell(entry["source_id"]), _cell(entry["preset_id"]),
            _cell(entry["strength_bin"]),
            f"{float(strength):.5f}" if strength is not None else "",
            f"{float(entry['delta_e']):.3f}" if entry["delta_e"] is not None else "",
            f"{nominal['dL']:.1f}", f"{measured['lightness']:.3f}",
            f"{nominal['dSat']:.1f}", f"{measured['saturation']:.3f}",
            f"{nominal['cast_hue']:.0f}", f"{nominal['cast_mag']:.1f}",
            f"{nominal['cast_a']:.3f}", f"{measured['cast_a']:.3f}",
            f"{nominal['cast_b']:.3f}", f"{measured['cast_b']:.3f}",
            f"{nominal['contrast']:.3f}", f"{measured['contrast']:.3f}",
            f"{nominal['shadow_dL']:.1f}", f"{nominal['highlight_dL']:.1f}",
            f"{nominal['hue_rot']:.1f}",
            f"{nominal.get('d_shadow', float('nan')):.3f}",
            f"{measured['d_shadow']:.4f}",
            f"{nominal.get('d_mid', float('nan')):.3f}", f"{measured['d_mid']:.4f}",
            f"{nominal.get('d_high', float('nan')):.3f}", f"{measured['d_high']:.4f}",
        ]) + " |")
    lines.append("")
    lines.append("### 逐列符号一致率\n")
    lines.append(
        "| 列（标称 / 实测） | 行数 | 近零剔除 | 计入 | 同号 | 同号率 | "
        "标称近零阈值 | 实测近零阈值 |"
    )
    lines.append("| " + " --- |" * 8)
    for row in ledger["sign_agreement"]:
        lines.append("| " + " | ".join([
            _cell(row["column"]), str(row["n_rows"]), str(row["n_near_zero_excluded"]),
            str(row["n_scored"]), str(row["n_same_sign"]),
            "" if row["rate"] is None else f"{row['rate']:.4f}",
            str(row["nominal_gate"]), str(row["measured_gate"]),
        ]) + " |")
    lines.append("")
    return "\n".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--campaign", required=True, action="append")
    parser.add_argument("--root", required=True, action="append", type=Path)
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--md", type=Path, default=None)
    parser.add_argument("--dsn", default=DSN)
    args = parser.parse_args()
    if len(args.campaign) != len(args.root):
        parser.error("--campaign and --root must be given the same number of times")
    ledger = build(list(zip(args.campaign, args.root)), args.dsn)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(
        json.dumps(ledger, ensure_ascii=False, sort_keys=True, indent=1) + "\n",
        encoding="utf-8",
    )
    if args.md is not None:
        args.md.parent.mkdir(parents=True, exist_ok=True)
        args.md.write_text(markdown(ledger), encoding="utf-8")
    print(json.dumps({
        "campaigns": ledger["campaigns"], "final_counts": ledger["final_counts"],
        "entries": len(ledger["entries"]),
        "skipped": len(ledger["skipped"]), "out": str(args.out),
        "sign_agreement": ledger["sign_agreement"],
    }, ensure_ascii=False, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
