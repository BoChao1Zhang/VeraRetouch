"""Offline backfill of the P2 §S2.1 visibility metric family over committed chains.

`backfill` walks committed local leaves of the given campaigns, resolves the
render_record of each leaf (``input_image_sha256`` = global_after blob,
``mask_sha256_or_global`` = mask alpha blob, ``artifact`` = final_after blob),
computes `dataset_build.agent_loop.visibility.visibility_metrics` and writes one
JSONL row per chain. Chains whose blobs are absent are skipped and counted.

`validate` joins that JSONL against the filled intent questionnaire CSVs through
their ``item_key.json`` and prints the rating-grouped tables.

Nothing here writes back into the agent-loop database or changes any gate.
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import os
import sys
from collections import Counter, defaultdict
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from dataset_build.agent_loop.render import load_alpha, load_rgb  # noqa: E402
from dataset_build.agent_loop.visibility import (  # noqa: E402
    COMPONENTS_CONTRACT, VISIBILITY_CONTRACT, VisibilityError,
    assert_component_columns, assert_visibility_columns, visibility_components,
    visibility_metrics,
)

DEFAULT_BLOB_ROOTS = (
    "/mnt/ramstage/agent_loop/local-v2-b15-smoke/blobs",
    "/mnt/ramstage/agent_loop/local-v2-pilot/blobs",
    "/mnt/ramstage/agent_loop/local-v2-b11-smoke/blobs",
    "/home/bc/data/agent_loop/local-v2-pilot/blobs",
    "/home/bc/data/agent_loop/a6-lane-smoke/blobs",
)
DEFAULT_CAMPAIGNS = ("local-v2-iter2", "local-v2-iter3", "local-v2-iter4")
DEFAULT_DSN = (
    "postgresql://research:research@127.0.0.1:5432/agent_loop"
    "?options=-c%20search_path%3Dagent_loop"
)

CHAIN_SQL = """
select b.campaign_id, b.branch_id, b.source_sha256, b.result_json,
       r.input_json, r.parameters_json, r.artifact_json
from agent_loop.agent_branch b
left join agent_loop.render_record r
       on r.render_hash = (b.result_json::json->'render'->>'render_hash')
where b.level = 'local'
  and b.campaign_id = any(%s)
  and (b.result_json::json->>'commit_status') = 'committed'
order by b.campaign_id, b.branch_id
"""


# ----------------------------------------------------------------- blob lookup
def blob_path(roots: Sequence[Path], sha256: str) -> Path | None:
    for root in roots:
        candidate = root / sha256[:2] / sha256[2:4] / sha256
        if candidate.exists():
            return candidate
    return None


def fetch_chains(dsn: str, campaigns: Sequence[str]) -> list[dict[str, Any]]:
    import psycopg

    with psycopg.connect(dsn) as conn:
        with conn.cursor() as cur:
            cur.execute(CHAIN_SQL, (list(campaigns),))
            rows = cur.fetchall()
    chains: list[dict[str, Any]] = []
    for campaign, branch_id, source, result_json, input_json, params_json, artifact_json in rows:
        result = json.loads(result_json)
        chains.append({
            "campaign_id": campaign,
            "branch_id": branch_id,
            "source_sha256": source,
            "global_branch_id": result.get("global_branch_id"),
            "intent": result.get("intent"),
            "intent_variant": result.get("intent_variant"),
            "mask_family": result.get("mask_family"),
            "mask_role": result.get("mask_role"),
            "global_strength_bin": result.get("global_strength_bin"),
            "input": json.loads(input_json) if input_json else None,
            "parameters": json.loads(params_json) if params_json else None,
            "artifact": json.loads(artifact_json) if artifact_json else None,
            "leaf_metrics": ((result.get("render") or {}).get("metrics") or {}),
        })
    return chains


# ----------------------------------------------------------------- one chain
_WORKER_ROOTS: list[Path] = []
_WORKER_KIND = "d1"


def _init_worker(roots: Sequence[str], kind: str = "d1") -> None:
    global _WORKER_ROOTS, _WORKER_KIND
    _WORKER_ROOTS = [Path(root) for root in roots]
    _WORKER_KIND = kind


def compute_chain(
    chain: Mapping[str, Any], roots: Sequence[Path], kind: str = "d1"
) -> dict[str, Any]:
    """Return one output row; `status` is `ok`, `blob_missing` or `error`.

    ``kind='d1'`` computes the §S2.1 CIEDE2000 family, ``kind='components'``
    the D1b Lab decomposition on the very same support sample.
    """
    base = {
        "campaign_id": chain["campaign_id"],
        "branch_id": chain["branch_id"],
        "source_sha256": chain["source_sha256"],
        "global_branch_id": chain.get("global_branch_id"),
        "intent": chain.get("intent"),
        "mask_family": chain.get("mask_family"),
        "mask_role": chain.get("mask_role"),
        "global_strength_bin": chain.get("global_strength_bin"),
        "local_strength_bin": (chain.get("parameters") or {}).get("strength_bin"),
        "local_strength": (chain.get("parameters") or {}).get("local_strength"),
        "de_masked": (chain.get("leaf_metrics") or {}).get("delta_e"),
    }
    request = chain.get("input") or {}
    before_sha = request.get("input_image_sha256")
    mask_sha = request.get("mask_sha256_or_global")
    after_sha = (chain.get("artifact") or {}).get("sha256")
    base.update({
        "global_after_sha256": before_sha,
        "final_after_sha256": after_sha,
        "mask_sha256": mask_sha,
    })
    if not (before_sha and mask_sha and after_sha):
        return {**base, "status": "blob_missing", "missing": "render_record_incomplete"}
    before_path = blob_path(roots, before_sha)
    after_path = blob_path(roots, after_sha)
    mask_path = blob_path(roots, mask_sha)
    missing = [
        name for name, path in
        (("global_after", before_path), ("final_after", after_path), ("mask", mask_path))
        if path is None
    ]
    if missing:
        return {**base, "status": "blob_missing", "missing": ",".join(missing)}
    try:
        before = load_rgb(before_path)
        after = load_rgb(after_path)
        alpha = load_alpha(mask_path, (before.shape[1], before.shape[0]))
        if kind == "components":
            metrics = visibility_components(
                before, after, alpha, seed_key=chain["branch_id"])
            assert_component_columns(metrics)
        else:
            metrics = visibility_metrics(
                before, after, alpha, seed_key=chain["branch_id"])
            assert_visibility_columns(metrics)
    except (VisibilityError, OSError, ValueError) as exc:
        return {**base, "status": "error", "error": f"{type(exc).__name__}: {exc}"}
    return {**base, "status": "ok", **metrics}


def _worker(chain: Mapping[str, Any]) -> dict[str, Any]:
    return compute_chain(chain, _WORKER_ROOTS, _WORKER_KIND)


def chain_from_row(row: Mapping[str, Any]) -> dict[str, Any]:
    """Rebuild a chain record from one already-backfilled JSONL row.

    Lets the D1b pass run over exactly the chains D1 resolved, with no second
    database read and no risk of a different chain set.
    """
    return {
        "campaign_id": row.get("campaign_id"),
        "branch_id": row.get("branch_id"),
        "source_sha256": row.get("source_sha256"),
        "global_branch_id": row.get("global_branch_id"),
        "intent": row.get("intent"),
        "mask_family": row.get("mask_family"),
        "mask_role": row.get("mask_role"),
        "global_strength_bin": row.get("global_strength_bin"),
        "input": {
            "input_image_sha256": row.get("global_after_sha256"),
            "mask_sha256_or_global": row.get("mask_sha256"),
        },
        "parameters": {
            "strength_bin": row.get("local_strength_bin"),
            "local_strength": row.get("local_strength"),
        },
        "artifact": {"sha256": row.get("final_after_sha256")},
        "leaf_metrics": {"delta_e": row.get("de_masked")},
    }


def read_chain_rows(path: Path, status: str | None = "ok") -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            row = json.loads(line)
            if status is None or row.get("status") == status:
                rows.append(row)
    return rows


# ----------------------------------------------------------------- backfill
def run_backfill(args: argparse.Namespace) -> int:
    roots = [Path(root) for root in args.blob_root]
    kind = getattr(args, "kind", "d1")
    source = getattr(args, "from_chains", None)
    if source:
        chains = [chain_from_row(row) for row in read_chain_rows(Path(source))]
    else:
        chains = fetch_chains(args.dsn, args.campaign)
    if args.limit:
        chains = chains[: args.limit]
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    counts: Counter[str] = Counter()
    missing: Counter[str] = Counter()
    rows: list[dict[str, Any]] = []
    if args.workers > 1:
        with ProcessPoolExecutor(
            max_workers=args.workers,
            initializer=_init_worker,
            initargs=([str(root) for root in roots], kind),
        ) as pool:
            rows = list(pool.map(_worker, chains, chunksize=4))
    else:
        rows = [compute_chain(chain, roots, kind) for chain in chains]
    rows.sort(key=lambda row: (row["campaign_id"], row["branch_id"]))
    with out_path.open("w", encoding="utf-8") as handle:
        for row in rows:
            counts[row["status"]] += 1
            if row["status"] == "blob_missing":
                missing[row.get("missing", "?")] += 1
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
    summary = {
        "contract": COMPONENTS_CONTRACT if kind == "components" else VISIBILITY_CONTRACT,
        "kind": kind,
        "source": source or f"postgres:{','.join(args.campaign)}",
        "campaigns": list(args.campaign),
        "chains_total": len(rows),
        "status_counts": dict(sorted(counts.items())),
        "blob_missing_detail": dict(sorted(missing.items())),
        "out": str(out_path),
    }
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


# ----------------------------------------------------------------- statistics
def spearman(xs: Sequence[float], ys: Sequence[float]) -> float | None:
    if len(xs) < 3:
        return None
    def rank(values: Sequence[float]) -> list[float]:
        order = sorted(range(len(values)), key=lambda i: values[i])
        ranks = [0.0] * len(values)
        index = 0
        while index < len(order):
            stop = index
            while stop + 1 < len(order) and values[order[stop + 1]] == values[order[index]]:
                stop += 1
            shared = (index + stop) / 2.0 + 1.0
            for pos in range(index, stop + 1):
                ranks[order[pos]] = shared
            index = stop + 1
        return ranks
    rx, ry = rank(list(xs)), rank(list(ys))
    n = len(rx)
    mx, my = sum(rx) / n, sum(ry) / n
    num = sum((a - mx) * (b - my) for a, b in zip(rx, ry))
    den = math.sqrt(sum((a - mx) ** 2 for a in rx) * sum((b - my) ** 2 for b in ry))
    return float(num / den) if den else None


def quantiles(values: Sequence[float]) -> dict[str, float | None]:
    if not values:
        return {"n": 0, "mean": None, "p25": None, "p50": None, "p75": None}
    array = np.asarray(values, dtype=np.float64)
    return {
        "n": int(array.size),
        "mean": float(array.mean()),
        "p25": float(np.quantile(array, 0.25)),
        "p50": float(np.quantile(array, 0.50)),
        "p75": float(np.quantile(array, 0.75)),
    }


METRIC_COLUMNS = (
    "de_masked", "de_in", "de_out", "de_contrast", "de_in_p50", "de_in_p90",
    "edge_de", "edge_step_p95", "floor_de_in", "floor_de_contrast",
    "de_in_over_floor", "de_in_minus_floor", "support_frac",
)

# D1b: the decomposition columns, reported next to the two D1 reference columns
# (`de_masked` = the shipping metric, `support_frac` = D1's strongest signal).
COMPONENT_METRIC_COLUMNS = (
    "dL_mean", "dL_p90", "dL_signed_mean",
    "dC_mean", "dC_p90", "dC_signed_mean", "dHue_mean",
    "dL_highlight_mean", "dL_highlight_signed_mean", "highlight_frac",
    "dL_grad", "dL_step_mean",
    "de_masked", "support_frac",
)

COLUMN_SETS = {
    "d1": METRIC_COLUMNS,
    "components": COMPONENT_METRIC_COLUMNS,
}


def load_rounds(specs: Iterable[str]) -> list[dict[str, Any]]:
    rounds = []
    for spec in specs:
        name, key_path, csv_path = spec.split("=", 2) if spec.count("=") >= 2 else \
            (Path(spec).name, spec, spec)
        key = json.loads(Path(key_path).read_text(encoding="utf-8"))
        ratings: dict[str, int] = {}
        with Path(csv_path).open(encoding="utf-8-sig") as handle:
            for line in csv.DictReader(handle):
                raw = (line.get("rating") or "").strip()
                if raw:
                    ratings[(line.get("item_id") or "").strip()] = int(raw)
        rounds.append({
            "name": name, "campaign": key.get("campaign"),
            "items": key["items"], "ratings": ratings,
            "item_key": key_path, "csv": csv_path,
        })
    return rounds


def run_validate(args: argparse.Namespace) -> int:
    columns = COLUMN_SETS[getattr(args, "columns", "d1")]
    by_branch: dict[str, dict[str, Any]] = {}
    with Path(args.chains).open(encoding="utf-8") as handle:
        for line in handle:
            row = json.loads(line)
            if row.get("status") == "ok":
                by_branch[row["branch_id"]] = row
    rounds = load_rounds(args.round)

    joined: list[dict[str, Any]] = []
    unmatched: Counter[str] = Counter()
    for entry in rounds:
        for item_id, rating in sorted(entry["ratings"].items()):
            item = entry["items"].get(item_id)
            if item is None:
                unmatched[f"{entry['name']}:unknown_item"] += 1
                continue
            row = by_branch.get(item["branch_id"])
            if row is None:
                unmatched[f"{entry['name']}:no_metrics"] += 1
                continue
            joined.append({
                "round": entry["name"], "campaign": entry["campaign"],
                "item_id": item_id, "rating": rating,
                "intent": item.get("intent"),
                "branch_id": item["branch_id"],
                **{col: row.get(col) for col in columns},
            })

    # runtime assertion: every requested column really reached the join
    absent = sorted(
        col for col in columns
        if not any(row.get(col) is not None for row in joined)
    )
    if absent:
        raise VisibilityError(f"columns absent from every joined row: {absent}")

    report: dict[str, Any] = {
        "contract": (
            COMPONENTS_CONTRACT if getattr(args, "columns", "d1") == "components"
            else VISIBILITY_CONTRACT
        ),
        "columns": list(columns),
        "rounds": [
            {"name": r["name"], "campaign": r["campaign"],
             "n_rated": len(r["ratings"]), "csv": r["csv"], "item_key": r["item_key"]}
            for r in rounds
        ],
        "n_rated_total": sum(len(r["ratings"]) for r in rounds),
        "n_joined": len(joined),
        "unmatched": dict(sorted(unmatched.items())),
        "rating_counts": dict(sorted(Counter(r["rating"] for r in joined).items())),
    }

    # rating group x metric distribution
    groups: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for row in joined:
        groups[row["rating"]].append(row)
    report["by_rating"] = {
        str(rating): {
            col: quantiles([r[col] for r in rows if r.get(col) is not None])
            for col in columns
        }
        for rating, rows in sorted(groups.items())
    }
    report["spearman_rating"] = {
        col: spearman(
            [r[col] for r in joined if r.get(col) is not None],
            [r["rating"] for r in joined if r.get(col) is not None],
        )
        for col in columns
    }

    # pre-registered contrast: rating == 4 vs rating in {3, 5}
    def split(pred) -> list[dict[str, Any]]:
        return [row for row in joined if pred(row["rating"])]

    four = split(lambda v: v == 4)
    other = split(lambda v: v in (3, 5))
    contrast: dict[str, Any] = {}
    for col in columns:
        a = [r[col] for r in four if r.get(col) is not None]
        b = [r[col] for r in other if r.get(col) is not None]
        qa, qb = quantiles(a), quantiles(b)
        rank_r = spearman(
            [r[col] for r in four + other if r.get(col) is not None],
            [1.0 if r["rating"] == 4 else 0.0
             for r in four + other if r.get(col) is not None],
        )
        contrast[col] = {
            "rating4": qa, "rating35": qb,
            "mean_gap": (qa["mean"] - qb["mean"])
            if qa["mean"] is not None and qb["mean"] is not None else None,
            "p50_gap": (qa["p50"] - qb["p50"])
            if qa["p50"] is not None and qb["p50"] is not None else None,
            "rank_r_is4": rank_r,
        }
    report["rating4_vs_rating35"] = contrast

    # per-intent stratification
    per_intent: dict[str, Any] = {}
    intents: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in joined:
        intents[str(row["intent"])].append(row)
    for intent, rows in sorted(intents.items()):
        per_intent[intent] = {
            "n": len(rows),
            "mean_rating": sum(r["rating"] for r in rows) / len(rows),
            "rate_4": sum(1 for r in rows if r["rating"] == 4) / len(rows),
            "spearman_rating": {
                col: spearman(
                    [r[col] for r in rows if r.get(col) is not None],
                    [r["rating"] for r in rows if r.get(col) is not None],
                )
                for col in columns
            },
            "by_rating_mean": {
                str(rating): {
                    col: quantiles(
                        [r[col] for r in rows if r["rating"] == rating
                         and r.get(col) is not None]
                    )["mean"]
                    for col in columns
                }
                for rating in sorted({r["rating"] for r in rows})
            },
        }
    report["per_intent"] = per_intent

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(
        json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    if args.joined_out:
        joined_path = Path(args.joined_out)
        joined_path.parent.mkdir(parents=True, exist_ok=True)
        with joined_path.open("w", encoding="utf-8") as handle:
            for row in sorted(joined, key=lambda r: (r["round"], r["item_id"])):
                handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
    print(json.dumps({
        "n_joined": report["n_joined"], "unmatched": report["unmatched"],
        "out": str(out),
    }, ensure_ascii=False, indent=2))
    return 0


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)

    back = sub.add_parser("backfill")
    back.add_argument("--dsn", default=os.environ.get(
        "VERARETOUCH_AGENT_POSTGRES_DSN", DEFAULT_DSN))
    back.add_argument("--campaign", action="append", default=None)
    back.add_argument("--blob-root", action="append", default=None)
    back.add_argument("--out", default="/home/bc/data/scratch/visibility_metrics/chains.jsonl")
    back.add_argument("--kind", choices=tuple(COLUMN_SETS), default="d1")
    back.add_argument("--from-chains", default=None,
                      help="reuse the chain set of an existing backfill JSONL "
                           "instead of querying postgres")
    back.add_argument("--workers", type=int, default=8)
    back.add_argument("--limit", type=int, default=0)
    back.set_defaults(func=run_backfill)

    val = sub.add_parser("validate")
    val.add_argument("--chains", default="/home/bc/data/scratch/visibility_metrics/chains.jsonl")
    val.add_argument("--round", action="append", required=True,
                     help="name=item_key.json=ratings.csv")
    val.add_argument("--columns", choices=tuple(COLUMN_SETS), default="d1")
    val.add_argument("--out", default="/home/bc/data/scratch/visibility_metrics/validation.json")
    val.add_argument("--joined-out", default=None)
    val.set_defaults(func=run_validate)

    args = parser.parse_args(argv)
    if getattr(args, "campaign", None) is None:
        args.campaign = list(DEFAULT_CAMPAIGNS)
    if getattr(args, "blob_root", None) is None:
        args.blob_root = list(DEFAULT_BLOB_ROOTS)
    return int(args.func(args))


if __name__ == "__main__":
    raise SystemExit(main())
