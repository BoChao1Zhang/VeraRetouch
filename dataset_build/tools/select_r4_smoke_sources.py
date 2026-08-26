"""EPR-047 c10: pick the R4/G4/L4 smoke sources deterministically.

The rule, in full, so a re-run is reproducible and an ad-hoc pick is not needed:

  1. read the annotated manifest (default `sources5k.annotated-v34.jsonl`);
  2. order every row by `sha1(source_id)` ascending, ties by `source_id` - the same
     sha1 rule family the campaign's splits use, never file order;
  3. keep a row only when BOTH run-period preconditions hold:
     * the offline annotation exists and its `provenance.reasoning_effort == "high"`,
       which is the hard gate `source_annotations.validate_source_annotation` enforces
       at run time (`docs/RETRIEVAL_RERANK_20260824.md` section 3.1);
     * the frozen `source_response_v2` probe cache carries this source's
       `source_response_v2.<sha16>.json.gz`, which the R4 recall node reads and never
       re-measures;
  4. take the first `--count` survivors.

Writes the run manifest (the exact JSONL `cli.py run --source-manifest` eats) and a
proof file listing, per picked source, its sha1 key, its effort tag and the probe cache
path with its byte size.

    python -m dataset_build.tools.select_r4_smoke_sources \
        --manifest /home/bc/data/agent_loop/local-v1/sources5k.annotated-v34.jsonl \
        --probe-cache /home/bc/data/agent_loop/local-v1/source_response_v2 \
        --count 5 --out <manifest.jsonl> --proof <proof.json>
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

DEFAULT_MANIFEST = Path(
    "/home/bc/data/agent_loop/local-v1/sources5k.annotated-v34.jsonl"
)
DEFAULT_PROBE_CACHE = Path("/home/bc/data/agent_loop/local-v1/source_response_v2")
#: The literal `source_annotations.validate_source_annotation` requires at run time.
REQUIRED_EFFORT = "high"
SELECTION_RULE = "sha1(source_id) ascending; high-effort annotation; probe cache present"


def sha1_key(source_id: str) -> str:
    return hashlib.sha1(str(source_id).encode("utf-8")).hexdigest()


def probe_path(cache_dir: Path, source_sha256: str) -> Path:
    return cache_dir / f"source_response_v2.{str(source_sha256)[:16]}.json.gz"


def _rows(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()
            if line.strip()]


def select(
    manifest: Path, cache_dir: Path, count: int
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, int]]:
    rows = _rows(manifest)
    ordered = sorted(rows, key=lambda row: (sha1_key(row["source_id"]),
                                            str(row["source_id"])))
    picked: list[dict[str, Any]] = []
    proof: list[dict[str, Any]] = []
    counts = {"scanned": 0, "no_annotation": 0, "effort_rejected": 0,
              "probe_missing": 0, "picked": 0}
    for row in ordered:
        if len(picked) >= count:
            break
        counts["scanned"] += 1
        annotation_path = Path(str(row.get("source_annotation_path") or ""))
        if not annotation_path.is_file():
            counts["no_annotation"] += 1
            continue
        annotation = json.loads(annotation_path.read_text(encoding="utf-8"))
        effort = str((annotation.get("provenance") or {}).get("reasoning_effort") or "")
        if effort != REQUIRED_EFFORT:
            counts["effort_rejected"] += 1
            continue
        source_sha = str(annotation.get("source_sha256") or "")
        cache = probe_path(cache_dir, source_sha)
        if not cache.is_file():
            counts["probe_missing"] += 1
            continue
        counts["picked"] += 1
        picked.append(row)
        proof.append({
            "rank": len(picked),
            "source_id": str(row["source_id"]),
            "sha1_source_id": sha1_key(row["source_id"]),
            "source_sha256": source_sha,
            "scene": str(row.get("scene") or annotation.get("scene") or ""),
            "annotation_path": str(annotation_path),
            "reasoning_effort": effort,
            "diagnose_prompt_revision": str(
                (annotation.get("provenance") or {}).get("diagnose_prompt_revision") or ""
            ),
            "enhancement_opportunities": len(
                (annotation.get("diagnosis") or {}).get(
                    "enhancement_opportunities"
                ) or []
            ),
            "probe_cache_path": str(cache),
            "probe_cache_bytes": cache.stat().st_size,
        })
    return picked, proof, counts


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="select_r4_smoke_sources")
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--probe-cache", type=Path, default=DEFAULT_PROBE_CACHE)
    parser.add_argument("--count", type=int, default=5)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--proof", type=Path, required=True)
    args = parser.parse_args(argv)

    picked, proof, counts = select(args.manifest, args.probe_cache, args.count)
    if len(picked) != args.count:
        raise SystemExit(
            f"only {len(picked)} of {args.count} sources satisfy the rule: {counts}"
        )
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(
        "".join(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n"
                for row in picked),
        encoding="utf-8",
    )
    payload = {
        "schema": "r4-smoke-source-selection-v1",
        "rule": SELECTION_RULE,
        "manifest": str(args.manifest),
        "manifest_rows": len(_rows(args.manifest)),
        "probe_cache": str(args.probe_cache),
        "required_reasoning_effort": REQUIRED_EFFORT,
        "count": int(args.count),
        "scan_counts": counts,
        "sources": proof,
        "out_manifest": str(args.out),
        "out_manifest_sha256": hashlib.sha256(
            args.out.read_bytes()
        ).hexdigest(),
    }
    args.proof.parent.mkdir(parents=True, exist_ok=True)
    args.proof.write_text(
        json.dumps(payload, ensure_ascii=False, sort_keys=True, indent=1) + "\n",
        encoding="utf-8",
    )
    print(json.dumps({"picked": len(picked), "counts": counts,
                      "out": str(args.out), "proof": str(args.proof)},
                     ensure_ascii=False, sort_keys=True, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = ["REQUIRED_EFFORT", "SELECTION_RULE", "main", "probe_path", "select",
           "sha1_key"]
