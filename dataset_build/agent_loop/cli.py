"""CLI for preflight, run/resume, status, and audit export."""
from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any

from .api_cache import ExactResponseCache
from .artifacts import ArtifactStore
from .candidates import LutCatalog
from .checkpoint import open_checkpointer
from .config import AgentLoopConfig, load_config
from .graph import build_graph, run_source
from .metrics import campaign_metrics
from .preflight import require_preflight, run_preflight
from .responses import CachedResponsesClient, ResponsesAdapter
from .runtime import build_terra_router, create_audit, create_services
from .scheduler import TerraLimiter
from .source_annotations import annotate_source, load_source_annotation
from .retention import apply_cleanup_manifest, plan_cleanup, write_cleanup_manifest


def _jsonl(path: Path) -> list[dict[str, Any]]:
    rows = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            row = json.loads(line)
            required = {"source_id", "source_path", "subject_path"}
            if not isinstance(row, dict) or not required.issubset(row):
                raise ValueError(f"invalid source manifest row {line_number}")
            rows.append(row)
    return rows


def _hydrate_annotations(manifest: Path, rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    hydrated = []
    for line_number, row in enumerate(rows, 1):
        value = row.get("source_annotation_path")
        if not isinstance(value, str) or not value:
            raise ValueError(f"source manifest row {line_number} lacks offline annotation")
        target = Path(value).expanduser()
        if not target.is_absolute():
            target = (manifest.parent / target).resolve()
        annotation = load_source_annotation(target, row)
        hydrated.append({
            **row, "source_annotation_path": str(target),
            "source_annotation": annotation,
            "source_sha256": str(annotation["source_sha256"]),
            "subject_sha256": str(annotation["subject_sha256"]),
        })
    return hydrated


def _preflight(config: AgentLoopConfig) -> int:
    audit = create_audit(config)
    artifacts = ArtifactStore(config.artifact_root, recorder=audit.record_artifact)
    catalog = LutCatalog.load(config.catalog, config.databuild_config)
    cache = ExactResponseCache(audit, lease_seconds=config.request_lease_seconds)
    results = []
    for endpoint in config.terra_lanes:
        client = CachedResponsesClient(ResponsesAdapter(endpoint, artifacts), cache)
        results.append(run_preflight(
            config, artifacts, audit, client, catalog,
            TerraLimiter(config.terra_concurrency_target, lane_id=endpoint.identity),
            endpoint=endpoint,
        ))
    payload = results[0] if len(results) == 1 else {"lanes": results}
    print(json.dumps(payload, ensure_ascii=False, sort_keys=True, indent=2))
    return 0


def _run(config: AgentLoopConfig, manifest: Path, limit: int, workers: int,
         resume: bool) -> int:
    sources = _jsonl(manifest)
    if limit > 0:
        sources = sources[:limit]
    sources = _hydrate_annotations(manifest, sources)
    services = create_services(config)
    landing = getattr(services, "landing", None)
    batch_size = config.artifact_landing.min_groups if landing is not None \
        else max(1, len(sources))
    with open_checkpointer(config) as checkpointer:
        graph = build_graph(services, checkpointer=checkpointer)
        results = []
        with ThreadPoolExecutor(max_workers=max(1, workers)) as pool:
            for start in range(0, len(sources), batch_size):
                batch = sources[start:start + batch_size]
                future_rows = {
                    pool.submit(run_source, graph, config, source, resume=resume): source
                    for source in batch
                }
                for future in as_completed(future_rows):
                    source = future_rows[future]
                    try:
                        result = future.result()
                        results.append({"source_id": source["source_id"],
                                        "status": result.get("terminal_status", "running")})
                    except Exception as exc:
                        pass_index = int(source.get("pass_index", 0))
                        source_sha = str(source["source_sha256"])
                        subject_sha = str(source["subject_sha256"])
                        run_revision = f"{config.thread_revision}:pass-{pass_index}"
                        services.audit.record_source_start({
                            "campaign_id": config.campaign_id,
                            "source_sha256": source_sha,
                            "prompt_revision": run_revision,
                            "thread_id": config.thread_id(source_sha, subject_sha, pass_index),
                            "source_id": source["source_id"],
                        })
                        services.audit.record_source_finish(
                            config.campaign_id, source_sha, run_revision, "error",
                            {"error_type": type(exc).__name__, "error": str(exc)}, None,
                        )
                        results.append({"source_id": source["source_id"],
                                        "status": "error", "error_type": type(exc).__name__})
                if landing is not None:
                    landing.maybe_land(completed_groups=len(batch))
        if landing is not None:
            landing.maybe_land(force=True)
    summary = Counter(row["status"] for row in results)
    print(json.dumps({"counts": dict(sorted(summary.items())), "sources": results},
                     ensure_ascii=False, sort_keys=True, indent=2))
    return 1 if summary.get("error") else 0


def _annotate_sources(
    config: AgentLoopConfig, manifest: Path, output_dir: Path,
    output_manifest: Path, limit: int, workers: int,
) -> int:
    sources = _jsonl(manifest)
    if limit > 0:
        sources = sources[:limit]
    audit = create_audit(config)
    require_preflight(config, audit)
    artifacts = ArtifactStore(config.artifact_root, recorder=audit.record_artifact)
    cache = ExactResponseCache(audit, lease_seconds=config.request_lease_seconds)
    catalog = LutCatalog.load(config.catalog, config.databuild_config)
    router = build_terra_router(config, artifacts, cache, audit)
    terra = router.lanes[0].client
    limiter = router.lanes[0].limiter
    output_dir.mkdir(parents=True, exist_ok=True)

    def annotate(index: int, source: dict[str, Any]):
        existing = source.get("source_annotation_path")
        if isinstance(existing, str) and existing:
            existing_path = Path(existing).expanduser()
            if not existing_path.is_absolute():
                existing_path = (manifest.parent / existing_path).resolve()
            try:
                load_source_annotation(existing_path, source)
            except ValueError:
                pass
            else:
                return index, {
                    **source, "source_annotation_path": str(existing_path),
                }, True
        annotation = annotate_source(
            config, artifacts, audit, terra, limiter, catalog, source, router
        )
        suffix = hashlib.sha256(str(source["source_id"]).encode()).hexdigest()[:16]
        target = (output_dir / f"source_annotation_{suffix}.json").resolve()
        temporary = target.with_suffix(".json.tmp")
        temporary.write_text(
            json.dumps(annotation, ensure_ascii=False, sort_keys=True, indent=2) + "\n",
            encoding="utf-8",
        )
        temporary.replace(target)
        return index, {**source, "source_annotation_path": str(target)}, False

    completed: dict[int, dict[str, Any]] = {}
    failures: dict[int, dict[str, Any]] = {}
    reused = 0
    output_manifest.parent.mkdir(parents=True, exist_ok=True)
    failure_manifest = output_manifest.with_suffix(output_manifest.suffix + ".failures.jsonl")

    if output_manifest.is_file():
        prior_by_id = {
            str(row["source_id"]): row for row in _jsonl(output_manifest)
        }
        for index, source in enumerate(sources):
            prior = prior_by_id.get(str(source["source_id"]))
            if prior is None:
                continue
            value = prior.get("source_annotation_path")
            if not isinstance(value, str) or not value:
                continue
            target = Path(value).expanduser()
            if not target.is_absolute():
                target = (output_manifest.parent / target).resolve()
            try:
                load_source_annotation(target, source)
            except ValueError:
                continue
            completed[index] = {
                **source, "source_annotation_path": str(target),
            }
            reused += 1

    def persist() -> None:
        temporary_manifest = output_manifest.with_suffix(output_manifest.suffix + ".tmp")
        temporary_manifest.write_text("".join(
            json.dumps(completed[index], ensure_ascii=False, sort_keys=True) + "\n"
            for index in sorted(completed)
        ), encoding="utf-8")
        temporary_manifest.replace(output_manifest)
        temporary_failures = failure_manifest.with_suffix(failure_manifest.suffix + ".tmp")
        temporary_failures.write_text("".join(
            json.dumps(failures[index], ensure_ascii=False, sort_keys=True) + "\n"
            for index in sorted(failures)
        ), encoding="utf-8")
        temporary_failures.replace(failure_manifest)

    persist()
    with ThreadPoolExecutor(max_workers=max(1, workers)) as pool:
        futures = {
            pool.submit(annotate, index, source): (index, source)
            for index, source in enumerate(sources)
            if index not in completed
        }
        for future in as_completed(futures):
            index, source = futures[future]
            try:
                _index, row, was_reused = future.result()
                completed[index] = row
                reused += int(was_reused)
                failures.pop(index, None)
            except Exception as exc:
                failures[index] = {
                    "source_id": str(source["source_id"]),
                    "source_path": str(source["source_path"]),
                    "error_type": type(exc).__name__, "error": str(exc),
                }
            persist()
    print(json.dumps({
        "annotations": len(completed), "generated": len(completed) - reused,
        "reused": reused, "failures": len(failures),
        "failure_manifest": str(failure_manifest.resolve()),
        "output_dir": str(output_dir.resolve()),
        "source_manifest": str(output_manifest.resolve()),
    }, ensure_ascii=False, sort_keys=True, indent=2))
    return 1 if failures else 0


def _status(config: AgentLoopConfig) -> int:
    audit = create_audit(config)
    print(json.dumps(
        campaign_metrics(audit, config.campaign_id),
        ensure_ascii=False, sort_keys=True, indent=2,
    ))
    return 0


def _export(config: AgentLoopConfig, output: Path) -> int:
    audit = create_audit(config)
    payload = {
        "schema": "local-retouch-audit-export-v1", "config": config.sanitized_dict(),
        "tables": audit.export_tables(),
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, ensure_ascii=False, sort_keys=True, indent=2),
                      encoding="utf-8")
    print(json.dumps({"output": str(output), "tables": {
        key: len(value) for key, value in payload["tables"].items()
    }}, sort_keys=True))
    return 0


def _cleanup(config: AgentLoopConfig, manifest_path: Path, apply: bool) -> int:
    audit = create_audit(config)
    artifacts = ArtifactStore(config.artifact_root, recorder=audit.record_artifact)
    if apply:
        result = apply_cleanup_manifest(manifest_path, audit, artifacts)
    else:
        result = plan_cleanup(
            audit, artifacts, rejected_days=config.rejected_asset_ttl_days
        )
        write_cleanup_manifest(manifest_path, result)
        result = {"manifest": str(manifest_path), "targets": len(result["targets"]),
                  "dry_run": True}
    print(json.dumps(result, sort_keys=True, indent=2))
    return 0


def parser() -> argparse.ArgumentParser:
    root = argparse.ArgumentParser(prog="python -m dataset_build.agent_loop.cli")
    root.add_argument("--config", type=Path, required=True)
    commands = root.add_subparsers(dest="command", required=True)
    commands.add_parser("preflight")
    annotate = commands.add_parser("annotate-sources")
    annotate.add_argument("--source-manifest", type=Path, required=True)
    annotate.add_argument("--out-dir", type=Path, required=True)
    annotate.add_argument("--out-manifest", type=Path, required=True)
    annotate.add_argument("--limit", type=int, default=0)
    annotate.add_argument("--workers", type=int, default=16)
    for name in ("run", "resume"):
        command = commands.add_parser(name)
        command.add_argument("--source-manifest", type=Path, required=True)
        command.add_argument("--limit", type=int, default=0)
        command.add_argument("--workers", type=int, default=32)
    commands.add_parser("status")
    export = commands.add_parser("export-audit")
    export.add_argument("--out", type=Path, required=True)
    cleanup = commands.add_parser("cleanup-rejected")
    cleanup.add_argument("--manifest", type=Path, required=True)
    cleanup.add_argument("--apply", action="store_true")
    return root


def main(argv: list[str] | None = None) -> int:
    args = parser().parse_args(argv)
    config = load_config(args.config)
    if args.command == "preflight":
        return _preflight(config)
    if args.command == "annotate-sources":
        return _annotate_sources(
            config, args.source_manifest, args.out_dir, args.out_manifest,
            args.limit, args.workers,
        )
    if args.command in {"run", "resume"}:
        return _run(config, args.source_manifest, args.limit, args.workers,
                    args.command == "resume")
    if args.command == "status":
        return _status(config)
    if args.command == "export-audit":
        return _export(config, args.out)
    if args.command == "cleanup-rejected":
        return _cleanup(config, args.manifest, args.apply)
    raise AssertionError(args.command)


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = ["main", "parser"]
