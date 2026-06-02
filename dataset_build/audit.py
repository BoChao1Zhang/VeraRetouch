"""
dataset_build/audit.py
======================
DATAGEN v2 shard contract auditor.

This is a lightweight pre/post-pilot gate. It validates the shard manifest and
per-sample invariants required by docs/plan/dataset/DATAGEN_REFACTOR_planA.md:

- schema/build version match config.
- S1/S7 degrade rows are kind=PARAM, provenance=DEGRADE, with teacher-renderable
  neg params and an audit DegradeSpec.
- S1 region-local rows persist raw pre-blur masks for composite replay.
- LUT rows carry resolved domain/size/sha metadata.
- S5 real JPG rows use after_source=real_jpg and an expert_after_path.

It loads no models and does not render pixels.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence

from dataset_build.contracts import AfterSource, MaskSource, Provenance, RecipeKind, StreamId
from dataset_build.pack import jsonl_to_sample, validate_manifest_index


def audit_dataset(
    out_root: str,
    config: Dict[str, Any],
    *,
    shard_paths: Optional[Sequence[str]] = None,
    strict_paths: bool = True,
    include_manifest: bool = True,
    max_issues: int = 1000,
) -> Dict[str, Any]:
    """Audit DATAGEN v2 manifest + sample-level shard invariants."""
    root = Path(out_root)
    issues: List[Dict[str, Any]] = []
    report: Dict[str, Any] = {
        "out_root": str(root),
        "ok": True,
        "manifest": None,
        "counts": {
            "rows": 0,
            "by_stream": {},
            "by_kind": {},
            "by_provenance": {},
            "region_local": 0,
            "lut": 0,
            "degrade": 0,
            "real_jpg": 0,
        },
        "issues": issues,
    }

    if include_manifest:
        manifest = validate_manifest_index(str(root), config)
        report["manifest"] = manifest
        for item in manifest.get("issues", []):
            _add_issue(issues, "manifest_" + str(item.get("kind")), item, max_issues)

    expected_build = str(config.get("build_version", "v2"))
    expected_schema = str(config.get("schema_version", "datagen_v2"))

    for shard in _resolve_shards(root, shard_paths, config):
        _audit_shard(
            Path(shard),
            expected_build=expected_build,
            expected_schema=expected_schema,
            strict_paths=strict_paths,
            report=report,
            max_issues=max_issues,
        )

    report["ok"] = not issues
    return report


def load_config(path: str) -> Dict[str, Any]:
    try:
        import yaml  # type: ignore
    except Exception as exc:  # pragma: no cover
        raise RuntimeError("PyYAML is required to read config.yaml") from exc
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def _resolve_shards(
    root: Path,
    shard_paths: Optional[Sequence[str]],
    config: Dict[str, Any],
) -> List[Path]:
    if shard_paths:
        return [Path(p) for p in shard_paths]
    storage = (config.get("storage", {}) or {})
    prefix = str(storage.get("shard_prefix", "shard"))
    shards_dir = root / "shards"
    out: List[Path] = []
    for stream_dir in sorted(shards_dir.glob("*")):
        if stream_dir.is_dir():
            out.extend(sorted(stream_dir.glob(f"{prefix}_*.jsonl")))
    return out


def _audit_shard(
    shard: Path,
    *,
    expected_build: str,
    expected_schema: str,
    strict_paths: bool,
    report: Dict[str, Any],
    max_issues: int,
) -> None:
    if not shard.exists():
        _add_issue(report["issues"], "missing_shard", {"path": str(shard)}, max_issues)
        return
    with shard.open("r", encoding="utf-8") as f:
        for line_index, line in enumerate(f):
            line = line.strip()
            if not line:
                continue
            try:
                sample = jsonl_to_sample(line)
            except Exception as exc:
                _add_issue(
                    report["issues"],
                    "invalid_sample_json",
                    {"path": str(shard), "line": line_index, "error": f"{type(exc).__name__}: {exc}"},
                    max_issues,
                )
                continue
            _count_sample(report["counts"], sample)
            _audit_sample(
                sample,
                shard=str(shard),
                line_index=line_index,
                expected_build=expected_build,
                expected_schema=expected_schema,
                strict_paths=strict_paths,
                issues=report["issues"],
                max_issues=max_issues,
            )


def _count_sample(counts: Dict[str, Any], sample: Any) -> None:
    counts["rows"] += 1
    counts["by_stream"][sample.stream.value] = counts["by_stream"].get(sample.stream.value, 0) + 1
    counts["by_kind"][sample.recipe.kind.value] = counts["by_kind"].get(sample.recipe.kind.value, 0) + 1
    counts["by_provenance"][sample.recipe.provenance.value] = (
        counts["by_provenance"].get(sample.recipe.provenance.value, 0) + 1
    )
    if sample.region_local:
        counts["region_local"] += 1
    if sample.recipe.kind == RecipeKind.LUT:
        counts["lut"] += 1
    if sample.recipe.provenance == Provenance.DEGRADE:
        counts["degrade"] += 1
    if sample.after_source == AfterSource.REAL_JPG:
        counts["real_jpg"] += 1


def _audit_sample(
    sample: Any,
    *,
    shard: str,
    line_index: int,
    expected_build: str,
    expected_schema: str,
    strict_paths: bool,
    issues: List[Dict[str, Any]],
    max_issues: int,
) -> None:
    ctx = {"path": shard, "line": line_index, "sample_id": sample.sample_id, "stream": sample.stream.value}
    if str(sample.build_version) != expected_build:
        _add_issue(issues, "sample_build_version_mismatch", {**ctx, "actual": sample.build_version, "expected": expected_build}, max_issues)
    if str(sample.schema_version) != expected_schema:
        _add_issue(issues, "sample_schema_version_mismatch", {**ctx, "actual": sample.schema_version, "expected": expected_schema}, max_issues)
    if sample.recipe.kind == RecipeKind.DEGRADE:
        _add_issue(issues, "legacy_degrade_kind", ctx, max_issues)

    if sample.recipe.provenance == Provenance.DEGRADE:
        _audit_degrade_sample(sample, ctx, strict_paths, issues, max_issues)
    if sample.recipe.kind == RecipeKind.LUT:
        _audit_lut_sample(sample, ctx, strict_paths, issues, max_issues)
    if sample.stream == StreamId.S5_GREYSKY_GLOBAL or sample.after_source == AfterSource.REAL_JPG:
        _audit_s5_after_source(sample, ctx, strict_paths, issues, max_issues)

    if strict_paths:
        _check_existing_path(issues, "missing_source_path", ctx, sample.source_path, max_issues)
        cgt = getattr(sample, "c_gt", None)
        if sample.region_local and cgt is not None:
            _check_existing_path(issues, "missing_cgt_path", ctx, cgt.cgt_path, max_issues)
            if cgt.cgt_patchgrid_path:
                _check_existing_path(issues, "missing_cgt_patchgrid_path", ctx, cgt.cgt_patchgrid_path, max_issues)


def _audit_degrade_sample(
    sample: Any,
    ctx: Dict[str, Any],
    strict_paths: bool,
    issues: List[Dict[str, Any]],
    max_issues: int,
) -> None:
    if sample.recipe.kind != RecipeKind.PARAM:
        _add_issue(issues, "degrade_not_param_kind", {**ctx, "kind": sample.recipe.kind.value}, max_issues)
    if not sample.recipe.params:
        _add_issue(issues, "degrade_missing_neg_params", ctx, max_issues)
    if sample.recipe.degrade is None:
        _add_issue(issues, "degrade_missing_spec", ctx, max_issues)
    if sample.stream not in (StreamId.S1_DEGRADE_LOCAL, StreamId.S7_DEGRADE_GLOBAL):
        _add_issue(issues, "degrade_unexpected_stream", ctx, max_issues)
    cgt = getattr(sample, "c_gt", None)
    if sample.region_local:
        if cgt is None or cgt.mask_source != MaskSource.DEGRADE:
            _add_issue(issues, "s1_mask_source_not_degrade", ctx, max_issues)
        raw_mask = cgt.raw_mask_path if cgt is not None else None
        if not raw_mask:
            _add_issue(issues, "s1_missing_raw_mask", ctx, max_issues)
        elif strict_paths:
            _check_existing_path(issues, "missing_raw_mask_path", ctx, raw_mask, max_issues)
    elif sample.stream == StreamId.S7_DEGRADE_GLOBAL:
        if cgt is not None and cgt.raw_mask_path:
            _add_issue(issues, "s7_unexpected_raw_mask", {**ctx, "raw_mask_path": cgt.raw_mask_path}, max_issues)


def _audit_lut_sample(
    sample: Any,
    ctx: Dict[str, Any],
    strict_paths: bool,
    issues: List[Dict[str, Any]],
    max_issues: int,
) -> None:
    meta = sample.recipe.meta or {}
    for key in ("domain_min", "domain_max", "lut_size", "lut_sha256"):
        if meta.get(key) in (None, "", []):
            _add_issue(issues, "lut_missing_" + key, ctx, max_issues)
    if sample.recipe.provenance != Provenance.LUT:
        _add_issue(issues, "lut_provenance_mismatch", {**ctx, "provenance": sample.recipe.provenance.value}, max_issues)
    if strict_paths and meta.get("path"):
        _check_existing_path(issues, "missing_lut_path", ctx, str(meta["path"]), max_issues)


def _audit_s5_after_source(
    sample: Any,
    ctx: Dict[str, Any],
    strict_paths: bool,
    issues: List[Dict[str, Any]],
    max_issues: int,
) -> None:
    if sample.after_source == AfterSource.REAL_JPG:
        path = sample.meta.get("expert_after_path")
        if not path:
            _add_issue(issues, "real_jpg_missing_expert_after_path", ctx, max_issues)
        elif strict_paths:
            _check_existing_path(issues, "missing_expert_after_path", ctx, str(path), max_issues)
    elif sample.stream == StreamId.S5_GREYSKY_GLOBAL:
        if sample.meta.get("expert_after_path"):
            _add_issue(issues, "s5_expert_path_without_real_jpg_after_source", ctx, max_issues)


def _check_existing_path(
    issues: List[Dict[str, Any]],
    kind: str,
    ctx: Dict[str, Any],
    path: Optional[str],
    max_issues: int,
) -> None:
    if not path:
        return
    if Path(str(path)).name == "_global_ones.png":
        return
    if not Path(str(path)).exists():
        _add_issue(issues, kind, {**ctx, "missing": str(path)}, max_issues)


def _add_issue(
    issues: List[Dict[str, Any]],
    kind: str,
    detail: Dict[str, Any],
    max_issues: int,
) -> None:
    if len(issues) >= max_issues:
        return
    item = {"kind": kind}
    for key, value in detail.items():
        item["detail_kind" if key == "kind" else key] = value
    issues.append(item)


def _main(argv: Optional[Sequence[str]] = None) -> int:
    default_config = str(Path(__file__).with_name("config.yaml"))
    ap = argparse.ArgumentParser(description="Audit DATAGEN v2 shard contracts.")
    ap.add_argument("--config", default=default_config)
    ap.add_argument("--out-root", required=True)
    ap.add_argument("--shard", action="append", help="Specific shard JSONL path; may repeat")
    ap.add_argument("--no-manifest", action="store_true")
    ap.add_argument("--no-strict-paths", action="store_true")
    ap.add_argument("--max-issues", type=int, default=1000)
    args = ap.parse_args(argv)

    cfg = load_config(args.config)
    report = audit_dataset(
        args.out_root,
        cfg,
        shard_paths=args.shard,
        strict_paths=not args.no_strict_paths,
        include_manifest=not args.no_manifest,
        max_issues=max(1, int(args.max_issues)),
    )
    print(json.dumps(report, ensure_ascii=False, sort_keys=True))
    return 0 if report.get("ok") else 1


if __name__ == "__main__":
    raise SystemExit(_main())
