"""Dry-run-first quarantine cleanup with an explicit deletion manifest."""
from __future__ import annotations

import hashlib
import json
import time
from pathlib import Path
from typing import Any

from .artifacts import ArtifactStore
from .persistence import AuditStore


def plan_cleanup(
    audit: AuditStore, artifacts: ArtifactStore, *, rejected_days: int,
    now: float | None = None,
) -> dict[str, Any]:
    cutoff = float(now if now is not None else time.time()) - rejected_days * 86400
    rows = [
        row for row in audit.artifacts_with_retention("quarantine")
        if float(row["created_at"]) < cutoff
    ]
    targets = [{
        "sha256": row["sha256"], "uri": row["uri"],
        "media_type": row["media_type"], "size": int(row["size"]),
    } for row in sorted(rows, key=lambda item: item["sha256"])]
    body = {
        "schema": "local-retouch-cleanup-v1", "dry_run": True,
        "artifact_root": str(artifacts.root), "cutoff": cutoff,
        "targets": targets,
    }
    body["manifest_sha256"] = hashlib.sha256(json.dumps(
        body, sort_keys=True, separators=(",", ":")
    ).encode()).hexdigest()
    return body


def write_cleanup_manifest(path: Path, manifest: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(manifest, sort_keys=True, indent=2), encoding="utf-8")


def apply_cleanup_manifest(
    path: Path, audit: AuditStore, artifacts: ArtifactStore
) -> dict[str, Any]:
    manifest = json.loads(path.read_text(encoding="utf-8"))
    digest = manifest.pop("manifest_sha256", None)
    actual = hashlib.sha256(json.dumps(
        manifest, sort_keys=True, separators=(",", ":")
    ).encode()).hexdigest()
    if digest != actual or manifest.get("schema") != "local-retouch-cleanup-v1":
        raise ValueError("cleanup manifest integrity check failed")
    if manifest.get("dry_run") is not True:
        raise ValueError("cleanup requires a dry-run manifest")
    if Path(manifest.get("artifact_root", "")).resolve() != artifacts.root:
        raise ValueError("cleanup manifest targets another artifact root")
    current = {
        row["sha256"]: row for row in audit.artifacts_with_retention("quarantine")
    }
    removed = []
    for target in manifest.get("targets", []):
        sha256 = str(target["sha256"])
        row = current.get(sha256)
        if row is None or row["retention"] != "quarantine":
            continue
        try:
            blob = artifacts.path_for(sha256)
        except FileNotFoundError:
            audit.mark_artifact_purged(sha256)
            removed.append(sha256)
            continue
        if blob.stat().st_size != int(target["size"]):
            raise ValueError(f"artifact size changed after dry run: {sha256}")
        blob.unlink()
        audit.mark_artifact_purged(sha256)
        removed.append(sha256)
    return {"manifest_sha256": digest, "removed": removed, "count": len(removed)}


__all__ = ["apply_cleanup_manifest", "plan_cleanup", "write_cleanup_manifest"]
