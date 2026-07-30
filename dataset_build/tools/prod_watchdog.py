#!/usr/bin/env python
"""Append one progress line per interval for a running production build.

The build itself only rewrites ``manifest.json`` at phase boundaries and land
checkpoints, so a remote observer polling the manifest alone cannot tell a slow
phase from a dead process.  This reads the manifest for phase/status and counts
the journals directly for live progress, adds the host facts that decide whether
to intervene (process alive, GPU load, tmpfs headroom) and appends one JSON
object per sample to a JSONL file.

Usage:
    prod_watchdog.py --output-root DIR --pid PID --status-file FILE
                     [--gpu-index N] [--interval SECONDS]
"""
from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import time
from collections import Counter
from pathlib import Path
from typing import Any


def _count_lines(path: Path) -> int:
    if not path.is_file():
        return 0
    total = 0
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            total += chunk.count(b"\n")
    return total


def _failure_summary(path: Path, tail_bytes: int = 4 * 1024 * 1024) -> dict[str, Any]:
    """Terminal/retryable counts plus the error-code histogram of the tail.

    The whole file is counted for totals; the histogram reads only the tail so a
    multi-hundred-megabyte journal cannot make the watchdog the slow part.
    """
    if not path.is_file():
        return {"lines": 0, "terminal": 0, "recent_codes": {}}
    size = path.stat().st_size
    terminal = 0
    lines = 0
    codes: Counter[str] = Counter()
    with open(path, "rb") as handle:
        for raw in handle:
            lines += 1
            if b'"terminal": true' in raw or b'"terminal":true' in raw:
                terminal += 1
        handle.seek(max(0, size - tail_bytes))
        if size > tail_bytes:
            handle.readline()
        for raw in handle:
            try:
                row = json.loads(raw)
            except ValueError:
                continue
            code = str(row.get("error_code") or "")
            if code:
                codes[code] += 1
    return {"lines": lines, "terminal": terminal, "recent_codes": dict(codes.most_common(8))}


def _gpu(index: int | None) -> dict[str, Any]:
    if index is None:
        return {}
    try:
        out = subprocess.run(
            [
                "nvidia-smi",
                f"--id={index}",
                "--query-gpu=utilization.gpu,memory.used",
                "--format=csv,noheader,nounits",
            ],
            capture_output=True, text=True, timeout=30, check=True,
        ).stdout.strip()
        util, used = (part.strip() for part in out.split(","))
        return {"gpu_util_pct": int(util), "gpu_mem_used_mib": int(used)}
    except Exception as exc:  # noqa: BLE001 - monitoring must not die on nvidia-smi
        return {"gpu_error": f"{type(exc).__name__}: {exc}"}


def _tmpfs(path: Path) -> dict[str, Any]:
    try:
        stat = os.statvfs(path)
        return {
            "tmpfs_free_gib": round(stat.f_bavail * stat.f_frsize / 1024**3, 2),
            "staged_gib": round(
                sum(f.stat().st_size for f in path.rglob("*") if f.is_file()) / 1024**3, 2
            ),
        }
    except Exception as exc:  # noqa: BLE001
        return {"tmpfs_error": f"{type(exc).__name__}: {exc}"}


def _alive(pid: int) -> bool:
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    return True


_STREAM_EVENT = re.compile(rb"untyped_stream_event")


def sample(output_root: Path, pid: int, gpu_index: int | None) -> dict[str, Any]:
    manifest_path = output_root / "manifest.json"
    manifest: dict[str, Any] = {}
    if manifest_path.is_file():
        try:
            manifest = json.loads(manifest_path.read_text())
        except ValueError:
            manifest = {"parse_error": True}
    completed = manifest.get("completed") or {}
    failures = _failure_summary(output_root / "failures.jsonl")
    row: dict[str, Any] = {
        "ts": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "build_id": manifest.get("build_id"),
        "phase": manifest.get("phase"),
        "status": manifest.get("status"),
        "alive": _alive(pid),
        "pid": pid,
        # Journals are the live counters; the manifest only refreshes at phase
        # boundaries and land checkpoints.
        "groups_journal": _count_lines(output_root / "groups.jsonl"),
        "sft_journal": _count_lines(output_root / "sft.jsonl"),
        "manifest_groups": completed.get("groups"),
        "manifest_sft": completed.get("sft"),
        "manifest_global": completed.get("global"),
        "manifest_local": completed.get("local"),
        "winner_abstained": completed.get("winner_abstained"),
        "failures": failures,
        "annotation": (manifest.get("annotation") or {}).get("pending"),
        "annotation_terminal": (manifest.get("annotation") or {}).get("terminal_failures"),
        "sam3": (manifest.get("sam3_relabel") or {}).get("terminal")
        if isinstance(manifest.get("sam3_relabel"), dict) else None,
    }
    row.update(_gpu(gpu_index))
    row.update(_tmpfs(output_root))
    return row


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--pid", type=int, required=True)
    parser.add_argument("--status-file", type=Path, required=True)
    parser.add_argument("--gpu-index", type=int, default=None)
    parser.add_argument("--interval", type=float, default=300.0)
    args = parser.parse_args(argv)

    args.status_file.parent.mkdir(parents=True, exist_ok=True)
    while True:
        row = sample(args.output_root, args.pid, args.gpu_index)
        with open(args.status_file, "a") as handle:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
            handle.flush()
        if not row["alive"]:
            # One final sample is written above, then the watchdog retires with
            # the build rather than polling a dead pid forever.
            return 0
        time.sleep(args.interval)


if __name__ == "__main__":
    raise SystemExit(main())
