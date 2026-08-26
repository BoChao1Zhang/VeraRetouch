"""EPR-034 batch pipeline: build the per-source `source_response_v2` cache.

For every source in the manifest this applies the **whole** LUT bank to a stratified
2048-pixel probe set at full strength once, caches the RGB result, and synthesises every
alpha point by linear RGB mixing (`render.py:147`). Nothing re-renders a LUT.

Usage
-----
    python -m dataset_build.tools.build_source_response_v2 \
        --sources /home/bc/data/agent_loop/local-v1/sources_full.jsonl \
        --out-dir /home/bc/data/agent_loop/local-v1/source_response_v2 \
        --device cuda:0 --limit 2

Manifest shapes
---------------
Two manifest shapes are consumed, and both resolve to the same ``source_sha256`` (so the
cache identity and the sampler seed do not depend on which manifest was used):

* the annotated manifests (``sources5k.annotated-v34.jsonl`` and friends) carry
  ``source_annotation_path``; the sha is read from that JSON's ``.source_sha256``;
* the full manifest ``sources_full.jsonl`` carries only
  ``{scene, source_id, source_path, subject, subject_path}``; the sha is read from the
  materialisation index ``materialized_full/index.jsonl`` (``--source-index``) by
  ``source_id``.

Reads only. The writes are the per-source cache files under ``--out-dir`` plus the
per-shard ``failures.jsonl`` sidecar.
"""
from __future__ import annotations

import argparse
import gzip
import json
import os
import time
from pathlib import Path
from typing import Any, Iterator, Sequence

import numpy as np
from dataclasses import dataclass
from PIL import Image, ImageOps

try:
    import tomllib
except ImportError:  # pragma: no cover
    import tomli as tomllib

from dataset_build.agent_loop.candidates import LutCatalog, LutRecord
from dataset_build.agent_loop.config import CatalogConfig
from dataset_build.agent_loop.models import GLOBAL_DELTA_E_TARGETS
from dataset_build.agent_loop.source_reach import catalog_digest
from dataset_build.agent_loop.source_response_v2 import (
    ALPHA_SEARCH_STEPS, RESPONSE_REVISION_V2, SAMPLER_REVISION_V2, SAMPLE_PIXELS_V2,
    SOURCE_RESPONSE_SCHEMA, VALIDATOR_CALLS, SourceResponseError, apply_lut_batch,
    assert_validators_wired, get_backend, measure_response, mix_alpha, response_row,
    rgb2lab, sample_probe_pixels, solve_alpha_hat, validate_sampling_report,
    validate_source_response,
)

DEFAULT_AGENT_CONFIG = Path("configs/agent_loop.local-v2-b12.toml")
DEFAULT_OUT_DIR = Path("/home/bc/data/agent_loop/local-v1/source_response_v2")
DEFAULT_SOURCE_INDEX = Path(
    "/home/bc/data/agent_loop/local-v1/materialized_full/index.jsonl"
)
# CLAUDE.md NFS discipline: reads go through the soft `/mnt/nfs-ro` mount, never through
# the hard `/mnt/nfs` mount. The manifest paths resolve into the hard mount.
NFS_WRITE_ROOT = "/mnt/nfs/"
NFS_READ_ROOT = "/mnt/nfs-ro/"


# --- inputs -------------------------------------------------------------------------
def read_path(path: str | Path) -> Path:
    resolved = Path(os.path.realpath(Path(path).expanduser()))
    text = str(resolved)
    if text.startswith(NFS_WRITE_ROOT):
        return Path(NFS_READ_ROOT + text[len(NFS_WRITE_ROOT):])
    return resolved


def load_rgb(path: Path) -> np.ndarray:
    with Image.open(path) as image:
        return np.asarray(
            ImageOps.exif_transpose(image).convert("RGB"), dtype=np.float32
        ) / 255.0


def load_matte(path: Path, size: tuple[int, int]) -> np.ndarray:
    with Image.open(path) as image:
        alpha = ImageOps.exif_transpose(image).convert("L")
        if alpha.size != size:
            alpha = alpha.resize(size, getattr(Image, "Resampling", Image).BILINEAR)
        return np.asarray(alpha, dtype=np.float32) / 255.0


def iter_manifest(path: Path, offset: int, limit: int | None) -> Iterator[dict[str, Any]]:
    seen = 0
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            if seen < offset:
                seen += 1
                continue
            if limit is not None and seen - offset >= limit:
                return
            seen += 1
            yield json.loads(line)


@dataclass(frozen=True, slots=True)
class SourceIdentity:
    """Everything a manifest row must resolve to before any pixel is read."""

    source_id: str
    source_sha256: str
    subject_sha256: str
    source_path: str
    subject_path: str | None
    origin: str            # which lookup produced `source_sha256`


class SourceIndex:
    """`source_id` -> materialisation record, loaded lazily from `index.jsonl`."""

    def __init__(self, path: Path) -> None:
        self.path = path
        self._rows: dict[str, dict[str, Any]] | None = None

    def _load(self) -> dict[str, dict[str, Any]]:
        if self._rows is None:
            rows: dict[str, dict[str, Any]] = {}
            resolved = read_path(self.path)
            if not resolved.is_file():
                raise SourceResponseError(
                    f"manifest rows carry no source_annotation_path and the source "
                    f"index {resolved} does not exist; pass --source-index"
                )
            with resolved.open("r", encoding="utf-8") as handle:
                for line in handle:
                    if not line.strip():
                        continue
                    row = json.loads(line)
                    rows[str(row["source_id"])] = row
            self._rows = rows
        return self._rows

    def get(self, source_id: str) -> dict[str, Any]:
        rows = self._load()
        try:
            return rows[source_id]
        except KeyError as exc:
            raise SourceResponseError(
                f"{source_id} is absent from the source index {self.path}"
            ) from exc


def resolve_identity(row: dict[str, Any], index: SourceIndex) -> SourceIdentity:
    """Resolve one manifest row to its frozen sha identity.

    `source_annotation_path` wins when present (the annotated manifests), otherwise the
    materialisation index is consulted by `source_id` (the full manifest). The two
    routes are byte-identical on the sources that carry both.
    """
    annotation_path = row.get("source_annotation_path")
    if annotation_path:
        annotation = json.loads(read_path(annotation_path).read_text("utf-8"))
        source_sha256 = str(annotation["source_sha256"])
        subject_sha256 = str(annotation.get("subject_sha256") or "")
        source_id = str(row.get("source_id") or annotation.get("source_id") or "")
        source_path = str(row.get("source_path") or annotation.get("source_path") or "")
        subject_path = row.get("subject_path") or annotation.get("subject_path")
        origin = "source_annotation_path"
    else:
        source_id = str(row.get("source_id") or "")
        if not source_id:
            raise SourceResponseError(
                "manifest row carries neither source_annotation_path nor source_id"
            )
        entry = index.get(source_id)
        source_sha256 = str(entry["source_sha256"])
        subject_sha256 = str(entry.get("subject_sha256") or "")
        source_path = str(row.get("source_path") or entry.get("source_path") or "")
        subject_path = row.get("subject_path") or entry.get("subject_path")
        origin = "source_index"
    if len(source_sha256) != 64:
        raise SourceResponseError(f"{source_id} resolved to a malformed source_sha256")
    if not source_path:
        raise SourceResponseError(f"{source_id} resolved to no source_path")
    return SourceIdentity(
        source_id=source_id, source_sha256=source_sha256,
        subject_sha256=subject_sha256, source_path=source_path,
        subject_path=str(subject_path) if subject_path else None, origin=origin,
    )


def load_catalog(agent_config: Path) -> tuple[LutCatalog, Path]:
    with agent_config.open("rb") as handle:
        agent = tomllib.load(handle)
    catalog_table = agent.get("catalog") or {}
    databuild = (agent_config.parent / str(
        (agent.get("agent_loop") or {})["databuild_config"]
    )).resolve()
    config = CatalogConfig(
        annotations=Path(str(catalog_table["annotations"])),
        global_major_limit=int(catalog_table.get("global_major_limit", 3)),
        global_per_major_limit=int(catalog_table.get("global_per_major_limit", 7)),
        local_limit=int(catalog_table.get("local_limit", 9)),
        reach_limit=int(catalog_table.get("reach_limit", 300)),
        cluster_artifact=(Path(str(catalog_table["cluster_artifact"]))
                          if catalog_table.get("cluster_artifact") else None),
        segment_fingerprints=(Path(str(catalog_table["segment_fingerprints"]))
                              if catalog_table.get("segment_fingerprints") else None),
    )
    return LutCatalog.load(config, databuild), databuild


# --- LUT bank on device -------------------------------------------------------------
class LutBank:
    """Every catalog LUT grid, grouped by cube size, resident on the target device."""

    def __init__(self, xp: Any, records: Sequence[LutRecord], databuild_config: Path) -> None:
        from dataset_build.agent_loop.source_reach import configured_lut_loader

        loader = configured_lut_loader(databuild_config)
        self.preset_ids = [row.preset_id for row in records]
        grids: dict[int, list[int]] = {}
        raw: list[np.ndarray] = []
        for index, row in enumerate(records):
            grid, domain_min, domain_max = loader.load(Path(row.path))
            if not (np.allclose(domain_min, 0.0) and np.allclose(domain_max, 1.0)):
                raise SourceResponseError(
                    f"{row.preset_id} carries a non-identity LUT domain; the packed "
                    "bank is expected to be [0,1]^3"
                )
            raw.append(np.asarray(grid, dtype=np.float32))
            grids.setdefault(int(grid.shape[0]), []).append(index)
        self.xp = xp
        self.groups: list[tuple[int, np.ndarray, Any]] = []
        for size in sorted(grids):
            members = np.asarray(grids[size], dtype=np.int64)
            stacked = np.stack([raw[i].reshape(-1, 3) for i in members], axis=0)
            self.groups.append((size, members, xp.asarray(stacked)))
        self.count = len(records)

    def apply_full(self, pixels: Any, chunk: int) -> Iterator[tuple[np.ndarray, Any]]:
        """Yield ``(catalog_indices, rendered)`` with rendered of shape (B, N, 3)."""
        for size, members, table in self.groups:
            for start in range(0, members.size, chunk):
                block = members[start:start + chunk]
                yield block, apply_lut_batch(
                    self.xp, pixels, table[start:start + chunk], size
                )


# --- per-source pipeline ------------------------------------------------------------
def build_source(
    xp: Any, bank: LutBank, row: dict[str, Any], identity: SourceIdentity, *, chunk: int,
    alpha_steps: int = ALPHA_SEARCH_STEPS,
) -> tuple[dict[str, Any], dict[str, float]]:
    clock = time.perf_counter()
    source_sha256 = identity.source_sha256
    rgb = load_rgb(read_path(identity.source_path))
    matte = None
    matte_error = None
    if identity.subject_path:
        try:
            matte = load_matte(
                read_path(identity.subject_path), (rgb.shape[1], rgb.shape[0])
            )
        except (FileNotFoundError, OSError, ValueError) as exc:
            matte_error = f"{type(exc).__name__}: {exc}"
    timings = {"read_seconds": time.perf_counter() - clock, "sample_seconds": 0.0,
               "lut_seconds": 0.0, "measure_seconds": 0.0, "bins_seconds": 0.0}

    clock = time.perf_counter()
    sample = sample_probe_pixels(rgb, matte, source_sha256)
    report = dict(sample.report)
    if matte_error:
        report["matte_error"] = matte_error
    validate_sampling_report(report)
    timings["sample_seconds"] = time.perf_counter() - clock

    before = xp.asarray(sample.pixels)
    before_lab = rgb2lab(xp, before)
    presets: dict[str, dict[str, Any]] = {}
    lut_clock = time.perf_counter()
    for block, full in bank.apply_full(before, chunk):
        timings["lut_seconds"] += time.perf_counter() - lut_clock
        started = time.perf_counter()
        measured = measure_response(xp, before, before_lab, full)
        timings["measure_seconds"] += time.perf_counter() - started
        d_full = np.asarray(measured["dE00"], dtype=np.float64)
        started = time.perf_counter()
        bins: dict[str, list[dict[str, Any] | None]] = {}
        for name, (low, high, inclusive) in GLOBAL_DELTA_E_TARGETS.items():
            center = (low + high) / 2.0
            alpha, achieved = solve_alpha_hat(
                xp, before, before_lab, full, center, steps=alpha_steps
            )
            mixed = mix_alpha(
                xp, before, full, xp.asarray(alpha.astype(np.float32)).reshape(-1, 1, 1)
            )
            bin_measured = measure_response(xp, before, before_lab, mixed)
            rows: list[dict[str, Any] | None] = []
            for position in range(int(block.size)):
                if d_full[position] < low:
                    rows.append(None)
                    continue
                value = float(achieved[position])
                in_band = low <= value <= high if inclusive else low <= value < high
                rows.append({
                    "alpha_hat": round(float(alpha[position]), 6),
                    "achieved_dE00": round(value, 5),
                    "status": "measured" if in_band else "off_band",
                    "response": response_row(bin_measured, position),
                })
            bins[name] = rows
        timings["bins_seconds"] += time.perf_counter() - started
        for position, catalog_index in enumerate(block.tolist()):
            preset_id = bank.preset_ids[catalog_index]
            entry_bins: dict[str, Any] = {}
            for name in GLOBAL_DELTA_E_TARGETS:
                value = bins[name][position]
                entry_bins[name] = value if value is not None else {
                    "alpha_hat": None, "achieved_dE00": None,
                    "status": "unreachable", "response": None,
                }
            presets[preset_id] = {
                "full_strength": response_row(measured, position),
                "bins": entry_bins,
            }
        lut_clock = time.perf_counter()
    payload = {
        "schema": SOURCE_RESPONSE_SCHEMA,
        "sampler_revision": SAMPLER_REVISION_V2,
        "response_revision": RESPONSE_REVISION_V2,
        "source_id": identity.source_id,
        "source_sha256": source_sha256,
        "subject_sha256": identity.subject_sha256,
        "identity_origin": identity.origin,
        "scene": str(row.get("scene") or ""),
        "requested_pixels": SAMPLE_PIXELS_V2,
        "alpha_search_steps": int(alpha_steps),
        "bin_targets": {name: list(target)
                        for name, target in GLOBAL_DELTA_E_TARGETS.items()},
        "sampling": report,
        "preset_count": len(presets),
        "presets": presets,
    }
    return payload, timings


def cache_path(out_dir: Path, source_sha256: str) -> Path:
    return out_dir / f"source_response_v2.{source_sha256[:16]}.json.gz"


def write_payload(out_dir: Path, payload: dict[str, Any]) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    target = cache_path(out_dir, payload["source_sha256"])
    temporary = target.with_suffix(".tmp")
    with gzip.open(temporary, "wt", encoding="utf-8", compresslevel=6) as handle:
        json.dump(payload, handle, separators=(",", ":"), allow_nan=False)
    temporary.replace(target)
    return target


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sources", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument("--config", type=Path, default=DEFAULT_AGENT_CONFIG)
    parser.add_argument(
        "--source-index", type=Path, default=DEFAULT_SOURCE_INDEX,
        help="materialisation index used to resolve source_sha256 for manifest rows "
             "that carry no source_annotation_path (loaded lazily)",
    )
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--offset", type=int, default=0)
    # Chunking only bounds device memory; the payload is byte-identical across chunk
    # sizes (verified 512 / 2048 / 4051 in the EPR-034 smoke).
    parser.add_argument("--preset-chunk", type=int, default=2048)
    parser.add_argument("--alpha-steps", type=int, default=ALPHA_SEARCH_STEPS)
    parser.add_argument(
        "--overwrite", action="store_true",
        help="recompute sources whose cache file already exists (default: skip them, "
             "so a shard can be resumed after a cancel)",
    )
    parser.add_argument("--report", type=Path, default=None)
    parser.add_argument(
        "--failures", type=Path, default=None,
        help="per-shard sidecar for sources that raised (default: "
             "<out-dir>/failures.offset<offset>.jsonl). A bad source is recorded and "
             "skipped; it never takes the shard down, and it is never silent.",
    )
    args = parser.parse_args(argv)

    # Runtime assertion wiring: both pre-registered validators must run and reject
    # before a single source is touched, and the call counters must keep rising.
    wired = assert_validators_wired()
    print(f"[startup] validators wired: {wired}", flush=True)

    catalog, databuild = load_catalog(args.config.resolve())
    records = tuple(sorted(catalog.records, key=lambda item: item.preset_id))
    digest = catalog_digest(records)
    print(f"[startup] catalog presets={len(records)} sha256={digest}", flush=True)

    xp = get_backend(args.device)
    started = time.perf_counter()
    bank = LutBank(xp, records, databuild)
    bank_seconds = time.perf_counter() - started
    print(f"[startup] LUT bank loaded in {bank_seconds:.1f}s "
          f"groups={[(s, int(m.size)) for s, m, _ in bank.groups]}", flush=True)

    peak_bytes = 0
    peak_reserved = 0
    if xp.is_torch:
        xp.mod.cuda.reset_peak_memory_stats()

    source_index = SourceIndex(args.source_index)
    failures_path = args.failures or (
        args.out_dir / f"failures.offset{args.offset}.jsonl"
    )

    rows: list[dict[str, Any]] = []
    failures: list[dict[str, Any]] = []
    skipped = 0
    origins: dict[str, int] = {}
    for index, row in enumerate(iter_manifest(args.sources, args.offset, args.limit)):
        source_id = str(row.get("source_id") or "")
        # A single bad source must not take the shard down, and must not be silent:
        # it is printed, counted into the summary, and appended to `failures.jsonl`.
        try:
            identity = resolve_identity(row, source_index)
            origins[identity.origin] = origins.get(identity.origin, 0) + 1
            if not args.overwrite:
                existing = cache_path(args.out_dir, identity.source_sha256)
                if existing.is_file():
                    skipped += 1
                    print(f"[{index}] {source_id} exists -> skip", flush=True)
                    continue
            before_calls = dict(VALIDATOR_CALLS)
            wall = time.perf_counter()
            payload, timings = build_source(
                xp, bank, row, identity,
                chunk=args.preset_chunk, alpha_steps=args.alpha_steps,
            )
            payload["catalog_sha256"] = digest
            payload["device"] = str(args.device)
            validate_source_response(payload)
            if (VALIDATOR_CALLS["sampling"] <= before_calls["sampling"]
                    or VALIDATOR_CALLS["schema"] <= before_calls["schema"]):
                raise SourceResponseError(
                    f"validators did not run for {source_id}; refusing to write"
                )
            serialize_clock = time.perf_counter()
            target = write_payload(args.out_dir, payload)
            timings["serialize_seconds"] = time.perf_counter() - serialize_clock
            elapsed = time.perf_counter() - wall
        except Exception as exc:  # noqa: BLE001 - recorded, counted, never swallowed
            record = {
                "index": index, "source_id": source_id,
                "source_path": str(row.get("source_path") or ""),
                "error": f"{type(exc).__name__}: {exc}",
            }
            failures.append(record)
            failures_path.parent.mkdir(parents=True, exist_ok=True)
            with failures_path.open("a", encoding="utf-8") as handle:
                handle.write(json.dumps(record, ensure_ascii=False) + "\n")
            print(f"[{index}] {source_id} FAILED -> {record['error']}", flush=True)
            continue
        if xp.is_torch:
            peak_bytes = max(peak_bytes, int(xp.mod.cuda.max_memory_allocated()))
            peak_reserved = max(peak_reserved, int(xp.mod.cuda.max_memory_reserved()))
        size = target.stat().st_size
        rows.append({
            "source_id": source_id, "source_sha256": payload["source_sha256"],
            "path": str(target), "bytes": int(size), "seconds": round(elapsed, 3),
            "preset_count": payload["preset_count"],
            "subject_share": payload["sampling"]["subject_share"],
            "matte_status": payload["sampling"]["matte_status"],
            **{key: round(value, 3) for key, value in timings.items()},
        })
        print(f"[{index}] {source_id} {elapsed:.2f}s {size} bytes -> {target}", flush=True)

    summary = {
        "sources": len(rows),
        "skipped_existing": skipped,
        "failed": len(failures),
        "failures_path": str(failures_path) if failures else None,
        "identity_origins": origins,
        "catalog_sha256": digest,
        "preset_count": len(records),
        "device": args.device,
        "bank_load_seconds": round(bank_seconds, 2),
        "gpu_peak_allocated_bytes": peak_bytes,
        "gpu_peak_reserved_bytes": peak_reserved,
        "gpu_max_memory_used_bytes": (
            int(xp.mod.cuda.mem_get_info()[1] - xp.mod.cuda.mem_get_info()[0])
            if xp.is_torch else 0
        ),
        "validator_calls": dict(VALIDATOR_CALLS),
        "failures": failures,
        "rows": rows,
    }
    print(json.dumps(summary, indent=2), flush=True)
    if args.report:
        args.report.parent.mkdir(parents=True, exist_ok=True)
        args.report.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
