"""B12 item 2: derive the v2 segment-fingerprint artifact (v1 + LUT histogram response).

The v1 table is a pure re-read of numbers that already live in the closed-v1
annotations. v2 keeps every v1 field byte for byte and adds one measured group per
LUT: the L* bin-share response of a frozen probe-pixel population.

Method, in full:

* Probe pixels: ``HISTOGRAM_PROBE_NAMES`` cached probe images (the frozen long-edge-768
  probe cache written by ``tools/lut_reannotate/pipeline.py probes``), ``PIXELS_PER_PROBE``
  deterministic pixels each, drawn by ``lut_render_distance.probe_pixels`` with the
  frozen ``DEFAULT_PIXEL_SEED``. Exactly the same probe-pixel machinery the render
  distance / clustering artifacts use.
* ``l_bins_in``: ``source_histogram.l_bin_shares`` of those pixels' L*.
* ``l_bins_out``: the same 8 bin shares after the pixels are pushed through the LUT with
  the packed-LUT CPU oracle (``lut_render_distance.render_all``, which already returns
  CIE Lab).
* ``delta`` / ``d_shadow`` / ``d_mid`` / ``d_high``:
  ``segment_fingerprints.derive_histogram_response``.

Deterministic: same annotations SHA + same probe SHA -> byte-identical output.

Usage::

    python -m dataset_build.tools.build_lut_histogram_fingerprints \
        --out /home/bc/data/scratch/lut_reannotate/out/segment_fingerprints.v2.jsonl
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from pathlib import Path
from typing import Any, Sequence

import numpy as np
from skimage.color import rgb2lab

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from dataset_build.agent_loop.candidates import _format_of  # noqa: E402
from dataset_build.agent_loop.lut_annotations import file_sha256  # noqa: E402
from dataset_build.agent_loop.segment_fingerprints import (  # noqa: E402
    DEFAULT_ANNOTATIONS, DEFAULT_FINGERPRINTS_V2, HISTOGRAM_AGGREGATES,
    HISTOGRAM_DERIVATION_REVISION, HISTOGRAM_GROUPS, SegmentFingerprintError,
    build_row, derive_histogram_response,
)
from dataset_build.agent_loop.source_histogram import (  # noqa: E402
    L_BIN_COUNT, l_bin_shares,
)
from dataset_build.tools.lut_render_distance import (  # noqa: E402
    DEFAULT_PIXEL_SEED, probe_pixels, render_all, sha256_file, write_json,
)

try:
    import tomllib
except ImportError:  # pragma: no cover
    import tomli as tomllib


# Frozen probe set of the histogram response. Four of the six bank probes: the two
# tonally decisive ones (neutral, skin) plus one warm and one cool chromatic probe.
PROBE_CACHE_DIR = Path("/home/bc/data/scratch/lut_reannotate/probes")
HISTOGRAM_PROBE_NAMES: tuple[str, ...] = ("neutral", "skin", "red", "blue")
PIXELS_PER_PROBE = 4096
DEFAULT_DATABUILD = REPO_ROOT / "databuild.prod-l8-local400k-20260812.toml"


class _Job:
    """Minimal `render_all` record: it only reads `preset_id` and `path`."""

    __slots__ = ("preset_id", "path")

    def __init__(self, preset_id: str, path: str) -> None:
        self.preset_id = preset_id
        self.path = path


def probe_paths() -> list[Path]:
    paths = [PROBE_CACHE_DIR / f"before_{name}.png" for name in HISTOGRAM_PROBE_NAMES]
    missing = [str(path) for path in paths if not path.is_file()]
    if missing:
        raise SystemExit(f"probe cache images missing: {missing}")
    return paths


def lut_paths(databuild: Path) -> dict[str, str]:
    """`preset_id -> renderable LUT path` from the preset bank's features.jsonl."""
    with databuild.open("rb") as handle:
        build = tomllib.load(handle)
    bank_dir = Path(str((build.get("presets") or {}).get("bank_dir") or ""))
    result: dict[str, str] = {}
    with (bank_dir / "features.jsonl").open("r", encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            row = json.loads(line)
            preset_id = str(row.get("preset_id") or "")
            path = str(row.get("path") or "")
            if preset_id and _format_of(row) == "lut" and Path(path).is_file():
                result[preset_id] = path
    if not result:
        raise SystemExit(f"no renderable LUT in {bank_dir / 'features.jsonl'}")
    return result


def read_annotations(path: Path) -> dict[str, dict[str, Any]]:
    """`preset_id -> hsl_features` for every successful closed-v1 annotation."""
    result: dict[str, dict[str, Any]] = {}
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            record = json.loads(line)
            if record.get("ok") is not True:
                continue
            preset_id = str(record.get("preset_id") or record.get("key") or "")
            if not preset_id:
                raise SystemExit(f"{path}:{line_number}: missing preset_id")
            if preset_id in result:
                raise SystemExit(f"{path}: duplicate preset_id {preset_id}")
            features = record.get("hsl_features")
            if not isinstance(features, dict):
                raise SystemExit(f"{preset_id}: missing hsl_features")
            result[preset_id] = features
    if not result:
        raise SystemExit(f"{path}: no successful annotation")
    return result


def quantiles(values: Sequence[float]) -> dict[str, float]:
    array = np.asarray(values, dtype=np.float64)
    if array.size == 0:
        return {}
    marks = (0.0, 0.05, 0.25, 0.50, 0.75, 0.95, 1.0)
    names = ("min", "p05", "p25", "p50", "p75", "p95", "max")
    return {
        name: round(float(np.quantile(array, mark)), 6)
        for name, mark in zip(names, marks)
    }


def build(args: argparse.Namespace) -> int:
    started = time.time()
    annotations_path = Path(args.annotations)
    annotations_sha = file_sha256(annotations_path)
    features = read_annotations(annotations_path)
    paths = lut_paths(Path(args.databuild))

    probes = probe_paths()
    pixels, probe_meta = probe_pixels(
        probes, PIXELS_PER_PROBE * len(probes), args.pixel_seed
    )
    probe_spec = {
        "histogram_revision": HISTOGRAM_DERIVATION_REVISION,
        "probes": probe_meta,
        "pixels_per_probe": PIXELS_PER_PROBE,
        "total_pixels": int(pixels.shape[0]),
        "pixel_seed": int(args.pixel_seed),
        "l_bin_count": L_BIN_COUNT,
        "groups": {name: list(group) for name, group in HISTOGRAM_GROUPS.items()},
        "pixels_sha256": hashlib.sha256(
            np.ascontiguousarray(pixels, dtype=np.float32).tobytes()
        ).hexdigest(),
    }
    probe_sha = hashlib.sha256(json.dumps(
        probe_spec, ensure_ascii=False, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")).hexdigest()

    covered = sorted(pid for pid in features if pid in paths)
    uncovered = sorted(pid for pid in features if pid not in paths)
    if not covered:
        raise SystemExit("no annotated preset resolves to a renderable LUT path")

    jobs = [_Job(pid, paths[pid]) for pid in covered]
    render_started = time.time()
    lab, failures = render_all(jobs, pixels, Path(args.databuild), args.workers)
    render_seconds = time.time() - render_started
    if failures:
        raise SystemExit(f"render failures ({len(failures)}): {failures[:5]}")

    l_bins_in = l_bin_shares(
        np.asarray(
            rgb2lab(pixels.reshape(-1, 1, 3).astype(np.float64)), dtype=np.float64
        ).reshape(-1, 3)[:, 0]
    )

    rows: dict[str, dict[str, Any]] = {}
    for index, preset_id in enumerate(covered):
        histogram = derive_histogram_response(
            l_bins_in, l_bin_shares(lab[index][:, 0].astype(np.float64))
        )
        rows[preset_id] = build_row(
            preset_id, features[preset_id], source_sha256=annotations_sha,
            histogram=histogram, probe_sha256=probe_sha,
        )
    if not rows:
        raise SegmentFingerprintError("no v2 row was produced")

    payload = b"".join(
        json.dumps(
            rows[preset_id], ensure_ascii=False, sort_keys=True, separators=(",", ":")
        ).encode("utf-8") + b"\n"
        for preset_id in sorted(rows)
    )
    output = Path(args.out)
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_name(f".{output.name}.tmp")
    temporary.write_bytes(payload)
    temporary.replace(output)

    distribution = {
        name: quantiles([row["histogram"][name] for row in rows.values()])
        for name in HISTOGRAM_AGGREGATES
    }
    distribution.update({
        f"delta[{index}]": quantiles(
            [row["histogram"]["delta"][index] for row in rows.values()]
        )
        for index in range(L_BIN_COUNT)
    })
    report = {
        "schema": "lut-segment-fingerprint-v2-report",
        "histogram_revision": HISTOGRAM_DERIVATION_REVISION,
        "inputs": {
            "annotations": str(annotations_path),
            "annotations_sha256": annotations_sha,
            "databuild_config": str(args.databuild),
            "annotated_ok": len(features),
            "annotated_without_renderable_path": len(uncovered),
        },
        "probe": {**probe_spec, "probe_sha256": probe_sha},
        "derivation": {
            "l_bins_in": l_bins_in,
            "groups": {name: list(group) for name, group in HISTOGRAM_GROUPS.items()},
            "delta": "l_bins_out - l_bins_in, per L* bin, sums to 0",
        },
        "output": {
            "path": str(output), "sha256": sha256_file(output),
            "records": len(rows), "bytes": len(payload),
        },
        "distribution": distribution,
        "seconds": {
            "render": round(render_seconds, 2), "total": round(time.time() - started, 2),
        },
    }
    report_path = Path(args.report) if args.report else output.with_suffix(".report.json")
    write_json(report_path, report)
    print(json.dumps({
        "output": report["output"], "report": str(report_path),
        "records": len(rows), "uncovered": len(uncovered),
        "probe_sha256": probe_sha,
        "distribution": {name: distribution[name] for name in HISTOGRAM_AGGREGATES},
        "seconds": report["seconds"],
    }, ensure_ascii=False, indent=2))
    return 0


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--annotations", type=Path, default=DEFAULT_ANNOTATIONS)
    parser.add_argument("--databuild", type=Path, default=DEFAULT_DATABUILD)
    parser.add_argument("--out", type=Path, default=DEFAULT_FINGERPRINTS_V2)
    parser.add_argument("--report", type=Path, default=None)
    parser.add_argument("--pixel-seed", type=int, default=DEFAULT_PIXEL_SEED)
    parser.add_argument("--workers", type=int, default=24)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    return build(parse_args(argv))


if __name__ == "__main__":
    raise SystemExit(main())
