"""LUT render-space pairwise distance, human-rating calibration, and clustering (EPR/A1e).

Renders a fixed deterministic probe-pixel set through every annotated LUT, converts to
CIE Lab, and defines the distance between two LUTs as the mean per-pixel CIEDE2000 of
their rendered probe pixels. Three subcommands:

    distances   render probe pixels for all LUTs, compute all within-``style_major``
                pairs, write ``render_distances.npz`` + manifest
    calibrate   join human ratings (pair questionnaire CSV + pair_key.json) against the
                render distances; emit per-pair table, equal-width bins, Spearman
                correlations, and the largest threshold whose <=d subset has
                ``rating>=3`` fraction <= the configured cap
    cluster     per-``style_major`` average-linkage clustering on the render distances at
                an explicit threshold; emits the ``clusters.*.jsonl`` schema + manifest

Usage:
    python -m dataset_build.tools.lut_render_distance distances \
        --config configs/agent_loop.terra-smoke.toml \
        --out-dir /home/bc/data/scratch/lut_clusters
    python -m dataset_build.tools.lut_render_distance calibrate \
        --csv docs/assets/questionnaire/questionnaire.csv \
        --pair-key docs/assets/lut_cluster_pilot_20260819/questionnaire/pair_key.json
    python -m dataset_build.tools.lut_render_distance cluster --threshold 1.5
"""
from __future__ import annotations

import argparse
import concurrent.futures as futures
import hashlib
import io
import json
import multiprocessing as mp
import sys
import time
import zipfile
from pathlib import Path
from typing import Any, Sequence

import numpy as np

try:
    import tomllib
except ImportError:  # pragma: no cover
    import tomli as tomllib

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from dataset_build.agent_loop.candidates import LutCatalog  # noqa: E402
from dataset_build.agent_loop.config import CatalogConfig  # noqa: E402

RENDER_SPEC = "lut-render-ciede2000-v1"
DEFAULT_PROBES = ("/home/bc/datasets/MMArt-PPR10k/global/230_7/before.jpg",)
DEFAULT_PIXELS = 4096
DEFAULT_PIXEL_SEED = 20260819
DEFAULT_OUT_DIR = Path("/home/bc/data/scratch/lut_clusters")
DISTANCE_NPZ = "render_distances.npz"
DISTANCE_MANIFEST = "render_distances.manifest.json"
PAIR_CHUNK = 128
# C1b item 9: a Spearman rho below this many joined pairs is not a measurement.
SPEARMAN_MIN_PAIRS = 3


# --------------------------------------------------------------------------- helpers


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_json(path: Path, payload: Any) -> None:
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8", newline="\n",
    )


def save_npz_deterministic(path: Path, arrays: dict[str, np.ndarray]) -> None:
    """np.savez-compatible archive with fixed zip metadata (byte-reproducible)."""
    with zipfile.ZipFile(path, "w", compression=zipfile.ZIP_STORED) as archive:
        for name in sorted(arrays):
            buffer = io.BytesIO()
            np.lib.format.write_array(buffer, np.asarray(arrays[name]), allow_pickle=False)
            info = zipfile.ZipInfo(filename=f"{name}.npy", date_time=(1980, 1, 1, 0, 0, 0))
            info.compress_type = zipfile.ZIP_STORED
            info.external_attr = 0o644 << 16
            archive.writestr(info, buffer.getvalue())


def load_catalog(config_path: Path) -> tuple[LutCatalog, Path, Path]:
    with config_path.open("rb") as handle:
        data = tomllib.load(handle)
    databuild = (config_path.parent / str(data["agent_loop"]["databuild_config"])).resolve()
    table = data.get("catalog") or {}
    config = CatalogConfig(
        annotations=Path(str(table["annotations"])).expanduser(),
        global_major_limit=int(table.get("global_major_limit", 8)),
        global_per_major_limit=int(table.get("global_per_major_limit", 4)),
        local_limit=int(table.get("local_limit", 12)),
        reach_limit=int(table.get("reach_limit", 300)),
    )
    return LutCatalog.load(config, databuild), databuild, config.annotations


def features_jsonl_path(databuild: Path) -> Path:
    """``<presets.bank_dir>/features.jsonl`` of one databuild config.

    C1b item 11: `LutCatalog.load` keeps only presets that appear in this file AND
    resolve to an existing renderable path, so the bank content decides which LUTs any
    metric / clustering run even saw. Recording only `annotations_sha256` leaves that
    half of the catalog identity out of every manifest.
    """
    with Path(databuild).open("rb") as handle:
        build = tomllib.load(handle)
    bank_dir = Path(str((build.get("presets") or {}).get("bank_dir") or ""))
    return bank_dir / "features.jsonl"


def features_inputs(databuild: Path) -> dict[str, Any]:
    """`{features_jsonl, features_jsonl_sha256, features_jsonl_rows}` for a manifest."""
    path = features_jsonl_path(databuild)
    if not path.is_file():
        return {
            "features_jsonl": str(path),
            "features_jsonl_sha256": None,
            "features_jsonl_rows": None,
        }
    with path.open("r", encoding="utf-8") as handle:
        rows = sum(1 for line in handle if line.strip())
    return {
        "features_jsonl": str(path),
        "features_jsonl_sha256": sha256_file(path),
        "features_jsonl_rows": rows,
    }


# --------------------------------------------------------------------------- probe set


def probe_pixels(probes: Sequence[Path], pixels: int, seed: int) -> tuple[np.ndarray, list[dict]]:
    """Deterministic RGB float32 pixel pool of shape (pixels, 3) plus per-probe manifest."""
    from PIL import Image

    if not probes:
        raise SystemExit("no probe image given")
    per_probe = [pixels // len(probes)] * len(probes)
    for index in range(pixels - sum(per_probe)):
        per_probe[index] += 1
    chunks: list[np.ndarray] = []
    meta: list[dict] = []
    for order, (probe, count) in enumerate(zip(probes, per_probe)):
        probe = Path(probe).expanduser()
        if not probe.is_file():
            raise SystemExit(f"probe image missing: {probe}")
        with Image.open(probe) as image:
            array = np.asarray(image.convert("RGB"), dtype=np.float32) / 255.0
        flat = array.reshape(-1, 3)
        rng = np.random.default_rng(seed + order)
        if count > flat.shape[0]:
            raise SystemExit(f"probe {probe} has only {flat.shape[0]} pixels, need {count}")
        index = np.sort(rng.choice(flat.shape[0], size=count, replace=False))
        chunks.append(flat[index])
        meta.append({
            "path": str(probe), "sha256": sha256_file(probe),
            "height": int(array.shape[0]), "width": int(array.shape[1]),
            "sampled_pixels": int(count), "pixel_seed": int(seed + order),
            "index_sha256": hashlib.sha256(index.astype(np.int64).tobytes()).hexdigest(),
        })
    return np.concatenate(chunks, axis=0).astype(np.float32), meta


# --------------------------------------------------------------------------- rendering


_PIXELS: np.ndarray | None = None
_LOADER = None
_LAB: np.ndarray | None = None
_PAIRS: dict[str, np.ndarray] = {}


def _init_render(pixels: np.ndarray, databuild: str) -> None:
    global _PIXELS, _LOADER
    from dataset_build.agent_loop.source_reach import configured_lut_loader

    _PIXELS = pixels.reshape(-1, 1, 3)
    _LOADER = configured_lut_loader(Path(databuild))


def _render_one(job: tuple[str, str]) -> tuple[str, np.ndarray | None, str]:
    from skimage.color import rgb2lab

    from dataset_build.src.construct.rendering import apply_lut_cpu_oracle

    preset_id, lut_path = job
    try:
        grid, dmin, dmax = _LOADER.load(Path(lut_path))
        rendered = apply_lut_cpu_oracle(_PIXELS, grid, domain_min=dmin, domain_max=dmax)
        lab = rgb2lab(rendered.reshape(-1, 3).astype(np.float64))
        return (preset_id, lab.astype(np.float32), "")
    except Exception as exc:  # pragma: no cover - reported, never silent
        return (preset_id, None, f"{type(exc).__name__}: {exc}")


def render_all(records: Sequence[Any], pixels: np.ndarray, databuild: Path,
               workers: int) -> tuple[np.ndarray, list[tuple[str, str]]]:
    jobs = [(row.preset_id, row.path) for row in records]
    lab = np.zeros((len(records), pixels.shape[0], 3), dtype=np.float32)
    failures: list[tuple[str, str]] = []
    context = mp.get_context("fork")
    with futures.ProcessPoolExecutor(
        max_workers=workers, mp_context=context,
        initializer=_init_render, initargs=(pixels, str(databuild)),
    ) as pool:
        for index, (preset_id, values, detail) in enumerate(
            pool.map(_render_one, jobs, chunksize=8)
        ):
            if values is None:
                failures.append((preset_id, detail))
                continue
            lab[index] = values
    return lab, failures


# --------------------------------------------------------------------------- distances


def _pair_chunk(job: tuple[str, int, int]) -> np.ndarray:
    from skimage.color import deltaE_ciede2000

    major, start, stop = job
    index = _PAIRS[major][start:stop]
    left = _LAB[index[:, 0]].astype(np.float64)
    right = _LAB[index[:, 1]].astype(np.float64)
    delta = deltaE_ciede2000(left.reshape(-1, 3), right.reshape(-1, 3))
    return delta.reshape(index.shape[0], -1).mean(axis=1).astype(np.float32)


def condensed_pairs(count: int) -> np.ndarray:
    """Row indices (i, j), i < j, in scipy condensed (squareform) order."""
    rows, cols = np.triu_indices(count, k=1)
    return np.stack([rows, cols], axis=1).astype(np.int64)


def compute_distances(lab: np.ndarray, records: Sequence[Any],
                      workers: int) -> tuple[list[str], list[int], np.ndarray, np.ndarray]:
    """Return (major names, per-major record counts, global row offsets, condensed dists)."""
    order: dict[str, list[int]] = {}
    for index, row in enumerate(records):
        order.setdefault(row.style_major, []).append(index)
    majors = sorted(order)
    pairs: dict[str, np.ndarray] = {}
    jobs: list[tuple[str, int, int]] = []
    for major in majors:
        local = np.asarray(order[major], dtype=np.int64)
        local_pairs = condensed_pairs(local.shape[0])
        pairs[major] = local[local_pairs] if local_pairs.size else local_pairs.reshape(0, 2)
        for start in range(0, pairs[major].shape[0], PAIR_CHUNK):
            jobs.append((major, start, min(start + PAIR_CHUNK, pairs[major].shape[0])))
    # Published to module globals so forked workers inherit them copy-on-write
    # instead of pickling the ~200 MB Lab cache once per worker.
    global _LAB, _PAIRS
    _LAB, _PAIRS = lab, pairs
    context = mp.get_context("fork")
    chunks: list[np.ndarray] = []
    with futures.ProcessPoolExecutor(max_workers=workers, mp_context=context) as pool:
        for values in pool.map(_pair_chunk, jobs, chunksize=1):
            chunks.append(values)
    flat = (np.concatenate(chunks) if chunks else np.zeros(0, dtype=np.float32))
    counts = [len(order[major]) for major in majors]
    offsets = np.zeros(len(majors) + 1, dtype=np.int64)
    for index, major in enumerate(majors):
        offsets[index + 1] = offsets[index] + pairs[major].shape[0]
    return majors, counts, offsets, flat


class DistanceStore:
    """Reader for ``render_distances.npz``."""

    def __init__(self, path: Path) -> None:
        with np.load(path, allow_pickle=False) as data:
            self.preset_ids = [str(value) for value in data["preset_ids"]]
            self.style_majors = [str(value) for value in data["style_majors"]]
            self.majors = [str(value) for value in data["majors"]]
            self.major_counts = data["major_counts"].astype(np.int64)
            self.offsets = data["cond_offsets"].astype(np.int64)
            self.distances = data["distances"].astype(np.float64)
        self.path = path
        self.rank: dict[str, dict[str, int]] = {}
        for major in self.majors:
            members = [pid for pid, mj in zip(self.preset_ids, self.style_majors) if mj == major]
            self.rank[major] = {pid: index for index, pid in enumerate(members)}
        self.major_index = {major: index for index, major in enumerate(self.majors)}

    def condensed(self, major: str) -> np.ndarray:
        index = self.major_index[major]
        return self.distances[self.offsets[index]:self.offsets[index + 1]]

    def members(self, major: str) -> list[str]:
        table = self.rank[major]
        return [pid for pid, _ in sorted(table.items(), key=lambda item: item[1])]

    def distance(self, preset_a: str, preset_b: str) -> float | None:
        major_a = self.style_majors[self.preset_ids.index(preset_a)] \
            if preset_a in self.preset_ids else None
        major_b = self.style_majors[self.preset_ids.index(preset_b)] \
            if preset_b in self.preset_ids else None
        if major_a is None or major_a != major_b:
            return None
        count = int(self.major_counts[self.major_index[major_a]])
        i = self.rank[major_a][preset_a]
        j = self.rank[major_a][preset_b]
        if i == j:
            return 0.0
        if i > j:
            i, j = j, i
        flat = count * i - (i * (i + 1)) // 2 + (j - i - 1)
        return float(self.condensed(major_a)[flat])


# --------------------------------------------------------------------------- subcommands


def cmd_distances(args: argparse.Namespace) -> int:
    started = time.time()
    catalog, databuild, annotations = load_catalog(args.config)
    records = list(catalog.records)
    pixels, probe_meta = probe_pixels(
        [Path(value) for value in args.probe], args.pixels, args.pixel_seed
    )
    render_started = time.time()
    lab, failures = render_all(records, pixels, databuild, args.workers)
    render_seconds = time.time() - render_started
    if failures:
        raise SystemExit(f"render failures ({len(failures)}): {failures[:5]}")

    pair_started = time.time()
    majors, counts, offsets, flat = compute_distances(lab, records, args.workers)
    pair_seconds = time.time() - pair_started

    args.out_dir.mkdir(parents=True, exist_ok=True)
    npz_path = args.out_dir / DISTANCE_NPZ
    save_npz_deterministic(npz_path, {
        "preset_ids": np.asarray([row.preset_id for row in records], dtype="<U64"),
        "style_majors": np.asarray([row.style_major for row in records], dtype="<U64"),
        "majors": np.asarray(majors, dtype="<U64"),
        "major_counts": np.asarray(counts, dtype=np.int64),
        "cond_offsets": offsets,
        "distances": flat,
    })
    quantiles = np.quantile(flat, [0.05, 0.25, 0.50, 0.75, 0.95]) if flat.size else np.zeros(5)
    manifest = {
        "schema": "lut-render-distances-v1",
        "render_spec": RENDER_SPEC,
        "distance": {
            "metric": "mean per-pixel CIEDE2000 (skimage.color.deltaE_ciede2000, kL=kC=kH=1)",
            "color_space": "sRGB -> CIE Lab (skimage.color.rgb2lab, D65)",
            "scope": "per style_major, all within-major pairs",
        },
        "probe": {
            "images": probe_meta,
            "total_pixels": int(pixels.shape[0]),
            "pixel_seed": int(args.pixel_seed),
            "sampling": "np.random.default_rng(seed+probe_order).choice(no replacement), sorted",
        },
        "inputs": {
            "agent_loop_config": str(args.config),
            "databuild_config": str(databuild),
            "annotations": str(annotations),
            "annotations_sha256": sha256_file(annotations),
            # C1b item 11: the preset bank decides which LUTs the catalog contains.
            **features_inputs(databuild),
        },
        "records": {"clustered": len(records), "style_majors": len(majors)},
        "pairs": {
            "count": int(flat.size),
            "quantiles": {
                "p5": round(float(quantiles[0]), 6), "p25": round(float(quantiles[1]), 6),
                "p50": round(float(quantiles[2]), 6), "p75": round(float(quantiles[3]), 6),
                "p95": round(float(quantiles[4]), 6),
            },
            "min": round(float(flat.min()), 6) if flat.size else 0.0,
            "max": round(float(flat.max()), 6) if flat.size else 0.0,
        },
        "artifacts": {
            "distances": {"path": str(npz_path), "sha256": sha256_file(npz_path)},
        },
    }
    manifest_path = args.out_dir / DISTANCE_MANIFEST
    write_json(manifest_path, manifest)
    print(json.dumps({
        "npz": str(npz_path), "npz_sha256": manifest["artifacts"]["distances"]["sha256"],
        "manifest": str(manifest_path), "manifest_sha256": sha256_file(manifest_path),
        "records": len(records), "majors": len(majors), "pairs": int(flat.size),
        "quantiles": manifest["pairs"]["quantiles"],
        "seconds": {
            "render": round(render_seconds, 2), "pairs": round(pair_seconds, 2),
            "total": round(time.time() - started, 2),
        },
    }, ensure_ascii=False, indent=2))
    return 0


def read_ratings(paths: Sequence[Path]) -> dict[str, int]:
    """Read ``<id>,rating[,notes]``; the id column is ``pair_id`` or ``item_id``.

    C1b item 8: the questionnaire builders emit ``item_id`` and
    ``lut_pair_questionnaire.read_ratings`` has always accepted both spellings. This
    reader accepted only ``pair_id``, so an ``item_id`` CSV silently produced zero
    ratings; a file with neither column now fails loud instead.
    """
    import csv

    ratings: dict[str, int] = {}
    for path in paths:
        with path.open("r", encoding="utf-8-sig", newline="") as handle:
            reader = csv.DictReader(handle)
            fields = set(reader.fieldnames or ())
            if not fields & {"pair_id", "item_id"}:
                raise SystemExit(f"{path}: no pair_id/item_id column")
            for row in reader:
                key = row.get("pair_id")
                if key is None or not str(key).strip():
                    key = row.get("item_id")
                pair_id = str(key or "").strip()
                raw = str(row.get("rating") or "").strip()
                if not pair_id or not raw:
                    continue
                if pair_id in ratings:
                    raise SystemExit(f"duplicate rating for {pair_id}")
                ratings[pair_id] = int(float(raw))
    return ratings


def spearman(x: Sequence[float], y: Sequence[float]) -> tuple[float, float]:
    from scipy.stats import spearmanr

    result = spearmanr(np.asarray(x, dtype=np.float64), np.asarray(y, dtype=np.float64))
    return float(result.statistic), float(result.pvalue)


def cmd_calibrate(args: argparse.Namespace) -> int:
    started = time.time()
    store = DistanceStore(args.distances or (args.out_dir / DISTANCE_NPZ))
    ratings = read_ratings(args.csv)
    key = json.loads(args.pair_key.read_text(encoding="utf-8"))
    pairs = key["pairs"]

    rows: list[dict[str, Any]] = []
    missing: list[str] = []
    for pair_id in sorted(ratings):
        entry = pairs.get(pair_id)
        if entry is None:
            missing.append(pair_id)
            continue
        render = store.distance(str(entry["preset_a"]), str(entry["preset_b"]))
        if render is None:
            missing.append(pair_id)
            continue
        rows.append({
            "pair_id": pair_id,
            "style_major": str(entry.get("style_major") or ""),
            "preset_a": str(entry["preset_a"]), "preset_b": str(entry["preset_b"]),
            "feature_dist": round(float(entry["distance"]), 6),
            "render_dist": round(render, 6),
            "rating": int(ratings[pair_id]),
        })

    render_values = [row["render_dist"] for row in rows]
    feature_values = [row["feature_dist"] for row in rows]
    rating_values = [row["rating"] for row in rows]

    bins: list[dict[str, Any]] = []
    if rows:
        low, high = min(render_values), max(render_values)
        span = (high - low) or 1.0
        edges = [low + span * index / args.bins for index in range(args.bins + 1)]
        edges[-1] = high
        for index in range(args.bins):
            lo, hi = edges[index], edges[index + 1]
            if index == args.bins - 1:
                members = [row for row in rows if lo <= row["render_dist"] <= hi]
            else:
                members = [row for row in rows if lo <= row["render_dist"] < hi]
            grades = [row["rating"] for row in members]
            bins.append({
                "bin": index, "low": round(lo, 6), "high": round(hi, 6), "n": len(members),
                "mean_rating": round(float(np.mean(grades)), 4) if grades else None,
                "frac_rating_ge3": (
                    round(float(np.mean([g >= 3 for g in grades])), 4) if grades else None
                ),
            })

    # C1b item 9: a Spearman rho needs at least three joined pairs. The old fallback
    # invented `(rho=0.0, p=1.0)`, which is indistinguishable from a real measured null
    # result. Under the floor both fields are null and `insufficient_pairs` says so.
    insufficient_pairs = len(rows) < SPEARMAN_MIN_PAIRS
    if insufficient_pairs:
        render_rho = render_p = feature_rho = feature_p = None
    else:
        render_rho, render_p = spearman(render_values, rating_values)
        feature_rho, feature_p = spearman(feature_values, rating_values)

    sweep: list[dict[str, Any]] = []
    threshold: float | None = None
    for candidate in sorted(set(render_values)):
        subset = [row["rating"] for row in rows if row["render_dist"] <= candidate]
        if len(subset) < args.min_pairs:
            continue
        frac = float(np.mean([grade >= 3 for grade in subset]))
        sweep.append({"d": round(candidate, 6), "n": len(subset), "frac_rating_ge3":
                      round(frac, 4)})
        if frac <= args.max_frac:
            threshold = candidate

    payload = {
        "schema": "lut-render-calibration-v1",
        "render_spec": RENDER_SPEC,
        "inputs": {
            "distances": str(store.path), "distances_sha256": sha256_file(store.path),
            "pair_key": str(args.pair_key), "pair_key_sha256": sha256_file(args.pair_key),
            "csv": [str(path) for path in args.csv],
            "csv_sha256": [sha256_file(path) for path in args.csv],
        },
        "counts": {"rated": len(ratings), "joined": len(rows), "missing": missing},
        "pairs": rows,
        "bins": bins,
        "spearman": {
            "render_dist_vs_rating": {
                "rho": None if render_rho is None else round(render_rho, 4),
                "p": None if render_p is None else round(render_p, 6),
            },
            "feature_dist_vs_rating": {
                "rho": None if feature_rho is None else round(feature_rho, 4),
                "p": None if feature_p is None else round(feature_p, 6),
            },
            "n": len(rows),
            "min_pairs": SPEARMAN_MIN_PAIRS,
            "insufficient_pairs": insufficient_pairs,
        },
        "threshold_rule": {
            "criterion": f"max d with frac(rating>=3 | render_dist<=d) <= {args.max_frac}",
            "min_pairs": args.min_pairs,
            "recommended_threshold": (round(threshold, 6) if threshold is not None else None),
            "sweep": sweep,
        },
    }
    args.out_dir.mkdir(parents=True, exist_ok=True)
    out_path = args.out_dir / args.out_name
    write_json(out_path, payload)
    print(json.dumps({
        "calibration": str(out_path), "calibration_sha256": sha256_file(out_path),
        "joined": len(rows), "missing": missing,
        "spearman": payload["spearman"],
        "bins": bins,
        "recommended_threshold": payload["threshold_rule"]["recommended_threshold"],
        "seconds": round(time.time() - started, 2),
    }, ensure_ascii=False, indent=2))
    return 0


def cmd_cluster(args: argparse.Namespace) -> int:
    from scipy.cluster.hierarchy import fcluster, linkage

    started = time.time()
    store = DistanceStore(args.distances or (args.out_dir / DISTANCE_NPZ))
    rows: list[tuple[str, str, str]] = []
    per_major: dict[str, dict[str, int]] = {}
    sizes: list[int] = []
    for major in store.majors:
        members = store.members(major)
        if len(members) == 1:
            buckets = {members[0]: [members[0]]}
        else:
            link = linkage(store.condensed(major), method="average")
            labels = fcluster(link, t=args.threshold, criterion="distance").astype(np.int64)
            grouped: dict[int, list[str]] = {}
            for label, preset_id in zip(labels.tolist(), members):
                grouped.setdefault(label, []).append(preset_id)
            buckets = {}
            for group in grouped.values():
                group = sorted(group)
                buckets[group[0]] = group
        group_sizes = [len(group) for group in buckets.values()]
        sizes.extend(group_sizes)
        per_major[major] = {
            "records": len(members), "clusters": len(buckets),
            "singletons": sum(1 for size in group_sizes if size == 1),
            "max_cluster_size": max(group_sizes),
        }
        for cluster_id, group in buckets.items():
            for preset_id in group:
                rows.append((preset_id, major, cluster_id))
    rows.sort()

    args.out_dir.mkdir(parents=True, exist_ok=True)
    tag = args.tag or f"render_t{args.threshold:g}"
    path = args.out_dir / f"clusters.{tag}.jsonl"
    path.write_text("".join(
        json.dumps({"preset_id": preset_id, "style_major": major, "cluster_id": cluster_id},
                   ensure_ascii=False, sort_keys=True, separators=(",", ":")) + "\n"
        for preset_id, major, cluster_id in rows
    ), encoding="utf-8", newline="\n")

    manifest = {
        "schema": "lut-render-clusters-v1",
        "render_spec": RENDER_SPEC,
        "linkage": {"method": "average", "metric": RENDER_SPEC, "scope": "per style_major",
                    "threshold_scope": "global"},
        "threshold": args.threshold,
        "inputs": {
            "distances": str(store.path), "distances_sha256": sha256_file(store.path),
        },
        "records": len(rows),
        "clusters": len(sizes),
        "singletons": sum(1 for size in sizes if size == 1),
        "max_cluster_size": max(sizes) if sizes else 0,
        "mean_cluster_size": round(float(np.mean(sizes)), 4) if sizes else 0.0,
        "per_major": per_major,
        "artifacts": {"clusters": {"path": str(path), "sha256": sha256_file(path),
                                   "lines": len(rows)}},
    }
    manifest_path = args.out_dir / f"clusters.{tag}.manifest.json"
    write_json(manifest_path, manifest)
    print(json.dumps({
        "clusters": str(path), "clusters_sha256": manifest["artifacts"]["clusters"]["sha256"],
        "manifest": str(manifest_path), "manifest_sha256": sha256_file(manifest_path),
        "threshold": args.threshold, "total_clusters": len(sizes),
        "singletons": manifest["singletons"], "max_cluster_size": manifest["max_cluster_size"],
        "seconds": round(time.time() - started, 2),
    }, ensure_ascii=False, indent=2))
    return 0


# --------------------------------------------------------------------------- cli


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    sub = parser.add_subparsers(dest="command", required=True)

    dist = sub.add_parser("distances", help="render probe pixels and compute all pair distances")
    dist.add_argument("--config", type=Path,
                      default=REPO_ROOT / "configs/agent_loop.terra-smoke.toml")
    dist.add_argument("--probe", action="append", default=None,
                      help=f"probe image (repeatable); default {DEFAULT_PROBES[0]}")
    dist.add_argument("--pixels", type=int, default=DEFAULT_PIXELS)
    dist.add_argument("--pixel-seed", type=int, default=DEFAULT_PIXEL_SEED)
    dist.add_argument("--workers", type=int, default=24)
    dist.set_defaults(func=cmd_distances)

    cal = sub.add_parser("calibrate", help="join human ratings against the render distances")
    cal.add_argument("--csv", type=Path, action="append", required=True)
    cal.add_argument("--pair-key", type=Path, required=True)
    cal.add_argument("--distances", type=Path, default=None)
    cal.add_argument("--bins", type=int, default=8)
    cal.add_argument("--max-frac", type=float, default=0.20)
    cal.add_argument("--min-pairs", type=int, default=1)
    cal.add_argument("--out-name", type=str, default="render_calibration.json")
    cal.set_defaults(func=cmd_calibrate)

    clu = sub.add_parser("cluster", help="average-linkage clustering at an explicit threshold")
    clu.add_argument("--threshold", type=float, required=True)
    clu.add_argument("--distances", type=Path, default=None)
    clu.add_argument("--tag", type=str, default=None)
    clu.set_defaults(func=cmd_cluster)

    args = parser.parse_args(argv)
    if args.command == "distances" and not args.probe:
        args.probe = list(DEFAULT_PROBES)
    return args


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    return int(args.func(args))


if __name__ == "__main__":
    raise SystemExit(main())
