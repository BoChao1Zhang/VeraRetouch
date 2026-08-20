"""P1 open-source perceptual-metric candidates on the 200 human-rated LUT pairs (A1h).

Scores four published perceptual distance definitions against the same 200 ratings, folds
and working point that ``lut_metric_ab`` (task A1f) used, so the numbers sit in one table
with the existing render-distance baselines.  No AUC anywhere.

Candidates
    M1  CAM16-UCS       per-pixel ``colour.difference.delta_E_CAM16UCS`` on the same sparse
                        4-probe x 4096-pixel set A1f used; mean and p95 columns
    M2  S-CIELAB-style  opponent-space spatial filtering of the rendered 256px images,
                        then per-pixel CIEDE2000; mean and p95 columns
    M3  LPIPS (AlexNet) ``lpips.LPIPS(net='alex')`` on the 4 rendered 256px image pairs,
                        mean over the 4 probes
    M4  ColorVideoVDP   ``pycvvdp.cvvdp(display_name='standard_fhd')`` on the same 4 image
                        pairs, mean JOD over probes; the reported score is ``10 - JOD``
                        so that, like every other column, larger == more different
Baselines (recomputed here from the A1f per-pair table, not re-derived)
    B1  render_mean_1probe   the current production distance
    B1b render_mean_4probe   the A1f 4-probe mean CIEDE2000

Evaluation is byte-for-byte the A1f protocol: label ``rating >= 3``, the stored A1f fold
assignment, working point = largest training-fold threshold whose ``rating>=3`` fraction is
<= ``MAX_FRAC`` over >= ``MIN_PAIRS`` training pairs, test folds report coverage, realised
``rating>=3`` violation rate and ``rating>=4`` rate.  Both constants are imported from
``lut_metric_ab``; a start-up assertion re-derives the published B1 row (rho 0.6344,
coverage 11, violation 0.3636) and aborts if this file's harness does not reproduce it.

Usage:
    python -m dataset_build.tools.lut_metric_p1 \
        --config configs/agent_loop.terra-smoke.toml \
        --out-dir /home/bc/data/scratch/lut_clusters/metric_ab/p1_candidates
"""
from __future__ import annotations

import argparse
import concurrent.futures as futures
import json
import multiprocessing as mp
import sys
import time
from pathlib import Path
from typing import Any, Sequence

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from dataset_build.tools.lut_metric_ab import (  # noqa: E402
    MAX_FRAC,
    MIN_PAIRS,
    DEFAULT_PROBES,
    working_threshold,
)
from dataset_build.tools.lut_render_distance import (  # noqa: E402
    features_inputs,
    load_catalog,
    probe_pixels,
    save_npz_deterministic,
    sha256_file,
    spearman,
    write_json,
)

METRIC_SPEC = "lut-metric-p1-v1"
DEFAULT_OUT_DIR = Path("/home/bc/data/scratch/lut_clusters/metric_ab/p1_candidates")
DEFAULT_LABELED = Path("/home/bc/data/scratch/lut_clusters/metric_ab/labeled_pairs.json")

# A1f published B1 row; the harness must reproduce it before anything else runs.
A1F_B1_EXPECTED = {"rho": 0.6344, "coverage_n": 11, "violation_rate_ge3": 0.3636}

PAIR_CHUNK = 8

# --- S-CIELAB (Zhang & Wandell 1996) opponent transform and filter parameters ---------
# Transcribed offline (no external lookup was permitted for this task card); the filter
# spreads are interpreted as plain Gaussian standard deviations in degrees of visual angle
# -- this is the "simplified Gaussian approximation" the task card allows, NOT the original
# MATLAB ``gauss(halfwidth, support)`` parameterisation.
XYZ_TO_OPPONENT = np.array([
    [0.279, 0.720, -0.107],
    [-0.449, 0.290, -0.077],
    [0.086, -0.590, 0.501],
], dtype=np.float64)
SCIELAB_FILTERS = (
    ((0.921, 0.105, -0.108), (0.0283, 0.133, 4.336)),   # luminance
    ((0.531, 0.330), (0.0392, 0.494)),                  # red-green
    ((0.488, 0.371), (0.0536, 0.386)),                  # blue-yellow
)


# --------------------------------------------------------------------------- inputs


def read_labeled_pairs(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    pairs = payload["pairs"]
    return {
        "pair_ids": [row["pair_id"] for row in pairs],
        "ratings": np.asarray([int(row["rating"]) for row in pairs], dtype=np.int64),
        "labels": np.asarray([int(row["label_distinguishable"]) for row in pairs],
                             dtype=np.int64),
        "assignment": np.asarray([int(row["fold"]) for row in pairs], dtype=np.int64),
        "preset_a": [row["preset_a"] for row in pairs],
        "preset_b": [row["preset_b"] for row in pairs],
        "style_major": [row["style_major"] for row in pairs],
        "render_mean_1probe": np.asarray([float(row["render_mean_1probe"]) for row in pairs]),
        "render_mean_4probe": np.asarray([float(row["render_mean"]) for row in pairs]),
        "render_p95_4probe": np.asarray([float(row["render_p95"]) for row in pairs]),
    }


def full_frames(probes: Sequence[Path], short_side: int) -> tuple[np.ndarray, list[dict]]:
    """Flat float32 RGB pool of every probe resized to ``short_side`` plus a manifest."""
    from PIL import Image

    chunks: list[np.ndarray] = []
    meta: list[dict] = []
    offset = 0
    for probe in probes:
        probe = Path(probe).expanduser()
        if not probe.is_file():
            raise SystemExit(f"probe image missing: {probe}")
        with Image.open(probe) as image:
            image = image.convert("RGB")
            width, height = image.size
            scale = short_side / min(width, height)
            size = (max(1, round(width * scale)), max(1, round(height * scale)))
            resized = image.resize(size, Image.LANCZOS)
            array = np.asarray(resized, dtype=np.float32) / 255.0
        flat = array.reshape(-1, 3)
        chunks.append(flat)
        meta.append({
            "path": str(probe), "sha256": sha256_file(probe),
            "source_height": int(height), "source_width": int(width),
            "height": int(array.shape[0]), "width": int(array.shape[1]),
            "offset": offset, "pixels": int(flat.shape[0]), "resample": "PIL.LANCZOS",
        })
        offset += flat.shape[0]
    return np.concatenate(chunks, axis=0).astype(np.float32), meta


# --------------------------------------------------------------------------- rendering


_PIXELS: np.ndarray | None = None
_LOADER = None


def _init_render(pixels: np.ndarray, databuild: str) -> None:
    global _PIXELS, _LOADER
    from dataset_build.agent_loop.source_reach import configured_lut_loader

    _PIXELS = pixels.reshape(-1, 1, 3)
    _LOADER = configured_lut_loader(Path(databuild))


def _render_rgb(job: tuple[str, str]) -> tuple[str, np.ndarray | None, str]:
    from dataset_build.src.construct.rendering import apply_lut_cpu_oracle

    preset_id, lut_path = job
    try:
        grid, dmin, dmax = _LOADER.load(Path(lut_path))
        rendered = apply_lut_cpu_oracle(_PIXELS, grid, domain_min=dmin, domain_max=dmax)
        return (preset_id, rendered.reshape(-1, 3).astype(np.float32), "")
    except Exception as exc:  # pragma: no cover - reported, never silent
        return (preset_id, None, f"{type(exc).__name__}: {exc}")


def render_presets(preset_paths: Sequence[tuple[str, str]], pixels: np.ndarray,
                   databuild: Path, workers: int) -> np.ndarray:
    """(n_presets, n_pixels, 3) float32 sRGB, rounded through float16 for cache parity."""
    out = np.zeros((len(preset_paths), pixels.shape[0], 3), dtype=np.float32)
    failures: list[tuple[str, str]] = []
    context = mp.get_context("fork")
    with futures.ProcessPoolExecutor(
        max_workers=workers, mp_context=context,
        initializer=_init_render, initargs=(pixels, str(databuild)),
    ) as pool:
        for index, (preset_id, values, detail) in enumerate(
            pool.map(_render_rgb, preset_paths, chunksize=4)
        ):
            if values is None:
                failures.append((preset_id, detail))
                continue
            out[index] = values.astype(np.float16).astype(np.float32)
    if failures:
        raise SystemExit(f"render failures ({len(failures)}): {failures[:5]}")
    return out


# --------------------------------------------------------------------------- M1 CAM16


def cam16ucs_all(rgb: np.ndarray) -> np.ndarray:
    """sRGB -> CAM16-UCS J'a'b' for every preset (n, pixels, 3)."""
    import colour

    out = np.zeros_like(rgb)
    for index in range(rgb.shape[0]):
        out[index] = colour.convert(
            rgb[index].astype(np.float64), "sRGB", "CAM16UCS"
        ).astype(np.float32)
    return out


_ARR_A: np.ndarray | None = None
_IDX: np.ndarray | None = None


def _chunk_cam16(job: tuple[int, int]) -> np.ndarray:
    from colour.difference import delta_E_CAM16UCS

    start, stop = job
    index = _IDX[start:stop]
    left = _ARR_A[index[:, 0]].astype(np.float64)
    right = _ARR_A[index[:, 1]].astype(np.float64)
    delta = delta_E_CAM16UCS(
        left.reshape(-1, 3), right.reshape(-1, 3)
    ).reshape(index.shape[0], -1)
    return np.stack([delta.mean(axis=1), np.percentile(delta, 95.0, axis=1)],
                    axis=1).astype(np.float64)


def _chunk_de00(job: tuple[int, int]) -> np.ndarray:
    from skimage.color import deltaE_ciede2000

    start, stop = job
    index = _IDX[start:stop]
    left = _ARR_A[index[:, 0]].astype(np.float64)
    right = _ARR_A[index[:, 1]].astype(np.float64)
    delta = deltaE_ciede2000(
        left.reshape(-1, 3), right.reshape(-1, 3)
    ).reshape(index.shape[0], -1)
    return np.stack([delta.mean(axis=1), np.percentile(delta, 95.0, axis=1)],
                    axis=1).astype(np.float64)


def pair_reduce(values: np.ndarray, index: np.ndarray, worker_fn, workers: int) -> np.ndarray:
    global _ARR_A, _IDX
    _ARR_A, _IDX = values, index
    jobs = [(start, min(start + PAIR_CHUNK, index.shape[0]))
            for start in range(0, index.shape[0], PAIR_CHUNK)]
    context = mp.get_context("fork")
    chunks: list[np.ndarray] = []
    with futures.ProcessPoolExecutor(max_workers=workers, mp_context=context) as pool:
        for out in pool.map(worker_fn, jobs, chunksize=1):
            chunks.append(out)
    _ARR_A, _IDX = None, None
    return np.concatenate(chunks)


# --------------------------------------------------------------------------- M2 S-CIELAB


_SC_RGB: np.ndarray | None = None
_SC_FRAMES: Sequence[dict] = ()
_SC_PPD: float = 30.0


def _scielab_one(position: int) -> tuple[int, np.ndarray, int, int]:
    """Filtered CIELAB for one preset plus (clipped, total) negative-XYZ counts."""
    import colour
    from scipy.ndimage import gaussian_filter
    from skimage.color import xyz2lab

    inverse = np.linalg.inv(XYZ_TO_OPPONENT)
    out = np.zeros((_SC_RGB.shape[1], 3), dtype=np.float32)
    clipped, total = 0, 0
    for frame in _SC_FRAMES:
        start, stop = frame["offset"], frame["offset"] + frame["pixels"]
        image = _SC_RGB[position, start:stop].reshape(frame["height"], frame["width"], 3)
        xyz = colour.sRGB_to_XYZ(image.astype(np.float64))
        opponent = xyz @ XYZ_TO_OPPONENT.T
        filtered = np.empty_like(opponent)
        for channel, (weights, spreads) in enumerate(SCIELAB_FILTERS):
            plane = opponent[:, :, channel]
            accumulated = np.zeros_like(plane)
            for weight, spread in zip(weights, spreads):
                accumulated += weight * gaussian_filter(
                    plane, sigma=float(spread) * _SC_PPD, mode="nearest", truncate=3.0
                )
            filtered[:, :, channel] = accumulated / float(sum(weights))
        back = filtered @ inverse.T
        clipped += int(np.sum(back < 0.0))
        total += int(back.size)
        out[start:stop] = xyz2lab(np.clip(back, 0.0, None)).reshape(-1, 3).astype(np.float32)
    return position, out, clipped, total


def scielab_lab(rgb: np.ndarray, frames: Sequence[dict], ppd: float,
                workers: int) -> tuple[np.ndarray, dict[str, Any]]:
    """Opponent-space spatially filtered CIELAB for every preset (n, pixels, 3)."""
    global _SC_RGB, _SC_FRAMES, _SC_PPD
    _SC_RGB, _SC_FRAMES, _SC_PPD = rgb, frames, ppd
    out = np.zeros_like(rgb)
    clipped, total = 0, 0
    context = mp.get_context("fork")
    with futures.ProcessPoolExecutor(max_workers=workers, mp_context=context) as pool:
        for position, values, one_clipped, one_total in pool.map(
            _scielab_one, range(rgb.shape[0]), chunksize=1
        ):
            out[position] = values
            clipped += one_clipped
            total += one_total
    _SC_RGB = None
    nonfinite = int(np.sum(~np.isfinite(out)))
    if nonfinite:
        raise SystemExit(f"S-CIELAB produced {nonfinite} non-finite values")
    return out, {
        "negative_xyz_components_clipped": clipped,
        "negative_xyz_components_total": total,
        "negative_xyz_clipped_frac": round(clipped / total, 6) if total else None,
        "nonfinite_lab_values": nonfinite,
    }


# --------------------------------------------------------------------------- M3/M4


def lpips_scores(rgb: np.ndarray, index: np.ndarray, frames: Sequence[dict],
                 device: str, batch: int) -> np.ndarray:
    import lpips
    import torch

    torch.manual_seed(0)
    model = lpips.LPIPS(net="alex").to(device).eval()
    totals = np.zeros(index.shape[0], dtype=np.float64)
    with torch.inference_mode():
        for frame in frames:
            start, stop = frame["offset"], frame["offset"] + frame["pixels"]
            shape = (frame["height"], frame["width"], 3)
            for begin in range(0, index.shape[0], batch):
                rows = index[begin:begin + batch]
                left = np.stack([rgb[a, start:stop].reshape(shape) for a in rows[:, 0]])
                right = np.stack([rgb[b, start:stop].reshape(shape) for b in rows[:, 1]])
                tensor_a = torch.from_numpy(left).permute(0, 3, 1, 2).to(device) * 2.0 - 1.0
                tensor_b = torch.from_numpy(right).permute(0, 3, 1, 2).to(device) * 2.0 - 1.0
                values = model(tensor_a, tensor_b).flatten().double().cpu().numpy()
                totals[begin:begin + batch] += values
    del model
    return totals / float(len(frames))


def cvvdp_scores(rgb: np.ndarray, index: np.ndarray, frames: Sequence[dict],
                 device: str, display: str) -> np.ndarray:
    import pycvvdp
    import torch

    metric = pycvvdp.cvvdp(display_name=display, heatmap=None,
                           device=torch.device(device), quiet=True)
    totals = np.zeros(index.shape[0], dtype=np.float64)
    quantised = np.clip(np.rint(rgb * 255.0), 0, 255).astype(np.uint8)
    for frame in frames:
        start, stop = frame["offset"], frame["offset"] + frame["pixels"]
        shape = (frame["height"], frame["width"], 3)
        for row in range(index.shape[0]):
            left = quantised[index[row, 0], start:stop].reshape(shape)
            right = quantised[index[row, 1], start:stop].reshape(shape)
            jod, _ = metric.predict(left, right, dim_order="HWC")
            totals[row] += float(jod)
    del metric
    return totals / float(len(frames))


# --------------------------------------------------------------------------- evaluation


def evaluate_column(name: str, scores: np.ndarray, labels: np.ndarray, ratings: np.ndarray,
                    assignment: np.ndarray, folds: int) -> dict[str, Any]:
    """A1f protocol for a fixed (unfitted) column; identical to the A1f B1/B2 path."""
    count = labels.shape[0]
    rows_all = np.arange(count)
    covered, violation, ge4 = 0, 0, 0
    per_fold: list[dict[str, Any]] = []
    for fold in range(folds):
        train = rows_all[assignment != fold]
        test = rows_all[assignment == fold]
        threshold = working_threshold(scores[train], labels[train], MAX_FRAC, MIN_PAIRS)
        selected = (np.zeros(test.shape[0], dtype=bool) if threshold is None
                    else scores[test] <= threshold)
        covered += int(selected.sum())
        violation += int(labels[test][selected].sum())
        ge4 += int(np.sum(ratings[test][selected] >= 4))
        per_fold.append({
            "fold": fold, "train_n": int(train.shape[0]), "test_n": int(test.shape[0]),
            "threshold": (None if threshold is None else round(float(threshold), 6)),
            "train_coverage": (0 if threshold is None
                               else int(np.sum(scores[train] <= threshold))),
            "test_coverage": int(selected.sum()),
            "test_violation_ge3": int(labels[test][selected].sum()),
            "test_ge4": int(np.sum(ratings[test][selected] >= 4)),
        })
    rho_full, p_full = spearman(scores, ratings)
    return {
        "candidate": name,
        "spearman_full_fit": {"rho": round(rho_full, 4), "p": round(p_full, 8)},
        "spearman_out_of_fold": {"rho": round(rho_full, 4), "p": round(p_full, 8),
                                 "note": "no fitted parameters: out-of-fold == full-fit"},
        "working_point": {
            "rule": f"score <= t, t = max train-fold threshold with frac(rating>=3) "
                    f"<= {MAX_FRAC} over >= {MIN_PAIRS} train pairs",
            "coverage_n": covered, "coverage_frac": round(covered / count, 4),
            "violation_rate_ge3": (round(violation / covered, 4) if covered else None),
            "violation_n_ge3": violation,
            "rate_ge4": (round(ge4 / covered, 4) if covered else None), "n_ge4": ge4,
        },
        "per_fold": per_fold,
    }


def assert_harness(data: dict[str, Any], folds: int) -> dict[str, Any]:
    """Runtime pre-registration guard: this file must reproduce the A1f B1 row."""
    row = evaluate_column("B1_render_mean_1probe", data["render_mean_1probe"],
                          data["labels"], data["ratings"], data["assignment"], folds)
    got = {
        "rho": row["spearman_full_fit"]["rho"],
        "coverage_n": row["working_point"]["coverage_n"],
        "violation_rate_ge3": row["working_point"]["violation_rate_ge3"],
    }
    if got != A1F_B1_EXPECTED:
        raise SystemExit(
            f"working-point harness does not reproduce the A1f B1 row: "
            f"expected {A1F_B1_EXPECTED}, got {got}"
        )
    return row


# --------------------------------------------------------------------------- driver


def package_versions() -> dict[str, str]:
    from importlib import metadata

    out: dict[str, str] = {}
    for name in ("colour-science", "lpips", "cvvdp", "torch", "scikit-image", "numpy",
                 "scipy", "pillow"):
        try:
            out[name] = metadata.version(name)
        except Exception:
            out[name] = "missing"
    return out


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", type=Path,
                        default=REPO_ROOT / "configs/agent_loop.terra-smoke.toml")
    parser.add_argument("--labeled-pairs", type=Path, default=DEFAULT_LABELED)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument("--probe", action="append", default=None)
    parser.add_argument("--pixels-per-probe", type=int, default=4096)
    parser.add_argument("--pixel-seed", type=int, default=20260819)
    parser.add_argument("--short-side", type=int, default=256)
    parser.add_argument("--ppd", type=float, default=30.0)
    parser.add_argument("--cvvdp-display", default="standard_fhd")
    parser.add_argument("--device", default="cuda:1")
    parser.add_argument("--lpips-batch", type=int, default=8)
    parser.add_argument("--workers", type=int, default=24)
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--skip", action="append", default=[],
                        choices=["M1", "M2", "M3", "M4"])
    args = parser.parse_args(argv)
    if not args.probe:
        args.probe = list(DEFAULT_PROBES)
    probes = [Path(value) for value in args.probe]

    started = time.time()
    data = read_labeled_pairs(args.labeled_pairs)
    pair_count = len(data["pair_ids"])
    b1_row = assert_harness(data, args.folds)

    catalog, databuild, annotations = load_catalog(args.config)
    path_of = {row.preset_id: row.path for row in catalog.records}
    presets = sorted(set(data["preset_a"]) | set(data["preset_b"]))
    missing = [preset for preset in presets if preset not in path_of]
    if missing:
        raise SystemExit(f"presets absent from the catalog: {missing[:5]}")
    slot = {preset: position for position, preset in enumerate(presets)}
    index = np.asarray([[slot[a], slot[b]]
                        for a, b in zip(data["preset_a"], data["preset_b"])], dtype=np.int64)
    jobs = [(preset, path_of[preset]) for preset in presets]

    columns: dict[str, np.ndarray] = {
        "B1_render_mean_1probe": data["render_mean_1probe"],
        "B1b_render_mean_4probe": data["render_mean_4probe"],
    }
    timings: dict[str, dict[str, float]] = {}
    notes: list[str] = []
    checks: dict[str, Any] = {"a1f_b1_row_reproduced": True}

    # C1b item 10: M1/M2/M3/M4 are hours apart and any one of them can die (CUDA OOM,
    # a missing optional package, a killed worker). Every finished stage is flushed to
    # `stage_progress.json` + `pair_scores.partial.npz` the moment it completes, so a
    # later failure costs only the unfinished stage instead of the whole run.
    args.out_dir.mkdir(parents=True, exist_ok=True)
    progress_path = args.out_dir / "stage_progress.json"
    partial_npz = args.out_dir / "pair_scores.partial.npz"
    completed: list[str] = []

    def flush_stage(stage: str) -> None:
        completed.append(stage)
        save_npz_deterministic(partial_npz, {
            "pair_index": index, "rating": data["ratings"], "fold": data["assignment"],
            **{name: values.astype(np.float64) for name, values in columns.items()},
        })
        write_json(progress_path, {
            "schema": f"{METRIC_SPEC}-progress",
            "completed_stages": list(completed),
            "skipped_stages": sorted(args.skip),
            "columns": sorted(columns),
            "pair_count": pair_count,
            "checks": checks,
            "seconds": dict(timings),
            "partial_npz": str(partial_npz),
            "elapsed_seconds": round(time.time() - started, 2),
        })

    flush_stage("inputs")

    # --- sparse probe set (M1 + the CIEDE2000 parity assertion) ------------------------
    sparse_meta: list[dict] = []
    if "M1" not in args.skip:
        pixels, sparse_meta = probe_pixels(
            probes, args.pixels_per_probe * len(probes), args.pixel_seed)
        mark = time.time()
        sparse_rgb = render_presets(jobs, pixels, databuild, args.workers)
        render_sparse_s = time.time() - mark

        mark = time.time()
        from skimage.color import rgb2lab
        sparse_lab = np.stack([
            rgb2lab(sparse_rgb[position].astype(np.float64)).astype(np.float32)
            for position in range(sparse_rgb.shape[0])
        ])
        de00 = pair_reduce(sparse_lab, index, _chunk_de00, args.workers)
        del sparse_lab
        parity = float(np.max(np.abs(de00[:, 0] - data["render_mean_4probe"])))
        checks["ciede2000_parity_max_abs_diff_vs_a1f"] = round(parity, 6)
        if parity > 5e-3:
            raise SystemExit(
                f"sparse CIEDE2000 does not reproduce the A1f render_mean column "
                f"(max abs diff {parity:.6f})")
        columns["B1c_render_mean_4probe_recomputed"] = de00[:, 0]

        mark = time.time()
        ucs = cam16ucs_all(sparse_rgb)
        del sparse_rgb
        cam16 = pair_reduce(ucs, index, _chunk_cam16, args.workers)
        del ucs
        cam16_s = time.time() - mark
        columns["M1_cam16ucs_mean"] = cam16[:, 0]
        columns["M1_cam16ucs_p95"] = cam16[:, 1]
        timings["M1_cam16ucs"] = {
            "render_seconds": round(render_sparse_s, 2),
            "metric_seconds": round(cam16_s, 2),
            "seconds_per_pair": round(cam16_s / pair_count, 4),
            "seconds_per_pair_incl_render": round(
                (cam16_s + render_sparse_s) / pair_count, 4),
        }
        flush_stage("M1")

    # --- full 256px frames (M2/M3/M4) --------------------------------------------------
    frames: list[dict] = []
    need_full = any(name not in args.skip for name in ("M2", "M3", "M4"))
    render_full_s = 0.0
    if need_full:
        flat, frames = full_frames(probes, args.short_side)
        mark = time.time()
        full_rgb = render_presets(jobs, flat, databuild, args.workers)
        render_full_s = time.time() - mark
        del flat

        if "M3" not in args.skip:
            mark = time.time()
            columns["M3_lpips_alex"] = lpips_scores(
                full_rgb, index, frames, args.device, args.lpips_batch)
            lpips_s = time.time() - mark
            timings["M3_lpips_alex"] = {
                "render_seconds": round(render_full_s, 2),
                "metric_seconds": round(lpips_s, 2),
                "seconds_per_pair": round(lpips_s / pair_count, 4),
                "seconds_per_pair_incl_render": round(
                    (lpips_s + render_full_s) / pair_count, 4),
            }
            flush_stage("M3")

        if "M4" not in args.skip:
            mark = time.time()
            jod = cvvdp_scores(full_rgb, index, frames, args.device, args.cvvdp_display)
            cvvdp_s = time.time() - mark
            columns["M4_cvvdp_10_minus_jod"] = 10.0 - jod
            columns["M4_cvvdp_jod_raw"] = jod
            timings["M4_cvvdp"] = {
                "render_seconds": round(render_full_s, 2),
                "metric_seconds": round(cvvdp_s, 2),
                "seconds_per_pair": round(cvvdp_s / pair_count, 4),
                "seconds_per_pair_incl_render": round(
                    (cvvdp_s + render_full_s) / pair_count, 4),
            }
            flush_stage("M4")

        if "M2" not in args.skip:
            mark = time.time()
            filtered, scielab_stats = scielab_lab(full_rgb, frames, args.ppd, args.workers)
            checks["scielab"] = scielab_stats
            filter_s = time.time() - mark
            del full_rgb
            mark = time.time()
            scielab = pair_reduce(filtered, index, _chunk_de00, args.workers)
            del filtered
            scielab_s = time.time() - mark
            columns["M2_scielab_de00_mean"] = scielab[:, 0]
            columns["M2_scielab_de00_p95"] = scielab[:, 1]
            timings["M2_scielab"] = {
                "render_seconds": round(render_full_s, 2),
                "filter_seconds": round(filter_s, 2),
                "metric_seconds": round(filter_s + scielab_s, 2),
                "seconds_per_pair": round((filter_s + scielab_s) / pair_count, 4),
                "seconds_per_pair_incl_render": round(
                    (filter_s + scielab_s + render_full_s) / pair_count, 4),
            }
            flush_stage("M2")
        else:
            del full_rgb

    # --- evaluation --------------------------------------------------------------------
    order = [name for name in (
        "B1_render_mean_1probe", "B1b_render_mean_4probe", "B1c_render_mean_4probe_recomputed",
        "M1_cam16ucs_mean", "M1_cam16ucs_p95", "M2_scielab_de00_mean",
        "M2_scielab_de00_p95", "M3_lpips_alex", "M4_cvvdp_10_minus_jod",
    ) if name in columns]
    results = [
        evaluate_column(name, columns[name], data["labels"], data["ratings"],
                        data["assignment"], args.folds)
        for name in order
    ]
    if results[0]["spearman_full_fit"]["rho"] != b1_row["spearman_full_fit"]["rho"]:
        raise SystemExit("B1 row drifted between the guard and the report")

    flush_stage("evaluate")
    scores_path = args.out_dir / "pair_scores.json"
    write_json(scores_path, {
        "schema": METRIC_SPEC,
        "pairs": [
            {
                "pair_id": data["pair_ids"][position],
                "rating": int(data["ratings"][position]),
                "label_distinguishable": int(data["labels"][position]),
                "fold": int(data["assignment"][position]),
                "style_major": data["style_major"][position],
                "preset_a": data["preset_a"][position],
                "preset_b": data["preset_b"][position],
                **{name: round(float(values[position]), 6)
                   for name, values in columns.items()},
            }
            for position in range(pair_count)
        ],
    })
    npz_path = args.out_dir / "pair_scores.npz"
    save_npz_deterministic(npz_path, {
        "pair_index": index, "rating": data["ratings"], "fold": data["assignment"],
        **{name: values.astype(np.float64) for name, values in columns.items()},
    })

    notes.append(
        "M2 uses a simplified Gaussian approximation of the Zhang & Wandell 1996 "
        "S-CIELAB filters: the published (weight, spread) pairs are applied as plain "
        f"Gaussians with sigma = spread * ppd, ppd = {args.ppd}; the original MATLAB "
        "gauss(halfwidth, support) parameterisation was not reproduced."
    )
    notes.append(
        "M2 clips the back-transformed XYZ at 0 before CIELAB (the inhibitory lobe of the "
        "luminance filter drives some components negative); the clipped fraction is "
        "reported under checks.scielab."
    )
    notes.append(
        "Full-frame renders are LUT-applied to the LANCZOS-resized source (resize first, "
        f"short side {args.short_side}), and every rendered value is rounded through "
        "float16 so that cached and freshly rendered runs are bit-identical."
    )
    notes.append(
        "M4 JOD is a similarity (10 == identical); the evaluated column is 10 - JOD so "
        "that every column is a distance. The raw mean JOD is kept as "
        "M4_cvvdp_jod_raw and is not evaluated separately."
    )
    notes.append(
        "M1/M2/M3/M4 have no fitted parameters, so out-of-fold Spearman equals the "
        "full-sample Spearman by construction; only the working point is cross-validated."
    )
    if args.skip:
        notes.append(f"skipped candidates: {sorted(args.skip)}")

    report = {
        "schema": METRIC_SPEC,
        "data": {
            "labeled_pairs": pair_count,
            "presets": len(presets),
            "rating_histogram": {str(value): int(np.sum(data["ratings"] == value))
                                 for value in range(1, 6)},
            "label_positive_ge3": int(data["labels"].sum()),
            "folds": args.folds,
            "fold_sizes": [int(np.sum(data["assignment"] == fold))
                           for fold in range(args.folds)],
            "fold_source": str(args.labeled_pairs),
        },
        "working_point": {"max_frac": MAX_FRAC, "min_pairs": MIN_PAIRS,
                          "source": "imported from dataset_build.tools.lut_metric_ab"},
        "render": {
            "sparse_probes": sparse_meta,
            "sparse_pixels_per_probe": args.pixels_per_probe,
            "pixel_seed": args.pixel_seed,
            "full_frames": frames,
            "full_short_side": args.short_side,
        },
        "candidates": {
            "M1": "colour.difference.delta_E_CAM16UCS on colour.convert(sRGB->CAM16UCS)",
            "M2": "opponent-space Gaussian filtering then skimage deltaE_ciede2000",
            "M3": "lpips.LPIPS(net='alex'), mean over the 4 probe frames",
            "M4": f"pycvvdp.cvvdp(display_name='{args.cvvdp_display}'), mean JOD over frames",
        },
        "checks": checks,
        "results": results,
        "seconds": timings | {"total": round(time.time() - started, 2)},
        "packages": package_versions(),
        "notes": notes,
        "inputs": {
            "config": str(args.config),
            "labeled_pairs": str(args.labeled_pairs),
            "labeled_pairs_sha256": sha256_file(args.labeled_pairs),
            "annotations": str(annotations), "annotations_sha256": sha256_file(annotations),
            "databuild_config": str(databuild),
            # C1b item 11: the preset bank decides which LUTs the catalog contains.
            **features_inputs(databuild),
        },
        "determinism": {
            "pixel_seed": args.pixel_seed, "torch_manual_seed": 0,
            "device": args.device, "ppd": args.ppd,
            "float16_render_rounding": True,
        },
    }
    report_path = args.out_dir / "metric_p1.json"
    write_json(report_path, report)
    manifest_path = args.out_dir / "manifest.json"
    write_json(manifest_path, {
        "schema": f"{METRIC_SPEC}-manifest",
        "inputs": report["inputs"],
        "packages": report["packages"],
        "determinism": report["determinism"],
        "artifacts": {
            "report": {"path": str(report_path), "sha256": sha256_file(report_path)},
            "pair_scores": {"path": str(scores_path), "sha256": sha256_file(scores_path)},
            "pair_scores_npz": {"path": str(npz_path), "sha256": sha256_file(npz_path)},
        },
    })
    print(json.dumps({
        "report": str(report_path), "report_sha256": sha256_file(report_path),
        "manifest": str(manifest_path),
        "table": [
            {
                "candidate": row["candidate"],
                "rho_full": row["spearman_full_fit"]["rho"],
                "rho_oof": row["spearman_out_of_fold"]["rho"],
                "coverage_n": row["working_point"]["coverage_n"],
                "violation_ge3": row["working_point"]["violation_rate_ge3"],
                "rate_ge4": row["working_point"]["rate_ge4"],
            }
            for row in results
        ],
        "seconds": report["seconds"],
    }, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
