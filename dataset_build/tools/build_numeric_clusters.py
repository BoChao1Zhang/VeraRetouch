"""EPR-035: the two numeric cluster tables (方案 §3.2 P0).

Builds, from program-computable evidence only:

    duplicate_cluster.numeric-v1   strength-normalised (ΔE00≈5.5) canonical-render
                                   distance + numeric-response shape distance,
                                   dual-threshold clustering
    style_family.numeric-v1        k-means over the normalised numeric style vector,
                                   k chosen from {32,48,64}

Every exploration parameter lives in ``configs/lut_numeric_clusters.epr035.toml``;
the pre-registration of thresholds, probe set and selection rule is
``experiments/prs/EPR-035_numeric-clusters/NOTES.md`` §1.

Subcommands (each one re-reads the cached render, nothing is recomputed silently):

    render      render the canonical probe through every LUT, solve the per-LUT
                strength-normalisation alpha, cache Lab + style vectors
    verify      §3.2 pre-check: render_t1 largest cluster + sampled singletons,
                distance quantiles before/after normalisation + fold rates
    duplicate   write clusters.duplicate.numeric-v1.jsonl
    family      write clusters.style_family.numeric-v1.jsonl (k sweep + selection)

Usage:
    python -m dataset_build.tools.build_numeric_clusters render
    python -m dataset_build.tools.build_numeric_clusters verify
    python -m dataset_build.tools.build_numeric_clusters duplicate
    python -m dataset_build.tools.build_numeric_clusters family
"""
from __future__ import annotations

import argparse
import concurrent.futures as futures
import hashlib
import json
import multiprocessing as mp
import shutil
import sys
import time
from collections import Counter
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
from dataset_build.tools.lut_render_distance import (  # noqa: E402
    features_inputs, features_jsonl_path, save_npz_deterministic, sha256_file, write_json,
)

DEFAULT_PARAMS = REPO_ROOT / "configs/lut_numeric_clusters.epr035.toml"
SOURCE_FILE = Path(__file__).resolve()

CACHE_NPZ = "numeric_probe_render.npz"
CACHE_MANIFEST = "numeric_probe_render.manifest.json"
VERIFY_JSON = "numeric_render_t1_precheck.json"
DUP_TABLE = "clusters.duplicate.numeric-v1.jsonl"
FAMILY_TABLE = "clusters.style_family.numeric-v1.jsonl"
FAMILY_SWEEP = "clusters.style_family.numeric-v1.sweep.json"

RENDER_CACHE_SCHEMA = "lut-numeric-probe-render-v1"
DUP_SCHEMA = "lut-duplicate-cluster-numeric-v1"
FAMILY_SCHEMA = "lut-style-family-numeric-v1"
DUP_ROW_KEYS = ("row_type", "preset_id", "cluster_id", "style_major", "cluster_size",
                "alpha_norm", "strength_capacity")
FAMILY_ROW_KEYS = ("row_type", "preset_id", "family_id", "family_name", "style_major",
                   "distance_to_centroid")

HUE_BAND_COUNT = 8

# EPR-035b M3: these three used to be TOML keys that were written into the manifest but
# never consumed by any branch of the code (changing them changed the recorded metadata
# only, not the behaviour). They are literals of the implementation now.
PROBE_SPEC = "canonical-lab-chart-v1"
DUP_SCOPE = "global"
TAU_SHAPE_RULE = "quantile-matched to tau_render over all bank pairs"


# ------------------------------------------------------------------ runtime assertions

_ASSERTIONS: Counter = Counter()


def assertion_counts() -> dict[str, int]:
    return dict(_ASSERTIONS)


def reset_assertions() -> None:
    _ASSERTIONS.clear()


def require_assertions(*names: str) -> None:
    """Fail if a pre-registered assertion was defined but never actually called."""
    missing = [name for name in names if _ASSERTIONS[name] == 0]
    if missing:
        raise SystemExit(f"pre-registered runtime assertion never ran: {missing}")


def assert_catalog_inputs(preset_ids: Sequence[str], cluster_ids: Sequence[str],
                          fingerprint_ids: Sequence[str], feature_ids: Sequence[str],
                          expect_rows: int) -> dict[str, int]:
    """Row counts + preset-id set identity across the four mounted inputs.

    EPR-035b M1: the three cache-driven subcommands used to hand the same
    ``cluster_rows`` list in as two different inputs and the catalog ids in as the
    feature ids, so three of the four columns could not disagree.  Every caller now
    passes the four real sources (catalog / render_t1 / segment_fingerprints /
    features.jsonl) and ``features_rows_matched`` is a real intersection.
    """
    _ASSERTIONS["assert_catalog_inputs"] += 1
    counts = {
        "catalog_records": len(preset_ids),
        "render_t1_rows": len(cluster_ids),
        "segment_fingerprint_rows": len(fingerprint_ids),
        "features_rows": len(feature_ids),
        "features_rows_matched": len(set(preset_ids) & set(feature_ids)),
    }
    if len(set(preset_ids)) != len(preset_ids):
        raise SystemExit("duplicate preset_id in catalog")
    if counts["catalog_records"] != expect_rows:
        raise SystemExit(f"catalog rows {counts['catalog_records']} != expected {expect_rows}")
    for name, ids in (("render_t1", cluster_ids), ("segment_fingerprints", fingerprint_ids),
                      ("features", feature_ids)):
        extra = set(preset_ids) - set(ids)
        if extra:
            raise SystemExit(f"{name} misses {len(extra)} catalog presets, e.g. {sorted(extra)[:3]}")
    return counts


def assert_normalization(achieved: np.ndarray, weak: np.ndarray, target: float,
                         tolerance: float) -> dict[str, float]:
    """Every non-weak LUT must sit within the pre-registered ΔE00 tolerance."""
    _ASSERTIONS["assert_normalization"] += 1
    solved = ~weak
    if not bool(solved.any()):
        raise SystemExit("no LUT reached the normalisation target")
    deviation = np.abs(achieved[solved] - target)
    worst = float(deviation.max())
    if worst > tolerance:
        raise SystemExit(f"normalisation deviation {worst:.4f} > tolerance {tolerance}")
    return {"max_abs_deviation": worst, "solved": int(solved.sum()), "weak": int(weak.sum())}


def assert_cache_params(cache_values: dict[str, float], param_values: dict[str, float],
                        tolerance: float = 1e-12) -> dict[str, float]:
    """Refuse to run when the TOML disagrees with what the cached render was built with.

    EPR-035b M2: ``duplicate`` / ``family`` / ``verify`` read a cached ``.npz`` that was
    produced by an earlier ``render`` invocation.  Editing ``[normalize]`` afterwards used
    to silently relabel the tables with a target the cache never used.
    """
    _ASSERTIONS["assert_cache_params"] += 1
    mismatched = {
        key: (cache_values.get(key), param_values[key])
        for key in sorted(param_values)
        if cache_values.get(key) is None
        or abs(float(cache_values[key]) - float(param_values[key])) > tolerance
    }
    if mismatched:
        raise SystemExit(
            "params disagree with the cached render (rerun `render` first): "
            + ", ".join(f"{key}: cache={pair[0]!r} params={pair[1]!r}"
                        for key, pair in mismatched.items())
        )
    return {key: float(value) for key, value in param_values.items()}


def assert_table_file(path: Path, expected_rows: int, schema: str,
                      row_keys: Sequence[str]) -> dict[str, Any]:
    """Re-read a written table and check the meta line, schema, row keys and row count."""
    _ASSERTIONS["assert_table_file"] += 1
    rows = 0
    number = -1
    meta: dict[str, Any] | None = None
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            number += 1
            row = json.loads(line)
            if number == 0:
                if row.get("row_type") != "meta" or row.get("schema") != schema:
                    raise SystemExit(f"{path}: first line is not the {schema} meta record")
                meta = row
                continue
            if row.get("row_type") != "row":
                raise SystemExit(f"{path}: line {number} is not a data row")
            missing = [key for key in row_keys if key not in row]
            if missing:
                raise SystemExit(f"{path}: line {number} misses keys {missing}")
            rows += 1
    if meta is None:
        raise SystemExit(f"{path}: no meta line")
    if rows != expected_rows:
        raise SystemExit(f"{path}: {rows} data rows != expected {expected_rows}")
    if int(meta.get("rows") or -1) != expected_rows:
        raise SystemExit(f"{path}: meta.rows != {expected_rows}")
    return {"rows": rows, "meta_keys": sorted(meta)}


# ----------------------------------------------------------------------------- params


def load_params(path: Path) -> dict[str, Any]:
    with Path(path).open("rb") as handle:
        return tomllib.load(handle)


def resolve_path(value: str) -> Path:
    candidate = Path(str(value)).expanduser()
    return candidate if candidate.is_absolute() else (REPO_ROOT / candidate)


# ------------------------------------------------------------------------ canonical probe


def canonical_probe(params: dict[str, Any]) -> tuple[np.ndarray, list[dict[str, Any]]]:
    """Deterministic synthetic probe: neutral ramp + Lab hue/chroma chart + memory colours."""
    from skimage.color import deltaE_ciede2000, lab2rgb, rgb2lab

    probe = params["probe"]
    entries: list[dict[str, Any]] = []
    rgb: list[np.ndarray] = []
    dropped: list[dict[str, float]] = []

    def _lab_point(lab: np.ndarray) -> np.ndarray | None:
        value = lab2rgb(lab.reshape(1, 1, 3)).reshape(3)
        back = rgb2lab(value.reshape(1, 1, 3)).reshape(1, 3)
        error = float(deltaE_ciede2000(lab.reshape(1, 3), back)[0])
        if error > float(probe["gamut_tolerance_de00"]):
            return None
        return value.astype(np.float32)

    for lightness in probe["neutral_l"]:
        lab = np.array([float(lightness), 0.0, 0.0], dtype=np.float64)
        value = _lab_point(lab)
        if value is None:  # pragma: no cover - neutrals are always in gamut
            raise SystemExit(f"neutral L*={lightness} out of gamut")
        entries.append({"kind": "neutral", "L": float(lightness), "C": 0.0, "h": None})
        rgb.append(value)

    for lightness in probe["chroma_l"]:
        for chroma in probe["chroma_c"]:
            for hue in probe["chroma_h"]:
                radians = np.deg2rad(float(hue))
                lab = np.array([
                    float(lightness), float(chroma) * np.cos(radians),
                    float(chroma) * np.sin(radians),
                ], dtype=np.float64)
                value = _lab_point(lab)
                if value is None:
                    dropped.append({"L": float(lightness), "C": float(chroma), "h": float(hue)})
                    continue
                entries.append({"kind": "chroma", "L": float(lightness),
                                "C": float(chroma), "h": float(hue)})
                rgb.append(value)

    for name, triple in zip(probe["memory_names"], probe["memory_rgb"]):
        entries.append({"kind": "memory", "name": str(name), "L": None, "C": None, "h": None})
        rgb.append(np.asarray(triple, dtype=np.float32))

    pixels = np.stack(rgb, axis=0).astype(np.float32)
    for index, entry in enumerate(entries):
        entry["index"] = index
        entry["rgb"] = [round(float(value), 6) for value in pixels[index]]
    meta = [{"entries": entries, "dropped_out_of_gamut": dropped}]
    return pixels, meta


def probe_sha256(pixels: np.ndarray) -> str:
    return hashlib.sha256(np.asarray(pixels, dtype=np.float32).tobytes()).hexdigest()


# ---------------------------------------------------------------------------- rendering


_PIXELS: np.ndarray | None = None
_LOADER = None


def _init_render(pixels: np.ndarray, databuild: str) -> None:
    global _PIXELS, _LOADER
    from dataset_build.agent_loop.source_reach import configured_lut_loader

    _PIXELS = pixels.reshape(-1, 1, 3)
    _LOADER = configured_lut_loader(Path(databuild))


def _render_one(job: tuple[str, str]) -> tuple[str, np.ndarray | None, str]:
    from dataset_build.src.construct.rendering import apply_lut_cpu_oracle

    preset_id, lut_path = job
    try:
        grid, dmin, dmax = _LOADER.load(Path(lut_path))
        rendered = apply_lut_cpu_oracle(_PIXELS, grid, domain_min=dmin, domain_max=dmax)
        return (preset_id, rendered.reshape(-1, 3).astype(np.float32), "")
    except Exception as exc:  # pragma: no cover - reported, never silent
        return (preset_id, None, f"{type(exc).__name__}: {exc}")


def render_probe_all(records: Sequence[Any], pixels: np.ndarray, databuild: Path,
                     workers: int) -> tuple[np.ndarray, list[tuple[str, str]]]:
    jobs = [(row.preset_id, row.path) for row in records]
    full = np.zeros((len(records), pixels.shape[0], 3), dtype=np.float32)
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
            full[index] = values
    return full, failures


# ------------------------------------------------------------- strength normalisation


def mean_de00(before_lab: np.ndarray, after_lab: np.ndarray) -> np.ndarray:
    """Mean CIEDE2000 per LUT over the probe axis. Shapes (P,3) / (N,P,3) -> (N,)."""
    from skimage.color import deltaE_ciede2000

    left = np.ascontiguousarray(
        np.broadcast_to(before_lab.astype(np.float64), after_lab.shape)
    )
    delta = deltaE_ciede2000(left.reshape(-1, 3), after_lab.reshape(-1, 3).astype(np.float64))
    return delta.reshape(after_lab.shape[0], -1).mean(axis=1)


def solve_normalisation(before_rgb: np.ndarray, full_rgb: np.ndarray, target: float,
                        tolerance: float, iterations: int) -> dict[str, np.ndarray]:
    """Vectorised bisection of the RGB blend alpha onto ``mean ΔE00 == target``.

    ``after = before*(1-alpha) + full*alpha`` is exactly ``render.apply_global_strength``
    (render.py:143-147), so one full-LUT render synthesises every strength.
    """
    from skimage.color import rgb2lab

    before_lab = rgb2lab(before_rgb.reshape(1, -1, 3).astype(np.float64)).reshape(-1, 3)
    count = full_rgb.shape[0]

    def de_at(alpha: np.ndarray) -> np.ndarray:
        blend = before_rgb[None, :, :] * (1.0 - alpha[:, None, None]) \
            + full_rgb * alpha[:, None, None]
        return mean_de00(before_lab, rgb2lab(np.clip(blend, 0.0, 1.0).astype(np.float64)))

    de_full = de_at(np.ones(count, dtype=np.float64))
    weak = de_full < (target - tolerance)
    low = np.zeros(count, dtype=np.float64)
    high = np.ones(count, dtype=np.float64)
    for _ in range(int(iterations)):
        mid = (low + high) / 2.0
        value = de_at(mid)
        below = value < target
        low = np.where(below, mid, low)
        high = np.where(below, high, mid)
    alpha = np.where(weak, 1.0, (low + high) / 2.0)
    achieved = de_at(alpha)
    monotone_violation = achieved > de_full + 1e-6
    return {
        "alpha": alpha, "achieved": achieved, "de_full": de_full, "weak": weak,
        "monotone_violation": monotone_violation,
    }


def normalised_lab(before_rgb: np.ndarray, full_rgb: np.ndarray,
                   alpha: np.ndarray) -> np.ndarray:
    from skimage.color import rgb2lab

    blend = before_rgb[None, :, :] * (1.0 - alpha[:, None, None]) + full_rgb * alpha[:, None, None]
    return rgb2lab(np.clip(blend, 0.0, 1.0).astype(np.float64)).astype(np.float32)


# -------------------------------------------------------------------- style vector


def _wrap_deg(values: np.ndarray) -> np.ndarray:
    return (values + 180.0) % 360.0 - 180.0


def style_vectors(before_lab: np.ndarray, lab: np.ndarray, entries: Sequence[dict[str, Any]],
                  params: dict[str, Any]) -> tuple[np.ndarray, list[str]]:
    """34-dim numeric style vector from the strength-normalised canonical render.

    9 ramp columns (dL / cast_a / cast_b per tonal segment) + contrast_delta
    + 8 hue bands x (ΔC*ab, sin Δh, 1−cos Δh).
    """
    probe = params["probe"]
    kinds = np.asarray([entry["kind"] for entry in entries])
    lightness = np.asarray([entry["L"] if entry["L"] is not None else np.nan
                            for entry in entries], dtype=np.float64)
    hue_nominal = np.asarray([entry["h"] if entry["h"] is not None else np.nan
                              for entry in entries], dtype=np.float64)

    neutral = kinds == "neutral"
    segments = {
        "shadows": neutral & (lightness <= float(probe["shadow_l_max"])),
        "mids": neutral & (lightness > float(probe["shadow_l_max"]))
        & (lightness < float(probe["highlight_l_min"])),
        "highlights": neutral & (lightness >= float(probe["highlight_l_min"])),
    }
    columns: list[np.ndarray] = []
    names: list[str] = []
    for segment in ("shadows", "mids", "highlights"):
        mask = segments[segment]
        columns.append((lab[:, mask, 0] - before_lab[None, mask, 0]).mean(axis=1))
        names.append(f"ramp_{segment}_dL")
    for segment in ("shadows", "mids", "highlights"):
        mask = segments[segment]
        columns.append(lab[:, mask, 1].mean(axis=1))
        names.append(f"ramp_{segment}_cast_a")
    for segment in ("shadows", "mids", "highlights"):
        mask = segments[segment]
        columns.append(lab[:, mask, 2].mean(axis=1))
        names.append(f"ramp_{segment}_cast_b")

    ramp_in = before_lab[neutral, 0].astype(np.float64)
    centred = ramp_in - ramp_in.mean()
    denominator = float((centred ** 2).sum())
    ramp_out = lab[:, neutral, 0].astype(np.float64)
    slope = (ramp_out - ramp_out.mean(axis=1, keepdims=True)) @ centred / denominator
    columns.append(slope - 1.0)
    names.append("contrast_delta")

    chroma_in = np.hypot(before_lab[:, 1], before_lab[:, 2]).astype(np.float64)
    hue_in = np.degrees(np.arctan2(before_lab[:, 2], before_lab[:, 1])).astype(np.float64)
    chroma_out = np.hypot(lab[:, :, 1], lab[:, :, 2]).astype(np.float64)
    hue_out = np.degrees(np.arctan2(lab[:, :, 2], lab[:, :, 1])).astype(np.float64)
    delta_hue = _wrap_deg(hue_out - hue_in[None, :])
    valid_hue = chroma_out >= float(probe["hue_chroma_floor"])
    bands = sorted({float(value) for value in hue_nominal[kinds == "chroma"]})
    if len(bands) != HUE_BAND_COUNT:
        raise SystemExit(f"expected {HUE_BAND_COUNT} hue bands, got {len(bands)}")
    for band in bands:
        mask = (kinds == "chroma") & (hue_nominal == band)
        columns.append((chroma_out[:, mask] - chroma_in[None, mask]).mean(axis=1))
        names.append(f"band_{int(band):03d}_dC")
        radians = np.deg2rad(delta_hue[:, mask])
        weights = valid_hue[:, mask].astype(np.float64)
        total = np.maximum(weights.sum(axis=1), 1e-8)
        columns.append((np.sin(radians) * weights).sum(axis=1) / total)
        names.append(f"band_{int(band):03d}_sin_dh")
        columns.append(((1.0 - np.cos(radians)) * weights).sum(axis=1) / total)
        names.append(f"band_{int(band):03d}_omcos_dh")
    return np.stack(columns, axis=1).astype(np.float64), names


TONAL_SEGMENTS = ("shadows", "mids", "highlights")


def chroma_chart_segments(entries: Sequence[dict[str, Any]],
                          params: dict[str, Any]) -> dict[str, np.ndarray]:
    """Boolean masks putting every chroma-chart probe point in one tonal segment.

    The chart lives on three nominal lightness planes (``[probe].chroma_l``), so the
    segment bounds already used for the neutral ramp (``shadow_l_max`` /
    ``highlight_l_min``) partition it exactly: L*=30 -> shadows, 50 -> mids, 70 ->
    highlights.
    """
    probe = params["probe"]
    kinds = np.asarray([entry["kind"] for entry in entries])
    lightness = np.asarray([entry["L"] if entry["L"] is not None else np.nan
                            for entry in entries], dtype=np.float64)
    chart = kinds == "chroma"
    masks = {
        "shadows": chart & (lightness <= float(probe["shadow_l_max"])),
        "mids": chart & (lightness > float(probe["shadow_l_max"]))
        & (lightness < float(probe["highlight_l_min"])),
        "highlights": chart & (lightness >= float(probe["highlight_l_min"])),
    }
    for segment, mask in masks.items():
        if not bool(mask.any()):
            raise SystemExit(f"chroma chart has no probe point in the {segment} segment")
    return masks


def segment_chroma_delta(before_lab: np.ndarray, lab: np.ndarray,
                         entries: Sequence[dict[str, Any]],
                         params: dict[str, Any]) -> dict[str, np.ndarray]:
    r"""True ``ΔC*ab = C*_out − C*_in`` per tonal segment, from the canonical render.

    §4.3 hard red line: HSL ``dSat`` is abolished, ``ΔC*ab`` is the authoritative chroma
    field.  ``segment_fingerprints.v2``'s ``dC`` is a mean of HSV ``d_sat_pct``, so the
    shape vector's chroma block is computed here instead, on this card's own canonical
    render, with the same ``hypot(a*, b*)`` definition the style vector's ``band_*_dC``
    columns use.

    Aggregation: plain arithmetic mean of the per-point ``C*out − C*in`` over every
    in-gamut chroma-chart point on that segment's lightness plane (all hues, all three
    nominal chroma rings).  ``c_in`` / ``c_out`` are the matching plain means, used for
    the hue validity gate.
    """
    masks = chroma_chart_segments(entries, params)
    chroma_in = np.hypot(before_lab[:, 1], before_lab[:, 2]).astype(np.float64)
    chroma_out = np.hypot(lab[:, :, 1], lab[:, :, 2]).astype(np.float64)
    delta, c_in, c_out = [], [], []
    for segment in TONAL_SEGMENTS:
        mask = masks[segment]
        delta.append((chroma_out[:, mask] - chroma_in[None, mask]).mean(axis=1))
        c_in.append(np.full(lab.shape[0], float(chroma_in[mask].mean())))
        c_out.append(chroma_out[:, mask].mean(axis=1))
    return {
        "delta_c_ab": np.stack(delta, axis=1),
        "c_in": np.stack(c_in, axis=1),
        "c_out": np.stack(c_out, axis=1),
    }


def hue_validity(chroma: dict[str, np.ndarray], floor: float) -> np.ndarray:
    """§4.3 hue validity gate: low ``C_in`` or low ``C_out`` -> hue unavailable."""
    return (chroma["c_in"] >= float(floor)) & (chroma["c_out"] >= float(floor))


def hue_band_rho(lab: np.ndarray, entries: Sequence[dict[str, Any]],
                 params: dict[str, Any]) -> tuple[np.ndarray, list[str]]:
    r"""Per-band ``ρ_h`` = fraction of that band's probe points above the chroma floor.

    Diagnostic only: reported next to the style vector, *not* a column of it (see
    NOTES §4 M6).
    """
    probe = params["probe"]
    kinds = np.asarray([entry["kind"] for entry in entries])
    hue_nominal = np.asarray([entry["h"] if entry["h"] is not None else np.nan
                              for entry in entries], dtype=np.float64)
    chroma_out = np.hypot(lab[:, :, 1], lab[:, :, 2]).astype(np.float64)
    valid = chroma_out >= float(probe["hue_chroma_floor"])
    bands = sorted({float(value) for value in hue_nominal[kinds == "chroma"]})
    columns, names = [], []
    for band in bands:
        mask = (kinds == "chroma") & (hue_nominal == band)
        columns.append(valid[:, mask].mean(axis=1))
        names.append(f"band_{int(band):03d}_rho")
    return np.stack(columns, axis=1).astype(np.float64), names


def robust_standardise(matrix: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    median = np.median(matrix, axis=0)
    quartiles = np.percentile(matrix, [25, 75], axis=0)
    scale = (quartiles[1] - quartiles[0]) / 1.349
    scale = np.where(scale < 1e-6, 1.0, scale)
    return (matrix - median) / scale, median, scale


def block_equalise(matrix: np.ndarray, names: Sequence[str]) -> np.ndarray:
    """Divide each named block by sqrt(dim) so ramp / contrast / band weigh equally."""
    blocks: dict[str, list[int]] = {}
    for index, name in enumerate(names):
        key = name.split("_")[0]
        blocks.setdefault(key, []).append(index)
    out = matrix.copy()
    for indices in blocks.values():
        out[:, indices] /= np.sqrt(len(indices))
    return out


# ------------------------------------------------------------------- shape distance


SHAPE_BLOCKS = ("tone", "chroma", "cast", "hue", "hist")


def shape_vectors(fingerprints: Sequence[dict[str, Any]], delta_c_ab: np.ndarray,
                  hue_valid: np.ndarray | None = None,
                  ) -> tuple[np.ndarray, dict[str, float], dict[str, Any]]:
    r"""Magnitude-free numeric-response shape vector.

    Four blocks (tone / cast / hue / hist) are read from ``segment_fingerprints.v2``;
    the **chroma** block is ``delta_c_ab`` -- the true ``ΔC*ab`` measured on this card's
    canonical render (§4.3: HSL ``dSat`` abolished, so the fingerprint's ``dC``
    (= mean HSV ``d_sat_pct``) is no longer read here at all).

    ``hue_valid`` is the §4.3 validity gate, one flag per (LUT, tonal segment).  A gated
    segment's ``(sin Δh, 1−cos Δh)`` pair is written as explicit zeros -- the origin of
    that sub-plane, i.e. "no hue evidence" rather than "no hue rotation measured".
    """
    order = TONAL_SEGMENTS
    delta_c_ab = np.asarray(delta_c_ab, dtype=np.float64)
    if delta_c_ab.shape != (len(fingerprints), len(order)):
        raise SystemExit(f"delta_c_ab shape {delta_c_ab.shape} != {(len(fingerprints), len(order))}")
    if hue_valid is None:
        gate = np.ones((len(fingerprints), len(order)), dtype=bool)
    else:
        gate = np.asarray(hue_valid, dtype=bool)
        if gate.shape != delta_c_ab.shape:
            raise SystemExit(f"hue_valid shape {gate.shape} != {delta_c_ab.shape}")
    tone, cast, hue, hist = [], [], [], []
    for number, row in enumerate(fingerprints):
        segments = row["segments"]
        tone.append([float(segments[key]["dL"]) for key in order])
        cast.append([float(segments[key]["cast_a"]) for key in order]
                    + [float(segments[key]["cast_b"]) for key in order])
        angles = np.deg2rad([float(segments[key]["d_hue"]) for key in order])
        weight = gate[number].astype(np.float64)
        hue.append(np.concatenate([np.sin(angles) * weight,
                                   (1.0 - np.cos(angles)) * weight]).tolist())
        hist.append([float(value) for value in row["histogram"]["delta"]])
    blocks = {
        "tone": np.asarray(tone, dtype=np.float64),
        "chroma": delta_c_ab,
        "cast": np.asarray(cast, dtype=np.float64),
        "hue": np.asarray(hue, dtype=np.float64),
        "hist": np.asarray(hist, dtype=np.float64),
    }
    gate_report = {
        "hue_gate_segments_total": int(gate.size),
        "hue_gate_unavailable": int((~gate).sum()),
        "hue_gate_unavailable_by_segment": {
            segment: int((~gate[:, index]).sum()) for index, segment in enumerate(order)},
        "hue_gate_rows_with_any_unavailable": int((~gate).any(axis=1).sum()),
    }
    scales: dict[str, float] = {}
    scaled = []
    for key in SHAPE_BLOCKS:
        block = blocks[key]
        norms = np.linalg.norm(block, axis=1)
        scale = float(np.median(norms))
        scale = scale if scale > 1e-9 else 1.0
        scales[key] = scale
        scaled.append(block / scale)
    matrix = np.concatenate(scaled, axis=1)
    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    norms = np.where(norms < 1e-12, 1.0, norms)
    return matrix / norms, scales, gate_report


def shape_distance_matrix(unit: np.ndarray) -> np.ndarray:
    cosine = np.clip(unit @ unit.T, -1.0, 1.0)
    return np.arccos(cosine) / np.pi


# ------------------------------------------------------------------ render distances


_LAB: np.ndarray | None = None
_PAIRS: np.ndarray | None = None


def _init_pairs(lab: np.ndarray, pairs: np.ndarray) -> None:  # pragma: no cover - fork path
    global _LAB, _PAIRS
    _LAB, _PAIRS = lab, pairs


def _pair_chunk(job: tuple[int, int]) -> np.ndarray:  # pragma: no cover - fork path
    from skimage.color import deltaE_ciede2000

    start, stop = job
    index = _PAIRS[start:stop]
    left = _LAB[index[:, 0]].astype(np.float64)
    right = _LAB[index[:, 1]].astype(np.float64)
    delta = deltaE_ciede2000(left.reshape(-1, 3), right.reshape(-1, 3))
    return delta.reshape(index.shape[0], -1).mean(axis=1).astype(np.float32)


def render_distance_condensed(lab: np.ndarray, workers: int, chunk: int) -> np.ndarray:
    """Condensed (scipy squareform order) mean-ΔE00 distance over all pairs."""
    global _LAB, _PAIRS

    count = lab.shape[0]
    rows, cols = np.triu_indices(count, k=1)
    pairs = np.stack([rows, cols], axis=1).astype(np.int32)
    del rows, cols
    _LAB, _PAIRS = lab, pairs
    if pairs.shape[0] <= chunk or workers <= 1:
        return _pair_chunk((0, pairs.shape[0]))
    jobs = [(start, min(start + chunk, pairs.shape[0]))
            for start in range(0, pairs.shape[0], chunk)]
    context = mp.get_context("fork")
    chunks: list[np.ndarray] = []
    with futures.ProcessPoolExecutor(max_workers=workers, mp_context=context) as pool:
        for values in pool.map(_pair_chunk, jobs, chunksize=1):
            chunks.append(values)
    return np.concatenate(chunks)


def bank_row_distances(query_lab: np.ndarray, bank_lab: np.ndarray,
                       rows_per_chunk: int = 8) -> np.ndarray:
    """(M, N) mean-ΔE00 of M query LUTs against the whole bank, chunked over queries."""
    from skimage.color import deltaE_ciede2000

    out = np.zeros((query_lab.shape[0], bank_lab.shape[0]), dtype=np.float32)
    bank = bank_lab.astype(np.float64)
    for start in range(0, query_lab.shape[0], rows_per_chunk):
        stop = min(start + rows_per_chunk, query_lab.shape[0])
        left = np.ascontiguousarray(np.broadcast_to(
            query_lab[start:stop, None, :, :].astype(np.float64),
            (stop - start, bank.shape[0], bank.shape[1], 3)))
        right = np.ascontiguousarray(np.broadcast_to(
            bank[None, :, :, :], left.shape))
        delta = deltaE_ciede2000(left.reshape(-1, 3), right.reshape(-1, 3))
        out[start:stop] = delta.reshape(stop - start, bank.shape[0], -1).mean(axis=2)
    return out


# ------------------------------------------------------------------------- clustering


def dual_threshold_clusters(render: np.ndarray, shape: np.ndarray, tau_render: float,
                            tau_shape: float, method: str, cut: float) -> np.ndarray:
    """Average linkage over ``max(d_render/tau_render, d_shape/tau_shape)`` cut at ``cut``."""
    from scipy.cluster.hierarchy import fcluster, linkage

    combined = np.maximum(render / float(tau_render), shape / float(tau_shape))
    link = linkage(combined.astype(np.float64), method=method)
    return fcluster(link, t=float(cut), criterion="distance").astype(np.int64)


def quantile_matched_tau(render: np.ndarray, shape: np.ndarray, tau_render: float) -> tuple[float, float]:
    fraction = float((render <= tau_render).mean())
    return float(np.quantile(shape, fraction)), fraction


def quantiles(values: np.ndarray) -> dict[str, float]:
    if values.size == 0:
        return {key: 0.0 for key in ("min", "p5", "p25", "p50", "p75", "p95", "max", "mean")}
    points = np.quantile(values, [0.05, 0.25, 0.50, 0.75, 0.95])
    return {
        "min": round(float(values.min()), 6), "p5": round(float(points[0]), 6),
        "p25": round(float(points[1]), 6), "p50": round(float(points[2]), 6),
        "p75": round(float(points[3]), 6), "p95": round(float(points[4]), 6),
        "max": round(float(values.max()), 6), "mean": round(float(values.mean()), 6),
    }


def size_histogram(sizes: Sequence[int]) -> dict[str, int]:
    buckets = {"1": 0, "2": 0, "3-4": 0, "5-8": 0, "9-16": 0, "17+": 0}
    for size in sizes:
        if size == 1:
            buckets["1"] += 1
        elif size == 2:
            buckets["2"] += 1
        elif size <= 4:
            buckets["3-4"] += 1
        elif size <= 8:
            buckets["5-8"] += 1
        elif size <= 16:
            buckets["9-16"] += 1
        else:
            buckets["17+"] += 1
    return buckets


# ----------------------------------------------------------------------------- k-means


def euclidean_matrix(left: np.ndarray, right: np.ndarray | None = None) -> np.ndarray:
    """Gram-trick euclidean distances (never materialises the N x M x D difference)."""
    right = left if right is None else right
    gram = left @ right.T
    square = (np.einsum("ij,ij->i", left, left)[:, None]
              + np.einsum("ij,ij->i", right, right)[None, :] - 2.0 * gram)
    return np.sqrt(np.maximum(square, 0.0))


def kmeans_plusplus(matrix: np.ndarray, k: int, rng: np.random.Generator) -> np.ndarray:
    centers = np.empty((k, matrix.shape[1]), dtype=np.float64)
    centers[0] = matrix[rng.integers(matrix.shape[0])]
    closest = ((matrix - centers[0]) ** 2).sum(axis=1)
    for index in range(1, k):
        total = float(closest.sum())
        if total <= 0.0:  # pragma: no cover - degenerate data
            centers[index] = matrix[rng.integers(matrix.shape[0])]
        else:
            centers[index] = matrix[rng.choice(matrix.shape[0], p=closest / total)]
        closest = np.minimum(closest, ((matrix - centers[index]) ** 2).sum(axis=1))
    return centers


def kmeans(matrix: np.ndarray, k: int, seed: int, n_init: int,
           iterations: int) -> tuple[np.ndarray, np.ndarray, float]:
    best: tuple[float, np.ndarray, np.ndarray] | None = None
    for attempt in range(n_init):
        rng = np.random.default_rng(seed + attempt)
        centers = kmeans_plusplus(matrix, k, rng)
        labels = np.full(matrix.shape[0], -1, dtype=np.int64)
        for _ in range(iterations):
            distances = euclidean_matrix(matrix, centers) ** 2
            new_labels = distances.argmin(axis=1)
            if np.array_equal(new_labels, labels):
                break
            labels = new_labels
            for index in range(k):
                members = matrix[labels == index]
                if members.shape[0] == 0:
                    centers[index] = matrix[distances.min(axis=1).argmax()]
                else:
                    centers[index] = members.mean(axis=0)
        inertia = float(((matrix - centers[labels]) ** 2).sum())
        if best is None or inertia < best[0] - 1e-9:
            best = (inertia, labels.copy(), centers.copy())
    assert best is not None
    return best[1], best[2], best[0]


def silhouette(distance: np.ndarray, labels: np.ndarray) -> float:
    """Mean silhouette on a precomputed distance matrix; clusters of size 1 score 0."""
    unique, compact = np.unique(labels, return_inverse=True)
    if unique.size < 2:
        return 0.0
    count = labels.shape[0]
    onehot = np.zeros((count, unique.size), dtype=np.float64)
    onehot[np.arange(count), compact] = 1.0
    sums = distance @ onehot                      # (N, K) distance mass per cluster
    sizes = onehot.sum(axis=0)                    # (K,)
    own_size = sizes[compact]
    own_sum = sums[np.arange(count), compact]
    with np.errstate(invalid="ignore", divide="ignore"):
        a = np.where(own_size > 1, own_sum / np.maximum(own_size - 1.0, 1.0), 0.0)
        others = sums / np.maximum(sizes[None, :], 1.0)
    others[np.arange(count), compact] = np.inf
    others[:, sizes == 0] = np.inf
    b = others.min(axis=1)
    scores = np.where((own_size > 1) & np.isfinite(b),
                      (b - a) / np.maximum(np.maximum(a, b), 1e-12), 0.0)
    return float(scores.mean())


def cluster_geometry(matrix: np.ndarray, labels: np.ndarray,
                     centers: np.ndarray) -> dict[str, float]:
    within: list[float] = []
    for index in range(centers.shape[0]):
        members = matrix[labels == index]
        if members.shape[0] < 2:
            continue
        within.append(float(np.linalg.norm(members - centers[index], axis=1).mean()))
    pairs = []
    for i in range(centers.shape[0]):
        for j in range(i + 1, centers.shape[0]):
            pairs.append(float(np.linalg.norm(centers[i] - centers[j])))
    sizes = np.bincount(labels, minlength=centers.shape[0])
    return {
        "mean_within_cluster_radius": round(float(np.mean(within)), 6) if within else 0.0,
        "mean_between_centroid_distance": round(float(np.mean(pairs)), 6) if pairs else 0.0,
        "within_over_between": round(float(np.mean(within) / np.mean(pairs)), 6)
        if within and pairs else 0.0,
        "min_cluster_size": int(sizes.min()), "max_cluster_size": int(sizes.max()),
        "singleton_clusters": int((sizes == 1).sum()),
    }


# -------------------------------------------------------------------------- family names


def family_names(raw: np.ndarray, names: Sequence[str], labels: np.ndarray,
                 params: dict[str, Any]) -> list[str]:
    """Deterministic ``<lightness>-<temperature>-<chroma>`` names from raw centroid means."""
    family = params["family"]
    column = {name: index for index, name in enumerate(names)}
    band_dc = [index for name, index in column.items() if name.endswith("_dC")]
    generated: list[str] = []
    for index in range(int(labels.max()) + 1):
        members = raw[labels == index]
        if members.shape[0] == 0:  # pragma: no cover - k-means never emits empty clusters
            generated.append(f"empty-{index}")
            continue
        mid_dl = float(members[:, column["ramp_mids_dL"]].mean())
        cast_b = float(members[:, column["ramp_mids_cast_b"]].mean())
        chroma = float(members[:, band_dc].mean())
        contrast = float(members[:, column["contrast_delta"]].mean())
        lightness = ("bright" if mid_dl > float(family["name_bright_dl"])
                     else "dark" if mid_dl < float(family["name_dark_dl"]) else "mid")
        temperature = ("warm" if cast_b > float(family["name_warm_cast_b"])
                       else "cool" if cast_b < float(family["name_cool_cast_b"])
                       else "neutralwb")
        saturation = ("vivid" if chroma > float(family["name_vivid_dc"])
                      else "desat" if chroma < float(family["name_desat_dc"]) else "even")
        crisp = ("crisp" if contrast > float(family["name_crisp_contrast"])
                 else "flat" if contrast < -float(family["name_crisp_contrast"]) else "soft")
        generated.append(f"{lightness}-{temperature}-{saturation}|{crisp}")
    final: list[str] = []
    seen: Counter = Counter()
    base_counts = Counter(name.split("|")[0] for name in generated)
    for name in generated:
        base, crisp = name.split("|")
        label = base if base_counts[base] == 1 else f"{base}-{crisp}"
        seen[label] += 1
        final.append(label if seen[label] == 1 else f"{label}-{seen[label]}")
    return final


# --------------------------------------------------------------------------- io helpers


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with Path(path).open("r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def load_catalog(config_path: Path) -> tuple[LutCatalog, Path, CatalogConfig]:
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
        segment_fingerprints=Path(str(table["segment_fingerprints"])).expanduser(),
    )
    return LutCatalog.load(config, databuild), databuild, config


def catalog_paths(config_path: Path) -> dict[str, Path]:
    """``databuild`` + ``segment_fingerprints`` of an agent-loop config, without
    paying for a full ``LutCatalog.load``."""
    with Path(config_path).open("rb") as handle:
        data = tomllib.load(handle)
    table = data.get("catalog") or {}
    return {
        "databuild": (Path(config_path).parent
                      / str(data["agent_loop"]["databuild_config"])).resolve(),
        "annotations": Path(str(table["annotations"])).expanduser(),
        "segment_fingerprints": Path(str(table["segment_fingerprints"])).expanduser(),
    }


def features_preset_ids(databuild: Path) -> list[str]:
    """Every ``preset_id`` in the bank's ``features.jsonl`` (the real 4th input)."""
    path = features_jsonl_path(databuild)
    ids: list[str] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                ids.append(str(json.loads(line)["preset_id"]))
    return ids


def check_cached_run(params: dict[str, Any], cache: "RenderCache",
                     ) -> tuple[dict[str, int], dict[str, Any]]:
    """The two start-up gates every cache-driven subcommand runs (M1 + M2).

    Returns ``(counts, provenance)``: the four-input row/id agreement and the
    params-vs-cache agreement that has to hold before anything is written.
    """
    inputs = params["inputs"]
    paths = catalog_paths(resolve_path(inputs["agent_loop_config"]))
    cluster_rows = load_jsonl(resolve_path(inputs["render_t1_clusters"]))
    fingerprint_rows = load_jsonl(paths["segment_fingerprints"])
    counts = assert_catalog_inputs(
        cache.preset_ids,
        [str(row["preset_id"]) for row in cluster_rows],
        [str(row["preset_id"]) for row in fingerprint_rows],
        features_preset_ids(paths["databuild"]),
        int(inputs["expect_rows"]),
    )
    agreed = assert_cache_params(cache.params, cache.declared_params(params))
    provenance = {
        "cache": str(cache.path), "cache_sha256": sha256_file(cache.path),
        "cache_chroma_source": cache.chroma_source,
        "cache_params_agreed": agreed,
        "segment_fingerprints": str(paths["segment_fingerprints"]),
        "features_jsonl": str(features_jsonl_path(paths["databuild"])),
        "render_t1_clusters": str(resolve_path(inputs["render_t1_clusters"])),
    }
    return counts, provenance


def backup_if_exists(path: Path) -> str | None:
    """Copy an existing artifact to ``<name>.bak.<stamp>`` *before* it is overwritten."""
    if not path.exists():
        return None
    stamp = time.strftime("%Y%m%dT%H%M%S")
    target = path.with_suffix(path.suffix + f".bak.{stamp}")
    shutil.copy2(path, target)
    return str(target)


def write_table(path: Path, meta: dict[str, Any], rows: Sequence[dict[str, Any]]) -> None:
    lines = [json.dumps(meta, ensure_ascii=False, sort_keys=True, separators=(",", ":"))]
    lines.extend(json.dumps(row, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
                 for row in rows)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8", newline="\n")


def sample_singletons(preset_ids: Sequence[str], salt: str, count: int) -> list[str]:
    """Deterministic sha256-ranked sample (no ad-hoc selection)."""
    ranked = sorted(
        preset_ids,
        key=lambda preset_id: (
            int(hashlib.sha256((salt + preset_id).encode("utf-8")).hexdigest()[:16], 16),
            preset_id,
        ),
    )
    return ranked[:count]


# -------------------------------------------------------------------------- subcommands


def cmd_render(args: argparse.Namespace) -> int:
    started = time.time()
    params = load_params(args.params)
    inputs = params["inputs"]
    config_path = resolve_path(inputs["agent_loop_config"])
    catalog, databuild, catalog_config = load_catalog(config_path)
    records = list(catalog.records)
    cluster_rows = load_jsonl(resolve_path(inputs["render_t1_clusters"]))
    fingerprint_rows = load_jsonl(Path(str(catalog_config.segment_fingerprints)))
    counts = assert_catalog_inputs(
        [row.preset_id for row in records],
        [str(row["preset_id"]) for row in cluster_rows],
        [str(row["preset_id"]) for row in fingerprint_rows],
        features_preset_ids(databuild),
        int(inputs["expect_rows"]),
    )

    pixels, probe_meta = canonical_probe(params)
    render_started = time.time()
    full_rgb, failures = render_probe_all(records, pixels, databuild, int(params["runtime"]["workers"]))
    if failures:
        raise SystemExit(f"render failures ({len(failures)}): {failures[:5]}")
    render_seconds = time.time() - render_started

    normalisation = solve_normalisation(
        pixels, full_rgb, float(params["normalize"]["target_de00"]),
        float(params["normalize"]["tolerance"]), int(params["normalize"]["bisect_iters"]),
    )
    norm_stats = assert_normalization(
        normalisation["achieved"], normalisation["weak"],
        float(params["normalize"]["target_de00"]), float(params["normalize"]["tolerance"]),
    )
    lab_norm = normalised_lab(pixels, full_rgb, normalisation["alpha"])
    from skimage.color import rgb2lab

    before_lab = rgb2lab(pixels.reshape(1, -1, 3).astype(np.float64)).reshape(-1, 3)
    lab_full = rgb2lab(np.clip(full_rgb, 0.0, 1.0).astype(np.float64)).astype(np.float32)
    entries = probe_meta[0]["entries"]
    style, style_names = style_vectors(before_lab, lab_norm, entries, params)
    rho, rho_names = hue_band_rho(lab_norm, entries, params)

    # §4.3 chroma block: true ΔC*ab off this card's canonical render, never HSL dSat.
    shape_params = params["shape"]
    chroma_source = str(shape_params["chroma_source"])
    source_lab = {"lab_full": lab_full, "lab_norm": lab_norm}[chroma_source]
    chroma = segment_chroma_delta(before_lab, source_lab, entries, params)
    hue_valid = hue_validity(chroma, float(shape_params["hue_chroma_floor"]))

    fingerprint_by_id = {str(row["preset_id"]): row for row in fingerprint_rows}
    shape, shape_scales, gate_report = shape_vectors(
        [fingerprint_by_id[row.preset_id] for row in records],
        chroma["delta_c_ab"], hue_valid,
    )

    out_dir = resolve_path(inputs["out_dir"])
    out_dir.mkdir(parents=True, exist_ok=True)
    cache_path = out_dir / CACHE_NPZ
    cache_backup = backup_if_exists(cache_path)
    save_npz_deterministic(cache_path, {
        "preset_ids": np.asarray([row.preset_id for row in records], dtype="<U64"),
        "style_majors": np.asarray([row.style_major for row in records], dtype="<U64"),
        "probe_rgb": pixels,
        "before_lab": before_lab.astype(np.float32),
        "lab_full": lab_full,
        "lab_norm": lab_norm,
        "alpha": normalisation["alpha"].astype(np.float64),
        "achieved_de00": normalisation["achieved"].astype(np.float64),
        "de00_full": normalisation["de_full"].astype(np.float64),
        "weak": normalisation["weak"].astype(np.int8),
        "style": style,
        "style_names": np.asarray(style_names, dtype="<U32"),
        "hue_band_rho": rho,
        "hue_band_rho_names": np.asarray(rho_names, dtype="<U32"),
        "shape_unit": shape,
        "shape_delta_c_ab": chroma["delta_c_ab"],
        "shape_c_in": chroma["c_in"],
        "shape_c_out": chroma["c_out"],
        "shape_hue_valid": hue_valid.astype(np.int8),
        # M2: what the cache was actually built with, so a consumer can refuse a
        # params file that has drifted away from it.
        "normalize_target_de00": np.asarray(
            [float(params["normalize"]["target_de00"])], dtype=np.float64),
        "normalize_tolerance": np.asarray(
            [float(params["normalize"]["tolerance"])], dtype=np.float64),
        "shape_hue_chroma_floor": np.asarray(
            [float(shape_params["hue_chroma_floor"])], dtype=np.float64),
        "shape_chroma_source": np.asarray([chroma_source], dtype="<U32"),
    })
    manifest = {
        "schema": RENDER_CACHE_SCHEMA,
        "probe": {
            "spec": PROBE_SPEC,
            "pixels": int(pixels.shape[0]),
            "sha256": probe_sha256(pixels),
            "dropped_out_of_gamut": len(probe_meta[0]["dropped_out_of_gamut"]),
            "kind_counts": dict(Counter(entry["kind"] for entry in entries)),
        },
        "normalization": {
            "target_de00": float(params["normalize"]["target_de00"]),
            "tolerance": float(params["normalize"]["tolerance"]),
            "blend": "after = before*(1-alpha) + full*alpha (render.py:143-147)",
            "alpha_quantiles": quantiles(normalisation["alpha"][~normalisation["weak"]]),
            "de00_full_quantiles": quantiles(normalisation["de_full"]),
            "weak_count": int(normalisation["weak"].sum()),
            "monotone_violations": int(normalisation["monotone_violation"].sum()),
            **norm_stats,
        },
        "shape": {
            "chroma_block": "delta_c_ab from the canonical render (§4.3; segment "
                            "fingerprint dC = HSV d_sat_pct is not read)",
            "chroma_source": chroma_source,
            "chroma_aggregation": "mean over the in-gamut chroma-chart points of the "
                                  "segment's nominal L* plane (all hues x all rings)",
            "block_scales": {key: round(value, 6) for key, value in shape_scales.items()},
            "hue_chroma_floor": float(shape_params["hue_chroma_floor"]),
            "delta_c_ab_quantiles": {
                segment: quantiles(chroma["delta_c_ab"][:, index])
                for index, segment in enumerate(TONAL_SEGMENTS)},
            "c_out_quantiles": {
                segment: quantiles(chroma["c_out"][:, index])
                for index, segment in enumerate(TONAL_SEGMENTS)},
            "c_in_mean": {segment: round(float(chroma["c_in"][0, index]), 6)
                          for index, segment in enumerate(TONAL_SEGMENTS)},
            **gate_report,
        },
        "style_columns": style_names,
        "hue_band_rho_mean": {name: round(float(rho[:, index].mean()), 6)
                              for index, name in enumerate(rho_names)},
        "inputs": {
            "params": str(args.params), "params_sha256": sha256_file(args.params),
            "source": str(SOURCE_FILE), "source_sha256": sha256_file(SOURCE_FILE),
            "agent_loop_config": str(config_path),
            "databuild_config": str(databuild),
            "annotations": str(catalog_config.annotations),
            "annotations_sha256": sha256_file(catalog_config.annotations),
            "segment_fingerprints": str(catalog_config.segment_fingerprints),
            "segment_fingerprints_sha256": sha256_file(Path(str(catalog_config.segment_fingerprints))),
            "render_t1_clusters": str(resolve_path(inputs["render_t1_clusters"])),
            **features_inputs(databuild),
        },
        "counts": counts,
        "assertions": assertion_counts(),
        "backup": cache_backup,
        "artifacts": {"cache": {"path": str(cache_path), "sha256": sha256_file(cache_path)}},
        "seconds": {"render": round(render_seconds, 2), "total": round(time.time() - started, 2)},
    }
    manifest_path = out_dir / CACHE_MANIFEST
    manifest["backup_manifest"] = backup_if_exists(manifest_path)
    write_json(manifest_path, manifest)
    require_assertions("assert_catalog_inputs", "assert_normalization")
    print(json.dumps({
        "cache": str(cache_path), "manifest": str(manifest_path),
        "shape": manifest["shape"],
        "probe_pixels": int(pixels.shape[0]),
        "records": len(records),
        "normalization": manifest["normalization"],
        "seconds": manifest["seconds"],
    }, ensure_ascii=False, indent=2))
    return 0


class RenderCache:
    def __init__(self, path: Path) -> None:
        with np.load(path, allow_pickle=False) as data:
            self.preset_ids = [str(value) for value in data["preset_ids"]]
            self.style_majors = [str(value) for value in data["style_majors"]]
            self.probe_rgb = data["probe_rgb"]
            self.before_lab = data["before_lab"]
            self.lab_full = data["lab_full"]
            self.lab_norm = data["lab_norm"]
            self.alpha = data["alpha"]
            self.achieved = data["achieved_de00"]
            self.de00_full = data["de00_full"]
            self.weak = data["weak"].astype(bool)
            self.style = data["style"]
            self.style_names = [str(value) for value in data["style_names"]]
            self.shape_unit = data["shape_unit"]
            self.hue_band_rho = data["hue_band_rho"]
            self.hue_band_rho_names = [str(value) for value in data["hue_band_rho_names"]]
            self.delta_c_ab = data["shape_delta_c_ab"]
            self.hue_valid = data["shape_hue_valid"].astype(bool)
            self.chroma_source = str(data["shape_chroma_source"][0])
            self.params = {
                "normalize_target_de00": float(data["normalize_target_de00"][0]),
                "normalize_tolerance": float(data["normalize_tolerance"][0]),
                "shape_hue_chroma_floor": float(data["shape_hue_chroma_floor"][0]),
            }
        self.path = path
        self.index = {preset_id: number for number, preset_id in enumerate(self.preset_ids)}

    def declared_params(self, params: dict[str, Any]) -> dict[str, float]:
        """The TOML values a consumer must agree with the cache on (M2)."""
        return {
            "normalize_target_de00": float(params["normalize"]["target_de00"]),
            "normalize_tolerance": float(params["normalize"]["tolerance"]),
            "shape_hue_chroma_floor": float(params["shape"]["hue_chroma_floor"]),
        }


def human_rating_join(path: Path, cache: "RenderCache", shape_all: np.ndarray,
                      tau_render: float, tau_shape: float) -> dict[str, Any]:
    """Join the existing 200 human-rated pairs onto the two numeric channels."""
    from scipy.stats import spearmanr

    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    ratings: list[int] = []
    before: list[float] = []
    after: list[float] = []
    shape: list[float] = []
    missing = 0
    for entry in payload["pairs"]:
        left, right = str(entry["preset_a"]), str(entry["preset_b"])
        if left not in cache.index or right not in cache.index:
            missing += 1
            continue
        i, j = cache.index[left], cache.index[right]
        before.append(float(bank_row_distances(cache.lab_full[[i]], cache.lab_full[[j]])[0, 0]))
        after.append(float(bank_row_distances(cache.lab_norm[[i]], cache.lab_norm[[j]])[0, 0]))
        shape.append(float(shape_all[i, j]))
        ratings.append(int(entry["rating"]))
    if len(ratings) < 3:
        return {"joined": len(ratings), "missing": missing}
    normalized = spearmanr(after, ratings)
    full_strength = spearmanr(before, ratings)
    shape_rho = spearmanr(shape, ratings)
    after_array = np.asarray(after)
    shape_array = np.asarray(shape)
    rating_array = np.asarray(ratings)
    inside = (after_array <= tau_render) & (shape_array <= tau_shape)
    return {
        "source": str(path), "joined": len(ratings), "missing": missing,
        "spearman_render_normalized_vs_rating": {
            "rho": round(float(normalized.statistic), 4), "p": round(float(normalized.pvalue), 8)},
        "spearman_render_full_strength_vs_rating": {
            "rho": round(float(full_strength.statistic), 4),
            "p": round(float(full_strength.pvalue), 8)},
        "spearman_shape_vs_rating": {
            "rho": round(float(shape_rho.statistic), 4), "p": round(float(shape_rho.pvalue), 8)},
        "pairs_inside_dual_threshold": int(inside.sum()),
        "rating_ge3_fraction_inside": (round(float((rating_array[inside] >= 3).mean()), 6)
                                       if int(inside.sum()) else None),
        "rating_ge3_fraction_all": round(float((rating_array >= 3).mean()), 6),
    }


def cmd_verify(args: argparse.Namespace) -> int:
    started = time.time()
    params = load_params(args.params)
    inputs = params["inputs"]
    out_dir = resolve_path(inputs["out_dir"])
    cache = RenderCache(out_dir / CACHE_NPZ)
    counts, provenance = check_cached_run(params, cache)
    cluster_rows = load_jsonl(resolve_path(inputs["render_t1_clusters"]))
    grouped: dict[tuple[str, str], list[str]] = {}
    for row in cluster_rows:
        key = (str(row.get("style_major") or ""), str(row["cluster_id"]))
        grouped.setdefault(key, []).append(str(row["preset_id"]))
    sizes = {key: len(value) for key, value in grouped.items()}
    largest_key = max(sizes, key=lambda key: (sizes[key], key))
    largest = sorted(grouped[largest_key])
    singletons = sorted(pid for key, value in grouped.items() if len(value) == 1 for pid in value)
    sampled = sample_singletons(singletons, str(params["verify"]["sample_salt"]),
                                int(params["verify"]["singleton_sample"]))

    shape_all = shape_distance_matrix(cache.shape_unit)
    tau_render = float(params["duplicate"]["tau_render"])
    workers = int(params["runtime"]["workers"])
    chunk = int(params["runtime"]["pair_chunk"])
    full_render = render_distance_condensed(cache.lab_norm, workers, chunk)
    full_render_before = render_distance_condensed(cache.lab_full, workers, chunk)
    shape_condensed = shape_all[np.triu_indices(shape_all.shape[0], k=1)]
    tau_shape, matched_fraction = quantile_matched_tau(full_render, shape_condensed, tau_render)

    def _group_report(name: str, members: Sequence[str]) -> dict[str, Any]:
        index = np.asarray([cache.index[preset_id] for preset_id in members], dtype=np.int64)
        pairs = np.triu_indices(index.shape[0], k=1)
        before = render_distance_condensed(cache.lab_full[index], 1, chunk)
        after = render_distance_condensed(cache.lab_norm[index], 1, chunk)
        shape = shape_all[np.ix_(index, index)][pairs]

        # Direct form of the hypothesis: does a member acquire a bank-wide partner
        # under the dual threshold once strength is normalised?
        bank_before = bank_row_distances(cache.lab_full[index], cache.lab_full)
        bank_after = bank_row_distances(cache.lab_norm[index], cache.lab_norm)
        bank_shape = shape_all[index]
        self_mask = np.zeros(bank_before.shape, dtype=bool)
        self_mask[np.arange(index.shape[0]), index] = True
        bank_before[self_mask] = np.inf
        bank_after[self_mask] = np.inf
        bank_shape = np.where(self_mask, np.inf, bank_shape)

        grid = []
        for tau in params["duplicate"]["tau_render_grid"]:
            tau = float(tau)
            partner_before = ((bank_before <= tau) & (bank_shape <= tau_shape)).any(axis=1)
            partner_after = ((bank_after <= tau) & (bank_shape <= tau_shape)).any(axis=1)
            grid.append({
                "tau_render": tau, "tau_shape": round(tau_shape, 6),
                "pair_fold_rate_full_strength": round(float(((before <= tau) & (shape <= tau_shape)).mean()), 6),
                "pair_fold_rate_normalized": round(float(((after <= tau) & (shape <= tau_shape)).mean()), 6),
                "pair_fold_rate_render_only_full_strength": round(float((before <= tau).mean()), 6),
                "pair_fold_rate_render_only_normalized": round(float((after <= tau).mean()), 6),
                "bank_partner_rate_full_strength": round(float(partner_before.mean()), 6),
                "bank_partner_rate_normalized": round(float(partner_after.mean()), 6),
                "bank_partner_rate_render_only_full_strength":
                    round(float((bank_before <= tau).any(axis=1).mean()), 6),
                "bank_partner_rate_render_only_normalized":
                    round(float((bank_after <= tau).any(axis=1).mean()), 6),
            })
        return {
            "group": name, "members": len(members), "pairs": int(before.size),
            "render_distance_full_strength": quantiles(before),
            "render_distance_normalized": quantiles(after),
            "shape_distance": quantiles(shape),
            "nearest_bank_render_distance_full_strength": quantiles(bank_before.min(axis=1)),
            "nearest_bank_render_distance_normalized": quantiles(bank_after.min(axis=1)),
            "nearest_bank_shape_distance": quantiles(bank_shape.min(axis=1)),
            "alpha": quantiles(cache.alpha[index]),
            "weak_members": int(cache.weak[index].sum()),
            "dual_threshold_grid": grid,
        }

    calibration_report = human_rating_join(
        resolve_path(inputs["calibration_pairs"]), cache, shape_all, tau_render, tau_shape
    )
    report = {
        "schema": "lut-numeric-precheck-v1",
        "human_rating_join": calibration_report,
        "hypothesis": "render_t1 84% singletons = full-strength magnitude hiding equal shape",
        "sampling": {
            "largest_cluster": {"style_major": largest_key[0], "cluster_id": largest_key[1],
                                "members": len(largest)},
            "singleton_rule": f"sha256('{params['verify']['sample_salt']}' + preset_id)[:16] ascending",
            "singleton_sample": len(sampled),
            "singleton_total": len(singletons),
        },
        "render_t1": {
            "rows": len(cluster_rows), "clusters": len(grouped),
            "singletons": len(singletons),
            "singleton_fraction_of_clusters": round(len(singletons) / max(len(grouped), 1), 6),
            "singleton_fraction_of_rows": round(len(singletons) / max(len(cluster_rows), 1), 6),
            "max_cluster_size": sizes[largest_key],
        },
        "thresholds": {
            "tau_render": tau_render, "tau_shape": round(tau_shape, 6),
            "tau_shape_rule": TAU_SHAPE_RULE,
            "matched_fraction": round(matched_fraction, 8),
        },
        "bank_pairs": {
            "count": int(full_render.size),
            "render_distance_full_strength": quantiles(full_render_before),
            "render_distance_normalized": quantiles(full_render),
            "shape_distance": quantiles(shape_condensed),
            "dual_threshold_grid": [
                {
                    "tau_render": float(tau),
                    "pairs_within_both_full_strength":
                        int(((full_render_before <= float(tau)) & (shape_condensed <= tau_shape)).sum()),
                    "pairs_within_both_normalized":
                        int(((full_render <= float(tau)) & (shape_condensed <= tau_shape)).sum()),
                    "pairs_within_render_only_full_strength":
                        int((full_render_before <= float(tau)).sum()),
                    "pairs_within_render_only_normalized": int((full_render <= float(tau)).sum()),
                }
                for tau in params["duplicate"]["tau_render_grid"]
            ],
        },
        "groups": [_group_report("render_t1_largest_cluster", largest),
                   _group_report("render_t1_singletons_sampled", sampled)],
        "counts": counts,
        "assertions": assertion_counts(),
        "inputs": {
            **provenance,
            "params": str(args.params), "params_sha256": sha256_file(args.params),
            "source": str(SOURCE_FILE), "source_sha256": sha256_file(SOURCE_FILE),
        },
        "seconds": round(time.time() - started, 2),
    }
    verify_path = out_dir / VERIFY_JSON
    report["backup"] = backup_if_exists(verify_path)
    write_json(verify_path, report)
    require_assertions("assert_catalog_inputs", "assert_cache_params")
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0


def cmd_duplicate(args: argparse.Namespace) -> int:
    started = time.time()
    params = load_params(args.params)
    inputs = params["inputs"]
    out_dir = resolve_path(inputs["out_dir"])
    cache = RenderCache(out_dir / CACHE_NPZ)
    counts, provenance = check_cached_run(params, cache)
    duplicate = params["duplicate"]
    workers = int(params["runtime"]["workers"])
    chunk = int(params["runtime"]["pair_chunk"])
    render = render_distance_condensed(cache.lab_norm, workers, chunk)
    shape_matrix = shape_distance_matrix(cache.shape_unit)
    upper = np.triu_indices(shape_matrix.shape[0], k=1)
    shape = shape_matrix[upper]
    del shape_matrix
    tau_render = float(duplicate["tau_render"])
    tau_shape, matched_fraction = quantile_matched_tau(render, shape, tau_render)
    labels = dual_threshold_clusters(render, shape, tau_render, tau_shape,
                                     str(duplicate["linkage_method"]),
                                     float(duplicate["fcluster_t"]))

    members: dict[int, list[str]] = {}
    for label, preset_id in zip(labels.tolist(), cache.preset_ids):
        members.setdefault(label, []).append(preset_id)
    cluster_id_of: dict[str, str] = {}
    size_of: dict[str, int] = {}
    for group in members.values():
        group = sorted(group)
        for preset_id in group:
            cluster_id_of[preset_id] = group[0]
            size_of[preset_id] = len(group)
    sizes = [len(group) for group in members.values()]

    rows = []
    for preset_id, style_major in sorted(zip(cache.preset_ids, cache.style_majors)):
        number = cache.index[preset_id]
        rows.append({
            "row_type": "row", "preset_id": preset_id,
            "cluster_id": cluster_id_of[preset_id], "style_major": style_major,
            "cluster_size": size_of[preset_id],
            "alpha_norm": round(float(cache.alpha[number]), 6),
            "strength_capacity": "weak" if bool(cache.weak[number]) else "normal",
        })
    meta = {
        "row_type": "meta", "schema": DUP_SCHEMA, "rows": len(rows),
        "clusters": len(sizes), "singletons": int(sum(1 for size in sizes if size == 1)),
        "max_cluster_size": int(max(sizes)),
        "normalize_target_de00": float(params["normalize"]["target_de00"]),
        "normalize_tolerance": float(params["normalize"]["tolerance"]),
        "tau_render": tau_render, "tau_shape": round(tau_shape, 6),
        "tau_shape_rule": TAU_SHAPE_RULE,
        "matched_fraction": round(matched_fraction, 8),
        "combined_distance": "max(d_render/tau_render, d_shape/tau_shape)",
        "linkage": {"method": str(duplicate["linkage_method"]), "criterion": "distance",
                    "t": float(duplicate["fcluster_t"]), "scope": DUP_SCOPE},
        "shape_chroma_block": f"delta_c_ab ({cache.chroma_source}, canonical render, §4.3)",
        "probe_spec": PROBE_SPEC,
        "probe_sha256": probe_sha256(cache.probe_rgb),
        "params_sha256": sha256_file(args.params),
        "source_sha256": sha256_file(SOURCE_FILE),
        "cache_sha256": sha256_file(cache.path),
    }
    path = out_dir / DUP_TABLE
    backup = backup_if_exists(path)
    write_table(path, meta, rows)
    table_check = assert_table_file(path, len(rows), DUP_SCHEMA, DUP_ROW_KEYS)

    major_span = Counter()
    for group in members.values():
        major_span[len({cache.style_majors[cache.index[pid]] for pid in group})] += 1
    manifest = {
        **{key: value for key, value in meta.items() if key != "row_type"},
        "size_histogram": size_histogram(sizes),
        "singleton_fraction": round(sum(1 for size in sizes if size == 1) / len(sizes), 6),
        "mean_cluster_size": round(float(np.mean(sizes)), 6),
        "clusters_by_style_major_span": {str(key): int(value) for key, value in sorted(major_span.items())},
        "pair_distances": {
            "count": int(render.size),
            "render_normalized": quantiles(render), "shape": quantiles(shape),
            "pairs_within_tau_render": int((render <= tau_render).sum()),
            "pairs_within_tau_shape": int((shape <= tau_shape).sum()),
            "pairs_within_both": int(((render <= tau_render) & (shape <= tau_shape)).sum()),
        },
        "counts": counts,
        "inputs": {**provenance, "params": str(args.params),
                   "source": str(SOURCE_FILE)},
        "table_check": table_check,
        "assertions": assertion_counts(),
        "backup": backup,
        "artifacts": {"table": {"path": str(path), "sha256": sha256_file(path)}},
        "seconds": round(time.time() - started, 2),
    }
    manifest_path = path.with_suffix(".manifest.json")
    manifest["backup_manifest"] = backup_if_exists(manifest_path)
    write_json(manifest_path, manifest)
    require_assertions("assert_catalog_inputs", "assert_cache_params", "assert_table_file")
    print(json.dumps(manifest, ensure_ascii=False, indent=2))
    return 0


def cmd_family(args: argparse.Namespace) -> int:
    started = time.time()
    params = load_params(args.params)
    inputs = params["inputs"]
    out_dir = resolve_path(inputs["out_dir"])
    cache = RenderCache(out_dir / CACHE_NPZ)
    counts, provenance = check_cached_run(params, cache)
    family = params["family"]
    standardised, median, scale = robust_standardise(cache.style)
    matrix = block_equalise(standardised, cache.style_names)

    calibration = json.loads(resolve_path(inputs["calibration_pairs"]).read_text(encoding="utf-8"))
    calibration_ids = sorted({str(entry[key]) for entry in calibration["pairs"]
                              for key in ("preset_a", "preset_b")} & set(cache.index))
    calibration_index = np.asarray([cache.index[pid] for pid in calibration_ids], dtype=np.int64)
    calibration_distance = euclidean_matrix(matrix[calibration_index])
    full_distance = euclidean_matrix(matrix)

    sweep = []
    fits: dict[int, tuple[np.ndarray, np.ndarray]] = {}
    for k in family["k_grid"]:
        k = int(k)
        labels, centers, inertia = kmeans(matrix, k, int(family["seed"]), int(family["n_init"]),
                                          int(family["lloyd_iters"]))
        fits[k] = (labels, centers)
        sweep.append({
            "k": k, "inertia": round(inertia, 6),
            "silhouette_full_bank": round(silhouette(full_distance, labels), 6),
            "silhouette_calibration_subset": round(
                silhouette(calibration_distance, labels[calibration_index]), 6),
            **cluster_geometry(matrix, labels, centers),
        })
    selection_metric = str(family["selection_metric"])
    best = max(sweep, key=lambda row: (row[selection_metric], -row["k"]))
    chosen = int(best["k"])
    labels, centers = fits[chosen]
    names = family_names(cache.style, cache.style_names, labels, params)
    distance_to_centroid = np.linalg.norm(matrix - centers[labels], axis=1)

    rows = []
    for preset_id, style_major in sorted(zip(cache.preset_ids, cache.style_majors)):
        number = cache.index[preset_id]
        rows.append({
            "row_type": "row", "preset_id": preset_id,
            "family_id": f"fam_{int(labels[number]):03d}",
            "family_name": names[int(labels[number])],
            "style_major": style_major,
            "distance_to_centroid": round(float(distance_to_centroid[number]), 6),
        })
    family_sizes = np.bincount(labels, minlength=chosen).tolist()
    meta = {
        "row_type": "meta", "schema": FAMILY_SCHEMA, "rows": len(rows),
        "k": chosen, "k_grid": [int(value) for value in family["k_grid"]],
        "selection_metric": selection_metric,
        "selection_scope": f"{len(calibration_ids)} presets from {inputs['calibration_pairs']}",
        "normalize_target_de00": float(params["normalize"]["target_de00"]),
        "normalize_tolerance": float(params["normalize"]["tolerance"]),
        "style_vector": {"dims": int(matrix.shape[1]), "columns": cache.style_names,
                         "standardisation": "robust z (median, IQR/1.349) then block/sqrt(dim)"},
        "kmeans": {"seed": int(family["seed"]), "n_init": int(family["n_init"]),
                   "lloyd_iters": int(family["lloyd_iters"]), "init": "k-means++"},
        "chroma_field": "band_*_dC = ΔC*ab on the normalised canonical render "
                        "(§4.3; no HSL dSat anywhere in this vector)",
        "probe_spec": PROBE_SPEC,
        "probe_sha256": probe_sha256(cache.probe_rgb),
        "params_sha256": sha256_file(args.params),
        "source_sha256": sha256_file(SOURCE_FILE),
        "cache_sha256": sha256_file(cache.path),
    }
    path = out_dir / FAMILY_TABLE
    backup = backup_if_exists(path)
    write_table(path, meta, rows)
    table_check = assert_table_file(path, len(rows), FAMILY_SCHEMA, FAMILY_ROW_KEYS)

    manifest = {
        **{key: value for key, value in meta.items() if key != "row_type"},
        "sweep": sweep,
        "family_sizes": {f"fam_{index:03d}": int(size) for index, size in enumerate(family_sizes)},
        "family_names": {f"fam_{index:03d}": names[index] for index in range(chosen)},
        "family_size_quantiles": quantiles(np.asarray(family_sizes, dtype=np.float64)),
        "style_major_purity": round(float(np.mean([
            Counter(cache.style_majors[cache.index[row["preset_id"]]]
                    for row in rows if row["family_id"] == f"fam_{index:03d}").most_common(1)[0][1]
            / max(family_sizes[index], 1) for index in range(chosen)
        ])), 6),
        "counts": counts,
        "inputs": {**provenance, "params": str(args.params),
                   "source": str(SOURCE_FILE)},
        "table_check": table_check,
        "assertions": assertion_counts(),
        "backup": backup,
        "robust_center": {name: round(float(value), 6)
                          for name, value in zip(cache.style_names, median)},
        "robust_scale": {name: round(float(value), 6)
                         for name, value in zip(cache.style_names, scale)},
        # M6: ρ_h is reported per band, but is deliberately not a column of the
        # clustering vector (NOTES §4 M6 records why).
        "hue_band_rho_mean": {name: round(float(cache.hue_band_rho[:, index].mean()), 6)
                              for index, name in enumerate(cache.hue_band_rho_names)},
        "hue_band_rho_quantiles": {
            name: quantiles(cache.hue_band_rho[:, index])
            for index, name in enumerate(cache.hue_band_rho_names)},
        "artifacts": {"table": {"path": str(path), "sha256": sha256_file(path)}},
        "seconds": round(time.time() - started, 2),
    }
    manifest_path = path.with_suffix(".manifest.json")
    manifest["backup_manifest"] = backup_if_exists(manifest_path)
    write_json(manifest_path, manifest)
    sweep_path = out_dir / FAMILY_SWEEP
    sweep_backup = backup_if_exists(sweep_path)
    write_json(sweep_path, {"schema": "lut-style-family-sweep-v1", "sweep": sweep,
                            "selected_k": chosen, "selection_metric": selection_metric,
                            "backup": sweep_backup})
    require_assertions("assert_catalog_inputs", "assert_cache_params", "assert_table_file")
    print(json.dumps(manifest, ensure_ascii=False, indent=2))
    return 0


# --------------------------------------------------------------------------------- cli


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--params", type=Path, default=DEFAULT_PARAMS)
    sub = parser.add_subparsers(dest="command", required=True)
    for name, handler, help_text in (
        ("render", cmd_render, "render the canonical probe and solve the normalisation alpha"),
        ("verify", cmd_verify, "render_t1 singleton hypothesis pre-check"),
        ("duplicate", cmd_duplicate, "write clusters.duplicate.numeric-v1.jsonl"),
        ("family", cmd_family, "write clusters.style_family.numeric-v1.jsonl"),
    ):
        command = sub.add_parser(name, help=help_text)
        command.set_defaults(func=handler)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    return int(args.func(args))


if __name__ == "__main__":
    raise SystemExit(main())
