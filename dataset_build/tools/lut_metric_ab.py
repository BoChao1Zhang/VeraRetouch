"""LUT perceptual-distance A/B (task card A1f): three candidates vs two baselines.

Reads the 200 human-rated LUT pairs (60 main + 140 ext), renders a 4-probe x 4096-pixel
probe set through every annotated LUT, and scores five distance/probability definitions
against the ratings with a 5-fold stratified cross-validation at a pre-registered working
point.  Label: ``rating >= 3`` == "distinguishable".

Candidates
    C1  enhanced render distance   max(z(mean dE2000), z(p95 dE2000)) over 4 probes
    C2  linear blend               w * z(render_mean_4probe) + (1-w) * z(feature_L2),
                                   w grid-searched on each training fold
    C3  logistic                   11 pair features, sklearn LogisticRegression (L2),
                                   C in {0.1, 1, 10} chosen inside the training fold
Baselines
    B1  single-probe render mean (the existing ``render_distances.npz`` column)
    B2  39-dim effect-feature L2 (the ``lut-effect-39d-v1`` pair distance)

Working point (pre-registered): "predicted mergeable" == score <= t, where t is the
largest training-fold threshold whose ``rating>=3`` fraction is <= ``--max-frac`` over at
least ``--min-pairs`` training pairs; the test folds report coverage, the realised
``rating>=3`` violation rate, and the ``rating>=4`` rate.  No AUC anywhere.

Usage:
    python -m dataset_build.tools.lut_metric_ab \
        --config configs/agent_loop.terra-smoke.toml \
        --out-dir /home/bc/data/scratch/lut_clusters/metric_ab
"""
from __future__ import annotations

import argparse
import concurrent.futures as futures
import csv
import json
import multiprocessing as mp
import sys
import time
from pathlib import Path
from typing import Any, Callable, Sequence

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from dataset_build.tools.cluster_lut_effects import build_features  # noqa: E402
from dataset_build.tools.lut_render_distance import (  # noqa: E402
    DistanceStore,
    features_inputs,
    load_catalog,
    probe_pixels,
    render_all,
    save_npz_deterministic,
    sha256_file,
    spearman,
    write_json,
)

METRIC_SPEC = "lut-metric-ab-v1"
DEFAULT_OUT_DIR = Path("/home/bc/data/scratch/lut_clusters/metric_ab")
DEFAULT_DISTANCES = Path("/home/bc/data/scratch/lut_clusters/render_distances.npz")
DEFAULT_PROBES = (
    "/home/bc/datasets/MMArt-PPR10k/global/230_7/before.jpg",
    "/home/bc/VeraRetouch/docs/assets/local_retouch_agent_loop_20260818/"
    "scene_samples_low_512/rendered/landscape/source.jpg",
    "/home/bc/data/scratch/mask_backfill/src/cedd6f30b59b2667.png",
    "/home/bc/data/scratch/mask_backfill/src/322fc6aa784fefd5.jpg",
)
RAMP_DIM = 15
BAND_COUNT = 8
PAIR_CHUNK = 32
C_GRID = (0.1, 1.0, 10.0)
W_GRID = tuple(round(0.1 * step, 1) for step in range(11))
C3_FEATURES = (
    "render_mean", "render_p95", "feat_l2", "ramp_l2",
    "band_mean_abs_dhue", "band_mean_abs_dsat", "band_mean_abs_dlum",
    "band_max_abs_dhue", "band_max_abs_dsat", "band_max_abs_dlum",
    "sat_shift_diff",
)


# --------------------------------------------------------------------------- inputs


def read_ratings(main_csv: Path, ext_csv: Path,
                 extra: Sequence[tuple[Path, str, str]] = ()) -> dict[str, int]:
    """pair_id -> rating; ext rows whose item_id is not ``ext_pair_*`` are ignored.

    ``extra`` appends further ``(path, id_column, id_prefix)`` sources (later annotation
    batches such as b30) after the two defaults; omitting it reproduces the A1f call.
    """
    ratings: dict[str, int] = {}
    sources = [(main_csv, "pair_id", "pair_"), (ext_csv, "item_id", "ext_pair_")]
    sources.extend(extra)
    for path, column, prefix in sources:
        with path.open("r", encoding="utf-8-sig", newline="") as handle:
            for row in csv.DictReader(handle):
                pair_id = str(row.get(column) or "").strip()
                raw = str(row.get("rating") or "").strip()
                if not pair_id.startswith(prefix) or not raw:
                    continue
                if pair_id in ratings:
                    raise SystemExit(f"duplicate rating for {pair_id}")
                ratings[pair_id] = int(float(raw))
    return ratings


def read_pair_keys(paths: Sequence[Path]) -> dict[str, dict]:
    pairs: dict[str, dict] = {}
    for path in paths:
        key = json.loads(path.read_text(encoding="utf-8"))
        for pair_id, entry in key["pairs"].items():
            if pair_id in pairs:
                raise SystemExit(f"duplicate pair key {pair_id}")
            pairs[pair_id] = entry
    return pairs


# --------------------------------------------------------------------------- pair dE


_LAB: np.ndarray | None = None
_IDX: np.ndarray | None = None


def _pair_stats(job: tuple[int, int]) -> np.ndarray:
    from skimage.color import deltaE_ciede2000

    start, stop = job
    index = _IDX[start:stop]
    left = _LAB[index[:, 0]].astype(np.float64)
    right = _LAB[index[:, 1]].astype(np.float64)
    delta = deltaE_ciede2000(
        left.reshape(-1, 3), right.reshape(-1, 3)
    ).reshape(index.shape[0], -1)
    return np.stack(
        [delta.mean(axis=1), np.percentile(delta, 95.0, axis=1)], axis=1
    ).astype(np.float32)


def pair_delta_stats(lab: np.ndarray, index: np.ndarray, workers: int) -> np.ndarray:
    """(n, 2) array of per-pair (mean dE2000, p95 dE2000) over the merged probe pixels."""
    global _LAB, _IDX
    _LAB, _IDX = lab, index
    jobs = [
        (start, min(start + PAIR_CHUNK, index.shape[0]))
        for start in range(0, index.shape[0], PAIR_CHUNK)
    ]
    context = mp.get_context("fork")
    chunks: list[np.ndarray] = []
    with futures.ProcessPoolExecutor(max_workers=workers, mp_context=context) as pool:
        for values in pool.map(_pair_stats, jobs, chunksize=1):
            chunks.append(values)
    return np.concatenate(chunks) if chunks else np.zeros((0, 2), dtype=np.float32)


# --------------------------------------------------------------------------- features


def pair_feature_table(features: np.ndarray, index: np.ndarray,
                       stats: np.ndarray, single_probe: np.ndarray) -> dict[str, np.ndarray]:
    """Per-pair columns; ``features`` is the de_med-normalised 39-dim effect matrix."""
    diff = features[index[:, 0]] - features[index[:, 1]]
    bands = diff[:, RAMP_DIM:].reshape(-1, BAND_COUNT, 3)
    sat_a = features[index[:, 0], RAMP_DIM:].reshape(-1, BAND_COUNT, 3)[:, :, 1]
    sat_b = features[index[:, 1], RAMP_DIM:].reshape(-1, BAND_COUNT, 3)[:, :, 1]
    return {
        "render_mean_1probe": single_probe.astype(np.float64),
        "render_mean": stats[:, 0].astype(np.float64),
        "render_p95": stats[:, 1].astype(np.float64),
        "feat_l2": np.linalg.norm(diff, axis=1),
        "ramp_l2": np.linalg.norm(diff[:, :RAMP_DIM], axis=1),
        "band_mean_abs_dhue": np.abs(bands[:, :, 0]).mean(axis=1),
        "band_mean_abs_dsat": np.abs(bands[:, :, 1]).mean(axis=1),
        "band_mean_abs_dlum": np.abs(bands[:, :, 2]).mean(axis=1),
        "band_max_abs_dhue": np.abs(bands[:, :, 0]).max(axis=1),
        "band_max_abs_dsat": np.abs(bands[:, :, 1]).max(axis=1),
        "band_max_abs_dlum": np.abs(bands[:, :, 2]).max(axis=1),
        "sat_shift_diff": np.abs(sat_a.mean(axis=1) - sat_b.mean(axis=1)),
    }


def zstats(values: np.ndarray) -> tuple[float, float]:
    scale = float(np.std(values))
    return float(np.mean(values)), (scale if scale > 0.0 else 1.0)


# --------------------------------------------------------------------------- fitting


def stratified_folds(labels: np.ndarray, folds: int, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    assignment = np.zeros(labels.shape[0], dtype=np.int64)
    for value in (0, 1):
        member = np.flatnonzero(labels == value)
        member = member[rng.permutation(member.shape[0])]
        assignment[member] = np.arange(member.shape[0]) % folds
    return assignment


def working_threshold(scores: np.ndarray, labels: np.ndarray,
                      max_frac: float, min_pairs: int) -> float | None:
    """Largest t with frac(label==1 | score<=t) <= max_frac over >= min_pairs pairs."""
    order = np.argsort(scores, kind="stable")
    ordered_scores, ordered_labels = scores[order], labels[order]
    running = np.cumsum(ordered_labels)
    best: float | None = None
    for position in range(ordered_scores.shape[0]):
        if position + 1 < min_pairs:
            continue
        if position + 1 < ordered_scores.shape[0] and (
            ordered_scores[position + 1] == ordered_scores[position]
        ):
            continue
        if running[position] / (position + 1) <= max_frac:
            best = float(ordered_scores[position])
    return best


def logistic_fit(matrix: np.ndarray, labels: np.ndarray, penalty_c: float):
    from sklearn.linear_model import LogisticRegression

    model = LogisticRegression(
        penalty="l2", C=penalty_c, solver="lbfgs", max_iter=5000, tol=1e-8,
        fit_intercept=True, random_state=0,
    )
    model.fit(matrix, labels)
    return model


def log_loss(probability: np.ndarray, labels: np.ndarray) -> float:
    clipped = np.clip(probability, 1e-12, 1.0 - 1e-12)
    return float(-np.mean(labels * np.log(clipped) + (1 - labels) * np.log(1 - clipped)))


def fit_candidate(name: str, table: dict[str, np.ndarray], labels: np.ndarray,
                  train: np.ndarray, folds: int, seed: int) -> tuple[Callable, dict]:
    """Return (scorer over row indices, fit metadata).  Higher score == more different."""
    if name == "B1":
        column = table["render_mean_1probe"]
        return (lambda rows: column[rows]), {}
    if name == "B2":
        column = table["feat_l2"]
        return (lambda rows: column[rows]), {}
    if name in {"C1", "C1_alt_linear"}:
        mean_mu, mean_sd = zstats(table["render_mean"][train])
        p95_mu, p95_sd = zstats(table["render_p95"][train])

        def scorer(rows: np.ndarray, name: str = name) -> np.ndarray:
            zm = (table["render_mean"][rows] - mean_mu) / mean_sd
            zp = (table["render_p95"][rows] - p95_mu) / p95_sd
            return np.maximum(zm, zp) if name == "C1" else 0.5 * zm + 0.5 * zp

        return scorer, {"z_render_mean": [mean_mu, mean_sd], "z_render_p95": [p95_mu, p95_sd]}
    if name == "C2":
        mean_mu, mean_sd = zstats(table["render_mean"][train])
        feat_mu, feat_sd = zstats(table["feat_l2"][train])

        def blend(rows: np.ndarray, weight: float) -> np.ndarray:
            zm = (table["render_mean"][rows] - mean_mu) / mean_sd
            zf = (table["feat_l2"][rows] - feat_mu) / feat_sd
            return weight * zm + (1.0 - weight) * zf

        best: tuple[float, float, float] | None = None
        for weight in W_GRID:
            scores = blend(train, weight)
            threshold = working_threshold(scores, labels[train], MAX_FRAC, MIN_PAIRS)
            coverage = 0 if threshold is None else int(np.sum(scores <= threshold))
            rho = spearman(scores, labels[train])[0] if scores.shape[0] > 2 else 0.0
            key = (coverage, rho, -weight)
            if best is None or key > best[0]:
                best = (key, weight, coverage)
        weight = best[1]
        return (lambda rows: blend(rows, weight)), {"w": weight, "train_coverage": best[2]}
    if name == "C3":
        matrix_all = np.stack([table[key] for key in C3_FEATURES], axis=1)
        mu = matrix_all[train].mean(axis=0)
        sd = matrix_all[train].std(axis=0)
        sd[sd <= 0.0] = 1.0
        inner = stratified_folds(labels[train], folds - 1, seed + 977)
        losses: list[tuple[float, float]] = []
        for penalty_c in C_GRID:
            fold_losses: list[float] = []
            for fold in range(folds - 1):
                inner_train = train[inner != fold]
                inner_test = train[inner == fold]
                if np.unique(labels[inner_train]).shape[0] < 2 or inner_test.shape[0] == 0:
                    continue
                model = logistic_fit(
                    (matrix_all[inner_train] - mu) / sd, labels[inner_train], penalty_c
                )
                probability = model.predict_proba((matrix_all[inner_test] - mu) / sd)[:, 1]
                fold_losses.append(log_loss(probability, labels[inner_test]))
            losses.append((float(np.mean(fold_losses)) if fold_losses else float("inf"),
                           penalty_c))
        penalty_c = min(losses)[1]
        model = logistic_fit((matrix_all[train] - mu) / sd, labels[train], penalty_c)

        def scorer(rows: np.ndarray) -> np.ndarray:
            return model.predict_proba((matrix_all[rows] - mu) / sd)[:, 1]

        meta = {
            "C": penalty_c,
            "inner_logloss": {str(value): round(loss, 6) for loss, value in losses},
            "weights": {
                key: round(float(value), 6)
                for key, value in zip(C3_FEATURES, model.coef_[0])
            },
            "intercept": round(float(model.intercept_[0]), 6),
            "z_mu": mu.tolist(), "z_sd": sd.tolist(),
        }
        return scorer, meta
    raise SystemExit(f"unknown candidate {name}")


MAX_FRAC = 0.20
MIN_PAIRS = 5


def evaluate(name: str, table: dict[str, np.ndarray], labels: np.ndarray,
             ratings: np.ndarray, assignment: np.ndarray, folds: int,
             seed: int) -> dict[str, Any]:
    count = labels.shape[0]
    rows_all = np.arange(count)
    oof = np.full(count, np.nan)
    per_fold: list[dict[str, Any]] = []
    covered, violation, ge4 = 0, 0, 0
    for fold in range(folds):
        train = rows_all[assignment != fold]
        test = rows_all[assignment == fold]
        scorer, meta = fit_candidate(name, table, labels, train, folds, seed)
        train_scores = scorer(train)
        threshold = working_threshold(train_scores, labels[train], MAX_FRAC, MIN_PAIRS)
        test_scores = scorer(test)
        oof[test] = test_scores
        selected = (
            np.zeros(test.shape[0], dtype=bool) if threshold is None
            else test_scores <= threshold
        )
        covered += int(selected.sum())
        violation += int(labels[test][selected].sum())
        ge4 += int(np.sum(ratings[test][selected] >= 4))
        per_fold.append({
            "fold": fold,
            "train_n": int(train.shape[0]), "test_n": int(test.shape[0]),
            "threshold": (None if threshold is None else round(threshold, 6)),
            "train_coverage": (
                0 if threshold is None else int(np.sum(train_scores <= threshold))
            ),
            "test_coverage": int(selected.sum()),
            "test_violation_ge3": int(labels[test][selected].sum()),
            "test_ge4": int(np.sum(ratings[test][selected] >= 4)),
            "fit": {key: value for key, value in meta.items()
                    if key in {"w", "C", "train_coverage"}},
            "weights": meta.get("weights"),
        })
    full_scorer, full_meta = fit_candidate(name, table, labels, rows_all, folds, seed)
    full_scores = full_scorer(rows_all)
    rho_full, p_full = spearman(full_scores, ratings)
    rho_oof, p_oof = spearman(oof, ratings)
    return {
        "candidate": name,
        "spearman_full_fit": {"rho": round(rho_full, 4), "p": round(p_full, 8)},
        "spearman_out_of_fold": {"rho": round(rho_oof, 4), "p": round(p_oof, 8)},
        "working_point": {
            "rule": f"score <= t, t = max train-fold threshold with frac(rating>=3) "
                    f"<= {MAX_FRAC} over >= {MIN_PAIRS} train pairs",
            "coverage_n": covered,
            "coverage_frac": round(covered / count, 4),
            "violation_rate_ge3": (round(violation / covered, 4) if covered else None),
            "violation_n_ge3": violation,
            "rate_ge4": (round(ge4 / covered, 4) if covered else None),
            "n_ge4": ge4,
        },
        "per_fold": per_fold,
        "full_fit": {key: value for key, value in full_meta.items() if key != "z_mu"},
    }


# --------------------------------------------------------------------------- driver


def build_pool(store: DistanceStore, labeled: set[tuple[int, int]],
               cutoff: float) -> tuple[np.ndarray, np.ndarray]:
    """Unlabeled within-major pairs with single-probe mean dE <= cutoff."""
    index_of = {preset: position for position, preset in enumerate(store.preset_ids)}
    rows: list[np.ndarray] = []
    values: list[np.ndarray] = []
    for major in store.majors:
        members = np.asarray([index_of[preset] for preset in store.members(major)],
                             dtype=np.int64)
        condensed = store.condensed(major)
        left, right = np.triu_indices(members.shape[0], k=1)
        keep = np.flatnonzero(condensed <= cutoff)
        if keep.size == 0:
            continue
        pairs = np.stack([members[left[keep]], members[right[keep]]], axis=1)
        mask = np.asarray(
            [(int(a), int(b)) not in labeled for a, b in pairs], dtype=bool
        )
        rows.append(pairs[mask])
        values.append(condensed[keep][mask])
    if not rows:
        return np.zeros((0, 2), dtype=np.int64), np.zeros(0)
    return np.concatenate(rows), np.concatenate(values)


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", type=Path,
                        default=REPO_ROOT / "configs/agent_loop.terra-smoke.toml")
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument("--distances", type=Path, default=DEFAULT_DISTANCES)
    parser.add_argument("--main-csv", type=Path,
                        default=REPO_ROOT / "docs/assets/questionnaire/questionnaire.csv")
    parser.add_argument("--ext-csv", type=Path,
                        default=REPO_ROOT / "docs/assets/questionnaire/questionnaire_ext.csv")
    parser.add_argument("--extra-csv", action="append", default=None,
                        metavar="PATH::COLUMN::PREFIX",
                        help="additional rating CSV, e.g. "
                             "'.../questionnaire_b30.csv::item_id::b30_pair_'")
    parser.add_argument("--pair-key", type=Path, action="append", default=None)
    parser.add_argument("--probe", action="append", default=None)
    parser.add_argument("--pixels-per-probe", type=int, default=4096)
    parser.add_argument("--pixel-seed", type=int, default=20260819)
    parser.add_argument("--workers", type=int, default=24)
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--seed", type=int, default=20260819)
    parser.add_argument("--pool-cutoff", type=float, default=3.0)
    parser.add_argument("--candidates", type=int, default=30)
    args = parser.parse_args(argv)
    if not args.probe:
        args.probe = list(DEFAULT_PROBES)
    if not args.pair_key:
        base = REPO_ROOT / "docs/assets/lut_cluster_pilot_20260819"
        args.pair_key = [base / "questionnaire/pair_key.json",
                         base / "questionnaire_ext/pair_key_ext.json"]

    started = time.time()
    catalog, databuild, annotations = load_catalog(args.config)
    records = list(catalog.records)
    features, bands, unnormalised = build_features(records)
    store = DistanceStore(args.distances)
    if [row.preset_id for row in records] != store.preset_ids:
        raise SystemExit("catalog order does not match the stored distance npz")
    index_of = {row.preset_id: position for position, row in enumerate(records)}

    extra_sources: list[tuple[Path, str, str]] = []
    for spec in (args.extra_csv or []):
        fields = spec.split("::")
        if len(fields) != 3:
            raise SystemExit(f"--extra-csv expects PATH::COLUMN::PREFIX, got {spec!r}")
        extra_sources.append((Path(fields[0]), fields[1], fields[2]))
    ratings_map = read_ratings(args.main_csv, args.ext_csv, extra_sources)
    keys = read_pair_keys(args.pair_key)
    pair_ids = sorted(ratings_map)
    labeled_index = np.asarray(
        [[index_of[keys[pid]["preset_a"]], index_of[keys[pid]["preset_b"]]] for pid in pair_ids],
        dtype=np.int64,
    )
    ratings = np.asarray([ratings_map[pid] for pid in pair_ids], dtype=np.int64)
    labels = (ratings >= 3).astype(np.int64)
    labeled_single = np.asarray(
        [store.distance(keys[pid]["preset_a"], keys[pid]["preset_b"]) for pid in pair_ids],
        dtype=np.float64,
    )
    labeled_set = {(int(a), int(b)) for a, b in labeled_index}
    labeled_set |= {(int(b), int(a)) for a, b in labeled_index}

    pool_index, pool_single = build_pool(store, labeled_set, args.pool_cutoff)

    pixels, probe_meta = probe_pixels(
        [Path(value) for value in args.probe],
        args.pixels_per_probe * len(args.probe), args.pixel_seed,
    )
    render_started = time.time()
    lab, failures = render_all(records, pixels, databuild, args.workers)
    if failures:
        raise SystemExit(f"render failures ({len(failures)}): {failures[:5]}")
    render_seconds = time.time() - render_started

    pair_started = time.time()
    every_index = np.concatenate([labeled_index, pool_index])
    stats = pair_delta_stats(lab, every_index, args.workers)
    pair_seconds = time.time() - pair_started
    del lab

    single = np.concatenate([labeled_single, pool_single])

    # C1b item 10: the render + pairwise CIEDE2000 pass above is the only expensive,
    # non-reproducible-in-seconds stage of this tool. It lands before the model fits,
    # the candidate sweep and the report, so a failure in any of those costs minutes of
    # arithmetic instead of the whole render.
    args.out_dir.mkdir(parents=True, exist_ok=True)
    stats_path = args.out_dir / "pair_delta_stats.npz"
    save_npz_deterministic(stats_path, {
        "pair_index": every_index,
        "delta_mean_p95": stats,
        "render_mean_1probe": single.astype(np.float32),
        "labeled_count": np.asarray([labeled_index.shape[0]], dtype=np.int64),
    })

    table_all = pair_feature_table(features, every_index, stats, single)
    split = labeled_index.shape[0]
    table = {key: value[:split] for key, value in table_all.items()}
    pool_table = {key: value[split:] for key, value in table_all.items()}

    assignment = stratified_folds(labels, args.folds, args.seed)
    results = [
        evaluate(name, table, labels, ratings, assignment, args.folds, args.seed)
        for name in ("B1", "B2", "C1", "C1_alt_linear", "C2", "C3")
    ]

    rows_all = np.arange(split)
    c3_scorer, c3_meta = fit_candidate("C3", table, labels, rows_all, args.folds, args.seed)
    c3_full = c3_scorer(rows_all)
    c3_threshold = working_threshold(c3_full, labels, MAX_FRAC, MIN_PAIRS)
    c3_pool_matrix = np.stack([pool_table[key] for key in C3_FEATURES], axis=1)
    mu = np.asarray(c3_meta["z_mu"])
    sd = np.asarray(c3_meta["z_sd"])
    c3_model = logistic_fit(
        (np.stack([table[key] for key in C3_FEATURES], axis=1) - mu) / sd,
        labels, c3_meta["C"],
    )
    pool_probability = c3_model.predict_proba((c3_pool_matrix - mu) / sd)[:, 1]
    anchor = c3_threshold if c3_threshold is not None else MAX_FRAC
    order = np.lexsort((pool_index[:, 1], pool_index[:, 0], np.abs(pool_probability - anchor)))
    chosen = order[: args.candidates]

    stability = {
        feature: {
            "min": round(min(fold["weights"][feature] for fold in results[5]["per_fold"]), 6),
            "max": round(max(fold["weights"][feature] for fold in results[5]["per_fold"]), 6),
            "mean": round(float(np.mean([fold["weights"][feature]
                                         for fold in results[5]["per_fold"]])), 6),
            "full_fit": c3_meta["weights"][feature],
        }
        for feature in C3_FEATURES
    }

    candidate_path = args.out_dir / "backfill_candidates.jsonl"
    candidate_path.write_text("".join(
        json.dumps({
            "preset_a": store.preset_ids[int(pool_index[row, 0])],
            "preset_b": store.preset_ids[int(pool_index[row, 1])],
            "style_major": store.style_majors[int(pool_index[row, 0])],
            "c3_probability": round(float(pool_probability[row]), 6),
            "abs_gap_to_threshold": round(float(abs(pool_probability[row] - anchor)), 6),
            "render_mean_1probe": round(float(pool_table["render_mean_1probe"][row]), 6),
            "render_mean_4probe": round(float(pool_table["render_mean"][row]), 6),
            "render_p95_4probe": round(float(pool_table["render_p95"][row]), 6),
            "feat_l2": round(float(pool_table["feat_l2"][row]), 6),
        }, ensure_ascii=False, sort_keys=True, separators=(",", ":")) + "\n"
        for row in chosen
    ), encoding="utf-8", newline="\n")

    pairs_path = args.out_dir / "labeled_pairs.json"
    write_json(pairs_path, {
        "schema": METRIC_SPEC,
        "pairs": [
            {
                "pair_id": pair_id, "rating": int(ratings[position]),
                "label_distinguishable": int(labels[position]),
                "fold": int(assignment[position]),
                "style_major": store.style_majors[int(labeled_index[position, 0])],
                "preset_a": store.preset_ids[int(labeled_index[position, 0])],
                "preset_b": store.preset_ids[int(labeled_index[position, 1])],
                **{key: round(float(value[position]), 6) for key, value in table.items()},
            }
            for position, pair_id in enumerate(pair_ids)
        ],
    })

    report = {
        "schema": METRIC_SPEC,
        "data": {
            "labeled_pairs": split,
            "rating_histogram": {str(value): int(np.sum(ratings == value))
                                 for value in range(1, 6)},
            "label_positive_ge3": int(labels.sum()),
            "folds": args.folds, "fold_seed": args.seed,
            "fold_sizes": [int(np.sum(assignment == fold)) for fold in range(args.folds)],
            "fold_positive": [int(labels[assignment == fold].sum())
                              for fold in range(args.folds)],
            "unlabeled_pool": int(pool_index.shape[0]),
            "pool_cutoff_single_probe": args.pool_cutoff,
        },
        "render": {
            "probes": probe_meta,
            "pixels_per_probe": args.pixels_per_probe,
            "total_pixels": int(pixels.shape[0]),
            "pixel_seed": args.pixel_seed,
            "metric": "per-pixel CIEDE2000 on sRGB->Lab; mean and p95 over merged probes",
        },
        "features": {
            "spec": "lut-effect-39d-v1 (de_med-normalised)",
            "band_order": list(bands),
            "unnormalised_records": unnormalised,
            "c3_features": list(C3_FEATURES),
        },
        "working_point": {"max_frac": MAX_FRAC, "min_pairs": MIN_PAIRS},
        "results": results,
        "c3_weight_stability": stability,
        "c3_full_fit": {
            "C": c3_meta["C"], "intercept": c3_meta["intercept"],
            "threshold": (None if c3_threshold is None else round(c3_threshold, 6)),
            "candidate_anchor": round(float(anchor), 6),
        },
        "artifacts": {
            "backfill_candidates": {"path": str(candidate_path),
                                    "sha256": sha256_file(candidate_path)},
            "labeled_pairs": {"path": str(pairs_path), "sha256": sha256_file(pairs_path)},
            "pair_delta_stats": {"path": str(stats_path), "sha256": sha256_file(stats_path)},
        },
        "inputs": {
            "config": str(args.config),
            "annotations": str(annotations), "annotations_sha256": sha256_file(annotations),
            "databuild_config": str(databuild),
            # C1b item 11: the preset bank decides which LUTs the catalog contains.
            **features_inputs(databuild),
            "distances": str(args.distances), "distances_sha256": sha256_file(args.distances),
            "main_csv": str(args.main_csv), "main_csv_sha256": sha256_file(args.main_csv),
            "ext_csv": str(args.ext_csv), "ext_csv_sha256": sha256_file(args.ext_csv),
            "extra_csv": [str(path) for path, _, _ in extra_sources],
            "extra_csv_sha256": [sha256_file(path) for path, _, _ in extra_sources],
            "extra_csv_spec": list(args.extra_csv or []),
            "pair_keys": [str(path) for path in args.pair_key],
            "pair_keys_sha256": [sha256_file(path) for path in args.pair_key],
        },
        "seconds": {
            "render": round(render_seconds, 2), "pairs": round(pair_seconds, 2),
            "total": round(time.time() - started, 2),
        },
    }
    report_path = args.out_dir / "metric_ab.json"
    write_json(report_path, report)
    manifest_path = args.out_dir / "manifest.json"
    write_json(manifest_path, {
        "schema": f"{METRIC_SPEC}-manifest",
        "inputs": report["inputs"],
        "artifacts": {
            **report["artifacts"],
            "report": {"path": str(report_path), "sha256": sha256_file(report_path)},
        },
        "determinism": {
            "pixel_seed": args.pixel_seed, "fold_seed": args.seed,
            "logistic_random_state": 0, "grids": {"C": list(C_GRID), "w": list(W_GRID)},
        },
    })
    print(json.dumps({
        "report": str(report_path), "report_sha256": sha256_file(report_path),
        "manifest": str(manifest_path), "manifest_sha256": sha256_file(manifest_path),
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
        "candidates": str(candidate_path), "pool": int(pool_index.shape[0]),
        "seconds": report["seconds"],
    }, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
