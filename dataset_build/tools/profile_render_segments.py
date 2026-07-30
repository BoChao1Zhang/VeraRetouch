#!/usr/bin/env python3
"""Profile canonical render/QA segments on one durable real group.

The tool is deliberately read-only with respect to databuild state. It reads a
completed group from ``groups.jsonl``, recreates its global LUT candidates in a
bounded temporary directory, warms both GPU models, and reports absolute wall
times without touching journals, manifests, or NFS datasets.
"""
from __future__ import annotations

import argparse
import json
import math
import os
import statistics
import tempfile
import time
from concurrent.futures import Future, ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Callable, Iterable, Sequence

import numpy as np

from construct.agent import _postprocess_candidate
from construct.canonical_masks import MaskAsset
from construct.canonical_qa import OneAlignScorer, _score_paths, _stats, rank_candidates
from construct.config import DatabuildConfig, load_config
from construct.presets import PresetCatalog, PresetRecord, TaxonomyLink
from construct.rendering import (
    LocalGpuOnlyRenderer,
    PreparedSource,
    RenderedCandidate,
    preprocess_source,
    save_candidate_jpeg,
)
from construct.visibility import (
    objective_edit_hints_from_lab,
    prepare_torch_lab_reference,
    prepare_working_after,
    prepare_working_reference,
    srgb_to_lab,
    visibility_and_hints_torch,
    visibility_metrics_from_lab,
)
from dataset_build.tools.archive_reader import open_image, open_rgb


class ProfileError(RuntimeError):
    """The selected fixture cannot exercise the canonical profiling path."""


@dataclass(frozen=True, slots=True)
class Fixture:
    group: dict[str, Any]
    source_path: str
    presets: tuple[PresetRecord, ...]
    masks: tuple[MaskAsset | None, ...]
    candidates: tuple[dict[str, Any], ...]


def _read_groups(path: Path) -> Iterable[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ProfileError(f"invalid JSONL at {path}:{line_number}") from exc
            if isinstance(row, dict):
                yield row


def _select_group(
    path: Path, group_id: str | None, render_mode: str
) -> dict[str, Any]:
    for group in _read_groups(path):
        if group_id is not None and group.get("group_id") != group_id:
            continue
        candidates = group.get("candidates")
        if group.get("render_mode") == render_mode and isinstance(candidates, list) \
                and len(candidates) == 8 \
                and all(candidate.get("format") == "lut" for candidate in candidates):
            return group
        if group_id is not None:
            raise ProfileError(
                f"selected group must be a complete {render_mode} LUT-only group"
            )
    detail = f" {group_id!r}" if group_id is not None else ""
    raise ProfileError(
        f"no matching {render_mode} LUT-only group{detail} in {path}"
    )


def _fixture(path: Path, group_id: str | None, render_mode: str) -> Fixture:
    group = _select_group(path, group_id, render_mode)
    source_path = group.get("source_path")
    candidates = group.get("candidates")
    if not isinstance(source_path, str) or not source_path:
        raise ProfileError("selected group has no source_path")
    presets: list[PresetRecord] = []
    masks: list[MaskAsset | None] = []
    for candidate in candidates:
        recipe = candidate.get("recipe")
        if not isinstance(recipe, dict):
            raise ProfileError("candidate recipe is missing")
        preset_id = candidate.get("preset_id")
        preset_path = recipe.get("preset_path")
        if not isinstance(preset_id, str) or not isinstance(preset_path, str):
            raise ProfileError("candidate preset identity is incomplete")
        path_value = Path(preset_path)
        if not path_value.is_file():
            raise ProfileError(f"preset path is not locally readable: {path_value}")
        presets.append(PresetRecord(
            preset_id=preset_id,
            path=path_value,
            format="lut",
            kind="lut",
            style_name=None,
            fidelity_de=None,
            render_engine="gpu_lut",
        ))
        if render_mode == "global":
            masks.append(None)
            continue
        mask_id = candidate.get("mask_id")
        cgt_path = candidate.get("cgt_path")
        slot_mode = candidate.get("slot_mode")
        region = candidate.get("region")
        if not all(isinstance(value, str) and value for value in (
            mask_id, cgt_path, slot_mode, region
        )):
            raise ProfileError("local candidate mask identity is incomplete")
        alpha = np.asarray(open_image(cgt_path).convert("L"), dtype=np.float32) / 255.0
        masks.append(MaskAsset(
            mask_id=mask_id,
            mode=slot_mode,
            effective_alpha=alpha,
            raw_alpha_mean=float(candidate["raw_alpha_mean"]),
            amount=float(candidate["amount"]),
            effective_alpha_mean=float(candidate["effective_alpha_mean"]),
            geometry=candidate.get("geometry"),
            region=region,
        ))
    if len({preset.preset_id for preset in presets}) != 8:
        raise ProfileError("selected group does not contain eight distinct presets")
    return Fixture(
        group, source_path, tuple(presets), tuple(masks), tuple(candidates)
    )


def _percentile(samples: Sequence[float], percentile: float) -> float:
    if not samples:
        raise ValueError("cannot summarize an empty sample")
    ordered = sorted(float(value) for value in samples)
    position = (len(ordered) - 1) * percentile
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return ordered[lower]
    fraction = position - lower
    return ordered[lower] * (1.0 - fraction) + ordered[upper] * fraction


def _summary(samples_ms: Sequence[float], *, candidates: int = 8) -> dict[str, Any]:
    values = [round(float(value), 6) for value in samples_ms]
    median = statistics.median(values)
    return {
        "samples_ms": values,
        "median_ms": round(median, 6),
        "p95_ms": round(_percentile(values, 0.95), 6),
        "per_candidate_median_ms": round(median / candidates, 6),
    }


def _measure(
    operation: Callable[[], Any],
    *,
    warmup: int,
    iterations: int,
) -> tuple[list[float], Any]:
    result: Any = None
    for _ in range(warmup):
        result = operation()
    samples: list[float] = []
    for _ in range(iterations):
        started = time.perf_counter_ns()
        result = operation()
        samples.append((time.perf_counter_ns() - started) / 1_000_000.0)
    return samples, result


def _expect_rendered(
    values: Sequence[RenderedCandidate | Exception],
    *,
    expected: int = 8,
) -> list[RenderedCandidate]:
    output: list[RenderedCandidate] = []
    for value in values:
        if isinstance(value, Exception):
            raise ProfileError(f"renderer returned {type(value).__name__}: {value}")
        output.append(value)
    if len(output) != expected:
        raise ProfileError(
            f"renderer returned {len(output)} candidates instead of {expected}"
        )
    return output


def _render_two_waves(
    renderer: LocalGpuOnlyRenderer,
    executor: ThreadPoolExecutor,
    source: PreparedSource,
    presets: Sequence[PresetRecord],
    masks: Sequence[MaskAsset | None],
    on_chunk: Callable[[list[int], list[RenderedCandidate]], None] | None = None,
) -> list[RenderedCandidate]:
    chunks = [list(range(worker, len(presets), 2)) for worker in range(2)]
    futures = {
        executor.submit(
            renderer.render_many,
            source,
            [(presets[index], masks[index]) for index in indexes],
        ): indexes
        for indexes in chunks
    }
    output: list[RenderedCandidate | None] = [None] * len(presets)
    for future in as_completed(futures):
        indexes = futures[future]
        rendered = _expect_rendered(future.result(), expected=len(indexes))
        for index, value in zip(indexes, rendered):
            output[index] = value
        if on_chunk is not None:
            on_chunk(indexes, rendered)
    if any(value is None for value in output):
        raise ProfileError("two-wave renderer lost a candidate")
    return [value for value in output if value is not None]


def _working_pairs(
    reference: Any,
    rendered: Sequence[RenderedCandidate],
    masks: Sequence[MaskAsset | None],
) -> list[tuple[Any, Any]]:
    return [
        prepare_working_after(
            reference,
            value.pixels,
            mask.effective_alpha if mask is not None else None,
        )
        for value, mask in zip(rendered, masks)
    ]


def _visibility(
    config: DatabuildConfig,
    working_reference: Any,
    torch_reference: Any,
    pair: tuple[Any, Any],
) -> None:
    after, weight = pair
    kwargs = {
        "weight": weight,
        "visible_de_min": config.render.visible_de_min,
        "visible_fraction_de": config.render.visible_fraction_de,
        "visible_fraction_min": config.render.visible_fraction_min,
    }
    if config.render.visibility_backend == "torch":
        metrics, _ = visibility_and_hints_torch(torch_reference, after, **kwargs)
    else:
        after_lab = srgb_to_lab(after)
        metrics = visibility_metrics_from_lab(working_reference.lab, after_lab, **kwargs)
        objective_edit_hints_from_lab(
            working_reference.lab, after_lab, weight=weight
        )
    if not metrics.accepted:
        raise ProfileError("fixture candidate failed the configured visibility gate")


def profile(
    *,
    config: DatabuildConfig,
    fixture: Fixture,
    warmup: int,
    iterations: int,
    scratch_root: Path,
    include_onealign: bool,
    iaa_batch: int | None = None,
    onealign_concurrency_probe: bool = False,
) -> dict[str, Any]:
    effective_iaa_batch = config.render.iaa_batch if iaa_batch is None else iaa_batch
    catalog = PresetCatalog.from_links(
        TaxonomyLink(preset=preset, major="profile", minor="profile")
        for preset in fixture.presets
    )
    renderer = LocalGpuOnlyRenderer.create(config)
    renderer.bind_catalog(catalog)
    renderer.assert_ready()
    scorer = OneAlignScorer.create("cuda:0") if include_onealign else None

    stages: dict[str, dict[str, Any]] = {}
    with tempfile.TemporaryDirectory(prefix="veraretouch-h1-", dir=scratch_root) as temp_name, \
            ThreadPoolExecutor(max_workers=2, thread_name_prefix="profile-render") as render_pool, \
            ThreadPoolExecutor(
                max_workers=config.render.postprocess_workers,
                thread_name_prefix="profile-postprocess",
            ) as postprocess_pool:
        temp = Path(temp_name)
        candidate_paths = [temp / f"candidate-{index:02d}.jpg" for index in range(8)]

        samples, prepared = _measure(
            lambda: preprocess_source(fixture.source_path, config.render.short_edge),
            warmup=warmup,
            iterations=iterations,
        )
        stages["source_decode_resize_group"] = _summary(samples, candidates=1)

        samples, working_reference = _measure(
            lambda: prepare_working_reference(
                prepared.pixels, config.render.diff_short_edge
            ),
            warmup=warmup,
            iterations=iterations,
        )
        stages["working_reference_group"] = _summary(samples, candidates=1)

        samples, torch_reference = _measure(
            lambda: prepare_torch_lab_reference(working_reference, renderer.device)
            if config.render.visibility_backend == "torch" else None,
            warmup=warmup,
            iterations=iterations,
        )
        stages["visibility_reference_group"] = _summary(samples, candidates=1)

        requests = list(zip(fixture.presets, fixture.masks))

        def cold_lut_load_all() -> list[Any]:
            loader = type(renderer._lut_loader)(config.presets.bank_dir)
            try:
                return [loader.load(preset.path) for preset in fixture.presets]
            finally:
                packed = getattr(loader, "_packed", None)
                if packed is not None:
                    packed.close()

        samples, _ = _measure(
            cold_lut_load_all, warmup=warmup, iterations=iterations
        )
        stages["renderer_lut_cold_load_group"] = _summary(samples)

        samples, _ = _measure(
            lambda: [
                renderer._lut_loader.load(preset.path)
                for preset in fixture.presets
            ],
            warmup=warmup,
            iterations=iterations,
        )
        stages["renderer_lut_cache_lookup_group"] = _summary(samples)

        torch = renderer._torch

        def upload_source_sync() -> Any:
            torch.cuda.synchronize(renderer.device)
            uploaded = renderer._upload(prepared.pixels)
            torch.cuda.synchronize(renderer.device)
            return uploaded

        samples, uploaded = _measure(
            upload_source_sync, warmup=warmup, iterations=iterations
        )
        stages["renderer_source_upload_sync_group"] = _summary(
            samples, candidates=1
        )

        def apply_luts_sync() -> list[Any]:
            torch.cuda.synchronize(renderer.device)
            values = [
                renderer._apply_lut(uploaded, preset)
                for preset in fixture.presets
            ]
            torch.cuda.synchronize(renderer.device)
            return values

        samples, applied = _measure(
            apply_luts_sync, warmup=warmup, iterations=iterations
        )
        stages["renderer_lut_operator_sync_group"] = _summary(samples)

        def transfer_outputs_sync() -> np.ndarray:
            torch.cuda.synchronize(renderer.device)
            batch = torch.stack([
                edited[0].permute(1, 2, 0) for edited, _diagnostics in applied
            ])
            arrays = batch.detach().cpu().numpy().astype(np.float32, copy=False)
            torch.cuda.synchronize(renderer.device)
            return arrays

        samples, _ = _measure(
            transfer_outputs_sync, warmup=warmup, iterations=iterations
        )
        stages["renderer_host_transfer_sync_group"] = _summary(samples)

        samples, one_wave_raw = _measure(
            lambda: renderer.render_many(prepared, requests),
            warmup=warmup,
            iterations=iterations,
        )
        rendered = _expect_rendered(one_wave_raw)
        stages["renderer_one_wave_group"] = _summary(samples)

        samples, _ = _measure(
            lambda: _render_two_waves(
                renderer, render_pool, prepared, fixture.presets, fixture.masks
            ),
            warmup=warmup,
            iterations=iterations,
        )
        stages["renderer_current_two_wave_group"] = _summary(samples)

        def resize_all() -> list[tuple[Any, Any]]:
            return _working_pairs(working_reference, rendered, fixture.masks)

        samples, working_pairs = _measure(
            resize_all, warmup=warmup, iterations=iterations
        )
        stages["working_after_resize_group_serial"] = _summary(samples)

        def visibility_all() -> None:
            for pair in working_pairs:
                _visibility(config, working_reference, torch_reference, pair)

        samples, _ = _measure(
            visibility_all, warmup=warmup, iterations=iterations
        )
        stages["visibility_hints_group_serial"] = _summary(samples)

        def jpeg_all() -> None:
            for path, value in zip(candidate_paths, rendered):
                save_candidate_jpeg(
                    value.pixels, path, quality=config.render.jpeg_quality
                )

        samples, _ = _measure(jpeg_all, warmup=warmup, iterations=iterations)
        stages["jpeg_encode_fsync_group_serial"] = _summary(samples)

        reference_bundle = SimpleNamespace(
            working=working_reference, torch=torch_reference
        )

        def postprocess_parallel(
            values: Sequence[RenderedCandidate] = rendered,
        ) -> list[Any]:
            futures = [
                postprocess_pool.submit(
                    _postprocess_candidate,
                    reference_bundle,
                    value.pixels,
                    mask.effective_alpha if mask is not None else None,
                    path,
                    config.render,
                )
                for path, value, mask in zip(
                    candidate_paths, values, fixture.masks
                )
            ]
            return [future.result() for future in futures]

        samples, postprocess_results = _measure(
            postprocess_parallel, warmup=warmup, iterations=iterations
        )
        stages["postprocess_current_parallel_group"] = _summary(samples)

        def qa_stats() -> None:
            _stats(fixture.source_path)
            for path in candidate_paths:
                _stats(str(path))

        samples, _ = _measure(qa_stats, warmup=warmup, iterations=iterations)
        stages["qa_stats_decode_group"] = _summary(samples)

        def qa_rgb_decode() -> None:
            open_rgb(fixture.source_path)
            for path in candidate_paths:
                open_rgb(path)

        samples, _ = _measure(qa_rgb_decode, warmup=warmup, iterations=iterations)
        stages["onealign_rgb_decode_group"] = _summary(samples)

        benchmark_candidates = [
            {**candidate, "after_path": str(path)}
            for candidate, path in zip(fixture.candidates, candidate_paths)
        ]
        cached_benchmark_candidates = [
            {**candidate, "_qa_stats": result.qa_stats}
            for candidate, result in zip(benchmark_candidates, postprocess_results)
        ]
        if scorer is not None:
            score_paths = [fixture.source_path] + [str(path) for path in candidate_paths]
            samples, _ = _measure(
                lambda: _score_paths(scorer, score_paths, effective_iaa_batch),
                warmup=warmup,
                iterations=iterations,
            )
            stages["onealign_decode_forward_group"] = _summary(samples)

            concurrency_probe = None
            if onealign_concurrency_probe:
                second_scorer = OneAlignScorer.create("cuda:0")

                samples, aggregate_scores = _measure(
                    lambda: _score_paths(
                        scorer,
                        score_paths * 2,
                        len(score_paths) * 2,
                    ),
                    warmup=warmup,
                    iterations=iterations,
                )
                stages["onealign_two_groups_aggregate"] = _summary(
                    samples, candidates=16
                )
                stages["onealign_two_groups_aggregate"]["per_group_median_ms"] = round(
                    stages["onealign_two_groups_aggregate"]["median_ms"] / 2.0,
                    6,
                )

                with ThreadPoolExecutor(
                    max_workers=2, thread_name_prefix="profile-onealign"
                ) as onealign_pool:
                    def score_two_concurrently() -> list[list[float | None]]:
                        futures = [
                            onealign_pool.submit(
                                _score_paths,
                                selected_scorer,
                                score_paths,
                                effective_iaa_batch,
                            )
                            for selected_scorer in (scorer, second_scorer)
                        ]
                        return [future.result() for future in futures]

                    samples, concurrent_scores = _measure(
                        score_two_concurrently,
                        warmup=warmup,
                        iterations=iterations,
                    )
                stages["onealign_two_groups_dual_scorer"] = _summary(
                    samples, candidates=16
                )
                stages["onealign_two_groups_dual_scorer"]["per_group_median_ms"] = round(
                    stages["onealign_two_groups_dual_scorer"]["median_ms"] / 2.0,
                    6,
                )
                baseline_scores = _score_paths(
                    scorer, score_paths, effective_iaa_batch
                )
                aggregate_first = aggregate_scores[:len(score_paths)]
                score_deltas = [
                    abs(float(left) - float(right))
                    for left, right in zip(baseline_scores, aggregate_first)
                ]
                concurrent_deltas = [
                    abs(float(left) - float(right))
                    for result in concurrent_scores
                    for left, right in zip(baseline_scores, result)
                ]
                torch = renderer._torch
                concurrency_probe = {
                    "aggregate_batch": len(score_paths) * 2,
                    "aggregate_max_abs_score_delta": round(
                        max(score_deltas, default=0.0), 6
                    ),
                    "dual_scorer_max_abs_score_delta": round(
                        max(concurrent_deltas, default=0.0), 6
                    ),
                    "cuda0_memory_allocated_bytes": (
                        int(torch.cuda.memory_allocated("cuda:0"))
                        if torch is not None else None
                    ),
                    "cuda0_memory_reserved_bytes": (
                        int(torch.cuda.memory_reserved("cuda:0"))
                        if torch is not None else None
                    ),
                }

            samples, ranked_result = _measure(
                lambda: rank_candidates(
                    fixture.source_path,
                    benchmark_candidates,
                    scorer,
                    batch_size=effective_iaa_batch,
                ),
                warmup=warmup,
                iterations=iterations,
            )
            stages["qa_rank_full_group"] = _summary(samples)

            samples, cached_ranked_result = _measure(
                lambda: rank_candidates(
                    fixture.source_path,
                    cached_benchmark_candidates,
                    scorer,
                    batch_size=effective_iaa_batch,
                ),
                warmup=warmup,
                iterations=iterations,
            )
            stages["qa_rank_cached_group"] = _summary(samples)

            qa_comparison = None
            if effective_iaa_batch != config.render.iaa_batch:
                baseline = rank_candidates(
                    fixture.source_path,
                    cached_benchmark_candidates,
                    scorer,
                    batch_size=config.render.iaa_batch,
                )
                baseline_by_id = {
                    candidate["candidate_id"]: candidate
                    for candidate in baseline.candidates
                }
                effective_by_id = {
                    candidate["candidate_id"]: candidate
                    for candidate in cached_ranked_result.candidates
                }
                score_deltas = []
                for candidate_id, baseline_candidate in baseline_by_id.items():
                    baseline_score = baseline_candidate["qa"]["onealign"]
                    effective_score = effective_by_id[candidate_id]["qa"]["onealign"]
                    if baseline_score is not None and effective_score is not None:
                        score_deltas.append(abs(effective_score - baseline_score))
                qa_comparison = {
                    "baseline_batch": config.render.iaa_batch,
                    "effective_batch": effective_iaa_batch,
                    "max_abs_onealign_delta": round(max(score_deltas, default=0.0), 6),
                    "baseline_winner_ids": list(baseline.winner_ids),
                    "effective_winner_ids": list(cached_ranked_result.winner_ids),
                    "baseline_rank_order": [
                        candidate["candidate_id"]
                        for candidate in sorted(
                            baseline.candidates, key=lambda candidate: candidate["rank"]
                        )
                    ],
                    "effective_rank_order": [
                        candidate["candidate_id"]
                        for candidate in sorted(
                            cached_ranked_result.candidates,
                            key=lambda candidate: candidate["rank"],
                        )
                    ],
                }

            def full_group_replay() -> None:
                current_source = preprocess_source(
                    fixture.source_path, config.render.short_edge
                )
                current_working = prepare_working_reference(
                    current_source.pixels, config.render.diff_short_edge
                )
                current_torch = (
                    prepare_torch_lab_reference(current_working, renderer.device)
                    if config.render.visibility_backend == "torch" else None
                )
                current_bundle = SimpleNamespace(
                    working=current_working, torch=current_torch
                )
                current_rendered = _expect_rendered(
                    renderer.render_many(
                        current_source,
                        list(zip(fixture.presets, fixture.masks)),
                    )
                )
                postprocess_futures: dict[int, Future[Any]] = {
                    index: postprocess_pool.submit(
                        _postprocess_candidate,
                        current_bundle,
                        value.pixels,
                        (
                            fixture.masks[index].effective_alpha
                            if fixture.masks[index] is not None else None
                        ),
                        candidate_paths[index],
                        config.render,
                    )
                    for index, value in enumerate(current_rendered)
                }
                current_candidates = []
                for index in range(8):
                    result = postprocess_futures[index].result()
                    current_candidates.append({
                        **benchmark_candidates[index],
                        "_qa_stats": result.qa_stats,
                    })
                rank_candidates(
                    fixture.source_path,
                    current_candidates,
                    scorer,
                    batch_size=effective_iaa_batch,
                )

            samples, _ = _measure(
                full_group_replay, warmup=warmup, iterations=iterations
            )
            stages["full_group_replay"] = _summary(samples)

    production_renderer_stage = (
        "renderer_one_wave_group"
        if all(mask is None for mask in fixture.masks)
        else "renderer_current_two_wave_group"
    )
    critical_names = [
        "source_decode_resize_group",
        "working_reference_group",
        "visibility_reference_group",
        production_renderer_stage,
        "postprocess_current_parallel_group",
    ]
    if include_onealign:
        critical_names.append("qa_rank_cached_group")
    estimated_ms = sum(stages[name]["median_ms"] for name in critical_names)
    for name in critical_names:
        stages[name]["pct_of_estimated_pipeline"] = round(
            100.0 * stages[name]["median_ms"] / estimated_ms, 3
        )

    torch = renderer._torch
    gpu_name = None
    if torch is not None and renderer.device.startswith("cuda"):
        gpu_name = torch.cuda.get_device_name(renderer.device)
    result = {
        "schema_version": 1,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "fixture": {
            "build_id": fixture.group.get("build_id"),
            "group_id": fixture.group.get("group_id"),
            "source_id": fixture.group.get("source_id"),
            "render_mode": fixture.group.get("render_mode"),
            "preset_ids": [preset.preset_id for preset in fixture.presets],
            "width": prepared.width,
            "height": prepared.height,
        },
        "runtime": {
            "warmup": warmup,
            "iterations": iterations,
            "device": renderer.device,
            "gpu_name": gpu_name,
            "onealign_included": include_onealign,
            "render": {
                "short_edge": config.render.short_edge,
                "diff_short_edge": config.render.diff_short_edge,
                "jpeg_quality": config.render.jpeg_quality,
                "gpu_concurrency": config.render.gpu_concurrency,
                "config_iaa_batch": config.render.iaa_batch,
                "profile_iaa_batch": effective_iaa_batch,
                "postprocess_workers": config.render.postprocess_workers,
                "visibility_backend": config.render.visibility_backend,
            },
        },
        "stages": stages,
        "estimated_pipeline": {
            "stage_names": critical_names,
            "median_ms": round(estimated_ms, 6),
            "groups_per_hour": round(3_600_000.0 / estimated_ms, 3),
        },
    }
    if include_onealign:
        result["qa_batch_comparison"] = qa_comparison
        result["onealign_concurrency_probe"] = concurrency_probe
    return result


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + f".tmp.{os.getpid()}")
    with tmp.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(tmp, path)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--groups-jsonl", type=Path, required=True)
    parser.add_argument("--group-id")
    parser.add_argument(
        "--render-mode", choices=("global", "local"), default="global"
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--scratch-root", type=Path, default=Path("/mnt/ramstage"))
    parser.add_argument("--warmup", type=int, default=2)
    parser.add_argument("--iterations", type=int, default=5)
    parser.add_argument(
        "--iaa-batch",
        type=int,
        help="profiling-only OneAlign batch override; does not mutate the config",
    )
    parser.add_argument("--skip-onealign", action="store_true")
    parser.add_argument("--onealign-concurrency-probe", action="store_true")
    args = parser.parse_args(argv)
    if args.warmup < 0 or args.iterations < 1:
        parser.error("--warmup must be >= 0 and --iterations must be >= 1")
    if args.iaa_batch is not None and args.iaa_batch < 1:
        parser.error("--iaa-batch must be >= 1")
    if args.skip_onealign and args.onealign_concurrency_probe:
        parser.error("--onealign-concurrency-probe requires OneAlign")
    if not args.groups_jsonl.is_file():
        parser.error("--groups-jsonl must be an existing file")
    if not args.scratch_root.is_dir():
        parser.error("--scratch-root must be an existing directory")
    try:
        config = load_config(args.config.resolve())
        fixture = _fixture(args.groups_jsonl, args.group_id, args.render_mode)
        result = profile(
            config=config,
            fixture=fixture,
            warmup=args.warmup,
            iterations=args.iterations,
            scratch_root=args.scratch_root,
            include_onealign=not args.skip_onealign,
            iaa_batch=args.iaa_batch,
            onealign_concurrency_probe=args.onealign_concurrency_probe,
        )
        _write_json(args.output, result)
    except (OSError, ProfileError, ValueError) as exc:
        parser.exit(1, f"profile failed: {exc}\n")
    print(json.dumps(result, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
