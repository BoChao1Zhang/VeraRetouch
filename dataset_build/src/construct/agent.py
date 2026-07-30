"""Single TOML-driven canonical databuild orchestrator."""
from __future__ import annotations

import argparse
import dataclasses
import gc
import hashlib
import json
import os
import shutil
import sys
import threading
from collections import deque
from concurrent.futures import Future, ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Mapping, Protocol

from dataset_build.tools.archive_reader import (
    default_db,
    invalidate_shared,
    path_exists,
    prefetch_name,
    read_bytes,
    set_prefetch_dir,
)
from dataset_build.tools.global_catalog import upsert as upsert_catalog
from dataset_build.tools.land import land
from dataset_build.tools.prefetch import prefetch

from .canonical_masks import MaskPlanError, build_mask_plan, pair_mask_slots
from .canonical_qa import (
    OneAlignScorer,
    OneAlignScorerPool,
    QaError,
    _stats,
    rank_candidates,
)
from .config import (
    DEFAULT_QA_SCORER_INSTANCES,
    DEFAULT_QA_WINNER_MARGIN_ABSTAIN,
    DEFAULT_QA_WINNER_MARGIN_LOW,
    DEFAULT_SOURCE_WINDOW,
    DatabuildConfig,
    load_config,
    redact_text,
)
from .legacy_import import (
    LegacyGroupInput,
    LegacySnapshot,
    convert_legacy_group,
    legacy_group_id,
    load_legacy_snapshot,
)
from .presets import CoverageSelector, PresetCatalog, PresetError
from .projection import ProjectionResult, project_artifacts
from .rendering import (
    LocalGpuOnlyRenderer,
    preprocess_source,
    save_candidate_jpeg,
    save_cgt_png,
)
from .responses import ResponsesAnnotator, preflight_openai_sdk
from .sources import (
    SourceInventoryResult,
    SourceRecord,
    allocate_sources,
    build_inventory,
    refresh_source_record,
)
from .state import ASSETS_LOST_CODE, ArtifactStore, StateError, file_digest, stable_id
from .visibility import (
    VisibilityError,
    hint_support,
    objective_edit_hints_from_lab,
    prepare_torch_lab_reference,
    prepare_working_after,
    prepare_working_reference,
    srgb_to_lab,
    visibility_and_hints_torch,
    visibility_metrics_from_lab,
)


# Land whenever the staged assets reach this much of the ramstage tmpfs, and once
# more when rendering ends.  ponytail: a module constant, not a config key — the
# mount is 24 GiB and landing is idempotent, so every value in the 8-20 GiB band
# behaves the same and nothing downstream reads it.
#
# 8, not 16, since the production layout runs two builds at once (one card each)
# on the same 24 GiB mount: the counter below is per-process, so two 16 GiB
# watermarks would ask for 32 GiB and fill the tmpfs instead of landing.  Two
# 8 GiB watermarks plus the two prefetch buffers (~1 GiB each) and the journals
# leave the mount with room to spare; the only cost is more frequent landing.
LAND_WATERMARK_BYTES = 8 * 1024**3
# Sources per prefetch buffer.  ponytail: a module constant for the same reason
# as the water mark — one chunk is ~1 GiB of a 24 GiB tmpfs, the buffer is
# rebuildable, and the only requirement is that a chunk take long enough to
# render that the next one finishes reading behind it.
PREFETCH_CHUNK = 256
# NFS roots.  Only ``default_dependencies`` wires them in, so any caller that
# builds ``PipelineDependencies`` by hand (every test) lands nothing, mirrors
# nothing and never touches NFS.
ARCHIVE_ROOT = Path("/mnt/nfs/bc/data/datasets")
MIRROR_ROOT = Path("/mnt/nfs/bc/data/builds")


class PipelineError(RuntimeError):
    """The canonical lifecycle cannot make safe progress."""


@dataclass(frozen=True, slots=True)
class _PostprocessResult:
    metrics: Any
    hints: dict[str, dict[str, float | str]]
    qa_stats: dict[str, float]


@dataclass(frozen=True, slots=True)
class _PostprocessReference:
    working: Any
    torch: Any | None


@dataclass(frozen=True, slots=True)
class _PendingCandidate:
    slot_index: int
    slot: Any
    slot_id: str
    preset_reservation: Any
    attempt_task_id: str
    candidate_id: str
    after_path: Path
    rendered: Any
    future: Future[_PostprocessResult]


@dataclass(frozen=True, slots=True)
class _CandidateAttempt:
    slot_index: int
    slot: Any
    slot_id: str
    preset_reservation: Any
    attempt_task_id: str
    candidate_id: str
    after_path: Path


@dataclass(frozen=True, slots=True)
class _BufferedFailure:
    row: dict[str, Any]
    durable: bool


@dataclass(frozen=True, slots=True)
class _DeferredGroupCommit:
    group: dict[str, Any]
    reservation: Any
    coverage: dict[str, Any]


@dataclass(frozen=True, slots=True)
class _DeferredSourceResult:
    status: str
    failures: tuple[_BufferedFailure, ...]
    group_commit: _DeferredGroupCommit | None = None
    error: Exception | None = None


@dataclass(frozen=True, slots=True)
class _SelectorTurn:
    ready: threading.Event
    following: threading.Event


def _write_cgt_once(mask: Any, path: Path) -> Path:
    """Encode one physical C_GT, or adopt the copy an earlier attempt fsynced.

    The reuse is keyed by ``mask_id``, which names a plan slot rather than the
    pixels behind it, so an existing file is only the right answer while nothing
    can re-plan a mask whose C_GT is already on disk.  Today that holds because
    SAM3 relabel only ever runs after ``build_mask_plan`` raised — that is,
    before ``_start_cgt_writes`` submitted anything for that source.  Any change
    that lets an already rendered source be re-planned has to invalidate (delete)
    the affected C_GT files first, or this returns a stale mask.
    """
    if path.is_file():
        return path
    return save_cgt_png(mask, path)


def _postprocess_candidate(
    reference: Any,
    after: Any,
    weight: Any,
    after_path: Path,
    render_config: Any,
) -> _PostprocessResult:
    """Run the CPU-heavy work for one already rendered candidate."""
    working_after, working_weight = prepare_working_after(
        reference.working, after, weight
    )
    metric_args = {
        "weight": working_weight,
        "visible_de_min": render_config.visible_de_min,
        "visible_fraction_de": render_config.visible_fraction_de,
        "visible_fraction_min": render_config.visible_fraction_min,
    }
    if render_config.visibility_backend == "torch":
        if reference.torch is None:
            raise VisibilityError("torch visibility reference is missing")
        metrics, hints = visibility_and_hints_torch(
            reference.torch, working_after, **metric_args
        )
    else:
        after_lab = srgb_to_lab(working_after)
        metrics = visibility_metrics_from_lab(
            reference.working.lab, after_lab, **metric_args
        )
        hints = objective_edit_hints_from_lab(
            reference.working.lab, after_lab, weight=working_weight,
            support=hint_support(reference.working, working_after),
        )
    if not metrics.accepted:
        raise PipelineError(
            "visibility gate rejected candidate "
            f"(de={metrics.visible_de:.6f}, "
            f"fraction={metrics.visible_fraction:.6f})"
        )
    save_candidate_jpeg(after, after_path, quality=render_config.jpeg_quality)
    return _PostprocessResult(
        metrics=metrics,
        hints=hints,
        qa_stats=_stats(str(after_path)),
    )


class Renderer(Protocol):
    device: str

    def bind_catalog(self, catalog: PresetCatalog) -> None: ...
    def assert_ready(self) -> None: ...
    def render(self, source: Any, preset: Any, mask: Any = None) -> Any: ...
    def render_many(self, source: Any, requests: Any) -> list[Any]: ...


class Scorer(Protocol):
    def score(self, path: str) -> float | None: ...


class QueueDrainer(Protocol):
    def drain(self, *, max_workers: int | None = None) -> dict[str, int]: ...


@dataclass(slots=True)
class PipelineDependencies:
    inventory_loader: Callable[[DatabuildConfig], SourceInventoryResult]
    catalog_loader: Callable[[DatabuildConfig], PresetCatalog]
    renderer_factory: Callable[[DatabuildConfig], Renderer]
    scorer_factory: Callable[[], Scorer]
    sdk_preflight: Callable[[], None]
    annotator_factory: Callable[[DatabuildConfig, ArtifactStore], QueueDrainer]
    relabeler: Callable[[list[SourceRecord], DatabuildConfig, int], Mapping[str, str]]
    projector: Callable[[ArtifactStore, Mapping[str, Any], DatabuildConfig], ProjectionResult]
    now: Callable[[], datetime]
    # Landing, mirroring and catalog refresh are off unless a root is supplied.
    archive_root: Path | None = None
    mirror_root: Path | None = None
    catalog_db: Path | None = None
    scorer_pool_factory: Callable[[int], Scorer] | None = None


def _default_relabeler(
    sources: list[SourceRecord], config: DatabuildConfig, attempt: int
) -> Mapping[str, str]:
    from dataset_build.source_qa.sam3_subject_instances import relabel_sources

    rows = [
        {"asset_id": source.source_id, "source_path": str(source.source_path)}
        for source in sources
    ]
    return relabel_sources(
        rows,
        cache_root=str(config.sources.subject_cache),
        device="cuda:0",
        seed=config.seed + attempt,
        vlm_base_url=config.annotation.local.base_url,
        vlm_api_key=config.annotation.local.api_key,
        vlm_model=config.annotation.local.model,
    )


def default_dependencies() -> PipelineDependencies:
    return PipelineDependencies(
        inventory_loader=lambda config: build_inventory(
            config.sources.subject_cache, config.sources.postgres_dsn
        ),
        catalog_loader=PresetCatalog.load,
        renderer_factory=LocalGpuOnlyRenderer.create,
        scorer_factory=lambda: OneAlignScorer.create("cuda:0"),
        sdk_preflight=preflight_openai_sdk,
        annotator_factory=lambda config, store: ResponsesAnnotator(config.annotation, store),
        relabeler=_default_relabeler,
        projector=lambda store, manifest, config: project_artifacts(
            store, manifest, config.viewer.postgres_dsn
        ),
        now=lambda: datetime.now(timezone.utc),
        archive_root=ARCHIVE_ROOT,
        mirror_root=MIRROR_ROOT,
        scorer_pool_factory=lambda instances: OneAlignScorerPool.create(
            "cuda:0", instances=instances
        ),
    )


def _atomic_copy(source: Path, target: Path) -> None:
    """Replace one small file in place: same-directory temp, fsync, rename."""
    target.parent.mkdir(parents=True, exist_ok=True)
    tmp = target.with_name(target.name + ".tmp")
    shutil.copyfile(source, tmp)
    with tmp.open("rb") as handle:
        os.fsync(handle.fileno())
    os.replace(tmp, target)


def _write_jsonl_atomic(path: Path, rows: list[Mapping[str, Any]]) -> None:
    """Replace a JSONL sidecar in place; a reader sees the old file or the new one."""
    tmp = path.with_name(path.name + ".tmp")
    with tmp.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(tmp, path)


def _artifact_names(root: Path) -> list[str]:
    names = [path.name for path in sorted(root.glob("*.jsonl"))]
    if (root / "manifest.json").is_file():
        names.append("manifest.json")
    return names


def mirror_artifacts(output_root: Path, mirror_dir: Path) -> list[str]:
    """Copy the authoritative ledgers off the tmpfs so a reboot cannot take them."""
    names = _artifact_names(Path(output_root))
    for name in names:
        _atomic_copy(Path(output_root) / name, Path(mirror_dir) / name)
    return names


def restore_mirror(mirror_dir: Path, output_root: Path) -> list[str]:
    """Rebuild a wiped output root from its mirror so the normal resume applies.

    Only a mirror carrying a manifest counts: without one there is nothing to
    resume from, and creating the output root here would defeat the preflight
    contract that a failed startup leaves no artifacts behind.
    """
    mirror_dir = Path(mirror_dir)
    if not (mirror_dir / "manifest.json").is_file():
        return []
    names = _artifact_names(mirror_dir)
    for name in names:
        _atomic_copy(mirror_dir / name, Path(output_root) / name)
    return names


def _timestamp(value: datetime) -> str:
    if value.tzinfo is None:
        value = value.replace(tzinfo=timezone.utc)
    return value.astimezone(timezone.utc).isoformat().replace("+00:00", "Z")


def _add_manifest_self_artifact(manifest: dict[str, Any]) -> None:
    """Record a verifiable scoped digest without claiming a self-referential hash."""
    artifacts = manifest.setdefault("artifacts", {})
    scoped_manifest = dict(manifest)
    scoped_artifacts = dict(artifacts)
    scoped_artifacts.pop("manifest.json", None)
    scoped_manifest["artifacts"] = scoped_artifacts
    canonical = json.dumps(
        scoped_manifest, ensure_ascii=False, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")
    entry = {
        "sha256": hashlib.sha256(canonical).hexdigest(),
        "hash_scope": "manifest excluding artifacts.manifest.json",
        "records": 1,
        "bytes": 0,
    }
    artifacts["manifest.json"] = entry
    for _ in range(8):
        size = len((json.dumps(
            manifest, ensure_ascii=False, sort_keys=True, indent=2
        ) + "\n").encode("utf-8"))
        if entry["bytes"] == size:
            break
        entry["bytes"] = size


def _empty_cuda_cache() -> None:
    """Hand the allocator's spare blocks back so a co-resident model can have them."""
    try:
        import torch

        torch.cuda.empty_cache()
    except Exception:  # noqa: BLE001 - a CPU-only run has nothing to give back
        pass


def _summary_count(text: str, key: str) -> int:
    """Read one integer back out of a journalled land-checkpoint summary.

    The summaries are the durable record of what each checkpoint published, so
    resuming a build inherits its predecessors' counts.  A summary written by an
    older revision simply has no such key and contributes nothing.
    """
    try:
        value = json.loads(text).get(key)
    except (json.JSONDecodeError, AttributeError):
        return 0
    return value if isinstance(value, int) and not isinstance(value, bool) else 0


def _load_existing_manifest(path: Path) -> dict[str, Any] | None:
    if not path.is_file():
        return None
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise StateError(f"cannot resume invalid manifest: {path}") from exc
    if not isinstance(value, dict):
        raise StateError("manifest must be a JSON object")
    return value


# Keys introduced into the config after builds were already durable.  A manifest
# written before the key existed cannot carry it, so an exact comparison would
# reject every such resume.  Only a key whose value provably leaves the journals
# alone may be listed here — ``render.source_window`` is scheduling width, and
# the ``source_window = 1`` oracle parity test shows it changes no journal line.
# The list is deliberately closed: any other missing key is still a difference.
_RESUME_NEUTRAL_DEFAULTS: dict[tuple[str, str], Any] = {
    ("render", "source_window"): DEFAULT_SOURCE_WINDOW,
    # ``render.qa_scorer_instances`` only says how many identical OneAlign copies
    # share the QA device.  Each group's images are ranked by one copy in one
    # batched forward, so the copy that serves a group cannot change its scores,
    # its ranks, or any journal line the ranking produces.
    ("render", "qa_scorer_instances"): DEFAULT_QA_SCORER_INSTANCES,
    # The winner-margin policy applies at the moment a winner is *chosen*, and a
    # chosen winner is already in ``groups.jsonl``.  A build that finished its
    # rendering phase — eval100 is the case in hand — therefore resumes into
    # annotation and landing with exactly the winner set it journalled, whatever
    # these two keys now say: they are neutral for it.  The limit of that claim
    # is a build resumed *mid-rendering*, whose remaining groups would be ranked
    # under the new policy while its earlier groups were not.  That is accepted
    # rather than prevented (no such build is active), and it is why these two
    # entries only complete a manifest that never had the keys, instead of
    # excusing a manifest whose values differ.
    ("render", "qa_winner_margin_abstain"): DEFAULT_QA_WINNER_MARGIN_ABSTAIN,
    ("render", "qa_winner_margin_low"): DEFAULT_QA_WINNER_MARGIN_LOW,
}


def _config_diff_paths(old: Any, new: Any, prefix: str = "") -> list[str]:
    """Dotted key paths where two effective configs disagree."""
    if isinstance(old, dict) and isinstance(new, dict):
        paths: list[str] = []
        for key in sorted(set(old) | set(new), key=str):
            path = f"{prefix}.{key}" if prefix else str(key)
            if key not in old or key not in new:
                paths.append(path)
            else:
                paths.extend(_config_diff_paths(old[key], new[key], path))
        return paths
    return [] if old == new else [prefix or "<config>"]


def _resume_config_differences(old_config: Any, new_config: Any) -> list[str]:
    """Compare a durable effective config with the one being resumed.

    Every key path is compared exactly, except that a neutral key the old
    manifest is simply missing is first completed with its current default, so a
    build created before that key was added stays resumable.
    """
    if isinstance(old_config, dict):
        completed = dict(old_config)
        for (table, key), default in _RESUME_NEUTRAL_DEFAULTS.items():
            section = completed.get(table)
            if isinstance(section, dict) and key not in section:
                completed[table] = {**section, key: default}
        old_config = completed
    return _config_diff_paths(old_config, new_config)


def _preflight_scorer(scorer: Scorer, inventory: SourceInventoryResult) -> None:
    """Exercise one real OneAlign forward before any authoritative artifact exists."""
    if not inventory.eligible:
        raise PipelineError("source inventory contains no eligible image for OneAlign preflight")
    sample_path = str(inventory.eligible[0].source_path)
    try:
        score = scorer.score(sample_path)
    except Exception as exc:  # noqa: BLE001 - startup boundary needs one classified error
        raise QaError(
            f"OneAlign live preflight failed: {type(exc).__name__}: {exc}"
        ) from exc
    if score is None or not isinstance(score, (int, float)) \
            or not 0.0 <= float(score) <= 100.0:
        raise QaError("OneAlign live preflight returned an invalid score")


class _DeferredScorer:
    """A scorer whose model arrives with the first ranking call.

    ``[render] qa_preflight_forward = false`` waives the startup forward, and
    with it the weight load that only exists to serve that forward: QA is the
    first place the model is genuinely needed, so a small iteration build stops
    paying half a minute before it renders anything.  The cost of the waiver is
    that a broken scorer now surfaces during rendering rather than before the
    output root exists, which is why the key defaults to true.
    """

    def __init__(self, factory: Callable[[], Scorer]) -> None:
        self._factory = factory
        self._scorer: Scorer | None = None
        self._lock = threading.Lock()

    def _resolve(self) -> Scorer:
        if self._scorer is None:
            with self._lock:
                if self._scorer is None:
                    self._scorer = self._factory()
        return self._scorer

    def score(self, path: str) -> float | None:
        return self._resolve().score(path)

    def __getattr__(self, name: str) -> Any:
        # Everything else the QA layer reaches for, including the optional
        # ``score_batch`` it probes with ``hasattr``.  Private names are refused
        # unresolved so copy/pickle protocol probes cannot load a model.
        if name.startswith("_"):
            raise AttributeError(name)
        return getattr(self._resolve(), name)


def _load_scorer(
    config: DatabuildConfig,
    dependencies: PipelineDependencies,
    inventory: SourceInventoryResult,
) -> Scorer:
    """Build the QA scorer, exercising one real forward unless the config waives it."""
    factory = dependencies.scorer_factory
    if dependencies.scorer_pool_factory is not None:
        # Copies are how QA throughput scales, so the count is its own key rather
        # than a function of ``gpu_concurrency``: that one sizes the render
        # semaphore on the other card and the two no longer move together.
        # ``qa_scorer_instances = 1`` is the explicit single-copy rollback.
        instances = config.render.qa_scorer_instances
        factory = lambda: dependencies.scorer_pool_factory(instances)
    if not config.render.qa_preflight_forward:
        return _DeferredScorer(factory)
    scorer = factory()
    _preflight_scorer(scorer, inventory)
    return scorer


class _SourcePrefetch:
    """Keep one chunk of source images ahead of the renderer, on the tmpfs.

    ``tools.prefetch`` reads a chunk in ``(shard, offset)`` order, which turns a
    round's scattered 51 ms preads into one sequential pass per shard; running it
    on a single background thread means chunk k+1 is read while chunk k renders.
    It is a cache and never an authority: every failure here degrades to exactly
    the archive read the buffer existed to avoid, so a prefetch problem can cost
    throughput but can never fail a render.
    """

    def __init__(self, directory: Path, db_path: Path | None) -> None:
        self.directory = Path(directory)
        self.directory.mkdir(parents=True, exist_ok=True)
        self._db_path = db_path
        self._executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="prefetch")
        self._pending: Future | None = None
        self.buffered = 0
        self.errors = 0

    def activate(self) -> None:
        """Point every ``read_bytes`` at this buffer for the rest of the run."""
        set_prefetch_dir(self.directory)

    def submit(self, source_paths: list[str]) -> None:
        if self._pending is not None:
            self.take()
        if source_paths:
            self._pending = self._executor.submit(
                prefetch, source_paths, self.directory, db_path=self._db_path
            )

    def take(self) -> list[Path]:
        """Wait for the outstanding chunk and report the copies it buffered."""
        pending, self._pending = self._pending, None
        if pending is None:
            return []
        try:
            fetched = list(pending.result().values())
        except Exception:  # noqa: BLE001 - a cold buffer is slower, never wrong
            self.errors += 1
            return []
        self.buffered += len(fetched)
        return fetched

    def path_for(self, source_path: object) -> Path:
        return self.directory / prefetch_name(str(source_path))

    def close(self) -> None:
        self._pending = None
        self._executor.shutdown(wait=True, cancel_futures=True)
        set_prefetch_dir(None)
        # The buffer is rebuildable by definition and a finished build must not
        # sit on the tmpfs it borrowed; a resume re-reads at sequential speed.
        shutil.rmtree(self.directory, ignore_errors=True)


class CanonicalPipeline:
    def __init__(
        self,
        config: DatabuildConfig,
        dependencies: PipelineDependencies,
        inventory: SourceInventoryResult,
        catalog: PresetCatalog,
        renderer: Renderer,
        scorer: Scorer,
        store: ArtifactStore,
        existing_manifest: Mapping[str, Any] | None,
    ) -> None:
        self.config = config
        self.dependencies = dependencies
        self.inventory = inventory
        self.catalog = catalog
        self.renderer: Renderer | None = renderer
        self.scorer: Scorer | None = scorer
        self.store = store
        self._source_context = threading.local()
        self._staged_lock = threading.Lock()
        # Both scorer copies can be inside a ranking call at once, so the window
        # has to fund a turn holder plus the sources decoding ahead of it.  It is
        # its own key: ``gpu_concurrency`` also sizes the render semaphore, and
        # the two no longer move together.  Injected dependencies (every test that
        # does not opt in) keep the pre-pool window.
        self._source_window = (
            config.render.source_window
            if dependencies.scorer_pool_factory is not None
            else min(2, config.render.gpu_concurrency)
        )
        self._source_executor = ThreadPoolExecutor(
            max_workers=self._source_window,
            thread_name_prefix="databuild-source",
        )
        self._render_executor = ThreadPoolExecutor(
            max_workers=config.render.gpu_concurrency,
            thread_name_prefix="databuild-render",
        )
        self._postprocess_executor = ThreadPoolExecutor(
            max_workers=config.render.postprocess_workers,
            thread_name_prefix="databuild-postprocess",
        )
        self.started_at = str(
            (existing_manifest or {}).get("started_at")
            or _timestamp(self.dependencies.now())
        )
        self._phase = "preflight"
        self._annotation_sync: dict[str, int] = {}
        # No archive root means nothing was ever landed there and nothing can be
        # prefetched from it, which is also what every hand-built dependency set
        # (that is, every test) gets.
        self._prefetch = (
            _SourcePrefetch(store.root / "prefetch", dependencies.catalog_db)
            if dependencies.archive_root is not None else None
        )
        self._recalibrate_staged()
        self.allocation = allocate_sources(
            inventory.eligible,
            build_id=config.build_id,
            seed=config.seed,
            target_groups=config.target_groups,
            mix=config.mix,
        )
        historical = tuple(store.groups.values())
        self.selectors = {
            "local": CoverageSelector(
                catalog,
                build_id=config.build_id,
                seed=config.seed,
                render_mode="local",
                preset_filter=config.preset_filter,
                historical_groups=historical,
            ) if self.allocation.local_target else None,
            "global": CoverageSelector(
                catalog,
                build_id=config.build_id,
                seed=config.seed,
                render_mode="global",
                preset_filter=config.preset_filter,
                historical_groups=historical,
            ) if self.allocation.global_target else None,
        }

    def _failure(
        self,
        *,
        event_type: str,
        stage: str,
        task_id: str,
        error_code: str,
        message: object,
        retryable: bool,
        terminal: bool,
        source: SourceRecord | None = None,
        group_id: str | None = None,
        candidate_id: str | None = None,
        round_number: int | None = None,
        attempt: int | None = None,
        endpoint_id: str | None = None,
        durable: bool = False,
    ) -> bool:
        source_id = source.source_id if source is not None else None
        event_id = stable_id(
            "failure", self.config.build_id, event_type, stage, task_id, error_code,
            round_number, attempt, group_id, candidate_id, endpoint_id,
        )
        row = {
            "build_id": self.config.build_id,
            "event_id": event_id,
            "event_type": event_type,
            "stage": stage,
            "task_id": task_id,
            "source_id": source_id,
            "source_path": str(source.source_path) if source is not None else None,
            "group_id": group_id,
            "candidate_id": candidate_id,
            "round": round_number,
            "attempt": attempt,
            "retryable": retryable,
            "error_code": error_code,
            "message": redact_text(message, self.config.secrets),
            "endpoint_id": endpoint_id,
            "terminal": terminal,
            "timestamp": _timestamp(self.dependencies.now()),
        }
        buffered = getattr(self._source_context, "failures", None)
        if buffered is not None:
            buffered.append(_BufferedFailure(row=row, durable=durable))
            return True
        return self.store.append_failure(row, durable=durable)

    def _live_groups(self) -> list[dict[str, Any]]:
        """Durable groups whose assets still exist; the rest are accounted, not used."""
        lost = self.store.lost_group_ids()
        return [
            row for row in self.store.groups.values()
            if str(row.get("group_id")) not in lost
        ]

    def _mode_groups(self, mode: str) -> list[dict[str, Any]]:
        return [row for row in self._live_groups() if row.get("render_mode") == mode]

    def _terminal_source_ids(self) -> set[str]:
        return {
            str(row["source_id"])
            for row in self.store.failures
            if row.get("terminal") and row.get("source_id")
            and row.get("stage") in {"rendering", "sam3_relabel"}
        }

    def _pending_sam3_ids(self) -> set[str]:
        queued = {
            str(row["source_id"])
            for row in self.store.failures
            if row.get("error_code") == "sam3_relabel_queued" and row.get("source_id")
        }
        return queued.difference(self.store.completed_sources()).difference(
            self._terminal_source_ids()
        )

    def _sam3_attempts(self, source_id: str) -> int:
        return sum(
            1 for row in self.store.failures
            if row.get("source_id") == source_id
            and row.get("stage") == "sam3_relabel"
            and row.get("event_type") == "sam3_attempt"
        )

    def _unconsumed_sam3_ready(self, source_id: str) -> bool:
        ready = [
            int(row.get("attempt") or 0) for row in self.store.failures
            if row.get("source_id") == source_id and row.get("event_type") == "sam3_ready"
        ]
        invalid = [
            int(row.get("attempt") or 0) for row in self.store.failures
            if row.get("source_id") == source_id
            and row.get("event_type") == "sam3_ready_invalid"
        ]
        return bool(ready) and max(ready) > max(invalid or [0])

    def _manifest(self, phase: str, status: str = "running") -> dict[str, Any]:
        local_groups = self._mode_groups("local")
        global_groups = self._mode_groups("global")
        lost_groups = self.store.lost_group_ids()
        live_groups = local_groups + global_groups
        candidates = [
            candidate
            for group in live_groups
            for candidate in group.get("candidates") or []
        ]
        terminal_failures = [row for row in self.store.failures if row.get("terminal")]
        annotation_failures = [
            row for row in terminal_failures if row.get("stage") == "annotation"
        ]
        preset_formats: dict[str, int] = {}
        preset_majors: dict[str, int] = {}
        preset_minors: dict[str, int] = {}
        for candidate in candidates:
            for target, key in (
                (preset_formats, "format"), (preset_majors, "major"),
                (preset_minors, "minor"),
            ):
                value = str(candidate.get(key) or "unknown")
                target[value] = target.get(value, 0) + 1
        completed = len(local_groups) + len(global_groups)
        local_initial = {row.source_id for row in self.allocation.local[:self.allocation.local_target]}
        global_initial = {
            row.source_id for row in self.allocation.global_[:self.allocation.global_target]
        }
        replacement_used = {
            "local": sum(group.get("source_id") not in local_initial for group in local_groups),
            "global": sum(group.get("source_id") not in global_initial for group in global_groups),
        }
        sam3_expected_ids = {
            str(row["source_id"])
            for row in self.store.failures
            if row.get("error_code") == "sam3_relabel_queued" and row.get("source_id")
        }
        sam3_completed_ids = {
            source_id for source_id in sam3_expected_ids
            if self._unconsumed_sam3_ready(source_id)
        }
        candidate_failures: dict[str, int] = {}
        for row in self.store.failures:
            if row.get("stage") == "rendering" and row.get("event_type") == "attempt":
                code = str(row.get("error_code") or "unknown")
                candidate_failures[code] = candidate_failures.get(code, 0) + 1
        ratio = {
            "local": len(local_groups) / completed if completed else 0.0,
            "global": len(global_groups) / completed if completed else 0.0,
        }
        return {
            "schema_version": self.config.schema_version,
            "build_id": self.config.build_id,
            "phase": phase,
            "status": status,
            "started_at": self.started_at,
            "updated_at": _timestamp(self.dependencies.now()),
            "effective_config": self.config.sanitized_dict(),
            "targets": {
                "groups": self.config.target_groups,
                "local": self.allocation.local_target,
                "global": self.allocation.global_target,
            },
            "completed": {
                "groups": completed,
                "local": len(local_groups),
                "global": len(global_groups),
                "actual_ratio": ratio,
                "candidates": len(candidates),
                "winner_top1": sum(bool(row.get("winner_ids")) for row in live_groups),
                "winner_top2": sum(len(row.get("winner_ids") or []) >= 2
                                   for row in live_groups),
                # Groups whose winner the margin policy refused.  Distinct from
                # the always-existing "no candidate cleared SFT_THRESHOLD" groups,
                # which carry no verdict at all.
                "winner_abstained": sum(row.get("winner_confidence") == "abstain"
                                        for row in live_groups),
                "sft": sum(
                    str(row.get("group_id")) not in lost_groups
                    for row in self.store.sft.values()
                ),
                "groups_assets_lost": len(lost_groups),
            },
            "landing": self._landing_counts(),
            "prefetch": self._prefetch_counts(),
            "sources": {
                "inventory": self.inventory.counts,
                "scene_metadata_status": self.inventory.scene_metadata_status,
                "eligible": len(self.inventory.eligible),
                "replacement_capacity": {
                    "local": max(0, len(self.allocation.local) - self.allocation.local_target),
                    "global": max(0, len(self.allocation.global_) - self.allocation.global_target),
                },
                "replacement_used": replacement_used,
                "exhaustion": {
                    "local_shortfall": max(0, self.allocation.local_target - len(local_groups)),
                    "global_shortfall": max(0, self.allocation.global_target - len(global_groups)),
                },
                "terminal": len(self._terminal_source_ids()),
            },
            "presets": {
                "inventory": {
                    "local": self.catalog.inventory_counts("local"),
                    "global": self.catalog.inventory_counts("global"),
                },
                "usage": {
                    "format": dict(sorted(preset_formats.items())),
                    "major": dict(sorted(preset_majors.items())),
                    "minor": dict(sorted(preset_minors.items())),
                    "selector": {
                        mode: selector.snapshot() if selector is not None else None
                        for mode, selector in self.selectors.items()
                    },
                    "candidate_failure_reasons": dict(sorted(candidate_failures.items())),
                },
            },
            "sam3_relabel": {
                "expected": len(sam3_expected_ids),
                "completed": len(sam3_completed_ids),
                "pending": len(self._pending_sam3_ids()),
                "attempt_events": sum(
                    row.get("event_type") == "sam3_attempt" for row in self.store.failures
                ),
                "terminal": sum(
                    row.get("stage") == "sam3_relabel" and row.get("terminal")
                    for row in self.store.failures
                ),
            },
            "annotation": {
                "pending": len(self.store.pending_annotation_tasks()),
                "terminal_failures": len(annotation_failures),
                "backends": self._annotation_counts(),
            },
            "failures": {
                "events": len(self.store.failures),
                "terminal": len(terminal_failures),
                "by_code": self._failure_counts(),
            },
        }

    def _landing_counts(self) -> dict[str, Any]:
        landed = [row for row in self.store.failures if row.get("stage") == "landing"]
        summaries = [str(row.get("message")) for row in landed]
        return {
            "checkpoints": len(landed),
            # One JSON summary per checkpoint, as it was journalled.
            "checkpoint_summaries": summaries,
            # Winners whose input image reached the SFT dataset as a member.  It
            # is the durable count rather than the winner count because a source
            # the archive cannot serve lands without I_in instead of failing the
            # checkpoint; a healthy build has the two agreeing.
            "i_in_members": sum(_summary_count(text, "i_in") for text in summaries),
            "sft_winners": sum(_summary_count(text, "winners") for text in summaries),
            "annotation_status": dict(self._annotation_sync),
        }

    def _prefetch_counts(self) -> dict[str, Any]:
        if self._prefetch is None:
            return {"enabled": False, "buffered": 0, "errors": 0}
        return {
            "enabled": True,
            "buffered": self._prefetch.buffered,
            # Non-zero means the round fell back to random archive reads, which
            # costs throughput and nothing else — worth seeing, never fatal.
            "errors": self._prefetch.errors,
        }

    def _failure_counts(self) -> dict[str, int]:
        result: dict[str, int] = {}
        for row in self.store.failures:
            code = str(row.get("error_code") or "unknown")
            result[code] = result.get(code, 0) + 1
        return dict(sorted(result.items()))

    def _annotation_counts(self) -> dict[str, Any]:
        sources: dict[str, int] = {}
        models: dict[str, int] = {}
        usage: dict[str, int] = {}
        for row in self.store.sft.values():
            source = str(row.get("annot_src") or "unknown")
            sources[source] = sources.get(source, 0) + 1
            meta = (row.get("qa") or {}).get("annotation") or {}
            model = str(meta.get("returned_model") or "unknown")
            models[model] = models.get(model, 0) + 1
            for key, value in (meta.get("usage") or {}).items():
                if isinstance(value, int):
                    usage[key] = usage.get(key, 0) + value
        return {
            "sources": dict(sorted(sources.items())),
            "models": dict(sorted(models.items())),
            "usage": dict(sorted(usage.items())),
        }

    def _write_phase(self, phase: str) -> None:
        self._phase = phase
        self.store.write_manifest(self._manifest(phase))
        self.store.checkpoint()

    @staticmethod
    def _group_assets(group: Mapping[str, Any]) -> list[str]:
        """Every physical file a group owns, in a stable order, without duplicates."""
        paths: dict[str, None] = {}
        for candidate in group.get("candidates") or []:
            for key in ("after_path", "cgt_path"):
                value = candidate.get(key)
                if value:
                    paths[str(value)] = None
        return list(paths)

    def _asset_directories(self) -> tuple[Path, ...]:
        directories = [
            self.store.assets_root / "candidates",
            self.store.assets_root / "masks",
        ]
        if self._prefetch is not None:
            # The buffer competes for the same tmpfs, so the water mark has to
            # count it; it is not an asset, so the orphan sweep must not reap it.
            directories.append(self._prefetch.directory)
        return tuple(directories)

    def _recalibrate_staged(self) -> None:
        """Re-measure the staging tree; the water mark's only full scan.

        Scanning per source cost minutes over a build (tens of thousands of files
        × 22k sources), so the counter is maintained incrementally instead and
        this runs only where a scan is already being paid for: once at startup,
        so a resume inherits the assets a previous run left behind, and inside
        the orphan sweep, which walks the same directories anyway.
        """
        sizes: dict[str, int] = {}
        for directory in self._asset_directories():
            if not directory.is_dir():
                continue
            with os.scandir(directory) as scan:
                for entry in scan:
                    if entry.is_file():
                        sizes[entry.path] = entry.stat().st_size
        with self._staged_lock:
            self._staged = sizes
            self._staged_bytes = sum(sizes.values())

    def _account_asset(self, path: Path) -> None:
        """Fold one freshly written asset in, replacing any size it overwrote.

        A refilled slot rewrites the same candidate JPEG and every group attempt
        rewrites the same C_GT, so a plain addition would drift upward forever.
        """
        size = path.stat().st_size
        with self._staged_lock:
            self._staged_bytes += size - self._staged.get(str(path), 0)
            self._staged[str(path)] = size

    def _staged_size(self) -> int:
        with self._staged_lock:
            return self._staged_bytes

    def _landed_datasets(self, root: Path) -> list[str]:
        """Every batch this build has published, read off the archive itself.

        ``_land_groups`` publishes into ``groups/<build_id>/<batch>`` and
        ``sft/<build_id>/<batch>``, so the layout is the record — and a more
        complete one than the journalled checkpoint summaries, which are written
        after the publish and therefore miss a batch that landed just before the
        crash.  Re-registering a batch that is already indexed is idempotent, so
        the list needs no bookkeeping.
        """
        names: list[str] = []
        for kind in ("groups", "sft"):
            base = root / kind / self.config.build_id
            if not base.is_dir():
                continue
            for manifest in sorted(base.glob("*/manifest.json")):
                names.append(manifest.parent.relative_to(root).as_posix())
        return names

    def _refresh_catalog(self) -> None:
        """Re-index this build's landed batches so their assets answer to staging paths."""
        root = self.dependencies.archive_root
        if root is None or not Path(root).is_dir():
            return
        # A checkpoint re-registers this build's own batches — a handful of
        # groups in an archive of hundreds; the full rebuild re-read all 5.5 M
        # members for them (~390 s of silence before annotation).  Anything else
        # in the archive was indexed by whoever published it.
        upsert_catalog(
            Path(root),
            self.dependencies.catalog_db or default_db(),
            self._landed_datasets(Path(root)),
        )
        # immutable=1 pins the pre-refresh snapshot; cached readers must reopen
        # or every landed path stays invisible to this process (eval100 全灭根因).
        invalidate_shared()

    def _verify_group_assets(self) -> None:
        """Account for groups whose assets died with the tmpfs before landing."""
        lost = self.store.lost_group_ids()
        suspects: list[tuple[dict[str, Any], list[str]]] = []
        for group in self.store.groups.values():
            if str(group.get("group_id")) in lost:
                continue
            missing = [
                path for path in self._group_assets(group) if not Path(path).is_file()
            ]
            if missing:
                suspects.append((group, missing))
        if not suspects:
            return
        # A landed asset only resolves through the archive, so the reverse map has
        # to be current before absence is called loss.  Without an archive root
        # nothing was ever landed and absence needs no second opinion.
        archived = self.dependencies.archive_root is not None
        if archived:
            self._refresh_catalog()
        for group, missing in suspects:
            if archived and all(
                path_exists(path, db_path=self.dependencies.catalog_db)
                for path in missing
            ):
                continue
            self._failure(
                event_type="terminal",
                stage="rendering",
                task_id=stable_id("group-assets", str(group["group_id"])),
                error_code=ASSETS_LOST_CODE,
                message="rendered assets are neither staged nor archived",
                retryable=False,
                terminal=True,
                group_id=str(group["group_id"]),
                durable=True,
            )

    def _stage_asset(
        self,
        directory: Path,
        candidate: Mapping[str, Any],
        sequence: int,
        aliases: dict[str, str],
    ) -> str:
        """Hardlink one candidate's files as one sample, keyed in production order.

        The packer emits members in member-name order, so the ordinal prefix is
        what makes the archived order the order the groups were built in — one
        group's eight candidates land contiguous and slot-ordered, which is how
        both the viewer and training read them back.
        """
        directory.mkdir(parents=True, exist_ok=True)
        key = f"{sequence:06d}_{candidate['candidate_id']}"
        for field, extension in (("after_path", ".jpg"), ("cgt_path", ".cgt.png")):
            original = candidate.get(field)
            if not original:
                continue
            link = directory / f"{key}{extension}"
            if not link.exists():
                # Two slots may share one C_GT; the second link is the same inode.
                os.link(str(original), link)
            aliases[str(link)] = str(original)
        return key

    def _stage_i_in(
        self, directory: Path, key: str, group: Mapping[str, Any], aliases: dict[str, str]
    ) -> int:
        """Materialise the winner's input image as one more member of its sample.

        ``read_bytes`` is local-first and then prefetch-first, so at this point in
        a round the bytes are normally already on the tmpfs and the SFT dataset
        gains its ``I_in`` member without a second archive read.  It is aliased to
        the original source path, exactly as ``tools/sft_pack.py`` does, so both
        the rebuild tool and this checkpoint teach the catalog the same key.

        A source that cannot be read is not worth failing a whole checkpoint for:
        that sample lands without ``I_in`` (sft.jsonl still carries the path) and
        the manifest's ``i_in_members`` count reports the shortfall.
        """
        source_path = str(group.get("source_path") or "")
        if not source_path:
            return 0
        try:
            payload = read_bytes(source_path, db_path=self.dependencies.catalog_db)
        except Exception:  # noqa: BLE001 - the SFT view degrades, the checkpoint does not
            return 0
        target = directory / f"{key}.in{Path(source_path).suffix.lower() or '.bin'}"
        target.write_bytes(payload)
        aliases[str(target)] = source_path
        return 1

    def _land_metadata(
        self, group: Mapping[str, Any], candidate: Mapping[str, Any], winner_rank: int | None
    ) -> dict[str, Any]:
        return {
            "build_id": self.config.build_id,
            "group_id": group.get("group_id"),
            "source_id": group.get("source_id"),
            # Not "source_path": that key names the archived member's own origin.
            "i_in_path": group.get("source_path"),
            "scene": group.get("scene"),
            "render_mode": group.get("render_mode"),
            "candidate_id": candidate.get("candidate_id"),
            "preset_id": candidate.get("preset_id"),
            "format": candidate.get("format"),
            "major": candidate.get("major"),
            "minor": candidate.get("minor"),
            "slot_id": candidate.get("slot_id"),
            "mask_id": candidate.get("mask_id"),
            "region": candidate.get("region"),
            "qa": candidate.get("qa"),
            "rank": candidate.get("rank"),
            "winner_rank": winner_rank,
            # Mirrors ``tools/sft_pack.py``'s sample metadata so a landed sample
            # and a repacked one describe the winner the same way.  ``.get``
            # keeps groups journalled before the policy existed landable.
            "winner_confidence": group.get("winner_confidence"),
        }

    def _land_groups(self) -> dict[str, Any] | None:
        """Publish every fully staged group into both datasets, then drop staging."""
        archive_root = Path(self.dependencies.archive_root)  # type: ignore[arg-type]
        # Landing is the only writer here and a half-built batch is always garbage,
        # so the staging tree is dropped first rather than reconciled — leftover
        # hardlinks would otherwise pin the bytes of already landed assets.
        staging_root = self.store.root / ".land"
        shutil.rmtree(staging_root, ignore_errors=True)
        pending = [
            group for group in self._live_groups()
            if all(Path(path).is_file() for path in self._group_assets(group))
        ]
        if not pending:
            return None
        groups_dir, sft_dir = staging_root / "groups", staging_root / "sft"
        aliases: dict[str, str] = {}
        by_key: dict[str, dict[str, Any]] = {}
        staged = winners_staged = i_in_members = 0
        for group in pending:
            winners = list(group.get("winner_ids") or [])
            for candidate in group["candidates"]:
                candidate_id = str(candidate["candidate_id"])
                rank = (
                    winners.index(candidate_id) + 1 if candidate_id in winners else None
                )
                meta = self._land_metadata(group, candidate, rank)
                by_key[self._stage_asset(groups_dir, candidate, staged, aliases)] = meta
                staged += 1
                if rank is not None:
                    winner_key = self._stage_asset(
                        sft_dir, candidate, winners_staged, aliases
                    )
                    by_key[winner_key] = meta
                    i_in_members += self._stage_i_in(sft_dir, winner_key, group, aliases)
                    winners_staged += 1

        def enrich(_key: str, members: Mapping[str, Path]) -> Mapping[str, object] | None:
            first = next(iter(members.values()))
            return by_key.get(Path(first).name.partition(".")[0])

        targets = [(groups_dir, f"groups/{self.config.build_id}")]
        if sft_dir.is_dir():
            # Only staged when this batch actually holds a winner; a batch without
            # one publishes the groups dataset alone rather than an empty tar.
            targets.append((sft_dir, f"sft/{self.config.build_id}"))
        published = []
        for directory, group_name in targets:
            published.append(land(
                directory,
                group_name,
                archive_root,
                plan_root=staging_root / "plans",
                meta_staging=staging_root / "meta",
                source_paths=aliases,
                enrich=enrich,
                keep_staging=True,
            ))
        # Both datasets verified, so the staging bytes are now redundant.
        for group in pending:
            for path in self._group_assets(group):
                Path(path).unlink(missing_ok=True)
        shutil.rmtree(staging_root, ignore_errors=True)
        result = {
            "groups": len(pending),
            "datasets": [str(item["group"]) for item in published],
            "members": sum(int(item["members"]) for item in published),
            "i_in": i_in_members,
            "winners": winners_staged,
        }
        self._failure(
            event_type="landed",
            stage="landing",
            task_id=stable_id("land", str(published[0]["group"])),
            error_code="land_checkpoint",
            message=json.dumps(result, sort_keys=True),
            retryable=False,
            terminal=False,
            durable=True,
        )
        return result

    def _winner_annotation_status(self) -> dict[str, dict[str, Any]]:
        """Each landed winner's final annotation outcome, keyed by candidate."""
        status: dict[str, dict[str, Any]] = {}
        for row in self.store.failures:
            candidate_id = row.get("candidate_id")
            if row.get("stage") == "annotation" and row.get("terminal") and candidate_id:
                status[str(candidate_id)] = {
                    "annotated": False,
                    "sft_id": None,
                    "annotation_failure_code": str(row.get("error_code") or "annotation_failed"),
                }
        # An SFT row is the definitive outcome: a task that produced one was not
        # abandoned, whatever earlier attempts recorded.
        for row in self.store.sft.values():
            candidate_id = row.get("candidate_id")
            if candidate_id:
                status[str(candidate_id)] = {
                    "annotated": True,
                    "sft_id": str(row.get("sft_id") or ""),
                    "annotation_failure_code": None,
                }
        return status

    def _sync_annotation_status(self) -> dict[str, int]:
        """Write each winner's annotation outcome into the published SFT metadata.

        The tar members were packed at QA time, when the winner was known but its
        text did not exist yet; ``metadata.jsonl`` is the one archived file the
        landing contract allows a producer to rewrite afterwards, so it is where
        "did this sample end up with an instruction" belongs.  Every member row of
        a winner's sample carries the flag, which is also what puts it in the
        catalog's sample payload on the next rebuild.

        A failed winner keeps its bytes: the tar is append-only and the sample is
        still a legitimate render, it simply has no training text.
        """
        root = self.dependencies.archive_root
        if root is None:
            return {}
        status = self._winner_annotation_status()
        batches = sorted(Path(root).glob(f"sft/{self.config.build_id}/batch-*"))
        rewritten = samples = 0
        for dataset in batches:
            metadata = dataset / "metadata.jsonl"
            if not metadata.is_file():
                continue
            rows = [
                json.loads(line)
                for line in metadata.read_text(encoding="utf-8").splitlines()
                if line.strip()
            ]
            # Only the sample-level ``.vrmeta.json`` row names the candidate, so
            # it is what maps the archive's sample key onto this build's winner.
            by_sample = {
                str(row["sample_id"]): str(row["candidate_id"])
                for row in rows
                if row.get("sample_id") and row.get("candidate_id")
            }
            samples += sum(1 for value in by_sample.values() if value in status)
            changed = False
            for row in rows:
                fields = status.get(by_sample.get(str(row.get("sample_id") or ""), ""))
                if fields is None or all(row.get(k) == v for k, v in fields.items()):
                    continue
                row.update(fields)
                changed = True
            # Rewriting an unchanged file would only churn the archive: a resume
            # re-derives the same outcome and has nothing to say.
            if changed:
                _write_jsonl_atomic(metadata, rows)
                rewritten += 1
        return {"datasets": rewritten, "samples": samples, "winners": len(status)}

    def _clean_orphan_assets(self) -> int:
        """Reclaim assets of abandoned group attempts: nothing durable names them.

        The rescan also re-bases the water mark, which is what accounts for the
        assets ``_land_groups`` just unlinked.
        """
        referenced = {
            path
            for group in self.store.groups.values()
            for path in self._group_assets(group)
        }
        self._recalibrate_staged()
        buffer = (
            str(self._prefetch.directory) + os.sep if self._prefetch is not None else None
        )
        orphans = [
            path for path in self._staged
            if path not in referenced and not (buffer and path.startswith(buffer))
        ]
        for path in orphans:
            os.unlink(path)
            self._staged_bytes -= self._staged.pop(path)
        return len(orphans)

    def _land_checkpoint(self, *, force: bool = False) -> dict[str, Any] | None:
        """Publish, mirror and reclaim — the whole durability step, in one place."""
        if self.dependencies.archive_root is None and self.dependencies.mirror_root is None:
            return None
        if not force and self._staged_bytes < LAND_WATERMARK_BYTES:
            return None
        result = None
        if self.dependencies.archive_root is not None:
            result = self._land_groups()
            self._clean_orphan_assets()
        if self.dependencies.mirror_root is not None:
            self.store.checkpoint()
            self.store.write_manifest(self._manifest(self._phase))
            mirror_artifacts(
                self.store.root, Path(self.dependencies.mirror_root) / self.config.build_id
            )
        return result

    def _queue_sam3(self, source: SourceRecord, error: MaskPlanError) -> None:
        task_id = stable_id("sam3", self.config.build_id, source.source_id)
        self._failure(
            event_type="queued",
            stage="sam3_relabel",
            task_id=task_id,
            error_code="sam3_relabel_queued",
            message=f"{error.code}: {error}",
            retryable=True,
            terminal=False,
            source=source,
            durable=True,
        )

    def _terminal_source(
        self, source: SourceRecord, stage: str, code: str, message: object
    ) -> None:
        self._failure(
            event_type="terminal",
            stage=stage,
            task_id=stable_id(stage, self.config.build_id, source.source_id),
            error_code=code,
            message=message,
            retryable=False,
            terminal=True,
            source=source,
            durable=True,
        )

    def _record_candidate_failure(
        self,
        *,
        source: SourceRecord,
        group_id: str,
        group_attempt: int,
        pending: _PendingCandidate,
        error: Exception,
    ) -> None:
        code = getattr(error, "code", None) or (
            "visibility_rejected" if isinstance(error, PipelineError)
            else "visibility_invalid_weights" if isinstance(error, VisibilityError)
            else "candidate_render_failed"
        )
        self._failure(
            event_type="attempt",
            stage="rendering",
            task_id=pending.attempt_task_id,
            error_code=str(code),
            message=error,
            retryable=True,
            terminal=False,
            source=source,
            group_id=group_id,
            round_number=group_attempt,
            attempt=pending.preset_reservation.attempt,
        )

    def _cgt_path(self, mask: Any) -> Path:
        return self.store.assets_root / "masks" / f"{mask.mask_id}.png"

    def _start_cgt_writes(self, slots: list[Any]) -> dict[str, Future[Path]]:
        """Encode this source's C_GT set once, ahead of and outside its turn.

        A local plan has seven physical masks behind eight slots (the two
        semantic slots share one), and every group attempt of the source reuses
        them, so the fan-out is keyed by ``mask_id`` and submitted once.  Doing it
        here — before the turn is even claimed — keeps 243 ms of PNG encoding per
        group off the critical section every other source is waiting on, while
        ``_render_source`` still joins the writes before anything durable can
        name them.
        """
        futures: dict[str, Future[Path]] = {}
        for slot in slots:
            mask = slot.mask
            if mask.mask_id in futures:
                continue
            futures[mask.mask_id] = self._postprocess_executor.submit(
                _write_cgt_once, mask, self._cgt_path(mask)
            )
        return futures

    def _start_candidate(
        self,
        *,
        source: SourceRecord,
        prepared: Any,
        postprocess_reference: Any,
        group_id: str,
        group_attempt: int,
        mode: str,
        reservation: Any,
        slot_index: int,
        slot: Any,
    ) -> _PendingCandidate | None:
        """Reserve and render one attempt for deferred serial resolution."""
        pending, _ = self._start_candidate_batch(
            source=source,
            prepared=prepared,
            postprocess_reference=postprocess_reference,
            group_id=group_id,
            group_attempt=group_attempt,
            mode=mode,
            reservation=reservation,
            indexed_slots=[(slot_index, slot)],
        )
        return pending[0] if pending else None

    def _start_candidate_batch(
        self,
        *,
        source: SourceRecord,
        prepared: Any,
        postprocess_reference: Any,
        group_id: str,
        group_attempt: int,
        mode: str,
        reservation: Any,
        indexed_slots: list[tuple[int, Any]],
    ) -> tuple[list[_PendingCandidate], bool]:
        """Reserve a slot wave, batch production renders, then defer ordered resolution."""
        if self.renderer is None:
            raise PipelineError("render resource is not loaded")
        attempts: list[_CandidateAttempt] = []
        exhausted = False
        for slot_index, slot in indexed_slots:
            slot_id = slot.slot_id if mode == "local" else str(slot)
            preset_reservation = reservation.reserve_candidate(slot_id)
            if preset_reservation is None:
                exhausted = True
                break
            preset = preset_reservation.link.preset
            candidate_id = stable_id("candidate", group_id, slot_id)
            attempts.append(_CandidateAttempt(
                slot_index=slot_index,
                slot=slot,
                slot_id=slot_id,
                preset_reservation=preset_reservation,
                attempt_task_id=stable_id(
                    "render-attempt", group_id, slot_id,
                    preset_reservation.attempt, preset.preset_id,
                ),
                candidate_id=candidate_id,
                after_path=self.store.assets_root / "candidates" / f"{candidate_id}.jpg",
            ))

        requests = [
            (
                attempt.preset_reservation.link.preset,
                attempt.slot.mask if mode == "local" else None,
            )
            for attempt in attempts
        ]
        try:
            render_many = getattr(self.renderer, "render_many")
        except AttributeError:
            render_many = None
        rendered_results: list[Any] = [None] * len(attempts)
        postprocess_futures: dict[int, Future[_PostprocessResult]] = {}

        def accept_rendered(index: int, rendered: Any) -> None:
            rendered_results[index] = rendered
            if isinstance(rendered, Exception):
                return
            mask = attempts[index].slot.mask if mode == "local" else None
            postprocess_futures[index] = self._postprocess_executor.submit(
                _postprocess_candidate,
                postprocess_reference,
                rendered.pixels,
                mask.effective_alpha if mask is not None else None,
                attempts[index].after_path,
                self.config.render,
            )

        # Unmasked LUT batches are faster as one render_many call: the two-wave
        # path uses the same physical GPU and otherwise uploads the source twice.
        # Mask validation and parameter replay retain the concurrent wave path.
        unmasked_lut_batch = all(
            preset.format == "lut" and mask is None
            for preset, mask in requests
        )
        if callable(render_many) and len(requests) > 1 \
                and self.config.render.gpu_concurrency > 1 \
                and not unmasked_lut_batch:
            worker_count = min(self.config.render.gpu_concurrency, len(requests))
            chunks = [list(range(worker, len(requests), worker_count))
                      for worker in range(worker_count)]
            render_futures = {
                self._render_executor.submit(
                    render_many, prepared, [requests[index] for index in indices]
                ): indices
                for indices in chunks
            }
            for future in as_completed(render_futures):
                indices = render_futures[future]
                try:
                    chunk_results = list(future.result())
                    if len(chunk_results) != len(indices):
                        raise PipelineError("batched renderer returned the wrong slot count")
                except Exception as exc:  # resolved and journaled in stable slot order
                    chunk_results = [exc] * len(indices)
                for index, rendered in zip(indices, chunk_results):
                    accept_rendered(index, rendered)
        elif callable(render_many):
            try:
                chunk_results = list(render_many(prepared, requests))
                if len(chunk_results) != len(attempts):
                    raise PipelineError("batched renderer returned the wrong slot count")
            except Exception as exc:  # resolved and journaled in stable slot order
                chunk_results = [exc] * len(attempts)
            for index, rendered in enumerate(chunk_results):
                accept_rendered(index, rendered)
        else:
            for index, (preset, mask) in enumerate(requests):
                try:
                    rendered = self.renderer.render(prepared, preset, mask)
                except Exception as exc:  # resolved and journaled in stable slot order
                    rendered = exc
                accept_rendered(index, rendered)

        pending: list[_PendingCandidate] = []
        for index, (attempt, rendered) in enumerate(zip(attempts, rendered_results)):
            if isinstance(rendered, Exception):
                future: Future = Future()
                future.set_exception(rendered)
                render_value = None
            else:
                future = postprocess_futures[index]
                render_value = rendered
            pending.append(_PendingCandidate(
                slot_index=attempt.slot_index,
                slot=attempt.slot,
                slot_id=attempt.slot_id,
                preset_reservation=attempt.preset_reservation,
                attempt_task_id=attempt.attempt_task_id,
                candidate_id=attempt.candidate_id,
                after_path=attempt.after_path,
                rendered=render_value,
                future=future,
            ))
        return pending, exhausted

    def _resolve_candidate(
        self,
        *,
        source: SourceRecord,
        prepared: Any,
        postprocess_reference: Any,
        group_id: str,
        group_attempt: int,
        mode: str,
        reservation: Any,
        pending: _PendingCandidate,
        before_retry: Callable[[], None] | None = None,
    ) -> dict[str, Any] | None:
        """Resolve one slot in serial order, retrying it without reordering events."""
        while True:
            try:
                result = pending.future.result()
            except Exception as exc:  # noqa: BLE001 - refill this exact slot
                self._record_candidate_failure(
                    source=source,
                    group_id=group_id,
                    group_attempt=group_attempt,
                    pending=pending,
                    error=exc,
                )
                if before_retry is not None:
                    before_retry()
                reservation.reject(pending.preset_reservation)
                replacement = self._start_candidate(
                    source=source,
                    prepared=prepared,
                    postprocess_reference=postprocess_reference,
                    group_id=group_id,
                    group_attempt=group_attempt,
                    mode=mode,
                    reservation=reservation,
                    slot_index=pending.slot_index,
                    slot=pending.slot,
                )
                if replacement is None:
                    return None
                pending = replacement
                continue

            preset_reservation = pending.preset_reservation
            preset = preset_reservation.link.preset
            if pending.rendered is None:
                raise PipelineError("successful candidate is missing its render result")
            mask = pending.slot.mask if mode == "local" else None
            self._account_asset(pending.after_path)
            # The PNG itself is encoded off the selector turn by ``_start_cgt_writes``
            # and joined before the group can be journaled; only its name is needed
            # to describe the candidate.
            cgt_path: Path | None = None
            if mask is not None:
                cgt_path = self._cgt_path(mask)
            recipe = {
                "preset_id": preset.preset_id,
                "preset_path": str(preset.path),
                "format": preset.format,
                "render_engine": pending.rendered.engine,
                "render_mode": mode,
            }
            candidate: dict[str, Any] = {
                "candidate_id": pending.candidate_id,
                "slot_id": pending.slot_id,
                "slot_index": pending.slot_index,
                "preset_id": preset.preset_id,
                "preset_path": str(preset.path),
                "format": preset.format,
                "kind": preset.kind,
                "style_name": preset.style_name,
                "major": preset_reservation.link.major,
                "minor": preset_reservation.link.minor,
                "after_path": str(pending.after_path),
                "render_engine": pending.rendered.engine,
                "render_diagnostics": pending.rendered.diagnostics,
                "visibility": {
                    "visible_de": round(result.metrics.visible_de, 6),
                    "visible_fraction": round(result.metrics.visible_fraction, 6),
                    "accepted": True,
                },
                "objective_hints": result.hints,
                "_qa_stats": result.qa_stats,
                "attempt_lineage": {
                    "group_attempt": group_attempt,
                    "preset_attempt": preset_reservation.attempt,
                    "preset_reservation_id": preset_reservation.reservation_id,
                },
                "recipe": recipe,
            }
            if mask is not None:
                recipe.update({"mask_id": mask.mask_id, "amount": mask.amount})
                candidate.update({
                    "slot_mode": pending.slot.mode,
                    "mode_index": pending.slot.mode_index,
                    "pairing_index": pending.slot.pairing_index,
                    "mask_id": mask.mask_id,
                    "cgt_path": str(cgt_path),
                    "subject": source.subject,
                    "region": mask.region,
                    "geometry": mask.geometry,
                    "raw_alpha_mean": round(mask.raw_alpha_mean, 8),
                    "amount": round(mask.amount, 8),
                    "effective_alpha_mean": round(mask.effective_alpha_mean, 8),
                })
            reservation.accept(preset_reservation)
            return candidate

    @staticmethod
    def _cancel_pending_candidates(
        pending: list[_PendingCandidate],
        *,
        reservation: Any | None = None,
    ) -> None:
        """Quiesce speculative CPU work before abandoning its reservation."""
        for candidate in pending:
            candidate.future.cancel()
        for candidate in pending:
            try:
                candidate.future.result()
            except Exception:
                pass
        if reservation is not None:
            for candidate in reversed(pending):
                reservation.cancel_speculative(candidate.preset_reservation)

    def _render_source(
        self,
        source: SourceRecord,
        mode: str,
        *,
        queue_mask_failure: bool = True,
        defer_commit: bool = False,
        selector_turn: _SelectorTurn | None = None,
    ) -> str | _DeferredGroupCommit:
        if self.renderer is None or self.scorer is None:
            raise PipelineError("render/QA resources are not loaded")
        selector = self.selectors[mode]
        if selector is None:
            raise PipelineError(f"missing {mode} selector")
        try:
            prepared = preprocess_source(source.source_path, self.config.render.short_edge)
        except Exception as exc:  # noqa: BLE001 - a corrupt source is replaced in its mode
            self._terminal_source(source, "rendering", "source_preprocess_failed", exc)
            return "terminal"
        working_reference = prepare_working_reference(
            prepared.pixels, self.config.render.diff_short_edge
        )
        torch_reference = (
            prepare_torch_lab_reference(working_reference, self.renderer.device)
            if self.config.render.visibility_backend == "torch" else None
        )
        postprocess_reference = _PostprocessReference(
            working=working_reference, torch=torch_reference
        )

        slots: list[Any]
        if mode == "local":
            try:
                plan = build_mask_plan(
                    source,
                    build_id=self.config.build_id,
                    seed=self.config.seed,
                    width=prepared.width,
                    height=prepared.height,
                    linear_target=self.config.masks.linear_target_alpha_mass,
                )
                slots = list(pair_mask_slots(
                    plan,
                    build_id=self.config.build_id,
                    seed=self.config.seed,
                    source_id=source.source_id,
                ))
            except MaskPlanError as exc:
                if queue_mask_failure:
                    self._queue_sam3(source, exc)
                    return "sam3_queued"
                return f"mask_failed:{exc.code}:{exc}"
        else:
            slots = [f"global-{index}" for index in range(8)]

        cgt_writes = self._start_cgt_writes(slots) if mode == "local" else {}
        if selector_turn is not None:
            selector_turn.ready.wait()
        selector_handed_off = False
        excluded_majors: list[str] = []
        cgt_error: Exception | None = None
        for group_attempt in range(len(selector.majors)):
            group_id = stable_id(
                "group", self.config.build_id, source.source_id, mode, group_attempt
            )
            reservation = None
            group_persisted = False
            try:
                reservation = selector.begin_group(
                    source.source_id, group_attempt, exclude_majors=excluded_majors
                )
                if cgt_error is not None:
                    # The C_GT futures are submitted once for the whole source, so
                    # a write that already failed cannot succeed on a later major.
                    # Fail this attempt where the barrier below would have failed
                    # it anyway, instead of rendering eight candidates first.
                    raise cgt_error
                candidates: list[dict[str, Any]] = []
                exhausted = False
                pending_candidates, exhausted = self._start_candidate_batch(
                    source=source,
                    prepared=prepared,
                    postprocess_reference=postprocess_reference,
                    group_id=group_id,
                    group_attempt=group_attempt,
                    mode=mode,
                    reservation=reservation,
                    indexed_slots=list(enumerate(slots)),
                )
                if not exhausted:
                    pending_index = 0
                    while pending_index < len(slots):
                        pending = pending_candidates[pending_index]

                        def rollback_later() -> None:
                            later = pending_candidates[pending_index + 1:]
                            self._cancel_pending_candidates(
                                later, reservation=reservation
                            )
                            del pending_candidates[pending_index + 1:]

                        candidate = self._resolve_candidate(
                            source=source,
                            prepared=prepared,
                            postprocess_reference=postprocess_reference,
                            group_id=group_id,
                            group_attempt=group_attempt,
                            mode=mode,
                            reservation=reservation,
                            pending=pending,
                            before_retry=rollback_later,
                        )
                        if candidate is None:
                            exhausted = True
                            self._cancel_pending_candidates(
                                pending_candidates[pending_index + 1:]
                            )
                            break
                        candidates.append(candidate)
                        if len(pending_candidates) < len(slots):
                            replacements, exhausted = self._start_candidate_batch(
                                source=source,
                                prepared=prepared,
                                postprocess_reference=postprocess_reference,
                                group_id=group_id,
                                group_attempt=group_attempt,
                                mode=mode,
                                reservation=reservation,
                                indexed_slots=[
                                    (slot_index, slots[slot_index])
                                    for slot_index in range(len(pending_candidates), len(slots))
                                ],
                            )
                            pending_candidates.extend(replacements)
                        if exhausted:
                            self._cancel_pending_candidates(
                                pending_candidates[pending_index + 1:]
                            )
                            break
                        pending_index += 1
                elif pending_candidates:
                    self._cancel_pending_candidates(pending_candidates)
                if exhausted or len(candidates) != 8:
                    raise PresetError("major could not yield eight visible candidates")
                # Join the off-turn C_GT encodes before the turn is released: no
                # journal line may name a mask that is not already fsynced, and a
                # write that failed has to surface where the group attempt can
                # still be retried in another major.  By now they have had the
                # whole render and postprocess wave to finish, so this is a
                # formality rather than a wait.  The failure is remembered so the
                # next major short-circuits above rather than re-rendering.
                try:
                    for cgt_write in cgt_writes.values():
                        self._account_asset(cgt_write.result())
                except Exception as exc:
                    cgt_error = exc
                    raise
                if selector_turn is not None:
                    # Accepted reservations contribute the same active counts that a
                    # serial commit would. The next source may select while this one
                    # runs IAA, without changing coverage order.
                    selector_turn.following.set()
                    selector_handed_off = True
                ranked = rank_candidates(
                    str(source.source_path), candidates, self.scorer,
                    batch_size=self.config.render.iaa_batch,
                    abstain_margin=self.config.render.qa_winner_margin_abstain,
                    low_margin=self.config.render.qa_winner_margin_low,
                )
                coverage = {
                    "major": reservation.major,
                    "coverage_cycle": reservation.coverage_cycle,
                    "coverage_position": reservation.coverage_position,
                    "reservation_id": reservation.reservation_id,
                }
                completed_at = _timestamp(self.dependencies.now())
                by_id = {row["candidate_id"]: row for row in ranked.candidates}
                group = {
                    "schema_version": 1,
                    "build_id": self.config.build_id,
                    "group_id": group_id,
                    "source_id": source.source_id,
                    "source_path": str(source.source_path),
                    "scene": source.scene,
                    "subject": source.subject,
                    "render_mode": mode,
                    "preset_filter": self.config.preset_filter,
                    "group_attempt": group_attempt,
                    **coverage,
                    "candidates": list(ranked.candidates),
                    "winner_ids": list(ranked.winner_ids),
                    "winner_ranks": [by_id[candidate_id]["rank"]
                                     for candidate_id in ranked.winner_ids],
                    # The rank1-rank2 OneAlign gap and the margin policy's verdict
                    # on it.  An abstaining group keeps ``winner_ids: []`` — the
                    # same shape a group with no candidate above SFT_THRESHOLD has
                    # always had — so nothing downstream needs a new state, and the
                    # verdict is what tells the two apart afterwards.
                    "winner_margin": ranked.winner_margin,
                    "winner_confidence": ranked.winner_confidence,
                    "source_onealign": ranked.source_score,
                    "stage_timestamps": {
                        "render_completed_at": completed_at,
                        "qa_completed_at": completed_at,
                    },
                }
                if defer_commit:
                    return _DeferredGroupCommit(
                        group=group,
                        reservation=reservation,
                        coverage=coverage,
                    )
                if not self.store.append_group(group):
                    raise StateError(f"unexpected existing group during render: {group_id}")
                self.store.checkpoint()
                group_persisted = True
                committed = reservation.commit()
                if committed != coverage:
                    raise StateError("coverage commit metadata changed after durable group append")
                return "completed"
            except QaError:
                if reservation is not None and not group_persisted:
                    reservation.abandon()
                raise
            except Exception as exc:  # noqa: BLE001 - restart group in another major
                if selector_handed_off:
                    if reservation is not None and not group_persisted:
                        reservation.abandon()
                    raise
                if group_persisted:
                    raise PipelineError(
                        f"durable group {group_id} could not commit in-memory coverage"
                    ) from exc
                major = reservation.major if reservation is not None else "unknown"
                if reservation is not None:
                    reservation.abandon()
                excluded_majors.append(major)
                self._failure(
                    event_type="group_attempt",
                    stage="rendering",
                    task_id=stable_id("render-group", group_id),
                    error_code="major_exhausted",
                    message=f"{major}: {type(exc).__name__}: {exc}",
                    retryable=True,
                    terminal=False,
                    source=source,
                    group_id=group_id,
                    round_number=group_attempt,
                    attempt=group_attempt + 1,
                )
        self._terminal_source(
            source, "rendering", "preset_inventory_exhausted",
            f"no taxonomy major yielded eight accepted {mode} candidates",
        )
        return "terminal"

    def _render_source_buffered(
        self,
        source: SourceRecord,
        mode: str,
        *,
        selector_turn: _SelectorTurn | None = None,
    ) -> _DeferredSourceResult:
        """Render one source off-thread without mutating the durable journals."""
        failures: list[_BufferedFailure] = []
        self._source_context.failures = failures
        try:
            try:
                result = self._render_source(
                    source, mode, defer_commit=True, selector_turn=selector_turn
                )
            except Exception as exc:  # flushed in source order before the error escapes
                self._failure(
                    event_type="source_worker",
                    stage="rendering",
                    task_id=stable_id(
                        "render-source", self.config.build_id, source.source_id, mode
                    ),
                    error_code="source_worker_failed",
                    message=f"{type(exc).__name__}: {exc}",
                    retryable=True,
                    terminal=False,
                    source=source,
                    durable=True,
                )
                return _DeferredSourceResult(
                    status="error", failures=tuple(failures), error=exc
                )
            if isinstance(result, _DeferredGroupCommit):
                return _DeferredSourceResult(
                    status="completed",
                    failures=tuple(failures),
                    group_commit=result,
                )
            return _DeferredSourceResult(status=result, failures=tuple(failures))
        finally:
            if selector_turn is not None and not selector_turn.following.is_set():
                selector_turn.ready.wait()
                selector_turn.following.set()
            del self._source_context.failures

    def _commit_source_result(self, result: _DeferredSourceResult) -> str:
        """Flush one buffered source outcome at its allocation-order boundary."""
        for failure in result.failures:
            self.store.append_failure(failure.row, durable=failure.durable)
        if result.error is not None:
            raise result.error
        commit = result.group_commit
        if commit is None:
            return result.status
        if not self.store.append_group(commit.group):
            commit.reservation.abandon()
            raise StateError(
                f"unexpected existing group during render: {commit.group['group_id']}"
            )
        self.store.checkpoint()
        committed = commit.reservation.commit()
        if committed != commit.coverage:
            raise PipelineError("coverage commit metadata changed after durable group append")
        return result.status

    def _discard_source_result(self, result: _DeferredSourceResult) -> None:
        """Keep diagnostics but abandon work beyond an allocation-order error."""
        for failure in result.failures:
            self.store.append_failure(failure.row, durable=failure.durable)
        if result.group_commit is not None:
            result.group_commit.reservation.abandon()

    def _chunk_paths(
        self,
        sources: tuple[SourceRecord, ...],
        start: int,
        skip: set[str],
        limit: int = PREFETCH_CHUNK,
    ) -> list[str]:
        """The source images one chunk of the allocation still has to read."""
        if limit <= 0:
            return []
        done = self.store.completed_sources() | skip
        return [
            str(source.source_path)
            for source in sources[start:start + PREFETCH_CHUNK]
            if source.source_id not in done
        ][:limit]

    def _account_prefetched(self) -> None:
        """Wait for the outstanding chunk and fold its bytes into the water mark."""
        assert self._prefetch is not None
        for path in self._prefetch.take():
            self._account_asset(path)

    def _rotate_prefetch(
        self,
        sources: tuple[SourceRecord, ...],
        start: int,
        skip: set[str],
        remaining: int | None = None,
    ) -> None:
        """Take delivery of the chunk about to render and queue the one behind it.

        The allocation order is known before the round starts, which is the whole
        premise: chunk k is already on the tmpfs when its first source is opened,
        and chunk k+1 is read sequentially while chunk k renders.  The very first
        chunk of a mode has nothing to hide behind, so it is fetched inline — the
        same bytes the renderer would otherwise take one random pread at a time.
        """
        if self._prefetch is None:
            return
        budget = PREFETCH_CHUNK * 2 if remaining is None else max(0, remaining)
        current = self._chunk_paths(
            sources, start, skip, limit=min(PREFETCH_CHUNK, budget)
        )
        # Outstanding here is chunk k, or — at the start of a mode — whatever the
        # previous mode's last look-ahead read.
        self._account_prefetched()
        if start == 0:
            self._prefetch.submit(current)
            self._account_prefetched()
        # Keep one replacement source warm even when the current chunk can fill
        # the target. A terminal source then preserves the double-buffer contract
        # without restoring the old 256-source overfetch on small builds.
        lookahead = (
            max(1, budget - len(current))
            if budget > 0 and current else budget
        )
        self._prefetch.submit(self._chunk_paths(
            sources,
            start + PREFETCH_CHUNK,
            skip,
            limit=min(PREFETCH_CHUNK, lookahead),
        ))

    def _discard_prefetched(self, source: SourceRecord) -> None:
        """Drop one source's buffered copy once it can no longer be read again."""
        if self._prefetch is None:
            return
        path = self._prefetch.path_for(source.source_path)
        path.unlink(missing_ok=True)
        with self._staged_lock:
            self._staged_bytes -= self._staged.pop(str(path), 0)

    def _fill_initial_mode(self, mode: str, sources: tuple[SourceRecord, ...], target: int) -> None:
        terminal = self._terminal_source_ids()
        pending = self._pending_sam3_ids() if mode == "local" else set()
        inflight: deque[tuple[SourceRecord, Future[_DeferredSourceResult]]] = deque()
        selector_tail = threading.Event()
        selector_tail.set()

        def finish_oldest() -> None:
            source, future = inflight.popleft()
            try:
                result = self._commit_source_result(future.result())
            except Exception:
                while inflight:
                    _, later = inflight.popleft()
                    self._discard_source_result(later.result())
                raise
            if result == "sam3_queued":
                pending.add(source.source_id)
            else:
                if result == "terminal":
                    terminal.add(source.source_id)
                self._discard_prefetched(source)
            if not inflight:
                self._land_checkpoint()

        for index, source in enumerate(sources):
            while inflight and (
                len(inflight) >= self._source_window
                or len(self._mode_groups(mode)) + len(pending) + len(inflight) >= target
                or self._staged_size() >= LAND_WATERMARK_BYTES
            ):
                finish_oldest()
            completed = len(self._mode_groups(mode))
            reserved = (len(pending) if mode == "local" else 0) + len(inflight)
            if completed + reserved >= target:
                break
            # After the target check so a finished mode reads nothing more, and
            # before the skip below so a chunk starting on a done source still
            # rotates.
            if index % PREFETCH_CHUNK == 0:
                self._rotate_prefetch(
                    sources,
                    index,
                    terminal | pending,
                    remaining=target - completed - reserved,
                )
            if source.source_id in self.store.completed_sources() \
                    or source.source_id in terminal or source.source_id in pending:
                continue
            following = threading.Event()
            turn = _SelectorTurn(ready=selector_tail, following=following)
            selector_tail = following
            inflight.append((
                source,
                self._source_executor.submit(
                    self._render_source_buffered, source, mode, selector_turn=turn
                ),
            ))
        while inflight:
            finish_oldest()

    def _release_heavy_resources(self) -> None:
        """Give the render GPUs back, for a phase that needs the cards elsewhere."""
        self.renderer = None
        self.scorer = None
        gc.collect()
        _empty_cuda_cache()

    def _record_sam3_attempt(
        self,
        source: SourceRecord,
        attempt: int,
        code: str,
        message: object,
        *,
        event_type: str = "sam3_attempt",
    ) -> None:
        self._failure(
            event_type=event_type,
            stage="sam3_relabel",
            task_id=stable_id("sam3", self.config.build_id, source.source_id),
            error_code=code,
            message=message,
            retryable=True,
            terminal=False,
            source=source,
            round_number=attempt,
            attempt=attempt,
            durable=True,
        )

    def _terminal_sam3(self, source: SourceRecord, message: object) -> None:
        self._terminal_source(source, "sam3_relabel", "sam3_relabel_failed", message)

    def _try_ready_relabel(self, source: SourceRecord, attempt: int) -> str:
        refreshed, reason = refresh_source_record(source)
        if refreshed is None:
            self._record_sam3_attempt(
                source, attempt, "sam3_integrity_failed", reason,
                event_type="sam3_ready_invalid",
            )
            return "retry"
        result = self._render_source(refreshed, "local", queue_mask_failure=False)
        if result == "completed":
            return "completed"
        if result.startswith("mask_failed:"):
            self._record_sam3_attempt(
                source, attempt, "sam3_geometry_failed", result,
                event_type="sam3_ready_invalid",
            )
            return "retry"
        return "terminal"

    def _drain_sam3_and_replacements(self) -> None:
        by_id = {source.source_id: source for source in self.allocation.local}
        target = self.allocation.local_target
        max_attempts = self.config.masks.sam3_relabel_attempts
        while len(self._mode_groups("local")) < target:
            pending_ids = self._pending_sam3_ids()
            pending = [by_id[source_id] for source_id in sorted(pending_ids) if source_id in by_id]

            made_progress = False
            for source in list(pending):
                if not self._unconsumed_sam3_ready(source.source_id):
                    continue
                attempt = self._sam3_attempts(source.source_id)
                status = self._try_ready_relabel(source, attempt)
                made_progress = True
                if status == "retry" and attempt >= max_attempts:
                    self._terminal_sam3(source, "ready relabel still violates canonical geometry")

            pending_ids = self._pending_sam3_ids()
            pending = [by_id[source_id] for source_id in sorted(pending_ids) if source_id in by_id]
            exhausted = [
                source for source in pending
                if self._sam3_attempts(source.source_id) >= max_attempts
                and not self._unconsumed_sam3_ready(source.source_id)
            ]
            for source in exhausted:
                self._terminal_sam3(source, "SAM3 relabel attempt budget exhausted")
                made_progress = True

            pending_ids = self._pending_sam3_ids()
            pending = [by_id[source_id] for source_id in sorted(pending_ids) if source_id in by_id]
            candidates = [
                source for source in pending
                if self._sam3_attempts(source.source_id) < max_attempts
                and not self._unconsumed_sam3_ready(source.source_id)
            ]
            if candidates:
                next_attempt = min(self._sam3_attempts(row.source_id) + 1 for row in candidates)
                batch = [
                    row for row in candidates
                    if self._sam3_attempts(row.source_id) + 1 == next_attempt
                ]
                # SAM3 loads next to the two OneAlign copies on cuda:0 (33.0 GB of
                # 95.6 GB) while the renderer keeps cuda:1, so a batch fits without
                # evicting anything: the 860 M-parameter detector is ~1.7 GB in
                # bf16 and its batch forward already falls back per image on OOM.
                # Dropping and reloading both models per batch instead cost
                # 11.5-11.8 s a round (six checkpoint-shard loads in a 30-group
                # run) and left cuda:0 briefly empty enough for the vGate
                # supervisor to place an 83 GB replica on it mid-run.
                _empty_cuda_cache()
                try:
                    statuses = self.dependencies.relabeler(batch, self.config, next_attempt)
                except Exception as exc:  # noqa: BLE001 - each source consumes one bounded attempt
                    statuses = {source.source_id: f"relabel_exception:{type(exc).__name__}:{exc}"
                                for source in batch}
                for source in batch:
                    status = str(statuses.get(source.source_id) or "missing_relabel_status")
                    self._record_sam3_attempt(
                        source, next_attempt, f"sam3_{status}", status
                    )
                    refreshed, reason = refresh_source_record(source)
                    if refreshed is None:
                        if next_attempt >= max_attempts:
                            self._terminal_sam3(source, f"{status}; integrity={reason}")
                        continue
                    self._record_sam3_attempt(
                        source, next_attempt, "sam3_relabel_ready", status,
                        event_type="sam3_ready",
                    )
                    result = self._try_ready_relabel(source, next_attempt)
                    if result == "retry" and next_attempt >= max_attempts:
                        self._terminal_sam3(source, "relabel cannot satisfy canonical geometry")
                made_progress = True

            pending_count = len(self._pending_sam3_ids())
            before = len(self._mode_groups("local"))
            if before + pending_count < target:
                self._fill_initial_mode("local", self.allocation.local, target)
                made_progress = made_progress or len(self._mode_groups("local")) > before \
                    or len(self._pending_sam3_ids()) > pending_count
            if not made_progress:
                break

        if self._pending_sam3_ids():
            raise PipelineError("SAM3 relabel queue remains unresolved")

    def _record_shortfalls(self) -> None:
        for mode, target in (
            ("local", self.allocation.local_target),
            ("global", self.allocation.global_target),
        ):
            completed = len(self._mode_groups(mode))
            if completed >= target:
                continue
            task_id = stable_id("target", self.config.build_id, mode)
            self._failure(
                event_type="terminal",
                stage="rendering",
                task_id=task_id,
                error_code=f"{mode}_target_shortfall",
                message=f"requested {target} {mode} groups, completed {completed}",
                retryable=False,
                terminal=True,
                durable=True,
            )

    def execute(self) -> dict[str, Any]:
        """Run the lifecycle with the prefetch buffer live for its whole duration."""
        if self._prefetch is not None:
            self._prefetch.activate()
        try:
            return self._run_phases()
        finally:
            self._source_executor.shutdown(wait=True, cancel_futures=True)
            self._render_executor.shutdown(wait=True, cancel_futures=True)
            self._postprocess_executor.shutdown(wait=True, cancel_futures=True)
            if self._prefetch is not None:
                self._prefetch.close()

    def _run_phases(self) -> dict[str, Any]:
        self._write_phase("preflight")
        # Resume boundary: a restarted build may have lost unlanded assets.
        self._verify_group_assets()
        self._write_phase("rendering")
        self._fill_initial_mode(
            "global", self.allocation.global_, self.allocation.global_target
        )
        self._fill_initial_mode(
            "local", self.allocation.local, self.allocation.local_target
        )

        self._write_phase("sam3_relabel")
        self._drain_sam3_and_replacements()
        self._record_shortfalls()
        self._land_checkpoint(force=True)

        self._release_heavy_resources()
        self._write_phase("annotation")
        # Annotation reads winner bytes by their staging path, which now lives in
        # the archive; the reverse map must know about this build's batches first.
        self._refresh_catalog()
        annotation = self.dependencies.annotator_factory(self.config, self.store).drain()
        if annotation.get("pending") or self.store.pending_annotation_tasks():
            self._write_phase("annotation")
            raise PipelineError("annotation queue remains unresolved")
        # Every winner's outcome is now final, which is the earliest the published
        # SFT datasets can be told which of their samples carry training text.
        self._annotation_sync = self._sync_annotation_status()

        self._write_phase("projection")
        terminal = any(row.get("terminal") for row in self.store.failures)
        final_status = "complete_with_failures" if terminal else "complete"
        final_manifest = self._manifest(final_status, final_status)
        final_manifest["ended_at"] = _timestamp(self.dependencies.now())
        projection = self.dependencies.projector(self.store, final_manifest, self.config)
        final_manifest["projection"] = dataclasses.asdict(projection)
        self.store.checkpoint()
        final_manifest["artifacts"] = {
            path.name: file_digest(path)
            for path in (self.store.groups_path, self.store.sft_path, self.store.failures_path)
            if path.exists()
        }
        _add_manifest_self_artifact(final_manifest)
        self.store.write_manifest(final_manifest)
        self.store.checkpoint()
        if self.dependencies.mirror_root is not None:
            mirror_artifacts(
                self.store.root, Path(self.dependencies.mirror_root) / self.config.build_id
            )
        return final_manifest


class LegacyImportPipeline:
    """Import an authorized legacy snapshot, then reuse the canonical Responses queue."""

    def __init__(
        self,
        config: DatabuildConfig,
        dependencies: PipelineDependencies,
        snapshot: LegacySnapshot,
        store: ArtifactStore,
        existing_manifest: Mapping[str, Any] | None,
    ) -> None:
        self.config = config
        self.dependencies = dependencies
        self.snapshot = snapshot
        self.store = store
        self.started_at = str(
            (existing_manifest or {}).get("started_at")
            or _timestamp(self.dependencies.now())
        )

    def _failure(
        self,
        group: LegacyGroupInput | None,
        *,
        code: str,
        message: object,
    ) -> None:
        source_asset_id = group.source_asset_id if group is not None else ""
        group_id = legacy_group_id(self.config, group) if group is not None else None
        source_id = (
            stable_id(
                "source", self.config.legacy_import.protocol, source_asset_id
            )
            if group is not None else None
        )
        task_id = stable_id("legacy-import", self.config.build_id, source_asset_id or code)
        if self.store.has_terminal_failure(task_id):
            return
        self.store.append_failure({
            "build_id": self.config.build_id,
            "event_id": stable_id("failure", self.config.build_id, task_id, code),
            "event_type": "terminal",
            "stage": "import",
            "task_id": task_id,
            "source_id": source_id,
            "source_path": (
                str(group.record.get("source")) if group is not None else None
            ),
            "group_id": group_id,
            "candidate_id": None,
            "round": None,
            "attempt": 1,
            "retryable": False,
            "error_code": code,
            "message": redact_text(message, self.config.secrets),
            "endpoint_id": None,
            "terminal": True,
            "timestamp": _timestamp(self.dependencies.now()),
        }, durable=True)

    def _annotation_counts(self) -> dict[str, Any]:
        sources: dict[str, int] = {}
        models: dict[str, int] = {}
        usage: dict[str, int] = {}
        for row in self.store.sft.values():
            source = str(row.get("annot_src") or "unknown")
            sources[source] = sources.get(source, 0) + 1
            meta = (row.get("qa") or {}).get("annotation") or {}
            model = str(meta.get("returned_model") or "unknown")
            models[model] = models.get(model, 0) + 1
            for key, value in (meta.get("usage") or {}).items():
                if isinstance(value, int):
                    usage[key] = usage.get(key, 0) + value
        return {
            "sources": dict(sorted(sources.items())),
            "models": dict(sorted(models.items())),
            "usage": dict(sorted(usage.items())),
        }

    def _manifest(self, phase: str, status: str = "running") -> dict[str, Any]:
        groups = list(self.store.groups.values())
        candidates = [candidate for group in groups for candidate in group["candidates"]]
        terminal = [row for row in self.store.failures if row.get("terminal")]
        by_code: dict[str, int] = {}
        for row in self.store.failures:
            code = str(row.get("error_code") or "unknown")
            by_code[code] = by_code.get(code, 0) + 1
        formats: dict[str, int] = {}
        modes: dict[str, int] = {}
        for candidate in candidates:
            fmt = str(candidate.get("format") or "unknown")
            mode = str(candidate.get("slot_mode") or "unknown")
            formats[fmt] = formats.get(fmt, 0) + 1
            modes[mode] = modes.get(mode, 0) + 1
        return {
            "schema_version": self.config.schema_version,
            "build_id": self.config.build_id,
            "phase": phase,
            "status": status,
            "started_at": self.started_at,
            "updated_at": _timestamp(self.dependencies.now()),
            "effective_config": self.config.sanitized_dict(),
            "targets": {
                "groups": self.config.target_groups,
                "local": self.config.target_groups,
                "global": 0,
                "sft": self.snapshot.sft_row_count,
            },
            "completed": {
                "groups": len(groups),
                "local": len(groups),
                "global": 0,
                "actual_ratio": {"local": 1.0 if groups else 0.0, "global": 0.0},
                "candidates": len(candidates),
                "winner_top1": sum(bool(group.get("winner_ids")) for group in groups),
                "winner_top2": sum(len(group.get("winner_ids") or []) >= 2 for group in groups),
                "sft": len(self.store.sft),
            },
            "sources": {
                "inventory": {
                    "legacy_source_groups": self.snapshot.source_group_count,
                    "groups_with_winners": len(self.snapshot.groups),
                    "groups_without_winners": self.snapshot.skipped_groups_without_winners,
                },
                "scene_metadata_status": "legacy_not_available",
                "eligible": len(self.snapshot.groups),
                "replacement_capacity": {"local": 0, "global": 0},
                "replacement_used": {"local": 0, "global": 0},
                "exhaustion": {
                    "local_shortfall": max(0, self.config.target_groups - len(groups)),
                    "global_shortfall": 0,
                },
                "terminal": sum(row.get("stage") == "import" for row in terminal),
            },
            "presets": {
                "inventory": {"local": {"xmp": len(candidates)}, "global": {}},
                "usage": {
                    "format": dict(sorted(formats.items())),
                    "major": {},
                    "minor": dict(sorted(modes.items())),
                    "selector": {"local": None, "global": None},
                    "candidate_failure_reasons": {},
                },
            },
            "sam3_relabel": {
                "expected": 0, "completed": 0, "pending": 0,
                "attempt_events": 0, "terminal": 0,
            },
            "annotation": {
                "pending": len(self.store.pending_annotation_tasks()),
                "terminal_failures": sum(
                    row.get("stage") == "annotation" for row in terminal
                ),
                "backends": self._annotation_counts(),
            },
            "failures": {
                "events": len(self.store.failures),
                "terminal": len(terminal),
                "by_code": dict(sorted(by_code.items())),
            },
            "legacy_import": {
                "protocol": self.config.legacy_import.protocol,
                "axis_fix_status": self.config.legacy_import.axis_fix_status,
                "input_root": str(self.config.legacy_import.input_root),
                "input_artifacts": self.snapshot.input_artifacts,
                "selection": "legacy winner set from unannotated sft.jsonl",
                "old_text_imported": False,
                "assets": "hardlink-or-copy into output assets; source paths remain external",
                "canonical_mask_contract": False,
                "viewer_projection_enabled": self.config.legacy_import.project_to_viewer,
            },
        }

    def _write_phase(self, phase: str) -> None:
        self.store.write_manifest(self._manifest(phase))
        self.store.checkpoint()

    def _import_groups(self) -> None:
        for index, group in enumerate(self.snapshot.groups, start=1):
            group_id = legacy_group_id(self.config, group)
            if group_id in self.store.groups:
                continue
            try:
                record = convert_legacy_group(
                    self.config,
                    group,
                    self.store,
                    timestamp=_timestamp(self.dependencies.now()),
                )
                self.store.append_group(record)
            except Exception as exc:  # noqa: BLE001 - preserve one structured import failure
                self._failure(
                    group,
                    code="legacy_group_import_failed",
                    message=f"{type(exc).__name__}: {exc}",
                )
            if index % 32 == 0:
                self.store.checkpoint()
        self.store.checkpoint()

    def execute(self) -> dict[str, Any]:
        self._write_phase("preflight")
        self._write_phase("import")
        self._import_groups()

        self._write_phase("annotation")
        annotation = self.dependencies.annotator_factory(self.config, self.store).drain()
        if annotation.get("pending") or self.store.pending_annotation_tasks():
            self._write_phase("annotation")
            raise PipelineError("legacy annotation queue remains unresolved")

        if len(self.store.groups) != self.config.target_groups \
                or len(self.store.sft) != self.snapshot.sft_row_count:
            self._failure(
                None,
                code="legacy_migration_target_shortfall",
                message=(
                    f"expected groups={self.config.target_groups}, sft={self.snapshot.sft_row_count}; "
                    f"completed groups={len(self.store.groups)}, sft={len(self.store.sft)}"
                ),
            )

        self._write_phase("projection")
        terminal = any(row.get("terminal") for row in self.store.failures)
        final_status = "complete_with_failures" if terminal else "complete"
        final_manifest = self._manifest(final_status, final_status)
        final_manifest["ended_at"] = _timestamp(self.dependencies.now())
        if self.config.legacy_import.project_to_viewer:
            projection = self.dependencies.projector(self.store, final_manifest, self.config)
        else:
            projection = ProjectionResult(True, {})
        final_manifest["projection"] = dataclasses.asdict(projection)
        self.store.checkpoint()
        final_manifest["artifacts"] = {
            path.name: file_digest(path)
            for path in (self.store.groups_path, self.store.sft_path, self.store.failures_path)
            if path.exists()
        }
        _add_manifest_self_artifact(final_manifest)
        self.store.write_manifest(final_manifest)
        self.store.checkpoint()
        return final_manifest


def run(
    config: DatabuildConfig,
    *,
    dependencies: PipelineDependencies | None = None,
) -> dict[str, Any]:
    """Preflight fully, then run or resume one canonical build."""
    dependencies = dependencies or default_dependencies()
    # A reboot empties the tmpfs root: restore the mirrored ledgers first and the
    # ordinary resume path then applies unchanged.  A live root is never touched.
    if dependencies.mirror_root is not None \
            and not (config.output_root / "manifest.json").is_file():
        restore_mirror(
            Path(dependencies.mirror_root) / config.build_id, config.output_root
        )
    existing = _load_existing_manifest(config.output_root / "manifest.json")
    if existing is not None:
        if existing.get("build_id") != config.build_id:
            raise StateError("output_root belongs to a different build_id")
        old_config = existing.get("effective_config")
        if old_config is not None:
            differences = _resume_config_differences(
                old_config, config.sanitized_dict()
            )
            if differences:
                raise StateError(
                    "resume config differs from the durable manifest: "
                    + ", ".join(differences)
                )

    # No authoritative artifact is opened until every startup dependency passes.
    dependencies.sdk_preflight()
    if config.legacy_import is not None:
        snapshot = load_legacy_snapshot(config)
        with ArtifactStore(config.output_root, config.build_id) as store:
            pipeline = LegacyImportPipeline(
                config, dependencies, snapshot, store, existing
            )
            return pipeline.execute()

    catalog = dependencies.catalog_loader(config)
    inventory = dependencies.inventory_loader(config)
    renderer = dependencies.renderer_factory(config)
    renderer.bind_catalog(catalog)
    renderer.assert_ready()
    scorer = _load_scorer(config, dependencies, inventory)

    with ArtifactStore(config.output_root, config.build_id) as store:
        pipeline = CanonicalPipeline(
            config, dependencies, inventory, catalog, renderer, scorer, store, existing
        )
        # CanonicalPipeline owns the heavy resources from here. Keeping these
        # aliases alive would defeat its release before the annotation phase.
        del renderer, scorer
        return pipeline.execute()


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="python -m construct.agent",
        description="Canonical GPU-only SFT databuild",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)
    run_parser = subparsers.add_parser("run")
    run_parser.add_argument("--config", required=True)
    args = parser.parse_args(argv)

    config: DatabuildConfig | None = None
    try:
        config = load_config(args.config)
        manifest = run(config)
    except Exception as exc:  # noqa: BLE001 - CLI boundary must redact every failure
        secrets = config.secrets if config is not None else ()
        print(
            "canonical databuild failed: "
            + redact_text(f"{type(exc).__name__}: {exc}", secrets),
            file=sys.stderr,
        )
        return 1
    print(json.dumps({
        "build_id": manifest["build_id"],
        "status": manifest["status"],
        "groups": manifest["completed"]["groups"],
        "sft": manifest["completed"]["sft"],
        "manifest": str(config.output_root / "manifest.json"),
    }, ensure_ascii=False, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
