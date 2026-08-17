"""Single TOML-driven canonical databuild orchestrator."""
from __future__ import annotations

import argparse
import dataclasses
import gc
import hashlib
import json
import os
import queue
import shutil
import sys
import threading
from collections import deque
from concurrent.futures import Future, ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Collection, Iterable, Mapping, Protocol

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
    DEFAULT_MAX_SOURCE_USES,
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
from .state import (
    ASSETS_LOST_CODE,
    ArtifactStore,
    StateError,
    file_digest,
    stable_id,
    write_json_atomic,
)
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
# 4, not 16, since the production layout runs two builds at once (one card each)
# on the same 24 GiB mount and this counter is per-process.  The watermark is not
# the peak: ``_land_groups`` stages winners under ``.land`` and, while the group
# candidates are hardlinks, each winner's ``I_in`` is a fresh copy of the source
# image, so a checkpoint adds roughly 0.6x the watermark again in real bytes.
# Measured on the first production checkpoint at 8 GiB: 7.8 GiB of assets became
# ~13 GiB at the peak, and two builds in lockstep took a 24 GiB mount to 716 MiB
# free.
#
# 8, not 4: a local-only build carries an unreclaimable staged base of ~5.2 GiB
# (masks/C_GT/in-flight winners that landing cannot release), so a 4 GiB mark sat
# permanently above threshold and the land loop thrashed — prod-l1 measured 183
# checkpoints per ~50 min and rendering throughput collapsed 3,170 -> ~250
# groups/h.  The mark must clear the unreclaimable base with room for one real
# batch; 8 GiB restores batching (peak ~13 GiB per build) and the tmpfs monitor
# alarms below 3 GiB free if two builds ever land in lockstep.
#
# The level it is compared against counts *only what landing can reclaim* — the
# staged assets.  Folding the prefetch buffer in (which ``_asset_directories``
# still does, deliberately, for the orphan sweep's benefit) made the mark a level
# with no lever: prod-l8 sat at 8.5 GiB of buffer with 0 bytes of assets, so both
# readers of this constant fired forever — checkpoints every group, and a source
# window pinned at 1 because ``_fill_initial_mode`` drains its whole queue on the
# same comparison.  The buffer has its own ceiling (``PREFETCH_BUFFER_BYTES``).
LAND_WATERMARK_BYTES = 8 * 1024**3
# Minimum spacing between two *unforced* checkpoints.  The water mark alone is
# not a rate limit: it is a level, and once the level sits above the mark (on
# prod-l8 the 8.5 GiB prefetch buffer alone does that, because
# ``_asset_directories`` counts the buffer) every single call passes it and the
# build lands one group per batch and re-mirrors the whole ledger each time —
# prod-l8 measured 910 checkpoints of a 391 MB ``groups.jsonl`` over 391 GB of
# NFS writes, starving the prefetch it shares the link with.  Both conditions
# must hold, because either one alone still degenerates: time alone lets a fast
# stretch land single groups, count alone lets a slow stretch land every 25
# groups no matter how long that took.  ``force=True`` (phase boundaries) is
# never throttled, and the free-space valve below overrides both.
LAND_MIN_INTERVAL_SECONDS = 300.0
LAND_MIN_GROUPS = 25
# Throttle override: landing is also the only path that reclaims staged bytes,
# so it must not be rate limited into a full tmpfs.  Measured against the mount
# holding the output root, not against the staged counter, because that counter
# includes the rebuildable prefetch buffer and is exactly what mis-fires above.
LAND_FREE_BYTES_FLOOR = 4 * 1024**3
# Incremental mirror.  ``MIRROR_VERIFY_WINDOW`` is the tail compared between the
# source and its mirror before every append: the ledgers are append-only, so an
# equal tail at the committed offset is the evidence that the mirror really is a
# byte prefix of the source and the delta may be appended behind it.
MIRROR_STATE_NAME = ".mirror_state.json"
MIRROR_VERIFY_WINDOW = 64 * 1024
MIRROR_COPY_CHUNK = 8 * 1024**2
# Periodic scrub.  The window above is a *tail*: damage anywhere before it is
# neither detected nor ever repaired, whereas the whole-file copy this replaced
# healed the mirror at every checkpoint.  Measured by the review (N-M1): flip
# byte 100 of a 320 KB mirror and five consecutive checkpoints report no event,
# the mirror stays diverged forever, and ``restore_mirror`` faithfully restores
# the bad byte into the ledger — silently wrong data if it lands inside a value,
# a mid-file ``StateError`` if it lands on a structural character.  So every
# K-th checkpoint copies the ledgers whole regardless of what the tail says.
# K=25 puts one 400 MB copy every 625 groups ≈ 0.64 MB/group, against the 405
# MB/group the incremental mirror replaced — 1/630 of the old cost, and the
# upper bound on how long a mid-file corruption can survive is now 25
# checkpoints instead of forever.
MIRROR_SCRUB_EVERY = 25
# Sources per prefetch buffer.  ponytail: a module constant for the same reason
# as the water mark — one chunk is ~1 GiB of a 24 GiB tmpfs, the buffer is
# rebuildable, and the only requirement is that a chunk take long enough to
# render that the next one finishes reading behind it.
PREFETCH_CHUNK = 256
# Ceiling on the buffer, which is *not* the water mark: landing cannot reclaim a
# single byte of it, so the two budgets are separate or the landing gate reads a
# level it has no lever over (prod-l8: 8.5 GiB of buffer permanently above an
# 8 GiB mark, source window pinned at 1).
#
# The buffer needs an explicit ceiling because nothing else bounds it.  A copy is
# dropped by ``_discard_prefetched`` when its source commits or goes terminal,
# but a source parked on the SAM3 queue keeps its copy so the drain can re-render
# it locally — and those never come back inside the rendering phase.  prod-l8
# measured 5,003 queued sources against 4,984 buffered files / 8.7 GiB: the
# buffer's real bound is "the whole local pool", i.e. unbounded for our purposes.
# ``_SourcePrefetch.close`` (rmtree at the end of rendering) is the only other
# reclaim and comes far too late.
#
# 3 GiB.  The floor is the working set that must never be evicted — the chunk
# being rendered plus the one read behind it, 512 files, measured on prod-l8 at
# 0.90 GiB (1.79 MB mean) and 2.38 GiB if every one of them sat at the p95 of
# 4.77 MB.  The ceiling is the 24 GiB tmpfs it shares with the staged assets (8
# GiB mark, ~13 GiB while a checkpoint stages winners) and the ledgers, which at
# the 400k target reach 7.4 GiB: 3 + 13 + 7.4 = 23.4 GiB.  Anything past ~1,700
# copies is parked SAM3 sources the drain may or may not reach, so the band
# between the two is theirs.  Evicting is never wrong — a miss is exactly the
# archive read the buffer existed to avoid — so this trades hit rate, never
# correctness.  It is a post-delivery trim rather than admission control, so one
# chunk may transiently overshoot before it is enforced.
PREFETCH_BUFFER_BYTES = 3 * 1024**3
# NFS roots.  Only ``default_dependencies`` wires them in, so any caller that
# builds ``PipelineDependencies`` by hand (every test) lands nothing, mirrors
# nothing and never touches NFS.
ARCHIVE_ROOT = Path("/mnt/nfs/bc/data/datasets")
MIRROR_ROOT = Path("/mnt/nfs/bc/data/builds")
# Stop-fill marker.  A file (or directory — the content is never read) at
# ``<output_root>/STOP_FILL`` retires the *rendering* half of a build while
# leaving the rest of the lifecycle exactly as it is: the annotation backlog is
# still drained, the shortfall is still journalled, and the projection still
# runs.  It exists because L8 has to stop at the groups it has and give both
# cards back to training, with ~50k winners still unannotated.
#
# **A marker file rather than a config key, deliberately.**  ``run`` compares the
# resumed manifest's ``effective_config`` against the current one path by path
# and refuses any difference (``_resume_config_differences``); a new key would
# make every live build unresumable the moment it was set, which is the exact
# opposite of what "stop rendering now" needs.  The marker is also reversible
# without touching a durable artifact: delete it, restart, and the build fills
# again, because nothing about it is written into the manifest's config.
STOP_FILL_MARKER = "STOP_FILL"


def _stop_fill_marker(output_root: Path) -> Path:
    return Path(output_root) / STOP_FILL_MARKER


def _stop_fill_requested(output_root: Path) -> bool:
    """Is the build being told to stop opening new render work?

    ``exists`` rather than ``is_file`` so ``touch``, ``echo >`` and ``mkdir`` are
    all valid ways to set it — an operator reaching for this is stopping a
    production build, and the answer must not depend on which one they typed.
    """
    return _stop_fill_marker(output_root).exists()


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
    pixels behind it, so an existing file is only the right answer while every
    re-plan of that slot produces the same pixels.  Two paths re-plan a source,
    and each is safe for its own reason:

    * **SAM3 relabel** only ever runs after ``build_mask_plan`` raised — that is,
      before ``_start_cgt_writes`` submitted anything for that source, so there
      is no file to go stale.
    * **Source reuse** (``[sources] max_source_uses`` above 1) re-plans a source
      that has already rendered, once per pass.  It is safe because the mask
      seeds in ``canonical_masks`` are derived from ``source_id`` alone and never
      from the pass index, so pass *k* rasterises the identical seven masks under
      the identical ``mask_id``s.  A pass whose files were already landed and
      unlinked simply re-encodes them.

    **Giving the mask plan any per-pass entropy breaks both claims at once** and
    would require invalidating (deleting) the affected C_GT files first, or this
    returns pixels that belong to another plan.
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
    def drain(
        self,
        *,
        max_workers: int | None = None,
        only: Collection[str] | None = None,
    ) -> dict[str, int]: ...


class _AnnotationDriver:
    """One background thread that annotates finished render passes.

    The renderer owns the GPUs and the annotator owns a relay socket, so making
    them take turns wasted whichever resource was idle: a build spent its whole
    render phase with the relay untouched and then sat on two idle cards for the
    length of the annotation phase.  This runs the annotation queue beside the
    render loop instead — one thread, one batch at a time, in the order the
    render passes finished.

    **Batches, not tasks.**  ``submit`` takes the whole of a finished pass, which
    is what keeps the relay's own batching intact (a task-at-a-time feed would
    hand ``ResponsesAnnotator.drain`` a pool of one and lose the concurrency the
    endpoint is configured for) and what makes the safety argument simple: the
    caller has already landed and re-indexed everything in the batch, so nothing
    in flight here can have its bytes moved by a later land checkpoint.

    **One thread, strictly serial.**  Two drains running at once could both pick
    up the same task and write two different SFT rows under one ``sft_id``, which
    the store rejects outright — after paying the relay twice.  The queue is
    therefore drained by a single thread and the closing drain is only started
    after this one has been joined.

    **Errors do not disappear.**  ``drain`` handles transport failure itself, so
    anything that escapes it is structural (a store conflict, an interrupt).  The
    first one is kept and re-raised on the *main* thread at the next ``submit``
    or at ``close``, which fails the build where a failure can still be seen,
    rather than leaving a dead thread and a build that renders on regardless.
    """

    def __init__(self, drainer: QueueDrainer) -> None:
        self._drainer = drainer
        self._queue: "queue.Queue[frozenset[str] | None]" = queue.Queue()
        self._lock = threading.Lock()
        self._inflight = 0
        self._batches = 0
        self._error: BaseException | None = None
        self._closed = False
        self._thread = threading.Thread(
            target=self._loop, name="databuild-annotate", daemon=False
        )
        self._thread.start()

    @property
    def inflight(self) -> int:
        """Tasks handed over and not yet resolved; the manifest's pipeline gauge."""
        with self._lock:
            return self._inflight

    @property
    def batches(self) -> int:
        with self._lock:
            return self._batches

    @property
    def error(self) -> BaseException | None:
        with self._lock:
            return self._error

    def submit(self, batch: frozenset[str]) -> None:
        error = self.error
        if error is not None:
            raise error
        if self._closed:
            raise PipelineError("annotation driver is closed")
        with self._lock:
            self._inflight += len(batch)
            self._batches += 1
        self._queue.put(batch)

    def close(self) -> None:
        """Stop accepting work and wait for the thread; safe to call twice."""
        if not self._closed:
            self._closed = True
            self._queue.put(None)
        self._thread.join()

    def _loop(self) -> None:
        while True:
            batch = self._queue.get()
            if batch is None:
                return
            try:
                if self.error is None:
                    self._drainer.drain(only=batch)
            except BaseException as exc:  # surfaced on the main thread; see class doc
                with self._lock:
                    if self._error is None:
                        self._error = exc
            finally:
                with self._lock:
                    self._inflight -= len(batch)


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


def _copy_stream(reader: Any, writer: Any, *, limit: int | None = None) -> int:
    """Copy up to ``limit`` bytes between two open handles; return what was written."""
    written = 0
    while limit is None or written < limit:
        want = MIRROR_COPY_CHUNK if limit is None else min(MIRROR_COPY_CHUNK, limit - written)
        chunk = reader.read(want)
        if not chunk:
            break
        writer.write(chunk)
        written += len(chunk)
    return written


def _atomic_copy(source: Path, target: Path, *, limit: int | None = None) -> int:
    """Replace one file in place: same-directory temp, fsync, rename.

    Returns the byte count actually copied, which is what the caller must record
    as the mirrored prefix: an append-only source can grow during the read, so
    the size it had before the copy is not a fact about the copy.
    """
    target.parent.mkdir(parents=True, exist_ok=True)
    tmp = target.with_name(target.name + ".tmp")
    with Path(source).open("rb") as reader, tmp.open("wb") as writer:
        written = _copy_stream(reader, writer, limit=limit)
        writer.flush()
        os.fsync(writer.fileno())
    os.replace(tmp, target)
    return written


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


def _read_mirror_state(mirror_dir: Path) -> tuple[dict[str, dict[str, Any]], int | None]:
    """The per-artifact ``(offset, tail digest)`` this mirror last committed,
    and the checkpoint sequence number that drives the scrub cadence.

    Unreadable, foreign or malformed state is *not* an error: it only costs one
    whole-file copy, whereas trusting it would be the one way to append behind
    an offset nobody verified.  The sequence number is ``None`` in exactly that
    case, which the caller reads as "scrub now" — a mirror whose position in the
    cadence is unknown is also a mirror nobody has verified past its tail.
    """
    try:
        payload = json.loads((mirror_dir / MIRROR_STATE_NAME).read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {}, None
    if not isinstance(payload, dict) or payload.get("version") != 1:
        return {}, None
    files = payload.get("files")
    if not isinstance(files, dict):
        return {}, None
    # Same ``bool``-is-an-``int`` trap as the offsets below, same treatment.
    seq = payload.get("checkpoints")
    if not isinstance(seq, int) or isinstance(seq, bool) or seq < 0:
        seq = None
    return {
        str(name): entry for name, entry in files.items() if isinstance(entry, dict)
    }, seq


def _mirror_state_entry(source: Path, size: int) -> dict[str, Any]:
    """Describe a mirrored prefix by its length and the digest of its tail.

    The tail is read from the *source*, which is the same bytes by construction
    (the mirror is a verified prefix) and lives on the tmpfs rather than behind
    NFS.
    """
    window = min(MIRROR_VERIFY_WINDOW, size)
    tail = b""
    if window:
        with source.open("rb") as handle:
            handle.seek(size - window)
            tail = handle.read(window)
    return {
        "bytes": size,
        "tail_bytes": window,
        "tail_sha256": hashlib.sha256(tail).hexdigest(),
    }


def _mirror_ledger(
    source: Path, target: Path, recorded: Mapping[str, Any] | None, *, scrub: bool = False
) -> tuple[dict[str, Any], dict[str, Any] | None, int]:
    """Extend one append-only ledger's mirror by its new suffix.

    Returns the new state entry, an event if this was not a plain append, and
    the bytes physically pushed to the mirror — which is the number worth
    watching, and is *not* derivable from the offsets when the previous state is
    missing (an unrecorded mirror can still be extended by its delta).

    The invariant every branch here restores is a single sentence: **the mirror
    is a byte-exact prefix of the source, and the recorded offset is its
    length**.  That is what makes ``restore_mirror`` produce a ledger the
    ordinary resume accepts, and it is strictly what the old whole-file copy
    also guaranteed — a copy of a file that is still being appended to is a
    prefix, not a snapshot of the end.

    Four ways the invariant can be found broken, each of which falls back to the
    whole-file copy rather than appending behind an unverified offset:

    * the source is *shorter* than what we mirrored (a resumed build writing a
      different file into the same name, or a truncated source);
    * the mirror is shorter than the recorded offset (truncated by something
      else, or an offset that never belonged to this file);
    * the tails disagree at the recorded offset (the prefix is not this
      source's), including the recorded digest disagreeing with the mirror;
    * the mirror is longer than its own recorded length *and* the surplus cannot
      be explained — see below, where it can.

    The one case that is repaired instead of re-copied is a mirror longer than
    the recorded offset: an append that was interrupted between the write and
    the state file leaves exactly that, and the surplus is by definition bytes
    nobody has claimed are durable, so it is truncated away and re-appended.

    ``scrub`` is the fifth way, and the only one that is not a symptom: the
    caller's cadence (``MIRROR_SCRUB_EVERY``) asks for the whole-file copy so
    that damage the tail window cannot see is healed anyway.  It is checked last
    so that a checkpoint which is *both* a scrub and an anomaly still journals
    the anomaly — the reason column is what tells the two apart afterwards.
    """
    source_size = source.stat().st_size
    target_size = target.stat().st_size if target.is_file() else -1
    recorded_bytes = recorded.get("bytes") if recorded is not None else None
    # ``bool`` is an ``int``: a state file carrying ``true`` would otherwise read
    # as the offset 1, and a one-byte window is a tail comparison that passes on
    # almost anything.  Anything that is not a plain non-negative integer costs a
    # whole-file copy instead.
    usable = isinstance(recorded_bytes, int) and not isinstance(recorded_bytes, bool) \
        and recorded_bytes >= 0
    committed = int(recorded_bytes) if usable else target_size  # type: ignore[arg-type]
    reason: str | None = None
    if target_size < 0:
        reason = "mirror_missing"
    elif recorded is not None and not usable:
        reason = "mirror_state_unusable"
    elif committed > source_size:
        reason = "source_shorter_than_mirror"
    elif committed > target_size:
        reason = "mirror_truncated"
    if reason is None and committed > 0:
        window = committed
        if recorded is not None and isinstance(recorded.get("tail_bytes"), int):
            window = min(int(recorded["tail_bytes"]), committed)
        window = max(1, min(window or MIRROR_VERIFY_WINDOW, MIRROR_VERIFY_WINDOW, committed))
        with target.open("rb") as handle:
            handle.seek(committed - window)
            mirrored_tail = handle.read(window)
        with source.open("rb") as handle:
            handle.seek(committed - window)
            source_tail = handle.read(window)
        if mirrored_tail != source_tail:
            reason = "prefix_diverged"
        elif recorded is not None and recorded.get("tail_sha256") not in (
            None, hashlib.sha256(mirrored_tail).hexdigest()
        ):
            reason = "mirror_tail_changed"
    if reason is None and scrub:
        reason = "scrub"

    event = {
        "artifact": target.name,
        "reason": reason,
        "action": "full_copy",
        # A mirror that does not exist yet *and* was never recorded is the first
        # checkpoint of a build, not an anomaly: copying it is the only thing
        # that could have happened, so it is accounted for but not journalled.
        "journal": not (reason == "mirror_missing" and recorded is None),
        "committed": committed,
        "mirror_bytes": target_size,
        "source_bytes": source_size,
    }
    if reason is not None:
        written = _atomic_copy(source, target)
        event["copied"] = written
        return _mirror_state_entry(source, written), event, written

    rolled_back = target_size > committed
    if rolled_back:
        with target.open("r+b") as handle:
            handle.truncate(committed)
            handle.flush()
            os.fsync(handle.fileno())
    written = 0
    if source_size > committed:
        with source.open("rb") as reader, target.open("r+b") as writer:
            reader.seek(committed)
            writer.seek(committed)
            written = _copy_stream(reader, writer)
            writer.flush()
            os.fsync(writer.fileno())
    landed = target.stat().st_size
    if landed != committed + written:
        # The mirror is not the length its own append says it is: nothing about
        # the offset is trustworthy any more, so stop reasoning and re-copy.
        event["reason"] = "mirror_size_after_append"
        event["mirror_bytes"] = landed
        copied = _atomic_copy(source, target)
        event["copied"] = copied
        return _mirror_state_entry(source, copied), event, written + copied
    state = _mirror_state_entry(source, landed)
    if rolled_back:
        event["reason"] = "interrupted_append"
        event["action"] = "rollback"
        event["appended"] = written
        return state, event, written
    return state, None, written


@dataclass(frozen=True, slots=True)
class MirrorReport:
    names: tuple[str, ...]
    # Bytes actually pushed to the mirror this pass, split by how they got
    # there.  ``copied_bytes`` staying near the ledger size checkpoint after
    # checkpoint is the signature of an incremental mirror that is not.
    appended_bytes: int
    copied_bytes: int
    # One entry per artifact that could not simply be extended.  Empty is the
    # healthy case; ``_land_checkpoint`` journals whatever is in here.
    events: tuple[dict[str, Any], ...]


def mirror_artifacts(output_root: Path, mirror_dir: Path) -> MirrorReport:
    """Carry the authoritative ledgers off the tmpfs so a reboot cannot take them.

    JSONL ledgers are append-only and grow without bound (391 MB on prod-l8 at
    a third of target), so they are extended by their new suffix rather than
    re-copied whole; ``manifest.json`` is rewritten in place and stays a normal
    atomic copy.  The cost of a checkpoint is therefore the bytes rendered since
    the last one, not the size of the build so far.

    Every ``MIRROR_SCRUB_EVERY``-th checkpoint the ledgers are copied whole
    anyway: appending behind a 64 KiB tail check leaves damage further back both
    undetected and unrepaired, so the cadence bounds how long such damage can
    survive.  The counter is the ``checkpoints`` sequence number in the state
    file, which is where a restart picks it back up; a state file that is
    missing or unusable has no position in the cadence and scrubs.
    """
    output_root, mirror_dir = Path(output_root), Path(mirror_dir)
    names = _artifact_names(output_root)
    mirror_dir.mkdir(parents=True, exist_ok=True)
    state, checkpoints = _read_mirror_state(mirror_dir)
    sequence = (checkpoints or 0) + 1
    scrub = checkpoints is None or sequence % MIRROR_SCRUB_EVERY == 0
    events: list[dict[str, Any]] = []
    appended = copied = 0
    for name in names:
        source, target = output_root / name, mirror_dir / name
        if not name.endswith(".jsonl"):
            copied += _atomic_copy(source, target)
            continue
        entry, event, written = _mirror_ledger(source, target, state.get(name), scrub=scrub)
        if event is not None and event["action"] == "full_copy":
            copied += written
        else:
            appended += written
        state[name] = entry
        if event is not None and event["journal"]:
            events.append(event)
    # Written last: a state file naming an offset the mirror has not reached
    # would send the next checkpoint appending into a hole.  The reverse (a
    # committed append the state file has not caught up with) is the repairable
    # ``interrupted_append`` case above.
    write_json_atomic(
        mirror_dir / MIRROR_STATE_NAME,
        {"version": 1, "checkpoints": sequence, "files": state},
    )
    return MirrorReport(
        names=tuple(names),
        appended_bytes=appended,
        copied_bytes=copied,
        events=tuple(events),
    )


def _whole_records_limit(path: Path) -> int | None:
    """Length of ``path`` up to its last newline, or ``None`` if it ends on one.

    An append to NFS is not atomic, so a mirror can end mid-record.  ``scan_jsonl``
    already tolerates a torn *unparseable* tail, but not the one cut that stays
    parseable: a record whose trailing newline is missing reads back as a whole
    record, and the journal's next append then glues the following record onto it
    — a corruption that only surfaces one resume later, in the middle of the file
    where nothing tolerates it.  Restoring on a record boundary removes the case.
    """
    size = path.stat().st_size
    if size == 0:
        return None
    with path.open("rb") as handle:
        handle.seek(size - 1)
        if handle.read(1) == b"\n":
            return None
        end = size
        while end > 0:
            start = max(0, end - MIRROR_COPY_CHUNK)
            handle.seek(start)
            block = handle.read(end - start)
            index = block.rfind(b"\n")
            if index >= 0:
                return start + index + 1
            end = start
    return 0


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
        source = mirror_dir / name
        limit = _whole_records_limit(source) if name.endswith(".jsonl") else None
        _atomic_copy(source, Path(output_root) / name, limit=limit)
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
    """Hand the allocator's spare blocks back so a co-resident model can have them.

    Nothing in here may *create* a CUDA context.  A stop-fill run loads no
    renderer and no scorer precisely so that ``nvidia-smi`` shows the build
    nowhere, and a call that initialised the driver just to find it had no blocks
    to release would put the process back on the card it was told to vacate.
    ``torch.cuda.is_initialized()`` is a Python-side flag on the lazy-init state
    — it calls nothing in the driver — so a process that never loaded a model
    returns here having touched no GPU at all.
    """
    try:
        import torch

        if torch.cuda.is_initialized():
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
    # ``sources.max_source_uses`` at its default of 1 is the pre-key pipeline
    # exactly: ``allocate_sources`` takes the same branch it always took, the
    # pool is walked once, and every derived ID drops the use suffix, so a
    # manifest that predates the key describes the same run this default
    # produces.  Only the missing-key case is excused — a manifest that already
    # says 1 and a config that now says 3 is a real difference, and the entry
    # above deliberately does not hide it.
    ("sources", "max_source_uses"): DEFAULT_MAX_SOURCE_USES,
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


def _sam3_ready_verdict(maxima: tuple[int | None, int | None] | None) -> bool:
    """``_unconsumed_sam3_ready``'s test, read off a (ready, invalid) maximum.

    Term for term the same comparison: a source with no ready event is not ready
    (``ready is None``), and an absent invalid event floors the comparison at
    zero exactly like the original ``max(invalid or [0])``.
    """
    if maxima is None:
        return False
    ready, invalid = maxima
    return ready is not None and ready > (0 if invalid is None else invalid)


class _Sam3Ledger:
    """SAM3 attempt counts and ready/invalid maxima, folded off the journal once.

    Both readings used to be answered per source with a full scan of
    ``failures`` — ``_sam3_attempts`` counting, ``_unconsumed_sam3_ready``
    maximising — and both are asked of *every* pending source inside
    ``_drain_sam3_and_replacements``'s loop.  At L8 shape (5,003 queued sources
    against 17,656 journal rows) the four comprehensions there measured 84.2 s
    per ``while`` iteration, on the main loop, and the journal only grows as the
    drain works.  This folds the same two quantities in one pass.

    **Why a prebuilt index is not enough.**  The drain *appends* to the journal
    while it iterates: ``_record_sam3_attempt`` writes ``sam3_attempt`` /
    ``sam3_ready``, ``_try_ready_relabel`` writes ``sam3_ready_invalid``, and a
    verdict can flip inside one loop body — the batch loop records ``sam3_ready``
    at attempt *n* and, three lines later, ``sam3_ready_invalid`` at the same *n*,
    which takes the source from ready back to consumed.  An index built once per
    round would answer with the state before those rows and be silently wrong.

    So this is incremental rather than prebuilt: ``refresh`` consumes exactly the
    rows appended since the last call, and every read point in the drain calls it
    first.  Because ``ArtifactStore.failures`` is append-only — rows are added at
    the end by ``append_failure`` and never reordered, rewritten or removed — a
    cursor into it is stable, each row is folded exactly once, and after any
    ``refresh`` the counts equal the full-scan answer over the whole journal.
    The identity/shrink guard rebuilds from scratch should a caller ever hand
    over a different or truncated list.

    Not thread safe, and deliberately not shared: the drain keeps its own
    instance on the main thread, while ``_sam3_ready_maxima`` builds a throwaway
    one per manifest.  Concurrent appends by other threads are harmless — the
    slice below is taken against a length captured first, so a row that lands
    mid-refresh is simply picked up by the next one.
    """

    __slots__ = ("_rows", "_pos", "attempts", "maxima")

    def __init__(self) -> None:
        self._rows: list[dict[str, Any]] | None = None
        self._pos = 0
        self.attempts: dict[str, int] = {}
        self.maxima: dict[str, tuple[int | None, int | None]] = {}

    def refresh(self, failures: list[dict[str, Any]]) -> "_Sam3Ledger":
        if failures is not self._rows or len(failures) < self._pos:
            self._rows = failures
            self._pos = 0
            self.attempts = {}
            self.maxima = {}
        end = len(failures)
        if end == self._pos:
            return self
        for row in failures[self._pos:end]:
            event = row.get("event_type")
            if event != "sam3_attempt" and event != "sam3_ready" \
                    and event != "sam3_ready_invalid":
                continue
            source_id = row.get("source_id")
            if not source_id:
                # Unreachable from the callers this feeds — they only ask about
                # source ids that came off a ``SourceRecord`` — and dropped
                # rather than keyed so a blank id cannot collide with a real one.
                continue
            key = str(source_id)
            if event == "sam3_attempt":
                # ``_sam3_attempts`` also required the stage, and only this
                # counter did: the maxima below are keyed on event type alone.
                if row.get("stage") == "sam3_relabel":
                    self.attempts[key] = self.attempts.get(key, 0) + 1
                continue
            attempt = int(row.get("attempt") or 0)
            ready, invalid = self.maxima.get(key, (None, None))
            if event == "sam3_ready":
                ready = attempt if ready is None else max(ready, attempt)
            else:
                invalid = attempt if invalid is None else max(invalid, attempt)
            self.maxima[key] = (ready, invalid)
        self._pos = end
        return self

    def attempts_of(self, source_id: str) -> int:
        return self.attempts.get(source_id, 0)

    def unconsumed_ready(self, source_id: str) -> bool:
        return _sam3_ready_verdict(self.maxima.get(source_id))


class CanonicalPipeline:
    def __init__(
        self,
        config: DatabuildConfig,
        dependencies: PipelineDependencies,
        inventory: SourceInventoryResult,
        catalog: PresetCatalog,
        renderer: Renderer | None,
        scorer: Scorer | None,
        store: ArtifactStore,
        existing_manifest: Mapping[str, Any] | None,
        *,
        stop_fill: bool | None = None,
    ) -> None:
        self.config = config
        self.dependencies = dependencies
        self.inventory = inventory
        self.catalog = catalog
        # ``None`` from the start is the stop-fill build: ``run`` skipped both
        # factories rather than loading a model it would never call.  Mid-run the
        # same two fields go to ``None`` in ``_release_heavy_resources``, which is
        # why they were already optional.
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
        # Stop-fill.  ``run`` decides it once, before the GPU factories, and hands
        # the same answer down here: it must not be re-derived, or a marker
        # dropped in the seconds between the two would give a pipeline that
        # believes it may render and a process that never loaded a renderer.
        # Callers that build the pipeline directly (every test) may leave it
        # ``None`` and get the marker's current answer.
        self._stop_fill_latched = (
            _stop_fill_requested(config.output_root) if stop_fill is None else stop_fill
        )
        self._stop_fill_at_start = self._stop_fill_latched
        self._stop_fill_recorded = False
        # What this process actually put on a card.  Journalled into the manifest
        # because "the build gave the GPUs back" is a claim about the process, not
        # about the marker, and the two are only the same if nothing loaded.
        self._gpu_resources_loaded = renderer is not None or scorer is not None
        # The pipelined annotation pass.  One annotator instance is shared by the
        # background driver and the closing drain: it caches SDK clients and
        # rebuilds the relay pool's removal state from the journal, and a second
        # instance would repeat both for no gain.  Both are created on first use,
        # so a build that renders nothing never opens a relay client at all.
        self._annotator: QueueDrainer | None = None
        self._driver: _AnnotationDriver | None = None
        # Task IDs already handed to the driver.  A task stays pending until its
        # SFT row or its terminal event exists, so without this a second pass
        # boundary would re-submit whatever the first one has not finished yet —
        # and two drains over one task is the duplicate-billing case the driver's
        # serial queue exists to prevent.
        self._annotation_submitted: set[str] = set()
        # Archive batches this process has already taught the catalog about.  See
        # ``_refresh_catalog``: the full re-registration is O(everything this
        # build ever landed), which is fine once per build and quadratic once per
        # render pass.
        self._registered_datasets: set[str] = set()
        # Land cadence marks.  Seeded from the resumed ledger rather than zero,
        # so a restart owes the same 25 *new* groups as an uninterrupted run
        # instead of checkpointing on its first committed group.
        self._last_land_at = self.dependencies.now()
        self._last_land_groups = len(store.groups)
        # No archive root means nothing was ever landed there and nothing can be
        # prefetched from it, which is also what every hand-built dependency set
        # (that is, every test) gets.
        self._prefetch = (
            _SourcePrefetch(store.root / "prefetch", dependencies.catalog_db)
            if dependencies.archive_root is not None else None
        )
        # Every staged path is classified against this once, in three places that
        # used to spell the test out (or, in the water mark's case, not make it at
        # all).  A trailing separator so a sibling directory sharing the prefix
        # cannot be mistaken for the buffer.
        self._buffer_prefix = (
            str(self._prefetch.directory) + os.sep if self._prefetch is not None else None
        )
        self._buffer_evicted = 0
        self._recalibrate_staged()
        self.allocation = allocate_sources(
            inventory.eligible,
            build_id=config.build_id,
            seed=config.seed,
            target_groups=config.target_groups,
            mix=config.mix,
            max_source_uses=config.sources.max_source_uses,
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

    def _journal_once(self, *, task_id: str, **fields: Any) -> None:
        """Append a phase-level event unless this exact task already has one.

        ``ArtifactStore.append_failure`` deduplicates *byte-identical* rows only,
        and every row carries a timestamp, so re-recording an event on a restart
        is a hard ``conflicting durable failure event`` rather than the no-op it
        reads as.  Nothing needed this before because every other durable event
        is written from a path a resume does not re-walk: a terminal source is
        refused before it renders, a lost group is skipped via ``lost_group_ids``,
        a queued SAM3 source is already on ``_pending_sam3_ids``.  The two events
        ``_run_phases`` writes for itself are the first that a restart genuinely
        reaches a second time.

        Measured on pristine HEAD, with no stop-fill anywhere in it: a build that
        journals a target shortfall and is then restarted dies **here**, before
        it does any work at all.  The frozen clock in the test fixtures hid it,
        and no production build had ever reached ``_record_shortfalls`` twice —
        L8 never finished its rendering phase, so it never reached it once.
        Stop-fill is what makes that reachable, because a stopped build records
        its shortfall and then restarts to keep draining ~50k annotations.

        Callers fold whatever legitimately varies into ``task_id`` — the group
        count for a shortfall, the shape of the stop for a stop-fill event — so
        this asks "have we already said exactly this", not "have we said
        anything".  A restart that changed nothing is silent; one that did
        journals the new number instead of being refused for disagreeing with
        the old one.
        """
        if any(row.get("task_id") == task_id for row in self.store.failures):
            return
        self._failure(task_id=task_id, **fields)

    def _source_reuse_counts(self) -> dict[str, Any]:
        """How many groups each source produced, and how much they repeat.

        Counted off the same ledger the orchestrator skips on (every durable
        group, lost ones included), so the histogram is the cursor state rather
        than a second opinion about it: with the switch at its default every
        bucket is ``"1"``.

        The preset columns exist because the diversity of a reused source is
        **not** guaranteed by anything in the selector — it is an emergent
        property of the bank being much larger than ``8 x max_source_uses``, and
        the real bank is not large enough for it to hold outright.  The selector
        only ever guarantees that the eight candidates *within one group* are
        distinct (``GroupReservation.commit``); across a source's groups it
        merely prefers least-used presets, and once every preset in a major has
        been used equally often it will hand out one the source already has.
        These two numbers are what makes that visible instead of assumed:

        ``duplicate_source_preset_pairs``
            draws of a ``(source, preset)`` pair the source had already drawn.
            In global mode such a draw is a bit-identical ``I_tar`` under a new
            ``candidate_id``, which nothing downstream de-duplicates
            (``sft_pack``/``q3vl.data.scan`` both key on IDs, not content).
        ``max_pair_overlap``
            largest preset intersection between any two groups of one source,
            out of 8.  ``8`` would mean a whole group was re-drawn.
        """
        used: dict[str, int] = {}
        drawn: dict[str, list[list[str]]] = {}
        for row in self.store.groups.values():
            source_id = str(row["source_id"])
            used[source_id] = used.get(source_id, 0) + 1
            drawn.setdefault(source_id, []).append([
                str(candidate.get("preset_id") or "")
                for candidate in (row.get("candidates") or ())
            ])
        histogram: dict[int, int] = {}
        draws = duplicate_pairs = max_overlap = 0
        for source_id, count in used.items():
            histogram[count] = histogram.get(count, 0) + 1
            groups = drawn[source_id]
            draws += sum(len(item) for item in groups)
            if count < 2:
                continue
            seen: dict[str, int] = {}
            for item in groups:
                for preset_id in item:
                    seen[preset_id] = seen.get(preset_id, 0) + 1
            duplicate_pairs += sum(n - 1 for n in seen.values() if n > 1)
            # A whole group re-drawn is the ceiling; once seen, no later source
            # can raise the maximum, so the quadratic part stops there.
            if max_overlap < 8:
                sets = [set(item) for item in groups]
                for first in range(len(sets)):
                    for second in range(first + 1, len(sets)):
                        max_overlap = max(max_overlap, len(sets[first] & sets[second]))
        budget = self.config.sources.max_source_uses
        observed = max(used.values(), default=0)
        return {
            "max_source_uses": budget,
            "distinct_sources": len(used),
            "groups": sum(used.values()),
            "max_observed": observed,
            # The cursor invariant, stated where it can be grepped instead of
            # only asserted in a docstring: a source cannot outrun its budget.
            "budget_exceeded": bool(budget) and observed > budget,
            "uses_histogram": {str(key): histogram[key] for key in sorted(histogram)},
            "duplicate_source_preset_pairs": duplicate_pairs,
            "duplicate_source_preset_rate": (
                round(duplicate_pairs / draws, 6) if draws else 0.0
            ),
            "max_pair_overlap": max_overlap,
        }

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
        """One source's attempt count, by scan.  See ``_Sam3Ledger.attempts_of``.

        Kept as the reference definition the equivalence tests measure the
        ledger against; the drain reads the ledger instead, because asking this
        of every pending source is what made that loop quadratic.
        """
        return sum(
            1 for row in self.store.failures
            if row.get("source_id") == source_id
            and row.get("stage") == "sam3_relabel"
            and row.get("event_type") == "sam3_attempt"
        )

    def _unconsumed_sam3_ready(self, source_id: str) -> bool:
        """One source's verdict, by scan.  See ``_Sam3Ledger.unconsumed_ready``.

        Kept for the same reason as ``_sam3_attempts``: it is the oracle the
        ledger's equivalence tests compare against.
        """
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

    def _sam3_ready_maxima(self) -> dict[str, tuple[int | None, int | None]]:
        """Per source, the highest ``sam3_ready`` / ``sam3_ready_invalid`` attempt.

        The manifest asks ``_unconsumed_sam3_ready`` the same question of every
        queued source, and that helper answers each one with two full scans of
        ``failures``.  At L8 shape — 5.0k queued sources against 17.7k failure
        rows — the product is 1.7e8 dict reads *per land checkpoint*, on the
        main loop, with the render threads idling behind it.  This is the same
        bookkeeping in one pass over the journal.

        ``None`` means "no such event for this source", which is what keeps
        "ready at attempt 0" apart from "never ready"; the verdict itself lives
        in ``_sam3_ready_beats_invalid`` so both readings stay one expression.

        The fold is ``_Sam3Ledger``'s, over a throwaway instance: the manifest
        wants a whole-journal answer and holds no state between checkpoints, so
        it pays one pass here rather than sharing the drain's cursor across
        threads.
        """
        return _Sam3Ledger().refresh(self.store.failures).maxima

    @staticmethod
    def _sam3_ready_beats_invalid(maxima: tuple[int | None, int | None] | None) -> bool:
        """``_unconsumed_sam3_ready``'s verdict, read off ``_sam3_ready_maxima``."""
        return _sam3_ready_verdict(maxima)

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
        # One journal pass for the whole set instead of two per queued source;
        # the verdict per source is unchanged.  See ``_sam3_ready_maxima``.
        sam3_ready_maxima = self._sam3_ready_maxima()
        sam3_completed_ids = {
            source_id for source_id in sam3_expected_ids
            if self._sam3_ready_beats_invalid(sam3_ready_maxima.get(source_id))
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
                    for row in self.store.sft_records()
                ),
                "groups_assets_lost": len(lost_groups),
            },
            "landing": self._landing_counts(),
            "prefetch": self._prefetch_counts(),
            # Why this build stopped where it did, and whether it was ever on a
            # card.  Reads the latch rather than calling ``_stop_fill``: writing a
            # manifest must not be the thing that appends the journal row.
            "stop_fill": {
                "requested": self._stop_fill_latched,
                # True is the no-GPU build; a marker that appeared mid-run leaves
                # this False and ``gpu_resources_loaded`` True, which is the
                # honest description of a process that did load the models.
                "at_start": self._stop_fill_at_start,
                "gpu_resources_loaded": self._gpu_resources_loaded,
                "marker": str(_stop_fill_marker(self.config.output_root)),
            },
            "sources": {
                "inventory": self.inventory.counts,
                "scene_metadata_status": self.inventory.scene_metadata_status,
                "eligible": len(self.inventory.eligible),
                "replacement_capacity": {
                    "local": max(0, len(self.allocation.local) - self.allocation.local_target),
                    "global": max(0, len(self.allocation.global_) - self.allocation.global_target),
                },
                "replacement_used": replacement_used,
                # Read the two keys above together with ``source_reuse`` below.
                # They still mean exactly what they always meant — sources held
                # back beyond a mode's initial quota, and groups that had to draw
                # on them — but with reuse on they stop being the whole story:
                # spare capacity now mostly lives in extra passes over the same
                # pool, and a pool smaller than its target reports capacity 0 and
                # used 0 while still meeting that target.
                "source_reuse": self._source_reuse_counts(),
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
                # Unchanged: the whole unresolved queue, whoever owns it.  A
                # manifest written mid-render now normally reports a non-zero
                # ``pending`` — that is the pipeline working, not a build with
                # annotation left undone, and ``inflight`` is what tells the two
                # apart.  The completion gate still reads ``pending`` after the
                # closing drain, where nothing is in flight and the two agree.
                "pending": len(self.store.pending_annotation_tasks()),
                "terminal_failures": len(annotation_failures),
                # Tasks handed to the background driver and not yet resolved, and
                # how many whole render passes have been handed over.  Both are
                # 0 in a build that never reached a pass boundary.
                "inflight": self._driver.inflight if self._driver is not None else 0,
                "batches": self._driver.batches if self._driver is not None else 0,
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
            return {"enabled": False, "buffered": 0, "errors": 0, "bytes": 0, "evicted": 0}
        return {
            "enabled": True,
            "buffered": self._prefetch.buffered,
            # Non-zero means the round fell back to random archive reads, which
            # costs throughput and nothing else — worth seeing, never fatal.
            "errors": self._prefetch.errors,
            # The buffer's own budget, reported next to the water mark's so the
            # two are never read as one number again: ``bytes`` is what the
            # ceiling governs, ``evicted`` is how often it bit.
            "bytes": self._buffer_size(),
            "evicted": self._buffer_evicted,
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
        for row in self.store.sft_records():
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
            # The buffer shares the tmpfs, so its files are tracked here; it is
            # neither an asset (the orphan sweep must not reap it) nor reclaimable
            # by landing (the water mark must not count it), so ``_buffer_bytes``
            # carries it separately from ``_staged_bytes``.
            directories.append(self._prefetch.directory)
        return tuple(directories)

    def _is_buffer(self, path: str) -> bool:
        """Is this tracked file a rebuildable prefetch copy rather than an asset?"""
        return self._buffer_prefix is not None and path.startswith(self._buffer_prefix)

    def _recalibrate_staged(self) -> None:
        """Re-measure the staging tree; the water mark's only full scan.

        Scanning per source cost minutes over a build (tens of thousands of files
        × 22k sources), so the counter is maintained incrementally instead and
        this runs only where a scan is already being paid for: once at startup,
        so a resume inherits the assets a previous run left behind, and inside
        the orphan sweep, which walks the same directories anyway.

        Two totals come out of the one walk, split by directory rather than by
        re-testing each path: landing reclaims the assets and nothing else, so
        the two budgets can never be compared against the same threshold.
        """
        sizes: dict[str, int] = {}
        staged = buffered = 0
        for directory in self._asset_directories():
            if not directory.is_dir():
                continue
            is_buffer = self._is_buffer(str(directory) + os.sep)
            with os.scandir(directory) as scan:
                for entry in scan:
                    if entry.is_file():
                        size = entry.stat().st_size
                        sizes[entry.path] = size
                        if is_buffer:
                            buffered += size
                        else:
                            staged += size
        with self._staged_lock:
            self._staged = sizes
            self._staged_bytes = staged
            self._buffer_bytes = buffered

    def _account_asset(self, path: Path) -> None:
        """Fold one freshly written asset in, replacing any size it overwrote.

        A refilled slot rewrites the same candidate JPEG and every group attempt
        rewrites the same C_GT, so a plain addition would drift upward forever.
        Prefetched copies arrive here too (``_account_prefetched``) and are booked
        against the buffer budget instead of the landable one.
        """
        size = path.stat().st_size
        key = str(path)
        with self._staged_lock:
            delta = size - self._staged.get(key, 0)
            if self._is_buffer(key):
                self._buffer_bytes += delta
            else:
                self._staged_bytes += delta
            self._staged[key] = size

    def _staged_size(self) -> int:
        """Bytes a checkpoint can actually hand back to the tmpfs."""
        with self._staged_lock:
            return self._staged_bytes

    def _buffer_size(self) -> int:
        with self._staged_lock:
            return self._buffer_bytes

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

    def _refresh_catalog(self, *, incremental: bool = False) -> None:
        """Re-index this build's landed batches so their assets answer to staging paths.

        ``incremental`` names only the batches this process has not registered
        yet.  The pipelined build refreshes once per render pass rather than once
        per build, and the full list is every batch the build ever published: by
        the twelfth pass of an L8 that is re-reading the whole build's index to
        learn about the last pass's worth of it, which is the ~390 s full rebuild
        this call was written to avoid, once per pass.  A landed batch is
        immutable apart from the ``metadata.jsonl`` rewrite in
        ``_sync_annotation_status``, which runs after the last refresh, so
        skipping one already registered cannot lose anything.  The closing
        refresh before the annotation phase stays full, which is also what
        re-registers everything a previous run of a resumed build landed.
        """
        root = self.dependencies.archive_root
        if root is None or not Path(root).is_dir():
            return
        # A checkpoint re-registers this build's own batches — a handful of
        # groups in an archive of hundreds; the full rebuild re-read all 5.5 M
        # members for them (~390 s of silence before annotation).  Anything else
        # in the archive was indexed by whoever published it.
        names = self._landed_datasets(Path(root))
        if incremental:
            names = [name for name in names if name not in self._registered_datasets]
            if not names:
                return
        upsert_catalog(
            Path(root),
            self.dependencies.catalog_db or default_db(),
            names,
        )
        self._registered_datasets.update(names)
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
        for row in self.store.sft_records():
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

    def _annotation_drainer(self) -> QueueDrainer:
        if self._annotator is None:
            self._annotator = self.dependencies.annotator_factory(self.config, self.store)
        return self._annotator

    def _annotation_bytes_settled(self, task: Mapping[str, Any]) -> bool:
        """Is this task's ``I_tar`` past the point where landing can move it?

        The one thing the render loop does that annotation cannot survive is the
        unlink at the end of ``_land_groups``: a checkpoint publishes a group's
        candidates into the archive and then deletes the staged copies, and a
        task encoding those bytes at that moment finds no local file and — until
        the next catalog refresh — no archive entry either, which
        ``_encode_image`` reports as ``annotation_image_invalid``, a *terminal*
        code.  One badly timed checkpoint would abandon a whole batch of winners.

        The test is the exact converse of what landing acts on.  ``_land_groups``
        only ever unlinks assets it can still see (its ``pending`` list requires
        every file of the group to exist), so an ``after_path`` that is already
        gone is an ``after_path`` no future checkpoint will touch: the bytes live
        in the archive, which is append-only, and the catalog refresh below makes
        them addressable.  A file still on the tmpfs is the opposite case and its
        task waits for a later boundary, by which time the ordinary land cadence
        will have retired it (measured on prod-l8: 1,789 checkpoints over 57,234
        groups, so the unlanded tail at a pass boundary is bounded by the 8 GiB
        staging water mark rather than by the pass).

        A build with no archive root never lands and never unlinks, so nothing
        can move and every task qualifies from the moment it exists.
        """
        if self.dependencies.archive_root is None:
            return True
        after = (task.get("candidate") or {}).get("after_path")
        return bool(after) and not Path(str(after)).is_file()

    def _flush_annotation_batch(self) -> None:
        """Hand the render pass that just finished to the background annotator.

        Called on the pass boundary inside ``_fill_mode``, which is the only
        place in the build where the render loop is quiet: every source future
        has been committed, so the queue this measures is a whole number of
        passes and nothing is being appended while it is measured.

        Two steps, in this order, and the order is the safety argument:

        1. ``_refresh_catalog`` teaches the reverse map about every batch landed
           since the last boundary.  Until it runs, ``read_bytes`` on a landed
           candidate resolves to nothing at all.
        2. Only then is the batch handed over — and only tasks whose bytes have
           settled (see ``_annotation_bytes_settled``) and that no earlier
           boundary already owns.

        Whole passes rather than single tasks: the relay is configured for 16
        concurrent requests and ``drain`` sizes its pool from the queue it is
        given, so a task-at-a-time feed would quietly run the annotation at
        concurrency one.  The measurement comes first and returns early when
        there is nothing new, so a pass whose groups are all still staged costs
        one queue walk and no catalog copy.
        """
        batch = frozenset(
            str(task["task_id"])
            for task in self.store.pending_annotation_tasks()
            if str(task["task_id"]) not in self._annotation_submitted
            and self._annotation_bytes_settled(task)
        )
        if not batch:
            return
        self._refresh_catalog(incremental=True)
        if self._driver is None:
            self._driver = _AnnotationDriver(self._annotation_drainer())
        self._annotation_submitted |= batch
        self._driver.submit(batch)

    def _close_annotation_driver(self) -> BaseException | None:
        """Join the background annotator and report what killed it, if anything."""
        if self._driver is None:
            return None
        self._driver.close()
        return self._driver.error

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
        orphans = [
            path for path in self._staged
            if path not in referenced and not self._is_buffer(path)
        ]
        for path in orphans:
            os.unlink(path)
            self._staged_bytes -= self._staged.pop(path)
        return len(orphans)

    def _trim_prefetch_buffer(self, keep: Iterable[str] = ()) -> int:
        """Evict the coldest buffered copies until the buffer is back under budget.

        The buffer has no other bound: a copy is released when its source commits
        or goes terminal, but a source parked on the SAM3 queue keeps its copy for
        the drain, and prod-l8 accumulated 8.7 GiB of exactly those.  Landing
        cannot touch any of it, so the ceiling has to be enforced here.

        Eviction is by mtime, which ``tools.prefetch`` maintains as a use stamp:
        it writes a fresh copy at fetch time and ``os.utime``s one it finds still
        valid, so the coldest entries are the parked ones the renderer walked past
        chunks ago — and the chunk about to render is protected outright.  Dropping
        a copy is never an error: ``read_bytes`` falls back to the archive, which
        is the read the buffer existed to save.

        Called with no prefetch in flight (``_rotate_prefetch`` has just taken
        delivery and not yet submitted), so nothing is writing into the directory
        while this runs.
        """
        if self._prefetch is None:
            return 0
        over = self._buffer_size() - PREFETCH_BUFFER_BYTES
        if over <= 0:
            return 0
        protected = {str(self._prefetch.path_for(path)) for path in keep}
        with self._staged_lock:
            candidates = [
                path for path in self._staged
                if self._is_buffer(path) and path not in protected
            ]

        def stamp(path: str) -> float:
            try:
                return os.stat(path).st_mtime
            except OSError:
                # Already gone: evict it first, it costs nothing and the counter
                # still has to come down by whatever it was booked at.
                return 0.0

        evicted = 0
        for path in sorted(candidates, key=stamp):
            if over <= 0:
                break
            try:
                os.unlink(path)
            except FileNotFoundError:
                pass
            with self._staged_lock:
                size = self._staged.pop(path, 0)
                self._buffer_bytes -= size
            over -= size
            evicted += 1
        self._buffer_evicted += evicted
        return evicted

    def _tmpfs_under_pressure(self) -> bool:
        """Is the mount holding the output root close enough to full to land now?

        Deliberately a question about free space rather than about the staged
        level: the level answers "how much could landing reclaim", which says
        nothing about how much room is left once the buffer and the ledgers have
        taken their share — and the ledgers, on a 400k build, outgrow the mark.
        """
        try:
            return shutil.disk_usage(self.store.root).free < LAND_FREE_BYTES_FLOOR
        except OSError:
            # An unreadable mount is not evidence of pressure, and this is only a
            # valve: the ordinary cadence still lands.
            return False

    def _land_cadence_ready(self) -> bool:
        """Enough time *and* enough new groups since the last checkpoint.

        Both, because each alone degenerates into the storm this exists to stop:
        during a fast stretch the group count is reached in seconds, during a
        slow one the clock runs out with a single group to land.  The free-space
        valve overrides both — landing is the only thing that reclaims the
        tmpfs, so it must never be throttled into an out-of-space.
        """
        groups = len(self.store.groups) - self._last_land_groups
        elapsed = (self.dependencies.now() - self._last_land_at).total_seconds()
        if groups >= LAND_MIN_GROUPS and elapsed >= LAND_MIN_INTERVAL_SECONDS:
            return True
        return self._tmpfs_under_pressure()

    def _staging_full(self) -> bool:
        """Is there a batch worth landing — or so little room that it cannot wait?

        The mark is a level of *reclaimable* bytes, so it can now sit below
        threshold while the mount fills with things landing does not own (the
        buffer, and a 400k build's ~7.6 GiB of ledgers).  The valve therefore has
        to be reachable from below the mark as well, or it would only ever be
        consulted in the one case that no longer needs it.  It stays second, so a
        healthy build answers this from a counter and never stats the mount.
        """
        return self._staged_size() >= LAND_WATERMARK_BYTES or self._tmpfs_under_pressure()

    def _land_checkpoint(self, *, force: bool = False) -> dict[str, Any] | None:
        """Publish, mirror and reclaim — the whole durability step, in one place."""
        if self.dependencies.archive_root is None and self.dependencies.mirror_root is None:
            return None
        if not force and (
            not self._staging_full() or not self._land_cadence_ready()
        ):
            return None
        # Marked before the work, not after: a checkpoint that takes minutes has
        # still *started* now, and the next one is owed a full interval from
        # here rather than from whenever this one happened to finish.
        self._last_land_at = self.dependencies.now()
        self._last_land_groups = len(self.store.groups)
        result = None
        if self.dependencies.archive_root is not None:
            result = self._land_groups()
            self._clean_orphan_assets()
        if self.dependencies.mirror_root is not None:
            self.store.checkpoint()
            self.store.write_manifest(self._manifest(self._phase))
            self._record_mirror_events(mirror_artifacts(
                self.store.root, Path(self.dependencies.mirror_root) / self.config.build_id
            ))
        return result

    def _record_mirror_events(self, report: MirrorReport) -> None:
        """Journal every artifact the incremental mirror could not simply extend.

        A mirror that silently fell back to a whole-file copy on every
        checkpoint is indistinguishable, from the outside, from one that is
        working — same bytes on NFS, just the old cost back.  These rows are the
        difference.  Non-terminal and non-retryable: the fallback already
        repaired the mirror, the record exists so the repair is visible.
        """
        for event in report.events:
            payload = json.dumps(event, sort_keys=True)
            self._failure(
                event_type="mirror",
                stage="mirror",
                # The payload and the timestamp are both inside the task id, so
                # two fallbacks can never collide on one event id while carrying
                # different messages (which the store rejects outright).
                task_id=stable_id(
                    "mirror", self.config.build_id, payload,
                    _timestamp(self.dependencies.now()),
                ),
                error_code=f"mirror_{event['action']}",
                message=payload,
                retryable=False,
                terminal=False,
                durable=True,
            )

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
        use_index: int = 0,
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
            # The use index is appended only when a source is on its second or
            # later pass, so the first pass keeps the group IDs — and therefore
            # the candidate IDs and asset filenames derived from them — that every
            # build before source reuse wrote.
            group_id = stable_id(
                "group", self.config.build_id, source.source_id, mode, group_attempt,
                *(() if not use_index else (use_index,)),
            )
            reservation = None
            group_persisted = False
            try:
                reservation = selector.begin_group(
                    source.source_id, group_attempt, exclude_majors=excluded_majors,
                    use_index=use_index,
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
                    # Which pass over the source pool produced this group.  Only
                    # written when it is not the first, so a build without source
                    # reuse journals the record shape it always journalled and a
                    # reader can treat "absent" as 0.
                    **({} if not use_index else {"source_use_index": use_index}),
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
        use_index: int = 0,
        selector_turn: _SelectorTurn | None = None,
    ) -> _DeferredSourceResult:
        """Render one source off-thread without mutating the durable journals."""
        failures: list[_BufferedFailure] = []
        self._source_context.failures = failures
        try:
            try:
                result = self._render_source(
                    source, mode, use_index=use_index,
                    defer_commit=True, selector_turn=selector_turn,
                )
            except Exception as exc:  # flushed in source order before the error escapes
                self._failure(
                    event_type="source_worker",
                    stage="rendering",
                    task_id=stable_id(
                        "render-source", self.config.build_id, source.source_id, mode,
                        *(() if not use_index else (use_index,)),
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
        *,
        use_index: int = 0,
        used: Mapping[str, int] | None = None,
    ) -> list[str]:
        """The source images one chunk of the allocation still has to read.

        The done test has to be the same one ``_fill_initial_mode`` applies, or a
        reuse pass would prefetch nothing at all: every source already owns a
        group by then, and a membership test would call the whole pool finished
        while the renderer went on reading each image from the archive.  ``used``
        is that caller's per-pass snapshot; a chunk only ever looks at sources at
        or ahead of the current position, so the snapshot is as current for them
        as a fresh scan would be.
        """
        if limit <= 0:
            return []
        if used is None:
            used = self.store.completed_source_uses()
        return [
            str(source.source_path)
            for source in sources[start:start + PREFETCH_CHUNK]
            if source.source_id not in skip
            and used.get(source.source_id, 0) <= use_index
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
        *,
        use_index: int = 0,
        used: Mapping[str, int] | None = None,
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
            sources, start, skip, limit=min(PREFETCH_CHUNK, budget),
            use_index=use_index, used=used,
        )
        # Outstanding here is chunk k, or — at the start of a mode — whatever the
        # previous mode's last look-ahead read.
        self._account_prefetched()
        if start == 0:
            self._prefetch.submit(current)
            self._account_prefetched()
        # Between taking delivery and queueing the next chunk is the one moment
        # in a round with nothing writing into the buffer, so it is where the
        # ceiling is enforced; the chunk about to render is held back from it.
        self._trim_prefetch_buffer(current)
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
            use_index=use_index, used=used,
        ))

    def _discard_prefetched(self, source: SourceRecord) -> None:
        """Drop one source's buffered copy once it can no longer be read again."""
        if self._prefetch is None:
            return
        path = self._prefetch.path_for(source.source_path)
        path.unlink(missing_ok=True)
        with self._staged_lock:
            self._buffer_bytes -= self._staged.pop(str(path), 0)

    def _fill_mode(
        self,
        mode: str,
        sources: tuple[SourceRecord, ...],
        target: int,
        *,
        annotate_passes: bool = False,
    ) -> None:
        """Walk the mode's source pool until its target is met or it stops paying.

        One pass is the whole of the pre-reuse behaviour, so ``max_source_uses =
        1`` calls ``_fill_initial_mode`` exactly once, measures nothing around it
        and nothing downstream can tell the wrapper is there.  Above one, the
        extra passes are what turns a pool smaller than the target into a build
        that still reaches it: the pool is re-walked in its original
        scene-stratified order rather than a source being repeated in place, so
        consecutive groups keep coming from different images and the coverage
        selector keeps handing each pass a different preset set.

        **The pass counter is not a fresh 0 on every call.**  It cannot be: this
        method is entered again on resume, and again after ``_drain_sam3_and_
        replacements`` rescues a source, and in both cases the early passes are
        already spent.  Two things keep it honest:

        * the walk *starts* at the least-used source's count, so passes in which
          every single source would be skipped are never walked at all;
        * the "this pass produced nothing, stop" guard only fires when the pass
          also skipped nobody **on the cursor** — a pass that produced nothing
          because its groups already exist is not an exhausted pool, and stopping
          there is what silently threw away every remaining pass of a resumed
          build.

        ``annotate_passes`` makes each completed pass the trigger for a batch of
        annotation (see ``_flush_annotation_batch``).  It is off by default, and
        the caller that leaves it off is ``_drain_sam3_and_replacements``: that
        loop re-enters here once per rescued source, so its "passes" are a
        trickle of replacements rather than a lap of the pool, and each one would
        buy a forced landing and a catalog copy for a handful of groups.  Those
        groups are annotated by the closing drain instead.
        """
        # Before the two ledger scans below, which are O(groups) each and pure
        # waste for a mode that will not open a single pass.  This is also the
        # whole of "a stop-fill build does not enter the fill": ``_run_phases``
        # keeps calling both modes so that its phase sequence — and therefore
        # what a resume has to redo — is byte for byte the one every other build
        # writes, and each call retires here.
        if self._stop_fill():
            return
        budget = self.config.sources.max_source_uses
        # Passes below this one are provably empty: every source already owns a
        # group for them.  Skipping them is an optimisation only — the cursor
        # test inside ``_fill_initial_mode`` would skip the same sources one at a
        # time — but without it a resumed L8 build re-walks the whole pool once
        # per spent pass before it renders anything.
        used = self.store.completed_source_uses()
        # Terminal sources are excluded or the minimum is pinned at 0 forever:
        # they never render, so they never leave zero, and L7 finished with 898
        # of them (all ``sam3_relabel_failed``, i.e. ``build_mask_plan`` raised
        # before anything was drawn).  With them in, a resume at pass 6 re-walks
        # six spent passes, and each idle step still pays a full ``_mode_groups``
        # scan — measured at 129.9 ms on 317,520 groups, so ~5.7 h before the
        # first image renders.  Dropping them cannot skip a pass that was owed:
        # a terminal source is refused inside the walk anyway.
        terminal = self._terminal_source_ids()
        use_index = min(
            (
                used.get(row.source_id, 0)
                for row in sources if row.source_id not in terminal
            ),
            default=0,
        )
        if budget:
            # A source whose groups were journalled and then *lost* still spends
            # its uses (the group_ids are taken), so the cursor can already sit
            # at the budget while the live count is short of target.  Opening a
            # pass the budget does not own would hand that source another group
            # — and at ``max_source_uses = 1`` that is the switch turning itself
            # on.  Clamping keeps the last permitted pass the last one.
            use_index = min(use_index, budget - 1)
        while True:
            # Nothing after the fill is read on the final permitted pass, so the
            # default path does not pay for a measurement it cannot use.
            final = bool(budget) and use_index + 1 >= budget
            before = None if final else len(self._mode_groups(mode))
            cursor_skips = self._fill_initial_mode(
                mode, sources, target, use_index=use_index
            )
            use_index += 1
            # Before every exit below, so the last pass of a mode is pipelined
            # exactly like the ones before it instead of falling to the closing
            # drain: at ``max_source_uses = 1`` the last pass is the only pass.
            if annotate_passes:
                self._flush_annotation_batch()
            # After the flush and before every other exit: a pass cut short by the
            # marker still owes its winners to the pipeline, and the tail of that
            # pass is the last batch this build will ever hand over.  Opening
            # another pass instead would render for hours on a card the operator
            # has already promised to training.
            if self._stop_fill():
                return
            if final:
                return
            if len(self._mode_groups(mode)) >= target:
                return
            if len(self._mode_groups(mode)) <= before and not cursor_skips:
                # Nothing rendered *and* nothing was held back for a later pass:
                # the pool is terminal, SAM3-blocked or out of presets, and
                # another lap would only re-walk the same refusals.
                return

    def _fill_initial_mode(
        self,
        mode: str,
        sources: tuple[SourceRecord, ...],
        target: int,
        *,
        use_index: int = 0,
    ) -> int:
        """Render one pass over ``sources``; return how many the cursor skipped.

        The return value is what lets ``_fill_mode`` tell "this pool is finished"
        apart from "this pass is finished": a source skipped because it already
        owns a group for *this* pass still owes the build a group on a later one.
        Sources skipped as terminal or SAM3-pending are deliberately not counted
        — those are refusals, not deferrals.
        """
        terminal = self._terminal_source_ids()
        pending = self._pending_sam3_ids() if mode == "local" else set()
        # One snapshot per pass, not one scan per source.  It stays exact for the
        # whole pass because ``sources`` holds each source_id once
        # (``_require_unique_source_ids`` guarantees it), so a source's count can
        # only change after its own single turn here — never before the test that
        # reads it.  The pre-reuse code re-derived ``completed_sources()`` on
        # every iteration and reached the same answer for the same reason; this
        # just stops paying O(groups) for it 26k times a pass.
        used = self.store.completed_source_uses()
        cursor_skips = 0
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
            # The in-pass safe boundary.  Breaking *before* a new source is
            # submitted, rather than abandoning one that is already rendering, is
            # what makes this graceful: the window still holds up to
            # ``_source_window`` sources, and the ``while inflight`` drain below
            # commits every one of them and takes the closing land checkpoint,
            # exactly as the end of an ordinary pass does.  Nothing is cancelled,
            # no group is half-journalled, and the marker costs at most one
            # window's worth of renders before the cards are free.
            if self._stop_fill():
                break
            while inflight and (
                len(inflight) >= self._source_window
                or len(self._mode_groups(mode)) + len(pending) + len(inflight) >= target
                # Draining to empty is the only thing that reaches the
                # ``not inflight`` checkpoint below, so this clause is how a
                # mid-pass land ever happens — and why it must ask about
                # reclaimable bytes only.  Counting the prefetch buffer here held
                # it true forever and collapsed the window to one source in
                # flight, which is what took prod-l8 to 15 groups/h.
                or self._staging_full()
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
                    use_index=use_index,
                    used=used,
                )
            # "This source already gave pass ``use_index`` its group."  At
            # ``use_index = 0`` that is the membership test this replaced, since
            # a count above zero is exactly membership.  Counted separately from
            # the terminal/pending refusals below: this one means "come back next
            # pass", and ``_fill_mode`` has to be able to tell the two apart.
            if used.get(source.source_id, 0) > use_index:
                cursor_skips += 1
                continue
            if source.source_id in terminal or source.source_id in pending:
                continue
            following = threading.Event()
            turn = _SelectorTurn(ready=selector_tail, following=following)
            selector_tail = following
            inflight.append((
                source,
                self._source_executor.submit(
                    self._render_source_buffered, source, mode,
                    use_index=use_index, selector_turn=turn,
                ),
            ))
        while inflight:
            finish_oldest()
        return cursor_skips

    def _stop_fill(self) -> bool:
        """Has this build been told to stop opening new render work?

        **Latched**, for two independent reasons.  It is polled once per source
        inside ``_fill_initial_mode`` — a pass over the L8 pool is 26k sources and
        many hours, so a per-pass check would not stop anything today — and a
        latch keeps that a single ``stat`` rather than one per source for the rest
        of the run.  More importantly it makes the answer monotone within a
        process: half a pass that believed the marker was set followed by half a
        pass that believed it was gone is a build in neither state.  Clearing the
        marker therefore takes effect at the next *start*, which is exactly the
        documented switch back (the GPUs cannot be un-released mid-run either).

        The one side effect is the journal row on the transition, written here
        because this is the moment the build learned, and written at most once per
        process: it is what puts "why did this build stop 338k groups short" in
        the same durable ledger as the shortfall it causes.  It is deliberately
        **not terminal** — the shortfall row is the terminal one, so
        ``complete_with_failures`` is reached by the rule that always reached it.
        """
        if not self._stop_fill_latched:
            if not _stop_fill_requested(self.config.output_root):
                return False
            self._stop_fill_latched = True
        if not self._stop_fill_recorded:
            self._stop_fill_recorded = True
            local, global_ = len(self._mode_groups("local")), len(self._mode_groups("global"))
            self._journal_once(
                # What varies between two stops of one build is where it was when
                # it learned, so that is the identity: the restart that only
                # drains adds nothing, and a build stopped again after filling
                # further records the new position.
                task_id=stable_id(
                    "stop-fill", self.config.build_id,
                    self._stop_fill_at_start, local, global_,
                ),
                event_type="stop_fill",
                stage="rendering",
                error_code="stop_fill_requested",
                message=json.dumps({
                    "marker": str(_stop_fill_marker(self.config.output_root)),
                    "at_start": self._stop_fill_at_start,
                    "phase": self._phase,
                    "local": local,
                    "global": global_,
                }, sort_keys=True),
                retryable=False,
                terminal=False,
                durable=True,
            )
        return True

    def _release_heavy_resources(self) -> None:
        """Give the render GPUs back, for a phase that needs the cards elsewhere."""
        if self.renderer is None and self.scorer is None:
            # A stop-fill build never loaded either, so there is no allocator to
            # drain — and no reason to import torch into a process whose whole
            # contract with the training queue is that it stays off the cards.
            return
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
        # A relabel candidate comes off ``_pending_sam3_ids``, which subtracts
        # every source that already owns a group, so this is 0 today.  Deriving
        # it rather than assuming it keeps the ID namespace right if that filter
        # ever loosens under source reuse.
        result = self._render_source(
            refreshed, "local", queue_mask_failure=False,
            use_index=self.store.completed_source_uses().get(source.source_id, 0),
        )
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
        # Both readings below used to scan the whole journal per pending source.
        # The ledger folds them incrementally, and every read re-refreshes it
        # first, so rows this loop appends to the journal — including a
        # ``sam3_ready`` consumed by a ``sam3_ready_invalid`` further down the
        # same loop body — are in the answer exactly as they were before.
        ledger = _Sam3Ledger()

        def attempts_of(source_id: str) -> int:
            return ledger.refresh(self.store.failures).attempts_of(source_id)

        def unconsumed_ready(source_id: str) -> bool:
            return ledger.refresh(self.store.failures).unconsumed_ready(source_id)

        while len(self._mode_groups("local")) < target:
            # A marker dropped *during* this phase, which on a real build is
            # hours long.  Retiring between rounds is the safe boundary here for
            # the same reason it is inside a fill pass: every source is either
            # still queued or already resolved, nothing is half-attempted, and no
            # attempt has been charged against a budget it did not spend.
            if self._stop_fill():
                return

            pending_ids = self._pending_sam3_ids()
            pending = [by_id[source_id] for source_id in sorted(pending_ids) if source_id in by_id]

            made_progress = False
            for source in list(pending):
                if not unconsumed_ready(source.source_id):
                    continue
                attempt = attempts_of(source.source_id)
                status = self._try_ready_relabel(source, attempt)
                made_progress = True
                if status == "retry" and attempt >= max_attempts:
                    self._terminal_sam3(source, "ready relabel still violates canonical geometry")

            pending_ids = self._pending_sam3_ids()
            pending = [by_id[source_id] for source_id in sorted(pending_ids) if source_id in by_id]
            exhausted = [
                source for source in pending
                if attempts_of(source.source_id) >= max_attempts
                and not unconsumed_ready(source.source_id)
            ]
            for source in exhausted:
                self._terminal_sam3(source, "SAM3 relabel attempt budget exhausted")
                made_progress = True

            pending_ids = self._pending_sam3_ids()
            pending = [by_id[source_id] for source_id in sorted(pending_ids) if source_id in by_id]
            candidates = [
                source for source in pending
                if attempts_of(source.source_id) < max_attempts
                and not unconsumed_ready(source.source_id)
            ]
            if candidates:
                next_attempt = min(attempts_of(row.source_id) + 1 for row in candidates)
                batch = [
                    row for row in candidates
                    if attempts_of(row.source_id) + 1 == next_attempt
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
                self._fill_mode("local", self.allocation.local, target)
                made_progress = made_progress or len(self._mode_groups("local")) > before \
                    or len(self._pending_sam3_ids()) > pending_count
            if not made_progress:
                break

        # Not under stop-fill: this error means "the drain ran and could not
        # finish", and a queue the drain was told not to touch is neither
        # unresolved nor this build's problem — it is a backlog handed to
        # whichever build starts without the marker.  Guarded rather than left to
        # the ``return`` above so a marker that appears after the last round's
        # check still lands a ``complete_with_failures`` instead of a crash.
        if self._pending_sam3_ids() and not self._stop_fill():
            raise PipelineError("SAM3 relabel queue remains unresolved")

    def _record_shortfalls(self) -> None:
        for mode, target in (
            ("local", self.allocation.local_target),
            ("global", self.allocation.global_target),
        ):
            completed = len(self._mode_groups(mode))
            if completed >= target:
                continue
            # The count is part of the event's identity, not just its message: a
            # build stopped at 61,500 and restarted at 61,500 says it once, while
            # one that filled further before stopping again journals the new
            # number rather than being refused for disagreeing with the old row.
            self._journal_once(
                task_id=stable_id("target", self.config.build_id, mode, completed),
                event_type="terminal",
                stage="rendering",
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
            # Before the executors and, more importantly, before ``run`` closes
            # the store: the driver's threads write to it, and a store closed
            # under them turns a build that merely failed into one that also
            # loses whatever the annotator was holding.  The error it may be
            # carrying is deliberately not re-raised here — a failure inside a
            # ``finally`` would replace the exception that is actually being
            # reported.  ``_run_phases`` raises it on the way out of the happy
            # path, and ``submit`` raises it at the next pass boundary.
            self._close_annotation_driver()
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
        # ``phase`` stays "rendering" for the whole of both calls even though
        # annotation is running underneath them: the phase names what owns the
        # GPUs and what a resume has to redo, and neither changes because a relay
        # socket is busy.  ``annotation.inflight`` in the manifest is where the
        # background work shows up.
        self._fill_mode(
            "global", self.allocation.global_, self.allocation.global_target,
            annotate_passes=True,
        )
        self._fill_mode(
            "local", self.allocation.local, self.allocation.local_target,
            annotate_passes=True,
        )

        self._write_phase("sam3_relabel")
        # The relabel drain is the *other* renderer, and under stop-fill it is
        # skipped whole rather than allowed to spin.  Both halves of it need a
        # card: ``dependencies.relabeler`` puts the 860 M-parameter SAM3 detector
        # on cuda:0, and every source it rescues is then re-rendered through
        # ``_try_ready_relabel`` -> ``_render_source``, which is precisely the
        # work the marker retired (and which would now raise "render/QA resources
        # are not loaded" against a build that deliberately loaded neither).
        #
        # The pending queue is therefore left exactly as it stands: those sources
        # keep their ``sam3_relabel_queued`` rows and no attempt is charged
        # against their budget, so a later build with the marker removed picks
        # them up unchanged.  The ``PipelineError`` an unresolved queue normally
        # raises does not apply — it means "the drain ran and could not finish",
        # and here the drain never ran.  ``sam3_relabel.pending`` in the manifest
        # keeps reporting the backlog, and the stop-fill event says why.
        if not self._stop_fill():
            self._drain_sam3_and_replacements()
        # Unchanged, and the point of leaving it here: the shortfall is measured
        # against the configured target whatever stopped the fill, so a build
        # retired at 61.5k of 400k journals one terminal ``local_target_shortfall``
        # saying so, and that row is what makes the final status
        # ``complete_with_failures`` by the rule that always made it so.
        self._record_shortfalls()
        self._land_checkpoint(force=True)

        self._release_heavy_resources()
        self._write_phase("annotation")
        # The cards are already back before this waits, so a batch still on the
        # relay costs nothing but wall clock.  Joining first also means the
        # closing drain is the only annotator running, which is what keeps two
        # drains from picking up one task and paying for it twice.
        driver_error = self._close_annotation_driver()
        if driver_error is not None:
            raise driver_error
        # Annotation reads winner bytes by their staging path, which now lives in
        # the archive; the reverse map must know about this build's batches first.
        # Full rather than incremental: this is also the refresh that adopts
        # everything a previous run of a resumed build landed.
        self._refresh_catalog()
        annotation = self._annotation_drainer().drain()
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
            # The one mirror whose events are *not* journalled: the manifest just
            # above froze the digests of these three files, so appending a
            # failure row here would leave the mirror one record behind the
            # digest it is published under.  A fallback at the last checkpoint of
            # a build is also the least interesting one — the mirror it produces
            # is a whole-file copy, which is correct by construction.
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
    # Decided once, here, and handed to the pipeline rather than re-derived by
    # it: this is the branch that must not load a model, and a second reading of
    # the marker could disagree with the first.
    #
    # **Every GPU load point in a canonical build is inside the else arm.**
    # ``renderer_factory`` is ``LocalGpuOnlyRenderer.create``, whose preflight
    # allocates a probe tensor on cuda:1 (that allocation *is* the context);
    # ``assert_ready`` re-reads ``torch.cuda.is_available``; ``_load_scorer``
    # instantiates one or two OneAlign copies on cuda:0 and, unless the config
    # waives it, runs a real forward through one.  The only two others in the
    # whole lifecycle are reached from the rendering and relabel phases, which
    # the marker retires: ``dependencies.relabeler`` (SAM3 on cuda:0, called only
    # from ``_drain_sam3_and_replacements``) and the torch helpers in
    # ``visibility``/``rendering``, called only from ``_render_source``.
    #
    # What is left running is relay and CPU work: the annotation drain reads
    # ``I_tar``/``I_in`` through ``archive_reader`` (tar + sqlite + PIL) and posts
    # them to an HTTP endpoint, landing hardlinks and packs, projection writes
    # Postgres.  Nothing on that path imports torch — ``_empty_cuda_cache`` is the
    # single call that could, and it is gated on ``torch.cuda.is_initialized()``
    # and skipped outright by ``_release_heavy_resources``.
    stop_fill = _stop_fill_requested(config.output_root)
    renderer: Renderer | None = None
    scorer: Scorer | None = None
    if not stop_fill:
        renderer = dependencies.renderer_factory(config)
        renderer.bind_catalog(catalog)
        renderer.assert_ready()
        scorer = _load_scorer(config, dependencies, inventory)

    with ArtifactStore(config.output_root, config.build_id) as store:
        pipeline = CanonicalPipeline(
            config, dependencies, inventory, catalog, renderer, scorer, store,
            existing, stop_fill=stop_fill,
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
