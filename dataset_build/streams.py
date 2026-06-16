"""
dataset_build/streams.py
========================
The 7 budget streams (S1..S7) + the accept/reject QA gate for the
VeraRetouch Direction-A, RECIPE-BASED, region-local-heavy 1,000,000 dataset.

Each :class:`Stream` deterministically PLANS ``(source, recipe, region_local)``
work items (scene-matched), runs the per-sample DAG

    decode -> render(TEACHER) -> [region_composite] -> C_GT -> annotate -> gate

and YIELDS accepted :class:`~dataset_build.contracts.Sample` records. Streams are
RESUMABLE: ``run(..., done_ids=...)`` skips ids already committed by the
:class:`~dataset_build.contracts.ShardWriter`.

Grounding (read before editing):
  - docs/plan/dataset/DATASET_BUILD_PLAN.md          §2 (streams), §3 (DAG), §7 (QA)
  - docs/plan/dataset/probe/probe_sources_budget.md  (per-stream construction)
  - docs/plan/dataset/probe/probe_sam3_masking.md    §5-7 (C_GT)
  - docs/plan/impl_plan_A_attention_context_4dlut.md §2.1-2.3 (3-aspect C_GT, Mode-1/2)

This module is IMPORT-LIGHT at top level: only stdlib + ``contracts`` +
``config`` helpers. Heavy collaborators (Renderer / ConceptMasker / Cleaner /
RecipeParser / Registry) are INJECTED at construction time; we never import
``torch`` / ``transformers`` / ``rawpy`` / ``cv2`` here. ``numpy`` is imported
lazily only inside the few methods that synthesize a degrade mask.

The collaborators are duck-typed against the abstract interfaces in
``contracts`` (Renderer, ConceptMasker, Cleaner, RecipeParser, QAGate). A
``MockModels`` bundle (see ``run.py --dry-run``) satisfies the same duck types
on CPU with no weights, so the full DAG is exercisable without GPUs.
"""

from __future__ import annotations

import abc
import hashlib
import math
import os
import random
import sys
import threading
from dataclasses import dataclass, field
from pathlib import Path
from typing import (
    Any,
    Dict,
    Iterable,
    Iterator,
    List,
    Optional,
    Protocol,
    Sequence,
    Tuple,
)

from dataset_build.contracts import (
    AfterSource,
    ASPECTS,
    CgtRef,
    DegradeSpec,
    MaskSource,
    PARAM_KEYS,
    Provenance,
    QualityScores,
    RawDecode,
    Recipe,
    RecipeAsset,
    RecipeKind,
    Sample,
    SceneMeta,
    SourceItem,
    StreamId,
)
from dataset_build.contracts import QAGate as QAGateABC
from dataset_build.contracts import Stream as StreamABC


# ===========================================================================
# Collaborator protocols (duck-typed; mirror the contracts.py interfaces).
# A protocol keeps this module decoupled from the concrete implementations
# (render.VeraRetouchRenderer / masking.Sam3Masker / vlm_clean.QwenVLCleaner /
#  recipes.DiskRecipeParser+DiskLutApplier) which are written by sibling agents
# and only imported lazily by run.py:_build_real_models.
# ===========================================================================


class _RendererProto(Protocol):
    def render(
        self,
        image_paths: Sequence[str],
        param_dicts: Sequence[Dict[str, Dict[str, float]]],
        batch_size: int = ...,
        chunk: int = ...,
        max_new_tokens: int = ...,
    ) -> List[Any]: ...


class _MaskerProto(Protocol):
    def mask(
        self,
        image: Any,
        concept: str,
        native_size: Optional[Tuple[int, int]] = ...,
        soft: bool = ...,
        reduce: str = ...,
        min_score: float = ...,
    ) -> Any: ...

    def masks(self, image: Any, concepts: Sequence[str], **kw: Any) -> Dict[str, Any]: ...

    def cgt_aspect_stack(self, image: Any, concept_map: Dict[str, List[str]]) -> Any: ...


class _CleanerProto(Protocol):
    def gen_instruction(self, image_path: str, scene_meta: Optional[dict] = ...) -> dict: ...

    def reason_params(self, image_path: str, instruction: str) -> dict: ...

    def verify(self, before_path: str, after_path: str, params: dict) -> dict: ...

    def tag_scene_region(self, image_path: str, instruction: str) -> dict: ...


class _RecipeParserProto(Protocol):
    def xmp_to_params(self, xmp_path: str) -> Dict[str, Dict[str, float]]: ...

    def lrtemplate_to_params(self, path: str) -> Dict[str, Dict[str, float]]: ...

    def sample_degrade_spec(
        self, aspects: Sequence[str], seed: int, region_local: bool
    ) -> DegradeSpec: ...

    def load_cube(self, path: str) -> Tuple[Any, Tuple[float, float, float], Tuple[float, float, float]]: ...


class _LutApplierProto(Protocol):
    def apply_lut(
        self,
        img: Any,
        lut: Any,
        domain_min: Tuple[float, float, float] = ...,
        domain_max: Tuple[float, float, float] = ...,
    ) -> Any: ...


# ===========================================================================
# Build context: the injected collaborators + the resolved config + caches.
# ===========================================================================


@dataclass
class BuildContext:
    """Everything a Stream needs to build a sample, injected once by run.py.

    Collaborators may be ``None`` only in narrow situations (e.g. ``cleaner``
    disabled); ``build_one`` degrades gracefully and records why on the sample.
    """

    config: Dict[str, Any]
    renderer: Optional[_RendererProto] = None
    masker: Optional[_MaskerProto] = None
    cleaner: Optional[_CleanerProto] = None
    parser: Optional[_RecipeParserProto] = None
    lut_applier: Optional[_LutApplierProto] = None
    cgt_writer: Optional[Any] = None  # pack.ShardWriter-compatible C_GT sink (write_cgt)
    # resolved recipe-asset lookup (recipe_id -> RecipeAsset) for LUT resolution
    recipe_assets: Dict[str, RecipeAsset] = field(default_factory=dict)
    lut_cache: Dict[str, Tuple[Any, Tuple[float, float, float], Tuple[float, float, float]]] = field(default_factory=dict)
    aspect_magnitude_cache: Dict[str, Dict[str, float]] = field(default_factory=dict)
    # Serializes the (non-thread-safe) GPU teacher renderer when run_stream builds
    # samples concurrently; cleaning (vLLM HTTP) parallelizes around it.
    render_lock: Any = field(default_factory=threading.Lock)
    # Per-run scratch dir for TRANSIENT verify PNGs (set by run.run_stream, swept at
    # start). Recipe-only storage is preserved: these are deleted immediately after
    # the judge reads them. None => verify disabled / no temp persistence.
    verify_tmp_dir: Optional[str] = None
    # VLM tag caches are shared by all stream workers in one build process.
    # Source tags are image-level and reusable; region tags are intent-specific
    # because they can drive SAM3 mask construction.
    source_tag_cache: Dict[str, dict] = field(default_factory=dict)
    region_tag_cache: Dict[str, dict] = field(default_factory=dict)
    tag_cache_inflight: Dict[str, threading.Event] = field(default_factory=dict)
    tag_cache_lock: Any = field(default_factory=threading.Lock)
    # Optional PRECOMPUTED per-source tag/aesthetic cache (tag_cache.CachedTagger).
    # When set (config tag_cache.use_cache:true), the inline source/region tag
    # sites read this INSTEAD of the live VLM tag call. None => no behavior change.
    cached_tagger: Optional[Any] = None
    # Decoupled core facade (dataset_build.core.Core). When set, the teacher
    # render path routes through ``core.render`` (which owns render_lock+renderer),
    # and ``cleaner``/``masker`` are the delegating ``core.vllm``/``core.sam3``
    # wrappers. None => fully inline legacy path (byte-identical fallback).
    core: Optional[Any] = None

    # ---- convenience config accessors (with plan defaults) ----
    @property
    def cgt_cfg(self) -> Dict[str, Any]:
        return self.config.get("cgt", {}) or {}

    @property
    def qa_cfg(self) -> Dict[str, Any]:
        return self.config.get("qa", {}) or {}

    @property
    def degrade_cfg(self) -> Dict[str, Any]:
        return self.config.get("degrade", {}) or {}

    @property
    def aspect_mix_cfg(self) -> Dict[str, Any]:
        return self.config.get("aspect_mix", {}) or {}

    @property
    def concept_map(self) -> Dict[str, List[str]]:
        cm = self.cgt_cfg.get("concept_map") or {}
        # ensure every aspect present so cgt_aspect_stack never KeyErrors
        return {a: list(cm.get(a, [])) for a in ASPECTS}

    @property
    def render_kw(self) -> Dict[str, int]:
        vr = (self.config.get("models", {}) or {}).get("veraretouch", {}) or {}
        return {
            "batch_size": int(vr.get("batch_size", 8)),
            "chunk": int(vr.get("chunk", 262144)),
            "max_new_tokens": int(vr.get("max_new_tokens", 768)),
        }


# ===========================================================================
# Small deterministic helpers (pure; no GPU, no heavy imports).
# ===========================================================================


def _stable_choice(items: Sequence[Any], key: str) -> Any:
    """Deterministically pick one element keyed by a string (sha1 -> index)."""
    if not items:
        raise ValueError("cannot choose from an empty sequence")
    h = int(hashlib.sha1(key.encode("utf-8")).hexdigest(), 16)
    return items[h % len(items)]


def _det_rng(seed: int, *parts: Any) -> random.Random:
    """A reproducible ``random.Random`` seeded by (seed, *parts)."""
    h = hashlib.sha1(("|".join([str(seed)] + [str(p) for p in parts])).encode()).hexdigest()
    return random.Random(int(h[:16], 16))


def _det_seed32(*parts: Any) -> int:
    """A reproducible 32-bit seed from *parts (replaces process-randomized hash())."""
    h = hashlib.sha1("\x1f".join(str(p) for p in parts).encode()).digest()
    return int.from_bytes(h[:4], "big")


def make_sample_id(stream: StreamId, source_id: str, recipe_key: str, idx: int) -> str:
    """Stable, collision-resistant id used for sharding + resumability."""
    raw = f"{stream.value}:{source_id}:{recipe_key}:{idx}"
    return f"{stream.value.lower()}_{hashlib.sha1(raw.encode()).hexdigest()[:16]}"


def scene_of(item: SourceItem) -> str:
    return (item.scene or "any").lower()


def _scene_compatible(src_scene: str, recipe_affinity: Optional[str]) -> bool:
    """Soft scene matching (probe_sources_budget §3). ``any`` matches everything."""
    aff = (recipe_affinity or "any").lower()
    if aff == "any" or src_scene == "any":
        return True
    return aff == src_scene


def _empty_params() -> Dict[str, Dict[str, float]]:
    """All-38-keys-zero param dict (the renderer's identity transform)."""
    return {k: {"value": 0.0} for k in PARAM_KEYS}


def _params_from_degrade_spec(spec: DegradeSpec) -> Dict[str, Dict[str, float]]:
    """Teacher-renderable neg/pos params for a DegradeSpec.

    `forward=False` means the stored params are neg_p: render(source, neg_p)
    produces the degraded input preview. Stage-0 later searches z* from that
    teacher-manifold input back to the clean source.
    """
    sign = 1.0 if spec.forward else -1.0
    out = _empty_params()
    for key, raw in (spec.op_params or {}).items():
        if key in out:
            out[key] = {"value": round(sign * float(raw), 3)}
    return out


def _file_sha256(path: str) -> Optional[str]:
    try:
        h = hashlib.sha256()
        with open(path, "rb") as f:
            for chunk in iter(lambda: f.read(1024 * 1024), b""):
                h.update(chunk)
        return h.hexdigest()
    except OSError:
        return None


# Sentinel distinguishing "caller passed no precomputed after" (=> render inline,
# the legacy run()/test path) from "caller passed an after, possibly None" (the
# batched run_stream path, which hoists the GPU render out of build_one).
_UNSET: Any = object()


def param_delta_aspect_magnitudes(params: Dict[str, Dict[str, float]]) -> Dict[str, float]:
    """L2 norm of the (raw/100) param deltas per aspect, for Mode-1 g(||dtheta||).

    Mirrors get_organized_dict's /100 normalization so magnitudes live on the
    same scale the renderer consumes (impl_plan_A §2.3).
    """
    from dataset_build.contracts import COLORMIXER_KEYS, COLORTEMP_KEYS, LIGHT_KEYS

    groups = {"L": LIGHT_KEYS, "GC": COLORTEMP_KEYS, "SC": COLORMIXER_KEYS}
    out: Dict[str, float] = {}
    for aspect, keys in groups.items():
        acc = 0.0
        for k in keys:
            v = float((params.get(k) or {}).get("value", 0.0)) / 100.0
            acc += v * v
        out[aspect] = math.sqrt(acc)
    return out


def magnitude_weight(norm: float, tau: float) -> float:
    """g(||dtheta_r||) = tanh(||dtheta_r|| / tau) (impl_plan_A §2.3, cgt.magnitude_tau)."""
    tau = tau if tau and tau > 0 else 1.0
    return math.tanh(norm / tau)


# ===========================================================================
# QA gate (contracts.QAGate). Pure: only reads sample.quality + config.
# ===========================================================================


class QAGate(QAGateABC):
    """Accept/reject gate (DATASET_BUILD_PLAN §7; thresholds in config.yaml:qa).

    Accept iff every ENABLED gate passes. A gate whose input is ``None`` is
    treated as "not measured" and skipped (so a --dry-run with no vLLM judge
    still accepts), EXCEPT the mandatory boolean gates when their config flag is
    on. On failure sets ``sample.quality.rejected_reason`` and returns False.
    """

    def __init__(self, config: Dict[str, Any]):
        qa = (config or {}).get("qa", {}) or {}
        self.mllm_score_min = float(qa.get("mllm_score_min", 0.70))
        self.require_look_match = bool(qa.get("require_look_match", True))
        self.require_param_sane = bool(qa.get("require_param_sane", True))
        self.mmart_require_processed_ok = bool(qa.get("mmart_require_processed_ok", True))
        self.mask_quality_min = float(qa.get("mask_quality_min", 0.30))
        self.histsim_min = float(qa.get("histsim_min", 0.50))
        self.er_recon_psnr_min = float(qa.get("er_recon_psnr_min", 25.0))
        cgt = (config or {}).get("cgt", {}) or {}
        self.min_coverage = float(cgt.get("min_coverage", 0.01))
        self.max_coverage = float(cgt.get("max_coverage", 0.95))

    def _reject(self, sample: Sample, reason: str) -> bool:
        sample.quality.rejected_reason = reason
        return False

    def accept(self, sample: Sample) -> bool:
        q = sample.quality

        # --- MLLM judge score (skip if not measured) ---
        if q.mllm_score is not None and q.mllm_score < self.mllm_score_min:
            return self._reject(sample, f"mllm_score<{self.mllm_score_min}")

        # --- mandatory boolean gates (only enforced when the judge ran) ---
        if self.require_look_match and q.look_match is False:
            return self._reject(sample, "look_match=False")
        if self.require_param_sane and q.param_sane is False:
            return self._reject(sample, "param_sane=False")

        # --- MMArt processed.jpg sanity (caveat a) ---
        if (
            sample.stream == StreamId.S3_MMART_LOCAL
            and self.mmart_require_processed_ok
            and q.processed_ok is False
        ):
            return self._reject(sample, "processed_ok=False")

        # --- mask quality floor (region-local only) ---
        if sample.region_local and q.mask_quality is not None and q.mask_quality < self.mask_quality_min:
            return self._reject(sample, f"mask_quality<{self.mask_quality_min}")

        # --- color-direction agreement (Track recon gate) ---
        if q.histsim is not None and q.histsim < self.histsim_min:
            return self._reject(sample, f"histsim<{self.histsim_min}")

        # --- Track-A E+R round-trip floor ---
        if q.er_recon_psnr is not None and q.er_recon_psnr < self.er_recon_psnr_min:
            return self._reject(sample, f"er_recon_psnr<{self.er_recon_psnr_min}")

        # --- coverage gate for region-local C_GT (DATASET_BUILD_PLAN §5) ---
        if sample.region_local and sample.c_gt is not None:
            cov = sample.c_gt.coverage
            if cov is not None:
                if cov < self.min_coverage:
                    return self._reject(sample, f"coverage<{self.min_coverage}")
                if cov > self.max_coverage:
                    return self._reject(sample, f"coverage>{self.max_coverage}")

        return True


# ===========================================================================
# Base stream: shared decode/render/composite/C_GT/annotate/gate machinery.
# ===========================================================================


class BaseStream(StreamABC):
    """Common per-sample DAG. Concrete streams override ``plan`` (and, where the
    pairing logic differs, parts of ``build_one``)."""

    stream_id: StreamId  # set by subclasses

    def __init__(self, ctx: BuildContext, gate: Optional[QAGate] = None):
        self.ctx = ctx
        self.gate = gate or QAGate(ctx.config)
        # The most recently rejected sample (its quality.rejected_reason is the
        # real gate reason). run.py reads this to log a precise reject row rather
        # than a generic placeholder.
        self.last_rejected: Optional[Sample] = None

    def _gate(self, sample: Sample) -> bool:
        """Apply the QA gate; remember the sample on reject for precise logging."""
        if self.gate.accept(sample):
            return True
        self.last_rejected = sample
        return False

    # ------------------------------------------------------------------ utils
    def _vlm_image_path(self, source: SourceItem, recipe: Recipe, sample: Sample) -> str:
        """Choose a decodable image for VLM annotation/tagging.

        Some quality anchors are RAW/DNG, which PIL/vLLM cannot read directly.
        For those samples, prefer an attached expert JPG preview when available.
        Verify still uses its own before/after path handling.
        """
        for p in (
            sample.meta.get("expert_after_path"),
            recipe.meta.get("expert_after_jpg"),
            (source.meta or {}).get("expert_after_jpg"),
        ):
            if p:
                return str(p)
        return source.path

    @staticmethod
    def _copy_vlm_tag(tag: dict) -> dict:
        out = dict(tag or {})
        if isinstance(out.get("sam3_concepts"), list):
            out["sam3_concepts"] = list(out["sam3_concepts"])
        return out

    def _source_tag_key(self, source: SourceItem, vlm_path: str) -> str:
        raw = "|".join(
            [
                "source",
                str(source.source_id or ""),
                os.path.abspath(str(vlm_path)),
                str(source.width or ""),
                str(source.height or ""),
            ]
        )
        return hashlib.sha1(raw.encode("utf-8")).hexdigest()

    def _region_tag_key(self, source: SourceItem, vlm_path: str, instruction: str) -> str:
        raw = "|".join(
            [
                "region",
                self._source_tag_key(source, vlm_path),
                hashlib.sha1((instruction or "").encode("utf-8")).hexdigest(),
            ]
        )
        return hashlib.sha1(raw.encode("utf-8")).hexdigest()

    def _cached_tag(self, cache_name: str, key: str, producer: Any) -> dict:
        """Return a cached VLM tag while coalescing same-key concurrent calls."""
        cache: Dict[str, dict] = getattr(self.ctx, cache_name)
        inflight_key = f"{cache_name}:{key}"
        owner = False
        with self.ctx.tag_cache_lock:
            if key in cache:
                return self._copy_vlm_tag(cache[key])
            evt = self.ctx.tag_cache_inflight.get(inflight_key)
            if evt is None:
                evt = threading.Event()
                self.ctx.tag_cache_inflight[inflight_key] = evt
                owner = True

        if not owner:
            evt.wait()
            with self.ctx.tag_cache_lock:
                return self._copy_vlm_tag(cache.get(key, {}))

        try:
            tag = producer() or {}
            tag = self._copy_vlm_tag(tag)
            with self.ctx.tag_cache_lock:
                cache[key] = tag
            return self._copy_vlm_tag(tag)
        finally:
            with self.ctx.tag_cache_lock:
                self.ctx.tag_cache_inflight.pop(inflight_key, None)
                evt.set()

    def _apply_vlm_tag(self, sample: Sample, tag: dict, meta_key: str) -> None:
        if not tag:
            return
        sample.meta[meta_key] = self._copy_vlm_tag(tag)
        sample.scene_meta.scene = tag.get("scene") or sample.scene_meta.scene
        sample.scene_meta.style = tag.get("style") or sample.scene_meta.style
        try:
            sample.scene_meta.masksubtype_hint = int(
                tag.get("masksubtype_hint", sample.scene_meta.masksubtype_hint)
            )
        except (TypeError, ValueError):
            pass
        # Cached tags may carry an aesthetic (model score preferred, else VLM
        # score); round-trips via pack.py:172 -> contracts QualityScores.aesthetic.
        if sample.quality.aesthetic is None:
            ae = tag.get("aesthetic")
            if ae is None:
                ae = tag.get("aesthetic_model")
            if ae is None:
                ae = tag.get("aesthetic_vlm")
            if ae is not None:
                try:
                    sample.quality.aesthetic = float(ae)
                except (TypeError, ValueError):
                    pass

    def _precomputed_tag(self, source: SourceItem, recipe: Recipe, sample: Sample) -> Optional[dict]:
        """Per-source PRECOMPUTED tags (tag_cache.CachedTagger) if enabled + hit.

        Image-level (scene/style/concepts/aesthetic), so it serves both source
        and region tag sites. ``None`` => cache disabled or miss (caller falls
        through to the live VLM / offline path: NO behavior change)."""
        tagger = self.ctx.cached_tagger
        if tagger is None:
            return None
        vlm_path = self._vlm_image_path(source, recipe, sample)
        try:
            return tagger.tags_for(vlm_path)
        except Exception:
            return None

    def _source_vlm_tag(self, source: SourceItem, recipe: Recipe, sample: Sample) -> dict:
        cached = self._precomputed_tag(source, recipe, sample)
        if cached is not None:
            return cached
        cl = self.ctx.cleaner
        if cl is None:
            return {}
        vlm_path = self._vlm_image_path(source, recipe, sample)
        key = self._source_tag_key(source, vlm_path)
        return self._cached_tag(
            "source_tag_cache",
            key,
            lambda: cl.tag_scene_region(vlm_path, ""),
        )

    def _region_vlm_tag(
        self,
        source: SourceItem,
        recipe: Recipe,
        sample: Sample,
        instruction: str,
    ) -> dict:
        cached = self._precomputed_tag(source, recipe, sample)
        if cached is not None:
            return cached
        cl = self.ctx.cleaner
        if cl is None:
            return {}
        vlm_path = self._vlm_image_path(source, recipe, sample)
        key = self._region_tag_key(source, vlm_path, instruction)
        return self._cached_tag(
            "region_tag_cache",
            key,
            lambda: cl.tag_scene_region(vlm_path, instruction),
        )

    def _native_size(self, source: SourceItem) -> Optional[Tuple[int, int]]:
        if source.height and source.width:
            return (int(source.height), int(source.width))
        return None

    def _render_after(
        self, source: SourceItem, params: Dict[str, Dict[str, float]]
    ) -> Optional[Any]:
        """Render the GLOBAL 'after' via the teacher. Returns np.uint8 HxWx3 or None.
        A teacher failure (e.g. the VLM not emitting a retouch token) must reject
        THIS sample, never crash the run."""
        core = getattr(self.ctx, "core", None)
        if core is not None and getattr(core, "render", None) is not None:
            return core.render.render_one(source.path, params,
                                          log_prefix=f"stream {self.stream_id}")
        if self.ctx.renderer is None:
            return None
        try:
            # GPU teacher is not thread-safe: serialize it across concurrent workers.
            with self.ctx.render_lock:
                outs = self.ctx.renderer.render(
                    [source.path], [params], **self.ctx.render_kw
                )
        except Exception as e:  # noqa: BLE001 - isolate per-sample teacher failures
            print(f"[stream {self.stream_id}] render failed for {source.path}: {e}", file=sys.stderr)
            return None
        return outs[0] if outs else None

    def _render_after_batch(
        self,
        sources: Sequence[SourceItem],
        params_list: Sequence[Optional[Dict[str, Dict[str, float]]]],
    ) -> List[Optional[Any]]:
        """Batched teacher render: ONE renderer.render() call for the whole batch.

        Returns one np.uint8 HxWx3 RGB (or None) per input, IN INPUT ORDER. Items
        whose params is None (LUT identity / degrade / skipped) are never sent to
        the GPU and come back None. The single render() call is held under
        render_lock (the GPU teacher is not thread-safe). A whole-batch teacher
        failure rejects only THIS call's items (-> None), never crashes the run.

        render() is shuffle=False so its output order == input order; we filter
        out None-param items before the call and scatter the results back to their
        original slots, so the i-th return aligns with sources[i].
        """
        core = getattr(self.ctx, "core", None)
        if core is not None and getattr(core, "render", None) is not None:
            cap = int(self.ctx.qa_cfg.get("verify_downscale_longedge", 768))
            return core.render.render_batch(
                [s.path for s in sources], list(params_list), cap,
                log_prefix=f"stream {self.stream_id}",
            )
        n = len(sources)
        results: List[Optional[Any]] = [None] * n
        if self.ctx.renderer is None:
            return results
        idxs: List[int] = []
        paths: List[str] = []
        pdicts: List[Dict[str, Dict[str, float]]] = []
        for i, (src, p) in enumerate(zip(sources, params_list)):
            if p is None:
                continue
            idxs.append(i)
            paths.append(src.path)
            pdicts.append(p)
        if not paths:
            return results
        try:
            with self.ctx.render_lock:
                outs = self.ctx.renderer.render(paths, pdicts, **self.ctx.render_kw)
        except Exception as e:  # noqa: BLE001 - isolate a whole-batch teacher failure
            print(f"[stream {self.stream_id}] batch render failed ({len(paths)}): {e}",
                  file=sys.stderr)
            return results
        # Downscale each after to the verify long-edge BEFORE it enters the pipeline
        # queue: the after is only consumed by the (downscaled) judge, and holding
        # native-res arrays for a whole batch x max_outstanding exhausts system RAM.
        cap = int(self.ctx.qa_cfg.get("verify_downscale_longedge", 768))
        for k, slot in enumerate(idxs):
            out = outs[k] if (k < len(outs)) else None
            results[slot] = self._downscale_rgb(out, cap) if out is not None else None
        return results

    def _load_lut_for_recipe(
        self, recipe: Recipe
    ) -> Optional[Tuple[Any, Tuple[float, float, float], Tuple[float, float, float]]]:
        """Resolve and validate a LUT recipe via parser.load_cube, cached by id/path."""
        if recipe.kind != RecipeKind.LUT or not recipe.lut_recipe_id:
            return None
        key = str(recipe.lut_recipe_id)
        if key in self.ctx.lut_cache:
            return self.ctx.lut_cache[key]
        if self.ctx.parser is None:
            return None
        asset = self.ctx.recipe_assets.get(key)
        path = (asset.path if asset is not None else None) or recipe.meta.get("path")
        if not path:
            return None
        try:
            resolved = self.ctx.parser.load_cube(str(path))
        except Exception as e:  # noqa: BLE001 - caller skips malformed LUT sample
            print(f"[stream {self.stream_id}] LUT validation failed for {path}: {e}",
                  file=sys.stderr)
            return None
        self.ctx.lut_cache[key] = resolved
        return resolved

    def _apply_lut_after(self, source: SourceItem, recipe: Recipe) -> Optional[Any]:
        """CPU/deterministic LUT after for verify. Never consumes a GPU render slot."""
        if self.ctx.lut_applier is None:
            return None
        resolved = self._load_lut_for_recipe(recipe)
        if resolved is None:
            return None
        lut, dmin, dmax = resolved
        try:
            import cv2
            import numpy as np
            import torch

            bgr = cv2.imread(source.path, cv2.IMREAD_COLOR)
            if bgr is None:
                return None
            rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
            x = (
                torch.from_numpy(rgb.astype("float32") / 255.0)
                .permute(2, 0, 1)
                .unsqueeze(0)
            )
            y = self.ctx.lut_applier.apply_lut(x, lut, dmin, dmax)
            out = (
                y.squeeze(0)
                .permute(1, 2, 0)
                .detach()
                .cpu()
                .numpy()
            )
            return (np.clip(out, 0.0, 1.0) * 255.0).round().astype("uint8")
        except Exception as e:  # noqa: BLE001 - isolate per-sample LUT failures
            print(f"[stream {self.stream_id}] LUT apply failed for {source.path}: {e}",
                  file=sys.stderr)
            return None

    @staticmethod
    def _downscale_rgb(arr: Any, longedge: int) -> Any:
        """Long-edge downscale of an RGB uint8 array (best-effort; returns input on
        failure or if already small). Delegates to the canonical core impl so the
        inline and core.render paths produce identical pixels."""
        from dataset_build.core.render_worker import downscale_rgb
        return downscale_rgb(arr, longedge)

    def _params_for_render(
        self, recipe: Recipe
    ) -> Optional[Dict[str, Dict[str, float]]]:
        """Params to feed the BATCHED teacher render for this recipe, or None to
        skip the GPU entirely.

        Only PARAM recipes produce a meaningful teacher 'after' worth rendering and
        judging. LUT recipes render an identity transform (the LUT is applied
        separately, not by the teacher) and DEGRADE recipes have no teacher after,
        so both return None and never hit the GPU. Degrade/PPR10K streams override
        this to None as well (their build_one never consumes an after).
        """
        if recipe.kind == RecipeKind.PARAM:
            return recipe.params or _empty_params()
        return None

    # ------------------------------------------------------- verify (judge)
    def _write_downscaled(self, src: Any, dst: str, longedge: int) -> bool:
        """Write a downscaled JPEG of ``src`` (np.uint8 RGB array OR an image path)
        to ``dst`` for the vLLM judge. Downscaling caps the base64 payload + vision
        tokens (native-res before+after would blow past max_model_len). Returns
        True on success, False if the source is not decodable (e.g. a .dng before)."""
        import cv2
        import numpy as np

        if isinstance(src, str):
            bgr = cv2.imread(src, cv2.IMREAD_COLOR)
            if bgr is None:
                return False
        else:
            arr = np.asarray(src)
            if arr.ndim != 3 or arr.shape[2] != 3:
                return False
            bgr = cv2.cvtColor(arr.astype("uint8"), cv2.COLOR_RGB2BGR)
        h, w = bgr.shape[:2]
        m = max(h, w)
        if m > longedge:
            s = longedge / float(m)
            bgr = cv2.resize(bgr, (max(1, int(round(w * s))), max(1, int(round(h * s)))),
                             interpolation=cv2.INTER_AREA)
        return bool(cv2.imwrite(dst, bgr, [int(cv2.IMWRITE_JPEG_QUALITY), 90]))

    def _maybe_verify(
        self,
        source: SourceItem,
        recipe: Recipe,
        sample: Sample,
        after_rgb: Optional[Any],
    ) -> None:
        """Wire the teacher 'after' into the vLLM judge (cleaner.verify) and record
        its scores. Gated by config.qa.enable_verify; sampled deterministically by
        verify_sample_rate so --resume reproduces the same decision.

        Modes (qa.verify_gate_mode):
          - "log"     : record the verdict in sample.meta['verify'] only (the QA
                        gate is NOT affected) -> measure score distributions first.
          - "enforce" : also write look_match/param_sane/processed_ok/mllm_score
                        into sample.quality so QAGate.accept() filters on them.

        Recipe-only storage is preserved: a downscaled before+after JPEG is written
        to a per-run scratch dir and DELETED in finally. No after pixels persist.
        """
        cl = self.ctx.cleaner
        qa = self.ctx.qa_cfg
        if cl is None or after_rgb is None or not qa.get("enable_verify", False):
            return
        tmp_dir = self.ctx.verify_tmp_dir
        if not tmp_dir:
            return
        rate = float(qa.get("verify_sample_rate", 1.0))
        if rate < 1.0:
            seed = int(self.ctx.config.get("seed", 0))
            if _det_rng(seed, sample.sample_id, "verify").random() >= rate:
                return

        longedge = int(qa.get("verify_downscale_longedge", 768))
        base = os.path.join(tmp_dir, sample.sample_id)
        before_tmp = base + "_b.jpg"
        after_tmp = base + "_a.jpg"
        try:
            # before must be a decodable sRGB image; .dng/raw sources -> skip verify.
            if not self._write_downscaled(source.path, before_tmp, longedge):
                return
            if not self._write_downscaled(after_rgb, after_tmp, longedge):
                return
            params = dict(sample.answer or (recipe.params if recipe else None) or {})
            params["_instruction"] = sample.instruction or ""
            vr = cl.verify(before_tmp, after_tmp, params)
            if vr:  # non-empty => judge actually ran (verify returns {} on failure)
                sample.meta["verify"] = vr  # always recorded for inspection
                if str(qa.get("verify_gate_mode", "log")).lower() == "enforce":
                    sample.quality.look_match = vr.get("look_match")
                    sample.quality.param_sane = vr.get("param_sane")
                    sample.quality.processed_ok = vr.get("processed_ok")
                    sample.quality.mllm_score = vr.get("score")
        except Exception as e:  # noqa: BLE001 - never reject on judge/infra failure
            print(f"[stream {self.stream_id}] verify failed for {sample.sample_id}: {e}",
                  file=sys.stderr)
        finally:
            for p in (before_tmp, after_tmp):
                try:
                    os.unlink(p)
                except OSError:
                    pass

    # -------------------------------------------------------------- C_GT build
    def _degrade_mask(
        self, native_size: Optional[Tuple[int, int]], spec: DegradeSpec, seed: int
    ) -> Any:
        """Synthesize the EXACT region mask M_r for an inverse-degradation sample
        (S1). Shape from config.degrade.region_shapes. Returns float32 [0,1] HxW.

        This is the only place we touch numpy; it is cheap and CPU-only.
        """
        import numpy as np

        H, W = native_size if native_size else (256, 256)
        rng = np.random.default_rng(seed)
        shapes = self.ctx.degrade_cfg.get(
            "region_shapes", ["random_blob", "radial", "linear"]
        )
        # SAM3-concept shapes are handled by the masker, not here; restrict to the
        # geometric shapes for the exact construction mask.
        geom = [s for s in shapes if s != "sam3_concept"] or ["random_blob"]
        shape = geom[int(rng.integers(0, len(geom)))]
        yy, xx = np.mgrid[0:H, 0:W].astype("float32")

        if shape == "radial":
            cy, cx = rng.uniform(0.25, 0.75) * H, rng.uniform(0.25, 0.75) * W
            r = rng.uniform(0.15, 0.40) * max(H, W)
            d = np.sqrt((yy - cy) ** 2 + (xx - cx) ** 2)
            m = np.clip(1.0 - d / (r + 1e-6), 0.0, 1.0)
        elif shape == "linear":
            ang = rng.uniform(0, math.pi)
            nx, ny = math.cos(ang), math.sin(ang)
            proj = (xx / W) * nx + (yy / H) * ny
            t = rng.uniform(0.3, 0.7)
            soft = 0.15
            m = np.clip((proj - (t - soft)) / (2 * soft), 0.0, 1.0).astype("float32")
        else:  # random_blob: union of a few soft discs
            m = np.zeros((H, W), dtype="float32")
            for _ in range(int(rng.integers(2, 5))):
                cy, cx = rng.uniform(0, H), rng.uniform(0, W)
                r = rng.uniform(0.08, 0.25) * max(H, W)
                d = np.sqrt((yy - cy) ** 2 + (xx - cx) ** 2)
                m = np.maximum(m, np.clip(1.0 - d / (r + 1e-6), 0.0, 1.0))
        return m.astype("float32")

    def _soft_blur(self, mask01: Any) -> Any:
        """PerTouch soft-boundary blur on stored C_GT (cgt.soft_blur_sigma_px)."""
        sigma = float(self.ctx.cgt_cfg.get("soft_blur_sigma_px", 0.0) or 0.0)
        if sigma <= 0:
            return mask01
        try:
            from scipy.ndimage import gaussian_filter  # type: ignore

            return gaussian_filter(mask01, sigma=sigma).astype("float32")
        except Exception:
            # scipy optional: a tiny separable box approximation keeps it dep-free.
            import numpy as np

            k = max(1, int(round(sigma)))
            m = mask01.astype("float32")
            for _ in range(2):
                m = np.pad(m, ((k, k), (k, k)), mode="edge")
                acc = np.zeros_like(mask01, dtype="float32")
                for dy in range(-k, k + 1):
                    for dx in range(-k, k + 1):
                        acc += m[k + dy : k + dy + mask01.shape[0],
                                 k + dx : k + dx + mask01.shape[1]]
                m = acc / ((2 * k + 1) ** 2)
            return m.astype("float32")

    def _coverage(self, mask01: Any) -> float:
        import numpy as np

        a = np.asarray(mask01)
        if a.size == 0:
            return 0.0
        try:
            import cv2

            mask = (a > 0.5).astype("uint8")
            return float(cv2.countNonZero(mask) / mask.size)
        except Exception:
            return float((a > 0.5).mean())

    def _select_aspects(self, seed: int, region_local: bool) -> List[str]:
        """Pick which of L/GC/SC this sample edits. Region-local oversamples SC
        (aspect_mix.region_local_sc_fraction; P0_001 §5)."""
        rng = _det_rng(seed, "aspects")
        combos = self.ctx.degrade_cfg.get(
            "combos", ["L", "GC", "SC", "L+GC", "L+SC", "GC+SC", "L+GC+SC"]
        )
        if region_local:
            sc_frac = float(self.ctx.aspect_mix_cfg.get("region_local_sc_fraction", 0.5))
            if rng.random() < sc_frac:
                sc_combos = [c for c in combos if "SC" in c] or ["SC"]
                return _stable_choice(sc_combos, f"{seed}:sc").split("+")
        return _stable_choice(combos, f"{seed}:any").split("+")

    def _build_cgt(
        self,
        sample_id: str,
        shard: str,
        source: SourceItem,
        params: Optional[Dict[str, Dict[str, float]]],
        region_local: bool,
        seed: int,
        *,
        concepts: Sequence[str] = (),
        degrade_spec: Optional[DegradeSpec] = None,
        mask_source: MaskSource = MaskSource.GLOBAL,
        exact_mask: Optional[Any] = None,
        recipe_key: Optional[str] = None,
    ) -> CgtRef:
        """Synthesize the Direction-A C_GT and (optionally) persist it.

        - GLOBAL streams (S5/S6/S7): C_GT == all-ones sentinel; no PNG written
          (DATASET_BUILD_PLAN §6 — one shared sentinel for 400k global samples).
        - Region-local with an EXACT mask (S1 degrade / S2-S3 composite mask):
          C_GT = M_r * g(||dtheta||), soft-blurred, persisted as native PNG +
          patch-grid npy.
        - mask_source records provenance (sam3 / degrade / ppr10k / global).
        """
        native = self._native_size(source)
        aspect_mag = self._aspect_magnitudes_for_recipe(params, degrade_spec, recipe_key)

        if not region_local:
            return CgtRef(
                mask_source=MaskSource.GLOBAL,
                concepts=list(concepts),
                aspect_magnitude=aspect_mag,
                coverage=1.0,
            )

        # ---- region-local: obtain the base mask M_r in [0,1] HxW ----
        import numpy as np

        mask: Optional[Any] = None
        used_source = mask_source

        if exact_mask is not None:
            mask = np.asarray(exact_mask, dtype="float32")
            used_source = mask_source if mask_source != MaskSource.GLOBAL else MaskSource.DEGRADE
        elif degrade_spec is not None:
            mask = self._degrade_mask(native, degrade_spec, seed)
            used_source = MaskSource.DEGRADE
        elif concepts and self.ctx.masker is not None:
            # SAM3 over the source: reduce concepts to the single soft union map.
            try:
                cmaps = self.ctx.masker.masks(source.path, list(concepts), native_size=native)
                stacked = [np.asarray(v, dtype="float32") for v in cmaps.values() if v is not None]
                if stacked:
                    mask = np.maximum.reduce(stacked)
                    used_source = mask_source
            except Exception:
                mask = None

        if mask is None:
            # Could not obtain a region mask -> degrade to global (honesty: DATASET
            # §5, C_GT is an optional accelerant). QAGate's coverage gate is skipped.
            return CgtRef(
                mask_source=MaskSource.GLOBAL,
                concepts=list(concepts),
                aspect_magnitude=aspect_mag,
                coverage=1.0,
                meta={"degraded_to_global": "no_mask"},
            ) if False else CgtRef(  # noqa: keep explicit branch readable
                mask_source=MaskSource.GLOBAL,
                concepts=list(concepts),
                aspect_magnitude=aspect_mag,
                coverage=1.0,
            )

        mask = np.clip(mask, 0.0, 1.0).astype("float32")

        # ---- Mode-1 magnitude weighting: C_GT = M_r * g(||dtheta||) ----
        tau = float(self.ctx.cgt_cfg.get("magnitude_tau", 1.0))
        if aspect_mag:
            edited = [a for a in ASPECTS if aspect_mag.get(a, 0.0) > 0.0] or list(ASPECTS)
            gnorm = math.sqrt(sum(aspect_mag.get(a, 0.0) ** 2 for a in edited))
            g = magnitude_weight(gnorm, tau)
        else:
            g = 1.0  # degrade samples: full strength inside the mask
        cgt = (mask * g).astype("float32")
        cgt = self._soft_blur(cgt)
        coverage = self._coverage(mask)

        # ---- persist (PNG native + patch-grid npy) via the injected sink ----
        cgt_path: Optional[str] = None
        patch_path: Optional[str] = None
        raw_mask_path: Optional[str] = None
        if self.ctx.cgt_writer is not None:
            try:
                write_out = self.ctx.cgt_writer.write_cgt(
                    self.stream_id, shard, sample_id, cgt,
                    patch_grid=int(self.ctx.cgt_cfg.get("patch_grid", 16)),
                    raw_mask01=mask,
                )
                cgt_path, patch_path = write_out[:2]
                raw_mask_path = write_out[2] if len(write_out) >= 3 else None
            except Exception:
                cgt_path, patch_path, raw_mask_path = None, None, None

        return CgtRef(
            cgt_path=cgt_path,
            cgt_patchgrid_path=patch_path,
            raw_mask_path=raw_mask_path,
            mask_source=used_source,
            concepts=list(concepts),
            mask_score=float(np.max(mask)) if mask.size else None,
            aspect_magnitude=aspect_mag,
            coverage=coverage,
            soft_blur_sigma_px=float(self.ctx.cgt_cfg.get("soft_blur_sigma_px", 0.0) or 0.0),
            magnitude_tau=float(self.ctx.cgt_cfg.get("magnitude_tau", 1.0) or 1.0),
        )

    def _aspect_magnitudes_for_recipe(
        self,
        params: Optional[Dict[str, Dict[str, float]]],
        degrade_spec: Optional[DegradeSpec],
        recipe_key: Optional[str],
    ) -> Dict[str, float]:
        if not params:
            return {}
        # O2: cache reused recipe magnitudes by stable source_recipe_id, but skip
        # degrade specs because their generated params vary per sample without a
        # stable recipe asset id.
        if degrade_spec is None and recipe_key:
            cached = self.ctx.aspect_magnitude_cache.get(recipe_key)
            if cached is not None:
                return dict(cached)
            mag = param_delta_aspect_magnitudes(params)
            self.ctx.aspect_magnitude_cache[recipe_key] = dict(mag)
            return mag
        return param_delta_aspect_magnitudes(params)

    # ----------------------------------------------------------- annotation
    def _annotate(
        self,
        source: SourceItem,
        recipe: Recipe,
        sample: Sample,
        after_rgb: Optional[Any],
    ) -> None:
        """Fill instruction/think/answer/scene + judge scores via the cleaner.

        Degrades gracefully when the cleaner is absent (dry-run): instruction is
        synthesized from scene + aspects, answer mirrors recipe.params, and the
        judge scores are left None (so the gate does not reject on missing data).
        """
        cl = self.ctx.cleaner
        if cl is None:
            self._offline_annotate(source, recipe, sample)
            return
        mode = str(self.ctx.qa_cfg.get("annotation_mode", "vlm")).strip().lower()
        if mode in {"template", "offline", "deterministic"}:
            self._offline_annotate(source, recipe, sample)
            self._maybe_verify(source, recipe, sample, after_rgb)
            return
        if not sample.instruction:
            try:
                vlm_path = self._vlm_image_path(source, recipe, sample)
                gi = cl.gen_instruction(vlm_path, scene_meta=sample.scene_meta.__dict__)
                sample.instruction = gi.get("instruction_long")
                sample.instruction_short = gi.get("instruction_short")
                sample.scene_meta.lang = gi.get("lang", sample.scene_meta.lang)
            except Exception:
                self._offline_annotate(source, recipe, sample)

        instr = sample.instruction or ""
        if not sample.think:
            try:
                vlm_path = self._vlm_image_path(source, recipe, sample)
                rp = cl.reason_params(vlm_path, instr)
                sample.think = rp.get("think")
                if self.ctx.qa_cfg.get("vlm_override_answer", False) or sample.answer is None:
                    sample.answer = rp.get("answer") or sample.answer
            except Exception:
                pass

        # judge: wire the teacher 'after' into cleaner.verify (persists a transient
        # downscaled before+after JPEG, deletes it after the judge reads it). Gated
        # by qa.enable_verify; in "log" mode it only records scores (no rejection).
        self._maybe_verify(source, recipe, sample, after_rgb)

        # tag scene/region (refines region hints). This is the 3rd image-bearing
        # vLLM call/sample; region tags only matter for region-local streams, so we
        # skip it for GLOBAL streams (S5/S6/S7) unless qa.tag_global. Big clean-stage
        # saving on the 400k global samples (vision-encode is the bottleneck).
        already_tagged = bool(sample.meta.get("vlm_region_tag"))
        if sample.region_local and not already_tagged:
            try:
                tg = self._region_vlm_tag(source, recipe, sample, instr)
                self._apply_vlm_tag(sample, tg, "vlm_region_tag")
            except Exception:
                pass
        elif (not sample.region_local) and self.ctx.qa_cfg.get("tag_global", False):
            try:
                tg = self._source_vlm_tag(source, recipe, sample)
                self._apply_vlm_tag(sample, tg, "vlm_source_tag")
            except Exception:
                pass

    def _vlm_region_concepts(
        self,
        source: SourceItem,
        recipe: Recipe,
        sample: Sample,
    ) -> List[str]:
        """Use VLM region tagging before SAM3 mask construction.

        This is where VLM improves C_GT quality for S2/S3: the returned
        sam3_concepts drive the construction mask instead of being only
        post-hoc metadata. Deterministic/template modes skip this path.
        """
        cl = self.ctx.cleaner
        mode = str(self.ctx.qa_cfg.get("annotation_mode", "vlm")).strip().lower()
        if cl is None or mode in {"template", "offline", "deterministic", "none"}:
            return []
        try:
            vlm_path = self._vlm_image_path(source, recipe, sample)
            if not sample.instruction:
                gi = cl.gen_instruction(vlm_path, scene_meta=sample.scene_meta.__dict__)
                sample.instruction = gi.get("instruction_long")
                sample.instruction_short = gi.get("instruction_short")
                sample.scene_meta.lang = gi.get("lang", sample.scene_meta.lang)
            tg = self._region_vlm_tag(source, recipe, sample, sample.instruction or "")
            self._apply_vlm_tag(sample, tg, "vlm_region_tag")
            concepts = tg.get("sam3_concepts", []) or []
            return [str(c).strip() for c in concepts if str(c).strip()]
        except Exception:
            return []

    def _offline_annotate(self, source: SourceItem, recipe: Recipe, sample: Sample) -> None:
        """Cleaner-free fallback annotation (dry-run / cleaner disabled)."""
        if sample.answer is None and recipe.params is not None:
            sample.answer = recipe.params
        if not sample.instruction:
            scene = sample.scene_meta.scene or scene_of(source)
            style = sample.scene_meta.style or recipe.meta.get("style")
            if not style:
                pack = str(recipe.meta.get("pack_id", "") or "")
                style = pack.rsplit("/", 1)[-1].lower() if pack else "balanced"
            if recipe.kind == RecipeKind.LUT:
                sample.instruction = f"Apply the stored {style} color look to this {scene} photo."
                sample.instruction_short = f"{style} color look"
            elif recipe.kind == RecipeKind.PARAM:
                sample.instruction = f"Apply the stored {style} retouching parameters to this {scene} photo."
                sample.instruction_short = f"{style} retouch"
            else:
                sample.instruction = f"Apply a {style} edit to this {scene} photo."
                sample.instruction_short = f"{style} {scene} edit"
        if not sample.think:
            sample.think = "Apply the stored recipe parameters to produce the target edit."

    # --------------------------------------------------------------- build_one
    def build_one(
        self,
        source: SourceItem,
        recipe: Recipe,
        region_local: bool,
        sample_id: str,
        shard: str,
        precomputed_after: Any = _UNSET,
    ) -> Optional[Sample]:
        """Default param/LUT pipeline; degrade & MMArt streams override pieces.

        Steps: resolve params -> [render after (teacher)] -> build C_GT ->
        annotate (+verify judge) -> gate. The GPU 'after' render is normally
        HOISTED to a single batched call in run.run_stream and threaded in via
        ``precomputed_after`` (so build_one is GPU-free and safe to run across the
        concurrent clean pool). When ``precomputed_after`` is unset (the legacy
        run()/test path) it falls back to a single-sample render here.
        """
        seed = int(self.ctx.config.get("seed", 0))
        params = self._resolve_params(recipe)

        sample = Sample(
            sample_id=sample_id,
            stream=self.stream_id,
            shard=shard,
            source_path=source.path,
            raw_decode=source.raw_decode,
            recipe=recipe,
            region_local=region_local,
            c_gt=CgtRef(),  # filled below
            answer=params if recipe.kind == RecipeKind.PARAM else None,
            scene_meta=SceneMeta(scene=scene_of(source)),
            quality=QualityScores(),
            source_id=source.source_id,
            recipe_asset_id=recipe.source_recipe_id,
            native_size=self._native_size(source),
            build_version=str(self.ctx.config.get("build_version", "v2")),
            schema_version=str(self.ctx.config.get("schema_version", "datagen_v2")),
            after_source=AfterSource.TEACHER,
        )

        # S5 real-JPG: record the gold expert 'after' path (set by Tier1ExpertStream)
        # so training loads the real JPG instead of re-rendering the XMP.
        if recipe.meta.get("expert_after_jpg"):
            sample.meta["expert_after_path"] = recipe.meta["expert_after_jpg"]
            sample.meta["after_source"] = "real_jpg"
            sample.after_source = AfterSource.REAL_JPG

        # global 'after' (teacher): from the batched render stage when provided,
        # else a single-sample render (legacy run()/test path).
        if precomputed_after is _UNSET:
            if recipe.kind == RecipeKind.LUT:
                after_rgb = self._apply_lut_after(source, recipe)
            else:
                after_rgb = self._render_after(source, params) if params is not None else None
        else:
            after_rgb = precomputed_after

        # Concepts for region-local masking. Quality-first path: let the VLM tag
        # the target region BEFORE SAM3 runs; fallback to the fixed SC vocab.
        if region_local:
            concepts = self._vlm_region_concepts(source, recipe, sample)
            if not concepts:
                concepts = self._region_concepts(seed, sample_id)
        else:
            concepts = []

        sample.c_gt = self._build_cgt(
            sample_id, shard, source, params, region_local, seed=int(seed) ^ _det_seed32(sample_id),
            concepts=concepts,
            mask_source=MaskSource.SAM3 if region_local else MaskSource.GLOBAL,
            recipe_key=recipe.source_recipe_id,
        )
        # mask_score -> mask_quality for the gate
        if sample.c_gt.mask_score is not None:
            sample.quality.mask_quality = sample.c_gt.mask_score

        self._annotate(source, recipe, sample, after_rgb)

        if not self._gate(sample):
            return None
        return sample

    # ----------------------------------------------------- subclass hooks
    def _resolve_params(self, recipe: Recipe) -> Optional[Dict[str, Dict[str, float]]]:
        """Return the param dict the teacher consumes, or None for non-param kinds.
        LUT kind has no params (renderer is param-mode); the LUT is applied by the
        LutApplier in run.py, so we pass identity params to skip the teacher."""
        if recipe.kind == RecipeKind.PARAM:
            return recipe.params or _empty_params()
        # LUT / DEGRADE: teacher renders identity (LUT applied separately; degrade
        # handled by DegradeStream.build_one override).
        return _empty_params()

    def _region_concepts(self, seed: int, sample_id: str) -> List[str]:
        """SC-leaning concept list for SAM3 region masking on param/LUT streams."""
        cm = self.ctx.concept_map
        vocab = cm.get("SC") or self.ctx.cgt_cfg.get("concept_map", {}).get("SC", [])
        if not vocab:
            return ["the main subject"]
        rng = _det_rng(seed, sample_id, "concepts")
        n = min(3, len(vocab))
        return rng.sample(list(vocab), n)

    # ------------------------------------------------------------------- run
    def run(
        self, budget: int, seed: int, done_ids: Iterable[str]
    ) -> Iterator[Sample]:
        done = set(done_ids)
        for pos, (source, recipe, region_local) in enumerate(self.plan(budget, seed)):
            key = recipe.source_recipe_id or recipe.kind.value
            # id derived from the DETERMINISTIC plan position so a resumed run
            # (same seed/plan) skips already-committed work idempotently.
            sid = make_sample_id(self.stream_id, source.source_id, key, pos)
            if sid in done:
                continue
            shard = "pending"  # ShardWriter assigns the real shard on write
            sample = self.build_one(source, recipe, region_local, sid, shard)
            if sample is not None:
                done.add(sample.sample_id)
                yield sample

    # plan() is abstract here (inherited from contracts.Stream)
    @abc.abstractmethod
    def plan(
        self, budget: int, seed: int
    ) -> Iterator[Tuple[SourceItem, Recipe, bool]]: ...


# ===========================================================================
# Concrete streams.
# ===========================================================================


class _PlanInputs:
    """A stream's planning inputs, resolved by run.py from the registry indexes.

    Attached to the stream before ``plan``/``run``. Holding it here (rather than
    in BuildContext) lets each stream own its own scene-matched pools.
    """

    def __init__(
        self,
        sources: Sequence[SourceItem],
        recipes: Sequence[RecipeAsset] = (),
        mmart_records: Sequence[Dict[str, Any]] = (),
    ):
        self.sources = list(sources)
        self.recipes = list(recipes)
        self.mmart_records = list(mmart_records)


class RecipeXSourceStream(BaseStream):
    """S2 (region-local) & S6 (global): apply a 全店素材/E18/GREYSKY look (param via
    the teacher, or LUT) to a scene-matched source.

    S2 composites the look inside a SAM3 mask (C_GT = mask); S6 is full-frame
    (C_GT = all-ones). Scene-matching (probe_sources_budget §3): a portrait
    recipe routes to a portrait source, scenery to scenery, ``any`` to anything.
    """

    def __init__(
        self,
        ctx: BuildContext,
        stream_id: StreamId,
        inputs: _PlanInputs,
        region_local: bool,
        gate: Optional[QAGate] = None,
    ):
        super().__init__(ctx, gate)
        self.stream_id = stream_id
        self.inputs = inputs
        self.region_local = region_local

    def _recipe_to_obj(self, asset: RecipeAsset) -> Recipe:
        if asset.kind == RecipeKind.LUT:
            meta = {"fmt": asset.fmt, "lut_size": asset.lut_size, "path": asset.path}
            if self.ctx.parser is not None:
                try:
                    if asset.recipe_id in self.ctx.lut_cache:
                        lut, dmin, dmax = self.ctx.lut_cache[asset.recipe_id]
                    else:
                        lut, dmin, dmax = self.ctx.parser.load_cube(asset.path)
                        self.ctx.lut_cache[asset.recipe_id] = (lut, dmin, dmax)
                    digest = _file_sha256(asset.path)
                    meta.update(
                        {
                            "lut_size": int(getattr(lut, "shape", [asset.lut_size])[0]),
                            "domain_min": list(dmin),
                            "domain_max": list(dmax),
                            "lut_sha256": digest,
                        }
                    )
                except Exception as e:  # noqa: BLE001 - malformed LUT skipped by plan()
                    raise ValueError(f"invalid LUT {asset.path}: {e}") from e
            return Recipe(
                kind=RecipeKind.LUT,
                lut_recipe_id=asset.recipe_id,
                provenance=Provenance.LUT,
                source_recipe_id=asset.recipe_id,
                meta=meta,
            )
        # PARAM: parse the xmp/lrtemplate now (CPU; parser injected) so the
        # recipe is fully self-describing for re-rendering.
        params: Optional[Dict[str, Dict[str, float]]] = None
        if self.ctx.parser is not None:
            try:
                if asset.fmt == "xmp":
                    params = self.ctx.parser.xmp_to_params(asset.path)
                elif asset.fmt == "lrtemplate":
                    params = self.ctx.parser.lrtemplate_to_params(asset.path)
            except Exception:
                params = None
        return Recipe(
            kind=RecipeKind.PARAM,
            params=params or _empty_params(),
            provenance=Provenance.PARAM,
            source_recipe_id=asset.recipe_id,
            meta={"fmt": asset.fmt, "pack_id": asset.pack_id, "style": asset.style},
        )

    def plan(self, budget: int, seed: int) -> Iterator[Tuple[SourceItem, Recipe, bool]]:
        sources = self.inputs.sources
        recipes = [r for r in self.inputs.recipes if not r.is_bw and not r.is_technical]
        if not sources or not recipes:
            return
        # group sources by scene for scene-matched routing.
        by_scene: Dict[str, List[SourceItem]] = {}
        for s in sources:
            by_scene.setdefault(scene_of(s), []).append(s)
        any_pool = sources

        emitted = 0
        ri = 0
        rng = _det_rng(seed, self.stream_id.value, "plan")
        # deterministic round-robin over recipes, scene-matched source draw.
        order = list(range(len(recipes)))
        rng.shuffle(order)
        while emitted < budget:
            asset = recipes[order[ri % len(order)]]
            ri += 1
            aff = (asset.scene_affinity or "any").lower()
            pool = by_scene.get(aff) if aff != "any" else None
            pool = pool or any_pool
            src = pool[_det_rng(seed, asset.recipe_id, emitted).randrange(len(pool))]
            if not _scene_compatible(scene_of(src), asset.scene_affinity):
                # soft skip: still allow ``any`` recipes everywhere
                if aff != "any":
                    continue
            try:
                recipe = self._recipe_to_obj(asset)
            except Exception as e:  # noqa: BLE001 - malformed recipes are skipped at build
                print(f"[stream {self.stream_id}] skipping recipe {asset.recipe_id}: {e}",
                      file=sys.stderr)
                continue
            yield src, recipe, self.region_local
            emitted += 1


class Tier1ExpertStream(BaseStream):
    """S5: GREYSKY real-expert GLOBAL triples (and FiveK/PPR10K expert XMP banks).

    These are the gold real-expert pairs: real before -> expert after via the
    expert XMP rendered by the teacher; C_GT = all-ones (global). Sources are
    reused across exposure jitter as augmentation views (recipe still renders at
    native res; jitter is an extra view, never a stored crop — gotcha 10).
    """

    def __init__(
        self,
        ctx: BuildContext,
        inputs: _PlanInputs,
        gate: Optional[QAGate] = None,
    ):
        super().__init__(ctx, gate)
        self.stream_id = StreamId.S5_GREYSKY_GLOBAL
        self.inputs = inputs

    def plan(self, budget: int, seed: int) -> Iterator[Tuple[SourceItem, Recipe, bool]]:
        # pair each gold source with its attached expert XMP (recorded in meta),
        # else round-robin the GREYSKY xmp bank.
        sources = self.inputs.sources
        xmps = [r for r in self.inputs.recipes if r.kind == RecipeKind.PARAM]
        if not sources:
            return
        emitted = 0
        si = 0
        while emitted < budget:
            src = sources[si % len(sources)]
            si += 1
            # prefer a paired XMP recorded on the source (registry sets meta)
            paired = src.meta.get("expert_xmp") if src.meta else None
            params: Optional[Dict[str, Dict[str, float]]] = None
            rid: Optional[str] = None
            if paired and self.ctx.parser is not None:
                try:
                    params = self.ctx.parser.xmp_to_params(paired)
                    rid = paired
                except Exception:
                    params = None
            if params is None and xmps:
                asset = _stable_choice(xmps, f"{src.source_id}:{emitted}")
                rid = asset.recipe_id
                if self.ctx.parser is not None:
                    try:
                        params = self.ctx.parser.xmp_to_params(asset.path)
                    except Exception:
                        params = None
            rmeta: Dict[str, Any] = {"tier": "gold", "expert": True}
            # Real expert JPG present (paired by the greysky index): use it as the
            # gold 'after' and SKIP the teacher render entirely. The XMP params stay
            # in the recipe so training can re-render OR load the real JPG from meta.
            real_jpg = (src.meta or {}).get("expert_after_jpg")
            if real_jpg:
                rmeta["expert_after_jpg"] = real_jpg
                rmeta["use_real_jpg"] = True
            recipe = Recipe(
                kind=RecipeKind.PARAM,
                params=params or _empty_params(),
                provenance=Provenance.PARAM,
                source_recipe_id=rid,
                meta=rmeta,
            )
            yield src, recipe, False  # global
            emitted += 1

    def _params_for_render(self, recipe: Recipe) -> Optional[Dict[str, Dict[str, float]]]:
        # Gold triples with a real expert JPG: the JPG IS the 'after' (recorded in
        # sample.meta for training); skip the teacher render entirely. Otherwise
        # render the expert XMP via the teacher like any PARAM recipe.
        if recipe.meta.get("use_real_jpg"):
            return None
        return super()._params_for_render(recipe)


class FivekGoldStream(BaseStream):
    """S8: fivek GOLD GLOBAL real before -> real expert-after pairs.

    Mirrors :class:`Tier1ExpertStream` (S5) gold real-JPG path but for the fivek
    ``fivek_gold`` corpus: I_in = before.jpg, I_tar = the real expert processed.jpg
    (after_source=REAL_JPG; the JPG IS the 'after', NO teacher render, NO LUT). The
    sample instruction comes from the en/ user_prompt text recorded on the source.
    The old-Lightroom config.lua params are NOT teacher CRS2012 params, so the recipe
    carries an EMPTY (identity) param dict and the .lua path is metadata only.
    C_GT = global all-ones sentinel (region_local=False).
    """

    def __init__(
        self,
        ctx: BuildContext,
        inputs: _PlanInputs,
        gate: Optional[QAGate] = None,
    ):
        super().__init__(ctx, gate)
        self.stream_id = StreamId.S8_FIVEK_GLOBAL
        self.inputs = inputs

    def plan(self, budget: int, seed: int) -> Iterator[Tuple[SourceItem, Recipe, bool]]:
        sources = self.inputs.sources
        if not sources:
            return
        emitted = 0
        si = 0
        while emitted < budget:
            src = sources[si % len(sources)]
            si += 1
            smeta = src.meta or {}
            recipe = Recipe(
                kind=RecipeKind.PARAM,
                params=_empty_params(),
                provenance=Provenance.PARAM,
                source_recipe_id=src.source_id,
                meta={
                    "tier": "gold",
                    "expert": True,
                    "origin": "fivek_gold",
                    # real expert JPG -> the gold 'after'; skips teacher render (S5 path)
                    "expert_after_jpg": smeta.get("expert_after_jpg"),
                    "use_real_jpg": True,
                    # old-Lightroom .lua params: METADATA ONLY (not teacher CRS2012 space)
                    "fivek_params_lua": smeta.get("fivek_params_lua"),
                    "instruction": smeta.get("instruction"),
                    "instruction_short": smeta.get("instruction_short"),
                },
            )
            yield src, recipe, False  # global
            emitted += 1

    def _params_for_render(self, recipe: Recipe) -> Optional[Dict[str, Dict[str, float]]]:
        # Real expert JPG IS the 'after'; never render the teacher for S8 gold rows.
        return None

    def build_one(
        self,
        source: SourceItem,
        recipe: Recipe,
        region_local: bool,
        sample_id: str,
        shard: str,
        precomputed_after: Any = _UNSET,
    ) -> Optional[Sample]:
        seed = int(self.ctx.config.get("seed", 0))
        sample = Sample(
            sample_id=sample_id,
            stream=self.stream_id,
            shard=shard,
            source_path=source.path,
            raw_decode=source.raw_decode,
            recipe=recipe,
            region_local=False,
            c_gt=CgtRef(),
            instruction=recipe.meta.get("instruction"),
            instruction_short=recipe.meta.get("instruction_short"),
            answer=None,  # no teacher params for gold real-JPG rows
            scene_meta=SceneMeta(scene=scene_of(source)),
            quality=QualityScores(),
            source_id=source.source_id,
            recipe_asset_id=recipe.source_recipe_id,
            native_size=self._native_size(source),
            build_version=str(self.ctx.config.get("build_version", "v2")),
            schema_version=str(self.ctx.config.get("schema_version", "datagen_v2")),
            after_source=AfterSource.REAL_JPG,
        )
        # Real expert JPG bypass (reproduce.py REAL_JPG branch loads this path).
        real_jpg = recipe.meta.get("expert_after_jpg")
        if real_jpg:
            sample.meta["expert_after_path"] = real_jpg
            sample.meta["after_source"] = "real_jpg"
        # carry the old-Lightroom params path as metadata only (never teacher params).
        if recipe.meta.get("fivek_params_lua"):
            sample.meta["fivek_params_lua"] = recipe.meta["fivek_params_lua"]
        # global C_GT sentinel (no PNG; all-ones).
        sample.c_gt = self._build_cgt(
            sample_id, shard, source, params=None, region_local=False,
            seed=int(seed) ^ _det_seed32(sample_id),
            mask_source=MaskSource.GLOBAL,
            recipe_key=recipe.source_recipe_id,
        )
        # annotate (instruction already from en/ text; cleaner may enrich think/scene).
        self._annotate(source, recipe, sample, after_rgb=None)
        if not self._gate(sample):
            return None
        return sample


class MMArtTextStream(BaseStream):
    """S3: MMArt text-attached, region-local.

    Each record = ``before.jpg`` + instruction + ``<think>`` + global ``<answer>``
    params. We RE-RENDER the after from the GLOBAL answer params via the teacher
    (caveat a — never trust processed.jpg without the verify gate), then
    composite inside the instruction-referenced region (SAM3 on instruction
    nouns / masksubtype_hint); C_GT = that mask.
    """

    def __init__(
        self,
        ctx: BuildContext,
        inputs: _PlanInputs,
        gate: Optional[QAGate] = None,
    ):
        super().__init__(ctx, gate)
        self.stream_id = StreamId.S3_MMART_LOCAL
        self.inputs = inputs

    def plan(self, budget: int, seed: int) -> Iterator[Tuple[SourceItem, Recipe, bool]]:
        recs = self.inputs.mmart_records
        srcs_by_id = {s.source_id: s for s in self.inputs.sources}
        if not recs:
            return
        emitted = 0
        for rec in recs:
            if emitted >= budget:
                break
            src = self._record_source(rec, srcs_by_id)
            if src is None:
                continue
            answer = self._parse_answer(rec)
            recipe = Recipe(
                kind=RecipeKind.PARAM,
                params=answer or _empty_params(),
                provenance=Provenance.PARAM,
                source_recipe_id=rec.get("id"),
                meta={
                    "origin": "mmart",
                    "instruction": rec.get("instruction"),
                    "think": rec.get("think"),
                    "masksubtype_hint": rec.get("masksubtype_hint", 0),
                },
            )
            yield src, recipe, True
            emitted += 1

    def _record_source(
        self, rec: Dict[str, Any], srcs_by_id: Dict[str, SourceItem]
    ) -> Optional[SourceItem]:
        # registry maps each MMArt record to a SourceItem (rebased before.jpg).
        sid = rec.get("source_id")
        if sid and sid in srcs_by_id:
            return srcs_by_id[sid]
        path = rec.get("before_path") or rec.get("image")
        if not path:
            return None
        return SourceItem(
            source_id=sid or hashlib.sha1(str(path).encode()).hexdigest()[:16],
            path=path,
            corpus="mmart",
            scene=rec.get("scene", "any"),
        )

    @staticmethod
    def _parse_answer(rec: Dict[str, Any]) -> Optional[Dict[str, Dict[str, float]]]:
        """The <answer> params, normalized to {key:{"value":raw}}. MMArt answers
        are single-quoted python-dicts; the registry/cleaner already parse them,
        but we defensively re-normalize here."""
        ans = rec.get("answer")
        if not isinstance(ans, dict):
            return None
        out: Dict[str, Dict[str, float]] = {}
        for k, v in ans.items():
            if k not in PARAM_KEYS:
                continue
            if isinstance(v, dict) and "value" in v:
                out[k] = {"value": float(v["value"])}
            else:
                try:
                    out[k] = {"value": float(v)}
                except (TypeError, ValueError):
                    continue
        return out or None

    def build_one(
        self,
        source: SourceItem,
        recipe: Recipe,
        region_local: bool,
        sample_id: str,
        shard: str,
        precomputed_after: Any = _UNSET,
    ) -> Optional[Sample]:
        # Reuse the base param pipeline but seed instruction/think from the record
        # and drive SAM3 concepts from the instruction nouns / masksubtype hint.
        seed = int(self.ctx.config.get("seed", 0))
        params = recipe.params or _empty_params()
        sample = Sample(
            sample_id=sample_id,
            stream=self.stream_id,
            shard=shard,
            source_path=source.path,
            raw_decode=source.raw_decode,
            recipe=recipe,
            region_local=region_local,
            c_gt=CgtRef(),
            instruction=recipe.meta.get("instruction"),
            think=recipe.meta.get("think"),
            answer=params,
            scene_meta=SceneMeta(
                scene=scene_of(source),
                masksubtype_hint=int(recipe.meta.get("masksubtype_hint", 0) or 0),
            ),
            quality=QualityScores(),
            source_id=source.source_id,
            recipe_asset_id=recipe.source_recipe_id,
            native_size=self._native_size(source),
            build_version=str(self.ctx.config.get("build_version", "v2")),
            schema_version=str(self.ctx.config.get("schema_version", "datagen_v2")),
            after_source=AfterSource.TEACHER,
        )
        if precomputed_after is _UNSET:
            after_rgb = self._render_after(source, params)
        else:
            after_rgb = precomputed_after
        concepts = self._vlm_region_concepts(source, recipe, sample) or self._mmart_concepts(sample)
        sample.c_gt = self._build_cgt(
            sample_id, shard, source, params, region_local,
            seed=int(seed) ^ _det_seed32(sample_id),
            concepts=concepts, mask_source=MaskSource.SAM3,
            recipe_key=recipe.source_recipe_id,
        )
        if sample.c_gt.mask_score is not None:
            sample.quality.mask_quality = sample.c_gt.mask_score
        # MMArt processed.jpg sanity: only the cleaner's verify can set processed_ok.
        if self.ctx.cleaner is not None and after_rgb is None:
            sample.quality.processed_ok = None
        self._annotate(source, recipe, sample, after_rgb)
        if not self._gate(sample):
            return None
        return sample

    def _mmart_concepts(self, sample: Sample) -> List[str]:
        # masksubtype_hint: 1=Subject 2=Sky 3=Person 0=global (probe_vllm_clean).
        hint = sample.scene_meta.masksubtype_hint
        cm = self.ctx.concept_map
        if hint == 2:
            return cm.get("GC", ["sky"])
        if hint in (1, 3):
            return cm.get("SC", ["person", "subject"])
        # else derive from the SC vocab (the most-local aspect)
        return cm.get("SC", ["the main subject"])[:3]


class InverseDegradeStream(BaseStream):
    """S1 (region-local, EXACT mask) & S7 (global): Track-B Gaussian-operator
    inverse-degradation.

    The 'before' is the degraded image, the 'after' is the clean source (or
    vice-versa). For S1 the degradation is applied ONLY inside an exact
    geometric/SAM3 mask M_r, and that mask is the EXACT C_GT (no segmentation
    noise — the strongest Direction-A signal). Self-supervised: needs no XMP and
    no segmentation model for the geometric shapes.
    """

    def __init__(
        self,
        ctx: BuildContext,
        stream_id: StreamId,
        inputs: _PlanInputs,
        region_local: bool,
        gate: Optional[QAGate] = None,
    ):
        super().__init__(ctx, gate)
        self.stream_id = stream_id
        self.inputs = inputs
        self.region_local = region_local

    def plan(self, budget: int, seed: int) -> Iterator[Tuple[SourceItem, Recipe, bool]]:
        sources = self.inputs.sources
        if not sources:
            return
        emitted = 0
        si = 0
        while emitted < budget:
            src = sources[si % len(sources)]
            si += 1
            aspects = self._select_aspects(seed ^ (emitted * 2654435761 & 0xFFFFFFFF), self.region_local)
            spec = self._make_spec(aspects, seed, emitted)
            recipe = Recipe(
                kind=RecipeKind.PARAM,
                params=_params_from_degrade_spec(spec),
                provenance=Provenance.DEGRADE,
                degrade=spec,
                meta={"aspects": aspects},
            )
            yield src, recipe, self.region_local
            emitted += 1

    def _make_spec(self, aspects: List[str], seed: int, idx: int) -> DegradeSpec:
        if self.ctx.parser is not None:
            try:
                return self.ctx.parser.sample_degrade_spec(
                    aspects, seed=seed ^ idx, region_local=self.region_local
                )
            except Exception:
                pass
        # parser-free fallback (dry-run): a minimal valid spec.
        return DegradeSpec(
            mode="gaussian_op",
            aspects=list(aspects),
            sigma_profile=self.ctx.degrade_cfg.get("sigma_profile", "aether_tab8"),
            forward=False,
            seed=seed ^ idx,
        )

    @staticmethod
    def _aspect_label(aspects: Sequence[str]) -> str:
        labels = {"L": "lighting", "GC": "global color", "SC": "specific colors"}
        picked = [labels.get(a, a) for a in aspects if a]
        return ", ".join(picked) if picked else "photo tone"

    def _offline_degrade_annotate(self, source: SourceItem, recipe: Recipe, sample: Sample) -> None:
        """Deterministic S1/S7 annotation.

        Inverse-degrade streams are self-supervised: the op_params and exact/global
        mask are already known from construction, so VLM image cleaning adds no
        required label and can OOM on giant native assets.
        """
        spec = recipe.degrade
        aspects = list((spec.aspects if spec else None) or recipe.meta.get("aspects", []) or [])
        aspect_text = self._aspect_label(aspects)
        scope = "the masked region" if sample.region_local else "the full image"
        scene = sample.scene_meta.scene or scene_of(source)
        sample.instruction = f"Restore the {aspect_text} of {scope} in this {scene} photo."
        sample.instruction_short = f"restore {aspect_text}"
        sample.think = (
            "This is a synthetic inverse-degradation sample; use the stored "
            "teacher-rendered degraded input and search the best renderer correction."
        )
        if sample.answer is None and spec is not None and spec.op_params:
            sample.answer = {
                k: {"value": float(v)} for k, v in spec.op_params.items() if k in PARAM_KEYS
            }

    def build_one(
        self,
        source: SourceItem,
        recipe: Recipe,
        region_local: bool,
        sample_id: str,
        shard: str,
        precomputed_after: Any = _UNSET,  # unused: degrade renders no teacher after
    ) -> Optional[Sample]:
        seed = int(self.ctx.config.get("seed", 0))
        spec = recipe.degrade
        params = recipe.params or (_params_from_degrade_spec(spec) if spec is not None else _empty_params())
        sample = Sample(
            sample_id=sample_id,
            stream=self.stream_id,
            shard=shard,
            source_path=source.path,
            raw_decode=source.raw_decode,
            recipe=recipe,
            region_local=region_local,
            c_gt=CgtRef(),
            scene_meta=SceneMeta(scene=scene_of(source)),
            quality=QualityScores(),
            source_id=source.source_id,
            native_size=self._native_size(source),
            build_version=str(self.ctx.config.get("build_version", "v2")),
            schema_version=str(self.ctx.config.get("schema_version", "datagen_v2")),
            after_source=AfterSource.TEACHER,
        )
        # answer params == +op_params warm-start; recipe.params is neg_p input render.
        if spec is not None and spec.op_params:
            sample.answer = {
                k: {"value": float(v)} for k, v in spec.op_params.items() if k in PARAM_KEYS
            }
        # EXACT construction mask (region-local) or all-ones (global).
        sample.c_gt = self._build_cgt(
            sample_id, shard, source,
            params=params, region_local=region_local,
            seed=int(seed) ^ _det_seed32(sample_id),
            degrade_spec=spec if region_local else None,
            mask_source=MaskSource.DEGRADE if region_local else MaskSource.GLOBAL,
            recipe_key=None,
        )
        after_rgb = None if precomputed_after is _UNSET else precomputed_after
        # exact masks are perfect -> mask_quality = 1.0 (no segmentation noise).
        if region_local:
            sample.quality.mask_quality = 1.0
        self._offline_degrade_annotate(source, recipe, sample)
        if after_rgb is not None:
            sample.meta["teacher_degraded_preview"] = True
        if not self._gate(sample):
            return None
        return sample


class PPR10KStream(BaseStream):
    """S4: PPR10K source + its OWN expert target XMP, C_GT = the real human mask.

    REVIVED (ppr10k now available; config models.ppr10k_masks.available=true). Each
    PPR10K source carries its own per-source target XMP and human mask on
    ``source.meta`` (registry ``_scan_ppr10k``):
      - meta['ppr10k_xmp']    -> teacher CRS2012 PARAM params (ctx.parser.xmp_to_params)
      - meta['ppr10k_mask']   -> the real human-region mask PNG (the EXACT C_GT)
      - meta['ppr10k_target'] -> the expert target PNG (provenance; I_tar is the
                                 teacher render of source+params, on the manifold)
    All 3 experts (a/b/c) are distinct sources (~8875 x 3 = 26,625). The XMP is the
    source's OWN target XMP (NOT the shared recipe_index pool). I_tar = teacher
    render(source, params); C_GT = the human mask (mask_source=PPR10K,
    mask_quality=1.0). region_local=True.
    """

    def __init__(
        self,
        ctx: BuildContext,
        inputs: _PlanInputs,
        gate: Optional[QAGate] = None,
    ):
        super().__init__(ctx, gate)
        self.stream_id = StreamId.S4_PPR10K_LOCAL
        self.inputs = inputs

    def plan(self, budget: int, seed: int) -> Iterator[Tuple[SourceItem, Recipe, bool]]:
        sources = self.inputs.sources
        if not sources:
            return
        emitted = 0
        si = 0
        while emitted < budget:
            src = sources[si % len(sources)]
            si += 1
            smeta = src.meta or {}
            xmp_path = smeta.get("ppr10k_xmp")
            if not xmp_path:
                continue
            # Parse the source's OWN target XMP into teacher CRS2012 PARAM params
            # (per-source, NOT the shared recipe_index pool).
            params: Optional[Dict[str, Dict[str, float]]] = None
            if self.ctx.parser is not None:
                try:
                    params = self.ctx.parser.xmp_to_params(xmp_path)
                except Exception:
                    params = None
            recipe = Recipe(
                kind=RecipeKind.PARAM,
                params=params or _empty_params(),
                provenance=Provenance.PPR10K,
                source_recipe_id=src.source_id,
                meta={
                    "origin": "ppr10k",
                    "expert": smeta.get("expert"),
                    "ppr10k_xmp": xmp_path,
                    "ppr10k_target": smeta.get("ppr10k_target"),
                },
            )
            yield src, recipe, True
            emitted += 1

    def _params_for_render(self, recipe: Recipe) -> Optional[Dict[str, Dict[str, float]]]:
        # PPR10K's after is not consumed by build_one (mask comes from the human
        # mask PNG); skip the teacher render. (S4 is currently disabled anyway.)
        return None

    def build_one(self, source, recipe, region_local, sample_id, shard, precomputed_after=_UNSET):  # type: ignore[override]
        seed = int(self.ctx.config.get("seed", 0))
        params = recipe.params or _empty_params()
        sample = Sample(
            sample_id=sample_id, stream=self.stream_id, shard=shard,
            source_path=source.path, raw_decode=source.raw_decode, recipe=recipe,
            region_local=True, c_gt=CgtRef(), answer=params,
            scene_meta=SceneMeta(scene=scene_of(source)),
            quality=QualityScores(),
            source_id=source.source_id, recipe_asset_id=recipe.source_recipe_id,
            native_size=self._native_size(source),
            build_version=str(self.ctx.config.get("build_version", "v2")),
            schema_version=str(self.ctx.config.get("schema_version", "datagen_v2")),
            after_source=AfterSource.TEACHER,
        )
        # The PPR10K human mask path is recorded on the source (registry).
        ppr_mask = (source.meta or {}).get("ppr10k_mask")
        exact = None
        if ppr_mask:
            exact = self._load_mask_png(ppr_mask, self._native_size(source))
        sample.c_gt = self._build_cgt(
            sample_id, shard, source, params, region_local=True,
            seed=int(seed) ^ _det_seed32(sample_id),
            exact_mask=exact, mask_source=MaskSource.PPR10K,
            recipe_key=recipe.source_recipe_id,
        )
        if exact is not None:
            sample.quality.mask_quality = 1.0
        self._annotate(source, recipe, sample, None)
        if not self._gate(sample):
            return None
        return sample

    @staticmethod
    def _load_mask_png(path: str, native: Optional[Tuple[int, int]]) -> Optional[Any]:
        try:
            import numpy as np
            from PIL import Image  # type: ignore

            im = Image.open(path).convert("L")
            if native:
                im = im.resize((native[1], native[0]), Image.BILINEAR)
            return (np.asarray(im, dtype="float32") / 255.0)
        except Exception:
            return None


# ===========================================================================
# Factory: build the right stream object for a StreamId given resolved inputs.
# ===========================================================================


def make_stream(
    stream_id: StreamId,
    ctx: BuildContext,
    inputs: _PlanInputs,
    gate: Optional[QAGate] = None,
) -> BaseStream:
    """Instantiate the concrete Stream for ``stream_id`` (used by run.py)."""
    if stream_id in (StreamId.S2_RECIPE_LOCAL, StreamId.S6_RECIPE_GLOBAL):
        return RecipeXSourceStream(
            ctx, stream_id, inputs,
            region_local=(stream_id == StreamId.S2_RECIPE_LOCAL), gate=gate,
        )
    if stream_id == StreamId.S5_GREYSKY_GLOBAL:
        return Tier1ExpertStream(ctx, inputs, gate)
    if stream_id == StreamId.S3_MMART_LOCAL:
        return MMArtTextStream(ctx, inputs, gate)
    if stream_id in (StreamId.S1_DEGRADE_LOCAL, StreamId.S7_DEGRADE_GLOBAL):
        return InverseDegradeStream(
            ctx, stream_id, inputs,
            region_local=(stream_id == StreamId.S1_DEGRADE_LOCAL), gate=gate,
        )
    if stream_id == StreamId.S4_PPR10K_LOCAL:
        return PPR10KStream(ctx, inputs, gate)
    if stream_id == StreamId.S8_FIVEK_GLOBAL:
        return FivekGoldStream(ctx, inputs, gate)
    raise ValueError(f"unknown stream {stream_id!r}")
