"""
dataset_build/contracts.py
==========================
Shared dataclasses + abstract interfaces for the VeraRetouch Direction-A,
RECIPE-BASED, region-local-heavy 1,000,000-sample dataset build.

This module is the single source of truth that every implementer (registry,
recipes, render, masking, vlm_clean, streams, pack) aligns to. It is
IMPORT-LIGHT on purpose:
  - NO `torch`, NO `transformers`, NO `rawpy`, NO `cv2`, NO model weights at
    module top level. Heavy deps are imported lazily INSIDE the concrete
    implementations (vera_renderer.py, sam3_masker.py, ...), never here.
  - Only stdlib (`dataclasses`, `enum`, `typing`, `abc`, `pathlib`) is used so
    this file imports on any env (base / fivek-cleaning / monetgpt_sam3 / vllm).

Grounding (read before editing):
  - docs/plan/dataset/DATASET_BUILD_PLAN.md           (the master plan)
  - docs/plan/research/data_construction_dossier.md   (§1.1 record, §2 tracks, §4 C_GT, §8 QA)
  - docs/plan/impl_plan_A_attention_context_4dlut.md  (§2.1-2.3 Direction-A C_GT, §2.5 record)
  - docs/plan/research/00_codebase_grounding.md        (§1-3 token/renderer facts)
  - docs/plan/dataset/probe/*.md                       (verified on-disk facts)
  - VeraRetouch param schema: data/infer_dataset.py get_organized_dict L229-296
    + data_samples/param.json (the 36-key target).

KEY FACTS this contract encodes:
  - The renderer (VeraRetouch param-mode) is a GLOBAL uniform transform
    (grounding §3). For C_GT to carry signal a sample must be REGION-LOCAL:
    render globally, then COMPOSITE inside a mask M_r; C_GT := that mask
    (MEMORY caveat b). `Sample.region_local` flags this.
  - Direction-A supervision signal is a single-channel C_GT in [0,1]^{H×W}
    (impl_plan_A §2.3), built as a 3-aspect stack (L/GC/SC) and collapsed
    via C_GT^{1ch} = max_r C_GT^r.
  - Storage is RECIPE-BASED (USER DECISION 1): each Sample stores a source
    path + a Recipe + mask refs; before/after are rendered ON THE FLY at
    NATIVE resolution (never crop/resize). We persist the small C_GT PNGs
    (cheap) but NOT the rendered before/after pixels.
  - VeraRetouch param.json values are RAW Lightroom units (e.g. Exposure 30);
    get_organized_dict divides by 100 internally. Store RAW units here.
"""

from __future__ import annotations

import abc
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, Dict, Iterable, Iterator, List, Optional, Sequence, Tuple

# ---------------------------------------------------------------------------
# 0. The VeraRetouch param schema (the 36 keys the renderer consumes).
#    Mirrors data/infer_dataset.py get_organized_dict (L229-296). Implementers
#    MUST emit param dicts of the form {key: {"value": <raw_LR_number>}} and
#    drop any CRS key not in PARAM_KEYS (param-mode is global; local-mask keys
#    like MaskGroupBasedCorrections are recorded as region hints, never sent
#    to the renderer).
# ---------------------------------------------------------------------------

LIGHT_KEYS: Tuple[str, ...] = (
    "Exposure2012", "Contrast2012", "Highlights2012", "Shadows2012",
    "Whites2012", "Blacks2012", "ParametricShadows", "ParametricDarks",
    "ParametricLights", "ParametricHighlights",
)
COLORTEMP_KEYS: Tuple[str, ...] = (
    "IncrementalTemperature", "IncrementalTint", "Vibrance", "Saturation",
)
_HSL_BANDS = ("Red", "Orange", "Yellow", "Green", "Aqua", "Blue", "Purple", "Magenta")
COLORMIXER_KEYS: Tuple[str, ...] = tuple(
    f"{prefix}{band}"
    for prefix in ("HueAdjustment", "SaturationAdjustment", "LuminanceAdjustment")
    for band in _HSL_BANDS
)
# NOTE: get_organized_dict (data/infer_dataset.py L229-296) lists 10 Light + 4
# Color&Temp + 24 Specific-Color = 38 keys. (Several probe docs say "36"; that is
# an arithmetic slip — the renderer code itself, read directly, has 38. Ground
# truth = the code.)
PARAM_KEYS: Tuple[str, ...] = LIGHT_KEYS + COLORTEMP_KEYS + COLORMIXER_KEYS  # 10 + 4 + 24 = 38
assert len(PARAM_KEYS) == 38, f"expected 38 VeraRetouch param keys, got {len(PARAM_KEYS)}"

# The 3 Direction-A aspects (the VLM retouch-token spine: grounding §1).
ASPECTS: Tuple[str, str, str] = ("L", "GC", "SC")  # light / global-color / specific-color


# ---------------------------------------------------------------------------
# 1. Enums
# ---------------------------------------------------------------------------

class RecipeKind(str, Enum):
    """How the 'after' is reproduced from the source.

    DATAGEN v2 keeps the active reproduction path to PARAM/LUT. DEGRADE remains
    only so old JSONL rows can still be parsed during migration; new S1/S7 rows
    use kind=PARAM with provenance=DEGRADE and keep DegradeSpec for audit.
    """
    PARAM = "param"        # CRS/XMP/lrtemplate -> VeraRetouch param.json (the teacher)
    LUT = "lut"            # .cube / .3dl 3D-LUT applied via trilinear interp
    DEGRADE = "degrade"    # LEGACY only; do not emit in DATAGEN v2 shards


class Provenance(str, Enum):
    """Where a recipe came from; separate from the active reproduction kind."""
    PARAM = "param"
    LUT = "lut"
    DEGRADE = "degrade"
    PPR10K = "ppr10k"


class AfterSource(str, Enum):
    """Which target-after source a trainer must use for this sample."""
    TEACHER = "teacher"
    REAL_JPG = "real_jpg"


class MaskSource(str, Enum):
    SAM3 = "sam3"                       # text-promptable SAM3 (monetgpt_sam3 env)
    GROUNDINGDINO_SAM2 = "groundingdino_sam2"  # fallback detector+segmenter
    PPR10K = "ppr10k"                   # real human-region mask (if re-downloaded)
    DEGRADE = "degrade"                 # exact construction mask (self-supervised)
    SAM3_ANCHORED = "sam3_anchored"     # WS-B: real-preset template pinned to a SAM3 region
    LR_GOLD = "lr_gold"                 # WS-B: real Lightroom local-mask render (gold)
    GLOBAL = "global"                   # all-ones (degenerate 3D-LUT case)


class RawDecode(str, Enum):
    """How a source path must be decoded to sRGB before rendering."""
    NONE = "none"          # already a jpg/png -> cv2.imread
    RAWPY = "rawpy"        # .dng/.cr2/.arw -> rawpy.postprocess (env fivek-cleaning)


class StreamId(str, Enum):
    """The 7 budget streams (DATASET_BUILD_PLAN.md §budget)."""
    S1_DEGRADE_LOCAL = "S1"   # inverse-degradation, region-local (exact mask)  340k
    S2_RECIPE_LOCAL = "S2"    # recipe x source, region-local composite (SAM3) 190k
    S3_MMART_LOCAL = "S3"     # MMArt text-attached, region-local                30k
    S4_PPR10K_LOCAL = "S4"    # PPR10K human-mask region-local (conditional)     40k
    S5_GREYSKY_GLOBAL = "S5"  # GREYSKY real-expert global triples               20k
    S6_RECIPE_GLOBAL = "S6"   # recipe x source, GLOBAL                         270k
    S7_DEGRADE_GLOBAL = "S7"  # inverse-degradation, GLOBAL                     110k
    S8_FIVEK_GLOBAL = "S8"    # fivek real before->expert-after GOLD global      20k


# ---------------------------------------------------------------------------
# 2. Source registry
# ---------------------------------------------------------------------------

@dataclass
class SourceItem:
    """One source IMAGE usable as a 'before' input (native resolution, never resized).

    Produced by the registry from the on-disk corpora (probe_sources_budget,
    probe_全店素材, probe_GREYSKY_raw_photogs). Stored one-per-row in
    source_index.jsonl.
    """
    source_id: str                       # stable hash id (sha1 of abs_path + size)
    path: str                            # ABS path to the source image / raw
    corpus: str                          # "tad66k" | "fivek" | "awards" | "korean"
                                         #  | "quandian" (全店素材) | "greysky" | "mmart"
                                         #  | "unsplash" | "ppr10k" | "fivek_gold"
    raw_decode: RawDecode = RawDecode.NONE
    width: Optional[int] = None          # native px (filled lazily; may be None pre-decode)
    height: Optional[int] = None
    scene: Optional[str] = None          # "portrait"|"landscape"|"street"|... (manifest/CLIP)
    is_portrait_pool: bool = False       # routing hint for scene-matching (skin-friendly)
    tags: List[str] = field(default_factory=list)
    bytes_size: Optional[int] = None
    meta: Dict[str, Any] = field(default_factory=dict)


@dataclass
class RecipeAsset:
    """One reusable 'look' recipe on disk (xmp / lrtemplate / cube / 3dl).

    Produced by the registry from 全店素材 / E18 / GREYSKY / preset_dataset_v1.
    Stored one-per-row in recipe_index.jsonl (post-dedup by preset stem).
    """
    recipe_id: str                       # stable hash id
    path: str                            # ABS path to the recipe file
    kind: RecipeKind                     # PARAM (xmp/lrtemplate) | LUT (cube/3dl)
    fmt: str                             # "xmp"|"lrtemplate"|"cube"|"3dl"
    pack_id: Optional[str] = None        # e.g. "H010" / "E18" / "GREYSKY/FILM"
    style: Optional[str] = None          # "warm vintage film" | "clean commercial" | ...
    scene_affinity: Optional[str] = None # "portrait"|"landscape"|"any" (scene-matching)
    is_bw: bool = False                  # ConvertToGrayscale -> drop for color C_GT pilot
    is_technical: bool = False           # Slog/REC709 conversion LUT -> drop
    has_local_mask: bool = False         # MaskGroup/Gradient/PaintBased present (region hint)
    lut_size: Optional[int] = None       # cube/3dl grid N (17/32/33/64)
    tags: List[str] = field(default_factory=list)
    meta: Dict[str, Any] = field(default_factory=dict)


# ---------------------------------------------------------------------------
# 3. Recipe (the edit specification — one of three kinds)
# ---------------------------------------------------------------------------

@dataclass
class DegradeSpec:
    """Track-A/B inverse-degradation / Gaussian-operator perturbation spec.

    The 'before' is the degraded image, the 'after' is the clean source (or
    vice-versa per `forward`). For region-local samples the degradation is
    applied ONLY inside `region_mask_ref`, which becomes the exact C_GT.
    """
    mode: str                            # "gaussian_op" (Track-B) | "er_invert" (Track-A)
    # Track-B: Gaussian-sampled raw CRS operators (subset of PARAM_KEYS), raw LR units.
    op_params: Dict[str, float] = field(default_factory=dict)
    sigma_profile: Optional[str] = None  # name of the per-op sigma table (P0_001 Tab 8)
    aspects: List[str] = field(default_factory=list)   # which of L/GC/SC are perturbed
    forward: bool = False                # False (canonical v2): invert (degrade) -> stored params = neg_p
                                         # (render(src, neg_p) = degraded input). True: clean->after via op.
                                         # Default is False so a spec missing the field never silently
                                         # flips to +op_params (the off-contract direction). See recipes.py
                                         # sample_degrade_spec + streams.py _params_from_degrade_spec.
    seed: Optional[int] = None           # reproducible sampling


@dataclass
class Recipe:
    """The edit recipe.

    DATAGEN v2 active reproduction is selected by `kind` in {PARAM, LUT}. A
    DEGRADE-origin sample stores kind=PARAM, params=teacher-renderable neg_p,
    provenance=DEGRADE, and keeps `degrade` only for audit / Stage-0 warm start.
    RECIPE-BASED: the recipe + source path are enough to re-render before/after
    at native resolution on demand.
    """
    kind: RecipeKind
    # kind == PARAM: raw VeraRetouch param.json dict {key: {"value": raw_LR_number}}.
    #   Only PARAM_KEYS allowed; missing keys default to {"value": 0}.
    params: Optional[Dict[str, Dict[str, float]]] = None
    # kind == LUT: ref to a RecipeAsset of kind LUT (resolved via recipe_index).
    lut_recipe_id: Optional[str] = None
    # kind == DEGRADE: the degradation spec.
    degrade: Optional[DegradeSpec] = None
    provenance: Provenance = Provenance.PARAM
    # Provenance: which on-disk recipe asset this came from (if any).
    source_recipe_id: Optional[str] = None
    meta: Dict[str, Any] = field(default_factory=dict)


# ---------------------------------------------------------------------------
# 4. C_GT reference + quality + scene metadata
# ---------------------------------------------------------------------------

@dataclass
class CgtRef:
    """Recipe-based reference to the Direction-A C_GT (NOT pixels of before/after).

    We DO persist the small single-channel C_GT maps (cheap, tens of KB native).
    Per probe_sam3 §7 and impl_plan_A §2.3:
      - cgt_path: 1-ch 8-bit PNG, native HxW, value=round(255*sigmoid) -> C_GT^{1ch}.
      - cgt_aspect_paths: optional 3 PNGs (L/GC/SC) for the per-aspect stack.
      - cgt_patchgrid_path: tiny .npy float16 at the VLM patch-grid res (~16x16),
        for the cheap map-supervision loss (the model's C(x) lives on the patch grid).
      - rle: optional inline COCO RLE for hard binary masks (smallest for binary).

    D7 ANTI-DOUBLE-COUNT CONTRACT (doc §4): the supervision target is the BAKED
    cgt PNG (cgt_path / cgt_patchgrid_path) -- it ALREADY includes g.soft_blur,
    i.e. C_GT = M_r * g(||dtheta||) soft-blurred (see streams.py _build_cgt).
    aspect_magnitude / soft_blur_sigma_px / magnitude_tau are METADATA for
    analysis only -- the loss MUST NOT re-apply aspect_magnitude (would
    double-count g).
    """
    cgt_path: Optional[str] = None
    cgt_aspect_paths: Optional[Dict[str, str]] = None   # {"L":..,"GC":..,"SC":..}
    cgt_patchgrid_path: Optional[str] = None
    raw_mask_path: Optional[str] = None                    # pre-blur M_r for composite replay
    rle: Optional[Dict[str, Any]] = None                # {"size":[H,W],"counts":...}
    mask_source: MaskSource = MaskSource.GLOBAL
    concepts: List[str] = field(default_factory=list)   # SAM3 text prompts used
    mask_score: Optional[float] = None                  # SAM3/GDINO confidence
    # Per-aspect edit magnitude g(||Delta theta_r||) used in Mode-1 weighting (impl_plan_A §2.3).
    # D7 METADATA ONLY: supervision target = the BAKED cgt PNG (already includes
    # g.soft_blur); the loss MUST NOT re-apply aspect_magnitude (would double-count g).
    aspect_magnitude: Dict[str, float] = field(default_factory=dict)
    coverage: Optional[float] = None                    # fraction of pixels with C_GT>0.5
    soft_blur_sigma_px: Optional[float] = None
    magnitude_tau: Optional[float] = None


@dataclass
class QualityScores:
    """vLLM-judge + heuristic gate outputs (data_construction_dossier §8, probe_vllm_clean §3c).
    Inclusion rule: see streams.QAGate.accept().
    """
    look_match: Optional[bool] = None
    param_sane: Optional[bool] = None
    processed_ok: Optional[bool] = None    # MMArt processed.jpg sanity (caveat a)
    mllm_score: Optional[float] = None      # 0..1 (Instruction-Compliance/Seamless/Preserve/Tech)
    aesthetic: Optional[float] = None       # LAION/Q-Align if computed
    mask_quality: Optional[float] = None    # mask confidence / largest-overlap check
    histsim: Optional[float] = None         # color-direction agreement (Track recon gate)
    er_recon_psnr: Optional[float] = None   # Track-A round-trip PSNR floor (P0_001 Tab 9)
    rejected_reason: Optional[str] = None


@dataclass
class SceneMeta:
    scene: Optional[str] = None            # portrait|landscape|street|food|product|wedding|night
    style: Optional[str] = None            # short style label
    lang: str = "en"
    masksubtype_hint: int = 0              # MMArt 1=Subject 2=Sky 3=Person 0=global
    tags: List[str] = field(default_factory=list)


# ---------------------------------------------------------------------------
# 5. THE Sample record (VeraRetouch-format + recipe-based + Direction-A).
# ---------------------------------------------------------------------------

@dataclass
class Sample:
    """One dataset record. Serialized one-per-line into shard JSONL manifests.

    Direction-A fields (the priority supervision signal) are first-class:
    region_local + c_gt + instruction/think/answer. before/after are NEVER
    stored as pixels — they are re-rendered from (source_path, recipe) at
    native resolution by vera_renderer + region_composite.
    """
    # --- identity / sharding ---
    sample_id: str
    stream: StreamId
    shard: str                             # e.g. "shard_00042"
    # --- recipe-based core (USER DECISION 1) ---
    source_path: str                       # ABS native-res 'before' source
    raw_decode: RawDecode
    recipe: Recipe
    # --- Direction-A core (USER DECISION 2) ---
    region_local: bool                     # True => composite-in-mask, C_GT carries signal
    c_gt: CgtRef
    # --- reasoning / instruction (offline vLLM; probe_vllm_clean) ---
    instruction: Optional[str] = None      # natural-language edit request (long form)
    instruction_short: Optional[str] = None
    think: Optional[str] = None            # <think> reasoning trace
    answer: Optional[Dict[str, Dict[str, float]]] = None  # <answer> = VeraRetouch params (raw)
    # --- metadata ---
    scene_meta: SceneMeta = field(default_factory=SceneMeta)
    quality: QualityScores = field(default_factory=QualityScores)
    # --- provenance / reproducibility ---
    source_id: Optional[str] = None
    recipe_asset_id: Optional[str] = None
    native_size: Optional[Tuple[int, int]] = None   # (H, W) of source
    build_version: str = "v2"
    schema_version: str = "datagen_v2"
    after_source: AfterSource = AfterSource.TEACHER
    meta: Dict[str, Any] = field(default_factory=dict)


# ===========================================================================
# ABSTRACT INTERFACES (the exact signatures every implementer must satisfy).
# Concrete classes live in sibling modules and may import torch/cv2/rawpy
# LAZILY inside methods. Keep this file dep-free.
# ===========================================================================

# ---- registry.py ---------------------------------------------------------
class Registry(abc.ABC):
    """Scans the on-disk corpora -> source_index.jsonl + recipe_index.jsonl + pack rows.
    Applies the junk filter (probe_全店素材 §4) and dedup-by-preset-stem (G3).
    """

    @abc.abstractmethod
    def scan_sources(self) -> Iterator[SourceItem]: ...

    @abc.abstractmethod
    def scan_recipes(self) -> Iterator[RecipeAsset]: ...

    @abc.abstractmethod
    def write_indexes(self, out_dir: str) -> Dict[str, int]:
        """Writes source_index.jsonl + recipe_index.jsonl; returns counts."""

    @abc.abstractmethod
    def load_sources(self, index_path: str,
                     corpus: Optional[str] = None,
                     scene: Optional[str] = None) -> List[SourceItem]: ...

    @abc.abstractmethod
    def load_recipes(self, index_path: str,
                     kind: Optional[RecipeKind] = None,
                     scene_affinity: Optional[str] = None) -> List[RecipeAsset]: ...


# ---- recipes.py ----------------------------------------------------------
class RecipeParser(abc.ABC):
    """Parses on-disk recipe files into VeraRetouch params / LUT tensors.
    Reuses presets/scripts/build_preset_dataset.parse_xmp_file (probe_recipe_parsers §1a).
    """

    @abc.abstractmethod
    def xmp_to_params(self, xmp_path: str) -> Dict[str, Dict[str, float]]:
        """XMP -> {key:{"value":raw_LR_number}} keeping only PARAM_KEYS, default 0
        (probe_recipe_parsers §1c). value = float(str.lstrip('+'))."""

    @abc.abstractmethod
    def lrtemplate_to_params(self, path: str) -> Dict[str, Dict[str, float]]:
        """Lua-table .lrtemplate -> same param dict (reuse pe_kg _lua_table_parser)."""

    @abc.abstractmethod
    def load_cube(self, path: str) -> Tuple[Any, Tuple[float, float, float], Tuple[float, float, float]]:
        """.cube -> (lut[N,N,N,3] float32, domain_min, domain_max). Returns torch tensor
        (import torch lazily). Red-fastest reshape; handle .3dl blue-fastest separately."""

    @abc.abstractmethod
    def sample_degrade_spec(self, aspects: Sequence[str], seed: int,
                            region_local: bool) -> DegradeSpec:
        """Track-B: Gaussian-sample raw CRS operators within per-op sigma (P0_001 Tab 8)."""


class LutApplier(abc.ABC):
    @abc.abstractmethod
    def apply_lut(self, img: Any, lut: Any,
                  domain_min: Tuple[float, float, float] = (0, 0, 0),
                  domain_max: Tuple[float, float, float] = (1, 1, 1)) -> Any:
        """img:[B,3,H,W] in [0,1] RGB -> same shape via 5D grid_sample trilinear,
        align_corners=True, padding='border'. NATIVE resolution (probe_recipe_parsers §2)."""


# ---- vera_renderer.py (TEACHER) ------------------------------------------
class Renderer(abc.ABC):
    """VeraRetouch param-mode renderer = the 'after' teacher (probe_veraretouch_renderer).
    GLOBAL uniform transform. Loads weights ONCE; never trained through (no_grad).
    """

    @abc.abstractmethod
    def render(self, image_paths: Sequence[str],
               param_dicts: Sequence[Dict[str, Dict[str, float]]],
               batch_size: int = 4, chunk: int = 262144,
               max_new_tokens: int = 256) -> List[Any]:
        """Returns list[np.uint8 HxWx3 RGB] 'after' at SOURCE native resolution
        (no resize/crop). param_dicts are RAW LR units. chunk caps peak memory."""


def region_composite(after_rgb_u8: Any, source_rgb_u8: Any, mask01: Any) -> Any:
    """source*(1-m) + after*m, all native HxW. C_GT == mask01 (MEMORY caveat b /
    probe_veraretouch_renderer §region-composite). Implemented in vera_renderer.py;
    declared here as the canonical signature. Imports numpy lazily there."""
    raise NotImplementedError("implement in dataset_build/vera_renderer.py")


# ---- sam3_masker.py ------------------------------------------------------
class ConceptMasker(abc.ABC):
    """Text-promptable masker -> soft single-channel C_GT (probe_sam3_masking §5).
    Construction-time ONLY; never run at VeraRetouch inference, never backprop.
    """

    @abc.abstractmethod
    def mask(self, image: Any, concept: str,
             native_size: Optional[Tuple[int, int]] = None,
             soft: bool = True, reduce: str = "max",
             min_score: float = 0.3) -> Any:
        """-> float32 [0,1] HxW at native res (target_sizes upsample; never resize image)."""

    @abc.abstractmethod
    def masks(self, image: Any, concepts: Sequence[str], **kw) -> Dict[str, Any]:
        """One soft map per concept (loops concepts; one text phrase per SAM3 call)."""

    @abc.abstractmethod
    def cgt_aspect_stack(self, image: Any,
                         concept_map: Dict[str, List[str]]) -> Any:
        """-> [3,H,W] float32, channels = (L, GC, SC) per the 3-aspect spine."""


# ---- vlm_clean.py --------------------------------------------------------
class Cleaner(abc.ABC):
    """Offline Qwen3-VL cleaner/annotator over the served vLLM endpoint
    (probe_vllm_clean §2-3). HTTP only; loads no weights locally.
    """

    @abc.abstractmethod
    def gen_instruction(self, image_path: str,
                        scene_meta: Optional[dict] = None) -> dict:
        """(a) -> {"instruction_long","instruction_short","lang"}"""

    @abc.abstractmethod
    def reason_params(self, image_path: str, instruction: str) -> dict:
        """(b) -> {"think": str, "answer": {<3-aspect params>}}"""

    @abc.abstractmethod
    def verify(self, before_path: str, after_path: str, params: dict) -> dict:
        """(c) -> {"look_match","param_sane","processed_ok","score","reason"}"""

    @abc.abstractmethod
    def tag_scene_region(self, image_path: str, instruction: str) -> dict:
        """(d) -> {"scene","style","region_local","sam3_concepts",
                   "groundingdino_prompt","masksubtype_hint"}"""


# ---- streams.py ----------------------------------------------------------
class QAGate(abc.ABC):
    """Accept/reject gate (data_construction_dossier §8; thresholds in config.yaml)."""

    @abc.abstractmethod
    def accept(self, sample: Sample) -> bool:
        """True iff sample passes all enabled gates (mllm_score>=thr, look_match &&
        param_sane, mask_quality>=thr, processed_ok for MMArt, er_recon_psnr floor).
        Sets sample.quality.rejected_reason on failure."""


class Stream(abc.ABC):
    """Produces Sample records for one StreamId. A stream draws (source, recipe,
    region) tuples, runs render -> [composite] -> C_GT -> annotate -> gate, and
    yields accepted Samples. Must be RESUMABLE (skip ids already in done-set).
    """
    stream_id: StreamId

    @abc.abstractmethod
    def plan(self, budget: int, seed: int) -> Iterator[Tuple[SourceItem, Recipe, bool]]:
        """Deterministically enumerate (source, recipe, region_local) work items
        up to `budget`, scene-matched (probe_sources_budget §3). Pure/no GPU."""

    @abc.abstractmethod
    def build_one(self, source: SourceItem, recipe: Recipe,
                  region_local: bool, sample_id: str, shard: str) -> Optional[Sample]:
        """Full per-sample pipeline; returns Sample or None if rejected. May use
        Renderer/ConceptMasker/Cleaner/QAGate (injected at construction)."""

    @abc.abstractmethod
    def run(self, budget: int, seed: int, done_ids: Iterable[str]) -> Iterator[Sample]:
        """plan -> build_one over not-yet-done items; yields accepted Samples."""


# ---- pack.py -------------------------------------------------------------
class ShardWriter(abc.ABC):
    """Writes accepted Samples to sharded JSONL + the C_GT PNG/npy sidecars,
    atomically and resumably (DATASET_BUILD_PLAN.md §storage/§resumability).
    """

    @abc.abstractmethod
    def write(self, sample: Sample) -> None:
        """Append sample to the current shard's .jsonl.tmp; rotate at shard_size."""

    @abc.abstractmethod
    def flush(self) -> None:
        """fsync + atomic rename .jsonl.tmp -> .jsonl; update manifest_index.jsonl."""

    @abc.abstractmethod
    def done_ids(self) -> Iterable[str]:
        """All sample_ids already committed (for resumability)."""

    @abc.abstractmethod
    def stats(self) -> Dict[str, Any]:
        """Per-stream counts, region-local fraction, reject rate, bytes-on-disk."""


def sample_to_jsonl(sample: Sample) -> str:
    """Canonical JSONL serialization of a Sample (dataclasses.asdict + enum->value).
    Implemented in pack.py; declared here so readers/writers agree on the schema."""
    raise NotImplementedError("implement in dataset_build/pack.py")


def jsonl_to_sample(line: str) -> Sample:
    """Inverse of sample_to_jsonl (for the training-time loader + eval)."""
    raise NotImplementedError("implement in dataset_build/pack.py")
