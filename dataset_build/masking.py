"""
dataset_build/masking.py
========================
Direction-A C_GT producer for the VeraRetouch RECIPE-BASED 1M dataset build.

This module implements the **construction/supervision-time** masker that turns a
source image (and optionally an expert before/after pair) into the single-channel
retouch-region context/attention map ``C_GT ∈ [0,1]^{H×W}`` at NATIVE resolution,
plus the optional 3-aspect (L / GC / SC) stack. It also handles compact C_GT
storage (1-ch native PNG + tiny patch-grid .npy + optional COCO-RLE) and the
concept-selection logic that maps scene/recipe tags to SAM3 text prompts.

It satisfies the abstract ``ConceptMasker`` interface declared in
``dataset_build/contracts.py`` (``Sam3Masker`` is the concrete class) and adds the
two C_GT builders the architect asked for:

  * ``cgt_from_concept(img, concepts)``  -> Mode-1 mask-derived C_GT (SAM3 union).
  * ``cgt_from_diff(before, after)``     -> Mode-2 expert-pair CIELAB diff C_GT
                                            (for GREYSKY / FiveK / PPR10K real pairs).

Grounding (read before editing):
  - docs/plan/dataset/probe/probe_sam3_masking.md   (§3 API, §4 load, §5 sig, §6 vocab, §7 storage, §8 gotchas)
  - docs/plan/impl_plan_A_attention_context_4dlut.md (§2.3 Mode-1/Mode-2 C_GT, max_r collapse)
  - dataset_build/contracts.py                       (ConceptMasker, CgtRef, MaskSource, ASPECTS)
  - dataset_build/config.yaml                        (models.sam3, cgt.*)

HARD RULES (from the build memory + probes):
  * Construction-time ONLY. SAM3 is a labeler: never backprop through it, never
    run it at VeraRetouch inference. The trained model derives C(x) from its own
    VLM attention (the differentiator vs PerTouch).
  * NATIVE resolution: never crop/resize the source image. SAM3 runs internally
    at 1008^2 and post_process upsamples masks back to native via target_sizes.
  * ``pred_masks`` are LOGITS -> sigmoid for the soft map.
  * Heavy deps (torch / transformers / SAM3 weights) are imported and loaded
    LAZILY inside methods. Module top-level stays import-light (numpy / cv2 /
    PIL / pycocotools only, all present in env monetgpt_sam3).

Run on its env:  /home/bc/miniconda3/envs/monetgpt_sam3/bin/python
"""

from __future__ import annotations

import base64
import json
import os
import zlib
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

# Contracts are import-light (stdlib only) -> safe to import at top level.
from .contracts import ASPECTS, CgtRef, ConceptMasker, MaskSource

__all__ = [
    "Sam3Masker",
    "select_concepts",
    "cgt_collapse",
    "apply_magnitude",
    "soft_blur",
    "coverage",
    "save_cgt",
    "load_cgt",
    "patch_grid_map",
    "rle_encode",
    "rle_decode",
]


# ===========================================================================
# 0. Concept vocabulary  ->  3-aspect spine (L / GC / SC).
#    Mirrors config.yaml cgt.concept_vocab + cgt.concept_map and probe §6.
#    Kept here as a self-contained default so this module is runnable without
#    threading config everywhere; run_build may override via set_concept_map().
# ===========================================================================

# Open-vocab seed set, grouped (probe_sam3 §6 / config cgt.concept_vocab).
CONCEPT_VOCAB: Dict[str, List[str]] = {
    "subject": ["person", "face", "skin", "hair", "eyes", "lips", "subject"],
    "nature": ["sky", "clouds", "foliage", "leaves", "grass", "trees",
               "flowers", "water", "sea", "sunset", "mountain", "snow", "sand"],
    "built": ["building", "wall", "street", "car", "food"],
    "compositional": ["foreground", "background", "shadows", "highlights"],
    "catch_all": ["the main subject", "the background"],
}

# Default aspect -> concept-list map (probe §6 / config cgt.concept_map).
#   L  : light / global    -> usually near-uniform => C_L ~ 1 (4D-LUT degrades to 3D-LUT, correct).
#   GC : global color&temp.
#   SC : specific color (the most LOCAL aspect, carries the most signal).
DEFAULT_CONCEPT_MAP: Dict[str, List[str]] = {
    "L": ["foreground", "background"],
    "GC": ["sky", "background", "building"],
    "SC": ["skin", "face", "foliage", "grass", "water", "sky", "flowers"],
}

# Scene -> preferred concept ordering for cgt_from_concept (single-channel path).
_SCENE_CONCEPTS: Dict[str, List[str]] = {
    "portrait": ["skin", "face", "person", "hair", "subject"],
    "wedding": ["person", "skin", "face", "subject", "the main subject"],
    "landscape": ["sky", "foliage", "grass", "water", "mountain", "sunset"],
    "street": ["building", "person", "car", "street", "the main subject"],
    "food": ["food", "the main subject"],
    "product": ["the main subject", "background"],
    "night": ["the main subject", "sky", "shadows"],
}

# Keyword hints scanned in recipe/style/tag strings -> SC-flavored concepts.
_TAG_CONCEPT_HINTS: List[Tuple[str, List[str]]] = [
    ("skin", ["skin", "face"]),
    ("portrait", ["skin", "face", "person"]),
    ("人像", ["skin", "face", "person"]),
    ("肤", ["skin", "face"]),
    ("新娘", ["person", "skin", "face"]),
    ("婚礼", ["person", "skin", "face"]),
    ("wedding", ["person", "skin", "face"]),
    ("sky", ["sky", "clouds"]),
    ("天空", ["sky", "clouds"]),
    ("sunset", ["sunset", "sky"]),
    ("日落", ["sunset", "sky"]),
    ("ocean", ["water", "sea"]),
    ("海洋", ["water", "sea"]),
    ("flower", ["flowers", "foliage"]),
    ("花", ["flowers", "foliage"]),
    ("landscape", ["sky", "foliage", "grass"]),
    ("风光", ["sky", "foliage", "grass"]),
    ("foliage", ["foliage", "grass", "trees"]),
    ("green", ["foliage", "grass"]),
    ("food", ["food"]),
]


def select_concepts(
    scene: Optional[str] = None,
    recipe_tags: Optional[Sequence[str]] = None,
    sam3_concepts: Optional[Sequence[str]] = None,
    aspect: Optional[str] = None,
    max_concepts: int = 4,
) -> List[str]:
    """Pick SAM3 text prompts from scene / recipe tags (probe §6, config scene_matching).

    Priority order:
      1. Explicit ``sam3_concepts`` (e.g. from the vLLM ``tag_scene_region`` call) win.
      2. ``aspect`` (one of L/GC/SC) -> that aspect's default concept list.
      3. Keyword hints found in ``recipe_tags`` (style/name strings, EN + 中文).
      4. The ``scene`` bucket's preferred concepts.
      5. Catch-all (``the main subject``).

    Returns a de-duplicated, order-preserved list capped at ``max_concepts``.

    Args:
        scene: portrait|landscape|street|food|product|night|wedding|... (or None).
        recipe_tags: free-text tags / style label / preset name tokens to scan.
        sam3_concepts: explicit prompts that override everything else.
        aspect: if given, restrict to that aspect's concept list (L/GC/SC).
        max_concepts: cap (CLIPTokenizer is short; one SAM3 call per concept).
    """
    out: List[str] = []

    def _add(items: Sequence[str]) -> None:
        for c in items:
            if c and c not in out:
                out.append(c)

    if sam3_concepts:
        _add(sam3_concepts)

    if aspect:
        _add(DEFAULT_CONCEPT_MAP.get(aspect.upper(), []))

    if recipe_tags:
        blob = " ".join(str(t) for t in recipe_tags).lower()
        for needle, concepts in _TAG_CONCEPT_HINTS:
            if needle.lower() in blob:
                _add(concepts)

    if scene:
        _add(_SCENE_CONCEPTS.get(scene.lower(), []))

    if not out:
        _add(CONCEPT_VOCAB["catch_all"])

    return out[:max_concepts]


# ===========================================================================
# 1. C_GT math helpers (Mode-1 magnitude weighting + collapse + blur + coverage).
#    Pure numpy; no model load. These are reused by streams.py.
# ===========================================================================

def cgt_collapse(aspect_stack: np.ndarray) -> np.ndarray:
    """Collapse a [3,H,W] (L,GC,SC) aspect stack to the single channel.

    ``C_GT^{1ch} = max_r C_GT^r`` (impl_plan_A §2.3). Input/output float32 in [0,1].
    """
    arr = np.asarray(aspect_stack, dtype=np.float32)
    if arr.ndim == 2:                       # already single-channel
        return np.clip(arr, 0.0, 1.0)
    if arr.ndim != 3:
        raise ValueError(f"aspect_stack must be [C,H,W] or [H,W], got {arr.shape}")
    return np.clip(arr.max(axis=0), 0.0, 1.0)


def apply_magnitude(mask01: np.ndarray, delta_norm: float, tau: float = 1.0) -> np.ndarray:
    """Mode-1 per-region magnitude weighting: ``C_GT^r = M_r * g(||dtheta_r||)``,
    with ``g = tanh(||dtheta_r|| / tau)`` normalizing edit magnitude to [0,1]
    (impl_plan_A §2.3; config cgt.magnitude_tau).

    Args:
        mask01: soft region mask in [0,1], HxW.
        delta_norm: ||dtheta_r|| (the per-region edit magnitude, raw units).
        tau: temperature; larger tau -> needs bigger edits to saturate.
    """
    g = float(np.tanh(abs(delta_norm) / max(tau, 1e-6)))
    return (np.asarray(mask01, dtype=np.float32) * g).astype(np.float32)


def soft_blur(mask01: np.ndarray, sigma_px: float = 2.0) -> np.ndarray:
    """PerTouch soft-boundary Gaussian blur on the stored C_GT (anti-overfit;
    config cgt.soft_blur_sigma_px). Uses cv2 (lazy import). sigma<=0 -> no-op.
    """
    if sigma_px is None or sigma_px <= 0:
        return np.asarray(mask01, dtype=np.float32)
    import cv2  # lazy
    m = np.asarray(mask01, dtype=np.float32)
    # ksize 0 lets OpenCV derive it from sigma; round to odd >=3.
    k = max(3, int(round(sigma_px * 6)) | 1)
    out = cv2.GaussianBlur(m, (k, k), sigmaX=float(sigma_px), sigmaY=float(sigma_px))
    return np.clip(out, 0.0, 1.0).astype(np.float32)


def coverage(mask01: np.ndarray, thr: float = 0.5) -> float:
    """Fraction of pixels with C_GT > thr (config cgt.min/max_coverage gating)."""
    m = np.asarray(mask01, dtype=np.float32)
    if m.size == 0:
        return 0.0
    return float((m > thr).mean())


# ===========================================================================
# 2. Mode-2: expert before/after CIELAB difference map (real pairs).
#    GREYSKY / FiveK / PPR10K supervised triples have a true 'after'; the C_GT
#    is the per-aspect CIELAB diff (impl_plan_A §2.3 Mode-2):
#      L  <- |dL*| (luminance change)
#      GC <- ||(da*, db*)|| (chroma / temp-tint shift)  -- global-color proxy
#      SC <- hue-specific delta (we proxy by the chroma diff masked to where the
#            hue ALSO rotates, i.e. a color-specific change rather than a pure
#            lightness change). Without per-band hue masks this is a heuristic
#            (Mode-2 is explicitly heuristic, impl_plan_A §2.3).
# ===========================================================================

def _to_lab(rgb_u8: np.ndarray) -> np.ndarray:
    """sRGB uint8 HxWx3 -> CIELAB float32 (L in [0,100], a/b ~[-128,127])."""
    import cv2  # lazy
    bgr = cv2.cvtColor(np.ascontiguousarray(rgb_u8), cv2.COLOR_RGB2BGR)
    lab = cv2.cvtColor(bgr, cv2.COLOR_BGR2LAB).astype(np.float32)
    # OpenCV 8-bit LAB packs L into [0,255], a/b into [0,255] with 128 offset.
    lab[..., 0] *= 100.0 / 255.0
    lab[..., 1] -= 128.0
    lab[..., 2] -= 128.0
    return lab


def cgt_from_diff(
    before_rgb_u8: np.ndarray,
    after_rgb_u8: np.ndarray,
    *,
    region_mask01: Optional[np.ndarray] = None,
    l_scale: float = 25.0,
    gc_scale: float = 20.0,
    sc_scale: float = 20.0,
    blur_sigma_px: float = 2.0,
    return_stack: bool = False,
) -> np.ndarray:
    """Mode-2 C_GT from a real expert before/after pair (impl_plan_A §2.3).

    Builds a 3-aspect (L/GC/SC) CIELAB-difference stack at NATIVE resolution and
    (by default) collapses it to the single channel via ``max_r``. Each aspect is
    squashed to [0,1] by ``tanh(diff / scale)`` (so the *scale* sets how big a
    Lab change saturates the map). If ``region_mask01`` is given (e.g. a PPR10K
    human mask), the diff is gated to that region.

    Args:
        before_rgb_u8, after_rgb_u8: native HxWx3 uint8 RGB, SAME size.
        region_mask01: optional HxW soft mask to multiply in (real-region anchor).
        l_scale/gc_scale/sc_scale: tanh saturation scales for L / GC / SC diffs.
        blur_sigma_px: soft-boundary blur (0 -> off).
        return_stack: if True return [3,H,W] (L,GC,SC); else single-channel HxW.

    Returns:
        float32 in [0,1]; HxW (default) or [3,H,W] if ``return_stack``.
    """
    b = np.asarray(before_rgb_u8)
    a = np.asarray(after_rgb_u8)
    if b.shape != a.shape:
        raise ValueError(f"before/after must match: {b.shape} vs {a.shape}")
    if b.ndim != 3 or b.shape[2] != 3:
        raise ValueError(f"expected HxWx3 RGB uint8, got {b.shape}")

    lab_b = _to_lab(b)
    lab_a = _to_lab(a)
    dL = np.abs(lab_a[..., 0] - lab_b[..., 0])                       # luminance
    da = lab_a[..., 1] - lab_b[..., 1]
    db = lab_a[..., 2] - lab_b[..., 2]
    dC = np.sqrt(da * da + db * db)                                  # chroma magnitude

    # SC proxy: a color-specific change is one where the *hue angle* rotated
    # (not just chroma scaled). Weight chroma diff by the absolute hue rotation.
    eps = 1e-6
    hue_b = np.arctan2(lab_b[..., 2], lab_b[..., 1])
    hue_a = np.arctan2(lab_a[..., 2], lab_a[..., 1])
    dhue = np.abs(np.arctan2(np.sin(hue_a - hue_b), np.cos(hue_a - hue_b)))  # [0,pi]
    sc_raw = dC * (dhue / np.pi)                                     # hue-rotation-weighted

    C_L = np.tanh(dL / max(l_scale, eps)).astype(np.float32)
    C_GC = np.tanh(dC / max(gc_scale, eps)).astype(np.float32)
    C_SC = np.tanh(sc_raw / max(sc_scale, eps)).astype(np.float32)

    stack = np.stack([C_L, C_GC, C_SC], axis=0)
    if region_mask01 is not None:
        rm = np.clip(np.asarray(region_mask01, dtype=np.float32), 0.0, 1.0)
        if rm.shape != stack.shape[1:]:
            raise ValueError(f"region_mask01 {rm.shape} != image {stack.shape[1:]}")
        stack = stack * rm[None, :, :]

    if blur_sigma_px and blur_sigma_px > 0:
        stack = np.stack([soft_blur(stack[i], blur_sigma_px) for i in range(3)], axis=0)

    stack = np.clip(stack, 0.0, 1.0).astype(np.float32)
    return stack if return_stack else cgt_collapse(stack)


# ===========================================================================
# 3. Sam3Masker  (concrete ConceptMasker)  -- lazy SAM3 detector.
# ===========================================================================

class Sam3Masker(ConceptMasker):
    """Offline text-promptable masker -> soft single-channel C_GT (probe §4/§5).

    Loads the HF SAM3 *video* model ONCE and uses its image ``detector_model``
    (a ``Sam3Model``); see probe §4 Option A and gotcha #1 (the on-disk checkpoint
    is the video model, weights are prefixed ``detector_model.*``). All forward
    passes run under ``torch.inference_mode``; NEVER backprop, NEVER run at
    VeraRetouch inference.

    The model is loaded lazily on first ``mask`` call so that importing this
    module (and the registry/streams that import it) stays weight-free for the
    parallel CPU subagents.
    """

    def __init__(
        self,
        model_dir: str = "/home/bc/data/models",
        device: str = "cuda",
        dtype: str = "bfloat16",
        score_threshold: float = 0.3,
        mask_threshold: float = 0.5,
        concept_map: Optional[Dict[str, List[str]]] = None,
    ) -> None:
        self.model_dir = model_dir
        self.device = device
        self.dtype = dtype
        self.score_threshold = float(score_threshold)
        self.mask_threshold = float(mask_threshold)
        self.concept_map: Dict[str, List[str]] = dict(concept_map or DEFAULT_CONCEPT_MAP)
        # Lazy handles (populated by _ensure_loaded()).
        self._det = None        # Sam3Model (the image detector)
        self._proc = None       # Sam3Processor
        self._torch = None

    # ----- lazy loading -----------------------------------------------------
    def _ensure_loaded(self) -> None:
        """Load SAM3 detector + processor exactly once (probe §4 Option A)."""
        if self._det is not None:
            return
        import torch  # lazy, heavy
        from transformers import Sam3Processor, Sam3VideoModel  # lazy, heavy

        self._torch = torch
        dtype = getattr(torch, self.dtype) if isinstance(self.dtype, str) else self.dtype
        vm = Sam3VideoModel.from_pretrained(
            self.model_dir, local_files_only=True, dtype=dtype
        ).eval()
        if self.device and self.device != "cpu":
            vm = vm.to(self.device)
        # The image detector is a Sam3Model built from cfg.detector_config (probe §4).
        self._det = vm.detector_model
        self._proc = Sam3Processor.from_pretrained(self.model_dir, local_files_only=True)

    # ----- helpers ----------------------------------------------------------
    @staticmethod
    def _as_pil(image: Any):
        """Accept a path / np.uint8 HxWx3 RGB / PIL.Image -> (PIL.Image, (H,W))."""
        from PIL import Image  # lazy
        if isinstance(image, (str, Path)):
            img = Image.open(str(image)).convert("RGB")
        elif isinstance(image, np.ndarray):
            img = Image.fromarray(np.ascontiguousarray(image).astype(np.uint8), "RGB")
        elif hasattr(image, "convert"):  # PIL.Image
            img = image.convert("RGB")
        else:
            raise TypeError(f"unsupported image type: {type(image)}")
        w, h = img.size
        return img, (h, w)

    # ----- ConceptMasker API ------------------------------------------------
    def mask(
        self,
        image: Any,
        concept: str,
        native_size: Optional[Tuple[int, int]] = None,
        soft: bool = True,
        reduce: str = "max",
        min_score: float = 0.3,
    ) -> np.ndarray:
        """One concept -> soft single-channel mask at native resolution.

        Returns float32 [0,1] HxW (soft) or float32 {0,1} HxW (hard, soft=False).
        Multiple SAM3 instances of the concept are combined by ``reduce``
        (``max`` = union-soft; ``sum`` = clipped accumulation). ``target_sizes``
        upsamples masks back to NATIVE res (probe §3/§8 gotcha #2); the source
        image itself is never resized.
        """
        self._ensure_loaded()
        torch = self._torch
        img, native = self._as_pil(image)
        H, W = native_size if native_size is not None else native

        inputs = self._proc(images=img, text=concept, return_tensors="pt")
        inputs = inputs.to(self._det.device)
        with torch.inference_mode():
            outputs = self._det(**inputs)
        results = self._proc.post_process_instance_segmentation(
            outputs,
            threshold=float(min(min_score, self.score_threshold)),
            mask_threshold=self.mask_threshold,
            target_sizes=[(H, W)],
        )
        res = results[0]
        masks = res.get("masks", None)
        scores = res.get("scores", None)
        if masks is None or len(masks) == 0:
            return np.zeros((H, W), dtype=np.float32)

        # masks: (num_instances, H, W). For the SOFT map we want sigmoid(logits);
        # post_process already applies sigmoid+threshold internally and returns
        # boolean/float masks. To keep a soft union we read pred per-instance.
        m = masks
        if hasattr(m, "detach"):
            m = m.detach().to(torch.float32).cpu().numpy()
        else:
            m = np.asarray(m, dtype=np.float32)
        if m.ndim == 2:
            m = m[None]

        # Filter instances by score.
        if scores is not None:
            sc = scores.detach().cpu().numpy() if hasattr(scores, "detach") else np.asarray(scores)
            keep = sc >= float(min_score)
            if keep.any():
                m = m[keep]
            else:
                return np.zeros((H, W), dtype=np.float32)

        if reduce == "sum":
            out = np.clip(m.sum(axis=0), 0.0, 1.0)
        else:  # "max" (default union-soft)
            out = m.max(axis=0)
        out = np.clip(out.astype(np.float32), 0.0, 1.0)

        if not soft:
            out = (out >= self.mask_threshold).astype(np.float32)
        return out

    def masks(self, image: Any, concepts: Sequence[str], **kw) -> Dict[str, np.ndarray]:
        """One soft map per concept (loops; one text phrase per SAM3 call, §3).

        Loads the image once and reuses the native size across concepts.
        """
        img, native = self._as_pil(image)
        kw.setdefault("native_size", native)
        return {c: self.mask(img, c, **kw) for c in concepts}

    def cgt_aspect_stack(
        self,
        image: Any,
        concept_map: Optional[Dict[str, List[str]]] = None,
    ) -> np.ndarray:
        """[3,H,W] float32 stack, channels = (L, GC, SC) per the 3-aspect spine.

        Each aspect channel = union (max) over its concept masks. The L channel
        is intentionally near-uniform for global edits (impl_plan_A §2.3).
        """
        cmap = concept_map or self.concept_map
        img, native = self._as_pil(image)
        H, W = native
        chans: List[np.ndarray] = []
        for aspect in ASPECTS:  # ("L","GC","SC")
            concepts = cmap.get(aspect, [])
            if not concepts:
                chans.append(np.zeros((H, W), dtype=np.float32))
                continue
            per = [self.mask(img, c, native_size=(H, W)) for c in concepts]
            chans.append(np.clip(np.stack(per, 0).max(0), 0.0, 1.0).astype(np.float32))
        return np.stack(chans, axis=0).astype(np.float32)

    # ----- Mode-1 convenience: concept-union single channel -----------------
    def cgt_from_concept(
        self,
        image: Any,
        concepts: Sequence[str],
        *,
        delta_norm: Optional[float] = None,
        tau: float = 1.0,
        blur_sigma_px: float = 2.0,
        min_score: float = 0.3,
    ) -> Tuple[np.ndarray, float]:
        """Mode-1 mask-derived single-channel C_GT from a concept union.

        ``C_GT = soft_blur( union_c M_c )`` and, if ``delta_norm`` is given, also
        magnitude-weighted by ``g = tanh(||dtheta||/tau)`` (impl_plan_A §2.3).
        Returns (C_GT float32 HxW in [0,1], best_concept_score).

        This is the workhorse for region-local recipe/degrade streams where the
        composite mask M_r IS the construction-time C_GT.
        """
        img, native = self._as_pil(image)
        H, W = native
        union = np.zeros((H, W), dtype=np.float32)
        best_score = 0.0
        for c in concepts:
            m = self.mask(img, c, native_size=(H, W), soft=True, min_score=min_score)
            cov = coverage(m, thr=self.mask_threshold)
            if cov > 0:
                best_score = max(best_score, cov)
            union = np.maximum(union, m)
        if delta_norm is not None:
            union = apply_magnitude(union, delta_norm, tau=tau)
        union = soft_blur(union, blur_sigma_px)
        return np.clip(union, 0.0, 1.0).astype(np.float32), float(best_score)


# ===========================================================================
# 4. C_GT storage  (1-ch native PNG + tiny patch-grid .npy + optional RLE).
#    Recipe-based: we DO persist the small C_GT maps (probe §7). All native res.
# ===========================================================================

def _u8(mask01: np.ndarray) -> np.ndarray:
    return np.clip(np.rint(np.asarray(mask01, dtype=np.float32) * 255.0), 0, 255).astype(np.uint8)


def save_cgt(
    mask01: np.ndarray,
    out_path: str,
    *,
    atomic: bool = True,
) -> str:
    """Save a single-channel C_GT as an 8-bit native-res PNG (probe §7).

    value = round(255 * C_GT). Writes .tmp then fsync+rename when ``atomic``
    (matches config storage.atomic_write). Returns the final path.
    """
    from PIL import Image  # lazy
    out_path = str(out_path)
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    arr = _u8(mask01)
    img = Image.fromarray(arr, mode="L")
    if atomic:
        tmp = out_path + ".tmp"
        img.save(tmp, format="PNG", optimize=True)
        with open(tmp, "rb") as f:
            os.fsync(f.fileno())
        os.replace(tmp, out_path)
    else:
        img.save(out_path, format="PNG", optimize=True)
    return out_path


def load_cgt(path: str) -> np.ndarray:
    """Load a 1-ch C_GT PNG -> float32 [0,1] HxW (inverse of save_cgt)."""
    from PIL import Image  # lazy
    img = Image.open(str(path)).convert("L")
    return (np.asarray(img, dtype=np.float32) / 255.0).astype(np.float32)


def save_cgt_stack(
    stack3: np.ndarray,
    out_paths: Dict[str, str],
    *,
    atomic: bool = True,
) -> Dict[str, str]:
    """Save a [3,H,W] (L,GC,SC) stack as three 1-ch PNGs (config cgt.store_aspect_stack).

    ``out_paths`` maps aspect -> path, e.g. {"L":..,"GC":..,"SC":..}.
    """
    arr = np.asarray(stack3, dtype=np.float32)
    if arr.ndim != 3 or arr.shape[0] != 3:
        raise ValueError(f"expected [3,H,W] stack, got {arr.shape}")
    written: Dict[str, str] = {}
    for i, aspect in enumerate(ASPECTS):
        if aspect in out_paths:
            written[aspect] = save_cgt(arr[i], out_paths[aspect], atomic=atomic)
    return written


def patch_grid_map(mask01: np.ndarray, grid: int = 16) -> np.ndarray:
    """Downsample C_GT to the VLM patch-grid resolution (config cgt.patch_grid).

    Area-averaging downsample to ``grid x grid`` (the model's C(x) lives on the
    patch grid and is bilinearly upsampled; probe §7 / dossier §0/§5). Returns
    float16 [0,1] for compact .npy storage.
    """
    import cv2  # lazy
    m = np.asarray(mask01, dtype=np.float32)
    small = cv2.resize(m, (grid, grid), interpolation=cv2.INTER_AREA)
    return np.clip(small, 0.0, 1.0).astype(np.float16)


def save_patch_grid(mask01: np.ndarray, out_path: str, grid: int = 16,
                    *, atomic: bool = True) -> str:
    """Save the tiny patch-grid map as a float16 .npy (probe §7)."""
    out_path = str(out_path)
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    pg = patch_grid_map(mask01, grid=grid)
    if atomic:
        tmp = out_path + ".tmp"
        np.save(tmp, pg)
        # np.save appends .npy if missing; normalize.
        if not os.path.exists(tmp) and os.path.exists(tmp + ".npy"):
            tmp = tmp + ".npy"
        with open(tmp, "rb") as f:
            os.fsync(f.fileno())
        os.replace(tmp, out_path)
    else:
        np.save(out_path, pg)
    return out_path


# ----- COCO-RLE (for hard binary construction masks; probe §7) -------------
def rle_encode(mask01: np.ndarray, thr: float = 0.5) -> Dict[str, Any]:
    """Binarize at ``thr`` and COCO-RLE encode via pycocotools (probe §7).

    Returns {"size":[H,W], "counts": <str>}. ``counts`` is decoded to a UTF-8 str
    so it is JSON-serializable inline in the manifest.
    """
    from pycocotools import mask as cocomask  # lazy
    m = (np.asarray(mask01, dtype=np.float32) >= thr).astype(np.uint8)
    rle = cocomask.encode(np.asfortranarray(m))
    counts = rle["counts"]
    if isinstance(counts, bytes):
        counts = counts.decode("ascii")
    return {"size": [int(m.shape[0]), int(m.shape[1])], "counts": counts}


def rle_decode(rle: Dict[str, Any]) -> np.ndarray:
    """Inverse of rle_encode -> float32 {0,1} HxW."""
    from pycocotools import mask as cocomask  # lazy
    counts = rle["counts"]
    if isinstance(counts, str):
        counts = counts.encode("ascii")
    r = {"size": [int(rle["size"][0]), int(rle["size"][1])], "counts": counts}
    return cocomask.decode(r).astype(np.float32)


# ----- CgtRef assembly (the contract record) -------------------------------
def build_cgt_ref(
    *,
    cgt_path: Optional[str] = None,
    cgt_aspect_paths: Optional[Dict[str, str]] = None,
    cgt_patchgrid_path: Optional[str] = None,
    rle: Optional[Dict[str, Any]] = None,
    mask_source: MaskSource = MaskSource.SAM3,
    concepts: Optional[Sequence[str]] = None,
    mask_score: Optional[float] = None,
    aspect_magnitude: Optional[Dict[str, float]] = None,
    cov: Optional[float] = None,
) -> CgtRef:
    """Assemble the contracts.CgtRef record (single source of truth for the schema)."""
    return CgtRef(
        cgt_path=cgt_path,
        cgt_aspect_paths=cgt_aspect_paths,
        cgt_patchgrid_path=cgt_patchgrid_path,
        rle=rle,
        mask_source=mask_source,
        concepts=list(concepts or []),
        mask_score=mask_score,
        aspect_magnitude=dict(aspect_magnitude or {}),
        coverage=cov,
    )


# ===========================================================================
# 5. Tiny CLI for a LIGHT self-test (no model weights).
#    Verifies the pure-numpy path (cgt_from_diff + storage + RLE round-trips)
#    on a synthetic pair. Heavy SAM3 path is exercised only in the GPU pilot.
# ===========================================================================

def _selftest() -> None:
    rng = np.random.default_rng(0)
    H, W = 64, 96
    before = rng.integers(0, 256, size=(H, W, 3), dtype=np.uint8)
    # 'after': brighten + warm-shift a corner region only.
    after = before.copy().astype(np.int16)
    after[:32, :48, 0] += 40   # warm R up
    after[:32, :48] += 25      # brighten
    after = np.clip(after, 0, 255).astype(np.uint8)

    # Mode-2 diff map.
    cgt = cgt_from_diff(before, after, blur_sigma_px=1.5)
    assert cgt.shape == (H, W) and cgt.dtype == np.float32
    assert 0.0 <= cgt.min() <= cgt.max() <= 1.0
    cov = coverage(cgt, thr=0.3)
    print(f"[diff] C_GT shape={cgt.shape} coverage>{0.3}={cov:.3f} "
          f"corner_mean={cgt[:32,:48].mean():.3f} rest_mean={cgt[32:,48:].mean():.3f}")
    assert cgt[:32, :48].mean() > cgt[32:, 48:].mean(), "edited corner must score higher"

    # 3-aspect stack + collapse.
    stack = cgt_from_diff(before, after, return_stack=True)
    assert stack.shape == (3, H, W)
    assert np.allclose(cgt_collapse(stack).max(), cgt.max(), atol=0.5)

    # Magnitude weighting.
    m = (cgt > 0.3).astype(np.float32)
    w = apply_magnitude(m, delta_norm=0.5, tau=1.0)
    assert abs(w.max() - np.tanh(0.5)) < 1e-5

    # Concept selection.
    cs = select_concepts(scene="portrait", recipe_tags=["warm film 人像"], max_concepts=3)
    print(f"[concepts] portrait+人像 -> {cs}")
    assert "skin" in cs or "face" in cs

    # Storage round-trips (PNG + patch-grid + RLE) in a temp dir.
    import tempfile
    with tempfile.TemporaryDirectory() as td:
        p = save_cgt(cgt, os.path.join(td, "a", "c.png"))
        back = load_cgt(p)
        assert back.shape == cgt.shape
        assert np.abs(back - cgt).mean() < 1.0 / 255 + 1e-3
        pg_path = save_patch_grid(cgt, os.path.join(td, "a", "c.npy"), grid=16)
        pg = np.load(pg_path)
        assert pg.shape == (16, 16) and pg.dtype == np.float16
        rle = rle_encode(cgt, thr=0.3)
        dec = rle_decode(rle)
        assert dec.shape == cgt.shape
        assert np.array_equal(dec, (cgt >= 0.3).astype(np.float32))
        print(f"[store] png={os.path.getsize(p)}B patchgrid={pg.shape} "
              f"rle_counts_len={len(rle['counts'])}")

    ref = build_cgt_ref(cgt_path="x.png", mask_source=MaskSource.SAM3,
                        concepts=cs, mask_score=0.8, cov=cov)
    assert ref.mask_source == MaskSource.SAM3
    print("[ok] masking.py light self-test passed (no model weights loaded).")


if __name__ == "__main__":
    import argparse

    ap = argparse.ArgumentParser(description="dataset_build/masking.py — Direction-A C_GT producer")
    ap.add_argument("--selftest", action="store_true",
                    help="run the light pure-numpy self-test (no SAM3 weights)")
    args = ap.parse_args()
    if args.selftest:
        _selftest()
    else:
        ap.print_help()
