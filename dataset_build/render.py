"""
dataset_build/render.py
=======================
The TEACHER + degradation-op module for the VeraRetouch Direction-A,
RECIPE-BASED, region-local-heavy dataset build.

This module supplies three things, all consistent with the abstract
``Renderer`` interface and the canonical ``region_composite`` signature in
``dataset_build/contracts.py``:

1. ``VeraRetouchRenderer`` — a programmatic, batched, native-resolution
   wrapper around the VeraRetouch param-mode renderer (the "after" teacher).
   It drives ``VeraRetouchForCausalLLM_Unified._generate`` directly (NOT the
   ``inference.py`` CLI), at the SOURCE native resolution (no resize/crop of
   the rendered image), batched over the VLM stage, under
   ``torch.inference_mode``.  Weights load ONCE, lazily — importing this module
   stays cheap (no torch/cv2/transformers at module top level).
   Grounded in: docs/plan/dataset/probe/probe_veraretouch_renderer.md,
   inference.py predict(), data/infer_dataset.py Infer_Param_Dataset,
   llava/model/VeraRetouch.py _generate, model/colormlp_v2.py forward_chunk.

2. ``region_composite`` — the Direction-A compositor. The renderer is a GLOBAL
   uniform transform; for C_GT to carry signal we render globally then blend
   the "after" inside a region mask M_r and use that mask as C_GT
   (MEMORY caveat b; probe §"Region composite"; impl_plan_A §2.3).

3. ``params_from_degrade_spec`` — projects a DegradeSpec to teacher-renderable
   PARAM params. The old CPU ``inverse_degradation`` kernels are kept below only
   as legacy utilities; DATAGEN v2 build paths do not call or export them.

DESIGN / IMPORT-WEIGHT CONTRACT
  - Top-level imports are stdlib + contracts ONLY.  ``numpy``, ``cv2``,
    ``torch``, ``transformers``, ``box``, repo modules (``llava``, ``data``,
    ``utils``) are imported lazily inside methods/functions.  This keeps
    ``import dataset_build.render`` usable on every conda env and avoids
    pulling 2.2 GB of weights at import time.
  - Heavy model weights are NEVER loaded at import; only the first call to a
    method that needs the model triggers the load.  Render runs under
    ``torch.inference_mode`` and is never trained through (the teacher is a
    forward-only target generator; cf. MEMORY: SA-LUT no_grad bridge).
  - cwd MUST be /home/bc/VeraRetouch when the model loads (repo-root-relative
    imports: ``from utils import ...``, ``from data.infer_dataset import ...``).

Run a CPU-only self-test of the weights-free ops (no model load):
    cd /home/bc/VeraRetouch
    python -m dataset_build.render --selftest
"""

from __future__ import annotations

import argparse
import os
from typing import Any, Dict, List, Optional, Sequence, Tuple

# contracts is import-light (stdlib only) — safe at top level.
from dataset_build.contracts import (
    COLORMIXER_KEYS,
    COLORTEMP_KEYS,
    DegradeSpec,
    LIGHT_KEYS,
    PARAM_KEYS,
    Renderer,
)

__all__ = [
    "VeraRetouchRenderer",
    "region_composite",
    "params_from_degrade_spec",
]

# Default model / render knobs (mirror config.yaml: models.veraretouch).
_DEFAULT_MODEL_PATH = "/home/bc/data/models/VeraRetouch"
_DEFAULT_CONFIG_ADD = "./configs/infer_config.yaml"
_DEFAULT_CHUNK = 262144          # 512*512 pixel tiling for native-res render
_DEFAULT_MAX_NEW_TOKENS = 256    # tight for data-gen (CLI default 4096 is wasteful)
_DEFAULT_BATCH = 4
_DEFAULT_NUM_WORKERS = 0


# ===========================================================================
# 1. The TEACHER: VeraRetouch param-mode renderer (lazy, batched, native-res).
# ===========================================================================

class VeraRetouchRenderer(Renderer):
    """Programmatic VeraRetouch param-mode renderer = the "after" teacher.

    GLOBAL uniform transform.  Construct once (defers the actual weight load to
    the first ``render`` call via ``_ensure_loaded``), then call ``render`` many
    times.  Outputs are native-resolution ``np.uint8 HxWx3 RGB`` (no resize/crop
    on the rendered image; the 512p control image is VLM-only).

    Parameters
    ----------
    model_path : str
        VeraRetouch checkpoint dir (config.yaml: models.veraretouch.model_path).
    config_add_path : str
        infer_config.yaml relative to cwd (which must be /home/bc/VeraRetouch).
    device : str
        "cuda" (default) or "cpu" (cpu only for tiny smoke tests; very slow).
    dtype : str
        "bfloat16" (default) | "float16" | "float32".
    max_new_tokens : int
        VLM decode cap; 256 is plenty for the short plan + 3 retouch tokens.
    greedy : bool
        temperature==0 deterministic teacher (config.yaml: greedy: true).
    """

    def __init__(
        self,
        model_path: str = _DEFAULT_MODEL_PATH,
        config_add_path: str = _DEFAULT_CONFIG_ADD,
        device: str = "cuda",
        dtype: str = "bfloat16",
        max_new_tokens: int = _DEFAULT_MAX_NEW_TOKENS,
        greedy: bool = True,
        num_workers: int = _DEFAULT_NUM_WORKERS,
    ) -> None:
        self.model_path = model_path
        self.config_add_path = config_add_path
        self.device = device
        self.dtype_str = dtype
        self.max_new_tokens = int(max_new_tokens)
        self.greedy = bool(greedy)
        self.num_workers = int(num_workers)
        # Heavy handles, populated lazily by _ensure_loaded().
        self._loaded = False
        self._torch = None
        self._cv2 = None
        self._np = None
        self.model = None
        self.tok = None
        self.image_processor = None
        self.collate = None
        self._torch_dtype = None

    # ---- lazy model load --------------------------------------------------
    def _ensure_loaded(self) -> None:
        """Import heavy deps + load weights ONCE (idempotent).

        cwd must be /home/bc/VeraRetouch (repo-root-relative imports). Mirrors
        the probe spec and inference.py predict() setup exactly.
        """
        if self._loaded:
            return

        import cv2  # noqa: F401
        import numpy as np
        import torch
        import yaml
        from box import Box
        from transformers import AutoTokenizer

        # Repo-root-relative imports (cwd == /home/bc/VeraRetouch).
        from llava.utils import disable_torch_init
        from llava.model.VeraRetouch import VeraRetouchForCausalLLM_Unified
        from llava.constants import (
            DEFAULT_RETOUCH_LIGHT_TOKEN,
            DEFAULT_RETOUCH_COLORTEMP_TOKEN,
            DEFAULT_RETOUCH_COLORMIXER_TOKEN,
        )
        from data.infer_dataset import DataCollatorForUnifiedTestDataset

        self._cv2 = cv2
        self._np = np
        self._torch = torch

        dtype_map = {
            "bfloat16": torch.bfloat16,
            "float16": torch.float16,
            "float32": torch.float32,
        }
        self._torch_dtype = dtype_map.get(self.dtype_str, torch.bfloat16)

        with open(self.config_add_path) as f:
            cfg = Box(yaml.safe_load(f))
        # Match the CLI param-mode setup (probe + inference.py L74-75).
        cfg.project_name = "test_no_instruct"
        cfg.freeze_retouch_decoder = True

        disable_torch_init()
        # model_max_length governs INPUT truncation and must NOT be tied to the
        # generation cap: the param-mode prompt is ~400-700 tokens, so a 256 cap
        # truncated it and the VLM never emitted the retouch tokens. Keep input
        # generous (4096, as inference.py does); max_new_tokens caps generation only.
        self.tok = AutoTokenizer.from_pretrained(
            self.model_path,
            model_max_length=max(4096, self.max_new_tokens),
            padding_side="right",
            use_fast=False,
        )
        self.model = (
            VeraRetouchForCausalLLM_Unified.from_pretrained(
                self.model_path, config_add=cfg, torch_dtype=self._torch_dtype
            )
            .to(self.device)
            .eval()
        )
        # Register the 3 retouch special-token ids (light / colortemp / colormixer).
        self.model.register_special_token_idx(
            self.tok(DEFAULT_RETOUCH_LIGHT_TOKEN, add_special_tokens=False).input_ids[0],
            self.tok(DEFAULT_RETOUCH_COLORTEMP_TOKEN, add_special_tokens=False).input_ids[0],
            self.tok(DEFAULT_RETOUCH_COLORMIXER_TOKEN, add_special_tokens=False).input_ids[0],
        )
        self.model.generation_config.pad_token_id = self.tok.pad_token_id
        self.image_processor = self.model.get_vision_tower().image_processor
        self.collate = DataCollatorForUnifiedTestDataset(tokenizer=self.tok)
        self._loaded = True

    # ---- the Renderer contract -------------------------------------------
    def render(
        self,
        image_paths: Sequence[str],
        param_dicts: Sequence[Dict[str, Dict[str, float]]],
        batch_size: int = _DEFAULT_BATCH,
        chunk: int = _DEFAULT_CHUNK,
        max_new_tokens: int = _DEFAULT_MAX_NEW_TOKENS,
    ) -> List[Any]:
        """Render the "after" for each (image_path, param_dict) at native res.

        Parameters
        ----------
        image_paths : Sequence[str]
            ABS native-resolution source images. DNG/CR2 must be pre-decoded to
            sRGB PNG/TIFF first (cv2.imread won't decode raw) — see registry /
            RawDecode.RAWPY (MEMORY caveat c; probe §Gotchas).
        param_dicts : Sequence[Dict[str, Dict[str, float]]]
            RAW Lightroom-unit CRS dicts {key: {"value": N}} (same length as
            image_paths). Only PARAM_KEYS are consumed; values are /100 inside
            the dataset (get_organized_dict L281-287). Pass RAW units.
        batch_size : int
            VLM-stage batch (the decoder loop is bs=1 internally; probe gotcha).
        chunk : int
            Pixel-tiling block for free-resolution native render (caps peak mem;
            output is bit-identical to non-chunked; probe §"Free-resolution").
        max_new_tokens : int
            VLM decode cap for this call (overrides the ctor default).

        Returns
        -------
        List[np.uint8 HxWx3 RGB]
            One "after" per input, at the SOURCE native resolution.
        """
        if len(image_paths) != len(param_dicts):
            raise ValueError(
                f"image_paths ({len(image_paths)}) and param_dicts "
                f"({len(param_dicts)}) must have equal length."
            )
        if not image_paths:
            return []

        self._ensure_loaded()
        torch = self._torch
        cv2 = self._cv2
        from torch.utils.data import DataLoader

        # Normalise every param dict to the canonical RAW-unit form.
        norm_params = [normalize_param_dict(p) for p in param_dicts]

        ds = _ParamDatasetFromDicts(
            img_paths=list(image_paths),
            param_dicts=norm_params,
            tokenizer=self.tok,
            image_processor=self.image_processor,
        )
        loader = DataLoader(
            ds,
            batch_size=batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            collate_fn=self.collate,
        )

        out: List[Any] = []
        for batch in loader:
            input_ids = batch["input_ids"].to(self.device)
            images = [t.to(self._torch_dtype).to(self.device) for t in batch["images"]]
            input_imgs = [t.to(self._torch_dtype).to(self.device) for t in batch["input_imgs"]]
            retouch_masks = [t.to(self._torch_dtype).to(self.device) for t in batch["retouch_masks"]]
            attn = batch["attention_mask"].to(self.device)

            bgr_list, _txt = self.model._generate(
                tokenizer=self.tok,
                inputs=input_ids,
                attention_mask=attn,
                images=images,
                image_sizes=batch["image_sizes"],
                retouch_masks=retouch_masks,
                input_imgs=input_imgs,
                do_sample=(not self.greedy),
                temperature=(0.0 if self.greedy else 0.2),
                top_p=1.0,
                num_beams=1,
                max_new_tokens=int(max_new_tokens),
                output_hidden_states=True,
                return_dict_in_generate=True,
                chunk=chunk,
            )
            # _generate yields BGR uint8 numpy (CLI writes with cv2). -> RGB.
            out += [cv2.cvtColor(b, cv2.COLOR_BGR2RGB) for b in bgr_list]
        return out


class _ParamDatasetFromDicts:
    """In-memory variant of ``Infer_Param_Dataset`` that takes param DICTS.

    Subclassing the real dataset avoids tmp-json churn over 1M samples (probe
    §"_ParamDatasetFromDicts"). We import the base class lazily so module import
    stays cheap; we therefore build the class on first instantiation.
    """

    def __new__(cls, img_paths, param_dicts, tokenizer, image_processor):
        from data.infer_dataset import Infer_Param_Dataset

        class _Impl(Infer_Param_Dataset):
            def __init__(self, img_paths, param_dicts, tokenizer, image_processor):
                super().__init__(
                    img_paths=img_paths,
                    instruction_paths=["<in-memory>"] * len(img_paths),
                    tokenizer=tokenizer,
                    image_processor=image_processor,
                )
                self._param_dicts = param_dicts

            def parse_json_file(self, file_path):  # noqa: ARG002 - signature kept
                # __getitem__ calls parse_json_file(self.instruction_paths[idx]);
                # we ignore the path and serve the matching in-memory dict by
                # locating its index. Because instruction_paths is a placeholder
                # list, resolve via identity is unsafe; instead we override
                # __getitem__ to thread the index directly.
                raise RuntimeError("parse_json_file should not be reached")

            def __getitem__(self, idx):
                # Re-implement only the param-sourcing line of the base
                # __getitem__; everything else (image tensors, prompt, mask)
                # is identical. We temporarily inject the dict via a closure.
                import cv2
                import os as _os
                from PIL import Image
                from torchvision import transforms
                from data.infer_dataset import (
                    TASK_PROFESSIONAL_RETOUCH_TOKEN,
                    conv_templates,
                    tokenizer_image_token,
                    IMAGE_TOKEN_INDEX,
                    process_images_,
                )

                json_content = self._param_dicts[idx]
                retouch_params = self.get_organized_dict(json_content)
                mask = self.get_mask_from_path(retouch_params)

                qs = (
                    f"{TASK_PROFESSIONAL_RETOUCH_TOKEN}\n"
                    "Now, you are acting as a Retouch Agent. I will provide an "
                    "image and an professional instruction (plain text "
                    "description or retouch operator parameters range from -1.0 "
                    "to 1.0), please give me a retouch plan and retouch tokens.\n"
                    f" Instruction: {retouch_params}"
                )
                conv = conv_templates[self.conv_mode].copy()
                conv.append_message(conv.roles[0], qs)
                conv.append_message(conv.roles[1], None)
                prompt = conv.get_prompt()
                input_ids = tokenizer_image_token(
                    prompt, self.tokenizer, IMAGE_TOKEN_INDEX, return_tensors="pt"
                )

                input_img_path = self.img_paths[idx]
                input_img_ = cv2.imread(input_img_path, cv2.IMREAD_COLOR)
                if input_img_ is None:
                    raise FileNotFoundError(
                        f"cv2.imread returned None for {input_img_path} "
                        "(raw files must be pre-decoded to sRGB PNG/TIFF)."
                    )
                input_img_tensor = transforms.ToTensor()(
                    cv2.cvtColor(input_img_, cv2.COLOR_BGR2RGB)
                )
                input_img = (input_img_tensor - 0.5) * 2  # native res, [-1,1]

                image = Image.open(input_img_path).convert("RGB")
                image_512p = self.resize2_512p(image)
                image_tensor = process_images_([image_512p], self.image_processor)[0]
                image_size = image_512p.size
                file_name = _os.path.basename(input_img_path).split(".")[0]
                return dict(
                    input_ids=input_ids.squeeze(),
                    image=image_tensor,
                    image_size=image_size,
                    file_name=file_name,
                    input_img=input_img,
                    retouch_mask=mask,
                    input_img_path=input_img_path,
                )

        return _Impl(img_paths, param_dicts, tokenizer, image_processor)


# ===========================================================================
# 2. Direction-A region compositor (the canonical contracts.region_composite).
# ===========================================================================

def region_composite(after_rgb_u8: Any, source_rgb_u8: Any, mask01: Any) -> Any:
    """Blend the GLOBAL "after" inside a region mask; C_GT == mask01.

    The renderer is a global uniform transform, so a region-local edit is
    realised as ``source*(1-m) + after*m`` and the mask M_r is exactly the
    Direction-A C_GT (MEMORY caveat b; probe §"Region composite";
    impl_plan_A §2.3).

    Parameters
    ----------
    after_rgb_u8, source_rgb_u8 : np.uint8 HxWx3
        The globally-rendered "after" and the original source, SAME native HxW.
    mask01 : np.ndarray HxW (or HxWx1) float in [0,1]
        The region mask (SAM3 soft / degrade exact). Resized to (H,W) if needed
        — we resize the MASK, never the images.

    Returns
    -------
    np.uint8 HxWx3
        The composited "after". For a soft mask the boundary is alpha-blended.
    """
    import numpy as np

    after = np.asarray(after_rgb_u8)
    source = np.asarray(source_rgb_u8)
    if after.shape != source.shape:
        raise ValueError(
            f"after {after.shape} and source {source.shape} must match (no image resize)."
        )
    h, w = source.shape[:2]

    m = np.asarray(mask01, dtype=np.float32)
    if m.ndim == 3:
        m = m[..., 0]
    if m.shape != (h, w):
        # Resize the MASK only (never the images), to native HxW.
        import cv2

        m = cv2.resize(m, (w, h), interpolation=cv2.INTER_LINEAR)
    m = np.clip(m, 0.0, 1.0)[..., None]

    comp = source.astype(np.float32) * (1.0 - m) + after.astype(np.float32) * m
    return comp.round().clip(0, 255).astype(np.uint8)


# ===========================================================================
# 3. Degradation ops (Track-B parametric; global + region-local).
# ===========================================================================

# CRS-unit -> normalized scale used by the renderer (get_organized_dict /100).
_HSL_BANDS = ("Red", "Orange", "Yellow", "Green", "Aqua", "Blue", "Purple", "Magenta")
# Approximate hue centres (degrees) for each Lightroom HSL band, for the local
# color-mixer ops in the weights-free degradation.
_BAND_HUE_DEG = {
    "Red": 0.0, "Orange": 30.0, "Yellow": 60.0, "Green": 120.0,
    "Aqua": 180.0, "Blue": 240.0, "Purple": 280.0, "Magenta": 320.0,
}


def normalize_param_dict(
    params: Optional[Dict[str, Dict[str, float]]]
) -> Dict[str, Dict[str, float]]:
    """Coerce a param dict to the canonical RAW-unit ``{key:{"value":float}}``.

    - Drops any key not in ``PARAM_KEYS`` (param-mode is global; local-mask CRS
      keys are recorded as region hints elsewhere, never sent to the renderer).
    - Accepts either ``{"k": {"value": N}}`` or the shorthand ``{"k": N}``.
    - Missing keys are NOT injected (get_organized_dict defaults them to 0).
    """
    out: Dict[str, Dict[str, float]] = {}
    if not params:
        return out
    valid = set(PARAM_KEYS)
    for k, v in params.items():
        if k not in valid:
            continue
        if isinstance(v, dict):
            val = float(v.get("value", 0.0))
        else:
            val = float(v)
        out[k] = {"value": val}
    return out


def params_from_degrade_spec(spec: DegradeSpec) -> Dict[str, Dict[str, float]]:
    """Project a ``DegradeSpec.op_params`` onto the renderer param schema.

    Used when a degrade sample's "after" should be produced by the TEACHER
    (param mode) rather than by the local CPU ops — keeps only PARAM_KEYS,
    raw LR units, and applies the ``forward`` sign (False => negate to invert).
    """
    sign = 1.0 if spec.forward else -1.0
    raw = {k: float(v) * sign for k, v in (spec.op_params or {}).items()}
    return normalize_param_dict(raw)


def inverse_degradation(img: Any, spec: DegradeSpec, mask01: Optional[Any] = None) -> Any:
    """Apply a ``DegradeSpec`` as concrete CRS-style image ops (weights-free).

    LEGACY ONLY. DATAGEN v2 degrade streams store teacher-renderable PARAM
    recipes via ``params_from_degrade_spec`` and do not call this function.
    It realises ``spec.op_params`` (raw Lightroom
    units, a subset of ``PARAM_KEYS``) plus optional blur/noise as deterministic
    numpy/cv2 ops on an sRGB image.

    ``spec.forward`` semantics (DegradeSpec doc):
      - ``forward=True``  : apply the operator (clean -> edited).
      - ``forward=False`` : INVERT the operator (apply the negative edit) to
        synthesise a degraded "before" from a clean source. The exact C_GT is
        ``mask01`` (region-local) or all-ones (global).

    Parameters
    ----------
    img : np.uint8 HxWx3 RGB (or float HxWx3 in [0,1]).
    spec : DegradeSpec
        ``op_params`` raw LR units; optional ``op_params['__blur_sigma']`` and
        ``op_params['__noise_std']`` extras (px / 0..1) drive blur+noise.
    mask01 : Optional[np.ndarray HxW] in [0,1]
        If given, the degradation is applied ONLY inside the mask
        (``out = src*(1-m) + degraded*m``) and that mask is the exact C_GT.
        If None, the op is global.

    Returns
    -------
    np.uint8 HxWx3 RGB
        The degraded (or edited) image at native resolution (no resize).
    """
    import numpy as np

    arr = np.asarray(img)
    was_uint8 = arr.dtype == np.uint8
    x = arr.astype(np.float32)
    if was_uint8:
        x = x / 255.0
    x = np.clip(x, 0.0, 1.0)
    src = x.copy()

    sign = 1.0 if spec.forward else -1.0
    ops = {k: float(v) * sign for k, v in (spec.op_params or {}).items()}

    x = _apply_light_ops(x, ops, np)
    x = _apply_colortemp_ops(x, ops, np)
    x = _apply_colormixer_ops(x, ops, np)

    # Optional non-CRS degradations (blur/noise) carried in op_params extras.
    blur_sigma = float((spec.op_params or {}).get("__blur_sigma", 0.0))
    if blur_sigma > 0.0:
        import cv2

        k = max(3, int(round(blur_sigma * 3)) * 2 + 1)
        x = cv2.GaussianBlur(x, (k, k), blur_sigma)
    noise_std = float((spec.op_params or {}).get("__noise_std", 0.0))
    if noise_std > 0.0:
        rng = np.random.default_rng(spec.seed if spec.seed is not None else 0)
        x = x + rng.normal(0.0, noise_std, size=x.shape).astype(np.float32)

    x = np.clip(x, 0.0, 1.0)

    if mask01 is not None:
        m = np.asarray(mask01, dtype=np.float32)
        if m.ndim == 3:
            m = m[..., 0]
        if m.shape != x.shape[:2]:
            import cv2

            m = cv2.resize(m, (x.shape[1], x.shape[0]), interpolation=cv2.INTER_LINEAR)
        m = np.clip(m, 0.0, 1.0)[..., None]
        x = src * (1.0 - m) + x * m

    x = np.clip(x, 0.0, 1.0)
    return (x * 255.0).round().clip(0, 255).astype(np.uint8)


# ---- internal op kernels (operate on float HxWx3 RGB in [0,1]) ------------

def _apply_light_ops(x: Any, ops: Dict[str, float], np: Any) -> Any:
    """Exposure / Contrast / Highlights / Shadows / Whites / Blacks + parametric.

    Values are raw LR units (~[-100,100]); /100 -> a unit-scale strength, same
    convention as get_organized_dict. Approximate but monotone & invertible.
    """
    # Exposure: stops-ish multiplicative push in linear-ish space.
    exp = ops.get("Exposure2012", 0.0) / 100.0
    if exp:
        x = x * (2.0 ** exp)

    # Contrast about mid-grey.
    con = ops.get("Contrast2012", 0.0) / 100.0
    if con:
        x = (x - 0.5) * (1.0 + con) + 0.5

    # Highlights / Whites lift the bright end; Shadows / Blacks lift the dark end.
    hi = (ops.get("Highlights2012", 0.0) + 0.5 * ops.get("Whites2012", 0.0)) / 100.0
    if hi:
        w = np.clip((x - 0.5) * 2.0, 0.0, 1.0)  # weight toward highlights
        x = x + hi * 0.5 * w
    sh = (ops.get("Shadows2012", 0.0) + 0.5 * ops.get("Blacks2012", 0.0)) / 100.0
    if sh:
        w = np.clip((0.5 - x) * 2.0, 0.0, 1.0)  # weight toward shadows
        x = x + sh * 0.5 * w

    # Parametric tone (four zones) — coarse, weighted bumps by luminance zone.
    para = {
        "ParametricBlacks": (0.00, 0.20),
        "ParametricShadows": (0.20, 0.40),
        "ParametricDarks": (0.30, 0.55),
        "ParametricLights": (0.45, 0.70),
        "ParametricHighlights": (0.70, 1.00),
    }
    lum = (0.299 * x[..., 0] + 0.587 * x[..., 1] + 0.114 * x[..., 2])
    for key, (lo, hi_z) in para.items():
        amt = ops.get(key, 0.0) / 100.0
        if amt:
            mid = 0.5 * (lo + hi_z)
            half = max(1e-3, 0.5 * (hi_z - lo))
            w = np.clip(1.0 - np.abs(lum - mid) / half, 0.0, 1.0)[..., None]
            x = x + amt * 0.3 * w
    return x


def _apply_colortemp_ops(x: Any, ops: Dict[str, float], np: Any) -> Any:
    """Temperature / Tint / Vibrance / Saturation (global color & temp)."""
    temp = ops.get("IncrementalTemperature", 0.0) / 100.0
    if temp:
        # Warm: +R, -B. Cool: -R, +B.
        x[..., 0] = x[..., 0] + 0.10 * temp
        x[..., 2] = x[..., 2] - 0.10 * temp
    tint = ops.get("IncrementalTint", 0.0) / 100.0
    if tint:
        # +Magenta (-G) / -Green (+G).
        x[..., 1] = x[..., 1] - 0.10 * tint

    sat = ops.get("Saturation", 0.0) / 100.0
    vib = ops.get("Vibrance", 0.0) / 100.0
    if sat or vib:
        lum = (0.299 * x[..., 0] + 0.587 * x[..., 1] + 0.114 * x[..., 2])[..., None]
        if sat:
            x = lum + (x - lum) * (1.0 + sat)
        if vib:
            # Vibrance scales more for low-saturation pixels (gentle on skin).
            chroma = np.linalg.norm(x - lum, axis=-1, keepdims=True)
            w = np.clip(1.0 - chroma, 0.0, 1.0)
            x = lum + (x - lum) * (1.0 + vib * (0.5 + 0.5 * w))
    return x


def _apply_colormixer_ops(x: Any, ops: Dict[str, float], np: Any) -> Any:
    """HSL Hue/Saturation/Luminance per the 8 Lightroom color bands.

    Coarse HSV-space band-weighted edit; sufficient as a degradation operator
    (the exact teacher path uses the renderer, not this kernel).
    """
    active = [
        b for b in _HSL_BANDS
        if ops.get(f"HueAdjustment{b}", 0.0)
        or ops.get(f"SaturationAdjustment{b}", 0.0)
        or ops.get(f"LuminanceAdjustment{b}", 0.0)
    ]
    if not active:
        return x

    import cv2

    x_clip = np.clip(x, 0.0, 1.0).astype(np.float32)
    hsv = cv2.cvtColor(x_clip, cv2.COLOR_RGB2HSV)  # H in [0,360), S,V in [0,1]
    h = hsv[..., 0]
    for b in active:
        centre = _BAND_HUE_DEG[b]
        # Circular distance to band centre (degrees), band half-width ~30deg.
        d = np.abs(((h - centre + 180.0) % 360.0) - 180.0)
        w = np.clip(1.0 - d / 30.0, 0.0, 1.0)
        hue_amt = ops.get(f"HueAdjustment{b}", 0.0) / 100.0
        sat_amt = ops.get(f"SaturationAdjustment{b}", 0.0) / 100.0
        lum_amt = ops.get(f"LuminanceAdjustment{b}", 0.0) / 100.0
        if hue_amt:
            hsv[..., 0] = (hsv[..., 0] + hue_amt * 30.0 * w) % 360.0
        if sat_amt:
            hsv[..., 1] = np.clip(hsv[..., 1] * (1.0 + sat_amt * w), 0.0, 1.0)
        if lum_amt:
            hsv[..., 2] = np.clip(hsv[..., 2] * (1.0 + lum_amt * 0.5 * w), 0.0, 1.0)
    return cv2.cvtColor(hsv, cv2.COLOR_HSV2RGB)


# ===========================================================================
# 4. CPU-only self-test (no model weights).
# ===========================================================================

def _selftest() -> int:
    """Exercise region_composite + param projection on a synthetic image.

    Loads NO model weights. The legacy CPU inverse_degradation path is not part
    of the DATAGEN v2 build contract and is intentionally not tested here.
    """
    import numpy as np

    rng = np.random.default_rng(0)
    src = (rng.uniform(0, 1, size=(64, 96, 3)) * 255).astype(np.uint8)

    # --- normalize_param_dict drops non-schema keys, accepts shorthand ---
    p = normalize_param_dict(
        {"Exposure2012": {"value": 30}, "Saturation": 20, "NotAKey": 5}
    )
    assert set(p) == {"Exposure2012", "Saturation"}, p
    assert p["Exposure2012"]["value"] == 30.0 and p["Saturation"]["value"] == 20.0

    spec_inv = DegradeSpec(
        mode="gaussian_op",
        op_params={"Exposure2012": 40.0, "Saturation": 30.0, "IncrementalTemperature": 25.0},
        aspects=["L", "GC"],
        forward=False,
        seed=7,
    )
    # --- params_from_degrade_spec keeps schema keys, applies sign ---
    proj = params_from_degrade_spec(spec_inv)
    assert proj["Exposure2012"]["value"] == -40.0  # forward=False negates
    assert proj["Saturation"]["value"] == -30.0

    # --- region_composite blends after inside mask; outside == source ---
    mask = np.zeros((64, 96), dtype=np.float32)
    mask[16:48, 24:72] = 1.0
    outside = (mask == 0)
    after = np.full_like(src, 128)
    comp = region_composite(after, src, mask)
    assert np.array_equal(comp[outside], src[outside])
    assert np.array_equal(comp[mask > 0], after[mask > 0])

    print("render.py self-test OK:")
    print(f"  PARAM_KEYS={len(PARAM_KEYS)} (LIGHT={len(LIGHT_KEYS)} "
          f"CT={len(COLORTEMP_KEYS)} CM={len(COLORMIXER_KEYS)})")
    print("  DATAGEN v2: degrade projected to teacher PARAM recipe; CPU inverse is legacy")
    return 0


def _build_arg_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(
        prog="dataset_build.render",
        description="VeraRetouch teacher renderer + degradation ops.",
    )
    ap.add_argument("--selftest", action="store_true",
                    help="Run the CPU-only weights-free self-test and exit.")
    ap.add_argument("--render", action="store_true",
                    help="LOADS WEIGHTS: render a single image with --image/--exposure.")
    ap.add_argument("--image", type=str, default=None, help="source image path (--render).")
    ap.add_argument("--exposure", type=float, default=30.0,
                    help="Exposure2012 raw LR units for the --render smoke test.")
    ap.add_argument("--out", type=str, default=None, help="output PNG path (--render).")
    ap.add_argument("--chunk", type=int, default=_DEFAULT_CHUNK)
    return ap


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = _build_arg_parser().parse_args(argv)
    if args.selftest:
        return _selftest()
    if args.render:
        if not args.image:
            raise SystemExit("--render requires --image")
        import cv2

        r = VeraRetouchRenderer()
        afters = r.render([args.image], [{"Exposure2012": {"value": args.exposure}}],
                          chunk=args.chunk)
        out = args.out or (os.path.splitext(args.image)[0] + "_after.png")
        cv2.imwrite(out, cv2.cvtColor(afters[0], cv2.COLOR_RGB2BGR))
        print(f"wrote {out}  shape={afters[0].shape}")
        return 0
    _build_arg_parser().print_help()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
