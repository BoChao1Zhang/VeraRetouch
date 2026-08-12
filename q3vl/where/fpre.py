"""Protocol 4.1 -- ``F_pre``: the last vision block's output, before the merger.

    F_pre in R^[B x H/16 x W/16 x 1024]

Two facts this module depends on, both re-verified on 2026-08-05 against the
installed transformers 4.57.1 and the local Qwen3-VL-4B-Instruct config:

1. ``vision_config.hidden_size == 1024`` and ``patch_size == 16``, so the
   pre-merger token grid is exactly ``H/16 x W/16``;
2. the pre-merger token *order* is ``(grid_h//2, grid_w//2, 2, 2)``, not plain
   row-major.  ``Qwen2VLImageProcessorFast`` builds the patch sequence with
   ``view(..., gh//m, m, ps, gw//m, m, ps).permute(0,1,4,7,5,8,3,2,6,9)`` and
   ``Qwen3VLVisionModel.fast_pos_embed_interpolate`` re-permutes the position
   grid the same way.  Reshaping ``F_pre`` with a naive ``view(gh, gw, C)``
   therefore scrambles the image; :func:`unshuffle_to_grid` is the inverse and
   :mod:`q3vl.where.tests.test_fpre` verifies it against the real processor.

Base SFT freezes ``patch_embed``, ``pos_embed`` and all 24 vision blocks
(``q3vl/train/freeze.py``), so ``F_pre`` is identical under the base weights and
under any Base-SFT checkpoint.  That is asserted once, on real weights, by
preflight item WA-P4b.
"""

from __future__ import annotations

from contextlib import contextmanager
from pathlib import Path
from typing import Any, Iterator

import torch

from .config import FPRE_DIM, SPATIAL_MERGE, VISION_PATCH

__all__ = [
    "unshuffle_to_grid",
    "shuffle_from_grid",
    "grid_from_geometry",
    "FPreHook",
    "MergerHook",
    "load_vision_tower",
    "extract_fpre",
]


def grid_from_geometry(out_h: int, out_w: int) -> tuple[int, int]:
    """``(grid_h, grid_w) = (H/16, W/16)`` -- protocol 4.1."""
    if out_h % VISION_PATCH or out_w % VISION_PATCH:
        raise ValueError(f"{out_h}x{out_w} is not a multiple of {VISION_PATCH}")
    return out_h // VISION_PATCH, out_w // VISION_PATCH


def unshuffle_to_grid(
    x: torch.Tensor, grid_h: int, grid_w: int, merge: int = SPATIAL_MERGE
) -> torch.Tensor:
    """``(grid_h*grid_w, C)`` in Qwen merge order -> ``(grid_h, grid_w, C)``."""
    if x.dim() != 2:
        raise ValueError(f"expected (N, C), got {tuple(x.shape)}")
    n, c = x.shape
    if n != grid_h * grid_w:
        raise ValueError(f"{n} tokens != grid {grid_h}x{grid_w}")
    if grid_h % merge or grid_w % merge:
        raise ValueError(f"grid {grid_h}x{grid_w} is not divisible by merge {merge}")
    return (
        x.reshape(grid_h // merge, grid_w // merge, merge, merge, c)
        .permute(0, 2, 1, 3, 4)
        .reshape(grid_h, grid_w, c)
    )


def shuffle_from_grid(
    g: torch.Tensor, merge: int = SPATIAL_MERGE
) -> torch.Tensor:
    """Inverse of :func:`unshuffle_to_grid` (used by the round-trip test)."""
    if g.dim() != 3:
        raise ValueError(f"expected (H, W, C), got {tuple(g.shape)}")
    gh, gw, c = g.shape
    return (
        g.reshape(gh // merge, merge, gw // merge, merge, c)
        .permute(0, 2, 1, 3, 4)
        .reshape(gh * gw, c)
    )


class FPreHook:
    """Forward hook on the last vision block; captures its output tensor."""

    def __init__(self, visual: torch.nn.Module, block_index: int = -1):
        self.visual = visual
        self.block_index = block_index
        self.block = visual.blocks[block_index]
        self.captured: torch.Tensor | None = None
        self._handle = None

    def _fn(self, _module, _inputs, output):
        self.captured = output[0] if isinstance(output, tuple) else output

    @contextmanager
    def attached(self) -> Iterator["FPreHook"]:
        self.captured = None
        self._handle = self.block.register_forward_hook(self._fn)
        try:
            yield self
        finally:
            self._handle.remove()
            self._handle = None

    def split(self, grid_thw: torch.Tensor) -> list[torch.Tensor]:
        """Split the captured sequence per image and unshuffle to ``(gh,gw,C)``."""
        if self.captured is None:
            raise RuntimeError("hook captured nothing; did the forward run?")
        x = self.captured
        if x.dim() == 3 and x.shape[0] == 1:      # (1, N, C) -> (N, C)
            x = x.squeeze(0)
        if x.shape[-1] != FPRE_DIM:
            raise AssertionError(
                f"F_pre is {x.shape[-1]}-dim, protocol 4.1 says {FPRE_DIM}"
            )
        out, off = [], 0
        for t, h, w in grid_thw.tolist():
            n = int(t) * int(h) * int(w)
            out.append(unshuffle_to_grid(x[off:off + n], int(h), int(w)))
            off += n
        if off != x.shape[0]:
            raise AssertionError(f"grid_thw covers {off} tokens, hook saw {x.shape[0]}")
        return out


class MergerHook:
    """Forward hook on ``visual.merger`` -- the H/32 image embedding the LLM sees.

    PR-AMORT needs this because the *only* spatial conditioning channel with
    demonstrated word specificity (P-W5: ``merger_out . E[subject noun]``,
    Delta=+0.021, p=1.4e-7) is scored in the merged space, and nothing in
    Where-B captured it: :class:`FPreHook` sits on the last vision **block**,
    i.e. one module *before* the merger, and ``EncodeResult`` carried only
    ``f_pre`` and ``h_where``.

    Crucially this is not a second visual forward.  ``Qwen3VLVisionModel.forward``
    ends with ``hidden_states = self.merger(hidden_states)`` and returns it, so
    the tensor is already computed in the same pass that produces ``F_pre``;
    it was merely never read.  Hooking it is therefore free and protocol-2.3
    safe (verified against transformers 4.57.1, 2026-08-10).

    ``dump_amort_cache.py`` obtained the identical tensor as ``visual(...)[0]``;
    hooking the merger reproduces that layout exactly, which is what lets the
    online path and E1's 400-sample cache be compared without a conversion.

    Ordering: the pre-merge sequence is blocked as ``(gh/m, gw/m, m, m)`` and
    ``Qwen3VLVisionPatchMerger`` folds every ``m*m`` **consecutive** tokens with
    ``x.view(-1, hidden*m*m)``.  One merged row is therefore exactly one ``m x m``
    block, and the merged rows run row-major over the ``(gh/m, gw/m)`` grid --
    so :meth:`split` reshapes directly and must **not** call
    :func:`unshuffle_to_grid` (that is the un-blocking the merger already did).
    """

    def __init__(self, visual: torch.nn.Module):
        self.visual = visual
        merger = getattr(visual, "merger", None)
        if merger is None:
            raise AttributeError(
                "vision tower has no .merger; expected Qwen3VLVisionModel with "
                "a Qwen3VLVisionPatchMerger (transformers 4.57.1)"
            )
        self.merger = merger
        self.merge = int(getattr(visual, "spatial_merge_size", SPATIAL_MERGE))
        self.captured: torch.Tensor | None = None
        self._handle = None

    def _fn(self, _module, _inputs, output):
        self.captured = output[0] if isinstance(output, tuple) else output

    @contextmanager
    def attached(self) -> Iterator["MergerHook"]:
        self.captured = None
        self._handle = self.merger.register_forward_hook(self._fn)
        try:
            yield self
        finally:
            self._handle.remove()
            self._handle = None

    def split(self, grid_thw: torch.Tensor) -> list[torch.Tensor]:
        """Per image ``(gh/m, gw/m, out_hidden)`` on the merged (H/32) grid."""
        if self.captured is None:
            raise RuntimeError("hook captured nothing; did the forward run?")
        x = self.captured
        if x.dim() == 3 and x.shape[0] == 1:
            x = x.squeeze(0)
        x = x.reshape(-1, x.shape[-1])
        m = self.merge
        out, off = [], 0
        for t, h, w in grid_thw.tolist():
            h, w, t = int(h), int(w), int(t)
            if h % m or w % m:
                raise ValueError(f"grid {h}x{w} is not divisible by merge {m}")
            gh, gw = h // m, w // m
            n = t * gh * gw
            out.append(x[off:off + n].reshape(gh, gw, x.shape[-1]))
            off += n
        if off != x.shape[0]:
            raise AssertionError(
                f"grid_thw implies {off} merged tokens, hook saw {x.shape[0]}"
            )
        return out


def load_vision_tower(
    model_dir: str | Path,
    dtype: torch.dtype = torch.float32,
    device: str = "cpu",
    attn_implementation: str = "sdpa",
) -> torch.nn.Module:
    """Build ``Qwen3VLVisionModel`` alone and load only ``model.visual.*``.

    The vision tower is ~0.3 GB in bf16, so this makes CPU-side structural and
    numeric validation possible while both GPUs are busy.  It is *the same*
    module and the same weights the full model uses.
    """
    from safetensors import safe_open
    from transformers import AutoConfig
    from transformers.models.qwen3_vl.modeling_qwen3_vl import Qwen3VLVisionModel

    model_dir = Path(model_dir)
    cfg = AutoConfig.from_pretrained(model_dir)
    vcfg = cfg.vision_config
    vcfg._attn_implementation = attn_implementation
    # NOT meta + to_empty: Qwen3VLVisionRotaryEmbedding registers ``inv_freq`` as
    # a *non-persistent* buffer, so it is absent from the checkpoint and
    # ``to_empty`` would leave it as uninitialised memory -- silently wrong
    # rotary positions with no error anywhere.
    with torch.device(device):
        model = Qwen3VLVisionModel(vcfg)

    import json
    index_path = model_dir / "model.safetensors.index.json"
    if index_path.exists():
        weight_map = json.loads(index_path.read_text())["weight_map"]
        files: dict[str, list[str]] = {}
        for k, f in weight_map.items():
            if k.startswith("model.visual."):
                files.setdefault(f, []).append(k)
    else:
        files = {"model.safetensors": None}  # type: ignore[dict-item]

    state: dict[str, torch.Tensor] = {}
    for fname, keys in files.items():
        with safe_open(str(model_dir / fname), framework="pt", device="cpu") as fh:
            it = keys if keys is not None else [k for k in fh.keys() if k.startswith("model.visual.")]
            for k in it:
                state[k[len("model.visual."):]] = fh.get_tensor(k).to(dtype)
    missing, unexpected = model.load_state_dict(state, strict=False)
    if unexpected:
        raise RuntimeError(f"unexpected vision weights: {unexpected[:5]}")
    if missing:
        raise RuntimeError(f"missing vision weights: {missing[:5]}")
    model = model.to(device=device, dtype=dtype).eval()
    for name, buf in model.named_buffers():
        if not torch.isfinite(buf).all():
            raise RuntimeError(f"non-finite buffer after load: {name}")
    return model


@torch.no_grad()
def extract_fpre(
    visual: torch.nn.Module,
    pixel_values: torch.Tensor,
    grid_thw: torch.Tensor,
    block_index: int = -1,
) -> list[torch.Tensor]:
    """Run the vision tower and return one ``(grid_h, grid_w, 1024)`` per image."""
    hook = FPreHook(visual, block_index)
    with hook.attached():
        visual(pixel_values.to(next(visual.parameters()).dtype), grid_thw=grid_thw)
    return hook.split(grid_thw)


def fpre_facts(visual: torch.nn.Module) -> dict[str, Any]:
    cfg = visual.config
    return {
        "depth": int(cfg.depth),
        "hidden_size": int(cfg.hidden_size),
        "patch_size": int(cfg.patch_size),
        "spatial_merge_size": int(cfg.spatial_merge_size),
        "n_blocks": len(visual.blocks),
        "hook_block": len(visual.blocks) - 1,
        "attn_implementation": getattr(cfg, "_attn_implementation", None),
    }
