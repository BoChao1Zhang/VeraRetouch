"""Decoupled core layer for the VeraRetouch data pipeline (see
``docs/concurrency/UNIFIED_CONCURRENCY_DESIGN_v2_2026-06-16.md``).

Phase 0 shipped the out-of-process vLLM broker (``core.broker``). Phase 2 adds
the synchronous ``core`` facade the business consumes — ``core.vllm`` /
``core.sam3`` / ``core.iqa`` / ``core.render`` / ``core.lr`` — relocating
ownership of the renderer + render_lock into ``core.render`` while preserving the
existing thread-based overlap orchestration and byte-identical output.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Optional

from .client import IqaClient, LrClient, Sam3Client, VLLMClient
from .render_worker import RenderClient, downscale_rgb

__all__ = [
    "Core", "build_core", "build_lr_client",
    "VLLMClient", "Sam3Client", "IqaClient", "LrClient", "RenderClient",
    "downscale_rgb",
]


def build_lr_client(max_concurrency: Optional[int] = None) -> LrClient:
    """The ``core.lr`` handle for the QA path (preset_qa stage2).

    ``max_concurrency`` is the LR pool admission cap (concurrent farm renders);
    defaults to ``source_qa.config.LR_MAX_CONCURRENCY`` when None. Build does not
    use LR, so this is constructed by the QA stage rather than ``build_core``.
    """
    if max_concurrency is None:
        try:
            from dataset_build.source_qa import config as _sqa_cfg
            max_concurrency = int(getattr(_sqa_cfg, "LR_MAX_CONCURRENCY", 3))
        except Exception:
            max_concurrency = 3
    return LrClient(max_concurrency=max_concurrency)


def _render_kw_from_config(config: Dict[str, Any]) -> Dict[str, int]:
    """Mirror ``streams.BuildContext.render_kw`` exactly (same keys/defaults)."""
    vr = (config.get("models", {}) or {}).get("veraretouch", {}) or {}
    return {
        "batch_size": int(vr.get("batch_size", 8)),
        "chunk": int(vr.get("chunk", 262144)),
        "max_new_tokens": int(vr.get("max_new_tokens", 768)),
    }


@dataclass
class Core:
    """The business's single synchronous interface to every heavy resource."""

    vllm: Optional[VLLMClient] = None
    sam3: Optional[Sam3Client] = None
    iqa: Optional[IqaClient] = None
    render: Optional[RenderClient] = None
    lr: Optional[LrClient] = None


def build_core(
    config: Dict[str, Any],
    *,
    renderer: Optional[Any] = None,
    masker: Optional[Any] = None,
    sam3_live_masker: Optional[Any] = None,
    cleaner: Optional[Any] = None,
    gpu: Optional[Any] = None,
    device: str = "cuda:0",
    iqa_scorer: Optional[Any] = None,
    lr_submit_fn: Optional[Any] = None,
) -> Core:
    """Assemble a :class:`Core` from already-constructed collaborators.

    ``render`` is always present (its ``renderer`` may be ``None`` — it then
    returns all-None like the historical inline path). ``vllm``/``sam3``/``iqa``
    exist only when their collaborator was built. ``gpu`` is a shared
    :class:`~dataset_build.core.gpu_compute.GpuCompute` so render / live-SAM3 /
    IQA serialize on a shared card via the lease->render_lock contract; pass
    ``None`` to leave render serialized by ``render_lock`` alone.
    """
    has_sam3 = masker is not None or sam3_live_masker is not None
    return Core(
        vllm=(VLLMClient(cleaner) if cleaner is not None else None),
        sam3=(Sam3Client(cached_masker=masker, live_masker=sam3_live_masker,
                         gpu=gpu, device=device) if has_sam3 else None),
        iqa=(IqaClient(iqa_scorer, gpu=gpu, device=device) if iqa_scorer is not None else None),
        render=RenderClient(renderer, _render_kw_from_config(config), gpu=gpu, device=device),
        lr=(LrClient(lr_submit_fn) if lr_submit_fn is not None else None),
    )
