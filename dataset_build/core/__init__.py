"""Decoupled core layer for the VeraRetouch data pipeline (see
``docs/concurrency/UNIFIED_CONCURRENCY_DESIGN_v2_2026-06-16.md``).

Phase 0 shipped the out-of-process vLLM broker (``core.broker``). Phase 2 adds
the synchronous ``core`` facade the business consumes — ``core.vllm`` /
``core.sam3`` / ``core.iqa`` / ``core.render`` / ``core.lr``.

渲染后端替换（2026-07）：teacher renderer（llava_qwen2 教师模型）已废弃。
``core.render_backend`` 是统一渲染入口（LR 农场 + 本地 gpu_render batch=16
@cuda:1 双路分流）；``core.render``（RenderClient）保留旧签名、内部代理到
render_backend，teacher 模型加载路径不再触发。
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Optional

from . import render_backend
from .client import IqaClient, LrClient, Sam3Client, VLLMClient
from .render_backend import RenderBackend, get_backend
from .render_worker import RenderClient, downscale_rgb

__all__ = [
    "Core", "build_core", "build_lr_client",
    "VLLMClient", "Sam3Client", "IqaClient", "LrClient", "RenderClient",
    "RenderBackend", "render_backend", "get_backend",
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

    ``render`` is always present, and is now a *proxy* to
    ``core.render_backend``（LR 农场 + 本地 gpu_render 双路）——``renderer``
    参数仅为签名兼容而保留，传入非 None 会被忽略并告警（teacher 已废弃，
    调用方不应再加载教师模型）。``vllm``/``sam3``/``iqa`` exist only when
    their collaborator was built. ``gpu``/``device`` 仍传给 SAM3/IQA 的
    lease 契约；render 不再参与 lease（本地渲固定 cuda:1，后端自持互斥）。
    """
    has_sam3 = masker is not None or sam3_live_masker is not None
    return Core(
        vllm=(VLLMClient(cleaner) if cleaner is not None else None),
        sam3=(Sam3Client(cached_masker=masker, live_masker=sam3_live_masker,
                         gpu=gpu, device=device) if has_sam3 else None),
        iqa=(IqaClient(iqa_scorer, gpu=gpu, device=device) if iqa_scorer is not None else None),
        # renderer 故意不再透传：RenderClient 内部走 render_backend。
        render=RenderClient(renderer, _render_kw_from_config(config), gpu=None, device=device),
        lr=(LrClient(lr_submit_fn) if lr_submit_fn is not None else None),
    )
