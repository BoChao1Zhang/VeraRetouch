"""Apply a complete preset inside per-image CGT masks.

This module is deliberately separate from ``local_gpu``'s Lightroom ``Local*``
correction replay.  A local preset has two image branches:

``base``
    The untouched input image.
``edited``
    The same image after the complete global preset and its residual LUT.

The branches are composited directly in sRGB with the CGT alpha.  There is no
smooth-gain or other spatial remapping in this path.

Tensor ABI: images are floating point ``(B, 3, H, W)`` tensors in ``[0, 1]``.
CGT alpha tensors are ``(B, 1, H, W)`` in ``[0, 1]``.
"""
from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from typing import Any

import torch


_MASK_TYPES = {"circulargradient", "gradient"}


def _validate_image_tensor(value: torch.Tensor, name: str) -> None:
    if not isinstance(value, torch.Tensor):
        raise TypeError(f"{name} must be a torch.Tensor")
    if value.ndim != 4:
        raise ValueError(f"{name} must be BCHW, got shape {tuple(value.shape)}")
    if value.shape[1] != 3:
        raise ValueError(f"{name} must have 3 RGB channels, got {value.shape[1]}")
    if value.shape[0] < 1 or value.shape[-2] < 1 or value.shape[-1] < 1:
        raise ValueError(f"{name} must have non-empty B, H, and W dimensions")
    if not value.is_floating_point():
        raise TypeError(f"{name} must be a floating point float01 tensor")


def _alpha_b1hw(alpha: torch.Tensor, h: int, w: int, *, device: torch.device,
                 dtype: torch.dtype) -> torch.Tensor:
    if not isinstance(alpha, torch.Tensor):
        alpha = torch.as_tensor(alpha, device=device, dtype=dtype)
    else:
        alpha = alpha.to(device=device, dtype=dtype)
    if alpha.ndim == 2:
        alpha = alpha.unsqueeze(0).unsqueeze(0)
    elif alpha.ndim == 3:
        alpha = alpha.unsqueeze(1)
    elif alpha.ndim != 4 or alpha.shape[1] != 1:
        raise ValueError(
            "alpha must have shape HW, BHW, or B1HW; "
            f"got {tuple(alpha.shape)}"
        )
    if alpha.shape[-2:] != (h, w):
        raise ValueError(
            f"alpha spatial shape must be {(h, w)}, got {tuple(alpha.shape[-2:])}"
        )
    return alpha


def composite_srgb(base_bchw: torch.Tensor, edited_bchw: torch.Tensor,
                   alpha: torch.Tensor) -> torch.Tensor:
    """Composite ``edited`` over ``base`` directly in sRGB.

    ``alpha`` accepts ``HW``, ``BHW``, or ``B1HW``.  Image and alpha batches
    may match, or either side may have batch size one.  Endpoint pixels are
    selected with ``torch.where`` so the two defining invariants are bit exact:
    ``alpha == 0`` returns the original base value and ``alpha == 1`` returns
    the edited value.
    """
    _validate_image_tensor(base_bchw, "base_bchw")
    _validate_image_tensor(edited_bchw, "edited_bchw")
    if base_bchw.shape != edited_bchw.shape:
        raise ValueError(
            "base_bchw and edited_bchw must have identical shapes, got "
            f"{tuple(base_bchw.shape)} and {tuple(edited_bchw.shape)}"
        )
    if base_bchw.device != edited_bchw.device:
        raise ValueError("base_bchw and edited_bchw must be on the same device")
    if base_bchw.dtype != edited_bchw.dtype:
        raise ValueError("base_bchw and edited_bchw must have the same dtype")

    b, _, h, w = base_bchw.shape
    a = _alpha_b1hw(alpha, h, w, device=base_bchw.device,
                    dtype=base_bchw.dtype)
    if b != a.shape[0] and b != 1 and a.shape[0] != 1:
        raise ValueError(
            "image and alpha batch sizes must match, or one must be 1; "
            f"got image B={b}, alpha B={a.shape[0]}"
        )

    mixed = torch.lerp(base_bchw, edited_bchw, a)
    return torch.where(a == 0, base_bchw,
                       torch.where(a == 1, edited_bchw, mixed))




def _as_spec_list(specs: Sequence[Mapping[str, Any]] | Mapping[str, Any]) -> list:
    if isinstance(specs, Mapping):
        return [specs]
    if isinstance(specs, (str, bytes)) or not isinstance(specs, Sequence):
        raise TypeError("specs must be a CGT mapping or a sequence of CGT mappings")
    result = list(specs)
    if not result:
        raise ValueError("specs must contain at least one CGT spec")
    return result


def _spec_amount(spec: Mapping[str, Any]) -> float:
    try:
        amount = float(str(spec.get("amount", 1.0)).lstrip("+"))
    except (TypeError, ValueError) as exc:
        raise ValueError("CGT spec 'amount' must be a finite number") from exc
    if not math.isfinite(amount):
        raise ValueError("CGT spec 'amount' must be a finite number")
    return amount


def _normalise_cgt_spec(
        spec: Mapping[str, Any],
) -> tuple[str, Any, float]:
    """归一化一个 CGT spec 为 ``(mask_type, payload, amount)``。

    payload：几何 spec 为 geom mapping；语义 spec 为 (H,W) float alpha 张量。
    """
    if not isinstance(spec, Mapping):
        raise TypeError(f"each CGT spec must be a mapping, got {type(spec).__name__}")

    # 语义 spec：显式 mask_type=semantic，携带 (H,W) float alpha 而非几何。
    if str(spec.get("mask_type") or "").strip().lower() == "semantic":
        alpha = spec.get("alpha")
        if alpha is None:
            raise ValueError("semantic CGT spec requires an 'alpha' array")
        if not isinstance(alpha, torch.Tensor):
            alpha = torch.as_tensor(alpha)
        if alpha.ndim != 2 or not alpha.is_floating_point():
            raise ValueError(
                "semantic CGT 'alpha' must be a 2-D floating point array, got "
                f"shape {tuple(alpha.shape)} dtype {alpha.dtype}")
        return "semantic", alpha, _spec_amount(spec)

    geom = spec.get("geom")
    if not isinstance(geom, Mapping) or not geom:
        raise ValueError("each CGT spec must contain a non-empty 'geom' mapping")

    mask_type = str(spec.get("mask_type") or "").strip().lower()
    if mask_type not in _MASK_TYPES:
        raise ValueError(
            "CGT mask_type must be 'circulargradient', 'gradient', or 'semantic', "
            f"got {mask_type!r}"
        )
    return mask_type, geom, _spec_amount(spec)


def raster_cgt_batch(specs: Sequence[Mapping[str, Any]] | Mapping[str, Any],
                     h: int, w: int, device: torch.device | str,
                     dtype: torch.dtype = torch.float32) -> torch.Tensor:
    """Raster per-image CGT specs to ``(N, 1, H, W)`` alpha tensors.

    The accepted canonical spec is the output of ``subject_geom``:
    ``{"mask_type": ..., "geom": {...}, "amount": 1.0}``.  ``amount`` is
    multiplied into alpha and clamped to ``[0, 1]``.  Geometry math is
    delegated to :func:`gpu_render.gpu.local_gpu.raster_alpha_t` with
    exp-radial smoothstep enabled explicitly, leaving genuine Lightroom Local*
    replay linear.

    语义 spec ``{"mask_type": "semantic", "alpha": (H,W) float, "amount": ...}``：
    alpha 可为降采样分辨率，双线性放大到 ``(h, w)``（对齐
    ``local_gpu.corr_alpha_t`` 的语义），×amount 后 clamp。
    """
    if h < 1 or w < 1:
        raise ValueError(f"h and w must be positive, got h={h}, w={w}")
    if not torch.empty((), dtype=dtype).is_floating_point():
        raise TypeError("dtype must be a floating point torch dtype")

    spec_list = _as_spec_list(specs)
    from gpu_render.gpu.local_gpu import raster_alpha_t

    alphas = []
    for spec in spec_list:
        mask_type, payload, amount = _normalise_cgt_spec(spec)
        if mask_type == "semantic":
            alpha = payload.to(device=device, dtype=torch.float32)
            if alpha.shape != (h, w):
                alpha = torch.nn.functional.interpolate(
                    alpha[None, None], size=(h, w), mode="bilinear",
                    align_corners=False)[0, 0]
        else:
            alpha = raster_alpha_t(
                mask_type, dict(payload), h, w, device, smoothstep=True)
        alphas.append((alpha * amount).clamp(0.0, 1.0))
    return torch.stack(alphas, dim=0).unsqueeze(1).to(dtype=dtype)




def _preset_without_locals(preset: Mapping[str, Any]) -> tuple[dict, dict]:
    if not isinstance(preset, Mapping):
        raise TypeError("preset must be a parsed preset mapping")
    if not isinstance(preset.get("attrs"), Mapping):
        raise ValueError("preset must contain an 'attrs' mapping")
    if not isinstance(preset.get("curves"), Mapping):
        raise ValueError("preset must contain a 'curves' mapping")

    clean = dict(preset)
    embedded = clean.pop("locals", None)
    if embedded is not None and not isinstance(embedded, (list, tuple)):
        raise ValueError("preset['locals'] must be a list/tuple when present")

    attrs = dict(clean["attrs"])
    local_attr_keys = tuple(sorted(key for key in attrs if str(key).startswith("Local")))
    for key in local_attr_keys:
        attrs.pop(key, None)
    clean["attrs"] = attrs

    return clean, {
        "preset_locals_stripped": bool(embedded) or bool(local_attr_keys),
        "stripped_local_corrections": len(embedded or ()),
        "stripped_local_attr_keys": local_attr_keys,
    }


def render_local_preset_tensor(
        base_bchw: torch.Tensor,
        preset: Mapping[str, Any],
        specs: Sequence[Mapping[str, Any]] | Mapping[str, Any],
        fits_dir: str,
        residual_id: str | None = None,
        fallback: str = "cpu",
) -> tuple[torch.Tensor, dict]:
    """Render a complete preset only inside each CGT spec.

    For an input batch ``B > 1``, exactly one spec is required per image.  For
    ``B == 1``, ``N`` specs produce an ``(N, 3, H, W)`` result while preset
    replay and residual correction run only once; the single base/edited pair
    is broadcast during composition.

    Structured preset locals are stripped from a shallow preset copy, as are
    any top-level ``Local*`` attributes captured by broad XMP parsing.  The
    caller's preset is never mutated.  A malformed non-sequence ``locals``
    payload is rejected instead of being silently replayed.

    Returns ``(out, info)``.  ``info['alpha']`` is the detached
    ``(N, 1, H, W)`` CGT tensor used for composition; replay diagnostics remain
    at the top level alongside local-stripping and residual metadata.
    """
    _validate_image_tensor(base_bchw, "base_bchw")
    if fallback not in {"cpu", "skip"}:
        raise ValueError("fallback must be 'cpu' or 'skip'")

    spec_list = _as_spec_list(specs)
    b, _, h, w = base_bchw.shape
    if b != 1 and len(spec_list) != b:
        raise ValueError(
            f"B={b} requires exactly one CGT spec per image, got {len(spec_list)}"
        )

    clean_preset, local_info = _preset_without_locals(preset)
    alpha = raster_cgt_batch(spec_list, h, w, base_bchw.device,
                             dtype=base_bchw.dtype).detach()

    # Rendering is an inference pipeline.  Detaching here prevents a caller's
    # training graph from being retained by the diagnostic alpha/output.
    with torch.no_grad():
        from gpu_render.gpu.gpu_replay import replay_batch

        base = base_bchw.detach()
        edited, replay_info = replay_batch(
            base.clone(), clean_preset, fits_dir=fits_dir, fallback=fallback
        )

        residual_applied = False
        if residual_id:
            from gpu_render.residual import load_residual

            residual = load_residual(residual_id)
            if residual is not None:
                from gpu_render.gpu.residual_gpu import apply_residual_batch

                edited = apply_residual_batch(edited, *residual)
                residual_applied = True

        out = composite_srgb(base, edited, alpha)

    info = dict(replay_info)
    info.update(local_info)
    info.update({
        "alpha": alpha,
        "input_batch": b,
        "output_batch": len(spec_list) if b == 1 else b,
        "expanded_single_input": b == 1 and len(spec_list) > 1,
        "residual_id": residual_id,
        "residual_applied": residual_applied,
    })
    return out, info


__all__ = [
    "composite_srgb",
    "raster_cgt_batch",
    "render_local_preset_tensor",
]
