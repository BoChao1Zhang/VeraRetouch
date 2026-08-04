"""Image preprocessing contract (spec 5).

    1. EXIF orientation applied;
    2. isotropic rescale so the *short* side is exactly 512;
    3. no square stretch, no centre crop;
    4. both sides aligned to factor 32 (= patch_size 16 * spatial_merge_size 2);
    5. long side capped at 2048;
    6. original aspect ratio > 4:1 -> reject (never squashed to fit);
    7. resulting H/W, image_grid_thw and visual token count recorded.

Note that with the short side pinned at 512 and aspect ratio <= 4, the long
side is <= 2048 by construction; the cap is asserted, not relied upon.

``max_pixels``/``min_pixels`` are deliberately *not* used: they are area
constraints and cannot pin the short side of a wide image (spec 5 note).
The processed image is handed to the HF image processor with
``do_resize=False`` so that the geometry computed here is the geometry used.
"""

from __future__ import annotations

import io
from dataclasses import asdict, dataclass
from typing import Any

from PIL import Image, ImageOps

from .constants import (
    IMAGE_ALIGN_FACTOR,
    IMAGE_ASPECT_TOLERANCE,
    IMAGE_LONG_SIDE_MAX,
    IMAGE_MAX_ASPECT_RATIO,
    IMAGE_SHORT_SIDE,
)

# PIL >= 9.1 moved the resampling enum; both spellings kept working, use the new one.
_RESAMPLE = Image.Resampling.BICUBIC


class ImageRejected(ValueError):
    """Sample must go to the rejection report rather than into training."""

    def __init__(self, reason: str, detail: str = ""):
        super().__init__(f"{reason}: {detail}" if detail else reason)
        self.reason = reason
        self.detail = detail


@dataclass(frozen=True)
class ImageGeometry:
    orig_h: int
    orig_w: int
    out_h: int
    out_w: int
    grid_h: int
    grid_w: int
    n_visual_tokens: int
    aspect_in: float
    aspect_out: float

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _align(x: int, factor: int = IMAGE_ALIGN_FACTOR) -> int:
    return max(factor, int(round(x / factor)) * factor)


def plan_geometry(
    h: int,
    w: int,
    short_side: int = IMAGE_SHORT_SIDE,
    long_side_max: int = IMAGE_LONG_SIDE_MAX,
    factor: int = IMAGE_ALIGN_FACTOR,
    max_aspect: float = IMAGE_MAX_ASPECT_RATIO,
    aspect_tol: float = IMAGE_ASPECT_TOLERANCE,
) -> ImageGeometry:
    """Compute the target size. Raises :class:`ImageRejected` for >4:1 images."""
    if h <= 0 or w <= 0:
        raise ImageRejected("image_corrupt", f"non-positive size {h}x{w}")
    if short_side % factor != 0:
        raise ValueError(f"short_side {short_side} must be a multiple of {factor}")

    aspect_in = max(h, w) / min(h, w)
    if aspect_in > max_aspect:
        raise ImageRejected("aspect_ratio", f"{aspect_in:.3f} > {max_aspect}")

    scale = short_side / min(h, w)
    if h <= w:
        out_h = short_side
        out_w = _align(w * scale, factor)
    else:
        out_w = short_side
        out_h = _align(h * scale, factor)

    long_side = max(out_h, out_w)
    if long_side > long_side_max:
        # Unreachable while max_aspect * short_side <= long_side_max; asserted
        # rather than silently squashed, per spec 5 item 6.
        raise ImageRejected(
            "long_side", f"{out_h}x{out_w} exceeds long-side cap {long_side_max}"
        )

    if min(out_h, out_w) != short_side:
        raise ImageRejected("short_side", f"{out_h}x{out_w} short side != {short_side}")
    if out_h % factor or out_w % factor:
        raise ImageRejected("alignment", f"{out_h}x{out_w} not aligned to {factor}")

    aspect_out = max(out_h, out_w) / min(out_h, out_w)
    if abs(aspect_out - aspect_in) / aspect_in > aspect_tol:
        raise ImageRejected(
            "aspect_distortion",
            f"in {aspect_in:.4f} -> out {aspect_out:.4f} (tol {aspect_tol})",
        )

    grid_h = out_h // (factor // 2)  # patch grid = out / patch_size(16)
    grid_w = out_w // (factor // 2)
    n_visual_tokens = (out_h // factor) * (out_w // factor)
    return ImageGeometry(
        orig_h=int(h),
        orig_w=int(w),
        out_h=int(out_h),
        out_w=int(out_w),
        grid_h=int(grid_h),
        grid_w=int(grid_w),
        n_visual_tokens=int(n_visual_tokens),
        aspect_in=float(aspect_in),
        aspect_out=float(aspect_out),
    )


def open_image(data: bytes | str | Image.Image) -> Image.Image:
    """Decode + EXIF-transpose + convert to RGB."""
    try:
        if isinstance(data, Image.Image):
            img = data
        elif isinstance(data, (bytes, bytearray, memoryview)):
            img = Image.open(io.BytesIO(bytes(data)))
        else:
            img = Image.open(data)
        img = ImageOps.exif_transpose(img)
        if img.mode != "RGB":
            img = img.convert("RGB")
        img.load()
    except ImageRejected:
        raise
    except Exception as exc:  # noqa: BLE001 - any decode failure is a rejection
        raise ImageRejected("image_corrupt", f"{type(exc).__name__}: {exc}") from exc
    return img


def prepare_image(data: bytes | str | Image.Image, **kwargs) -> tuple[Image.Image, ImageGeometry]:
    """Full spec-5 pipeline: decode -> EXIF -> plan -> isotropic bicubic resize."""
    img = open_image(data)
    w, h = img.size
    geom = plan_geometry(h, w, **kwargs)
    if (h, w) != (geom.out_h, geom.out_w):
        img = img.resize((geom.out_w, geom.out_h), _RESAMPLE)
    if img.size != (geom.out_w, geom.out_h):
        raise ImageRejected("resize_failed", f"{img.size} != {(geom.out_w, geom.out_h)}")
    return img, geom


def assert_grid_matches(geom: ImageGeometry, grid_thw) -> None:
    """Cross-check the HF image processor's grid against our own plan."""
    t, gh, gw = (int(x) for x in grid_thw)
    if t != 1:
        raise ImageRejected("grid_temporal", f"expected T=1 for a still image, got {t}")
    if (gh, gw) != (geom.grid_h, geom.grid_w):
        raise ImageRejected(
            "grid_mismatch",
            f"processor grid {(gh, gw)} != planned {(geom.grid_h, geom.grid_w)} "
            f"for {geom.out_h}x{geom.out_w}",
        )
