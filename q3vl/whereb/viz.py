"""Spatial-field visualisation, under the 2026-08-05 red lines.

Four rules, each of which has already cost this campaign a wrong conclusion:

1. **No per-image min-max colouring.**  Measured on RO-9c: ``expand2square``'s
   pad cells are ~5.3 of a 16x16 grid, carry **53-74%** of the attention mass and
   **93%** of the source argmax.  A per-image min-max normalisation puts those
   cells in the denominator, so the field renders as "one sink and nothing else"
   while the real signal inside the valid region is perfectly visible.  The
   entire "common-mode removal is required to unlock grounding" conclusion came
   from that artifact.
2. **The colour scale is computed over valid cells only**, and pad cells are
   drawn explicitly (white, or hatched) -- never silently filled.
3. **Overlaying a grid field on the image uses the strict inverse mapping**, not
   a resize.  6-09's figures were off by about a third of a cell systematically.
4. **Colouring is colouring, arithmetic is arithmetic.**  Every number that
   enters a criterion comes from the raw, un-normalised field; the report states
   the two separately.

This module owns the rules; :mod:`q3vl.whereb.metrics` owns the numbers.  Nothing
here is allowed to feed a criterion, which is why nothing here returns one.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import torch

__all__ = ["FieldRender", "color_scale", "render_field", "grid_to_img",
           "overlay_grid_on_image", "PerImageMinMaxError"]


class PerImageMinMaxError(RuntimeError):
    """Raised when someone asks for the banned normalisation."""


@dataclass
class FieldRender:
    """A field prepared for display.  Deliberately carries no criterion."""

    rgba: np.ndarray               # (H, W, 4) float in [0, 1]
    vmin: float                    # scale endpoints, over VALID cells only
    vmax: float
    n_valid: int
    n_pad: int
    scale_source: str              # "valid_cells" | "fixed"
    raw_stats: dict[str, float]    # from the UN-normalised field

    def to_dict(self) -> dict[str, Any]:
        return {"vmin": self.vmin, "vmax": self.vmax, "n_valid": self.n_valid,
                "n_pad": self.n_pad, "scale_source": self.scale_source,
                "raw_stats": dict(self.raw_stats)}


def color_scale(field: torch.Tensor, valid: torch.Tensor | None = None,
                *, mode: str = "valid_cells",
                fixed: tuple[float, float] | None = None,
                allow_all_valid: bool = False) -> tuple[float, float]:
    """Colour-scale endpoints.

    ``mode="valid_cells"`` uses only the cells ``valid`` marks True -- the whole
    point of rule 1.  ``mode="per_image_minmax"`` is refused outright rather than
    made available with a warning; a banned default that is one keyword away is
    not banned.
    """
    if mode == "per_image_minmax":
        raise PerImageMinMaxError(
            "per-image min-max colouring is a red line (CLAUDE.md 2026-08-05): "
            "pad cells carried 53-74% of the attention mass on RO-9c and "
            "dominated the denominator, hiding the real signal. Use "
            "mode='valid_cells' (or a fixed campaign-wide scale)."
        )
    if mode == "fixed":
        if fixed is None:
            raise ValueError("mode='fixed' needs an explicit (vmin, vmax)")
        return float(fixed[0]), float(fixed[1])
    if mode != "valid_cells":
        raise ValueError(f"unknown colour-scale mode {mode!r}")
    f = field.reshape(-1).double()
    if valid is None:
        # Review nit N24: without a validity mask "valid_cells" silently becomes
        # the whole-field min-max this module exists to ban -- the `mode`
        # keyword was guarded and `valid` was not, so the ban was one argument
        # away from being undone.  A caller that genuinely has no pad cells says
        # so explicitly.  This module is shared with §13 and Stage-What, where
        # pad cells DO exist, so the default cannot be the permissive one.
        if not allow_all_valid:
            raise PerImageMinMaxError(
                "color_scale needs a `valid` mask: with valid=None the "
                "'valid_cells' scale degenerates into the banned whole-field "
                "min-max (CLAUDE.md 2026-08-05). Pass the mask, or pass "
                "allow_all_valid=True to state that this field has no pad cells."
            )
        v = f
    else:
        m = valid.reshape(-1).bool()
        if not bool(m.any()):
            raise ValueError("no valid cells to build a colour scale from")
        v = f[m]
    return float(v.min()), float(v.max())


def _viridis(x: np.ndarray) -> np.ndarray:
    """Small perceptually-ordered ramp; avoids a matplotlib dependency."""
    stops = np.array([
        [0.267, 0.005, 0.329], [0.283, 0.141, 0.458], [0.254, 0.265, 0.530],
        [0.207, 0.372, 0.553], [0.164, 0.471, 0.558], [0.128, 0.567, 0.551],
        [0.135, 0.659, 0.518], [0.267, 0.749, 0.441], [0.478, 0.821, 0.318],
        [0.741, 0.873, 0.150], [0.993, 0.906, 0.144],
    ])
    idx = np.clip(x, 0.0, 1.0) * (len(stops) - 1)
    lo = np.floor(idx).astype(int)
    hi = np.clip(lo + 1, 0, len(stops) - 1)
    t = (idx - lo)[..., None]
    return stops[lo] * (1 - t) + stops[hi] * t


def render_field(
    field: torch.Tensor,
    valid: torch.Tensor | None = None,
    *,
    mode: str = "valid_cells",
    fixed: tuple[float, float] | None = None,
    pad_style: str = "white",
    allow_all_valid: bool = False,
) -> FieldRender:
    """Colour a ``(H, W)`` field.  Pad cells are drawn, never filled in.

    ``raw_stats`` come from the **un-normalised** field, so a report can quote a
    number without ever quoting the colour mapping (rule 4).
    """
    if field.dim() != 2:
        raise ValueError(f"expected a 2D field, got {tuple(field.shape)}")
    if pad_style not in ("white", "hatch"):
        raise ValueError(f"unknown pad_style {pad_style!r}")
    f = field.detach().double().cpu()
    if valid is None and not allow_all_valid:
        raise PerImageMinMaxError(
            "render_field needs a `valid` mask (review nit N24); pass "
            "allow_all_valid=True only if this field genuinely has no pad cells."
        )
    v = (torch.ones_like(f, dtype=torch.bool) if valid is None
         else valid.detach().reshape(f.shape).bool().cpu())
    vmin, vmax = color_scale(f, v, mode=mode, fixed=fixed, allow_all_valid=True)

    span = max(vmax - vmin, 1e-12)
    norm = ((f.numpy() - vmin) / span).clip(0.0, 1.0)
    rgba = np.ones((*f.shape, 4), dtype=float)
    rgba[..., :3] = _viridis(norm)
    mask = v.numpy()
    if pad_style == "white":
        rgba[~mask, :3] = 1.0
    else:                                    # hatched: visible, not invented
        yy, xx = np.mgrid[0:f.shape[0], 0:f.shape[1]]
        stripe = ((yy + xx) % 2 == 0)
        rgba[~mask, :3] = 1.0
        rgba[~mask & stripe, :3] = 0.0
    rgba[~mask, 3] = 1.0

    raw = f[v] if bool(v.any()) else f.reshape(-1)
    return FieldRender(
        rgba=rgba, vmin=vmin, vmax=vmax,
        n_valid=int(v.sum()), n_pad=int((~v).sum()), scale_source=mode,
        raw_stats={
            "min": float(raw.min()), "max": float(raw.max()),
            "mean": float(raw.mean()), "std": float(raw.std(unbiased=False)),
        },
    )


def grid_to_img(grid_h: int, grid_w: int, out_h: int, out_w: int) -> dict[str, Any]:
    """The strict inverse of the image -> grid mapping (rule 3).

    Cell ``(i, j)`` covers pixels ``[i*out_h/grid_h, (i+1)*out_h/grid_h)`` and
    likewise in x.  Returned as explicit integer edges so an overlay lands on
    cell boundaries instead of being interpolated onto them; a plain ``resize``
    is off by up to half a cell and 6-09's figures showed it.
    """
    if out_h % grid_h or out_w % grid_w:
        raise ValueError(
            f"{out_h}x{out_w} is not an integer multiple of the {grid_h}x{grid_w} "
            "grid; the inverse mapping would not be exact"
        )
    sy, sx = out_h // grid_h, out_w // grid_w
    return {"scale_y": sy, "scale_x": sx,
            "y_edges": [i * sy for i in range(grid_h + 1)],
            "x_edges": [j * sx for j in range(grid_w + 1)]}


def overlay_grid_on_image(
    field: torch.Tensor, image: torch.Tensor, *, alpha: float = 0.5,
    valid: torch.Tensor | None = None, mode: str = "valid_cells",
    allow_all_valid: bool = False,
) -> tuple[np.ndarray, FieldRender]:
    """Blend a grid field onto a ``(3, H, W)`` image via the exact inverse map."""
    if image.dim() != 3 or image.shape[0] != 3:
        raise ValueError(f"expected (3, H, W) image, got {tuple(image.shape)}")
    gh, gw = field.shape[-2:]
    out_h, out_w = image.shape[-2:]
    m = grid_to_img(gh, gw, out_h, out_w)
    r = render_field(field, valid, mode=mode, allow_all_valid=allow_all_valid)
    # nearest-neighbour expansion by an integer factor == the exact inverse map
    big = np.repeat(np.repeat(r.rgba[..., :3], m["scale_y"], axis=0),
                    m["scale_x"], axis=1)
    img = image.detach().float().cpu().numpy().transpose(1, 2, 0)
    return (1 - alpha) * img + alpha * big, r
