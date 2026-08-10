"""Long-tail composite panels.

Colour rules (CLAUDE.md 2026-08-05), and how each is met here:

* **no per-image min-max.**  Nothing in this module computes a scale from the
  image it is drawing.  Masks are drawn on a fixed ``(0, 1)`` scale because they
  *are* probabilities, and ``s`` on the fixed ``(-S_SCALE, +S_SCALE)`` scale that
  ``s = 3 tanh(q/3)`` guarantees by construction.  A fixed campaign-wide scale is
  the stronger of the two sanctioned options in :mod:`q3vl.whereb.viz`, and it has
  the property the report needs anyway: two samples' ``s`` panels are directly
  comparable, which a per-image scale would destroy.
* **pad cells.**  Where-B's grid is ``(H/16, W/16)`` of a natively-sized spec-5
  image (``q3vl.where.fpre.grid_from_geometry``), so every cell covers real image
  content -- there is no ``expand2square`` and therefore no pad cell.  That is
  stated to :func:`q3vl.whereb.viz.render_field` explicitly via
  ``allow_all_valid=True`` (its nit-N24 escape hatch), never by passing
  ``valid=None`` and hoping.
* **overlays use the exact inverse map.**  Expansion is integer nearest-neighbour
  through :func:`q3vl.whereb.viz.grid_to_img`; no ``resize`` appears anywhere in
  this file.
* **colouring is not arithmetic.**  Every number printed on a panel is passed in
  by the caller, already computed from the raw fields; this module computes no
  metric and returns none.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

__all__ = ["render_tail_panel", "PanelInputs"]

_ALPHA = 0.55


def _to_hw3(img: np.ndarray) -> np.ndarray:
    a = np.asarray(img, dtype=np.float32)
    if a.ndim == 3 and a.shape[0] == 3:
        a = a.transpose(1, 2, 0)
    return np.clip(a, 0.0, 1.0)


def _overlay(field: np.ndarray, image_hw3: np.ndarray, *,
             vmin: float, vmax: float, alpha: float = _ALPHA) -> np.ndarray:
    """Blend a ``(gh, gw)`` grid field onto an image through the exact map."""
    import torch

    from q3vl.whereb.viz import grid_to_img, render_field

    f = torch.as_tensor(np.asarray(field, dtype=np.float32))
    gh, gw = f.shape[-2:]
    out_h, out_w = image_hw3.shape[:2]
    m = grid_to_img(gh, gw, out_h, out_w)
    r = render_field(f, None, mode="fixed", fixed=(vmin, vmax),
                     allow_all_valid=True)
    big = np.repeat(np.repeat(r.rgba[..., :3], m["scale_y"], axis=0),
                    m["scale_x"], axis=1)
    return (1 - alpha) * image_hw3 + alpha * big


class PanelInputs(dict):
    """What one panel needs.  A plain dict so it survives json round-trips."""


#: first available wins; the report is Chinese, so a panel caption rendered in
#: DejaVu comes out as a row of tofu boxes and the class label is unreadable
_CJK_FONTS = ("Noto Sans CJK JP", "Noto Sans CJK SC", "WenQuanYi Micro Hei",
              "WenQuanYi Zen Hei", "Microsoft YaHei", "Source Han Sans CN")
_font_choice: str | None | bool = False          # False = not yet resolved


def _cjk_font() -> str | None:
    global _font_choice
    if _font_choice is not False:
        return _font_choice                                   # type: ignore[return-value]
    import matplotlib.font_manager as fm

    have = {f.name for f in fm.fontManager.ttflist}
    _font_choice = next((f for f in _CJK_FONTS if f in have), None)
    if _font_choice is None:                                   # pragma: no cover
        import warnings

        warnings.warn("no CJK font found; panel captions will show tofu boxes. "
                      "Install fonts-noto-cjk or pass ASCII captions.")
    return _font_choice                                        # type: ignore[return-value]


def render_tail_panel(
    out_path: str | Path,
    *,
    image: np.ndarray | None,
    mask_gt_hi: np.ndarray | None,
    mask_pred_hi: np.ndarray | None = None,
    mask_pred_low: np.ndarray | None = None,
    s_low: np.ndarray | None = None,
    s_star_low: np.ndarray | None = None,
    title: str = "",
    caption: str = "",
    s_scale: float | None = None,
) -> Path:
    """One composite row: ``I_in | GT | GT overlay | pred | s`` + caption.

    Missing panels are drawn as an explicit "not available" tile rather than
    omitted, so the reader can see *what was not measured* -- the same reason
    :mod:`attribution` distinguishes ``not_tested`` from ``absent``.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    from q3vl.whereb.config import S_SCALE

    font = _cjk_font()
    if font:
        plt.rcParams["font.sans-serif"] = [font, *plt.rcParams["font.sans-serif"]]
        plt.rcParams["axes.unicode_minus"] = False

    smax = float(S_SCALE if s_scale is None else s_scale)
    img = None if image is None else _to_hw3(image)

    tiles: list[tuple[str, Any, dict[str, Any]]] = [("I_in", img, {})]
    tiles.append(("GT mask (.cgt)", None if mask_gt_hi is None
                  else np.asarray(mask_gt_hi, dtype=np.float32),
                  {"cmap": "magma", "vmin": 0.0, "vmax": 1.0}))
    if img is not None and mask_gt_hi is not None:
        g = np.asarray(mask_gt_hi, dtype=np.float32)[..., None]
        tiles.append(("GT overlay", np.clip(img * (0.35 + 0.65 * g), 0, 1), {}))
    else:
        tiles.append(("GT overlay", None, {}))
    tiles.append(("pred mask hi", None if mask_pred_hi is None
                  else np.asarray(mask_pred_hi, dtype=np.float32),
                  {"cmap": "magma", "vmin": 0.0, "vmax": 1.0}))
    tiles.append(("pred mask low (F_pre)", None if mask_pred_low is None
                  else np.asarray(mask_pred_low, dtype=np.float32),
                  {"cmap": "magma", "vmin": 0.0, "vmax": 1.0}))
    if s_low is not None and img is not None:
        tiles.append((f"s over I_in  [-{smax:g}, {smax:g}]",
                      _overlay(s_low, img, vmin=-smax, vmax=smax), {}))
    else:
        tiles.append(("s over I_in", None if s_low is None
                      else np.asarray(s_low, dtype=np.float32),
                      {"cmap": "viridis", "vmin": -smax, "vmax": smax}))
    if s_star_low is not None:
        tiles.append((f"oracle s*  [-{smax:g}, {smax:g}]",
                      np.asarray(s_star_low, dtype=np.float32),
                      {"cmap": "viridis", "vmin": -smax, "vmax": smax}))

    n = len(tiles)
    fig, axes = plt.subplots(1, n, figsize=(3.1 * n, 3.9))
    if n == 1:
        axes = [axes]
    for ax, (name, data, kw) in zip(axes, tiles):
        ax.set_title(name, fontsize=8)
        ax.set_xticks([])
        ax.set_yticks([])
        if data is None:
            ax.text(0.5, 0.5, "not available\n(needs --checkpoint)", ha="center",
                    va="center", fontsize=8, color="0.35")
            ax.set_facecolor("0.94")
            continue
        im = ax.imshow(data, interpolation="nearest", **kw)
        if kw.get("cmap"):
            cb = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.02)
            cb.ax.tick_params(labelsize=6)
    fig.suptitle(title, fontsize=9)
    if caption:
        fig.text(0.01, 0.015, caption, fontsize=7, va="bottom", ha="left",
                 family=("sans-serif" if font else "monospace"))
    fig.tight_layout(rect=(0, 0.13, 1, 0.95))
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=110)
    plt.close(fig)
    return out_path


def panel_caption(row: Mapping[str, Any], labels: Mapping[str, str],
                  attribution: Mapping[str, Any],
                  extra: Sequence[str] = ()) -> str:
    """The text block under a panel: classes, raw numbers, mechanisms."""
    def num(key: str, digits: int = 3) -> str:
        v = row.get(key)
        return "n/a" if v is None else f"{float(v):.{digits}f}"

    cls = "  ".join(f"{k}={v}" for k, v in labels.items())
    ev = attribution.get("evidence", {})
    gap = ev.get("context_gap")
    lines = [
        f"class : {cls}",
        (f"metric: softIoU={num('soft_iou')}  hardIoU={num('grid_hard_iou')}  "
         f"边界F1={num('grid_boundary_f1')}  中心先验 hardIoU="
         f"{num('center_prior_hard_iou')}"),
        (f"        oracle softIoU={num('oracle_soft_iou')}  "
         f"std(s)/std(s*)={num('s_std_ratio')}  hi-lo={num('hi_lo_soft_iou_drop', 4)}  "
         f"GT-gen gap={'n/a' if gap is None else format(float(gap), '.3f')}"),
        f"归因  : primary={attribution.get('primary')}  "
        f"all={','.join(attribution.get('mechanisms', []))}",
    ]
    lines.extend(extra)
    return "\n".join(lines)
