"""Success/failure panels for a PR-AMORT arm.

Three visualisation red lines are load-bearing here, and each one has already
produced a wrong conclusion in this project at least once:

1. **No per-image min-max colouring.**  Every mask panel is drawn on a fixed
   ``0..1`` scale.  The historical damage: RO-9c's pad cells carried 53-74% of
   the attention mass, min-max put the denominator under their control, and the
   raw field "looked like a single sink" -- which is where the entire (false)
   "common-mode removal unlocks grounding" thesis came from.
2. **The colour scale reads valid cells only**, and pad cells are drawn white
   rather than filled in.  In this representation there are no pad cells (each
   sample keeps its native grid), which is asserted, not assumed.
3. **Overlays use the exact inverse map** ``grid_to_img``, never a resize -- a
   direct resize put about a third of an earlier figure's cells in the wrong
   place.

Panels are five-up: image / GT / prediction / centre prior / prediction under a
swapped subject.  The last two are the ones that make a failure legible: an
over-covering centre blob and a prediction that does not move when the subject
changes are the W01/W02 signature, and side by side they are unmistakable.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import torch

from q3vl.whereb.viz import overlay_grid_on_image, render_field

from .evaluate import center_prior_unit

__all__ = ["write_panels"]

#: every mask panel shares this scale; nothing is normalised per image
FIXED = (0.0, 1.0)


def _panel(ax, arr: np.ndarray, title: str) -> None:
    ax.imshow(arr)
    ax.set_title(title, fontsize=8)
    ax.set_xticks([])
    ax.set_yticks([])


@torch.no_grad()
def write_panels(
    model, builder, dataset, indices: Sequence[int],
    rows: Mapping[str, Sequence[Mapping[str, Any]]],
    out_dir: Path, *, n_each: int = 6, main: str = "generated",
) -> list[Path]:
    """Pick the best/worst by soft-IoU on the main context and draw them."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    ctx = rows.get(main) or rows.get("gt") or []
    live = [r for r in ctx if not r.get("uncovered") and not r.get("is_fake")]
    if not live:
        return []
    # ranked by the primary (matched-area top-k) column, so the panels show
    # the same best/worst the board reports
    live = sorted(live, key=lambda r: r["hard_iou"])
    worst = live[:n_each]
    best = live[-n_each:][::-1]
    by_id = {dataset.record(i)["sample_id"]: i for i in indices}
    shuffled_rows = {r["sample_id"]: r for r in rows.get("shuffled", [])}

    written: list[Path] = []
    for tag, group in (("success", best), ("failure", worst)):
        for r in group:
            sid = r["sample_id"]
            if sid not in by_id:
                continue
            s = dataset[by_id[sid]]
            built = builder.build([s], [main if main in rows else "gt"])
            if not built:
                continue
            x = built[0]
            cond = model.cond_of(x.cond_h, x.cond_mask, x.word_ids, x.word_offsets)
            if x.route_semantic and model.sem is not None:
                out = model.forward_sem(x.feat, cond, sim=x.sim, center=x.center,
                                        geom=getattr(x, 'geom', None))
            else:
                out = model.forward_geo(x.feat, cond, x.phi_dir, sim=x.sim,
                                        center=x.center,
                                        geom=getattr(x, 'geom', None),
                                        grid_h=x.grid_h, grid_w=x.grid_w,
                                        h_where=x.cond_h, h_mask=x.cond_mask)
            m = out["m_low"].float().cpu()
            gt = x.gt_low.float().cpu()
            gh, gw = x.grid_h, x.grid_w
            cp = center_prior_unit(gh, gw).cpu()

            # the swapped-subject panel, when a partner exists
            m_shuf = None
            try:
                b2 = builder.build([s], ["shuffled"])
                x2 = b2[0]
                c2 = model.cond_of(x2.cond_h, x2.cond_mask, x2.word_ids,
                                   x2.word_offsets)
                if x2.route_semantic and model.sem is not None:
                    o2 = model.forward_sem(x2.feat, c2, sim=x2.sim, center=x2.center,
                                           geom=getattr(x2, 'geom', None))
                else:
                    o2 = model.forward_geo(x2.feat, c2, x2.phi_dir, sim=x2.sim,
                                           center=x2.center,
                                           geom=getattr(x2, 'geom', None),
                                           grid_h=x2.grid_h, grid_w=x2.grid_w,
                                           h_where=x2.cond_h, h_mask=x2.cond_mask)
                m_shuf = o2["m_low"].float().cpu()
            except Exception:
                m_shuf = None

            img = s.image_tensor()
            # there are no pad cells in this representation; assert it
            valid = torch.ones(gh, gw, dtype=torch.bool)
            over, _ = overlay_grid_on_image(m, img, valid=valid, mode="fixed",
                                            fixed=FIXED, alpha=0.55)

            fig, ax = plt.subplots(1, 5, figsize=(17, 3.4))
            _panel(ax[0], img.permute(1, 2, 0).numpy(), f"{sid[:22]}\ninput")
            _panel(ax[1], render_field(gt, valid, mode="fixed", fixed=FIXED).rgba,
                   f"GT .cgt (area={float((gt>0.5).float().mean()):.3f})")
            _panel(ax[2], render_field(m, valid, mode="fixed", fixed=FIXED).rgba,
                   f"pred  top-k IoU={r['hard_iou']:.3f}")
            _panel(ax[3], render_field(cp, valid, mode="fixed", fixed=FIXED).rgba,
                   f"centre prior={r['center_prior_hard_iou']:.3f}")
            if m_shuf is not None:
                _panel(ax[4], render_field(m_shuf, valid, mode="fixed",
                                           fixed=FIXED).rgba,
                       "pred | swapped subject")
            else:
                _panel(ax[4], over, "pred over input (exact inverse map)")
            fig.suptitle(
                f"{tag}  {model.arm}  family={r.get('family')}  head={r.get('head')}  "
                f"corr(centre)={r['corr_pred_center']:.2f} corr(GT)={r['corr_pred_gt']:.2f}"
                "   [colour scale fixed 0..1, no per-image min-max]", fontsize=9)
            fig.tight_layout()
            p = out_dir / f"{tag}_{model.arm}_{sid[:24]}.png"
            fig.savefig(p, dpi=110)
            plt.close(fig)
            written.append(p)
    return written
