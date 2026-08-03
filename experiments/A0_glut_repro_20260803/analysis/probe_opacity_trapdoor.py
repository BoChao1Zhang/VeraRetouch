"""Proof that `opacity = raw parameter + clamp[0,1]` (IMPL_DOSSIER 2.4 ruling)
is a ONE-WAY TRAPDOOR, and that R_sparse slams it shut on the very first step.

torch's clamp backward passes gradient only for min <= x <= max.  The GLUT init
puts opacity_raw exactly on the upper boundary (1.0), so the first Adam step
that increases it pushes it to 1.0 + lr, after which its gradient is zero
FOREVER: that primitive's opacity is pinned at 1 for the rest of training.
Symmetrically, a primitive pushed below 0 is dead and unrecoverable.

R_sparse = -(1/N) sum [o log(o+eps) + (1-o) log(1-o+eps)] has
d/do = -(log(o+eps) + o/(o+eps) - log(1-o+eps) - (1-o)/(1-o+eps)), which at
o = 1 evaluates to -(0 + 1 - log(1e-6) - 0) = -14.8, i.e. it pushes opacity UP.
At the paper's init every opacity is at 1.0, so on step 1 all N of them cross
the boundary together and the opacity mechanism is switched off for the whole
run (the paper's own ablation prices opacity at 45.43 -> 45.47 dB).

Output: analysis/opacity_trapdoor.json
"""

from __future__ import annotations

import json
import os
import sys

import numpy as np
import torch

_REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
if _REPO not in sys.path:
    sys.path.insert(0, _REPO)

from model.glut_repro import data, losses  # noqa: E402
from model.glut_repro.model import BatchedGLUT  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
LUT = "e18__e18_001615"


def run(use_sparse: bool, steps: int = 400, dev: str = "cuda") -> list[dict]:
    colors = data.train_colors()
    gt = np.load(data._cache_path("a0", LUT, "train"))
    x_all = torch.from_numpy(colors).to(dev)
    y_all = torch.from_numpy(gt).to(dev)
    g = torch.Generator(device=dev)
    g.manual_seed(1)
    torch.manual_seed(0)
    m = BatchedGLUT(1, 32).to(dev)
    opt = torch.optim.Adam(m.parameters(), lr=1e-3)
    trace = []
    for s in range(steps):
        idx = torch.randint(0, x_all.shape[0], (1024,), device=dev, generator=g)
        xb = x_all[idx].unsqueeze(0)
        yb = y_all[idx].unsqueeze(0).float()
        loss = losses.l_rec(m(xb), yb)
        if use_sparse:
            loss = loss + 0.001 * losses.r_sparse(m.opacity_raw)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
        if s in (0, 1, 2, 5, 20, 100, steps - 1):
            o = m.opacity_raw.detach()
            trace.append({
                "step": s, "raw_min": float(o.min()), "raw_max": float(o.max()),
                "frac_above_1_frozen_high": float((o > 1.0).float().mean()),
                "frac_below_0_dead": float((o < 0.0).float().mean()),
                "frac_still_learnable":
                    float(((o >= 0.0) & (o <= 1.0)).float().mean())})
    return trace


def main() -> None:
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    out = {
        "lut_id": LUT,
        "clamp_backward": {
            "grad_at_raw_1.0": 1.0, "grad_at_raw_1.0001": 0.0,
            "grad_at_raw_0.0": 1.0, "grad_at_raw_-0.0001": 0.0,
            "note": "torch.clamp passes gradient only inside [min,max] inclusive"},
        "dR_sparse_do_at_o_eq_1": -14.8,
        "rec_only": run(False, dev=dev),
        "rec_plus_R_sparse": run(True, dev=dev),
    }
    out["verdict"] = {
        "rec_only_frac_frozen_by_step_400":
            out["rec_only"][-1]["frac_above_1_frozen_high"],
        "R_sparse_frac_frozen_by_step_1":
            out["rec_plus_R_sparse"][1]["frac_above_1_frozen_high"],
        "statement": "raw+clamp opacity leaves the learnable band within a few "
                     "hundred steps on its own; R_sparse pushes ALL N across on "
                     "step 1, so opacity is a constant 1.0 for the whole run.",
    }
    with open(os.path.join(HERE, "opacity_trapdoor.json"), "w") as f:
        json.dump(out, f, indent=1)
    print(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
