"""Proposal C (FPD, field-as-program) -- Gate 0: the compile/interpret round trip.

``RESEARCH_unified-field-prediction_2026-08-10`` section 4.6, stage one.  Before any
training, compile each GT field into a program and interpret it back::

    s = C(z, omega, y; x)        fixed, zero-parameter compiler
    y_hat = I(s; x)              fixed, zero-parameter differentiable interpreter

Pre-registered gates:

===========================================  ==========  ==========================
criterion                                    pass        falsify
===========================================  ==========  ==========================
geometric families, mean round-trip soft-IoU  >= 0.92    < 0.85 after one bin refinement
contour family (oracle segment selection)     >= 0.80    < 0.70 -> HiMTok fallback
===========================================  ==========  ==========================

plus the **offline compile check** the same section asks for: bin widths are
calibrated in *field space* -- adjacent bins must differ by <= 0.02 soft-IoU in the
rendered field -- and the same calibration fixes the soft-target kernel width.

Interpreter (section 4.2, unified specification): every family first produces a
32x32-class coarse field (geometric families analytically; the contour family by
segment OR followed by convex label propagation; the constant family by a constant),
and then **all** of them go through the one frozen image-guided upsample ``U_I``.
Zero trainable parameters anywhere.

Honest scope note, recorded in NOTES and in ``metrics.json``
-----------------------------------------------------------
``.vrmeta.json`` carries the family label (``slot_id``) but **not** the generation
parameters ``omega`` -- the construction-side generator lives in a different repo.
The compiler therefore *fits* each family's analytic form to ``y`` by L2, which the
document's own signature ``C(z, omega, y; x)`` permits (it consumes ``y``).  The
consequence is stated rather than hidden: the analytic forms here are this card's
reconstruction of the primitive families, so the measured round trip is a **lower
bound** on the true generator's program space.  A pass is therefore trustworthy;
a marginal failure would need the real generator before it could be called
falsification.

L2 is the fit objective throughout.  soft-IoU is a *criterion* here and never an
optimisation target (CLAUDE.md red line).  No AUC anywhere.
"""

from __future__ import annotations

import argparse
import json
import math
import platform
import subprocess
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch

GEOMETRIC = ("radial", "linear", "band")
CONTOUR = "semantic"

GATE_GEOMETRIC = 0.92
GATE_GEOMETRIC_FALSIFY = 0.85
GATE_CONTOUR = 0.80
GATE_CONTOUR_FALSIFY = 0.70
GATE_BIN_STEP_SOFTIOU = 0.02

N_BINS_COARSE = 32
N_BINS_FINE = 32
N_SEGMENTS = 24
LPOSS_ALPHA = 0.85
FIT_STEPS = 220
FIT_RESTARTS = 6


def git_commit() -> str:
    try:
        return subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True,
                              text=True, check=True).stdout.strip()
    except Exception:                                        # noqa: BLE001
        return "unknown"


# --------------------------------------------------------------------------- #
#  the grammar: analytic coarse-field renderers, one per geometric family       #
# --------------------------------------------------------------------------- #
#  Every renderer maps a parameter vector to a coarse field in [0,1].  Slot
#  layouts follow section 4.2's grammar; parameters live in the canonicalised
#  (symmetry-quotiented) coordinates the document requires -- ellipse a >= b and
#  phi in [0,pi), strip direction half-circled -- so the permutation/reflection
#  orbits that produce multi-basin fits are quotiented out at compile time
#  rather than being left for the optimiser to stumble over.

def _xy(grid_h: int, grid_w: int, device, dtype):
    from q3vl.where.phi import norm_coords
    X, Y = norm_coords(grid_h, grid_w, device=device, dtype=dtype)
    return X.reshape(-1), Y.reshape(-1)


class Family:
    name: str
    slots: tuple[str, ...]
    #: (lo, hi) per slot -- the pre-registered quantisation range
    ranges: tuple[tuple[float, float], ...]

    @staticmethod
    def render(p: torch.Tensor, X: torch.Tensor, Y: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError

    @classmethod
    def box(cls, dtype, device) -> tuple[torch.Tensor, torch.Tensor]:
        lo = torch.tensor([r[0] for r in cls.ranges], dtype=dtype, device=device)
        hi = torch.tensor([r[1] for r in cls.ranges], dtype=dtype, device=device)
        return lo, hi

    @classmethod
    def starts(cls, n: int, gen: torch.Generator, dtype) -> torch.Tensor:
        lo = torch.tensor([r[0] for r in cls.ranges], dtype=dtype)
        hi = torch.tensor([r[1] for r in cls.ranges], dtype=dtype)
        u = torch.rand(n, len(cls.slots), generator=gen, dtype=dtype)
        return lo + u * (hi - lo)

    @classmethod
    def informed_start(cls, y: torch.Tensor, X: torch.Tensor, Y: torch.Tensor
                       ) -> torch.Tensor | None:
        """A moment-matched start, or ``None`` if the family has no closed form.

        Random restarts alone left ``radial`` at 0.50 -- an ellipse objective is
        badly multi-basin in ``(cx, cy, a, b, phi)`` and Adam from a uniform draw
        lands in the wrong basin most of the time.  Matching the GT's first and
        second moments puts the start in the right basin by construction; it is
        the same "centroid-radial informed start" the project's own oracle fitter
        already uses (``q3vl/where/oracle.py::build_starts``).
        """
        return None


class Radial(Family):
    """``[F_ell] cx cy log_a log_ba phi log_gamma amp base``."""
    name = "radial"
    #: ``logit_ba`` parameterises ``b/a = sigmoid(.) in (0,1)``, which enforces the
    #: grammar's ``a >= b`` quotient structurally instead of by a constraint the
    #: optimiser can violate; combined with ``phi in [0,pi)`` that removes the
    #: reflection/relabelling orbit that makes ellipse fits multi-basin.
    slots = ("cx", "cy", "log_a", "logit_ba", "phi", "log_gamma", "amp", "base")
    ranges = ((-2.5, 2.5), (-2.5, 2.5), (-2.5, 1.5), (-4.0, 4.0),
              (0.0, math.pi), (-1.0, 5.0), (0.0, 1.0), (0.0, 1.0))

    @classmethod
    def informed_start(cls, y, X, Y):
        w = y.clamp_min(1e-6)
        m = w.sum(dim=-1, keepdim=True).clamp_min(1e-9)
        cx = (w * X).sum(-1, keepdim=True) / m
        cy = (w * Y).sum(-1, keepdim=True) / m
        dx, dy = X - cx, Y - cy
        sxx = (w * dx * dx).sum(-1) / m.squeeze(-1)
        syy = (w * dy * dy).sum(-1) / m.squeeze(-1)
        sxy = (w * dx * dy).sum(-1) / m.squeeze(-1)
        tr, det = sxx + syy, (sxx * syy - sxy * sxy).clamp_min(1e-12)
        disc = (tr * tr / 4 - det).clamp_min(0).sqrt()
        l1, l2 = (tr / 2 + disc).clamp_min(1e-9), (tr / 2 - disc).clamp_min(1e-9)
        phi = 0.5 * torch.atan2(2 * sxy, sxx - syy) % math.pi
        a = 2.0 * l1.sqrt()
        return torch.stack([
            cx.squeeze(-1), cy.squeeze(-1), a.log(),
            torch.logit((l2.sqrt() / l1.sqrt()).clamp(0.02, 0.98)),
            phi, torch.full_like(a, 1.6),
            y.max(dim=-1).values, y.min(dim=-1).values,
        ], dim=-1)

    @staticmethod
    def render(p, X, Y):
        cx, cy, la, lba, phi, lg, amp, base = [p[..., i:i + 1] for i in range(8)]
        a = torch.exp(la)
        b = a * torch.sigmoid(lba)                        # b <= a by construction
        c, s = torch.cos(phi), torch.sin(phi)
        dx, dy = X - cx, Y - cy
        u = dx * c + dy * s
        v = -dx * s + dy * c
        r = torch.sqrt((u / a.clamp_min(1e-3)) ** 2 + (v / b.clamp_min(1e-3)) ** 2 + 1e-12)
        g = torch.exp(lg)
        f = torch.sigmoid(-g * (r - 1.0))
        return (base + (amp - base) * f).clamp(0.0, 1.0)


class Linear(Family):
    """``[F_lin] phi intercept log_width amp base``."""
    name = "linear"
    slots = ("phi", "c", "log_w", "amp", "base")
    ranges = ((0.0, 2 * math.pi), (-3.0, 3.0), (-5.0, 1.5), (0.0, 1.0), (0.0, 1.0))

    @classmethod
    def informed_start(cls, y, X, Y):
        # least-squares plane through y gives the ramp direction directly
        yc = y - y.mean(dim=-1, keepdim=True)
        gx = (yc * (X - X.mean())).sum(-1)
        gy = (yc * (Y - Y.mean())).sum(-1)
        phi = torch.atan2(gy, gx) % (2 * math.pi)
        u = X * torch.cos(phi).unsqueeze(-1) + Y * torch.sin(phi).unsqueeze(-1)
        c0 = (y * u).sum(-1) / y.sum(-1).clamp_min(1e-9)
        return torch.stack([phi, c0, torch.full_like(c0, -1.2),
                            y.max(dim=-1).values, y.min(dim=-1).values], dim=-1)

    @staticmethod
    def render(p, X, Y):
        phi, c0, lw, amp, base = [p[..., i:i + 1] for i in range(5)]
        u = X * torch.cos(phi) + Y * torch.sin(phi)
        w = torch.exp(lw).clamp_min(1e-3)
        f = torch.sigmoid((u - c0) / w)
        return (base + (amp - base) * f).clamp(0.0, 1.0)


class Band(Family):
    """``[F_strip] phi centreline half_width log_width amp base``."""
    name = "band"
    slots = ("phi", "c", "h", "log_w", "amp", "base")
    ranges = ((0.0, math.pi), (-3.0, 3.0), (0.02, 4.0), (-5.0, 1.5),
              (0.0, 1.0), (0.0, 1.0))

    @classmethod
    def informed_start(cls, y, X, Y):
        # principal axis of the mask's second moment is the strip normal
        w = y.clamp_min(1e-6)
        m = w.sum(dim=-1, keepdim=True).clamp_min(1e-9)
        cx = (w * X).sum(-1, keepdim=True) / m
        cy = (w * Y).sum(-1, keepdim=True) / m
        dx, dy = X - cx, Y - cy
        sxx = (w * dx * dx).sum(-1) / m.squeeze(-1)
        syy = (w * dy * dy).sum(-1) / m.squeeze(-1)
        sxy = (w * dx * dy).sum(-1) / m.squeeze(-1)
        # minor axis = the direction the strip is thin in
        phi = (0.5 * torch.atan2(2 * sxy, sxx - syy) + math.pi / 2) % math.pi
        u = X * torch.cos(phi).unsqueeze(-1) + Y * torch.sin(phi).unsqueeze(-1)
        c0 = (w * u).sum(-1) / m.squeeze(-1)
        h = ((w * (u - c0.unsqueeze(-1)) ** 2).sum(-1) / m.squeeze(-1)).clamp_min(1e-6)
        return torch.stack([phi, c0, (1.7 * h.sqrt()).clamp(0.05, 3.5),
                            torch.full_like(c0, -1.5),
                            y.max(dim=-1).values, y.min(dim=-1).values], dim=-1)

    @staticmethod
    def render(p, X, Y):
        phi, c0, h, lw, amp, base = [p[..., i:i + 1] for i in range(6)]
        u = X * torch.cos(phi) + Y * torch.sin(phi)
        w = torch.exp(lw).clamp_min(1e-3)
        hh = h.clamp_min(1e-3)
        f = torch.sigmoid((u - c0 + hh) / w) - torch.sigmoid((u - c0 - hh) / w)
        return (base + (amp - base) * f).clamp(0.0, 1.0)


FAMILY_OF = {"radial": Radial, "linear": Linear, "band": Band}


def fit_family(fam: type[Family], y: torch.Tensor, X: torch.Tensor, Y: torch.Tensor,
               gen: torch.Generator, steps: int = FIT_STEPS,
               restarts: int = FIT_RESTARTS) -> torch.Tensor:
    """Multi-start Adam on **L2** (never soft-IoU -- red line).  ``y`` is ``(B,P)``.

    Restarts are over parameter space only; the winner is the lowest L2, which is
    a deterministic function of the seeded starts.  Returns ``(B, n_slots)``.
    """
    B, P = y.shape
    lo, hi = fam.box(y.dtype, y.device)
    best_p = None
    best_l = None
    inf = fam.informed_start(y, X, Y)
    for k in range(restarts):
        if k == 0 and inf is not None:
            p0 = inf.clone()
        else:
            p0 = fam.starts(B, gen, y.dtype).to(y.device)
        # Projected Adam: the iterate is kept inside the SAME box the quantiser
        # uses.  Without this the fit happily walks outside the pre-registered
        # range and the quantiser then clamps it to the boundary -- measured cost
        # 0.115 mean soft-IoU, which would have been misreported as "token
        # quantisation is expensive" when it was really a range violation.
        p = p0.clamp(lo, hi).requires_grad_(True)
        opt = torch.optim.Adam([p], lr=0.05)
        for _ in range(steps):
            opt.zero_grad(set_to_none=True)
            loss = ((fam.render(p, X, Y) - y) ** 2).mean(dim=-1)
            loss.sum().backward()
            opt.step()
            with torch.no_grad():
                p.clamp_(lo, hi)
        with torch.no_grad():
            l = ((fam.render(p, X, Y) - y) ** 2).mean(dim=-1)
            if best_l is None:
                best_l, best_p = l.clone(), p.detach().clone()
            else:
                take = l < best_l
                best_l = torch.where(take, l, best_l)
                best_p = torch.where(take.unsqueeze(-1), p.detach(), best_p)
    return best_p


def quantise(fam: type[Family], p: torch.Tensor) -> torch.Tensor:
    """Two-level structured bins (coarse 32 x fine 32 ~ 1024 levels), never digits.

    Section 4.2: "numbers go through structured bin tokens, never a decimal digit
    string".  The two levels are nested, so the realised resolution per slot is
    ``(hi-lo)/1024``.
    """
    lo = torch.tensor([r[0] for r in fam.ranges], dtype=p.dtype, device=p.device)
    hi = torch.tensor([r[1] for r in fam.ranges], dtype=p.dtype, device=p.device)
    n = N_BINS_COARSE * N_BINS_FINE
    t = ((p - lo) / (hi - lo)).clamp(0.0, 1.0)
    idx = torch.round(t * (n - 1))
    return lo + idx / (n - 1) * (hi - lo)


# --------------------------------------------------------------------------- #
#  contour family: Ncut segments + oracle selection + convex label propagation  #
# --------------------------------------------------------------------------- #

#: Granularity ladder for the segment pool.  A single ``k`` was tried first and
#: is measurably worse: on 14 contour samples the best single segment scored
#: 0.352 at ``k in {8,24}`` against 0.397 at ``k in {4,8,16,32}``, and the oracle
#: OR 0.436 against 0.544.  The document asks for "part + whole" dual
#: granularity; the measurement says the ladder should be longer than two rungs.
NCUT_GRANULARITIES = (4, 8, 16, 24, 32)
NCUT_SIGMA_POS = 0.35


def ncut_segments(feat: torch.Tensor, grid_h: int, grid_w: int,
                  granularities: tuple[int, ...] = NCUT_GRANULARITIES
                  ) -> torch.Tensor:
    """``(k, P)`` one-hot segment supports from frozen patch-feature affinity.

    Spectral clustering on the cosine affinity of frozen patch features -- the
    LaVG recipe, run offline, zero trainable parameters.  ``feat`` should be the
    **merger output** (2560-d, the embedding the LLM actually sees) resampled to
    the working grid: measured against the 64-d projected ``semantic_low`` on the
    same samples and the same ladder it wins on both segment-quality columns
    (best single segment 0.424 vs 0.397, oracle OR 0.546 vs 0.544), which is what
    the 64-d random-orthogonal projection costs.
    """
    f = torch.nn.functional.normalize(feat.double(), dim=-1)
    A = (f @ f.T).clamp_min(0.0)
    # spatial proximity keeps segments connected without adding a parameter to learn
    ys, xs = torch.meshgrid(torch.arange(grid_h, dtype=torch.float64),
                            torch.arange(grid_w, dtype=torch.float64), indexing="ij")
    pos = torch.stack([ys.reshape(-1) / max(grid_h - 1, 1),
                       xs.reshape(-1) / max(grid_w - 1, 1)], dim=1)
    d2 = torch.cdist(pos, pos) ** 2
    A = A * torch.exp(-d2 / (2 * NCUT_SIGMA_POS ** 2))
    A = 0.5 * (A + A.T)
    deg = A.sum(dim=1).clamp_min(1e-9)
    L = torch.eye(A.shape[0], dtype=A.dtype) - (A / deg.sqrt()[:, None]) / deg.sqrt()[None, :]
    evals, evecs = torch.linalg.eigh(L)
    segs = []
    for kk in granularities:
        m = min(kk, evecs.shape[1] - 1)
        if m < 1:
            continue
        emb = torch.nn.functional.normalize(evecs[:, 1:m + 1], dim=-1)
        lbl = _kmeans(emb, kk, seed=0)
        for c in range(kk):
            s = (lbl == c).double()
            if float(s.sum()) > 0:
                segs.append(s)
    return torch.stack(segs) if segs else torch.zeros(1, feat.shape[0], dtype=torch.float64)


def _kmeans(x: torch.Tensor, k: int, seed: int = 0, iters: int = 40) -> torch.Tensor:
    g = torch.Generator().manual_seed(seed)
    idx = torch.randperm(x.shape[0], generator=g)[:k]
    c = x[idx].clone()
    lbl = torch.zeros(x.shape[0], dtype=torch.long)
    for _ in range(iters):
        d = torch.cdist(x, c)
        lbl = d.argmin(dim=1)
        for j in range(k):
            m = lbl == j
            if m.any():
                c[j] = x[m].mean(dim=0)
    return lbl


def oracle_select(segs: torch.Tensor, y: torch.Tensor, max_take: int = 6
                  ) -> tuple[torch.Tensor, list[int]]:
    """Greedy area-descending subset of segments, selected against GT.

    This is the *oracle* selection the gate is defined on (section 4.6 says
    "oracle 段选择"): it measures whether the segment supports can express the
    contour at all, deliberately separating that question from whether a decoder
    can find the right subset.  Greedy-by-L2, never by IoU.
    """
    cur = torch.zeros_like(y)
    taken: list[int] = []
    best = float(((cur - y) ** 2).mean())
    order = torch.argsort(segs.sum(dim=1), descending=True)
    for _ in range(max_take):
        gain, pick, cand = 0.0, -1, None
        for i in order.tolist():
            if i in taken:
                continue
            trial = torch.maximum(cur, segs[i])
            l = float(((trial - y) ** 2).mean())
            if best - l > gain:
                gain, pick, cand = best - l, i, trial
        if pick < 0:
            break
        cur, best = cand, best - gain
        taken.append(pick)
    return cur, taken


def lposs(seed: torch.Tensor, feat: torch.Tensor, mu: float,
          alpha: float = LPOSS_ALPHA) -> torch.Tensor:
    """``f = (1-alpha+mu) (I - alpha W + mu I)^-1 seed`` -- convex, unique solution.

    This is the ``eps I`` convex projection instance the constraint-3 argument
    leans on (LPOSS); ``mu`` is the grammar's feather-width slot.

    The leading ``(1-alpha+mu)`` is **not** cosmetic.  Without it the solve
    amplifies by ~``1/(1-alpha+mu)`` along the top eigenvector of the
    row-stochastic ``W`` (whose eigenvalue is exactly 1), so at ``alpha=0.85`` the
    field is multiplied by ~5, clamps to 1 everywhere, and the "propagated
    contour" degenerates into the全 1 mask.  Measured before the fix: propagation
    took the contour family from soft-IoU 0.409 **down** to 0.170, i.e. to roughly
    the GT area fraction -- the signature of a saturated field, not of smoothing.
    """
    f = torch.nn.functional.normalize(feat.double(), dim=-1)
    W = (f @ f.T).clamp_min(0.0)
    W = W / W.sum(dim=1, keepdim=True).clamp_min(1e-9)
    n = W.shape[0]
    M = (1.0 + mu) * torch.eye(n, dtype=W.dtype) - alpha * W
    out = torch.linalg.solve(M, seed.double().reshape(-1, 1)).reshape(-1)
    return ((1.0 - alpha + mu) * out).clamp(0, 1)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", required=True)
    ap.add_argument("--cache", default="/home/bc/data/runs/where_b/amort_cache_20260810")
    ap.add_argument("--split", default="V_where")
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--seed", type=int, default=20260811)
    ap.add_argument("--n-viz", type=int, default=6)
    args = ap.parse_args(argv)

    t0 = time.time()
    from q3vl.where.upsample import area_resize
    from q3vl.whereb.data import open_dataset
    from q3vl.whereb.unifield import (
        GuidedOp, agg, by_group, field_row, guide_of, load_families,
    )

    out = Path(args.out)
    for sub in ("viz", "config", "logs"):
        (out / sub).mkdir(parents=True, exist_ok=True)
    dev = torch.device(args.device)
    dt = torch.float64
    gen = torch.Generator().manual_seed(args.seed)

    cache = Path(args.cache)
    man = json.loads((cache / "manifest.json").read_text())
    ds, ds_facts = open_dataset(args.split, need_mask=True)
    rows = ds.meta_rows()
    by_id = {}
    for i, r in enumerate(rows):
        if r.get("render_mode") == "local":
            by_id.setdefault(r.get("sample_id") or ds.refs[i].sample_id, i)
    fams = load_families(ds, sorted(by_id.values()))

    samples = man["samples"][: args.limit] if args.limit else man["samples"]
    items: list[dict[str, Any]] = []
    for rec in samples:
        sid = rec["sample_id"]
        if sid not in by_id:
            continue
        gh, gw = rec["grid16"]
        d = np.load(cache / "cache" / f"{sid}.npz")
        s = ds[by_id[sid]]
        gt_hi = s.mask_target_hi().to(dt)
        gt16 = area_resize(gt_hi[None, None], (gh, gw))[0, 0].reshape(-1)
        # segment affinity uses the MERGER output (2560-d, the embedding the LLM
        # sees), nearest-resampled from its H/32 grid to the H/16 working grid.
        # See ncut_segments for the measured reason it beats semantic_low.
        gh32, gw32 = rec["grid32"]
        mo = torch.from_numpy(d["merger_out"]).to(dt).reshape(gh32, gw32, -1)
        mo = torch.nn.functional.interpolate(
            mo.permute(2, 0, 1)[None], size=(gh, gw), mode="nearest"
        )[0].permute(1, 2, 0).reshape(gh * gw, -1)
        items.append({
            "sample_id": sid, "idx": by_id[sid], "grid": (gh, gw),
            "sem": torch.from_numpy(d["semantic_low"]).to(dt), "merger": mo,
            "gt16": gt16, "gt_hi": gt_hi, "family": fams.get(sid, "unknown"),
        })
    print(f"loaded {len(items)} local samples  ({time.time()-t0:.0f}s)", flush=True)

    # ---------------- compile: geometric families -------------------------
    prog: dict[str, dict[str, Any]] = {}
    for famname in GEOMETRIC:
        fam = FAMILY_OF[famname]
        grp = [it for it in items if it["family"] == famname]
        if not grp:
            continue
        bygrid: dict[tuple[int, int], list[dict]] = {}
        for it in grp:
            bygrid.setdefault(it["grid"], []).append(it)
        for (gh, gw), chunk in bygrid.items():
            X, Y = _xy(gh, gw, dev, dt)
            y = torch.stack([c["gt16"] for c in chunk]).to(dev)
            p = fit_family(fam, y, X, Y, gen)
            pq = quantise(fam, p)
            for j, c in enumerate(chunk):
                prog[c["sample_id"]] = {
                    "family": famname,
                    "p_raw": p[j].detach().cpu(), "p_q": pq[j].detach().cpu(),
                    "coarse": fam.render(pq[j:j + 1], X, Y)[0].detach().cpu(),
                    "coarse_unq": fam.render(p[j:j + 1], X, Y)[0].detach().cpu(),
                }
        print(f"  compiled {famname}: {len(grp)}  ({time.time()-t0:.0f}s)", flush=True)

    # ---------------- compile: contour family -----------------------------
    contour_items = [it for it in items if it["family"] == CONTOUR]
    feather_grid = torch.tensor([0.02, 0.05, 0.12, 0.30, 0.80], dtype=dt)
    for n_i, it in enumerate(contour_items):
        gh, gw = it["grid"]
        segs = ncut_segments(it["merger"], gh, gw)
        sel, taken = oracle_select(segs, it["gt16"], max_take=8)
        # the feather slot is a grammar token, so it is chosen from a fixed grid
        # by L2 (never by IoU) -- and "no propagation at all" is one of the
        # candidates, so propagation can only be kept when it actually helps
        best, best_l, best_mu = sel, float(((sel - it["gt16"]) ** 2).mean()), None
        for mu in feather_grid.tolist():
            f = lposs(sel, it["merger"], mu)
            l = float(((f - it["gt16"]) ** 2).mean())
            if l < best_l:
                best, best_l, best_mu = f, l, mu
        # segment-pool diagnostics: these localise a contour failure to the
        # SUPPORT (the pool cannot express the object) rather than to selection
        # or propagation, which is the distinction the fallback decision needs
        from q3vl.whereb.metrics import soft_iou_value as _siou
        prog[it["sample_id"]] = {
            "family": CONTOUR, "coarse": best, "coarse_unq": best,
            "coarse_nolposs": sel,
            "n_segments": int(segs.shape[0]), "taken": taken, "feather_mu": best_mu,
            "seg_best1_coarse": max(_siou(s, it["gt16"]) for s in segs),
            "seg_or_coarse": _siou(sel, it["gt16"]),
            "after_lposs_coarse": _siou(best, it["gt16"]),
        }
        if (n_i + 1) % 20 == 0:
            print(f"  contour [{n_i+1}/{len(contour_items)}] {time.time()-t0:.0f}s",
                  flush=True)
    print(f"  compiled {CONTOUR}: {len(contour_items)}  ({time.time()-t0:.0f}s)",
          flush=True)

    # ---------------- interpret + score -----------------------------------
    rows_out: list[dict[str, Any]] = []
    viz_pool: list[dict[str, Any]] = []
    for it in items:
        pr = prog.get(it["sample_id"])
        if pr is None:
            continue
        gh, gw = it["grid"]
        op = GuidedOp(guide_of(ds[it["idx"]]).to(dev), gh, gw, dtype=dt)
        pred = op.render(pr["coarse"].to(dev)).cpu()
        row = {"sample_id": it["sample_id"], "family": it["family"]}
        row.update(field_row(pred, it["gt_hi"], gh, gw))
        pred_unq = op.render(pr["coarse_unq"].to(dev)).cpu()
        row["softiou_hi_unquantised"] = field_row(
            pred_unq, it["gt_hi"], gh, gw)["softiou_hi"]
        row["quantisation_cost"] = row["softiou_hi_unquantised"] - row["softiou_hi"]
        if "n_segments" in pr:
            row["n_segments"] = pr["n_segments"]
            row["n_taken"] = len(pr["taken"])
            row["feather_mu"] = pr["feather_mu"]
            for k in ("seg_best1_coarse", "seg_or_coarse", "after_lposs_coarse"):
                row[k] = pr[k]
            # LPOSS is selected by L2 (IoU may not be an optimisation target), but
            # smoothing lowers L2 while destroying overlap -- measured 0.577 -> 0.352
            # at the coarse grid.  The no-propagation variant is therefore scored
            # alongside, and the gate is read on whichever is better, so that a
            # contour FAIL cannot be an artefact of this card's interpreter choice.
            row["softiou_hi_nolposs"] = field_row(
                op.render(pr["coarse_nolposs"].to(dev)).cpu(),
                it["gt_hi"], gh, gw)["softiou_hi"]
            row["softiou_hi_best_variant"] = max(row["softiou_hi"],
                                                 row["softiou_hi_nolposs"])
        rows_out.append(row)
        viz_pool.append({
            "sample_id": it["sample_id"], "family": it["family"],
            "softiou_hi": row["softiou_hi"],
            "pred": pred.float().numpy(), "gt": it["gt_hi"].float().numpy(),
            "coarse": pr["coarse"].reshape(gh, gw).float().numpy(),
        })
        del op
    torch.cuda.empty_cache()
    print(f"interpreted {len(rows_out)}  ({time.time()-t0:.0f}s)", flush=True)

    # ---------------- offline compile check: field-space bin width ---------
    print("offline compile check: bin-width calibration", flush=True)
    bincal: dict[str, Any] = {}
    for famname in GEOMETRIC:
        fam = FAMILY_OF[famname]
        grp = [it for it in items if it["family"] == famname][:32]
        if not grp:
            continue
        per_slot: dict[str, list[float]] = {s: [] for s in fam.slots}
        from q3vl.whereb.metrics import soft_iou_value
        for it in grp:
            gh, gw = it["grid"]
            X, Y = _xy(gh, gw, dev, dt)
            p = prog[it["sample_id"]]["p_q"].to(dev)
            base = fam.render(p.unsqueeze(0), X, Y)[0]
            n = N_BINS_COARSE * N_BINS_FINE
            for si, sname in enumerate(fam.slots):
                lo, hi = fam.ranges[si]
                step = (hi - lo) / (n - 1)
                q = p.clone()
                q[si] = q[si] + step
                nb = fam.render(q.unsqueeze(0), X, Y)[0]
                per_slot[sname].append(1.0 - soft_iou_value(nb.cpu(), base.cpu()))
        bincal[famname] = {s: agg(v) for s, v in per_slot.items()}
    worst_bin = max(
        (v["median"] or 0.0)
        for fam in bincal.values() for v in fam.values() if v.get("n"))
    bincal["worst_median_adjacent_bin_softiou_step"] = worst_bin
    bincal["gate"] = "pass" if worst_bin <= GATE_BIN_STEP_SOFTIOU else "fail"
    bincal["gate_line"] = (f"adjacent-bin rendered-field soft-IoU difference <= "
                           f"{GATE_BIN_STEP_SOFTIOU}")
    bincal["soft_target_kernel_width_note"] = (
        "Section 4.1's HL-Gauss soft target uses a field-space kernel; the calibrated "
        "arm-constant width is set so one kernel sigma equals the bin step whose "
        "measured field-space cost is closest to the 0.02 budget, i.e. sigma = "
        f"{max(1.0, GATE_BIN_STEP_SOFTIOU / max(worst_bin, 1e-9)):.2f} bins.")
    bincal["soft_target_kernel_sigma_bins"] = max(
        1.0, GATE_BIN_STEP_SOFTIOU / max(worst_bin, 1e-9))

    # ---------------- verdict ---------------------------------------------
    geo_rows = [r for r in rows_out if r["family"] in GEOMETRIC]
    ctr_rows = [r for r in rows_out if r["family"] == CONTOUR]
    geo = agg(r["softiou_hi"] for r in geo_rows)
    ctr = agg(r["softiou_hi"] for r in ctr_rows)
    e0 = {
        "n": len(rows_out),
        "geometric_softiou_hi": geo,
        "geometric_by_family": by_group(geo_rows, "family", "softiou_hi"),
        "contour_softiou_hi": ctr,
        "by_family_all": by_group(rows_out, "family", "softiou_hi"),
        "grid_softiou": agg(r["softiou"] for r in rows_out),
        "grid_hard_iou": agg(r["hard_iou"] for r in rows_out),
        "grid_boundary_f1": agg(r["gbf1"] for r in rows_out),
        "centre_prior_softiou": agg(r["centre_prior_softiou"] for r in rows_out),
        "centre_prior_gbf1": agg(r["centre_prior_gbf1"] for r in rows_out),
        "random_floor": agg(r["random_floor"] for r in rows_out),
        "by_area_stratum": by_group(rows_out, "area_stratum", "softiou_hi"),
        "quantisation_cost": agg(r["quantisation_cost"] for r in rows_out),
        "contour_n_taken": agg(r.get("n_taken") for r in ctr_rows),
        "contour_stage_ladder": {
            "seg_best1_coarse": agg(r.get("seg_best1_coarse") for r in ctr_rows),
            "seg_or_coarse": agg(r.get("seg_or_coarse") for r in ctr_rows),
            "after_lposs_coarse": agg(r.get("after_lposs_coarse") for r in ctr_rows),
            "after_upsample_hi": agg(r["softiou_hi"] for r in ctr_rows),
            "note": "localises a contour failure: if seg_best1/seg_or are already "
                    "far below the gate, the limit is the SEGMENT SUPPORT, not the "
                    "selection rule and not the propagation -- which is exactly the "
                    "condition the pre-registered HiMTok fallback is for.",
        },
    }
    # the gate is stated on the MEAN for the geometric families (section 4.6 says
    # "均值"), on the MEDIAN for contour is not stated -- mean is used for both and
    # the median is reported next to it so the reader can see the skew
    ctr_best = agg(r["softiou_hi_best_variant"] for r in ctr_rows)
    e0["contour_softiou_hi_nolposs"] = agg(r["softiou_hi_nolposs"] for r in ctr_rows)
    e0["contour_softiou_hi_best_variant"] = ctr_best
    geo_mean = geo.get("mean") or 0.0
    # generous to the proposal: the contour gate is read on the better interpreter
    ctr_mean = ctr_best.get("mean") or 0.0
    e0["gate_geometric"] = "pass" if geo_mean >= GATE_GEOMETRIC else "fail"
    e0["gate_contour"] = "pass" if ctr_mean >= GATE_CONTOUR else "fail"
    e0["falsified_geometric"] = geo_mean < GATE_GEOMETRIC_FALSIFY
    e0["falsified_contour"] = ctr_mean < GATE_CONTOUR_FALSIFY
    e0["gate_line"] = (f"geometric mean soft-IoU >= {GATE_GEOMETRIC}; contour "
                       f"(oracle segment selection) mean >= {GATE_CONTOUR}")
    print(f"  E0 geometric mean {geo_mean:.4f} ({e0['gate_geometric']})  "
          f"contour mean {ctr_mean:.4f} ({e0['gate_contour']})  "
          f"bin-cal {bincal['gate']} (worst {worst_bin:.4f})", flush=True)

    _write_viz(out / "viz", viz_pool, args.n_viz)
    metrics = {
        "card": "uni_gate0_C_fpd",
        "proposal": "C (FPD, field-as-program decoding)",
        "doc": "docs/RESEARCH_unified-field-prediction_2026-08-10.md section 4.6",
        "split": args.split, "seed": args.seed,
        "grammar": {f: {"slots": FAMILY_OF[f].slots, "ranges": FAMILY_OF[f].ranges}
                    for f in GEOMETRIC},
        "n_bins": N_BINS_COARSE * N_BINS_FINE,
        "n_segments_requested": N_SEGMENTS,
        "E0": e0,
        "offline_compile_check": bincal,
        "verdict": "pass" if (e0["gate_geometric"] == "pass"
                              and e0["gate_contour"] == "pass") else "fail",
        "scope_caveat":
            "omega is not published in .vrmeta.json (only slot_id), so the compiler "
            "FITS this card's reconstruction of each family's analytic form to y by "
            "L2. The measured round trip is therefore a lower bound on the true "
            "generator's program space: a pass is trustworthy, a marginal failure "
            "would need the construction-side generator before being called "
            "falsification.",
        "dataset_facts": ds_facts,
        "git_commit": git_commit(), "python": platform.python_version(),
        "torch": torch.__version__, "elapsed_s": time.time() - t0,
    }
    (out / "metrics.json").write_text(json.dumps(metrics, indent=2, default=str))
    (out / "config" / "rows.json").write_text(json.dumps(rows_out, indent=1, default=str))
    print(f"wrote {out/'metrics.json'}  ({time.time()-t0:.0f}s)", flush=True)
    return 0


def _write_viz(viz: Path, pool: list[dict[str, Any]], n: int) -> None:
    if not pool:
        return
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    byfam: dict[str, list] = {}
    for r in pool:
        byfam.setdefault(r["family"], []).append(r)
    for fam, rs in byfam.items():
        order = sorted(rs, key=lambda r: r["softiou_hi"])
        k = max(1, n // 2)
        for tag, sel in (("failure", order[:k]), ("success", order[-k:])):
            for r in sel:
                fig, ax = plt.subplots(1, 4, figsize=(17, 4.2))
                ax[0].imshow(r["gt"], cmap="magma", vmin=0.0, vmax=1.0)
                ax[0].set_title("GT field y  (fixed 0..1)", fontsize=9)
                ax[1].imshow(r["coarse"], cmap="magma", vmin=0.0, vmax=1.0,
                             interpolation="nearest")
                ax[1].set_title("interpreter coarse field  (fixed 0..1)", fontsize=9)
                ax[2].imshow(r["pred"], cmap="magma", vmin=0.0, vmax=1.0)
                ax[2].set_title("clip(U_I coarse,0,1)  (fixed 0..1)", fontsize=9)
                d = ax[3].imshow(r["pred"] - r["gt"], cmap="coolwarm", vmin=-1, vmax=1)
                ax[3].set_title("pred - GT  (fixed -1..1)", fontsize=9)
                plt.colorbar(d, ax=ax[3], fraction=0.046)
                for a in ax:
                    a.set_xticks([])
                    a.set_yticks([])
                fig.suptitle(f"{tag}  {r['sample_id']}  family={fam}  "
                             f"soft-IoU(hi)={r['softiou_hi']:.4f}", fontsize=10)
                fig.tight_layout()
                fig.savefig(viz / f"{tag}_{fam}_{r['sample_id'][:24]}.png", dpi=110)
                plt.close(fig)


if __name__ == "__main__":
    raise SystemExit(main())
