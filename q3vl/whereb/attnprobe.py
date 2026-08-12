"""Analysis side of PR-ATT1-E1: head scan, criteria, learnability up-probe.

Reads what :mod:`q3vl.whereb.attnread` exported and turns it into the
preregistered numbers.  Every criterion here obeys the 2026-08-05 red lines:

* **no AUC anywhere** -- the three sanctioned columns are grid soft-IoU,
  grid boundary F1, and the zero-parameter centre-prior baseline;
* thresholding is always **matched-area top-k**, never a per-field tuned
  threshold, so a field cannot buy coverage it did not earn;
* every criterion row carries the centre-prior column **and** the shuffled
  column -- they are criteria, not appendices;
* the fields that enter a number are the **raw, un-normalised** attention
  values on the surviving cells.  The only rescaling in this module is the
  per-head **arm constant** used to make heads commensurable before they are
  summed (:func:`head_norm_constants`), which is the s-cache contract's
  sanctioned form ("normalisation may only use whole-arm constants").

One reading decision is load-bearing and is stated here rather than buried:
**"grid soft-IoU" of an attention field means soft-IoU between the matched-area
top-k binarisation of the field and the soft GT grid.**  Soft-IoU of a raw
attention field against a mask would be meaningless -- attention values live
around 1e-3 while the GT lives around 1, so ``sum(min)/sum(max)`` would report
roughly the attention mass and rank heads by how much total mass they put on the
image, not by where they put it.  Top-k binarisation is also invariant to any
monotone per-field transform, which is exactly why the red line names it.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass, field
from typing import Any, Mapping, Sequence

import numpy as np
import torch

__all__ = [
    "fit_oof_split", "gt_merged_grid", "field_scores", "head_norm_constants",
    "combine_heads", "signed_rank_z", "paired_wilcoxon", "max_stat_fwer",
    "GatedLinearHead", "HeadStackCNN", "train_learnable_head",
    "EXCLUDE_FIRST_LAYERS", "TOP_K_HEADS",
]

#: PROPOSAL section 4 P-W3: the layer scan drops the first two layers before
#: ranking.  Recorded as a constant so the REPORT can state it and so the
#: deepstack caveat below stays attached to it.
#:
#: Caveat (verified 2026-08-10 in ``Qwen3VLTextModel.forward``): deepstack
#: re-injects visual features at language layers **0, 1, 2**, so this filter
#: removes two of the three injection layers and keeps the third.  The PROPOSAL
#: believed injection happened at layers 5/11/17 (those are vision-tower block
#: indices) and set the filter without knowing that.  The filter is kept as
#: preregistered; layer 2 is flagged in the report instead of being dropped
#: after the fact.
EXCLUDE_FIRST_LAYERS = 2

#: PROPOSAL section 2.2 step 7 probe tier A: soft-IoU-weighted convex top-k.
TOP_K_HEADS = 8


# --- folds ------------------------------------------------------------------

def fit_oof_split(
    rows: Sequence[Mapping[str, Any]],
    *,
    seed: str = "verasplit-v1",
    fold_names: tuple[str, str] = ("fit", "oof"),
) -> dict[str, str]:
    """Split samples into two folds **by ``source_image_id``**, never within one.

    Why this is not read off the S/P sidecar, despite the card asking for it:
    ``tools/data_splits/splits.sqlite3`` assigns each ``source_id`` to
    train/val/test, and V_where's 162 sources land 146/10/4 -- with 2 sources not
    present at all, because the sidecar's ``meta.builds`` stops at g3/l4 while
    V_where contains g4/l5/l6.  There is no 448/448 partition inside it to read.

    So the fold key is generated with the sidecar's **own published rule family**
    (``sha1(seed:source_id)``, ``meta.split_seed = "verasplit-v1"``) rather than
    an invented one, and groups are then walked in hash order and greedily given
    to whichever fold currently holds fewer **local** samples.  Local count is
    the balancing target because the local subset is the only one that carries a
    GT mask and therefore the only one that enters a spatial criterion.

    Group integrity is the property that actually matters here: V_where's 896
    samples come from 162 source images, so a per-sample split would put near
    duplicates of the same photograph on both sides and inflate every OOF number.
    """
    groups: dict[str, list[Mapping[str, Any]]] = {}
    for r in rows:
        key = str(r.get("source_image_id") or f"__nosrc__{r['sample_id']}")
        groups.setdefault(key, []).append(r)

    def hkey(src: str) -> str:
        return hashlib.sha1(f"{seed}:{src}".encode()).hexdigest()

    order = sorted(groups, key=hkey)
    n_local = [0, 0]
    n_total = [0, 0]
    out: dict[str, str] = {}
    for src in order:
        members = groups[src]
        loc = sum(1 for m in members if m.get("render_mode") == "local")
        side = 0 if (n_local[0], n_total[0]) <= (n_local[1], n_total[1]) else 1
        n_local[side] += loc
        n_total[side] += len(members)
        for m in members:
            out[str(m["sample_id"])] = fold_names[side]
    return out


# --- GT on the merged grid --------------------------------------------------

def gt_merged_grid(mask_low: torch.Tensor, grid_h: int, grid_w: int) -> torch.Tensor:
    """Where-A's ``.masklow.npy`` (``out/16``) -> the image-token grid (``out/32``).

    The published low view is on the ``F_pre`` patch grid, which is exactly twice
    the image-token grid in each direction, so this is a single exact 2x2 area
    mean -- "area-mean downsample of the soft GT to the merged grid", with no
    interpolation and no resampling phase error.
    """
    from q3vl.where.upsample import area_resize

    m = torch.as_tensor(mask_low, dtype=torch.float32)
    if m.shape[-2] != grid_h * 2 or m.shape[-1] != grid_w * 2:
        raise ValueError(
            f"mask_low {tuple(m.shape)} is not 2x the merged grid {(grid_h, grid_w)}"
        )
    return area_resize(m.reshape(1, 1, *m.shape[-2:]), (grid_h, grid_w))[0, 0]


# --- per-head scoring -------------------------------------------------------

def field_scores(
    fields: np.ndarray,
    gt: np.ndarray,
    valid: np.ndarray,
    *,
    eps: float = 1e-8,
) -> np.ndarray:
    """Matched-area top-k grid soft-IoU for **every** head at once.

    ``fields`` is ``(..., n_cells)`` (any leading shape, typically
    ``(n_layers, n_heads)``), ``gt`` and ``valid`` are ``(n_cells,)``.  Only
    ``valid`` cells take part: invalid cells can neither be selected by the top-k
    nor contribute to the union, so a sink cell cannot be spent as coverage.

    Returns ``(...)`` of soft-IoU between the top-k mask and the **soft** GT.
    """
    f = np.asarray(fields, dtype=np.float64)
    lead = f.shape[:-1]
    n = f.shape[-1]
    g = np.asarray(gt, dtype=np.float64).reshape(-1)
    v = np.asarray(valid, dtype=bool).reshape(-1)
    if g.size != n or v.size != n:
        raise ValueError("fields / gt / valid disagree on the number of cells")

    gv = g[v]
    k = int((gv > 0.5).sum())
    k = max(1, min(k, int(v.sum())))

    flat = f.reshape(-1, n)[:, v]                      # (m, n_valid)
    m, nv = flat.shape
    # top-k by partition; ties broken by index, which is deterministic and, being
    # the same rule for every field including the centre prior, is not a knob.
    idx = np.argpartition(-flat, kth=k - 1, axis=1)[:, :k]
    pred = np.zeros((m, nv), dtype=np.float64)
    np.put_along_axis(pred, idx, 1.0, axis=1)

    gt_b = np.broadcast_to(gv, (m, nv))
    inter = np.minimum(pred, gt_b).sum(axis=1)
    union = np.maximum(pred, gt_b).sum(axis=1)
    return (inter / (union + eps)).reshape(lead)


def head_norm_constants(
    stack: Sequence[np.ndarray],
    n_cells: Sequence[int],
) -> tuple[np.ndarray, np.ndarray]:
    """Whole-arm per-head ``(mu, sigma)`` for making heads commensurable.

    Heads differ by an order of magnitude in how much mass they put on the image
    at all, so a convex combination of raw rows would be a report on head mass
    rather than on head agreement.  The fix has to be a **whole-arm constant**
    (s-cache contract: "normalisation may only use whole-arm constants; per-image
    is a red line"), so the statistics are pooled over the whole fit fold.

    The per-image factor applied first is ``n_cells``, i.e. the field is expressed
    as *attention relative to uniform over the image*.  That is a deterministic
    geometric factor, not a data-dependent per-image statistic: without it the
    same head would sit at a different scale on a 16x16 image than on a 16x64 one
    purely because the mass is spread over four times as many cells, and a single
    arm constant could not serve both.

    Returns ``mu, sigma`` shaped like one sample's ``(n_layers, n_heads)``.
    """
    tot = None
    sq = None
    cnt = 0
    for arr, n in zip(stack, n_cells):
        a = np.asarray(arr, dtype=np.float64) * float(n)   # (L, H, n_cells)
        s = a.mean(axis=-1)
        s2 = (a ** 2).mean(axis=-1)
        tot = s if tot is None else tot + s
        sq = s2 if sq is None else sq + s2
        cnt += 1
    if not cnt:
        raise ValueError("empty fit fold")
    mu = tot / cnt
    var = np.maximum(sq / cnt - mu ** 2, 0.0)
    sigma = np.sqrt(var)
    sigma[sigma <= 0] = 1.0
    return mu, sigma


def combine_heads(
    fields: np.ndarray,
    idx: Sequence[tuple[int, int]],
    weights: Sequence[float],
    mu: np.ndarray,
    sigma: np.ndarray,
    n_cells: int,
) -> np.ndarray:
    """Convex combination of the selected heads, on the arm-constant z scale."""
    w = np.asarray(weights, dtype=np.float64)
    if w.size != len(idx):
        raise ValueError("weights and idx disagree")
    s = w.sum()
    if s <= 0:
        raise ValueError("non-positive weight sum; convexity would be undefined")
    w = w / s
    out = np.zeros(fields.shape[-1], dtype=np.float64)
    for (l, h), wi in zip(idx, w):
        z = (np.asarray(fields[l, h], dtype=np.float64) * n_cells - mu[l, h]) / sigma[l, h]
        out += wi * z
    return out


# --- statistics -------------------------------------------------------------

def signed_rank_z(diffs: np.ndarray) -> float:
    """Standardised Wilcoxon signed-rank statistic of paired differences.

    ``W = sum sign(d) * rank(|d|)`` over the non-zero differences, divided by
    ``sqrt(sum rank^2)``.  Written out rather than taken from scipy because the
    max-stat permutation below has to recompute it under sign flips, and the
    statistic and its null must come from the same formula.  Ties in ``|d|`` get
    mid-ranks, which is the standard treatment and keeps the flip-null exact.
    """
    d = np.asarray(diffs, dtype=np.float64)
    d = d[d != 0]
    if d.size == 0:
        return 0.0
    a = np.abs(d)
    order = np.argsort(a, kind="mergesort")
    ranks = np.empty(a.size, dtype=np.float64)
    sorted_a = a[order]
    i = 0
    while i < a.size:                       # mid-ranks for ties
        j = i
        while j + 1 < a.size and sorted_a[j + 1] == sorted_a[i]:
            j += 1
        ranks[order[i:j + 1]] = 0.5 * (i + j) + 1.0
        i = j + 1
    w = float(np.sum(np.sign(d) * ranks))
    denom = float(np.sqrt(np.sum(ranks ** 2)))
    return w / denom if denom > 0 else 0.0


def paired_wilcoxon(a: Sequence[float], b: Sequence[float]) -> dict[str, Any]:
    """Median/mean paired ``a - b`` with the two-sided signed-rank p-value."""
    x = np.asarray(a, dtype=np.float64)
    y = np.asarray(b, dtype=np.float64)
    if x.shape != y.shape:
        raise ValueError("unpaired inputs")
    d = x - y
    out: dict[str, Any] = {
        "n": int(d.size),
        "delta_median": float(np.median(d)) if d.size else None,
        "delta_mean": float(np.mean(d)) if d.size else None,
        "median_a": float(np.median(x)) if x.size else None,
        "median_b": float(np.median(y)) if y.size else None,
        "z": signed_rank_z(d),
        "n_nonzero": int((d != 0).sum()),
        "frac_positive": float((d > 0).mean()) if d.size else None,
    }
    try:
        from scipy.stats import wilcoxon

        if int((d != 0).sum()) > 0:
            res = wilcoxon(x, y, zero_method="wilcox", alternative="two-sided")
            out["p_value"] = float(res.pvalue)
            out["statistic"] = float(res.statistic)
        else:
            out["p_value"] = 1.0
    except Exception as exc:                              # pragma: no cover
        out["p_value"] = None
        out["p_error"] = f"{type(exc).__name__}: {exc}"
    return out


def max_stat_fwer(
    diffs_by_key: Mapping[str, np.ndarray],
    *,
    n_perm: int = 20000,
    seed: int = 20260810,
) -> dict[str, Any]:
    """Westfall-Young max-statistic FWER correction across a family of tests.

    All keys must be measured on the **same samples in the same order**: one
    sign-flip vector is drawn per permutation and applied to every key at once,
    which preserves the dependence between pools instead of pretending they are
    independent (Bonferroni would; these pools are slices of one forward pass and
    are strongly dependent, so Bonferroni would be badly conservative).

    The per-key statistic is |signed-rank z|, exactly the statistic
    :func:`paired_wilcoxon` reports, so the corrected p-value tests the same
    hypothesis the uncorrected one does.
    """
    keys = list(diffs_by_key)
    if not keys:
        return {"keys": [], "n_perm": n_perm}
    mats = [np.asarray(diffs_by_key[k], dtype=np.float64) for k in keys]
    n = mats[0].size
    for m in mats:
        if m.size != n:
            raise ValueError("max-stat FWER needs the same samples in every key")
    obs = {k: abs(signed_rank_z(m)) for k, m in zip(keys, mats)}

    rng = np.random.default_rng(seed)
    ge = {k: 0 for k in keys}
    for _ in range(n_perm):
        signs = rng.choice(np.array([-1.0, 1.0]), size=n)
        stats = [abs(signed_rank_z(m * signs)) for m in mats]
        mx = max(stats)
        for k in keys:
            if mx >= obs[k] - 1e-12:
                ge[k] += 1
    return {
        "keys": keys,
        "n_perm": n_perm,
        "seed": seed,
        "observed_abs_z": obs,
        "p_fwer": {k: (ge[k] + 1) / (n_perm + 1) for k in keys},
        "method": "westfall_young_max_abs_signed_rank_z_sign_flip",
    }


# --- learnability up-probe --------------------------------------------------

class GatedLinearHead(torch.nn.Module):
    """Per-head sigmoid gate on a linear combination -- PROPOSAL tier B, ~1.2k params."""

    def __init__(self, n_layers: int = 36, n_heads: int = 32):
        super().__init__()
        self.gate = torch.nn.Parameter(torch.zeros(n_layers, n_heads))
        self.scale = torch.nn.Parameter(torch.ones(1))
        self.bias = torch.nn.Parameter(torch.zeros(1))

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        """``z``: ``(n_layers, n_heads, n_cells)`` arm-normalised -> ``(n_cells,)`` logit."""
        g = torch.sigmoid(self.gate).reshape(-1, 1)
        return (z.reshape(g.shape[0], -1) * g).sum(dim=0) * self.scale + self.bias

    def n_params(self) -> int:
        return sum(p.numel() for p in self.parameters())


class HeadStackCNN(torch.nn.Module):
    """Selected heads as channels -> 1x1 -> 3x3 -> 1 -- PROPOSAL tier C, ~27k params.

    The F-LMM precedent this copies (2406.05821) learns head selection *inside* a
    small conv stack rather than choosing heads up front, which is why the input
    is the top-128 head stack rather than the top-8 convex field.
    """

    def __init__(self, n_in: int = 128, mid: int = 64, mid2: int = 32):
        super().__init__()
        self.c1 = torch.nn.Conv2d(n_in, mid, 1)
        self.c2 = torch.nn.Conv2d(mid, mid2, 3, padding=1)
        self.c3 = torch.nn.Conv2d(mid2, 1, 1)
        self.act = torch.nn.GELU()

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        """``z``: ``(n_in, H, W)`` -> ``(H, W)`` logit."""
        x = z.unsqueeze(0)
        x = self.act(self.c1(x))
        x = self.act(self.c2(x))
        return self.c3(x)[0, 0]

    def n_params(self) -> int:
        return sum(p.numel() for p in self.parameters())


def train_learnable_head(
    model: torch.nn.Module,
    fit_items: Sequence[tuple[torch.Tensor, torch.Tensor, torch.Tensor]],
    *,
    epochs: int = 30,
    lr: float = 3e-2,
    seed: int = 20260810,
    device: str = "cpu",
) -> dict[str, Any]:
    """Fit a small head with **soft-target BCE** on valid cells only.

    Red line: IoU is never the optimisation target (it is an "hedge by covering
    more" engine).  Soft-target BCE is the PROPOSAL's prescribed main term and is
    volume-unbiased under soft labels (Bertels 2211.04161).  Plain BCE, not the
    balanced variant -- balanced BCE is the over-coverage engine that lifted the
    positive prior from 2% to 50% in W01/W02 and it is explicitly abolished.

    ``fit_items`` are ``(z, gt, valid)`` triples already on the arm-normalised
    scale.  Checkpoint selection is by **fit-fold BCE**, never by a validation
    loss on the fold the number is reported on.
    """
    torch.manual_seed(seed)
    model = model.to(device).train()
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    hist: list[float] = []
    best = float("inf")
    best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
    order = np.arange(len(fit_items))
    rng = np.random.default_rng(seed)
    for _ in range(epochs):
        rng.shuffle(order)
        tot, nb = 0.0, 0
        for i in order:
            z, gt, valid = fit_items[int(i)]
            z, gt, valid = z.to(device), gt.to(device), valid.to(device)
            logit = model(z).reshape(-1)
            t = gt.reshape(-1)
            v = valid.reshape(-1)
            loss = torch.nn.functional.binary_cross_entropy_with_logits(
                logit[v], t[v].clamp(0.0, 1.0)
            )
            opt.zero_grad()
            loss.backward()
            opt.step()
            tot += float(loss.detach())
            nb += 1
        ep = tot / max(nb, 1)
        hist.append(ep)
        if ep < best:
            best = ep
            best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
    model.load_state_dict(best_state)
    return {"epochs": epochs, "lr": lr, "seed": seed,
            "fit_bce_history": hist, "best_fit_bce": best,
            "n_params": int(sum(p.numel() for p in model.parameters())),
            "selection": "best fit-fold BCE (val loss is banned for selection)"}
