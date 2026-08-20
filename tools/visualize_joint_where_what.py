#!/usr/bin/env python3
"""Joint where + what rendering on a split (normal-only).

``--alpha-source`` picks which alpha the prediction side composes with:

* ``pred`` (default) -- the where arm's inferred field, six panels, grid-level
  soft-IoU reported.  Only ``V_what`` has such a field directory;
* ``gt`` -- ``alpha_pred := alpha_gt``, five panels, **no** soft-IoU column
  (with ``alpha_pred == alpha_gt`` it is 1.0 by definition, so it is neither
  computed nor printed) and no predfield directory is opened.  This is the mode
  for splits that have no where prediction on disk (e.g. ``train``); the panel
  titles and the text bar state ``alpha = GT`` on every figure.

The where side (``--alpha-source pred`` only) is the ST_LANG arm's
already-inferred field directory
``predfield_stlang_v2seg`` (two sources in one directory: ``stlang_m_low`` =
408 predicted local fields, ``style_definition_ones`` = 489 style rows whose
field is the generating law's constant one -- the two are never merged and the
source is printed on every panel).  The what side is the MLP row
``whatb_QDEC_E031_MLP_NOL8_L3``, rebuilt from its own ``run_setup.json`` /
``best.pt['config']`` and run on the ``V_what.generated.none`` z cache.

Frozen image formation, used on both sides::

    I_hat = (1 - alpha) * I + alpha * f_hat(I)          short side 512

with ``alpha`` = the upsampled predicted field and ``f_hat`` = the what model's
GLUT transform on the prediction side, and ``alpha`` = GT alpha, ``f_hat`` =
the sample's true LUT (``LutBank``) on the GT side.

Everything numeric is imported, never re-implemented: ``SampleStore`` (images +
GT alpha), ``LutBank`` (the data law), ``ZCache`` (z), ``area_resize`` (the one
sanctioned resize operator), ``criteria.compose_hat`` (the formation),
``colorimetry.delta_e00`` (CIEDE2000), ``metrics.soft_iou_value`` (min/max
soft-IoU).

Visualisation discipline
------------------------
* the dE00 panels use ONE global colour scale for every sample; ``vmax`` is the
  p99 of the pooled per-pixel dE00 over the whole scored population and is
  printed on every colour bar -- unless the scoring pass covered a pre-drawn
  subsample instead of the whole population, in which case ``vmax`` is the
  :data:`REFERENCE_VMAX` of the V_what full-population run and both that fact
  and the subsample's own pooled p99 are printed on the bar and stored in the
  manifest,
* alpha panels use a fixed ``[0, 1]`` scale, never a per-image min-max,
* non-finite / out-of-range field cells are masked and drawn white, and counted,
* fields reach image resolution through ``area_resize`` only,
* every number printed on a figure is computed from the raw field / raw dE00,
  never from the colour-mapped array.

CPU only.  ``CUDA_VISIBLE_DEVICES`` is cleared at import time.
"""

from __future__ import annotations

import os

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")
os.environ["CUDA_VISIBLE_DEVICES"] = ""

import argparse
import hashlib
import json
import textwrap
import time
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import torch

ROOT = Path("/home/bc/VeraRetouch")
WHAT_RUN = Path("/home/bc/data/runs/what_b/whatb_QDEC_E031_MLP_NOL8_L3")
PREDFIELD = Path("/home/bc/data/runs/whatb/predfield_stlang_v2seg")
ZCACHE = Path("/home/bc/data/runs/whatb/zcache_v2seg/V_what.generated.none.zcache.pt")
BANK_DIR = Path("/var/cache/veradata/preset_bank_full")
DEFAULT_OUT = ROOT / "docs/assets/joint_stlang_mlp_20260817"

#: the quantiles of the per-sample dE00 mean the selection walks, per class
QUANTILES: tuple[float, ...] = (0.05, 0.25, 0.50, 0.75, 0.95)

#: the value range of ``q3vl.whereb.amort.data.family_labels``
FAMILY_VALUES: tuple[str, ...] = ("radial", "semantic", "band", "linear",
                                  "unknown")

#: pooled-pixel dE00 histogram used for the global colour scale
HIST_BINS = 20000
HIST_HI = 200.0

#: dE00 vmax of the V_what full-population run
#: (``docs/assets/joint_stlang_mlp_20260817/manifest.json`` ->
#: ``colour_scales.de00.vmax``, p99 of the pooled per-pixel dE00 over all 567
#: scored V_what normal-only rows).  Borrowed by a pre-sampled run so figures
#: from different output directories share one colour scale; every figure that
#: borrows it says so on the colour bar.
REFERENCE_VMAX = 20.82
REFERENCE_VMAX_SOURCE = ("docs/assets/joint_stlang_mlp_20260817/manifest.json "
                         "colour_scales.de00.vmax (V_what normal-only, n=567, "
                         "alpha_source=pred)")

# --------------------------------------------------------------------------- #
# run-time witnesses: "defined but never wired" has happened three times
# --------------------------------------------------------------------------- #
_CALLED: dict[str, int] = {"delta_e00": 0, "soft_iou": 0, "area_resize": 0,
                           "compose_hat": 0, "lut_f_star": 0}


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", type=Path, default=DEFAULT_OUT,
                    help="output directory (PNGs + manifest.json + per_sample.jsonl)")
    # deferred import: q3vl.whatb pulls torch in, and parse_args runs before the
    # tool's own heavy imports on purpose (argument coherence first)
    from q3vl.whatb.splits import (
        DATASET_VERSION_CHOICES as _DVC,
        DEFAULT_DATASET_VERSION as _DDV,
    )
    ap.add_argument("--dataset-version", default=_DDV, choices=list(_DVC),
                    help="sft2seg index口径 the split's rows are read from "
                         "(q3vl.whatb.splits.DATASET_VERSIONS); same spelling, "
                         "choices and default as the arm runners.  Recorded in "
                         "manifest.json.  The boards rendered before 2026-08-18 "
                         "11:49 were drawn on 'v20260804' -- pass it to redraw "
                         "them")
    ap.add_argument("--split", default="V_what",
                    help="split to score.  The index rows, the GT alpha store "
                         "and the summary figure all follow this flag; the z "
                         "cache of the same split must be passed with --zcache. "
                         "--alpha-source pred additionally needs a --predfield "
                         "directory covering the split (only V_what has one)")
    ap.add_argument("--alpha-source", choices=("pred", "gt"), default="pred",
                    help="which alpha the prediction side composes with.  "
                         "'pred' (default) = the --predfield field, six panels, "
                         "grid soft-IoU reported.  'gt' = alpha_pred := alpha_gt, "
                         "five panels (Input | GT alpha | GT edited | pred edited "
                         "| dE00), no predfield is opened and soft-IoU is neither "
                         "computed nor printed (it is 1.0 by definition when the "
                         "two alphas are the same array)")
    ap.add_argument("--presample", choices=("auto", "off"), default="auto",
                    help="'auto' (default): under --select random with --n-total "
                         "> 0 the --n-total rows per rendered class are drawn "
                         "from the split's normal-only INDEX rows before the "
                         "scoring pass, so only those rows are scored, and every "
                         "population statistic in the manifest is a statistic of "
                         "that subsample, not of the split.  'off': score the "
                         "whole normal-only population first and draw afterwards")
    ap.add_argument("--de00-vmax", default="auto",
                    help="dE00 colour-scale vmax.  'auto' (default) = this run's "
                         "own pooled-pixel p99, except when the scoring pass was "
                         "pre-sampled, where the V_what full-population "
                         f"{REFERENCE_VMAX} is borrowed instead (the subsample's "
                         "own pooled p99 is still computed and stored).  A float "
                         "forces the value")
    ap.add_argument("--n-per-bin", type=int, default=2,
                    help="samples rendered per quantile bin per class "
                         "(--select quantile only; ignored by --select random)")
    ap.add_argument("--select", choices=("quantile", "random"), default="quantile",
                    help="how the rendered rows are picked out of a class.  "
                         "'quantile' (default) = the historical stratified rule: "
                         "--n-per-bin rows inside a +-1%% rank window around each "
                         "of the 5 dE00 quantiles.  'random' = one seeded "
                         "without-replacement draw of --n-total rows from ALL of "
                         "the class's scored rows, no stratification, so the draw "
                         "follows the class's own dE00 distribution instead of "
                         "clustering on 5 quantile anchors")
    ap.add_argument("--sample-seed", type=int, default=0,
                    help="seed of the --select random draw (numpy default_rng); "
                         "same seed + same scored population => same sample_ids.  "
                         "Unused by --select quantile")
    ap.add_argument("--n-total", type=int, default=0,
                    help="rows drawn per rendered class by --select random; "
                         "0 = unset (an error under --select random).  Unused by "
                         "--select quantile")
    ap.add_argument("--family-cap", action="append", default=None,
                    metavar="FAMILY=N",
                    help="mask-family quota on the pre-sample draw, e.g. "
                         "'band=4'.  Repeatable and/or comma separated "
                         "('band=4,linear=10').  FAMILY is one of "
                         f"{'/'.join(FAMILY_VALUES)}; the label comes from "
                         "q3vl.whereb.amort.data.family_labels (fetched live "
                         "from the construction side).  Empty (default) = the "
                         "draw is not family aware and behaves exactly as "
                         "before.  Requires --select random with --n-total > 0 "
                         "and an active pre-sample (--presample auto)")
    ap.add_argument("--exclude-ids", type=Path, default=None,
                    help="file of sample_id s (one per line, whitespace "
                         "separated) removed from the SAMPLING POOL, i.e. from "
                         "the split's normal-only index rows before the "
                         "pre-sample draw / the quantile walk sees them.  Same "
                         "effect as pointing the run at a filtered split index, "
                         "without moving the index root.  Default unset: the "
                         "pool is the whole normal-only population and every "
                         "byte of the run is what it was before this flag "
                         "existed")
    ap.add_argument("--family-label", action="store_true",
                    help="fetch the mask family of every drawn row and print "
                         "mask_family=<label> on the figure + record it on the "
                         "manifest, WITHOUT changing the draw.  --family-cap "
                         "already does this as a side effect of quota'ing the "
                         "draw; this flag is the label without the quota.  "
                         "Default off: no label is fetched and the figures are "
                         "byte-identical to a run without the flag")
    ap.add_argument("--family-cap-pool-mult", type=int, default=4,
                    help="candidate pool = max(mult * --n-total, "
                         "--family-cap-pool-min) index rows, clipped to the "
                         "class population; only the pool is family-labelled "
                         "(--family-cap only)")
    ap.add_argument("--family-cap-pool-min", type=int, default=400,
                    help="floor of the candidate pool size (--family-cap only)")
    ap.add_argument("--classes", choices=("both", "local", "style"), default="both",
                    help="which class(es) to RENDER.  This filters the selection "
                         "step only: the scoring pass, the pooled-pixel colour "
                         "scale and the summary figure always cover the whole "
                         "normal-only population (both classes), so figures from "
                         "different --classes runs share one colour scale.  Under "
                         "an active pre-sample (see --presample) it filters the "
                         "scoring pass too: only the drawn rows of the rendered "
                         "class(es) are scored")
    ap.add_argument("--predfield", type=Path, default=PREDFIELD)
    ap.add_argument("--what-run", type=Path, default=WHAT_RUN)
    ap.add_argument("--zcache", type=Path, default=ZCACHE)
    ap.add_argument("--bank-dir", type=Path, default=BANK_DIR)
    ap.add_argument("--point-chunk", type=int, default=200_000,
                    help="colour queries per carrier chunk (memory only)")
    ap.add_argument("--dpi", type=int, default=150)
    ap.add_argument("--limit", type=int, default=0,
                    help="score only the first N rows (smoke runs); 0 = all")
    ap.add_argument("--no-bank-hash", action="store_true",
                    help="skip the 2.7 GB luts.npz sha256")
    return ap.parse_args(argv)


def parse_family_cap(specs: Sequence[str] | None) -> dict[str, int]:
    """``['band=4', 'linear=10,radial=8']`` -> ``{'band': 4, ...}``.

    Empty / ``None`` gives ``{}``, which is the "not family aware" switch: the
    caller then runs the historical draw untouched.
    """
    caps: dict[str, int] = {}
    for spec in specs or []:
        for part in str(spec).split(","):
            part = part.strip()
            if not part:
                continue
            if "=" not in part:
                raise SystemExit(
                    f"--family-cap takes FAMILY=N, got {part!r}")
            name, _, val = part.partition("=")
            name = name.strip()
            if name not in FAMILY_VALUES:
                raise SystemExit(
                    f"--family-cap: unknown family {name!r}; the label range of "
                    f"family_labels is {list(FAMILY_VALUES)}")
            try:
                cap = int(val.strip())
            except ValueError:
                raise SystemExit(
                    f"--family-cap {name}: {val.strip()!r} is not an integer")
            if cap < 0:
                raise SystemExit(f"--family-cap {name}: N must be >= 0")
            if name in caps and caps[name] != cap:
                raise SystemExit(
                    f"--family-cap {name} given twice with different values "
                    f"({caps[name]} and {cap})")
            caps[name] = cap
    return caps


def sha256_file(path: Path, *, chunk: int = 1 << 22) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        while True:
            b = fh.read(chunk)
            if not b:
                break
            h.update(b)
    return h.hexdigest()


# --------------------------------------------------------------------------- #
# model
# --------------------------------------------------------------------------- #
def load_what_model(run_dir: Path):
    """Rebuild the arm's model from the checkpoint's own config dict."""
    from q3vl.whatb.scripts.run_epr030_arm import Epr030Config, Epr030Model

    ckpt = torch.load(run_dir / "best.pt", map_location="cpu", weights_only=False)
    cfg_dict = dict(ckpt["config"])
    setup = json.loads((run_dir / "run_setup.json").read_text(encoding="utf-8"))
    if setup["config"] != cfg_dict:
        raise AssertionError(
            f"{run_dir}: best.pt['config'] and run_setup.json['config'] disagree; "
            "the model would be rebuilt from an unknown recipe")
    fields = set(Epr030Config.__dataclass_fields__)
    unknown = {k: v for k, v in cfg_dict.items() if k not in fields}
    cfg = Epr030Config(**{k: v for k, v in cfg_dict.items() if k in fields})
    model = Epr030Model(cfg)
    model.load_state_dict(ckpt["model"], strict=True)
    model.eval()
    for p in model.parameters():
        p.requires_grad_(False)
    used = {k: cfg_dict[k] for k in sorted(fields & set(cfg_dict))}
    return model, cfg, {
        "checkpoint": str(run_dir / "best.pt"),
        "step": int(ckpt.get("step", -1)),
        "selection_headline": float(ckpt.get("headline", float("nan"))),
        "hyperparameters_used_to_rebuild": used,
        "config_keys_not_dataclass_fields": sorted(unknown),
        "model_config": model.config,
        "base_vlm_checkpoint": setup["checkpoint"],
        "readout": setup["readout"],
    }


# --------------------------------------------------------------------------- #
# per-sample inference
# --------------------------------------------------------------------------- #
@dataclass
class Scored:
    sample_id: str
    task_type: str
    minor: str | None
    lut_id: str
    source_image_id: str
    field_source: str
    instruction: str
    de00_mean: float
    de00_p50: float
    de00_p90: float
    de00_max: float
    soft_iou_grid: float | None
    alpha_pred_mean: float
    alpha_gt_mean: float
    grid_h: int
    grid_w: int
    img_h: int
    img_w: int
    n_field_cells_invalid: int


@torch.no_grad()
def score_one(row, *, store, bank, zc, model, predfield: Path, field_source: str,
              record: dict[str, Any], point_chunk: int,
              alpha_source: str = "pred"):
    """Everything one row contributes: the panels' tensors + its numbers.

    ``alpha_source == "pred"``: six panels, ``alpha_pred`` = the upsampled
    predicted field, grid soft-IoU computed.  ``alpha_source == "gt"``:
    five panels, ``alpha_pred := alpha_gt`` (no predfield is opened), soft-IoU
    left ``None`` -- identical alphas make it 1.0 by definition, so reporting it
    would be reporting the definition.
    """
    from q3vl.whatb import criteria as C
    from q3vl.whatb.colorimetry import delta_e00_srgb
    from q3vl.where.upsample import area_resize
    from q3vl.whereb.metrics import soft_iou_value

    img, alpha_gt = store.load(row)                       # (3,H,W), (1,H,W) | 1.0
    h, w = int(img.shape[-2]), int(img.shape[-1])

    if alpha_source == "gt":
        field = None
        invalid = None
        n_invalid = 0
        a_pred = (torch.full((1, h, w), float(alpha_gt))
                  if isinstance(alpha_gt, float) else alpha_gt)     # (1,H,W)
        grid_h, grid_w = h, w
    else:
        raw = np.load(predfield / f"{row.sample_id}.npy")
        if raw.ndim != 2:
            raise ValueError(f"{row.sample_id}: field is {raw.shape}, expected 2-d")
        field = torch.from_numpy(np.asarray(raw, dtype=np.float32))
        invalid = ~torch.isfinite(field) | (field < 0.0) | (field > 1.0)
        n_invalid = int(invalid.sum())
        field_clean = torch.where(invalid, torch.zeros_like(field), field)

        a_pred = area_resize(field_clean[None, None], (h, w))[0]        # (1,H,W)
        _CALLED["area_resize"] += 1
        grid_h, grid_w = int(field.shape[0]), int(field.shape[1])

    z = zc.vector(row.sample_id)
    f_hat = model.transform_image(z, img, point_chunk=point_chunk)
    pred_edited = C.compose_hat(img, a_pred, f_hat)
    _CALLED["compose_hat"] += 1
    gt_edited = bank.f_star_image(img, alpha_gt, row.lut_id)
    _CALLED["lut_f_star"] += 1

    de = delta_e00_srgb(pred_edited.permute(1, 2, 0), gt_edited.permute(1, 2, 0))
    _CALLED["delta_e00"] += 1
    de_flat = de.reshape(-1)

    # grid-level soft-IoU: the raw field against GT alpha area-averaged onto the
    # same grid (the where board's own `grid_soft_iou` quantity).  Under
    # alpha_source == "gt" the two arguments would be the same array, so the
    # quantity is not computed at all.
    alpha_gt_mean = (float(alpha_gt) if isinstance(alpha_gt, float)
                     else float(alpha_gt.mean()))
    if alpha_source == "gt":
        siou = None
    else:
        if isinstance(alpha_gt, float):
            gt_low = torch.full_like(field_clean, float(alpha_gt))
        else:
            gt_low = area_resize(alpha_gt[None], tuple(field_clean.shape))[0, 0]
            _CALLED["area_resize"] += 1
        siou = float(soft_iou_value(field_clean, gt_low))
        _CALLED["soft_iou"] += 1

    scored = Scored(
        sample_id=row.sample_id, task_type=row.task_type,
        minor=(str(record.get("minor")) if record.get("minor") is not None else None),
        lut_id=row.lut_id, source_image_id=row.source_image_id,
        field_source=field_source,
        instruction=str(record.get("instruction") or ""),
        de00_mean=float(de.mean()),
        de00_p50=float(de_flat.median()),
        de00_p90=float(torch.quantile(de_flat.float(), 0.90)) if de_flat.numel() < 16_000_000
        else float(np.quantile(de_flat.numpy(), 0.90)),
        de00_max=float(de.max()),
        soft_iou_grid=siou,
        alpha_pred_mean=float(a_pred.mean()),
        alpha_gt_mean=alpha_gt_mean,
        grid_h=grid_h, grid_w=grid_w,
        img_h=h, img_w=w,
        n_field_cells_invalid=n_invalid,
    )
    panels = {
        "image": img, "alpha_gt": alpha_gt, "alpha_pred": a_pred,
        "gt_edited": gt_edited, "pred_edited": pred_edited, "de00": de,
        "invalid_hi": area_resize(invalid.float()[None, None], (h, w))[0, 0]
        if n_invalid else None,
    }
    return scored, panels


# --------------------------------------------------------------------------- #
# selection
# --------------------------------------------------------------------------- #
def nearest_rank(sorted_vals: list[float], q: float) -> tuple[int, float]:
    """Nearest-rank order statistic (``q`` a fraction).  Returns (rank, value)."""
    n = len(sorted_vals)
    i = min(n - 1, max(0, int(round(q * (n - 1)))))
    return i, sorted_vals[i]


def select_by_quantile(rows: Sequence[Scored], *, n_per_bin: int
                       ) -> list[dict[str, Any]]:
    """Deterministic quantile coverage, worst cases included by construction.

    Rule (recorded verbatim in the manifest):

    1. sort the class's rows by ``de00_mean`` ascending, ties broken by
       ``sample_id`` ascending;
    2. for each quantile ``q``, the anchor rank is the **nearest-rank** order
       statistic ``i = round(q * (n - 1))``;
    3. the bin is the closed rank window ``[i - hw, i + hw]`` with
       ``hw = max(n_per_bin - 1, round(0.01 * n))`` -- a fixed +-1% rank window,
       never a value-tuned one;
    4. inside the window the rows are ordered by ``sample_id`` ascending and the
       first ``n_per_bin`` not already taken by an earlier quantile are kept.
    """
    ordered = sorted(rows, key=lambda r: (r.de00_mean, r.sample_id))
    vals = [r.de00_mean for r in ordered]
    n = len(ordered)
    hw = max(n_per_bin - 1, int(round(0.01 * n)))
    taken: set[str] = set()
    out: list[dict[str, Any]] = []
    for q in QUANTILES:
        i, v = nearest_rank(vals, q)
        lo, hi = max(0, i - hw), min(n - 1, i + hw)
        window = sorted(ordered[lo:hi + 1], key=lambda r: r.sample_id)
        picked = 0
        for r in window:
            if picked >= n_per_bin:
                break
            if r.sample_id in taken:
                continue
            taken.add(r.sample_id)
            picked += 1
            out.append({
                "quantile": q, "anchor_rank": i, "anchor_de00_mean": v,
                "window_ranks": [lo, hi], "rank_in_class": ordered.index(r),
                "row": r,
            })
        if picked < n_per_bin:
            raise RuntimeError(
                f"quantile {q}: only {picked} of {n_per_bin} rows available in "
                f"rank window [{lo},{hi}] after de-duplication")
    return out


def select_random(rows: Sequence[Scored], *, n_total: int, seed: int
                  ) -> list[dict[str, Any]]:
    """Seeded without-replacement draw over the class's whole population.

    Rule (recorded verbatim in the manifest):

    1. the class's rows are put in one canonical order: ``sample_id`` ascending
       -- the draw must not depend on the order rows happened to be scored in;
    2. ``numpy.random.default_rng(seed).permutation(n)`` is taken and its first
       ``n_total`` entries index that canonical order.  No stratification, no
       quantile anchors, no rank window: every row of the class has the same
       inclusion probability ``n_total / n``;
    3. the drawn rows are emitted in ``sample_id`` ascending order.  Each row
       still carries ``rank_in_class``, its 0-based rank when the class is
       sorted by ``de00_mean`` ascending (ties by ``sample_id``), so where the
       draw landed in the distribution is readable without a quantile label.

    Same ``seed`` and same scored population give the same ``sample_id`` set.
    """
    if n_total <= 0:
        raise RuntimeError("--select random needs --n-total > 0")
    pool = sorted(rows, key=lambda r: r.sample_id)
    n = len(pool)
    if n_total > n:
        raise RuntimeError(
            f"--n-total {n_total} exceeds the {n} scored rows of this class")
    ordered = sorted(rows, key=lambda r: (r.de00_mean, r.sample_id))
    rank_of = {r.sample_id: i for i, r in enumerate(ordered)}

    idx = np.random.default_rng(seed).permutation(n)[:n_total]
    drawn = sorted((pool[int(i)] for i in idx), key=lambda r: r.sample_id)
    if len({r.sample_id for r in drawn}) != n_total:
        raise AssertionError("the draw was not without replacement")
    return [{"select": "random", "sample_seed": seed, "n_total": n_total,
             "pool_size": n, "rank_in_class": rank_of[r.sample_id], "row": r}
            for r in drawn]


def presample_index_rows(rows, *, classes: Sequence[str], n_total: int, seed: int
                         ) -> tuple[list[Any], dict[str, Any]]:
    """Draw ``n_total`` INDEX rows per class before anything is scored.

    Same draw rule as :func:`select_random`, applied one level earlier:

    1. the class's normal-only index rows are put in ``sample_id`` ascending
       order (``local`` = ``task_type != "style"``);
    2. ``numpy.random.default_rng(seed).permutation(n)`` is taken and its first
       ``n_total`` entries index that order; every row of the class has the same
       inclusion probability ``n_total / n``;
    3. only the drawn rows are scored.

    Consequence, recorded on the manifest and on the summary figure: every
    population number the run reports (quantiles, statistics, the pooled-pixel
    p99) is a number OF THIS SUBSAMPLE, not of the split.
    """
    if n_total <= 0:
        raise RuntimeError("pre-sampling needs --n-total > 0")
    picked: list[Any] = []
    facts: dict[str, Any] = {}
    for cls in classes:
        want_style = cls == "style"
        pool = sorted((r for r in rows if (r.task_type == "style") == want_style),
                      key=lambda r: r.sample_id)
        n = len(pool)
        if n_total > n:
            raise RuntimeError(
                f"--n-total {n_total} exceeds the {n} normal-only {cls} index "
                f"rows of this split")
        idx = np.random.default_rng(seed).permutation(n)[:n_total]
        drawn = sorted((pool[int(i)] for i in idx), key=lambda r: r.sample_id)
        if len({r.sample_id for r in drawn}) != n_total:
            raise AssertionError("the pre-sample draw was not without replacement")
        facts[cls] = {"class_population": n, "n_drawn": len(drawn),
                      "sample_ids": [r.sample_id for r in drawn]}
        picked.extend(drawn)
    return picked, facts


def presample_index_rows_family_capped(
        rows, *, classes: Sequence[str], n_total: int, seed: int,
        family_cap: dict[str, int], split: str, pool_mult: int, pool_min: int
) -> tuple[list[Any], dict[str, Any], dict[str, str], dict[str, Any]]:
    """:func:`presample_index_rows` with a per-mask-family quota.

    The mask family (``radial`` / ``semantic`` / ``band`` / ``linear`` /
    ``unknown``) is **not** in the split index, the ``.rec.json`` or the
    maskview meta; it is fetched live from the construction side by
    ``q3vl.whereb.amort.data.family_labels``, which is imported, never
    re-implemented (it owns the thread-local ``MaskResolver``: sqlite objects
    may not cross threads).

    Rule (recorded verbatim in the manifest):

    1. the class's normal-only index rows are put in ``sample_id`` ascending
       order (``local`` = ``task_type != "style"``) and
       ``numpy.random.default_rng(seed).permutation(n)`` is taken -- the same
       permutation :func:`presample_index_rows` uses;
    2. its first ``n_cand = min(n, max(pool_mult * n_total, pool_min))`` entries
       are the CANDIDATE POOL.  Only the pool is family-labelled: labelling all
       42752 ``train`` local rows to keep 100 is not paid for;
    3. the pool is walked **in permutation order** (a uniform random order, so
       every step below is a seeded draw, not a re-ranking).  Pass 1 keeps, for
       each family named in ``--family-cap``, its first ``min(cap, available)``
       rows.  Pass 2 fills the remaining ``n_total - kept`` slots from the rows
       whose family is NOT capped, again in permutation order -- so the
       uncapped families keep their own natural proportions relative to each
       other, no per-family balancing is applied to them;
    4. ``unknown`` is an ordinary family: capped if named, otherwise it takes
       part in pass 2.  It is counted separately in the manifest;
    5. if the pool cannot fill ``n_total`` the run **raises**: a short draw is
       never emitted silently;
    6. the drawn rows are scored and rendered; the emission order is
       ``sample_id`` ascending.
    """
    if n_total <= 0:
        raise RuntimeError("pre-sampling needs --n-total > 0")

    from q3vl.whereb.amort.data import family_labels
    from q3vl.whereb.data import open_dataset

    ds, ds_info = open_dataset(split, need_mask=False)
    pos = {r["sample_id"]: i for i, r in enumerate(ds.meta_rows())}

    picked: list[Any] = []
    facts: dict[str, Any] = {}
    family_by_id: dict[str, str] = {}
    for cls in classes:
        want_style = cls == "style"
        pool = sorted((r for r in rows if (r.task_type == "style") == want_style),
                      key=lambda r: r.sample_id)
        n = len(pool)
        if n_total > n:
            raise RuntimeError(
                f"--n-total {n_total} exceeds the {n} normal-only {cls} index "
                f"rows of this split")
        n_cand = min(n, max(pool_mult * n_total, pool_min))
        perm = np.random.default_rng(seed).permutation(n)
        cand = [pool[int(i)] for i in perm[:n_cand]]        # permutation order

        absent = [r.sample_id for r in cand if r.sample_id not in pos]
        if absent:
            raise RuntimeError(
                f"{len(absent)} candidate sample_id(s) of class {cls} are not in "
                f"the whereb {split} index, so no family label can be fetched "
                f"for them (first: {absent[:3]})")
        fam = family_labels(ds, [pos[r.sample_id] for r in cand])
        if len(fam) != len(cand):
            raise AssertionError(
                f"family_labels returned {len(fam)} labels for {len(cand)} "
                f"candidates")
        family_by_id.update(fam)

        cand_counts: dict[str, int] = {}
        for r in cand:
            f = fam[r.sample_id]
            cand_counts[f] = cand_counts.get(f, 0) + 1

        kept: list[Any] = []
        counts: dict[str, int] = {}
        for r in cand:                                   # pass 1: capped only
            f = fam[r.sample_id]
            if f not in family_cap:
                continue
            if counts.get(f, 0) >= family_cap[f]:
                continue
            counts[f] = counts.get(f, 0) + 1
            kept.append(r)
        for r in cand:                                   # pass 2: uncapped fill
            if len(kept) >= n_total:
                break
            f = fam[r.sample_id]
            if f in family_cap:
                continue
            counts[f] = counts.get(f, 0) + 1
            kept.append(r)
        if len(kept) < n_total:
            raise RuntimeError(
                f"class {cls}: the family-capped draw could only fill "
                f"{len(kept)} of {n_total} rows from a candidate pool of "
                f"{n_cand} (pool families {cand_counts}, caps {family_cap}, "
                f"kept {counts}).  Raise --family-cap-pool-mult / "
                f"--family-cap-pool-min, or raise the caps; a short draw is not "
                f"emitted silently")
        for f, cap in family_cap.items():                # runtime assertion
            if counts.get(f, 0) > cap:
                raise AssertionError(
                    f"class {cls}: family {f} kept {counts[f]} rows against a "
                    f"cap of {cap}; the quota was defined but not honoured")

        drawn = sorted(kept, key=lambda r: r.sample_id)
        if len({r.sample_id for r in drawn}) != n_total:
            raise AssertionError("the pre-sample draw was not without replacement")
        facts[cls] = {
            "class_population": n,
            "candidate_pool_size": n_cand,
            "candidate_pool_rule": (f"min({n}, max({pool_mult} * {n_total}, "
                                    f"{pool_min})) first entries of "
                                    f"default_rng({seed}).permutation({n})"),
            "candidate_family_counts": dict(sorted(cand_counts.items())),
            "n_drawn": len(drawn),
            "final_family_counts": dict(sorted(counts.items())),
            "n_unknown_family_drawn": counts.get("unknown", 0),
            "n_unknown_family_in_pool": cand_counts.get("unknown", 0),
            "family_by_sample_id": {r.sample_id: fam[r.sample_id] for r in drawn},
            "sample_ids": [r.sample_id for r in drawn],
        }
        picked.extend(drawn)
    return picked, facts, family_by_id, ds_info


def fetch_family_labels(rows, *, split: str) -> tuple[dict[str, str], dict[str, Any]]:
    """``sample_id -> family`` for exactly ``rows``, with no effect on the draw.

    The same label source :func:`presample_index_rows_family_capped` uses --
    ``q3vl.whereb.amort.data.family_labels``, imported and never
    re-implemented, because it owns the thread-local ``MaskResolver`` (sqlite
    objects may not cross threads).  Called only under ``--family-label``: a
    run without the flag fetches nothing and draws the figures it drew before.
    """
    from q3vl.whereb.amort.data import family_labels
    from q3vl.whereb.data import open_dataset

    ds, ds_info = open_dataset(split, need_mask=False)
    pos = {r["sample_id"]: i for i, r in enumerate(ds.meta_rows())}
    absent = [r.sample_id for r in rows if r.sample_id not in pos]
    if absent:
        raise RuntimeError(
            f"{len(absent)} drawn sample_id(s) are not in the whereb {split} "
            f"index, so no family label can be fetched for them "
            f"(first: {absent[:3]})")
    fam = family_labels(ds, [pos[r.sample_id] for r in rows])
    if len(fam) != len(rows):
        raise AssertionError(
            f"family_labels returned {len(fam)} labels for {len(rows)} rows")
    unknown = {v for v in fam.values()} - set(FAMILY_VALUES)
    if unknown:
        raise AssertionError(f"family_labels returned labels outside the "
                             f"documented range: {sorted(unknown)}")
    return fam, ds_info


# --------------------------------------------------------------------------- #
# rendering
# --------------------------------------------------------------------------- #
#: filled by :func:`_setup_fonts` -- the explicit family lists the text calls use
_FONT_SANS: list[str] = ["DejaVu Sans"]
_FONT_MONO: list[str] = ["DejaVu Sans Mono"]


def _setup_fonts() -> list[str]:
    """Append the installed CJK face to the two font stacks.

    Instructions and ``minor`` labels are Chinese; without a fallback face every
    glyph renders as a tofu box and the text bar carries no information.  The
    fallback only takes effect when the family list is passed to the text call
    itself -- the ``font.monospace`` alias list is not a fallback chain.
    """
    import matplotlib
    import matplotlib.font_manager as fm

    global _FONT_SANS, _FONT_MONO
    have = {f.name for f in fm.fontManager.ttflist}
    cjk = [n for n in ("Noto Sans CJK JP", "Noto Sans CJK SC",
                       "WenQuanYi Micro Hei", "WenQuanYi Zen Hei") if n in have]
    _FONT_SANS = ["DejaVu Sans"] + cjk
    _FONT_MONO = ["DejaVu Sans Mono"] + cjk
    matplotlib.rcParams["font.family"] = _FONT_SANS
    matplotlib.rcParams["axes.unicode_minus"] = False
    return cjk


def _alpha_rgba(field_hi: torch.Tensor, invalid_hi: torch.Tensor | None):
    """Fixed [0,1] viridis mapping; invalid cells go white."""
    import matplotlib

    a = np.clip(field_hi.detach().cpu().numpy(), 0.0, 1.0)
    rgba = matplotlib.colormaps["viridis"](a)
    if invalid_hi is not None:
        m = invalid_hi.detach().cpu().numpy() > 0.5
        rgba[m] = (1.0, 1.0, 1.0, 1.0)
    return rgba


def render_sample(*, out_png: Path, scored: Scored, panels: dict[str, Any],
                  pick: dict[str, Any], vmin: float, vmax: float, dpi: int,
                  alpha_source: str = "pred",
                  vmax_note: str = "vmax = p99 of pooled pixels",
                  family: str | None = None) -> None:
    import matplotlib

    matplotlib.use("Agg")
    _setup_fonts()
    import matplotlib.pyplot as plt
    from matplotlib.patches import FancyBboxPatch

    img = panels["image"].detach().cpu().numpy().transpose(1, 2, 0)
    gt_ed = panels["gt_edited"].detach().cpu().numpy().transpose(1, 2, 0)
    pr_ed = panels["pred_edited"].detach().cpu().numpy().transpose(1, 2, 0)
    de = panels["de00"].detach().cpu().numpy()
    inv = panels["invalid_hi"]

    a_gt = panels["alpha_gt"]
    if isinstance(a_gt, float):
        a_gt_hi = torch.full((scored.img_h, scored.img_w), float(a_gt))
    else:
        a_gt_hi = a_gt[0]
    a_pred_hi = panels["alpha_pred"][0]

    # the two layouts differ by ONE panel: with alpha_source == "gt" there is no
    # predicted alpha to show, so the "pred alpha" column is absent (it would be
    # a copy of the GT alpha column)
    if alpha_source == "gt":
        kinds = ["input", "alpha_gt", "gt_edited", "pred_edited", "de00"]
    else:
        kinds = ["input", "alpha_gt", "alpha_pred", "gt_edited", "pred_edited",
                 "de00"]
    n_col = len(kinds)

    h, w = scored.img_h, scored.img_w
    panel_h = 3.4                                    # inches, image row height
    panel_w = panel_h * (w / h)
    fig_w = n_col * panel_w + 2.1
    # the gt bar carries one row more (the alpha_source statement is a line of
    # its own so it cannot be pushed off the right edge of a portrait figure)
    bar_h = 1.9 if alpha_source == "gt" else 1.55
    fig = plt.figure(figsize=(fig_w, panel_h + 0.35 + bar_h),
                     constrained_layout=False)
    gs = fig.add_gridspec(2, n_col + 1, height_ratios=(panel_h, bar_h),
                          width_ratios=(1,) * n_col + (0.075,),
                          left=0.012, right=0.955, top=0.90, bottom=0.045,
                          wspace=0.035, hspace=0.14)

    title_of = {
        "input": "Input",
        "alpha_gt": ("GT alpha  [alpha = GT]" if alpha_source == "gt"
                     else "GT alpha"),
        "alpha_pred": ("pred alpha (style: definition 1)"
                       if scored.field_source == "style_definition_ones"
                       else "pred alpha (ST_LANG)"),
        "gt_edited": "GT edited",
        "pred_edited": "pred edited",
        "de00": "|pred-GT| dE00",
    }

    for col, kind in enumerate(kinds):
        ax = fig.add_subplot(gs[0, col])
        ax.set_axis_off()
        ax.set_title(title_of[kind], fontsize=11, pad=6)
        if kind == "input":
            ax.imshow(np.clip(img, 0, 1))
            sub = f"{h}x{w}  short side 512"
        elif kind == "alpha_gt":
            ax.imshow(_alpha_rgba(a_gt_hi, None))
            # under alpha_source == "gt" the panel is both alphas at once; that
            # is said in the panel title and on its own line of the text bar,
            # not in this caption, which has only one panel's width and would
            # run under the next panel on a portrait sample
            sub = f"alpha_mean={scored.alpha_gt_mean:.3f}  scale [0,1]"
        elif kind == "alpha_pred":
            ax.imshow(_alpha_rgba(a_pred_hi, inv))
            sub = (f"alpha_mean={scored.alpha_pred_mean:.3f}  "
                   f"grid {scored.grid_h}x{scored.grid_w}  scale [0,1]")
            if scored.n_field_cells_invalid:
                sub += f"  invalid cells={scored.n_field_cells_invalid} (white)"
        elif kind == "gt_edited":
            ax.imshow(np.clip(gt_ed, 0, 1))
            sub = f"lut {scored.lut_id}"
        elif kind == "pred_edited":
            ax.imshow(np.clip(pr_ed, 0, 1))
            sub = "f_hat = E031_MLP_NOL8_L3"
            if alpha_source == "gt":
                sub += "  alpha = GT"
        else:
            im = ax.imshow(de, cmap="magma", vmin=vmin, vmax=vmax)
            sub = (f"mean={scored.de00_mean:.3f}  p50={scored.de00_p50:.3f}  "
                   f"max={scored.de00_max:.3f}")
            cax = fig.add_subplot(gs[0, n_col])
            cb = fig.colorbar(im, cax=cax)
            # the label is vertical text: its length runs along the figure
            # height, so a note longer than the default one is put on a second
            # line instead of running off the top and bottom of the figure
            head = f"dE00, global scale [{vmin:.2f}, {vmax:.2f}] "
            cb.set_label(f"{head}({vmax_note})" if len(vmax_note) <= 32
                         else f"{head}\n({vmax_note})", fontsize=8)
            cb.ax.tick_params(labelsize=7)
        ax.text(0.015, 0.015, sub, transform=ax.transAxes, fontsize=8,
                color="white", va="bottom", ha="left",
                bbox={"facecolor": "#111827", "alpha": 0.75, "pad": 2.2,
                      "edgecolor": "none"})

    tax = fig.add_subplot(gs[1, :])
    tax.set_axis_off()
    tax.add_patch(FancyBboxPatch((0.002, 0.02), 0.996, 0.96,
                                 transform=tax.transAxes,
                                 boxstyle="round,pad=0.008,rounding_size=0.004",
                                 facecolor="#f4f7f9", edgecolor="#334e5c",
                                 linewidth=1.2))
    siou = ("n/a" if scored.soft_iou_grid is None
            else f"{scored.soft_iou_grid:.4f}")
    line1 = (f"sample_id={scored.sample_id}   task_type={scored.task_type}   "
             f"minor={scored.minor}   lut_id={scored.lut_id}   "
             f"field_source={scored.field_source}")
    # no quantile label is invented for a random draw: the row simply has none
    sel = (f"bin=p{int(pick['quantile'] * 100):02d}" if "quantile" in pick
           else f"select=random(seed={pick['sample_seed']}, "
                f"n={pick['n_total']}/{pick['pool_size']})")
    if alpha_source == "gt":
        # no soft-IoU column: alpha_pred IS alpha_gt here, so the value is 1.0
        # by definition and printing it would be printing the definition
        line_alpha = ("alpha = GT: alpha_pred := alpha_GT, this split has no "
                      "where predicted field   |   soft-IoU not computed "
                      "(alpha_pred == alpha_GT makes it 1.0 by definition)")
        line2 = (f"dE00 mean={scored.de00_mean:.4f}   "
                 f"alpha_mean(GT)={scored.alpha_gt_mean:.4f}   "
                 f"{sel}   "
                 f"rank_in_class={pick['rank_in_class']}")
    else:
        line_alpha = None
        line2 = (f"dE00 mean={scored.de00_mean:.4f}   "
                 f"soft-IoU(grid, pred alpha vs GT alpha, min/max)={siou}   "
                 f"alpha_mean(pred)={scored.alpha_pred_mean:.4f}   "
                 f"alpha_mean(GT)={scored.alpha_gt_mean:.4f}   "
                 f"{sel}   "
                 f"rank_in_class={pick['rank_in_class']}")
    # the family label goes at the FRONT of the shorter of the two data lines:
    # line1 already runs past the right edge of a portrait figure, so anything
    # appended to it is clipped.  Only printed when a label was actually
    # fetched, so a run without --family-cap draws the byte-identical figure it
    # drew before.
    if family is not None:
        line2 = f"mask_family={family}   " + line2
    # the instruction is wrapped to the figure's own width (two lines, then
    # truncated) so it never runs past the text box on a portrait sample
    per_line = max(60, int(fig_w * 15.0))
    flat = " ".join(scored.instruction.split())
    lines = textwrap.wrap(f"instruction: {flat}", width=per_line)[:2]
    if len(" ".join(lines)) < len(f"instruction: {flat}") and lines:
        lines[-1] = lines[-1][:max(0, per_line - 4)] + " ..."
    if line_alpha is None:
        y1, y2, y_instr = 0.90, 0.635, 0.375
    else:
        y1, y_alpha, y2, y_instr = 0.93, 0.745, 0.56, 0.335
        tax.text(0.012, y_alpha, line_alpha, transform=tax.transAxes, va="top",
                 fontsize=9.5, color="#7a2f12", family=_FONT_MONO)
    tax.text(0.012, y1, line1, transform=tax.transAxes, va="top", fontsize=9.5,
             fontweight="bold", color="#16242b", family=_FONT_MONO)
    tax.text(0.012, y2, line2, transform=tax.transAxes, va="top", fontsize=9.5,
             color="#16242b", family=_FONT_MONO)
    tax.text(0.012, y_instr, "\n".join(lines), transform=tax.transAxes,
             va="top", fontsize=9, color="#16242b", family=_FONT_SANS,
             linespacing=1.35)

    fig.savefig(out_png, dpi=dpi, facecolor="white")
    plt.close(fig)


def render_summary(*, out_png: Path, rows: Sequence[Scored],
                   quantiles: dict[str, dict[str, float]], dpi: int,
                   population: str = "V_what normal-only") -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    allv = np.array([r.de00_mean for r in rows], dtype=np.float64)
    locv = np.array([r.de00_mean for r in rows if r.task_type != "style"],
                    dtype=np.float64)
    styv = np.array([r.de00_mean for r in rows if r.task_type == "style"],
                    dtype=np.float64)
    hi = float(allv.max())
    bins = np.linspace(0.0, hi * 1.02 + 1e-6, 61)

    fig, axes = plt.subplots(1, 2, figsize=(15.5, 5.0))
    ax = axes[0]
    ax.hist(allv, bins=bins, color="#3f6f8f", edgecolor="white", linewidth=0.5)
    for q, v in quantiles.get("all", {}).items():
        ax.axvline(v, color="#b23a48", linewidth=1.2, linestyle="--")
        ax.text(v, ax.get_ylim()[1] * 0.97, f" {q} = {v:.2f}", rotation=90,
                va="top", ha="left", fontsize=8, color="#b23a48")
    # the population label can be a sentence (split + alpha source + the
    # subsample warning); wrapped so it cannot run over the next axes' title
    ax.set_title("\n".join(textwrap.wrap(
        f"per-sample dE00 mean, {population} (n={len(allv)})", width=66)),
        fontsize=12)
    ax.set_xlabel("dE00 mean of |pred edited - GT edited|")
    ax.set_ylabel("samples")

    ax = axes[1]
    for v, name, color in ((locv, "local", "#2d6a4f"), (styv, "style", "#7b508f")):
        if v.size == 0:                       # a class the run did not score
            continue
        ax.hist(v, bins=bins, histtype="step", linewidth=1.8, color=color,
                label=f"{name} (n={len(v)})")
        for q in QUANTILES:
            qq = quantiles.get(name, {}).get(f"p{int(q * 100):02d}")
            if qq is None:
                continue
            ax.axvline(qq, color=color, linewidth=0.9, linestyle=":", alpha=0.8)
    ax.legend(fontsize=10)
    ax.set_title("local vs style, same bins; dotted lines = "
                 "p05/p25/p50/p75/p95 of each class", fontsize=12)
    ax.set_xlabel("dE00 mean of |pred edited - GT edited|")
    ax.set_ylabel("samples")

    fig.tight_layout()
    fig.savefig(out_png, dpi=dpi, facecolor="white")
    plt.close(fig)


# --------------------------------------------------------------------------- #
# statistics
# --------------------------------------------------------------------------- #
def stats_of(vals: Sequence[float]) -> dict[str, Any]:
    if not vals:
        return {"n": 0}
    a = np.asarray(sorted(vals), dtype=np.float64)
    out: dict[str, Any] = {"n": int(a.size), "mean": float(a.mean()),
                           "min": float(a[0]), "max": float(a[-1])}
    for q in (0.05, 0.25, 0.50, 0.75, 0.90, 0.95, 0.99):
        i = min(a.size - 1, max(0, int(round(q * (a.size - 1)))))
        out[f"p{int(q * 100):02d}"] = float(a[i])
    return out


def hist_percentile(counts: np.ndarray, edges: np.ndarray, q: float) -> float:
    """Percentile of a pooled distribution held as a histogram (upper edge)."""
    total = counts.sum()
    if total <= 0:
        return float("nan")
    cum = np.cumsum(counts)
    i = int(np.searchsorted(cum, q * total, side="left"))
    i = min(i, len(edges) - 2)
    return float(edges[i + 1])


# --------------------------------------------------------------------------- #
# main
# --------------------------------------------------------------------------- #
def main(argv: Sequence[str] | None = None) -> int:
    t_start = time.time()
    args = parse_args(argv)
    if os.environ.get("CUDA_VISIBLE_DEVICES", "") != "":
        raise RuntimeError("this tool is CPU-only; CUDA_VISIBLE_DEVICES must be empty")
    # argument coherence is checked before the ~6-minute scoring pass, not after
    if args.select == "random" and args.n_total <= 0:
        raise SystemExit("--select random requires --n-total > 0")
    if args.select == "quantile" and (args.n_total or args.sample_seed):
        print("note: --select quantile ignores --n-total / --sample-seed; the "
              "manifest records them as unused", flush=True)
    if str(args.de00_vmax) != "auto":
        try:
            forced_vmax: float | None = float(args.de00_vmax)
        except ValueError:
            raise SystemExit("--de00-vmax takes 'auto' or a float")
    else:
        forced_vmax = None
    presample = (args.presample == "auto" and args.select == "random"
                 and args.n_total > 0)
    family_cap = parse_family_cap(args.family_cap)
    if family_cap and not presample:
        raise SystemExit(
            "--family-cap applies to the pre-sample draw, so it needs "
            "--select random, --n-total > 0 and --presample auto")
    if family_cap and (args.family_cap_pool_mult < 1 or args.family_cap_pool_min < 1):
        raise SystemExit("--family-cap-pool-mult / --family-cap-pool-min must be >= 1")

    from q3vl.whatb import splits as S
    from q3vl.whatb.evaldata import SampleStore
    from q3vl.whatb.lutdata import LutBank
    from q3vl.whatb.zcache import ZCache

    out_dir: Path = args.out
    out_dir.mkdir(parents=True, exist_ok=True)

    # -- inputs ------------------------------------------------------------- #
    model, cfg, model_facts = load_what_model(args.what_run)
    zc = ZCache(args.zcache)
    zfacts = zc.assert_belongs_to(checkpoint=model_facts["base_vlm_checkpoint"],
                                  readout_kind=cfg.readout,
                                  context_source=cfg.context, control_tag="none")
    pf_manifest: dict[str, Any] = {}
    field_source: dict[str, str] = {}
    if args.alpha_source == "pred":
        pf_manifest = json.loads(
            (args.predfield / "manifest.json").read_text(encoding="utf-8"))
        with (args.predfield / "per_sample.jsonl").open(encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                if line:
                    r = json.loads(line)
                    field_source[str(r["sample_id"])] = str(r["source"])

    store = SampleStore(args.split)
    bank = LutBank(args.bank_dir)

    render_classes = (("local", "style") if args.classes == "both"
                      else (args.classes,))

    # the index口径 is selected before the index is read
    ver = S.use_dataset_version(args.dataset_version)
    rows = S.normal_only(S.load_index(args.split))
    exclude_facts: dict[str, Any] | None = None
    excluded_ids: set[str] = set()
    if args.exclude_ids is not None:
        excluded_ids = {tok for tok in
                        args.exclude_ids.read_text(encoding="utf-8").split()
                        if tok}
        n_pool_before = len(rows)
        rows = [r for r in rows if r.sample_id not in excluded_ids]
        exclude_facts = {
            "file": str(args.exclude_ids),
            "file_sha256": hashlib.sha256(
                args.exclude_ids.read_bytes()).hexdigest(),
            "n_ids_in_file": len(excluded_ids),
            "pool_before": n_pool_before,
            "pool_after": len(rows),
            "n_removed_from_pool": n_pool_before - len(rows),
            "stage": "applied to the split's normal-only index rows BEFORE the "
                     "pre-sample draw / the quantile walk, so the draw is the "
                     "draw over the filtered population",
        }
        print(f"--exclude-ids: {len(excluded_ids)} id(s) listed; "
              f"{args.split} normal-only pool {n_pool_before} -> {len(rows)}",
              flush=True)
        if not rows:
            raise RuntimeError("--exclude-ids removed the entire sampling pool")
    if args.limit:
        rows = rows[:args.limit]
    presample_facts: dict[str, Any] | None = None
    family_by_id: dict[str, str] = {}
    family_ds_info: dict[str, Any] | None = None
    if presample and family_cap:
        rows, presample_facts, family_by_id, family_ds_info = (
            presample_index_rows_family_capped(
                rows, classes=render_classes, n_total=args.n_total,
                seed=args.sample_seed, family_cap=family_cap, split=args.split,
                pool_mult=args.family_cap_pool_mult,
                pool_min=args.family_cap_pool_min))
        print(f"pre-sampled {len(rows)} index rows with family caps "
              f"{family_cap} ({args.n_total} per class of "
              f"{list(render_classes)}, seed={args.sample_seed}); "
              f"final families "
              f"{ {c: f['final_family_counts'] for c, f in presample_facts.items()} }",
              flush=True)
    elif presample:
        # the whole point: `train` normal-only local is 42752 rows, and scoring
        # all of them to then draw 100 costs 427x what the 100 figures cost
        rows, presample_facts = presample_index_rows(
            rows, classes=render_classes, n_total=args.n_total,
            seed=args.sample_seed)
        print(f"pre-sampled {len(rows)} index rows "
              f"({args.n_total} per class of {list(render_classes)}, "
              f"seed={args.sample_seed}) out of the split's normal-only "
              f"population; only these rows are scored", flush=True)

    # runtime assertion: the exclusion list is not merely read, it landed on the
    # rows that will actually be rendered.
    if excluded_ids:
        leaked = sorted({r.sample_id for r in rows} & excluded_ids)
        if leaked:
            raise AssertionError(
                f"{len(leaked)} excluded sample_id(s) survived into the drawn "
                f"set (first: {leaked[:3]}); --exclude-ids was defined but not "
                f"honoured")
    if args.family_label and not family_cap:
        family_by_id, family_ds_info = fetch_family_labels(rows, split=args.split)
        got = Counter(family_by_id.values())
        print(f"--family-label: fetched {len(family_by_id)} labels "
              f"{dict(sorted(got.items()))}", flush=True)

    records = {}
    with_rec = [r for r in rows if r.has_record]
    for r, rec in zip(with_rec, S.iter_records(with_rec)):
        records[r.sample_id] = rec

    # -- score every row ---------------------------------------------------- #
    scored: list[Scored] = []
    skipped: list[dict[str, str]] = []
    hist = np.zeros(HIST_BINS, dtype=np.int64)
    edges = np.linspace(0.0, HIST_HI, HIST_BINS + 1)
    row_by_id = {r.sample_id: r for r in rows}
    for i, row in enumerate(rows):
        if args.alpha_source == "gt":
            src = "gt_alpha"
        else:
            src = field_source.get(row.sample_id)
            if src is None:
                skipped.append({"sample_id": row.sample_id,
                                "reason": "no field source row"})
                continue
            if not (args.predfield / f"{row.sample_id}.npy").is_file():
                skipped.append({"sample_id": row.sample_id,
                                "reason": "missing .npy field"})
                continue
        if row.sample_id not in zc:
            skipped.append({"sample_id": row.sample_id, "reason": "missing z vector"})
            continue
        try:
            s, panels = score_one(row, store=store, bank=bank, zc=zc, model=model,
                                  predfield=args.predfield, field_source=src,
                                  record=records.get(row.sample_id, {}),
                                  point_chunk=args.point_chunk,
                                  alpha_source=args.alpha_source)
        except (KeyError, IOError, OSError, ValueError) as exc:
            skipped.append({"sample_id": row.sample_id,
                            "reason": f"{type(exc).__name__}: {exc}"})
            continue
        scored.append(s)
        de = panels["de00"].reshape(-1).numpy()
        hist += np.histogram(de, bins=edges)[0]
        del panels                       # 567 x ~40 MB of panels does not fit
        if (i + 1) % 50 == 0:
            print(f"scored {len(scored)}/{i + 1} rows "
                  f"({time.time() - t_start:.1f}s)", flush=True)

    if not scored:
        raise RuntimeError("no rows scored")

    # the witness list is mode-dependent: with alpha_source == "gt" there is no
    # field to upsample and no soft-IoU to take, so requiring those two calls
    # would be requiring a call the mode does not make
    required_witnesses = (["delta_e00", "compose_hat", "lut_f_star"]
                          if args.alpha_source == "gt"
                          else sorted(_CALLED))
    for name in required_witnesses:
        if _CALLED[name] == 0:
            raise AssertionError(
                f"the {name} criterion function was never called; a metric that "
                "is defined but not wired is the failure this assertion exists for")

    # -- global colour scale ------------------------------------------------ #
    vmin = 0.0
    batch_pooled_p99 = hist_percentile(hist, edges, 0.99)
    if forced_vmax is not None:
        vmax = forced_vmax
        vmax_source = "explicit --de00-vmax"
        vmax_note = (f"vmax forced by --de00-vmax; this batch's pooled p99 = "
                     f"{batch_pooled_p99:.2f}")
    elif presample:
        vmax = REFERENCE_VMAX
        vmax_source = f"borrowed from the V_what full-population run: {REFERENCE_VMAX_SOURCE}"
        vmax_note = (f"vmax borrowed from V_what run; "
                     f"this batch pooled p99 = {batch_pooled_p99:.2f}")
    else:
        vmax = batch_pooled_p99
        vmax_source = "this run's own pooled-pixel p99"
        vmax_note = "vmax = p99 of pooled pixels"
    n_over = int(hist[np.searchsorted(edges, vmax, side="left"):].sum())

    # -- selection ---------------------------------------------------------- #
    by_class = {"local": [s for s in scored if s.task_type != "style"],
                "style": [s for s in scored if s.task_type == "style"]}
    picks: list[tuple[str, dict[str, Any]]] = []
    n_total_clamped: dict[str, int] = {}
    for cls in render_classes:
        if args.select == "random":
            n_draw = args.n_total
            if presample and n_draw > len(by_class[cls]):
                # a pre-drawn row that got skipped (missing z, unreadable member)
                # shrinks the pool below --n-total; the shortfall is recorded
                n_total_clamped[cls] = len(by_class[cls])
                n_draw = len(by_class[cls])
            chosen = select_random(by_class[cls], n_total=n_draw,
                                   seed=args.sample_seed)
        else:
            chosen = select_by_quantile(by_class[cls], n_per_bin=args.n_per_bin)
        for p in chosen:
            picks.append((cls, p))

    quantile_table = {
        name: {f"p{int(q * 100):02d}":
               nearest_rank(sorted(s.de00_mean for s in rowset), q)[1]
               for q in QUANTILES}
        for name, rowset in (("all", scored), ("local", by_class["local"]),
                             ("style", by_class["style"]))
        if rowset
    }

    # -- render ------------------------------------------------------------- #
    figures: list[dict[str, Any]] = []
    for cls, pick in picks:
        s: Scored = pick["row"]
        # the panels are recomputed for the selected rows only: keeping all 567
        # sets of six full-resolution tensors alive does not fit in memory, and
        # the second pass is the same deterministic forward on the same inputs
        s2, panels = score_one(row_by_id[s.sample_id], store=store, bank=bank,
                               zc=zc, model=model, predfield=args.predfield,
                               field_source=s.field_source,
                               record=records.get(s.sample_id, {}),
                               point_chunk=args.point_chunk,
                               alpha_source=args.alpha_source)
        if abs(s2.de00_mean - s.de00_mean) > 1e-6:
            raise AssertionError(
                f"{s.sample_id}: re-inference gave dE00 mean {s2.de00_mean} vs "
                f"{s.de00_mean} in the scoring pass")
        if "quantile" in pick:
            png = out_dir / (f"joint_{cls}_p{int(pick['quantile'] * 100):02d}_"
                             f"{s.sample_id[-12:]}.png")
        else:
            png = out_dir / f"joint_{cls}_rnd_{s.sample_id[-12:]}.png"
        render_sample(out_png=png, scored=s, panels=panels,
                      pick=pick, vmin=vmin, vmax=vmax, dpi=args.dpi,
                      alpha_source=args.alpha_source, vmax_note=vmax_note,
                      family=family_by_id.get(s.sample_id))
        entry = {"figure": png.name, "class": cls}
        if family_by_id:
            entry["mask_family"] = family_by_id.get(s.sample_id)
        if "quantile" in pick:
            entry.update({
                "quantile": pick["quantile"],
                "anchor_de00_mean": pick["anchor_de00_mean"],
                "anchor_rank": pick["anchor_rank"],
                "window_ranks": pick["window_ranks"],
            })
        else:
            entry.update({"select": "random", "sample_seed": pick["sample_seed"],
                          "pool_size": pick["pool_size"]})
        figures.append({
            **entry,
            "rank_in_class": pick["rank_in_class"],
            "sample_id": s.sample_id, "task_type": s.task_type,
            "minor": s.minor, "lut_id": s.lut_id,
            "field_source": s.field_source,
            "de00_mean": s.de00_mean, "soft_iou_grid": s.soft_iou_grid,
            "alpha_pred_mean": s.alpha_pred_mean,
            "alpha_gt_mean": s.alpha_gt_mean,
        })
        print(f"wrote {png.name}", flush=True)

    summary_png = out_dir / "summary_de00_distribution.png"
    population_label = f"{args.split} normal-only"
    if args.alpha_source == "gt":
        population_label += ", alpha = GT"
    if presample:
        population_label += (f", RANDOM SUBSAMPLE {args.n_total}/class "
                             f"(seed {args.sample_seed}), not the whole split")
    render_summary(out_png=summary_png, rows=scored, quantiles=quantile_table,
                   dpi=args.dpi, population=population_label)
    print(f"wrote {summary_png.name}", flush=True)

    # -- per-sample jsonl --------------------------------------------------- #
    with (out_dir / "per_sample.jsonl").open("w", encoding="utf-8") as fh:
        for s in sorted(scored, key=lambda r: r.sample_id):
            fh.write(json.dumps(s.__dict__, ensure_ascii=False) + "\n")

    # -- manifest ----------------------------------------------------------- #
    stlang_rows = [s for s in scored if s.field_source == "stlang_m_low"]
    ones_rows = [s for s in scored if s.field_source == "style_definition_ones"]
    inputs = {
        "what_best_pt": str(args.what_run / "best.pt"),
        "what_run_setup": str(args.what_run / "run_setup.json"),
    }
    if args.alpha_source == "pred":
        inputs["predfield_manifest"] = str(args.predfield / "manifest.json")
        inputs["predfield_per_sample"] = str(args.predfield / "per_sample.jsonl")
    inputs.update({
        "zcache": str(args.zcache),
        "lut_bank_meta": str(args.bank_dir / "luts_meta.json"),
        "script": str(Path(__file__).resolve()),
    })
    if not args.no_bank_hash:
        inputs["lut_bank_npz"] = str(args.bank_dir / "luts.npz")
    sha = {}
    for k, p in inputs.items():
        print(f"sha256 {k} ...", flush=True)
        sha[k] = sha256_file(Path(p))

    manifest = {
        "product": ("what(E031_MLP_NOL8_L3) rendering with alpha = GT"
                    if args.alpha_source == "gt" else
                    "joint where(ST_LANG field) + what(E031_MLP_NOL8_L3) rendering"),
        "generated_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "split": args.split,
        "population": "winner_confidence == normal",
        # which sft2seg index口径 that population was read from: name + root +
        # that口径's measured per-split n / normal_n + the exclusion sha256
        "dataset_version": ver.name,
        "dataset_root": str(ver.root),
        "dataset_version_facts": ver.facts(),
        "alpha_source": {
            "flag": args.alpha_source,
            "meaning": ("alpha_pred := alpha_gt; no where predicted field is "
                        "read, because this split has none on disk.  The "
                        "figures carry five panels (Input | GT alpha | GT "
                        "edited | pred edited | |pred-GT| dE00) and say "
                        "'alpha = GT' on the alpha panel and in the text bar; "
                        "the dE00 therefore varies with f_hat alone, the alpha "
                        "being identical on both sides"
                        if args.alpha_source == "gt" else
                        "alpha_pred = the --predfield field upsampled with "
                        "area_resize; six panels; grid soft-IoU reported"),
            "soft_iou_reported": args.alpha_source != "gt",
            "soft_iou_omission_reason":
                ("alpha_pred and alpha_gt are the same array, so soft-IoU is "
                 "1.0 by definition; it is neither computed nor printed"
                 if args.alpha_source == "gt" else None),
        },
        "sampling_scope": {
            "presample": presample,
            "flag": args.presample,
            "note": (f"the scoring pass covered a random subsample of "
                     f"{args.n_total} row(s) per rendered class drawn from the "
                     f"split's normal-only index rows with seed "
                     f"{args.sample_seed}; EVERY statistic, quantile and pooled "
                     f"percentile in this manifest is a statistic OF THAT "
                     f"SUBSAMPLE, not of the {args.split} split"
                     if presample else
                     "the scoring pass covered the whole normal-only population "
                     "of the split (both classes)"),
            "draw_rule": ((presample_index_rows_family_capped.__doc__
                           if family_cap else presample_index_rows.__doc__)
                          if presample else None),
            "per_class": presample_facts,
        },
        "family_sampling": ({
            "family_cap": dict(sorted(family_cap.items())),
            "family_cap_flag": args.family_cap,
            "label_source": "q3vl.whereb.amort.data.family_labels (imported, "
                            "not re-implemented): the family survives only on "
                            "the construction side, so it is fetched live "
                            "through q3vl.where.maskdata.MaskResolver with one "
                            "resolver per thread",
            "label_range": list(FAMILY_VALUES),
            "whereb_dataset": family_ds_info,
            "pool_mult": args.family_cap_pool_mult,
            "pool_min": args.family_cap_pool_min,
            "candidate_pool_size": {c: f["candidate_pool_size"]
                                    for c, f in (presample_facts or {}).items()},
            "candidate_family_counts": {c: f["candidate_family_counts"]
                                        for c, f in (presample_facts or {}).items()},
            "final_family_counts": {c: f["final_family_counts"]
                                    for c, f in (presample_facts or {}).items()},
            "n_unknown_family": {c: {"pool": f["n_unknown_family_in_pool"],
                                     "drawn": f["n_unknown_family_drawn"]}
                                 for c, f in (presample_facts or {}).items()},
            "family_by_sample_id_drawn": {
                c: f["family_by_sample_id"]
                for c, f in (presample_facts or {}).items()},
            "family_by_sample_id_candidate_pool": dict(sorted(family_by_id.items())),
            "printed_on_every_figure": "mask_family=<label> on the text bar's "
                                       "first line",
        } if family_cap else {
            "family_cap": {},
            "family_label_flag": bool(args.family_label),
            "label_source": ("q3vl.whereb.amort.data.family_labels (imported, "
                             "not re-implemented); labels are FETCHED ONLY, the "
                             "draw is not family aware"
                             if args.family_label else None),
            "label_range": list(FAMILY_VALUES) if args.family_label else None,
            "whereb_dataset": family_ds_info,
            "family_counts_drawn": (
                dict(sorted(Counter(family_by_id.values()).items()))
                if args.family_label else None),
            "family_by_sample_id_drawn": (dict(sorted(family_by_id.items()))
                                          if args.family_label else None),
            "printed_on_every_figure": ("mask_family=<label> on the text bar's "
                                        "first line" if args.family_label
                                        else None),
            "note": ("--family-cap was not given: the draw is not family aware"
                     if args.family_label else
                     "--family-cap was not given: the draw is not family aware "
                     "and no family label was fetched"),
        }),
        "sampling_pool_exclusion": exclude_facts or {
            "flag": None,
            "note": "--exclude-ids was not given: the sampling pool is the "
                    "split's whole normal-only population",
        },
        "n_scored": len(scored),
        "n_skipped": len(skipped),
        "skipped": skipped,
        "device": "cpu",
        "elapsed_s": round(time.time() - t_start, 1),
        "inputs": {"paths": inputs, "sha256": sha},
        "image_formation": "I_hat = (1-alpha) * I + alpha * f_hat(I), short side 512",
        "gt_formation": "I_star = (1-alpha_gt) * I + alpha_gt * L_l(I), the same "
                        "mix_alpha implementation the dataset generates with",
        "upsampling": "q3vl/where/upsample.py:area_resize (area down / bilinear up); "
                      "no PIL/cv2 resize is used anywhere",
        "error_metric": "CIELab dE00 (CIEDE2000, q3vl/whatb/colorimetry.py:delta_e00) "
                        "per pixel between pred edited and GT edited",
        "soft_iou": ({
            "computed": False,
            "reason": "alpha_source == gt: the predicted alpha IS the GT alpha, "
                      "so the min/max soft-IoU is 1.0 by definition on every "
                      "row.  It is not computed, not stored and not drawn.",
        } if args.alpha_source == "gt" else {
            "form": "min/max (q3vl/whereb/metrics.py:soft_iou_value); the product "
                    "form is banned",
            "support": "grid level: the raw predicted field against GT alpha "
                       "area-averaged onto the same (grid_h, grid_w)",
            "style_rows_note": "a style row's field is the constant 1 definition "
                               "and its GT alpha is 1 everywhere, so its soft-IoU "
                               "is 1.0 by construction and is not a prediction score",
        }),
        "colour_scales": {
            "de00": {"vmin": vmin, "vmax": vmax,
                     "vmax_source": vmax_source,
                     "vmax_note_printed_on_every_colourbar": vmax_note,
                     "this_batch_pooled_p99": batch_pooled_p99,
                     "reference_vmax": REFERENCE_VMAX,
                     "reference_vmax_provenance": REFERENCE_VMAX_SOURCE,
                     "rule": "global, identical on every sample; the pooled "
                             "per-pixel dE00 p99 is read off a 20000-bin "
                             "histogram on [0, 200] and is stored as "
                             "this_batch_pooled_p99 whether or not it is the "
                             "vmax actually used",
                     "n_pixels_above_vmax": n_over,
                     "n_pixels_pooled": int(hist.sum()),
                     "p99_of_per_sample_means":
                         stats_of([s.de00_mean for s in scored])["p99"]},
            "alpha": {"vmin": 0.0, "vmax": 1.0,
                      "rule": "fixed [0,1] on every alpha panel; no per-image "
                              "min-max, invalid field cells drawn white"},
        },
        "invalid_cells": {
            "rule": "a field cell that is non-finite or outside [0,1] is masked, "
                    "drawn white on the alpha panel and counted per sample",
            "n_samples_with_invalid_cells":
                sum(1 for s in scored if s.n_field_cells_invalid),
            "n_cells_total": sum(s.n_field_cells_invalid for s in scored),
        },
        "selection_rule": {
            "select": args.select,
            "quantiles": (list(QUANTILES) if args.select == "quantile"
                          else "unused (--select random)"),
            "n_per_bin": (args.n_per_bin if args.select == "quantile"
                          else "unused (--select random)"),
            "sample_seed": (args.sample_seed if args.select == "random"
                            else "unused (--select quantile)"),
            "n_total": (args.n_total if args.select == "random"
                        else "unused (--select quantile)"),
            "n_total_note": "--n-total is per rendered class, not a grand total",
            "n_total_clamped_to_scored_pool": n_total_clamped or None,
            "classes": list(render_classes),
            "classes_flag": args.classes,
            "classes_note": ("--classes filtered the pre-sample draw, hence the "
                             "scoring pass, the pooled-pixel colour scale and "
                             "the summary figure too: only the drawn rows of "
                             "the listed class(es) were scored"
                             if presample else
                             "--classes filters the selection step only; scoring, "
                             "the pooled-pixel colour scale and the summary figure "
                             "always cover both classes"),
            "text": (select_random.__doc__ if args.select == "random"
                     else select_by_quantile.__doc__),
            "selected_sample_ids": sorted(p["row"].sample_id for _, p in picks),
            "n_selected": len(picks),
        },
        "quantiles_of_de00_mean": quantile_table,
        "statistics": {
            "de00_mean_all": stats_of([s.de00_mean for s in scored]),
            "de00_mean_local": stats_of([s.de00_mean for s in by_class["local"]]),
            "de00_mean_style": stats_of([s.de00_mean for s in by_class["style"]]),
            **({"alpha_gt_mean_all":
                stats_of([s.alpha_gt_mean for s in scored])}
               if args.alpha_source == "gt" else {
                "de00_mean_field_source_stlang_m_low":
                    stats_of([s.de00_mean for s in stlang_rows]),
                "de00_mean_field_source_style_definition_ones":
                    stats_of([s.de00_mean for s in ones_rows]),
                "soft_iou_grid_stlang_m_low":
                    stats_of([s.soft_iou_grid for s in stlang_rows]),
                "soft_iou_grid_style_definition_ones":
                    stats_of([s.soft_iou_grid for s in ones_rows]),
            }),
            "alpha_pred_mean_all": stats_of([s.alpha_pred_mean for s in scored]),
        },
        "where_side": ({
            "field": None,
            "note": "no where predicted field was used: --alpha-source gt.  The "
                    f"predfield directory {args.predfield} covers V_what only "
                    "(its producer pins EVAL_SPLIT='V_what'), so on "
                    f"split={args.split} there is no predicted field on disk and "
                    "none was inferred by this tool.",
        } if args.alpha_source == "gt" else {
            "arm": pf_manifest.get("arm"),
            "checkpoint": pf_manifest.get("checkpoint"),
            "checkpoint_sha256": pf_manifest.get("checkpoint_sha256"),
            "field_key": pf_manifest.get("field_key"),
            "n_by_source": pf_manifest.get("n_by_source"),
            "sources_note": pf_manifest.get("composite_note"),
            "n_scored_by_source": {
                "stlang_m_low": len(stlang_rows),
                "style_definition_ones": len(ones_rows),
            },
        }),
        "what_side": model_facts,
        "zcache_facts": zfacts,
        "criterion_call_counts": dict(_CALLED),
        "criterion_witnesses_required": required_witnesses,
        "figures": figures,
        "summary_figure": summary_png.name,
        "per_sample_jsonl": "per_sample.jsonl",
        "argv": {k: (str(v) if isinstance(v, Path) else v)
                 for k, v in vars(args).items()},
    }
    (out_dir / "manifest.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False, default=str),
        encoding="utf-8")
    print(json.dumps({"n_scored": len(scored), "n_skipped": len(skipped),
                      "vmax": vmax, "elapsed_s": manifest["elapsed_s"]}),
          flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
