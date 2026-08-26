#!/usr/bin/env python3
"""EPR-050 v3: subject-aware masked five-step LUT degradation + exact restoration.

Chain of five kinds, every alpha field computed from the *before* image x0:

  luminance band  highlights  |  luminance band  midtones
  luminance band  shadows     |  global a == 1
  geometry {linear | radial | subject mask}

  y_k = mix_alpha(y_{k-1}, L_k(y_{k-1}), a_k),  y_0 = x0,  y_N == after

The ORDER in which the chain consumes those five kinds is a config value,
`[run] step_order` (v3.3); there is no hard-coded ordering fallback.  Degradation
is the inverse construction of an edit, so the order is semantic: the v3.2 order
was lum_high, lum_mid, lum_shadow, global, geom; the v3.3 order is geom,
lum_high, lum_mid, lum_shadow, global, which makes the restoration direction
(global inverse first, subject inverse last) the normal editing order.
Only the consumption order changes -- the alpha fields are still all derived from
x0 before the chain starts, so putting the geometry step first does not alter any
mask definition.

Restoration (path E) unwinds in reverse, solving at every step

  F_{a_k}(x) = (1 - a_k) x + a_k L_k(x) = y_k     for x

by damped Gauss-Newton.  `(1-a)y + a L^-1(y)` is NOT the inverse of F_a (path N,
falsified in v2: 8-bit p50 error 1.8-33 levels in the soft band versus ~1e-5 for
path E); path N is still computed for the first `--n-naive` samples as a control
column and nothing else.

LUT pool: `recovered >= [pool] recovered_min and clip <= [pool] clip_max`, both
read from the --config TOML.  clip is the fraction of grid
nodes pinned at 0 or 1, so clip == 0 means no node touches the [0,1] boundary and
the map cannot destroy highlight or shadow detail by saturation.  The control
pool (`recovered < 0.5`) is kept and reported side by side.

Numerics are shared with v2 (apply_lut / grid_det / self_check / Gauss-Newton);
`mix_alpha` is imported from the main chain (q3vl/whatb/lutdata.py) so endpoint
snapping is bit-identical to the dataset generator.  Results are written to
pairs.jsonl before any plotting.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import random
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image, ImageDraw

REPO = Path(__file__).resolve().parents[2]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "dataset_build/tools"))
import archive_reader as AR  # noqa: E402  source_path is a prefix key, not a local path
from q3vl.whatb.lutdata import mix_alpha  # noqa: E402  main chain, read-only
# v3.5: the geometry masks are the MAIN CHAIN's, imported and never
# re-implemented (the pixgt.py precedent).  raster_geometry is pure numpy and
# runs once per sample on the CPU, which is negligible next to the chain itself.
# `_semantic_alpha` is private but is imported rather than copied for exactly the
# same reason: a copy would silently drift from the main chain's edge treatment.
from dataset_build.src.construct.canonical_masks import (  # noqa: E402
    _semantic_alpha, raster_geometry)
from dataset_build.src.construct.subject_geom import (  # noqa: E402
    band_geom, linear_geom, radial_geom)

BANK = "/var/cache/veradata/preset_bank_full"
EPR_DIR = REPO / "experiments/prs/EPR-050_masked-lut-degradation"
POOL_JSON = EPR_DIR / "lut_invert_joined.json"
SUBJECT_CACHE = Path("/home/bc/data/datasets/vera_directionA_1M/subject_cache")
SPLIT_TABLE = REPO / "tools/data_splits/splits_sources.csv"
TRAIN_BUCKET_MAX = 89

LEVEL = 255.0                 # every error is reported in 8-bit levels
SAT_HI, SAT_LO = 254 / 255.0, 1 / 255.0
# The five step kinds are fixed; their ORDER along the chain is a config value
# ([run] step_order, v3.3) because "degradation is the inverse construction of an
# edit" makes the order semantic, not cosmetic.  STEP_KIND is the ordered list and
# is filled by load_config(); STEP_KINDS is the unordered set it must be a
# permutation of.  There is no hard-coded ordering fallback.
STEP_KINDS = ("lum_high", "lum_mid", "lum_shadow", "global", "geom", "hue")
STEP_KIND: list[str] = []

# --------------------------------------------------------------------------- #
# Every experiment-semantic number lives in the TOML passed via --config.  These
# are populated by load_config() and are deliberately left as None until then: a
# missing key must raise, never fall back to a stale hard-coded default.
# --------------------------------------------------------------------------- #
ERR_VMAX = CONV_TOL = INV_ITERS = INV_EARLY = PIX_CHUNK = SAT_WARN = None
MASK_ATTEMPTS = MASK_MIN_BAND = N_STEPS = None
COVER_EPS = COVER_MIN = SPAN_LONG_MIN = None
LIN_REF = LIN_FEATHER = LIN_MIN_SHORT = LIN_OFFSET = None
LIN_PROFILE = None
GEOM_SAMPLER = GEOM_WEIGHTS = None
# v3.5 geometry masks: the four MAIN-CHAIN families, rasterised by the main
# chain's own code (see the imports at the top).  Nothing about their shape is
# defined here -- semantic / radial / band / linear are exactly what
# dataset_build/src/construct produces, so the recorded geometry dicts are the
# main chain's (Left/Right/Top/Bottom/Angle/Feather/Flipped for circulargradient,
# ZeroX/ZeroY/FullX/FullY for gradient) -- i.e. the LR/ACR parameter table.
#
# Terminology note kept for the record: the v3.1-lineage linear implementation is
# an OFFSET RAMP -- one-sided, with a separately sampled feather width.  It was
# historically called "band", which was wrong (a band falls off from a centre line
# to BOTH sides).  It survives only as `linear_profile = "offset_ramp"`, which
# exists purely to reproduce v3.1-v3.4 data.
#
# IMPORTANT, coordinate conventions: the main chain rasterises on
# `np.mgrid[0:h,0:w] / (width, height)` (top-left origin, NOT endpoint-closed),
# while the legacy geo_mask() below uses torch.linspace(0,1) (endpoint-closed).
# The two must never be mixed inside one mode: main-chain kinds go exclusively
# through mainchain_geom(), legacy kinds exclusively through geo_mask().
GEOM_KINDS = ("semantic", "radial", "band", "linear")   # main-chain modes
# v3.6: the WIDTH-class parameters of the main-chain geometries may come from this
# EPR's config instead of subject_geom's built-in constants (RADIAL_FEATHERS,
# LINEAR_RAMP).  PLACEMENT stays the main chain's: the centre, the angle and the
# subject-side constraint are all computed by subject_geom from the subject's PCA
# and bbox and are never touched here (user ruling: "摆位必须主体感知，禁随意生成").
# GEOM_WIDTH_SOURCE = "mainchain" reproduces v3.5 exactly.
GEOM_WIDTH_SOURCE = GEOM_FEATHER = GEOM_LINEAR_RAMP = GEOM_WIDTH_TRIES = None
FULL_EPS = FULL_OF_MASK_MAX = FULL_OF_MASK_MIN = SEMANTIC_COVER_MIN = None
# D1 (v3.6): families excused from the `zero_band` floor.  A gradient filter that
# runs across the whole frame legitimately has no alpha == 0 region; the floor is
# a v3-era sampling guard, not a statement about the shape being malformed.  The
# alpha == 0 bit-exactness assertion then holds vacuously on those rows, which is
# made public through `alpha.zero_band_px` / `band_exempt` exactly as A13/A41.
ZERO_BAND_EXEMPT: tuple = ()
# v3.6.1: a floor on the radial ellipse's semi-axes, in units of the frame's
# corresponding side.  The main chain sizes the ellipse from the subject's PCA
# extents times adaptive_margin(area), whose cap is 1.6, so a small subject still
# gets a small ellipse.  The floor scales the ellipse ABOUT ITS OWN CENTRE and by
# ONE factor for both semi-axes, so the subject-aware centre, angle and aspect
# ratio are all untouched.  (0.0, 0.0) == the v3.6 behaviour.
RADIAL_AXIS_MIN: tuple = (0.0, 0.0)
RAD_CENTER = RAD_RADIUS = RAD_FEATHER = None
LUM_Q_LO = LUM_Q_HI = LUM_WIDTH = SUBJ_FEATHER = None
HUE_ANCHORS = HUE_SAT_LO = HUE_SAT_HI = None
DE00_LO = DE00_HI = DE00_TOL = DE00_DRAW_LO = DE00_DRAW_HI = None
BISECT_ITERS = DE00_SUB = ONLY_COMPRESS = None
POOL_REC_MIN = POOL_CLIP_MAX = POOL_CTRL_REC_MAX = None
SEED = SAMPLE_SALT = DE00_SALT = SPLIT_SEED = None
# CLI run control (--repeat-salt), NOT an experiment-semantic parameter, hence not
# in the TOML and not part of the resume identity: it only lets the same source be
# degraded again as an independent chain.  None == the pre-repeat behaviour.
REPEAT_SALT = None
# CLI run control (--force-major), same status as REPEAT_SALT: pins the chain's
# major so repeat diversity isolates the mask/order/strength contribution from the
# style contribution.  None == the pre-existing uniform-over-major draw.
FORCE_MAJOR = None
CFG: dict = {}


def _sec(cfg: dict, name: str) -> dict:
    if name not in cfg or not isinstance(cfg[name], dict):
        die(f"config: missing section [{name}]")
    return cfg[name]


def _num(cfg: dict, sec: str, key: str) -> float:
    s = _sec(cfg, sec)
    if key not in s:
        die(f"config: missing key [{sec}] {key}")
    v = s[key]
    if not isinstance(v, (int, float)) or isinstance(v, bool):
        die(f"config: [{sec}] {key} must be a number, got {v!r}")
    return float(v)


def _int(cfg: dict, sec: str, key: str) -> int:
    v = _num(cfg, sec, key)
    if v != int(v):
        die(f"config: [{sec}] {key} must be an integer, got {v!r}")
    return int(v)


def _rng2(cfg: dict, sec: str, key: str) -> tuple[float, float]:
    s = _sec(cfg, sec)
    if key not in s:
        die(f"config: missing key [{sec}] {key}")
    v = s[key]
    if (not isinstance(v, (list, tuple)) or len(v) != 2
            or not all(isinstance(x, (int, float)) and not isinstance(x, bool) for x in v)):
        die(f"config: [{sec}] {key} must be a [lo, hi] pair of numbers, got {v!r}")
    if float(v[0]) > float(v[1]):
        die(f"config: [{sec}] {key} has lo > hi: {v!r}")
    return float(v[0]), float(v[1])


def _str(cfg: dict, sec: str, key: str, allowed=None) -> str:
    s = _sec(cfg, sec)
    if key not in s:
        die(f"config: missing key [{sec}] {key}")
    v = s[key]
    if not isinstance(v, str):
        die(f"config: [{sec}] {key} must be a string, got {v!r}")
    if allowed and v not in allowed:
        die(f"config: [{sec}] {key} must be one of {allowed}, got {v!r}")
    return v


def _steporder(cfg: dict, sec: str, key: str, allowed: tuple[str, ...],
               n: int, required: tuple[str, ...] = ("geom",)) -> list[str]:
    """The chain order: `n` DISTINCT kinds drawn from `allowed`.

    v3.3 made the order a config value over a fixed set of 5.  v3.4 adds a sixth
    kind (`hue`) that only some chains use, so the rule is "distinct subset of
    length chain_len" rather than "permutation of everything"; typos and repeats
    still raise.  `required` names the kinds other code indexes by name.
    """
    s = _sec(cfg, sec)
    if key not in s:
        die(f"config: missing key [{sec}] {key}")
    v = s[key]
    if not isinstance(v, list) or not all(isinstance(x, str) for x in v):
        die(f"config: [{sec}] {key} must be a list of strings, got {v!r}")
    bad = [x for x in v if x not in allowed]
    if bad:
        die(f"config: [{sec}] {key} has unknown step kind(s) {bad}; "
            f"allowed are {list(allowed)}")
    if len(set(v)) != len(v):
        die(f"config: [{sec}] {key} repeats a step kind: {v!r}")
    if len(v) != n:
        die(f"config: [{sec}] {key} has {len(v)} steps but [pool] chain_len is {n}")
    miss = [x for x in required if x not in v]
    if miss:
        die(f"config: [{sec}] {key} must contain {list(required)}, missing {miss}")
    return list(v)


def _anchors(cfg: dict, sec: str, key: str) -> list[tuple[float, float]]:
    """[[degrees, weight], ...] for the v3.4 hue weighting curve.

    Degrees must be in [0, 360) and strictly increasing (they are interpolated
    circularly, so the last anchor wraps round to the first); weights in [0, 1].
    """
    s = _sec(cfg, sec)
    if key not in s:
        die(f"config: missing key [{sec}] {key}")
    v = s[key]
    if (not isinstance(v, list) or len(v) < 2
            or not all(isinstance(p, list) and len(p) == 2
                       and all(isinstance(q, (int, float)) and not isinstance(q, bool)
                               for q in p) for p in v)):
        die(f"config: [{sec}] {key} must be a list of [degrees, weight] pairs, got {v!r}")
    out = [(float(d), float(w)) for d, w in v]
    for d, w in out:
        if not (0.0 <= d < 360.0):
            die(f"config: [{sec}] {key} degree {d} outside [0, 360)")
        if not (0.0 <= w <= 1.0):
            die(f"config: [{sec}] {key} weight {w} outside [0, 1]")
    if any(out[i][0] >= out[i + 1][0] for i in range(len(out) - 1)):
        die(f"config: [{sec}] {key} degrees must be strictly increasing, got "
            f"{[d for d, _ in out]}")
    return out


def _bool(cfg: dict, sec: str, key: str) -> bool:
    s = _sec(cfg, sec)
    if key not in s:
        die(f"config: missing key [{sec}] {key}")
    if not isinstance(s[key], bool):
        die(f"config: [{sec}] {key} must be a boolean, got {s[key]!r}")
    return s[key]


def load_config(path: Path) -> dict:
    """Populate every experiment-semantic global from the TOML.  Strict: a missing
    or mistyped key raises here rather than silently reverting to an old default."""
    global ERR_VMAX, CONV_TOL, INV_ITERS, INV_EARLY, PIX_CHUNK, SAT_WARN
    global MASK_ATTEMPTS, MASK_MIN_BAND, N_STEPS
    global COVER_EPS, COVER_MIN, SPAN_LONG_MIN
    global LIN_REF, LIN_FEATHER, LIN_MIN_SHORT, LIN_OFFSET
    global LIN_PROFILE, GEOM_SAMPLER, GEOM_WEIGHTS
    global GEOM_WIDTH_SOURCE, GEOM_FEATHER, GEOM_LINEAR_RAMP, GEOM_WIDTH_TRIES
    global FULL_EPS, FULL_OF_MASK_MAX, FULL_OF_MASK_MIN, SEMANTIC_COVER_MIN
    global ZERO_BAND_EXEMPT, RADIAL_AXIS_MIN
    global RAD_CENTER, RAD_RADIUS, RAD_FEATHER
    global LUM_Q_LO, LUM_Q_HI, LUM_WIDTH, SUBJ_FEATHER
    global HUE_ANCHORS, HUE_SAT_LO, HUE_SAT_HI
    global DE00_LO, DE00_HI, DE00_TOL, DE00_DRAW_LO, DE00_DRAW_HI
    global BISECT_ITERS, DE00_SUB, ONLY_COMPRESS
    global POOL_REC_MIN, POOL_CLIP_MAX, POOL_CTRL_REC_MAX
    global SEED, SAMPLE_SALT, DE00_SALT, SPLIT_SEED, CFG, STEP_KIND
    try:
        import tomllib
    except ModuleNotFoundError:                     # pragma: no cover
        import tomli as tomllib                     # type: ignore
    if not path.exists():
        die(f"config not found: {path}")
    CFG = tomllib.loads(path.read_text())

    POOL_REC_MIN = _num(CFG, "pool", "recovered_min")
    POOL_CLIP_MAX = _num(CFG, "pool", "clip_max")
    POOL_CTRL_REC_MAX = _num(CFG, "pool", "ctrl_recovered_max")
    N_STEPS = _int(CFG, "pool", "chain_len")

    LUM_Q_LO = _rng2(CFG, "masks", "lum_q_lo")
    LUM_Q_HI = _rng2(CFG, "masks", "lum_q_hi")
    LUM_WIDTH = _rng2(CFG, "masks", "lum_width")
    COVER_EPS = _num(CFG, "masks", "cover_eps")
    COVER_MIN = _num(CFG, "masks", "cover_min")
    MASK_MIN_BAND = _num(CFG, "masks", "min_band")
    MASK_ATTEMPTS = _int(CFG, "masks", "attempts_max")
    SUBJ_FEATHER = _rng2(CFG, "masks", "subject_feather_frac")
    # v3.5: which linear parameterisation is in force.  Strict like every other
    # key -- the archived v3.2 / v3.3 configs carry an explicit
    # `linear_profile = "band"` so reproducing old data still states its shape
    # rather than inheriting it from a default.
    # v3.5 geometry sampler.  "subject_prob" is the v3.1-v3.4 two-stage draw
    # (Bernoulli on subject, then a uniform pick between linear and radial);
    # "weights" is the v3.5 categorical draw that also carries the new `band`
    # v3.5 geometry sampler.  "subject_prob" is the v3.1-v3.4 two-stage draw
    # (Bernoulli on subject, then a uniform pick between linear and radial) on
    # this EPR's own shapes; "mainchain_weights" is the v3.5 categorical draw over
    # the four MAIN-CHAIN families.  Strict, and the archived configs pin
    # "subject_prob" explicitly -- the two samplers consume the rng differently
    # and use different rasterisers, so this can never be inferred.
    GEOM_SAMPLER = _str(CFG, "masks", "geom_sampler",
                        ("mainchain_weights", "subject_prob"))
    GEOM_WEIGHTS = None
    if GEOM_SAMPLER == "mainchain_weights":
        gw = _sec(CFG, "masks").get("geom_weights")
        if not isinstance(gw, dict) or not gw:
            die("config: [masks] geom_weights must be a non-empty table")
        bad = [k for k in gw if k not in GEOM_KINDS]
        if bad:
            die(f"config: [masks] geom_weights has unknown kinds {bad}; "
                f"expected a subset of {sorted(GEOM_KINDS)}")
        if any(float(v) < 0 for v in gw.values()) or sum(map(float, gw.values())) <= 0:
            die(f"config: [masks] geom_weights must be non-negative and sum > 0, "
                f"got {gw!r}")
        GEOM_WEIGHTS = {k: float(gw[k]) for k in sorted(gw) if float(gw[k]) > 0}
        # v3.6 keys.  They are read ONLY under the main-chain sampler, so the
        # archived v3.2 / v3.3 / v3.4 configs (geom_sampler = "subject_prob")
        # stay byte-identical and keep reproducing their own data.  The archived
        # v3.5 config carries them pinned to the v3.5 behaviour
        # (mainchain widths / no full-strength cap / no semantic gate).
        GEOM_WIDTH_SOURCE = _str(CFG, "masks", "geom_width_source",
                                 ("mainchain", "config"))
        FULL_EPS = _num(CFG, "masks", "full_eps")
        FULL_OF_MASK_MAX = _num(CFG, "masks", "full_frac_of_mask_max")
        FULL_OF_MASK_MIN = _num(CFG, "masks", "full_frac_of_mask_min")
        SEMANTIC_COVER_MIN = _num(CFG, "masks", "semantic_cover_min")
        zbe = _sec(CFG, "masks").get("zero_band_exempt_geoms")
        if not isinstance(zbe, list) or any(k not in GEOM_KINDS for k in zbe):
            die(f"config: [masks] zero_band_exempt_geoms must be a list of "
                f"{sorted(GEOM_KINDS)} (use [] for none), got {zbe!r}")
        ZERO_BAND_EXEMPT = tuple(zbe)
        # v3.6.1.  Accepts a [lo, hi] sampling range or a single number; the
        # archived v3.6 config pins the scalar 0.0, i.e. the floor is off and no
        # rng is consumed for it, so that run reproduces byte-for-byte.
        ram = _sec(CFG, "masks").get("radial_axis_min_frac")
        if isinstance(ram, list):
            RADIAL_AXIS_MIN = _rng2(CFG, "masks", "radial_axis_min_frac")
        elif isinstance(ram, (int, float)):
            RADIAL_AXIS_MIN = (float(ram), float(ram))
        else:
            die("config: [masks] radial_axis_min_frac must be a number or a "
                f"[lo, hi] pair (0.0 = off), got {ram!r}")
        if not (0.0 <= RADIAL_AXIS_MIN[0] <= RADIAL_AXIS_MIN[1] < 1.0):
            die(f"config: [masks] need 0 <= radial_axis_min_frac <= 1, got "
                f"{RADIAL_AXIS_MIN}")
        if not (0.0 < FULL_EPS <= 1.0):
            die(f"config: [masks] full_eps must be in (0, 1], got {FULL_EPS}")
        if not (0.0 <= FULL_OF_MASK_MIN < FULL_OF_MASK_MAX <= 1.0):
            die(f"config: [masks] need 0 <= full_frac_of_mask_min < "
                f"full_frac_of_mask_max <= 1, got {FULL_OF_MASK_MIN} / "
                f"{FULL_OF_MASK_MAX}")
        if GEOM_WIDTH_SOURCE == "config":
            GEOM_FEATHER = _rng2(CFG, "masks", "geom_feather")
            GEOM_LINEAR_RAMP = _rng2(CFG, "masks", "geom_linear_ramp")
            GEOM_WIDTH_TRIES = _int(CFG, "masks", "geom_width_tries")
            if GEOM_WIDTH_TRIES < 1:
                die("config: [masks] geom_width_tries must be >= 1")

    # Which linear parameterisation the LEGACY sampler uses.  Only "offset_ramp"
    # is supported now: the v3.5 linear shape is the main chain's `gradient`,
    # reached through mainchain_weights, not through a profile switch.
    LIN_PROFILE = _str(CFG, "masks", "linear_profile", ("offset_ramp",))
    if GEOM_SAMPLER == "subject_prob":
        SUBJ_FEATHER = _rng2(CFG, "masks", "subject_feather_frac")
        LIN_REF = _str(CFG, "masks", "linear_feather_ref", ("short", "long"))
        LIN_FEATHER = _rng2(CFG, "masks", "linear_feather")
        LIN_MIN_SHORT = _num(CFG, "masks", "linear_feather_min_short")
        LIN_OFFSET = _rng2(CFG, "masks", "linear_offset")
    SPAN_LONG_MIN = _num(CFG, "masks", "span_long_min")
    if GEOM_SAMPLER == "subject_prob":
        RAD_CENTER = _rng2(CFG, "masks", "radial_center")
        RAD_RADIUS = _rng2(CFG, "masks", "radial_radius")
        RAD_FEATHER = _rng2(CFG, "masks", "radial_feather")
    # v3.4 hue step.  Only read when the chain actually contains it, so a 5-step
    # v3.3 config stays valid without carrying dead keys.
    if "hue" in _sec(CFG, "run").get("step_order", []):
        HUE_ANCHORS = _anchors(CFG, "masks", "hue_anchors")
        HUE_SAT_LO = _num(CFG, "masks", "hue_sat_lo")
        HUE_SAT_HI = _num(CFG, "masks", "hue_sat_hi")
        if not (0.0 <= HUE_SAT_LO < HUE_SAT_HI <= 1.0):
            die(f"config: [masks] need 0 <= hue_sat_lo < hue_sat_hi <= 1, got "
                f"{HUE_SAT_LO} / {HUE_SAT_HI}")

    DE00_DRAW_LO = _num(CFG, "strength", "de00_target_low")
    DE00_DRAW_HI = _num(CFG, "strength", "de00_target_high")
    DE00_LO = _num(CFG, "strength", "band_low")
    DE00_HI = _num(CFG, "strength", "band_high")
    DE00_TOL = _num(CFG, "strength", "tolerance")
    BISECT_ITERS = _int(CFG, "strength", "bisect_iters")
    ONLY_COMPRESS = _bool(CFG, "strength", "only_compress")
    DE00_SUB = _int(CFG, "strength", "calib_max_px")

    ERR_VMAX = _num(CFG, "render", "err_vmax")

    SEED = _int(CFG, "run", "seed")
    SAMPLE_SALT = _str(CFG, "run", "sample_salt")
    DE00_SALT = _str(CFG, "run", "de00_salt")
    SPLIT_SEED = _str(CFG, "run", "split_salt")
    INV_ITERS = _int(CFG, "run", "inv_iters")
    INV_EARLY = _num(CFG, "run", "inv_early_levels") / LEVEL
    CONV_TOL = _num(CFG, "run", "conv_tol")
    PIX_CHUNK = _int(CFG, "run", "pix_chunk")
    SAT_WARN = _num(CFG, "run", "sat_warn")
    # v3.3: the chain order is a config value.  Degradation is the inverse
    # construction of an edit, so which step runs first is part of the protocol.
    # v3.4: `hue` joins the kind set and the chain may be 5 or 6 steps long.
    STEP_KIND = _steporder(CFG, "run", "step_order", STEP_KINDS, N_STEPS)
    if not ONLY_COMPRESS:
        die("config: [strength] only_compress = false is not supported -- s is "
            "parameterised on (0,1] and cannot amplify a weak edit")
    if not (DE00_LO <= DE00_DRAW_LO <= DE00_DRAW_HI <= DE00_HI):
        die(f"config: [strength] draw range [{DE00_DRAW_LO}, {DE00_DRAW_HI}] must sit "
            f"inside the band [{DE00_LO}, {DE00_HI}]")
    return CFG
BISECT_ITERS = 60             # configs/lut_numeric_clusters.epr035.toml


def die(msg: str):
    """N7: explicit raise, so `python -O` cannot strip a guard."""
    raise RuntimeError(msg)


class MaskRejected(RuntimeError):
    """The pre-registered mask criterion (every masked step needs >=1% alpha==0 and
    >=1% soft band) is unsatisfiable for this source within MASK_ATTEMPTS draws.

    Raised instead of die() so the *sampling* layer can exclude the source and draw
    a replacement.  The criterion itself is never relaxed; every exclusion is
    written to skipped.json with the failing step, the realised band fractions and
    the source's luminance spread, so the count is visible and never silent.
    """

    def __init__(self, gid: str, diag: dict):
        super().__init__(f"{gid}: no mask set with a populated zero band and soft "
                         f"band in {MASK_ATTEMPTS} attempts; {json.dumps(diag)}")
        self.gid, self.diag = gid, diag


# --------------------------------------------------------------------------- #
# LUT numerics
# --------------------------------------------------------------------------- #
def apply_lut(vol: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
    """vol: (1,3,D_b,D_g,D_r) -- the bank stores grid[b,g,r,c]
    (q3vl/whatb/lutdata.py:94-96).  x: (1,M,3) in [0,1] -> (1,M,3).

    grid_sample's last-dim order is (x,y,z) -> (W,H,D) = (r,g,b) so the RGB
    triple goes in unpermuted.  Getting this backwards silently transposes R and
    B, which flips det(J) at every node -- that is what self_check() catches.
    """
    lo, m = x.shape[0], x.shape[1]
    g = (x * 2 - 1).view(lo, m, 1, 1, 3)
    out = F.grid_sample(vol, g, mode="bilinear", align_corners=True,
                        padding_mode="border")
    return out.view(lo, 3, m).permute(0, 2, 1).contiguous()


def grid_det(v: torch.Tensor) -> torch.Tensor:
    """v: (L,N,N,N,3) stored [b,g,r,channel] -> det(J) at interior nodes."""
    n = v.shape[1]
    h = 1.0 / (n - 1)
    d_b = (v[:, 2:, 1:-1, 1:-1] - v[:, :-2, 1:-1, 1:-1]) / (2 * h)
    d_g = (v[:, 1:-1, 2:, 1:-1] - v[:, 1:-1, :-2, 1:-1]) / (2 * h)
    d_r = (v[:, 1:-1, 1:-1, 2:] - v[:, 1:-1, 1:-1, :-2]) / (2 * h)
    return torch.linalg.det(torch.stack([d_r, d_g, d_b], dim=-1)).reshape(v.shape[0], -1)


def _gn(fwd, y: torch.Tensor, n: int, iters: int) -> tuple[torch.Tensor, int]:
    """Damped Gauss-Newton for fwd(x) = y.  Returns (x, iterations actually used).

    N2: the finite-difference step is clamped to [0,1], so at a gamut boundary
    the realised spacing is smaller than 2d.  Dividing by the *realised* spacing
    turns the central difference into a one-sided one automatically instead of
    silently shrinking the derivative.
    """
    x = y.clone()
    d = 0.5 / (n - 1)
    eye = torch.eye(3, device=y.device).expand(y.shape[0], y.shape[1], 3, 3)
    used = iters
    for it in range(iters):
        r = fwd(x) - y
        if float(r.abs().max()) < INV_EARLY:
            used = it
            break
        cols = []
        for axis in range(3):
            e = torch.zeros(3, device=y.device)
            e[axis] = d
            xp, xm = (x + e).clamp(0, 1), (x - e).clamp(0, 1)
            h = (xp[..., axis] - xm[..., axis]).unsqueeze(-1).clamp_min(1e-8)
            cols.append((fwd(xp) - fwd(xm)) / h)
        jac = torch.stack(cols, dim=-1)
        jtj = jac.transpose(-1, -2) @ jac + eye * 1e-3
        step = torch.linalg.solve(jtj, jac.transpose(-1, -2) @ r.unsqueeze(-1)).squeeze(-1)
        x = (x - step).clamp(0, 1)
    return x, used


def invert_lut(vol: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    """Root-find for L(x) = y (path N only)."""
    return _gn(lambda t: apply_lut(vol, t), y, vol.shape[-1], INV_ITERS)[0]


def invert_blend(vol: torch.Tensor, y: torch.Tensor,
                 alpha: torch.Tensor) -> tuple[torch.Tensor, int]:
    """Root-find for F_alpha(x) = (1-alpha) x + alpha L(x) = y, per-pixel alpha.

    Endpoint snapping is re-applied to the result so alpha==0 pixels come back
    bit-exact; alpha==1 reduces to the plain LUT inverse.
    """
    fwd = lambda t: mix_alpha(t, apply_lut(vol, t), alpha)  # noqa: E731
    x, used = _gn(fwd, y, vol.shape[-1], INV_ITERS)
    return torch.where(alpha == 0, y, x), used


def self_check(dev: str) -> dict:
    """Identity LUT must round-trip, have det(J)==1 and report zero fold-over."""
    n = 17
    ax = torch.linspace(0, 1, n, device=dev)
    b, g, r = torch.meshgrid(ax, ax, ax, indexing="ij")   # stored order [b,g,r]
    ident = torch.stack([r, g, b], dim=-1)[None]          # channels are RGB
    vol = ident.permute(0, 4, 1, 2, 3).contiguous()
    x = torch.rand((1, 4096, 3), device=dev)

    err = (apply_lut(vol, x) - x).abs().max().item()
    if not err < 1e-4:
        die(f"identity LUT does not round-trip: max err {err}")
    det = grid_det(ident)
    dmed = det.median().item()
    if not abs(dmed - 1.0) < 1e-3:
        die(f"identity det(J)={dmed}, expected 1")
    fold = (det <= 0).float().mean().item()
    if not fold < 1e-6:
        die(f"identity LUT reports fold-over: {fold}")

    # analytically invertible per-channel gamma LUT
    gam = torch.stack([r ** 1.6, g ** 0.7, b ** 1.3], dim=-1)[None]
    gvol = gam.permute(0, 4, 1, 2, 3).contiguous()
    a0 = torch.zeros((1, 4096, 1), device=dev)
    a1 = torch.ones((1, 4096, 1), device=dev)
    ah = torch.full((1, 4096, 1), 0.5, device=dev)

    e0 = (invert_blend(gvol, x, a0)[0] - x).abs().max().item()
    if e0 != 0.0:
        die(f"alpha=0 inverse is not bit-exact: {e0}")
    gx = apply_lut(gvol, x)
    if not torch.equal(mix_alpha(x, gx, a1), gx):
        die("mix_alpha does not snap alpha==1")
    x1, i1 = invert_blend(gvol, gx, a1)
    e1 = ((x1 - x).abs().max() * LEVEL).item()
    if not e1 < 0.5:
        die(f"alpha=1 inverse off by {e1} levels on a monotone gamma LUT")
    yh = mix_alpha(x, gx, ah)
    xh, ih = invert_blend(gvol, yh, ah)
    eh = ((xh - x).abs().max() * LEVEL).item()
    if not eh < 0.5:
        die(f"alpha=0.5 blended inverse off by {eh} levels")

    out = dict(identity_roundtrip=err, identity_det_med=dmed, identity_fold=fold,
               gamma_alpha0_bitexact=e0, gamma_alpha1_err_levels=e1,
               gamma_alpha1_iters=i1, gamma_alpha05_err_levels=eh,
               gamma_alpha05_iters=ih, inv_iters_cap=INV_ITERS,
               inv_early_stop_levels=INV_EARLY * LEVEL)
    print(f"self-check ok: {json.dumps(out)}", flush=True)
    return out


def chunked(fn, y: torch.Tensor, *extra):
    """Run a (1,M,3) -> (1,M,3) op in pixel chunks to bound peak memory.

    If `fn` returns a tuple its second element is taken as an iteration count and
    the maximum over chunks is returned.
    """
    m = y.shape[1]
    slices = [slice(s, min(s + PIX_CHUNK, m)) for s in range(0, m, PIX_CHUNK)]
    parts, iters = [], 0
    for sl in slices:
        ex = [e[:, sl] if torch.is_tensor(e) and e.dim() > 1 and e.shape[1] == m else e
              for e in extra]
        out = fn(y[:, sl], *ex)
        if isinstance(out, tuple):
            out, it = out
            iters = max(iters, it)
        parts.append(out)
    cat = torch.cat(parts, dim=1) if len(parts) > 1 else parts[0]
    return (cat, iters) if iters or len(extra) else cat


# --------------------------------------------------------------------------- #
# masks
# --------------------------------------------------------------------------- #
def smoothstep(t: torch.Tensor) -> torch.Tensor:
    """Saturates to exact 0 / exact 1 outside [0,1] -- required for endpoint snap."""
    t = t.clamp(0.0, 1.0)
    return t * t * (3.0 - 2.0 * t)


def luminance(img: torch.Tensor) -> torch.Tensor:
    w = torch.tensor([0.2126, 0.7152, 0.0722], device=img.device, dtype=img.dtype)
    return (img * w).sum(-1)


def lum_bands(x0: torch.Tensor, q_lo: float, q_hi: float, width: float):
    """highlights / midtones / shadows, smoothstep edges, overlap allowed."""
    lum = luminance(x0).float()
    t_lo = torch.quantile(lum.flatten(), q_lo).item()
    t_hi = torch.quantile(lum.flatten(), q_hi).item()
    a = smoothstep((lum - (t_lo - width)) / (2 * width))   # 0 below t_lo, 1 above
    b = smoothstep((lum - (t_hi - width)) / (2 * width))   # 0 below t_hi, 1 above
    return dict(lum_high=b, lum_mid=a * (1.0 - b), lum_shadow=1.0 - a), (t_lo, t_hi)


def hsv_hue_sat(x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """HSV hue (degrees, [0,360)) and saturation from **sRGB values as stored**.

    Colour-space note (v3.4): the hue is taken straight from the sRGB triplet
    already in [0,1] -- the same numbers the LUT chain and any downstream
    training/inference consume -- with the textbook HSV formulas
    (S = (max-min)/max, hue from which channel is the max).  No linearisation, no
    new dependency, no other colour space.  Hue is undefined where max == min;
    those pixels get hue 0 and S = 0 and are removed by the saturation gate.
    """
    cmax, amax_i = x.max(-1)
    cmin = x.min(-1).values
    d = cmax - cmin
    sat = torch.where(cmax > 0, d / cmax.clamp_min(1e-12),
                      torch.zeros_like(cmax))
    r, g, b = x[..., 0], x[..., 1], x[..., 2]
    dz = d.clamp_min(1e-12)
    hr = ((g - b) / dz) % 6.0
    hg = ((b - r) / dz) + 2.0
    hb = ((r - g) / dz) + 4.0
    h = torch.where(amax_i == 0, hr, torch.where(amax_i == 1, hg, hb)) * 60.0
    h = torch.where(d > 0, h, torch.zeros_like(h)) % 360.0
    return h, sat


def hue_weight(h: torch.Tensor, anchors: list[tuple[float, float]]) -> torch.Tensor:
    """Piecewise-linear weight over the hue circle, from the config anchors.

    The last anchor wraps round to the first (the segment 300 -> 0+360 here), so
    the curve is continuous across the red end of the circle.
    """
    degs = [d for d, _ in anchors] + [anchors[0][0] + 360.0]
    wts = [w for _, w in anchors] + [anchors[0][1]]
    out = torch.full_like(h, float(wts[0]))
    hs = torch.where(h < degs[0], h + 360.0, h)      # rotate onto [d0, d0+360)
    for i in range(len(degs) - 1):
        d0, d1, w0, w1 = degs[i], degs[i + 1], wts[i], wts[i + 1]
        t = ((hs - d0) / (d1 - d0)).clamp(0.0, 1.0)
        seg = (hs >= d0) & (hs <= d1)
        out = torch.where(seg, w0 + (w1 - w0) * t, out)
    return out


def hue_mask(x0: torch.Tensor) -> torch.Tensor:
    """v3.4 hue-selective alpha: hue weight x chroma gate.

    The chroma gate is mandatory: hue is meaningless for a grey pixel, so the
    weight is ramped in by a smoothstep on HSV saturation between
    [masks] hue_sat_lo and hue_sat_hi.  Below hue_sat_lo the alpha is EXACTLY 0
    (smoothstep saturates), which is what keeps the per-step alpha == 0
    bit-exactness guard non-vacuous on this step.
    """
    h, s = hsv_hue_sat(x0)
    gate = smoothstep((s - HUE_SAT_LO) / (HUE_SAT_HI - HUE_SAT_LO))
    return hue_weight(h, HUE_ANCHORS) * gate


def feather(m: torch.Tensor, rad: int) -> torch.Tensor:
    """Two box blurs ~ a Gaussian.  Keeps exact 0 / 1 away from the edge, which is
    what gives the subject mask a genuine soft band without touching its core."""
    if rad < 1:
        return m
    k = 2 * rad + 1
    t = m[None, None]
    for _ in range(2):
        t = F.avg_pool2d(F.pad(t, (rad,) * 4, mode="replicate"), k, stride=1)
    return t[0, 0].clamp(0, 1)


def span_long_frac(m: torch.Tensor) -> float:
    """v3.2: how far the acted region (alpha > COVER_EPS) reaches along the image's
    LONG axis, as a fraction of that axis -- the bounding-box extent of the acted
    rows (or columns) rather than its area.  This is the direct check for
    "the gradient traverses >= 60% of the long side"; in v3.1 the same words were
    implemented as a feather-width floor, which cannot coexist with a feather
    sampled at 10-30% of the short side (A24).
    """
    a = m > COVER_EPS
    if not bool(a.any()):
        return 0.0
    h, w = a.shape
    proj = a.any(1) if h >= w else a.any(0)      # collapse across the short axis
    idx = torch.nonzero(proj, as_tuple=False)[:, 0]
    return float((int(idx.max()) - int(idx.min()) + 1) / proj.numel())


def geom_shape_stats(m: torch.Tensor) -> dict:
    """The shape numbers the geometry step is judged on.

    v3.6 adds `full_frac_of_mask` = frac(alpha >= full_eps) / frac(alpha >
    cover_eps): the share of the MASK (not of the frame) that sits at full
    strength.  The user's criterion is on that ratio -- at most 30% of the mask
    may be flat at full strength, the other >=70% has to be transition -- which
    is why the denominator is the mask and not the picture.
    """
    cov = float((m > COVER_EPS).float().mean())
    full = float((m >= FULL_EPS).float().mean()) if FULL_EPS is not None else 0.0
    return dict(
        frac_zero=float((m == 0).float().mean()),
        frac_mid=float(((m > 0) & (m < 1)).float().mean()),
        coverage=cov,
        full_frac=full,
        full_frac_of_mask=(full / cov) if cov > 0 else 0.0,
        mid_frac_of_mask=(float(((m > COVER_EPS) & (m < FULL_EPS)).float().mean())
                          / cov) if (cov > 0 and FULL_EPS is not None) else 0.0,
    )


def geom_step_why(m: torch.Tensor, kind: str) -> tuple[list[str], dict]:
    """Which pre-registered criteria the geometry alpha fails, and the numbers.

    Identical in content to the generic per-step check for every criterion that
    existed before v3.6; `full_of_mask` is the v3.6 addition and is inert when
    the config pins `full_frac_of_mask_max = 1.0` (the ratio is <= 1 by
    construction, and the test is strict).
    """
    st = geom_shape_stats(m)
    exempt = kind in ("subject", "semantic")     # subject-shaped: area is the subject's
    why = []
    # D1: `linear` (a full-frame LR gradient filter) may be excused from the
    # alpha == 0 floor by config; radial / band keep it (they have a zero region
    # by construction).  The exemption is a config list, so replaying an archived
    # config still enforces exactly what that run enforced.
    if st["frac_zero"] < MASK_MIN_BAND and kind not in ZERO_BAND_EXEMPT:
        why.append("zero_band")
    if st["frac_mid"] < MASK_MIN_BAND:
        why.append("soft_band")
    if not exempt and st["coverage"] < COVER_MIN:
        why.append("coverage")
    if not exempt and SPAN_LONG_MIN > 0:
        st["span_long"] = span_long_frac(m)
        if st["span_long"] < SPAN_LONG_MIN:
            why.append("span_long")
    if (not exempt and FULL_OF_MASK_MAX is not None
            and st["full_frac_of_mask"] > FULL_OF_MASK_MAX):
        why.append("full_of_mask")
    # D2: the target band is [full_frac_of_mask_min, full_frac_of_mask_max] --
    # a real LR mask is a solid core with a wide falloff, not an all-diffuse
    # field.  Pinned to 0.0 in the archived configs, where the test is inert.
    if (not exempt and FULL_OF_MASK_MIN
            and st["full_frac_of_mask"] < FULL_OF_MASK_MIN):
        why.append("full_of_mask_low")
    return why, {k: (round(v, 6) if isinstance(v, float) else v)
                 for k, v in st.items()}


def _grow_ellipse(geom: dict, min_frac: float) -> tuple[dict, float]:
    """v3.6.1 size floor for the main chain's `circulargradient`.

    Scales the ellipse ABOUT ITS OWN CENTRE by a single factor so that each
    semi-axis reaches `min_frac` of the frame's corresponding side (rx is in
    units of the frame width, ry of its height -- the main chain rasterises on
    x = col/width, y = row/height).  One factor for both axes keeps the aspect
    ratio, and the centre and Angle are not touched, so the subject-aware
    placement the main chain computed survives untouched.  Returns the geometry
    and the factor actually applied (1.0 when the ellipse is already big enough).
    """
    left, right = float(geom["Left"]), float(geom["Right"])
    top, bottom = float(geom["Top"]), float(geom["Bottom"])
    cx, cy = (left + right) / 2.0, (top + bottom) / 2.0
    rx, ry = abs(right - left) / 2.0, abs(bottom - top) / 2.0
    if rx <= 0 or ry <= 0:
        return dict(geom), 1.0
    k = max(1.0, min_frac / rx, min_frac / ry)
    if k == 1.0:
        return dict(geom), 1.0
    out = dict(geom)
    out["Left"] = round(cx - rx * k, 4)
    out["Right"] = round(cx + rx * k, 4)
    out["Top"] = round(cy - ry * k, 4)
    out["Bottom"] = round(cy + ry * k, 4)
    return out, k


def _set_ramp_len(geom: dict, length: float) -> dict:
    """v3.6 width control for the main chain's `gradient`: keep the ramp's MIDPOINT
    and its DIRECTION (both are the main chain's subject-side placement) and only
    change how long the Zero -> Full ramp is."""
    zx, zy = float(geom["ZeroX"]), float(geom["ZeroY"])
    fx, fy = float(geom["FullX"]), float(geom["FullY"])
    mx, my = (zx + fx) / 2.0, (zy + fy) / 2.0
    dx, dy = fx - zx, fy - zy
    n = (dx * dx + dy * dy) ** 0.5
    if n <= 0:
        die("gradient geometry has a zero-length Zero->Full vector")
    ux, uy = dx / n, dy / n
    out = dict(geom)
    out["ZeroX"] = round(mx - length / 2.0 * ux, 4)
    out["ZeroY"] = round(my - length / 2.0 * uy, 4)
    out["FullX"] = round(mx + length / 2.0 * ux, 4)
    out["FullY"] = round(my + length / 2.0 * uy, 4)
    return out


def python_rng(source_id: str, *parts) -> "random.Random":
    """A python Random for the main-chain samplers, keyed on content like every
    other draw in this tool (sha1 rule family, per CLAUDE.md)."""
    key = ":".join([SAMPLE_SALT, source_id + _rep(), *map(str, parts)])
    return random.Random(int(hashlib.sha1(key.encode()).hexdigest()[:16], 16))


def mainchain_geom(kind: str, hard: np.ndarray, bbox: tuple, area: float,
                   prng, h: int, w: int, dev: str):
    """v3.5: the geometry mask IS the main chain's.  Returns (recorded_params,
    alpha) or (recorded_params, None) when the main-chain sampler declares the
    geometry infeasible for this subject.

    Deliberate difference from the main chain, recorded in NOTES: the main
    chain's `linear_strength` amount normalisation (target_alpha_mass = 0.5) is
    NOT applied.  EPR-050 already calibrates a single global strength s by
    bisection ([strength] in the config); stacking two strength mechanisms would
    make each one fight the other's target.
    """
    if kind == "semantic":
        # main chain's _semantic_alpha: gaussian blur multiplied BACK by the hard
        # mask, so the edit cannot bleed outside the selected subject instance.
        a = _semantic_alpha(hard, prng)
        return (dict(mode="semantic", mask_type=None, geom=None,
                     subject_area=round(float(area), 6),
                     width_source=GEOM_WIDTH_SOURCE),
                torch.from_numpy(np.ascontiguousarray(a)).to(dev))
    if kind == "radial":
        spec = radial_geom(hard, prng, apply_inside=True)
    elif kind == "band":
        spec = band_geom(hard, prng, apply_inside=True)
    elif kind == "linear":
        rooms = {"left": bbox[0], "right": 1.0 - bbox[2],
                 "top": bbox[1], "bottom": 1.0 - bbox[3]}
        sides = [s for s, room in rooms.items() if room >= 0.20]
        spec = (None if not sides else
                linear_geom(bbox, prng, apply_subject_side=True, area=area,
                            side=prng.choice(sides)))
    else:
        die(f"unknown main-chain geometry kind {kind!r}")
    if spec is None:
        return dict(mode=kind, mask_type=None, geom=None), None

    def raster(g: dict) -> torch.Tensor:
        return torch.from_numpy(np.ascontiguousarray(
            raster_geometry(spec["mask_type"], g, h, w))).to(dev)

    # the recorded parameters ARE the main chain's geometry dict, i.e. the
    # LR/ACR parameter table (Left/Right/Top/Bottom/Angle/Feather/Flipped for
    # circulargradient, ZeroX/ZeroY/FullX/FullY for gradient)
    gp = dict(mode=spec["_mode"], mask_type=spec["mask_type"],
              what=spec.get("what"), geom=dict(spec["geom"]),
              subject_area=round(float(area), 6),
              width_source=GEOM_WIDTH_SOURCE)
    if "_side" in spec:
        gp["side"] = spec["_side"]
    if GEOM_WIDTH_SOURCE != "config":
        return gp, raster(spec["geom"])

    # ---- v3.6: redraw the WIDTH parameter at the SAME placement ------------
    # The placement (centre / angle / subject side) is the main chain's and is
    # never resampled here.  Only the width-class parameter moves: `Feather` for
    # circulargradient, the Zero->Full ramp length for gradient.  The first draw
    # that satisfies the geometry criteria is taken; if the whole width budget is
    # spent the last draw is returned with its diagnosis and the caller's attempt
    # loop redraws everything (that is the "宽度档打满才整体重抽" rule).
    # v3.6.1: the radial size floor is drawn ONCE per attempt (it is a size, not
    # a width: the width retries below only redraw `Feather`).  Only `radial` is
    # affected -- band's major axis is BAND_AXIS_LEN = 1.6 and already spans the
    # frame.  Disabled (0.0) consumes no rng, which is what keeps the archived
    # v3.6 config bit-reproducible.
    base_geom = dict(spec["geom"])
    axis_scale = 1.0
    axis_floor = None
    if kind == "radial" and RADIAL_AXIS_MIN[1] > 0.0:
        axis_floor = float(prng.uniform(*RADIAL_AXIS_MIN))
        base_geom, axis_scale = _grow_ellipse(base_geom, axis_floor)
        gp.update(geom=dict(base_geom), axis_min_frac=round(axis_floor, 4),
                  axis_scale=round(axis_scale, 4))

    trace = []
    g = dict(base_geom)
    m = None
    for t in range(1, GEOM_WIDTH_TRIES + 1):
        if spec["mask_type"] == "circulargradient":
            wval = float(prng.uniform(*GEOM_FEATHER))
            g = dict(base_geom)
            g["Feather"] = round(wval, 3)
        else:
            wval = float(prng.uniform(*GEOM_LINEAR_RAMP))
            g = _set_ramp_len(base_geom, wval)
        m = raster(g)
        why, st = geom_step_why(m, kind)
        trace.append(dict(width=round(wval, 4), why=why,
                          coverage=st["coverage"],
                          full_frac_of_mask=st["full_frac_of_mask"]))
        gp.update(geom=dict(g), width_value=round(wval, 4), width_tries=t,
                  width_trace=trace, width_ok=not why, shape=st)
        if not why:
            break
    return gp, m


def geom_alpha_from_row(row: dict, h: int, w: int, dev: str,
                        subject_hard: np.ndarray | None = None) -> torch.Tensor:
    """Rebuild a journalled geometry alpha WITHOUT resampling anything.

    v3.5 / v3.6 rows carry the main chain's own geometry dict, so the rasteriser
    is replayed on the recorded LR parameters (`Left/Right/Top/Bottom/Angle/
    Feather/Flipped` or `ZeroX/ZeroY/FullX/FullY`) and the result is exact.  The
    `semantic` family has no geometry dict -- its alpha is `_semantic_alpha`'s
    blur of the hard subject mask, whose radius is drawn from the content-keyed
    prng, so it is replayed with the SAME key, including the accepted attempt
    index that the row records.  Legacy (v3.1-v3.4) rows go through geo_mask().
    """
    m = row["mask"]
    gp = m.get("geom_params") or {}
    if not gp.get("mode"):                       # v3.1-v3.4 legacy shapes
        subj = (torch.from_numpy(subject_hard).to(dev)
                if subject_hard is not None else None)
        return geo_mask(h, w, m["geom"], gp, dev, subj)
    if gp["mode"] == "semantic":
        if subject_hard is None:
            die(f"{row['id']}: semantic alpha needs the hard subject mask")
        # the key is rebuilt from the row itself (not from the process-wide
        # REPEAT_SALT, which a drawing tool does not carry): a repeat row's id is
        # "<source_id>.rep<k>" while its content key is "<source_id>:rep<k>"
        rid = row["id"]
        sid, _, rp = rid.partition(".rep")
        key = ":".join([SAMPLE_SALT, sid + (f":rep{rp}" if rp else ""),
                        "geom", str(int(m["attempts"]))])
        prng = random.Random(int(hashlib.sha1(key.encode()).hexdigest()[:16], 16))
        a = _semantic_alpha(subject_hard, prng)
        return torch.from_numpy(np.ascontiguousarray(a)).to(dev)
    a = raster_geometry(gp["mask_type"], gp["geom"], h, w)
    return torch.from_numpy(np.ascontiguousarray(a)).to(dev)


def geo_mask(h: int, w: int, kind: str, p: dict, dev: str,
             subject: torch.Tensor | None = None) -> torch.Tensor:
    yy = torch.linspace(0, 1, h, device=dev).view(h, 1).expand(h, w)
    xx = torch.linspace(0, 1, w, device=dev).view(1, w).expand(h, w)
    if kind == "radial":
        d = (((xx - p["cx"]) / p["rx"]) ** 2 + ((yy - p["cy"]) / p["ry"]) ** 2).sqrt()
        return 1.0 - smoothstep((d - (1.0 - p["feather"])) / p["feather"])
    if kind == "linear":
        # offset_ramp (v3.1 lineage).  A ONE-SIDED smoothstep whose cut line sits
        # at `offset` of the projection span -- despite the old config value being
        # called "band", it is not a band: there is no centre line and no
        # two-sided falloff.  Kept only to reproduce v3.1-v3.4 data.
        # The gradient is parameterised in PIXELS, not in normalised frame units,
        # so "feather >= 60% of the long side" is a statement about the picture
        # rather than about the aspect ratio.
        th = p["angle"]
        proj = xx * (w - 1) * np.cos(th) + yy * (h - 1) * np.sin(th)
        # normalise the *origin* over the frame: an un-normalised projection puts
        # the whole image on one side of the offset for angles in the third
        # quadrant and alpha comes out identically 0 (hit 4/5 of the first v2 smoke)
        lo, hi = proj.min(), proj.max()
        cut = lo + p["offset"] * (hi - lo)
        return smoothstep((proj - cut) / p["feather_px"] + 0.5)
    if kind == "subject":
        if subject is None:
            die("subject geometry requested without a subject mask")
        return feather(subject, p["feather_px"])
    die(f"unknown geometry {kind}")


def load_subject(path: Path, h: int, w: int, dev: str) -> torch.Tensor:
    with Image.open(path) as im:
        m = im.convert("L").resize((w, h), Image.BILINEAR)
    return torch.from_numpy(np.asarray(m, dtype=np.float32) / 255.0).to(dev)


# --------------------------------------------------------------------------- #
# stats
# --------------------------------------------------------------------------- #
def band_stats(err: torch.Tensor, sel: torch.Tensor) -> dict:
    """err: (M,) 8-bit levels (per-pixel RGB L-infinity); sel: (M,) bool."""
    n = int(sel.sum())
    if n == 0:
        return dict(n=0)
    e = err[sel].float()
    qs = torch.quantile(e, torch.tensor([0.5, 0.95, 0.99], device=e.device))
    return dict(n=n, p50=round(float(qs[0]), 6), p95=round(float(qs[1]), 6),
                p99=round(float(qs[2]), 6), max=round(float(e.max()), 6),
                mean=round(float(e.mean()), 6))


def alpha_bands(a: torch.Tensor) -> dict:
    return {"a0": a == 0, "mid": (a > 0) & (a < 1), "a1": a == 1}


def gamut_cover(x: torch.Tensor, sel: torch.Tensor, n: int) -> float:
    if int(sel.sum()) == 0:
        return 0.0
    idx = (x[sel] * (n - 1)).floor().clamp(0, n - 2).to(torch.int64)
    packed = idx[:, 0] * (n - 1) ** 2 + idx[:, 1] * (n - 1) + idx[:, 2]
    return round(float(torch.unique(packed).numel() / (n - 1) ** 3), 8)


def sat_frac(x: torch.Tensor) -> float:
    """Fraction of pixels with any channel at the 8-bit rail."""
    return float(((x >= SAT_HI) | (x <= SAT_LO)).any(-1).float().mean())


def sat_contrast(x: torch.Tensor) -> tuple[float, float]:
    """v3.1 side columns (recorded, never a criterion).

    saturation = mean HSV S = (max-min)/max ; contrast = std of Rec.709 luminance.
    """
    cmax = x.amax(-1)
    cmin = x.amin(-1)
    sat = float(((cmax - cmin) / cmax.clamp_min(1e-6)).mean())
    return sat, float(luminance(x).std())


def de00(a: torch.Tensor, b: torch.Tensor) -> np.ndarray:
    """CIEDE2000 between two (M,3) RGB tensors in [0,1].  skimage, as in
    dataset_build/tools/build_numeric_clusters.py:334."""
    from skimage.color import deltaE_ciede2000, rgb2lab
    la = rgb2lab(a.clamp(0, 1).detach().cpu().numpy().astype(np.float64).reshape(1, -1, 3))
    lb = rgb2lab(b.clamp(0, 1).detach().cpu().numpy().astype(np.float64).reshape(1, -1, 3))
    return deltaE_ciede2000(la.reshape(-1, 3), lb.reshape(-1, 3))


def run_chain(vols, alphas, x: torch.Tensor, s: float = 1.0) -> torch.Tensor:
    """Forward five-step chain at global strength s (alphas scaled by s)."""
    y = x
    for k in range(N_STEPS):
        a = alphas[k] if s == 1.0 else alphas[k] * s
        y = mix_alpha(y, chunked(lambda t, v=vols[k]: apply_lut(v, t), y), a)
    return y


def de00_target_for(source_id: str) -> float:
    """A20: per-sample amplitude target, drawn U(DE00_LO, DE00_DRAW_HI).

    A single fixed target would pin all 100 samples to the same edit amplitude and
    flatten the strength spectrum of the dataset; the cap stays at DE00_DRAW_HI so
    the amplitude ceiling the band expresses is still enforced.  Keyed on content
    (sha1 rule family, per CLAUDE.md) so the draw is deterministic and does not
    shift when the source list changes.
    """
    h = int(hashlib.sha1(
        f"{DE00_SALT}:{source_id}{_rep()}".encode()).hexdigest()[:8], 16)
    return DE00_DRAW_LO + (h / float(1 << 32)) * (DE00_DRAW_HI - DE00_DRAW_LO)


def calibrate_strength(vols, alphas, xf: torch.Tensor, acted: torch.Tensor,
                       rng, target: float) -> dict:
    """v3.1: bisect one global strength s in (0,1] onto the median CIEDE2000 of
    the acted region, targeting the band [DE00_LO, DE00_HI].

    Only ever compresses.  If the edit at s == 1 is already at or below the top of
    the band it is left alone (`s == 1`), and if it is below the bottom of the band
    it is still left alone and flagged `under_target` -- weak edits are never
    amplified.  This mirrors the `weak` branch of solve_normalisation() in
    dataset_build/tools/build_numeric_clusters.py:352-380.

    The chain is pointwise (every step is a LUT plus a per-pixel alpha, no spatial
    coupling), so bisecting on a random pixel subset is the same computation as
    bisecting on the whole frame restricted to that subset; the achieved value is
    then re-measured on the full acted region.
    """
    idx = torch.nonzero(acted, as_tuple=False)[:, 0]
    if idx.numel() > DE00_SUB:               # only bites on very large frames
        sel = torch.from_numpy(
            rng.choice(idx.numel(), size=DE00_SUB, replace=False)).to(idx.device)
        idx = idx[sel]
    xs = xf[:, idx]
    als = [a[:, idx] for a in alphas]

    def de_at(s: float) -> float:
        return float(np.median(de00(xs[0], run_chain(vols, als, xs, s)[0])))

    de_full = de_at(1.0)
    it = 0
    if de_full <= target:
        # Only ever compresses: an edit already at or below its own target is left
        # at full strength rather than amplified.
        s, ach = 1.0, de_full
    else:
        lo, hi, s, ach = 0.0, 1.0, 1.0, de_full
        for it in range(1, BISECT_ITERS + 1):
            s = (lo + hi) / 2.0
            ach = de_at(s)
            # The band check is part of the stop rule, not just the tolerance: a
            # target drawn at the very bottom of the range could otherwise stop at
            # target - DE00_TOL, i.e. just under the band floor.
            if abs(ach - target) <= DE00_TOL and DE00_LO <= ach <= DE00_HI:
                break
            if ach < target:
                lo = s
            else:
                hi = s
    return dict(s=round(float(s), 8), de00_acted_at_s1=round(de_full, 6),
                de00_bisect_achieved=round(ach, 6), bisect_iters=it,
                de00_target=round(float(target), 6),
                under_target=bool(de_full < target),
                below_band_at_s1=bool(de_full < DE00_LO),
                over_target_at_s1=bool(de_full > target),
                calib_px=int(idx.numel()), target_band=[DE00_LO, DE00_HI],
                draw_range=[DE00_DRAW_LO, DE00_DRAW_HI])


_VIRIDIS = np.array([
    [68, 1, 84], [72, 40, 120], [62, 74, 137], [49, 104, 142], [38, 130, 142],
    [31, 158, 137], [53, 183, 121], [109, 205, 89], [180, 222, 44], [253, 231, 37],
], dtype=np.float32)


def colorize(err: np.ndarray, vmax: float | None = None) -> np.ndarray:
    """Fixed absolute scale [0, vmax] in 8-bit levels.  Never per-image min-max."""
    vmax = ERR_VMAX if vmax is None else vmax
    t = np.clip(err / vmax, 0.0, 1.0) * (len(_VIRIDIS) - 1)
    i = np.floor(t).astype(np.int32).clip(0, len(_VIRIDIS) - 2)
    f = (t - i)[..., None]
    return (_VIRIDIS[i] * (1 - f) + _VIRIDIS[i + 1] * f).astype(np.uint8)


def to_png(arr: torch.Tensor) -> Image.Image:
    return Image.fromarray((arr.clamp(0, 1).cpu().numpy() * 255.0).round().astype(np.uint8))


def gray_png(arr: torch.Tensor) -> Image.Image:
    return Image.fromarray((arr.clamp(0, 1).cpu().numpy() * 255.0).round().astype(np.uint8))


# --------------------------------------------------------------------------- #
# pools / sources
# --------------------------------------------------------------------------- #
def ckey(s: str) -> int:
    """N3: content sampling key.  sha1 rule family, per CLAUDE.md.

    Deliberately NOT affected by --repeat-salt: source ordering and the
    main/ctrl pool assignment must stay pinned to the source itself, so every
    repeat of one source lands in the same pool and the source list does not
    reshuffle between repeats.
    """
    return int(hashlib.sha1(f"{SAMPLE_SALT}:{s}".encode()).hexdigest()[:8], 16)


def _rep() -> str:
    """Content-key suffix for a repeat run.  Empty string when --repeat-salt is
    absent, so every key below is byte-identical to the pre-repeat tool."""
    return "" if REPEAT_SALT is None else f":rep{REPEAT_SALT}"


def sample_gid(source_id: str) -> str:
    """Row id / asset prefix.  Unchanged without --repeat-salt."""
    return source_id if REPEAT_SALT is None else f"{source_id}.rep{REPEAT_SALT}"


def chain_key(source_id: str) -> int:
    """Per-sample rng key.  Equals ckey(source_id) without --repeat-salt; with a
    salt it moves, which is exactly what makes a repeat an independent chain."""
    return int(hashlib.sha1(
        f"{SAMPLE_SALT}:{source_id}{_rep()}".encode()).hexdigest()[:8], 16)


def split_bucket(source_id: str) -> int:
    return int(hashlib.sha1(f"{SPLIT_SEED}:{source_id}".encode()).hexdigest()[:8], 16) % 100


def load_pools():
    rows = json.load(open(POOL_JSON))
    by_name = {r["name"]: r for r in rows}
    # v3.2: pool widened to recovered >= 0.95 (420 LUTs, 8 majors with >= 5, vs 179
    # / 6 at the 0.99 cut).  clip == 0 stays: a grid with no node pinned at 0 or 1
    # cannot destroy highlight or shadow detail by saturation.
    main = [r for r in rows if r["recovered"] >= POOL_REC_MIN
            and r["clip"] <= POOL_CLIP_MAX]
    ctrl = [r for r in rows if r["recovered"] < POOL_CTRL_REC_MAX]

    def group(rs):
        d = {}
        for r in rs:
            d.setdefault(r["major"], []).append(r["name"])
        return {k: sorted(v) for k, v in d.items() if len(v) >= N_STEPS}
    return by_name, group(main), group(ctrl)


def rec_band(rec: float) -> str:
    """v3.4: three main-pool bands after [pool] recovered_min dropped to 0.85.

    `0.50-0.85` exists only to name the gap: nothing lands in it, because the
    main pool starts at 0.85 and the control pool stops at 0.50.
    """
    if rec >= 0.99:
        return "ge0.99"
    if rec >= 0.95:
        return "0.95-0.99"
    if rec >= 0.85:
        return "0.85-0.95"
    if rec < 0.5:
        return "lt0.50"
    return "0.50-0.85"


def build_source_pool(cache_path: Path) -> list[dict]:
    """Sources that have a ready subject mask AND are in the S-split train half.

    `source_path` in the subject cache is a prefix key into the databuild archive,
    not necessarily a local file -- resolution goes through archive_reader and the
    path is never "corrected".
    """
    if cache_path.exists():
        return json.loads(cache_path.read_text())
    tbl = {}
    if SPLIT_TABLE.exists():
        tbl = {r["source_id"]: r["split"] for r in csv.DictReader(open(SPLIT_TABLE))}
    rows, unresolved, holdout = [], 0, 0
    for d in sorted(SUBJECT_CACHE.iterdir()):
        j = d / "subject.json"
        if not j.exists():
            continue
        try:
            m = json.loads(j.read_text())
        except Exception:
            continue
        if m.get("status") != "ready" or not (d / "subject.png").exists():
            continue
        sp, sid = m.get("source_path"), m.get("asset_id")
        if not sp or not sid:
            continue
        if not (os.path.exists(sp) or AR.path_exists(sp)):
            unresolved += 1
            continue
        # frozen side table is authoritative when it knows the id; otherwise the
        # documented inline sha1 rule (tools/data_splits/README.md), never ad-hoc
        if sid in tbl:
            split, rule = tbl[sid], "frozen_table"
        else:
            split = "train" if split_bucket(sid) <= TRAIN_BUCKET_MAX else "holdout"
            rule = "sha1_verasplit-v1"
        if split != "train":
            holdout += 1
            continue
        rows.append(dict(source_id=sid, source_path=sp, subject_png=str(d / "subject.png"),
                         subject_area=m.get("area"), split_rule=rule,
                         split_bucket=split_bucket(sid)))
    rows.sort(key=lambda r: (ckey(r["source_id"]), r["source_id"]))
    cache_path.write_text(json.dumps(rows))
    print(f"source pool: {len(rows)} train sources with subject masks "
          f"(unresolved {unresolved}, non-train {holdout})", flush=True)
    return rows


def open_source(sp: str, max_side: int) -> np.ndarray:
    im = AR.open_rgb(sp) if not os.path.exists(sp) else Image.open(sp).convert("RGB")
    with im:
        if max_side and max(im.size) > max_side:
            s = max_side / max(im.size)
            im = im.resize((round(im.width * s), round(im.height * s)), Image.LANCZOS)
        return np.asarray(im.convert("RGB"), dtype=np.uint8)


# --------------------------------------------------------------------------- #
# one sample
# --------------------------------------------------------------------------- #
def build_one(rec, z, by_name, pool_by_major, pool_tag, rng, dev, out_assets,
              max_side, subject_prob, want_naive):
    sid = rec["source_id"]
    gid = sample_gid(sid)          # == sid unless --repeat-salt is given
    src = open_source(rec["source_path"], max_side)
    h, w = src.shape[:2]
    x0 = torch.from_numpy(src.astype(np.float32) / 255.0).to(dev)
    subj_full = load_subject(Path(rec["subject_png"]), h, w, dev)
    # v3.5: the main-chain samplers work off the HARD subject mask plus its bbox
    # and area, exactly as canonical_masks.build_mask_plan does.
    hard_np = (subj_full > 0.5).float().cpu().numpy()
    ys_, xs_ = np.nonzero(hard_np > 0.5)
    if len(xs_):
        bbox = (float(xs_.min()) / w, float(ys_.min()) / h,
                float(xs_.max() + 1) / w, float(ys_.max() + 1) / h)
    else:
        bbox = (0.0, 0.0, 1.0, 1.0)
    subj_area = float(hard_np.mean())

    # ---- five alpha fields, all computed from the before image x0 -----------
    # Rejection rule (recorded, not silent): a masked step whose alpha has no
    # untouched region or no soft band measures nothing, so the mask parameters
    # are redrawn.  Step 4 is alpha == 1 everywhere by spec and is exempt.
    rej: dict[str, int] = {}
    last_bad: dict = {}
    sem_dropped = False              # v3.6 semantic gate, see the draw below
    # v3.4: the hue field has no sampled parameters, so it is computed once
    # outside the rejection loop (it is the same on every attempt)
    m_hue = hue_mask(x0) if "hue" in STEP_KIND else None
    for attempt in range(1, MASK_ATTEMPTS + 1):
        # v3.1: wider soft ramps, so each luminance band can clear the 35% area floor
        q_lo = float(rng.uniform(*LUM_Q_LO))
        q_hi = float(rng.uniform(*LUM_Q_HI))
        width = float(rng.uniform(*LUM_WIDTH))
        bands, (t_lo, t_hi) = lum_bands(x0, q_lo, q_hi, width)

        if GEOM_SAMPLER == "mainchain_weights":
            # v3.5: one categorical draw over the four MAIN-CHAIN families.
            # Deliberately a DIFFERENT rng consumption pattern from the v3.1-v3.4
            # two-stage draw, which is why the sampler is an explicit config key.
            # v3.6 semantic gate (replaces "semantic is exempt from everything"):
            # the subject has to own at least [masks] semantic_cover_min of the
            # frame before the semantic family is allowed at all.  When it does
            # not, that family is DROPPED from the candidate set and the
            # remaining weights are renormalised -- counted, never silent.
            # semantic_cover_min = 0.0 (the archived v3.5 pin) never drops it,
            # so the draw is bit-identical to v3.5.
            names = list(GEOM_WEIGHTS)
            sem_dropped = bool(SEMANTIC_COVER_MIN > 0 and "semantic" in names
                               and subj_area < SEMANTIC_COVER_MIN)
            if sem_dropped:
                names = [k for k in names if k != "semantic"]
            wts = np.array([GEOM_WEIGHTS[k] for k in names], dtype=np.float64)
            kind = str(rng.choice(names, p=wts / wts.sum()))
            # The main-chain samplers take a python random.Random.  Seed it from
            # the same sha1 content-key family as everything else, plus the
            # attempt index so a rejected geometry really is redrawn.
            prng = python_rng(sid, "geom", attempt)
            gp, m_geo = mainchain_geom(kind, hard_np, bbox, subj_area, prng, h, w, dev)
            if m_geo is None:
                # the main-chain sampler declared the geometry infeasible for this
                # subject (e.g. no side with >= 0.20 room); counted, never silent
                rej["geometry_unsatisfied"] = rej.get("geometry_unsatisfied", 0) + 1
                last_bad = dict(step="geom", kind=kind, why="mainchain_returned_none")
                continue
        else:
            use_subject = bool(rng.random() < subject_prob)
            kind = "subject" if use_subject else str(rng.choice(["linear", "radial"]))
            if kind == "radial":
                # v3.2: feather floor relaxed by the same factor as the linear one
                gp = dict(cx=float(rng.uniform(*RAD_CENTER)),
                          cy=float(rng.uniform(*RAD_CENTER)),
                          rx=float(rng.uniform(*RAD_RADIUS)),
                          ry=float(rng.uniform(*RAD_RADIUS)),
                          feather=float(rng.uniform(*RAD_FEATHER)))
            elif kind == "linear":
                # v3.2: the feather is sampled in units of the SHORT side,
                # [10%, 30%].  The "traverses >= 60% of the long side" requirement
                # is no longer a feather-width rule (it would contradict this
                # range); it is checked directly on the acted region's span below.
                ref = min(h, w) if LIN_REF == "short" else max(h, w)
                fpx = float(rng.uniform(*LIN_FEATHER)) * ref
                fpx = max(fpx, LIN_MIN_SHORT * min(h, w))
                gp = dict(angle=float(rng.uniform(0, 2 * np.pi)),
                          offset=float(rng.uniform(*LIN_OFFSET)),
                          feather_px=round(fpx, 3),
                          feather_frac_long=round(fpx / max(h, w), 4),
                          feather_frac_short=round(fpx / min(h, w), 4))
            else:
                gp = dict(feather_px=int(round(max(h, w)
                                               * float(rng.uniform(*SUBJ_FEATHER)))),
                          subject_png=rec["subject_png"],
                          subject_area=rec["subject_area"])
            m_geo = geo_mask(h, w, kind, gp, dev, subj_full)
        use_subject = (kind in ("subject", "semantic"))

        # v3.3: the alpha fields themselves are unchanged -- the luminance bands
        # are still computed from the BEFORE image x0 -- only the order in which
        # the chain consumes them comes from [run] step_order.
        by_kind = dict(lum_high=bands["lum_high"], lum_mid=bands["lum_mid"],
                       lum_shadow=bands["lum_shadow"],
                       **{"global": torch.ones((h, w), device=dev)}, geom=m_geo)
        if "hue" in STEP_KIND:
            # v3.4: content-determined, no sampled parameters -- it is identical
            # on every attempt, so it never drives the rejection loop
            by_kind["hue"] = m_hue
        fields = [by_kind[k] for k in STEP_KIND]
        bad = {}
        for k, f in enumerate(fields):
            if STEP_KIND[k] == "global":                 # alpha == 1 by spec
                continue
            if STEP_KIND[k] == "geom":
                # v3.6: one place decides what a geometry alpha must satisfy, so
                # the width loop inside mainchain_geom and this acceptance test
                # can never disagree.  For every criterion that existed before
                # v3.6 the content is identical to the generic branch below.
                why, st = geom_step_why(f, kind)
                if why:
                    bad["geom"] = dict(st, why=why)
                continue
            fz = float((f == 0).float().mean())
            fm = float(((f > 0) & (f < 1)).float().mean())
            cov = float((f > COVER_EPS).float().mean())
            why = []
            # v3.4 (A41 ruling): the hue step is exempt from ALL THREE shape
            # criteria, because for this step every one of them is a statement
            # about the picture's colour content rather than about the mask
            # being malformed:
            #   zero_band  -- a fully saturated picture has no grey pixel, so the
            #                 hue alpha legitimately has no exact-zero region;
            #   soft_band  -- a greyscale picture gates to alpha == 0 everywhere,
            #                 i.e. the hue step is a no-op, which is correct
            #                 (nobody opens the HSL panel on a black-and-white
            #                 frame);
            #   coverage   -- the acted area is decided by how much colour of the
            #                 weighted hues is actually in the frame.
            # Enforcing them would systematically drop the fully-grey and
            # fully-saturated ends of the source distribution, i.e. bake a
            # content bias into training data, which is worse than the resulting
            # vacuous assertion.  The band pixel counts are recorded either way
            # (see `alpha.zero_band_px` / `soft_band_px`) so the fact that the
            # alpha == 0 bit-exactness guard is vacuous on a given row is public.
            hue_step = STEP_KIND[k] == "hue"
            if not hue_step and fz < MASK_MIN_BAND:
                why.append("zero_band")
            if not hue_step and fm < MASK_MIN_BAND:
                why.append("soft_band")
            # v3.1 area floor.  (The geometry step is handled above by
            # geom_step_why(); what is left here is the luminance bands and the
            # hue step, and the hue step is exempt by the A41 ruling.)
            if not hue_step and cov < COVER_MIN:
                why.append("coverage")
            if why:
                bad[STEP_KIND[k]] = dict(frac_zero=round(fz, 6), frac_mid=round(fm, 6),
                                         coverage=round(cov, 6), why=why)
        if not bad:
            break
        key = ",".join(sorted(bad))
        rej[key] = rej.get(key, 0) + 1
        last_bad = bad
        # v3.4: the hue field is content-only -- it is byte-identical on every
        # attempt -- so if it is what failed, all MASK_ATTEMPTS retries are
        # guaranteed to fail the same way.  Stop now instead of burning 40 draws,
        # and say so in the diagnosis rather than reporting "39 failures" for
        # what is really one deterministic failure.  The accepted set is
        # unchanged: this source was going to be excluded either way, and each
        # source draws from its own rng, so nothing downstream shifts.
        if "hue" in bad:
            lum = luminance(x0).float().flatten()
            raise MaskRejected(gid, dict(
                attempts=attempt, last_bad=bad, reject_by_step=rej,
                deterministic_step="hue",
                lum_std=round(float(lum.std()), 6),
                lum_q25=round(float(torch.quantile(lum, 0.25)), 6),
                lum_q75=round(float(torch.quantile(lum, 0.75)), 6),
                width_range=list(LUM_WIDTH), min_band=MASK_MIN_BAND,
                cover_min=COVER_MIN, cover_eps=COVER_EPS))
    else:
        lum = luminance(x0).float().flatten()
        raise MaskRejected(gid, dict(
            attempts=MASK_ATTEMPTS, last_bad=last_bad, reject_by_step=rej,
            lum_std=round(float(lum.std()), 6),
            lum_q25=round(float(torch.quantile(lum, 0.25)), 6),
            lum_q75=round(float(torch.quantile(lum, 0.75)), 6),
            width_range=[0.12, 0.24], min_band=MASK_MIN_BAND,
            cover_min=COVER_MIN, cover_eps=COVER_EPS))

    # measured on the accepted, pre-scale mask: multiplying by s < 1 would shrink
    # the alpha > COVER_EPS set and understate the geometry's reach
    geom_span = span_long_frac(fields[STEP_KIND.index("geom")])
    # v3.6: the shape numbers the criteria are actually written in are the
    # PRE-SCALE ones (the criteria run inside the rejection loop, before the
    # global strength s multiplies every field).  Recording the post-scale field
    # instead would report full_frac_of_mask ~ 0 for every s < 1 row, which is a
    # statement about the amplitude calibration, not about the mask's shape.
    geom_shape_pre = {k: round(v, 6) for k, v in geom_shape_stats(
        fields[STEP_KIND.index("geom")]).items()}
    # v3.4 hue census (recorded, never a criterion): how much of the frame is
    # chromatic at all, and where it sits on the hue circle.  This is what makes
    # "this source has blue in it / this one does not" checkable from the journal.
    hue_info = None
    if m_hue is not None:
        hh, ss = hsv_hue_sat(x0)
        gated = smoothstep((ss - HUE_SAT_LO) / (HUE_SAT_HI - HUE_SAT_LO))
        chroma = gated > 0
        nch = int(chroma.sum())
        bins = {"red_0_60": (0, 60), "yellow_60_120": (60, 120),
                "green_120_180": (120, 180), "cyan_180_230": (180, 230),
                "blue_230_300": (230, 300), "magenta_300_360": (300, 360)}
        hue_info = dict(
            sat_lo=HUE_SAT_LO, sat_hi=HUE_SAT_HI,
            anchors=[[d, w] for d, w in HUE_ANCHORS],
            chroma_frac=round(float(chroma.float().mean()), 6),
            gate_zero_frac=round(float((gated == 0).float().mean()), 6),
            weight_mean=round(float(m_hue.mean()), 6),
            weight_max=round(float(m_hue.max()), 6),
            # hue histogram over the CHROMATIC pixels only (grey pixels have no hue)
            hue_frac={k: (round(float((((hh >= lo) & (hh < hi)) & chroma).float()
                                      .sum() / nch), 6) if nch else 0.0)
                      for k, (lo, hi) in bins.items()},
            # share of the whole frame that is both chromatic and blue-ish
            blue_frac_frame=round(float((((hh >= 230) & (hh < 300)) & chroma)
                                        .float().mean()), 6))
    alphas = [f.reshape(1, -1, 1).contiguous() for f in fields]

    # ---- five LUTs, same major, all distinct --------------------------------
    majors = sorted(pool_by_major)
    if FORCE_MAJOR is not None:
        # CLI run control: pin the style so repeat diversity can only come from
        # LUT order / masks / strength.  Absent -> the uniform draw below, unchanged.
        if FORCE_MAJOR not in pool_by_major:
            die(f"--force-major {FORCE_MAJOR!r}: not a usable major of the "
                f"{pool_tag} pool (usable: {majors})")
        majors = [FORCE_MAJOR]
    major = majors[int(rng.integers(len(majors)))]
    names = pool_by_major[major]
    pick = rng.choice(len(names), size=N_STEPS, replace=False)
    luts = [names[int(i)] for i in pick]
    grids = [int(z[n].shape[0]) for n in luts]
    vols = [torch.from_numpy(z[n][None]).to(dev, torch.float32)
            .permute(0, 4, 1, 2, 3).contiguous() for n in luts]

    xf = x0.reshape(1, -1, 3)
    base_sat = sat_frac(xf[0])

    # ---- v3.1 coverage census + global strength calibration -----------------
    cover_pre = [round(float((a[0, :, 0] > COVER_EPS).float().mean()), 6) for a in alphas]
    # Acted region per spec = composite alpha > COVER_EPS.  Step 4 is alpha == 1
    # everywhere, so that set is the whole frame by construction; the fraction is
    # recorded anyway so the degeneracy is visible, and a "local" variant that
    # drops the global step is recorded next to it.
    comp = torch.stack([a[0, :, 0] for a in alphas], 0).amax(0)
    comp_local = torch.stack([alphas[k][0, :, 0] for k in range(N_STEPS)
                              if STEP_KIND[k] != "global"], 0).amax(0)
    acted, acted_local = comp > COVER_EPS, comp_local > COVER_EPS
    calib = calibrate_strength(vols, alphas, xf, acted, rng, de00_target_for(sid))
    calib.update(acted_frac=round(float(acted.float().mean()), 6),
                 acted_local_frac=round(float(acted_local.float().mean()), 6))
    s_glob = calib["s"]
    if s_glob != 1.0:
        # F_{s*alpha} is the same family of blend maps, so every downstream step
        # (inversion, the alpha==0 bit-exactness guards) is unaffected; alpha==0
        # stays exactly 0 because s*0 == 0.
        alphas = [a * s_glob for a in alphas]
        fields = [f * s_glob for f in fields]

    # ---- forward chain ------------------------------------------------------
    ys = [xf]
    steps = []
    for k in range(N_STEPS):
        prev = ys[-1]
        y = mix_alpha(prev, chunked(lambda t, v=vols[k]: apply_lut(v, t), prev), alphas[k])
        # guard: alpha == 0 must not move a single bit, forward direction
        z0 = (alphas[k][0, :, 0] == 0)
        if int(z0.sum()) and float((y[0][z0] - prev[0][z0]).abs().max()) != 0.0:
            die(f"{gid}: step {k+1} moved pixels with alpha == 0")
        s = sat_frac(y[0])
        a = alphas[k][0, :, 0]
        steps.append(dict(
            step=k + 1, kind=STEP_KIND[k], lut=luts[k], grid=grids[k],
            recovered=by_name[luts[k]]["recovered"], clip=by_name[luts[k]]["clip"],
            de_med=by_name[luts[k]]["de_med"],
            alpha=dict(mean=round(float(a.mean()), 6),
                       frac_zero=round(float((a == 0).float().mean()), 6),
                       frac_mid=round(float(((a > 0) & (a < 1)).float().mean()), 6),
                       frac_one=round(float((a == 1).float().mean()), 6),
                       # A13/A41: absolute pixel counts, not just fractions, so a
                       # reader can see when a per-step guard is asserted over an
                       # EMPTY set (0 px) instead of that fact hiding behind a
                       # rounded 0.000000 fraction.  band_exempt marks the steps
                       # whose band criteria were not enforced.
                       zero_band_px=int((a == 0).sum()),
                       soft_band_px=int(((a > 0) & (a < 1)).sum()),
                       # D1 (v3.6): the geometry step is also marked exempt when
                       # its family is on [masks] zero_band_exempt_geoms, so a
                       # vacuous alpha==0 assertion is never read as a pass.
                       band_exempt=bool(STEP_KIND[k] == "hue"
                                        or (STEP_KIND[k] == "geom"
                                            and kind in ZERO_BAND_EXEMPT))),
            sat_frac=round(s, 8), sat_delta=round(s - base_sat, 8),
            sat_warn=bool(s - base_sat > SAT_WARN),
            gamut_cover=gamut_cover(prev[0], a > 0, grids[k]),
            coverage_pre_scale=cover_pre[k],
            coverage=round(float((a > COVER_EPS).float().mean()), 6),
            # which steps the COVER_MIN floor was not applied to (subject geom,
            # and the v3.4 hue step); the coverage itself is reported either way
            cover_exempt=bool(STEP_KIND[k] == "hue"
                              or (STEP_KIND[k] == "geom" and use_subject)),
        ))
        ys.append(y)
    after = ys[-1]

    # ---- v3.1 achieved amplitude + side columns (recorded, not a criterion) --
    calib["de00_acted_after"] = round(
        float(np.median(de00(xf[0][acted], after[0][acted]))), 6)
    calib["de00_local_acted_after"] = round(
        float(np.median(de00(xf[0][acted_local], after[0][acted_local]))), 6)
    calib["in_band"] = bool(DE00_LO <= calib["de00_acted_after"] <= DE00_HI)
    s_b, c_b = sat_contrast(xf[0])
    s_a, c_a = sat_contrast(after[0])
    side = dict(sat_mean_before=round(s_b, 6), sat_mean_after=round(s_a, 6),
                sat_mean_delta=round(s_a - s_b, 6),
                sat_mean_ratio=round(s_a / max(s_b, 1e-6), 6),
                contrast_before=round(c_b, 6), contrast_after=round(c_a, 6),
                contrast_delta=round(c_a - c_b, 6),
                contrast_ratio=round(c_a / max(c_b, 1e-6), 6))

    # ---- path E: unwind in reverse, solving F_{a_k}(x) = y_k -----------------
    est = [None] * (N_STEPS + 1)
    est[N_STEPS] = after
    for k in range(N_STEPS - 1, -1, -1):
        xh, iters = chunked(lambda t, a, v=vols[k]: invert_blend(v, t, a),
                            est[k + 1], alphas[k])
        est[k] = xh
        resid = (mix_alpha(xh, chunked(lambda t, v=vols[k]: apply_lut(v, t), xh),
                           alphas[k]) - est[k + 1]).abs().amax(-1)[0] * LEVEL
        fail = resid > CONV_TOL
        # N1: a failed pixel sitting on the boundary of [0,1]^3 has no preimage
        # inside the gamut; a failed interior pixel is the solver giving up.
        on_edge = ((xh[0] <= 0) | (xh[0] >= 1)).any(-1)
        d_err = (xh - ys[k]).abs().amax(-1)[0] * LEVEL
        ab = alpha_bands(alphas[k][0, :, 0])
        # guard: alpha == 0 must come back bit-exact, backward direction
        z0 = ab["a0"]
        if int(z0.sum()) and float((xh[0][z0] - est[k + 1][0][z0]).abs().max()) != 0.0:
            die(f"{gid}: step {k+1} inverse moved pixels with alpha == 0")
        steps[k].update(
            inv_iters=iters,
            conv_fail_frac=round(float(fail.float().mean()), 8),
            conv_fail_sat_frac=round(float((fail & on_edge).float().mean()), 8),
            conv_fail_interior_frac=round(float((fail & ~on_edge).float().mean()), 8),
            resid_p95=round(float(torch.quantile(resid.float(), 0.95)), 6),
            # depth-resolved error: |x_hat_k - y_k| right after unwinding step k+1
            depth_err={b: band_stats(d_err, s) for b, s in ab.items()},
            depth_err_all=band_stats(d_err, torch.ones_like(z0)),
        )

    # ---- path N (control column, first --n-naive samples only) --------------
    n_stats = None
    if want_naive:
        cur = after
        for k in range(N_STEPS - 1, -1, -1):
            cur = mix_alpha(cur, chunked(lambda t, v=vols[k]: invert_lut(v, t), cur),
                            alphas[k])
        errn = (cur - xf).abs().amax(-1)[0] * LEVEL
        # banded on the GEOMETRY step's alpha (the last step under the v3.2 order,
        # the first under v3.3) so the control column stays comparable across
        # step orders rather than following whichever step happens to be last
        n_stats = {b: band_stats(errn, s) for b, s in
                   alpha_bands(alphas[STEP_KIND.index("geom")][0, :, 0]).items()}
        n_stats["all"] = band_stats(errn, torch.ones(errn.shape, dtype=torch.bool, device=dev))

    # ---- final error, stratified by every step's alpha band -----------------
    err_e = (est[0] - xf).abs().amax(-1)[0] * LEVEL
    all_zero = torch.ones_like(err_e, dtype=torch.bool)
    for a in alphas:
        all_zero &= (a[0, :, 0] == 0)
    # NOTE: step 4 is alpha == 1 everywhere, so this set is empty by construction.
    # The meaningful bit-exactness guard is the per-step one asserted above.
    if int(all_zero.sum()) and float(err_e[all_zero].max()) != 0.0:
        die(f"{gid}: composite alpha==0 region not bit-exact")

    err_by_step = {}
    for k, a in enumerate(alphas):
        err_by_step[f"step{k+1}_{STEP_KIND[k]}"] = {
            b: band_stats(err_e, s) for b, s in alpha_bands(a[0, :, 0]).items()}

    row = dict(
        id=gid, source_path=rec["source_path"], subject_png=rec["subject_png"],
        subject_area=rec["subject_area"], split_rule=rec["split_rule"],
        split_bucket=rec["split_bucket"], size=[h, w], pool=pool_tag, major=major,
        luts=luts, grids=grids, rec_band=rec_band(min(s["recovered"] for s in steps)),
        mask=dict(q_lo=q_lo, q_hi=q_hi, width=width, t_lo=t_lo, t_hi=t_hi,
                  geom=kind, geom_params=gp, uses_subject=use_subject,
                  attempts=attempt, subject_exempt_from_coverage=use_subject,
                  cover_min=COVER_MIN, cover_eps=COVER_EPS,
                  geom_span_long=round(geom_span, 6), span_long_min=SPAN_LONG_MIN,
                  # v3.6 shape record: the accepted geometry alpha's own numbers
                  # (frame coverage, and the full-strength / transition split
                  # measured INSIDE the mask), plus the thresholds in force.
                  geom_shape=geom_shape_pre,          # pre-scale = 判据口径
                  geom_shape_post_scale={k: round(v, 6) for k, v in
                                         geom_shape_stats(
                                             fields[STEP_KIND.index("geom")]
                                         ).items()},
                  full_eps=FULL_EPS, full_frac_of_mask_max=FULL_OF_MASK_MAX,
                  semantic_gate=dict(cover_min=SEMANTIC_COVER_MIN,
                                     subject_area=round(subj_area, 6),
                                     dropped=sem_dropped),
                  hue=hue_info),
        calib=calib, side=side,
        steps=steps,
        base_sat_frac=round(base_sat, 8),
        sat_warn_steps=[s["step"] for s in steps if s["sat_warn"]],
        composite_alpha0_px=int(all_zero.sum()),
        err_norm="per-pixel RGB L-inf, 8-bit levels",
        err_E_all=band_stats(err_e, torch.ones_like(all_zero)),
        err_E_by_step_alpha=err_by_step,
        err_N=n_stats,
        conv_fail_frac_chain=round(float(max(s["conv_fail_frac"] for s in steps)), 8),
    )

    imgs = {"src": to_png(x0), "after": to_png(after.reshape(h, w, 3)),
            "restored_e": to_png(est[0].reshape(h, w, 3)),
            "err_e": Image.fromarray(colorize(err_e.reshape(h, w).cpu().numpy()))}
    for k, f in enumerate(fields):
        imgs[f"a{k+1}"] = gray_png(f)
    if want_naive:
        imgs["err_n"] = Image.fromarray(
            colorize(errn.reshape(h, w).cpu().numpy()))
        imgs["restored_n"] = to_png(cur.reshape(h, w, 3))
    for k, v in imgs.items():
        v.save(out_assets / f"{gid}.{k}.png")
    del vols
    return row


# --------------------------------------------------------------------------- #
# montage
# --------------------------------------------------------------------------- #
_FONTS = ["/home/bc/.local/share/fonts/windows/msyh.ttc",
          "/home/bc/.local/share/fonts/windows/STXIHEI.TTF",
          "/usr/share/fonts/truetype/arphic/uming.ttc"]


def _font(size: int):
    from PIL import ImageFont
    for p in _FONTS:
        if Path(p).exists():
            try:
                return ImageFont.truetype(p, size)
            except Exception:
                pass
    return ImageFont.load_default()


KIND_LABEL = {"lum_high": "highlight", "lum_mid": "midtone",
              "lum_shadow": "shadow", "global": "global=1", "geom": "geom",
              "hue": "hue"}


def cols() -> list[str]:
    """One alpha column per chain step, so the sheet widens with chain_len."""
    return (["src"] + [f"a{k+1}" for k in range(len(STEP_KIND))]
            + ["after", "restored_e", "err_e", "err_n"])


def titles() -> dict:
    """a1..aN are chain POSITIONS, so their labels follow [run] step_order."""
    t = {"src": "before", "after": f"after ({len(STEP_KIND)} steps)",
         "restored_e": "restore E", "err_e": "err E", "err_n": "err N"}
    for k, kind in enumerate(STEP_KIND):
        t[f"a{k+1}"] = f"a{k+1} {KIND_LABEL[kind]}"
    return t


def montage(rows, assets, out_path, panel_w=230):
    fh_, fb = _font(12), _font(14)
    COLS = cols()
    tiles, hs = [], []
    for r in rows:
        row = []
        for c in COLS:
            p = assets / f"{r['id']}.{c}.png"
            if not p.exists():
                row.append(None)
                continue
            im = Image.open(p).convert("RGB")
            s = panel_w / im.width
            row.append(im.resize((panel_w, max(1, round(im.height * s))), Image.LANCZOS))
        hs.append(max(i.height for i in row if i is not None))
        tiles.append(row)
    hdr, cap = 22, 56
    W, H = panel_w * len(COLS), hdr + sum(h + cap for h in hs)
    sheet = Image.new("RGB", (W, H), "white")
    d = ImageDraw.Draw(sheet)
    TITLES = titles()
    for k, c in enumerate(COLS):
        d.text((k * panel_w + 4, 3), TITLES[c], fill="black", font=fb)
    bar = colorize(np.linspace(0, ERR_VMAX, 96)[None].repeat(11, 0))
    for k, c in enumerate(COLS):
        if c.startswith("err"):
            x0 = k * panel_w + panel_w - 150
            sheet.paste(Image.fromarray(bar), (x0, 5))
            d.text((x0 - 10, 3), "0", fill="black", font=fh_)
            d.text((x0 + 100, 3), f"{ERR_VMAX:.0f}+ lv", fill="black", font=fh_)
    y = hdr
    for r, row, h in zip(rows, tiles, hs):
        for k, im in enumerate(row):
            if im is None:
                d.text((k * panel_w + panel_w // 2 - 12, y + h // 2), "n/a",
                       fill="gray", font=fb)
            else:
                sheet.paste(im, (k * panel_w, y))
                # step 4's alpha is 1 everywhere, i.e. a pure white panel; without
                # a frame it is indistinguishable from an empty cell
                d.rectangle([k * panel_w, y, k * panel_w + panel_w - 1,
                             y + im.height - 1], outline=(150, 150, 150))
        d.text((4, y + h + 3),
               f"{r['id']} pool={r['pool']} major={r['major']} rec_band={r['rec_band']} "
               f"geom={r['mask']['geom']} subject={r['mask']['uses_subject']} "
               f"luts={','.join(x[-6:] for x in r['luts'])} "
               f"| sat_base={r['base_sat_frac']:.4f} sat_warn_steps={r['sat_warn_steps']} "
               f"| composite_a0_px={r['composite_alpha0_px']}"
               + ("" if "calib" not in r else
                  f" | s={r['calib']['s']:.4f} dE00 {r['calib']['de00_acted_at_s1']:.2f}"
                  f"->{r['calib']['de00_acted_after']:.2f} (tgt "
                  f"{r['calib']['de00_target']:.2f}, local "
                  f"{r['calib']['de00_local_acted_after']:.2f}) in_band="
                  f"{r['calib']['in_band']} under={r['calib']['under_target']}"),
               fill="black", font=fh_)
        e = r["err_E_all"]
        d.text((4, y + h + 18),
               f"restore E vs before, per-pixel RGB L-inf (8-bit levels): "
               f"p50 {e['p50']:.4g} | p95 {e['p95']:.4g} | p99 {e['p99']:.4g} | max {e['max']:.4g}"
               + ("" if not r["err_N"] else
                  f"      N: p50 {r['err_N']['all']['p50']:.4g} | p95 "
                  f"{r['err_N']['all']['p95']:.4g} | max {r['err_N']['all']['max']:.4g}"),
               fill="black", font=fh_)
        d.text((4, y + h + 33),
               "per-step  " + "   ".join(
                   f"{s['step']}{s['kind'][:4]}: it={s['inv_iters']} "
                   f"fail={s['conv_fail_frac']:.1e}(sat {s['conv_fail_sat_frac']:.1e}) "
                   f"cov={s.get('coverage_pre_scale', float('nan')):.2f} "
                   f"dsat={s['sat_delta']:+.4f}" for s in r["steps"])
               + ("" if "side" not in r else
                  f"   | dSat={r['side']['sat_mean_delta']:+.4f} "
                  f"dContrast={r['side']['contrast_delta']:+.4f}"),
               fill="black", font=fh_)
        y += h + cap
    sheet.save(out_path, quality=92)


# --------------------------------------------------------------------------- #
# main
# --------------------------------------------------------------------------- #
def main():
    ap = argparse.ArgumentParser()
    # Run control only.  Everything with experiment semantics lives in --config;
    # there is no CLI override for those, so a run cannot silently disagree with
    # the manifest that gets frozen next to its output.
    ap.add_argument("--config", required=True,
                    help="experiment TOML; every numeric parameter is read from it")
    ap.add_argument("--out", required=True)
    ap.add_argument("--n", type=int, default=None,
                    help="override [run] n for a smoke; nothing else is overridable")
    ap.add_argument("--smoke", action="store_true",
                    help="shorthand for --n 5 --no-montage")
    ap.add_argument("--montage-dir",
                    default=str(REPO / "docs/assets/epr050_degrade_20260825"))
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--montage-rows", type=int, default=20)
    ap.add_argument("--montage-ctrl", type=int, default=6)
    ap.add_argument("--no-montage", action="store_true")
    # --- repeat controls (CLI run control, NOT experiment semantics) ----------
    # They carry no numeric parameter of the experiment, so they stay out of the
    # TOML and out of the resume identity; without them the tool behaves exactly
    # as before (see NOTES A47's no-drift check).
    ap.add_argument("--repeat-salt", type=int, default=None,
                    help="degrade the same source again as an INDEPENDENT chain: "
                         "the per-sample rng key becomes sha1(sample_salt:<id>:rep<k>) "
                         "and the row id becomes <id>.rep<k>.  Absent = unchanged.")
    ap.add_argument("--force-major", default=None,
                    help="pin every chain to this major (must be a usable major of "
                         "the pool the row lands in).  Absent = the uniform "
                         "over-major draw, unchanged.")
    ap.add_argument("--only-sources", default=None,
                    help="comma-separated source_ids; restrict the walked source "
                         "pool to these (content-key order preserved).  Absent = "
                         "the whole pool, i.e. unchanged.")
    ap.add_argument("--montage-only", action="store_true",
                    help="redraw the sheet from an existing pairs.jsonl; computes "
                         "and appends nothing, so the resume identity guard is "
                         "not applicable")
    args = ap.parse_args()

    global REPEAT_SALT, FORCE_MAJOR
    REPEAT_SALT = args.repeat_salt
    FORCE_MAJOR = args.force_major
    cfg_path = Path(args.config)
    cfg = load_config(cfg_path)
    cfg_text = cfg_path.read_text()
    n_target = args.n if args.n is not None else _int(cfg, "run", "n")
    ctrl_frac = _num(cfg, "pool", "ctrl_frac")
    # legacy sampler only; under mainchain_weights the kind comes from geom_weights
    subject_prob = (_num(cfg, "masks", "subject_prob")
                    if GEOM_SAMPLER == "subject_prob" else None)
    max_side = _int(cfg, "render", "max_side")
    n_naive = _int(cfg, "run", "n_naive")
    if args.smoke:
        n_target = 5 if args.n is None else n_target
        args.no_montage = True

    dev = args.device if torch.cuda.is_available() else "cpu"
    out = Path(args.out)
    assets = out / "assets"
    assets.mkdir(parents=True, exist_ok=True)

    # B3 + v3.2: freeze the run identity.  The whole config file is stored inside
    # the manifest, so a resume is validated on config *content*, not on the CLI
    # line -- renaming or editing the TOML in place is caught.
    ident = dict(config=cfg,
                 config_sha256=hashlib.sha256(cfg_text.encode()).hexdigest(),
                 config_path=str(cfg_path),
                 tool_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                 bank_mtime=os.path.getmtime(f"{BANK}/luts.npz"),
                 pool_sha256=hashlib.sha256(POOL_JSON.read_bytes()).hexdigest())
    ap_path = out / "run_args.json"
    if args.montage_only:
        rows = [json.loads(l) for l in open(out / "pairs.jsonl")]
        md = Path(args.montage_dir)
        md.mkdir(parents=True, exist_ok=True)
        ctrl = [r for r in rows if r["pool"] == "ctrl"][: args.montage_ctrl]
        mainr = [r for r in rows if r["pool"] == "main"][: args.montage_rows - len(ctrl)]
        pick = sorted(ctrl + mainr, key=lambda r: (r["pool"], r["id"]))
        p = md / f"montage_v3_n{len(pick)}.jpg"
        montage(pick, out / "assets", p)
        print(f"montage-only: {p}  (ctrl {len(ctrl)} / main {len(mainr)})", flush=True)
        return
    if ap_path.exists():
        old = json.loads(ap_path.read_text())
        drift = {k: (old.get(k), ident[k]) for k in ident
                 if k not in ("config", "config_path") and old.get(k) != ident[k]}
        # config content diff, section by section, so the message names the key
        cdrift = {}
        oc = old.get("config", {})
        for sec in sorted(set(oc) | set(cfg)):
            for key in sorted(set(oc.get(sec, {})) | set(cfg.get(sec, {}))):
                a, b = oc.get(sec, {}).get(key), cfg.get(sec, {}).get(key)
                if a != b:
                    cdrift[f"{sec}.{key}"] = (a, b)
        if drift or cdrift:
            die(f"resume refused: run identity changed.\n  meta drift: {drift}\n"
                f"  config drift: {cdrift}\n  use a fresh --out")
    else:
        ap_path.write_text(json.dumps(ident, ensure_ascii=False, indent=1))
    # The repeat controls are recorded next to the identity but are NOT part of it:
    # several --repeat-salt runs are meant to append to one journal, so comparing
    # them would refuse the second run.  Kept as an append-only ledger.
    if args.repeat_salt is not None or args.only_sources or args.force_major:
        led_path = out / "repeat_runs.json"
        led = json.loads(led_path.read_text()) if led_path.exists() else []
        led.append(dict(repeat_salt=args.repeat_salt,
                        only_sources=args.only_sources,
                        force_major=args.force_major, n=n_target,
                        tool_sha256=ident["tool_sha256"],
                        config_sha256=ident["config_sha256"],
                        ts=time.strftime("%Y-%m-%dT%H:%M:%S")))
        led_path.write_text(json.dumps(led, ensure_ascii=False, indent=1))

    checks = self_check(dev)
    (out / "self_check.json").write_text(json.dumps(checks, indent=1))

    by_name, main_pool, ctrl_pool = load_pools()
    print(f"pools: main {len(main_pool)} majors / {sum(map(len, main_pool.values()))} luts; "
          f"ctrl {len(ctrl_pool)} majors / {sum(map(len, ctrl_pool.values()))} luts", flush=True)

    # The pool is walked in content-key order and consumed until `--n` samples are
    # ACCEPTED; a source whose luminance spread cannot satisfy the pre-registered
    # band criterion is excluded (skipped.json) and the next source takes its slot.
    srcs = build_source_pool(out / "source_pool.json")
    if args.only_sources:
        want = [s for s in args.only_sources.split(",") if s]
        have = {r["source_id"] for r in srcs}
        missing = [s for s in want if s not in have]
        if missing:
            die(f"--only-sources: not in the source pool: {missing}")
        srcs = [r for r in srcs if r["source_id"] in set(want)]
        print(f"--only-sources: {len(srcs)} sources kept", flush=True)
    if len(srcs) < n_target:
        print(f"WARNING: only {len(srcs)} subject-masked train sources available "
              f"for n {n_target}", flush=True)

    z = np.load(f"{BANK}/luts.npz")
    pairs = out / "pairs.jsonl"
    done = set()
    if pairs.exists():
        for line in open(pairs):
            try:
                done.add(json.loads(line)["id"])
            except Exception:
                pass
    print(f"resume: {len(done)} rows already present", flush=True)
    skip_path = out / "skipped.json"
    skipped = json.loads(skip_path.read_text()) if skip_path.exists() else []
    skipped_ids = {s["id"] for s in skipped}

    t0 = time.time()
    k = len(done)                      # accepted-row counter, drives --n-naive
    with open(pairs, "a") as fh:
        for rec in srcs:
            if k >= n_target:
                break
            gid = sample_gid(rec["source_id"])
            if gid in done:
                continue
            if gid in skipped_ids:
                continue
            # N3: pool assignment and rng seed keyed on content, not position, so
            # the sample set does not shift when the source list changes.
            # --repeat-salt moves only the rng key (chain_key), never the pool key
            # (ckey), so every repeat of a source stays in the same pool.
            key = ckey(rec["source_id"])
            is_ctrl = (key % 1000) < round(ctrl_frac * 1000)
            rng = np.random.default_rng(SEED + chain_key(rec["source_id"]))
            try:
                row = build_one(rec, z, by_name, ctrl_pool if is_ctrl else main_pool,
                                "ctrl" if is_ctrl else "main", rng, dev, assets,
                                max_side, subject_prob, k < n_naive)
            except MaskRejected as e:
                # counted and excluded, never silent: the band criterion is kept
                # exactly as pre-registered and the next source takes the slot
                skipped.append(dict(id=e.gid, source_path=rec["source_path"],
                                    pool="ctrl" if is_ctrl else "main",
                                    reason="mask_band_criterion", **e.diag))
                skipped_ids.add(e.gid)
                skip_path.write_text(json.dumps(skipped, ensure_ascii=False, indent=1))
                print(f"  SKIP {e.gid}: {json.dumps(e.diag, ensure_ascii=False)}",
                      flush=True)
                continue
            fh.write(json.dumps(row, ensure_ascii=False) + "\n")
            fh.flush()
            os.fsync(fh.fileno())
            k += 1
            print(f"  [{k}/{n_target}] {row['id']} {row['pool']} {row['major']} "
                  f"geom={row['mask']['geom']} E.p50={row['err_E_all']['p50']:.4g} "
                  f"E.p95={row['err_E_all']['p95']:.4g} "
                  f"maxfail={row['conv_fail_frac_chain']:.2e} ({time.time()-t0:.1f}s)",
                  flush=True)
    print(f"skipped {len(skipped)} sources (mask band criterion) -> {skip_path}",
          flush=True)

    rows = [json.loads(l) for l in open(pairs)]
    # N5: uniform-over-major sampling is a design choice (coverage of every
    # major); record what it actually drew against the pool sizes.
    # A major name can exist in BOTH pools, so samples/LUTs are counted per pool
    # and the pool size is reported per pool too.  (Reporting one combined
    # `distinct_luts` against only the main pool's size made majors like
    # 暖调高亮 read as "38 distinct LUTs out of a pool of 6".)
    draw = {}
    for r in rows:
        d = draw.setdefault(r["major"], {})
        e = d.setdefault(r["pool"], {"samples": 0, "luts": set()})
        e["samples"] += 1
        e["luts"].update(r["luts"])
    summary = {}
    for m, per in draw.items():
        rec = {}
        for pl, v in per.items():
            src = main_pool if pl == "main" else ctrl_pool
            rec[pl] = dict(samples=v["samples"], distinct_luts=len(v["luts"]),
                           pool_size=len(src.get(m, [])))
        rec["samples_total"] = sum(v["samples"] for v in per.values())
        summary[m] = rec
    # what the >= chain_len major threshold actually left usable, per the v3.4
    # ruling ("实际可用数落盘")
    summary["_usable"] = dict(
        chain_len=N_STEPS,
        main_majors=len(main_pool), main_luts=sum(len(v) for v in main_pool.values()),
        ctrl_majors=len(ctrl_pool), ctrl_luts=sum(len(v) for v in ctrl_pool.values()),
        main_pool_sizes={k: len(v) for k, v in sorted(main_pool.items())},
        ctrl_pool_sizes={k: len(v) for k, v in sorted(ctrl_pool.items())})
    (out / "major_draw.json").write_text(json.dumps(summary, ensure_ascii=False, indent=1))
    print(f"=== {len(rows)} rows, {time.time()-t0:.1f}s ===", flush=True)
    print("major draw:", json.dumps(summary, ensure_ascii=False), flush=True)

    if not args.no_montage:
        md = Path(args.montage_dir)
        md.mkdir(parents=True, exist_ok=True)
        # B2: stratified pick, otherwise the control pool fills the whole sheet
        ctrl = [r for r in rows if r["pool"] == "ctrl"][: args.montage_ctrl]
        mainr = [r for r in rows if r["pool"] == "main"][: args.montage_rows - len(ctrl)]
        pick = sorted(ctrl + mainr, key=lambda r: (r["pool"], r["id"]))
        p = md / f"montage_v3_n{len(pick)}.jpg"
        montage(pick, assets, p)
        print(f"montage: {p}  (ctrl {len(ctrl)} / main {len(mainr)})", flush=True)


if __name__ == "__main__":
    main()
