#!/usr/bin/env python
"""Runner for EPR-028 R1 (G4D): the conditional RGB-``s`` carrier.

Spec: ``experiments/prs/EPR-028_4d-gaussian-conditional-slice/PROPOSAL_R1.md``
-- **the only implementation authority**.  ``PROPOSAL.md`` (R0) survives only in
its cross-arm frozen block, its §1.2 data counts and its §4 external-fact record.

    CUDA_VISIBLE_DEVICES=0 PYTHONPATH=/home/bc/VeraRetouch \\
    /home/bc/envs/q3vl_sft/bin/python -m q3vl.whatb.scripts.run_g4d_arm \\
        --gate-stage G1 --glut4d-mode A1 --out /home/bc/data/runs/epr028_g1

What R1 changed in *this* file (the carrier changed in ``arms/g4d.py``)
-----------------------------------------------------------------------
* **paired anchors** (§8.1): a step is ``L LUTs x Q colours x 4 s-anchors`` with
  ``s in {0, 1, u, 1-u}`` per colour, not ``s ~ U(0,1)`` drawn independently per
  point.  Hard mining is therefore done **per colour group** -- selecting a
  colour keeps all four of its anchors, so an endpoint is never dropped alone.
* **``R_line``** (§4.4) replaces R0's deleted ``L_m4d`` and costs **zero extra
  forwards**: ``f(x,0)`` and ``f(x,1)`` are already in the batch.
* **image formation per arm** (§5): every ``compose_headline`` call now carries
  ``mode=``.  R0's unconditional outer ``mix_alpha`` on top of a carrier that had
  already eaten ``s`` was the double-alpha bug (§0.1).
* **condition source** (§2.2 / §6): ``--cond oracle`` is a learnable
  ``nn.Embedding(n_lut, cond_dim)`` indexed by ``lut_id`` (Experiment C); the
  ``z``-cache path is ``--cond vlm`` (Experiment Z).  An oracle board is
  ``published = false`` and carries ``oracle_reference = true``: its headline is
  a learnable approximation of ``B4_oracle`` and may not be read next to any
  arm's headline (§9.1).
* **gate stages** (§9): ``--gate-stage {G1,G2,G3}`` is a preset that pins
  ``(LUT pool, per step, steps, board)`` in one place and writes the resolution
  into ``run_setup.json``.  G1/G2 need neither the ``z`` cache nor images nor a
  predicted field -- only the preset bank ``luts.npz``.
* **regulariser cost** (§8.3): ``L_s4d`` differences 256 random 4D base points
  per sample; the ``17^4`` lattice is evaluation-only.  ``L_img`` samples
  ``--img-pixels`` pixels on at most ``--img-batch`` samples.
* **NaN re-check every quick eval** (§10): a NaN prediction scores ``dE00 = 0.0``
  rather than NaN, so a guard that only looks once publishes a ``headline = 0.0``
  board.  Every quick eval re-checks ``L_rec``; a non-finite value stops the run
  with a non-zero rc and writes ``void_reason`` into ``metrics.json``.

What it does NOT own
--------------------
The ``z`` cache is :mod:`q3vl.whatb.zcache` -- one implementation for all six
arms; ``ConditionStore`` here is a five-line adapter onto it, not a second
reader.  The image / GT-alpha loader for this arm's own eval bundle
(``EvalSample``) is still local.

Everything is device-explicit and nothing calls ``.cpu()`` on a metric path.
"""

from __future__ import annotations

import argparse
import hashlib
import io
import json
import math
import random
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import torch

from q3vl.whatb import caliber as K
from q3vl.whatb import colorspan, criteria, publish, queries, splits
from q3vl.whatb.arms import g4d
from q3vl.whatb.guards import (
    DegeneracyThresholds,
    DegenerateTransform,
    degeneracy_check_ran,
    record_step_witness,
)
from q3vl.whatb.lutdata import BANK_DIR, LutBank, apply_lut_volume, mix_alpha
from q3vl.whatb.readout import WhatReadoutSpec
from q3vl.whatb.zcache import ZCacheDir
from q3vl.where.upsample import area_resize   # frozen block's resampling operator

BASE_CKPT = "/home/bc/data/runs/q3vl_base_sft_v2seg_20260814/checkpoint-4976"
MASKVIEW_ROOT = Path("/mnt/nfs-ro/bc/data/datasets/where_a-20260805/maskviews")
#: frozen block: 93934 train normal-only, 32x256 = 8192 colours, 2936 steps per
#: epoch, 40 epochs, 117,440 steps.  This is a *record* of the口径 the published
#: boards were run on (``splits.TRAIN_NORMAL_N`` = the v20260804 version's n),
#: not the horizon: the run's own horizon is measured from the active口径 and
#: ``measured.train_matches_frozen`` says whether the two agree.
FROZEN = {
    "train_normal_only_n": splits.TRAIN_NORMAL_N,
    "batch_samples": queries.BATCH_SAMPLES,
    "queries_per_sample": queries.QUERIES_PER_SAMPLE,
    "colors_per_step": queries.COLORS_PER_STEP,
    "steps_per_epoch": math.ceil(splits.TRAIN_NORMAL_N / queries.BATCH_SAMPLES),
    "epochs": 40,
    "total_steps": math.ceil(splits.TRAIN_NORMAL_N / queries.BATCH_SAMPLES) * 40,
    "clamp_default": "two",
    # R1 §5 replaced R0's single outer compositor: the formation is per arm.
    "headline_formation": ("per arm (R1 §5): A0 = I + S * [T(I) - I]; "
                           "A1/A2/A3 = f(I, S)"),
}
CONTROL_TAGS = ("none", "shuffle", "irrelevant", "const")
_CONTROL_COLUMN = {"shuffle": "N1_shuffle", "irrelevant": "N2_irrelevant",
                   "const": "N3_const"}

#: R1 §8.1: every colour carries exactly four ``s`` anchors ``{0, 1, u, 1-u}``.
S_ANCHORS: int = 4
#: exit code of the R1 §10 NaN re-check.  Distinct from the degeneracy guard's
#: ``2`` so a log line says which gate stopped the run.
RC_NONFINITE: int = 3

#: R1 §9's gate ladder, as one table.  ``--gate-stage`` resolves this into
#: ``(n_lut_pool, b_samples, q_colors, total_steps, needs_eval_board)`` and the
#: resolution is written verbatim into ``run_setup.json``.
#:
#: ``n_lut_pool = None`` means "every LUT in the training bucket" (G3).
#: ``colours per step = b_samples * q_colors * S_ANCHORS``:
#: G1 1x2048x4 = 8,192, G2 32x512x4 = 65,536, G3 256x2048x4 = 2,097,152.
#: All three are ``--cond oracle`` and therefore ``published = false`` with
#: ``oracle_reference = true`` (R1 §9.1).
GATE_STAGES: dict[str, dict[str, Any]] = {
    "G1": {"n_lut_pool": 1, "b_samples": 1, "q_colors": 2048,
           "total_steps": 2000, "needs_eval_board": False, "mining": False,
           "note": "single-LUT overfit, no VLM, no mining (R1 §9-G1)"},
    "G2": {"n_lut_pool": 32, "b_samples": 32, "q_colors": 512,
           "total_steps": 4000, "needs_eval_board": False, "mining": True,
           "note": "32-LUT oracle, A1/A2/A3 x 3 seeds (R1 §9-G2)"},
    "G3": {"n_lut_pool": None, "b_samples": 256, "q_colors": 2048,
           "total_steps": 18760, "needs_eval_board": True, "mining": True,
           "note": "full oracle carrier board, 2,097,152 colours/step (R1 §9-G3)"},
}
GATE_STAGE_CHOICES: tuple[str, ...] = ("G1", "G2", "G3", "none")

#: R1 §9.1: G1-G3 are ``E[lut_id]`` runs.  The three negative controls and the
#: predicted field are the **G4** board's columns (they need a text read-out and
#: the where arm's product); an oracle run cannot compute them, can never
#: publish, and records the waiver on its own board rather than pretending.
ORACLE_WAIVED_CRITERIA: tuple[str, ...] = (
    "N1_shuffle_delta", "N1_shuffle_M",
    "N2_irrelevant_delta", "N2_irrelevant_M",
    "N3_const_delta", "N3_const_M",
    "field_pred",
)

#: The degeneracy verdict is only BINDING once the arm has had a real chance to
#: leave its own initialisation.  §3.5 zero-initialises the last layer of every
#: generator head so that step 0 is EXACTLY the identity in all five modes
#: (proposition 2) -- which means ``cross_std`` ("one transform for every
#: sample") is 0 **by construction** at step 0 and only grows as the generator
#: learns z.  Measured on CPU 2026-08-15 with this arm's own optimiser, its own
#: loss and 16 real V_what conditions: cross_std = 0 at step 0, 4e-5 at step 1,
#: and it crosses the 1e-4 floor at step 3 only when a condition is seen
#: repeatedly; the 10-step smoke (32 fresh samples per step, 160 samples total)
#: reached 2.3e-5 at step 5.  The frozen schedule puts the first quick eval at
#: one epoch, which is the horizon the floor was written for.  Below it the
#: guard still RUNS (all three numbers computed, printed, recorded, and the
#: in-process witness set, so ``publish.assert_publishable`` still sees it), but
#: it does not kill the process -- and that is only allowed on a ``--smoke``
#: run, which cannot produce a published board (``board["published"]``).
DEGENERACY_BINDING_MIN_STEPS: int = FROZEN["steps_per_epoch"]


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #
def build_arg_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(
        prog="run_g4d_arm",
        description="EPR-028 R1 G4D: conditional RGB-s carrier (A0..A3)")
    # -- the arm's own flags (R1 §3) --------------------------------------- #
    ap.add_argument("--carrier", choices=("glut4d", "glut3d"), default="glut4d",
                    help="glut3d forces --glut4d-mode A0 (the plain 3D GLUT)")
    ap.add_argument("--glut4d-mode", choices=g4d.G4D_MODES, default="A3",
                    help="R1 §3: A0 3D-LUT / A1 3D-ExplicitGate / A2 4D-BlockDiag "
                         "(beta forced to 0 in the forward) / A3 4D-Joint.  The "
                         "paper question is the single difference A3 - A2")
    ap.add_argument("--glut4d-field", choices=g4d.FIELD_CHOICES, default="gt",
                    help="field the carrier eats; training is fixed to gt, "
                         "evaluation runs all four rows regardless")
    ap.add_argument("--glut4d-n", type=int, default=48)
    ap.add_argument("--glut4d-marg-norm", choices=g4d.MARG_NORM_CHOICES,
                    default="peak",
                    help="R1 §4.2: peak-normalised is the default and the only "
                         "main-arm value; 'full' is a flag-only appendix row")
    ap.add_argument("--clamp", choices=("two", "one"), default="two")
    ap.add_argument("--w-img", type=float, default=0.0)
    ap.add_argument("--alpha-s", type=float, default=0.0)
    ap.add_argument("--lam-hc", type=float, default=g4d.LAMBDA_HC)
    ap.add_argument("--lam-sparse", type=float, default=g4d.LAMBDA_SPARSE)
    ap.add_argument("--lam-line", type=float, default=g4d.LAMBDA_LINE,
                    help="R1 §4.4 R_line weight, pre-registered 0.1 (ablation "
                         "row 0 / 0.1 / 1.0).  0 switches the term OFF and it is "
                         "then not computed at all")
    ap.add_argument("--loss-level", type=int, default=None, choices=(1, 3, 4),
                    help="the GLUT ladder.  Default: the EPR-030 caliber's pure "
                         "L1 (1: lambda_hc = lambda_sparse = 0), or 4 when "
                         "--w-img > 0")
    ap.add_argument("--reg-grid", type=int, default=17,
                    help="side of the 17^4 EVALUATION lattice; L_s4d's training "
                         "step spacing is 1/(reg_grid-1) (R1 §8.3)")
    ap.add_argument("--reg-points", type=int, default=256,
                    help="R1 §8.3: random 4D base points per sample for L_s4d's "
                         "finite differences.  The 17^4 lattice is never a "
                         "per-step training cost")
    ap.add_argument("--img-pixels", type=int, default=768,
                    help="R1 §8.3: pixels sampled per image for L_img (512-1024)")
    ap.add_argument("--img-batch", type=int, default=4,
                    help="R1 §8.3: samples per step that pay the L_img forward")
    # -- condition source (R1 §2.2 / §6) ----------------------------------- #
    ap.add_argument("--cond", choices=("oracle", "vlm"), default="oracle",
                    help="oracle = Experiment C, a learnable nn.Embedding("
                         "n_lut, cond_dim) indexed by lut_id (no z cache, no "
                         "text read-out); vlm = Experiment Z, pi(z_color) out of "
                         "the shared cache.  An oracle board is never published "
                         "and carries oracle_reference = true (R1 §9.1)")
    ap.add_argument("--gate-stage", choices=GATE_STAGE_CHOICES, default="none",
                    help="R1 §9 preset: pins (LUT pool, per step, steps, board) "
                         "in one place; the resolution is written into "
                         "run_setup.json")
    ap.add_argument("--g1-lut-id", default=None,
                    help="the single LUT of --gate-stage G1; default = the first "
                         "lut_id of the training bucket in sorted order")
    # -- shared conditioning / generator ---------------------------------- #
    ap.add_argument("--cond-dim", type=int, default=64)
    ap.add_argument("--gen-width", type=int, choices=(128, 64), default=128)
    ap.add_argument("--readout", default="seg_color",
                    choices=("seg_color", "color_span_pool", "color_close",
                             "im_end", "seg_where", "qtok"))
    ap.add_argument("--base-ckpt", default=BASE_CKPT)
    ap.add_argument("--z-cache", default=None,
                    help="root of the shared z cache (q3vl/whatb/zcache.py): the "
                         "leaves are <root>/<split>__<tag>")
    # -- optimisation (§3.3) ----------------------------------------------- #
    # the ONE table (arms/carrier.py BATCH_SPLITS); a free BxQ outside it still
    # parses (the smoke path this runner already had) via caliber.parse_batch_split
    ap.add_argument("--batch-split", default="32x256")
    ap.add_argument("--epochs", type=int, default=FROZEN["epochs"])
    ap.add_argument("--steps", type=int, default=None,
                    help="override the frozen 117,440 (smoke only; recorded)")
    # one spelling across the six arms; --lr stays as this arm's old name
    ap.add_argument("--base-lr", "--lr", dest="lr", type=float, default=K.BASE_LR)
    ap.add_argument("--pi-lr-scale", type=float, default=0.1)
    ap.add_argument("--max-grad-norm", type=float, default=1.0)
    ap.add_argument("--seed", type=int, default=20260810)
    ap.add_argument("--precision", choices=("bf16", "fp32"), default="bf16")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    # -- data / evaluation -------------------------------------------------- #
    ap.add_argument("--train-split", default="train")
    ap.add_argument("--eval-split", default="V_what")
    ap.add_argument("--lut-bank", default=str(BANK_DIR))
    ap.add_argument("--lut-resample", choices=("none",), default="none")
    ap.add_argument("--lib-sample", type=int, default=1137,
                    help="|Lib_tr| for B1/B2/B4/B6 (§4.C protocol: 1137 of 3149)")
    ap.add_argument("--eval-limit", type=int, default=None)
    ap.add_argument("--bucket-record-limit", type=int, default=0,
                    help="B3 pools from the first N train records (0 = all 93934); "
                         "a truncated pool is recorded on the board")
    ap.add_argument("--eval-short-side", type=int, default=512)
    ap.add_argument("--pred-field-dir", default=None,
                    help="where-arm m_pix products; field_pred is a required "
                         "column, so a published board needs this")
    ap.add_argument("--quick-eval-every", type=int, default=2000)
    ap.add_argument("--quick-eval-n", type=int, default=16)
    ap.add_argument("--degeneracy-point-std", type=float, default=1e-3)
    ap.add_argument("--degeneracy-identity-dev", type=float, default=1e-3)
    ap.add_argument("--degeneracy-cross-std", type=float, default=1e-4)
    # -- plumbing ----------------------------------------------------------- #
    ap.add_argument("--out", required=True)
    ap.add_argument("--smoke", action="store_true",
                    help="tiny CPU run: synthetic conditions are allowed and the "
                         "board is marked published=False")
    ap.add_argument("--eval-only", action="store_true")
    ap.add_argument("--no-train", action="store_true")
    ap.add_argument("--dry-run", action="store_true",
                    help="resolve the caliber, write run_setup.json, then stop")
    # --data / --zcache-root-l8; --batch-split and --base-lr are this runner's
    # own flags (declared above) and are not re-declared here
    K.add_caliber_arguments(ap, batch_split=False, base_lr=False)
    return ap


# --------------------------------------------------------------------------- #
# small helpers
# --------------------------------------------------------------------------- #
def _sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for block in iter(lambda: fh.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def source_freeze() -> dict[str, str]:
    """sha256 of the files this run's behaviour depends on (submit-time freeze)."""
    root = Path(__file__).resolve().parents[1]
    names = ["arms/g4d.py", "scripts/run_g4d_arm.py", "glut.py", "generator.py",
             "criteria.py", "publish.py", "guards.py", "lutdata.py", "queries.py",
             "colorimetry.py", "colorspan.py", "splits.py", "readout.py",
             "zcache.py", "evaldata.py"]
    return {n: _sha256_file(root / n) for n in names if (root / n).exists()}


def write_json(path: Path, obj: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2, ensure_ascii=False, default=str),
                    encoding="utf-8")


def parse_batch_split(text: str) -> tuple[int, int]:
    """``BxQ`` through the ONE shared table (:mod:`q3vl.whatb.caliber`)."""
    return K.parse_batch_split(text)


def resolve_loss_level(args) -> int:
    """The GLUT ladder this run is on.

    R1 §4.4: the main arm is the EPR-030 caliber's **pure L1**
    (:data:`q3vl.whatb.caliber.PURE_L1_LOSS_LEVEL` = 1, so
    ``lambda_hc = lambda_sparse = 0``).  ``--w-img > 0`` still lifts the ladder
    to 4, which is what puts ``L_img`` on the first step row.
    """
    if args.loss_level is not None:
        return int(args.loss_level)
    return 4 if args.w_img > 0 else K.PURE_L1_LOSS_LEVEL


def effective_lambdas(args) -> tuple[float, float]:
    """``(lam_hc, lam_sparse)`` after the ladder gate (``carrier.py:347/:351``).

    At the R1 default level 1 both are ``0.0``; the flags' own values are only
    reached from ``--loss-level 3`` upwards.
    """
    level = resolve_loss_level(args)
    return (K.effective_lambda_hc(args.lam_hc, level),
            K.effective_lambda_sparse(args.lam_sparse, level))


def resolve_gate_stage(args, lut_pool: Sequence[str]) -> dict[str, Any]:
    """R1 §9's preset, resolved against the measured training LUT bucket.

    Returns the full resolution -- pool, per-step structure, step count, board
    policy -- which the caller both *uses* and writes verbatim into
    ``run_setup.json``.  ``--gate-stage none`` returns the flags' own numbers so
    every run records the same shape of block.
    """
    ids = sorted(str(i) for i in lut_pool)
    stage = str(args.gate_stage)
    if stage == "none":
        b_samples, q_colors = parse_batch_split(args.batch_split)
        return {
            "stage": "none", "preset": None, "note": "flags only, no §9 preset",
            "lut_ids": ids, "n_lut_pool": len(ids),
            "b_samples": b_samples, "q_colors": q_colors,
            "s_anchors": S_ANCHORS,
            "colors_per_step": b_samples * q_colors * S_ANCHORS,
            "n_pairs_s": b_samples * q_colors * S_ANCHORS,
            "total_steps": int(args.steps) if args.steps else None,
            "needs_eval_board": True, "mining": True,
            "published": False if args.cond == "oracle" else None,
            "oracle_reference": args.cond == "oracle",
        }
    preset = dict(GATE_STAGES[stage])
    want = preset["n_lut_pool"]
    if want is None:
        picked = ids
    elif stage == "G1":
        first = str(args.g1_lut_id) if args.g1_lut_id else (ids[0] if ids else "")
        if first not in ids:
            raise SystemExit(
                f"--g1-lut-id {first!r} is not in the training LUT bucket "
                f"(n = {len(ids)}); G1 overfits one LUT that the training "
                "population actually contains")
        picked = [first]
    else:
        if len(ids) < int(want):
            raise SystemExit(
                f"--gate-stage {stage} wants {want} LUTs and the training "
                f"bucket has {len(ids)}")
        rng = np.random.default_rng(int(args.seed))
        idx = rng.choice(len(ids), size=int(want), replace=False)
        picked = [ids[int(i)] for i in sorted(idx)]
    b_samples = int(preset["b_samples"])
    q_colors = int(preset["q_colors"])
    total = int(args.steps) if args.steps else int(preset["total_steps"])
    return {
        "stage": stage, "preset": {k: v for k, v in GATE_STAGES[stage].items()},
        "note": preset["note"],
        "lut_ids": picked, "n_lut_pool": len(picked),
        "n_lut_pool_available": len(ids),
        "b_samples": b_samples, "q_colors": q_colors, "s_anchors": S_ANCHORS,
        "colors_per_step": b_samples * q_colors * S_ANCHORS,
        "n_pairs_s": b_samples * q_colors * S_ANCHORS,
        "total_steps": total,
        "steps_overridden": bool(args.steps),
        "needs_eval_board": bool(preset["needs_eval_board"]),
        "mining": bool(preset["mining"]),
        # R1 §9.1: every gate stage is an E[lut_id] run.
        "published": False,
        "oracle_reference": True,
    }


def pred_field_sources(pred_dir: Path | None) -> dict[str, str]:
    """``sample_id -> which product wrote that ``.npy``, per the product itself.

    The ``field_pred`` directory can hold more than one kind of field: the where
    arm predicts only ``render_mode == "local"`` rows, and V_what's ``style``
    rows carry a constant-one field that is the data law (``mask is None ->
    out = edited``), not a prediction.  The product tags every row in its own
    ``per_sample.jsonl``; reading the tag here (instead of re-deriving it from
    ``task_type``) means the board reports what the artefact claims, and an
    untagged file shows up as ``"unlabelled"`` rather than as a guess.
    """
    if pred_dir is None:
        return {}
    path = Path(pred_dir) / "per_sample.jsonl"
    if not path.is_file():
        return {}
    out: dict[str, str] = {}
    with path.open(encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            rec = json.loads(line)
            sid = rec.get("sample_id")
            if sid:
                out[str(sid)] = str(rec.get("source") or "unlabelled")
    return out


def degeneracy_binding(args, total_steps: int) -> tuple[bool, str]:
    """Does the degeneracy verdict kill the process at the first quick eval?

    Binding unless the run is BOTH a ``--smoke`` run AND shorter than
    :data:`DEGENERACY_BINDING_MIN_STEPS` -- see that constant for the measured
    reason.  A short run without ``--smoke`` stays binding: it can publish a
    board, and a published board must never rest on a verdict that was waived.
    """
    if int(total_steps) >= DEGENERACY_BINDING_MIN_STEPS:
        return True, (f"total_steps {total_steps} >= "
                      f"{DEGENERACY_BINDING_MIN_STEPS} (one frozen epoch)")
    if not args.smoke:
        return True, (f"total_steps {total_steps} < "
                      f"{DEGENERACY_BINDING_MIN_STEPS} but this is not a "
                      "--smoke run, so the board can be published and the "
                      "verdict binds")
    return False, (f"--smoke with total_steps {total_steps} < "
                   f"{DEGENERACY_BINDING_MIN_STEPS} (one frozen epoch): the "
                   "generator's zero-initialised heads (§3.5, f = identity at "
                   "step 0) make cross_std ~0 by construction this early, so "
                   "the three numbers are recorded and printed but do not stop "
                   "the run.  A --smoke board is never published.")


def values_to_volume(values: torch.Tensor, n: int) -> torch.Tensor:
    """``(n^3, 3)`` grid values (R,G,B major) -> ``(1, 3, D_b, D_g, D_r)``.

    The storage order matches ``lutdata.apply_lut_volume`` (which mirrors
    ``rendering.py:390-405``'s ``permute(3,0,1,2)`` on a ``grid[b,g,r]`` array),
    so a synthesised transform (B1's library mean) is applied through exactly the
    same trilinear operator as a real preset.
    """
    v = values.reshape(n, n, n, 3).permute(2, 1, 0, 3)      # (r,g,b,·) -> (b,g,r,·)
    return v.permute(3, 0, 1, 2).unsqueeze(0).contiguous()


# --------------------------------------------------------------------------- #
# R1 §10 -- the NaN re-check, and R1 §8.1 -- the paired-anchor sampler
# --------------------------------------------------------------------------- #
class NonFiniteLoss(SystemExit):
    """``L_rec`` went non-finite; the run stops with :data:`RC_NONFINITE`."""

    def __init__(self, void_reason: Mapping[str, Any]) -> None:
        super().__init__(RC_NONFINITE)
        self.void_reason = dict(void_reason)


def assert_l_rec_finite(value: Any, *, step: int, where: str,
                        run_dir: Path | None = None,
                        extra: Mapping[str, Any] | None = None) -> float | None:
    """R1 §10's采信 discipline: re-check ``L_rec`` at EVERY quick eval.

    A NaN prediction scores ``dE00 = 0.0``, not NaN (the Lab conversion and the
    arctan2 of the hue difference both swallow it), so a board built on a NaN
    checkpoint reads as a *perfect* headline.  EPR-030 was bitten twice.  This
    checks the number itself, writes ``void_reason`` into ``metrics.json`` and
    raises :class:`NonFiniteLoss` (``SystemExit(3)``) -- a non-zero rc, never a
    warning.
    """
    if value is None:
        # "not measured" is a different failure from "measured and non-finite";
        # criteria.assert_criteria_ran / assert_publishable own the first one.
        return None
    try:
        v = float(value)
    except (TypeError, ValueError):
        v = float("nan")
    if math.isfinite(v):
        return v
    reason = {"void_reason": "L_rec is not finite",
              "L_rec": None if math.isnan(v) else v,
              "raw": repr(value), "step": int(step), "where": where,
              "rc": RC_NONFINITE,
              "authority": "R1 §10 (a NaN prediction scores dE00 = 0.0, so a "
                           "board built on it publishes headline = 0.0)",
              **dict(extra or {})}
    if run_dir is not None:
        write_json(Path(run_dir) / "metrics.json",
                   {"arm": g4d.ARM, "published": False, "quick": False,
                    "void": True, **reason})
    print(f"[EPR-028] NON-FINITE L_rec at {where} (step {step}): stopping with "
          f"rc={RC_NONFINITE}", file=sys.stderr, flush=True)
    raise NonFiniteLoss(reason)


def paired_anchor_batch(sampler: queries.QuerySampler, b: int, q: int, *,
                        device: torch.device,
                        dtype: torch.dtype = torch.float32
                        ) -> tuple[torch.Tensor, torch.Tensor]:
    """R1 §8.1: ``(x (B, Q*4, 3), s (B, Q*4))`` with four anchors per colour.

    Each colour ``c`` draws one ``u ~ U(0,1)`` and is evaluated at
    ``s in {0, 1, u, 1-u}`` -- the identity endpoint, the LUT endpoint and two
    complementary interior points, all on the *same* colour.  The anchor axis is
    the fastest-varying one, so ``view(B, Q, 4)`` recovers the colour groups.

    The draw comes from the sampler's private generator (the where-side N1
    lesson: adding a draw must not shift any other random decision).
    """
    colors = sampler.sample(b, q, device=device, dtype=dtype)          # (B,Q,3)
    u = torch.rand(b, q, generator=sampler.generator, dtype=dtype).to(device)
    zeros = torch.zeros_like(u)
    s = torch.stack([zeros, zeros + 1.0, u, 1.0 - u], dim=-1)          # (B,Q,4)
    x = colors.unsqueeze(2).expand(b, q, S_ANCHORS, 3).reshape(b, q * S_ANCHORS, 3)
    return x.contiguous(), s.reshape(b, q * S_ANCHORS).contiguous()


def select_hard_colour_groups(err: torch.Tensor, ratio: float, b: int, q: int
                              ) -> torch.Tensor:
    """R1 §8.1's mining granularity: whole colour groups, never single anchors.

    ``err`` is the per-pair error ``(B, Q*4)``.  It is reduced over the anchor
    axis **first** (mean over the four ``s`` of one colour) and
    :func:`q3vl.whatb.queries.select_hard` then runs on the ``(B*Q,)`` colour
    errors, so a selected colour keeps all four of its anchors and an unselected
    one keeps none.  Selecting on the flattened ``(B, Q*4)`` would split colour
    groups and silently drop ``s = 0`` / ``s = 1`` endpoints.
    """
    group = err.reshape(b, q, S_ANCHORS).mean(dim=-1).reshape(-1)      # (B*Q,)
    keep = torch.zeros_like(group, dtype=torch.bool)
    idx = queries.select_hard(group, float(ratio))
    if idx.numel():
        keep[idx] = True
    return keep.reshape(b, q, 1).expand(b, q, S_ANCHORS).reshape(b, q * S_ANCHORS)


def r_line_from_anchors(y_hat: torch.Tensor, s: torch.Tensor, b: int, q: int
                        ) -> torch.Tensor:
    """R1 §4.4's ``R_line``, at **zero extra forward cost**.

    ``y_hat`` is ``(B, Q*4, 3)`` from the paired-anchor batch, so ``f(x,0)`` and
    ``f(x,1)`` are anchors 0 and 1 of the same colour group.  The two interior
    anchors (``u`` and ``1-u``) each get one chord residual and the two are
    averaged.
    """
    y = y_hat.reshape(b, q, S_ANCHORS, 3)
    ss = s.reshape(b, q, S_ANCHORS)
    f0, f1 = y[:, :, 0, :], y[:, :, 1, :]
    return 0.5 * (g4d.r_line(y[:, :, 2, :], f0, f1, ss[:, :, 2])
                  + g4d.r_line(y[:, :, 3, :], f0, f1, ss[:, :, 3]))


def l_s4d_random_points(arm: g4d.G4DArm, params: g4d.G4DParams, *,
                        n_points: int, step: float,
                        generator: torch.Generator,
                        device: torch.device) -> torch.Tensor:
    """R1 §8.3: the 4D smoothness term on RANDOM base points, not on ``17^4``.

    ``17^4 = 83,521`` points per sample per step is 1.5e13 Gaussian-point
    combinations over the frozen horizon before any Cholesky or backward pass.
    This differences ``n_points`` uniformly drawn 4D base points along each of
    the four axes with the lattice's own spacing ``step = 1/(reg_grid - 1)``.
    The full lattice stays on the evaluation path (:func:`g4d.grid_4d`).
    """
    b = params.batch_size
    lo = 1.0 - float(step)
    x = torch.rand(b, int(n_points), 3, generator=generator).to(device) * lo
    s = torch.rand(b, int(n_points), generator=generator).to(device) * lo
    base = arm.transform(params, x, s)
    total = base.new_zeros(())
    for axis in range(3):
        shifted = x.clone()
        shifted[..., axis] = shifted[..., axis] + float(step)
        d = arm.transform(params, shifted, s) - base
        total = total + (d * d).mean()
    d_s = arm.transform(params, x, s + float(step)) - base
    return total + (d_s * d_s).mean()


# --------------------------------------------------------------------------- #
# conditions -- the oracle embedding (R1 §6 Experiment C)
# --------------------------------------------------------------------------- #
class OracleConditionStore(torch.nn.Module):
    """``lut_id -> cond (cond_dim,)`` -- a learnable ``nn.Embedding``.

    R1 §2.2 / §6: Experiment C conditions the generator on the LUT identity, so
    a collapse cannot be blamed on four things at once (``pi(z_color)``, the
    generator, the geometry, the colour heads).  The embedding **is trained**,
    at the generator's own lr -- ``--pi-lr-scale`` belongs to the ``pi(z_color)``
    projection of Experiment Z and does not apply here.

    R1 §9.1: because the condition is the GT ``lut_id``, the headline of any run
    that uses this store is a learnable approximation of ``B4_oracle``.  Such a
    board is ``published = false`` and carries ``oracle_reference = true``; it
    may not be read next to any arm's headline.
    """

    kind = "oracle"
    #: the publication gate reads this on both stores; an oracle store is not
    #: "synthetic" (it is a real, trained condition) -- it is *unpublishable*,
    #: which is a different flag and is set from ``--cond`` at the board.
    synthetic = False

    def __init__(self, lut_ids: Sequence[str], *, cond_dim: int = 64,
                 seed: int = 20260810, init_std: float = 1.0) -> None:
        super().__init__()
        self.lut_ids = [str(i) for i in lut_ids]
        if not self.lut_ids:
            raise ValueError("the oracle condition needs a non-empty LUT pool")
        self.index = {lid: i for i, lid in enumerate(self.lut_ids)}
        self.cond_dim = int(cond_dim)
        self.init_std = float(init_std)
        self.embedding = torch.nn.Embedding(len(self.lut_ids), int(cond_dim))
        gen = torch.Generator().manual_seed(int(seed))
        with torch.no_grad():
            self.embedding.weight.copy_(
                torch.randn(len(self.lut_ids), int(cond_dim), generator=gen)
                * float(init_std))

    def ids_to_index(self, lut_ids: Sequence[str], *,
                     device: Any = "cpu") -> torch.Tensor:
        missing = sorted({str(i) for i in lut_ids} - set(self.index))
        if missing:
            raise KeyError(
                f"{len(missing)} lut_id(s) are outside this run's oracle pool "
                f"(n = {len(self.lut_ids)}); first few: {missing[:5]}")
        return torch.as_tensor([self.index[str(i)] for i in lut_ids],
                               dtype=torch.long, device=device)

    def get(self, lut_ids: Sequence[str], control: str = "none", *,
            device: Any = "cpu", dtype: torch.dtype = torch.float32
            ) -> torch.Tensor:
        if control != "none":
            raise KeyError(
                f"the oracle condition has no {control!r} control: the three "
                "negative controls are text-side and belong to the G4 board "
                "(R1 §9.1)")
        idx = self.ids_to_index(lut_ids, device=self.embedding.weight.device)
        return self.embedding(idx).to(device=device, dtype=dtype)

    def facts(self) -> dict[str, Any]:
        return {"kind": self.kind, "n_lut": len(self.lut_ids),
                "cond_dim": self.cond_dim,
                "n_params": int(self.embedding.weight.numel()),
                "init": f"N(0, {self.init_std}) from a private generator",
                "trained": True, "lr_group": "generator (NOT pi_lr_scale)",
                "synthetic": False, "publishable": False,
                "authority": "R1 §2.2 / §6 Experiment C; §9.1 oracle_reference"}


# --------------------------------------------------------------------------- #
# conditions -- adapter over the ONE shared cache (q3vl/whatb/zcache.py)
# --------------------------------------------------------------------------- #
class ConditionStore:
    """``sample_id -> z (2560,)`` per control tag -- an adapter over the ONE cache.

    The reader, the on-disk layout (``<root>/<split>__<tag>/``) and the three
    start-up assertions (``checkpoint``, ``readout_kind``, a 1% ``verify_plan``
    replay of the recorded read-out positions) all live in
    :mod:`q3vl.whatb.zcache`; the frozen block says 共同依赖只写一份 and this arm
    used to carry its own ``<split>.<control>.pt`` reader.

    ``synthetic=True`` (``--smoke`` with no cache) derives a deterministic
    pseudo-condition from the sample id: it is NOT a read-out, it exists so the
    CPU plumbing test has distinct conditions per sample, and it forces
    ``published=False`` on any board.
    """

    def __init__(self, root: str | Path | None, *, split: str, checkpoint: str,
                 readout_kind: str, dim: int = 2560, synthetic: bool = False,
                 tags: Sequence[str] = ("none", "shuffle", "irrelevant", "const"),
                 required: Sequence[str] = ("none",), data: str = "v2seg",
                 zcache_root_l8: str | Path | None = None,
                 seed: int = 20260810) -> None:
        self.root = Path(root) if root else None
        self.split = split
        self.checkpoint = checkpoint
        self.readout_kind = readout_kind
        self.dim = int(dim)
        self.data = str(data)
        self.synthetic = bool(synthetic or root is None)
        # ``--data v2seg+l8``: the ``none`` slot is the union of two caches.
        # ZCacheDir is opened with tags=() so the sft2seg member is read once,
        # by caliber.open_train_z (= run_carrier_arm.open_z_caches).
        union = (not self.synthetic and self.data != "v2seg"
                 and "none" in tuple(tags))
        self.dir = ZCacheDir(self.root, split=split, checkpoint=checkpoint,
                             readout_kind=readout_kind,
                             tags=(() if union else tuple(tags)),
                             required=(() if union else tuple(required)),
                             synthetic=self.synthetic)
        self.union_record: dict[str, Any] | None = None
        if union:
            cache, rec = K.open_train_z(
                self.root, split, data=self.data, checkpoint=checkpoint,
                readout_kind=readout_kind, zcache_root_l8=zcache_root_l8,
                seed=int(seed))
            self.dir.caches["none"] = cache
            self.dir.record["none"] = rec
            self.union_record = rec

    def get(self, sample_ids: Sequence[str], control: str = "none", *,
            device: Any = "cpu", dtype: torch.dtype = torch.float32) -> torch.Tensor:
        return self.dir.batch(list(sample_ids), control, device=device, dtype=dtype)

    kind = "vlm"

    def facts(self) -> dict[str, Any]:
        return {"kind": self.kind,
                "root": str(self.root) if self.root else None,
                "split": self.split, "checkpoint": self.checkpoint,
                "readout_kind": self.readout_kind, "data": self.data,
                "union": self.union_record,
                "synthetic": self.synthetic, "loaded": dict(self.dir.record)}


# --------------------------------------------------------------------------- #
# what a training step is conditioned on
# --------------------------------------------------------------------------- #
class OracleBatchSource:
    """One step = ``b_samples`` LUTs out of the gate stage's pool (R1 §9).

    No records, no ``z`` cache, no images -- G1/G2 must start with nothing on
    disk but the preset bank.  When the pool is not larger than the batch (G1's
    1, G2's 32) every step sees the whole pool; G3 walks a reshuffled pool with
    a cursor, so no LUT is seen twice before all of them are seen once.
    """

    kind = "oracle"

    def __init__(self, store: OracleConditionStore, *, b_samples: int,
                 seed: int) -> None:
        self.store = store
        self.pool = list(store.lut_ids)
        self.b = int(b_samples)
        self.rng = random.Random(int(seed))
        self.order = list(range(len(self.pool)))
        self.cursor = len(self.order)

    def batch(self, device: torch.device
              ) -> tuple[list[str], list[splits.IndexRow] | None, torch.Tensor]:
        if self.b >= len(self.pool):
            picked = list(self.pool)
            if self.b > len(self.pool):                 # cycle the short pool
                picked = [self.pool[i % len(self.pool)] for i in range(self.b)]
        else:
            if self.cursor + self.b > len(self.order):
                self.rng.shuffle(self.order)
                self.cursor = 0
            picked = [self.pool[i]
                      for i in self.order[self.cursor:self.cursor + self.b]]
            self.cursor += self.b
        return picked, None, self.store.get(picked, device=device)

    def facts(self) -> dict[str, Any]:
        return {"kind": self.kind, "n_pool": len(self.pool),
                "b_samples": self.b,
                "whole_pool_every_step": self.b >= len(self.pool)}


class RecordBatchSource:
    """One step = ``b_samples`` training records + their cached ``z`` (R1 §6 Z)."""

    kind = "vlm"

    def __init__(self, rows: Sequence[splits.IndexRow],
                 conditions: ConditionStore, *, b_samples: int,
                 seed: int) -> None:
        self.rows = list(rows)
        self.conditions = conditions
        self.b = int(b_samples)
        self.rng = random.Random(int(seed))
        self.order = list(range(len(self.rows)))
        self.cursor = len(self.order)

    def batch(self, device: torch.device
              ) -> tuple[list[str], list[splits.IndexRow] | None, torch.Tensor]:
        if self.cursor + self.b > len(self.order):
            self.rng.shuffle(self.order)
            self.cursor = 0
        picked = [self.rows[i] for i in self.order[self.cursor:self.cursor + self.b]]
        self.cursor += self.b
        z = self.conditions.get([r.sample_id for r in picked], "none",
                                device=device)
        return [r.lut_id for r in picked], picked, z

    def facts(self) -> dict[str, Any]:
        return {"kind": self.kind, "n_rows": len(self.rows),
                "b_samples": self.b}


# --------------------------------------------------------------------------- #
# evaluation data (CONTRACT GAP: no shared image / GT-alpha loader)
# --------------------------------------------------------------------------- #
@dataclass
class EvalSample:
    row: splits.IndexRow
    image: torch.Tensor          # (3, H, W) sRGB in [0,1]
    alpha: torch.Tensor          # (H, W) GT alpha; style samples are all ones
    alpha_mean: float
    minor: str
    mask_type: str | None


class EvalData:
    """Images (from the dataset shards) and GT alpha (from ``maskviews``).

    Image: the index row's ``members.image`` gives ``(shard, offset, length)``;
    the shard is read through ``splits.ro_path`` so every read is on the *soft*
    read-only mount.  Alpha: ``<split>/shards/shard-*.tar`` at the offset the
    split's ``indexes/shard-*.idx.jsonl`` records for ``<sample_id>.maskhi.png``
    (mode ``L``, already short side 512 -- HANDOFF §七).  ``style`` samples have
    no mask member and ``alpha == 1`` everywhere, by construction of the data law.

    Resampling to the evaluation resolution uses the frozen operator
    (``q3vl/where/upsample.py:54-62``: ``area`` down, ``bilinear`` up).
    """

    def __init__(self, split: str, *, short_side: int = 512,
                 mask_root: Path = MASKVIEW_ROOT) -> None:
        self.split = split
        self.short_side = int(short_side)
        self.mask_root = Path(mask_root)
        self._mask_index: dict[str, dict[str, Any]] | None = None

    # -- alpha -------------------------------------------------------------- #
    def _mask_entries(self) -> dict[str, dict[str, Any]]:
        if self._mask_index is not None:
            return self._mask_index
        out: dict[str, dict[str, Any]] = {}
        idx_dir = self.mask_root / self.split / "indexes"
        for path in sorted(idx_dir.glob("*.idx.jsonl")):
            with path.open(encoding="utf-8") as fh:
                for line in fh:
                    if not line.strip():
                        continue
                    d = json.loads(line)
                    if d.get("suffix") == ".maskhi.png":
                        out[str(d["sample_id"])] = d
        self._mask_index = out
        return out

    def _read_member(self, path: Path, offset: int, length: int) -> bytes:
        with path.open("rb") as fh:
            fh.seek(int(offset))
            blob = fh.read(int(length))
        if len(blob) != int(length):
            raise IOError(f"short read of {path}:{offset}")
        return blob

    # -- one sample --------------------------------------------------------- #
    def load(self, row: splits.IndexRow, record: Mapping[str, Any] | None = None
             ) -> EvalSample:
        from PIL import Image

        member = row.raw["members"]["image"]
        blob = self._read_member(splits.ro_path(member["shard"]),
                                 member["offset"], member["length"])
        img = Image.open(io.BytesIO(blob)).convert("RGB")
        arr = torch.from_numpy(np.array(img, dtype=np.uint8)).float() / 255.0
        image = arr.permute(2, 0, 1)                                  # (3,H,W)
        image = self._resize(image.unsqueeze(0)).squeeze(0)

        if row.is_style:
            alpha = torch.ones(image.shape[-2:], dtype=image.dtype)
        else:
            ent = self._mask_entries().get(row.sample_id)
            if ent is None:
                raise KeyError(f"{row.sample_id}: local sample with no .maskhi.png "
                               f"under {self.mask_root / self.split}")
            shard = self.mask_root / self.split / "shards" / f"{ent['shard']}.tar"
            mblob = self._read_member(shard, ent["offset"], ent["length"])
            m = Image.open(io.BytesIO(mblob)).convert("L")
            a = torch.from_numpy(np.array(m, dtype=np.uint8)).float() / 255.0
            alpha = self._resize(a[None, None])[0, 0]
        rec = record or {}
        return EvalSample(row=row, image=image, alpha=alpha,
                          alpha_mean=float(alpha.mean()),
                          minor=str(rec.get("minor", "")),
                          mask_type=rec.get("mask_type"))

    def load_alpha(self, row: splits.IndexRow,
                   size: tuple[int, int] | None = None) -> torch.Tensor:
        """Just the GT alpha of ``row`` (the shuffled-field row needs no image)."""
        from PIL import Image

        if row.is_style:
            hw = size or (self.short_side, self.short_side)
            return torch.ones(hw, dtype=torch.float32)
        ent = self._mask_entries().get(row.sample_id)
        if ent is None:
            raise KeyError(f"{row.sample_id}: local sample with no .maskhi.png")
        shard = self.mask_root / self.split / "shards" / f"{ent['shard']}.tar"
        blob = self._read_member(shard, ent["offset"], ent["length"])
        m = Image.open(io.BytesIO(blob)).convert("L")
        a = torch.from_numpy(np.array(m, dtype=np.uint8)).float() / 255.0
        a = self._resize(a[None, None])
        if size is not None and tuple(a.shape[-2:]) != tuple(size):
            a = area_resize(a, tuple(size))
        return a[0, 0]

    def _resize(self, x: torch.Tensor) -> torch.Tensor:
        """Frozen block: the resampling operator is ``upsample.area_resize``."""
        h, w = x.shape[-2:]
        short = min(h, w)
        if short == self.short_side:
            return x
        scale = self.short_side / short
        size = (max(1, int(round(h * scale))), max(1, int(round(w * scale))))
        return area_resize(x, size)


# --------------------------------------------------------------------------- #
# training
# --------------------------------------------------------------------------- #
def lut_values(bank: LutBank, lut_ids: Sequence[str], x: torch.Tensor) -> torch.Tensor:
    """``L_l(x)`` per sample: ``x`` is ``(B, Q, 3)`` -> ``(B, Q, 3)``."""
    return torch.stack([bank.apply(x[i], lut_ids[i]) for i in range(len(lut_ids))])


def build_param_groups(arm: g4d.G4DArm, cond, args) -> list[dict[str, Any]]:
    """The arm's own groups, plus the oracle embedding at the generator's lr.

    R1 §6: ``--pi-lr-scale`` is the ``pi(z_color)`` projection's scale (Experiment
    Z).  The oracle embedding is not that projection -- it is the condition
    itself -- so it trains at ``--base-lr``, in its own named group so the board
    can show which parameters moved.
    """
    groups = list(arm.param_groups(args.lr, pi_lr_scale=args.pi_lr_scale))
    if isinstance(cond, OracleConditionStore):
        groups.append({"params": list(cond.parameters()), "lr": float(args.lr),
                       "name": "cond_oracle"})
    return groups


def train(args, arm: g4d.G4DArm, bank: LutBank, source, run_dir: Path,
          device: torch.device, eval_hook, *, cond=None,
          b_samples: int = 32, q_colors: int = 256, total_steps: int = 1,
          steps_per_epoch: int = 1, mining: bool = True) -> dict[str, Any]:
    """R1 §8.1's loop: ``L x Q x 4`` paired anchors, colour-group mining, Adam+cosine."""
    loss_level = resolve_loss_level(args)
    lam_hc, lam_sparse = effective_lambdas(args)
    lam_line = float(args.lam_line)
    extra_cols = tuple(n for n, on in (("L_s4d", args.alpha_s),
                                       ("L_img", args.w_img)) if on)
    n_pairs_s = int(b_samples) * int(q_colors) * S_ANCHORS

    opt = torch.optim.Adam(build_param_groups(arm, cond, args))
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=max(1, total_steps))
    sampler = queries.QuerySampler(seed=args.seed, q=q_colors)
    reg_gen = torch.Generator().manual_seed(int(args.seed) + 7)
    reg_step = 1.0 / max(1, int(args.reg_grid) - 1)

    # §3.3 "精度 bf16" -- the GENERATOR runs in bf16 on CUDA; the carrier always
    # opts out of autocast (glut.py: einsum is on autocast's low-precision list
    # and exp(logpdf) in bf16 is not this carrier), so the maths stays fp32.
    use_amp = device.type == "cuda" and args.precision == "bf16"

    steps_path = run_dir / "steps.jsonl"
    steps_path.parent.mkdir(parents=True, exist_ok=True)
    fh = steps_path.open("a", encoding="utf-8")
    t0 = time.time()
    try:
        for step in range(total_steps):
            lut_ids, picked, z = source.batch(device)
            epoch = step / max(1, steps_per_epoch)
            ratio = queries.mining_ratio(epoch) if mining else 0.0

            if use_amp:
                with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                    params = arm.theta(z)
            else:
                params = arm.theta(z)

            # R1 §8.1: L x Q x 4 paired anchors, s in {0, 1, u, 1-u} per colour.
            x0, s0 = paired_anchor_batch(sampler, b_samples, q_colors, device=device)
            y0 = g4d.target_4d(x0, s0, lut_values(bank, lut_ids, x0))
            if ratio > 0:
                x1, s1 = paired_anchor_batch(sampler, b_samples, q_colors,
                                             device=device)
                y1 = g4d.target_4d(x1, s1, lut_values(bank, lut_ids, x1))
                # ruling 11.1-4 with R1 §8.1's granularity: the probe error is
                # reduced over the four anchors of a colour FIRST, so top-r
                # selects colours, never single (x, s) pairs.  Selecting on the
                # flat (B, Q*4) error would split the group and drop endpoints.
                with torch.no_grad():
                    probe = arm.transform(params.detach(), x0, s0)
                    err = (probe - y0).abs().mean(-1)
                keep = select_hard_colour_groups(err, ratio, b_samples, q_colors)
                x = torch.where(keep.unsqueeze(-1), x0, x1)
                s = torch.where(keep, s0, s1)
                y = torch.where(keep.unsqueeze(-1), y0, y1)
            else:
                x, s, y = x0, s0, y0

            y_hat, aux = arm.transform(params, x, s, return_aux=True)
            # R1 §4.4: zero extra forward -- f(x,0) and f(x,1) are anchors 0/1.
            line = (r_line_from_anchors(y_hat, s, b_samples, q_colors)
                    if lam_line else None)
            extra: dict[str, tuple[torch.Tensor, float]] = {}
            if args.alpha_s:
                extra["L_s4d"] = (l_s4d_random_points(
                    arm, params, n_points=int(args.reg_points), step=reg_step,
                    generator=reg_gen, device=device), args.alpha_s)
            if args.w_img > 0:
                extra["L_img"] = (image_term(arm, params, picked, bank, args, device),
                                  args.w_img)
            loss, cols = g4d.total_loss(y_hat, y, aux.opacity, lam_hc=lam_hc,
                                        lam_sparse=lam_sparse,
                                        lam_line=lam_line, line=line, extra=extra)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            # ``clip_grad_norm_`` returns the PRE-clip total norm: that is the
            # R1 §10 ``gnorm`` column (a spike is this campaign's NaN precursor).
            gnorm = torch.nn.utils.clip_grad_norm_(
                (p for g in opt.param_groups for p in g["params"]),
                float(args.max_grad_norm) if args.max_grad_norm else float("inf"))
            opt.step()
            sched.step()

            telemetry = aux.columns(
                weight_underflow_tau=arm.cfg.weight_underflow_tau)
            row = {"step": step, "epoch": epoch, **cols,
                   # R1 §8.1 counts the batch in (x, s) PAIRS ("2,097,152 色/步"
                   # = 256 x 2048 x 4); the distinct-colour count is kept beside
                   # it so neither number can be read as the other.
                   "n_colors": n_pairs_s,
                   "n_colors_distinct": int(b_samples * q_colors),
                   "n_luts_in_batch": len(set(lut_ids)),
                   "mining_ratio": float(ratio),
                   "lr": float(opt.param_groups[0]["lr"]),
                   **{k2: float(v.detach() if torch.is_tensor(v) else v)
                      for k2, v in telemetry.items()},
                   "gnorm": float(gnorm)}
            if int(row["n_pairs_s"]) != n_pairs_s:
                raise AssertionError(
                    f"n_pairs_s is {row['n_pairs_s']} and the resolved sampler "
                    f"structure is {b_samples} x {q_colors} x {S_ANCHORS} = "
                    f"{n_pairs_s}; R1 §8.1's column exists to pin exactly this")
            if step == 0:
                # tier 3 of the board-time row resolution: a quick eval can land
                # before the file is flushed, and an unflushed file is not a
                # "the loss never ran" failure.
                record_step_witness(row)
                # the pre-registered first-row column set, asserted where it is
                # produced -- "defined but not wired" has cost this campaign five
                # times, and a board-time-only assertion pays a whole run for it.
                g4d.assert_step_row(steps_row=row, loss_level=loss_level,
                                    mode=args.glut4d_mode,
                                    r_line=bool(lam_line), extra=extra_cols,
                                    forbidden=tuple(
                                        k for k in g4d.OFF_BY_DEFAULT_LOSS_COLUMNS
                                        if k not in extra_cols))
            fh.write(json.dumps(row) + "\n")
            fh.flush()

            if args.quick_eval_every and (step + 1) % args.quick_eval_every == 0:
                eval_hook(step + 1, cols)
    finally:
        fh.close()
    return {"total_steps": total_steps, "steps_per_epoch": steps_per_epoch,
            "loss_level": loss_level, "extra_loss_columns": list(extra_cols),
            "lam_line": lam_line, "mining": bool(mining),
            "n_pairs_s": n_pairs_s,
            "batch_structure": {"lut_per_step": int(b_samples),
                                "colors_per_lut": int(q_colors),
                                "s_anchors": S_ANCHORS,
                                "s_anchor_set": "{0, 1, u, 1-u}, u ~ U(0,1)"},
            "condition": source.facts(),
            "precision": {"flag": args.precision, "autocast_used": bool(use_amp),
                          "scope": "generator forward only; the carrier is fp32"},
            "wall_time_s": time.time() - t0,
            "sampler": sampler.facts()}


def image_term(arm: g4d.G4DArm, params: g4d.G4DParams,
               picked: Sequence[splits.IndexRow] | None, bank: LutBank, args,
               device: torch.device) -> torch.Tensor:
    """``L_img`` (ablation row only): the headline formation, in the loss.

    R1 §8.3 caps the cost: at most ``--img-batch`` samples per step and
    ``--img-pixels`` pixels per sample (512-1024), **never** a full short-side-512
    image loss on ``B = 256``.  The pixels are drawn from the sampler's own
    stream, the field is the GT alpha at those pixels, and the target is the
    data law evaluated at exactly the same pixels, so the term and the headline
    remain the same quantity.  The formation is per arm (R1 §5).
    """
    if not picked:
        raise SystemExit(
            "--w-img > 0 needs training records (images + GT alpha); the oracle "
            "condition source has none.  Run L_img with --cond vlm.")
    data = getattr(image_term, "_data", None)
    if data is None or data.split != args.train_split:
        data = EvalData(args.train_split, short_side=args.eval_short_side)
        image_term._data = data                       # type: ignore[attr-defined]
    gen = getattr(image_term, "_gen", None)
    if gen is None:
        gen = torch.Generator().manual_seed(int(args.seed) + 11)
        image_term._gen = gen                         # type: ignore[attr-defined]
    n_img = max(1, min(int(args.img_batch), len(picked)))
    n_px = max(1, int(args.img_pixels))
    total = None
    for i in range(n_img):
        row = picked[i]
        sample = data.load(row)
        img = sample.image.to(device)                      # (3, H, W)
        alpha = sample.alpha.to(device)                    # (H, W)
        flat = img.reshape(3, -1).transpose(0, 1)          # (H*W, 3)
        idx = torch.randint(0, flat.shape[0], (n_px,), generator=gen).to(device)
        x = flat[idx]                                      # (P, 3)
        s = alpha.reshape(-1)[idx]                         # (P,)
        f = arm.transform(params.select(i), x.unsqueeze(0), s.unsqueeze(0))[0]
        i_hat = g4d.compose_headline(x, s.unsqueeze(-1), f, mode=arm.cfg.mode)
        i_star = mix_alpha(x, bank.apply(x, row.lut_id), s.unsqueeze(-1))
        term = g4d.l_img(i_hat, i_star)
        total = term if total is None else total + term
    return total / n_img


# --------------------------------------------------------------------------- #
# evaluation
# --------------------------------------------------------------------------- #
class Evaluator:
    """Builds the per-sample rows the board is made of (§4.B-§4.E + §3.6)."""

    def __init__(self, args, arm: g4d.G4DArm, bank: LutBank,
                 conditions: ConditionStore, device: torch.device) -> None:
        self.args = args
        self.arm = arm
        self.bank = bank
        self.conditions = conditions
        self.device = device
        #: R1 §5: the image formation is per arm and ``mode`` has no default.
        self.mode = str(args.glut4d_mode)
        self.grid17 = queries.uniform_grid(17, device=device)
        self.grid9 = queries.uniform_grid(9, device=device)
        self.data = EvalData(args.eval_split, short_side=args.eval_short_side)
        self.sampler = queries.QuerySampler(seed=args.seed + 1, q=4096)
        self.unseen = self.sampler.sample_heldout(1, q=4096, device=device)[0]
        self.lib: criteria.LibraryValues | None = None
        self.lib_mean_volume: torch.Tensor | None = None
        self.bucket_pools: dict[str, list[str]] = {}
        self.notes: dict[str, Any] = {}

    # -- library ------------------------------------------------------------ #
    def prepare_library(self, train_rows: Sequence[splits.IndexRow]) -> None:
        """``Lib_tr`` on the 9^3 grid (§4.C protocol) + B1's mean transform."""
        ids = sorted({r.lut_id for r in train_rows})
        rng = np.random.default_rng(self.args.seed)
        if self.args.lib_sample and len(ids) > self.args.lib_sample:
            pick = rng.choice(len(ids), size=int(self.args.lib_sample), replace=False)
            ids = [ids[int(i)] for i in sorted(pick)]
        self.lib = criteria.LibraryValues.build(self.bank, ids, self.grid9)
        mean17 = torch.stack([self.bank.apply(self.grid17, lid) for lid in ids]).mean(0)
        self.lib_mean_volume = values_to_volume(mean17, 17)
        self.notes["lib_tr"] = {"n_lut": len(ids), "grid": "9^3 (de76 protocol)",
                                "b1_mean_grid": "17^3 volume, same trilinear operator"}

    def prepare_buckets(self, train_rows: Sequence[splits.IndexRow], *,
                        limit: int | None = None) -> None:
        rows = list(train_rows if limit is None else train_rows[:limit])
        self.bucket_pools = splits.bucket_pools(splits.iter_records(rows))
        self.notes["b3_pools"] = {"n_buckets": len(self.bucket_pools),
                                  "n_records": len(rows)}

    # -- one sample --------------------------------------------------------- #
    def row_for(self, sample: EvalSample, *, pred_field: torch.Tensor | None,
                shuffle_alpha: torch.Tensor, controls: Mapping[str, torch.Tensor],
                z: torch.Tensor, z_zero: torch.Tensor | None = None,
                z_mean: torch.Tensor | None = None,
                b2_ids: Sequence[str] = (), b3_ids: Sequence[str | None] = (),
                b4_id: str | None = None, b6_id: str | None = None
                ) -> dict[str, Any]:
        arm, dev = self.arm, self.device
        row = sample.row
        img = sample.image.to(dev)
        alpha = sample.alpha.to(dev)
        a4 = alpha[None, None]
        img4 = img.unsqueeze(0)
        lut_hi = self.bank.apply_image(img, row.lut_id)
        i_star = mix_alpha(img, lut_hi, alpha[None])

        with torch.no_grad():
            params = arm.theta(z.unsqueeze(0).to(dev))

            # function-value columns (§4.B + this arm's s axis)
            lut_grid = self.bank.apply(self.grid17, row.lut_id)
            out: dict[str, Any] = {
                "sample_id": row.sample_id,
                "winner_confidence": row.winner_confidence,
                "task_type": row.task_type,
                "alpha_mean": sample.alpha_mean,
                "mask_type": sample.mask_type,
            }
            f_s1 = arm.transform(params, self.grid17, 1.0)[0]
            out["grid_error"] = float(criteria.function_distance(f_s1, lut_grid))
            for s_val, key in g4d.S_AXIS_GRID:
                f_s = arm.transform(params, self.grid17, s_val)[0]
                target = mix_alpha(self.grid17, lut_grid,
                                   torch.full_like(self.grid17[:, :1], s_val))
                out[key] = float(criteria.function_distance(f_s, target))
            f_unseen = arm.transform(params, self.unseen, 1.0)[0]
            out["unseen_color_error"] = float(criteria.function_distance(
                f_unseen, self.bank.apply(self.unseen, row.lut_id)))
            hist_c, hist_w = queries.image_histogram_colors(img)
            f_hist = arm.transform(params, hist_c, 1.0)[0]
            out["img_error"] = float(criteria.function_distance(
                f_hist, self.bank.apply(hist_c, row.lut_id), hist_w))

            # headline + the field rows (§4.E)
            fields = {"gt": a4,
                      "const": torch.full_like(a4, sample.alpha_mean),
                      "shuffle": shuffle_alpha.to(dev)[None, None]}
            if pred_field is not None:
                fields["pred"] = pred_field.to(dev)[None, None]
            for name, field in fields.items():
                f_img = arm.apply_to_image(params, img4, field)
                # R1 §5: per arm.  A0 mixes with the row's OWN field (that is
                # what a field row varies); A1/A2/A3 already ate it inside f, so
                # compose_headline is the identity and there is no second alpha.
                i_hat = g4d.compose_headline(img4, field, f_img, mode=self.mode)
                e = float(criteria.image_delta_e00(i_hat[0], i_star))
                out[f"field_{name}"] = e
                if name == "gt":
                    out["E_arm"] = e
                    out["headline_alpha_inside_only"] = float(
                        criteria.image_delta_e00(f_img[0], i_star))
                    out.update(criteria.locality_errors(i_hat[0], i_star, img, alpha))
            if "pred" not in fields:
                out["field_pred"] = None

            # trivial baselines, all in the headline caliber (dE00 / GT alpha)
            out["E_B0_identity"] = float(criteria.image_delta_e00(img, i_star))
            if self.lib_mean_volume is not None:
                mean_img = apply_lut_volume(
                    self.lib_mean_volume.to(device=img.device, dtype=img.dtype),
                    img.permute(1, 2, 0)).permute(2, 0, 1)
                out["E_B1_libmean"] = float(criteria.image_delta_e00(
                    mix_alpha(img, mean_img, alpha[None]), i_star))
            if b2_ids:
                out["E_B2_librandom_repeats"] = [self._lut_headline(img, alpha, i_star, lid)
                                                 for lid in b2_ids]
            if b3_ids:
                vals = [self._lut_headline(img, alpha, i_star, lid) if lid else None
                        for lid in b3_ids]
                out["E_B3_bucket_retrieval_repeats"] = [v for v in vals if v is not None]
            if b4_id:
                out["E_B4_oracle"] = self._lut_headline(img, alpha, i_star, b4_id)
            if b6_id:
                out["E_B6_libfill"] = self._lut_headline(img, alpha, i_star, b6_id)

            # negative controls (§4.D): both columns, always together
            for tag, name in _CONTROL_COLUMN.items():
                z_c = controls.get(tag)
                if z_c is None:
                    continue
                p_c = arm.theta(z_c.unsqueeze(0).to(dev))
                f_c = arm.apply_to_image(p_c, img4, a4)
                out[f"E_{name}"] = float(criteria.image_delta_e00(
                    g4d.compose_headline(img4, a4, f_c, mode=self.mode)[0], i_star))
                out[f"M_{name}"] = float(criteria.function_distance(
                    f_s1, arm.transform(p_c, self.grid17, 1.0)[0]))

            # the two internal B1 analogues (§4.D tail)
            for key, z_alt in (("cond_zero", z_zero), ("cond_mean", z_mean)):
                if z_alt is None:
                    continue
                p_a = arm.theta(z_alt.unsqueeze(0).to(dev))
                f_a = arm.apply_to_image(p_a, img4, a4)
                out[key] = float(criteria.image_delta_e00(
                    g4d.compose_headline(img4, a4, f_a, mode=self.mode)[0], i_star))
        return out

    def _lut_headline(self, img: torch.Tensor, alpha: torch.Tensor,
                      i_star: torch.Tensor, lut_id: str) -> float:
        hat = mix_alpha(img, self.bank.apply_image(img, lut_id), alpha[None])
        return float(criteria.image_delta_e00(hat, i_star))


def evaluate(args, arm: g4d.G4DArm, bank: LutBank, conditions: ConditionStore,
             device: torch.device, *, rows: Sequence[splits.IndexRow],
             train_rows: Sequence[splits.IndexRow], quick: bool = False,
             evaluator: Evaluator | None = None, n_low_in_split: int = 0
             ) -> tuple[dict[str, Any], Evaluator]:
    ev = evaluator or Evaluator(args, arm, bank, conditions, device)
    if ev.lib is None:
        ev.prepare_library(train_rows)
    if not ev.bucket_pools:
        ev.prepare_buckets(train_rows, limit=args.bucket_record_limit or None)

    ids = [r.sample_id for r in rows]
    oracle = isinstance(conditions, OracleConditionStore)
    controls: dict[str, torch.Tensor] = {}
    if oracle:
        # R1 §9.1: the three negative controls are text-side and belong to the
        # G4 board.  Recorded as a waiver, never as a silently missing column.
        ev.notes["negative_controls"] = {
            "computed": False,
            "reason": "cond=oracle (E[lut_id]); the shuffle / irrelevant / const "
                      "controls are text conditions and belong to G4 (R1 §9.1)",
            "waived": list(ORACLE_WAIVED_CRITERIA)}
        z_all = conditions.get([r.lut_id for r in rows], device=device)
    else:
        for tag in ("shuffle", "irrelevant", "const"):
            try:
                controls[tag] = conditions.get(ids, tag)
            except (FileNotFoundError, KeyError) as exc:        # loud, not silent
                ev.notes.setdefault("missing_controls", {})[tag] = str(exc)
        z_all = conditions.get(ids, "none")
    z_mean = z_all.mean(dim=0)
    z_zero = torch.zeros_like(z_mean)

    b2 = criteria.library_random_draw(ev.lib.lut_ids, len(rows), seed=args.seed) \
        if ev.lib is not None else []
    records = list(splits.iter_records(list(rows)))
    minors = [str(rec.get("minor", "")) for rec in records]
    b3 = criteria.bucket_draw(minors, ev.bucket_pools, seed=args.seed) \
        if ev.bucket_pools else []

    b4: dict[str, tuple[str, float]] = {}
    b6: dict[str, tuple[str, float]] = {}
    if ev.lib is not None:
        # B4 is keyed by sample; B6 ("nearest OTHER library LUT") has to be keyed
        # by lut_id or `exclude_self` never finds the row to exclude.
        by_lut = {r.lut_id: bank.apply(ev.grid9, r.lut_id) for r in rows}
        b4_by_lut = criteria.oracle_lut_ids(ev.lib, by_lut, metric="de76")
        b6_by_lut = criteria.oracle_lut_ids(ev.lib, by_lut, metric="de76",
                                            exclude_self=True)
        b4 = {r.sample_id: b4_by_lut[r.lut_id] for r in rows}
        b6 = {r.sample_id: b6_by_lut[r.lut_id] for r in rows}

    pred_dir = Path(args.pred_field_dir) if args.pred_field_dir else None
    pred_sources = pred_field_sources(pred_dir)
    n_pred_missing = 0
    out_rows: list[dict[str, Any]] = []
    for i, row in enumerate(rows):
        sample = ev.data.load(row, records[i])
        shuffle_alpha = ev.data.load_alpha(rows[(i + 1) % len(rows)],
                                           size=tuple(sample.alpha.shape))
        pred = None
        pred_source = None
        if pred_dir is not None:
            p = pred_dir / f"{row.sample_id}.npy"
            if p.exists():
                pf = torch.from_numpy(np.load(p)).float()
                if pf.dim() != 2:
                    raise SystemExit(
                        f"{p} is {tuple(pf.shape)}; the field_pred product is "
                        "2-D (grid_h, grid_w) and this call site does pf[None, "
                        "None] -- a 3-D array would silently become a batch")
                if pf.shape != sample.alpha.shape:
                    # frozen block: the where arm's m_pix is resampled to the
                    # carrier's resolution with area_resize, and the resolution
                    # is recorded on the board (eval_config.short_side).
                    pf = area_resize(pf[None, None], tuple(sample.alpha.shape))[0, 0]
                pred = pf
                pred_source = pred_sources.get(row.sample_id, "unlabelled")
            else:
                n_pred_missing += 1
        r = ev.row_for(
            sample, pred_field=pred, shuffle_alpha=shuffle_alpha,
            controls={k: v[i] for k, v in controls.items()}, z=z_all[i],
            z_zero=z_zero, z_mean=z_mean,
            b2_ids=[rep[i] for rep in b2], b3_ids=[rep[i] for rep in b3],
            b4_id=b4.get(row.sample_id, (None, 0.0))[0],
            b6_id=b6.get(row.sample_id, (None, 0.0))[0])
        r["field_pred_source"] = pred_source
        out_rows.append(r)

    extra_columns: dict[str, dict[str, Any]] = {}
    normal = [r for r in out_rows if r.get("winner_confidence") == "normal"]
    for key in (*[k for _, k in g4d.S_AXIS_GRID], "headline_alpha_inside_only",
                "field_pred", "cond_zero", "cond_mean"):
        vals = [r.get(key) for r in normal]
        if any(v is not None for v in vals):
            extra_columns[key] = criteria.describe(vals)
    # `field_pred` pools TWO products: the where arm's prediction on the local
    # rows and, on the style rows, the constant-one field that IS the data law
    # (`mask is None -> out = edited`), not a prediction.  The pooled column is
    # the pre-registered one; these split it so a board can never present the
    # definition as a where-arm result.  The tag comes from the product's own
    # per_sample.jsonl, not from a guess made here.
    for tag in sorted({str(r.get("field_pred_source")) for r in normal
                       if r.get("field_pred_source") and r.get("field_pred") is not None}):
        vals = [r.get("field_pred") for r in normal
                if r.get("field_pred_source") == tag]
        extra_columns[f"field_pred__{tag}"] = {
            **criteria.describe(vals), "field_pred_source": tag,
            "quantity": "the field_pred row restricted to one product source"}
    board = criteria.build_board(out_rows, arm=g4d.ARM, split=args.eval_split,
                                 extra_columns=extra_columns, seed=args.seed)
    board["quick"] = bool(quick)
    # R1 §9.1: an E[lut_id] headline is a learnable approximation of B4_oracle,
    # so it can never be published and must say so on its own face.
    board["oracle_reference"] = bool(oracle)
    if oracle:
        board["oracle_reference_note"] = (
            "the condition is the GT lut_id: this headline is a learnable "
            "approximation of B4_oracle and may NOT be read next to any arm's "
            "headline (R1 §9.1)")
        board["criteria_waived"] = list(ORACLE_WAIVED_CRITERIA)
    board["published"] = bool(not quick and not args.smoke and not oracle
                              and args.eval_short_side == 512
                              and args.eval_limit is None
                              and not conditions.synthetic)
    # `low` rows are filtered before evaluation (never scored, never in a GT),
    # so build_board sees none of them; the split's own count is recorded here
    # rather than left at 0, which would read as "this split has no low rows".
    board["n_low_excluded"] = int(n_low_in_split) or board.get("n_low_excluded", 0)
    board["n_low_excluded_note"] = (
        "low rows are dropped from the split before evaluation (campaign data "
        "discipline); the count is the split's, not a post-hoc filter")
    board["eval_notes"] = ev.notes
    by_source: dict[str, int] = {}
    for r in normal:
        if r.get("field_pred") is not None:
            k = str(r.get("field_pred_source"))
            by_source[k] = by_source.get(k, 0) + 1
    board["eval_config"] = {"short_side": args.eval_short_side,
                            "field_rows": list(g4d.FIELD_CHOICES),
                            "pred_field_dir": args.pred_field_dir,
                            "pred_field_by_source_normal_only": by_source,
                            "pred_field_n_missing": int(n_pred_missing),
                            "alpha_source": "GT alpha (.maskhi, short side 512)",
                            "mode": str(args.glut4d_mode),
                            "cond": str(args.cond),
                            "headline_formation": FROZEN["headline_formation"]}
    return board, ev


# --------------------------------------------------------------------------- #
# the gate-stage read-out (R1 §9.1: G1 / G2 read function values, not a board)
# --------------------------------------------------------------------------- #
def gate_grid_metrics(arm: g4d.G4DArm, cond: OracleConditionStore,
                      bank: LutBank, lut_ids: Sequence[str],
                      device: torch.device, *, grid_n: int = 17
                      ) -> dict[str, Any]:
    """R1 §9.1's G1 / G2 read-out: ``grid_de00`` on the LUT lattice, per ``s``.

    Five ``s`` columns (0 / 0.25 / 0.5 / 0.75 / 1) plus their pool, against the
    data law's own target ``(1-s) x + s L_l(x)``.  No images, no ``z`` cache, no
    predicted field -- G1 and G2 must run with nothing but ``luts.npz``.
    """
    grid = queries.uniform_grid(int(grid_n), device=device)
    per_s: dict[str, list[float]] = {key: [] for _, key in g4d.S_AXIS_GRID}
    every: list[float] = []
    with torch.no_grad():
        for lid in lut_ids:
            params = arm.theta(cond.get([lid], device=device))
            lut_grid = bank.apply(grid, lid)
            for s_val, key in g4d.S_AXIS_GRID:
                f_s = arm.transform(params, grid, s_val)[0]
                target = mix_alpha(grid, lut_grid,
                                   torch.full_like(grid[:, :1], s_val))
                v = float(criteria.function_distance(f_s, target))
                per_s[key].append(v)
                every.append(v)
    out: dict[str, Any] = {key: criteria.describe(vals)
                           for key, vals in per_s.items()}
    out["grid_de00_mean"] = criteria.describe(every)
    out["n_lut"] = len(list(lut_ids))
    out["grid"] = f"{grid_n}^3 uniform sRGB lattice x 5 s values"
    return out


# --------------------------------------------------------------------------- #
# main
# --------------------------------------------------------------------------- #
def main(argv: Sequence[str] | None = None) -> int:
    args = build_arg_parser().parse_args(argv)
    # the index口径 first: every n / steps_per_epoch below is counted in it
    dataset_ver = K.apply_dataset_version(args)
    if args.carrier == "glut3d":
        # R1 §3: the plain 3D GLUT arm is A0 (A1 is the same carrier with the
        # explicit s gate on top, which is a 4D-target arm).
        args.glut4d_mode = "A0"
    if args.glut4d_field != "gt" and not (args.eval_only or args.no_train):
        raise SystemExit(
            "--glut4d-field is fixed to 'gt' during training (§3.5: 训练固定 gt); "
            "the four field rows are all computed at evaluation time regardless, "
            "so use --eval-only to score a single non-GT field.")
    torch.manual_seed(args.seed)
    device = torch.device(args.device)
    run_dir = Path(args.out)
    run_dir.mkdir(parents=True, exist_ok=True)
    oracle = args.cond == "oracle"

    # --data's population is MEASURED (each source counted against its own
    # on-disk declaration); the merged n is never written down.
    train_rows = (splits.normal_only(splits.load_index(args.train_split))
                  if args.data == "v2seg"
                  else K.train_normal_rows(args.data, split=args.train_split))
    eval_index = splits.load_index(args.eval_split)
    eval_rows = splits.normal_only(eval_index)
    n_low_eval = len(eval_index) - len(eval_rows)
    if args.eval_limit:
        eval_rows = eval_rows[: int(args.eval_limit)]

    # (1) R1 §9's gate ladder, resolved against the MEASURED training LUT bucket
    train_lut_ids = sorted({r.lut_id for r in train_rows})
    gate = resolve_gate_stage(args, train_lut_ids)
    b_samples, q_colors = int(gate["b_samples"]), int(gate["q_colors"])
    needs_board = bool(gate["needs_eval_board"])

    # (2) the colour-span start-up assertion -- BEFORE any dataloader.  It pins
    # the colour-TEXT encoder, which only Experiment Z consumes; an oracle run
    # records why it does not run rather than skipping silently.
    if oracle:
        colorspan_record = {
            "check": "colorspan_vs_tokenizer", "ran": False,
            "skipped_reason": "cond=oracle: no colour text is read out "
                              "(R1 §6 Experiment C)",
            "publishable": False}
    else:
        colorspan_record = run_colorspan_assertion(args, eval_rows)

    bank = LutBank(args.lut_bank, resample=args.lut_resample)
    cfg = g4d.G4DConfig(mode=args.glut4d_mode, n_gauss=args.glut4d_n,
                        cond_dim=args.cond_dim, hidden=args.gen_width,
                        clamp=args.clamp,
                        marg_norm=args.glut4d_marg_norm, init_seed=args.seed)
    arm = g4d.G4DArm(cfg).to(device)

    # (3) the condition source (R1 §2.2 / §6).  The oracle path opens NO z cache
    # and touches no image: G1 / G2 must start with nothing but luts.npz.
    conditions: Any
    train_conditions: Any
    if oracle:
        store = OracleConditionStore(gate["lut_ids"], cond_dim=args.cond_dim,
                                     seed=args.seed).to(device)
        conditions = train_conditions = store
        source = OracleBatchSource(store, b_samples=b_samples, seed=args.seed)
    else:
        conditions = ConditionStore(
            args.z_cache, split=args.eval_split, checkpoint=args.base_ckpt,
            readout_kind=args.readout, dim=2560,
            synthetic=bool(args.smoke and not args.z_cache),
            required=("none", "shuffle", "irrelevant", "const"))
        train_conditions = ConditionStore(
            args.z_cache, split=args.train_split, checkpoint=args.base_ckpt,
            readout_kind=args.readout, dim=2560,
            synthetic=bool(args.smoke and not args.z_cache), tags=("none",),
            # ``--data v2seg+l8``: one condition over two caches, through the
            # shared MultiZCache (caliber.open_train_z); sft2seg opens once.
            data=args.data, zcache_root_l8=args.zcache_root_l8, seed=args.seed)
        source = RecordBatchSource(train_rows, train_conditions,
                                   b_samples=b_samples, seed=args.seed)
    thresholds = DegeneracyThresholds(point_std=args.degeneracy_point_std,
                                      identity_dev=args.degeneracy_identity_dev,
                                      cross_std=args.degeneracy_cross_std)

    # (4) run_setup.json
    steps_per_epoch = K.assert_steps_per_epoch(
        math.ceil(len(train_rows) / b_samples), n_train=len(train_rows),
        batch_samples=b_samples, where="g4d steps_per_epoch")
    total_steps = int(gate["total_steps"] or steps_per_epoch * int(args.epochs))
    loss_level = resolve_loss_level(args)
    lam_hc_eff, lam_sparse_eff = effective_lambdas(args)
    extra_cols = tuple(n for n, on in (("L_s4d", args.alpha_s),
                                       ("L_img", args.w_img)) if on)
    caliber = K.horizon_record(
        data=args.data, n_train=len(train_rows), batch_split=args.batch_split,
        batch_samples=b_samples, queries_per_sample=q_colors,
        steps_per_epoch=steps_per_epoch, total_steps=total_steps,
        base_lr=args.lr, epochs=int(args.epochs),
        zcache_root_l8=args.zcache_root_l8, loss_level=loss_level,
        lambda_hc=lam_hc_eff, lambda_sparse=lam_sparse_eff)
    caliber["train_source_facts"] = (
        None if args.data == "v2seg" else
        splits.train_source_facts(args.data, split=args.train_split))
    caliber["z_cache_train"] = getattr(train_conditions, "union_record", None)
    gate_record = {k: v for k, v in gate.items() if k != "lut_ids"}
    gate_record["lut_ids_n"] = len(gate["lut_ids"])
    gate_record["lut_ids_head"] = list(gate["lut_ids"])[:8]
    setup = {
        "caliber": caliber,
        "arm": g4d.ARM, "axes": list(g4d.ARM_AXES),
        "spec": "PROPOSAL_R1.md (R1, 2026-08-16) -- the only authority",
        "argv": list(sys.argv[1:] if argv is None else argv),
        "flags": vars(args),
        "frozen_block": FROZEN,
        "dataset_version": dataset_ver.facts(),
        "gate_stage": gate_record,
        "measured": {
            "train_split": args.train_split,
            "train_normal_only_n": len(train_rows),
            "train_matches_frozen": len(train_rows) == FROZEN["train_normal_only_n"],
            "train_lut_bucket_n": len(train_lut_ids),
            "train_lut_bucket_matches_r1_3149": len(train_lut_ids) == 3149,
            "steps_per_epoch": steps_per_epoch, "total_steps": total_steps,
            "batch_split": [b_samples, q_colors],
            "s_anchors": S_ANCHORS,
            "colors_per_step": b_samples * q_colors * S_ANCHORS,
            "n_pairs_s": b_samples * q_colors * S_ANCHORS,
            "eval_split": args.eval_split, "eval_normal_only_n": len(eval_rows),
            "eval_board": needs_board,
            "train_split_facts": splits.split_facts(splits.load_index(args.train_split)),
        },
        "model": arm.config,
        "readout": WhatReadoutSpec(kind=args.readout).to_dict(),
        "colorspan_assertion": colorspan_record,
        "degeneracy_thresholds": thresholds.as_dict(),
        "cond": args.cond,
        "conditions": conditions.facts(),
        "train_conditions": train_conditions.facts(),
        "lut_bank": bank.facts(),
        "loss_level": loss_level,
        "lam_line": float(args.lam_line),
        "criteria_required": g4d.required_criteria(),
        "criteria_waived": list(ORACLE_WAIVED_CRITERIA) if oracle else [],
        "first_row_columns": list(g4d.step_columns(
            loss_level, mode=args.glut4d_mode, r_line=bool(args.lam_line),
            extra=extra_cols)),
        "source_sha256": source_freeze(),
        "torch": torch.__version__, "device": str(device),
    }
    write_json(run_dir / "run_setup.json", setup)
    write_json(run_dir / "config" / "loss_preregistration.json",
               g4d.loss_preregistration(args))
    if args.dry_run:
        print(json.dumps({"arm": g4d.ARM, "dry_run": True, "data": args.data,
                          "mode": args.glut4d_mode, "cond": args.cond,
                          "gate_stage": gate["stage"],
                          "n_train": len(train_rows),
                          "n_lut_pool": gate["n_lut_pool"],
                          "batch_structure": [b_samples, q_colors, S_ANCHORS],
                          "colours_per_step": b_samples * q_colors * S_ANCHORS,
                          "n_pairs_s": b_samples * q_colors * S_ANCHORS,
                          "steps_per_epoch": steps_per_epoch,
                          "total_steps": total_steps, "base_lr": args.lr,
                          "lam_line": float(args.lam_line),
                          "eval_board": needs_board,
                          "published": False if oracle else None,
                          "loss_level": loss_level}, indent=2), flush=True)
        return 0

    # (5) quick eval: the degeneracy guard fires at the first one
    state: dict[str, Any] = {"first_quick": True, "best": None}
    quick_rows = eval_rows[: int(args.quick_eval_n)]
    quick_path = run_dir / "quick_eval.jsonl"
    binding, binding_reason = degeneracy_binding(args, total_steps)
    if oracle and int(gate["n_lut_pool"]) < 2:
        # "one transform for every sample" IS G1's design (a single-LUT overfit),
        # so the cross-sample spread floor cannot bind there.  Recorded, not
        # relaxed: the three numbers are still measured, printed and written.
        binding = False
        binding_reason = (f"cond=oracle with an {gate['n_lut_pool']}-LUT pool "
                          f"(gate stage {gate['stage']}): cross_std is 0 by "
                          "construction, so the verdict is recorded but does not "
                          "stop the run.  Such a board is never published.")

    def quick_cond(rows: Sequence[splits.IndexRow]) -> torch.Tensor:
        if oracle:
            return conditions.get(gate["lut_ids"][: max(1, len(rows))],
                                  device=device)
        return conditions.get([r.sample_id for r in rows], "none", device=device)

    def quick_eval(step: int, loss_cols: Mapping[str, Any] | None = None) -> None:
        # R1 §10: EVERY quick eval re-checks L_rec, not just the first one.  A
        # NaN prediction scores dE00 = 0.0, so an unchecked run publishes a
        # headline of 0.0 (EPR-030 was bitten twice).
        if loss_cols is not None:
            assert_l_rec_finite(loss_cols.get("L_rec"), step=step,
                                where=f"quick_eval@step{step} (training L_rec)",
                                run_dir=run_dir,
                                extra={"L_total": loss_cols.get("L_total")})
        if state["first_quick"]:
            # `exit_process=binding`: a non-binding call still measures, prints
            # and records the witness (guards.record_degeneracy_check), it just
            # raises DegenerateTransform instead of SystemExit(2).
            try:
                report = g4d.assert_not_degenerate(
                    arm, quick_cond(quick_rows),
                    queries.uniform_grid(9, device=device), s=1.0,
                    thresholds=thresholds, where=f"quick_eval@step{step}",
                    exit_process=binding)
            except DegenerateTransform as exc:
                report = exc.report
                print(f"[EPR-028] degeneracy verdict NOT BINDING: {binding_reason}",
                      file=sys.stderr, flush=True)
            write_json(run_dir / "degeneracy_first_quick_eval.json",
                       {**report.as_dict(), "step": step, "binding": binding,
                        "binding_reason": binding_reason,
                        "binding_min_steps": DEGENERACY_BINDING_MIN_STEPS,
                        "total_steps": int(total_steps)})
            state["first_quick"] = False

        if not needs_board:
            # R1 §9.1: G1 / G2 read function values, not a board.
            grid = gate_grid_metrics(arm, conditions, bank, gate["lut_ids"],
                                     device, grid_n=int(args.reg_grid))
            state["grid"] = grid
            score = (grid.get("grid_de00_mean") or {}).get("mean")
            assert_l_rec_finite(score, step=step,
                                where=f"quick_eval@step{step} (grid_de00_mean)",
                                run_dir=run_dir)
            rec = {"step": step, "grid_de00_mean": score,
                   **{k: (grid.get(k) or {}).get("mean")
                      for _, k in g4d.S_AXIS_GRID}}
            with quick_path.open("a", encoding="utf-8") as fh:
                fh.write(json.dumps(rec) + "\n")
            # checkpoint selection: the quick-eval read-out, never a val loss
            if score is not None and (state["best"] is None or score < state["best"]):
                state["best"] = score
                torch.save({"step": step, "state_dict": arm.state_dict(),
                            "grid_de00_mean": score}, run_dir / "best.pt")
            return

        board, ev = evaluate(args, arm, bank, conditions, device, rows=quick_rows,
                             train_rows=train_rows, quick=True,
                             evaluator=state.get("evaluator"))
        state["evaluator"] = ev
        # "定义了没接线" has cost this campaign five times: assert the whole
        # pre-registered table at the FIRST quick eval, minus the columns that
        # depend on an external product (field_pred needs the where arm's m_pix)
        # or on a text read-out this run does not have (R1 §9.1).
        waived = {"field_pred", *(ORACLE_WAIVED_CRITERIA if oracle else ())}
        criteria.assert_criteria_ran(
            board, g4d.ARM,
            required=[k for k in g4d.required_criteria() if k not in waived])
        head = ((board.get("contexts") or {}).get("all") or {}) \
            .get("headline_normal_only", {})
        assert_l_rec_finite(head.get("mean"), step=step,
                            where=f"quick_eval@step{step} (headline)",
                            run_dir=run_dir)
        rec = {"step": step, "headline_normal_only": head.get("mean"),
               "n": head.get("n")}
        with quick_path.open("a", encoding="utf-8") as fh:
            fh.write(json.dumps(rec) + "\n")
        # checkpoint selection: the quick-eval hard gate + headline, never val loss
        if head.get("mean") is not None and (state["best"] is None
                                             or head["mean"] < state["best"]):
            state["best"] = head["mean"]
            torch.save({"step": step, "state_dict": arm.state_dict(),
                        "headline_normal_only": head["mean"]}, run_dir / "best.pt")

    train_report: dict[str, Any] = {"skipped": True}
    if not (args.eval_only or args.no_train):
        train_report = train(args, arm, bank, source, run_dir, device, quick_eval,
                             cond=conditions if oracle else None,
                             b_samples=b_samples, q_colors=q_colors,
                             total_steps=total_steps,
                             steps_per_epoch=steps_per_epoch,
                             mining=bool(gate["mining"]))
        write_json(run_dir / "train_report.json", train_report)
    elif degeneracy_check_ran() is None:
        # --eval-only / --no-train must not be a way to publish a board from a
        # process that never ran the guard (W4).
        report = g4d.assert_not_degenerate(
            arm, quick_cond(quick_rows), queries.uniform_grid(9, device=device),
            s=1.0, thresholds=thresholds, where="eval_only")
        write_json(run_dir / "degeneracy_eval_only.json", report.as_dict())

    witness = degeneracy_check_ran() or {}
    degeneracy_block = {**witness, "binding": binding,
                        "binding_reason": binding_reason,
                        "binding_min_steps": DEGENERACY_BINDING_MIN_STEPS,
                        "total_steps": int(total_steps)}

    # (6a) G1 / G2: no board at all -- metrics.json is the function-value
    # read-out plus the telemetry the next gate judges on (R1 §9.1).
    if not needs_board:
        grid = state.get("grid") or gate_grid_metrics(
            arm, conditions, bank, gate["lut_ids"], device,
            grid_n=int(args.reg_grid))
        first_row, row_source = g4d.assert_step_row(
            steps_path=run_dir / "steps.jsonl", loss_level=loss_level,
            mode=args.glut4d_mode, r_line=bool(args.lam_line), extra=extra_cols,
            forbidden=tuple(k for k in g4d.OFF_BY_DEFAULT_LOSS_COLUMNS
                            if k not in extra_cols)) \
            if not (args.eval_only or args.no_train) else ({}, "eval_only")
        assert_l_rec_finite((grid.get("grid_de00_mean") or {}).get("mean"),
                            step=int(total_steps), where="final grid read-out",
                            run_dir=run_dir)
        out = {"arm": g4d.ARM, "spec": "PROPOSAL_R1.md §9.1",
               "gate_stage": gate_record, "mode": args.glut4d_mode,
               "cond": args.cond,
               "published": False,
               "published_reason": "gate stage G1/G2 read function values only; "
                                   "the condition is E[lut_id] (R1 §9.1)",
               "oracle_reference": True,
               "grid_metrics": grid,
               "first_step_row": {"source": row_source, "row": first_row},
               "degeneracy": degeneracy_block,
               "train_report": train_report}
        write_json(run_dir / "metrics.json", out)
        print(f"[EPR-028] {gate['stage']} grid_de00_mean = "
              f"{(grid.get('grid_de00_mean') or {}).get('mean')} "
              f"(n_lut={grid.get('n_lut')}) -> {run_dir / 'metrics.json'}")
        return 0

    # (6b) the board, then the publication gate, then metrics.json
    if oracle:
        keep = set(conditions.lut_ids)
        kept = [r for r in eval_rows if r.lut_id in keep]
        n_dropped = len(eval_rows) - len(kept)
        eval_rows = kept
        if not eval_rows:
            raise SystemExit(
                "no evaluation row's lut_id is inside the oracle pool; an "
                "E[lut_id] condition cannot score a LUT it never saw")
        state["eval_rows_dropped_not_in_oracle_pool"] = n_dropped
    board, _ = evaluate(args, arm, bank, conditions, device, rows=eval_rows,
                        train_rows=train_rows, n_low_in_split=n_low_eval,
                        evaluator=state.get("evaluator"))
    board["degeneracy"] = degeneracy_block
    if oracle:
        board["eval_rows_dropped_not_in_oracle_pool"] = int(
            state.get("eval_rows_dropped_not_in_oracle_pool", 0))
    # publish.assert_publishable only checks that the guard RAN.  A published
    # board must additionally carry a guard that PASSED: the waiver above is
    # reachable only under --smoke / a single-LUT oracle pool, and this is the
    # assertion that says so.
    if board.get("published") and not witness.get("ok", False):
        raise SystemExit(
            "[EPR-028] refusing to publish: the degenerate-solution guard did "
            f"not pass in this process (witness={witness or 'never ran'}, "
            f"binding={binding}: {binding_reason})")
    waived = {*(ORACLE_WAIVED_CRITERIA if oracle else ())}
    report = publish.assert_publishable(
        board, g4d.ARM, steps_path=run_dir / "steps.jsonl",
        eval_only=bool(args.eval_only or args.no_train), loss_level=loss_level,
        extra_step_columns=tuple(g4d.step_extra_columns(args.glut4d_mode))
        + (("R_line",) if args.lam_line else ()) + extra_cols,
        required=[k for k in g4d.required_criteria() if k not in waived])
    if not (args.eval_only or args.no_train):
        g4d.assert_step_row(steps_path=run_dir / "steps.jsonl",
                            loss_level=loss_level, mode=args.glut4d_mode,
                            r_line=bool(args.lam_line), extra=extra_cols,
                            forbidden=tuple(k for k in g4d.OFF_BY_DEFAULT_LOSS_COLUMNS
                                            if k not in extra_cols))
    board["publish_report"] = report
    board["train_report"] = train_report
    head = board["contexts"]["all"]["headline_normal_only"]
    # checked BEFORE the board lands: a NaN checkpoint scores dE00 = 0.0, and a
    # 0.0 headline on disk is exactly the artefact R1 §10 forbids.
    assert_l_rec_finite(head.get("mean"), step=int(total_steps),
                        where="final board headline", run_dir=run_dir)
    write_json(run_dir / "metrics.json", board)
    tag = " (oracle_reference: NOT comparable to any arm headline)" if oracle else ""
    print(f"[EPR-028] headline_normal_only = {head.get('mean')} (n={head.get('n')})"
          f"{tag} -> {run_dir / 'metrics.json'}")
    return 0


def run_colorspan_assertion(args, rows: Sequence[splits.IndexRow]) -> dict[str, Any]:
    """Frozen block item 7, before the dataloader exists.

    Reads the ``color`` text of up to 512 records of the evaluation split and
    pins this package's piecewise encoder against the base checkpoint's
    tokenizer, length first and then token by token.  A smoke run whose tokenizer
    is not mounted records the reason and forfeits publication -- it never
    silently skips.
    """
    texts: list[str] = []
    for rec in splits.iter_records(list(rows[:512])):
        t = rec.get("color")
        if isinstance(t, str) and t:
            texts.append(t)
    try:
        from transformers import AutoTokenizer

        tok = AutoTokenizer.from_pretrained(args.base_ckpt)
    except Exception as exc:                                   # noqa: BLE001
        if not args.smoke:
            raise
        return {"check": "colorspan_vs_tokenizer", "skipped_reason": str(exc),
                "publishable": False, "n_texts_available": len(texts)}
    return colorspan.assert_color_span_encoding(tok, texts,
                                                tokenizer_path=args.base_ckpt)


if __name__ == "__main__":   # pragma: no cover
    raise SystemExit(main())
