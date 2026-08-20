#!/usr/bin/env python
"""EPR-026 INTERPC runner -- ``L = L_GLUT + lambda_int * L_interp``.

Four stages, selected with ``--stage``:

``setup`` (default)
    Validate every flag, run the frozen-block colour-span assertion against the
    base checkpoint's tokenizer **before** any data loader is built, construct
    and hash the pairing index, and write ``run_setup.json`` +
    ``loss_preregistration.json``.  Nothing is trained; this is the artefact a
    queue submission freezes against (source sha256 included).

``self-test``
    A synthetic CPU end-to-end: a handful of steps through :func:`train_step`
    with mining on, the first-quick-eval degeneracy guard, the IP-A / IP-B
    columns, a board and the publication gate.  No GPU, no NFS, no tokenizer --
    it exercises the wiring, not the science.

``train``
    The real loop.  Needs the ``z`` cache (see "contract gaps" below).

``eval``
    Two boards over the same ``V_what`` normal-only rows and the same
    checkpoint, written as two files because they answer two questions:

    * ``board_functionspace.json`` -- EPR-026's own P1 board: ``E^grid``, the
      unseen colour column, IP-A (six alphas) and IP-B (K = 20);
    * ``metrics.json`` -- the **twelve pre-registered keys** every arm publishes
      (headline ``dE00(Î, I*)`` on GT alpha at short side 512, B0-B4/B6, the
      three negative controls) plus the four P1 columns, built by the shared
      :func:`q3vl.whatb.criteria.build_board` from images the shared
      :class:`q3vl.whatb.evaldata.SampleStore` loads.  This is the board
      ``--publish`` gates and the row a cross-arm paired delta reads.

Shared infrastructure this runner consumes (it implements neither)
------------------------------------------------------------------
* the ``z`` cache is :mod:`q3vl.whatb.zcache` -- one reader, one on-disk layout
  (``<root>/<split>__<tag>``) and one set of start-up assertions for all six
  arms.  ``--z-cache DIR`` is required and the run refuses to start without it.
* the image / GT-alpha loader is :class:`q3vl.whatb.evaldata.SampleStore`.  It
  is what makes the **checkpoint selection** here identical to the other five
  arms: every ``--eval-every`` steps the loop recomputes
  ``.contexts.all.headline_normal_only`` on a fixed V_what normal-only subset
  and keeps ``best.pt`` (HANDOFF section 4.H; never a val loss).  The ``eval``
  stage reads the same loader for the full-split image board, so the selection
  scalar and the published headline are the same quantity computed by the same
  code on the same GT.
* the board itself is :func:`q3vl.whatb.criteria.build_board` and the gate is
  :func:`q3vl.whatb.publish.assert_publishable` (through
  ``assert_publishable_interpc``) -- this runner registers columns, it does not
  define criteria.

Discipline: this file starts no GPU process unless ``--device`` says so;
everything above is CPU-clean.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
import time
from dataclasses import replace
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import torch

from q3vl.whatb import caliber as K
from q3vl.whatb.arms.interpc import (
    ARM,
    AXES,
    ALTERNATE_CHOICES,
    AcaiCritic,
    AlphaSampler,
    EmaTeacher,
    FitBatch,
    INTERP_ALPHA_MODES,
    INTERP_DIST_CHOICES,
    INTERP_RAMP_CHOICES,
    INTERP_STAGE_CHOICES,
    INTERP_TARGET_CHOICES,
    INTERP_WHERE_CHOICES,
    InterpcArm,
    InterpcConfig,
    PairBatch,
    PairIndex,
    acai_critic_loss,
    assert_context_caches,
    assert_interp_step_columns,
    assert_publishable_interpc,
    build_optimizer,
    build_scheduler,
    interp_extra_columns,
    ipa_pair_errors,
    ipb_pair_path,
    mine_step,
    quick_eval_guard,
    run_setup_block,
    train_step,
)
from q3vl.whatb.criteria import (
    LibraryValues,
    assert_criteria_ran,
    build_board,
    bucket_draw,
    compose_hat,
    function_distance,
    image_delta_e00,
    library_random_draw,
    locality_errors,
    oracle_lut_ids,
)
from q3vl.whatb.degeneracy import DegenerateTransform, degeneracy_check_ran
from q3vl.whatb.zcache import CONTROL_TAGS, ZCacheDir
from q3vl.whatb.queries import QuerySampler, uniform_grid

REPO = Path("/home/bc/VeraRetouch")
BASE_CHECKPOINT = "/home/bc/data/runs/q3vl_base_sft_v2seg_20260814/checkpoint-4976"
DEFAULT_OUT = REPO / "experiments/prs/EPR-026_interp-consistency-supervision"
#: the files whose bytes define this run (frozen at submission, ruling: "进程
#: 启动后禁改源码")
SOURCE_FILES = ("q3vl/whatb/arms/interpc.py",
                "q3vl/whatb/scripts/run_interpc_arm.py",
                "q3vl/whatb/glut.py", "q3vl/whatb/generator.py",
                "q3vl/whatb/criteria.py", "q3vl/whatb/publish.py",
                "q3vl/whatb/guards.py", "q3vl/whatb/lutdata.py",
                "q3vl/whatb/colorimetry.py", "q3vl/whatb/queries.py",
                "q3vl/whatb/colorspan.py", "q3vl/whatb/readout.py",
                "q3vl/whatb/splits.py",
                # the shared z cache and eval loader are consumed, not copied,
                # so they belong in this run's freeze
                "q3vl/whatb/zcache.py", "q3vl/whatb/evaldata.py")

#: The degeneracy verdict is only BINDING once the arm has had a real chance to
#: leave its own initialisation.  The generator's heads are zero-initialised, so
#: ``cross_std`` ("one transform for every sample") is 0 **by construction** at
#: step 0 and only grows as the generator learns z.  Measured on CPU 2026-08-15
#: with this arm's own optimiser, its own loss, the real z cache and the real
#: LUT bank, ``guards.measure_degeneracy``'s ``cross_std`` (floor 1e-4) walks::
#:
#:     step   0  4.57e-4 pass   step   4  2.75e-4 pass   step   9  7.56e-5 fail
#:     step  14  5.18e-5 fail   step  49  4.97e-5 fail   step 100  3.99e-5 fail
#:     step 200  5.88e-5 fail   step 300  5.10e-5 fail   step 325  5.63e-5 fail
#:     step 350  2.33e-3 pass   step 400  2.27e-3 pass   step 550  1.07e-2 pass
#:
#: i.e. a start-up trough between roughly step 5 and step 330, and the old
#: trigger (``step + 1 >= min(50, total)``) sat at the bottom of it for EVERY
#: run with ``total >= 50``, the 117,440-step one included.  The frozen schedule
#: puts the first quick eval at one epoch, which is the horizon the floor was
#: written for.  Below that horizon the guard still RUNS (all three numbers
#: computed, printed, recorded, and the in-process witness set, so
#: ``publish.assert_publishable`` still sees it), but it does not kill the
#: process -- and only on a run that cannot publish a board.
#: This is EPR-028's constant (``run_g4d_arm.py:110``) applied to this arm; the
#: criteria, the thresholds and the witness are untouched.
DEGENERACY_BINDING_MIN_STEPS: int = 2936


def degeneracy_binding(args, total_steps: int) -> tuple[bool, str]:
    """Does the degeneracy verdict kill the process at the first quick eval?

    Binding unless the run is BOTH a ``--smoke`` run AND shorter than
    :data:`DEGENERACY_BINDING_MIN_STEPS` -- see that constant for the measured
    reason.  A short run without ``--smoke`` stays binding: it can publish a
    board, and a published board must never rest on a verdict that was waived.

    This runner has **no** ``--smoke`` flag (the campaign runs its short pass
    without one on purpose, so that every start-up assertion really executes),
    so ``getattr`` is the honest read and the waiver is currently unreachable
    here: INTERPC is always binding.  The branch is kept byte-for-byte with
    EPR-028's so that adding the flag later cannot make the two arms drift.
    """
    if int(total_steps) >= DEGENERACY_BINDING_MIN_STEPS:
        return True, (f"total_steps {total_steps} >= "
                      f"{DEGENERACY_BINDING_MIN_STEPS} (one frozen epoch)")
    if not bool(getattr(args, "smoke", False)):
        return True, (f"total_steps {total_steps} < "
                      f"{DEGENERACY_BINDING_MIN_STEPS} but this is not a "
                      "--smoke run, so the board can be published and the "
                      "verdict binds")
    return False, (f"--smoke with total_steps {total_steps} < "
                   f"{DEGENERACY_BINDING_MIN_STEPS} (one frozen epoch): the "
                   "generator's zero-initialised heads (f = identity at step 0) "
                   "make cross_std ~0 by construction this early, so the three "
                   "numbers are recorded and printed but do not stop the run.  "
                   "A --smoke board is never published.")


# --------------------------------------------------------------------------- #
# provenance
# --------------------------------------------------------------------------- #
def provenance() -> dict[str, Any]:
    out: dict[str, Any] = {"git_commit": "unknown", "working_tree_dirty": None,
                           "source_sha256": None, "source_files": list(SOURCE_FILES)}
    try:
        out["git_commit"] = subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=REPO).decode().strip()
        st = subprocess.check_output(["git", "status", "--porcelain"],
                                     cwd=REPO).decode().strip()
        out["working_tree_dirty"] = bool(st)
    except Exception:                                    # pragma: no cover
        pass
    h = hashlib.sha256()
    for rel in SOURCE_FILES:
        p = REPO / rel
        if p.is_file():
            h.update(rel.encode())
            h.update(p.read_bytes())
    out["source_sha256"] = h.hexdigest()
    return out


# --------------------------------------------------------------------------- #
# the z cache + the headline selection set (both shared, not re-implemented)
# --------------------------------------------------------------------------- #
def open_z(root: str | Path, split: str, cfg: InterpcConfig, *, checkpoint: str,
           required: Sequence[str] = ("none",),
           tags: Sequence[str] = CONTROL_TAGS) -> ZCacheDir:
    """One split's caches, with the three start-up assertions of HANDOFF 4.H.

    The reader is :mod:`q3vl.whatb.zcache` -- the single implementation the
    frozen block asks for.  This arm used to carry its own copy of it.

    ``tags=()`` opens nothing: the ``--data v2seg+l8`` path fills the ``none``
    slot from :func:`q3vl.whatb.caliber.open_train_z` instead, so the sft2seg
    member is opened exactly once.
    """
    return ZCacheDir(root, split=split, checkpoint=checkpoint,
                     readout_kind=cfg.readout, context_source=cfg.context,
                     tags=tuple(tags), required=required, seed=cfg.seed)


@torch.no_grad()
def _transform_image(arm: InterpcArm, z: torch.Tensor, img: torch.Tensor,
                     chunk: int = 65536) -> torch.Tensor:
    """``f_hat`` applied to every pixel of ``(3, H, W)``, chunked."""
    c, h, w = img.shape
    pts = img.permute(1, 2, 0).reshape(-1, 3)
    outs = []
    zz = z.reshape(1, -1)
    for i in range(0, pts.shape[0], chunk):
        outs.append(arm(zz, pts[i:i + chunk].unsqueeze(0))[0])
    return torch.cat(outs, dim=0).reshape(h, w, 3).permute(2, 0, 1)


@torch.no_grad()
def selection_headline(arm: InterpcArm, samples: Sequence[Mapping[str, Any]], *,
                       bank: Any, device: Any) -> dict[str, Any]:
    """``.contexts.all.headline_normal_only`` on a fixed V_what normal-only subset.

    The **same** scalar the other five arms select on (HANDOFF 4.H: one
    quick-eval hard gate + headline selection for every arm, never val loss).
    ``I_hat = (1-a) I + a f_hat(I)`` and ``I* = (1-a) I + a L_l(I)`` with GT
    alpha at short side 512 -- ``criteria.compose_hat`` / ``image_delta_e00``,
    the one implementation.
    """
    from q3vl.whatb.criteria import compose_hat, image_delta_e00

    arm.eval()
    errs: list[float] = []
    for s in samples:
        img = s["image"].to(device)
        alpha = s["alpha"] if isinstance(s["alpha"], float) else s["alpha"].to(device)
        i_star = bank.f_star_image(img, alpha, s["lut_id"])
        i_hat = compose_hat(img, alpha, _transform_image(arm, s["z"].to(device), img))
        errs.append(float(image_delta_e00(i_hat, i_star)))
    arm.train()
    n = len(errs)
    return {"mean": (sum(errs) / n) if n else None, "n": n}


def load_selection_samples(split: str, cache: ZCacheDir, *, limit: int,
                           device: Any) -> list[dict[str, Any]]:
    """The fixed selection subset: V_what **normal-only**, in index order."""
    from q3vl.whatb.evaldata import SampleStore
    from q3vl.whatb.splits import load_index, normal_only

    store = SampleStore(split)
    out: list[dict[str, Any]] = []
    for row in normal_only(load_index(split)):
        if len(out) >= limit:
            break
        if row.sample_id not in cache.cache("none"):
            continue
        img, alpha = store.load(row, device=device)
        out.append({"sample_id": row.sample_id, "lut_id": row.lut_id,
                    "image": img, "alpha": alpha,
                    "z": cache.vector(row.sample_id, device=device)})
    return out


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #
def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(
        prog="run_interpc_arm",
        description="EPR-026 INTERPC: EPR-024 + lambda_int * L_interp")
    ap.add_argument("--stage", default="setup",
                    choices=("setup", "self-test", "train", "eval"))
    ap.add_argument("--out-root", default=str(DEFAULT_OUT))
    ap.add_argument("--run-name", default=None)
    ap.add_argument("--device", default="cpu",
                    help="cpu / cuda:0.  The default is cpu on purpose: nothing "
                         "in this arm needs a GPU until --stage train.")
    ap.add_argument("--checkpoint", default=BASE_CHECKPOINT)
    ap.add_argument("--z-cache", default=None)
    ap.add_argument("--dry-run", action="store_true",
                    help="resolve the caliber, write run_setup.json, then stop")
    # --data / --zcache-root-l8; --batch-split and --base-lr are this arm's own
    # flags (registered in the EPR-024 group below) so they are not re-declared
    K.add_caliber_arguments(ap, batch_split=False, base_lr=False)
    ap.add_argument("--bank-dir", default="/var/cache/veradata/preset_bank_full")
    ap.add_argument("--train-split", default="train")
    ap.add_argument("--eval-split", default="V_what")
    ap.add_argument("--publish", action="store_true",
                    help="run the publication gate on the board (it will refuse "
                         "without the image-space headline)")
    ap.add_argument("--eval-every", type=int, default=2000,
                    help="steps between quick evals; each one recomputes the "
                         "headline on the selection subset and updates best.pt "
                         "(HANDOFF 4.H: the same gate for all six arms, never "
                         "val loss).  0 disables it and --publish then refuses.")
    ap.add_argument("--select-samples", type=int, default=64,
                    help="size of the fixed V_what normal-only subset the "
                         "selection headline is computed on")
    ap.add_argument("--colorspan-samples", type=int, default=256)
    ap.add_argument("--skip-colorspan-assert", action="store_true",
                    help="only for a run with no tokenizer available; the fact "
                         "is written into run_setup.json as a refusal, not a pass")

    # ---- EPR-024 inherited flags (EPR-024:610) ----------------------------
    g = ap.add_argument_group("EPR-024 baseline (inherited verbatim)")
    g.add_argument("--readout", default="seg_color",
                   choices=("seg_color", "color_span_pool", "color_close",
                            "im_end", "seg_where", "qtok"))
    g.add_argument("--readout-qtok", type=int, default=0)
    g.add_argument("--cond-dim", type=int, default=64)
    g.add_argument("--n-gauss", type=int, default=48)
    g.add_argument("--gen-width", type=int, default=128, choices=(128, 64))
    g.add_argument("--gen-mode", default="full", choices=("full", "affine_only"))
    g.add_argument("--loss-level", type=int, default=3, choices=(1, 2, 3, 4))
    g.add_argument("--lambda-img", type=float, default=0.0)
    g.add_argument("--clamp", default="two", choices=("two", "one"))
    # the ONE table (arms/carrier.py BATCH_SPLITS); EPR-030 runs on 256x8192
    g.add_argument("--batch-split", default="32x256",
                   choices=list(K.batch_split_choices()))
    g.add_argument("--hc-eps", type=float, default=1e-3)
    g.add_argument("--no-hc-mask", action="store_true")
    g.add_argument("--context", default="generated", choices=("teacher", "generated"))
    g.add_argument("--lut-resample", default="none", choices=("none",))
    g.add_argument("--no-mining", action="store_true")
    # one spelling across the six arms; --lr stays as this arm's old name
    g.add_argument("--base-lr", "--lr", dest="lr", type=float, default=K.BASE_LR)
    g.add_argument("--pi-lr-scale", type=float, default=0.1)
    g.add_argument("--shared-geom-lr-scale", type=float, default=0.1)
    g.add_argument("--epochs", type=int, default=40)
    g.add_argument("--max-steps", type=int, default=None,
                   help="cap the loop (smoke runs only; the frozen budget is "
                        "117,440 steps and any capped run is off the U4 grid)")
    g.add_argument("--seed", type=int, default=20260810)
    g.add_argument("--amp-dtype", default="bfloat16",
                   choices=("bfloat16", "float32"))
    g.add_argument("--max-grad-norm", type=float, default=1.0)

    # ---- EPR-026 flags (EPR-026:463) --------------------------------------
    i = ap.add_argument_group("EPR-026 (the one change)")
    i.add_argument("--interp-weight", type=float, default=0.0,
                   help="lambda_int.  0 = the EPR-024 baseline row (bit-for-bit); "
                        "the main arm is 1; the sweep is {0, 0.1, 1, 10}")
    i.add_argument("--interp-alpha", default="beta0.5", choices=INTERP_ALPHA_MODES)
    i.add_argument("--interp-p-end", type=float, default=1.0 / 3.0)
    i.add_argument("--interp-where", default="post_pi", choices=INTERP_WHERE_CHOICES)
    i.add_argument("--interp-dist", default="l1", choices=INTERP_DIST_CHOICES)
    i.add_argument("--interp-target", default="gt_mix", choices=INTERP_TARGET_CHOICES)
    i.add_argument("--interp-ema-decay", type=float, default=0.999)
    i.add_argument("--interp-ramp", default="const", choices=INTERP_RAMP_CHOICES)
    i.add_argument("--interp-stage", default="joint", choices=INTERP_STAGE_CHOICES)
    i.add_argument("--interp-finetune-start", type=int, default=None)
    i.add_argument("--interp-pairs-per-step", type=int, default=None)
    i.add_argument("--interp-hc", action="store_true")
    i.add_argument("--interp-alternate", default="none", choices=ALTERNATE_CHOICES)
    i.add_argument("--acai-lambda", type=float, default=0.5)
    i.add_argument("--acai-gamma", type=float, default=0.2)
    i.add_argument("--jac-decay", type=float, default=0.01)
    i.add_argument("--jac-lambda", type=float, default=1.0)
    i.add_argument("--degenerate-weight-tau", type=float, default=1e-3)
    i.add_argument("--ipb-k", type=int, default=20)
    i.add_argument("--eval-grid-n", type=int, default=17)
    i.add_argument("--eval-pairs", type=int, default=None,
                   help="cap the IP-A/IP-B pair count (default: all 1311)")
    i.add_argument("--dlib-lib-size", type=int, default=0,
                   help="size of Lib_tr for the d_lib(alpha) column (0 = off).  "
                        "The column costs O(pairs x (K+1) x |Lib| x |X|) colour "
                        "differences -- the full 1137-LUT library on all 1311 "
                        "pairs is ~1e11 dE00 evaluations, so the subsample size "
                        "is a flag and not a silent default.")
    i.add_argument("--eval-pairs-per-source", default="all", choices=("all", "one"),
                   help="'all' = EPR-026 section 3.6's n (1311 pairs / 120 "
                        "sources); 'one' = one seeded pair per source (120), the "
                        "protocol EPR-024's board uses.  A cross-arm paired delta "
                        "on IP-A needs both arms on the same pair set.")

    # ---- the twelve-key image board (the same one the other five arms publish)
    h = ap.add_argument_group("headline board (metrics.json)")
    h.add_argument("--headline-samples", type=int, default=0,
                   help="rows of the image-space board (0 = every normal-only "
                        "row of --eval-split).  A capped board is marked "
                        "published=false and --publish then only exercises the "
                        "assertions, it does not stand as a result.")
    h.add_argument("--lib-size", type=int, default=1137,
                   help="|Lib_tr| for B1/B2/B4/B6 (section 4.C measured its "
                        "floors on 1137 train lut_id)")
    h.add_argument("--libmean-grid", type=int, default=33,
                   help="the grid B1's library mean is baked on (affonly and "
                        "idgate use 33, carrier 65; it is on the board)")
    h.add_argument("--select-metric", default="de76", choices=("de76", "de00"),
                   help="B4/B6 pick their LUT on the 9^3 grid in this metric -- "
                        "de76 is the protocol the pre-registered floors were "
                        "measured with")
    h.add_argument("--repeats", type=int, default=8,
                   help="R draws per sample for B2 / B3")
    return ap


def config_from_args(a: argparse.Namespace, *, train_n: int | None = None
                     ) -> InterpcConfig:
    """``Namespace -> InterpcConfig``.

    ``train_n`` is the population MEASURED from ``--data``
    (:func:`q3vl.whatb.caliber.train_normal_rows`); the horizon follows from it
    (``steps_per_epoch = ceil(n / B)``) instead of from a literal.  It defaults
    to the active index口径's declared count so ``--stage setup`` / ``self-test``
    need no index read.
    """
    b, q = K.parse_batch_split(a.batch_split)
    n = int(train_n if train_n is not None else K.default_train_normal_n())
    steps_per_epoch = K.steps_per_epoch_of(n, b)
    return InterpcConfig(
        data=a.data, train_n=n,
        readout=a.readout, readout_qtok=a.readout_qtok, cond_dim=a.cond_dim,
        n_gauss=a.n_gauss, gen_width=a.gen_width, gen_mode=a.gen_mode,
        loss_level=a.loss_level, lambda_img=a.lambda_img, clamp=a.clamp,
        batch_samples=b, queries_per_sample=q, hc_eps=a.hc_eps,
        hc_mask=not a.no_hc_mask, context=a.context, lut_resample=a.lut_resample,
        mining=not a.no_mining, lr=a.lr, pi_lr_scale=a.pi_lr_scale,
        geometry_lr_scale=a.shared_geom_lr_scale, max_grad_norm=a.max_grad_norm,
        epochs=a.epochs, total_steps=steps_per_epoch * a.epochs,
        steps_per_epoch=steps_per_epoch, seed=a.seed, amp_dtype=a.amp_dtype,
        interp_weight=a.interp_weight, interp_alpha=a.interp_alpha,
        interp_p_end=a.interp_p_end, interp_where=a.interp_where,
        interp_dist=a.interp_dist, interp_target=a.interp_target,
        interp_ema_decay=a.interp_ema_decay, interp_ramp=a.interp_ramp,
        interp_stage=a.interp_stage, interp_finetune_start=a.interp_finetune_start,
        interp_pairs_per_step=a.interp_pairs_per_step, interp_hc=a.interp_hc,
        alternate=a.interp_alternate, acai_lambda=a.acai_lambda,
        acai_gamma=a.acai_gamma, jac_decay=a.jac_decay, jac_lambda=a.jac_lambda,
        degenerate_weight_tau=a.degenerate_weight_tau, eval_grid_n=a.eval_grid_n,
        ipb_k=a.ipb_k)


# --------------------------------------------------------------------------- #
# stage: setup
# --------------------------------------------------------------------------- #
def colorspan_startup_assertion(a: argparse.Namespace, texts: Sequence[str] | None
                                ) -> dict[str, Any]:
    """Frozen block section 3.2, run **before** any data loader is built."""
    if a.skip_colorspan_assert:
        return {"skipped": True,
                "why": ("--skip-colorspan-assert: this run did NOT verify the "
                        "<color> span token ids against the base tokenizer")}
    from transformers import AutoTokenizer

    from q3vl.whatb.colorspan import assert_color_span_encoding

    tok = AutoTokenizer.from_pretrained(a.checkpoint)
    if not texts:
        raise SystemExit("[interpc] the colour-span assertion needs the split's "
                         "<color> texts; none were loaded")
    return assert_color_span_encoding(tok, list(texts), n_sample=a.colorspan_samples,
                                      tokenizer_path=a.checkpoint)


def load_pair_index(split: str) -> PairIndex:
    from q3vl.whatb.splits import load_index

    return PairIndex.build(load_index(split), split=split)


def stage_setup(a: argparse.Namespace, cfg: InterpcConfig, out: Path) -> int:
    from q3vl.whatb.splits import dataset_version_facts as S_dataset_version_facts
    from q3vl.whatb.splits import load_index, split_facts

    rows = load_index(a.train_split)
    eval_rows = load_index(a.eval_split)
    pairs = PairIndex.build(rows, split=a.train_split)
    eval_pairs = PairIndex.build(eval_rows, split=a.eval_split)
    (out / "pair_index.json").write_text(pairs.to_json(), encoding="utf-8")

    texts: list[str] = []
    if not a.skip_colorspan_assert:
        from q3vl.whatb.splits import iter_records

        sample = list(eval_rows)[: max(a.colorspan_samples * 2, 8)]
        texts = [str(r.get("color", "")) for r in iter_records(sample) if r.get("color")]
    setup = run_setup_block(
        cfg, InterpcArm(cfg), pair_index=pairs, alpha_sampler=AlphaSampler(cfg),
        extra={
            "stage": "setup",
            "checkpoint": a.checkpoint,
            "provenance": provenance(),
            "dataset_version": S_dataset_version_facts(),
            "splits": {a.train_split: split_facts(rows),
                       a.eval_split: split_facts(eval_rows)},
            "eval_pair_index": eval_pairs.facts(),
            "colorspan_check": colorspan_startup_assertion(a, texts),
            "flags": vars(a),
        })
    (out / "run_setup.json").write_text(json.dumps(setup, indent=2, ensure_ascii=False),
                                        encoding="utf-8")
    (out / "loss_preregistration.json").write_text(
        json.dumps(loss_preregistration(cfg), indent=2, ensure_ascii=False),
        encoding="utf-8")
    print(json.dumps({"stage": "setup", "out": str(out),
                      "pair_index": pairs.facts(),
                      "eval_pair_index": eval_pairs.facts(),
                      "step_columns": list(cfg.step_columns())},
                     indent=2, ensure_ascii=False))
    return 0


def loss_preregistration(cfg: InterpcConfig) -> dict[str, Any]:
    """What this run promises to compute, written before it computes anything."""
    return {
        "arm": ARM, "axes": list(AXES),
        "total_loss": "L = L_GLUT + lambda_int * L_interp",
        "L_GLUT": "L_rec + 10 * L_hc + 0.001 * R_sparse  (GLUT Eq.6-8, section 4.1)",
        "L_interp": ("|| f_{G(u_alpha)}(x) - ((1-a) L_a(x) + a L_b(x)) ||_1, "
                     "u_alpha = (1-a) pi(z_a) + a pi(z_b), a ~ Beta(beta,beta)"),
        "lambda_int": cfg.interp_weight,
        "step_columns": list(cfg.step_columns()),
        "required_criteria": ["headline_normal_only", "B0_identity", "B1_libmean",
                              "B2_librandom", "B3_bucket_retrieval", "B4_oracle",
                              "N1_shuffle_delta", "N1_shuffle_M",
                              "N2_irrelevant_delta", "N2_irrelevant_M",
                              "N3_const_delta", "N3_const_M",
                              "interp_grid", "path_len", "mono_rate", "oob_rate"],
        "headline": ".contexts.all.headline_normal_only (dE00, GT alpha, short side 512)",
        "banned": ["AUC in any form", "pooled (low-mixed) headline",
                   "per-image min-max / softmax normalisation",
                   "percentile-trimmed means", "IoU as an objective",
                   "cross-step-count comparison"],
    }


# --------------------------------------------------------------------------- #
# batch assembly (shared by train and self-test)
# --------------------------------------------------------------------------- #
def lut_values(bank, x: torch.Tensor, lut_ids: Sequence[str]) -> torch.Tensor:
    """``(B, Q, 3)`` targets ``L_l(x)`` with the generator's own operator."""
    return torch.stack([bank.apply(x[i], lid) for i, lid in enumerate(lut_ids)], dim=0)


def make_pair_batch(arm_cfg: InterpcConfig, cache: Any, bank: Any,
                    draws: Sequence[tuple[str, tuple[str, str], tuple[str, str]]],
                    x: torch.Tensor, alpha: torch.Tensor, device: Any) -> PairBatch:
    ids_a = [d[1][0] for d in draws]
    ids_b = [d[2][0] for d in draws]
    lut_a = [d[1][1] for d in draws]
    lut_b = [d[2][1] for d in draws]
    return PairBatch(
        z_a=cache.z(ids_a, device=device), z_b=cache.z(ids_b, device=device),
        x=x, values_a=lut_values(bank, x, lut_a), values_b=lut_values(bank, x, lut_b),
        alpha=alpha, source_ids=tuple(d[0] for d in draws))


# --------------------------------------------------------------------------- #
# stage: train
# --------------------------------------------------------------------------- #
def stage_train(a: argparse.Namespace, cfg: InterpcConfig, out: Path) -> int:
    from q3vl.whatb.lutdata import LutBank
    from q3vl.whatb.splits import load_index, normal_only, train_source_facts

    # -- the population FIRST: the horizon is a function of it ---------------
    # every source is counted against its own on-disk declaration; the merged n
    # is never written down (q3vl/whatb/splits.py train_normal_rows).
    rows = (normal_only(load_index(a.train_split)) if a.data == "v2seg"
            else K.train_normal_rows(a.data, split=a.train_split))
    cfg = config_from_args(a, train_n=len(rows))
    caliber = K.horizon_record(
        data=a.data, n_train=len(rows), batch_split=cfg.batch_split,
        batch_samples=cfg.batch_samples,
        queries_per_sample=cfg.queries_per_sample,
        steps_per_epoch=cfg.steps_per_epoch, total_steps=cfg.total_steps,
        base_lr=cfg.lr, epochs=cfg.epochs, zcache_root_l8=a.zcache_root_l8,
        loss_level=cfg.loss_level, lambda_hc=cfg.lambda_hc_effective,
        lambda_sparse=cfg.lambda_sparse_effective)
    caliber["train_source_facts"] = (
        None if a.data == "v2seg" else train_source_facts(a.data, split=a.train_split))
    if a.dry_run:
        (out / "run_setup.json").write_text(
            json.dumps({"arm": ARM, "stage": "train", "dry_run": True,
                        "flags": vars(a), "config": cfg.as_dict(),
                        "caliber": caliber}, indent=2, ensure_ascii=False,
                       default=str), encoding="utf-8")
        print(json.dumps({"arm": ARM, "dry_run": True, "data": a.data,
                          "n_train": len(rows), "batch_split": cfg.batch_split,
                          "colours_per_step": cfg.colors_per_step,
                          "steps_per_epoch": cfg.steps_per_epoch,
                          "total_steps": cfg.total_steps, "base_lr": cfg.lr,
                          "loss_level": cfg.loss_level}, indent=2))
        return 0

    if a.z_cache is None:
        raise SystemExit(
            "[interpc] --stage train needs --z-cache DIR (the shared cache "
            "q3vl/whatb/zcache.py reads; build it with "
            "q3vl/whatb/scripts/build_zcache.py).")
    # ``--data v2seg``: the split's own cache, unchanged.  ``--data v2seg+l8``:
    # the union, through the shared MultiZCache (caliber.open_train_z).  The
    # sft2seg member is then opened once, by that call, not twice.
    if a.data == "v2seg":
        cache = open_z(a.z_cache, a.train_split, cfg, checkpoint=a.checkpoint)
        caliber["z_cache_train"] = cache.facts()
    else:
        cache = open_z(a.z_cache, a.train_split, cfg, checkpoint=a.checkpoint,
                       required=(), tags=())
        member, zrec = K.open_train_z(
            a.z_cache, a.train_split, data=a.data, checkpoint=a.checkpoint,
            readout_kind=cfg.readout, context=cfg.context,
            zcache_root_l8=a.zcache_root_l8, seed=cfg.seed)
        cache.caches["none"] = member
        cache.record["none"] = zrec
        caliber["z_cache_train"] = zrec
    # the four control caches live on the *evaluation* split (n = V_what 897);
    # asserting them here, before the loop, is HANDOFF 4.H's start-up check.
    eval_cache = open_z(a.z_cache, a.eval_split, cfg, checkpoint=a.checkpoint,
                        required=CONTROL_TAGS)
    assert_context_caches(eval_cache.meta, checkpoint=a.checkpoint,
                          readout_kind=cfg.readout)

    device = torch.device(a.device)
    torch.manual_seed(cfg.seed)
    bank = LutBank(a.bank_dir, resample=cfg.lut_resample)
    pairs = PairIndex.build(rows, split=a.train_split)

    arm = InterpcArm(cfg).to(device)
    opt = build_optimizer(arm, cfg)
    sched = build_scheduler(opt, cfg)
    teacher = EmaTeacher(arm, cfg.interp_ema_decay) if cfg.interp_target == "ema_teacher" else None
    critic = AcaiCritic().to(device) if cfg.alternate == "acai" else None
    critic_opt = torch.optim.Adam(critic.parameters(), lr=cfg.lr) if critic else None
    jac_state: dict[str, float] = {}
    queries = QuerySampler(seed=cfg.seed, q=cfg.queries_per_sample)
    alphas = AlphaSampler(cfg)
    rng = np.random.default_rng(cfg.seed)

    # the caliber, on the artefact, BEFORE the first step (the campaign rule)
    (out / "run_setup.json").write_text(
        json.dumps(run_setup_block(
            cfg, arm, pair_index=pairs, alpha_sampler=alphas,
            extra={"stage": "train", "checkpoint": a.checkpoint,
                   "provenance": provenance(), "flags": vars(a),
                   "caliber": caliber}),
            indent=2, ensure_ascii=False, default=str), encoding="utf-8")
    (out / "loss_preregistration.json").write_text(
        json.dumps(loss_preregistration(cfg), indent=2, ensure_ascii=False),
        encoding="utf-8")

    steps_path = out / "steps.jsonl"
    steps_path.unlink(missing_ok=True)
    total = int(a.max_steps or cfg.total_steps)
    order = rng.permutation(len(rows))
    cursor = 0
    guard_done = False
    t0 = time.time()

    # checkpoint selection (HANDOFF 4.H, same gate as the other five arms):
    # periodic quick eval -> .contexts.all.headline_normal_only -> best.pt.
    # Never val loss; the last-step weights are kept separately as arm_last.pt.
    select = load_selection_samples(a.eval_split, eval_cache,
                                    limit=int(a.select_samples), device=device)
    if not select:
        raise SystemExit(
            f"[interpc] the selection set is empty: no {a.eval_split} normal-only "
            "sample is in the z cache.  Without it there is no headline to select "
            "on and a cross-arm paired delta would compare a headline-selected "
            "checkpoint against a last-step one.")
    quick_path = out / "quick_eval.jsonl"
    quick_path.unlink(missing_ok=True)
    best: dict[str, Any] = {"headline": None, "step": None}
    eval_every = int(a.eval_every)
    binding, binding_reason = degeneracy_binding(a, total)

    with steps_path.open("w", encoding="utf-8") as log:
        for step in range(total):
            if cursor + cfg.batch_samples > len(order):
                order = rng.permutation(len(rows))
                cursor = 0
            picks = [rows[int(i)] for i in order[cursor: cursor + cfg.batch_samples]]
            cursor += cfg.batch_samples
            sample_ids = [r.sample_id for r in picks]
            lut_ids = [r.lut_id for r in picks]
            z = cache.z(sample_ids, device=device)

            x = queries.sample(len(picks), device=device)
            y_t = lut_values(bank, x, lut_ids)
            r = 0.0
            if cfg.mining:
                from q3vl.whatb.arms.interpc import _mining_ratio_for

                r = _mining_ratio_for(cfg, step, None)
                x_fresh = queries.sample(len(picks), device=device)
                t_fresh = lut_values(bank, x_fresh, lut_ids)
                x, y_t, _ = mine_step(arm, z, x, y_t, x_fresh, t_fresh, r)

            fit = FitBatch(z=z, x=x, target=y_t, lut_ids=tuple(lut_ids),
                           sample_ids=tuple(sample_ids))
            pair = None
            if cfg.interp_enabled or cfg.alternate == "acai":
                draws = pairs.draw(cfg.pairs_per_step, rng)
                p = len(draws)
                if p > x.shape[0]:
                    raise SystemExit(
                        "[interpc] --interp-pairs-per-step > the fit batch: the "
                        "interpolation term must share the fit stream's colours "
                        "(EPR-026:355), so P <= B.")
                pair = make_pair_batch(cfg, cache, bank, draws, x[:p],
                                       alphas.sample(p, device=device), device)

            loss, row = train_step(arm, fit, cfg, step=step, pair=pair,
                                   teacher=teacher, critic=critic,
                                   jac_state=jac_state, mining_r=r)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(arm.parameters(), cfg.max_grad_norm)
            opt.step()
            sched.step()
            if teacher is not None:
                teacher.update(arm)
            if critic is not None and critic_opt is not None and pair is not None:
                closs, crow = acai_critic_loss(arm, pair, cfg, critic)
                critic_opt.zero_grad(set_to_none=True)
                closs.backward()
                critic_opt.step()
                row.update(crow)
            row["lr"] = float(opt.param_groups[0]["lr"])
            row["elapsed_s"] = round(time.time() - t0, 3)
            log.write(json.dumps(row, ensure_ascii=False) + "\n")
            log.flush()
            if step == 0:
                assert_interp_step_columns(cfg, steps_row=row)
                (out / "step0_witness.json").write_text(
                    json.dumps(arm.step0_witness(z), indent=2), encoding="utf-8")
            due = eval_every > 0 and ((step + 1) % eval_every == 0
                                      or step + 1 == total)
            # The degeneracy guard fires at the FIRST quick eval -- the call
            # point the other five arms already use (carrier :394 / affonly :896
            # / idgate :655 / g4d :982 / qdual :831).  It used to fire at
            # ``step + 1 >= min(50, total)``, i.e. at step 49 for EVERY run with
            # total >= 50, which is the bottom of this arm's start-up trough
            # (4.97e-5 against a 1e-4 floor; see DEGENERACY_BINDING_MIN_STEPS).
            # ``or step + 1 == total`` keeps the guard reachable with
            # ``--eval-every 0``: "the check ran" must not depend on a flag.
            if not guard_done and (due or step + 1 == total):
                # exit_process=binding: a non-binding call still measures,
                # prints and records the shared witness
                # (guards.record_degeneracy_check); it raises
                # DegenerateTransform instead of SystemExit(2) so that a run too
                # short for the floor to mean anything is not killed by it.
                try:
                    witness = quick_eval_guard(arm, z, cfg,
                                               where=f"quick_eval@step{step}",
                                               exit_process=binding)
                except DegenerateTransform as exc:
                    witness = {"where": f"quick_eval@step{step}",
                               "ok": False, **exc.report.as_dict()}
                    print(f"[interpc] degeneracy verdict NOT BINDING: "
                          f"{binding_reason}", file=sys.stderr, flush=True)
                (out / "degeneracy_first_quick_eval.json").write_text(
                    json.dumps({**witness, "step": step, "binding": binding,
                                "binding_reason": binding_reason,
                                "binding_min_steps": DEGENERACY_BINDING_MIN_STEPS,
                                "total_steps": int(total)},
                               indent=2, ensure_ascii=False), encoding="utf-8")
                guard_done = True

            if not due:
                continue
            hn = selection_headline(arm, select, bank=bank, device=device)
            qrow = {"step": step + 1,
                    "contexts": {"all": {"headline_normal_only": hn}}}
            with quick_path.open("a", encoding="utf-8") as qfh:
                qfh.write(json.dumps(qrow, ensure_ascii=False) + "\n")
            print(json.dumps(qrow), flush=True)
            h = hn.get("mean")
            if h is not None and (best["headline"] is None or h < best["headline"]):
                best = {"headline": float(h), "step": int(step + 1)}
                torch.save({"model": arm.state_dict(), "config": cfg.as_dict(),
                            "step": int(step + 1),
                            "headline_normal_only": float(h)}, out / "best.pt")
    torch.save({"model": arm.state_dict(), "config": cfg.as_dict()},
               out / "arm_last.pt")
    (out / "selection.json").write_text(
        json.dumps({"best": best, "n_selection_samples": len(select),
                    "eval_every": eval_every, "split": a.eval_split,
                    "criterion": ".contexts.all.headline_normal_only.mean",
                    "degeneracy": {**(degeneracy_check_ran() or {"ran": False}),
                                   "binding": binding,
                                   "binding_reason": binding_reason,
                                   "binding_min_steps": DEGENERACY_BINDING_MIN_STEPS,
                                   "total_steps": int(total)}},
                   indent=2), encoding="utf-8")
    print(json.dumps({"stage": "train", "steps": total, "out": str(out),
                      "best": best}))
    return 0


# --------------------------------------------------------------------------- #
# stage: eval (function space -- the headline needs the image pipeline)
# --------------------------------------------------------------------------- #
def select_eval_pairs(index: PairIndex, *, mode: str = "all", seed: int = 20260810,
                      limit: int | None = None) -> list[tuple[int, int, int]]:
    """``(source_idx, i, j)`` triples for IP-A / IP-B.

    ``all`` -- every unordered pair with different ``lut_id``: EPR-026 section
    3.6's own n (V_what normal-only = **1311** pairs over 120 sources).
    ``one`` -- one seeded pair per source (**120**), which is the protocol
    EPR-024's board uses (``carrier.same_source_pairs`` with
    ``interp_pairs = 120``).  A cross-arm *paired* delta on IP-A needs both arms
    on the same pair set, so the mode is a flag rather than a silent default.
    """
    import random

    out: list[tuple[int, int, int]] = []
    if mode == "one":
        rng = random.Random(seed)
        for s in range(len(index)):
            cand = index.all_pairs(s)
            if cand:
                i, j = rng.choice(cand)
                out.append((s, i, j))
    elif mode == "all":
        out = [(s, i, j) for s in range(len(index)) for i, j in index.all_pairs(s)]
    else:
        raise ValueError("--eval-pairs-per-source must be 'all' or 'one'")
    return out if limit is None else out[: int(limit)]


def select_library_ids(lut_ids: Sequence[str], n: int, *, seed: int = 20260810
                       ) -> list[str]:
    """Section 4.C's ``Lib_tr``: ``n`` seeded ``lut_id`` out of train's 3149.

    One selection for every column that reads a library (B1/B2/B4/B6 and
    ``d_lib``), so those columns cannot end up on different libraries.
    """
    rng = np.random.default_rng(seed)
    pool = sorted(set(lut_ids))
    if n >= len(pool):
        return pool
    return [pool[int(i)] for i in rng.choice(len(pool), size=n, replace=False)]


def build_library(bank: Any, lut_ids: Sequence[str], *, n: int, grid_n: int,
                  device: Any, seed: int = 20260810):
    """``Lib_tr`` on the **evaluation grid** for the ``d_lib(alpha)`` column.

    ``d_lib(alpha) = min_l D_X(f_alpha, L_l)`` is defined on the same ``X`` as
    the rest of IP-B, so the library is evaluated on ``grid_n^3`` = the path's
    own grid.  Section 4.C's floor protocol samples 1137 ``lut_id`` out of
    train's 3149; this takes ``n`` of them (seeded), because the column costs one
    library scan per alpha per pair.
    """
    take = select_library_ids(lut_ids, n, seed=seed)
    return LibraryValues.build(bank, take, uniform_grid(grid_n).to(device))


def library_mean_volume(bank: Any, lut_ids: Sequence[str], *, grid_n: int,
                        device: Any) -> torch.Tensor:
    """B1's ``L_bar(x) = mean_l L_l(x)`` baked as one ``grid_n^3`` LUT volume.

    The accumulation (never ``|Lib| x |X|`` in memory) and the axis permute are
    ``run_affonly_arm.mean_lut_volume`` verbatim: ``.cube`` grids are stored
    ``grid[b, g, r]`` while ``uniform_grid`` indexes ``(r, g, b)``, and a wrong
    permute gives a plausible, completely wrong LUT.

    This is a fourth copy of those six lines (``carrier.library_mean_volume``,
    ``affonly.mean_lut_volume``, ``idgate.LibraryContext.mean_volume``).
    Unifying them means editing five runners hours before the wave launches, so
    it is not done here; the *shared* piece is
    ``criteria.LibraryValues.mean_transform`` and the operator is the shared
    ``lutdata.apply_lut_volume``.
    """
    x = uniform_grid(grid_n, device=device)                       # index (r, g, b)
    acc = torch.zeros_like(x)
    for lid in lut_ids:
        acc += bank.apply(x, lid)
    acc /= max(1, len(lut_ids))
    vals = acc.reshape(grid_n, grid_n, grid_n, 3)                 # [i_r, i_g, i_b, c]
    grid_bgr = vals.permute(2, 1, 0, 3).contiguous()              # [i_b, i_g, i_r, c]
    return grid_bgr.permute(3, 0, 1, 2)[None].contiguous()        # (1,3,D_b,D_g,D_r)


@torch.no_grad()
def image_board_rows(arm: InterpcArm, rows: Sequence[Any], *, cache: Any, bank: Any,
                     store: Any, records: Mapping[str, Mapping[str, Any]],
                     lib: Any, lib_mean_vol: torch.Tensor,
                     pools: Mapping[str, Sequence[str]] | None,
                     grid: torch.Tensor, device: Any, repeats: int = 8,
                     seed: int = 20260810, select_metric: str = "de76",
                     base: Mapping[str, Mapping[str, Any]] | None = None,
                     ) -> list[dict[str, Any]]:
    """One row per sample carrying every column :func:`build_board` reads.

    The same quantities, the same shared helpers and the same order as the other
    five arms (``run_carrier_arm`` / ``run_affonly_arm``): the frozen formation
    ``I_hat = (1-a) I + a f_hat(I)`` on **GT alpha at short side 512**
    (``criteria.compose_hat`` / ``image_delta_e00``, never a private copy), B4 /
    B6 chosen on the 9^3 grid in dE76 and then *re-measured* in the headline
    quantity, and the three negative controls read from the same
    :class:`~q3vl.whatb.zcache.ZCacheDir` the training loop used.

    ``base`` merges the already-computed function-space columns of the same
    sample in, so ``grid_error`` / ``unseen_color_error`` are one computation
    shared by the two boards rather than two that can disagree.
    """
    from q3vl.whatb.lutdata import apply_lut_volume

    minors = [str((records.get(r.sample_id) or {}).get("minor") or "") for r in rows]
    draws_b2 = library_random_draw(lib.lut_ids, len(rows), repeats=repeats, seed=seed)
    draws_b3 = (bucket_draw(minors, pools, repeats=repeats, seed=seed + 1)
                if pools else None)
    targets9 = {r.sample_id: bank.apply(lib.x, r.lut_id) for r in rows}
    b4 = oracle_lut_ids(lib, targets9, metric=select_metric)
    # B6 is keyed by lut_id, not by sample: exclude_self can only drop the
    # target's own library row if the key IS that row's id.
    b6 = oracle_lut_ids(lib, {r.lut_id: targets9[r.sample_id] for r in rows},
                        metric=select_metric, exclude_self=True)
    mean_vol = lib_mean_vol.to(device=device)

    out: list[dict[str, Any]] = []
    for i, r in enumerate(rows):
        img, alpha = store.load(r, device=device)
        z = cache.z([r.sample_id], device=device)
        i_star = bank.f_star_image(img, alpha, r.lut_id)

        def err(values: torch.Tensor) -> float:
            """Headline quantity for one prediction of ``f_hat(I)``."""
            return float(image_delta_e00(compose_hat(img, alpha, values), i_star))

        f_img = _transform_image(arm, z[0], img)
        i_hat = compose_hat(img, alpha, f_img)
        row: dict[str, Any] = dict((base or {}).get(r.sample_id) or {})
        row.update({
            "sample_id": r.sample_id, "winner_confidence": r.winner_confidence,
            "task_type": r.task_type, "lut_id": r.lut_id,
            "source_image_id": r.source_image_id, "minor": minors[i],
            "alpha_mean": (1.0 if isinstance(alpha, float) else float(alpha.mean())),
            "E_arm": float(image_delta_e00(i_hat, i_star)),
            "E_B0_identity": float(image_delta_e00(img, i_star)),
            "E_B1_libmean": err(apply_lut_volume(
                mean_vol, img.permute(1, 2, 0)).permute(2, 0, 1)),
            "E_B4_oracle": err(bank.apply_image(img, b4[r.sample_id][0])),
            "B4_lut_id": b4[r.sample_id][0],
            "B4_select_de76_9grid": b4[r.sample_id][1],
            "E_B2_librandom_repeats": [err(bank.apply_image(img, d[i]))
                                       for d in draws_b2],
        })
        if r.lut_id in b6:
            row["E_B6_libfill"] = err(bank.apply_image(img, b6[r.lut_id][0]))
        if draws_b3 is not None:
            vals = [err(bank.apply_image(img, d[i])) for d in draws_b3
                    if d[i] is not None]
            row["E_B3_bucket_retrieval_repeats"] = vals or None
            row["B3_bucket_missing"] = int(repeats - len(vals))

        f_grid = arm(z, grid)[0]
        for tag, name in (("shuffle", "N1_shuffle"), ("irrelevant", "N2_irrelevant"),
                          ("const", "N3_const")):
            if tag not in cache:
                continue
            z_c = cache.z([r.sample_id], tag=tag, device=device)
            row[f"E_{name}"] = err(_transform_image(arm, z_c[0], img))
            row[f"M_{name}"] = float(function_distance(f_grid, arm(z_c, grid)[0]))

        if isinstance(alpha, torch.Tensor):
            row.update(locality_errors(i_hat, i_star, img, alpha))
        out.append(row)
    return out


def evaluate_function_space(arm: InterpcArm, cfg: InterpcConfig, cache: Any,
                            bank: Any, index: PairIndex, *, device: Any,
                            max_pairs: int | None = None,
                            pair_mode: str = "all",
                            library: Any = None) -> dict[str, Any]:
    """IP-A + IP-B over the evaluation split's same-source pairs.

    V_what normal-only has 120 sources / 1311 unordered pairs; every one of its
    531 ``lut_id`` appears in train and none of its 163 source images does, so
    this measures interpolation **between two seen LUTs on an unseen image** --
    that sentence belongs in the RESULT method note (``EPR-026:568-571``).
    """
    x = uniform_grid(cfg.eval_grid_n).to(device)
    ipa: list[dict[str, Any]] = []
    ipb: list[dict[str, Any]] = []
    for s_idx, i, j in select_eval_pairs(index, mode=pair_mode, seed=cfg.seed,
                                         limit=max_pairs):
        (sid_a, lut_a), (sid_b, lut_b) = index.samples[s_idx][i], index.samples[s_idx][j]
        z_a = cache.z([sid_a], device=device)
        z_b = cache.z([sid_b], device=device)
        va, vb = bank.apply(x, lut_a), bank.apply(x, lut_b)
        ipa.append(ipa_pair_errors(arm, z_a, z_b, va, vb, x))
        ipb.append(ipb_pair_path(arm, z_a, z_b, x, k=cfg.ipb_k,
                                 tau=cfg.degenerate_weight_tau, library=library))
    return {"ipa": ipa, "ipb": ipb, "n_pairs": len(ipa), "pair_mode": pair_mode,
            "d_lib_library": (None if library is None
                              else {"n_lut": len(library.lut_ids),
                                    "n_colors": int(library.x.shape[0])})}


def stage_eval(a: argparse.Namespace, cfg: InterpcConfig, out: Path) -> int:
    from q3vl.whatb.evaldata import SampleStore
    from q3vl.whatb.lutdata import LutBank
    from q3vl.whatb.splits import bucket_pools, iter_records, load_index, normal_only

    if a.z_cache is None:
        raise SystemExit("[interpc] --stage eval needs --z-cache DIR (see --help)")
    cache = open_z(a.z_cache, a.eval_split, cfg, checkpoint=a.checkpoint,
                   required=CONTROL_TAGS)
    device = torch.device(a.device)
    bank = LutBank(a.bank_dir, resample=cfg.lut_resample)
    rows = normal_only(load_index(a.eval_split))
    index = PairIndex.build(rows, split=a.eval_split)
    arm = InterpcArm(cfg).to(device)
    # headline-selected weights, like the other five arms; arm_last.pt is only
    # the fallback for a run that predates the selection gate (and it says so on
    # the artefact rather than passing silently for the published one).
    ckpt = out / "best.pt"
    selected_from = "best.pt"
    if not ckpt.is_file():
        ckpt = out / "arm_last.pt"
        selected_from = "arm_last.pt"
    if ckpt.is_file():
        blob = torch.load(ckpt, map_location=device)
        arm.load_state_dict(blob["model"])
        checkpoint_record = {
            "file": ckpt.name, "step": blob.get("step"),
            "headline_normal_only": blob.get("headline_normal_only"),
            "selection": ".contexts.all.headline_normal_only.mean",
            "headline_selected": selected_from == "best.pt"}
    else:
        checkpoint_record = {"file": None, "headline_selected": False}
    if a.publish and not checkpoint_record["headline_selected"]:
        raise SystemExit(
            "[interpc] --publish with no best.pt: this board would carry the "
            "last-step weights while the other five arms carry headline-selected "
            "ones, and the cross-arm paired delta would not be step-matched "
            "(HANDOFF 4.H).  Re-run --stage train with --eval-every > 0.")

    z0 = cache.z([r.sample_id for r in rows[: cfg.batch_samples]], device=device)
    quick_eval_guard(arm, z0, cfg, where=f"eval@{a.eval_split}")

    train_rows = normal_only(load_index(a.train_split))
    train_luts = sorted({r.lut_id for r in train_rows})
    library = None
    if a.dlib_lib_size > 0:
        library = build_library(bank, train_luts, n=a.dlib_lib_size,
                                grid_n=cfg.eval_grid_n, device=device, seed=cfg.seed)
    res = evaluate_function_space(arm, cfg, cache, bank, index, device=device,
                                  max_pairs=a.eval_pairs,
                                  pair_mode=a.eval_pairs_per_source,
                                  library=library)
    grid = uniform_grid(cfg.eval_grid_n).to(device)
    unseen = QuerySampler(seed=cfg.seed).sample_heldout(1, 4096, device=device)[0]
    board_rows: list[dict[str, Any]] = []
    for r in rows:
        z = cache.z([r.sample_id], device=device)
        with torch.no_grad():
            f_grid = arm(z, grid)[0]
            f_unseen = arm(z, unseen)[0]
        board_rows.append({
            "sample_id": r.sample_id, "winner_confidence": r.winner_confidence,
            "task_type": r.task_type,
            "grid_error": float(function_distance(f_grid, bank.apply(grid, r.lut_id))),
            "unseen_color_error": float(
                function_distance(f_unseen, bank.apply(unseen, r.lut_id))),
        })
    extra = interp_extra_columns(res["ipa"], res["ipb"], cfg)
    interp_block = {"interp_where": cfg.interp_where, "n_pairs": res["n_pairs"],
                    "pair_mode": res["pair_mode"],
                    "d_lib_library": res["d_lib_library"],
                    "alphas": list(res["ipa"][0]["alphas"]) if res["ipa"] else []}
    board = build_board(board_rows, arm=ARM, split=a.eval_split, extra_columns=extra)
    board["interp"] = dict(interp_block)
    board["quick"] = True
    board["checkpoint"] = checkpoint_record
    board["headline_absent_because"] = (
        "this file is EPR-026's own function-space board (E^grid / unseen colour "
        "/ IP-A / IP-B); its rows carry no image column by construction.  The "
        "twelve pre-registered keys -- headline included -- are in metrics.json, "
        "built in the same process from the same checkpoint by the same "
        "criteria.build_board, and that is the board --publish gates.")
    # the result lands before the optional stage below (campaign rule): a failure
    # in the image board cannot destroy the function-space numbers already computed
    (out / "board_functionspace.json").write_text(
        json.dumps(board, indent=2, ensure_ascii=False), encoding="utf-8")

    # --- the twelve-key image board, on the other five arms' own protocol ----
    n_head = (len(rows) if a.headline_samples <= 0
              else min(int(a.headline_samples), len(rows)))
    head_rows = list(rows[:n_head])
    lib_ids = select_library_ids(train_luts, a.lib_size, seed=cfg.seed)
    lib9 = LibraryValues.build(bank, lib_ids, uniform_grid(9).to(device))
    lib_mean_vol = library_mean_volume(bank, lib_ids, grid_n=a.libmean_grid,
                                       device=device)
    pools = bucket_pools(iter_records(train_rows))
    records = {r.sample_id: rec for r, rec in zip(head_rows, iter_records(head_rows))}
    img_rows = image_board_rows(
        arm, head_rows, cache=cache, bank=bank, store=SampleStore(a.eval_split),
        records=records, lib=lib9, lib_mean_vol=lib_mean_vol, pools=pools,
        grid=grid, device=device, repeats=a.repeats, seed=cfg.seed,
        select_metric=a.select_metric,
        base={r["sample_id"]: r for r in board_rows})
    hboard = build_board(img_rows, arm=ARM, split=a.eval_split, extra_columns=extra)
    hboard["interp"] = dict(interp_block)
    hboard["checkpoint"] = checkpoint_record
    hboard["published"] = bool(n_head == len(rows))
    hboard["headline_protocol"] = {
        "formation": "I_hat = (1-a) I + a f_hat(I); I* = (1-a) I + a L_l(I)",
        "alpha": "GT alpha at short side 512 (q3vl.whatb.evaldata.SampleStore)",
        "metric": "mean dE00 over pixels (criteria.image_delta_e00)",
        "n_rows": n_head, "n_rows_available": len(rows),
        "lib_size": len(lib_ids), "libmean_grid": int(a.libmean_grid),
        "select_metric": a.select_metric, "n_repeats": int(a.repeats),
        "bucket_pools": {"n_buckets": len(pools), "source": "train records, minor"},
        "note": ("the same quantity the training loop selects best.pt on "
                 "(selection_headline); published=true only when n_rows == "
                 "n_rows_available, i.e. the whole normal-only split"),
    }
    # the board is on disk BEFORE any assertion runs: a refused board is still
    # readable, which is how a failed assertion gets diagnosed
    (out / "metrics.json").write_text(
        json.dumps(hboard, indent=2, ensure_ascii=False), encoding="utf-8")
    # the required table is checked on EVERY eval, --publish or not: "defined but
    # not wired" is exactly the failure this stage had (the pre-registration at
    # loss_preregistration() promises sixteen keys and the board carried two)
    hboard["criteria_assertion"] = assert_criteria_ran(hboard, ARM, axes=AXES)
    # publish.assert_publishable only checks that the guard RAN.  A published
    # board must additionally carry a guard that PASSED (EPR-028's rule, applied
    # to this arm): the waiver in degeneracy_binding is reachable only on a run
    # that cannot publish, and this is the assertion that says so.
    witness = degeneracy_check_ran() or {}
    hboard["degeneracy"] = dict(witness)
    if hboard.get("published") and not witness.get("ok", False):
        raise SystemExit(
            "[interpc] refusing to publish: the degenerate-solution guard did "
            f"not pass in this process (witness={witness or 'never ran'})")
    if a.publish:
        hboard["publication"] = assert_publishable_interpc(
            hboard, cfg, steps_path=out / "steps.jsonl")
    (out / "metrics.json").write_text(
        json.dumps(hboard, indent=2, ensure_ascii=False), encoding="utf-8")

    hn = ((hboard.get("contexts") or {}).get("all") or {}).get(
        "headline_normal_only") or {}
    print(json.dumps({"stage": "eval", "n_pairs": res["n_pairs"],
                      "functionspace_columns": sorted(board["criteria_columns"]),
                      "metrics_columns": sorted(hboard["criteria_columns"]),
                      "headline_normal_only": {"n": hn.get("n"),
                                               "mean": hn.get("mean")},
                      "published": hboard["published"],
                      "publish_gate": bool(a.publish)}, indent=2))
    return 0


# --------------------------------------------------------------------------- #
# stage: self-test (synthetic, CPU, no NFS / tokenizer / GPU)
# --------------------------------------------------------------------------- #
def stage_self_test(a: argparse.Namespace, cfg: InterpcConfig, out: Path) -> int:
    """Exercise the wiring end to end on tensors this function makes up itself."""
    torch.manual_seed(cfg.seed)
    cfg = replace(cfg, batch_samples=4, queries_per_sample=2048, eval_grid_n=5,
                  ipb_k=4, total_steps=8, steps_per_epoch=2)
    device = torch.device("cpu")
    arm = InterpcArm(cfg)
    opt = build_optimizer(arm, cfg)
    alphas = AlphaSampler(cfg)
    queries = QuerySampler(seed=cfg.seed, q=cfg.queries_per_sample)
    rng = np.random.default_rng(cfg.seed)
    z_all = torch.randn(8, 2560)

    # a synthetic "LUT library": three fixed channel gains
    gains = torch.tensor([[1.1, 0.9, 1.0], [0.8, 1.0, 1.2], [1.0, 1.2, 0.8]])

    def target(x: torch.Tensor, idx: Sequence[int]) -> torch.Tensor:
        return torch.stack([(x[i] * gains[k % 3]).clamp(0, 1)
                            for i, k in enumerate(idx)], dim=0)

    steps_path = out / "steps.jsonl"
    steps_path.unlink(missing_ok=True)
    with steps_path.open("w", encoding="utf-8") as log:
        for step in range(cfg.total_steps):
            idx = [int(v) for v in rng.integers(0, 8, size=cfg.batch_samples)]
            z = z_all[idx]
            x = queries.sample(cfg.batch_samples, device=device)
            fit = FitBatch(z=z, x=x, target=target(x, idx),
                           lut_ids=tuple(f"lut{k % 3}" for k in idx))
            pair = None
            if cfg.interp_enabled:
                p = min(cfg.pairs_per_step, cfg.batch_samples)
                ia = [int(v) for v in rng.integers(0, 8, size=p)]
                ib = [(k + 1) % 8 for k in ia]
                xp = x[:p]
                pair = PairBatch(z_a=z_all[ia], z_b=z_all[ib], x=xp,
                                 values_a=target(xp, ia), values_b=target(xp, ib),
                                 alpha=alphas.sample(p))
            loss, row = train_step(arm, fit, cfg, step=step, pair=pair)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            log.write(json.dumps(row) + "\n")
            if step == 0:
                assert_interp_step_columns(cfg, steps_row=row)
    quick_eval_guard(arm, z_all[:4], cfg, where="self_test_quick_eval")

    x_eval = uniform_grid(cfg.eval_grid_n)
    ipa = [ipa_pair_errors(arm, z_all[i: i + 1], z_all[i + 1: i + 2],
                           target(x_eval[None], [i])[0], target(x_eval[None], [i + 1])[0],
                           x_eval) for i in range(3)]
    ipb = [ipb_pair_path(arm, z_all[i: i + 1], z_all[i + 1: i + 2], x_eval,
                         k=cfg.ipb_k) for i in range(3)]
    rows = [{"sample_id": f"s{i}", "winner_confidence": "normal",
             "task_type": "style" if i % 2 else "local",
             "E_arm": 1.0 + 0.1 * i, "E_B0_identity": 3.0, "E_B1_libmean": 2.5,
             "E_B2_librandom_repeats": [3.5, 3.4], "E_B4_oracle": 0.9,
             "E_B3_bucket_retrieval_repeats": [3.1, 3.3],
             "E_N1_shuffle": 1.4, "M_N1_shuffle": 2.0,
             "E_N2_irrelevant": 1.5, "M_N2_irrelevant": 2.1,
             "E_N3_const": 1.6, "M_N3_const": 2.2} for i in range(6)]
    board = build_board(rows, arm=ARM, split="self_test",
                        extra_columns=interp_extra_columns(ipa, ipb, cfg))
    board["interp"] = {"interp_where": cfg.interp_where}
    board["published"] = True
    report = assert_publishable_interpc(board, cfg, steps_path=steps_path)
    (out / "self_test_board.json").write_text(
        json.dumps(board, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps({"stage": "self-test", "ok": True,
                      "criteria": report["shared"]["criteria"]["computed"],
                      "degeneracy": report["degeneracy"],
                      "interp_grid": board["criteria_columns"]["interp_grid"]["mean"]},
                     indent=2))
    return 0


# --------------------------------------------------------------------------- #
def main(argv: Sequence[str] | None = None) -> int:
    a = build_parser().parse_args(argv)
    K.apply_dataset_version(a)          # the index口径, before anything counts n
    cfg = config_from_args(a)
    name = a.run_name or f"interpc_l{a.interp_weight:g}_{a.interp_alpha}"
    out = Path(a.out_root) / "runs" / name
    out.mkdir(parents=True, exist_ok=True)
    if a.stage == "setup":
        return stage_setup(a, cfg, out)
    if a.stage == "self-test":
        return stage_self_test(a, cfg, out)
    if a.stage == "train":
        return stage_train(a, cfg, out)
    return stage_eval(a, cfg, out)


if __name__ == "__main__":                        # pragma: no cover
    raise SystemExit(main())
