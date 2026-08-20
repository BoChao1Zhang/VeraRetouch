#!/usr/bin/env python
"""Runner for the IDGATE arm (EPR-027): identity-anchored strength gate.

    CUDA_VISIBLE_DEVICES=0 PYTHONPATH=/home/bc/VeraRetouch \\
    /home/bc/envs/q3vl_sft/bin/python -m q3vl.whatb.scripts.run_idgate_arm \\
        --run-dir /home/bc/data/runs/epr027_idgate \\
        --z-cache /home/bc/data/cache/whatb_z \\
        --checkpoint /home/bc/data/runs/q3vl_base_sft_v2seg_20260814/checkpoint-4976 \\
        --eval-bundle /home/bc/data/cache/whatb_eval/V_what

Everything the run promises is written **before** the first optimisation step
(``config/run_setup.json`` + ``config/loss_preregistration.json``, with a sha256 of
the whole ``q3vl/whatb`` tree so the frozen source is on the artefact), and nothing
reaches ``metrics.json`` without :func:`q3vl.whatb.arms.idgate.publish_arm_board`.

Order of the start-up checks (each one has cost this campaign a run):

1. the frozen batch organisation -- 32 x 256 = 8192, 2936 steps/epoch, 117,440 steps;
   without it the cross-arm paired deltas are not step-matched (U4);
2. the ``<color>`` span encoder against the checkpoint's own tokenizer, 256 sampled
   texts, length first then token by token (frozen block item 7) -- **before** the
   dataloader exists;
3. the z-cache's ``checkpoint`` / ``readout_kind`` fields against this run's base;
4. at the FIRST quick eval: the three degeneracy checks and the ``u = 1`` gate wiring
   assertion.  A degenerate transform leaves via ``SystemExit(2)``; the where side
   burned 2.6 GPU-hours on a constant field for want of exactly this.

Data seams:

``--z-cache DIR``      the shared cache (:mod:`q3vl.whatb.zcache`), addressed as
                       ``DIR/<split>__<tag>/{z.npy,index.jsonl,meta.json}``, fp32
                       (ruling 11.1-5).  This arm used to carry its own reader
                       with its own layout; there is now one implementation and
                       one set of start-up assertions for all six arms.
``--eval-bundle DIR``  ``DIR/index.jsonl`` (``sample_id``, ``lut_id``, ``task_type``,
                       ``winner_confidence``, ``minor``, ``source_image_id``) +
                       ``DIR/<sample_id>.npz`` with ``image`` ``(3,H,W)`` float32 sRGB
                       at short side 512, ``alpha`` ``(H,W)``, optional ``alpha_pred``
                       / ``alpha_shuffle`` (already ``area_resize``-d to the same grid,
                       ``q3vl/where/upsample.py:54-62``), and ``z`` / ``z_null`` /
                       ``z_N1_shuffle`` / ``z_N2_irrelevant`` / ``z_N3_const``.

``--smoke`` replaces both with synthetic tensors and a handful of real bank LUTs so the
whole path -- assertions, training step, quick eval, board, publication gate -- runs on
CPU in seconds.  A smoke run stamps ``"smoke": true`` on every artefact it writes; its
numbers are not results and the board says so.
"""

from __future__ import annotations

import argparse
import contextlib
import hashlib
import json
import math
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import torch

from q3vl.whatb import caliber as K
from q3vl.whatb import criteria as _criteria
from q3vl.whatb import queries as _queries
from q3vl.whatb import splits as _splits
from q3vl.whatb.arms import idgate as ig
from q3vl.whatb.lutdata import BANK_DIR, LutBank
from q3vl.whatb.readout import WhatReadoutSpec
from q3vl.whatb.zcache import ZCache, blob_path, leaf_dir, resolve_leaf

REPO = Path("/home/bc/VeraRetouch")
DEFAULT_CHECKPOINT = "/home/bc/data/runs/q3vl_base_sft_v2seg_20260814/checkpoint-4976"
#: the two tiny splits this runner accepts for ``--smoke`` only; they are NOT in
#: the shared ``BATCH_SPLITS`` table because nothing publishes on them.
SMOKE_BATCH_SPLITS: tuple[str, ...] = ("8x64", "2x16")

#: ``--no-gate`` is section 4.1 row 1 (= EPR-024): there is no gate in the graph at
#: all.  ``IdGateArm.apply_transform`` ignores ``u`` entirely when ``cfg.gate`` is
#: false (``arms/idgate.py`` -- ``u`` is only consulted behind ``if self.cfg.gate``),
#: so the three P2 field columns, whose whole content is "hand the gate a per-pixel
#: ``u`` field and see what changes", have nothing to measure: all three would return
#: the ungated headline number and differ from it, and from each other, by exactly 0.
#: ``arms/idgate.py:1496`` already refuses to compute them for that reason
#: (``if with_fields and cfg.gate``).  They are waived on this 档 only, by name, and
#: the waiver is written onto the board as ``criteria_waived`` -- the shape
#: ``scripts/run_g4d_arm.py``'s ``ORACLE_WAIVED_CRITERIA`` uses.
#:
#: Nothing else is waived.  In particular ``gate_identity_check``, ``gate_u_hist``,
#: ``strength_dE_u`` and ``dlib_u`` ARE still computed on this 档 (verified on the
#: ``--smoke --no-gate`` board: every one carries ``n > 0``), so they stay required.
NO_GATE_WAIVED_CRITERIA: tuple[str, ...] = (
    "field_gt", "field_const", "field_shuffle",
)


def waived_criteria(cfg: ig.IdGateConfig) -> tuple[str, ...]:
    """The pre-registered keys this 档 structurally cannot compute (empty with the gate)."""
    return () if cfg.gate else NO_GATE_WAIVED_CRITERIA


# --------------------------------------------------------------------------- #
# provenance
# --------------------------------------------------------------------------- #
def provenance() -> dict[str, Any]:
    """Commit, working-tree state and a content hash of ``q3vl/whatb``.

    The arm's source is not committed, so a bare HEAD would describe code that did not
    run.  ``whatb_source_sha256`` is the freeze the operations rule refers to ("no
    source edits after the process starts").
    """
    out: dict[str, Any] = {"git_commit": "unknown", "working_tree_dirty": None,
                           "changed_files": [], "whatb_source_sha256": None,
                           "python": sys.version.split()[0],
                           "torch": torch.__version__}
    try:
        out["git_commit"] = subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=REPO).decode().strip()
        st = subprocess.check_output(["git", "status", "--porcelain"],
                                     cwd=REPO).decode().strip()
        out["working_tree_dirty"] = bool(st)
        out["changed_files"] = [ln[3:] for ln in st.splitlines()][:80]
    except Exception:
        pass
    try:
        h = hashlib.sha256()
        for f in sorted(Path(REPO, "q3vl/whatb").rglob("*.py")):
            h.update(f.read_bytes())
        out["whatb_source_sha256"] = h.hexdigest()
    except Exception:
        pass
    return out


# --------------------------------------------------------------------------- #
# arguments
# --------------------------------------------------------------------------- #
def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(
        prog="run_idgate_arm",
        description="EPR-027 IDGATE: identity-anchored strength gate outside GLUT.")

    g = ap.add_argument_group("run")
    g.add_argument("--run-dir", type=Path, required=True)
    g.add_argument("--device", default="cuda:0")
    g.add_argument("--seed", type=int, default=20260810)
    g.add_argument("--precision", choices=("bf16", "fp32"), default="bf16")
    g.add_argument("--dry-run", action="store_true",
                   help="build everything, write run_setup.json, do not train")
    g.add_argument("--smoke", action="store_true",
                   help="synthetic CPU end-to-end; the frozen-organisation check is "
                        "relaxed and every artefact is stamped smoke=true")
    g.add_argument("--eval-only", action="store_true")

    g = ap.add_argument_group("data")
    g.add_argument("--split", default="train")
    g.add_argument("--eval-split", default="V_what")
    g.add_argument("--checkpoint", default=DEFAULT_CHECKPOINT)
    g.add_argument("--z-cache", type=Path, default=None)
    g.add_argument("--context", choices=("teacher", "generated"),
                   default="generated",
                   help="which read-out context this run consumes; the "
                        "cache's own context_source field is asserted "
                        "against it at start-up (HANDOFF 4.H)")
    g.add_argument("--eval-bundle", type=Path, default=None)
    # --data / --zcache-root-l8; --batch-split and --base-lr are declared in the
    # optimiser group below (this arm owns their spelling), so they are off here
    K.add_caliber_arguments(ap, batch_split=False, base_lr=False)
    g.add_argument("--bank-dir", type=Path, default=BANK_DIR)
    g.add_argument("--lut-resample", choices=("none",), default="none",
                   help="ruling 11.1-3: LUTs are evaluated on their own grid")
    g.add_argument("--colorspan-samples", type=int, default=256)
    g.add_argument("--library-size", type=int, default=1137,
                   help="|Lib_tr| for the B columns (section 4.C measured its floors "
                        "on 1137 lut_id drawn from train)")

    g = ap.add_argument_group("carrier / generator (unchanged from EPR-024)")
    g.add_argument("--n-gauss", type=int, default=48)
    g.add_argument("--cond-dim", type=int, default=64)
    g.add_argument("--gen-width", type=int, choices=(128, 64), default=128)
    g.add_argument("--clamp", choices=("two", "one"), default="two")
    g.add_argument("--no-residual", dest="residual", action="store_false", default=True)

    g = ap.add_argument_group("gate (EPR-027:462)")
    g.add_argument("--gate", dest="gate", action="store_true", default=True)
    g.add_argument("--no-gate", dest="gate", action="store_false",
                   help="ablation row 1: bit-identical to EPR-024")
    g.add_argument("--gate-u-source", choices=ig.U_SOURCES, default="sample")
    g.add_argument("--gate-clamp", choices=ig.GATE_CLAMP_CHOICES, default="after")
    g.add_argument("--gate-p-end", type=float, default=0.2)
    g.add_argument("--gate-ku", type=int, default=4)
    g.add_argument("--gate-u-dist", choices=ig.U_DISTS, default="uniform01")
    g.add_argument("--gate-lambda-sign", choices=ig.LAMBDA_SIGNS, default="fixed")
    g.add_argument("--gate-null-prompt", choices=tuple(ig.NULL_PROMPTS), default="keep")
    g.add_argument("--gate-zhead", choices=("none", "linear"), default="none")
    g.add_argument("--gate-zhead-weight", type=float, default=0.1)
    g.add_argument("--gate-field-src", choices=ig.FIELD_SRCS, default="gt")
    g.add_argument("--gate-u-eval", default=",".join(str(u) for u in ig.DEFAULT_U_EVAL))
    g.add_argument("--gate-identity-atol", type=float, default=None,
                   help="0 reproduces EPR-027:461's literal '== 0'; unset uses the "
                        "float-arithmetic floor 4*eps and records it")

    g = ap.add_argument_group("loss")
    g.add_argument("--loss-level", type=int, choices=(1, 2, 3), default=3,
                   help="1 = the single L1 term (lambda_hc = lambda_sparse = "
                        "lambda_mono = 0, asserted before step 0)")
    g.add_argument("--lambda-hc", type=float, default=10.0)
    g.add_argument("--lambda-sparse", type=float, default=0.001)
    g.add_argument("--hc-eps", type=float, default=1e-3)
    g.add_argument("--l-rec-scale", type=float, default=1.0,
                   help="ablation row 1' (identity G1): 0.5")
    g.add_argument("--chroma-weight-src", choices=("target", "y_u"), default="target",
                   help="ablation row 1'' (identity G1'): y_u")

    g = ap.add_argument_group("optimiser (GLUT section 4.1 + App A.1)")
    # one spelling across the six arms; --lr stays as this arm's old name
    g.add_argument("--base-lr", "--lr", dest="lr", type=float, default=K.BASE_LR)
    g.add_argument("--pi-lr-scale", type=float, default=0.1)
    g.add_argument("--epochs", type=int, default=ig.FROZEN_EPOCHS)
    # the ONE table (arms/carrier.py BATCH_SPLITS) + the two --smoke-only splits
    # this runner already accepted; EPR-030 runs on 256x8192
    g.add_argument("--batch-split", default="32x256",
                   choices=list(K.batch_split_choices(SMOKE_BATCH_SPLITS)),
                   help="B x Q from q3vl.whatb.arms.carrier.BATCH_SPLITS; "
                        f"{SMOKE_BATCH_SPLITS} are --smoke-only and are not in "
                        "that table")
    g.add_argument("--mining", dest="mining", action="store_true", default=True)
    g.add_argument("--no-mining", dest="mining", action="store_false")
    g.add_argument("--max-steps", type=int, default=None,
                   help="stop early (smoke / debugging); recorded on the board")

    g = ap.add_argument_group("evaluation")
    g.add_argument("--eval-every", type=int, default=5000)
    g.add_argument("--quick-eval-n", type=int, default=64)
    g.add_argument("--grid-n", type=int, default=17)
    g.add_argument("--floor-grid-n", type=int, default=9)
    g.add_argument("--b1-grid-n", type=int, default=33)
    g.add_argument("--n-repeats", type=int, default=8)
    g.add_argument("--interp-pairs", type=int, default=64)
    g.add_argument("--strength-n", type=int, default=64)
    return ap


def config_from_args(args: argparse.Namespace, *, train_n: int | None = None
                     ) -> ig.IdGateConfig:
    """``Namespace -> IdGateConfig``.  ``train_n`` is MEASURED by the caller.

    ``--batch-split`` resolves through the ONE shared table
    (:func:`q3vl.whatb.caliber.parse_batch_split`); the horizon
    (``steps_per_epoch``, ``total_steps``) follows from ``train_n`` and ``B``.
    """
    b, q = K.parse_batch_split(args.batch_split)
    return ig.IdGateConfig(
        data=args.data,
        train_n=(int(train_n) if train_n is not None
                 else K.default_train_normal_n()),
        n_gauss=args.n_gauss, cond_dim=args.cond_dim, gen_width=args.gen_width,
        clamp=args.clamp, residual=args.residual,
        gate=args.gate, gate_u_source=args.gate_u_source, gate_clamp=args.gate_clamp,
        gate_p_end=args.gate_p_end, gate_ku=args.gate_ku,
        gate_u_dist=args.gate_u_dist, gate_lambda_sign=args.gate_lambda_sign,
        gate_null_prompt=args.gate_null_prompt, gate_zhead=args.gate_zhead,
        gate_zhead_weight=args.gate_zhead_weight, gate_field_src=args.gate_field_src,
        gate_u_eval=tuple(float(v) for v in str(args.gate_u_eval).split(",") if v != ""),
        gate_identity_atol=args.gate_identity_atol,
        lambda_hc=args.lambda_hc, lambda_sparse=args.lambda_sparse,
        hc_eps_c=args.hc_eps, loss_level=args.loss_level,
        l_rec_scale=args.l_rec_scale, chroma_weight_src=args.chroma_weight_src,
        lr=args.lr, pi_lr_scale=args.pi_lr_scale, epochs=args.epochs,
        batch_samples=b, queries=q, seed=args.seed, mining=args.mining,
        precision=args.precision, grid_n=args.grid_n, floor_grid_n=args.floor_grid_n,
        b1_grid_n=args.b1_grid_n, n_repeats=args.n_repeats,
    )


# --------------------------------------------------------------------------- #
# data seams (provisional; see the module docstring)
# --------------------------------------------------------------------------- #
def open_z_cache(root: str | Path, split: str, *, checkpoint: str,
                 readout_kind: str = "seg_color", context: str = "generated",
                 tag: str = "none") -> ZCache:
    """One leaf of the shared cache, with HANDOFF 4.H's start-up assertions.

    The reader is :mod:`q3vl.whatb.zcache` -- this arm used to carry a private
    copy with its own on-disk layout (``<split>.<context>.f32.npy``); the frozen
    block says 共同依赖只写一份, so the layout and the assertions are now the
    single shared ones and the arm only names the split it wants.
    """
    leaf = resolve_leaf(root, split, tag, context)
    if leaf is None:
        raise FileNotFoundError(
            f"no {tag!r} z cache for split {split!r} under {root} "
            f"(want {leaf_dir(root, split, tag)} or "
            f"{blob_path(root, split, tag, context)})")
    cache = ZCache(leaf)
    cache.assert_belongs_to(checkpoint=checkpoint, readout_kind=readout_kind,
                            context_source=context, control_tag=tag)
    return cache


def load_eval_bundle(root: Path, *, device: str = "cpu", limit: int | None = None,
                     short_side: int = 512) -> list[ig.EvalSample]:
    """Read the eval bundle described in the module docstring into EvalSamples."""
    root = Path(root)
    index = root / "index.jsonl"
    if not index.is_file():
        raise FileNotFoundError(
            f"{index} does not exist; --eval-bundle expects index.jsonl + "
            "<sample_id>.npz (schema in this runner's docstring)")
    rows = [json.loads(ln) for ln in index.read_text(encoding="utf-8").splitlines()
            if ln.strip()]
    if limit:
        rows = rows[:limit]
    out: list[ig.EvalSample] = []
    for r in rows:
        blob = np.load(root / f"{r['sample_id']}.npz")
        t = lambda k: (torch.from_numpy(np.asarray(blob[k], dtype=np.float32)).to(device)
                       if k in blob.files else None)
        ctrl = {k: t(f"z_{k}") for k in ("N1_shuffle", "N2_irrelevant", "N3_const")}
        out.append(ig.EvalSample(
            sample_id=str(r["sample_id"]),
            winner_confidence=str(r.get("winner_confidence", "normal")),
            task_type=str(r.get("task_type", "style")),
            lut_id=str(r["lut_id"]), image=t("image"), alpha=t("alpha"), z=t("z"),
            z_null=t("z_null"),
            z_ctrl={k: v for k, v in ctrl.items() if v is not None},
            alpha_pred=t("alpha_pred"), alpha_shuffle=t("alpha_shuffle"),
            minor=r.get("minor"), source_image_id=r.get("source_image_id")))
    return [align_fields(s, short_side=short_side) for s in out]


def align_fields(sample: ig.EvalSample, *, short_side: int = 512) -> ig.EvalSample:
    """Enforce the frozen field calibration on one sample.

    Frozen block ("where 分支输出的消费口径"): the section E strata are always measured
    on **GT alpha at short side 512**, and any field that has to be resampled goes
    through ``q3vl/where/upsample.py:54-62``'s ``area_resize`` (area down, bilinear up)
    -- the same operator the where arms publish their ``m_pix`` with.  The GT field is
    never resampled here: a GT alpha that does not already match its image means the
    bundle was built at another scale, and silently resizing it would move ``E_in`` /
    ``E_band`` / ``E_out`` without anything on the board saying so.
    """
    from q3vl.where.upsample import area_resize

    h, w = int(sample.image.shape[-2]), int(sample.image.shape[-1])
    if min(h, w) != int(short_side):
        raise ValueError(
            f"{sample.sample_id}: image short side is {min(h, w)}, the frozen headline "
            f"resolution is {short_side} (area_resize the bundle, not the criterion)")
    if tuple(sample.alpha.shape[-2:]) != (h, w):
        raise ValueError(
            f"{sample.sample_id}: GT alpha {tuple(sample.alpha.shape)} does not match "
            f"the image grid {(h, w)}; the section E strata are defined on GT alpha at "
            "short side 512 and this runner will not resample GT")
    fix = lambda t: (None if t is None else
                     (t if tuple(t.shape[-2:]) == (h, w) else
                      area_resize(t.reshape(1, 1, *t.shape[-2:]), (h, w))[0, 0]))
    return ig.EvalSample(**{**sample.__dict__,
                            "alpha_pred": fix(sample.alpha_pred),
                            "alpha_shuffle": fix(sample.alpha_shuffle)})


def synthetic_eval_samples(n: int, bank: LutBank, lut_ids: Sequence[str], *,
                           hw: tuple[int, int] = (24, 32), seed: int = 20260810,
                           device: str = "cpu") -> list[ig.EvalSample]:
    """Smoke-mode stand-ins.  Random images / fields / read-outs, real bank LUTs."""
    g = torch.Generator().manual_seed(int(seed))
    h, w = hw
    out: list[ig.EvalSample] = []
    for i in range(int(n)):
        style = (i % 2 == 0)
        alpha = torch.ones(h, w) if style else torch.rand(h, w, generator=g)
        out.append(ig.EvalSample(
            sample_id=f"smoke_{i:04d}", winner_confidence="normal",
            task_type="style" if style else "local",
            lut_id=lut_ids[i % len(lut_ids)],
            image=torch.rand(3, h, w, generator=g).to(device),
            alpha=alpha.to(device),
            z=torch.randn(2560, generator=g).to(device),
            z_ctrl={k: torch.randn(2560, generator=g).to(device)
                    for k in ("N1_shuffle", "N2_irrelevant", "N3_const")},
            alpha_pred=torch.rand(h, w, generator=g).to(device),
            alpha_shuffle=torch.rand(h, w, generator=g).to(device),
            minor=f"bucket_{i % 3}", source_image_id=f"src_{i // 2}"))
    return out


def batch_lut_values(bank: LutBank, lut_ids: Sequence[str], x: torch.Tensor
                     ) -> torch.Tensor:
    """``(B, Q, 3)`` of ``L_l(x)``, one LUT per row, on ``x``'s device."""
    out = torch.empty_like(x)
    for i, lid in enumerate(lut_ids):
        out[i] = bank.apply(x[i], str(lid))
    return out


# --------------------------------------------------------------------------- #
# start-up assertions
# --------------------------------------------------------------------------- #
def run_colorspan_assertion(checkpoint: str, texts: Sequence[str], *, n_sample: int
                            ) -> dict[str, Any]:
    """Frozen block item 7, run **before** the dataloader is built."""
    from transformers import AutoTokenizer

    from q3vl.whatb.colorspan import assert_color_span_encoding

    tok = AutoTokenizer.from_pretrained(checkpoint, trust_remote_code=True)
    return assert_color_span_encoding(tok, list(texts), n_sample=int(n_sample),
                                      tokenizer_path=str(checkpoint))


def write_json(path: Path, obj: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2, default=str), encoding="utf-8")


# --------------------------------------------------------------------------- #
# evaluation driver
# --------------------------------------------------------------------------- #
def evaluate(
    model: ig.IdGateArm,
    samples: Sequence[ig.EvalSample],
    *,
    bank: LutBank,
    grids: ig.GridSpec,
    library: ig.LibraryContext | None,
    cfg: ig.IdGateConfig,
    split: str,
    u_hist: Sequence[float],
    quick: bool,
    published: bool,
    interp_pairs: int = 64,
    strength_n: int = 64,
) -> dict[str, Any]:
    """Per-sample rows + the arm's own columns + the board (never writes anything)."""
    rng = np.random.default_rng(cfg.seed)
    rows = [ig.evaluate_sample(model, s, bank=bank, grids=grids, library=library,
                               rng=rng) for s in samples]

    subset = list(samples)[:int(strength_n)] or list(samples)
    # --gate-u-eval is the pre-registered ladder: its out-of-[0,1] points are the
    # section G(d) extrapolation column, the whole ladder is where d_lib is read
    extrap = tuple(u for u in cfg.gate_u_eval if u < 0.0 or u > 1.0) or ig.EXTRAP_U
    strength = ig.strength_columns(model, subset, bank=bank, grid=grids.grid,
                                   library=library, extrap=extrap,
                                   dlib_u=cfg.gate_u_eval or ig.DLIB_U, seed=cfg.seed)
    pairs = _same_source_pairs(samples, limit=int(interp_pairs))
    interp = (ig.interpolation_columns(model, pairs, bank=bank, grid=grids.grid)
              if pairs else None)
    ident = ig.gate_identity_check(model, grids.grid,
                                   model.theta(samples[0].z.unsqueeze(0)))
    # HANDOFF section 2.3: the degenerate-weight rate is on EVERY board
    degen = ig.degeneracy_columns(model, subset, grid=grids.grid)
    extra = ig.extra_criteria_columns(gate_identity=ident,
                                      u_histogram=ig.u_histogram(u_hist),
                                      strength=strength, interpolation=interp,
                                      degeneracy=degen)
    board = ig.build_arm_board(rows, split=split, extra_columns=extra, cfg=cfg,
                               quick=quick, published=published)
    board["per_sample_rows"] = rows
    board["n_interp_pairs"] = len(pairs)
    return board


def _same_source_pairs(samples: Sequence[ig.EvalSample], *, limit: int
                       ) -> list[tuple[ig.EvalSample, ig.EvalSample]]:
    """Section 4.F IP-A pairs: two different LUTs under the same source image."""
    by_src: dict[str, list[ig.EvalSample]] = {}
    for s in samples:
        by_src.setdefault(str(s.source_image_id), []).append(s)
    pairs: list[tuple[ig.EvalSample, ig.EvalSample]] = []
    for group in by_src.values():
        for i in range(len(group)):
            for j in range(i + 1, len(group)):
                if group[i].lut_id != group[j].lut_id:
                    pairs.append((group[i], group[j]))
                if len(pairs) >= limit:
                    return pairs
    return pairs


# --------------------------------------------------------------------------- #
# main
# --------------------------------------------------------------------------- #
def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    dataset_ver = K.apply_dataset_version(args)
    # -- the population FIRST: the horizon is ceil(n / B) on it, not on a
    # -- literal.  Every source is counted against its own on-disk declaration
    # -- (q3vl/whatb/splits.py train_normal_rows); --smoke has no split to read.
    index_rows: list[Any] | None = None
    if not args.smoke:
        index_rows = (_splits.normal_only(_splits.load_index(args.split))
                      if args.data == "v2seg"
                      else K.train_normal_rows(args.data, split=args.split))
    cfg = config_from_args(
        args, train_n=(len(index_rows) if index_rows is not None else None))
    run_dir = Path(args.run_dir)
    (run_dir / "config").mkdir(parents=True, exist_ok=True)
    device = "cpu" if args.smoke else args.device
    if device == "cpu":
        # the box runs two GPU arms plus other agents; an unbounded CPU thread pool
        # here just fights them (load average 124 on 48 cores while this was written)
        torch.set_num_threads(min(4, torch.get_num_threads()))
    torch.manual_seed(cfg.seed)

    setup: dict[str, Any] = {
        "arm": ig.ARM, "epr": ig.EPR, "smoke": bool(args.smoke),
        "args": {k: (str(v) if isinstance(v, Path) else v) for k, v in vars(args).items()},
        "config": cfg.to_dict(), "provenance": provenance(),
        "device": device, "started": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "frozen_organisation": None, "colorspan_check": None, "z_cache": None,
        "degeneracy_thresholds": cfg.degeneracy.as_dict(),
        # the read-out contract this arm consumes (EPR-024 owns the kind itself;
        # EPR-027 only reads it -- recorded so the board says which one was used)
        "readout": WhatReadoutSpec("seg_color").to_dict(),
        "null_prompt": cfg.null_prompt,
        "z_null_in_training": False,
    }

    # (1) frozen batch organisation
    if args.smoke:
        setup["frozen_organisation"] = {"skipped": "--smoke (synthetic CPU run)"}
    else:
        setup["frozen_organisation"] = ig.assert_frozen_organisation(cfg)

    bank = LutBank(args.bank_dir, resample=args.lut_resample)

    # (2) the colour-span start-up assertion, before any dataloader
    # (``ZCache`` for --data v2seg, ``MultiZCache`` for the union)
    z_train: Any = None
    if args.smoke:
        setup["colorspan_check"] = {"skipped": "--smoke (no tokenizer, no split texts)"}
    else:
        if args.z_cache is None:
            raise SystemExit("--z-cache is required (or --smoke); the condition z is a "
                             "frozen-VLM read-out and this runner does not run the VLM")
        if args.data == "v2seg":
            z_train = open_z_cache(args.z_cache, args.split,
                                   checkpoint=args.checkpoint,
                                   context=args.context)
            setup["z_cache_train_caliber"] = {"data": args.data, "members": 1}
        else:
            # ``--data v2seg+l8``: one condition over two caches, through the
            # shared MultiZCache (caliber.open_train_z -> the same call the
            # EPR-030 runs make).  Every member keeps its own assert_belongs_to.
            z_train, zrec = K.open_train_z(
                args.z_cache, args.split, data=args.data,
                checkpoint=args.checkpoint, readout_kind="seg_color",
                context=args.context, zcache_root_l8=args.zcache_root_l8,
                seed=cfg.seed)
            setup["z_cache_train_caliber"] = zrec
        # the colour texts come from the cache when it carries them and from the
        # split's own records otherwise (the .pt encoding keeps its diagnostics
        # in a sidecar and has none).  An empty list is NOT a way to skip the
        # frozen-block assertion -- assert_color_span_encoding raises on one.
        texts = z_train.color_texts()
        colorspan_source = "z cache"
        if not texts:
            colorspan_source = f"{args.split} records"
            rows0 = _splits.load_index(args.split)[: 4 * int(args.colorspan_samples)]
            texts = [str(r.get("color", "")) for r in _splits.iter_records(rows0)]
            texts = [t for t in texts if t]
        setup["colorspan_check"] = {
            **run_colorspan_assertion(args.checkpoint, texts,
                                      n_sample=args.colorspan_samples),
            "texts_from": colorspan_source}
        setup["z_cache"] = z_train.assert_belongs_to(
            checkpoint=args.checkpoint, readout_kind="seg_color",
            context_source=args.context, control_tag="none")
        null_path = resolve_leaf(args.z_cache, f"{args.split}_null", "none",
                                 args.context)
        if null_path is not None:
            setup["z_null_cache"] = open_z_cache(
                args.z_cache, f"{args.split}_null", checkpoint=args.checkpoint,
                context=args.context).facts()
        else:
            setup["z_null_cache"] = {
                "absent": str(null_path),
                "note": "NOTES 5's fallback: training runs with lambda = 1, where "
                        "z_lambda == z bit for bit, so the null read-out is only "
                        "needed by the lambda / zhead evaluation rows"}

    # (3) model / optimiser
    model = ig.IdGateArm(cfg).to(device)
    optimizer = ig.build_optimizer(model, cfg)
    total_steps = int(args.max_steps or cfg.total_steps)
    scheduler = ig.build_scheduler(optimizer, cfg.total_steps)
    setup["param_counts"] = model.param_counts
    setup["total_optimizer_steps"] = total_steps
    setup["total_optimizer_steps_frozen"] = cfg.total_steps
    setup["model"] = model.config

    # the shared EPR-030 caliber, asserted and recorded before the first step
    setup["caliber"] = K.horizon_record(
        data=args.data, n_train=cfg.train_n, batch_split=cfg.batch_split,
        batch_samples=cfg.batch_samples, queries_per_sample=cfg.queries,
        steps_per_epoch=cfg.steps_per_epoch, total_steps=cfg.total_steps,
        base_lr=cfg.lr, epochs=cfg.epochs, zcache_root_l8=args.zcache_root_l8,
        loss_level=cfg.loss_level, lambda_hc=cfg.lambda_hc_effective,
        lambda_sparse=cfg.lambda_sparse_effective)
    setup["caliber"]["train_source_facts"] = (
        None if (args.data == "v2seg" or args.smoke) else
        _splits.train_source_facts(args.data, split=args.split))
    setup["dataset_version"] = dataset_ver.facts()

    write_json(run_dir / "config" / "run_setup.json", setup)
    write_json(run_dir / "config" / "loss_preregistration.json",
               ig.loss_preregistration(cfg))
    print(json.dumps({"arm": ig.ARM, "total_optimizer_steps": total_steps,
                      "param_counts": model.param_counts, "smoke": bool(args.smoke)}),
          flush=True)
    if args.dry_run:
        print(json.dumps({"arm": ig.ARM, "dry_run": True, "data": args.data,
                          "n_train": cfg.train_n,
                          "batch_split": cfg.batch_split,
                          "colours_per_step": cfg.colors_per_step,
                          "steps_per_epoch": cfg.steps_per_epoch,
                          "total_steps": cfg.total_steps, "base_lr": cfg.lr,
                          "loss_level": cfg.loss_level}, indent=2), flush=True)
        return 0

    # (4) data
    if args.smoke:
        lut_ids = bank.lut_ids()[:8]
        eval_samples = synthetic_eval_samples(max(4, args.quick_eval_n), bank, lut_ids,
                                              seed=cfg.seed, device=device)
        train_rows = [(f"smoke_{i:04d}", lut_ids[i % len(lut_ids)]) for i in range(64)]
        def z_of(sid: str) -> torch.Tensor:
            # a *stable* per-id seed: PYTHONHASHSEED makes builtin hash() differ
            # between processes, which would make a smoke run unreproducible
            h = int(hashlib.sha256(str(sid).encode()).hexdigest()[:8], 16)
            return torch.randn(2560, generator=torch.Generator().manual_seed(h))
    else:
        rows = index_rows or []
        train_rows = [(r.sample_id, r.lut_id) for r in rows if r.sample_id in z_train]
        if len(train_rows) != len(rows):
            print(f"[idgate] {len(rows) - len(train_rows)} train rows have no z cache "
                  "entry and are skipped -- this is counted, never silent", flush=True)
        z_of = z_train.get
        if args.eval_bundle is None:
            raise SystemExit("--eval-bundle is required (or --smoke): the board needs "
                             "images and GT alpha at short side 512")
        eval_samples = load_eval_bundle(args.eval_bundle, device=device,
                                        short_side=cfg.field_resolution)

    grids = ig.GridSpec.build(cfg, device=device, n_heldout=4096 if not args.smoke else 128)
    lib_ids = sorted({lid for _, lid in train_rows})[:args.library_size]
    bucket_pools: dict[str, list[str]] = {}
    if args.smoke:
        for i, lid in enumerate(lib_ids):
            bucket_pools.setdefault(f"bucket_{i % 3}", []).append(lid)
    else:
        # B3's pools come from the sft2seg records' own ``minor``; an L8 row
        # publishes no record shard (splits.IndexRow.has_record), so it
        # contributes no bucket -- filtered here rather than KeyError-ing.
        idx = {r.sample_id: r for r in _splits.load_index(args.split)
               if r.has_record}
        bucket_pools = _splits.bucket_pools(
            _splits.iter_records([idx[sid] for sid, _ in train_rows[:20000]
                                  if sid in idx]))
    library = ig.LibraryContext.build(bank, lib_ids, grids.floor_grid,
                                      bucket_pools=bucket_pools,
                                      mean_grid_n=cfg.b1_grid_n)

    if args.eval_only:
        # the degeneracy guard is NOT tied to the training loop: --eval-only
        # would otherwise publish a board from a process that never asked
        # whether the transform is a constant field (W4).
        guard = ig.first_quick_eval_guard(
            model, torch.stack([s.z for s in eval_samples[:8]]).to(device),
            grids.grid, where="eval_only", extra={"run_dir": str(run_dir)})
        write_json(run_dir / "quick_eval" / "guard_eval_only.json", guard)
        board = evaluate(model, eval_samples, bank=bank, grids=grids, library=library,
                         cfg=cfg, split=args.eval_split, u_hist=[0.0, 1.0], quick=False,
                         published=True, interp_pairs=args.interp_pairs,
                         strength_n=args.strength_n)
        write_json(run_dir / "board_raw.json", board)
        board["publication"] = ig.publish_arm_board(board, cfg=cfg, eval_only=True)
        write_json(run_dir / "metrics.json", board)
        return 0

    # (5) training
    steps_path = run_dir / "steps.jsonl"
    if steps_path.exists():
        steps_path.unlink()          # noclobber: never append to a previous run's log
    sampler = _queries.QuerySampler(seed=cfg.seed, q=cfg.queries)
    u_gen = torch.Generator().manual_seed(cfg.seed + 1)
    order_rng = np.random.default_rng(cfg.seed)
    u_seen: list[float] = []
    best = {"step": -1, "headline": math.inf}
    guard_done = False
    first_board_done = False

    # bf16 forward with fp32 master weights (EPR-027:450).  The GLUT carrier disables
    # autocast for its own maths, so Eq.1's Mahalanobis form and exp(logpdf) stay fp32
    # while the projection and the generator MLP run in bf16.
    autocast = (torch.autocast(device_type=torch.device(device).type,
                               dtype=torch.bfloat16)
                if (cfg.precision == "bf16" and torch.device(device).type == "cuda")
                else contextlib.nullcontext())
    setup["autocast"] = {"enabled": not isinstance(autocast, contextlib.nullcontext),
                         "dtype": cfg.precision,
                         "carrier": "glut_forward disables autocast internally"}
    write_json(run_dir / "config" / "run_setup.json", setup)

    order = order_rng.permutation(len(train_rows))
    cursor = 0
    for step in range(total_steps):
        if cursor + cfg.batch_samples > len(order):
            order = order_rng.permutation(len(train_rows))
            cursor = 0
        take = order[cursor:cursor + cfg.batch_samples]
        cursor += cfg.batch_samples
        batch = [train_rows[int(i)] for i in take]
        sids = [s for s, _ in batch]
        lut_ids_b = [l for _, l in batch]
        z = torch.stack([z_of(sid) for sid in sids]).to(device)
        x = sampler.sample(len(batch), cfg.queries, device=device)
        epoch = step / max(1, cfg.steps_per_epoch)
        u_draw = ig.sample_u(len(batch), cfg.k_u, generator=u_gen, p_end=cfg.gate_p_end,
                             dist=cfg.gate_u_dist, device=device)
        u_seen.extend(u_draw.u.reshape(-1).tolist())

        with autocast:
            loss, row = ig.train_step(
                model, z=z,
                lut_values_fn=lambda xx: batch_lut_values(bank, lut_ids_b, xx),
                lut_ids=lut_ids_b, x=x, u_draw=u_draw, epoch=epoch, sampler=sampler,
                mining_ratio=_queries.mining_ratio(epoch) if cfg.mining else 0.0)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        if cfg.grad_clip:
            torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)
        optimizer.step()
        scheduler.step()

        row = {"step": step, "lr": float(optimizer.param_groups[0]["lr"]), **row}
        if step == 0:
            ig.record_first_step(row)      # tier 3 of the first-step contract
        with steps_path.open("a", encoding="utf-8") as fh:
            fh.write(json.dumps(row, default=str) + "\n")

        due = (step + 1) % max(1, args.eval_every) == 0 or step + 1 == total_steps
        if not due:
            continue
        if not guard_done:
            # FIRST quick eval: a degenerate transform must not survive it
            guard = ig.first_quick_eval_guard(
                model, torch.stack([s.z for s in eval_samples[:8]]).to(device),
                grids.grid, where=f"quick_eval@step{step}",
                extra={"run_dir": str(run_dir)})
            write_json(run_dir / "quick_eval" / f"guard_step{step}.json", guard)
            guard_done = True
        qboard = evaluate(model, eval_samples[:args.quick_eval_n], bank=bank,
                          grids=grids, library=library, cfg=cfg, split=args.eval_split,
                          u_hist=u_seen, quick=True, published=False,
                          interp_pairs=min(8, args.interp_pairs),
                          strength_n=min(8, args.strength_n))
        if not first_board_done:
            # "定义了没接线" has cost this campaign five times: the FIRST quick
            # eval asserts the WHOLE pre-registered table, so a column that was
            # defined but never wired fails in epoch 1 rather than after the
            # 117,440-step horizon (the shape run_g4d_arm.py uses).
            waived = waived_criteria(cfg)
            qboard["criteria_waived"] = list(waived)
            qboard["first_board_assertion"] = _criteria.assert_criteria_ran(
                qboard, ig.EPR,
                required=[k for k in ig.REQUIRED_CRITERIA if k not in waived])
            first_board_done = True
        write_json(run_dir / "quick_eval" / f"board_step{step}.json", qboard)
        headline = ((qboard.get("contexts") or {}).get("all") or {}).get(
            "headline_normal_only") or {}
        h = headline.get("mean")
        if h is not None and float(h) < best["headline"]:
            best = {"step": step, "headline": float(h)}
            torch.save({"model": model.state_dict(), "step": step, "headline": float(h),
                        "config": cfg.to_dict()}, run_dir / "best.pt")

    # (6) the board, written before the gate so a failed assertion cannot destroy it
    board = evaluate(model, eval_samples, bank=bank, grids=grids, library=library,
                     cfg=cfg, split=args.eval_split, u_hist=u_seen, quick=False,
                     published=True, interp_pairs=args.interp_pairs,
                     strength_n=args.strength_n)
    board["best_checkpoint"] = best
    board["smoke"] = bool(args.smoke)
    board["criteria_waived"] = list(waived_criteria(cfg))
    if not cfg.gate:
        board["criteria_waived_reason"] = (
            "--no-gate (section 4.1 row 1 = EPR-024): u is not consumed anywhere in the "
            "graph, so the three P2 field columns have no quantity to measure; "
            "arms/idgate.py refuses to compute them under this 档")
    write_json(run_dir / "board_raw.json", board)
    board["publication"] = ig.publish_arm_board(board, cfg=cfg, steps_path=steps_path,
                                                waived=waived_criteria(cfg))
    write_json(run_dir / "metrics.json", board)
    print(json.dumps({"headline_normal_only":
                      ((board["contexts"]["all"]).get("headline_normal_only") or {}
                       ).get("mean"),
                      "best": best, "smoke": bool(args.smoke)}), flush=True)
    return 0


if __name__ == "__main__":     # pragma: no cover
    raise SystemExit(main())
