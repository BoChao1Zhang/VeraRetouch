"""EPR-024 entry point: the instruction-conditioned CGLUT carrier arm.

    python -m q3vl.whatb.scripts.run_carrier_arm \\
        --run-name whatb_EPR024_main \\
        --zcache-root /home/bc/data/caches/whatb_z_20260815 \\
        --readout seg_color --cond-dim 64 --n-gauss 48 --gen-width 128 \\
        --loss-level 3 --clamp two --batch-split 32x256 --context generated

Order of operations, and why it is this order
---------------------------------------------
1. **The colour-span assertion runs before the dataloader is built** (frozen
   block §3.2, verbatim): 256 colour texts of the split are encoded by
   :mod:`q3vl.whatb.colorspan` and compared, length first then token by token,
   with ``tok("<color>" + t + "</color>")``.  Any inequality is an
   ``AssertionError`` and the run refuses to start; the sampled count and the
   mismatch count go into ``run_setup.json``.
2. **The z caches are asserted next** (§3.8): each of the four
   (``none`` / ``shuffle`` / ``irrelevant`` / ``const``) caches must carry this
   run's base checkpoint and ``--readout`` value, and 1% of its rows are replayed
   through ``verify_plan`` so the recorded readout index really carries the token
   id it claims.
3. ``run_setup.json`` and ``config/loss_preregistration.json`` are written
   **before** the first step, with the sha256 of both source files: the campaign
   rule is that source may not change once the process is running.
4. The **first quick eval** runs the three degeneracy assertions
   (:func:`q3vl.whatb.arms.carrier.quick_eval`).  A flat / identity /
   sample-invariant transform leaves via ``SystemExit(2)`` right there instead of
   burning the rest of the horizon.
5. Checkpoint selection reads ``.contexts.all.headline_normal_only`` on the
   selection subset of V_what.  **Never** validation loss.
6. Every ``metrics.json`` write goes through
   :func:`q3vl.whatb.publish.assert_publishable`, which fetches the first
   ``steps.jsonl`` row itself and distinguishes "nobody handed me a row" from
   "the row has no loss columns".

Nothing in this file computes a criterion of its own; the board comes from
:mod:`q3vl.whatb.criteria` through the arm module.
"""

from __future__ import annotations

import argparse
import json
import math
import random
import sys
import time
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import numpy as np
import torch

from q3vl.whatb import criteria as C
from q3vl.whatb.arms import carrier as A
from q3vl.whatb.colorspan import assert_color_span_encoding
from q3vl.whatb.degeneracy import DegeneracyThresholds
from q3vl.whatb.queries import DEFAULT_SEED, QuerySampler
from q3vl.whatb.guards import degeneracy_check_ran
from q3vl.whatb.zcache import MultiZCache, blob_path, leaf_dir, resolve_leaf
from q3vl.whatb import splits as S
from q3vl.whatb.splits import (
    IndexRow,
    bucket_pools,
    color_texts_of,
    extra_train_rows,
    iter_records,
    load_index,
    load_index_cached,
    normal_only,
    split_facts,
    train_source_facts,
)

DEFAULT_OUT_ROOT = "/home/bc/data/runs/what_b"


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #
def build_parser(arm: Any = A) -> argparse.ArgumentParser:
    """The CLI.  ``arm`` is the arm module whose flag surface is registered.

    EPR-030 reuses this whole entry point with its own arm module (see
    ``run_epr030_arm.py``): every arm-specific symbol below is looked up on
    ``arm``, so a second arm adds a module, not a second copy of the runner.
    """
    ap = argparse.ArgumentParser(
        prog=getattr(arm, "RUNNER_PROG", "run_carrier_arm"),
        description=getattr(arm, "RUNNER_DESC",
                            "EPR-024 instruction-conditioned CGLUT carrier arm"))
    arm.add_arguments(ap)
    g = ap.add_argument_group("run")
    g.add_argument("--run-name", default=f"whatb_{arm.ARM_NAME}")
    g.add_argument("--out-root", default=DEFAULT_OUT_ROOT)
    g.add_argument("--checkpoint", default=arm.BASE_CHECKPOINT)
    g.add_argument("--zcache-root", default=None,
                   help="directory holding <split>__<control_tag>/ caches")
    g.add_argument("--zcache-root-l8", default=str(S.L8_ZCACHE_ROOT),
                   help="z cache root of the L8 training source; only read when "
                        "--data names it (the leaf is <root>/<context>/l8_train__none)")
    g.add_argument("--bank-dir", default=str(arm.BANK_DIR))
    g.add_argument("--train-split", default="train")
    g.add_argument("--eval-split", default="V_what")
    g.add_argument("--eval-every", type=int, default=0,
                   help="steps between selection evals (0 = one epoch = "
                        "ceil(train_n / B), which is what the horizon is counted in)")
    g.add_argument("--select-samples", type=int, default=64,
                   help="V_what subset used for headline-based checkpoint selection")
    g.add_argument("--quick-samples", type=int, default=32)
    g.add_argument("--select-interp-pairs", type=int, default=8,
                   help="IP-A pairs during selection (the published board uses "
                        "--interp-pairs)")
    g.add_argument("--eval-n", type=int, default=0,
                   help="cap the final evaluation (0 = the whole normal-only split)")
    g.add_argument("--eval-only", action="store_true")
    g.add_argument("--resume", default=None)
    g.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    g.add_argument("--log-every", type=int, default=50)
    g.add_argument("--point-chunk", type=int, default=None)
    g.add_argument("--img-loss-short-side", type=int, default=arm.SHORT_SIDE,
                   help="resolution L_img is composed at under --loss-level 4; "
                        "the default is the frozen headline resolution")
    g.add_argument("--dry-run", action="store_true",
                   help="write run_setup.json and the pre-registration, then stop")
    return ap


# --------------------------------------------------------------------------- #
# start-up assertions
# --------------------------------------------------------------------------- #
def run_colorspan_assertion(checkpoint: str, rows: Sequence[IndexRow], *,
                            n_sample: int = 256, seed: int = DEFAULT_SEED
                            ) -> dict[str, Any]:
    """Frozen block §3.2, before the dataloader exists.  Never skipped."""
    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained(checkpoint, trust_remote_code=True)
    rng = random.Random(seed)
    picked = rng.sample(list(rows), min(int(n_sample), len(rows)))
    # a row carries its <color> inline (L8 manifest) or in a record shard
    # (sft2seg); the assertion is over the same population that is trained on.
    texts = color_texts_of(picked)
    return assert_color_span_encoding(tok, texts, n_sample=n_sample, seed=seed,
                                      tokenizer_path=str(checkpoint))


def open_z_caches(root: str | Path, split: str, cfg: A.CarrierConfig, *,
                  checkpoint: str, tags: Iterable[str] = A.CONTROL_TAGS,
                  required: Iterable[str] = ("none",), arm: Any = A,
                  extra_sources: Sequence[tuple[Any, str]] = ()
                  ) -> tuple[dict[str, A.ZCache], dict[str, Any]]:
    """Open and assert the caches of one split.  Returns ``(caches, record)``.

    ``extra_sources`` is a sequence of ``(root, split)`` pairs whose ``none``
    cache is unioned onto this split's through :class:`MultiZCache` -- the
    ``--data v2seg+l8`` training condition.  Every member is opened with the real
    reader and goes through its own ``assert_belongs_to``; only the ``none`` tag
    takes extras, because the three control caches (N1/N2/N3) are an evaluation
    construct and exist for the eval split alone.
    """
    root = Path(root)
    caches: dict[str, A.ZCache] = {}
    record: dict[str, Any] = {}
    for tag in tags:
        # frozen block ⑥: the three controls are regenerated reasonings, so their
        # context_source is always "generated" whatever --context says.
        ctx = cfg.context if tag == "none" else "generated"
        # either encoding of the ONE cache: the canonical directory, or the
        # offline producer's <split>.<context>.<tag>.zcache.pt blob
        path = resolve_leaf(root, split, tag, ctx)
        if path is None:
            want = f"{leaf_dir(root, split, tag)} or {blob_path(root, split, tag, ctx)}"
            if tag in set(required):
                raise FileNotFoundError(
                    f"the {tag!r} z cache for split {split} is missing ({want}); "
                    "the arm consumes z from cache and cannot generate it here")
            record[tag] = {"path": want, "present": False}
            continue
        cache = arm.ZCache(path)
        rec = cache.assert_belongs_to(checkpoint=checkpoint,
                                      readout_kind=cfg.readout,
                                      context_source=ctx, control_tag=tag,
                                      seed=cfg.seed)
        if tag == "none" and extra_sources:
            members = [cache]
            rec = {"members": [rec]}
            for extra_root, extra_split in extra_sources:
                extra_path = resolve_leaf(extra_root, extra_split, tag, ctx)
                if extra_path is None:
                    raise FileNotFoundError(
                        f"the {tag!r} z cache for split {extra_split} is missing "
                        f"under {extra_root} "
                        f"({leaf_dir(Path(extra_root) / ctx, extra_split, tag)}); "
                        "--data names that source, and the arm consumes z from "
                        "cache and cannot generate it here")
                member = arm.ZCache(extra_path)
                rec["members"].append(member.assert_belongs_to(
                    checkpoint=checkpoint, readout_kind=cfg.readout,
                    context_source=ctx, control_tag=tag, seed=cfg.seed))
                members.append(member)
            cache = MultiZCache(members, control_tag=tag)
            rec = cache.assert_belongs_to(
                checkpoint=checkpoint, readout_kind=cfg.readout,
                context_source=ctx, control_tag=tag, seed=cfg.seed)
        rec["present"] = True
        record[tag] = rec
        caches[tag] = cache
    return caches, record


def train_zcache_sources(cfg: A.CarrierConfig, args) -> list[tuple[Any, str]]:
    """The ``(root, split)`` pairs the non-sft2seg training sources live in."""
    out: list[tuple[Any, str]] = []
    for name in S.DATA_SOURCES[getattr(cfg, "data", "v2seg")]:
        if name == "l8":
            out.append((args.zcache_root_l8, S.L8_SPLIT))
        else:                                            # pragma: no cover
            raise ValueError(f"unknown training source {name!r}")
    return out


# --------------------------------------------------------------------------- #
# data assembly
# --------------------------------------------------------------------------- #
def record_fields(rows: Sequence[IndexRow], keys: Sequence[str] = ("minor", "color")
                  ) -> dict[str, dict[str, Any]]:
    """``sample_id -> {key: value}`` for the record fields the arm needs."""
    out: dict[str, dict[str, Any]] = {}
    for row, rec in zip(rows, iter_records(rows)):
        out[row.sample_id] = {k: rec.get(k) for k in keys}
    return out


def load_eval_samples(rows: Sequence[IndexRow], *, store: A.SampleStore,
                      caches: Mapping[str, A.ZCache], fields: Mapping[str, Mapping[str, Any]],
                      device: Any, limit: int | None = None,
                      arm: Any = A) -> list[A.EvalSample]:
    """Materialise :class:`~q3vl.whatb.arms.carrier.EvalSample` rows."""
    out: list[A.EvalSample] = []
    for row in rows[: limit or len(rows)]:
        if row.sample_id not in caches["none"]:
            continue
        img, alpha = store.load(row, device=device)
        controls = {tag: c.vector(row.sample_id, device=device)
                    for tag, c in caches.items()
                    if tag != "none" and row.sample_id in c}
        out.append(arm.EvalSample(
            sample_id=row.sample_id, task_type=row.task_type,
            winner_confidence=row.winner_confidence, lut_id=row.lut_id,
            source_image_id=row.source_image_id,
            minor=(fields.get(row.sample_id) or {}).get("minor"),
            image=img, alpha=alpha,
            z=caches["none"].vector(row.sample_id, device=device),
            z_controls=controls))
    return out


# --------------------------------------------------------------------------- #
# evaluation
# --------------------------------------------------------------------------- #
def evaluate_and_publish(model: A.CarrierModel, cfg: A.CarrierConfig,
                         samples: Sequence[A.EvalSample], *, bank, lib,
                         lib_mean_volume: torch.Tensor,
                         pools: Mapping[str, Sequence[str]] | None,
                         split: str, steps_path: Path | None,
                         published: bool = True, eval_only: bool = False,
                         facts: Mapping[str, Any] | None = None,
                         point_chunk: int | None = None,
                         interp_samples: Sequence[A.EvalSample] | None = None,
                         interp_limit: int | None = None,
                         arm: Any = A) -> dict[str, Any]:
    """rows -> board -> publication gate.  The gate runs before anything is written.

    ``interp_samples`` lets the selection eval build IP-A pairs from the whole
    split while scoring only its own subset: a 64-sample slice can easily contain
    no two samples of the same source, and the interpolation columns are
    pre-registered, so falling back to "no pairs" would turn a healthy run into a
    publication failure.
    """
    rows = arm.evaluate_samples(model, cfg, samples, bank=bank, lib=lib,
                                lib_mean_volume=lib_mean_volume, bucket_pools=pools,
                                seed=cfg.seed, point_chunk=point_chunk)
    extra = arm.interpolation_columns(model, cfg, interp_samples or samples, bank=bank,
                                      limit=interp_limit)
    board = arm.build_arm_board(rows, split=split, extra_columns=extra,
                                published=published, seed=cfg.seed, facts=facts)
    board["publication"] = arm.publish_board(board, cfg, steps_path=steps_path,
                                             eval_only=eval_only)
    return board


def selection_board(model: A.CarrierModel, cfg: A.CarrierConfig,
                    samples: Sequence[A.EvalSample], *, bank, split: str,
                    arm: Any = A) -> dict[str, Any]:
    """The headline, and only the headline, on the selection subset.

    Checkpoint selection reads exactly ``.contexts.all.headline_normal_only``
    (§4.B), so the per-epoch eval computes that one column instead of the whole
    board -- the arm's own transform on the image, composed the frozen way,
    against the dataset's own target.  The FULL gated board still runs at the
    first selection point, so a missing pre-registered column is caught in the
    first epoch rather than at the end of the horizon.
    """
    rows: list[dict[str, Any]] = []
    with torch.no_grad():
        for s in samples:
            img = s.image.to(device=model.device)
            alpha = s.alpha if isinstance(s.alpha, float) else s.alpha.to(model.device)
            i_star = bank.f_star_image(img, alpha, s.lut_id)
            f = model.transform_image(s.z, img)
            rows.append({"sample_id": s.sample_id, "task_type": s.task_type,
                         "winner_confidence": s.winner_confidence,
                         "E_arm": float(C.image_delta_e00(
                             C.compose_hat(img, alpha, f), i_star))})
    return C.build_board(rows, arm=arm.ARM, split=split, seed=cfg.seed)


def selection_headline(board: Mapping[str, Any]) -> float:
    """``.contexts.all.headline_normal_only.mean`` -- the ONE selection number."""
    hn = ((board.get("contexts") or {}).get("all") or {}).get("headline_normal_only") or {}
    val = hn.get("mean")
    if val is None:
        raise KeyError("the selection board has no .contexts.all.headline_normal_only")
    return float(val)


# --------------------------------------------------------------------------- #
# main
# --------------------------------------------------------------------------- #
def main(argv: list[str] | None = None, *, arm: Any = A) -> int:
    """Train / evaluate one arm.  ``arm`` supplies every arm-specific symbol.

    The default is EPR-024's carrier module, so this file's behaviour is
    unchanged; EPR-030 passes its own module (same pipeline, different backbone
    and loss) instead of forking the runner.
    """
    args = build_parser(arm).parse_args(argv)
    cfg = arm.config_from_args(args)
    device = torch.device(args.device)
    torch.manual_seed(cfg.seed)
    np.random.seed(cfg.seed)

    run_dir = Path(args.out_root) / args.run_name
    (run_dir / "config").mkdir(parents=True, exist_ok=True)
    steps_path = run_dir / "steps.jsonl"

    # -- 1. splits (each index is read exactly once) -------------------------
    data = getattr(cfg, "data", "v2seg")
    train_index = list(load_index_cached(args.train_split))
    eval_index = load_index(args.eval_split)
    train_rows = normal_only(train_index)
    eval_rows = normal_only(eval_index)
    facts = {args.train_split: split_facts(train_index),
             args.eval_split: split_facts(eval_index)}
    if args.train_split == "train":
        # the second source is unioned on only for the frozen train split; a
        # hand-picked --train-split is a debugging path and stays single-source.
        extra = extra_train_rows(data)
        if extra:
            seen = {r.sample_id for r in train_rows}
            for row in extra:
                if row.sample_id in seen:
                    raise AssertionError(
                        f"{row.sample_id} appears in two training sources; the z "
                        "caches are keyed by sample_id, so one of the two z "
                        "vectors would be silently unreachable")
                seen.add(row.sample_id)
            train_rows = list(train_rows) + list(extra)
            facts["train_sources"] = train_source_facts(data,
                                                        split=args.train_split)
    if len(train_rows) != cfg.train_n and args.train_split == "train":
        raise AssertionError(
            f"--data {data} measures {len(train_rows)} normal training rows; this "
            f"run's config says {cfg.train_n}.  Refusing to run a horizon that is "
            "not the one the population implies.")
    # the horizon is counted in the measured population, not in a remembered
    # number: "predeclared criteria must have a run-time assertion".
    want_spe = int(math.ceil(len(train_rows) / cfg.batch_samples))
    if args.train_split == "train" and cfg.steps_per_epoch != want_spe:
        raise AssertionError(
            f"steps_per_epoch is {cfg.steps_per_epoch} but ceil({len(train_rows)} / "
            f"{cfg.batch_samples}) = {want_spe}; the epoch length and the training "
            "population disagree")
    eval_every = int(args.eval_every) or cfg.steps_per_epoch

    # -- 2. the colour-span assertion, BEFORE any loader ---------------------
    colorspan = run_colorspan_assertion(args.checkpoint, train_rows, seed=cfg.seed)

    # -- 3. the z caches ------------------------------------------------------
    if not args.zcache_root:
        raise SystemExit(
            "--zcache-root is required: the condition z = norm(h[-1]) at <seg_color> "
            "is produced by a frozen-VLM pass and consumed here from cache "
            "(EPR-024 §3.6-②).")
    train_caches, train_rec = open_z_caches(
        args.zcache_root, args.train_split, cfg, checkpoint=args.checkpoint,
        tags=("none",), arm=arm,
        extra_sources=(train_zcache_sources(cfg, args)
                       if args.train_split == "train" else ()))
    eval_caches, eval_rec = open_z_caches(args.zcache_root, args.eval_split, cfg,
                                          checkpoint=args.checkpoint, arm=arm)

    # -- 4. model, optimiser, schedule ---------------------------------------
    model = arm.CarrierModel(cfg).to(device)
    if cfg.cond_trainmean:
        model.set_train_mean_z(train_caches["none"].mean(device=device))
    opt = arm.build_optimizer(model, cfg)
    sched = arm.build_scheduler(opt, cfg)
    bank = arm.open_bank(cfg, args.bank_dir)
    sampler = QuerySampler(seed=cfg.seed, q=cfg.queries_per_sample)
    # --loss-level 4 composes Î on the training images themselves; only that
    # level pays for the decode, so the store is built only when it is asked for.
    train_store = (arm.SampleStore(args.train_split, short_side=args.img_loss_short_side)
                   if cfg.loss_level >= 4 else None)

    # -- 5. the run record, before the first step ----------------------------
    trainable = [r for r in train_rows if r.sample_id in train_caches["none"]]
    if not trainable:
        raise SystemExit("no train sample has a cached z; nothing to train on")
    z0 = train_caches["none"].batch([r.sample_id for r in trainable[:8]], device=device)
    setup = arm.run_setup_record(
        cfg, model, checkpoint=args.checkpoint, colorspan_check=colorspan,
        zcaches={args.train_split: train_rec, args.eval_split: eval_rec},
        split_facts=facts, bank_facts=bank.facts(),
        thresholds=DegeneracyThresholds(),
        step0_witness=model.step0_maxabs_f_minus_id(z0),
        extra={"argv": list(argv if argv is not None else sys.argv[1:]),
               "run_dir": str(run_dir), "device": str(device),
               "n_train_with_z": len(trainable),
               "data": data, "n_train_normal": len(train_rows),
               "steps_per_epoch": cfg.steps_per_epoch,
               "total_steps": cfg.total_steps, "eval_every": eval_every})
    (run_dir / "run_setup.json").write_text(json.dumps(setup, indent=2, default=str),
                                            encoding="utf-8")
    (run_dir / "config" / "loss_preregistration.json").write_text(
        json.dumps(arm.loss_preregistration(cfg), indent=2), encoding="utf-8")
    if args.dry_run:
        print(json.dumps({"run_dir": str(run_dir), "total_steps": cfg.total_steps,
                          "step0_maxabs_f_minus_id": setup["step0_maxabs_f_minus_id"]},
                         indent=2))
        return 0

    # -- 6. evaluation fixtures ----------------------------------------------
    store = arm.SampleStore(args.eval_split)
    fields = record_fields(eval_rows)
    # B3's pool and Lib_tr are both defined over the WHOLE train index (the
    # measured 77 buckets / 3149 ids / 1137-id Lib_tr are that population), not
    # over the normal-only training subset.
    # ``train_index`` and not ``train_rows``: both populations are defined over
    # the WHOLE sft2seg train index (the measured 77 buckets / 3149 ids /
    # 1137-id Lib_tr are that population).  The five plain baselines are not a
    # function of what the arm trains on, so adding a training source does not
    # move them -- and a moved baseline would silently rebase every published Δ.
    pools = bucket_pools(iter_records([r for r in train_index if r.has_record]))
    lib_ids = _library_ids(train_index, cfg.lib_size, cfg.seed)
    lib = C.LibraryValues.build(bank, lib_ids, model.query_grid9)
    lib_vol, _ = arm.library_mean_volume(bank, lib_ids, cfg.bake_grid, device=device)

    # Images stay on the host: 567 x 3 x 512 x 640 fp32 is ~2.2 GB and
    # `evaluate_samples` moves each sample onto the model's device as it goes.
    eval_samples = load_eval_samples(eval_rows, store=store, caches=eval_caches,
                                     fields=fields, device="cpu",
                                     limit=args.eval_n or None, arm=arm)
    select_samples = eval_samples[: args.select_samples]

    # -- 7. train -------------------------------------------------------------
    best = {"headline": float("inf"), "step": -1}
    if not args.eval_only:
        rng = random.Random(cfg.seed)
        order: list[IndexRow] = []
        first_quick = True
        first_board = True
        with steps_path.open("w", encoding="utf-8") as steps_fh:
            for step in range(cfg.total_steps):
                if len(order) < cfg.batch_samples:
                    order = list(trainable)
                    rng.shuffle(order)
                batch = [order.pop() for _ in range(cfg.batch_samples)]
                z = train_caches["none"].batch([r.sample_id for r in batch], device=device)
                images = alphas = None
                if train_store is not None:
                    loaded = [train_store.load(r) for r in batch]
                    images = [im for im, _ in loaded]
                    alphas = [a for _, a in loaded]
                row = arm.train_step(model, cfg, opt, sched, step=step, z=z,
                                     lut_ids=[r.lut_id for r in batch], bank=bank,
                                     sampler=sampler, images=images, alphas=alphas)
                if step == 0 or (step + 1) % args.log_every == 0:
                    steps_fh.write(json.dumps(row) + "\n")
                    steps_fh.flush()
                if step == 0:
                    print(json.dumps(row), flush=True)

                due = (step + 1) % eval_every == 0 or step + 1 == cfg.total_steps
                if not due:
                    continue
                qz = train_caches["none"].batch(
                    [r.sample_id for r in trainable[: args.quick_samples]], device=device)
                qrow = arm.quick_eval(model, cfg, z=qz,
                                      lut_ids=[r.lut_id for r in trainable[: args.quick_samples]],
                                      bank=bank, step=step + 1, first=first_quick)
                first_quick = False
                if first_board:
                    # the whole gated board once, early: a pre-registered column
                    # that was never wired must fail in epoch 1, not at step 117k
                    board = evaluate_and_publish(
                        model, cfg, select_samples, bank=bank, lib=lib,
                        lib_mean_volume=lib_vol, pools=pools, split=args.eval_split,
                        steps_path=steps_path, published=False,
                        point_chunk=args.point_chunk, interp_samples=eval_samples,
                        interp_limit=args.select_interp_pairs, arm=arm)
                    first_board = False
                else:
                    board = selection_board(model, cfg, select_samples, bank=bank,
                                            split=args.eval_split, arm=arm)
                h = selection_headline(board)
                print(json.dumps({"step": step + 1, "quick": qrow, "headline": h}),
                      flush=True)
                if h < best["headline"]:
                    best = {"headline": h, "step": step + 1}
                    torch.save({"model": model.state_dict(), "step": step + 1,
                                "headline": h, "config": cfg.as_dict()},
                               run_dir / "best.pt")
        if (run_dir / "best.pt").is_file():
            model.load_state_dict(torch.load(run_dir / "best.pt",
                                             map_location=device)["model"])

    # -- 7b. the degeneracy guard must have RUN before anything is published --
    # It is deliberately not tied to the training loop: --eval-only would
    # otherwise reach the board without ever being asked whether the transform
    # is a constant field (PRND / CONDINST, 2.6 GPU-hours).  The witness is the
    # shared one, so this fires only when the loop did not already.
    if degeneracy_check_ran() is None:
        arm.quick_eval(model, cfg,
                       z=torch.stack([s.z for s in select_samples[: args.quick_samples]],
                                     dim=0).to(device),
                       lut_ids=[s.lut_id for s in select_samples[: args.quick_samples]],
                       bank=bank, step=-1, first=True)

    # -- 8. the published board ----------------------------------------------
    board = evaluate_and_publish(
        model, cfg, eval_samples, bank=bank, lib=lib, lib_mean_volume=lib_vol,
        pools=pools, split=args.eval_split, steps_path=steps_path,
        published=True, eval_only=args.eval_only, arm=arm,
        facts={"selection": best, "library": {"n_lut": len(lib_ids),
                                              "bake_grid": cfg.bake_grid},
               "wall_clock": time.time()},
        point_chunk=args.point_chunk)
    (run_dir / "metrics.json").write_text(json.dumps(board, indent=2, default=str),
                                          encoding="utf-8")
    print(json.dumps({"headline_normal_only": selection_headline(board),
                      "n": board["n_normal"], "best": best}, indent=2), flush=True)
    return 0


def _library_ids(rows: Sequence[IndexRow], n: int, seed: int) -> list[str]:
    """``Lib_tr``: the measured protocol draws 2500 index rows -> 1137 unique ids.

    A cap below that draws uniformly from the unique ids rather than taking an
    alphabetical prefix -- ``lut_id`` is a content hash, but a prefix is still a
    deterministic subset of the library rather than a sample of it.
    """
    rng = random.Random(seed)
    picked = rng.sample(list(rows), min(2500, len(rows)))
    ids = sorted({r.lut_id for r in picked if r.lut_id})
    if n and len(ids) > n:
        ids = sorted(rng.sample(ids, int(n)))
    return ids


if __name__ == "__main__":
    raise SystemExit(main())
