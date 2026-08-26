"""EPR-033 entry point: stage-2 LoRA on the conditional path of the span-pool arm.

    python -m q3vl.whatb.scripts.run_lora_span_arm \\
        --run-name whatb_EPR033_lora_spanpool_s2 --lora \\
        --stage1 /home/bc/data/runs/what_b/whatb_CARRIER_ro_color_span_pool/best.pt \\
        --readout color_span_pool --n-gauss 32 --loss-level 1 --epochs 4 ...

The one structural change
-------------------------
Everything is the construction that produced ``whatb_CARRIER_ro_color_span_pool``
(headline 4.0656 on V_what normal-only, n=567) -- same ``pi``, same generator,
same carrier, same pure-L1 loss, same 32x256 batch, same 17^3 board -- with a
single difference: the language tower's **last eight blocks carry LoRA on
q/k/v/o**, and the VLM forward happens inside the training step so the task loss
reaches those adapters.  ``--no-lora`` is the paired control: the same
checkpoint, the same T2 steps, the same schedule, z read from the same cache the
adapters would have been asked to reproduce.  ``Delta_LoRA`` is the difference of
those two headlines and nothing else.

Order of operations
-------------------
1. The colour-span assertion and the four z-cache assertions run first, through
   :mod:`q3vl.whatb.scripts.run_carrier_arm`'s own functions -- this arm reuses
   that runner's helpers rather than restating them.
1b. ``--carrier affonly_frozen`` then runs two more before the first step:
   ``A_geom_frozen`` (the four shared-geometry tables have ``requires_grad ==
   False`` **and** are bit-for-bit the stage-1 tables) and ``A_stage1`` (this
   carrier's ``f(x)`` on the 17^3 grid equals a freshly rebuilt
   :class:`~q3vl.whatb.arms.affonly.AffineOnlyHead`'s, bitwise, for the cached
   stage-1 conditions).  Both hold with or without ``--lora``.
2. ``--lora`` then runs four more, in this order, before the first step:
   ``A_lora`` / ``A_freeze`` (:func:`q3vl.whatb.loraspan.attach_lora`),
   ``A_reply`` (bitwise, on every plan this arm ever builds),
   ``A_step0`` (adapter on == adapter off, exactly),
   ``A_cache`` (adapter off == the cached z, to a batch-shape-measured bar).
   Each writes its measured numbers into ``run_setup.json``.
3. ``run_setup.json`` and ``config/loss_preregistration.json`` are written before
   the first step, with the sha256 of the sources.
4. The first quick eval runs the three degeneracy assertions.
5. Checkpoint selection reads ``.contexts.all.headline_normal_only``.  Never
   validation loss.

Nothing here computes a criterion of its own.  ``q3vl/whatb/arms/carrier.py``,
``q3vl/whatb/scripts/run_carrier_arm.py`` and ``q3vl/whatb/glut.py`` are imported
and **not modified**: three jobs were running against them when this arm was
written (PIDs 1884539 / 1884689 / 1884898, 2026-08-24) and the campaign rule is
that a running process's source is frozen.
"""

from __future__ import annotations

# The campaign-wide R6 guard: sqlite3 must be imported BEFORE torch, and this is
# an entry point (build_zcache.py:166-172 carries the same three lines and the
# libstdc++ CXXABI trace behind them).  This arm reads the published shard store
# and imports build_zcache's gate verdict, so it needs the guard for both.
import sqlite3  # noqa: F401  (import order is the point)

import argparse
import json
import math
import random
import sys
import time
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import torch

from q3vl.whatb import caliber as K
from q3vl.whatb import criteria as C
from q3vl.whatb import loraspan as L
from q3vl.whatb import splits as S
from q3vl.whatb.arms import carrier as A
from q3vl.whatb.degeneracy import DegeneracyThresholds
from q3vl.whatb.guards import degeneracy_check_ran
from q3vl.whatb.queries import QuerySampler
from q3vl.whatb.readout import WhatReadoutBuilder, WhatReadoutSpec
from q3vl.whatb.scripts import run_carrier_arm as R
from q3vl.whatb.splits import (
    IndexRow,
    bucket_pools,
    iter_records,
    load_index,
    load_index_cached,
    normal_only,
    split_facts,
)

ARM = L.ARM
ARM_NAME = L.ARM_NAME
READOUT = "color_span_pool"


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #
def build_parser() -> argparse.ArgumentParser:
    ap = R.build_parser(arm=A)
    ap.prog = "run_lora_span_arm"
    ap.description = ("EPR-033 stage-2: LoRA on the last eight language blocks "
                      "behind the color_span_pool readout")
    g = ap.add_argument_group("EPR-033 LoRA stage 2")
    g.add_argument("--lora", dest="lora", action="store_true", default=True,
                   help="unfreeze the adapters and run the VLM inside the step "
                        "(the treatment row)")
    g.add_argument("--no-lora", dest="lora", action="store_false",
                   help="the paired control: same checkpoint, same T2 steps, "
                        "backbone frozen, z read from cache")
    g.add_argument("--carrier", default=L.CARRIER_DEFAULT, choices=list(L.CARRIER_CHOICES),
                   help="which carrier the same LoRA ablation is run on: "
                        "'carrier' = Full Generation (22N+12, the EPR-024 form), "
                        "'affonly_frozen' = the JetLUT-p1 image-domain form "
                        "(AFFONLY shared analytic geometry loaded from --stage1 "
                        "and frozen, 12N+12 generated)")
    g.add_argument("--stage1", default=None,
                   help="the stage-1 checkpoint pi + generator are warm-started "
                        "from; both rows load the SAME file.  Default: the run "
                        "dir of the --carrier 档 (CARRIER: "
                        f"{L.STAGE1_RUN_DIR}, affonly_frozen: "
                        f"{L.AFFONLY_STAGE1_RUN_DIR})")
    g.add_argument("--lora-r", type=int, default=L.LORA_R)
    g.add_argument("--lora-alpha", type=int, default=L.LORA_ALPHA)
    g.add_argument("--lora-dropout", type=float, default=L.LORA_DROPOUT)
    g.add_argument("--lora-last-n", type=int, default=L.LORA_LAST_N_BLOCKS)
    g.add_argument("--lora-lr", type=float, default=L.LORA_LR)
    g.add_argument("--vlm-dtype", default="bfloat16", choices=["bfloat16", "float32"],
                   help="the cache this arm must reproduce was written in "
                        "bfloat16 (build_report.json)")
    g.add_argument("--vlm-attn", default="eager",
                   help="attention implementation; eager is the campaign default "
                        "(FA2/SDPA must not be fallen back to silently)")
    g.add_argument("--vlm-micro-batch", type=int, default=4,
                   help="samples per padded VLM forward inside one training "
                        "step.  Measured on one H100, 32x256, 12 steps: 2 -> "
                        "56.4 GiB / 4.74 s per step, 4 -> 59.4 GiB / 4.41 s, "
                        "8 -> 62.4 GiB / 4.33 s (allocated, training phase only)")
    g.add_argument("--model-dir", default=None,
                   help="processor directory (defaults to --checkpoint)")
    g.add_argument("--parity-samples", type=int, default=8,
                   help="samples A_step0 and A_cache are measured on")
    g.add_argument("--parity-tol", type=float, default=1e-3)
    g.add_argument("--parity-factor", type=float, default=4.0)
    g.add_argument("--live-z-select", type=int, default=128,
                   help="how many V_what samples get a live-LoRA z at each "
                        "selection eval (the headline is scored on the first "
                        "--select-samples of them; IP-A pairs are drawn from all)")
    g.add_argument("--diag-every", type=int, default=200,
                   help="steps between the lora_grad_norm / z_drift columns")
    # N1/N2/N3 have no off switch on purpose: they are pre-registered columns
    # and ``criteria.assert_criteria_ran`` refuses a board without them, so a
    # "skip the controls" flag can only ever throw a finished run away at the
    # publication gate.  On the smoke fixtures the three cost 30 s.
    g.add_argument("--smoke", type=int, default=0,
                   help="run N steps with small eval fixtures and stop; prints "
                        "the wall-clock and peak-memory row")
    return ap


def resolve_stage1(args) -> str:
    """``--stage1``, or the ``--carrier`` 档's own stage-1 run dir.

    Resolved rather than defaulted on the flag so that ``--carrier
    affonly_frozen`` without ``--stage1`` cannot silently warm-start from the
    Full-Generation board -- the two checkpoints have different key names and
    the load would raise, but the message would be about tensor names rather
    than about the 档.
    """
    if args.stage1:
        return str(args.stage1)
    root = (L.AFFONLY_STAGE1_RUN_DIR if args.carrier == "affonly_frozen"
            else L.STAGE1_RUN_DIR)
    return str(Path(root) / "best.pt")


def apply_smoke(args) -> None:
    """``--smoke N`` -> the small-fixture defaults, unless the flag was given."""
    n = int(args.smoke)
    if n <= 0:
        return
    args.max_steps = n
    args.eval_every = max(1, n // 2)
    args.log_every = 1
    args.select_samples = 8
    args.quick_samples = 8
    args.select_interp_pairs = 2
    args.interp_pairs = 2
    args.eval_n = 64
    args.live_z_select = 64
    args.lib_size = 16
    args.diag_every = max(1, n // 4)
    args.parity_samples = min(int(args.parity_samples), 4)


# --------------------------------------------------------------------------- #
# the live-z side
# --------------------------------------------------------------------------- #
def cache_row(cache: Any, sample_id: str) -> dict[str, Any]:
    """The two fields a reply plan is replayed from, through the public reader.

    ``ZCache.field`` is the sanctioned accessor; reaching into ``cache.rows`` /
    ``cache._index`` would couple this arm to the internals of a module three
    live jobs import.
    """
    return {"sample_id": str(sample_id),
            "reply_token_ids": cache.field(sample_id, "reply_token_ids"),
            "readout_index": cache.field(sample_id, "readout_index")}


class LiveZ:
    """``sample_id -> z`` computed now, through the adapters, from cached replies.

    The reply is **not** regenerated: the plan comes from the z cache's own
    ``reply_token_ids`` (:func:`q3vl.whatb.loraspan.plan_from_cache_row`, which
    checks it token for token), so the only thing that moves between stage 1 and
    stage 2 is the tower the forward runs through.  Regenerating would confound
    "the adapters changed the hidden state" with "the adapters changed which
    words were written".
    """

    def __init__(self, *, split: str, encoder: L.SpanPoolEncoder, collator,
                 builder: WhatReadoutBuilder, dataset, shim,
                 control_instructions: Mapping[str, Mapping[str, str]] | None = None):
        self.split = split
        self.encoder = encoder
        self.collator = collator
        self.builder = builder
        self.dataset = dataset
        self.shim = shim
        self.control_instructions = dict(control_instructions or {})
        self._pos = {ref.sample_id: i for i, ref in enumerate(dataset.refs)}
        self._prompt_cache: dict[tuple[str, str], list[int]] = {}

    def __contains__(self, sample_id: str) -> bool:
        return str(sample_id) in self._pos

    def item(self, row: Mapping[str, Any], *, tag: str = "none") -> L.SpanItem:
        sid = str(row["sample_id"])
        s = self.dataset[self._pos[sid]]
        if tag == "none":
            instr = None
        else:
            table = self.control_instructions.get(tag) or {}
            if sid not in table:
                raise KeyError(
                    f"{sid}: the {tag!r} cache has a z for this sample but the "
                    "perturbed instruction its forward was conditioned on could "
                    "not be rebuilt.  Re-encoding it under the TRUE instruction "
                    "would void the control (build_zcache frozen block 6), so "
                    "the run stops instead.")
            instr = table[sid]
        enc = self.collator.encode_one(self.shim(s, instruction=instr))
        pids = enc["input_ids"][: enc["n_prompt_tokens"]]
        plan = L.plan_from_cache_row(self.builder, row, source="generated",
                                     control_tag=tag)
        return L.SpanItem(sample_id=sid, image=s.image, prompt_ids=pids, plan=plan)

    def z(self, rows: Sequence[Mapping[str, Any]], *, tag: str = "none",
          grad: bool = True) -> torch.Tensor:
        return self.encoder.z([self.item(r, tag=tag) for r in rows], grad=grad)


def control_instruction_table(split: str, tags: Sequence[str], *,
                              seed: int) -> dict[str, dict[str, str]]:
    """The perturbed instruction each control row's forward was conditioned on.

    Rebuilt with :mod:`q3vl.whatb.scripts.build_zcache`'s own functions and the
    cache's own seed, replayed in the cache's own row order -- ``irrelevant``
    breaks ties with a seeded draw, so the order is part of the definition.  The
    reconstruction is not trusted on its own: ``A_cache`` re-encodes control rows
    with the adapter off and compares against the cached z, which is exactly the
    measurement that says whether the prompt came out right.
    """
    from q3vl.whatb.scripts.build_zcache import (
        CONST_PHRASE, control_instruction, plan_rows, shuffle_partner, _records,
    )

    rows = plan_rows(split, rows_filter="all")
    records = _records(rows)
    instructions = {sid: str(rec.get("instruction", ""))
                    for sid, rec in records.items()}
    partners = shuffle_partner(rows, seed=seed)
    fillers = sorted({str(rec.get("where", "")).strip()
                      for rec in records.values()
                      if str(rec.get("where", "")).strip()})
    out: dict[str, dict[str, str]] = {}
    for tag in tags:
        if tag == "none":
            continue
        rng = random.Random(seed)
        table: dict[str, str] = {}
        for row in rows:
            instr = control_instruction(tag, row, instructions=instructions,
                                        partners=partners, fillers=fillers, rng=rng)
            if instr:
                table[row.sample_id] = instr
        out[tag] = table
        if tag == "const" and set(table.values()) != {CONST_PHRASE}:  # pragma: no cover
            raise AssertionError("the const control is not a single fixed phrase")
    return out


def refresh_live_z(samples: Sequence[A.EvalSample], live: LiveZ,
                   caches: Mapping[str, Any], *, tags: Sequence[str],
                   micro: int = 8) -> dict[str, int]:
    """Overwrite ``EvalSample.z`` / ``.z_controls`` with adapter-current values."""
    counts: dict[str, int] = {}
    for tag in tags:
        cache = caches.get(tag)
        if cache is None:
            continue
        owners = [s for s in samples if s.sample_id in cache]
        rows = [cache_row(cache, s.sample_id) for s in owners]
        zs: list[torch.Tensor] = []
        for i in range(0, len(rows), micro):
            zs.append(live.z(rows[i:i + micro], tag=tag, grad=False).cpu())
        if not zs:
            continue
        z = torch.cat(zs, dim=0)
        for s, v in zip(owners, z):
            if tag == "none":
                s.z = v
            else:
                s.z_controls[tag] = v
        counts[tag] = len(owners)
    return counts


# --------------------------------------------------------------------------- #
# main
# --------------------------------------------------------------------------- #
def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    apply_smoke(args)
    ver = K.apply_dataset_version(args)
    if args.readout != READOUT:
        raise SystemExit(
            f"--readout must be {READOUT!r} for this arm: EPR-033 is the LoRA "
            f"row of the span-pool readout board, got {args.readout!r}")
    cfg = A.config_from_args(args)
    device = torch.device(args.device)
    torch.manual_seed(cfg.seed)
    np.random.seed(cfg.seed)

    run_dir = Path(args.out_root) / args.run_name
    (run_dir / "config").mkdir(parents=True, exist_ok=True)
    steps_path = run_dir / "steps.jsonl"

    # -- 1. splits ------------------------------------------------------------
    train_index = list(load_index_cached(args.train_split))
    eval_index = load_index(args.eval_split)
    train_rows = normal_only(train_index)
    eval_rows = normal_only(eval_index)
    facts = {args.train_split: split_facts(train_index),
             args.eval_split: split_facts(eval_index)}
    if len(train_rows) != cfg.train_n and args.train_split == "train":
        raise AssertionError(
            f"the index measures {len(train_rows)} normal training rows, this "
            f"run's config says {cfg.train_n}")
    want_spe = int(math.ceil(len(train_rows) / cfg.batch_samples))
    if args.train_split == "train" and cfg.steps_per_epoch != want_spe:
        raise AssertionError(
            f"steps_per_epoch {cfg.steps_per_epoch} != ceil({len(train_rows)} / "
            f"{cfg.batch_samples}) = {want_spe}")
    eval_every = int(args.eval_every) or cfg.steps_per_epoch

    # -- 2. the colour-span assertion, before any loader ----------------------
    colorspan = R.run_colorspan_assertion(args.checkpoint, train_rows, seed=cfg.seed)

    # -- 3. the z caches ------------------------------------------------------
    if not args.zcache_root:
        raise SystemExit(
            "--zcache-root is required: this arm replays the cache's reply spans "
            "and (in the control row) its z, and asserts against its z in both.")
    train_caches, train_rec = R.open_z_caches(
        args.zcache_root, args.train_split, cfg, checkpoint=args.checkpoint,
        tags=("none",), arm=A, dataset_version=ver.name)
    eval_tags = A.CONTROL_TAGS
    eval_caches, eval_rec = R.open_z_caches(
        args.zcache_root, args.eval_split, cfg, checkpoint=args.checkpoint,
        tags=eval_tags, arm=A, dataset_version=ver.name)

    # -- 4. model, warm start, optimiser --------------------------------------
    assertions: dict[str, Any] = {}
    stage1_path = resolve_stage1(args)
    model = L.build_model(cfg, carrier=args.carrier).to(device)
    stage1 = torch.load(stage1_path, map_location=device, weights_only=False)
    try:
        stage1_facts = L.load_stage1(model, stage1, carrier=args.carrier,
                                     path=stage1_path)
    except AssertionError as exc:
        raise SystemExit(str(exc)) from exc
    # the shared analytic geometry is frozen BEFORE the optimiser is built, so
    # the four tables can never enter a param group in the first place
    if args.carrier == "affonly_frozen":
        assertions["freeze_shared_geometry"] = L.freeze_shared_geometry(model)
    carrier_rec = L.carrier_facts(model, args.carrier)

    # -- 5. the LoRA side ------------------------------------------------------
    vlm = encoder = live_train = live_eval = None
    lora_facts: dict[str, Any] = {"lora": False}
    lora_params: list[torch.Tensor] = []
    if args.lora:
        from q3vl.train.collator import Sft2SegCollator
        from q3vl.train.modeling import load_model, load_processor
        from q3vl.whereb.data import _PromptShim, open_dataset

        model_dir = args.model_dir or args.checkpoint
        processor, special_ids = load_processor(model_dir, 2048)
        base = load_model(args.checkpoint, attn_implementation=args.vlm_attn,
                          dtype=args.vlm_dtype).to(device).eval()
        vlm, facts_lora = L.attach_lora(
            base, r=args.lora_r, alpha=args.lora_alpha, dropout=args.lora_dropout,
            last_n=args.lora_last_n)
        vlm.train()
        lora_params = L.trainable_lora_parameters(vlm)
        encoder = L.SpanPoolEncoder(vlm, processor, device=device,
                                    micro_batch=args.vlm_micro_batch)
        collator = Sft2SegCollator(processor, max_length=2048, system_prompt=None)
        builder = WhatReadoutBuilder(collator.tokenizer, WhatReadoutSpec(kind=READOUT),
                                     control_color="regenerated")
        builder.run_colorspan_assertion(train_caches["none"].color_texts(256))
        ds_train, ds_train_info = open_dataset(args.train_split, need_mask=False)
        ds_eval, ds_eval_info = open_dataset(args.eval_split, need_mask=False)
        ctl = (control_instruction_table(args.eval_split,
                                         [t for t in eval_tags if t != "none"],
                                         seed=int(eval_caches["none"].meta.get(
                                             "seed", cfg.seed)))
               if len(eval_tags) > 1 else {})
        live_train = LiveZ(split=args.train_split, encoder=encoder, collator=collator,
                           builder=builder, dataset=ds_train, shim=_PromptShim)
        live_eval = LiveZ(split=args.eval_split, encoder=encoder, collator=collator,
                          builder=builder, dataset=ds_eval, shim=_PromptShim,
                          control_instructions=ctl)
        lora_facts = {"lora": True, **facts_lora.to_dict(),
                      "lora_lr": float(args.lora_lr),
                      "encoder": encoder.facts(),
                      "special_token_ids": special_ids,
                      "dataset": {args.train_split: ds_train_info,
                                  args.eval_split: ds_eval_info}}

    opt = L.build_optimizer(model, cfg, lora_params=lora_params,
                            lora_lr=args.lora_lr)
    sched = A.build_scheduler(opt, cfg)
    bank = A.open_bank(cfg, args.bank_dir)
    sampler = QuerySampler(seed=cfg.seed, q=cfg.queries_per_sample)

    trainable = [r for r in train_rows if r.sample_id in train_caches["none"]]
    if not trainable:
        raise SystemExit("no train sample has a cached z; nothing to train on")
    if args.lora:
        trainable = [r for r in trainable if r.sample_id in live_train]
        if not trainable:
            raise SystemExit(
                "no cached train sample is reachable in the shard dataset; the "
                "cache and the dataset disagree")

    # -- 6. the start-up assertions -------------------------------------------
    # 6a. the carrier 档's two, before the LoRA four: they are statements about
    # the model stage 2 starts from, and they hold with or without --lora.
    if args.carrier == "affonly_frozen":
        n_geom = max(2, int(args.parity_samples))
        z_geom = train_caches["none"].batch(
            [r.sample_id for r in trainable[:n_geom]], device=device)
        assertions["A_geom_frozen"] = L.assert_geometry_frozen(
            model, stage1, carrier=args.carrier)
        assertions["A_stage1"] = L.assert_stage1_forward_identity(
            model, stage1, z_geom)
        print(json.dumps({"carrier_startup_assertions": {
            k: assertions[k] for k in ("freeze_shared_geometry", "A_geom_frozen",
                                       "A_stage1") if k in assertions}},
            indent=2, default=str), flush=True)

    if args.lora:
        # A_lora / A_freeze already ran inside attach_lora; their measured
        # records are carried onto the artefact next to the three below.
        assertions.update(dict(facts_lora.assertions))
        cache = train_caches["none"]
        probe_rows = [cache_row(cache, r.sample_id)
                      for r in trainable[: max(1, int(args.parity_samples))]]
        probe_items = [live_train.item(r) for r in probe_rows]   # A_reply, bitwise
        assertions["A_reply"] = {
            "n_checked": len(probe_items), "bitwise": True,
            "scope": "every plan this arm builds goes through the same check "
                     "(loraspan.plan_from_cache_row), not only these rows"}
        assertions["A_step0"] = L.assert_lora_step0_identity(encoder, probe_items)
        z_ref = cache.batch([r["sample_id"] for r in probe_rows], device=device)
        assertions["A_cache"] = {args.train_split: L.assert_cache_parity(
            encoder, probe_items, z_ref, tol=args.parity_tol,
            factor=args.parity_factor)}
        # the same measurement on each control leaf.  A control's forward was
        # conditioned on a PERTURBED instruction which the cache does not store;
        # control_instruction_table rebuilds it, and re-encoding the row with the
        # adapter off against the cached z is the only thing that says whether the
        # rebuild is the one the cache was written under.  Getting it wrong would
        # put the TRUE instruction back into N1/N2/N3 and void them silently
        # (build_zcache frozen block 6).
        n_ctl = max(1, int(args.parity_samples))
        for tag in eval_tags:
            ctl_cache = eval_caches.get(tag)
            if ctl_cache is None:
                continue
            ids = [s for s in ctl_cache.ids if s in live_eval][:n_ctl]
            if not ids:
                raise SystemExit(
                    f"the {tag!r} cache has no sample this arm can re-encode; "
                    "its parity could not be measured")
            ctl_items = [live_eval.item(cache_row(ctl_cache, s), tag=tag)
                         for s in ids]
            assertions["A_cache"][f"{args.eval_split}__{tag}"] = L.assert_cache_parity(
                encoder, ctl_items,
                ctl_cache.batch(ids, device=device),
                tol=args.parity_tol, factor=args.parity_factor)
        print(json.dumps({"lora_startup_assertions": assertions}, indent=2),
              flush=True)

    # -- 7. the run record, before the first step -----------------------------
    z0 = train_caches["none"].batch([r.sample_id for r in trainable[:8]], device=device)
    setup = A.run_setup_record(
        cfg, model, checkpoint=args.checkpoint, colorspan_check=colorspan,
        zcaches={args.train_split: train_rec, args.eval_split: eval_rec},
        split_facts=facts, bank_facts=bank.facts(),
        thresholds=DegeneracyThresholds(),
        step0_witness=model.step0_maxabs_f_minus_id(z0),
        extra={"arm": ARM, "arm_name": ARM_NAME,
               "argv": list(argv if argv is not None else sys.argv[1:]),
               "run_dir": str(run_dir), "device": str(device),
               "stage1": stage1_facts, "stage2_steps": cfg.total_steps,
               "carrier_mode": carrier_rec,
               "lora": lora_facts, "startup_assertions": assertions,
               "source_sha256": {
                   "loraspan.py": A._sha256(Path(L.__file__)),
                   "run_lora_span_arm.py": A._sha256(Path(__file__))},
               "n_train_with_z": len(trainable),
               "dataset_version": ver.facts(),
               "n_train_normal": len(train_rows),
               "steps_per_epoch": cfg.steps_per_epoch,
               "total_steps": cfg.total_steps, "eval_every": eval_every,
               "optimizer_groups": [{"name": g.get("name"), "lr": g["lr"],
                                     "n_tensors": len(g["params"])}
                                    for g in opt.param_groups]})
    (run_dir / "run_setup.json").write_text(json.dumps(setup, indent=2, default=str),
                                            encoding="utf-8")
    (run_dir / "config" / "loss_preregistration.json").write_text(
        json.dumps(A.loss_preregistration(cfg), indent=2), encoding="utf-8")
    if args.dry_run:
        print(json.dumps({"run_dir": str(run_dir), "total_steps": cfg.total_steps,
                          "carrier": args.carrier, "stage1": stage1_path,
                          "theta_gen_dim": carrier_rec["theta_gen_dim"],
                          "lora": lora_facts.get("n_lora_params"),
                          "assertions": assertions}, indent=2, default=str))
        return 0

    # -- 8. evaluation fixtures ------------------------------------------------
    store = A.SampleStore(args.eval_split)
    fields = R.record_fields(eval_rows)
    pools = bucket_pools(iter_records([r for r in train_index if r.has_record]))
    lib_ids = R._library_ids(train_index, cfg.lib_size, cfg.seed)
    lib = C.LibraryValues.build(bank, lib_ids, model.query_grid9)
    lib_vol, _ = A.library_mean_volume(bank, lib_ids, cfg.bake_grid, device=device)
    eval_samples = R.load_eval_samples(eval_rows, store=store, caches=eval_caches,
                                       fields=fields, device="cpu",
                                       limit=args.eval_n or None, arm=A)
    select_pool = eval_samples[: max(int(args.live_z_select), int(args.select_samples))]
    select_samples = select_pool[: args.select_samples]
    # pre-registered column with a run-time assertion: IP-A needs a pair inside
    # the pool the selection eval draws from, or the first board is void.
    if not A.same_source_pairs(select_pool, limit=1, seed=cfg.seed):
        raise SystemExit(
            f"--live-z-select {args.live_z_select} yields no same-source pair with "
            "different lut_ids, so the pre-registered IP-A column could not be "
            "computed at the selection evals.  Raise it.")

    # -- 9. train --------------------------------------------------------------
    best = {"headline": float("inf"), "step": -1}
    t_train = time.time()
    step_wall: list[float] = []
    train_peak: dict[str, float | None] = {}
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()
    if not args.eval_only:
        rng = random.Random(cfg.seed)
        order: list[IndexRow] = []
        first_quick = True
        first_board = True
        cache = train_caches["none"]
        with steps_path.open("w", encoding="utf-8") as steps_fh:
            for step in range(cfg.total_steps):
                t_step = time.time()
                if len(order) < cfg.batch_samples:
                    order = list(trainable)
                    rng.shuffle(order)
                batch = [order.pop() for _ in range(cfg.batch_samples)]
                sids = [r.sample_id for r in batch]
                if args.lora:
                    rows = [cache_row(cache, s) for s in sids]
                    z = live_train.z(rows, tag="none", grad=True)
                else:
                    z = cache.batch(sids, device=device)
                row = A.train_step(model, cfg, opt, sched, step=step, z=z,
                                   lut_ids=[r.lut_id for r in batch], bank=bank,
                                   sampler=sampler, images=None, alphas=None)
                row["arm"] = ARM
                row["lora"] = bool(args.lora)
                if args.lora:
                    row["lr_lora"] = float(opt.param_groups[2]["lr"])
                    # train_step clips in place, so what is on ``.grad`` now is
                    # post-clip; ``gnorm`` is the pre-clip total, which gives the
                    # scale back exactly (clip_grad_norm_ multiplies every grad
                    # by min(1, max_norm / total)).  Both are recorded rather
                    # than one being silently relabelled as the other.
                    post = L.lora_grad_norm(lora_params)
                    scale = (max(1.0, float(row["gnorm"]) / cfg.max_grad_norm)
                             if cfg.max_grad_norm > 0 else 1.0)
                    row["lora_grad_norm"] = post * scale
                    row["lora_grad_norm_clipped"] = post
                    if step == 0 or (step + 1) % max(1, args.diag_every) == 0:
                        with torch.no_grad(), L.adapter_disabled(vlm):
                            z_frozen = live_train.z(rows[:8], tag="none", grad=False)
                        row["z_drift_cos"] = L.z_cos_drift(z[:8], z_frozen)
                        row["z_drift_l2"] = float(
                            (z[:8].detach() - z_frozen).norm(dim=-1).mean())
                # the L_rec guard: a non-finite reconstruction loss is the one
                # thing that makes every later column meaningless
                if not math.isfinite(float(row["L_rec"])):
                    raise SystemExit(
                        f"step {step}: L_rec is {row['L_rec']}; the run is stopped "
                        "here rather than writing a board on a diverged model")
                step_wall.append(time.time() - t_step)
                if step == 0 or (step + 1) % args.log_every == 0:
                    steps_fh.write(json.dumps(row) + "\n")
                    steps_fh.flush()
                if step == 0:
                    print(json.dumps(row), flush=True)

                due = (step + 1) % eval_every == 0 or step + 1 == cfg.total_steps
                if due and torch.cuda.is_available() and not train_peak:
                    # the number a queue submission needs is the TRAINING peak,
                    # measured before the first eval adds its own allocations
                    train_peak = {
                        "train_peak_memory_gib":
                            round(torch.cuda.max_memory_allocated() / 2 ** 30, 2),
                        "train_peak_memory_reserved_gib":
                            round(torch.cuda.max_memory_reserved() / 2 ** 30, 2)}
                if not due:
                    continue
                qids = [r.sample_id for r in trainable[: args.quick_samples]]
                if args.lora:
                    qz = live_train.z([cache_row(cache, s) for s in qids],
                                      tag="none", grad=False).to(device)
                else:
                    qz = cache.batch(qids, device=device)
                qrow = A.quick_eval(model, cfg, z=qz,
                                    lut_ids=[r.lut_id for r in trainable[: args.quick_samples]],
                                    bank=bank, step=step + 1, first=first_quick)
                if not math.isfinite(float(qrow["l1_grid"])):
                    raise SystemExit(f"step {step + 1}: quick-eval l1_grid is not finite")
                first_quick = False
                if args.lora:
                    refresh_live_z(select_pool, live_eval, eval_caches,
                                   tags=("none",), micro=args.vlm_micro_batch)
                if first_board:
                    board = R.evaluate_and_publish(
                        model, cfg, select_samples, bank=bank, lib=lib,
                        lib_mean_volume=lib_vol, pools=pools, split=args.eval_split,
                        steps_path=steps_path, published=False,
                        point_chunk=args.point_chunk, interp_samples=select_pool,
                        interp_limit=args.select_interp_pairs, arm=A)
                    first_board = False
                else:
                    board = R.selection_board(model, cfg, select_samples, bank=bank,
                                              split=args.eval_split, arm=A)
                h = R.selection_headline(board)
                print(json.dumps({"step": step + 1, "quick": qrow, "headline": h}),
                      flush=True)
                if h < best["headline"]:
                    best = {"headline": h, "step": step + 1}
                    ckpt: dict[str, Any] = {"model": model.state_dict(),
                                            "step": step + 1, "headline": h,
                                            "config": cfg.as_dict()}
                    if args.lora:
                        ckpt["lora"] = {k: v.detach().cpu()
                                        for k, v in vlm.state_dict().items()
                                        if "lora_" in k}
                    torch.save(ckpt, run_dir / "best.pt")
        if (run_dir / "best.pt").is_file():
            saved = torch.load(run_dir / "best.pt", map_location=device,
                               weights_only=False)
            model.load_state_dict(saved["model"])
            if args.lora and saved.get("lora"):
                miss, unexp = vlm.load_state_dict(saved["lora"], strict=False)
                if unexp:
                    raise SystemExit(
                        f"the saved adapter does not fit the attached one: "
                        f"unexpected {list(unexp)[:4]}")

    # -- 10. the z the published board is scored on ----------------------------
    # Before the degeneracy guard, not after: --eval-only reaches the guard
    # without the training loop ever having refreshed anything, and a guard run
    # on stale cached z would be answering a question about a model that is not
    # the one being published.
    live_counts: dict[str, int] = {}
    if args.lora:
        t_z = time.time()
        live_counts = refresh_live_z(eval_samples, live_eval, eval_caches,
                                     tags=eval_tags, micro=args.vlm_micro_batch)
        live_counts["seconds"] = round(time.time() - t_z, 1)

    # -- 10b. the degeneracy guard must have RUN before anything is published ---
    if degeneracy_check_ran() is None:
        A.quick_eval(model, cfg,
                     z=torch.stack([s.z for s in select_samples[: args.quick_samples]],
                                   dim=0).to(device),
                     lut_ids=[s.lut_id for s in select_samples[: args.quick_samples]],
                     bank=bank, step=-1, first=True)

    # -- 11. the published board -----------------------------------------------
    wall = {
        "train_seconds": round(time.time() - t_train, 1),
        "s_per_step_mean": round(float(np.mean(step_wall)), 4) if step_wall else None,
        "s_per_step_p50": round(float(np.median(step_wall)), 4) if step_wall else None,
        "n_steps_timed": len(step_wall),
        "vlm_micro_batch": int(args.vlm_micro_batch),
        **train_peak,
        "peak_memory_gib": (round(torch.cuda.max_memory_allocated() / 2 ** 30, 2)
                            if torch.cuda.is_available() else None),
        "peak_memory_reserved_gib": (round(torch.cuda.max_memory_reserved() / 2 ** 30, 2)
                                     if torch.cuda.is_available() else None),
    }
    board = R.evaluate_and_publish(
        model, cfg, eval_samples, bank=bank, lib=lib, lib_mean_volume=lib_vol,
        pools=pools, split=args.eval_split, steps_path=steps_path,
        published=True, eval_only=args.eval_only, arm=A,
        facts={"arm": ARM, "arm_name": ARM_NAME, "lora": lora_facts,
               "carrier_mode": carrier_rec,
               "stage1": stage1_facts, "startup_assertions": assertions,
               "selection": best, "live_z": live_counts, "wall": wall,
               "library": {"n_lut": len(lib_ids), "bake_grid": cfg.bake_grid},
               "wall_clock": time.time()},
        point_chunk=args.point_chunk)
    (run_dir / "metrics.json").write_text(json.dumps(board, indent=2, default=str),
                                          encoding="utf-8")
    print(json.dumps({"arm": ARM, "lora": bool(args.lora), "carrier": args.carrier,
                      "headline_normal_only": R.selection_headline(board),
                      "n": board["n_normal"], "best": best, "wall": wall},
                     indent=2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
