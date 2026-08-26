#!/usr/bin/env python
"""Build the ``z`` cache the six whatb arms consume (HANDOFF 步骤 0-7).

    z = norm(hidden_states[-1]) at <seg_color>, (2560,) fp32

Two jobs in one entry point, because they differ only in where the reply span
comes from:

``--tag none``  (the main cache: train + V_what)
    **No generation.**  The where-side job already produced, for every sample of
    every split, the base's own greedy continuation and split it into
    ``where_ids`` / ``color_ids``::

        /home/bc/data/runs/where_b/genwhere_v2seg/{train,V_where}/
        producer q3vl.whereb.gencontext/2, checkpoint field =
        q3vl_base_sft_v2seg_20260814/checkpoint-4976 = this campaign's base

    Those are the exact token ids the ``generated`` context is defined as, so
    this job re-uses them and runs **one frozen forward per sample** to read the
    hidden state at ``<seg_color>``.  Re-generating 93,934 reasonings would cost
    ~8.7 h at the measured 3 samples/s; the forward pass is ~1 h.
    ``--context teacher`` takes the reply from the record's own ``where`` /
    ``color`` text instead, same single forward.

``--tag shuffle|irrelevant|const``  (N1/N2/N3, V_what only)
    Frozen block ⑥: a control **must re-generate the reasoning** under the
    perturbed instruction -- reusing the true reasoning would leave the true
    instruction's semantics sitting at the ``<seg_color>`` position and void the
    control.  So these three do generate, but only over V_what (897 x 3 ≈ 2,700
    samples).  ``context_source`` is therefore always ``generated`` for them.

Perturbations (EPR-024 §3.7-D, ``PROPOSAL.md:671-673``):

    shuffle     another sample's instruction, same ``task_type``, different
                ``lut_id``, same split (deterministic, seeded)
    irrelevant  an equal-length non-colour English sentence.  NOTES 9 of the
                proposal left the corpus open; the conservative default
                implemented here is ``--irrelevant-source where_span`` = another
                sample's ``<where>`` span text (pure spatial description, no
                colour words), and the choice is written into the cache meta and
                the report.  ``--irrelevant-source file`` reads one sentence per
                line instead.
    const       the fixed phrase ``Please edit this photo.``

Output (one directory per ``(split, tag)``, the layout ``q3vl.whatb.zcache``
reads)::

    <out-root>/<split>__<tag>/{z.npy,index.jsonl,meta.json}

``context_source`` is **not** in the leaf name, because the arms address a cache
by ``(split, tag)`` and assert the context.  Give the two contexts two roots::

    <cache-root>/generated/<split>__<tag>      --zcache-root .../generated
    <cache-root>/teacher/<split>__<tag>        --zcache-root .../teacher

so a run started with ``--context teacher`` against the generated root dies on
the start-up assertion instead of quietly training on the other condition.

``--readout-multi kind1,kind2,...``  (one forward, several readout positions)
    EPR-024 §4.3's readout-position ablation needs the same samples read out at
    ``color_close`` / ``im_end`` / ``color_span_pool`` / ``seg_where`` /
    ``seg_color``.  The five reply spans are **prefixes of one another** (see
    :data:`SUPERSET_KIND`) and the base is causal, so one forward over the
    longest span (``im_end``) carries every one of the five positions.  Each
    kind is written to its own root, because the leaf name ``<split>__<tag>``
    carries no kind::

        <out-root>/<kind>/<split>__<tag>/{z.npy,index.jsonl,meta.json}

    Two runtime assertions, not two comments: the prefix property is checked per
    sample per kind (:func:`assert_prefix_plan`), and ``--causality-gate N``
    re-runs the gate kind through its OWN forward on the first ``N`` samples and
    kills the job before anything is written if the two z disagree.

    **What "disagree" has to mean here.**  Measured on this base (V_what, eager,
    4 samples, ``<seg_color>`` row, ``max|z|`` ~ 47-62):

        dtype      repeat   batch 4 vs 1   superset vs own forward
        float32    0.0      1.2e-4..7.5e-4   0.0  (exactly, 4/4)
        bfloat16   0.0      1.25 .. 2.5      1.1 .. 4.25

    The base is prefix-invariant -- in fp32 the superset forward reproduces the
    per-kind forward **exactly**.  In bf16 the same z moves by O(1) when only the
    padding shape changes, so a fixed ``1e-3`` is a floor no bf16 run can meet,
    and every bf16 cache already carries that much batch-shape noise against any
    other bf16 build of itself.  The gate therefore accepts
    ``max|dz| <= max(--causality-gate-tol, --causality-gate-factor x control)``
    where the control is the same superset tokens re-encoded at batch 1, and
    writes all three numbers (delta, control, ``max|z|``) into the setup, the
    cache ``meta.json`` and the report.  A wiring error (reading the wrong row)
    is O(|z|) and still dies here; an fp32 run still has to come out at ~0.

    With ``--readout-multi`` empty the job is the single-``--readout`` one it
    always was, down to the forward it runs.

Local disk only -- these are GB-scale and ``/mnt/nfs`` is the hard mount.

**Run this file by absolute path, never ``python -m``.**  ``q3vl/whatb/__init__``
imports torch eagerly, so ``-m q3vl.whatb.scripts.build_zcache`` loads torch
while walking the package chain -- before this module's own body runs -- and the
sqlite3-before-torch guard below is then already too late (campaign bug R6; the
where side documents the same trap at ``q3vl/whereb/__init__.py:14-20``).
Measured: ``-m`` dies with ``CXXABI_1.3.15 not found``; by path it runs.

DO NOT SUBMIT THIS BLIND -- a producer is already running (2026-08-15 15:10)
---------------------------------------------------------------------------
``q status`` shows ``ZCACHE_A`` (gpu0) and ``ZCACHE_B`` (gpu1), both ~1 h in,
running ``/home/bc/data/runs/whatb/zcache_v2seg/_src/build_z.py`` over the six
stages (train teacher/generated + V_what teacher/generated + the three
controls).  That producer writes the ``.pt`` encoding
(``<split>.<context>.<tag>.zcache.pt``), which :class:`q3vl.whatb.zcache.ZCache`
reads directly -- so **nothing needs rebuilding**; point the arms at
``--z-cache /home/bc/data/runs/whatb/zcache_v2seg``.

This entry point stays for the cases that producer does not cover (another
split, another ``--readout``, a re-run of one control after the NOTES 9 ruling)
and because it carries the reuse optimisation below.  The commands are a draft
for those cases, not something to run alongside the job in flight.

Queue submission (draft; D-20: ``rm -f`` the log first, never ``pgrep``)::

    ROOT=/home/bc/data/caches/whatb_z_20260815
    LOG=/home/bc/data/runs/whatb_zcache/logs
    PY=/home/bc/envs/q3vl_sft/bin/python
    B=/home/bc/VeraRetouch/q3vl/whatb/scripts/build_zcache.py

    # (1) gpu0 -- train, normal-only 93,934, forward only (re-uses the where-side
    #     spans).  ~1 h estimated; 962 MB of z.
    rm -f $LOG/train_none_generated.log
    q submit ZCACHE_TRAIN_GEN 0 $LOG/train_none_generated.log --mem-peak 24 \\
      --desc "whatb z: train normal-only 93934 / tag none / generated (forward only)" \\
      -- env PYTHONPATH=/home/bc/VeraRetouch $PY $B --split train --tag none \\
         --context generated --rows normal --out-root $ROOT/generated \\
         --device cuda:0 --batch-size 16

    # (2..6) gpu1 -- V_what 897 x {none generated, none teacher, N1, N2, N3}.
    #        The four generated ones re-generate the reasoning (V_what is not in
    #        the where-side store); ~9 MB each.
    rm -f $LOG/vwhat_none_generated.log
    q submit ZCACHE_VWHAT_GEN 1 $LOG/vwhat_none_generated.log --mem-peak 30 \\
      --desc "whatb z: V_what 897 / tag none / generated" \\
      -- env PYTHONPATH=/home/bc/VeraRetouch $PY $B --split V_what --tag none \\
         --context generated --out-root $ROOT/generated --device cuda:0 \\
         --batch-size 16 --gen-batch-size 32

    rm -f $LOG/vwhat_none_teacher.log
    q submit ZCACHE_VWHAT_TEACHER 1 $LOG/vwhat_none_teacher.log --mem-peak 24 \\
      --desc "whatb z: V_what 897 / tag none / teacher" \\
      -- env PYTHONPATH=/home/bc/VeraRetouch $PY $B --split V_what --tag none \\
         --context teacher --out-root $ROOT/teacher --device cuda:0 --batch-size 16

    for TAG in shuffle irrelevant const; do
      rm -f $LOG/vwhat_$TAG.log
      q submit ZCACHE_VWHAT_${TAG:u} 1 $LOG/vwhat_$TAG.log --mem-peak 30 \\
        --desc "whatb z: V_what 897 / control $TAG (reasoning re-generated)" \\
        -- env PYTHONPATH=/home/bc/VeraRetouch $PY $B --split V_what --tag $TAG \\
           --out-root $ROOT/generated --device cuda:0 --batch-size 16 \\
           --gen-batch-size 32
    done

``--mem-peak`` is declared, not measured: 24 GB for a pure forward (8 GB of bf16
weights + a batch of 16 sequences of ~1.5-2k tokens under FA2) and 30 GB for the
jobs that also generate (the KV cache of 512 new tokens x 32 sequences).  Both
sit under the 65 GB co-existence line, so two of them share a card.
"""

from __future__ import annotations

# The campaign-wide R6 guard: sqlite3 must be imported BEFORE torch, and this is
# an entry point (see make_generated_context.py's module docstring for the
# libstdc++ CXXABI trace).  The published-store reads below need it.
import sqlite3  # noqa: F401  (import order is the point)

import argparse
import json
import random
import sys
import time
from pathlib import Path
from typing import Any, Iterator, Sequence

import numpy as np
import torch

_REPO = Path(__file__).resolve().parents[3]
if (_REPO / "q3vl").is_dir() and str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

from q3vl.whatb import caliber as K                           # noqa: E402
from q3vl.whatb import splits as S                            # noqa: E402
from q3vl.whatb.readout import (                              # noqa: E402
    WHATB_READOUT_KINDS,
    ReplyPlan,
    WhatReadoutBuilder,
    WhatReadoutSpec,
    readout_vector,
)
from q3vl.whatb.zcache import (                               # noqa: E402
    CONTROL_TAGS,
    Z_DIM,
    write_z_cache,
)

__all__ = ["build_parser", "plan_rows", "shuffle_partner", "control_instruction",
           "parse_readout_multi", "assert_prefix_plan", "readout_index_field",
           "resolve_outputs", "prompt_ids", "causality_gate_verdict",
           "SUPERSET_KIND", "MULTI_KINDS", "main"]

BASE_CHECKPOINT = "/home/bc/data/runs/q3vl_base_sft_v2seg_20260814/checkpoint-4976"
MODEL_DIR = "/home/bc/data/models/Qwen3-VL-4B-Instruct"
GENCTX_ROOT = "/home/bc/data/runs/where_b/genwhere_v2seg"
DEFAULT_OUT = "/home/bc/data/caches/whatb_z_20260815"
CONST_PHRASE = "Please edit this photo."
DEFAULT_SEED = 20260810

#: ``--readout-multi``: the readout kind whose reply span contains every other
#: kind's span as a **prefix**, so one forward serves all of them.
#:
#:     color_span_pool  w + c
#:     color_close      w + c
#:     seg_where        w + c + [<seg_where>]
#:     seg_color        w + c + [<seg_where>, <seg_color>]
#:     im_end           w + c + [<seg_where>, <seg_color>, <im_end>]
#:
#: (``q3vl/whatb/readout.py:165-189`` for the first two whatb builds itself,
#: ``q3vl/whereb/readout.py:319-352`` for the three it delegates.)  The base is
#: causal, so the hidden state at position i of the superset forward is the
#: hidden state at position i of every prefix forward -- which is what the
#: ``--causality-gate`` measures at run time instead of assuming.
SUPERSET_KIND = "im_end"

#: the kinds ``--readout-multi`` may carry.  ``qtok`` is excluded: its plan
#: **appends** K query rows the VLM wrapper adds after the reply span
#: (``whereb/readout.py:354-362``, ``n_appended``), so it is not a prefix of the
#: superset and its z is ``(K, 2560)``, not ``(2560,)`` (EPR-024 NOTES-10).
MULTI_KINDS: tuple[str, ...] = tuple(k for k in WHATB_READOUT_KINDS if k != "qtok")



# --------------------------------------------------------------------------- #
# planning (no torch, unit-testable)
# --------------------------------------------------------------------------- #
def plan_rows(split: str, *, rows_filter: str = "all",
              limit: int | None = None) -> list[S.IndexRow]:
    """The frozen row set of one split.  ``normal`` = ``winner_confidence`` normal."""
    rows = S.load_index(split)
    if rows_filter == "normal":
        rows = S.normal_only(rows)
    elif rows_filter != "all":
        raise ValueError(f"--rows must be 'all' or 'normal', got {rows_filter!r}")
    if limit:
        rows = rows[:limit]
    return rows


def shuffle_partner(rows: Sequence[S.IndexRow], *, seed: int = DEFAULT_SEED
                    ) -> dict[str, str]:
    """N1: ``sample_id -> partner sample_id``, same task_type, different lut_id.

    Deterministic given ``(rows, seed)``.  A sample whose task_type pool has no
    other lut_id gets no partner and is **counted and skipped**, never silently
    paired with itself (that would make N1 a copy of the true condition).
    """
    by_type: dict[str, list[S.IndexRow]] = {}
    for r in rows:
        by_type.setdefault(r.task_type, []).append(r)
    rng = random.Random(seed)
    out: dict[str, str] = {}
    for r in rows:
        pool = [p for p in by_type[r.task_type] if p.lut_id != r.lut_id]
        if not pool:
            continue
        out[r.sample_id] = pool[rng.randrange(len(pool))].sample_id
    return out


def control_instruction(tag: str, row: S.IndexRow, *, instructions: dict[str, str],
                        partners: dict[str, str], fillers: Sequence[str],
                        rng: random.Random) -> str | None:
    """The perturbed instruction for one control, or ``None`` to skip the row."""
    if tag == "const":
        return CONST_PHRASE
    if tag == "shuffle":
        partner = partners.get(row.sample_id)
        return None if partner is None else instructions.get(partner)
    if tag == "irrelevant":
        if not fillers:
            return None
        want = len(instructions.get(row.sample_id, "").split())
        # equal-length (in words) stand-in, deterministic: the closest filler,
        # ties broken by the seeded draw rather than by file order
        best = min(fillers, key=lambda s: (abs(len(s.split()) - want),
                                           rng.random()))
        return best
    raise ValueError(f"unknown control tag {tag!r}")


def parse_readout_multi(raw: str | None) -> list[str]:
    """``--readout-multi`` -> the ordered, de-duplicated kind list ([] = off)."""
    kinds: list[str] = []
    for part in str(raw or "").split(","):
        k = part.strip()
        if not k:
            continue
        if k not in WHATB_READOUT_KINDS:
            raise SystemExit(
                f"--readout-multi {k!r}: unknown readout kind; expected one of "
                f"{MULTI_KINDS}")
        if k not in MULTI_KINDS:
            raise SystemExit(
                f"--readout-multi {k!r}: this kind is not exportable from the "
                "shared forward.  Its plan appends query rows after the reply "
                "span, so it is not a prefix of the superset and its z is "
                "(K, 2560); run it as its own --readout job.")
        if k not in kinds:
            kinds.append(k)
    return kinds


def assert_prefix_plan(super_ids: Sequence[int], plan: ReplyPlan, *, kind: str,
                       sample_id: str) -> None:
    """Runtime proof that ``plan`` reads a **prefix** of the superset forward.

    Wired, not assumed: this is the whole licence for reading five kinds out of
    one forward, so it runs on every plan of every sample rather than on a
    sample of them (the campaign's "defined but not wired" failure mode).
    """
    ids = [int(t) for t in plan.token_ids]
    head = [int(t) for t in super_ids[:len(ids)]]
    if plan.n_appended:
        raise AssertionError(
            f"{sample_id}: readout {kind!r} appends {plan.n_appended} rows after "
            "the reply span, so it is not a prefix of the superset forward")
    if head != ids:
        n = next((i for i, (a, b) in enumerate(zip(head, ids)) if a != b),
                 min(len(head), len(ids)))
        raise AssertionError(
            f"{sample_id}: readout {kind!r} is not a prefix of the "
            f"{SUPERSET_KIND!r} superset span (first difference at token {n}: "
            f"superset has {head[n:n + 1]}, plan has {ids[n:n + 1]}; lengths "
            f"{len(super_ids)} vs {len(ids)})")
    if plan.end > len(super_ids):
        raise AssertionError(
            f"{sample_id}: readout {kind!r} slice {plan.start}:{plan.end} runs "
            f"past the superset span ({len(super_ids)} tokens)")


def readout_index_field(plan: ReplyPlan) -> int | list[int]:
    """The schema's ``readout_index``: ``int``, or ``[start, end]`` when pooled.

    ``zcache.py:21`` spells the field "int (or [start, end] for a pooled kind)"
    and ``ZCache._plan_of`` (``zcache.py:453-472``) rebuilds a pooled plan only
    from the two-element form; writing the bare ``start`` for
    ``color_span_pool`` would have the start-up replay verify a one-row slice of
    a span that was mean-pooled over ``end - start`` rows.
    """
    if plan.pool:
        return [int(plan.start), int(plan.end)]
    return int(plan.start)


def causality_gate_verdict(*, delta: float, control: float, tol: float,
                           factor: float) -> tuple[float, bool]:
    """``(threshold, passed)`` for gate G1 -- see the module docstring's table.

    ``control`` is the same token sequence re-encoded at batch 1: the part of
    ``delta`` that is the arithmetic's own reproducibility floor rather than the
    prefix.  In fp32 the control is ~1e-4 and the prefix delta is exactly 0, so
    ``tol`` decides; in bf16 the control is O(1) and it decides.
    """
    thr = max(float(tol), float(factor) * float(control))
    return thr, float(delta) <= thr


def resolve_outputs(*, out_root: str | Path, split: str, tag: str, readout: str,
                    readout_multi: str | None
                    ) -> tuple[list[str], bool, dict[str, Path], str]:
    """``(kinds, multi, out_dirs, superset_kind)`` for one invocation.

    With ``--readout-multi`` empty this is the single-kind job it always was:
    one kind, the historical ``<out-root>/<split>__<tag>`` leaf, and the forward
    driven by that kind's own span (``superset_kind`` == the kind).
    """
    multi_kinds = parse_readout_multi(readout_multi)
    multi = bool(multi_kinds)
    kinds = multi_kinds or [readout]
    out_dirs = {
        k: (Path(out_root) / k / f"{split}__{tag}" if multi
            else Path(out_root) / f"{split}__{tag}")
        for k in kinds}
    return kinds, multi, out_dirs, (SUPERSET_KIND if multi else kinds[0])


def prompt_ids(collator, shim, sample, instruction: str | None) -> Sequence[int]:
    """The prompt half of one sample's encoding, **under ``instruction``**.

    Both phases go through here, and the instruction is a required argument, so
    the forward cannot quietly fall back to the sample's own instruction.  It
    did: the control caches' phase-2 forward was encoded from an un-perturbed
    ``_PromptShim(s)``, which puts the TRUE instruction back into the prompt the
    readout position attends to and voids N1/N2/N3 (frozen block ⑥).
    """
    enc = collator.encode_one(shim(sample, instruction=instruction))
    return enc["input_ids"][:enc["n_prompt_tokens"]]


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #
def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(
        prog="build_zcache",
        description="build the whatb z cache (one frozen forward per sample)")
    # the index口径, spelled / defaulted exactly as on the eight arm runners:
    # which rows this cache covers is a property of the口径, so it is selected
    # here and written onto the artefact rather than inherited from a default.
    K.add_caliber_arguments(ap, group="index口径 (shared with the arm runners)",
                            data=False, batch_split=False, base_lr=False)
    ap.add_argument("--split", required=True, choices=S.SPLITS)
    ap.add_argument("--tag", default="none", choices=CONTROL_TAGS)
    ap.add_argument("--context", default="generated",
                    choices=("teacher", "generated"),
                    help="only read for --tag none; the three controls are "
                         "always 'generated' (frozen block 6)")
    ap.add_argument("--rows", default="all", choices=("all", "normal"),
                    help="'normal' = winner_confidence normal only (train: 93934)")
    ap.add_argument("--readout", default="seg_color")
    ap.add_argument("--readout-qtok", type=int, default=0)
    ap.add_argument("--readout-multi", default="",
                    help="comma-separated readout kinds exported from ONE "
                         f"forward (the {SUPERSET_KIND!r} span; every other "
                         "kind's span is a prefix of it, asserted per sample).  "
                         "Each kind gets its own root: <out-root>/<kind>/"
                         "<split>__<tag>.  Empty = the single --readout job, "
                         f"unchanged.  Allowed: {','.join(MULTI_KINDS)}")
    ap.add_argument("--causality-gate", type=int, default=32,
                    help="--readout-multi only: on the first N samples, compare "
                         "the kind read out of the superset forward against the "
                         "same kind read out of its OWN forward, and die if they "
                         "differ.  Not a post-hoc check -- it runs before "
                         "anything is written.")
    ap.add_argument("--causality-gate-tol", type=float, default=1e-3,
                    help="max|dz| the causality gate accepts outright (default "
                         "1e-3).  Above it the gate compares against the "
                         "batch-shape control measured on the same samples, see "
                         "--causality-gate-factor.")
    ap.add_argument("--causality-gate-factor", type=float, default=4.0,
                    help="the gate also accepts max|dz| <= factor * the "
                         "batch-shape control (the SAME token sequence encoded "
                         "at batch 1 instead of --batch-size).  In bf16 that "
                         "control is O(1) on a |z| of O(50) -- see the module "
                         "docstring -- so a fixed 1e-3 is a floor no bf16 run "
                         "can meet, while a wiring error (wrong row) is O(|z|) "
                         "and still dies here.")
    ap.add_argument("--causality-gate-kind", default="seg_color",
                    help="which kind the gate compares (default seg_color; "
                         "falls back to the first --readout-multi kind)")
    ap.add_argument("--checkpoint", default=BASE_CHECKPOINT)
    ap.add_argument("--model-dir", default=MODEL_DIR)
    ap.add_argument("--genctx-root", default=GENCTX_ROOT,
                    help="where-side generated-context store; --tag none "
                         "--context generated re-uses its token ids instead of "
                         "generating again")
    ap.add_argument("--genctx", default="auto", choices=("auto", "reuse", "generate"),
                    help="auto = re-use the where-side spans when that split is "
                         "in the store (train, V_where), generate otherwise")
    ap.add_argument("--out-root", default=DEFAULT_OUT)
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--dtype", default="bfloat16")
    ap.add_argument("--attn", default="flash_attention_2")
    ap.add_argument("--batch-size", type=int, default=8)
    ap.add_argument("--gen-batch-size", type=int, default=32,
                    help="batch for the control re-generation phase")
    ap.add_argument("--max-new-tokens", type=int, default=512)
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--seed", type=int, default=DEFAULT_SEED)
    ap.add_argument("--irrelevant-source", default="where_span",
                    choices=("where_span", "file"),
                    help="N2 corpus.  Proposal NOTES 9 is open; the default is "
                         "another sample's <where> span (spatial, colour-free) "
                         "and the choice is recorded in the cache meta")
    ap.add_argument("--irrelevant-file", default=None)
    ap.add_argument("--report", default=None)
    ap.add_argument("--plan-only", action="store_true",
                    help="resolve the row set and the perturbations, write the "
                         "report, touch no GPU")
    return ap


# --------------------------------------------------------------------------- #
# the job
# --------------------------------------------------------------------------- #
def _records(rows: Sequence[S.IndexRow]) -> dict[str, dict[str, Any]]:
    return {r.sample_id: rec for r, rec in zip(rows, S.iter_records(rows))}


def _genctx_records(root: Path, split: str, checkpoint: str,
                    keep: set[str] | None = None) -> dict[str, dict[str, Any]]:
    """The where-side generated spans, asserted to come from this base.

    Only the three fields the readout needs are kept: the full records carry
    ``generated_text`` and 300-token id lists for 159,215 samples.
    """
    from q3vl.whereb.stores import GenContextStore

    store = GenContextStore(Path(root) / split)
    out: dict[str, dict[str, Any]] = {}
    seen_ckpt: set[str] = set()
    for rec in store.iter_records():
        if "color_ids" not in rec:
            raise KeyError(
                f"{rec.get('sample_id')}: the generated-context record is schema "
                f"{rec.get('schema_version')!r}, which predates the <color> "
                "segment.  The what side needs the two-segment (v2) store.")
        seen_ckpt.add(str(rec.get("checkpoint")))
        sid = str(rec["sample_id"])
        if keep is not None and sid not in keep:
            continue
        out[sid] = {"where_ids": [int(t) for t in rec["where_ids"]],
                    "color_ids": [int(t) for t in rec["color_ids"]],
                    "n_generated_tokens": int(rec.get("n_generated_tokens", 0))}
    if seen_ckpt != {str(checkpoint)}:
        raise AssertionError(
            f"{root/split}: the generated context was produced by "
            f"{sorted(seen_ckpt)}, this cache's base is {checkpoint!r}.  Reading "
            "one base's reasoning into another base's condition is exactly the "
            "silent failure the checkpoint field exists to stop.")
    return out


def _batched(xs: Sequence[Any], n: int) -> Iterator[list[Any]]:
    for i in range(0, len(xs), n):
        yield list(xs[i:i + n])


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    # the口径 is resolved BEFORE the first load_index: which rows this cache
    # covers is decided here, so it must be decided before anything is read.
    ver = K.apply_dataset_version(args)
    t_start = time.time()
    tag = args.tag
    context_source = "generated" if tag != "none" else args.context
    # Where the reply span comes from.  "reuse" is the whole efficiency argument
    # of this job (train: 93,934 forwards instead of 93,934 generations); it is
    # only available where the where-side store actually has the split.
    genctx_dir = Path(args.genctx_root) / args.split
    if tag != "none" or context_source == "teacher":
        span_source = "control_regenerate" if tag != "none" else "record_teacher"
    elif args.genctx == "generate":
        span_source = "regenerate"
    elif genctx_dir.is_dir():
        span_source = "reuse_where_side"
    elif args.genctx == "reuse":
        raise SystemExit(
            f"--genctx reuse asked for, but {genctx_dir} does not exist.  The "
            "where-side store only carries the splits that job was run on "
            "(train, V_where); pass --genctx generate for the others.")
    else:
        span_source = "regenerate"
    regenerate = span_source in ("control_regenerate", "regenerate")

    # -- which kinds this job exports, and where each one lands ---------------
    kinds, multi, out_dirs, super_kind = resolve_outputs(
        out_root=args.out_root, split=args.split, tag=tag, readout=args.readout,
        readout_multi=args.readout_multi)
    if multi and int(args.readout_qtok):
        raise SystemExit("--readout-qtok is not read under --readout-multi")
    if multi and int(args.causality_gate) < 1:
        raise SystemExit(
            "--readout-multi without --causality-gate N>=1: the prefix property "
            "is the licence for reading N kinds out of one forward, and a gate "
            "that never runs is the failure mode this campaign has already paid "
            "for three times.")
    # one root per kind: the leaf name is <split>__<tag> (zcache.py:126-141) and
    # carries no kind, so five kinds in one root would be five collisions.
    out_dir = out_dirs[kinds[0]]
    for k, d in out_dirs.items():
        if d.exists():
            raise SystemExit(
                f"{d} already exists.  Move it aside (do NOT delete -- long-job "
                "discipline) before re-running.")
    gate_kind = (args.causality_gate_kind if args.causality_gate_kind in kinds
                 else kinds[0]) if multi else None
    if tag != "none" and args.split == "train":
        raise SystemExit(
            "the three controls are evaluation-only columns (n = V_what 897); "
            "re-generating 93,934 reasonings three times over is 26 GPU-hours "
            "for a column nothing reads.")

    rows = plan_rows(args.split, rows_filter=args.rows, limit=args.limit)
    records = _records(rows)
    instructions = {sid: str(rec.get("instruction", ""))
                    for sid, rec in records.items()}
    partners = shuffle_partner(rows, seed=args.seed) if tag == "shuffle" else {}
    fillers: list[str] = []
    if tag == "irrelevant":
        if args.irrelevant_source == "file":
            if not args.irrelevant_file:
                raise SystemExit("--irrelevant-source file needs --irrelevant-file")
            fillers = [ln.strip() for ln in
                       Path(args.irrelevant_file).read_text(encoding="utf-8").splitlines()
                       if ln.strip()]
        else:
            fillers = sorted({str(rec.get("where", "")).strip()
                              for rec in records.values()
                              if str(rec.get("where", "")).strip()})

    setup: dict[str, Any] = {
        "split": args.split, "tag": tag, "context_source": context_source,
        "rows": args.rows, "n_rows": len(rows), "checkpoint": args.checkpoint,
        # which index口径 the row set was planned in: name + root + that口径's
        # measured per-split n / normal_n + the exclusion list's sha256.  A
        # cache is only usable by a run of the SAME口径, and
        # run_carrier_arm.open_z_caches asserts exactly this field.
        "dataset_version": ver.name,
        "dataset_root": str(ver.root),
        "dataset_version_facts": ver.facts(),
        "n_index_rows": len(S.load_index(args.split)),
        "readout": args.readout, "readout_qtok": int(args.readout_qtok),
        "readout_multi": kinds if multi else None,
        "superset_kind": SUPERSET_KIND if multi else None,
        "out_dir": str(out_dir),
        "out_dirs": {k: str(d) for k, d in out_dirs.items()},
        # filled in by the gate itself while the job runs; a None here on a
        # --readout-multi artefact means the gate did not run.
        "causality_gate": ({"kind": gate_kind, "n_requested": int(args.causality_gate),
                            "tol": float(args.causality_gate_tol),
                            "factor": float(args.causality_gate_factor),
                            "n_compared": 0, "max_abs_delta": None,
                            "max_abs_delta_batch_control": None,
                            "max_abs_z": None, "threshold": None,
                            "passed": None} if multi else None),
        "seed": int(args.seed),
        "span_source": span_source, "regenerate": regenerate,
        "genctx_root": (args.genctx_root if span_source == "reuse_where_side"
                        else None),
        "irrelevant_source": args.irrelevant_source if tag == "irrelevant" else None,
        "n_fillers": len(fillers) or None,
        "n_shuffle_partners": len(partners) or None,
        "batch_size": int(args.batch_size),
        "device": args.device, "dtype": args.dtype, "attn": args.attn,
    }
    print(json.dumps(setup, indent=2, ensure_ascii=False), flush=True)
    if args.plan_only:
        if args.report:
            Path(args.report).parent.mkdir(parents=True, exist_ok=True)
            Path(args.report).write_text(
                json.dumps({"setup": setup, "plan_only": True}, indent=2,
                           ensure_ascii=False), encoding="utf-8")
        return 0

    # -- the model ----------------------------------------------------------
    from q3vl.train.collator import Sft2SegCollator
    from q3vl.train.modeling import load_model, load_processor
    from q3vl.whereb.config import (
        COLOR_CONTEXT_MAX_TOKENS as COLOR_MAX,
        WHERE_CONTEXT_MAX_TOKENS as WHERE_MAX,
    )
    from q3vl.whereb.context import encode_where_span, extract_segment
    from q3vl.whereb.data import _PromptShim, open_dataset
    from q3vl.whereb.gencontext import SegmentIds
    from q3vl.whereb.hiddens import EncodeItem, FrozenVLM

    processor, special_ids = load_processor(args.model_dir, 2048)
    model = load_model(args.checkpoint, attn_implementation=args.attn,
                       dtype=args.dtype).to(args.device).eval()
    vlm = FrozenVLM(model, processor, device=args.device)
    collator = Sft2SegCollator(processor, max_length=2048, system_prompt=None)
    tokenizer = collator.tokenizer
    tags = SegmentIds(tokenizer)

    control_color = "own" if tag == "none" else "regenerated"
    # one builder per kind: each keeps its own facts()/counts, so a per-kind
    # flag (missing_tag, null_context) is not pooled across the五档.
    specs = {k: WhatReadoutSpec(kind=k,
                                qtok=0 if multi else int(args.readout_qtok))
             for k in kinds}
    builders = {k: WhatReadoutBuilder(tokenizer, specs[k],
                                      control_color=control_color)
                for k in kinds}
    spec = specs[kinds[0]]
    builder = builders[kinds[0]]
    # the forward is driven by the superset span.  In the single-kind job that
    # IS the kind's own span, so that path stays byte-for-byte what it was.
    super_builder = builders.get(super_kind) or WhatReadoutBuilder(
        tokenizer, WhatReadoutSpec(kind=super_kind), control_color=control_color)
    color_texts = [str(rec.get("color", "")) for rec in records.values()
                   if rec.get("color")][:256]
    colorspan_check = builder.run_colorspan_assertion(color_texts)
    for k, b in builders.items():
        if b is not builder:
            b.run_colorspan_assertion(color_texts)

    dataset, ds_info = open_dataset(args.split, need_mask=False)
    by_id = {r.sample_id: r for r in rows}
    order = [i for i in range(len(dataset)) if dataset.refs[i].sample_id in by_id]

    genctx = (_genctx_records(Path(args.genctx_root), args.split, args.checkpoint,
                              keep=set(by_id))
              if span_source == "reuse_where_side" else {})

    rng = random.Random(args.seed)
    skipped: dict[str, int] = {}
    index_rows: dict[str, list[dict[str, Any]]] = {k: [] for k in kinds}
    vectors: dict[str, list[np.ndarray]] = {k: [] for k in kinds}
    n_done = 0
    t0 = time.time()
    gate_left = int(args.causality_gate) if multi else 0
    gate_n, gate_max, gate_ctl, gate_absz = 0, 0.0, 0.0, 0.0

    for chunk in _batched(order, args.batch_size):
        samples = [dataset[i] for i in chunk]

        # phase 1: re-generate the reasoning.  Controls always do (frozen block
        # 6: the perturbed instruction must drive the reasoning); the ``none``
        # cache only does it for a split the where-side job never covered.
        gen_ids: dict[str, list[int]] = {}
        # the instruction the FORWARD is conditioned on, per sample.  A control's
        # forward must carry the same perturbed instruction its generation did:
        # re-encoding the sample's own instruction here would put the TRUE
        # instruction's semantics back into the prompt the readout position
        # attends to, i.e. void the control (frozen block ⑥).
        prompt_instr: dict[str, str | None] = {}
        if regenerate:
            items, keep_samples = [], []
            for s in samples:
                row = by_id[s.sample_id]
                instr = (None if tag == "none" else control_instruction(
                    tag, row, instructions=instructions, partners=partners,
                    fillers=fillers, rng=rng))
                if tag != "none" and not instr:
                    skipped["no_perturbation"] = skipped.get("no_perturbation", 0) + 1
                    continue
                prompt_instr[s.sample_id] = instr
                items.append(EncodeItem(
                    sample_id=s.sample_id, image=s.image,
                    prompt_ids=prompt_ids(collator, _PromptShim, s, instr)))
                keep_samples.append(s)
            if not items:
                continue
            outs = vlm.generate_where(items, max_new_tokens=args.max_new_tokens,
                                      eos_token_id=tags.eos)
            for s, ids in zip(keep_samples, outs):
                gen_ids[s.sample_id] = [int(t) for t in ids]
            samples = keep_samples

        # phase 2: one frozen forward, read the hidden at the readout index
        items, plans, metas = [], [], []
        for s in samples:
            sid = s.sample_id
            rec = records[sid]
            if context_source == "teacher":
                where_ids = encode_where_span(tokenizer, str(rec.get("where", "")))
                color_ids = builder.color_ids_from_text(str(rec.get("color", "")))
                n_generated = 0
            elif span_source == "reuse_where_side":
                g = genctx.get(sid)
                if g is None:
                    skipped["no_genctx"] = skipped.get("no_genctx", 0) + 1
                    continue
                where_ids = [int(t) for t in g["where_ids"]]
                color_ids = [int(t) for t in g["color_ids"]]
                n_generated = int(g.get("n_generated_tokens", 0))
            else:
                # the where side's own boundaries, not new numbers: a control's
                # span must be carved exactly like the reused ones
                ids = gen_ids[sid]
                w = extract_segment(ids, tags.where_close, WHERE_MAX,
                                    eos_id=tags.eos)
                start = w.end if w.stop_reason == "closed" else 0
                c = extract_segment(ids, tags.color_close, COLOR_MAX,
                                    open_id=tags.color_open, start=start,
                                    eos_id=tags.eos)
                where_ids = list(w.token_ids)
                color_ids = list(c.token_ids)
                n_generated = len(ids)
            if not color_ids:
                skipped["empty_color_span"] = skipped.get("empty_color_span", 0) + 1
                continue
            plans_k = {k: b.plan_for(sample_id=sid, where_ids=where_ids,
                                     color_ids=color_ids, source=context_source,
                                     control_tag=tag)
                       for k, b in builders.items()}
            super_plan = plans_k.get(super_kind) or super_builder.plan_for(
                sample_id=sid, where_ids=where_ids, color_ids=color_ids,
                source=context_source, control_tag=tag)
            if multi:
                for k, p in plans_k.items():
                    assert_prefix_plan(super_plan.token_ids, p, kind=k,
                                       sample_id=sid)
            pids = prompt_ids(collator, _PromptShim, s, prompt_instr.get(sid))
            items.append(EncodeItem(sample_id=sid, image=s.image,
                                    prompt_ids=pids,
                                    where_ids=super_plan.token_ids))
            plans.append(plans_k)
            metas.append({"sample_id": sid, "n_generated_tokens": n_generated,
                          "rec": rec, "image": s.image, "prompt_ids": pids})
        if not items:
            continue

        results = vlm.encode(items)
        for res, plans_k, meta in zip(results, plans, metas):
            rec = meta["rec"]
            for k, plan in plans_k.items():
                h = res.h_where[:len(plan.token_ids)] if multi else res.h_where
                z = readout_vector(h, plan).float().cpu().numpy()
                if z.shape != (Z_DIM,):
                    raise AssertionError(f"z is {z.shape}, expected ({Z_DIM},)")
                vectors[k].append(z.astype(np.float32))
                index_rows[k].append({
                    "sample_id": meta["sample_id"], "split": args.split,
                    "checkpoint": args.checkpoint, "readout_kind": k,
                    "context_source": context_source, "control_tag": tag,
                    "reply_token_ids": [int(t) for t in plan.token_ids],
                    "readout_index": readout_index_field(plan),
                    "expected_ids": [int(t) for t in plan.expected_ids],
                    "n_generated_tokens": int(meta["n_generated_tokens"]),
                    "color_text": str(rec.get("color", "")),
                    "lut_id": str(rec.get("lut_id", "")),
                    "minor": rec.get("minor"),
                })

        # -- gate G1: the superset forward vs the kind's OWN forward ----------
        # Runs on the first N samples of the job, before a single vector is
        # written, and kills the job on violation.  "one forward, five kinds"
        # rests entirely on the base being causal; this measures it, next to the
        # batch-shape control that says how much of the measured delta is the
        # arithmetic's own reproducibility floor rather than the prefix.
        if gate_left > 0:
            take = min(gate_left, len(items))
            g_items = [EncodeItem(sample_id=metas[i]["sample_id"],
                                  image=metas[i]["image"],
                                  prompt_ids=metas[i]["prompt_ids"],
                                  where_ids=plans[i][gate_kind].token_ids)
                       for i in range(take)]
            g_results = vlm.encode(g_items)
            # the control: the SAME superset tokens, encoded alone.  Nothing
            # about the sequence changes, only the batch it is padded in.
            c_results = [vlm.encode([items[i]])[0] for i in range(take)]
            for i, g_res in enumerate(g_results):
                plan = plans[i][gate_kind]
                n = len(plan.token_ids)
                z_super = readout_vector(results[i].h_where[:n], plan).float()
                z_solo = readout_vector(g_res.h_where, plan).float()
                z_ctl = readout_vector(c_results[i].h_where[:n], plan).float()
                gate_max = max(gate_max, float((z_super - z_solo).abs().max()))
                gate_ctl = max(gate_ctl, float((z_super - z_ctl).abs().max()))
                gate_absz = max(gate_absz, float(z_super.abs().max()))
                gate_n += 1
            gate_left -= take
            thr, ok = causality_gate_verdict(
                delta=gate_max, control=gate_ctl,
                tol=args.causality_gate_tol, factor=args.causality_gate_factor)
            setup["causality_gate"].update(
                {"n_compared": gate_n, "max_abs_delta": gate_max,
                 "max_abs_delta_batch_control": gate_ctl,
                 "max_abs_z": gate_absz, "threshold": thr, "passed": ok})
            if not ok:
                raise SystemExit(
                    f"causality gate FAILED after {gate_n} samples: --readout "
                    f"{gate_kind} read out of the {super_kind!r} superset forward "
                    f"differs from its own forward by max|dz| = {gate_max:.3e}, "
                    f"over the threshold {thr:.3e} (= max(tol {args.causality_gate_tol:.3e}, "
                    f"{args.causality_gate_factor} x batch-shape control "
                    f"{gate_ctl:.3e})); max|z| = {gate_absz:.3e}.  The one-forward "
                    "export is not valid here; nothing was written.")
            if gate_left <= 0:
                print(json.dumps({"causality_gate": setup["causality_gate"]}),
                      flush=True)
        n_done += len(results)
        if n_done % 500 < args.batch_size:
            rate = n_done / max(1e-9, time.time() - t0)
            print(json.dumps({"done": n_done, "of": len(order),
                              "samples_per_s": round(rate, 2),
                              "eta_s": round((len(order) - n_done) / max(rate, 1e-9))}),
                  flush=True)

    if multi and gate_n == 0:
        raise SystemExit(
            "--readout-multi ran to the end without the causality gate ever "
            f"comparing a sample (requested N={args.causality_gate}, kind "
            f"{gate_kind!r}).  A gate that is defined but never fires is not a "
            "gate; nothing was written.")

    zs: dict[str, np.ndarray] = {}
    for k in kinds:
        zs[k] = (np.stack(vectors[k]).astype(np.float32) if vectors[k]
                 else np.zeros((0, Z_DIM), dtype=np.float32))
        write_z_cache(out_dirs[k], index_rows[k], zs[k],
                      checkpoint=args.checkpoint,
                      readout_kind=k, context_source=context_source,
                      control_tag=tag, split=args.split,
                      readout_qtok=int(specs[k].qtok),
                      extra_meta={"irrelevant_source": setup["irrelevant_source"],
                                  "rows_filter": args.rows, "seed": int(args.seed),
                                  "genctx_root": setup["genctx_root"],
                                  # the consumer's口径 gate (run_carrier_arm.
                                  # assert_zcache_dataset_version) reads these two
                                  "dataset_version": setup["dataset_version"],
                                  "dataset_root": setup["dataset_root"],
                                  "dataset_version_facts":
                                      setup["dataset_version_facts"],
                                  "readout_multi": kinds if multi else None,
                                  "superset_kind": super_kind if multi else None,
                                  "causality_gate": setup["causality_gate"]})

    z = zs[kinds[0]]
    report = {
        "setup": setup, "dataset": ds_info,
        "n_written": int(z.shape[0]), "n_planned": len(order),
        "n_written_by_kind": {k: int(v.shape[0]) for k, v in zs.items()},
        "skipped": skipped, "colorspan_check": colorspan_check,
        "readout": builder.facts(),
        "readout_by_kind": {k: b.facts() for k, b in builders.items()},
        "superset_readout": super_builder.facts() if multi else None,
        "causality_gate": setup["causality_gate"],
        "vlm": vlm.facts(),
        "special_token_ids": special_ids,
        "seconds": round(time.time() - t_start, 1),
        "samples_per_second": round(z.shape[0] / max(1e-9, time.time() - t0), 3),
        "peak_memory_gib": (round(torch.cuda.max_memory_allocated() / 2 ** 30, 2)
                            if torch.cuda.is_available() else None),
    }
    blob = json.dumps(report, indent=2, ensure_ascii=False, default=str)
    paths = ([Path(args.report)] if args.report
             else [d / "build_report.json" for d in out_dirs.values()])
    for path in paths:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(blob, encoding="utf-8")
    print(json.dumps({k: v for k, v in report.items() if k != "setup"},
                     indent=2, ensure_ascii=False, default=str), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
