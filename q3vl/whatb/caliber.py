"""The EPR-030 run caliber, in one place, for the EPR-025..029 arms.

EPR-030 (``run_epr030_arm.py`` + ``arms/carrier.py``) settled four run-level
settings on 2026-08-16.  None of them is an experimental variable of any arm --
they are the conditions every arm is measured under -- so they live here once
and the five runners call in, instead of each growing its own copy:

1. **pure L1**: ``--loss-level 1`` means ``lambda_hc = lambda_sparse =
   lambda_mono = 0``.  The gating rule is EPR-024's own
   (``arms/carrier.py:347`` ``LAMBDA_HC if loss_level >= 2 else 0.0``,
   ``:351`` ``LAMBDA_SPARSE if loss_level >= 3 else 0.0``), re-exported as
   :func:`effective_lambda_hc` / :func:`effective_lambda_sparse`; the run-time
   check is EPR-030's own :func:`q3vl.whatb.losses_l0.assert_l0_pure`.
2. **the colour batch**: ``--batch-split`` resolves through
   :data:`q3vl.whatb.arms.carrier.BATCH_SPLITS` -- the one table, 17 entries
   including ``256x8192`` (2,097,152 colours/step).  No arm keeps a second one.
3. **the base lr**: :data:`q3vl.whatb.arms.carrier.BASE_LR` (1e-3, GLUT App
   A.1); the flag is spelled ``--base-lr`` on every arm.
4. **the training corpora**: ``--dataset-version {v20260804, cut-p45}`` (which
   sft2seg index口径 the v2seg rows are read from -- see
   :data:`q3vl.whatb.splits.DATASET_VERSIONS`) and ``--data {v2seg, v2seg+l8}``.
   The口径 travels into ``run_setup.json`` via :func:`horizon_record`.  The
   population is
   *measured* by :func:`q3vl.whatb.splits.train_normal_rows` (each source
   counted against its own on-disk declaration) and the z of the union comes
   from :class:`q3vl.whatb.zcache.MultiZCache` through
   :func:`q3vl.whatb.scripts.run_carrier_arm.open_z_caches` -- the same call
   the EPR-030 runs make.  **No merged n is written down anywhere.**

Plus the assertion those four imply: ``steps_per_epoch == ceil(n_train / B)``
(:func:`assert_steps_per_epoch`), which every runner calls before step 0.

Nothing here computes a criterion, touches a loss term, or knows what any arm's
experimental variable is.
"""

from __future__ import annotations

import argparse
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Sequence

from q3vl.whatb import splits as S
from q3vl.whatb import zcache as _zcache
from q3vl.whatb.arms.carrier import (
    BASE_LR,
    BATCH_SPLIT_COLORS,
    BATCH_SPLITS,
    EPOCHS,
    FROZEN_BATCH_SPLITS,
)
from q3vl.whatb.losses_l0 import assert_l0_pure
from q3vl.whatb.queries import DEFAULT_SEED

__all__ = [
    "BASE_LR",
    "BATCH_SPLITS",
    "BATCH_SPLIT_COLORS",
    "DATA_CHOICES",
    "DATASET_VERSION_CHOICES",
    "DATASET_VERSION_EXPLICIT_ATTR",
    "DEFAULT_BATCH_SPLIT",
    "DEFAULT_DATA",
    "DEFAULT_DATASET_VERSION",
    "EPOCHS",
    "EPR030_BATCH_SPLIT",
    "FROZEN_TRAIN_NORMAL_N",
    "FROZEN_BATCH_SPLITS",
    "PURE_L1_LOSS_LEVEL",
    "add_caliber_arguments",
    "apply_dataset_version",
    "dataset_version_given",
    "dataset_version_record",
    "default_train_normal_n",
    "assert_steps_per_epoch",
    "batch_split_choices",
    "effective_lambda_hc",
    "effective_lambda_sparse",
    "extra_zcache_sources",
    "horizon_record",
    "open_train_z",
    "parse_batch_split",
    "pure_l1_record",
    "steps_per_epoch_of",
    "train_normal_rows",
]

#: ``--data`` values, from the one table in :mod:`q3vl.whatb.splits`
DATA_CHOICES: tuple[str, ...] = S.DATA_CHOICES
#: ``--dataset-version`` values -- the index口径 table in :mod:`q3vl.whatb.splits`
DATASET_VERSION_CHOICES: tuple[str, ...] = S.DATASET_VERSION_CHOICES
#: the口径 every arm runs on unless ``--dataset-version`` says otherwise
DEFAULT_DATASET_VERSION = S.DEFAULT_DATASET_VERSION
#: every arm's default stays the frozen sft2seg split alone; ``v2seg+l8`` is asked for
DEFAULT_DATA = "v2seg"
#: the frozen colour batch stays every arm's default
DEFAULT_BATCH_SPLIT = "32x256"
#: the split EPR-030 settled on (2,097,152 colours/step)
EPR030_BATCH_SPLIT = "256x8192"
#: ``--loss-level`` value that means "the single L1 term"
PURE_L1_LOSS_LEVEL = 1
#: the **original**口径's sft2seg normal-only count (``v20260804``).  It is what
#: the published boards were run on and what the runners' "frozen block" records
#: compare against; it is NOT a fallback for the active口径 -- use
#: :func:`default_train_normal_n` for that.
FROZEN_TRAIN_NORMAL_N: int = S.TRAIN_NORMAL_N


def default_train_normal_n() -> int:
    """The active口径's declared sft2seg train normal-only n.

    The *fallback* for a code path that has not read an index yet (``--stage
    setup``, a parser unit test); a real run measures its population -- see
    :func:`train_normal_rows`.
    """
    return S.active_dataset_version().train_normal_n


def dataset_version_record(root: str | Path | None = None) -> dict[str, Any]:
    """The口径 block ``run_setup.json`` carries (name / root / n / exclusion sha)."""
    return S.dataset_version_facts(root)


#: ``args`` attribute :class:`_RecordExplicit` sets when ``--dataset-version``
#: was actually typed on the command line.  argparse cannot tell a default apart
#: from a value that happens to equal it, and the two mean different things here:
#: ``--dataset-root <old root>`` alone is a complete, unambiguous request, while
#: ``--dataset-root <old root> --dataset-version cut-p45`` is a contradiction.
DATASET_VERSION_EXPLICIT_ATTR = "dataset_version_explicit"


class _RecordExplicit(argparse.Action):
    """Store the value and record that the user typed the flag."""

    def __call__(self, parser, namespace, values, option_string=None):
        setattr(namespace, self.dest, values)
        setattr(namespace, DATASET_VERSION_EXPLICIT_ATTR, True)


def dataset_version_given(args) -> bool:
    """True when ``--dataset-version`` was typed (not merely defaulted)."""
    return bool(getattr(args, DATASET_VERSION_EXPLICIT_ATTR, False))


def apply_dataset_version(args) -> Any:
    """Resolve ``--dataset-version`` / ``--dataset-root`` into the process口径.

    Called once, at the top of a runner's ``main`` and **before** any index is
    read, so ``n`` / ``steps_per_epoch`` / ``total_steps`` are all derived from
    the same口径.  A runner that also owns a ``--dataset-root`` gets it filled in
    with the resolved root (an explicitly given root wins and must itself be a
    registered口径).

    ``--dataset-root`` **alone** resolves the口径 from the root: the
    ``--dataset-version`` default is a string, never ``None``, so comparing
    against it would reject a command line that named exactly one of the two.
    Naming **both** and disagreeing is still a refusal -- that is a real
    contradiction and stays loud (:func:`dataset_version_given`).
    """
    root = getattr(args, "dataset_root", None)
    if root:
        ver = S.version_for_root(root)
        named = getattr(args, "dataset_version", None)
        if (dataset_version_given(args) and named is not None
                and str(named) != ver.name):
            raise SystemExit(
                f"--dataset-root {root} is dataset version {ver.name!r} but "
                f"--dataset-version says {named!r}; pass one of the two")
    else:
        ver = S.dataset_version(getattr(args, "dataset_version",
                                        S.DEFAULT_DATASET_VERSION))
    S.use_dataset_version(ver)
    if hasattr(args, "dataset_root"):
        args.dataset_root = str(ver.root)
    if hasattr(args, "dataset_version"):
        args.dataset_version = ver.name
    return ver


# --------------------------------------------------------------------------- #
# 1. the colour batch
# --------------------------------------------------------------------------- #
def batch_split_choices(extra: Iterable[str] = ()) -> tuple[str, ...]:
    """``sorted(BATCH_SPLITS)`` plus an arm's own non-published entries.

    ``extra`` exists for the tiny smoke splits two runners already accept
    (``8x64`` / ``2x16``); they are not in the shared table because they are not
    a caliber anything publishes on.
    """
    return tuple(sorted(BATCH_SPLITS)) + tuple(
        e for e in extra if e not in BATCH_SPLITS)


def parse_batch_split(name: str) -> tuple[int, int]:
    """``"256x8192" -> (256, 8192)`` through the shared table.

    A name in :data:`BATCH_SPLITS` returns that table's pair (so a typo cannot
    silently run a different batch); a well-formed ``BxQ`` outside it is parsed
    and returned, which is the smoke path the runners already had.
    """
    key = str(name).strip().lower()
    if key in BATCH_SPLITS:
        return BATCH_SPLITS[key]
    b, sep, q = key.partition("x")
    if not sep or not b.isdigit() or not q.isdigit() or int(b) < 1 or int(q) < 1:
        raise ValueError(
            f"--batch-split {name!r} is neither one of {sorted(BATCH_SPLITS)} "
            "nor a well-formed BxQ")
    return int(b), int(q)


def colours_per_step(batch_split: str) -> int:
    b, q = parse_batch_split(batch_split)
    return b * q


# --------------------------------------------------------------------------- #
# 2. the horizon
# --------------------------------------------------------------------------- #
def steps_per_epoch_of(n_train: int, batch_samples: int) -> int:
    """``ceil(n / B)`` -- one epoch is one pass over the training population."""
    return int(math.ceil(int(n_train) / int(batch_samples)))


def assert_steps_per_epoch(value: int, *, n_train: int, batch_samples: int,
                           where: str = "") -> int:
    """The run-time assertion.  A pre-registered number without one is a comment."""
    want = steps_per_epoch_of(n_train, batch_samples)
    if int(value) != want:
        raise AssertionError(
            f"{where or 'steps_per_epoch'} is {int(value)} but "
            f"ceil({int(n_train)} / {int(batch_samples)}) = {want}; the epoch "
            "length and the training population disagree, so the horizon is not "
            "the one the data implies")
    return want


# --------------------------------------------------------------------------- #
# 3. pure L1
# --------------------------------------------------------------------------- #
def effective_lambda_hc(lambda_hc: float, loss_level: int) -> float:
    """``arms/carrier.py:347``, verbatim: the ladder gates the weight."""
    return float(lambda_hc) if int(loss_level) >= 2 else 0.0


def effective_lambda_sparse(lambda_sparse: float, loss_level: int) -> float:
    """``arms/carrier.py:351``, verbatim."""
    return float(lambda_sparse) if int(loss_level) >= 3 else 0.0


def pure_l1_record(*, loss_level: int, lambda_hc: float, lambda_sparse: float,
                   lambda_mono: float = 0.0) -> dict[str, Any]:
    """``run_setup.json``'s pure-L1 block, with the assertion actually run.

    At ``--loss-level 1`` the three optional weights must be exactly 0, and
    :func:`q3vl.whatb.losses_l0.assert_l0_pure` (EPR-030's own check) is what
    says so -- before the first step, not in a docstring.
    """
    rec: dict[str, Any] = {
        "loss_level": int(loss_level),
        "pure_l1": int(loss_level) == PURE_L1_LOSS_LEVEL,
        "lambda_hc_effective": float(lambda_hc),
        "lambda_sparse_effective": float(lambda_sparse),
        "lambda_mono_effective": float(lambda_mono),
    }
    if rec["pure_l1"]:
        rec["assert_l0_pure"] = assert_l0_pure(
            lambda_hc=lambda_hc, lambda_sparse=lambda_sparse,
            lambda_mono=lambda_mono, loss="l0")
    return rec


# --------------------------------------------------------------------------- #
# 4. the training corpora
# --------------------------------------------------------------------------- #
def train_normal_rows(data: str, *, split: str = "train",
                      root: str | Path | None = None) -> list[S.IndexRow]:
    """The measured training population of ``--data`` (never a literal n)."""
    return S.train_normal_rows(data, split=split, root=root)


def extra_zcache_sources(data: str, *, zcache_root_l8: str | Path | None = None
                         ) -> tuple[tuple[Any, str], ...]:
    """``(root, split)`` of every non-sft2seg source ``--data`` names."""
    if data not in S.DATA_SOURCES:
        raise ValueError(f"--data must be one of {S.DATA_CHOICES}, got {data!r}")
    out: list[tuple[Any, str]] = []
    for name in S.DATA_SOURCES[data]:
        if name == "l8":
            out.append((zcache_root_l8 or S.L8_ZCACHE_ROOT, S.L8_SPLIT))
        else:                                            # pragma: no cover
            raise ValueError(f"unknown training source {name!r}")
    return tuple(out)


@dataclass(frozen=True)
class _ZShim:
    """The three fields ``run_carrier_arm.open_z_caches`` reads off a config."""

    context: str
    readout: str
    seed: int


def open_train_z(root: str | Path, split: str, *, data: str, checkpoint: str,
                 readout_kind: str = "seg_color", context: str = "generated",
                 zcache_root_l8: str | Path | None = None,
                 seed: int = DEFAULT_SEED) -> tuple[Any, dict[str, Any]]:
    """The training condition of ``--data``: ONE cache, or a :class:`MultiZCache`.

    This is :func:`q3vl.whatb.scripts.run_carrier_arm.open_z_caches` -- the exact
    call the EPR-030 runs make -- driven by a three-field shim instead of a
    ``CarrierConfig``, because the five arms carry five different config types.
    Every member goes through its own ``assert_belongs_to``; an overlapping
    ``sample_id`` is rejected by :class:`~q3vl.whatb.zcache.MultiZCache`.
    """
    from q3vl.whatb.scripts import run_carrier_arm as R

    caches, record = R.open_z_caches(
        root, split, _ZShim(context=context, readout=readout_kind, seed=int(seed)),
        checkpoint=checkpoint, tags=("none",), required=("none",),
        # only ``arm.ZCache`` is looked up, and the shared reader is the one
        # every arm already uses
        arm=_zcache,
        extra_sources=extra_zcache_sources(data, zcache_root_l8=zcache_root_l8))
    return caches["none"], record["none"]


# --------------------------------------------------------------------------- #
# 5. the record every arm writes
# --------------------------------------------------------------------------- #
def horizon_record(*, data: str, n_train: int, batch_split: str,
                   batch_samples: int, queries_per_sample: int,
                   steps_per_epoch: int, total_steps: int, base_lr: float,
                   epochs: int | None = None,
                   zcache_root_l8: str | Path | None = None,
                   loss_level: int | None = None,
                   lambda_hc: float | None = None,
                   lambda_sparse: float | None = None,
                   lambda_mono: float = 0.0) -> dict[str, Any]:
    """The block ``run_setup.json`` carries so the caliber is on the artefact.

    ``steps_per_epoch`` is asserted here, not described: the same
    :func:`assert_steps_per_epoch` every runner calls.
    """
    assert_steps_per_epoch(steps_per_epoch, n_train=n_train,
                           batch_samples=batch_samples, where="steps_per_epoch")
    rec: dict[str, Any] = {
        "data": data,
        "data_sources": ["v2seg", *S.DATA_SOURCES[data]],
        # which sft2seg index口径 the v2seg rows came from.  L8 has no口径: its
        # rows come from l8_train.manifest.jsonl, which no sft2seg filter touches.
        "dataset_version": S.active_dataset_version().facts(),
        "l8_manifest": str(S.L8_MANIFEST),
        "train_normal_n_measured": int(n_train),
        "batch_split": str(batch_split),
        "batch_samples": int(batch_samples),
        "queries_per_sample": int(queries_per_sample),
        "colors_per_step": int(batch_samples) * int(queries_per_sample),
        "colours_per_step": int(batch_samples) * int(queries_per_sample),
        "batch_split_step_matched_to_epr024":
            str(batch_split) in FROZEN_BATCH_SPLITS,
        "steps_per_epoch": int(steps_per_epoch),
        "total_steps": int(total_steps),
        "base_lr": float(base_lr),
        "zcache_root_l8": (str(zcache_root_l8) if zcache_root_l8
                           else str(S.L8_ZCACHE_ROOT)),
        "source": "q3vl/whatb/caliber.py (EPR-030 caliber, shared by EPR-025..029)",
    }
    if epochs is not None:
        rec["epochs"] = int(epochs)
    if loss_level is not None:
        rec.update(pure_l1_record(loss_level=loss_level,
                                  lambda_hc=float(lambda_hc or 0.0),
                                  lambda_sparse=float(lambda_sparse or 0.0),
                                  lambda_mono=float(lambda_mono)))
    return rec


# --------------------------------------------------------------------------- #
# 6. the flags
# --------------------------------------------------------------------------- #
def add_caliber_arguments(ap, *, group: str = "EPR-030 caliber (shared)",
                          default_data: str = DEFAULT_DATA,
                          default_batch_split: str | None = DEFAULT_BATCH_SPLIT,
                          default_base_lr: float | None = BASE_LR,
                          batch_split_extra: Sequence[str] = (),
                          data: bool = True, batch_split: bool = True,
                          base_lr: bool = True,
                          dataset_version: bool = True,
                          default_dataset_version: str = DEFAULT_DATASET_VERSION,
                          ) -> Any:
    """Register ``--dataset-version`` / ``--data`` / ``--zcache-root-l8`` /
    ``--batch-split`` / ``--base-lr``.

    One spelling on all arms.  A runner that already owns one of the flags
    (its own ``--batch-split`` choices, say) turns that one off and keeps the
    rest, so no flag is ever declared twice.
    """
    g = ap.add_argument_group(group)
    if dataset_version:
        g.add_argument("--dataset-version", default=default_dataset_version,
                       action=_RecordExplicit,
                       choices=list(DATASET_VERSION_CHOICES),
                       help="sft2seg index口径 (q3vl.whatb.splits.DATASET_VERSIONS).  "
                            "v20260804 = the published index (train normal "
                            "93934); cut-p45 = that index with the small-area "
                            "band/radial/semantic masks dropped (80269).  Both "
                            "index the same shards and keep the same sha1 split "
                            "assignment.  Recorded in run_setup.json")
    if data:
        g.add_argument("--data", default=default_data, choices=list(DATA_CHOICES),
                       help="training sources.  v2seg = the frozen sft2seg train "
                            "split (normal-only); v2seg+l8 adds the L8 normal "
                            "rows.  n / steps_per_epoch / total_steps are "
                            "measured from this, never written down")
        g.add_argument("--zcache-root-l8", default=str(S.L8_ZCACHE_ROOT),
                       help="z cache root of the L8 training source; only read "
                            "when --data names it (leaf: "
                            "<root>/<context>/l8_train__none)")
    if batch_split:
        g.add_argument("--batch-split", default=default_batch_split,
                       choices=list(batch_split_choices(batch_split_extra)),
                       help="B x Q from q3vl.whatb.arms.carrier.BATCH_SPLITS "
                            "(the one table); EPR-030 runs on 256x8192")
    if base_lr:
        g.add_argument("--base-lr", type=float, default=default_base_lr,
                       help="GLUT App A.1 base learning rate")
    return g
