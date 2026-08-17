"""The M-series qualifier (proposal §2.5) as a harness, not a set of scripts.

Nine arms decide whether the 21-d geometry code is worth extracting at all, and
seven of them are the *same trained weights* scored under a different code.  So
the arm is a **code transform** plus a scoring convention, and the only honest
way to keep them comparable is to make them one object:

    M0        no injection -- the resumed checkpoint's own board
    M1-full   GT code, PCH-Full          (trained)
    M1-lite   GT code, PCH-Lite          (trained; G6 asks |full-lite| <= 0.005)
    M1-prob   GT code, label-smoothed 0.9/0.1 + conf ~ Beta(5,2)   (trained)
    M2        M1 weights, eval-set code derangement    (G2: content specificity)
    M2b       M1 weights, dataset-mean constant code   (D-13: the Delta_const column)
    M3        M1 weights, all-empty code               (G4: the null path is inert)
    M4        M1 weights, conf in {0,.25,.5,.75,1}     (conf sweep)
    M5        M1 weights, one group at a time          (per-group attribution)

Three runtime assertions live here rather than in a checklist, because all three
have already failed silently at least once in this campaign:

* **the transform must actually run.**  A pre-registered criterion that is
  defined, imported and never called has happened three times (SHAPE3's
  ``shape_residual`` most recently), and an M2 board whose derangement never
  fired looks exactly like an M2 board where the code did not matter.
  :func:`assert_mseries_wired` refuses to publish a board in that state.
* **the shuffle/derangement controls must conserve what they claim to.**
  EPR-003 registered "preserving the active-bit count" and asserted it nowhere.
* **c_cont must live in the domain its producer declared** (s-cache contract),
  and the Delta-logit domain must ship with the delivery, so that "the injection
  did nothing" can be distinguished from "the tap was dead".

Code sources are pluggable on purpose.  The GT code itself is being repaired
under AMD-8 (dir/extent/cont from the construction-side geometry parameters
instead of vrmeta ``region``, 82% of which is the degenerate value "center" that
the data-discipline red line forbids as a direction label), and that repair is a
separate deliverable.  Everything here therefore reads codes through
:class:`CodeSource`, and :class:`FileCodeSource` -- a ``codes.jsonl`` keyed by
``sample_id`` -- is the seam the repaired producer, the A1/B1/C arms, and AMD-1's
"M1 weights, zero-shot, on somebody else's codes" reference column all plug into.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import torch

from .geocode import (DISC_GROUPS, GROUP_NAMES, GROUP_SPANS, CodePool, ContNorm,
                      GeoCode, condition_dropout, constant_code, contract_facts,
                      group_only, label_smooth, null_like, sample_conf_beta,
                      scale_conf)

__all__ = ["MSeriesSpec", "M_ARMS", "arm_spec", "CodeSource",
           "VrmetaLegacyCodeSource", "ParsedTextCodeSource", "FileCodeSource",
           "ConstructionCodeSource",
           "MSeriesHarness", "TRAIN_RECIPE", "train_config_kwargs",
           "prepare_pool", "assert_marginals_preserved", "assert_mseries_wired",
           "assert_no_auc", "make_code_source"]


# --- arm table --------------------------------------------------------------

@dataclass(frozen=True)
class MSeriesSpec:
    name: str
    #: what the arm does to the code before it reaches PCH
    transform: str
    #: eval-only arms reuse M1's weights; trained arms need a training run
    eval_only: bool
    #: None = take the capacity from the CLI (M2..M5 inherit M1's)
    pch_size: str | None = None
    #: does the transform need the whole eval set's codes materialised first?
    needs_pool: bool = False
    inject: bool = True
    note: str = ""

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


M_ARMS: dict[str, MSeriesSpec] = {
    "M0": MSeriesSpec("M0", "none", eval_only=True, inject=False,
                      note="frozen unconditional baseline; with --freeze-base "
                           "this is literally the resumed checkpoint's board"),
    "M1-full": MSeriesSpec("M1-full", "identity", eval_only=False,
                           pch_size="full", note="GT code, upper bound (Gate-0)"),
    "M1-lite": MSeriesSpec("M1-lite", "identity", eval_only=False,
                           pch_size="lite", note="G6 capacity ablation"),
    "M1-prob": MSeriesSpec("M1-prob", "prob", eval_only=False, pch_size="full",
                           note="label smoothing 0.9/0.1 + conf ~ Beta(5,2): the "
                                "deployment code distribution, not the training one"),
    "M2": MSeriesSpec("M2", "derangement", eval_only=True, needs_pool=True,
                      note="G2 content specificity; marginals preserved"),
    "M2b": MSeriesSpec("M2b", "constant", eval_only=True, needs_pool=True,
                       note="D-13 Delta_const column"),
    "M3": MSeriesSpec("M3", "null", eval_only=True,
                      note="G4 |M3 - M0| <= 0.003, else the null path has side "
                           "effects and every Delta in the series is void"),
    "M4": MSeriesSpec("M4", "conf_scale", eval_only=True,
                      note="conf sweep {0,.25,.5,.75,1}"),
    "M5": MSeriesSpec("M5", "group_only", eval_only=True,
                      note="single-group injection; interpretable only if M3 "
                           "passed G4 AND the training log shows >=5% null "
                           "dropout coverage for that group"),
}


def arm_spec(name: str) -> MSeriesSpec:
    if name not in M_ARMS:
        raise ValueError(f"unknown M-series arm {name!r}; expected one of "
                         f"{sorted(M_ARMS)}")
    return M_ARMS[name]


# --- §2.4 training recipe ---------------------------------------------------

#: Proposal §2.4, verbatim.  Kept as data so the delivery can show the recipe it
#: ran under next to the recipe that was registered.
TRAIN_RECIPE: dict[str, Any] = {
    "trainable": "PCH only (VLM and dense head frozen; --freeze-base)",
    "loss": "the head's existing BCE-with-logits family, unchanged; no dice "
            "(D-5's exemption is for B1's coarse auxiliary branch, not here), "
            "no IoU as an optimisation target",
    "optimizer": "AdamW",
    "learning_rate": 3e-4,
    "weight_decay": 1e-4,
    "scheduler": "cosine",
    "warmup_steps": 300,
    "effective_batch": 64,
    "max_steps": 10_000,
    "cond_dropout_all": 0.15,
    "cond_dropout_group": 0.05,
    "checkpoint_selection": "S-val soft-IoU behind the hard gates; never a val loss",
}


def train_config_kwargs(max_steps: int = TRAIN_RECIPE["max_steps"],
                        effective_batch: int = TRAIN_RECIPE["effective_batch"]
                        ) -> dict[str, Any]:
    """§2.4 as ``AmortTrainConfig`` keyword arguments.

    ``warmup_ratio`` rather than a step count because that is the knob
    ``make_scheduler`` takes; 300/10000 = 0.03 exactly, and the ratio is
    recomputed here so a shortened smoke run still warms up over 300 steps'
    worth of *its* schedule rather than silently over three.
    """
    return {
        "learning_rate": TRAIN_RECIPE["learning_rate"],
        "weight_decay": TRAIN_RECIPE["weight_decay"],
        "scheduler": TRAIN_RECIPE["scheduler"],
        "warmup_ratio": TRAIN_RECIPE["warmup_steps"] / max(1, int(max_steps)),
        "effective_batch": int(effective_batch),
        "max_steps": int(max_steps),
    }


# --- code sources -----------------------------------------------------------

class CodeSource:
    """``sample_id (+ context text) -> GeoCode``.  Producers implement this."""

    name = "abstract"
    #: producer-side meta.norm for c_cont (s-cache contract)
    cont_norm: ContNorm = ContNorm()

    def code_for(self, sample_id: str, text: str = "") -> GeoCode:
        raise NotImplementedError

    def pool_for(self, sample_ids: Sequence[str]) -> CodePool:
        return CodePool({sid: self.code_for(sid) for sid in sample_ids},
                        source=self.name)

    def facts(self) -> dict[str, Any]:
        return {"code_source": self.name, "cont_norm": self.cont_norm.to_dict()}


class VrmetaLegacyCodeSource(CodeSource):
    """``.vrmeta.json`` slot_id/region -- the code EPR-001/002/003 actually ran.

    **AMD-8 supersedes it.**  Two defects, both recorded rather than papered
    over: ``region`` is 82% the degenerate value "center" (which the data
    discipline forbids as a direction label), and the extent slots plus the
    continuous triple are simply absent.  So the confidence this source declares
    is ``conf_ext = conf_cont = 0`` -- a group with no evidence takes the null
    path, which is the whole reason the contract carries a per-group confidence.
    Claiming conf=1 on an all-zero extent group would present *missing* extent to
    the module as a confident "no extent", and the M5 attribution row would then
    be reading a fabrication.

    Kept so the completed-version rerun can be scored against exactly the code
    the v0 series was scored against; the repaired producer arrives as a
    ``codes.jsonl`` and enters through :class:`FileCodeSource`.
    """

    name = "vrmeta-legacy (AMD-8 superseded)"

    def __init__(self, builder):
        self.builder = builder

    def code_for(self, sample_id: str, text: str = "") -> GeoCode:
        v = self.builder._vrmeta_code(sample_id)
        if v is None:
            return GeoCode.null(meta={"source": self.name, "reason": "no vrmeta"})
        d = torch.as_tensor(np.asarray(v, dtype=np.float32))
        conf = torch.zeros(len(GROUP_NAMES))
        for i, g in enumerate(DISC_GROUPS):
            a, b = GROUP_SPANS[g]
            conf[i] = 1.0 if float(d[a:b].sum()) > 0 else 0.0
        return GeoCode(d, torch.zeros(3), conf, bool(float(conf.max()) > 0),
                       {"source": self.name,
                        "amd8_pending": True,
                        "extent_and_cont": "absent from vrmeta -> conf 0 -> null path"})

    def facts(self) -> dict[str, Any]:
        return {**super().facts(),
                "warning": "AMD-8: dir comes from vrmeta `region` (82% 'center'), "
                           "extent and cont are absent.  Use only to reproduce the "
                           "v0 series' conditions."}


class ParsedTextCodeSource(CodeSource):
    """The frozen-vocabulary parse of the ``<where>`` span (the v0 parsed arm).

    Confidence here is presence-based (a group with an active slot gets 1.0),
    **not** A1's §2.3 rule.  A1's confidence is a masked-softmax span minimum
    computed during constrained decoding and cannot be reconstructed from the
    text afterwards; when the A1 arm runs it writes its own ``codes.jsonl`` with
    the real numbers and enters through :class:`FileCodeSource`.
    """

    name = "parsed-text (presence conf; NOT the A1 §2.3 rule)"

    def code_for(self, sample_id: str, text: str = "") -> GeoCode:
        from .geomparse import geom_features

        return GeoCode.from_multihot(geom_features(text or ""),
                                     meta={"source": self.name})

    def pool_for(self, sample_ids: Sequence[str]) -> CodePool:
        raise NotImplementedError(
            "the parsed source cannot build a pool: its code depends on the "
            "context text, which differs per eval mode, so a pool built here "
            "would silently mix teacher-forced and generated codes.  M2/M2b on a "
            "parsed arm must read a codes.jsonl produced under one fixed context")


class ConstructionCodeSource(CodeSource):
    """AMD-8's GT code: the construction-side geometry parameters.

    Reads through :class:`~q3vl.whereb.amort.data.ConstructGeomStore`, which is
    EPR-009's deliverable: the sqlite sidecar reader (thread-local connection,
    keyed by ``candidate_id``) plus the call into
    :func:`geomparse.geom_features_from_construction`.  Imported, never
    re-implemented -- a second reader would be a second place for the bucket
    edges and the frame-aspect ``size`` argument to drift, which is exactly what
    AMD-6/AMD-8 exist to prevent (omitting ``size`` alone moves a 45 deg axis to
    33.7 deg in a 3:2 frame and flips orientation buckets).

    Unlike the v0 conditioning path, which keeps only the 21-d multi-hot, this
    source keeps the whole ``(c_disc, c_cont, conf)`` triple -- the §2.2 module
    is the first consumer that has somewhere to put the other two.
    """

    name = "construction geometry (AMD-8)"

    def __init__(self, store, dataset, id_to_index: Mapping[str, int],
                 cont_norm: ContNorm | None = None):
        self.store = store
        self.dataset = dataset
        self.id_to_index = dict(id_to_index or {})
        if cont_norm is not None:
            self.cont_norm = cont_norm
        self.n_missing = 0
        self._cache: dict[str, GeoCode] = {}

    def code_for(self, sample_id: str, text: str = "") -> GeoCode:
        got = self._cache.get(sample_id)
        if got is not None:
            return got
        idx = self.id_to_index.get(sample_id)
        triple = None
        if idx is not None and self.dataset is not None:
            triple = self.store.code(self.dataset.record(idx))
        if triple is None:
            # never a `region` backfill and never a dropped sample: an
            # unanswerable sample is an ABSTAIN and takes the null path
            self.n_missing += 1
            out = GeoCode.null(meta={"source": self.name,
                                     "reason": "no construction parameters"})
        else:
            d, cont, conf = triple
            out = GeoCode(d, cont, conf, bool(float(np.max(conf)) > 0),
                          {"source": self.name})
        self._cache[sample_id] = out
        return out

    def facts(self) -> dict[str, Any]:
        return {**super().facts(),
                "n_missing_parameters": self.n_missing,
                "store": {"path": getattr(self.store, "path", None),
                          "n_hit": getattr(self.store, "n_hit", None),
                          "n_miss": getattr(self.store, "n_miss", None)}}


class FileCodeSource(CodeSource):
    """A ``codes.jsonl`` keyed by ``sample_id`` -- the arm-to-arm seam.

    This is how the AMD-8 repaired GT code, A1's parsed codes, B1's readout codes
    and C's bridge codes all reach the module, and how AMD-1's reference column
    is produced (M1's trained weights, zero-shot, on another arm's code file --
    reported, never adjudicated).
    """

    def __init__(self, path: str | Path, cont_norm: ContNorm | None = None):
        self.pool = CodePool.read(path)
        self.path = str(path)
        self.name = f"codes.jsonl ({self.path})"
        meta = Path(str(path) + ".norm.json")
        if cont_norm is not None:
            self.cont_norm = cont_norm
        elif meta.exists():
            self.cont_norm = ContNorm.from_dict(json.loads(meta.read_text()))

    def code_for(self, sample_id: str, text: str = "") -> GeoCode:
        code = self.pool.codes.get(sample_id)
        if code is None:
            # never drop the sample and never crash: an absent code is an
            # ABSTAIN, which the contract routes to the null path
            return GeoCode.null(meta={"source": self.name, "reason": "absent"})
        return code

    def pool_for(self, sample_ids: Sequence[str]) -> CodePool:
        return CodePool({sid: self.code_for(sid) for sid in sample_ids},
                        source=self.name)

    def facts(self) -> dict[str, Any]:
        return {**super().facts(), "path": self.path, "pool": self.pool.stats()}


# --- the harness ------------------------------------------------------------

class MSeriesHarness:
    """Wraps an :class:`~q3vl.whereb.amort.data.AmortBatchBuilder`.

    It delegates everything about images, text and masks to the inner builder and
    owns exactly one thing: the ``GeoCode`` each sample is injected with.  That
    separation is deliberate -- the M series must not be able to change the data
    the baseline was measured on, only the code.
    """

    def __init__(self, inner, source: CodeSource, spec: MSeriesSpec, *,
                 train_mode: bool = False, seed: int = 20260812,
                 conf_scale: float = 1.0, group: str = "shape",
                 pool: CodePool | None = None,
                 cond_dropout_all: float = TRAIN_RECIPE["cond_dropout_all"],
                 cond_dropout_group: float = TRAIN_RECIPE["cond_dropout_group"]):
        self.inner = inner
        self.source = source
        self.spec = spec
        self.train_mode = bool(train_mode)
        self.rng = np.random.default_rng(seed)
        self.seed = int(seed)
        self.conf_scale = float(conf_scale)
        self.group = group
        self.pool = pool
        self.mapping: dict[str, str] | None = None
        self.cond_dropout_all = float(cond_dropout_all)
        self.cond_dropout_group = float(cond_dropout_group)

        # observables.  The point of counting is that "the transform ran" must be
        # a fact in the delivery, not an inference from the code being present.
        self.n_built = 0
        self.n_transformed = 0
        self.n_null = 0
        self.n_dropout_all = 0
        self.dropout_hits: dict[str, int] = {g: 0 for g in GROUP_NAMES}
        self.cont_domain_reports: list[dict[str, float]] = []

        if spec.needs_pool:
            if pool is None:
                raise ValueError(
                    f"arm {spec.name} needs the eval set's codes materialised "
                    "first (a derangement and a mean are properties of the SET); "
                    "call prepare_pool() before building")
            if spec.transform == "derangement":
                self.mapping = pool.derangement(seed)

    # -- pass-through -------------------------------------------------------
    def __getattr__(self, item):
        # everything not overridden belongs to the inner builder (device,
        # genctx, context_for, ...).  `.get` rather than `[...]` so an attribute
        # touched before __init__ finished raises AttributeError instead of a
        # KeyError nobody can read.
        inner = self.__dict__.get("inner")
        if inner is None:
            raise AttributeError(item)
        return getattr(inner, item)

    @property
    def allow_context_fallback(self) -> bool:
        return self.inner.allow_context_fallback

    @allow_context_fallback.setter
    def allow_context_fallback(self, v: bool) -> None:
        self.inner.allow_context_fallback = bool(v)

    # -- code assembly ------------------------------------------------------
    def code_for(self, sample_id: str, text: str = "") -> GeoCode:
        base = self.source.code_for(sample_id, text)
        t = self.spec.transform
        if t in ("none",):
            return base
        if t == "identity":
            out = base
        elif t == "prob":
            out = sample_conf_beta(label_smooth(base), self.rng)
        elif t == "derangement":
            other = self.mapping[sample_id] if self.mapping else sample_id
            out = self.pool.codes[other].replace(
                meta={"source": self.source.name, "deranged_from": other})
        elif t == "constant":
            out = constant_code(self.pool.mean_code())
        elif t == "null":
            out = null_like(base)
        elif t == "conf_scale":
            out = scale_conf(base, self.conf_scale)
        elif t == "group_only":
            out = group_only(base, self.group)
        else:                                            # pragma: no cover
            raise ValueError(f"unknown transform {t!r}")
        self.n_transformed += 1

        if self.train_mode and self.cond_dropout_all + self.cond_dropout_group > 0:
            out, hits = condition_dropout(
                out, self.rng, p_all=self.cond_dropout_all,
                p_group=self.cond_dropout_group)
            self.n_dropout_all += int(hits["all"])
            for g in GROUP_NAMES:
                self.dropout_hits[g] += int(hits[g])
        if not out.valid:
            self.n_null += 1
        self.cont_domain_reports.append(
            self.source.cont_norm.assert_in_domain(out.c_cont))
        return out

    # -- builder API --------------------------------------------------------
    def build(self, samples, modes):
        inputs = self.inner.build(samples, modes)
        for x in inputs:
            text = ""
            try:
                text = x.meta.get("context_text", "") or ""
            except Exception:                            # pragma: no cover
                text = ""
            code = self.code_for(x.sample_id, text)
            x.geom = code.to(device=self.inner.device)
            self.n_built += 1
        return inputs

    # -- reporting ----------------------------------------------------------
    def dropout_coverage(self) -> dict[str, float]:
        """Fraction of samples that saw each group on the null path.

        M5's pre-condition (§2.5): below 5% for a group, that group's
        single-injection row is marked uninterpretable rather than reported --
        the module was never trained to serve it alone.  All-group dropouts are
        already counted in every group's tally, which is what makes this the
        coverage number the pre-condition asks for.
        """
        n = max(1, self.n_built)
        return {g: self.dropout_hits[g] / n for g in GROUP_NAMES}

    def facts(self) -> dict[str, Any]:
        inner = {}
        try:
            inner = self.inner.facts()
        except Exception:                                # pragma: no cover
            pass
        return {
            **inner,
            "m_series": {
                "arm": self.spec.to_dict(),
                "seed": self.seed,
                "train_mode": self.train_mode,
                "conf_scale": self.conf_scale,
                "group": self.group,
                "n_codes_built": self.n_built,
                "n_transform_calls": self.n_transformed,
                "n_null_codes": self.n_null,
                "cond_dropout": {"p_all": self.cond_dropout_all,
                                 "p_group": self.cond_dropout_group,
                                 "n_all_hits": self.n_dropout_all,
                                 "per_group_hits": dict(self.dropout_hits),
                                 "per_group_coverage": self.dropout_coverage()},
                "pool": (self.pool.stats() if self.pool is not None else None),
                "derangement_fixed_points": (
                    0 if self.mapping is None
                    else sum(1 for k, v in self.mapping.items() if k == v)),
                "contract": contract_facts(),
                **self.source.facts(),
            },
        }


def make_code_source(kind: str, *, builder=None, code_file: str | Path | None = None,
                     geometry_db: str | Path | None = None, dataset=None,
                     id_to_index: Mapping[str, int] | None = None) -> CodeSource:
    """``--code-source`` -> a producer.  One place, so the run record names it."""
    if kind == "file":
        if not code_file:
            raise ValueError("--code-source file needs --code-file")
        return FileCodeSource(code_file)
    if kind == "construction":
        from .data import ConstructGeomStore

        if geometry_db is None:
            from q3vl.whereb.config import CONSTRUCT_GEOM_DB

            geometry_db = CONSTRUCT_GEOM_DB
        # the store raises if the sidecar is absent -- the parameters are NOT in
        # .vrmeta.json, so a missing file means the export step never ran and the
        # arm would otherwise train on an all-null code without saying so
        return ConstructionCodeSource(ConstructGeomStore(geometry_db), dataset,
                                      id_to_index or {})
    if kind == "vrmeta":
        if builder is None:
            raise ValueError("--code-source vrmeta needs the batch builder")
        return VrmetaLegacyCodeSource(builder)
    if kind == "parsed":
        return ParsedTextCodeSource()
    raise ValueError(f"unknown code source {kind!r}")


def prepare_pool(source: CodeSource, sample_ids: Sequence[str]) -> CodePool:
    """Materialise the eval set's codes (M2/M2b) and check the marginals.

    The derangement is a permutation of this pool, so the per-slot activation
    rates are preserved by construction; the assertion below is what makes that
    a checked fact rather than a claim about an algorithm.
    """
    pool = source.pool_for(list(sample_ids))
    if len(pool) < 2:
        raise ValueError("a derangement/mean needs at least two codes")
    return pool


# --- runtime assertions -----------------------------------------------------

def assert_marginals_preserved(pool: CodePool, mapping: Mapping[str, str],
                               *, tol: float = 1e-6) -> dict[str, Any]:
    """M2 must move codes between samples, not change the code distribution."""
    before = pool.marginals()
    ids = pool.ids()
    after = np.stack([pool.codes[mapping[i]].c_disc.cpu().numpy() for i in ids]).mean(0)
    delta = float(np.abs(before - after).max())
    if delta > tol:
        raise AssertionError(
            f"the M2 derangement changed the code marginals by {delta:.2e}; "
            "G2 compares M1 against a same-distribution control, so a shifted "
            "marginal would confound 'the content was wrong' with 'the codes "
            "were different codes'")
    return {"max_marginal_delta": delta, "n": len(ids)}


def assert_no_auc(board: Mapping[str, Any]) -> None:
    """AUC in any variant is a red line (three separate misleading readings).

    Cheap, and it catches the realistic failure: a metrics dict merged in from a
    helper that computes one "just for reference".
    """
    def walk(node, path=""):
        if isinstance(node, Mapping):
            for k, v in node.items():
                if "auc" in str(k).lower():
                    raise AssertionError(
                        f"board carries an AUC-like key at {path}/{k}; AUC is "
                        "banned outright in this campaign (subject prior "
                        "masquerading as instruction understanding; a centre "
                        "prior at 0.836 beating every readout; blind to a 2.2dB "
                        "difference)")
                walk(v, f"{path}/{k}")
        elif isinstance(node, (list, tuple)):
            for i, v in enumerate(node):
                walk(v, f"{path}[{i}]")

    walk(board)


def assert_board_columns(board: Mapping[str, Any]) -> dict[str, Any]:
    """The three mandatory spatial columns + the normal-only headline + no AUC.

    Applies to M0 as much as to the injected arms: the baseline every Delta is
    taken against has to be quotable under the same convention, or the
    subtraction is between two different numbers.
    """
    contexts = board.get("contexts", {})
    main = board.get("main_context") or ("generated" if "generated" in contexts
                                         else "gt")
    ctx = contexts.get(main, {})
    if not ctx.get("headline_normal_only"):
        raise AssertionError(
            f"context {main!r} carries no `headline_normal_only`; the reporting "
            "convention is normal-only (K6) and the pooled figure over-reads by "
            "~0.031, so a board without it cannot be quoted")
    checks: dict[str, Any] = {"main_context": main}
    for col in ("topk_iou", "grid_boundary_f1", "center_prior_topk_iou"):
        n = int(ctx.get(col, {}).get("n", 0))
        checks[col] = n
        if n == 0:
            raise AssertionError(
                f"the board carries 0 values for {col!r}.  All three of "
                "matched-area top-k IoU, grid-level boundary F1 and the "
                "centre-prior baseline are mandatory columns -- a localisation "
                "claim without the paired centre-prior Delta is exactly what the "
                "red line forbids")
    assert_no_auc(board)
    checks["no_auc"] = True
    return checks


def assert_mseries_wired(board: Mapping[str, Any], harness: MSeriesHarness,
                         *, min_codes: int = 1) -> dict[str, Any]:
    """Refuse a board the M series cannot adjudicate.

    Checks, in order of how they have actually failed here:

    1. the code transform ran at all (the "defined but not wired" class);
    2. the injected arms really injected (n_codes_built > 0);
    3. the headline is the normal-only figure (U7/K6: pooled is 0.031 optimistic);
    4. the three mandatory spatial columns carry values -- matched-area top-k
       IoU, grid-level boundary F1, and the centre-prior baseline;
    5. no AUC anywhere.
    """
    spec = harness.spec
    report: dict[str, Any] = {"arm": spec.name, "checks": {}}

    if spec.inject:
        if harness.n_built < min_codes:
            raise AssertionError(
                f"arm {spec.name} built {harness.n_built} codes: the injection "
                "path never ran, so whatever this board measured, it is not the "
                "arm it is labelled as")
        if spec.transform not in ("none", "identity") and harness.n_transformed == 0:
            raise AssertionError(
                f"arm {spec.name} declares transform {spec.transform!r} and it "
                "was never applied; this is the third occurrence in this "
                "campaign of a criterion that exists, is imported, and is not "
                "called")
    report["checks"]["n_codes_built"] = harness.n_built
    report["checks"]["n_transform_calls"] = harness.n_transformed

    report["checks"].update(assert_board_columns(board))
    if harness.mapping is not None and harness.pool is not None:
        report["checks"]["marginals"] = assert_marginals_preserved(
            harness.pool, harness.mapping)
    report["cont_domain"] = {
        "n_checks": len(harness.cont_domain_reports),
        "violations": sum(1 for r in harness.cont_domain_reports
                          if r.get("violated")),
    }
    return report
