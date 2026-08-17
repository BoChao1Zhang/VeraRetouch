"""GeoCode contract v1.0 -- the only thing the injection module is allowed to see.

Proposal ``docs/PROPOSAL_geometry-injection_2026-08-11.md`` §2.1, as amended by
**AMD-6** (21 discrete slots = 5 shape + 9 direction + 7 extent).  The extraction
arms (A1 / B1 / C) produce this and nothing else; PCH consumes this and nothing
else, which is what makes the three arms comparable at all (D-1: one shared
injector, per-arm weights).

    c_disc : float32 (21,)   [0:5] shape, [5:14] direction, [14:21] extent
                             values in [0,1] -- GT gives 0/1, readouts give
                             probabilities
    c_cont : float32 (3,)    (cx, cy, cover) in [0,1]; z-scored with **arm
                             constants** only, never per image, and the producer
                             must declare the domain (:class:`ContNorm`)
    conf   : float32 (4,)    group confidence [shape, dir, extent, cont] in [0,1]
                             (§2.3).  No NaN, no default -- parse failure, missing
                             reasoning span or a multi-span conflict is all-zero
    valid  : bool            false <=> conf is all zero (redundant on purpose; the
                             consumer asserts the two agree)

The slot vocabulary is **not duplicated here**.  ``geomparse.GEOM_SLOTS`` is the
single authority named by AMD-6 (A1 schema enums, A1-free regexes and the B1
self-generated tagger all derive from it), so this module *derives* the group
spans from it and asserts the (5, 9, 7) layout at import time.  A copy would be
a fourth list to drift, which is the failure AMD-6 exists to prevent.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import numpy as np
import torch

from .geomparse import GEOM_DIM, GEOM_SLOTS

__all__ = [
    "GROUP_NAMES", "DISC_GROUPS", "GROUP_SPANS", "GROUP_SIZES", "GROUP_INDEX",
    "contract_facts", "assert_contract",
    "ContNorm", "GeoCode", "CodePool",
    "label_smooth", "sample_conf_beta", "scale_conf", "group_only", "null_like",
    "constant_code", "condition_dropout", "assert_bitcount_conserved",
    "sattolo_derangement", "assert_derangement",
]

#: group order is the order the tokens are assembled in (§2.2 step 1)
GROUP_NAMES: tuple[str, ...] = ("shape", "dir", "ext", "cont")
#: the three groups that live in ``c_disc``; ``cont`` is the (cx, cy, cover) triple
DISC_GROUPS: tuple[str, ...] = ("shape", "dir", "ext")
GROUP_INDEX: dict[str, int] = {g: i for i, g in enumerate(GROUP_NAMES)}

_PREFIX = {"shape": "shape_", "dir": "dir_", "ext": "ext_"}
#: AMD-6's numbers, written down once so the assertion has something to fail against
EXPECTED_SIZES: dict[str, int] = {"shape": 5, "dir": 9, "ext": 7}
N_CONT = 3


def _derive_spans() -> dict[str, tuple[int, int]]:
    """``{group: (start, stop)}`` read off ``GEOM_SLOTS`` by slot-name prefix."""
    spans: dict[str, tuple[int, int]] = {}
    for g, pre in _PREFIX.items():
        idx = [i for i, (name, _) in enumerate(GEOM_SLOTS) if name.startswith(pre)]
        if not idx:
            raise AssertionError(f"GEOM_SLOTS carries no {pre!r} slot")
        if idx != list(range(idx[0], idx[-1] + 1)):
            raise AssertionError(
                f"the {g!r} slots are not contiguous in GEOM_SLOTS ({idx}); the "
                "GeoCode contract slices c_disc by span, so a scattered group "
                "would silently mix groups")
        spans[g] = (idx[0], idx[-1] + 1)
    return spans


GROUP_SPANS: dict[str, tuple[int, int]] = _derive_spans()
GROUP_SIZES: dict[str, int] = {g: b - a for g, (a, b) in GROUP_SPANS.items()}


def assert_contract() -> dict[str, Any]:
    """AMD-6: the 21-d contract must agree with ``GEOM_SLOTS`` slot by slot.

    Runs at import, so *any* consumer of this module trips the moment the slot
    table and the contract drift apart -- rather than at the end of a 10k-step
    run, in a board whose direction group has quietly moved by one index.
    """
    if GEOM_DIM != 21:
        raise AssertionError(
            f"AMD-6 fixes the discrete code at 21 slots; GEOM_SLOTS has "
            f"{GEOM_DIM}.  Update the proposal AND the readout heads together, "
            "never one of them")
    for g, want in EXPECTED_SIZES.items():
        if GROUP_SIZES[g] != want:
            raise AssertionError(
                f"AMD-6 fixes |{g}| = {want}; GEOM_SLOTS gives {GROUP_SIZES[g]}")
    order = [GROUP_SPANS[g] for g in DISC_GROUPS]
    if order != sorted(order) or order[0][0] != 0 or order[-1][1] != GEOM_DIM:
        raise AssertionError(
            f"c_disc must be shape|dir|ext back to back over [0,{GEOM_DIM}); got "
            f"{dict(zip(DISC_GROUPS, order))}")
    return contract_facts()


def contract_facts() -> dict[str, Any]:
    return {
        "geom_dim": GEOM_DIM,
        "group_spans": {g: list(s) for g, s in GROUP_SPANS.items()},
        "group_sizes": dict(GROUP_SIZES),
        "n_cont": N_CONT,
        "n_conf": len(GROUP_NAMES),
        "slot_names": [n for n, _ in GEOM_SLOTS],
        "authority": "q3vl.whereb.amort.geomparse.GEOM_SLOTS (AMD-6)",
    }


assert_contract()


# --- the continuous triple's producer/consumer contract ---------------------

@dataclass(frozen=True)
class ContNorm:
    """``meta.norm`` for ``c_cont``: arm constants + the declared domain.

    The s-cache contract in one object.  Two halves, both mandatory:

    * **producer** states the domain the whole arm's raw values live in;
    * **consumer** asserts the raw data it was handed really lives there, and
      **raises** instead of clamping -- a clamp silently amputates the negative
      half of a zero-crossing field and then presents itself as a quality
      problem (the contract's named silent failure).

    The z-score uses arm-wide constants only.  Per-image standardisation is a red
    line here for the same reason as in :mod:`simfield`: the whole point of
    ``cover`` is that it is comparable *across* samples.
    """

    mean: tuple[float, ...] = (0.0,) * N_CONT
    std: tuple[float, ...] = (1.0,) * N_CONT
    #: declared domain of the RAW values (contract §2.1 says [0,1])
    domain: tuple[float, float] = (0.0, 1.0)
    fitted: bool = False
    n_samples: int = 0
    source: str = "identity (unfitted): c_cont enters as the raw [0,1] triple"
    #: the assertion is on the domain, not on tail thickness -- see simfield
    tol: float = 1e-4

    def z(self, c_cont: torch.Tensor) -> torch.Tensor:
        m = torch.as_tensor(self.mean, dtype=c_cont.dtype, device=c_cont.device)
        s = torch.as_tensor(self.std, dtype=c_cont.dtype, device=c_cont.device)
        return (c_cont - m) / s.clamp_min(1e-6)

    def assert_in_domain(self, c_cont: torch.Tensor, *, enforce: bool = True
                         ) -> dict[str, float]:
        v = c_cont.detach().float()
        lo, hi = self.domain
        vmin, vmax = float(v.min()), float(v.max())
        if not np.isfinite([vmin, vmax]).all():
            raise AssertionError("c_cont carries non-finite values (NaN is "
                                 "forbidden by the GeoCode contract)")
        bad = bool(vmin < lo - self.tol or vmax > hi + self.tol)
        rep = {"min": vmin, "max": vmax, "domain_lo": lo, "domain_hi": hi,
               "violated": bad}
        if enforce and bad:
            raise AssertionError(
                f"c_cont lives outside the domain its producer declared: "
                f"[{vmin:.4f}, {vmax:.4f}] vs [{lo}, {hi}] ({self.source}).  "
                "Refusing to clamp -- a clamp would hide the shift and make "
                "every injection Delta unreadable")
        return rep

    @staticmethod
    def fit(values: Iterable[Sequence[float]], *, source: str = "",
            domain: tuple[float, float] | None = None) -> "ContNorm":
        """Arm constants from a sample of the arm's own codes.

        ``domain`` defaults to the **contract's** [0,1] rather than to the
        observed range: the observed range of a 256-sample fit is not the domain,
        and asserting against it would fire on the first legitimately larger
        ``cover`` -- the classic way an assertion becomes noise.  The observed
        range is recorded in ``source`` instead, where a reader can see it.
        """
        arr = np.asarray([list(v) for v in values], dtype=np.float64)
        if arr.ndim != 2 or arr.shape[1] != N_CONT:
            raise ValueError(f"expected (n, {N_CONT}) continuous values, got "
                             f"{arr.shape}")
        return ContNorm(
            mean=tuple(float(x) for x in arr.mean(0)),
            std=tuple(float(max(x, 1e-6)) for x in arr.std(0)),
            domain=domain or (0.0, 1.0),
            fitted=True, n_samples=int(arr.shape[0]),
            source=(source or "fitted over the arm (never per image)")
            + f"; observed [{float(arr.min()):.4f}, {float(arr.max()):.4f}]")

    def to_dict(self) -> dict[str, Any]:
        return {"mean": list(self.mean), "std": list(self.std),
                "domain": list(self.domain), "fitted": self.fitted,
                "n_samples": self.n_samples, "source": self.source,
                "tol": self.tol}

    @classmethod
    def from_dict(cls, d: Mapping[str, Any]) -> "ContNorm":
        return cls(mean=tuple(d.get("mean", (0.0,) * N_CONT)),
                   std=tuple(d.get("std", (1.0,) * N_CONT)),
                   domain=tuple(d.get("domain", (0.0, 1.0))),
                   fitted=bool(d.get("fitted", False)),
                   n_samples=int(d.get("n_samples", 0)),
                   source=str(d.get("source", "")),
                   tol=float(d.get("tol", 1e-4)))


# --- the code itself --------------------------------------------------------

def _as_vec(x, n: int, name: str, device=None) -> torch.Tensor:
    t = torch.as_tensor(x, dtype=torch.float32)
    t = t.reshape(-1)
    if t.numel() != n:
        raise ValueError(f"{name} must have {n} entries, got {t.numel()}")
    return t.to(device) if device is not None else t


@dataclass
class GeoCode:
    """One sample's geometry code.  Validated on construction, always."""

    c_disc: torch.Tensor
    c_cont: torch.Tensor
    conf: torch.Tensor
    valid: bool
    meta: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        self.c_disc = _as_vec(self.c_disc, GEOM_DIM, "c_disc")
        self.c_cont = _as_vec(self.c_cont, N_CONT, "c_cont")
        self.conf = _as_vec(self.conf, len(GROUP_NAMES), "conf")
        self.valid = bool(self.valid)
        self.validate()

    # -- contract ----------------------------------------------------------
    def validate(self) -> None:
        for name, t, hi in (("c_disc", self.c_disc, 1.0),
                            ("c_cont", self.c_cont, 1.0),
                            ("conf", self.conf, 1.0)):
            if not torch.isfinite(t).all():
                raise AssertionError(
                    f"{name} carries NaN/Inf; the GeoCode contract forbids it "
                    "outright (a NaN code silently poisons every downstream "
                    "field and lands as 'the injector did nothing')")
            if float(t.min()) < -1e-6 or float(t.max()) > hi + 1e-6:
                raise AssertionError(
                    f"{name} must live in [0,{hi}]; got "
                    f"[{float(t.min()):.4f}, {float(t.max()):.4f}]")
        has_conf = bool(float(self.conf.max()) > 0.0)
        if has_conf != self.valid:
            raise AssertionError(
                f"valid={self.valid} but conf={self.conf.tolist()}: the contract "
                "makes valid == (conf is not all zero) and the consumer asserts "
                "the two agree, because a 'valid' sample on the null path and an "
                "invalid one carrying a code are two different bugs that look "
                "identical downstream")

    # -- accessors ---------------------------------------------------------
    def group(self, name: str) -> torch.Tensor:
        if name == "cont":
            return self.c_cont
        a, b = GROUP_SPANS[name]
        return self.c_disc[a:b]

    def conf_of(self, name: str) -> float:
        return float(self.conf[GROUP_INDEX[name]])

    def n_active(self) -> int:
        """Active bits, thresholded at 0.5 -- the quantity the shuffle control
        must conserve (EPR-003's missing runtime assertion)."""
        return int((self.c_disc > 0.5).sum())

    def to(self, device=None, dtype=None) -> "GeoCode":
        kw: dict[str, Any] = {}
        if device is not None:
            kw["device"] = device
        if dtype is not None:
            kw["dtype"] = dtype
        if not kw:
            return self
        out = GeoCode.__new__(GeoCode)
        out.c_disc = self.c_disc.to(**kw)
        out.c_cont = self.c_cont.to(**kw)
        out.conf = self.conf.to(**kw)
        out.valid = self.valid
        out.meta = dict(self.meta)
        return out

    def replace(self, **kw) -> "GeoCode":
        d = {"c_disc": self.c_disc, "c_cont": self.c_cont, "conf": self.conf,
             "valid": self.valid, "meta": dict(self.meta)}
        d.update(kw)
        if "conf" in kw and "valid" not in kw:
            d["valid"] = bool(float(torch.as_tensor(d["conf"]).max()) > 0.0)
        return GeoCode(**d)

    def to_json(self) -> dict[str, Any]:
        return {"c_disc": [round(float(x), 6) for x in self.c_disc],
                "c_cont": [round(float(x), 6) for x in self.c_cont],
                "conf": [round(float(x), 6) for x in self.conf],
                "valid": self.valid, "meta": self.meta}

    # -- constructors ------------------------------------------------------
    @classmethod
    def null(cls, *, device=None, meta: dict[str, Any] | None = None) -> "GeoCode":
        """The all-zero code.  conf = 0 => valid = False => PCH's null path.

        Note this is *not* "zero contribution": §2.2's degeneracy insurance sends
        conf=0 to a **learned null constant**, which is what makes the null path
        trainable at all (condition dropout shapes it).  The exact-no-op
        guarantee is separate and structural (tanh(0)=0 + zero_conv at init).
        """
        z = torch.zeros(GEOM_DIM, device=device)
        return cls(z, torch.zeros(N_CONT, device=device),
                   torch.zeros(len(GROUP_NAMES), device=device), False,
                   meta or {"source": "null"})

    @classmethod
    def from_multihot(cls, v, *, c_cont=None, conf=None,
                      meta: dict[str, Any] | None = None) -> "GeoCode":
        """A 21-d multi-hot (+ optional continuous triple) -> a validated code.

        ``conf`` defaults to the §2.3 GT rule (1.0 everywhere), except that a
        group with no evidence at all -- no active slot, or an absent continuous
        triple -- gets 0 rather than a confident all-zero code.  Claiming
        conf=1 on a group the producer never filled is how a *missing* extent
        would masquerade as a confident "no extent".
        """
        d = _as_vec(v, GEOM_DIM, "c_disc")
        cont = (torch.zeros(N_CONT) if c_cont is None
                else _as_vec(c_cont, N_CONT, "c_cont"))
        if conf is None:
            c = [1.0 if float(d[a:b].sum()) > 0 else 0.0
                 for g, (a, b) in ((g, GROUP_SPANS[g]) for g in DISC_GROUPS)]
            c.append(0.0 if c_cont is None else 1.0)
            cf = torch.tensor(c, dtype=torch.float32)
        else:
            cf = _as_vec(conf, len(GROUP_NAMES), "conf")
        return cls(d, cont, cf, bool(float(cf.max()) > 0.0), meta or {})


# --- transforms (the M-series arms are all one of these) --------------------

def label_smooth(code: GeoCode, on: float = 0.9, off: float = 0.1) -> GeoCode:
    """M1-prob: 0/1 -> 0.9/0.1.  Deployment codes are probabilities, and an
    injector trained only on hard codes has never seen the distribution it will
    be asked to serve (risk 16)."""
    d = torch.where(code.c_disc > 0.5,
                    torch.full_like(code.c_disc, on),
                    torch.full_like(code.c_disc, off))
    return code.replace(c_disc=d, meta={**code.meta, "label_smooth": [on, off]})


def sample_conf_beta(code: GeoCode, rng: np.random.Generator,
                     a: float = 5.0, b: float = 2.0) -> GeoCode:
    """M1-prob: conf ~ Beta(5,2) on the groups that carry evidence.

    Groups whose conf is already 0 stay 0 -- resampling them would invent
    confidence for a group the producer never filled.
    """
    draws = rng.beta(a, b, size=len(GROUP_NAMES)).astype(np.float32)
    conf = torch.where(code.conf > 0, torch.from_numpy(draws).to(code.conf),
                       torch.zeros_like(code.conf))
    return code.replace(conf=conf, meta={**code.meta, "conf_beta": [a, b]})


def scale_conf(code: GeoCode, s: float) -> GeoCode:
    """M4: the conf sweep, {0, .25, .5, .75, 1}."""
    return code.replace(conf=code.conf * float(s),
                        meta={**code.meta, "conf_scale": float(s)})


def group_only(code: GeoCode, group: str) -> GeoCode:
    """M5: one group injected, the other three on the null path."""
    if group not in GROUP_NAMES:
        raise ValueError(f"unknown group {group!r}; expected {GROUP_NAMES}")
    keep = GROUP_INDEX[group]
    conf = torch.zeros_like(code.conf)
    conf[keep] = code.conf[keep]
    return code.replace(conf=conf, meta={**code.meta, "group_only": group})


def null_like(code: GeoCode) -> GeoCode:
    """M3: the all-empty code.  G4 asks whether the null path has side effects."""
    return GeoCode.null(device=code.c_disc.device,
                        meta={**code.meta, "forced": "null"})


def constant_code(mean: GeoCode) -> GeoCode:
    """M2b / Delta_const: every sample gets the dataset-mean code.

    The registered second control column (D-13).  Shuffle answers "does the
    *content* matter"; constant answers "does anything beyond the mere presence
    of a code matter", and the two fail differently.
    """
    return mean.replace(meta={**mean.meta, "forced": "constant"})


def condition_dropout(code: GeoCode, rng: np.random.Generator, *,
                      p_all: float = 0.15, p_group: float = 0.05
                      ) -> tuple[GeoCode, dict[str, bool]]:
    """§2.4 conditioning dropout: p=0.15 all groups to null + p=0.05 per group.

    This is what *shapes* the null path, so M5 (single-group injection) is only
    interpretable when the training log shows every group actually spent >=5% of
    its samples on it -- which is why the per-group draws are returned rather
    than swallowed.
    """
    hit_all = bool(rng.random() < p_all)
    if hit_all:
        out = null_like(code)
        return out, {"all": True, **{g: True for g in GROUP_NAMES}}
    conf = code.conf.clone()
    hits: dict[str, bool] = {"all": False}
    for g in GROUP_NAMES:
        h = bool(rng.random() < p_group)
        hits[g] = h
        if h:
            conf[GROUP_INDEX[g]] = 0.0
    return code.replace(conf=conf), hits


def assert_bitcount_conserved(before: GeoCode, after: GeoCode, *,
                              where: str = "") -> None:
    """The EPR-003 gap: a shuffle/derangement control must conserve *how much*
    signal is present, or "the geometry was wrong" is confounded with "there was
    less of it".  EPR-003 declared this and never asserted it at runtime.
    """
    nb, na = before.n_active(), after.n_active()
    if nb != na:
        raise AssertionError(
            f"{where or 'code control'} changed the active-bit count "
            f"{nb} -> {na}; a same-density control is a pre-registered property "
            "of every shuffle arm, not an implementation detail")


def sattolo_derangement(ids: Sequence[str], seed: int) -> dict[str, str]:
    """A permutation with **no fixed point**, via Sattolo's algorithm.

    Sattolo produces a single cycle, so ``m[i] != i`` holds by construction
    rather than by rejection sampling -- and because it is a permutation of the
    same multiset, every marginal (per-slot activation rate, conf histogram) is
    preserved exactly, which is what "保边缘分布" asks for.
    """
    if len(ids) < 2:
        raise ValueError("a derangement needs at least two samples")
    rng = np.random.default_rng(seed)
    order = list(ids)
    perm = list(range(len(order)))
    for i in range(len(perm) - 1, 0, -1):
        j = int(rng.integers(0, i))          # strictly < i  => single cycle
        perm[i], perm[j] = perm[j], perm[i]
    return {order[i]: order[perm[i]] for i in range(len(order))}


def assert_derangement(mapping: Mapping[str, str]) -> None:
    fixed = [k for k, v in mapping.items() if k == v]
    if fixed:
        raise AssertionError(
            f"{len(fixed)} sample(s) kept their own code under the M2 "
            f"derangement (e.g. {fixed[:3]}); a partial derangement inflates "
            "Delta(M1-M2) by exactly the fraction that was not deranged")
    if len(set(mapping.values())) != len(mapping):
        raise AssertionError("the derangement is not a permutation, so the "
                             "code marginals are no longer preserved")


# --- the pool the eval-only arms are built from -----------------------------

@dataclass
class CodePool:
    """Every eval sample's code, materialised once.

    M2/M2b need the *set* of codes (a derangement and a mean), so they cannot be
    computed sample by sample.  Building the pool up front also makes the
    marginal-preservation and bit-count assertions checkable rather than
    aspirational.
    """

    codes: dict[str, GeoCode]
    source: str = ""

    def __len__(self) -> int:
        return len(self.codes)

    def ids(self) -> list[str]:
        return sorted(self.codes)

    def matrix(self) -> np.ndarray:
        return np.stack([self.codes[i].c_disc.cpu().numpy() for i in self.ids()])

    def marginals(self) -> np.ndarray:
        return self.matrix().mean(0)

    def mean_code(self) -> GeoCode:
        ids = self.ids()
        if not ids:
            raise ValueError("empty code pool")
        d = np.stack([self.codes[i].c_disc.cpu().numpy() for i in ids]).mean(0)
        c = np.stack([self.codes[i].c_cont.cpu().numpy() for i in ids]).mean(0)
        f = np.stack([self.codes[i].conf.cpu().numpy() for i in ids]).mean(0)
        return GeoCode(torch.from_numpy(d.astype(np.float32)),
                       torch.from_numpy(c.astype(np.float32)),
                       torch.from_numpy(f.astype(np.float32)),
                       bool(f.max() > 0),
                       {"source": f"mean of {len(ids)} codes ({self.source})"})

    def derangement(self, seed: int) -> dict[str, str]:
        m = sattolo_derangement(self.ids(), seed)
        assert_derangement(m)
        return m

    def stats(self) -> dict[str, Any]:
        if not self.codes:
            return {"n": 0}
        m = self.matrix()
        conf = np.stack([self.codes[i].conf.cpu().numpy() for i in self.ids()])
        return {
            "n": len(self.codes),
            "source": self.source,
            "n_valid": int(sum(c.valid for c in self.codes.values())),
            "active_bits_mean": float((m > 0.5).sum(1).mean()),
            "marginals": [round(float(x), 6) for x in m.mean(0)],
            "conf_mean": [round(float(x), 6) for x in conf.mean(0)],
            "group_coverage": {g: float((m[:, GROUP_SPANS[g][0]:GROUP_SPANS[g][1]]
                                         > 0.5).any(1).mean())
                               for g in DISC_GROUPS},
        }

    # -- io ---------------------------------------------------------------
    def write(self, path: str | Path) -> Path:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w", encoding="utf-8") as fh:
            for sid in self.ids():
                fh.write(json.dumps({"sample_id": sid,
                                     **self.codes[sid].to_json()}) + "\n")
        return path

    @classmethod
    def read(cls, path: str | Path) -> "CodePool":
        """Read a ``codes.jsonl`` -- the seam every extraction arm writes to.

        This is also AMD-1's eval-only entry point: M1's trained weights are
        pointed at another arm's code file and scored zero-shot, reported as a
        reference column only.
        """
        codes: dict[str, GeoCode] = {}
        for line in Path(path).read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            row = json.loads(line)
            sid = row["sample_id"]
            # tolerate the two spellings a producer plausibly writes -- the
            # contract's own names, or a bare multi-hot under `code`/`geom`.
            # Anything else is refused rather than guessed at.
            disc = next((row[k] for k in ("c_disc", "code", "geom", "features")
                         if k in row), None)
            if disc is None:
                raise ValueError(
                    f"{path}: row for {sid} has no discrete code (expected one "
                    "of c_disc/code/geom/features)")
            cont = next((row[k] for k in ("c_cont", "cont") if k in row),
                        [0.0] * N_CONT)
            conf = row.get("conf")
            if conf is None:
                # a producer that ships no confidence is asserting nothing about
                # it; §2.3's presence rule is the only defensible default and it
                # is recorded as such rather than silently applied as 1.0
                codes[sid] = GeoCode.from_multihot(
                    disc, c_cont=cont,
                    meta={"source": str(path), "conf": "derived (presence rule)",
                          **row.get("meta", {})})
                continue
            codes[sid] = GeoCode(disc, cont, conf,
                                 bool(row.get("valid", max(conf) > 0)),
                                 {"source": str(path), **row.get("meta", {})})
        return cls(codes, source=str(path))
