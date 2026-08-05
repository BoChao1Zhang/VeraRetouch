"""Reader for the published generated-``<color>`` context (amendment A-4).

The producer is the **extended** Where-B generation job
(``q3vl.whereb.scripts.make_generated_context``, schema ``genwhere/2``): the Base
SFT model already emits ``<where>...</where><color>...</color>`` in one greedy
pass, so keeping the second segment costs nothing beyond writing it down.  WB-IMPL
owns that job; this module owns the *contract* Stage-What reads it under, and
states it as assertions rather than as an assumption:

* ``schema_version`` must be the v2 schema -- a v1 record has no ``<color>``
  segment at all, and reading one would silently give every sample an empty
  generated context;
* ``mode`` must be the one the arm asked for.  ``C01``/``C02`` need the
  ``forced_color_prefix`` generation (no ``<where>`` in the prompt); every other
  arm needs ``with_where_prefix``.  Handing an arm the wrong one is invisible in
  the loss and would put the where reasoning back into the strict no-where
  control through the token ids;
* every sample of the split must be present.  A missing record has no legitimate
  substitute -- falling back to the GT span is the one thing amendment A-4
  forbids -- so it raises.

Only ids are read; the hidden states are re-derived by the same
:meth:`q3vl.what.hiddens.WhatVLM.encode` that produces the teacher context.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Iterable, Iterator

from q3vl.whereb.stores import PublishedStore

from .config import GENCTX_MODES, SCHEMA_COLOR_GENCTX

__all__ = ["ColorGenContextStore", "COLOR_GENCTX_SUFFIX"]

COLOR_GENCTX_SUFFIX = ".genwhere.json"
#: the record fields Stage-What consumes.  Named here so that a schema change on
#: the producer side surfaces as one failure with a list, not as a KeyError deep
#: inside a batch build.
REQUIRED_FIELDS = ("sample_id", "schema_version", "mode", "color_ids")


class ColorGenContextStore(PublishedStore):
    """One published ``(split, mode)`` generation, read by ``sample_id``."""

    SUFFIX = COLOR_GENCTX_SUFFIX

    def __init__(self, root: str | Path, *, mode: str, verify: bool = True):
        if mode not in GENCTX_MODES:
            raise ValueError(f"unknown genctx mode {mode!r}; have {GENCTX_MODES}")
        super().__init__(root, verify=verify)
        self.mode = mode
        self._checked = False

    def record(self, sample_id: str) -> dict[str, Any]:
        rec = self.read_json(sample_id, self.SUFFIX)
        missing = [f for f in REQUIRED_FIELDS if f not in rec]
        if missing:
            raise KeyError(
                f"{sample_id}: generated-context record is missing {missing}.  "
                f"Stage-What needs schema {SCHEMA_COLOR_GENCTX} (the v2 genwhere "
                "record that keeps the <color> segment); a v1 record would give "
                "every sample an empty generated context."
            )
        if rec["schema_version"] != SCHEMA_COLOR_GENCTX:
            raise ValueError(
                f"{sample_id}: schema {rec['schema_version']!r} != "
                f"{SCHEMA_COLOR_GENCTX!r} (amendment A-4 reads the v2 record)"
            )
        if rec["mode"] != self.mode:
            raise ValueError(
                f"{sample_id}: record was generated in mode {rec['mode']!r} but "
                f"this arm requires {self.mode!r}.  C01/C02 must consume the "
                "forced-<color>-prefix generation; every other arm the "
                "with-<where>-prefix one (amendment A-4 item 4)."
            )
        return rec

    def color_ids(self, sample_id: str) -> list[int]:
        return [int(t) for t in self.record(sample_id)["color_ids"]]

    def assert_covers(self, sample_ids: Iterable[str]) -> dict[str, Any]:
        """Every sample must have a record -- there is no GT fallback (A-4).

        Called once at job start rather than per batch: discovering a coverage
        hole forty minutes into a run, on a sample the loop then cannot build, is
        the failure mode Where-B's ``run_where_b.py`` already refuses to allow.
        """
        ids = list(sample_ids)
        present = self.sample_ids
        missing = [s for s in ids if s not in present]
        if missing:
            raise RuntimeError(
                f"{len(missing)}/{len(ids)} samples have no generated <color> "
                f"context in {self.root} (mode={self.mode}); first few: "
                f"{missing[:5]}.  Amendment A-4 forbids falling back to the GT "
                "span, so the generation job must be completed for this split "
                "before any arm starts."
            )
        # Review N-23: existence is only one of the three contracts.  Reading one
        # record here moves the schema and mode checks from "the first batch,
        # minutes into the run" to "before anything expensive starts" -- and the
        # failure they catch ("the whole split is the wrong generation") is a
        # property of the directory, so one sample settles it.
        probe: dict[str, Any] | None = None
        if ids:
            rec = self.record(ids[0])
            probe = {"sample_id": ids[0], "schema_version": rec["schema_version"],
                     "mode": rec["mode"], "n_color_ids": len(rec["color_ids"])}
        self._checked = True
        return {"n_requested": len(ids), "n_present": len(ids), "mode": self.mode,
                "root": str(self.root), "coverage": 1.0, "probe": probe}

    def iter_records(self) -> Iterator[dict[str, Any]]:
        for sid, suf in sorted(self.rows):
            if suf == self.SUFFIX:
                yield self.read_json(sid, suf)

    def summary(self) -> dict[str, Any]:
        n = fail = trunc = 0
        reasons: dict[str, int] = {}
        for r in self.iter_records():
            n += 1
            fail += int(r.get("color_format_failure", False))
            trunc += int(r.get("color_truncated", False))
            k = str(r.get("color_stop_reason"))
            reasons[k] = reasons.get(k, 0) + 1
        return {
            "n": n, "mode": self.mode,
            "color_format_failure_rate": fail / n if n else None,
            "color_truncation_rate": trunc / n if n else None,
            "color_stop_reasons": dict(sorted(reasons.items())),
            "coverage_checked": self._checked,
            **self.facts(),
        }
