"""N-24 -- the ``<color>`` token boundary must be verified before an arm starts.

``gt_color_context`` *raises* when a GT ``<color>`` span exceeds
``COLOR_CONTEXT_MAX_TOKENS``.  That is the right behaviour -- the boundary is an
assertion about the corpus, and silently shortening a teacher context would be a
worse failure -- but it means **one over-long sample kills an arm mid-epoch**,
hours in, with the other eleven arms queued behind it.

The boundary itself was derived from 3,745 sampled records (max 324 against a
boundary of 384).  The full corpus is 169k records, so the sampled maximum is an
estimate, and the estimate is the thing standing between the campaign and a
mid-epoch crash.  The scan is a pure record read -- no images, no model -- and it
belongs *before* the first arm, not in the post-run job list where ``WT-J9``
originally sat.

Two pieces, deliberately separated:

* :func:`scan_split` / the ``scan_color_boundary`` script -- the (IO-heavy) scan,
  run once, publishing a report;
* :func:`require_color_boundary_scan` -- the (free) gate, called by every arm's
  runner, which refuses to start if the report is missing, stale in schema, or
  failing.
"""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any, Iterable, Sequence

from .config import COLOR_CONTEXT_MAX_TOKENS, REPORT_DIR

__all__ = ["SCHEMA", "DEFAULT_REPORT_PATH", "scan_split", "summarise_scan",
           "require_color_boundary_scan", "ColorBoundaryError"]

SCHEMA = "q3vl.what.color_boundary/1"
DEFAULT_REPORT_PATH = REPORT_DIR / "preflight" / "color_boundary_scan.json"
#: every split whose GT ``<color>`` span is ever tokenised as a teacher context
SCANNED_SPLITS = ("train", "V_where", "V_what", "T_final", "T_lut_unseen")


class ColorBoundaryError(RuntimeError):
    """Raised when the boundary scan is missing, unusable, or failing."""


def scan_split(dataset, *, split: str, boundary: int = COLOR_CONTEXT_MAX_TOKENS,
               progress_every: int = 20000) -> dict[str, Any]:
    """Max / percentiles of ``tokens.color`` over a whole split.

    Uses the record's own ``tokens.color`` count -- written by the sft2seg build
    with the same tokeniser the collator uses -- so the scan costs one record
    read per sample and no tokenisation.  Samples whose record predates that
    field are counted separately rather than skipped silently: an unmeasured
    sample is exactly the one that could be over the boundary.
    """
    lengths: list[int] = []
    over: list[dict[str, Any]] = []
    n_missing = 0
    t0 = time.time()
    for i in range(len(dataset)):
        rec = dataset.record(i)
        n = (rec.get("tokens") or {}).get("color")
        if n is None:
            n_missing += 1
            continue
        n = int(n)
        lengths.append(n)
        # the record counts the body; the span adds <color> and </color>
        if n + 2 > boundary:
            over.append({"sample_id": rec.get("sample_id"), "tokens_color": n})
        if progress_every and (i + 1) % progress_every == 0:
            print(json.dumps({"split": split, "done": i + 1, "of": len(dataset),
                              "elapsed_s": round(time.time() - t0, 1)}), flush=True)
    return summarise_scan(split, lengths, over, n_missing, boundary,
                          round(time.time() - t0, 1))


def summarise_scan(split: str, lengths: Sequence[int], over: Sequence[dict[str, Any]],
                   n_missing: int, boundary: int, elapsed_s: float) -> dict[str, Any]:
    s = sorted(lengths)

    def q(p: float) -> int | None:
        return s[min(len(s) - 1, int(p * (len(s) - 1)))] if s else None

    return {
        "split": split, "n": len(s), "n_missing_tokens_field": n_missing,
        "boundary": boundary,
        "min": s[0] if s else None, "p50": q(0.50), "p95": q(0.95),
        "p99": q(0.99), "max": s[-1] if s else None,
        "max_span_with_tags": (s[-1] + 2) if s else None,
        "n_over_boundary": len(over),
        "over_boundary": list(over[:32]),
        "headroom": (boundary - (s[-1] + 2)) if s else None,
        "ok": (not over) and n_missing == 0 and bool(s),
        "elapsed_s": elapsed_s,
    }


def require_color_boundary_scan(path: str | Path | None = None,
                                boundary: int = COLOR_CONTEXT_MAX_TOKENS
                                ) -> dict[str, Any]:
    """The pre-run gate.  Raises unless a passing scan exists for every split.

    Deliberately strict about three separate things, because each of them has a
    different failure story and "the file was there" covers none of them:

    * **present** -- an arm that starts without the scan is an arm that might
      crash at hour four;
    * **for this boundary** -- a scan taken against a different
      ``COLOR_CONTEXT_MAX_TOKENS`` says nothing about the current one;
    * **complete** -- every split that is ever tokenised as teacher context, with
      no records missing the ``tokens.color`` field.
    """
    p = Path(path or DEFAULT_REPORT_PATH)
    if not p.exists():
        raise ColorBoundaryError(
            f"{p} does not exist.  N-24: the <color> boundary "
            f"({boundary}) is an assertion about the corpus and "
            "gt_color_context raises when a sample exceeds it, so one over-long "
            "record would kill this arm mid-epoch.  Run "
            "`python -m q3vl.what.scripts.scan_color_boundary` first."
        )
    try:
        report = json.loads(p.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ColorBoundaryError(f"{p} is unreadable: {exc}") from None
    if report.get("schema") != SCHEMA:
        raise ColorBoundaryError(
            f"{p}: schema {report.get('schema')!r} != {SCHEMA!r}; re-run the scan")
    if int(report.get("boundary", -1)) != int(boundary):
        raise ColorBoundaryError(
            f"{p}: scanned against boundary {report.get('boundary')}, but "
            f"COLOR_CONTEXT_MAX_TOKENS is now {boundary}.  A scan against a "
            "different boundary proves nothing about this one; re-run it."
        )
    splits = report.get("splits") or {}
    missing = [s for s in SCANNED_SPLITS if s not in splits]
    if missing:
        raise ColorBoundaryError(f"{p}: splits {missing} were not scanned")
    failing = {s: {k: v[k] for k in ("n_over_boundary", "max",
                                     "n_missing_tokens_field") if k in v}
               for s, v in splits.items() if not v.get("ok")}
    if failing:
        raise ColorBoundaryError(
            f"{p}: the <color> boundary scan does not pass: {failing}.  Either "
            "raise COLOR_CONTEXT_MAX_TOKENS (it is imported from "
            "q3vl.whereb.config, so both stages move together) and re-run the "
            "scan, or exclude the offending samples explicitly -- do not start "
            "an arm that will raise on them."
        )
    return {
        "path": str(p), "schema": report.get("schema"), "boundary": boundary,
        "generated_at": report.get("generated_at"),
        "splits": {s: {k: v.get(k) for k in
                       ("n", "max", "max_span_with_tags", "headroom", "ok")}
                   for s, v in splits.items()},
        "ok": True,
    }


def merge_report(per_split: Iterable[dict[str, Any]],
                 boundary: int = COLOR_CONTEXT_MAX_TOKENS) -> dict[str, Any]:
    splits = {r["split"]: r for r in per_split}
    return {
        "schema": SCHEMA,
        "boundary": boundary,
        "generated_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "splits": splits,
        "ok": all(v.get("ok") for v in splits.values()),
        "global_max": max((v["max"] for v in splits.values()
                           if v.get("max") is not None), default=None),
    }
