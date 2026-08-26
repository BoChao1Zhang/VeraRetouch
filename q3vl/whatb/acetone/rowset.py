"""``A_rows``: the row set this experiment scores is EPR-033's row set.

The reference board ``/home/bc/data/runs/what_b/whatb_EPR033_lora_spanpool_s2``
carries no per-sample ids, so the identity of the two row sets is established
constructively rather than by comparing id lists after the fact:

* EPR-033 built its rows as ``normal_only(load_index("V_what"))`` filtered by
  membership in its ``V_what__none`` z cache
  (``q3vl/whatb/scripts/run_carrier_arm.py:293-314``);
* it recorded the split's measured counts in ``run_setup.json``
  (``split_facts.V_what``) and the resulting board counts in ``metrics.json``
  (``n_rows`` / ``n_normal`` / ``n_low_excluded``).

So this module rebuilds the same two ingredients, asserts the rebuilt
``split_facts`` equals the recorded one field for field, asserts every rebuilt
row is in the same z cache (which makes the filter a no-op on both sides, hence
the sets equal), and asserts the resulting count equals the board's ``n_rows``.
Any drift -- a different口径 (``cut-p45`` gives 777 / 496 instead of 897 / 567),
a re-published index, a different z cache -- stops the run.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Sequence

from q3vl.whatb import splits as S
from q3vl.whatb.splits import IndexRow

__all__ = [
    "DATASET_VERSION",
    "EVAL_SPLIT",
    "REFERENCE_RUN",
    "REFERENCE_ZCACHE",
    "N_EXPECTED",
    "RowSetMismatch",
    "eval_rows",
    "assert_rows",
]

#: task card §3: v20260804, not cut-p45
DATASET_VERSION = "v20260804"
EVAL_SPLIT = "V_what"

#: the board every column of this experiment is pinned to
REFERENCE_RUN = Path("/home/bc/data/runs/what_b/whatb_EPR033_lora_spanpool_s2")

#: the z cache EPR-033 filtered its eval rows through (run_setup.json:z_caches)
REFERENCE_ZCACHE = Path(
    "/home/bc/data/caches/whatb_z_multi_20260822/generated/color_span_pool/"
    "V_what__none")

N_EXPECTED = 567


class RowSetMismatch(AssertionError):
    """The rebuilt row set is not EPR-033's row set."""


def eval_rows() -> list[IndexRow]:
    """``normal_only(load_index("V_what"))`` under the v20260804口径."""
    S.use_dataset_version(DATASET_VERSION, force=True)
    return S.normal_only(S.load_index(EVAL_SPLIT))


def _zcache_ids(path: Path = REFERENCE_ZCACHE) -> set[str] | None:
    idx = Path(path) / "index.jsonl"
    if not idx.is_file():
        return None
    out: set[str] = set()
    with idx.open("r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if line:
                out.add(str(json.loads(line)["sample_id"]))
    return out


def _reference() -> tuple[dict[str, Any], dict[str, Any]]:
    setup = json.loads((REFERENCE_RUN / "run_setup.json").read_text(encoding="utf-8"))
    board = json.loads((REFERENCE_RUN / "metrics.json").read_text(encoding="utf-8"))
    return setup, board


def assert_rows(rows: Sequence[IndexRow] | None = None) -> dict[str, Any]:
    """Run ``A_rows`` and return the report that goes on the board."""
    rows = list(rows) if rows is not None else eval_rows()
    setup, board = _reference()

    facts = S.split_facts(S.load_index(EVAL_SPLIT))
    want = dict(setup["split_facts"][EVAL_SPLIT])
    if facts != want:
        raise RowSetMismatch(
            f"A_rows: the rebuilt {EVAL_SPLIT} index measures {facts} and "
            f"EPR-033's run_setup.json recorded {want}.  The row set is not the "
            "reference board's row set; scoring would compare two populations.")

    n_board = int(board["n_rows"])
    if len(rows) != n_board or len(rows) != N_EXPECTED:
        raise RowSetMismatch(
            f"A_rows: rebuilt {len(rows)} normal rows, EPR-033's board carries "
            f"{n_board}, the task card pre-registers {N_EXPECTED}")

    zids = _zcache_ids()
    missing = sorted({r.sample_id for r in rows} - zids) if zids is not None else []
    if zids is not None and missing:
        raise RowSetMismatch(
            f"A_rows: {len(missing)} of the rebuilt rows are absent from "
            f"{REFERENCE_ZCACHE}, which EPR-033 filtered its eval rows through; "
            f"first five: {missing[:5]}")

    return {
        "dataset_version": DATASET_VERSION, "split": EVAL_SPLIT,
        "n_rows": len(rows), "n_expected": N_EXPECTED,
        "reference_run": str(REFERENCE_RUN),
        "reference_n_rows": n_board,
        "reference_n_low_excluded": int(board.get("n_low_excluded", 0)),
        "split_facts": facts,
        "zcache": str(REFERENCE_ZCACHE),
        "zcache_present": zids is not None,
        "zcache_n": (len(zids) if zids is not None else None),
        "n_rows_outside_zcache": len(missing),
        "passed": True,
    }
