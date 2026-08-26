"""Rows -> board -> ``metrics.json``, for both AceTone rows.

The board is built by ``q3vl.whatb.criteria.build_board`` and gated by
``q3vl.whatb.publish.assert_publishable`` -- the same two calls every arm makes.
Two of that gate's four checks do not apply to an external baseline and are
turned off *explicitly*, with the reason recorded on the artefact:

``eval_only=True``               there is no training side and no ``steps.jsonl``;
``require_degeneracy_check``     the degenerate-solution guard watches a *trained*
                                 transform drift to a constant.  The predictions
                                 here are produced by frozen external weights in
                                 another process, so the witness cannot fire in
                                 this one.  The board carries the guard's own
                                 substitute instead: the standard deviation of
                                 the predicted 17³ function values across
                                 samples, next to the identity distance.

``required`` is the four baselines + the headline; the twelve-key table belongs
to the six trained arms and an external row does not pre-register N1/N2/N3.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

from q3vl.whatb import criteria as C
from q3vl.whatb import publish as P

from . import scoring as SC
from .bridge import repo_facts

__all__ = ["REQUIRED", "build_and_publish"]

#: what this board must carry before it may be written
REQUIRED: tuple[str, ...] = ("headline_normal_only", "B0_identity",
                             "B3_bucket_retrieval", "B4_oracle", "B6_libfill")


def _column(values: Sequence[float | None], quantity: str) -> dict[str, Any]:
    return {**C.describe(values), "quantity": quantity}


def build_and_publish(rows: Sequence[Mapping[str, Any]], *, arm: str,
                      run_dir: Path | str,
                      extra_columns: Mapping[str, Mapping[str, Any]] | None = None,
                      facts: Mapping[str, Any] | None = None,
                      seed: int = SC.SEED) -> dict[str, Any]:
    """Write ``<run_dir>/metrics.json`` after the four assertions have run."""
    run_dir = Path(run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)

    cols: dict[str, dict[str, Any]] = dict(extra_columns or {})
    board = C.build_board(rows, arm=arm, split="V_what",
                          extra_columns=cols, seed=seed)
    board["published"] = True
    board["arm_name"] = arm

    a_baseline = SC.assert_baselines(board)
    a_finite = SC.finite_report(rows)
    report = P.assert_publishable(
        board, arm, eval_only=True, require_degeneracy_check=False,
        required=list(REQUIRED))

    board["facts"] = {
        **dict(facts or {}),
        "acetone": repo_facts(),
        "assertions": {"A_baseline": a_baseline, "A_finite": a_finite,
                       **{k: v for k, v in dict(facts or {}).items()
                          if k.startswith("A_")}},
        "publication_exemptions": {
            "eval_only": "external frozen weights; this run takes no gradient step",
            "require_degeneracy_check": (
                "the guard's witness is per-process and the predictions come "
                "from another process; the transform's spread across samples is "
                "reported as pred_grid_std instead"),
        },
    }
    board["publication"] = report
    (run_dir / "metrics.json").write_text(json.dumps(board, indent=2, default=str),
                                          encoding="utf-8")
    (run_dir / "rows.jsonl").write_text(
        "".join(json.dumps(r, default=str) + "\n" for r in rows), encoding="utf-8")
    return board


def spread_columns(rows: Sequence[Mapping[str, Any]], key: str = "grid_error"
                   ) -> dict[str, Any]:
    """A constant-prediction witness that does not need the in-process guard."""
    vals = np.asarray([float(r[key]) for r in rows if r.get(key) is not None],
                      dtype=np.float64)
    return {"n": int(vals.size), "mean": float(vals.mean()) if vals.size else None,
            "std": float(vals.std(ddof=1)) if vals.size > 1 else 0.0,
            "quantity": f"spread of {key} across samples"}
