"""Scoring an external prediction with the campaign's own criteria.

Nothing in this module belongs to AceTone.  A predicted LUT arrives as a plain
``(D, D, D, 3)`` array stored ``grid[b, g, r]`` (:mod:`q3vl.whatb.acetone.bridge`
is the only place that knows how it got there) and is turned into board columns
by exactly the functions every other arm uses:

* image formation -- ``q3vl.whatb.criteria.compose_hat`` (=
  ``q3vl.whatb.lutdata.mix_alpha``), ``Î = (1-a) I + a f̂(I)``;
* headline -- ``q3vl.whatb.criteria.image_delta_e00``, mean ΔE00 against
  ``I* = bank.f_star_image(I, a, lut_id)``;
* function column -- ``q3vl.whatb.criteria.function_distance`` on the 17³ grid;
* baselines -- ``B0`` / ``B3`` / ``B4`` / ``B6`` built the way
  ``q3vl/whatb/arms/carrier.py:evaluate_samples`` builds them, from the same
  ``Lib_tr`` draw (``run_carrier_arm._library_ids``, 1137 ids, seed 20260810),
  the same 9³ / ΔE76 selection protocol and the same 8 bucket draws;
* board + gate -- ``criteria.build_board`` / ``publish.assert_publishable``.

``AceTone``'s own ``delta_e_between_images`` is never called.

The two pre-registered guards this module owns:

``A_baseline``  the four baseline means must equal EPR-033's board to 1e-4.
``A_finite``    every non-finite predicted grid value and every non-finite
                per-sample scalar is counted and lands on the board; a column
                with a non-finite value is never silently reduced to 0.0.
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import torch

from q3vl.whatb import criteria as C
from q3vl.whatb import splits as S
from q3vl.whatb.evaldata import SampleStore
from q3vl.whatb.lutdata import LutBank, apply_lut_volume
from q3vl.whatb.queries import uniform_grid
from q3vl.whatb.scripts import run_carrier_arm as R

from .bridge import grid_to_volume
from .rowset import REFERENCE_RUN, assert_rows, eval_rows

__all__ = [
    "SEED",
    "LIB_SIZE",
    "N_REPEATS",
    "REFERENCE_BASELINES",
    "BaselineMismatch",
    "Fixtures",
    "build_fixtures",
    "score_rows",
    "assert_baselines",
    "finite_report",
]

#: ``cfg.seed`` of the reference run (run_setup.json:config.seed)
SEED = 20260810
#: ``cfg.lib_size`` / ``cfg.n_repeats`` of the reference run
LIB_SIZE = 1137
N_REPEATS = 8

#: the four columns the task card pins, read off EPR-033's own metrics.json
REFERENCE_BASELINES: dict[str, float] = {
    "B0_identity": 8.2926,
    "B3_bucket_retrieval": 6.1553,
    "B4_oracle": 0.8253,
    "B6_libfill": 4.2345,
}


class BaselineMismatch(AssertionError):
    """A baseline column does not reproduce the reference board."""


@dataclass
class Fixtures:
    """Everything the baselines and the headline need, built once."""

    rows: list[Any]
    store: SampleStore
    bank: LutBank
    lib: C.LibraryValues
    lib_ids: list[str]
    pools: dict[str, list[str]]
    minors: dict[str, Any]
    grid9: torch.Tensor
    grid17: torch.Tensor
    device: torch.device
    a_rows: dict[str, Any] = field(default_factory=dict)

    def facts(self) -> dict[str, Any]:
        return {"seed": SEED, "lib_size": LIB_SIZE, "n_repeats": N_REPEATS,
                "n_lut": len(self.lib_ids), "n_buckets": len(self.pools),
                "grid_floor": 9, "grid_headline": 17,
                "device": str(self.device),
                "bank": self.bank.facts()}


def build_fixtures(*, device: Any = "cpu", limit: int | None = None) -> Fixtures:
    """The reference run's evaluation fixtures, rebuilt.

    ``limit`` is for smoke runs only; a limited fixture set can never satisfy
    ``A_rows`` and the caller has to say so on the artefact.
    """
    dev = torch.device(device)
    a_rows = assert_rows()
    rows = eval_rows()
    if limit:
        rows = rows[:limit]

    train_index = list(S.load_index_cached("train"))
    lib_ids = R._library_ids(train_index, LIB_SIZE, SEED)
    bank = LutBank()
    grid9 = uniform_grid(9, device=dev)
    grid17 = uniform_grid(17, device=dev)
    lib = C.LibraryValues.build(bank, lib_ids, grid9)
    pools = S.bucket_pools(S.iter_records([r for r in train_index if r.has_record]))
    minors = R.record_fields(rows)
    store = SampleStore("V_what")
    return Fixtures(rows=rows, store=store, bank=bank, lib=lib, lib_ids=lib_ids,
                    pools=pools, minors=minors, grid9=grid9, grid17=grid17,
                    device=dev, a_rows=a_rows)


def _headline_error(img, alpha, f_img, i_star) -> float:
    return float(C.image_delta_e00(C.compose_hat(img, alpha, f_img), i_star))


def _apply_grid(grid: np.ndarray, img: torch.Tensor) -> torch.Tensor:
    """``f̂(I)`` for a ``(3,H,W)`` image and a predicted ``grid[b,g,r]`` LUT."""
    vol = grid_to_volume(grid).to(device=img.device)
    return apply_lut_volume(vol, img.permute(1, 2, 0)).permute(2, 0, 1)


def _apply_grid_points(grid: np.ndarray, x: torch.Tensor) -> torch.Tensor:
    vol = grid_to_volume(grid).to(device=x.device)
    return apply_lut_volume(vol, x)


@torch.no_grad()
def score_rows(fx: Fixtures, preds: Mapping[str, np.ndarray], *,
               extra_preds: Mapping[str, Mapping[str, np.ndarray]] | None = None,
               control_preds: Mapping[str, Mapping[str, np.ndarray]] | None = None,
               with_baselines: bool = True,
               progress_every: int = 25) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """One row per sample + the ``A_finite`` report.

    ``preds`` is the headline prediction, ``sample_id -> (D,D,D,3)``.
    ``extra_preds`` are further LUT sets scored on the same formation and
    published as their own columns (row C uses ``resample_only``).
    ``control_preds`` are negative controls: each gets an ``E_<name>`` /
    ``M_<name>`` pair, paired against the headline by the board builder.
    """
    extra_preds = dict(extra_preds or {})
    control_preds = dict(control_preds or {})
    dev = fx.device
    rows = fx.rows
    ids = [r.sample_id for r in rows]

    draws_b3 = C.bucket_draw([(fx.minors.get(s) or {}).get("minor") or ""
                              for s in ids], fx.pools,
                             repeats=N_REPEATS, seed=SEED + 1) \
        if with_baselines else None

    if with_baselines:
        targets9 = {r.sample_id: fx.bank.apply(fx.grid9, r.lut_id) for r in rows}
        oracle = C.oracle_lut_ids(fx.lib, targets9, metric="de76", exclude_self=False)
        fill = C.oracle_lut_ids(fx.lib,
                                {r.lut_id: targets9[r.sample_id] for r in rows},
                                metric="de76", exclude_self=True)
    else:
        oracle = fill = {}

    finite: dict[str, Any] = {"n_pred_grids": 0, "n_pred_grids_nonfinite": 0,
                              "n_pred_values_nonfinite": 0,
                              "nonfinite_sample_ids": [],
                              "n_scalar_nonfinite": 0,
                              "scalar_nonfinite_keys": {}}
    out: list[dict[str, Any]] = []
    for i, r in enumerate(rows):
        if progress_every and i % progress_every == 0:
            print(f"[score] {i}/{len(rows)}", flush=True)
        grid = np.asarray(preds[r.sample_id], dtype=np.float32)
        finite["n_pred_grids"] += 1
        n_bad = int((~np.isfinite(grid)).sum())
        if n_bad:
            finite["n_pred_grids_nonfinite"] += 1
            finite["n_pred_values_nonfinite"] += n_bad
            finite["nonfinite_sample_ids"].append(r.sample_id)

        img, alpha = fx.store.load(r, device=dev)
        if isinstance(alpha, torch.Tensor):
            alpha = alpha.to(device=dev)
        i_star = fx.bank.f_star_image(img, alpha, r.lut_id)
        f_arm = _apply_grid(grid, img)
        i_hat = C.compose_hat(img, alpha, f_arm)

        row: dict[str, Any] = {
            "sample_id": r.sample_id, "winner_confidence": r.winner_confidence,
            "task_type": r.task_type, "lut_id": r.lut_id,
            "source_image_id": r.source_image_id,
            "minor": (fx.minors.get(r.sample_id) or {}).get("minor"),
            "lut_size": fx.bank.size(r.lut_id),
            "E_arm": float(C.image_delta_e00(i_hat, i_star)),
        }
        y17 = fx.bank.apply(fx.grid17, r.lut_id)
        f17 = _apply_grid_points(grid, fx.grid17)
        row["grid_error"] = float(C.function_distance(f17, y17))

        if with_baselines:
            row["E_B0_identity"] = float(C.image_delta_e00(img, i_star))
            row["E_B4_oracle"] = _headline_error(
                img, alpha, fx.bank.apply_image(img, oracle[r.sample_id][0]), i_star)
            row["E_B6_libfill"] = _headline_error(
                img, alpha, fx.bank.apply_image(img, fill[r.lut_id][0]), i_star)
            row["B4_lut_id"] = oracle[r.sample_id][0]
            row["B6_lut_id"] = fill[r.lut_id][0]
            assert draws_b3 is not None
            vals = [_headline_error(img, alpha, fx.bank.apply_image(img, lid), i_star)
                    for k in range(N_REPEATS)
                    if (lid := draws_b3[k][i]) is not None]
            row["E_B3_bucket_retrieval_repeats"] = vals or None
            row["B3_bucket_missing"] = int(N_REPEATS - len(vals))

        for name, table in extra_preds.items():
            g = np.asarray(table[r.sample_id], dtype=np.float32)
            row[name] = _headline_error(img, alpha, _apply_grid(g, img), i_star)
            row[f"{name}_grid_error"] = float(
                C.function_distance(_apply_grid_points(g, fx.grid17), y17))

        for name, table in control_preds.items():
            g = np.asarray(table[r.sample_id], dtype=np.float32)
            row[f"E_{name}"] = _headline_error(img, alpha, _apply_grid(g, img), i_star)
            row[f"M_{name}"] = float(
                C.function_distance(f17, _apply_grid_points(g, fx.grid17)))

        for key, value in list(row.items()):
            if isinstance(value, float) and not math.isfinite(value):
                finite["n_scalar_nonfinite"] += 1
                finite["scalar_nonfinite_keys"][key] = \
                    finite["scalar_nonfinite_keys"].get(key, 0) + 1
        out.append(row)

    finite["all_finite"] = (finite["n_pred_grids_nonfinite"] == 0
                            and finite["n_scalar_nonfinite"] == 0)
    return out, finite


def finite_report(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    """``A_finite`` over an already-built row list (board-side re-check)."""
    bad: dict[str, int] = {}
    for r in rows:
        for k, v in r.items():
            if isinstance(v, float) and not math.isfinite(v):
                bad[k] = bad.get(k, 0) + 1
            elif isinstance(v, list):
                for x in v:
                    if isinstance(x, float) and not math.isfinite(x):
                        bad[k] = bad.get(k, 0) + 1
    return {"n_rows": len(rows), "n_nonfinite": sum(bad.values()),
            "keys": dict(sorted(bad.items())), "all_finite": not bad}


def assert_baselines(board: Mapping[str, Any], *, tol: float = 1e-4
                     ) -> dict[str, Any]:
    """``A_baseline``: the four columns must equal the reference board.

    The comparison is against the reference ``metrics.json`` itself (full
    precision) and, separately, against the task card's four rounded literals --
    a board that agrees with one and not the other is still a stop.
    """
    ref_path = REFERENCE_RUN / "metrics.json"
    ref = json.loads(ref_path.read_text(encoding="utf-8"))["criteria_columns"]
    cols = board["criteria_columns"]
    report: dict[str, Any] = {"reference": str(ref_path), "tol": tol,
                              "columns": {}}
    bad: list[str] = []
    for key, card in REFERENCE_BASELINES.items():
        got = cols.get(key, {}).get("mean")
        want = ref.get(key, {}).get("mean")
        d_ref = None if (got is None or want is None) else abs(float(got) - float(want))
        d_card = None if got is None else abs(round(float(got), 4) - card)
        report["columns"][key] = {"got": got, "reference": want,
                                  "task_card": card,
                                  "abs_delta_vs_reference": d_ref,
                                  "abs_delta_vs_task_card": d_card}
        if d_ref is None or d_ref > tol or d_card is None or d_card > 5e-5:
            bad.append(key)
    if bad:
        raise BaselineMismatch(
            "A_baseline: " + ", ".join(
                f"{k}: got {report['columns'][k]['got']!r}, EPR-033 has "
                f"{report['columns'][k]['reference']!r}" for k in bad)
            + ".  The row set or the口径 differs from the reference board; every "
              "cross-arm delta read off this board would be void.")
    report["passed"] = True
    return report
