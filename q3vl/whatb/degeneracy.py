"""Degenerate-solution guard -- **alias module, no second implementation.**

The three assertions the task card asks for (the transform is flat across query
colours / is the identity / is the same for every sample, any one of them
exiting the process at the first quick eval with the numbers printed) are
implemented once, in :mod:`q3vl.whatb.guards`.  This module exists only so the
name ``q3vl.whatb.degeneracy`` resolves to that one implementation instead of
inviting a parallel copy; a second copy would drift on the thresholds, which are
the part that ends up in ``run_setup.json``.

Usage from an arm's first quick eval::

    from q3vl.whatb.degeneracy import assert_transform_not_degenerate
    report = assert_transform_not_degenerate(f_hat, x_queries,
                                             where="quick_eval@step%d" % step)
    run_setup["degeneracy"] = report.as_dict()      # thresholds travel with it
"""

from __future__ import annotations

from .guards import (
    DegeneracyCheckNotRun,
    DegeneracyReport,
    DegeneracyThresholds,
    DegenerateTransform,
    assert_transform_not_degenerate,
    clear_degeneracy_check,
    degeneracy_check_ran,
    measure_degeneracy,
    record_degeneracy_check,
)

__all__ = [
    "DegeneracyCheckNotRun",
    "DegeneracyReport",
    "DegeneracyThresholds",
    "DegenerateTransform",
    "assert_transform_not_degenerate",
    "clear_degeneracy_check",
    "degeneracy_check_ran",
    "measure_degeneracy",
    "record_degeneracy_check",
]
