from __future__ import annotations

import numpy as np

from dataset_build.tools.evaluate_mask_feather import build_candidates, feather_binary


def _subject_mask() -> np.ndarray:
    yy, xx = np.mgrid[0:180, 0:240]
    return ((xx - 118) / 42.0) ** 2 + ((yy - 92) / 65.0) ** 2 <= 1.0


def test_strict_feather_has_exact_zero_exterior() -> None:
    hard = _subject_mask()
    alpha = feather_binary(hard, feather_in=0.02, feather_out=0.0)

    assert np.all(alpha[~hard] == 0.0)
    assert alpha[hard].max() == 1.0
    assert np.any((alpha[hard] > 0.0) & (alpha[hard] < 1.0))


def test_bounded_feather_is_nonzero_only_near_boundary() -> None:
    hard = _subject_mask()
    alpha = feather_binary(hard, feather_in=0.02, feather_out=0.01)

    assert alpha[~hard].max() > 0.0
    assert alpha[0, 0] == 0.0
    assert alpha[-1, -1] == 0.0


def test_candidate_allocation_is_one_one_three_three() -> None:
    candidates = build_candidates(_subject_mask())
    counts = {
        kind: sum(candidate.kind == kind for candidate in candidates)
        for kind in ("semantic", "radial", "band", "linear")
    }

    assert len(candidates) == 8
    assert counts == {"semantic": 1, "radial": 1, "band": 3, "linear": 3}
    assert candidates[0].feather.feather_out == 0.0
    assert any(candidate.feather.feather_out > 0.0 for candidate in candidates)
