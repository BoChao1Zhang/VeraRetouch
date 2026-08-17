"""One pass of the whole data+criteria chain on real LUTs from the real bank.

Query colours -> ``L_l(x)`` -> the data law -> the headline formation -> dE00 ->
the baseline columns -> the board -> ``assert_publishable``.  Skips when the LUT
bank is not on this host; nothing here needs a GPU.
"""

from __future__ import annotations

import numpy as np
import pytest
import torch

from q3vl.whatb import criteria as C
from q3vl.whatb import publish as P
from q3vl.whatb.lutdata import BANK_DIR, LutBank
from q3vl.whatb.queries import QuerySampler, uniform_grid

pytestmark = pytest.mark.skipif(
    not (BANK_DIR / "luts_meta.json").exists(), reason="LUT bank not on this host")


@pytest.fixture(scope="module")
def bank():
    return LutBank()


@pytest.fixture(scope="module")
def library(bank):
    ids = bank.lut_ids()[:24]
    return ids, C.LibraryValues.build(bank, ids, uniform_grid(9,
                                                              dtype=torch.float32))


def test_bank_domains_are_the_identity_normalisation(bank):
    for lid in bank.lut_ids()[:200]:
        e = bank.entry(lid)
        assert e.dmin == (0.0, 0.0, 0.0) and e.dmax == (1.0, 1.0, 1.0)


def test_query_colours_through_a_real_lut(bank):
    x = QuerySampler(seed=1).sample(4, 256)[0]
    lid = bank.lut_ids()[0]
    y = bank.apply(x, lid)
    assert y.shape == x.shape
    assert float(y.min()) >= 0.0 and float(y.max()) <= 1.0
    assert float((y - x).abs().max()) > 0.0        # a real LUT is not identity


def test_headline_chain_on_synthetic_images(bank, library):
    ids, lib = library
    g = torch.Generator().manual_seed(0)
    rows = []
    for i, lid in enumerate(ids[:8]):
        img = torch.rand(3, 24, 32, generator=g)
        alpha = torch.rand(24, 32, generator=g).round()      # style-ish 0/1 field
        i_star = bank.f_star_image(img, alpha, lid)

        # a perfect arm, and the two trivial predictions
        i_hat = C.compose_hat(img, alpha, bank.apply_image(img, lid))
        e_arm = float(C.image_delta_e00(i_hat, i_star))
        e_b0 = float(C.image_delta_e00(C.compose_hat(img, alpha, img), i_star))
        wrong = ids[(i + 5) % len(ids)]
        e_b2 = float(C.image_delta_e00(
            C.compose_hat(img, alpha, bank.apply_image(img, wrong)), i_star))

        rows.append({"sample_id": f"s{i}", "winner_confidence": "normal",
                     "task_type": "local", "E_arm": e_arm,
                     "E_B0_identity": e_b0, "E_B1_libmean": e_b2,
                     "E_B2_librandom_repeats": [e_b2] * 8,
                     "E_B3_bucket_retrieval_repeats": [e_b2] * 8,
                     "E_B4_oracle": e_arm,
                     "E_N1_shuffle": e_b2, "M_N1_shuffle": 1.0,
                     "E_N2_irrelevant": e_b2, "M_N2_irrelevant": 1.0,
                     "E_N3_const": e_b2, "M_N3_const": 1.0})

    # the perfect arm reproduces the dataset's own GT to float precision
    assert max(r["E_arm"] for r in rows) < 1e-4
    assert min(r["E_B0_identity"] for r in rows) > 0.0

    board = C.build_board(rows, arm="EPR-024", split="V_what")
    board["published"] = True
    cols = board["criteria_columns"]
    assert cols["headline_normal_only"]["n"] == 8
    assert cols["B0_identity"]["paired_delta_arm_minus_baseline"]["delta"] < 0

    rep = P.assert_publishable(board, "EPR-024",
                               steps_row={c: 0.0 for c in P.FIRST_STEP_COLUMNS},
                               axes=())
    assert rep["criteria"]["computed"]["B4_oracle"] == 8


def test_oracle_is_the_best_library_member(bank, library):
    ids, lib = library
    target = lib.values[7]
    picked = C.oracle_lut_ids(lib, {"t": target})["t"]
    assert picked[0] == ids[7] and picked[1] == pytest.approx(0.0, abs=1e-6)
    d = lib.distance_to(target)
    assert float(d.min()) == pytest.approx(0.0, abs=1e-6)
    assert int(torch.argmin(d)) == 7


def test_library_mean_is_a_valid_mapping(bank, library):
    ids, lib = library
    m = lib.mean_transform()
    assert m.shape == (729, 3)
    assert float(m.min()) >= 0.0 and float(m.max()) <= 1.0
