"""The criteria layer: keys, board shape, paired statistics, banned-by-absence."""

from __future__ import annotations

import re
from pathlib import Path

import numpy as np
import pytest
import torch

from q3vl.whatb import criteria as C


# --------------------------------------------------------------------------- #
# key tables
# --------------------------------------------------------------------------- #
def test_the_twelve_pre_registered_keys_are_frozen():
    assert C.PREREGISTERED_KEYS == (
        "headline_normal_only",
        "B0_identity", "B1_libmean", "B2_librandom", "B3_bucket_retrieval",
        "B4_oracle",
        "N1_shuffle_delta", "N1_shuffle_M",
        "N2_irrelevant_delta", "N2_irrelevant_M",
        "N3_const_delta", "N3_const_M")
    assert len(C.PREREGISTERED_KEYS) == 12
    assert C.REQUIRED_COMMON == frozenset(C.PREREGISTERED_KEYS)


def test_required_tables_per_arm():
    p1 = set(C.required_criteria("EPR-024"))
    assert p1 == set(C.PREREGISTERED_KEYS) | {"interp_grid", "path_len",
                                              "mono_rate", "oob_rate"}
    assert set(C.required_criteria("EPR-028")) == \
        set(C.PREREGISTERED_KEYS) | set(C.REQUIRED_P2P3)
    # EPR-027 spans P1 and P2: union, not either
    assert set(C.required_criteria("EPR-027")) == \
        set(C.PREREGISTERED_KEYS) | set(C.REQUIRED_P1) | set(C.REQUIRED_P2P3)
    with pytest.raises(KeyError, match="ARM_AXES"):
        C.required_criteria("EPR-999")
    assert set(C.required_criteria("EPR-999", axes=("P1",))) == p1


def test_banned_columns_are_not_implemented():
    """§4.I is enforced by absence: no function may exist with these names."""
    src = Path(C.__file__).read_text()
    defs = re.findall(r"^\s*def\s+(\w+)", src, re.MULTILINE)
    banned = ("auc", "roc", "minmax", "min_max", "softmax_norm", "trim", "iou")
    assert not [d for d in defs if any(b in d.lower() for b in banned)], defs
    assert "headline_pooled" not in src


def test_no_cpu_round_trip_in_the_metric_path():
    """Criteria are computed where the tensor already is (the top-k lesson).

    Parsed, not grepped: the prose in these modules talks about ``.cpu()``
    precisely because it must not be called.
    """
    import ast

    for mod in ("criteria", "colorimetry", "lutdata"):
        path = Path(C.__file__).with_name(f"{mod}.py")
        for node in ast.walk(ast.parse(path.read_text())):
            if not isinstance(node, ast.Call):
                continue
            fn = node.func
            if isinstance(fn, ast.Attribute):
                assert fn.attr != "cpu", f"{mod}: .cpu() in the metric path"
                if fn.attr == "to" and node.args:
                    a = node.args[0]
                    assert not (isinstance(a, ast.Constant) and a.value == "cpu"), \
                        f"{mod}: .to('cpu') in the metric path"


# --------------------------------------------------------------------------- #
# per-sample primitives
# --------------------------------------------------------------------------- #
def test_compose_hat_is_the_frozen_formation():
    img = torch.rand(3, 8, 8)
    f = torch.rand(3, 8, 8)
    a = torch.rand(1, 8, 8)
    got = C.compose_hat(img, a, f)
    assert torch.allclose(got, img * (1 - a) + f * a, atol=1e-7)
    assert torch.equal(C.compose_hat(img, 0.0, f), img)
    assert torch.equal(C.compose_hat(img, 1.0, f), f)


def test_image_delta_e00_is_zero_on_identity_and_positive_otherwise():
    img = torch.rand(3, 5, 6, dtype=torch.float64)
    assert float(C.image_delta_e00(img, img)) == 0.0
    assert float(C.image_delta_e00(img, img * 0.5)) > 0.0
    assert C.image_delta_e00(img, img).dim() == 0


def test_function_distance_weighted_and_uniform():
    a = torch.rand(64, 3, dtype=torch.float64)
    b = torch.rand(64, 3, dtype=torch.float64)
    uni = C.function_distance(a, b)
    w = torch.zeros(64, dtype=torch.float64)
    w[3] = 1.0
    one = C.function_distance(a, b, w)
    from q3vl.whatb.colorimetry import delta_e00_srgb

    assert abs(float(one) - float(delta_e00_srgb(a[3], b[3]))) < 1e-9
    assert float(uni) > 0.0
    assert float(C.function_distance(a, b, metric="de76")) > 0.0


def test_locality_strata_and_out_of_mask_is_zero_for_mask_blend():
    img = torch.rand(3, 16, 16)
    f = torch.rand(3, 16, 16)
    alpha = torch.zeros(16, 16)
    alpha[:4] = 1.0                       # inside
    alpha[4:8] = 0.5                      # band
    i_hat = C.compose_hat(img, alpha, f)
    i_star = C.compose_hat(img, alpha, f)          # perfect arm
    out = C.locality_errors(i_hat, i_star, img, alpha)
    assert out["loc_in_n_pixels"] == 4 * 16
    assert out["loc_band_n_pixels"] == 4 * 16
    assert out["loc_out_n_pixels"] == 8 * 16
    assert out["loc_in"] == pytest.approx(0.0, abs=1e-6)
    # F1 mask blend: outside the mask the output IS the input, by construction
    assert out["loc_out"] == pytest.approx(0.0, abs=1e-6)


# --------------------------------------------------------------------------- #
# statistics
# --------------------------------------------------------------------------- #
def test_paired_stats_reports_delta_ci_and_wilcoxon():
    rng = np.random.default_rng(0)
    b = rng.normal(10.0, 1.0, 200)
    a = b - 0.5
    st = C.paired_stats(list(a), list(b), n_boot=2000)
    assert st["n"] == 200
    assert st["delta"] == pytest.approx(-0.5, abs=1e-9)
    assert st["ci95"][0] < -0.5 < st["ci95"][1] or st["ci95"][1] < 0
    assert st["p_wilcoxon"] < 1e-6
    assert "wilcoxon" in st["test"]


def test_paired_stats_drops_unpaired_and_survives_all_zero():
    st = C.paired_stats([1.0, None, 3.0], [1.0, 2.0, 3.0], n_boot=100)
    assert st["n"] == 2 and st["n_dropped"] == 1
    assert st["delta"] == 0.0 and st["p_wilcoxon"] == 1.0
    assert C.paired_stats([], [], n_boot=10)["n"] == 0


def test_bootstrap_does_not_touch_the_global_numpy_stream():
    np.random.seed(0)
    first = np.random.rand(3)
    np.random.seed(0)
    C.paired_stats([1.0, 2.0, 3.0], [0.0, 1.0, 2.0], n_boot=1000)
    assert np.array_equal(first, np.random.rand(3))


def test_describe_handles_empty_and_none():
    assert C.describe([])["n"] == 0
    d = C.describe([1.0, None, 3.0])
    assert d["n"] == 2 and d["mean"] == pytest.approx(2.0)


# --------------------------------------------------------------------------- #
# retrieval baselines
# --------------------------------------------------------------------------- #
class _StubBank:
    """A tiny library: LUT k multiplies the image by (k+1)/n."""

    def __init__(self, n=6):
        self.n = n

    def apply(self, x, lut_id):
        k = int(lut_id.split("_")[1])
        return (x * (k + 1) / self.n).clamp(0, 1)

    def evaluate_library(self, lut_ids, x):
        return torch.stack([self.apply(x, l) for l in lut_ids])


def test_library_values_mean_and_oracle():
    bank = _StubBank()
    ids = [f"lut_{k}" for k in range(6)]
    x = torch.rand(64, 3, dtype=torch.float64)
    lib = C.LibraryValues.build(bank, ids, x)
    assert lib.values.shape == (6, 64, 3)
    assert torch.allclose(lib.mean_transform(),
                          torch.stack([bank.apply(x, i) for i in ids]).mean(0))
    target = bank.apply(x, "lut_3")
    got = C.oracle_lut_ids(lib, {"lut_3": target})
    assert got["lut_3"][0] == "lut_3" and got["lut_3"][1] == pytest.approx(0.0)
    # B6: the nearest OTHER library LUT is a neighbour, and strictly worse
    other = C.oracle_lut_ids(lib, {"lut_3": target}, exclude_self=True)
    assert other["lut_3"][0] in ("lut_2", "lut_4") and other["lut_3"][1] > 0.0


def test_bucket_draw_respects_the_bucket_and_flags_empty_pools():
    pools = {"warm_01": ["a", "b"], "cool_02": ["c"]}
    rows = C.bucket_draw(["warm_01", "cool_02", "missing"], pools, repeats=4)
    assert len(rows) == 4 and all(len(r) == 3 for r in rows)
    assert all(r[0] in ("a", "b") for r in rows)
    assert all(r[1] == "c" for r in rows)
    assert all(r[2] is None for r in rows)          # counted, never substituted


def test_library_random_draw_is_reproducible():
    ids = list("abcdef")
    a = C.library_random_draw(ids, 5, repeats=3, seed=1)
    b = C.library_random_draw(ids, 5, repeats=3, seed=1)
    assert a == b and len(a) == 3 and len(a[0]) == 5


# --------------------------------------------------------------------------- #
# the board
# --------------------------------------------------------------------------- #
def _rows(n=12, with_all=True):
    rng = np.random.default_rng(3)
    rows = []
    for i in range(n):
        conf = "normal" if i % 4 else "low"
        row = {"sample_id": f"s{i}", "winner_confidence": conf,
               "task_type": "style" if i % 2 else "local",
               "E_arm": float(4 + rng.normal(0, 0.2))}
        if with_all:
            row.update({
                "E_B0_identity": float(30 + rng.normal()),
                "E_B1_libmean": float(25 + rng.normal()),
                "E_B2_librandom_repeats": [float(35 + rng.normal())
                                           for _ in range(8)],
                "E_B3_bucket_retrieval_repeats": [float(20 + rng.normal())
                                                  for _ in range(8)],
                "E_B4_oracle": float(9 + rng.normal()),
                "E_B6_libfill": float(10 + rng.normal()),
                "E_N1_shuffle": float(6 + rng.normal(0, 0.2)),
                "M_N1_shuffle": float(3 + rng.normal(0, 0.2)),
                "E_N2_irrelevant": float(7 + rng.normal(0, 0.2)),
                "M_N2_irrelevant": float(4 + rng.normal(0, 0.2)),
                "E_N3_const": float(8 + rng.normal(0, 0.2)),
                "M_N3_const": float(5 + rng.normal(0, 0.2)),
                "grid_error": float(3 + rng.normal(0, 0.1)),
                "img_error": float(2 + rng.normal(0, 0.1)),
            })
        rows.append(row)
    return rows


def test_board_is_normal_only_and_splits_by_task_type():
    rows = _rows()
    board = C.build_board(rows, arm="EPR-024", split="V_what")
    assert board["n_rows"] == 12 and board["n_normal"] == 9
    assert board["n_low_excluded"] == 3
    ctx = board["contexts"]
    assert ctx["all"]["headline_normal_only"]["n"] == 9
    assert (ctx["style"]["headline_normal_only"]["n"]
            + ctx["local"]["headline_normal_only"]["n"] == 9)
    # no pooled headline anywhere on the board
    assert "headline" not in ctx["all"]
    assert all("pooled" not in k for k in board["criteria_columns"])


def test_board_columns_carry_n_and_paired_deltas():
    board = C.build_board(_rows(), arm="EPR-024", split="V_what")
    cols = board["criteria_columns"]
    for key in C.PREREGISTERED_KEYS:
        assert key in cols, key
        assert cols[key]["n"] > 0, key
    assert cols["B0_identity"]["paired_delta_arm_minus_baseline"]["delta"] < 0
    assert cols["B2_librandom"]["n_repeats"] == 8
    assert cols["B3_bucket_retrieval"]["repeat_std"] >= 0.0
    assert cols["N1_shuffle_delta"]["delta"] > 0          # control is worse
    assert cols["N1_shuffle_M"]["mean"] > 0


def test_assert_criteria_ran_fails_on_a_missing_column():
    board = C.build_board(_rows(), arm="EPR-024", split="V_what")
    rep = C.assert_criteria_ran(board, "EPR-024", axes=())
    assert rep["headline_normal_only_n"] == 9
    with pytest.raises(C.CriterionNotComputed, match="interp_grid"):
        C.assert_criteria_ran(board, "EPR-024")           # P1 keys not wired


def test_assert_criteria_ran_accepts_registered_extra_columns():
    board = C.build_board(
        _rows(), arm="EPR-024", split="V_what",
        extra_columns={"interp_grid": {"n": 120, "mean": 3.2},
                       "path_len": {"n": 120, "mean": 9.1},
                       "mono_rate": {"n": 120, "mean": 0.8,
                                     "random_floor": 0.5},
                       "oob_rate": {"n": 120, "mean": 0.01}})
    rep = C.assert_criteria_ran(board, "EPR-024")
    assert rep["computed"]["path_len"] == 120


def test_extra_column_without_n_is_refused():
    with pytest.raises(ValueError, match="carries no 'n'"):
        C.build_board(_rows(), arm="EPR-024", split="V_what",
                      extra_columns={"interp_grid": {"mean": 1.0}})


def test_board_without_normal_rows_cannot_be_asserted():
    rows = [dict(r, winner_confidence="low") for r in _rows()]
    board = C.build_board(rows, arm="EPR-024", split="V_what")
    with pytest.raises(C.CriterionNotComputed, match="headline_normal_only"):
        C.assert_criteria_ran(board, "EPR-024", axes=())


# --------------------------------------------------------------------------- #
# path quantities (§4.F-B)
# --------------------------------------------------------------------------- #
def test_path_quantities_on_a_straight_interpolation():
    x = torch.rand(200, 3, dtype=torch.float64)
    a = x
    b = (x * 0.5 + 0.25)
    alphas = torch.linspace(0, 1, 21, dtype=torch.float64)
    path = torch.stack([(1 - t) * a + t * b for t in alphas])
    q = C.path_quantities(path)
    assert q["k_steps"] == 20
    assert q["rho"] == pytest.approx(1.0, abs=0.05)        # near-straight path
    assert q["sigma_bar"] < 0.15
    assert q["oob_rate"] == 0.0
    assert q["mono_rate_random_floor"] == 0.5
    assert 0.0 <= q["mono_rate"] <= 1.0


def test_path_quantities_flag_a_collapsed_path():
    x = torch.rand(50, 3, dtype=torch.float64)
    path = x[None].repeat(11, 1, 1)
    q = C.path_quantities(path)
    assert q["path_len"] == 0.0 and q["chord"] == 0.0
    assert q["rho"] is None                    # undefined for the collapsed solution
    assert q["jump_max"] == 0.0


def test_jump_max_is_not_percentile_trimmed():
    x = torch.rand(50, 3, dtype=torch.float64)
    path = torch.stack([x, x, x, torch.rand(50, 3, dtype=torch.float64), x])
    q = C.path_quantities(path)
    assert q["jump_max"] > 0.0
    assert "no percentile trimming" in q["note"]
