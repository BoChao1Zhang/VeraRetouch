"""EPR-034: the external baseline's contract with the campaign's口径.

Four things are pinned here, all of them things that have silently drifted in
this campaign before:

1. **the axis order** -- the LUT array this repository hands the tokenizer is
   the array AceTone's own ``.cube`` reader produces, byte for byte, and this
   repository's applier evaluates it as the analytic identity;
2. **the prompt and the generation kwargs** -- read straight out of
   ``eval/predict_lut_ddp.py`` with ``ast`` and compared to the constants in
   :mod:`q3vl.whatb.acetone.bridge`, so a re-worded prompt cannot pass;
3. **the guards fire** -- ``A_baseline`` raises on a board that does not
   reproduce EPR-033 and ``A_finite`` counts a NaN instead of dropping it;
4. **the banned list** -- no AUC anywhere in the package, and the scoring path
   never imports AceTone's own metrics.
"""

from __future__ import annotations

import ast
import json
import math
from pathlib import Path

import numpy as np
import pytest
import torch

from q3vl.whatb.acetone import bridge

PACKAGE = Path(bridge.__file__).resolve().parent
EVAL_SCRIPT = bridge.ACETONE_REPO / "eval" / "predict_lut_ddp.py"

repo_available = pytest.mark.skipif(
    not bridge.ACETONE_REPO.is_dir(),
    reason="the AceTone clone is not on this box")
bank_available = pytest.mark.skipif(
    not (Path("/var/cache/veradata/preset_bank_full") / "luts_meta.json").is_file(),
    reason="the LUT bank is not mounted")


# --------------------------------------------------------------------------- #
# 1. axis order
# --------------------------------------------------------------------------- #
def test_identity_grid_is_the_identity_under_this_repos_applier():
    from q3vl.whatb.lutdata import apply_lut_volume

    grid = bridge.identity_grid(32)
    x = torch.rand(4096, 3)
    y = apply_lut_volume(bridge.grid_to_volume(grid), x)
    assert float((y - x).abs().max()) < 2e-3        # 32-point grid, linear interp


def test_asymmetric_grid_moves_only_the_channel_it_names():
    from q3vl.whatb.lutdata import apply_lut_volume

    lin = np.linspace(0.0, 1.0, 32, dtype=np.float32)
    b, g, r = np.meshgrid(lin, lin, lin, indexing="ij")
    grid = np.stack([r * 0.5, g, b], -1).astype(np.float32)
    x = torch.rand(4096, 3)
    y = apply_lut_volume(bridge.grid_to_volume(grid), x)
    assert float((y[:, 0] - x[:, 0] * 0.5).abs().max()) < 2e-3
    assert float((y[:, 1] - x[:, 1]).abs().max()) < 2e-3
    assert float((y[:, 2] - x[:, 2]).abs().max()) < 2e-3


def test_grid_to_volume_rejects_a_non_cubic_grid():
    with pytest.raises(ValueError):
        bridge.grid_to_volume(np.zeros((8, 8, 9, 3), dtype=np.float32))


@repo_available
def test_resize_lut_is_acetones_own_and_preserves_the_identity():
    src = bridge.identity_grid(33)
    out = bridge.resize_lut_acetone(src, 32)
    assert out.shape == (32, 32, 32, 3) and out.dtype == np.float32
    assert float(np.abs(out - bridge.identity_grid(32)).max()) < 1e-5


@repo_available
@bank_available
def test_the_two_cube_readers_produce_the_same_array():
    from q3vl.whatb.lutdata import LutBank

    bank = LutBank()
    checked = 0
    for lid in bank.lut_ids():
        entry = bank.entry(lid)
        if entry.suffix != ".cube":
            continue
        rep = bridge.cube_reader_parity(entry.path)
        assert rep["max_abs_delta"] == 0.0, rep
        checked += 1
        if checked >= 3:
            break
    assert checked == 3


# --------------------------------------------------------------------------- #
# 2. the prompt and the generation kwargs, read out of the clone
# --------------------------------------------------------------------------- #
def _eval_script_ast() -> ast.Module:
    return ast.parse(EVAL_SCRIPT.read_text(encoding="utf-8"))


@repo_available
def test_prompt_is_verbatim():
    strings = [n.value for n in ast.walk(_eval_script_ast())
               if isinstance(n, ast.Constant) and isinstance(n.value, str)
               and "un-touched raw image" in n.value]
    assert len(strings) == 1, strings
    assert strings[0] == bridge.PST_PROMPT


@repo_available
def test_generation_kwargs_are_verbatim():
    calls = [n for n in ast.walk(_eval_script_ast())
             if isinstance(n, ast.Call) and isinstance(n.func, ast.Attribute)
             and n.func.attr == "generate"]
    assert calls, "eval/predict_lut_ddp.py no longer calls .generate"
    for call in calls:
        kw = {k.arg: ast.literal_eval(k.value) for k in call.keywords
              if k.arg is not None}
        assert kw == bridge.GENERATION_KWARGS, kw


@repo_available
def test_the_mm_token_regex_is_the_upstream_one():
    from q3vl.whatb.acetone import pst_infer

    patterns = [n.value for n in ast.walk(_eval_script_ast())
                if isinstance(n, ast.Constant) and isinstance(n.value, str)
                and n.value.startswith("<MM")]
    assert patterns == [pst_infer.MM_RE.pattern], patterns


@repo_available
def test_upstream_still_pads_with_the_first_token_and_takes_64():
    src = EVAL_SCRIPT.read_text(encoding="utf-8")
    assert "num_missing = 64 - len(prediction_ids_flatten)" in src
    assert "prediction_ids_flatten[:64].reshape(4,4,4)" in src


# --------------------------------------------------------------------------- #
# 3. the guards
# --------------------------------------------------------------------------- #
def _reference_board() -> dict:
    from q3vl.whatb.acetone.rowset import REFERENCE_RUN

    return json.loads((REFERENCE_RUN / "metrics.json").read_text(encoding="utf-8"))


reference_available = pytest.mark.skipif(
    not (Path("/home/bc/data/runs/what_b/whatb_EPR033_lora_spanpool_s2")
         / "metrics.json").is_file(),
    reason="the EPR-033 reference board is not on this box")


@reference_available
def test_assert_baselines_accepts_the_reference_board_itself():
    from q3vl.whatb.acetone import scoring as SC

    rep = SC.assert_baselines(_reference_board())
    assert rep["passed"] is True
    assert set(rep["columns"]) == set(SC.REFERENCE_BASELINES)


@reference_available
@pytest.mark.parametrize("key", ["B0_identity", "B3_bucket_retrieval",
                                 "B4_oracle", "B6_libfill"])
def test_assert_baselines_refuses_a_perturbed_column(key):
    from q3vl.whatb.acetone import scoring as SC

    board = _reference_board()
    board["criteria_columns"][key]["mean"] += 0.01
    with pytest.raises(SC.BaselineMismatch):
        SC.assert_baselines(board)


@reference_available
def test_assert_baselines_refuses_a_missing_column():
    from q3vl.whatb.acetone import scoring as SC

    board = _reference_board()
    board["criteria_columns"].pop("B6_libfill")
    with pytest.raises(SC.BaselineMismatch):
        SC.assert_baselines(board)


def test_finite_report_counts_a_nan_instead_of_dropping_it():
    from q3vl.whatb.acetone import scoring as SC

    rows = [{"sample_id": "a", "E_arm": 1.0},
            {"sample_id": "b", "E_arm": float("nan")},
            {"sample_id": "c", "E_arm": 2.0, "grid_error": float("inf")}]
    rep = SC.finite_report(rows)
    assert rep["all_finite"] is False
    assert rep["keys"] == {"E_arm": 1, "grid_error": 1}
    assert rep["n_nonfinite"] == 2

    clean = SC.finite_report([{"sample_id": "a", "E_arm": 1.0}])
    assert clean["all_finite"] is True and clean["n_nonfinite"] == 0


def test_derangement_never_pairs_a_sample_with_itself():
    from q3vl.whatb.acetone.pst_infer import _derangement

    for n in (2, 5, 64, 567):
        perm = _derangement(n, 20260826)
        assert sorted(perm) == list(range(n))
        assert all(i != j for i, j in enumerate(perm))
    assert _derangement(567, 20260826) == _derangement(567, 20260826)


@pytest.mark.skipif(
    not (Path("/mnt/nfs-ro/bc/data/datasets/sft2seg-20260804")
         / "splits" / "V_what.index.jsonl").is_file(),
    reason="the v20260804 index is not readable")
def test_a_rows_rebuilds_the_reference_row_set():
    from q3vl.whatb.acetone import rowset

    rep = rowset.assert_rows()
    assert rep["n_rows"] == rowset.N_EXPECTED == 567
    assert rep["n_rows_outside_zcache"] == 0
    assert rep["split_facts"]["n"] == 897


# --------------------------------------------------------------------------- #
# 4. the banned list, by absence
# --------------------------------------------------------------------------- #
def test_no_auc_anywhere_in_the_package():
    for path in sorted(PACKAGE.glob("*.py")):          # the package, not its tests
        src = path.read_text(encoding="utf-8")
        tree = ast.parse(src)
        names = [n.name.lower() for n in ast.walk(tree)
                 if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef))]
        assert not any("auc" in n for n in names), path
        assert "roc_auc" not in src and "auroc" not in src.lower(), path


def test_the_scoring_path_never_imports_acetones_own_metrics():
    banned = {"skimage", "lpips", "dataset.lut3d", "eval.color_similarity"}
    for name in ("scoring.py", "publish.py", "run_tokenizer_row.py",
                 "run_pst_row.py", "rowset.py"):
        tree = ast.parse((PACKAGE / name).read_text(encoding="utf-8"))
        for node in ast.walk(tree):
            mods: list[str] = []
            if isinstance(node, ast.Import):
                mods = [a.name for a in node.names]
            elif isinstance(node, ast.ImportFrom) and node.module:
                mods = [node.module]
            for m in mods:
                assert m.split(".")[0] not in {b.split(".")[0] for b in banned}, \
                    f"{name} imports {m}"


def test_scoring_uses_the_shared_criteria_module_for_every_metric():
    src = (PACKAGE / "scoring.py").read_text(encoding="utf-8")
    for call in ("C.image_delta_e00", "C.compose_hat", "C.function_distance",
                 "C.bucket_draw", "C.oracle_lut_ids"):
        assert call in src, call
    assert "def delta_e" not in src and "def dE" not in src


def test_required_columns_are_the_four_baselines_plus_the_headline():
    from q3vl.whatb.acetone.publish import REQUIRED

    assert set(REQUIRED) == {"headline_normal_only", "B0_identity",
                             "B3_bucket_retrieval", "B4_oracle", "B6_libfill"}


# --------------------------------------------------------------------------- #
# 5. the published boards, when they exist
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("run", ["whatb_ACETONE_tokenizer", "whatb_ACETONE_pst"])
def test_published_board_carries_every_required_column(run):
    path = Path("/home/bc/data/runs/what_b") / run / "metrics.json"
    if not path.is_file():
        pytest.skip(f"{run} has not been published yet")
    from q3vl.whatb.acetone.publish import REQUIRED
    from q3vl.whatb.acetone.scoring import REFERENCE_BASELINES

    board = json.loads(path.read_text(encoding="utf-8"))
    assert board["n_rows"] == board["n_normal"] == 567
    for key in REQUIRED:
        if key == "headline_normal_only":
            assert board["contexts"]["all"][key]["n"] == 567
            continue
        assert board["criteria_columns"][key]["n"] == 567
    for key, want in REFERENCE_BASELINES.items():
        assert abs(board["criteria_columns"][key]["mean"] - want) < 5e-5
    finite = board["facts"]["assertions"]["A_finite"]
    assert finite["all_finite"] is True, finite
    head = board["contexts"]["all"]["headline_normal_only"]["mean"]
    assert math.isfinite(head)
