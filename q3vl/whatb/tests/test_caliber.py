"""The shared EPR-030 caliber (``q3vl/whatb/caliber.py``) and the five arms.

Two things are pinned here:

1. the caliber resolves to the EPR-030 numbers (``256x8192`` -> 2,097,152
   colours/step; ``n = 119,828`` -> 469 steps/epoch -> 18,760 steps at 40
   epochs; ``--loss-level 1`` -> every optional weight 0);
2. the two step-matched splits (``32x256`` / ``64x128``) resolve to exactly the
   pre-2026-08-16 values on every one of the five arms -- the arithmetic half
   of the "behaviour unchanged" claim (the run-level half is the ``steps.jsonl``
   comparison recorded in ``experiments/prs/EPR-030_shared-query-backbone/NOTES.md``).
"""

from __future__ import annotations

import math

import pytest

from q3vl.whatb import caliber as K
from q3vl.whatb.arms import affonly as AFF
from q3vl.whatb.arms import carrier as CAR
from q3vl.whatb.arms import idgate as IG
from q3vl.whatb.arms import interpc as IC


# --------------------------------------------------------------------------- #
# 1. the colour batch comes from the ONE table
# --------------------------------------------------------------------------- #
def test_the_table_is_carriers_and_is_not_copied() -> None:
    assert K.BATCH_SPLITS is CAR.BATCH_SPLITS
    assert K.FROZEN_BATCH_SPLITS is CAR.FROZEN_BATCH_SPLITS
    assert K.BASE_LR == CAR.BASE_LR == 1e-3
    assert K.EPR030_BATCH_SPLIT in K.BATCH_SPLITS


def test_parse_batch_split_matches_the_pre_change_literal_parse() -> None:
    """The old parse was ``tuple(int(v) for v in name.split("x"))``."""
    for name in K.BATCH_SPLITS:
        b, q = name.split("x")
        assert K.parse_batch_split(name) == (int(b), int(q))
    assert K.parse_batch_split("256x8192") == (256, 8192)
    assert K.colours_per_step("256x8192") == 2_097_152
    assert K.colours_per_step("32x256") == K.colours_per_step("64x128") == 8192
    # the two --smoke-only splits two runners accept are still parseable
    assert K.parse_batch_split("8x64") == (8, 64)
    with pytest.raises(ValueError):
        K.parse_batch_split("not-a-split")


def test_batch_split_choices_carry_the_epr030_row_and_the_frozen_pair() -> None:
    ch = K.batch_split_choices(("8x64", "2x16"))
    for name in ("32x256", "64x128", "256x8192", "8x64", "2x16"):
        assert name in ch
    assert len(ch) == len(K.BATCH_SPLITS) + 2


# --------------------------------------------------------------------------- #
# 2. the horizon
# --------------------------------------------------------------------------- #
def test_steps_per_epoch_and_its_runtime_assertion() -> None:
    assert K.steps_per_epoch_of(93934, 32) == 2936
    assert K.steps_per_epoch_of(93934, 64) == 1468
    assert K.steps_per_epoch_of(119828, 256) == 469
    assert 469 * 40 == 18_760
    assert K.assert_steps_per_epoch(469, n_train=119828, batch_samples=256) == 469
    with pytest.raises(AssertionError, match="epoch length"):
        K.assert_steps_per_epoch(468, n_train=119828, batch_samples=256)


def test_horizon_record_asserts_and_records() -> None:
    rec = K.horizon_record(
        data="v2seg+l8", n_train=119828, batch_split="256x8192",
        batch_samples=256, queries_per_sample=8192, steps_per_epoch=469,
        total_steps=18760, base_lr=1e-3, epochs=40, loss_level=1,
        lambda_hc=0.0, lambda_sparse=0.0)
    assert rec["colours_per_step"] == rec["colors_per_step"] == 2_097_152
    assert (rec["steps_per_epoch"], rec["total_steps"]) == (469, 18760)
    assert rec["train_normal_n_measured"] == 119828
    assert rec["base_lr"] == 1e-3 and rec["data_sources"] == ["v2seg", "l8"]
    assert rec["batch_split_step_matched_to_epr024"] is False
    assert rec["pure_l1"] is True
    with pytest.raises(AssertionError):
        K.horizon_record(data="v2seg", n_train=119828, batch_split="256x8192",
                         batch_samples=256, queries_per_sample=8192,
                         steps_per_epoch=470, total_steps=18800, base_lr=1e-3)


# --------------------------------------------------------------------------- #
# 3. pure L1
# --------------------------------------------------------------------------- #
def test_the_ladder_rule_is_carriers_own() -> None:
    """``carrier.py:347`` / ``:351``, on the same inputs."""
    for level in (1, 2, 3, 4):
        cfg = CAR.CarrierConfig(loss_level=level)
        assert K.effective_lambda_hc(CAR.LAMBDA_HC, level) == cfg.lambda_hc
        assert K.effective_lambda_sparse(CAR.LAMBDA_SPARSE, level) == cfg.lambda_sparse


def test_pure_l1_record_runs_the_assertion() -> None:
    rec = K.pure_l1_record(loss_level=1, lambda_hc=0.0, lambda_sparse=0.0)
    assert rec["pure_l1"] is True and "assert_l0_pure" in rec
    off = K.pure_l1_record(loss_level=3, lambda_hc=10.0, lambda_sparse=0.001)
    assert off["pure_l1"] is False and "assert_l0_pure" not in off
    with pytest.raises(AssertionError):
        K.pure_l1_record(loss_level=1, lambda_hc=10.0, lambda_sparse=0.0)


# --------------------------------------------------------------------------- #
# 4. the corpora
# --------------------------------------------------------------------------- #
def test_data_choices_and_extra_sources() -> None:
    assert K.DATA_CHOICES == ("v2seg", "v2seg+l8")
    assert K.extra_zcache_sources("v2seg") == ()
    (root, split), = K.extra_zcache_sources("v2seg+l8")
    assert split == "l8_train"
    assert str(root).endswith("zcache_l8")
    assert str(K.extra_zcache_sources(
        "v2seg+l8", zcache_root_l8="/tmp/other")[0][0]) == "/tmp/other"
    with pytest.raises(ValueError):
        K.extra_zcache_sources("nope")


# --------------------------------------------------------------------------- #
# 5. the five arms on the EPR-030 caliber
# --------------------------------------------------------------------------- #
def test_every_runner_declares_the_four_caliber_flags() -> None:
    import importlib

    want = {"data", "zcache_root_l8", "batch_split", "loss_level", "dry_run"}
    for name in ("affonly", "interpc", "idgate", "g4d", "qdual"):
        mod = importlib.import_module(f"q3vl.whatb.scripts.run_{name}_arm")
        build = next(getattr(mod, n) for n in
                     ("build_parser", "build_argparser", "build_arg_parser")
                     if hasattr(mod, n))
        ap = build()
        dests = {a.dest for a in ap._actions}
        assert want <= dests, (name, sorted(want - dests))
        # --base-lr is spelled the same way everywhere (dest may be `lr`)
        opts = {o for a in ap._actions for o in a.option_strings}
        assert "--base-lr" in opts, name
        assert "--data" in opts and "--zcache-root-l8" in opts, name


@pytest.mark.parametrize("split,b,q", [("32x256", 32, 256), ("64x128", 64, 128)])
def test_frozen_pair_is_unchanged_on_every_arm(split: str, b: int, q: int) -> None:
    """The arithmetic half of "32x256 / 64x128 behave exactly as before"."""
    assert K.parse_batch_split(split) == (b, q)
    assert b * q == 8192
    spe = K.steps_per_epoch_of(93934, b)
    assert (spe, spe * 40) == ((2936, 117440) if b == 32 else (1468, 58720))

    aff = AFF.AffineOnlyConfig(batch_samples=b, queries=q)
    assert (aff.colors_per_step, aff.batch_split) == (8192, split)
    assert aff.step_matched_to_epr024 is True
    assert (aff.lambda_hc_effective, aff.lambda_sparse_effective) == (10.0, 0.001)
    assert aff.steps_per_epoch == spe

    ic = IC.InterpcConfig(batch_samples=b, queries_per_sample=q)
    assert (ic.colors_per_step, ic.batch_split) == (8192, split)
    assert ic.step_matched_to_epr024 is True
    assert (ic.lambda_hc_effective, ic.lambda_sparse_effective) == (10.0, 0.001)

    ig = IG.IdGateConfig(batch_samples=b, queries=q)
    assert (ig.colors_per_step, ig.batch_split) == (8192, split)
    assert ig.step_matched_to_epr024 is True
    assert (ig.lambda_hc_effective, ig.lambda_sparse_effective) == (10.0, 0.001)
    assert ig.steps_per_epoch == spe

    from q3vl.whatb.scripts import run_g4d_arm as G4
    a = G4.build_arg_parser().parse_args(["--out", "/tmp/x",
                                          "--batch-split", split])
    assert G4.parse_batch_split(a.batch_split) == (b, q)
    # EPR-028 R1 §4.4 moved this arm's DEFAULT ladder onto the caliber's pure
    # L1 (the other four arms' config objects still default to 3, which is why
    # they are asserted at (10.0, 0.001) above).  The batch arithmetic -- what
    # this test is about -- is unchanged.
    assert G4.resolve_loss_level(a) == K.PURE_L1_LOSS_LEVEL == 1
    assert G4.effective_lambdas(a) == (0.0, 0.0)
    a3 = G4.build_arg_parser().parse_args(
        ["--out", "/tmp/x", "--batch-split", split, "--loss-level", "3"])
    assert G4.effective_lambdas(a3) == (10.0, 0.001)

    from q3vl.whatb.scripts import run_qdual_arm as QD
    aq = QD.build_parser().parse_args(["--batch-split", split])
    assert (aq.batch_samples, aq.queries) == (b, q)
    assert aq.lr == 1e-3


def test_every_arm_reaches_469_and_18760_on_the_epr030_caliber() -> None:
    n, b, q = 119828, 256, 8192
    assert K.steps_per_epoch_of(n, b) == 469
    assert 469 * 40 == 18760

    aff = AFF.AffineOnlyConfig(batch_samples=b, queries=q, train_n=n,
                               data="v2seg+l8", loss_level=1,
                               total_steps=469 * 40)
    assert (aff.steps_per_epoch, aff.total_steps) == (469, 18760)
    assert aff.colors_per_step == 2_097_152
    assert (aff.lambda_hc_effective, aff.lambda_sparse_effective) == (0.0, 0.0)

    ic = IC.InterpcConfig(batch_samples=b, queries_per_sample=q, train_n=n,
                          data="v2seg+l8", loss_level=1,
                          steps_per_epoch=469, total_steps=469 * 40)
    assert (ic.steps_per_epoch, ic.total_steps) == (469, 18760)
    assert ic.colors_per_step == 2_097_152
    assert (ic.lambda_hc_effective, ic.lambda_sparse_effective) == (0.0, 0.0)

    ig = IG.IdGateConfig(batch_samples=b, queries=q, train_n=n,
                         data="v2seg+l8", loss_level=1)
    assert (ig.steps_per_epoch, ig.total_steps) == (469, 18760)
    assert ig.colors_per_step == 2_097_152
    assert (ig.lambda_hc_effective, ig.lambda_sparse_effective) == (0.0, 0.0)

    from q3vl.whatb.scripts import run_g4d_arm as G4
    a = G4.build_arg_parser().parse_args(
        ["--out", "/tmp/x", "--batch-split", "256x8192", "--loss-level", "1",
         "--data", "v2seg+l8", "--base-lr", "1e-3"])
    bb, qq = G4.parse_batch_split(a.batch_split)
    assert (bb, qq) == (256, 8192) and G4.effective_lambdas(a) == (0.0, 0.0)
    assert math.ceil(n / bb) * a.epochs == 18760 and a.lr == 1e-3

    from q3vl.whatb.scripts import run_qdual_arm as QD
    aq = QD.build_parser().parse_args(
        ["--batch-split", "256x8192", "--loss-level", "1", "--data", "v2seg+l8",
         "--base-lr", "1e-3"])
    assert (aq.batch_samples, aq.queries) == (256, 8192)
    assert math.ceil(n / aq.batch_samples) * aq.epochs == 18760
    assert aq.lr == 1e-3
