"""Pre-step-0 Where-A oracle coverage check (the C04 post-mortem, 2026-08-11).

C04 (OracleWhere-SB48) trained for 3h42m / 3,440 steps and then died in
``WhatBatchBuilder._oracle_latent`` on ONE sample --
``sft_ef61ee381c35f9dc5fd29e611402686b``, whose published Where-A record carries
a ``band`` fit but no ``cband12`` one (1 of 75,544 local train samples).  No
checkpoint was written; the card had been burning for four hours.

``run_what`` already asserted that the generated ``<color>`` context covers the
split before the first step, on the explicit grounds that "a coverage hole
discovered mid-run has no legal repair".  The oracle store had no equivalent:
the 2026-08-10 verification asked whether a *record exists*, which this sample
passes.  These tests pin the missing half -- presence is not coverage -- and the
two policies that replaced the four-hour crash.
"""

from __future__ import annotations

import pytest

from q3vl.what.data import (
    ORACLE_UNCOVERED_DROP,
    ORACLE_UNCOVERED_FAIL,
    ORACLE_UNCOVERED_MAX_FRAC,
    oracle_uncovered,
    resolve_oracle_coverage,
)


class _Ref:
    def __init__(self, sample_id: str, build: str):
        self.sample_id = sample_id
        self.meta = {"build": build}


class _Dataset:
    """Just enough WhatDataset to exercise the coverage path."""

    def __init__(self, refs):
        self.refs = list(refs)
        self.rejections: list[dict] = []

    def __len__(self):
        return len(self.refs)

    # the real method, copied in behaviour: drop by id, record every removal
    def exclude(self, sample_ids, reason):
        drop = set(sample_ids)
        keep = []
        for ref in self.refs:
            if ref.sample_id in drop:
                self.rejections.append({"sample_id": ref.sample_id,
                                        "reason": reason})
            else:
                keep.append(ref)
        n = len(self.refs) - len(keep)
        self.refs = keep
        return n


class _Store:
    """``fits`` keyed by sample -> readout, mirroring the published schema."""

    def __init__(self, fits):
        self.fits = fits

    def latent(self, sample_id, readout, **kw):
        if sample_id not in self.fits:
            raise KeyError(sample_id)          # no record at all
        return self.fits[sample_id].get(readout)


def _split(n_local=8, n_global=4):
    refs = [_Ref(f"loc{i}", "l1") for i in range(n_local)]
    refs += [_Ref(f"glo{i}", "g1") for i in range(n_global)]
    return refs


def _store_all_ok(refs, readout="cband12"):
    return _Store({r.sample_id: {readout: object()}
                   for r in refs if r.meta["build"] == "l1"})


def test_global_samples_are_not_uncovered():
    """A global edit has no ROI; its absence from the store is not a hole.

    This is why the check keys on the build code: if it flagged every sample
    without a record it would flag all 83,671 global train samples.
    """
    refs = _split()
    ds = _Dataset(refs)
    assert oracle_uncovered(ds, _store_all_ok(refs), "cband12") == []


def test_record_present_but_wrong_readout_is_uncovered():
    """The exact C04 shape: a record exists, but not for the asked readout.

    ``store.has(...)`` -- the 2026-08-10 check -- says yes here.  Coverage says
    no, which is the whole point of the new check.
    """
    refs = _split()
    fits = {r.sample_id: {"cband12": object(), "band": object()}
            for r in refs if r.meta["build"] == "l1"}
    fits["loc3"] = {"band": object()}          # band only, like the real sample
    ds = _Dataset(refs)
    assert oracle_uncovered(ds, _Store(fits), "cband12") == ["loc3"]
    # ... and the same store is fully covered for the readout it does carry
    assert oracle_uncovered(ds, _Store(fits), "band") == []


def test_missing_record_for_a_local_sample_is_uncovered():
    refs = _split()
    fits = {r.sample_id: {"cband12": object()}
            for r in refs if r.meta["build"] == "l1"}
    del fits["loc5"]
    assert oracle_uncovered(_Dataset(refs), _Store(fits), "cband12") == ["loc5"]


def test_rejected_fit_is_uncovered():
    """``OracleStore.latent`` returns None for a fit whose status is not ok."""
    refs = _split()
    fits = {r.sample_id: {"cband12": object()}
            for r in refs if r.meta["build"] == "l1"}
    fits["loc1"] = {"cband12": None}
    assert oracle_uncovered(_Dataset(refs), _Store(fits), "cband12") == ["loc1"]


def test_fail_policy_raises_before_step_zero():
    """The default turns a 3h42m crash into a startup refusal."""
    refs = _split()
    fits = {r.sample_id: {"cband12": object()}
            for r in refs if r.meta["build"] == "l1"}
    fits["loc3"] = {"band": object()}
    ds = _Dataset(refs)
    with pytest.raises(RuntimeError, match="no usable Where-A oracle fit"):
        resolve_oracle_coverage(ds, _Store(fits), "cband12", ORACLE_UNCOVERED_FAIL)
    assert len(ds) == 12, "a refusal must not mutate the population"


def test_clean_split_is_a_no_op_under_both_policies():
    for policy in (ORACLE_UNCOVERED_FAIL, ORACLE_UNCOVERED_DROP):
        refs = _split()
        ds = _Dataset(refs)
        facts = resolve_oracle_coverage(ds, _store_all_ok(refs), "cband12", policy)
        assert facts["n_uncovered"] == 0 and facts["n_dropped"] == 0
        assert len(ds) == 12 and ds.rejections == []


def test_drop_policy_excludes_and_records_every_id():
    # roughly the real train proportions (75,544 local / 83,671 global), scaled
    # so that one hole sits under the cap exactly as the real one does
    refs = _split(n_local=12_000, n_global=8_000)
    fits = {r.sample_id: {"cband12": object()}
            for r in refs if r.meta["build"] == "l1"}
    fits["loc7"] = {"band": object()}
    ds = _Dataset(refs)
    facts = resolve_oracle_coverage(ds, _Store(fits), "cband12",
                                    ORACLE_UNCOVERED_DROP)
    assert facts["n_uncovered"] == 1 and facts["n_dropped"] == 1
    assert facts["uncovered_ids"] == ["loc7"]
    assert len(ds) == 19_999
    assert "loc7" not in {r.sample_id for r in ds.refs}
    # the removal is recorded, not silent -- these facts reach run_setup.json
    assert ds.rejections == [{"sample_id": "loc7",
                              "reason": "oracle_uncovered_cband12"}]


def test_drop_refuses_a_hole_big_enough_to_be_a_population_change():
    """``drop`` is for a data blemish.  A real hole needs a ruling, not a flag."""
    refs = _split(n_local=100, n_global=0)
    fits = {r.sample_id: {"cband12": object()} for r in refs}
    for i in range(5):                       # 5% >> the 0.01% cap
        fits[f"loc{i}"] = {"band": object()}
    ds = _Dataset(refs)
    with pytest.raises(RuntimeError, match="population change"):
        resolve_oracle_coverage(ds, _Store(fits), "cband12", ORACLE_UNCOVERED_DROP)
    assert len(ds) == 100, "a refusal must not mutate the population"


def test_cap_is_small_enough_to_be_meaningful():
    """The cap must not be able to swallow a percent of the split."""
    assert 0 < ORACLE_UNCOVERED_MAX_FRAC <= 1e-3


def test_unknown_policy_is_rejected():
    refs = _split()
    with pytest.raises(ValueError, match="unknown oracle_uncovered policy"):
        resolve_oracle_coverage(_Dataset(refs), _store_all_ok(refs),
                                "cband12", "ignore")


def test_real_train_split_arithmetic_leaves_the_lr_schedule_intact():
    """Dropping the one real hole must not move ``total_optimizer_steps``.

    Protocol 11's paired comparison pins the LR schedule, so an exclusion that
    changed the step count would not be a blemish repair -- it would break the
    pairing with C01/C02.  Measured 2026-08-11: train = 159,215 samples,
    micro_batch 16, effective batch 32, teacher_fraction 0.5.
    """
    from q3vl.whereb.context import BalancedContextSampler

    def steps(n):
        return len(BalancedContextSampler(n, 16, seed=20260804)) // 2

    assert steps(159_215) == 4975          # what C01/C02/C04 actually ran
    assert steps(159_214) == 4975          # with the one uncovered sample gone
