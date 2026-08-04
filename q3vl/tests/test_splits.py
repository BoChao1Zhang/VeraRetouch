"""Unit tests for the LUT reserve and the four-way evaluation partition.

Run: ``python -m pytest q3vl/tests/test_splits.py -q``
"""

from __future__ import annotations

from q3vl.data.splits import (
    SPLITS,
    audit,
    choose_lut_reserve,
    partition_eval,
    source_groups,
)


def _row(sft_id, source, lut, build="g1", major="m0", path=None):
    return {
        "sft_id": sft_id, "source_id": source, "lut_id": lut, "build": build,
        "major": major, "i_in_path": path or f"/img/{source}.jpg",
    }


def _corpus(n_sources=30, n_luts=30):
    """A miniature of the real shape: eval is ~2% of the corpus and the per-LUT
    training cost varies by an order of magnitude, so only the cheap LUTs fit in
    a 5% reserve budget."""
    train, evals = [], []
    k = 0
    for lut in range(n_luts):
        for _ in range(5 + (lut % 10) * 10):     # 5 .. 95 training rows per LUT
            train.append(_row(f"t{k}", f"src_t{k % 200}", f"lut{lut}",
                              major=f"m{lut % 3}"))
            k += 1
    j = 0
    for s in range(n_sources):
        for lut in range(n_luts):
            evals.append(_row(f"e{j}", f"src_e{s}", f"lut{lut}", major=f"m{lut % 3}"))
            j += 1
    return train, evals


def test_reserve_respects_the_training_budget():
    train, evals = _corpus()
    res = choose_lut_reserve(train, evals, target_eval=10_000, budget_fraction=0.05)
    assert res.train_cost <= 0.05 * len(train) + 1e-9
    assert res.train_fraction <= 0.05
    assert res.lut_ids


def test_reserve_is_deterministic():
    train, evals = _corpus()
    a = choose_lut_reserve(train, evals, target_eval=60, budget_fraction=0.05)
    b = choose_lut_reserve(list(reversed(train)), list(reversed(evals)),
                           target_eval=60, budget_fraction=0.05)
    assert a.lut_ids == b.lut_ids


def test_reserve_covers_every_major_when_affordable():
    train, evals = _corpus()
    res = choose_lut_reserve(train, evals, target_eval=60, budget_fraction=0.05)
    covered = {m["major"] for m in res.per_major if m["luts_reserved"]}
    assert covered == {"m0", "m1", "m2"}


def test_source_groups_merge_shared_input_files():
    rows = [
        _row("a", "src_1", "lutA", path="/img/shared.jpg"),
        _row("b", "src_2", "lutB", path="/img/shared.jpg"),
        _row("c", "src_3", "lutC", path="/img/other.jpg"),
    ]
    groups = source_groups(rows)
    assert groups["a"] == groups["b"], "same file under two source ids must not split"
    assert groups["c"] != groups["a"]


def test_partition_is_a_disjoint_cover_and_respects_roles():
    train, evals = _corpus()
    res = choose_lut_reserve(train, evals, target_eval=80, budget_fraction=0.05)
    groups = source_groups(evals)
    plan = partition_eval(evals, res.lut_ids, groups)

    assigned = set(plan.assignment) | set(plan.unused)
    assert assigned == {r["sft_id"] for r in evals}
    assert not (set(plan.assignment) & set(plan.unused))
    assert set(plan.sizes) == set(SPLITS)

    by_id = {r["sft_id"]: r for r in evals}
    # every T_lut_unseen sample carries a reserved LUT, and no other set does
    for sft_id, split in plan.assignment.items():
        reserved = by_id[sft_id]["lut_id"] in res.lut_ids
        assert reserved == (split == "T_lut_unseen"), (sft_id, split)
    # a select-role source never hosts a reserved sample
    for sft_id in plan.unused:
        assert plan.group_role[groups[sft_id]] in ("V_where", "V_what")


def test_audit_reports_zero_leakage_on_a_clean_corpus():
    train, evals = _corpus()
    res = choose_lut_reserve(train, evals, target_eval=80, budget_fraction=0.05)
    groups_all = source_groups(train + evals)
    plan = partition_eval(evals, res.lut_ids, groups_all)
    train_ids = {r["sft_id"] for r in train if r["lut_id"] not in res.lut_ids}
    rows_by_id = {r["sft_id"]: r for r in train + evals}
    report = audit(rows_by_id, plan.assignment, train_ids, groups_all)

    assert all(v == 0 for v in report["sample_id_overlap"].values())
    assert all(v == 0 for v in
               report["protocol_group_overlap_source_lut_build"].values())
    assert report["train_x_T_lut_unseen_lut_overlap"] == 0
    assert report["select_vs_test_source_overlap"] == 0
    assert all(v == 0 for v in report["train_x_eval_sample_overlap"].values())
    # V_where and V_what never share a source; T_final and T_lut_unseen may
    assert report["source_overlap"]["V_where|V_what"] == 0
