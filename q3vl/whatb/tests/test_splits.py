"""Split index / record reading and the B3 bucket pools.

The synthetic half runs anywhere; the mounted half re-measures the frozen data
facts (n = 93934 normal train rows, 567 normal V_what rows, 0 lut_id overlap
with T_lut_unseen) and skips when the read-only mount is absent.
"""

from __future__ import annotations

import json

import pytest

from q3vl.whatb.splits import (
    DATASET_ROOT,
    TRAIN_NORMAL_N,
    IndexRow,
    bucket_pools,
    load_index,
    normal_only,
    read_record,
    ro_path,
    split_facts,
    split_index_path,
)

MOUNTED = (DATASET_ROOT / "splits" / "train.index.jsonl").exists()


def _row(i, conf="normal", task="style", lut="rcp_a", src="src_1"):
    return IndexRow.from_json({"sample_id": f"s{i}", "split": "train",
                               "lut_id": lut, "source_image_id": src,
                               "task_type": task, "winner_confidence": conf})


def test_ro_path_rewrites_only_the_hard_mount():
    assert str(ro_path("/mnt/nfs/bc/x.tar")) == "/mnt/nfs-ro/bc/x.tar"
    assert str(ro_path("/home/bc/x.tar")) == "/home/bc/x.tar"


def test_unknown_split_is_refused():
    with pytest.raises(ValueError, match="unknown split"):
        split_index_path("V_nope")


def test_normal_only_and_facts():
    rows = [_row(0), _row(1, conf="low"), _row(2, task="local", lut="rcp_b")]
    assert len(normal_only(rows)) == 2
    f = split_facts(rows)
    assert f == {"n": 3, "n_normal": 2, "n_low": 1, "n_style": 2, "n_local": 1,
                 "n_normal_style": 1, "n_normal_local": 1, "uniq_lut_id": 2,
                 "uniq_lut_id_normal": 2, "uniq_source": 1}


def test_bucket_pools_are_built_from_the_record_label():
    recs = [{"minor": "warm_01", "lut_id": "a"}, {"minor": "warm_01", "lut_id": "b"},
            {"minor": "warm_01", "lut_id": "a"}, {"minor": "cool_02", "lut_id": "c"},
            {"minor": None, "lut_id": "d"}, {"minor": "cool_02"}]
    pools = bucket_pools(recs)
    assert pools == {"cool_02": ["c"], "warm_01": ["a", "b"]}


def test_record_reader_uses_offset_and_length(tmp_path):
    payload = json.dumps({"sample_id": "s0", "minor": "m", "color": "c"}).encode()
    blob = b"x" * 37 + payload + b"y" * 11
    (tmp_path / "shard.tar").write_bytes(blob)
    row = IndexRow.from_json({
        "sample_id": "s0", "members": {"record": {
            "shard": str(tmp_path / "shard.tar"), "offset": 37,
            "length": len(payload)}}})
    assert read_record(row)["minor"] == "m"


@pytest.mark.skipif(not MOUNTED, reason="sft2seg splits not mounted")
def test_frozen_data_facts_hold_on_the_mounted_splits():
    v = load_index("V_what")
    assert len(v) == 897
    assert len(normal_only(v)) == 567
    f = split_facts(v)
    assert (f["n_normal_style"], f["n_normal_local"]) == (321, 246)
    assert f["uniq_lut_id"] == 531 and f["uniq_source"] == 163

    t = load_index("T_lut_unseen")
    assert len(t) == 433 and len(normal_only(t)) == 252
    assert split_facts(t)["uniq_lut_id"] == 259

    train = load_index("train")
    assert len(train) == 159215
    assert len(normal_only(train)) == TRAIN_NORMAL_N == 93934
    tf = split_facts(train)
    assert tf["n_normal_style"] == 51182 and tf["n_normal_local"] == 42752
    assert tf["uniq_lut_id"] == 3149
    assert not ({r.lut_id for r in train} & {r.lut_id for r in t})


@pytest.mark.skipif(not MOUNTED, reason="sft2seg splits not mounted")
def test_records_carry_the_b3_labels():
    rows = load_index("V_what")[:8]
    for rec in (read_record(r) for r in rows):
        assert isinstance(rec["minor"], str) and isinstance(rec["major"], str)
        assert isinstance(rec["color"], str) and isinstance(rec["instruction"], str)
        assert rec["preset_path"].endswith((".cube", ".3dl"))
