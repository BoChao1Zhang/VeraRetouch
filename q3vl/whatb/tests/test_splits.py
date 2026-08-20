"""Split index / record reading and the B3 bucket pools.

The synthetic half runs anywhere; the mounted half re-measures the data facts of
**every registered index口径** (``DATASET_VERSIONS``) and skips when the
read-only mount is absent.  The numbers below were measured on 2026-08-18 and
are what :func:`train_normal_rows` asserts against at run time.
"""

from __future__ import annotations

import json

import pytest

from q3vl.whatb import splits as S
from q3vl.whatb.splits import (
    DATASET_VERSIONS,
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

MOUNTED = S.dataset_available() and S.dataset_available(
    DATASET_VERSIONS["v20260804"].root)

#: measured 2026-08-18, per口径: split -> the split_facts keys this file pins
FACTS = {
    "v20260804": {
        "V_what": (897, 567, 321, 246, 531, 163),
        "T_lut_unseen": (433, 252, 144, 108, 259, 212),
        "train": (159215, 93934, 51182, 42752, 3149, 27104),
    },
    "cut-p45": {
        "V_what": (777, 496, 321, 175, 475, 161),
        "T_lut_unseen": (371, 221, 144, 77, 232, 190),
        "train": (135697, 80269, 51182, 29087, 3103, 26489),
    },
}
_FACT_KEYS = ("n", "n_normal", "n_normal_style", "n_normal_local",
              "uniq_lut_id", "uniq_source")


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


def test_the_version_table_pins_the_frozen_block_number():
    """``TRAIN_NORMAL_N`` is the ORIGINAL口径's count -- the published boards'."""
    assert DATASET_VERSIONS["v20260804"].train_normal_n == TRAIN_NORMAL_N == 93934
    assert DATASET_VERSIONS["cut-p45"].train_normal_n == 80269
    assert S.DEFAULT_DATASET_VERSION == "cut-p45"
    assert DATASET_ROOT == DATASET_VERSIONS["cut-p45"].root
    # cut-p45 drops local-mask rows only: the style rows are identical
    assert (DATASET_VERSIONS["cut-p45"].excluded_n
            == DATASET_VERSIONS["v20260804"].n["train"]
            + sum(DATASET_VERSIONS["v20260804"].n[s_] for s_ in
                  ("V_what", "V_where", "T_final", "T_lut_unseen"))
            - DATASET_VERSIONS["cut-p45"].n["train"]
            - sum(DATASET_VERSIONS["cut-p45"].n[s_] for s_ in
                  ("V_what", "V_where", "T_final", "T_lut_unseen")))


def test_an_unregistered_root_is_refused():
    with pytest.raises(AssertionError, match="not a registered dataset version"):
        S.version_for_root("/home/bc/data/datasets/no-such-cut")


@pytest.mark.skipif(not MOUNTED, reason="sft2seg splits not mounted")
@pytest.mark.parametrize("version", sorted(DATASET_VERSIONS))
def test_data_facts_hold_on_the_mounted_splits(version):
    ver = DATASET_VERSIONS[version]
    for split, want in FACTS[version].items():
        f = split_facts(load_index(split, ver.root))
        assert tuple(f[k] for k in _FACT_KEYS) == want, (version, split)
        assert ver.n[split] == f["n"] and ver.normal_n[split] == f["n_normal"]

    train = load_index("train", ver.root)
    t = load_index("T_lut_unseen", ver.root)
    assert not ({r.lut_id for r in train} & {r.lut_id for r in t})
    # the口径's own declaration is what train_normal_rows asserts against
    assert S.train_normal_n("v2seg", root=ver.root) == ver.train_normal_n


@pytest.mark.skipif(not MOUNTED, reason="sft2seg splits not mounted")
def test_cut_p45_is_a_strict_subset_of_the_published_index():
    a = {r.sample_id for r in load_index("train", DATASET_VERSIONS["v20260804"].root)}
    b = {r.sample_id for r in load_index("train", DATASET_VERSIONS["cut-p45"].root)}
    assert b < a


@pytest.mark.skipif(not MOUNTED, reason="sft2seg splits not mounted")
def test_records_carry_the_b3_labels():
    rows = load_index("V_what")[:8]
    for rec in (read_record(r) for r in rows):
        assert isinstance(rec["minor"], str) and isinstance(rec["major"], str)
        assert isinstance(rec["color"], str) and isinstance(rec["instruction"], str)
        assert rec["preset_path"].endswith((".cube", ".3dl"))
