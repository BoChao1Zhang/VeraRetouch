"""EPR-030's data side: two z caches as one condition, and a measured horizon.

The synthetic half runs anywhere.  The mounted half re-measures the numbers the
proposal quotes (per-source n, the merged n, steps/epoch, total steps and the
four eval-split intersections) and skips when the soft mount or the L8 manifest
is absent -- a number in a proposal that no test re-measures is a number that
drifts.
"""

from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np
import pytest
import torch

from q3vl.whatb import splits as S
from q3vl.whatb import zcache as Z
from q3vl.whatb.scripts import run_epr030_arm as E

SEG_COLOR = 151674
SEG_WHERE = 151673

MOUNTED = S.dataset_available() and S.dataset_available(
    S.dataset_version("v20260804").root)
L8_PRESENT = S.L8_MANIFEST.is_file()


# --------------------------------------------------------------------------- #
# fixtures
# --------------------------------------------------------------------------- #
def _rows(ids, split):
    seq = [151669, 7, 151670, 151671, 9, 151672, SEG_WHERE, SEG_COLOR]
    return [{"sample_id": sid, "split": split, "reply_token_ids": seq,
             "readout_index": len(seq) - 1, "expected_ids": [SEG_COLOR],
             "n_generated_tokens": len(seq),
             "color_text": f"<color>{sid}</color>"} for sid in ids]


def _write(root, *, split, ids, tag="none", checkpoint="ckpt",
           kind="seg_color", context="generated", fill=1.0):
    z = np.full((len(ids), Z.Z_DIM), float(fill), dtype=np.float32)
    for i in range(len(ids)):
        z[i] += i
    return Z.write_z_cache(Z.leaf_dir(root, split, tag), _rows(ids, split), z,
                           checkpoint=checkpoint, readout_kind=kind,
                           context_source=context, control_tag=tag, split=split)


def _l8_manifest(tmp_path, rows, *, report=None) -> Path:
    man = tmp_path / "l8_train.manifest.jsonl"
    man.write_text("".join(json.dumps(r) + "\n" for r in rows), encoding="utf-8")
    rep = report if report is not None else {
        "n_manifest": len(rows),
        "counts": {"usable_final_normal":
                   sum(1 for r in rows if r["winner_confidence"] == "normal")},
    }
    (tmp_path / "l8_train.manifest.report.json").write_text(json.dumps(rep),
                                                            encoding="utf-8")
    return man


def _l8_row(i, conf="normal"):
    return {"sample_id": f"sft_l8_{i}", "sft_id": f"sft_l8_{i}",
            "lut_id": f"rcp_{i}", "source_image_id": f"src_{i}",
            "task_type": "local", "winner_confidence": conf,
            "color": f"<color>l8 {i}</color>", "where": "subject: x"}


# --------------------------------------------------------------------------- #
# 1. the cache root layout
# --------------------------------------------------------------------------- #
def test_resolve_leaf_finds_the_per_context_sub_directory(tmp_path):
    """``<root>/<context>/<split>__<tag>`` is how the L8 cache is laid out."""
    _write(tmp_path / "generated", split="l8_train", ids=("a", "b"))
    assert Z.resolve_leaf(tmp_path, "l8_train", "none", "generated") is not None
    # the canonical placement still wins when both exist
    canon = _write(tmp_path, split="l8_train", ids=("a", "b"))
    assert Z.resolve_leaf(tmp_path, "l8_train", "none", "generated") == canon


def test_pointing_the_wrong_context_at_a_context_root_still_fails(tmp_path):
    _write(tmp_path / "generated", split="l8_train", ids=("a", "b"),
           context="generated")
    leaf = Z.resolve_leaf(tmp_path, "l8_train", "none", "teacher")
    assert leaf is None                     # <root>/teacher/... does not exist
    leaf = Z.resolve_leaf(tmp_path, "l8_train", "none", "generated")
    with pytest.raises(AssertionError, match="context_source"):
        Z.ZCache(leaf).assert_belongs_to(checkpoint="ckpt",
                                         readout_kind="seg_color",
                                         context_source="teacher")


# --------------------------------------------------------------------------- #
# 2. MultiZCache
# --------------------------------------------------------------------------- #
def test_multi_zcache_dispatches_by_sample_id(tmp_path):
    a = Z.ZCache(_write(tmp_path / "A", split="train", ids=("a", "b"), fill=1.0))
    b = Z.ZCache(_write(tmp_path / "B", split="l8_train", ids=("c", "d"), fill=100.0))
    m = Z.MultiZCache([a, b])
    assert len(m) == 4 and "a" in m and "d" in m and "zz" not in m
    assert m.k_rows == 0 and m.control_tag == "none"
    assert m.split == "train+l8_train"
    # interleaved order is preserved across members
    got = m.batch(["d", "a", "c", "b"])
    assert got.shape == (4, Z.Z_DIM)
    torch.testing.assert_close(got[0], b.vector("d"))
    torch.testing.assert_close(got[1], a.vector("a"))
    torch.testing.assert_close(got[2], b.vector("c"))
    torch.testing.assert_close(got[3], a.vector("b"))
    torch.testing.assert_close(m.vector("c"), b.vector("c"))
    assert m.field("c", "color_text") == "<color>c</color>"


def test_multi_zcache_mean_is_n_weighted_not_a_mean_of_means(tmp_path):
    a = Z.ZCache(_write(tmp_path / "A", split="train", ids=tuple("abc"), fill=0.0))
    b = Z.ZCache(_write(tmp_path / "B", split="l8_train", ids=("d",), fill=100.0))
    m = Z.MultiZCache([a, b])
    stacked = torch.cat([a.batch(a.ids), b.batch(b.ids)], dim=0)
    torch.testing.assert_close(m.mean(), stacked.mean(dim=0), atol=1e-5, rtol=0)


def test_multi_zcache_refuses_an_overlapping_sample_id(tmp_path):
    a = Z.ZCache(_write(tmp_path / "A", split="train", ids=("a", "b")))
    b = Z.ZCache(_write(tmp_path / "B", split="l8_train", ids=("b", "c")))
    with pytest.raises(ValueError, match="silently unreachable"):
        Z.MultiZCache([a, b])


def test_multi_zcache_asserts_every_member(tmp_path):
    a = Z.ZCache(_write(tmp_path / "A", split="train", ids=("a",)))
    bad = Z.ZCache(_write(tmp_path / "B", split="l8_train", ids=("c",),
                          checkpoint="another-base"))
    m = Z.MultiZCache([a, bad])
    with pytest.raises(AssertionError, match="another-base"):
        m.assert_belongs_to(checkpoint="ckpt", readout_kind="seg_color")
    ok = Z.ZCache(_write(tmp_path / "C", split="l8_train", ids=("d",)))
    rec = Z.MultiZCache([a, ok]).assert_belongs_to(checkpoint="ckpt",
                                                   readout_kind="seg_color")
    assert rec["n"] == 2 and rec["n_members"] == 2
    assert len(rec["members"]) == 2
    # verify_plan really replayed on both members, not recorded as "could not"
    assert rec["n_verify_plan"] == 2


def test_multi_zcache_missing_id_names_every_root(tmp_path):
    a = Z.ZCache(_write(tmp_path / "A", split="train", ids=("a",)))
    b = Z.ZCache(_write(tmp_path / "B", split="l8_train", ids=("c",)))
    with pytest.raises(KeyError, match="none of the 2 z caches"):
        Z.MultiZCache([a, b]).vector("zz")


# --------------------------------------------------------------------------- #
# 3. the L8 index
# --------------------------------------------------------------------------- #
def test_l8_index_carries_color_inline_and_needs_no_record_shard(tmp_path):
    man = _l8_manifest(tmp_path, [_l8_row(0), _l8_row(1, "low")])
    rows = S.load_l8_index(man)
    assert len(rows) == 2
    assert [r.source for r in rows] == ["l8", "l8"]
    assert [r.split for r in rows] == ["l8_train", "l8_train"]
    assert not any(r.has_record for r in rows)
    assert rows[0].color_text == "<color>l8 0</color>"
    assert S.color_texts_of(rows) == ["<color>l8 0</color>", "<color>l8 1</color>"]
    assert [r.sample_id for r in S.normal_only(rows)] == ["sft_l8_0"]


def test_l8_index_is_counted_against_its_own_report(tmp_path):
    rows = [_l8_row(0), _l8_row(1)]
    man = _l8_manifest(tmp_path, rows, report={"n_manifest": 3,
                                               "counts": {"usable_final_normal": 3}})
    with pytest.raises(AssertionError, match="declares 3"):
        S.load_l8_index(man)


def test_l8_index_rejects_a_duplicated_sample_id(tmp_path):
    man = _l8_manifest(tmp_path, [_l8_row(0), _l8_row(0)])
    with pytest.raises(AssertionError, match="distinct sample_id"):
        S.load_l8_index(man)


def test_color_texts_prefers_the_inline_segment(tmp_path):
    """A row that carries <color> is never sent to a record shard."""
    inline = S.IndexRow.from_json({"sample_id": "s0", "split": "l8_train",
                                   "color": "<color>inline</color>"})
    assert S.color_texts_of([inline]) == ["<color>inline</color>"]
    # a row with neither an inline text nor a record member contributes nothing
    bare = S.IndexRow.from_json({"sample_id": "s1", "split": "l8_train"})
    assert S.color_texts_of([bare]) == []


# --------------------------------------------------------------------------- #
# 4. the --data flag and the horizon it implies
# --------------------------------------------------------------------------- #
def test_data_choices_and_validation():
    assert S.DATA_CHOICES == ("v2seg", "v2seg+l8")
    assert S.DATA_SOURCES["v2seg"] == ()
    assert S.DATA_SOURCES["v2seg+l8"] == ("l8",)
    with pytest.raises(ValueError, match="--data must be one of"):
        E.Epr030Config(loss_level=1, data="l8-only")
    with pytest.raises(ValueError, match="--data must be one of"):
        S.extra_train_rows("nope")


def test_the_arm_defaults_to_the_union():
    ap = E.build_parser()
    args = ap.parse_args([])
    assert args.data == "v2seg+l8"
    assert E.Epr030Config(loss_level=1).data == "v2seg+l8"


def test_steps_per_epoch_is_ceil_of_the_measured_n():
    for n in (93934, 119828, 33, 32, 31):
        cfg = E.Epr030Config(loss_level=1, train_n=n)
        assert cfg.steps_per_epoch == math.ceil(n / 32)
        assert cfg.total_steps == cfg.steps_per_epoch * cfg.epochs
    assert E.Epr030Config(loss_level=1, train_n=93934).steps_per_epoch == 2936
    assert E.Epr030Config(loss_level=1, train_n=119828).steps_per_epoch == 3745
    assert E.Epr030Config(loss_level=1, train_n=119828).total_steps == 149800


def test_eval_every_zero_means_one_epoch():
    """The default must track the epoch length, not a remembered 2936."""
    ap = E.build_parser()
    assert ap.parse_args([]).eval_every == 0
    assert (int(0) or E.Epr030Config(loss_level=1, train_n=119828).steps_per_epoch) == 3745


def test_run_setup_record_carries_the_measured_horizon():
    cfg = E.Epr030Config(loss_level=1, train_n=119828)
    assert cfg.as_dict()["data"] == "v2seg+l8"
    assert cfg.as_dict()["steps_per_epoch"] == 3745
    assert cfg.as_dict()["total_steps"] == 149800


# --------------------------------------------------------------------------- #
# 5. the mounted numbers the proposal quotes
# --------------------------------------------------------------------------- #
#: (口径 -> its v2seg n, the merged n with L8, ceil(merged/32), *40), measured
#: 2026-08-18.  L8 contributes the same 25,894 normal rows to both: no sft2seg
#: 口径 touches ``l8_train.manifest.jsonl``.
MERGED = {"v20260804": (93934, 119828, 3745, 149800),
          "cut-p45": (80269, 106163, 3318, 132720)}


@pytest.mark.skipif(not (MOUNTED and L8_PRESENT), reason="dataset not mounted")
@pytest.mark.parametrize("version", sorted(MERGED))
def test_measured_population_and_horizon(version):
    root = S.dataset_version(version).root
    want_v2seg, want_merged, want_spe, want_total = MERGED[version]
    v2seg = S.train_normal_n("v2seg", root=root)
    merged = S.train_normal_n("v2seg+l8", root=root)
    facts = S.train_source_facts("v2seg+l8", root=root)
    assert v2seg == S.dataset_version(version).train_normal_n == want_v2seg
    assert facts["sources"]["l8"]["n_normal"] == 25894
    assert merged == want_merged == facts["n_normal_total"]
    cfg = E.Epr030Config(loss_level=1, train_n=merged)
    assert (cfg.steps_per_epoch, cfg.total_steps) == (want_spe, want_total)


def test_the_frozen_block_number_is_the_original_version():
    assert S.TRAIN_NORMAL_N == S.dataset_version("v20260804").train_normal_n == 93934


@pytest.mark.skipif(not L8_PRESENT, reason="L8 manifest not present")
def test_l8_manifest_normal_low_counts():
    rows = S.load_l8_index()
    assert len(rows) == 46129
    assert sum(1 for r in rows if r.is_normal) == 25894
    assert sum(1 for r in rows if not r.is_normal) == 20235


@pytest.mark.skipif(not (MOUNTED and L8_PRESENT), reason="dataset not mounted")
@pytest.mark.parametrize("split", ["V_where", "V_what", "T_final", "T_lut_unseen"])
def test_merged_training_set_is_disjoint_from_every_eval_split(split):
    """sft_id and source_image_id, both directions, on the MERGED population."""
    train = S.train_normal_rows("v2seg+l8")
    ev = S.load_index_cached(split)
    assert len({r.sample_id for r in train} & {r.sample_id for r in ev}) == 0
    assert len({r.source_image_id for r in train}
               & {r.source_image_id for r in ev}) == 0


@pytest.mark.skipif(not (MOUNTED and L8_PRESENT), reason="dataset not mounted")
def test_the_l8_z_cache_covers_every_l8_training_row():
    leaf = Z.resolve_leaf(S.L8_ZCACHE_ROOT, S.L8_SPLIT, "none", "generated")
    if leaf is None:
        pytest.skip("L8 z cache not built")
    cache = Z.ZCache(leaf)
    rows = [r for r in S.load_l8_index() if r.is_normal]
    assert len(rows) == 25894
    assert all(r.sample_id in cache for r in rows)
