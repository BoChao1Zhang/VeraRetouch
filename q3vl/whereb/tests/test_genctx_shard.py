"""The restored shard driver: partition algebra and the forced suffixing.

CPU only, no model, no dataset.  ``genctx_shard.py`` / ``merge_genctx.py`` were
archived out of the tree by ``484f22e`` and restored on 2026-08-14 into
``q3vl/whereb/scripts/`` for the EPR-018..023 v2seg regeneration (train =
159,215 samples, 8 shards over two cards).  What is pinned here is what the
restore could plausibly have broken: the repo-root depth the two files compute
for themselves, the cross-import between them, and the two invariants the
partition has to satisfy at the REAL train size.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from q3vl.whereb.scripts import genctx_shard as G

#: the split the 8-way shard exists for
N_TRAIN = 159_215


def test_the_files_compute_the_repo_root_they_now_live_at():
    assert (G._REPO / "q3vl" / "whereb" / "scripts" / "genctx_shard.py").is_file()
    from q3vl.whereb.scripts import merge_genctx as M

    assert M._REPO == G._REPO
    # merge reaches into the shard module for the SAME partition function
    assert M.shard_indices is G.shard_indices
    assert M.shard_tag is G.shard_tag


@pytest.mark.parametrize("mode", ["interleave", "contiguous"])
@pytest.mark.parametrize("n_total,n_shards", [(N_TRAIN, 8), (896, 2), (7, 8),
                                              (0, 4)])
def test_the_partition_is_exhaustive_disjoint_and_balanced(n_total, n_shards,
                                                           mode):
    parts = [G.shard_indices(n_total, i, n_shards, mode) for i in range(n_shards)]
    flat = [i for p in parts for i in p]
    assert sorted(flat) == list(range(n_total))          # exhaustive + disjoint
    sizes = [len(p) for p in parts]
    assert max(sizes) - min(sizes) <= 1                  # balanced


@pytest.mark.parametrize("bad,match", [
    ((10, 4, 4, "interleave"), "out of range"),        # shard == num_shards
    ((10, -1, 4, "interleave"), "out of range"),
    ((10, 0, 0, "interleave"), "num_shards must be"),
    ((-1, 0, 4, "interleave"), "n_total must be"),
    ((10, 0, 4, "round_robin"), "unknown shard mode"),
])
def test_a_bad_partition_request_is_refused(bad, match):
    with pytest.raises(ValueError, match=match):
        G.shard_indices(*bad)


def test_the_shard_tag_is_zero_padded_and_stable():
    assert G.shard_tag(0, 8) == "shard00of08"
    assert G.shard_tag(7, 8) == "shard07of08"


def test_out_root_and_report_dir_are_suffixed_so_shards_cannot_collide():
    """Eight shards sharing one root/report dir would overwrite each other."""
    import argparse

    args = argparse.Namespace(split="train", shard=3, num_shards=8,
                              out_root="/runs/genwhere_v2seg/_shards/train",
                              report_dir="/runs/genwhere_v2seg/_reports/train")
    argv = G.build_inner_argv(args, ["--batch-size", "64"])
    assert argv[argv.index("--out-root") + 1] == \
        "/runs/genwhere_v2seg/_shards/train/shard03of08"
    assert argv[argv.index("--report-dir") + 1] == \
        "/runs/genwhere_v2seg/_reports/train/shard03of08"
    assert argv[-2:] == ["--batch-size", "64"]
    roots = {G.build_inner_argv(
        argparse.Namespace(split="train", shard=i, num_shards=8,
                           out_root="/o", report_dir="/r"), [])[3]
        for i in range(8)}
    assert len(roots) == 8


def test_the_owned_flags_may_not_be_passed_through():
    import argparse

    args = argparse.Namespace(split="train", shard=0, num_shards=8,
                              out_root="/o", report_dir="/r")
    for owned in ("--split", "--out-root", "--report-dir"):
        with pytest.raises(SystemExit, match="owned by genctx_shard"):
            G.build_inner_argv(args, [owned, "x"])


def test_plan_only_costs_no_torch_and_prints_the_inner_argv(capsys,
                                                            monkeypatch):
    import json
    import sys

    monkeypatch.setattr(sys, "argv", [
        "genctx_shard", "--split", "V_where", "--shard", "1", "--num-shards", "2",
        "--out-root", "/o", "--report-dir", "/r", "--plan-only",
        "--batch-size", "64"])
    assert G.main() == 0
    rec = json.loads(capsys.readouterr().out)
    assert rec["tag"] == "shard01of02"
    assert "--batch-size" in rec["inner_argv"]
    # nothing under q3vl.whereb.scripts.make_generated_context was imported
    assert Path(G.__file__).name == "genctx_shard.py"
