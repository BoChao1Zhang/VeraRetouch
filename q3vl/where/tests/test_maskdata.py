"""GT mask source: eligibility, projection, and (if the NFS build is mounted)
a live locate-and-read of the real ``.cgt.png``."""

from __future__ import annotations

import json

import numpy as np
import pytest
import torch

from q3vl.where.config import HEADLINE_WINNER_CONFIDENCE, LOCAL_BUILDS, SFT2SEG_ROOT
from q3vl.where.maskdata import (
    MaskResolver, aspect_check, eligibility, mask_stats, mask_views, split_index_path,
)

HAVE_DATA = split_index_path("V_where").exists()


def _record(**kw):
    rec = {
        "sample_id": "sft_x", "build": "l1", "render_mode": "local",
        "winner_confidence": "normal", "source_sample_id": "batch-0000_000000_candidate_x",
        "image": {"out_h": 512, "out_w": 768, "oriented_h": 1024, "oriented_w": 1536,
                  "origin": {"root": "/nowhere"}},
    }
    rec.update(kw)
    return rec


def test_eligibility_keeps_only_local_builds():
    for b in LOCAL_BUILDS:
        assert eligibility(_record(build=b))[0]
    for b in ("g1", "g2", "g3", "g4"):
        ok, reason = eligibility(_record(build=b))
        assert not ok and reason.endswith("not_local")


def test_eligibility_keeps_low_confidence_by_default():
    """D1 as adjudicated 2026-08-05: `winner_confidence` ranks which candidate
    won, the Where-A label is where the region is -- orthogonal, so `low` trains.
    The headline population is narrowed at *reporting* time instead."""
    assert eligibility(_record(winner_confidence="low"))[0]
    ok, reason = eligibility(_record(winner_confidence="low"), exclude_low=True)
    assert not ok and reason == "winner_confidence_low"
    assert HEADLINE_WINNER_CONFIDENCE == ("normal",)


def test_eligibility_requires_a_locator_and_geometry():
    r = _record()
    r["image"] = {"out_h": 512, "out_w": 768}
    assert eligibility(r)[1] == "no_origin_locator"
    r2 = _record()
    r2["image"] = {"origin": {"root": "/x"}}
    assert eligibility(r2)[1] == "no_geometry"


def test_mask_views_shapes_and_area_average():
    mask = np.zeros((1024, 1536), dtype=np.float32)
    mask[:512] = 1.0
    hi, low = mask_views(mask, 512, 768)
    assert hi.shape == (512, 768)
    assert low.shape == (32, 48)
    assert float(hi[:256].mean()) == pytest.approx(1.0, abs=1e-4)
    assert float(hi[256:].mean()) == pytest.approx(0.0, abs=1e-4)
    assert float(low.mean()) == pytest.approx(0.5, abs=1e-3)
    assert float(low.min()) >= 0.0 and float(low.max()) <= 1.0


def test_mask_views_preserve_soft_edges():
    ramp = np.tile(np.linspace(0, 1, 1536, dtype=np.float32), (1024, 1))
    hi, low = mask_views(ramp, 512, 768)
    assert 0.05 < float(((low > 0.04) & (low < 0.96)).float().mean())
    assert float(low[:, 0].mean()) < float(low[:, -1].mean())


def test_mask_stats_flags_degenerate():
    assert mask_stats(torch.zeros(8, 8))["degenerate"]
    assert not mask_stats(torch.rand(8, 8))["degenerate"]


def test_aspect_check():
    rec = _record()
    good = aspect_check(rec, np.zeros((1024, 1536), dtype=np.float32))
    assert good["ok"] and good["rel_error"] < 1e-9
    bad = aspect_check(rec, np.zeros((1024, 1024), dtype=np.float32))
    assert not bad["ok"]


@pytest.mark.skipif(not HAVE_DATA, reason="sft2seg NFS dataset not mounted")
def test_live_resolve_and_load():
    """Locate a real ``.cgt.png`` from a real record and check the contract."""
    from q3vl.train.shards import ShardIndex, ShardStore

    index = ShardIndex.load(split_index_path("V_where"))
    store = ShardStore("/", verify="checksum")
    resolver = MaskResolver()
    n = 0
    for ref in index.samples:
        if ref.meta.get("build") not in LOCAL_BUILDS:
            continue
        rec = json.loads(store.read(ref.members["record"]).decode("utf-8"))
        if not eligibility(rec)[0]:
            continue
        mref = resolver.resolve(rec)
        assert mref.member.member.endswith(".cgt.png")
        assert mref.member.member.startswith(rec["source_sample_id"])
        assert mref.member.checksum and mref.member.checksum.startswith("sha256:")
        mask = resolver.load(mref)                 # checksum-verified read
        assert mask.dtype == np.float32
        assert 0.0 <= float(mask.min()) and float(mask.max()) <= 1.0
        ac = aspect_check(rec, mask)
        assert ac["ok"], ac
        hi, low = mask_views(mask, rec["image"]["out_h"], rec["image"]["out_w"])
        assert low.shape == (rec["image"]["out_h"] // 16, rec["image"]["out_w"] // 16)
        n += 1
        if n >= 5:
            break
    assert n == 5
    resolver.close()
    store.close()


@pytest.mark.skipif(not SFT2SEG_ROOT.exists(), reason="sft2seg NFS dataset not mounted")
def test_split_files_are_the_frozen_authority():
    for split in ("train", "V_where", "V_what", "T_final", "T_lut_unseen"):
        assert split_index_path(split).exists(), split


# --- REVIEW-impl-WhereA N-3: the published mask views now have a consumer ---

def test_maskview_store_roundtrip(tmp_path):
    from q3vl.where.maskdata import MaskViewStore
    from q3vl.where.packing import pack_maskviews

    gh, gw = 8, 12
    rows = []
    for i in range(3):
        sid = f"sft_{i:032x}"
        low = torch.rand(gh, gw)
        hi = torch.rand(gh * 16, gw * 16)
        rows.append((sid, low, hi, {"build": "l1", "grid": [gh, gw],
                                    "aspect": {"ok": True},
                                    "mask_raw_shape": [1024, 1536]}))
    out = tmp_path / "maskviews"
    pack_maskviews(rows, out, source_label="test")

    store = MaskViewStore(out)
    assert len(store) == 3
    assert rows[0][0] in store
    assert "sft_nope" not in store

    hi, low, meta = store.get(rows[1][0])
    assert low.shape == (gh, gw) and hi.shape == (gh * 16, gw * 16)
    # float16 on disk, so compare at that precision
    assert torch.allclose(low, rows[1][1], atol=1e-3)
    assert torch.allclose(hi, rows[1][2], atol=1.0 / 255 + 1e-6)
    assert meta["build"] == "l1" and meta["schema_version"].startswith("q3vl.where_a.maskview")
    assert store.n_hits == 1

    assert store.get("sft_absent") is None          # clean fallback, not a crash
    assert store.n_misses == 1
    store.close()


def test_maskview_store_rejects_an_empty_root(tmp_path):
    from q3vl.where.maskdata import MaskLookupError, MaskViewStore
    (tmp_path / "indexes").mkdir(parents=True)
    with pytest.raises(MaskLookupError):
        MaskViewStore(tmp_path)


def test_maskview_root_can_be_the_parent_of_several_splits(tmp_path):
    """A calibration run reads `train` for the epoch and `V_where` for the final
    refit.  Pointing at the parent keeps both on published shards; a train-only
    root would silently send the evaluation split back to decoding .cgt.png."""
    from q3vl.where.packing import pack_maskviews
    from q3vl.where.pipeline import WhereADataSource

    gh, gw = 8, 12
    for split, n in (("train", 3), ("V_where", 2)):
        rows = [(f"sft_{split[:2]}{i:030x}", torch.rand(gh, gw),
                 torch.rand(gh * 16, gw * 16), {"build": "l1", "grid": [gh, gw]})
                for i in range(n)]
        pack_maskviews(rows, tmp_path / "maskviews" / split, source_label="test")

    src = WhereADataSource(None, None, device="cpu", verify="none",
                           maskview_root=str(tmp_path / "maskviews"))
    assert len(src._store_for("train")) == 3
    assert len(src._store_for("V_where")) == 2
    assert src._store_for("T_final") is None          # not packed -> clean fallback
    stats = src.maskview_stats()
    assert stats["train"]["n_samples"] == 3
    assert stats["T_final"] == {"unavailable": True}
    src.close()


def test_maskview_root_can_still_be_a_single_split(tmp_path):
    """Backwards compatible: pointing straight at one split's shards works."""
    from q3vl.where.packing import pack_maskviews
    from q3vl.where.pipeline import WhereADataSource

    rows = [(f"sft_{i:032x}", torch.rand(4, 4), torch.rand(64, 64), {}) for i in range(2)]
    root = tmp_path / "one"
    pack_maskviews(rows, root, source_label="test")
    src = WhereADataSource(None, None, device="cpu", verify="none", maskview_root=str(root))
    assert src.maskviews is not None and len(src.maskviews) == 2
    src.close()
