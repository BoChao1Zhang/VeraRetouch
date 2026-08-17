"""``geom_source="construction"``: the AMD-8 GT code on the training data path.

The hermetic half builds a two-row sidecar and drives the real
``AmortBatchBuilder`` methods over it -- bound onto a stub so the test does not
have to load a VLM to check a lookup.  The integration half is skipped unless the
real sidecar and the EPR-004 recompute board are both on disk, and then checks
the one thing that actually matters: the code the training path hands the model
is **bit-identical** to the code the recomputed capture table was scored on.  A
data path that quietly disagrees with its own measurement board is how "the GT
arm did not move" becomes unfalsifiable.
"""

from __future__ import annotations

import json
import sqlite3
from pathlib import Path

import numpy as np
import pytest
import torch

from q3vl.whereb.amort.data import (GEOM_SOURCES, AmortBatchBuilder,
                                    ConstructGeomStore, check_geom_source)
from q3vl.whereb.amort.geomparse import (GEOM_DIM, GEOM_SLOTS, describe,
                                         geom_features_from_construction)

_NAMES = [n for n, _ in GEOM_SLOTS]

RECOMPUTE = Path("/home/bc/VeraRetouch/experiments/prs/EPR-004_a1-pilot/"
                 "per_sample_recomputed.jsonl")
SIDECAR = Path("/home/bc/data/runs/where_b/construct_geometry.sqlite3")


# -- stub: the two real methods, none of the VLM ----------------------------

class _StubDataset:
    def __init__(self, records):
        self._r = list(records)

    def record(self, i):
        return self._r[i]


class _StubBuilder:
    """Carries exactly the attributes ``_geom_of``/``_construction_code`` read."""

    _construction_code = AmortBatchBuilder._construction_code
    _geom_of = AmortBatchBuilder._geom_of

    def __init__(self, store, records, *, source="construction", shuffle=False):
        self.geom_store = store
        self.dataset = _StubDataset(records)
        self.id_to_index = {r["sample_id"]: i for i, r in enumerate(records)}
        self._construct_cache = {}
        self._construct_errors = {}
        self._vrmeta_cache = {}
        self.geom_inject = True
        self.geom_source = source
        self.geom_shuffle = shuffle
        self._geom_rng = np.random.default_rng(0)
        self._geom_seen = self._geom_nonzero = 0
        self.device = torch.device("cpu")


def _sidecar(tmp_path, rows) -> ConstructGeomStore:
    path = tmp_path / "geom.sqlite3"
    db = sqlite3.connect(path)
    db.execute("CREATE TABLE candidate_geometry (candidate_id TEXT PRIMARY KEY,"
               " build_id TEXT, slot_id TEXT, slot_mode TEXT, region TEXT,"
               " raw_alpha_mean REAL, effective_alpha_mean REAL, amount REAL,"
               " geometry TEXT)")
    db.executemany("INSERT INTO candidate_geometry VALUES (?,?,?,?,?,?,?,?,?)",
                   rows)
    db.commit()
    db.close()
    return ConstructGeomStore(path)


def _record(sample_id, candidate_id, w=1000, h=1000):
    return {"sample_id": sample_id, "candidate_id": candidate_id,
            "image": {"oriented_w": w, "oriented_h": h}}


BAND = json.dumps({"Angle": 117.24, "Bottom": 0.9577, "Feather": 55.0,
                   "Flipped": "true", "Left": -1.0821, "Midpoint": 50.0,
                   "Right": 2.1179, "Roundness": 0.0, "Top": 0.3917})


# -- 1. hermetic ------------------------------------------------------------

def test_construction_branch_matches_the_function(tmp_path):
    store = _sidecar(tmp_path, [
        ("cand_b", "b", "band-0", "band", "center", 0.6, 0.59, 1.0, BAND),
    ])
    rec = _record("s_b", "cand_b", 360, 540)
    b = _StubBuilder(store, [rec])
    got = b._geom_of("", "s_b").numpy()
    want, _, _ = geom_features_from_construction(
        "band-0", json.loads(BAND), (360.0, 540.0), alpha_mean=0.59)
    assert np.array_equal(got, want)
    assert set(describe(got)) == {"shape_band", "dir_vertical", "ext_moderate"}


def test_semantic_keeps_its_shape_slot_and_zeroes_the_rest(tmp_path):
    store = _sidecar(tmp_path, [
        ("cand_s", "b", "semantic-1", "semantic", "lower", 0.01, 0.008, 1.0, None),
    ])
    b = _StubBuilder(store, [_record("s_s", "cand_s")])
    got = b._geom_of("", "s_s").numpy()
    assert describe(got) == ["shape_semantic"]
    # and nothing that looks like a region backfill
    assert not any(n.startswith("dir_") for n in describe(got))


def test_missing_row_degrades_to_zeros_not_to_region(tmp_path):
    store = _sidecar(tmp_path, [
        ("cand_b", "b", "band-0", "band", "center", 0.6, 0.59, 1.0, BAND)])
    rec = _record("s_missing", "cand_absent")
    rec["region"] = "lower left"          # present, and must stay ignored
    b = _StubBuilder(store, [rec])
    got = b._geom_of("", "s_missing").numpy()
    assert got.shape == (GEOM_DIM,)
    assert got.sum() == 0.0
    assert store.n_miss == 1


def test_lookup_is_cached_per_sample(tmp_path):
    store = _sidecar(tmp_path, [
        ("cand_b", "b", "band-0", "band", "center", 0.6, 0.59, 1.0, BAND)])
    b = _StubBuilder(store, [_record("s_b", "cand_b")])
    for _ in range(5):
        b._geom_of("", "s_b")
    assert store.n_hit == 1
    assert b._geom_seen == 5 and b._geom_nonzero == 5


def test_size_comes_from_the_record_not_from_a_default(tmp_path):
    """Aspect ratio moves the axis bucket; a dropped ``size`` is a silent lesion."""
    geom = json.dumps({"Left": -1.6, "Right": 1.6, "Top": 0.4, "Bottom": 0.6,
                       "Angle": 30.0, "Flipped": "true"})
    store = _sidecar(tmp_path, [
        ("c1", "b", "band-0", "band", "center", 0.2, 0.2, 1.0, geom),
        ("c2", "b", "band-0", "band", "center", 0.2, 0.2, 1.0, geom)])
    b = _StubBuilder(store, [_record("sq", "c1", 100, 100),
                             _record("wide", "c2", 1000, 100)])
    assert "dir_diagonal" in describe(b._geom_of("", "sq").numpy())
    assert "dir_horizontal" in describe(b._geom_of("", "wide").numpy())


def test_shuffle_control_still_applies_to_the_new_source(tmp_path):
    store = _sidecar(tmp_path, [
        ("cand_b", "b", "band-0", "band", "center", 0.6, 0.59, 1.0, BAND)])
    b = _StubBuilder(store, [_record("s_b", "cand_b")], shuffle=True)
    v = b._geom_of("", "s_b").numpy()
    plain, _, _ = geom_features_from_construction(
        "band-0", json.loads(BAND), (360.0, 540.0), alpha_mean=0.59)
    assert v.sum() == plain.sum()          # same active-bit count
    assert not np.array_equal(v, plain)    # ... and it did permute


def test_unknown_geom_source_is_rejected():
    assert GEOM_SOURCES == ("parsed", "vrmeta", "construction")
    for ok in GEOM_SOURCES:
        assert check_geom_source(ok) == ok
    for typo in ("constructionn", "Construction", "vr_meta", ""):
        with pytest.raises(ValueError, match="geom_source"):
            check_geom_source(typo)


def test_cli_choices_and_the_guard_agree():
    """One list, two places -- the campaign has shipped a drifted pair before."""
    import ast

    src = Path("/home/bc/VeraRetouch/q3vl/whereb/scripts/run_amort_arm.py")
    tree = ast.parse(src.read_text())
    found = None
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        if not any(isinstance(a, ast.Constant) and a.value == "--geom-source"
                   for a in node.args):
            continue
        for kw in node.keywords:
            if kw.arg == "choices":
                found = tuple(e.value for e in kw.value.elts)
    assert found == GEOM_SOURCES


def test_absent_sidecar_fails_loudly(tmp_path):
    with pytest.raises(FileNotFoundError, match="export_construct_geometry"):
        ConstructGeomStore(tmp_path / "nope.sqlite3")


# -- 2. integration: the training path vs the EPR-004 recompute board --------

pytestmark_integration = pytest.mark.skipif(
    not (RECOMPUTE.is_file() and SIDECAR.is_file()),
    reason="needs the real sidecar and the EPR-004 recompute board")


@pytestmark_integration
def test_matches_the_recomputed_board_bit_for_bit():
    """>=20 samples, all four families, code == the board it was scored on."""
    from q3vl.whereb.data import open_dataset

    board = [json.loads(l) for l in RECOMPUTE.open()]
    by_family: dict[str, list] = {}
    for r in board:
        by_family.setdefault(r["family"], []).append(r)
    assert set(by_family) == {"band", "linear", "radial", "semantic"}
    picked = [r for fam in sorted(by_family) for r in by_family[fam][:6]]
    assert len(picked) >= 20

    ds, _ = open_dataset("train", need_mask=True, exclude_low=True)
    rows = ds.meta_rows()
    id_to_index = {}
    wanted = {r["sample_id"] for r in picked}
    for i, row in enumerate(rows):
        sid = row.get("sample_id")
        if sid in wanted:
            id_to_index[sid] = i
    assert len(id_to_index) == len(wanted)

    store = ConstructGeomStore(SIDECAR)
    b = _StubBuilder(store, [])
    b.dataset = ds
    b.id_to_index = id_to_index

    mismatched = []
    for r in picked:
        got = b._geom_of("", r["sample_id"]).numpy()
        want = np.asarray(r["gt_construction"], dtype=np.float32)
        if not np.array_equal(got >= 0.5, want >= 0.5):
            mismatched.append((r["sample_id"], describe(got),
                               [_NAMES[k] for k, x in enumerate(want) if x]))
    assert not mismatched, mismatched
    assert store.n_miss == 0


@pytestmark_integration
def test_smoke_fifty_batches_threaded():
    """50 x 8 samples through the real sidecar, from 4 threads.

    The thread count is the point: the store holds sqlite connections, and the
    campaign has already paid for sharing one across threads once (72% of 75,544
    family lookups came back ``unknown``, with a perfectly normal-looking
    histogram).  Also pins the non-zero rate against the recompute board, so a
    lookup that silently stops resolving cannot pass as "the code is sparse".
    """
    from concurrent.futures import ThreadPoolExecutor

    from q3vl.whereb.data import open_dataset

    ds, _ = open_dataset("train", need_mask=True, exclude_low=True)
    rows = ds.meta_rows()
    local = [i for i, r in enumerate(rows) if r.get("render_mode") == "local"]
    rng = np.random.default_rng(20260812)
    pick = [local[int(k)] for k in rng.choice(len(local), size=400, replace=False)]

    store = ConstructGeomStore(SIDECAR)
    b = _StubBuilder(store, [])
    b.dataset = ds
    b.id_to_index = {rows[i]["sample_id"]: i for i in pick}
    ids = [rows[i]["sample_id"] for i in pick]

    def one_batch(k):
        return [b._geom_of("", s).numpy() for s in ids[k * 8:(k + 1) * 8]]

    with ThreadPoolExecutor(max_workers=4) as ex:
        codes = np.stack([v for batch in ex.map(one_batch, range(50))
                          for v in batch])

    assert codes.shape == (400, GEOM_DIM)
    assert store.n_miss == 0
    dir_cols = [i for i, n in enumerate(_NAMES) if n.startswith("dir_")]
    ext_cols = [i for i, n in enumerate(_NAMES) if n.startswith("ext_")]
    # board: shape 100%, dir/ext 85.6% (the semantic family declares neither)
    assert (codes.sum(axis=1) > 0).mean() == 1.0
    assert 0.78 <= (codes[:, dir_cols] >= 0.5).any(axis=1).mean() <= 0.92
    assert 0.78 <= (codes[:, ext_cols] >= 0.5).any(axis=1).mean() <= 0.92
    # and the degeneracy AMD-8 exists to remove
    assert (codes[:, _NAMES.index("dir_center")] >= 0.5).mean() < 0.4
