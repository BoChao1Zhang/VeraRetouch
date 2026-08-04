"""Protocol 2.3 -- Where-A derivatives are published as indexed tar shards."""

from __future__ import annotations

import json

import numpy as np
import pytest
import torch

from q3vl.data.shardio import read_member
from q3vl.where.packing import (
    load_masklow, maskview_payloads, oracle_payloads, pack_maskviews, pack_oracle,
    verify_published, write_basis,
)
from q3vl.where.projector import BasisProjector


def _rows(n=4, gh=8, gw=12):
    for i in range(n):
        sid = f"sft_{i:032x}"
        low = torch.rand(gh, gw)
        hi = torch.rand(gh * 16, gw * 16)
        yield sid, low, hi, {"build": "l1", "grid": [gh, gw], "out": [gh * 16, gw * 16]}


def test_member_names_obey_the_key_policy():
    names = [p for p, _ in maskview_payloads("sft_abc123", torch.zeros(2, 2),
                                             torch.zeros(4, 4), {})]
    assert names == ["sft_abc123.masklow.npy", "sft_abc123.maskhi.png",
                     "sft_abc123.maskmeta.json"]
    # all three share the WebDataset key, so they group into one sample
    assert len({n.split(".")[0] for n in names}) == 1


def test_pack_and_verify_maskviews(tmp_path):
    rows = list(_rows())
    out = tmp_path / "maskviews"
    manifest = pack_maskviews(rows, out, source_label="test")
    assert manifest["status"] == "complete"
    assert manifest["compression"] == "none"
    assert manifest["sample_count"] == len(rows)
    assert manifest["member_count"] == 3 * len(rows)

    rep = verify_published(out, n_random=12)
    assert rep["ok"], rep
    assert rep["checksum_failures"] == []

    # index rows carry everything 2.3 asks for
    idx = sorted((out / "indexes").glob("shard-*.idx.jsonl"))
    assert idx
    row = json.loads(idx[0].read_text().splitlines()[0])
    for k in ("shard", "member", "offset_data", "length", "size", "sha256",
              "schema_version", "sample_id"):
        assert k in row

    # a random read reproduces the payload
    low_rows = [json.loads(l) for p in idx for l in p.read_text().splitlines()
                if l.strip() and json.loads(l)["suffix"] == ".masklow.npy"]
    r = low_rows[0]
    data = read_member(out, r["shard"], r["offset_data"], r["length"], r["sha256"])
    arr = load_masklow(data)
    assert arr.dtype == np.float16 and arr.shape == (8, 12)
    want = next(x for x in rows if x[0] == r["sample_id"])[1]
    assert np.allclose(arr.astype(np.float32), want.numpy(), atol=1e-3)


def test_pack_oracle(tmp_path):
    rows = [
        (f"sft_{i:032x}",
         {"band": {"status": "ok", "loss": 0.1}, "cband12": {"status": "ok", "loss": 0.2}},
         {"build": "l2"})
        for i in range(3)
    ]
    out = tmp_path / "oracle"
    manifest = pack_oracle(rows, out, source_label="test")
    assert manifest["sample_count"] == 3
    assert verify_published(out, n_random=3)["ok"]
    names = [p for p, _ in oracle_payloads("sft_x", {}, {})]
    assert names == ["sft_x.oracle.json"]


def test_write_basis(tmp_path):
    proj = BasisProjector(seed=1)
    meta = write_basis(tmp_path / "basis", "BA-3-Joint", proj,
                       {"note": "unit test", "objective": "mse"})
    assert meta["arm"] == "BA-3-Joint"
    assert meta["shape"] == [64, 1024]
    w = np.load(tmp_path / "basis" / "B.npy")
    assert w.shape == (64, 1024)
    assert np.allclose(w, proj.weight.detach().numpy())
    on_disk = json.loads((tmp_path / "basis" / "basis.json").read_text())
    assert on_disk["projector"]["digest"] == proj.digest()


def test_publication_is_atomic(tmp_path):
    """A failed pack must not leave a half-published root behind."""
    out = tmp_path / "boom"

    def bad():
        yield "sft_ok.masklow.npy", b"x" * 10
        raise RuntimeError("boom")

    with pytest.raises(RuntimeError):
        pack_maskviews(((sid, low, hi, m) for sid, low, hi, m in []), out)  # empty
    assert not out.exists()
