"""Interop with the Where-A / Where-B published shards (protocol 2.3).

These tests build *real* published datasets with the Where-A packer and read
them back through the Where-B stores, so a schema change on either side breaks a
test instead of a training run.
"""

from __future__ import annotations

import json

import numpy as np
import pytest
import torch

from q3vl.where.config import FitConfig
from q3vl.where.oracle import fit_latent
from q3vl.where.packing import pack_maskviews, pack_oracle
from q3vl.whereb.gencontext import SUFFIX as GEN_SUFFIX
from q3vl.whereb.gencontext import publish_generated
from q3vl.whereb.stores import GenContextStore, MaskViewStore, OracleStore, PublishedStore


def _fit(readout: str, seed: int = 0):
    g = torch.Generator().manual_seed(seed)
    phi = torch.randn(24, 71, generator=g, dtype=torch.float64)
    target = torch.sigmoid(phi[:, 0] * 3).clamp(0, 1)
    return fit_latent(phi, target, readout, FitConfig(n_random=1, max_iter=8))


@pytest.fixture(scope="module")
def oracle_root(tmp_path_factory):
    root = tmp_path_factory.mktemp("oracle") / "V_where"
    rows = []
    for i in range(3):
        fits = {r: _fit(r, seed=i).to_dict() for r in ("band", "cband12")}
        rows.append((f"sft_{i:04d}", fits, {"build": "l1", "winner_confidence": "normal"}))
    pack_oracle(iter(rows), root, source_label="test.oracle")
    return root


def test_oracle_store_reads_back_a_where_a_latent(oracle_root):
    st = OracleStore(oracle_root)
    assert len(st.sample_ids) == 3
    lat = st.latent("sft_0000", "band")
    assert lat is not None
    assert lat.readout == "band"
    assert lat.w_raw.shape == (71,)
    assert set(lat.rho) == {"mu", "h_raw", "k_raw", "pi_raw"}
    assert abs(float(lat.w_dir.norm()) - 1.0) < 1e-6


def test_oracle_store_round_trips_the_numbers(oracle_root):
    """The shard stores float64; the default read is float32 (the training dtype)."""
    st = OracleStore(oracle_root)
    stored = st.payload("sft_0001")["fits"]["cband12"]["latent"]
    exact = st.latent("sft_0001", "cband12", dtype=torch.float64)
    assert abs(float(exact.w0) - stored["w0"]) < 1e-12
    assert np.allclose(exact.w_raw.numpy(), np.asarray(stored["w_raw"]), atol=1e-12)
    lat = st.latent("sft_0001", "cband12")
    assert lat.w0.dtype == torch.float32
    assert abs(float(lat.w0) - stored["w0"]) < 1e-6


def test_oracle_store_returns_none_for_a_rejected_fit(tmp_path):
    root = tmp_path / "oracle"
    fit = _fit("band").to_dict()
    fit["status"] = "rejected"
    fit["reject_reason"] = "loss_above_threshold"
    pack_oracle(iter([("sft_x", {"band": fit}, {})]), root, source_label="t")
    st = OracleStore(root)
    assert st.status("sft_x", "band") == "rejected"
    assert st.latent("sft_x", "band") is None                 # never a zero vector
    assert st.latent("sft_x", "band", require_ok=False) is not None


def test_oracle_store_coverage_report(oracle_root):
    st = OracleStore(oracle_root)
    cov = st.coverage(["sft_0000", "sft_0001", "missing"], "band")
    assert cov["n_requested"] == 3 and cov["n_present"] == 2
    assert abs(cov["present_rate"] - 2 / 3) < 1e-9


def test_maskview_store_reads_both_projections(tmp_path):
    root = tmp_path / "mv"
    low = np.random.RandomState(0).rand(4, 6).astype(np.float32)
    hi = np.random.RandomState(1).rand(64, 96).astype(np.float32)
    pack_maskviews(iter([("sft_a", low, hi, {"build": "l2"})]), root,
                   source_label="t")
    st = MaskViewStore(root)
    assert st.mask_low("sft_a").shape == (4, 6)
    assert st.mask_hi("sft_a").shape == (64, 96)
    assert float(st.mask_hi("sft_a").max()) <= 1.0
    assert st.meta("sft_a")["build"] == "l2"
    assert np.allclose(st.mask_low("sft_a").numpy(), low.astype(np.float16), atol=1e-3)


def test_published_store_refuses_an_unpublished_directory(tmp_path):
    with pytest.raises(FileNotFoundError, match="manifest"):
        PublishedStore(tmp_path / "nothing")


def test_published_store_refuses_an_incomplete_manifest(tmp_path):
    root = tmp_path / "half"
    (root / "indexes").mkdir(parents=True)
    (root / "manifest.json").write_text(json.dumps({"status": "partial"}))
    with pytest.raises(RuntimeError, match="atomically"):
        PublishedStore(root)


def test_gen_context_store_round_trip(tmp_path):
    root = tmp_path / "gen"
    records = [
        {"schema_version": "q3vl.where_b.genwhere/1", "sample_id": "sft_a",
         "generated_ids": [10, 11, 12], "generated_text": "<where>x</where>",
         "where_ids": [10, 11, 12], "format_failure": False, "truncated": False,
         "stop_reason": "closed"},
        {"schema_version": "q3vl.where_b.genwhere/1", "sample_id": "sft_b",
         "generated_ids": [10] * 200, "generated_text": "runaway",
         "where_ids": [10] * 96, "format_failure": True, "truncated": True,
         "stop_reason": "no_close_tag"},
    ]
    publish_generated(iter(records), root, "V_where")
    st = GenContextStore(root)
    assert st.record("sft_a")["generated_ids"] == [10, 11, 12]
    s = st.summary()
    assert s["n"] == 2
    assert abs(s["format_failure_rate"] - 0.5) < 1e-9
    assert s["stop_reasons"]["no_close_tag"] == 1
    assert st.has("sft_b", GEN_SUFFIX)


def test_published_store_verifies_checksums(oracle_root):
    st = OracleStore(oracle_root, verify=True)
    data = st.read("sft_0000", ".oracle.json")
    assert json.loads(data)["sample_id"] == "sft_0000"
    with pytest.raises(KeyError):
        st.read("does_not_exist", ".oracle.json")
