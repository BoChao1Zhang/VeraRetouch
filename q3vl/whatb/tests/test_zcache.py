"""The ONE z cache: schema, dtype discipline, start-up assertions, uniqueness.

The last three tests are the cross-arm ones -- they read the six arms' source
with ``ast`` and fail if a second z-cache reader, a second image loader or a
second CIELab conversion grows back (the frozen block's 共同依赖只写一份, and
the review that found five parallel readers with four different assertion sets).
"""

from __future__ import annotations

import ast
import json
from pathlib import Path

import numpy as np
import pytest
import torch

from q3vl.whatb import zcache as Z

PACKAGE = Path(Z.__file__).resolve().parent
SEG_COLOR = 151674
SEG_WHERE = 151673


def _rows(ids, *, bad_index=False):
    out = []
    for sid in ids:
        seq = [151669, 7, 151670, 151671, 9, 151672, SEG_WHERE, SEG_COLOR]
        out.append({"sample_id": sid, "split": "V_what",
                    "reply_token_ids": seq,
                    "readout_index": 0 if bad_index else len(seq) - 1,
                    "expected_ids": [SEG_COLOR],
                    "n_generated_tokens": len(seq),
                    "color_text": f"<color>{sid}</color>"})
    return out


def _write(root, split="V_what", tag="none", ids=("a", "b", "c"), *,
           checkpoint="ckpt", kind="seg_color", context="generated",
           dtype=np.float32, k=None, bad_index=False):
    shape = (len(ids), Z.Z_DIM) if k is None else (len(ids), k, Z.Z_DIM)
    z = np.arange(int(np.prod(shape)), dtype=np.float64).reshape(shape).astype(dtype)
    return Z.write_z_cache(Z.leaf_dir(root, split, tag), _rows(ids, bad_index=bad_index),
                           z, checkpoint=checkpoint, readout_kind=kind,
                           context_source=context, control_tag=tag, split=split)


# --------------------------------------------------------------------------- #
# schema + round trip
# --------------------------------------------------------------------------- #
def test_round_trip_and_the_frozen_field_list(tmp_path):
    d = _write(tmp_path)
    rows = [json.loads(l) for l in (d / "index.jsonl").read_text().splitlines()]
    # EPR-024:607 names these fields; every one of them is on every row
    for field in Z.ZCACHE_FIELDS:
        assert field in rows[0], field
    cache = Z.ZCache(d)
    assert len(cache) == 3 and "b" in cache and "zz" not in cache
    assert cache.vector("b").shape == (Z.Z_DIM,)
    assert cache.batch(["a", "c"]).shape == (2, Z.Z_DIM)
    assert cache.mean().shape == (Z.Z_DIM,)
    assert cache.k_rows == 0 and cache.control_tag == "none"
    assert cache.color_texts(2) == ["<color>a</color>", "<color>b</color>"]
    assert float(cache.vector("b")[0]) == float(Z.Z_DIM)


def test_qtok_shape_is_carried(tmp_path):
    d = _write(tmp_path, k=4)
    cache = Z.ZCache(d)
    assert cache.k_rows == 4
    assert cache.vector("a").shape == (4, Z.Z_DIM)
    assert cache.batch(["a", "b"]).shape == (2, 4, Z.Z_DIM)


def test_missing_sample_raises_instead_of_dropping(tmp_path):
    cache = Z.ZCache(_write(tmp_path))
    with pytest.raises(KeyError, match="not in the z cache"):
        cache.vector("nope")
    with pytest.raises(KeyError):
        cache.batch(["a", "nope"])


# --------------------------------------------------------------------------- #
# dtype discipline (ruling 11.1-5): REJECT, never upcast
# --------------------------------------------------------------------------- #
def test_a_bf16_cache_is_rejected_not_upcast(tmp_path):
    """The defect this module was created for: ``np.asarray(z, np.float32)``."""
    d = Z.leaf_dir(tmp_path, "V_what", "none")
    d.mkdir(parents=True)
    (d / "index.jsonl").write_text(
        json.dumps({"sample_id": "a", "row": 0}) + "\n", encoding="utf-8")
    np.save(d / "z.npy", np.zeros((1, Z.Z_DIM), dtype=np.float16))
    (d / "meta.json").write_text(json.dumps(
        {"checkpoint": "ckpt", "readout_kind": "seg_color", "split": "V_what",
         "control_tag": "none", "context_source": "generated", "n": 1}))
    with pytest.raises(Z.ZCacheDtypeError, match="fp32"):
        Z.ZCache(d)


def test_the_writer_refuses_a_narrow_dtype_but_accepts_float64(tmp_path):
    with pytest.raises(Z.ZCacheDtypeError, match="fp32"):
        _write(tmp_path / "a", dtype=np.float16)
    d = Z.write_z_cache(Z.leaf_dir(tmp_path / "b", "V_what", "none"), _rows(["a"]),
                        np.zeros((1, Z.Z_DIM), dtype=np.float64),
                        checkpoint="c", readout_kind="seg_color",
                        context_source="generated", control_tag="none",
                        split="V_what")
    assert np.load(d / "z.npy").dtype == np.float32


def test_the_writer_refuses_a_wrong_shape_or_an_unknown_tag(tmp_path):
    with pytest.raises(ValueError, match="2560"):
        Z.write_z_cache(tmp_path / "x", _rows(["a"]),
                        np.zeros((1, 8), dtype=np.float32), checkpoint="c",
                        readout_kind="seg_color", context_source="generated",
                        control_tag="none", split="V_what")
    with pytest.raises(ValueError, match="control_tag"):
        Z.write_z_cache(tmp_path / "y", _rows(["a"]),
                        np.zeros((1, Z.Z_DIM), dtype=np.float32), checkpoint="c",
                        readout_kind="seg_color", context_source="generated",
                        control_tag="nope", split="V_what")


# --------------------------------------------------------------------------- #
# the three start-up assertions (HANDOFF 4.H)
# --------------------------------------------------------------------------- #
def test_assert_belongs_to_checks_checkpoint_readout_context_and_tag(tmp_path):
    cache = Z.ZCache(_write(tmp_path))
    rec = cache.assert_belongs_to(checkpoint="ckpt", readout_kind="seg_color",
                                  context_source="generated", control_tag="none",
                                  verify_frac=1.0)
    assert rec["n"] == 3 and rec["n_verify_plan"] == 3
    with pytest.raises(AssertionError, match="checkpoint"):
        cache.assert_belongs_to(checkpoint="other", readout_kind="seg_color")
    with pytest.raises(AssertionError, match="readout_kind"):
        cache.assert_belongs_to(checkpoint="ckpt", readout_kind="im_end")
    with pytest.raises(AssertionError, match="context_source"):
        cache.assert_belongs_to(checkpoint="ckpt", readout_kind="seg_color",
                                context_source="teacher")
    with pytest.raises(AssertionError, match="control_tag"):
        cache.assert_belongs_to(checkpoint="ckpt", readout_kind="seg_color",
                                control_tag="shuffle")


def test_verify_plan_catches_a_readout_index_that_moved(tmp_path):
    """The recorded index must carry the token id the kind promises."""
    cache = Z.ZCache(_write(tmp_path, bad_index=True))
    with pytest.raises(AssertionError, match="expected"):
        cache.assert_belongs_to(checkpoint="ckpt", readout_kind="seg_color",
                                verify_frac=1.0)


# --------------------------------------------------------------------------- #
# the four caches of one split
# --------------------------------------------------------------------------- #
def test_zcachedir_opens_and_asserts_every_tag_at_construction(tmp_path):
    for tag in Z.CONTROL_TAGS:
        _write(tmp_path, tag=tag)
    d = Z.ZCacheDir(tmp_path, split="V_what", checkpoint="ckpt",
                    readout_kind="seg_color", required=Z.CONTROL_TAGS)
    assert set(d.caches) == set(Z.CONTROL_TAGS)
    assert d.z(["a", "b"], tag="shuffle").shape == (2, Z.Z_DIM)
    assert all(r["present"] for r in d.record.values())
    with pytest.raises(AssertionError, match="checkpoint"):
        Z.ZCacheDir(tmp_path, split="V_what", checkpoint="other",
                    readout_kind="seg_color")
    with pytest.raises(FileNotFoundError, match="missing"):
        Z.ZCacheDir(tmp_path, split="train", checkpoint="ckpt",
                    readout_kind="seg_color")


def test_a_control_written_as_teacher_forced_is_refused(tmp_path):
    """Frozen block 6: N1/N2/N3 must be RE-GENERATED reasoning."""
    _write(tmp_path, tag="none")
    _write(tmp_path, tag="shuffle", context="teacher")
    with pytest.raises(AssertionError, match="context_source"):
        Z.ZCacheDir(tmp_path, split="V_what", checkpoint="ckpt",
                    readout_kind="seg_color", tags=("none", "shuffle"))


def test_synthetic_is_deterministic_condition_dependent_and_never_a_readout():
    a = Z.SyntheticZCache(tag="none", split="V_what")
    b = Z.SyntheticZCache(tag="shuffle", split="V_what")
    assert torch.equal(a.vector("s1"), a.vector("s1"))
    assert not torch.equal(a.vector("s1"), a.vector("s2"))
    assert not torch.equal(a.vector("s1"), b.vector("s1"))
    assert a.facts()["synthetic"] is True
    k = Z.SyntheticZCache(tag="none", k_rows=4)
    assert k.vector("s1").shape == (4, Z.Z_DIM)
    lut = Z.SyntheticZCache(tag="none", key_fn=lambda t, s: "lut:one")
    assert torch.equal(lut.vector("s1"), lut.vector("s2"))     # learnable stand-in


# --------------------------------------------------------------------------- #
# cross-arm: exactly ONE implementation of each shared dependency
# --------------------------------------------------------------------------- #
def _module_sources():
    for path in sorted(PACKAGE.rglob("*.py")):
        if "tests" in path.parts:
            continue
        yield path, ast.parse(path.read_text(encoding="utf-8"), filename=str(path))


def test_only_one_z_cache_reader_exists():
    """Five arms grew five readers with four different assertion sets."""
    offenders = []
    for path, tree in _module_sources():
        if path.name == "zcache.py":
            continue
        for node in ast.walk(tree):
            if not isinstance(node, ast.ClassDef):
                continue
            body = ast.get_source_segment(path.read_text(encoding="utf-8"), node) or ""
            reads_z = ("z.npy" in body or "np.load" in body or "torch.load" in body)
            asserts_ckpt = "checkpoint" in body and "readout_kind" in body
            if reads_z and asserts_ckpt:
                offenders.append(f"{path.name}:{node.lineno} {node.name}")
    assert not offenders, (
        "a second z-cache reader grew back (共同依赖只写一份): " + ", ".join(offenders))


#: image / GT-alpha loaders that are NOT yet folded into ``evaldata.py``.
#: EPR-024 (and now EPR-026, which consumes the shared one for its headline
#: selection) read ``evaldata.SampleStore``; these two still carry their own,
#: each keyed to its own eval bundle.  Merging them was not part of the
#: z-cache/B1 pass -- the list is here so a *new* copy fails this test instead of
#: appearing silently.
_KNOWN_LOCAL_ALPHA_LOADERS = {
    ("run_g4d_arm.py", "EvalData"),
    ("run_qdual_arm.py", "AlphaStore"),
}


def test_no_new_image_and_alpha_loader_appears():
    offenders = []
    for path, tree in _module_sources():
        if path.name == "evaldata.py":
            continue
        src = path.read_text(encoding="utf-8")
        for node in ast.walk(tree):
            if isinstance(node, ast.ClassDef):
                body = ast.get_source_segment(src, node) or ""
                if ".maskhi.png" in body and "Image.open" in body:
                    if (path.name, node.name) in _KNOWN_LOCAL_ALPHA_LOADERS:
                        continue
                    offenders.append(f"{path.name}:{node.lineno} {node.name}")
    assert not offenders, (
        "a new image / GT-alpha loader grew back (use q3vl.whatb.evaldata): "
        + ", ".join(offenders))


def test_only_one_cielab_conversion_exists():
    """No arm may spell its own sRGB -> Lab (the two口径 would drift)."""
    offenders = []
    for path, tree in _module_sources():
        if path.name == "colorimetry.py":
            continue
        src = path.read_text(encoding="utf-8")
        for node in ast.walk(tree):
            if isinstance(node, ast.FunctionDef):
                body = ast.get_source_segment(src, node) or ""
                if "6.0 / 29.0" in body or "0.04045" in body or "_M_SRGB_TO_XYZ" in body:
                    offenders.append(f"{path.name}:{node.lineno} {node.name}")
    assert not offenders, (
        "a second CIELab conversion grew back: " + ", ".join(offenders))


# --------------------------------------------------------------------------- #
# the producer's .pt encoding (2026-08-15 offline job) -- same reader
# --------------------------------------------------------------------------- #
def _blob(root, split="V_what", tag="none", context="generated", *,
          ids=("a", "b"), checkpoint="ckpt", dtype=torch.float32, sidecar=True):
    """The exact three top-level keys ``build_z.py`` writes."""
    root = Path(root)
    root.mkdir(parents=True, exist_ok=True)
    path = Z.blob_path(root, split, tag, context)
    z = torch.arange(len(ids) * Z.Z_DIM, dtype=torch.float32).reshape(
        len(ids), Z.Z_DIM).to(dtype)
    torch.save({"meta": {"checkpoint": checkpoint, "readout_kind": "seg_color",
                         "context_source": context, "control_tag": tag,
                         "split": split, "n": len(ids), "dim": Z.Z_DIM,
                         "dtype": "float32", "seg_color_id": SEG_COLOR,
                         "reply_layout": "where_span + color_span + "
                                         "[<seg_where>, <seg_color>]",
                         "producer": "scratchpad/zcache/build_z.py"},
                "sample_ids": list(ids), "z": z}, path)
    if sidecar:
        (root / (path.name.replace(".pt", "") + ".report.json")).write_text(
            json.dumps({"rows": [{"sample_id": s, "readout_index": 7,
                                  "reply_tokens": 8} for s in ids]}),
            encoding="utf-8")
    return path


def test_the_producer_blob_reads_through_the_same_class(tmp_path):
    path = _blob(tmp_path)
    cache = Z.ZCache(path)
    assert len(cache) == 2 and "a" in cache
    assert cache.vector("b").shape == (Z.Z_DIM,)
    assert float(cache.vector("b")[0]) == float(Z.Z_DIM)
    assert cache.batch(["b", "a"]).shape == (2, Z.Z_DIM)
    rec = cache.assert_belongs_to(checkpoint="ckpt", readout_kind="seg_color",
                                  context_source="generated", control_tag="none")
    # no reply_token_ids in the blob: the absence is recorded, never a silent pass
    assert rec["n_verify_plan"] == 0 and "verify_plan_note" in rec
    assert "build_z.py" in rec["verify_plan_note"]
    with pytest.raises(AssertionError, match="checkpoint"):
        cache.assert_belongs_to(checkpoint="other", readout_kind="seg_color")


def test_the_producer_blob_is_rejected_when_it_is_not_fp32(tmp_path):
    path = _blob(tmp_path, dtype=torch.bfloat16)
    with pytest.raises(Z.ZCacheDtypeError, match="fp32"):
        Z.ZCache(path)


def test_zcachedir_finds_either_encoding(tmp_path):
    for tag in Z.CONTROL_TAGS:
        _blob(tmp_path, tag=tag)
    d = Z.ZCacheDir(tmp_path, split="V_what", checkpoint="ckpt",
                    readout_kind="seg_color", required=Z.CONTROL_TAGS)
    assert set(d.caches) == set(Z.CONTROL_TAGS)
    assert d.z(["a"], tag="const").shape == (1, Z.Z_DIM)
    # the canonical directory wins when both exist
    _write(tmp_path, tag="none")
    d2 = Z.ZCacheDir(tmp_path, split="V_what", checkpoint="ckpt",
                     readout_kind="seg_color", tags=("none",))
    assert d2.cache("none").path.is_dir()


def test_all_six_arm_seams_resolve_the_same_cache(tmp_path):
    """One cache root, six arms, both encodings -- the point of this module."""
    from q3vl.whatb.arms import carrier as A, interpc as IC
    from q3vl.whatb.readout import WhatReadoutSpec
    from q3vl.whatb.scripts import run_affonly_arm as RA
    from q3vl.whatb.scripts import run_carrier_arm as RC
    from q3vl.whatb.scripts import run_idgate_arm as RI
    from q3vl.whatb.scripts import run_interpc_arm as RP
    from q3vl.whatb.scripts.run_g4d_arm import ConditionStore
    from q3vl.whatb.scripts.run_qdual_arm import ZStore

    for tag in Z.CONTROL_TAGS:
        _blob(tmp_path, split="V_what", tag=tag, ids=("a", "b"), checkpoint="CKPT")
    _blob(tmp_path, split="train", tag="none", ids=("a", "b"), checkpoint="CKPT")

    caches, _ = RC.open_z_caches(tmp_path, "V_what", A.CarrierConfig(),
                                 checkpoint="CKPT", tags=Z.CONTROL_TAGS,
                                 required=Z.CONTROL_TAGS)
    assert sorted(caches) == sorted(Z.CONTROL_TAGS)
    assert caches["shuffle"].vector("b").shape == (Z.Z_DIM,)

    assert RA.open_z(None, root=tmp_path, tag="const", checkpoint="CKPT",
                     readout="seg_color", split="V_what").get("a").shape == (Z.Z_DIM,)
    assert len(RI.open_z_cache(tmp_path, "train", checkpoint="CKPT")) == 2
    assert RP.open_z(tmp_path, "V_what", IC.InterpcConfig(), checkpoint="CKPT",
                     required=Z.CONTROL_TAGS).z(["a", "b"],
                                                tag="irrelevant").shape == (2, Z.Z_DIM)
    assert ConditionStore(tmp_path, split="V_what", checkpoint="CKPT",
                          readout_kind="seg_color",
                          required=Z.CONTROL_TAGS).get(["a"], "none").shape == (1, Z.Z_DIM)
    assert ZStore(mode="zcache", root=tmp_path, split="V_what", checkpoint="CKPT",
                  readout=WhatReadoutSpec(), tags=Z.CONTROL_TAGS,
                  required=Z.CONTROL_TAGS).preload(
                      Z.CONTROL_TAGS)["controls"]["none"]["n"] == 2


# --------------------------------------------------------------------------- #
# the index口径 gate on the consumer side (DATA-P45 B3)
# --------------------------------------------------------------------------- #
def _write_versioned(root, version, *, split="V_what", tag="none",
                     ids=("a", "b"), checkpoint="ckpt"):
    """A cache directory that carries the口径 build_zcache.py writes."""
    z = np.zeros((len(ids), Z.Z_DIM), dtype=np.float32)
    return Z.write_z_cache(
        Z.leaf_dir(root, split, tag), _rows(ids), z, checkpoint=checkpoint,
        readout_kind="seg_color", context_source="generated", control_tag=tag,
        split=split,
        extra_meta=None if version is None else {"dataset_version": version})


def test_a_zcache_from_another_dataset_version_is_refused(tmp_path):
    from q3vl.whatb import splits as S
    from q3vl.whatb.arms import carrier as A
    from q3vl.whatb.scripts import run_carrier_arm as RC

    _write_versioned(tmp_path, "v20260804")
    cache = Z.ZCache(Z.leaf_dir(tmp_path, "V_what", "none"))

    # matching口径 -> recorded, no noise
    rec = RC.assert_zcache_dataset_version(cache, split="V_what",
                                           dataset_version="v20260804")
    assert rec == {"checked": True, "status": "match",
                   "cache_dataset_version": "v20260804",
                   "run_dataset_version": "v20260804"}

    # a different口径 -> non-zero exit, naming both sides
    with pytest.raises(SystemExit) as e:
        RC.assert_zcache_dataset_version(cache, split="V_what",
                                         dataset_version="cut-p45")
    assert "v20260804" in str(e.value) and "cut-p45" in str(e.value)

    # and through the seam the runners actually call
    with pytest.raises(SystemExit):
        RC.open_z_caches(tmp_path, "V_what", A.CarrierConfig(), checkpoint="ckpt",
                         tags=("none",), dataset_version="cut-p45")
    caches, record = RC.open_z_caches(tmp_path, "V_what", A.CarrierConfig(),
                                      checkpoint="ckpt", tags=("none",),
                                      dataset_version="v20260804")
    assert record["none"]["dataset_version_check"]["status"] == "match"
    assert set(caches) == {"none"}
    # the default is the process口径, not "whatever the cache says"
    S.use_dataset_version("cut-p45", force=True)
    with pytest.raises(SystemExit):
        RC.assert_zcache_dataset_version(cache, split="V_what")


def test_a_zcache_without_a_dataset_version_is_warned_about_not_assumed(tmp_path,
                                                                        capsys):
    from q3vl.whatb.scripts import run_carrier_arm as RC

    _write_versioned(tmp_path, None)
    cache = Z.ZCache(Z.leaf_dir(tmp_path, "V_what", "none"))
    rec = RC.assert_zcache_dataset_version(cache, split="V_what",
                                           dataset_version="cut-p45")
    assert rec["status"] == "unknown"
    assert rec["cache_dataset_version"] is None
    assert "UNKNOWN" in capsys.readouterr().err

    # a non-sft2seg split (the L8 union) has no口径 and is not checked
    assert RC.assert_zcache_dataset_version(
        cache, split="l8_train", dataset_version="cut-p45")["checked"] is False
