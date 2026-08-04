"""Review blocker B2: the job entry points must be able to build their dataset.

``make_generated_context.py`` and ``make_oracle_latents.py`` each constructed
``WhereBDataset(split)`` with neither a maskview store nor a mask resolver, so
both raised on their first local sample.  These tests pin the contract that
replaced it -- ``open_dataset`` -- without touching NFS, plus the guard rails
that make a misconfigured dataset fail at construction rather than 40 minutes in.
"""

from __future__ import annotations

import ast
import inspect
from pathlib import Path

import pytest
import torch

from q3vl.train.imageproc import ImageGeometry
from q3vl.whereb.data import WhereBDataset, WhereBSample, open_dataset

SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"


def _geom(h=64, w=96) -> ImageGeometry:
    return ImageGeometry(orig_h=h, orig_w=w, out_h=h, out_w=w,
                         grid_h=h // 16, grid_w=w // 16,
                         n_visual_tokens=(h // 32) * (w // 32),
                         aspect_in=w / h, aspect_out=w / h)


def _sample(**kw) -> WhereBSample:
    base = dict(sample_id="s", image=None, geometry=_geom(), instruction="i",
                where_text="w", meta={"render_mode": "local"}, grid_h=4, grid_w=6)
    base.update(kw)
    return WhereBSample(**base)


# --- the guard rails --------------------------------------------------------

def test_dataset_refuses_to_be_built_without_a_mask_source():
    with pytest.raises(ValueError, match="need_mask"):
        WhereBDataset("V_where", index_path=Path("/nonexistent.jsonl"))


def test_dataset_signature_exposes_need_mask():
    assert "need_mask" in inspect.signature(WhereBDataset.__init__).parameters


def test_open_dataset_is_the_documented_factory():
    sig = inspect.signature(open_dataset)
    for name in ("need_mask", "maskview_root", "limit", "verify"):
        assert name in sig.parameters, name


# --- a local sample without a loaded mask must not be faked as all-ones -----

def test_local_sample_with_need_mask_false_refuses_to_invent_a_target():
    s = _sample(mask_loaded=False)
    with pytest.raises(RuntimeError, match="need_mask=False"):
        s.mask_target_hi()


def test_local_sample_with_a_loaded_mask_returns_it():
    m = torch.rand(64, 96)
    assert torch.equal(_sample(mask_hi=m).mask_target_hi(), m)


def test_global_sample_is_all_ones_even_without_a_mask():
    s = _sample(meta={"render_mode": "global"}, mask_loaded=False)
    t = s.mask_target_hi()
    assert t.shape == (64, 96) and bool((t == 1).all())


def test_local_sample_that_should_have_a_mask_but_has_none_raises():
    with pytest.raises(RuntimeError, match="no GT mask"):
        _sample(mask_hi=None, mask_loaded=True).mask_target_hi()


# --- the two scripts actually use the factory now ---------------------------

def _calls_open_dataset(path: Path) -> bool:
    tree = ast.parse(path.read_text())
    return any(isinstance(n, ast.Call) and getattr(n.func, "id", "") == "open_dataset"
               for n in ast.walk(tree))


def _bare_wherebdataset_calls(path: Path) -> list[int]:
    tree = ast.parse(path.read_text())
    out = []
    for n in ast.walk(tree):
        if isinstance(n, ast.Call) and getattr(n.func, "id", "") == "WhereBDataset":
            kw = {k.arg for k in n.keywords}
            if not (kw & {"maskviews", "mask_resolver", "need_mask"}):
                out.append(n.lineno)
    return out


@pytest.mark.parametrize("script", ["make_generated_context.py",
                                    "make_oracle_latents.py",
                                    "run_where_b.py"])
def test_job_scripts_build_their_dataset_through_the_factory(script):
    path = SCRIPTS / script
    assert _calls_open_dataset(path), f"{script} does not call open_dataset()"
    assert not _bare_wherebdataset_calls(path), (
        f"{script} still constructs WhereBDataset without a mask source "
        f"at lines {_bare_wherebdataset_calls(path)} (review blocker B2)"
    )


def test_generation_job_asks_for_no_mask_at_all():
    """159,215 pointless `.cgt.png` lookups is the other half of B2."""
    src = (SCRIPTS / "make_generated_context.py").read_text()
    assert "need_mask=False" in src


def test_training_job_checks_generated_context_coverage():
    """Nit N2: a missing genctx record must fail at startup, not after hours."""
    from q3vl.whereb.scripts.run_where_b import assert_genctx_coverage

    class FakeRef:
        def __init__(self, sid):
            self.sample_id = sid

    class FakeDS:
        refs = [FakeRef("a"), FakeRef("b")]

    class FakeStore:
        def __init__(self, ids):
            self.sample_ids = set(ids)

    info = assert_genctx_coverage(FakeDS(), FakeStore(["a", "b", "c"]), "train")
    assert info["n_missing"] == 0
    with pytest.raises(SystemExit, match="no generated"):
        assert_genctx_coverage(FakeDS(), FakeStore(["a"]), "train")


def test_shuffle_records_carry_both_halves_of_the_swap():
    src = inspect.getsource(WhereBDataset.shuffle_records)
    assert '"where"' in src and '"instruction"' in src
