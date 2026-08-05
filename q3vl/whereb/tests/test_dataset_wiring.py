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
import os
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


# --- campaign bug R6: sqlite3 must be imported before torch -----------------

def _first_import_lines(path: Path) -> dict[str, int]:
    """``{top-level module: first line it is imported on}`` via AST, not regex."""
    tree = ast.parse(path.read_text())
    lines: dict[str, int] = {}
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for a in node.names:
                lines.setdefault(a.name.split(".")[0], node.lineno)
        elif isinstance(node, ast.ImportFrom) and node.module and node.level == 0:
            lines.setdefault(node.module.split(".")[0], node.lineno)
    return lines


@pytest.mark.parametrize("script", ["run_where_b.py", "make_generated_context.py",
                                    "make_oracle_latents.py"])
def test_entry_points_import_sqlite3_before_torch(script):
    """``import torch`` poisons ``import sqlite3`` in the campaign env.

    Measured 2026-08-05 on ``/home/bc/envs/q3vl_sft``::

        import sqlite3; import torch   -> fine
        import torch;   import sqlite3 -> ImportError (libstdc++ CXXABI_1.3.15)

    torch loads a libstdc++ that shadows the one ``_sqlite3``'s dependency chain
    (libicui18n) needs.  Both of this package's routes to a published shard --
    ``q3vl.data.shardio`` and ``q3vl.where.maskdata`` -- import sqlite3 at module
    level, so a torch-first process cannot open a store at all.  The guard is one
    import at the top of each entry point; this test is what stops someone
    tidying it away.  Campaign bug R6 (found by WHAT-IMPL).
    """
    lines = _first_import_lines(SCRIPTS / script)
    assert "sqlite3" in lines, f"{script} lost its sqlite3 guard (campaign bug R6)"
    assert "torch" in lines, f"{script} no longer imports torch -- retune this test"
    assert lines["sqlite3"] < lines["torch"], (
        f"{script}: sqlite3 must be imported before torch "
        f"(sqlite3 at line {lines['sqlite3']}, torch at {lines['torch']})"
    )


@pytest.mark.parametrize("script", ["run_where_b.py", "make_generated_context.py",
                                    "make_oracle_latents.py"])
def test_the_guard_precedes_every_first_party_import(script):
    """A ``q3vl.*`` import can pull torch in transitively, so the guard has to
    come before those too, not merely before the literal ``import torch``."""
    lines = _first_import_lines(SCRIPTS / script)
    first_party = {m: n for m, n in lines.items() if m == "q3vl"}
    assert first_party, script
    for mod, lineno in first_party.items():
        assert lines["sqlite3"] < lineno, (
            f"{script}: sqlite3 (line {lines['sqlite3']}) must precede "
            f"{mod} (line {lineno})"
        )


def test_the_guard_is_not_applied_to_library_modules():
    """A guard inside a library module is worse than the bug it fixes.

    ``import sqlite3`` at the top of a library makes that library unimportable in
    *any* torch-first process, instead of failing only where a store is actually
    opened.  WHAT-IMPL tried it and reverted; this pins the boundary so nobody
    re-adds it here.  ``q3vl/whereb/stores.py`` is deliberately absent from the
    list: it reaches sqlite3 through ``q3vl.data.shardio``, which is the S0-DATA
    task's module, not ours to guard.
    """
    pkg = SCRIPTS.parent
    for path in sorted(pkg.glob("*.py")):
        lines = _first_import_lines(path)
        assert "sqlite3" not in lines, (
            f"{path.name} is a library module and must not carry the guard "
            "(it would make the module unimportable after torch)"
        )


def test_importing_the_scripts_package_does_not_load_torch():
    """The guard is only reachable if the package chain has not loaded torch yet.

    ``python -m q3vl.whereb.scripts.<job>`` imports ``q3vl`` -> ``q3vl.whereb``
    -> ``q3vl.whereb.scripts`` before running the entry point's own body.  While
    ``q3vl/whereb/__init__.py`` eagerly re-exported ``.model`` (which imports
    torch), the sqlite3 guard at the top of each script was already too late and
    the job still died on ``import sqlite3`` -- measured, not theorised.  The
    re-exports are lazy (PEP 562) for exactly this reason.
    """
    import subprocess
    import sys

    env = {k: v for k, v in os.environ.items() if k != "LD_LIBRARY_PATH"}
    probe = (
        "import sys, importlib;"
        "importlib.import_module('q3vl.whereb.scripts');"
        "print('torch' in sys.modules)"
    )
    out = subprocess.run([sys.executable, "-c", probe], capture_output=True,
                         text=True, cwd=Path(__file__).resolve().parents[3], env=env)
    assert out.returncode == 0, out.stderr[-500:]
    assert out.stdout.strip() == "False", (
        "importing q3vl.whereb.scripts pulled torch in; the entry-point sqlite3 "
        "guard can no longer run first (campaign bug R6)"
    )


def test_the_lazy_re_exports_still_work():
    """Laziness must not cost the public API."""
    import q3vl.whereb as wb

    assert wb.arm_config("W01").structure == "MC8-Joint"
    assert wb.WhereBModel is not None
    assert set(wb.__all__) == {
        "ARMS", "ARM_IDS", "STRUCTURES", "ArmConfig", "TrainConfig", "arm_config",
        "WhereBModel", "WhereBOutput", "MODEL_INPUT_KEYS", "parameter_table",
    }
    with pytest.raises(AttributeError):
        wb.no_such_symbol
