"""The shared seams the six EPR-018..023 agents plug into, end to end on CPU.

Covers what ``test_arms_registry`` does not: the batch-builder seam
(``readout`` / ``pixgt`` -> ``AmortSampleInputs``), the evaluator seam
(``per_sample_row`` / ``criteria_columns``), the trainer's optimizer+scheduler
pass-through, and the entry script's flag surface.
"""

from __future__ import annotations

import inspect
import types

import pytest
import torch

from q3vl.whereb.amort import arms as A
from q3vl.whereb.amort.data import AmortBatchBuilder, AmortSampleInputs

# the whereb suite's fake tokenizer lives in its conftest, which this package
# does not inherit; import it rather than writing a second one that could drift
_WHEREB_CONFTEST = __import__("q3vl.whereb.tests.conftest", fromlist=["x"])
FakeTokenizer = _WHEREB_CONFTEST.FakeTokenizer


@pytest.fixture()
def tokenizer():
    return FakeTokenizer()


# --- AmortSampleInputs: additive fields, live arms unaffected ---------------

def _minimal_inputs(**kw):
    base = dict(
        sample_id="s", feat=torch.zeros(1, 4, 2, 3), sim=None, center=None,
        cond_h=torch.zeros(1, 3, 8), cond_mask=torch.ones(1, 3, dtype=torch.bool),
        word_ids=torch.tensor([0]), word_offsets=torch.tensor([0]),
        phi_dir=torch.zeros(6, 71), guide_hi=None, gt_low=torch.zeros(2, 3),
        gt_hi=None, gt_partner_low=None, grid_h=2, grid_w=3, is_fake=False,
        family="radial", route_semantic=False)
    base.update(kw)
    return AmortSampleInputs(**base)


def test_new_fields_default_to_none_so_the_live_arms_see_nothing():
    x = _minimal_inputs()
    assert x.h_cond is None and x.readout is None
    assert x.gt_pix is None and x.gt_pix_source == "" and x.pixgt is None


def test_new_fields_round_trip():
    x = _minimal_inputs(h_cond=torch.zeros(1, 8), gt_pix=torch.zeros(8, 12),
                        gt_pix_source="render", readout={"kind": "seg_where"})
    assert x.h_cond.shape == (1, 8) and x.gt_pix.shape == (8, 12)
    assert x.readout["kind"] == "seg_where"


# --- builder seam: _plan_for ------------------------------------------------

class _Stub:
    """Just enough of an AmortBatchBuilder to run the two new private methods."""

    _plan_for = AmortBatchBuilder._plan_for
    _record_of = AmortBatchBuilder._record_of

    def __init__(self, **kw):
        self.readout = None
        self.genctx = None
        self.color_texts = {}
        self.pixgt = None
        self.dataset = None
        self.families = {}
        self.id_to_index = {}
        self.__dict__.update(kw)


class _Sample:
    def __init__(self, sid="s1", color=None):
        self.sample_id = sid
        self.meta = {"candidate_id": "c1"}
        if color is not None:
            self.color_text = color


def test_no_readout_means_the_pre_existing_path(tokenizer):
    from q3vl.whereb.context import gt_context

    b = _Stub()
    assert b._plan_for(_Sample(), gt_context(tokenizer, "s1", "left")) is None


def test_readout_plan_uses_the_prebuilt_colour_text(tokenizer):
    from q3vl.whereb.context import gt_context
    from q3vl.whereb.readout import ReadoutBuilder, ReadoutSpec

    ro = ReadoutBuilder(tokenizer, ReadoutSpec("seg_where"))
    b = _Stub(readout=ro, color_texts={"s1": "warmer"})
    p = b._plan_for(_Sample(), gt_context(tokenizer, "s1", "left"))
    assert p.token_ids[p.start] == ro.tags.seg_where
    assert p.source == "teacher"


def test_readout_plan_reads_the_generated_colour_ids_from_the_cache(tokenizer):
    from q3vl.whereb.context import generated_context
    from q3vl.whereb.readout import ReadoutBuilder, ReadoutSpec

    color_ids = tokenizer("<color> warmer </color>")["input_ids"]

    class _Gen:
        def color_ids(self, sid):
            assert sid == "s1"
            return list(color_ids)

    ro = ReadoutBuilder(tokenizer, ReadoutSpec("color_close"))
    b = _Stub(readout=ro, genctx=_Gen())
    ids = tokenizer("<where> left </where>")["input_ids"]
    ctx = generated_context("s1", ids, tokenizer("</where>")["input_ids"][0])
    p = b._plan_for(_Sample(), ctx)
    assert p.token_ids[-len(color_ids):] == list(color_ids)
    assert p.token_ids[p.start] == ro.tags.color_close


def test_missing_colour_text_names_the_fix(tokenizer):
    from q3vl.whereb.context import gt_context
    from q3vl.whereb.readout import ReadoutBuilder, ReadoutSpec

    b = _Stub(readout=ReadoutBuilder(tokenizer, ReadoutSpec("seg_where")))
    with pytest.raises(KeyError, match="color_texts_of"):
        b._plan_for(_Sample(), gt_context(tokenizer, "s1", "left"))


def test_record_is_only_fetched_when_the_cgt_branch_is_reachable():
    from q3vl.whereb.amort.pixgt import PixGTProvider

    calls = []

    class _DS:
        def record(self, i):
            calls.append(i)
            return {"candidate_id": "c1"}

    b = _Stub(pixgt=PixGTProvider(raster_fallback="maskhi512"), dataset=_DS(),
              id_to_index={"s1": 3}, families={"s1": "radial"})
    assert b._record_of(_Sample()) is None and calls == []

    b.pixgt = PixGTProvider(raster_fallback="cgt1024")
    assert b._record_of(_Sample()) == {"candidate_id": "c1"} and calls == [3]


def test_builder_build_uses_the_plan_token_ids():
    """`build` must feed `plan.token_ids`, not `ctx.token_ids`."""
    src = inspect.getsource(AmortBatchBuilder.build)
    assert "where_ids = ctx.token_ids if plan is None else plan.token_ids" in src
    assert "readout_hidden(e.h_where, plan)" in src


def test_builder_facts_are_additive_only():
    src = inspect.getsource(AmortBatchBuilder.facts)
    assert "if self.readout is not None" in src
    assert "if self.pixgt is not None" in src


# --- evaluator seam ---------------------------------------------------------

def test_arm_row_is_empty_for_a_live_arm():
    from q3vl.whereb.amort.evaluate import _arm_row

    class _M:
        is_new_arm = False

    assert _arm_row(_M(), {}, None) == {}


def test_arm_row_calls_the_hook(monkeypatch):
    from q3vl.whereb.amort.evaluate import _arm_row

    mod = types.ModuleType("m")
    mod.ARM = "FAKEARM2"
    mod.CRITERIA = ("fake2",)
    mod.build_head = mod.forward = mod.compute_loss = (lambda *a, **k: None)
    mod.per_sample_row = lambda model, out, x: {"fake2_iou": 0.5}
    import sys

    monkeypatch.setitem(sys.modules, "m", mod)
    monkeypatch.setitem(A.ARM_MODULES, "FAKEARM2", "m")
    monkeypatch.setitem(A.ARM_CRITERIA, "FAKEARM2", ("fake2",))
    A._CACHE.pop("FAKEARM2", None)

    class _M:
        is_new_arm = True
        arm = "FAKEARM2"

    assert _arm_row(_M(), {}, None) == {"fake2_iou": 0.5}
    A._CACHE.pop("FAKEARM2", None)


def test_criteria_columns_hook_is_invoked_on_the_headline_rows():
    """It must aggregate `live_ref` -- the same rows the headline uses."""
    from q3vl.whereb.amort import evaluate

    src = inspect.getsource(evaluate.evaluate_arm)
    assert 'arm_hook(model.arm, "criteria_columns")' in src
    assert '_cc(live_ref)' in src
    assert 'board["criteria_columns"].update' in src
    # ... and it runs BEFORE the assertion that reads the column
    assert src.index('_cc(live_ref)') < src.index("assert_criteria_ran(")


def test_forward_geo_is_given_h_cond_by_both_callers():
    from q3vl.whereb.amort import evaluate, trainer

    for mod in (trainer.compute_micro_batch, evaluate.evaluate_context):
        src = inspect.getsource(mod)
        assert "h_cond=getattr(x, \"h_cond\", None)" in src, mod


# --- trainer seam -----------------------------------------------------------

def test_trainer_passes_the_spec_and_scheduler_kwargs_through(monkeypatch):
    import q3vl.whereb.amort.trainer as T

    seen = {}
    real = T.make_scheduler

    def spy(opt, total, ratio, kind, **kw):
        seen.update(kind=kind, kw=kw)
        return real(opt, total, ratio, kind, **kw)

    monkeypatch.setattr(T, "make_scheduler", spy)

    model = _FactsModel()
    cfg = T.AmortTrainConfig(arm="P1", scheduler="warmup_multistep", max_steps=10,
                             micro_batch=2, effective_batch=4)
    tr = T.AmortTrainer(model, _FakeBuilder(), None, list(range(8)), cfg,
                        _weights(), run_dir=_tmp(), device="cpu",
                        optimizer_spec=T.OptimizerSpec(type="sgd", lr=0.01,
                                                       grouping="norm_bias"),
                        scheduler_kwargs={"milestones": [5], "gamma": 0.1,
                                          "warmup_steps": 2,
                                          "warmup_factor": 1e-3})
    assert isinstance(tr.optimizer, torch.optim.SGD)
    assert seen["kind"] == "warmup_multistep"
    assert seen["kw"]["milestones"] == [5]
    s = tr.setup()
    assert s["optimizer_spec"]["type"] == "sgd"
    assert s["scheduler_kwargs"]["milestones"] == [5]


def test_trainer_defaults_record_nothing_extra():
    import q3vl.whereb.amort.trainer as T

    model = _FactsModel()
    cfg = T.AmortTrainConfig(arm="P1", max_steps=10, micro_batch=2,
                             effective_batch=4)
    tr = T.AmortTrainer(model, _FakeBuilder(), None, list(range(8)), cfg,
                        _weights(), run_dir=_tmp(), device="cpu")
    assert isinstance(tr.optimizer, torch.optim.AdamW)
    s = tr.setup()
    assert "optimizer_spec" not in s and "scheduler_kwargs" not in s


def test_grad_clipping_off_means_inf_not_zero():
    import q3vl.whereb.amort.trainer as T

    src = inspect.getsource(T.AmortTrainer.run) if hasattr(T.AmortTrainer, "run") \
        else inspect.getsource(T.AmortTrainer)
    assert 'float("inf")' in src and "_mgn if _mgn > 0" in src


class _FakeBuilder:
    genctx = None

    def facts(self):
        return {}


class _FactsModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.lin = torch.nn.Linear(4, 4)
        self.norm = torch.nn.LayerNorm(4)

    def facts(self):
        return {"arm": "P1"}


def _weights():
    from q3vl.whereb.amort.losses import LossWeights

    return LossWeights()


def _tmp():
    import tempfile
    from pathlib import Path

    return Path(tempfile.mkdtemp())


# --- entry script flag surface ---------------------------------------------

def test_run_amort_arm_exposes_the_shared_flags():
    import argparse
    import contextlib
    import io

    from q3vl.whereb.scripts.run_amort_arm import main

    # --help exits 0 after printing the parser; capture and inspect the text
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf), pytest.raises(SystemExit) as ei:
        main(["--help"])
    assert ei.value.code == 0
    txt = buf.getvalue()
    for flag in ("--cond-readout", "--readout-qtok", "--readout-nseg",
                 "--newarm-legacy-routing", "--pixgt-source", "--pixgt-fallback",
                 "--no-pixgt", "--weight-decay", "--scheduler", "--warmup-ratio",
                 "--max-grad-norm", "--want-hi"):
        assert flag in txt, flag
    for arm in A.NEW_ARMS:
        assert arm in txt, arm
    assert isinstance(argparse.ArgumentParser, type)


def test_run_amort_arm_rejects_an_unregistered_arm():
    import contextlib
    import io

    from q3vl.whereb.scripts.run_amort_arm import main

    with contextlib.redirect_stderr(io.StringIO()), pytest.raises(SystemExit):
        main(["--arm", "NOPE"])


# --- loss_preregistration.json ----------------------------------------------

class _Weights:
    uniq_iou = 0.0

    def to_dict(self):
        return {"bce": 1.0}


def test_a_live_arm_still_publishes_the_seven_term_record():
    from q3vl.whereb.scripts.run_amort_arm import _loss_preregistration

    args = types.SimpleNamespace(arm="UNIQ", seed=7)
    rec = _loss_preregistration(args, _Weights(),
                                types.SimpleNamespace(is_new_arm=False))
    assert rec["form"].startswith("L = bce*BCE_soft")
    assert rec["dice_as_target"] is False and rec["seed"] == 7


@pytest.mark.parametrize("arm", A.NEW_ARMS)
def test_every_new_arm_declares_its_own_loss_record(arm):
    """The shared file must carry the ARM's loss, not the live arms'."""
    from q3vl.whereb.scripts.run_amort_arm import _loss_preregistration

    hook = A.arm_hook(arm, "loss_preregistration")
    assert hook is not None, f"{arm} has no loss_preregistration hook"
    args = types.SimpleNamespace(arm=arm, seed=7)
    rec = _loss_preregistration(args, _Weights(),
                                types.SimpleNamespace(is_new_arm=True))
    assert rec["arm"] == arm
    assert "bce*BCE_soft + sdf*SDF_boundary" not in str(rec.get("form", ""))
    for key in ("iou_as_field_target", "iou_as_selection_target",
                "dice_as_target"):
        assert isinstance(rec[key], bool), key
    assert rec["seed"] == 7 and "provenance" in rec


def test_the_three_dice_arms_say_so():
    """SEGSAM 0.5 / SAMDEC 1.0 / CONDINST (dice IS the target)."""
    for arm in ("SEGSAM", "SAMDEC", "CONDINST"):
        rec = A.arm_hook(arm, "loss_preregistration")(None)
        assert rec["dice_as_target"] is True, arm
        assert rec["dice_red_line_waiver"], arm
    for arm in ("PRND", "LIIF", "MATTE"):
        rec = A.arm_hook(arm, "loss_preregistration")(None)
        assert rec["dice_as_target"] is False, arm


def test_a_new_arm_without_the_hook_cannot_start(monkeypatch):
    from q3vl.whereb.scripts.run_amort_arm import _loss_preregistration

    mod = A.load_arm("LIIF")
    monkeypatch.delattr(mod, "loss_preregistration")
    args = types.SimpleNamespace(arm="LIIF", seed=7)
    with pytest.raises(SystemExit, match="loss_preregistration"):
        _loss_preregistration(args, _Weights(),
                              types.SimpleNamespace(is_new_arm=True))


def test_an_incomplete_new_arm_record_cannot_start(monkeypatch):
    from q3vl.whereb.scripts.run_amort_arm import _loss_preregistration

    mod = A.load_arm("LIIF")
    monkeypatch.setattr(mod, "loss_preregistration",
                        lambda args=None: {"form": "L = ?"})
    args = types.SimpleNamespace(arm="LIIF", seed=7)
    with pytest.raises(SystemExit, match="iou_as_field_target"):
        _loss_preregistration(args, _Weights(),
                              types.SimpleNamespace(is_new_arm=True))


# --- --genctx-dir and the no-silent-teacher-fallback guard -------------------

class _OKStore:
    def __init__(self, path):
        self.path = path
        self.rows = {("s", ".genwhere.json")}


class _MissingStore:
    def __init__(self, path):
        raise FileNotFoundError(f"no manifest.json under {path}")


def test_genctx_dir_flag_exists_and_names_the_config_default():
    import contextlib
    import io

    from q3vl.whereb.scripts.run_amort_arm import main

    buf = io.StringIO()
    with contextlib.redirect_stdout(buf), pytest.raises(SystemExit):
        main(["--help"])
    txt = buf.getvalue()
    assert "--genctx-dir" in txt and "GENCTX_DIR" in txt


def test_genctx_dir_defaults_to_the_config_root_and_is_overridable(tmp_path):
    from q3vl.whereb.config import GENCTX_DIR
    from q3vl.whereb.scripts.run_amort_arm import genctx_dir_of

    from pathlib import Path

    assert genctx_dir_of(types.SimpleNamespace(genctx_dir=None)) == Path(GENCTX_DIR)
    assert genctx_dir_of(types.SimpleNamespace(genctx_dir=str(tmp_path))) == tmp_path


def test_open_genctx_opens_root_slash_split(tmp_path):
    from q3vl.whereb.scripts.run_amort_arm import open_genctx

    args = types.SimpleNamespace(arm="LIIF", train_context="mixed")
    st = open_genctx(_OKStore, tmp_path, "train", args=args, required=True)
    assert st.path == tmp_path / "train"


@pytest.mark.parametrize(
    "arm,train_context,eval_only,required",
    [
        ("LIIF", "mixed", False, True),        # the pre-registered condition
        ("SEGSAM", "mixed", False, True),
        ("LIIF", "gt", False, False),          # teacher-only row, on purpose
        ("LIIF", "shuffled", False, False),    # a ceiling row
        ("LIIF", "mixed", True, False),        # eval_only trains nothing
        ("UNIQ", "mixed", False, False),       # the live four keep the old path
        ("P1", "mixed", False, False),
    ])
def test_when_the_generated_cache_is_required(arm, train_context, eval_only,
                                              required):
    from q3vl.whereb.scripts.run_amort_arm import genctx_is_required

    args = types.SimpleNamespace(arm=arm, train_context=train_context,
                                 eval_only=eval_only)
    assert genctx_is_required(args) is required


def test_a_new_arm_refuses_the_silent_teacher_fallback(tmp_path, capsys):
    """Missing cache + --train-context mixed on a new arm = no run at all."""
    from q3vl.whereb.scripts.run_amort_arm import genctx_is_required, open_genctx

    args = types.SimpleNamespace(arm="LIIF", train_context="mixed",
                                 eval_only=False, genctx_dir=str(tmp_path))
    with pytest.raises(SystemExit) as ei:
        open_genctx(_MissingStore, tmp_path, "train", args=args,
                    required=genctx_is_required(args))
    msg = str(ei.value)
    assert "--genctx-dir" in msg and "--train-context gt" in msg
    assert "train_generated_context" in msg          # names what used to happen


def test_a_live_arm_still_warns_and_returns_none(tmp_path, capsys):
    from q3vl.whereb.scripts.run_amort_arm import genctx_is_required, open_genctx

    args = types.SimpleNamespace(arm="UNIQ", train_context="mixed",
                                 eval_only=False, genctx_dir=None)
    got = open_genctx(_MissingStore, tmp_path, "train", args=args,
                      required=genctx_is_required(args))
    assert got is None
    assert "WARN no genctx" in capsys.readouterr().out


def test_main_wires_the_guard_to_the_flag():
    """The three helpers are CALLED by main, not merely defined (the campaign's
    recurring 'defined, never wired' failure)."""
    import inspect as _inspect

    from q3vl.whereb.scripts import run_amort_arm as R

    src = _inspect.getsource(R.main)
    for name in ("genctx_dir_of(args)", "genctx_is_required(args)",
                 "open_genctx("):
        assert name in src, name


def test_genctx_checkpoint_guard_refuses_a_mismatch():
    from q3vl.whereb.scripts.run_amort_arm import _assert_genctx_checkpoint

    class _Store:
        def __init__(self, ckpt):
            self.ckpt = ckpt

        def iter_records(self):
            yield {"checkpoint": self.ckpt}

    args = types.SimpleNamespace(checkpoint="/runs/v2seg/checkpoint-4976")
    _assert_genctx_checkpoint(args, _Store("/runs/v2seg/checkpoint-4976/"), None)
    with pytest.raises(SystemExit, match="different experiment"):
        _assert_genctx_checkpoint(args, _Store("/runs/old/checkpoint-4976"), None)
