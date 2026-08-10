"""Task card EXEC-5 -- the offline evaluation entry point, before it is queued.

Three things this file pins, each of which was a live failure mode rather than a
hypothetical:

* **the LPIPS rescale.**  ``preflight/wt_g7_lpips_closure.json`` recorded
  ``input_convention: "[-1,1] ... the caller must rescale"``.  Everything in
  :mod:`q3vl.what.metrics` is ``[0, 1]``, and a missing ``x * 2 - 1`` does not
  crash -- it silently reports the LPIPS of a half-contrast pair.  So the
  rescale is a named function with a test, not a lambda nobody can reach.
* **the 12.2 composite mask.**  Protocol 12.2 composites every arm through the
  same frozen Where mask so that C01/C02 are a clean lower bound rather than a
  mask ablation.  Under deviation D-EXEC4 there is no frozen mask, so the choice
  must be *declared*: ``--composite-mask`` has no default and every row records
  which mask it got.
* **the natural-mask cross-check.**  The natural query half is drawn from the
  supervision mask; evaluating an arm with a different one silently changes what
  "natural" means in every LUT metric.
* **where ``I_tar`` comes from.**  ``image.baked`` is the contract-sized copy of
  ``I_in`` (``q3vl/data/bake.py``), bit-identical to the member the dataset
  already loads as the input -- reading it as ``I_tar`` scores every arm against
  its own input, and the whole of protocol 12.2 turns into "how little did you
  change the image".  ``I_tar`` is the build's rendered candidate, the ``.jpg``
  member.  Measured on V_what: mean |I_in - I_tar| inside the GT mask is 6-10x
  the outside value, which is what a local edit looks like.
"""

from __future__ import annotations

import json

import pytest
import torch

from q3vl.what.scripts.evaluate_what import (
    COMPOSITE_GT,
    COMPOSITE_MASKS,
    COMPOSITE_MODEL,
    COMPOSITE_ONES,
    _composite_mask,
    lpips_closure,
    main,
)


# --- WT-G7: the LPIPS backend's input convention -----------------------------

class _RecordingNet:
    """Stands in for ``lpips.LPIPS``: records what it was actually handed."""

    def __init__(self):
        self.calls: list[tuple[torch.Tensor, torch.Tensor]] = []

    def __call__(self, a, b):
        self.calls.append((a, b))
        return torch.tensor([[[[0.25]]], [[[0.75]]]])


def test_lpips_closure_rescales_zero_one_images_to_minus_one_one():
    net = _RecordingNet()
    fn = lpips_closure(net)
    black = torch.zeros(1, 3, 4, 4)
    white = torch.ones(1, 3, 4, 4)

    out = fn(black, white)

    a, b = net.calls[0]
    assert float(a.min()) == -1.0 and float(a.max()) == -1.0, "0.0 must map to -1"
    assert float(b.min()) == 1.0 and float(b.max()) == 1.0, "1.0 must map to +1"
    # and the closure reduces to a scalar, the way image_metrics consumes it
    assert out.ndim == 0 and float(out) == pytest.approx(0.5)


def test_lpips_closure_is_not_the_identity_on_the_input_range():
    """A rescale that got dropped would leave [0,1] untouched -- catch that."""
    net = _RecordingNet()
    lpips_closure(net)(torch.full((1, 3, 2, 2), 0.5), torch.zeros(1, 3, 2, 2))
    a, _ = net.calls[0]
    assert float(a.mean()) == pytest.approx(0.0), "0.5 must map to the midpoint 0"


def test_evaluate_what_wires_the_named_closure_not_a_bare_net_call():
    """AST-level: the wiring, not just the helper, has to use the rescale.

    The helper can be perfect and still unused -- that is exactly what review
    NF-2 found for the whole evaluation library, so the caller gets its own
    assertion.
    """
    import ast
    from pathlib import Path

    from q3vl.what.scripts import evaluate_what as mod

    tree = ast.parse(Path(mod.__file__).read_text(encoding="utf-8"))
    calls = [n for n in ast.walk(tree) if isinstance(n, ast.Call)]
    named = {ast.unparse(n.func) for n in calls}
    assert "lpips_closure" in named, "main() must build lpips_fn via lpips_closure"
    # and the backend is only ever constructed inside that call's arguments
    for call in calls:
        if ast.unparse(call.func).endswith("lpips_lib.LPIPS"):
            parents = [c for c in calls
                       if ast.unparse(c.func) == "lpips_closure"
                       and any(call is sub for sub in ast.walk(c))]
            assert parents, "a raw LPIPS net must not reach lpips_fn unrescaled"


# --- protocol 12.2: which mask composes I_out --------------------------------

class _Sample:
    def __init__(self, mask=None):
        self.mask_hi = mask


def test_composite_gt_uses_the_gt_mask_for_every_arm():
    m = torch.rand(4, 4)
    got, label = _composite_mask(_Sample(m), None, composite=COMPOSITE_GT,
                                 where_source="none", device="cpu")
    assert torch.equal(got, m) and label == "gt"


def test_composite_gt_on_a_global_sample_is_labelled_not_guessed():
    """A global edit has no ROI; the row must say so rather than read 'gt'."""
    got, label = _composite_mask(_Sample(None), None, composite=COMPOSITE_GT,
                                 where_source="none", device="cpu")
    assert got is None and label == "ones_global"


def test_composite_model_reproduces_the_confounded_pre_exec5_behaviour():
    m = torch.rand(4, 4)
    # oracle arm: the model input *is* the GT mask
    _, label = _composite_mask(_Sample(None), m, composite=COMPOSITE_MODEL,
                               where_source="oracle", device="cpu")
    assert label == "gt"
    # no-where arm with no frozen checkpoint: nothing to composite with
    got, label = _composite_mask(_Sample(m), None, composite=COMPOSITE_MODEL,
                                 where_source="none", device="cpu")
    assert got is None and label == "ones", (
        "this is the confound --composite-mask exists to make visible")


def test_composite_ones_ignores_every_mask_it_is_given():
    got, label = _composite_mask(_Sample(torch.rand(4, 4)), torch.rand(4, 4),
                                 composite=COMPOSITE_ONES, where_source="oracle",
                                 device="cpu")
    assert got is None and label == "ones"


def test_composite_choices_are_exactly_the_three_declared_ones():
    assert set(COMPOSITE_MASKS) == {COMPOSITE_GT, COMPOSITE_MODEL, COMPOSITE_ONES}


# --- protocol 12.2: I_tar is the build's render, not the baked input ---------

_REC = {
    "sample_id": "sft_deadbeef",
    "source_sample_id": "batch-0000_000001_candidate_abc",
    "image": {
        "baked": {"shard": "/nfs/sft2seg/images/shards/shard-00003.tar",
                  "member": "sft_deadbeef.jpg", "offset": 17, "length": 5,
                  "size": 5, "sha256": "aa"},
        "origin": {"root": "/nfs/sft/prod-l2-local17k-20260731/batch-0000",
                   "shard": "shard-00000", "suffix": ".in.png",
                   "member": "batch-0000_000001_candidate_abc.in.png"},
    },
}


def test_target_locator_points_at_the_build_render_not_image_baked():
    from q3vl.what.data import TARGET_SUFFIX, WhatDataset

    loc = WhatDataset._target_locator(_REC)
    assert loc == {"root": _REC["image"]["origin"]["root"],
                   "source_sample_id": _REC["source_sample_id"],
                   "suffix": TARGET_SUFFIX}
    assert TARGET_SUFFIX == ".jpg"
    # the trap: image.baked is I_IN, and it is the very member the dataset loads
    assert "shard-00003" not in json.dumps(loc), (
        "image.baked is the contract-sized copy of I_in, not I_tar")


def test_target_locator_is_none_when_the_record_cannot_reach_the_build():
    from q3vl.what.data import WhatDataset

    assert WhatDataset._target_locator({"sample_id": "x"}) is None
    assert WhatDataset._target_locator(
        {"sample_id": "x", "image": {"origin": {"root": "/r"}}}) is None


def test_load_target_image_never_reads_image_baked():
    """Regression guard: the bug was one dict lookup, and it type-crashed."""
    import ast
    from pathlib import Path

    from q3vl.what import data as data_mod

    tree = ast.parse(Path(data_mod.__file__).read_text(encoding="utf-8"))
    fn = next(n for n in ast.walk(tree)
              if isinstance(n, ast.FunctionDef) and n.name == "load_target_image")
    src = ast.unparse(fn)
    assert "baked" not in src
    assert "target_locator" in src and "prepare_image" in src


# --- the CLI refuses to guess ------------------------------------------------

def _argv(monkeypatch, *args):
    monkeypatch.setattr("sys.argv", ["evaluate_what", *args])


def test_composite_mask_has_no_default(monkeypatch, tmp_path):
    _argv(monkeypatch, "--checkpoints", str(tmp_path / "what_final.pt"),
          "--out-dir", str(tmp_path / "out"), "--where-readout", "cband12",
          "--natural-mask-source", "global_uniform")
    with pytest.raises(SystemExit) as e:
        main()
    assert e.value.code == 2                       # argparse's "required" exit


def test_no_where_checkpoint_requires_an_explicit_readout(monkeypatch, tmp_path):
    _argv(monkeypatch, "--checkpoints", str(tmp_path / "what_final.pt"),
          "--out-dir", str(tmp_path / "out"), "--composite-mask", "gt",
          "--natural-mask-source", "global_uniform")
    with pytest.raises(SystemExit) as e:
        main()
    assert "--where-readout is required" in str(e.value)


def test_no_where_checkpoint_requires_an_explicit_natural_mask_source(
        monkeypatch, tmp_path):
    _argv(monkeypatch, "--checkpoints", str(tmp_path / "what_final.pt"),
          "--out-dir", str(tmp_path / "out"), "--composite-mask", "gt",
          "--where-readout", "cband12")
    with pytest.raises(SystemExit) as e:
        main()
    assert "natural_mask_source" in str(e.value)


def test_natural_mask_source_must_match_the_run_that_trained_the_checkpoint(
        monkeypatch, tmp_path):
    run = tmp_path / "C01"
    run.mkdir()
    (run / "run_setup.json").write_text(json.dumps(
        {"arm": "C01", "deviation": {"id": "D-EXEC4",
                                     "natural_mask_source": "global_uniform"}}),
        encoding="utf-8")
    _argv(monkeypatch, "--checkpoints", str(run / "what_final.pt"),
          "--out-dir", str(tmp_path / "out"), "--composite-mask", "gt",
          "--where-readout", "cband12",
          "--natural-mask-source", "oracle_gt_mask")
    with pytest.raises(SystemExit) as e:
        main()
    assert "trained with natural_mask_source='global_uniform'" in str(e.value)
