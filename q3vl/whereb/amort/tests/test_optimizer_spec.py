"""``OptimizerSpec`` / ``build_optimizer`` -- the per-arm optimizer entry.

CPU only.  The load-bearing test is the first one: the default spec must
reproduce the construction it replaced **exactly** (class, group order, group
membership, lr, weight decay, and betas left at the class default), or the four
live arms are no longer the runs their boards were measured on.
"""

from __future__ import annotations

import pytest
import torch
import torch.nn as nn

from q3vl.whereb.amort.trainer import (AmortTrainConfig, OptimizerSpec,
                                       build_optimizer)


class _Toy(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = nn.Conv2d(3, 4, 3, bias=True)        # dim 4 weight, dim 1 bias
        self.norm = nn.GroupNorm(2, 4)                   # dim 1 weight + bias
        self.ln2d = _LayerNorm2d(4)                      # vendored-norm lookalike
        self.fc = nn.Linear(4, 2)                        # dim 2 weight, dim 1 bias
        self.frozen = nn.Parameter(torch.zeros(5))
        self.frozen.requires_grad_(False)


class _LayerNorm2d(nn.Module):
    """SAM's ``common.py:31-43`` shape: a norm whose class name ends in Norm2d."""

    def __init__(self, c):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(c))
        self.bias = nn.Parameter(torch.zeros(c))


CFG = AmortTrainConfig(arm="P1", learning_rate=3e-4, weight_decay=0.01)


def _legacy(model, cfg):
    """The construction `build_optimizer` replaced, transcribed verbatim."""
    return torch.optim.AdamW(
        [
            {"params": [p for n, p in model.named_parameters()
                        if p.requires_grad and p.dim() > 1],
             "weight_decay": cfg.weight_decay},
            {"params": [p for n, p in model.named_parameters()
                        if p.requires_grad and p.dim() <= 1],
             "weight_decay": 0.0},
        ],
        lr=cfg.learning_rate,
    )


def _shape(opt):
    return [(g["lr"], g["weight_decay"], g.get("betas"),
             [tuple(p.shape) for p in g["params"]]) for g in opt.param_groups]


@pytest.mark.parametrize("spec", [None, OptimizerSpec()])
def test_default_is_bit_identical_to_the_old_hard_coded_adamw(spec):
    m = _Toy()
    got = build_optimizer(m, CFG, spec)
    want = _legacy(m, CFG)
    assert type(got) is type(want) is torch.optim.AdamW
    assert _shape(got) == _shape(want)
    assert got.param_groups[0]["betas"] == (0.9, 0.999)   # torch default, untouched


def test_frozen_parameters_never_enter_a_group():
    m = _Toy()
    opt = build_optimizer(m, CFG)
    ids = {id(p) for g in opt.param_groups for p in g["params"]}
    assert id(m.frozen) not in ids


def test_lisa_spec_adamw_betas_and_zero_decay():
    """EPR-018: AdamW, lr 3e-4, wd 0.0, betas (0.9, 0.95)."""
    opt = build_optimizer(_Toy(), CFG,
                          OptimizerSpec(type="adamw", lr=3e-4, weight_decay=0.0,
                                        betas=(0.9, 0.95)))
    assert isinstance(opt, torch.optim.AdamW)
    assert opt.param_groups[0]["betas"] == (0.9, 0.95)
    assert [g["weight_decay"] for g in opt.param_groups] == [0.0, 0.0]
    assert all(g["lr"] == 3e-4 for g in opt.param_groups)


def test_sam_spec_lr_and_decay_override_the_config():
    """EPR-019: lr 8e-4, wd 0.1, campaign grouping kept."""
    opt = build_optimizer(_Toy(), CFG,
                          OptimizerSpec(type="adamw", lr=8e-4, weight_decay=0.1))
    assert [g["weight_decay"] for g in opt.param_groups] == [0.1, 0.0]
    assert all(g["lr"] == 8e-4 for g in opt.param_groups)


def test_liif_spec_plain_adam_one_group_no_decay():
    """EPR-021: ``torch.optim.Adam(params, lr=1e-4)`` -- no grouping at all."""
    opt = build_optimizer(_Toy(), CFG,
                          OptimizerSpec(type="adam", lr=1e-4, weight_decay=0.0,
                                        grouping="none"))
    assert type(opt) is torch.optim.Adam
    assert len(opt.param_groups) == 1
    assert opt.param_groups[0]["weight_decay"] == 0.0


def test_detectron2_spec_sgd_with_norm_exempt_grouping():
    """EPR-020 / EPR-023: SGD 0.01 / momentum 0.9 / wd 1e-4, norm affine wd 0."""
    m = _Toy()
    opt = build_optimizer(m, CFG,
                          OptimizerSpec(type="sgd", lr=0.01, weight_decay=1e-4,
                                        momentum=0.9, nesterov=False,
                                        grouping="norm_bias",
                                        norm_weight_decay=0.0))
    assert type(opt) is torch.optim.SGD
    decayed, norms = opt.param_groups
    assert decayed["weight_decay"] == 1e-4 and norms["weight_decay"] == 0.0
    assert decayed["momentum"] == 0.9 and decayed["nesterov"] is False
    norm_ids = {id(p) for p in norms["params"]}
    for mod in (m.norm, m.ln2d):
        for p in mod.parameters():
            assert id(p) in norm_ids
    # detectron2 ties BIAS decay to WEIGHT_DECAY (defaults.py:577), so the
    # conv/fc biases stay in the decayed group -- unlike the campaign's "dim"
    dec_ids = {id(p) for p in decayed["params"]}
    assert id(m.conv.bias) in dec_ids and id(m.fc.bias) in dec_ids


def test_spec_validation():
    with pytest.raises(ValueError, match="unknown optimizer type"):
        OptimizerSpec(type="lion")
    with pytest.raises(ValueError, match="unknown grouping"):
        OptimizerSpec(grouping="weird")


def test_spec_is_json_serialisable_for_run_setup():
    import json

    d = OptimizerSpec(type="sgd", betas=None).to_dict()
    assert json.loads(json.dumps(d))["type"] == "sgd"
    assert OptimizerSpec(betas=(0.9, 0.95)).to_dict()["betas"] == [0.9, 0.95]
