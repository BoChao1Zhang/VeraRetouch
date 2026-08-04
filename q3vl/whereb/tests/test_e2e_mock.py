"""Mock closed loop: W01 and W08 take real optimisation steps on synthetic data.

The synthetic GT mask is produced *from a known latent* through the same
analytic chain the model has to invert, so a falling loss means the predicted
``(w0, w_dir, alpha, rho)`` path carries signal.  A model that only learned a
constant would flat-line here, and the zero-init assertion below proves it
starts from exactly that constant.
"""

from __future__ import annotations

import pytest
import torch

from q3vl.whereb.config import ConnectorConfig, TrainConfig, arm_config
from q3vl.whereb.losses import schedule_weights
from q3vl.whereb.model import WhereBModel
from q3vl.whereb.trainer import build_optimizer, compute_batch

from .conftest import make_mock_sample, mock_batch


def _small_cfg(arm: str):
    cfg = arm_config(arm)
    return type(cfg)(
        arm=arm, seed=cfg.seed,
        connector=ConnectorConfig(dim=64, n_blocks=2, n_heads=4, ffn=128,
                                  pos_bands=8),
    )


def _run(arm: str, n_steps: int = 12, lr: float = 3e-3):
    cfg = _small_cfg(arm)
    readout = cfg.readout
    torch.manual_seed(0)
    samples = [make_mock_sample(f"s{i}", readout, seed=i) for i in range(4)]
    batch = mock_batch(samples, readout, modes=["gt", "gt", "generated", "generated"])
    model = WhereBModel(cfg)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01)
    history = []
    for step in range(n_steps):
        w = schedule_weights(step, n_steps)
        total, stats, _ = compute_batch(model, batch, cfg, w)
        opt.zero_grad(set_to_none=True)
        total.backward()
        gnorm = float(torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0))
        opt.step()
        history.append({"step": step, "loss": float(total.detach()), "grad_norm": gnorm,
                        "stage": w["stage"], **{k: v for k, v in stats.items()
                                                if k in ("soft_iou", "L_s", "L_dir")}})
    return model, history, batch, cfg


@pytest.mark.parametrize("arm", ["W01", "W08"])
def test_mock_closed_loop_decreases_a_finite_loss(arm):
    _model, hist, _batch, _cfg = _run(arm)
    assert all(torch.isfinite(torch.tensor(h["loss"])) for h in hist)
    assert all(h["grad_norm"] > 0 for h in hist)
    assert hist[-1]["loss"] < hist[0]["loss"], [h["loss"] for h in hist]
    # the schedule actually switched inside the run
    assert {h["stage"] for h in hist} == {1, 2}


@pytest.mark.parametrize("arm", ["W01", "W08"])
def test_mock_closed_loop_improves_the_oracle_auxiliaries(arm):
    _model, hist, _b, _c = _run(arm, n_steps=16)
    assert hist[-1]["L_dir"] < hist[0]["L_dir"]
    assert hist[-1]["L_s"] <= hist[0]["L_s"] * 1.05


@pytest.mark.parametrize("arm", ["W01", "W08"])
def test_zero_init_gates_make_the_first_prediction_input_independent(arm):
    cfg = _small_cfg(arm)
    model = WhereBModel(cfg).eval()
    a = [make_mock_sample(f"a{i}", cfg.readout, seed=i) for i in range(2)]
    b = [make_mock_sample(f"b{i}", cfg.readout, seed=50 + i) for i in range(2)]
    ba = mock_batch(a, cfg.readout)
    bb = mock_batch(b, cfg.readout)
    with torch.no_grad():
        oa = model(**ba.inputs)
        ob = model(**bb.inputs)
    assert torch.allclose(oa.w_raw, ob.w_raw, atol=0, rtol=0)
    assert torch.allclose(oa.w0, ob.w0, atol=0, rtol=0)
    for k in oa.rho:
        assert torch.allclose(oa.rho[k], ob.rho[k], atol=0, rtol=0)
    # ... and every sample in a batch gets the same latent at step 0
    assert torch.allclose(oa.w_raw[0], oa.w_raw[1], atol=0, rtol=0)


def test_after_training_the_prediction_depends_on_the_input():
    model, _hist, _b, cfg = _run("W01", n_steps=8)
    a = mock_batch([make_mock_sample("a", cfg.readout, seed=1)], cfg.readout)
    b = mock_batch([make_mock_sample("b", cfg.readout, seed=2)], cfg.readout)
    with torch.no_grad():
        oa, ob = model(**a.inputs), model(**b.inputs)
    assert not torch.allclose(oa.w_raw, ob.w_raw)
    gates = model.gate_values()["stream0"]
    assert any(v != 0.0 for v in gates["gate_text"] + gates["gate_vis"])


def test_null_context_is_survivable_end_to_end():
    cfg = _small_cfg("W03")
    torch.manual_seed(0)
    samples = [make_mock_sample(f"s{i}", cfg.readout, seed=i) for i in range(2)]
    batch = mock_batch(samples, cfg.readout, modes=["null", "null"])
    batch.inputs["h_where_mask"][:] = False
    model = WhereBModel(cfg)
    total, _stats, _ = compute_batch(model, batch, cfg, schedule_weights(0, 10))
    total.backward()
    assert torch.isfinite(total)
    assert all(torch.isfinite(p.grad).all() for p in model.parameters()
               if p.grad is not None)


def test_global_sample_without_an_oracle_still_contributes_a_mask_loss():
    cfg = _small_cfg("W01")
    s_loc = make_mock_sample("loc", cfg.readout, seed=1)
    s_glob = make_mock_sample("glob", cfg.readout, seed=2, is_global=True)
    batch = mock_batch([s_loc, s_glob], cfg.readout)
    model = WhereBModel(cfg)
    total, stats, losses = compute_batch(model, batch, cfg, schedule_weights(0, 10))
    assert stats["n_with_oracle"] == 1
    assert losses[1].has_oracle is False
    assert set(losses[1].parts) == {"L_mask"}
    assert torch.isfinite(total)


def test_optimizer_groups_exclude_gates_and_norms_from_weight_decay():
    model = WhereBModel(_small_cfg("W05"))
    opt = build_optimizer(model, TrainConfig(micro_batch=4))
    assert len(opt.param_groups) == 2
    assert opt.param_groups[0]["weight_decay"] == 0.01
    assert opt.param_groups[1]["weight_decay"] == 0.0
    decayed = {id(p) for p in opt.param_groups[0]["params"]}
    for name, p in model.named_parameters():
        if name.endswith(("gate_text", "gate_vis", "positions")) or p.dim() <= 1:
            assert id(p) not in decayed, name
