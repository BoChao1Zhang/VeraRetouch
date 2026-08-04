"""Review blocker B4: the analytic path is float32 under any autocast context.

``s_low = 3 tanh((w0 + alpha * (phi_dir @ w_dir)) / 3)`` contains a matmul, the
first entry on autocast's bf16 allow-list.  Wrapping the loss computation in
``torch.autocast`` therefore demoted the one quantity Stage-Where-B is about,
while ``evaluate_context`` (no autocast) kept it in fp32 -- so training and the
§5.6 gate were optimising and measuring two different functions, and the damage
fell on the four CBand12 arms (sigma down to 0.025) far more than on the four
Band arms, i.e. straight into the §5.3 controlled comparison.

These tests assert the fix at three levels: the primitive, the per-sample
analytic path, and a real ``compute_batch`` under autocast.
"""

from __future__ import annotations

import pytest
import torch

from q3vl.where.readout import param_shapes
from q3vl.whereb.config import ConnectorConfig, arm_config
from q3vl.whereb.fields import phi_dir_fast, predict_fields, s_from_params
from q3vl.whereb.losses import schedule_weights
from q3vl.whereb.model import WhereBModel
from q3vl.whereb.trainer import compute_batch

from .conftest import make_mock_sample, mock_batch


def _band_params(seed: int = 0) -> dict[str, torch.Tensor]:
    g = torch.Generator().manual_seed(seed)
    p = {"w0": torch.randn((), generator=g),
         "w_raw": torch.randn(71, generator=g),
         "alpha_raw": torch.tensor(1.0)}
    for k, shape in param_shapes("band").items():
        p[k] = torch.randn((), generator=g) if shape == () else torch.randn(12, generator=g)
    return p


def test_matmul_would_be_demoted_without_the_guard():
    """The premise of B4, asserted so the test cannot rot into a tautology."""
    a, b = torch.randn(8, 71), torch.randn(71)
    with torch.autocast(device_type="cpu", dtype=torch.bfloat16):
        assert (a @ b).dtype == torch.bfloat16          # this is the trap
        assert torch.tanh(a).dtype == torch.float32     # elementwise ops are safe


def test_s_from_params_stays_float32_inside_autocast():
    phi = torch.randn(40, 71)
    w0, alpha = torch.tensor(0.3), torch.tensor(2.0)
    w_dir = torch.nn.functional.normalize(torch.randn(71), dim=0)
    outside = s_from_params(phi, w0, alpha, w_dir)
    with torch.autocast(device_type="cpu", dtype=torch.bfloat16):
        inside = s_from_params(phi, w0, alpha, w_dir)
    assert outside.dtype == inside.dtype == torch.float32
    assert torch.equal(outside, inside)                 # bit-identical, not close


def test_predict_fields_is_bitwise_identical_inside_and_outside_autocast():
    torch.manual_seed(0)
    gh, gw = 6, 8
    phi = phi_dir_fast(torch.randn(gh * gw, 64), torch.rand(3, gh, gw), gh, gw)
    guide = torch.rand(1, 1, gh * 8, gw * 8)
    params = _band_params()
    outside = predict_fields(phi, params, "band", gh, gw, guide_hi=guide)
    with torch.autocast(device_type="cpu", dtype=torch.bfloat16):
        inside = predict_fields(phi, params, "band", gh, gw, guide_hi=guide)
    for k in ("s_low", "m_low", "s_hi", "m_hi"):
        assert inside[k].dtype == torch.float32, k
        assert torch.equal(outside[k], inside[k]), k


@pytest.mark.parametrize("readout", ["band", "cband12"])
def test_cband12_is_the_arm_that_bf16_would_have_hurt(readout):
    """Quantifies the §5.3 bias the bug would have introduced, and its absence."""
    torch.manual_seed(1)
    gh, gw = 6, 8
    phi = phi_dir_fast(torch.randn(gh * gw, 64), torch.rand(3, gh, gw), gh, gw)
    params = {"w0": torch.tensor(0.0), "w_raw": torch.randn(71),
              "alpha_raw": torch.tensor(1.5)}
    if readout == "cband12":
        # sigma near its 0.025 floor, and alternating payloads so the mixture is
        # actually a function of s rather than a constant
        params["sig_raw"] = torch.full((12,), -3.0)
        params["o_raw"] = torch.zeros(12)
        params["c_raw"] = torch.tensor([4.0, -4.0] * 6)
    else:
        params["mu"] = torch.tensor(0.0)
        params["h_raw"] = torch.tensor(0.0)
        params["k_raw"] = torch.tensor(0.0)
        params["pi_raw"] = torch.tensor(2.0)
    fp32 = predict_fields(phi, params, readout, gh, gw)
    # simulate the bug: feed a bf16-rounded s through the same readout
    from q3vl.where.readout import apply_readout

    rho = {k: v for k, v in params.items() if k not in ("w0", "w_raw", "alpha_raw")}
    m_bf16 = apply_readout(readout, fp32["s_low"].bfloat16().float(), rho)
    delta = float((m_bf16 - fp32["m_low"]).abs().max())
    if readout == "cband12":
        assert delta > 1e-3, "the CBand12 sensitivity this test documents vanished"
    # and the shipped path never produces that delta at all
    with torch.autocast(device_type="cpu", dtype=torch.bfloat16):
        guarded = predict_fields(phi, params, readout, gh, gw)
    assert torch.equal(guarded["m_low"], fp32["m_low"])


def test_predict_fields_require_dtype_rejects_a_wrong_precision():
    phi = torch.randn(12, 71, dtype=torch.float64)
    with pytest.raises(AssertionError, match="float32"):
        predict_fields(phi, _band_params(), "band", 3, 4,
                       require_dtype=torch.float32)


def _small_cfg(arm: str):
    c = arm_config(arm)
    return type(c)(arm=arm, seed=c.seed,
                   connector=ConnectorConfig(dim=32, n_blocks=2, n_heads=4,
                                             ffn=64, pos_bands=4))


@pytest.mark.parametrize("arm", ["W01", "W02"])
def test_compute_batch_keeps_the_analytic_path_fp32_under_autocast(arm):
    cfg = _small_cfg(arm)
    torch.manual_seed(0)
    samples = [make_mock_sample(f"s{i}", cfg.readout, seed=i) for i in range(2)]
    batch = mock_batch(samples, cfg.readout)
    model = WhereBModel(cfg)
    w = schedule_weights(0, 10)
    with torch.autocast(device_type="cpu", dtype=torch.bfloat16):
        total, stats, losses = compute_batch(model, batch, cfg, w)
    assert stats["s_dtype"] == "torch.float32"
    assert total.dtype == torch.float32
    assert all(l.mask_term.dtype == torch.float32 for l in losses)


def test_compute_batch_reports_the_effective_aux_scale():
    cfg = _small_cfg("W01")
    s_loc = make_mock_sample("loc", cfg.readout, seed=1)
    s_glob = make_mock_sample("glob", cfg.readout, seed=2, is_global=True)
    batch = mock_batch([s_loc, s_glob], cfg.readout)
    _total, stats, _ = compute_batch(WhereBModel(cfg), batch, cfg,
                                     schedule_weights(0, 10))
    assert stats["n_with_oracle"] == 1 and stats["n"] == 2
    assert stats["oracle_fraction"] == 0.5
    assert stats["aux_effective_scale"] == 1.0        # ruling D-B16
