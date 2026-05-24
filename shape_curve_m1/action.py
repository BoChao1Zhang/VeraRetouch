from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn
import torch.nn.functional as F


G_LUT = 17
K_SPLINE = 10
R_FREE = 4
M_ATOMS = 16
HSL_SCALE = torch.tensor([8.0, 0.08, 0.08])
WB_SCALE = torch.tensor([0.08, 0.08])
TAIL_COLOR_SCALE = 0.08
G_MAX_TAIL = 0.06


@dataclass
class ShapeCurveAction:
    curves: torch.Tensor
    hsl: torch.Tensor
    wb: torch.Tensor
    dict_coef: torch.Tensor
    tail_gate: torch.Tensor
    tail_color: torch.Tensor
    tail_alpha: torch.Tensor
    tail_beta: torch.Tensor
    tail_gamma: torch.Tensor


@dataclass(frozen=True)
class RawLayout:
    curve_delta: slice
    curve_black_white: slice
    hsl: slice
    wb: slice
    dictionary: slice
    tail_gate: slice
    tail_color: slice
    tail_alpha: slice
    tail_beta: slice
    tail_gamma: slice
    dim: int


def raw_layout() -> RawLayout:
    start = 0
    curve_delta = slice(start, start + 3 * 7)
    start = curve_delta.stop
    curve_black_white = slice(start, start + 3 * 2)
    start = curve_black_white.stop
    hsl = slice(start, start + 8 * 3)
    start = hsl.stop
    wb = slice(start, start + 2)
    start = wb.stop
    dictionary = slice(start, start + M_ATOMS)
    start = dictionary.stop
    tail_gate = slice(start, start + R_FREE)
    start = tail_gate.stop
    tail_color = slice(start, start + R_FREE * 3)
    start = tail_color.stop
    tail_alpha = slice(start, start + R_FREE * K_SPLINE)
    start = tail_alpha.stop
    tail_beta = slice(start, start + R_FREE * K_SPLINE)
    start = tail_beta.stop
    tail_gamma = slice(start, start + R_FREE * K_SPLINE)
    start = tail_gamma.stop
    return RawLayout(
        curve_delta=curve_delta,
        curve_black_white=curve_black_white,
        hsl=hsl,
        wb=wb,
        dictionary=dictionary,
        tail_gate=tail_gate,
        tail_color=tail_color,
        tail_alpha=tail_alpha,
        tail_beta=tail_beta,
        tail_gamma=tail_gamma,
        dim=start,
    )


def softplus_inverse(x: torch.Tensor | float) -> torch.Tensor:
    x_t = torch.as_tensor(x)
    return torch.log(torch.expm1(x_t).clamp_min(1e-8))


def identity_raw(batch: int, device: torch.device | str) -> torch.Tensor:
    layout = raw_layout()
    raw = torch.zeros(batch, layout.dim, device=device)
    raw[:, layout.curve_delta] = softplus_inverse(torch.tensor(1.0, device=device))
    raw[:, layout.curve_black_white] = -8.0
    raw[:, layout.tail_gate] = -6.0
    return raw


def decode_raw_action(raw: torch.Tensor) -> ShapeCurveAction:
    layout = raw_layout()
    batch = raw.shape[0]
    deltas = raw[:, layout.curve_delta].view(batch, 3, 7)
    dy = F.softplus(deltas) + 1e-4
    interior = torch.cumsum(dy, dim=-1)
    interior = interior / interior[..., -1:].clamp_min(1e-6)
    y = torch.cat([torch.zeros_like(interior[..., :1]), interior], dim=-1)
    bw = raw[:, layout.curve_black_white].view(batch, 3, 2)
    y_min = torch.sigmoid(bw[..., 0:1]) * 0.08
    y_max = 1.0 - torch.sigmoid(bw[..., 1:2]) * 0.08
    curves = y_min + y * (y_max - y_min)

    hsl_scale = HSL_SCALE.to(raw.device, raw.dtype)
    wb_scale = WB_SCALE.to(raw.device, raw.dtype)
    hsl = torch.tanh(raw[:, layout.hsl]).view(batch, 8, 3) * hsl_scale.view(1, 1, 3)
    wb = torch.tanh(raw[:, layout.wb]) * wb_scale.view(1, 2)
    dict_coef = torch.tanh(raw[:, layout.dictionary])
    tail_gate = G_MAX_TAIL * torch.sigmoid(raw[:, layout.tail_gate])
    tail_color = torch.tanh(raw[:, layout.tail_color]).view(batch, R_FREE, 3) * TAIL_COLOR_SCALE
    tail_alpha = raw[:, layout.tail_alpha].view(batch, R_FREE, K_SPLINE)
    tail_beta = raw[:, layout.tail_beta].view(batch, R_FREE, K_SPLINE)
    tail_gamma = raw[:, layout.tail_gamma].view(batch, R_FREE, K_SPLINE)
    return ShapeCurveAction(
        curves=curves,
        hsl=hsl,
        wb=wb,
        dict_coef=dict_coef,
        tail_gate=tail_gate,
        tail_color=tail_color,
        tail_alpha=tail_alpha,
        tail_beta=tail_beta,
        tail_gamma=tail_gamma,
    )


def make_raw_parameter(
    batch: int,
    device: torch.device | str,
    include_dictionary: bool,
    include_free_tail: bool,
) -> nn.Parameter:
    raw = identity_raw(batch, device)
    if include_dictionary:
        raw[:, raw_layout().dictionary] = 0.0
    if include_free_tail:
        layout = raw_layout()
        raw[:, layout.tail_gate] = -3.0
        raw[:, layout.tail_color] = 0.01 * torch.randn(batch, R_FREE * 3, device=device)
        raw[:, layout.tail_alpha] = 0.05 * torch.randn(batch, R_FREE * K_SPLINE, device=device)
        raw[:, layout.tail_beta] = 0.05 * torch.randn(batch, R_FREE * K_SPLINE, device=device)
        raw[:, layout.tail_gamma] = 0.05 * torch.randn(batch, R_FREE * K_SPLINE, device=device)
    return nn.Parameter(raw)


def mask_raw_for_config(raw: torch.Tensor, include_dictionary: bool, include_free_tail: bool) -> torch.Tensor:
    layout = raw_layout()
    masked = raw.clone()
    if not include_dictionary:
        masked[:, layout.dictionary] = 0.0
    if not include_free_tail:
        masked[:, layout.tail_gate] = -20.0
        masked[:, layout.tail_color] = 0.0
        masked[:, layout.tail_alpha] = 0.0
        masked[:, layout.tail_beta] = 0.0
        masked[:, layout.tail_gamma] = 0.0
    return masked


def action_main_l1(pred: ShapeCurveAction, target: ShapeCurveAction) -> torch.Tensor:
    curve = (pred.curves - target.curves).abs().mean()
    hsl_scale = HSL_SCALE.to(pred.hsl.device, pred.hsl.dtype).view(1, 1, 3)
    hsl = ((pred.hsl - target.hsl) / hsl_scale).abs().mean()
    wb_scale = WB_SCALE.to(pred.wb.device, pred.wb.dtype).view(1, 2)
    wb = ((pred.wb - target.wb) / wb_scale).abs().mean()
    return (curve + hsl + wb) / 3.0


def action_vector_mae(pred: ShapeCurveAction, target: ShapeCurveAction) -> dict[str, float]:
    hsl_scale = HSL_SCALE.to(pred.hsl.device, pred.hsl.dtype).view(1, 1, 3)
    wb_scale = WB_SCALE.to(pred.wb.device, pred.wb.dtype).view(1, 2)
    return {
        "curve_hsl_wb_l1": float(action_main_l1(pred, target).detach().cpu()),
        "dict_coef_mae": float((pred.dict_coef - target.dict_coef).abs().mean().detach().cpu()),
        "curve_l1": float((pred.curves - target.curves).abs().mean().detach().cpu()),
        "hsl_norm_l1": float(((pred.hsl - target.hsl) / hsl_scale).abs().mean().detach().cpu()),
        "wb_norm_l1": float(((pred.wb - target.wb) / wb_scale).abs().mean().detach().cpu()),
    }
