from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
from itertools import product
from pathlib import Path
from typing import Any

import torch

from .action import K_SPLINE, R_FREE, decode_raw_action, identity_raw, raw_layout, softplus_inverse
from .m1_2_common import now_stamp, select_device, write_json
from .render import bspline_basis, identity_lut, lut_stats, softclip_identity


TARGET_MIN = 0.005
TARGET_MAX = 0.02


@dataclass(frozen=True)
class TailScaleConfig:
    g_max: float
    tail_color_scale: float
    factor_norm_mode: str


def run_tail_scale_sweep(
    output: Path,
    device_name: str = "cuda:0",
    count: int = 512,
    seed: int = 20260525,
    configs: list[TailScaleConfig] | None = None,
) -> dict[str, Any]:
    started = now_stamp()
    device = select_device(device_name)
    raw = sample_tail_raw(count, seed, device)
    configs = configs or default_sweep()
    rows = []
    for config in configs:
        stats = evaluate_tail_config(raw, config)
        rows.append(
            {
                "config": asdict(config),
                "stats": stats,
                "pass": bool(
                    TARGET_MIN <= stats["free_tail_energy"] <= TARGET_MAX
                    and stats["clipping_ratio"] <= 0.01
                    and stats["gamut_violation_ratio"] <= 0.01
                ),
            }
        )
    passing = [row for row in rows if row["pass"]]
    best = _select_best(rows)
    result = {
        "experiment": "M1.2_free_tail_scale_calibration",
        "started_at": started,
        "ended_at": now_stamp(),
        "count": count,
        "seed": seed,
        "device": str(device),
        "target": {
            "free_tail_energy_min": TARGET_MIN,
            "free_tail_energy_max": TARGET_MAX,
            "max_clipping_ratio": 0.01,
            "max_gamut_violation_ratio": 0.01,
        },
        "rows": rows,
        "best": best,
        "closed": bool(passing),
        "selected_passing_config": passing[0] if passing else None,
        "free_tail_unvalidated": not bool(passing),
    }
    write_json(output, result)
    return result


def sample_tail_raw(count: int, seed: int, device: torch.device) -> torch.Tensor:
    gen = torch.Generator(device=device).manual_seed(seed)
    layout = raw_layout()
    raw = identity_raw(count, device)
    raw[:, layout.curve_delta] = softplus_inverse(torch.tensor(1.0, device=device))
    raw[:, layout.curve_black_white] = -3.0
    raw[:, layout.hsl] = 0.0
    raw[:, layout.wb] = 0.0
    raw[:, layout.dictionary] = 0.0
    raw[:, layout.tail_gate] = 2.5 + 0.5 * torch.randn(count, R_FREE, generator=gen, device=device)
    raw[:, layout.tail_color] = 2.0 * torch.randn(count, R_FREE * 3, generator=gen, device=device)
    raw[:, layout.tail_alpha] = torch.randn(count, R_FREE * K_SPLINE, generator=gen, device=device)
    raw[:, layout.tail_beta] = torch.randn(count, R_FREE * K_SPLINE, generator=gen, device=device)
    raw[:, layout.tail_gamma] = torch.randn(count, R_FREE * K_SPLINE, generator=gen, device=device)
    return raw


def evaluate_tail_config(raw: torch.Tensor, config: TailScaleConfig) -> dict[str, float]:
    base_action = decode_raw_action(raw)
    lut_main = identity_lut(device=raw.device, dtype=raw.dtype).unsqueeze(0).repeat(raw.shape[0], 1, 1, 1, 1)
    tail = free_tail_lut_scaled(raw, config)
    pre = lut_main + tail
    final = softclip_identity(pre)
    luts = {
        "main": lut_main,
        "dict": torch.zeros_like(lut_main),
        "tail": tail,
        "pre": pre,
        "final": final,
        "identity": identity_lut(device=raw.device, dtype=raw.dtype).unsqueeze(0),
    }
    stats = lut_stats(base_action, luts)
    return {key: float(value.detach().float().mean().cpu()) for key, value in stats.items()}


def free_tail_lut_scaled(raw: torch.Tensor, config: TailScaleConfig) -> torch.Tensor:
    layout = raw_layout()
    batch = raw.shape[0]
    basis = bspline_basis(device=raw.device).to(raw.dtype)
    gate = config.g_max * torch.sigmoid(raw[:, layout.tail_gate])
    color = torch.tanh(raw[:, layout.tail_color]).view(batch, R_FREE, 3) * config.tail_color_scale
    alpha = raw[:, layout.tail_alpha].view(batch, R_FREE, K_SPLINE)
    beta = raw[:, layout.tail_beta].view(batch, R_FREE, K_SPLINE)
    gamma = raw[:, layout.tail_gamma].view(batch, R_FREE, K_SPLINE)
    u = torch.einsum("gk,brk->brg", basis, alpha)
    v = torch.einsum("gk,brk->brg", basis, beta)
    w = torch.einsum("gk,brk->brg", basis, gamma)
    u = normalize_factor(u, config.factor_norm_mode)
    v = normalize_factor(v, config.factor_norm_mode)
    w = normalize_factor(w, config.factor_norm_mode)
    return torch.einsum("br,brc,bri,brj,brk->bijkc", gate, color, u, v, w)


def normalize_factor(x: torch.Tensor, mode: str) -> torch.Tensor:
    if mode == "l2":
        return x / (x.norm(dim=-1, keepdim=True) + 1e-6)
    if mode == "sqrt_l2":
        scale = x.norm(dim=-1, keepdim=True) / (x.shape[-1] ** 0.5)
        return x / (scale + 1e-6)
    if mode == "linf":
        return x / (x.abs().amax(dim=-1, keepdim=True) + 1e-6)
    if mode == "none":
        return x
    raise ValueError(mode)


def default_sweep() -> list[TailScaleConfig]:
    g_max = (0.08, 0.16, 0.32, 0.64, 1.0)
    color = (0.1, 0.2, 0.4, 0.8, 1.2)
    modes = ("l2", "sqrt_l2", "linf", "none")
    return [TailScaleConfig(*values) for values in product(g_max, color, modes)]


def _select_best(rows: list[dict[str, Any]]) -> dict[str, Any] | None:
    if not rows:
        return None

    def key(row: dict[str, Any]):
        stats = row["stats"]
        energy = stats["free_tail_energy"]
        target_gap = 0.0 if TARGET_MIN <= energy <= TARGET_MAX else min(abs(energy - TARGET_MIN), abs(energy - TARGET_MAX))
        clip_over = max(0.0, stats["clipping_ratio"] - 0.01)
        gamut_over = max(0.0, stats["gamut_violation_ratio"] - 0.01)
        return (clip_over, gamut_over, target_gap)

    return min(rows, key=key)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run M1.2 free_tail scale calibration.")
    parser.add_argument("--output", default="m1_results/tail_scale_sweep_20260525.json")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--count", type=int, default=512)
    parser.add_argument("--seed", type=int, default=20260525)
    parser.add_argument("--smoke", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.smoke:
        args.count = min(args.count, 64)
    result = run_tail_scale_sweep(Path(args.output), args.device, args.count, args.seed)
    print(result["closed"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
