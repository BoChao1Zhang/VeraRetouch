from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F

from .action import action_main_l1, decode_raw_action, make_raw_parameter, mask_raw_for_config
from .metrics import metric_summary
from .m1_2_common import (
    A2_THRESHOLDS,
    M12DataConfig,
    dataset_batches,
    load_clean_tier_a,
    load_lpips_model,
    now_stamp,
    passes_a2,
    select_device,
    summarize_metric_tensors,
    write_json,
)
from .render import gamut_penalty, lut_stats, render_hybrid, smoothness2_3d


@dataclass(frozen=True)
class GridFitConfig:
    name: str
    final_grid_weight: float = 0.0
    pre_grid_weight: float = 0.0
    gt_action_weight: float = 0.0
    smoothness_weight: float = 0.005
    gamut_weight: float = 1.0
    dict_l1_weight: float = 0.002
    tail_gate_weight: float = 0.01
    fit_steps: int = 100
    lr: float = 0.05


DEFAULT_GRID_CONFIGS = (
    GridFitConfig("A2_final_grid", final_grid_weight=1.0),
    GridFitConfig("A2_pre_grid", pre_grid_weight=1.0),
    GridFitConfig("A2_gt_action_aux", final_grid_weight=0.5, pre_grid_weight=0.5, gt_action_weight=0.2),
)


def fit_grid_supervised_batch(
    batch: dict[str, torch.Tensor],
    config: GridFitConfig,
    dictionary: torch.Tensor,
    rho: torch.Tensor,
    lpips_model,
) -> dict[str, torch.Tensor]:
    source = batch["source"]
    target = batch["target"]
    gt_action = decode_raw_action(batch["raw"])
    raw = make_raw_parameter(source.shape[0], source.device, include_dictionary=True, include_free_tail=True)
    opt = torch.optim.AdamW([raw], lr=config.lr)
    for _ in range(config.fit_steps):
        opt.zero_grad(set_to_none=True)
        masked = mask_raw_for_config(raw, True, True)
        action = decode_raw_action(masked)
        pred, luts = render_hybrid(source, action, dictionary, rho, True, True)
        loss = F.mse_loss(pred, target) + 0.15 * F.l1_loss(pred, target)
        if config.final_grid_weight:
            loss = loss + config.final_grid_weight * F.mse_loss(luts["final"], batch["gt_final_lut"])
        if config.pre_grid_weight:
            loss = loss + config.pre_grid_weight * F.mse_loss(luts["pre"], batch["gt_pre_lut"])
        if config.gt_action_weight:
            loss = loss + config.gt_action_weight * _action_aux_loss(action, gt_action)
        loss = loss + config.smoothness_weight * smoothness2_3d(luts["final"]).mean()
        loss = loss + config.gamut_weight * gamut_penalty(luts["pre"])
        loss = loss + config.dict_l1_weight * action.dict_coef.abs().mean()
        loss = loss + config.tail_gate_weight * action.tail_gate.mean()
        loss.backward()
        opt.step()

    with torch.no_grad():
        action = decode_raw_action(mask_raw_for_config(raw, True, True))
        pred, luts = render_hybrid(source, action, dictionary, rho, True, True)
        aux = lut_stats(action, luts)
        aux["final_grid_mse"] = (luts["final"] - batch["gt_final_lut"]).pow(2).mean(dim=(1, 2, 3, 4))
        aux["pre_grid_mse"] = (luts["pre"] - batch["gt_pre_lut"]).pow(2).mean(dim=(1, 2, 3, 4))
        aux["action_aux_l1"] = _action_aux_per_image(action, gt_action)
        return metric_summary(pred, target, aux, lpips_model)


def run_grid_supervised(
    data_config: M12DataConfig,
    output: Path,
    configs: tuple[GridFitConfig, ...] = DEFAULT_GRID_CONFIGS,
) -> dict[str, Any]:
    started = now_stamp()
    device = select_device(data_config.device)
    lpips_model = load_lpips_model(device)
    data = load_clean_tier_a(data_config)
    rows = []
    for config in configs:
        metric_rows = []
        for batch in dataset_batches(data, data_config.batch_size):
            metric_rows.append(fit_grid_supervised_batch(batch, config, data.dictionary, data.rho, lpips_model))
        metrics = summarize_metric_tensors(metric_rows)
        rows.append({"name": config.name, "config": asdict(config), "metrics": metrics, "pass": passes_a2(metrics)})
    passing = [row for row in rows if row["pass"]]
    result = {
        "experiment": "M1.2_A2_grid_supervised_inverse_fitting",
        "started_at": started,
        "ended_at": now_stamp(),
        "data_config": data_config.to_dict(),
        "thresholds": A2_THRESHOLDS,
        "gt_action_stats": data.gt_stats,
        "sampling": data.sampling,
        "rows": rows,
        "closed": bool(passing),
        "selected_passing_config": passing[0] if passing else None,
    }
    write_json(output, result)
    return result


def _action_aux_loss(pred, target) -> torch.Tensor:
    return _action_aux_per_image(pred, target).mean()


def _action_aux_per_image(pred, target) -> torch.Tensor:
    curve = (pred.curves - target.curves).abs().mean(dim=(1, 2))
    hsl = (pred.hsl - target.hsl).abs().mean(dim=(1, 2))
    wb = (pred.wb - target.wb).abs().mean(dim=1)
    dict_coef = (pred.dict_coef - target.dict_coef).abs().mean(dim=1)
    tail_gate = (pred.tail_gate - target.tail_gate).abs().mean(dim=1)
    tail_color = (pred.tail_color - target.tail_color).abs().mean(dim=(1, 2))
    return (curve + hsl + wb + dict_coef + tail_gate + tail_color) / 6.0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run M1.2 Tier A grid-supervised inverse fitting.")
    parser.add_argument("--data-root", default="~/retouching/monetGPT/data")
    parser.add_argument("--output", default="m1_results/a2_grid_supervised_20260525.json")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--count", type=int, default=128)
    parser.add_argument("--image-size", type=int, default=64)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--seed", type=int, default=20260525)
    parser.add_argument("--fit-steps", type=int, default=100)
    parser.add_argument("--lr", type=float, default=0.05)
    parser.add_argument("--smoke", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.smoke:
        args.count = min(args.count, 8)
        args.batch_size = min(args.batch_size, 4)
        args.fit_steps = min(args.fit_steps, 4)
    configs = tuple(
        GridFitConfig(
            cfg.name,
            final_grid_weight=cfg.final_grid_weight,
            pre_grid_weight=cfg.pre_grid_weight,
            gt_action_weight=cfg.gt_action_weight,
            fit_steps=args.fit_steps,
            lr=args.lr,
        )
        for cfg in DEFAULT_GRID_CONFIGS
    )
    result = run_grid_supervised(
        M12DataConfig(
            data_root=Path(args.data_root).expanduser(),
            count=args.count,
            image_size=args.image_size,
            batch_size=args.batch_size,
            seed=args.seed,
            device=args.device,
        ),
        Path(args.output),
        configs,
    )
    print(result["closed"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
