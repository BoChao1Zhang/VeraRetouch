from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
from itertools import product
from pathlib import Path
from typing import Any

import torch

from .fit import FitConfig, fit_batch
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


@dataclass(frozen=True)
class A2SweepConfig:
    gamut_weight: float
    smoothness_weight: float
    dict_l1_weight: float
    fit_steps: int
    lr: float

    @property
    def name(self) -> str:
        return (
            f"A2-g{self.gamut_weight:g}-s{self.smoothness_weight:g}-"
            f"d{self.dict_l1_weight:g}-steps{self.fit_steps}-lr{self.lr:g}"
        )


def run_a2_sweep(
    data_config: M12DataConfig,
    output: Path,
    sweep: list[A2SweepConfig],
) -> dict[str, Any]:
    started = now_stamp()
    device = select_device(data_config.device)
    lpips_model = load_lpips_model(device)
    data = load_clean_tier_a(data_config)
    rows = []
    for item in sweep:
        fit_config = FitConfig(
            item.name,
            include_dictionary=True,
            include_free_tail=True,
            smoothness_weight=item.smoothness_weight,
            gamut_weight=item.gamut_weight,
            dict_l1_weight=item.dict_l1_weight,
        )
        metric_rows = []
        for batch in dataset_batches(data, data_config.batch_size):
            _, metrics = fit_batch(
                batch["source"],
                batch["target"],
                fit_config,
                data.dictionary,
                data.rho,
                lpips_model,
                item.fit_steps,
                item.lr,
            )
            metric_rows.append(metrics)
        metrics = summarize_metric_tensors(metric_rows)
        rows.append(
            {
                "config": asdict(item),
                "name": item.name,
                "metrics": metrics,
                "pass": passes_a2(metrics),
            }
        )
        _write_partial(output, started, data_config, data, rows, complete=False)

    passing = [row for row in rows if row["pass"]]
    best = _select_best(rows)
    result = {
        "experiment": "M1.2_A2_loss_constraint_sweep",
        "started_at": started,
        "ended_at": now_stamp(),
        "data_config": data_config.to_dict(),
        "thresholds": A2_THRESHOLDS,
        "gt_action_stats": data.gt_stats,
        "sampling": data.sampling,
        "rows": rows,
        "best": best,
        "closed": bool(passing),
        "selected_passing_config": passing[0] if passing else None,
    }
    write_json(output, result)
    return result


def default_sweep(smoke: bool, full_grid: bool) -> list[A2SweepConfig]:
    if smoke:
        return [
            A2SweepConfig(0.5, 0.01, 0.0005, 4, 0.07),
            A2SweepConfig(1.0, 0.005, 0.002, 4, 0.05),
        ]
    if not full_grid:
        return [
            A2SweepConfig(0.5, 0.01, 0.0005, 80, 0.07),
            A2SweepConfig(1.0, 0.01, 0.0005, 80, 0.07),
            A2SweepConfig(2.0, 0.01, 0.0005, 80, 0.07),
            A2SweepConfig(4.0, 0.01, 0.0005, 80, 0.07),
            A2SweepConfig(8.0, 0.01, 0.0005, 80, 0.07),
            A2SweepConfig(2.0, 0.0, 0.002, 80, 0.07),
            A2SweepConfig(2.0, 0.002, 0.002, 80, 0.07),
            A2SweepConfig(2.0, 0.005, 0.002, 80, 0.07),
            A2SweepConfig(2.0, 0.01, 0.002, 80, 0.07),
            A2SweepConfig(4.0, 0.0, 0.002, 80, 0.07),
            A2SweepConfig(4.0, 0.002, 0.002, 80, 0.07),
            A2SweepConfig(4.0, 0.005, 0.002, 80, 0.07),
            A2SweepConfig(4.0, 0.01, 0.002, 80, 0.07),
            A2SweepConfig(4.0, 0.005, 0.0005, 80, 0.07),
            A2SweepConfig(4.0, 0.005, 0.002, 80, 0.07),
            A2SweepConfig(4.0, 0.005, 0.005, 80, 0.07),
            A2SweepConfig(4.0, 0.005, 0.01, 80, 0.07),
            A2SweepConfig(8.0, 0.005, 0.002, 80, 0.07),
            A2SweepConfig(8.0, 0.005, 0.005, 80, 0.07),
            A2SweepConfig(8.0, 0.005, 0.01, 80, 0.07),
            A2SweepConfig(4.0, 0.005, 0.005, 120, 0.05),
            A2SweepConfig(4.0, 0.005, 0.005, 120, 0.07),
            A2SweepConfig(8.0, 0.005, 0.005, 120, 0.05),
            A2SweepConfig(8.0, 0.005, 0.005, 120, 0.07),
        ]
    gamut = (0.5, 1.0, 2.0, 4.0)
    smooth = (0.0, 0.002, 0.005, 0.01)
    dict_l1 = (0.0005, 0.002, 0.005, 0.01)
    steps = (80, 120)
    lrs = (0.05, 0.07)
    return [A2SweepConfig(*values) for values in product(gamut, smooth, dict_l1, steps, lrs)]


def _write_partial(
    output: Path,
    started: str,
    data_config: M12DataConfig,
    data,
    rows: list[dict[str, Any]],
    complete: bool,
) -> None:
    passing = [row for row in rows if row["pass"]]
    write_json(
        output,
        {
            "experiment": "M1.2_A2_loss_constraint_sweep",
            "started_at": started,
            "ended_at": now_stamp() if complete else None,
            "complete": complete,
            "data_config": data_config.to_dict(),
            "thresholds": A2_THRESHOLDS,
            "gt_action_stats": data.gt_stats,
            "sampling": data.sampling,
            "rows": rows,
            "best": _select_best(rows),
            "closed": bool(passing),
            "selected_passing_config": passing[0] if passing else None,
        },
    )


def _select_best(rows: list[dict[str, Any]]) -> dict[str, Any] | None:
    if not rows:
        return None

    def key(row: dict[str, Any]):
        metrics = row["metrics"]
        clip_over = max(0.0, metrics["clipping_ratio"] - A2_THRESHOLDS["clipping_ratio"])
        de_over = max(0.0, metrics["mean_deltaE2000"] - A2_THRESHOLDS["mean_deltaE2000"])
        p95_over = max(0.0, metrics["p95_deltaE2000"] - A2_THRESHOLDS["p95_deltaE2000"])
        return (clip_over, de_over, p95_over, -metrics["PSNR"], metrics["LPIPS"])

    return min(rows, key=key)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run M1.2 A2 loss/constraint sweep.")
    parser.add_argument("--data-root", default="~/retouching/monetGPT/data")
    parser.add_argument("--output", default="m1_results/a2_loss_sweep_20260525.json")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--count", type=int, default=128)
    parser.add_argument("--image-size", type=int, default=64)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--seed", type=int, default=20260525)
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--full-grid", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.smoke:
        args.count = min(args.count, 8)
        args.batch_size = min(args.batch_size, 4)
    result = run_a2_sweep(
        M12DataConfig(
            data_root=Path(args.data_root).expanduser(),
            count=args.count,
            image_size=args.image_size,
            batch_size=args.batch_size,
            seed=args.seed,
            device=args.device,
        ),
        Path(args.output),
        default_sweep(args.smoke, args.full_grid),
    )
    print(result["closed"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
