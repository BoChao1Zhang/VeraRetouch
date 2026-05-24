from __future__ import annotations

import argparse
import json
from pathlib import Path

from .data import DataRequirements, discover_m1_data
from .runner import M1RunConfig, run_m1


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m shape_curve_m1",
        description="Run ShapeCurve M1 action-space viability checks.",
    )
    parser.add_argument(
        "--version",
        action="store_true",
        help="Print the ShapeCurve M1 runner version and exit.",
    )
    subparsers = parser.add_subparsers(dest="command")

    check_data = subparsers.add_parser(
        "check-data",
        help="Discover and validate M1 data tiers.",
    )
    check_data.add_argument(
        "--data-root",
        default="~/retouching/monetGPT/data",
        help="Root containing ppr10k and fivek_mmart_like directories.",
    )
    check_data.add_argument(
        "--json",
        action="store_true",
        help="Emit machine-readable JSON.",
    )

    run = subparsers.add_parser(
        "run-m1",
        help="Run the full M1 action-space viability gate.",
    )
    run.add_argument("--data-root", default="~/retouching/monetGPT/data")
    run.add_argument("--output", default="m1_results/latest_m1.json")
    run.add_argument("--device", default="cuda:0")
    run.add_argument("--image-size", type=int, default=64)
    run.add_argument("--batch-size", type=int, default=16)
    run.add_argument("--fit-steps", type=int, default=80)
    run.add_argument("--fit-lr", type=float, default=0.07)
    run.add_argument("--tier-a-count", type=int, default=500)
    run.add_argument("--tier-b-count", type=int, default=500)
    run.add_argument("--fivek-count", type=int, default=100)
    run.add_argument("--ppr10k-count", type=int, default=100)
    run.add_argument("--tiny-train-count", type=int, default=2000)
    run.add_argument("--tiny-val-count", type=int, default=500)
    run.add_argument("--tiny-epochs", type=int, default=5)
    run.add_argument("--seed", type=int, default=20260525)
    run.add_argument(
        "--smoke",
        action="store_true",
        help="Run a tiny development check. This is not valid for M1 acceptance.",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.version:
        from . import __version__

        print(__version__)
        return 0
    if args.command == "check-data":
        report = discover_m1_data(Path(args.data_root).expanduser(), DataRequirements())
        payload = report.to_dict()
        if args.json:
            print(json.dumps(payload, indent=2, sort_keys=True))
        else:
            print(report.format_text())
        return 0 if report.ready else 2
    if args.command == "run-m1":
        if args.smoke:
            args.tier_a_count = 8
            args.tier_b_count = 8
            args.fivek_count = 4
            args.ppr10k_count = 4
            args.tiny_train_count = 16
            args.tiny_val_count = 8
            args.fit_steps = min(args.fit_steps, 4)
            args.tiny_epochs = 1
            args.batch_size = min(args.batch_size, 4)
        config = M1RunConfig(
            data_root=Path(args.data_root).expanduser(),
            output=Path(args.output),
            device=args.device,
            image_size=args.image_size,
            batch_size=args.batch_size,
            fit_steps=args.fit_steps,
            fit_lr=args.fit_lr,
            tier_a_count=args.tier_a_count,
            tier_b_count=args.tier_b_count,
            fivek_count=args.fivek_count,
            ppr10k_count=args.ppr10k_count,
            tiny_train_count=args.tiny_train_count,
            tiny_val_count=args.tiny_val_count,
            tiny_epochs=args.tiny_epochs,
            seed=args.seed,
            smoke=args.smoke,
        )
        result = run_m1(config)
        print(json.dumps(result, indent=2, sort_keys=True))
        return 0 if result.get("M1_CONCLUSION") == "PASS" else 1
    parser.print_help()
    return 0
