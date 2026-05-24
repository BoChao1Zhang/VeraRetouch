from __future__ import annotations

import argparse
import json
from pathlib import Path

from .data import DataRequirements, discover_m1_data


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
    parser.print_help()
    return 0
