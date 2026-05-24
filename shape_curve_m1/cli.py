from __future__ import annotations

import argparse


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
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.version:
        from . import __version__

        print(__version__)
        return 0
    parser.print_help()
    return 0
