"""Command-line interface (sub-commands are added as the pipeline stages are built)."""

from __future__ import annotations

import argparse
from collections.abc import Sequence


def build_parser() -> argparse.ArgumentParser:
    """Create the top-level argument parser."""
    parser = argparse.ArgumentParser(
        prog="flight_emissions",
        description="Estimate fuel burn and emissions from ADS-B trajectories.",
    )
    parser.add_subparsers(dest="command", required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """Entry point for ``python -m flight_emissions``."""
    parser = build_parser()
    parser.parse_args(argv)
    return 0
