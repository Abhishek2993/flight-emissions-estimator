r"""Command-line interface.

Examples::

    python -m flight_emissions run --input data/sample --out results/
    python -m flight_emissions run --input data/sample --out results/ --figures-dir docs/figures
    python -m flight_emissions build-sample --out data/sample
    python -m flight_emissions fetch-opensky --start "2024-07-06 06:30" --stop "2024-07-06 08:15" \\
        --callsign AFR34ZG --typecode A321 --origin LFPG --destination LFBO --out data/opensky
"""

from __future__ import annotations

import argparse
import logging
import warnings
from collections.abc import Sequence
from pathlib import Path

from flight_emissions.config import PipelineConfig


def build_parser() -> argparse.ArgumentParser:
    """Create the argument parser."""
    parser = argparse.ArgumentParser(
        prog="flight_emissions",
        description="Estimate fuel burn and emissions from ADS-B trajectories.",
    )
    parser.add_argument("-v", "--verbose", action="store_true", help="debug logging")
    sub = parser.add_subparsers(dest="command", required=True)

    run = sub.add_parser("run", help="run the pipeline on a file or directory of flights")
    run.add_argument("--input", required=True, type=Path, help="parquet/CSV file or directory")
    run.add_argument("--out", required=True, type=Path, help="output directory")
    run.add_argument(
        "--figures-dir",
        type=Path,
        default=None,
        help="where to save PNG figures (default: OUT/figures)",
    )
    run.add_argument("--no-figures", action="store_true", help="skip figure generation")
    run.add_argument(
        "--reference",
        type=Path,
        default=Path("data/reference/icao_reference.csv"),
        help="ICAO reference table (CSV)",
    )
    run.add_argument(
        "--fdr",
        type=Path,
        default=Path("data/reference/a320_fdr_fuelflow.parquet"),
        help="flight-data-recorder sample for the fuel-model check",
    )
    run.add_argument(
        "--step",
        type=int,
        default=PipelineConfig.resample_step_s,
        help="resampling step in seconds (default: %(default)s)",
    )
    run.add_argument(
        "--mass-fraction",
        type=float,
        default=PipelineConfig.mass_fraction,
        help="take-off mass as a fraction of MTOW (default: %(default)s)",
    )
    run.add_argument(
        "--load-factor",
        type=float,
        default=PipelineConfig.load_factor,
        help="passenger load factor for per-pax CO2 (default: %(default)s)",
    )
    run.add_argument(
        "--fallback-typecode",
        default=None,
        help="type to model when a flight's type code is missing or unsupported",
    )

    sample = sub.add_parser("build-sample", help="rebuild data/sample from traffic's bundled data")
    sample.add_argument("--out", type=Path, default=Path("data/sample"))
    sample.add_argument(
        "--fdr-out", type=Path, default=Path("data/reference/a320_fdr_fuelflow.parquet")
    )

    osky = sub.add_parser(
        "fetch-opensky",
        help="download one flight from OpenSky (needs OPENSKY_USERNAME / OPENSKY_PASSWORD)",
    )
    osky.add_argument("--start", required=True, help="UTC start, e.g. '2024-07-06 06:30'")
    osky.add_argument("--stop", required=True, help="UTC end")
    osky.add_argument("--callsign")
    osky.add_argument("--icao24")
    osky.add_argument("--typecode", help="ICAO type designator (ADS-B does not carry it)")
    osky.add_argument("--origin")
    osky.add_argument("--destination")
    osky.add_argument("--out", type=Path, default=Path("data/opensky"))
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """Entry point for ``python -m flight_emissions``."""
    args = build_parser().parse_args(argv)
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(levelname)-7s %(name)s: %(message)s",
    )
    for noisy in ("traffic", "impunity", "matplotlib", "fontTools"):
        logging.getLogger(noisy).setLevel(logging.WARNING)
    warnings.filterwarnings("ignore", message="Aircraft: using synonym")

    if args.command == "run":
        from flight_emissions.pipeline import run_pipeline

        config = PipelineConfig(
            resample_step_s=args.step,
            mass_fraction=args.mass_fraction,
            load_factor=args.load_factor,
            fallback_typecode=args.fallback_typecode,
        )
        outputs = run_pipeline(
            args.input,
            args.out,
            config,
            reference_path=args.reference,
            fdr_path=args.fdr,
            figures_dir=args.figures_dir,
            make_figures=not args.no_figures,
        )
        print((Path(args.out) / "summary.md").read_text())
        n_ok = int((outputs.summary["status"] == "ok").sum())
        print(f"{n_ok}/{len(outputs.results)} flights processed; results in {args.out}")
        return 0

    if args.command == "build-sample":  # pragma: no cover - needs traffic sample data
        from flight_emissions.sample_data import build_fdr_reference, build_sample

        paths = build_sample(args.out)
        build_fdr_reference(args.fdr_out)
        print(f"wrote {len(paths)} flights to {args.out} and {args.fdr_out}")
        return 0

    if args.command == "fetch-opensky":  # pragma: no cover - needs network and credentials
        from flight_emissions.ingest import load_opensky

        df = load_opensky(
            args.start,
            args.stop,
            callsign=args.callsign,
            icao24=args.icao24,
            typecode=args.typecode,
            origin=args.origin,
            destination=args.destination,
        )
        args.out.mkdir(parents=True, exist_ok=True)
        path = args.out / f"{df['flight_id'].iloc[0]}.parquet"
        df.to_parquet(path, index=False)
        print(f"wrote {len(df)} points to {path}")
        return 0
    return 1  # pragma: no cover
