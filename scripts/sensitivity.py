"""Sensitivity of total fuel to the take-off mass assumption.

Runs the pipeline on every flight with several take-off mass fractions and writes
``results/sensitivity.csv`` and ``results/sensitivity.md`` (quoted in the README)::

    python scripts/sensitivity.py --input data/sample --out results/
"""

from __future__ import annotations

import argparse
import logging
import warnings
from pathlib import Path

import pandas as pd

from flight_emissions.config import PipelineConfig
from flight_emissions.ingest import load_flights
from flight_emissions.pipeline import process_flight

FRACTIONS = (0.70, 0.85, 1.00)


def sensitivity(input_path: Path, fractions: tuple[float, ...] = FRACTIONS) -> pd.DataFrame:
    """Total fuel (kg) per flight for each mass fraction."""
    flights = load_flights(input_path)
    rows = []
    for raw in flights:
        row: dict[str, object] = {"flight_id": raw["flight_id"].iloc[0]}
        for f in fractions:
            r = process_flight(raw, PipelineConfig(mass_fraction=f))
            row[f"fuel_kg_at_{f:.2f}"] = r.phase_table["fuel_kg"].sum() if r.ok else float("nan")
        rows.append(row)
    return pd.DataFrame(rows)


def to_markdown(table: pd.DataFrame, fractions: tuple[float, ...] = FRACTIONS) -> str:
    """Markdown table with the change relative to the default fraction (0.85)."""
    base = "fuel_kg_at_0.85"
    head = "| Flight | " + " | ".join(f"{f:.2f} x MTOW" for f in fractions) + " |"
    lines = [head, "|---|" + "---:|" * len(fractions)]
    for _, r in table.iterrows():
        cells = []
        for f in fractions:
            v = r[f"fuel_kg_at_{f:.2f}"]
            delta = 100 * (v / r[base] - 1)
            cells.append(f"{v:,.0f} kg" + ("" if abs(f - 0.85) < 1e-9 else f" ({delta:+.1f} %)"))
        lines.append(f"| {r['flight_id']} | " + " | ".join(cells) + " |")
    return "\n".join(lines) + "\n"


def main() -> None:
    """Command-line entry point."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=Path("data/sample"))
    parser.add_argument("--out", type=Path, default=Path("results"))
    args = parser.parse_args()
    logging.basicConfig(level=logging.WARNING)
    warnings.filterwarnings("ignore", message="Aircraft: using synonym")
    table = sensitivity(args.input)
    args.out.mkdir(parents=True, exist_ok=True)
    table.to_csv(args.out / "sensitivity.csv", index=False)
    (args.out / "sensitivity.md").write_text(to_markdown(table))
    print(to_markdown(table))


if __name__ == "__main__":
    main()
