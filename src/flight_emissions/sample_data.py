"""Build the small offline sample shipped in ``data/sample`` from ``traffic``'s bundled data.

The ``traffic`` library ships a few real trajectories as test data. This module extracts five
of them, attaches the metadata that ADS-B does not transmit (aircraft type, origin and
destination), normalises them to the canonical schema and writes one parquet file per flight.

Aircraft type codes were looked up by transponder address (icao24) in the public
`wiedehopf/tar1090-db <https://github.com/wiedehopf/tar1090-db>`_ aircraft database; the
registration is recorded in :data:`SAMPLE_FLIGHTS` so the lookup can be re-checked.
For the FlightRadar24 samples the type and airports come from the file itself.

It also extracts the A320 flight-data-recorder sample (measured fuel flow and mass, no
position) used by :func:`flight_emissions.validate.check_fuel_model_against_fdr`.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path

import pandas as pd

from flight_emissions.ingest import normalise

logger = logging.getLogger(__name__)

CATALOGUE_NAME = "_catalogue.csv"
"""Catalogue file written next to the sample; files starting with "_" are not loaded as flights."""


@dataclass(frozen=True)
class SampleFlight:
    """A flight from the ``traffic`` sample data with the metadata ADS-B does not carry."""

    flight_id: str
    source: str  # "samples:<name>" or "fr24:<id>"
    typecode: str
    registration: str
    origin: str
    destination: str
    description: str


SAMPLE_FLIGHTS: tuple[SampleFlight, ...] = (
    SampleFlight(
        "AFR34ZG_20240706",
        "samples:full_flight_short",
        "A321",
        "F-GMZC",
        "LFPG",
        "LFBO",
        "Air France Paris CDG - Toulouse, raw Mode S decoded with rs1090 (2024)",
    ),
    SampleFlight(
        "THY9BP_20240917",
        "fr24:372355e5",
        "B738",
        "TC-JVF",
        "LTFM",
        "ENGM",
        "Turkish Airlines Istanbul - Oslo (FlightRadar24 export)",
    ),
    SampleFlight(
        "EDW24_20240406",
        "fr24:34a8254b",
        "A343",
        "HB-JMF",
        "LSZH",
        "MMUN",
        "Edelweiss Zurich - Cancun, transatlantic (FlightRadar24 export)",
    ),
    SampleFlight(
        "ELY1747_20191103",
        "samples:elal747",
        "B744",
        "4X-ELC",
        "LIRF",
        "LLBG",
        "El Al Rome - Tel Aviv, B747 farewell flight that drew a 747 over Israel",
    ),
    SampleFlight(
        "TRA051_20180530",
        "samples:belevingsvlucht",
        "B738",
        "PH-HZO",
        "EHAM",
        "EHAM",
        "Transavia sightseeing round trip from Amsterdam (circular reference case)",
    ),
)


def _load_traffic_flight(source: str) -> pd.DataFrame:  # pragma: no cover - needs traffic data
    kind, name = source.split(":", 1)
    if kind == "samples":
        from traffic.data import samples

        return getattr(samples, name).data
    if kind == "fr24":
        from traffic.data.samples.fr24 import flight

        return flight(name).data
    raise ValueError(f"unknown sample source: {source}")


def merge_messages_per_second(df: pd.DataFrame) -> pd.DataFrame:
    """Collapse raw Mode S messages to one row per second.

    Raw decoders emit one row per message, and each message type carries different fields
    (position, velocity, identification). Taking the first non-missing value of every
    column within each second rebuilds complete state vectors without inventing data.
    """
    df = df.copy()
    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True).dt.floor("1s")
    return df.groupby("timestamp", as_index=False, sort=True).first()


def build_sample(out_dir: str | Path) -> list[Path]:  # pragma: no cover - needs traffic data
    """Write one canonical parquet file per sample flight and a ``_catalogue.csv`` table.

    Args:
        out_dir: Destination directory (created if needed).

    Returns:
        Paths of the written parquet files.
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    written: list[Path] = []
    rows = []
    for spec in SAMPLE_FLIGHTS:
        raw = _load_traffic_flight(spec.source)
        if spec.source == "samples:full_flight_short":
            raw = merge_messages_per_second(raw)
        df = normalise(
            raw,
            typecode=spec.typecode,
            flight_id=spec.flight_id,
            origin=spec.origin,
            destination=spec.destination,
        )
        path = out_dir / f"{spec.flight_id}.parquet"
        df.to_parquet(path, index=False, compression="zstd")
        written.append(path)
        rows.append(
            {
                "flight_id": spec.flight_id,
                "callsign": df["callsign"].iloc[0],
                "icao24": df["icao24"].iloc[0],
                "registration": spec.registration,
                "typecode": spec.typecode,
                "origin": spec.origin,
                "destination": spec.destination,
                "date_utc": df["timestamp"].iloc[0].strftime("%Y-%m-%d"),
                "n_points": len(df),
                "source": spec.source,
                "description": spec.description,
            }
        )
        logger.info("wrote %s (%d points)", path, len(df))
    pd.DataFrame(rows).to_csv(out_dir / CATALOGUE_NAME, index=False)
    return written


def build_fdr_reference(path: str | Path) -> Path:  # pragma: no cover - needs traffic data
    """Extract the A320 flight-data-recorder sample (measured fuel flow and mass).

    Columns: ``timestamp``, ``altitude_ft``, ``groundspeed_kt``, ``cas_kt``, ``mass_kg``,
    ``fuel_flow_kgh`` (total for both engines).
    """
    from traffic.data import samples

    raw = samples.fuelflow_a320.data
    df = pd.DataFrame(
        {
            "timestamp": pd.to_datetime(raw["timestamp"], utc=True),
            "altitude_ft": raw["altitude"].astype(float),
            "groundspeed_kt": raw["groundspeed"].astype(float),
            "cas_kt": raw["CAS"].astype(float),
            "mass_kg": raw["weight"].astype(float),
            "fuel_flow_kgh": raw["fuelflow"].astype(float),
        }
    )
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(path, index=False, compression="zstd")
    return path
