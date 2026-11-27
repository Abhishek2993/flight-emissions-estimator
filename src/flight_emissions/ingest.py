"""Load trajectories and normalise them to the pipeline's canonical schema.

Every downstream stage works on a :class:`pandas.DataFrame` with exactly these columns
(units in the names, so nobody has to guess):

========================  ===========================================================
``timestamp``             UTC, ``datetime64[ns, UTC]``
``lat`` / ``lon``         WGS-84 degrees
``altitude_ft``           barometric (pressure) altitude, feet
``groundspeed_kt``        ground speed, knots
``vertical_rate_fpm``     rate of climb (negative = descent), feet per minute
``track``                 true track angle, degrees
``icao24``                24-bit transponder address, lower-case hex
``callsign``              ATC callsign
``typecode``              ICAO aircraft type designator, e.g. ``A321``
========================  ===========================================================

plus the per-flight metadata ``flight_id``, ``origin`` and ``destination`` (ICAO airport
codes, may be empty). Data sources differ in naming (``latitude`` vs ``lat``,
``vertical_speed`` vs ``vertical_rate``...), so :func:`normalise` maps the known aliases.
"""

from __future__ import annotations

import logging
import os
from collections.abc import Iterable
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np
import pandas as pd

if TYPE_CHECKING:  # pragma: no cover - imports only needed for type hints
    from traffic.core import Flight

logger = logging.getLogger(__name__)

CANONICAL_COLUMNS: list[str] = [
    "timestamp",
    "lat",
    "lon",
    "altitude_ft",
    "groundspeed_kt",
    "vertical_rate_fpm",
    "track",
    "icao24",
    "callsign",
    "typecode",
]
META_COLUMNS: list[str] = ["flight_id", "origin", "destination"]
NUMERIC_COLUMNS: list[str] = [
    "lat",
    "lon",
    "altitude_ft",
    "groundspeed_kt",
    "vertical_rate_fpm",
    "track",
]

#: Accepted source column names for each canonical column, in order of preference.
COLUMN_ALIASES: dict[str, tuple[str, ...]] = {
    "timestamp": ("timestamp", "time", "ts", "datetime"),
    "lat": ("lat", "latitude"),
    "lon": ("lon", "longitude", "lng", "long"),
    "altitude_ft": ("altitude_ft", "altitude", "baro_altitude_ft", "alt"),
    "groundspeed_kt": ("groundspeed_kt", "groundspeed", "gs", "ground_speed"),
    "vertical_rate_fpm": ("vertical_rate_fpm", "vertical_rate", "vertical_speed", "roc", "vrate"),
    "track": ("track", "true_track", "course"),
    "icao24": ("icao24", "icao", "hex", "transponder"),
    "callsign": ("callsign", "flight", "ident"),
    "typecode": ("typecode", "aircraft_type", "type", "actype"),
    "flight_id": ("flight_id", "id"),
    "origin": ("origin", "departure", "adep"),
    "destination": ("destination", "arrival", "ades"),
}

OPENSKY_ENV_VARS = ("OPENSKY_USERNAME", "OPENSKY_PASSWORD")


class IngestError(ValueError):
    """Raised when input data cannot be mapped to the canonical schema."""


def _first_present(df: pd.DataFrame, names: Iterable[str]) -> str | None:
    for name in names:
        if name in df.columns:
            return name
    return None


def _to_utc(series: pd.Series) -> pd.Series:
    """Convert datetimes or epoch seconds to ``datetime64[ns, UTC]``."""
    if pd.api.types.is_numeric_dtype(series):
        out = pd.to_datetime(series.astype(float), unit="s", utc=True)
    else:
        out = pd.to_datetime(series, utc=True)
    return out.astype("datetime64[ns, UTC]")


def _clean_str(value: Any) -> str | None:
    if value is None or (isinstance(value, float) and np.isnan(value)) or value is pd.NA:
        return None
    text = str(value).strip()
    return text or None


def normalise(
    df: pd.DataFrame,
    *,
    typecode: str | None = None,
    callsign: str | None = None,
    icao24: str | None = None,
    flight_id: str | None = None,
    origin: str | None = None,
    destination: str | None = None,
) -> pd.DataFrame:
    """Map a raw trajectory table onto the canonical schema.

    Keyword arguments override (or supply) the per-flight constants when the raw data
    does not carry them, which is common: ADS-B messages contain no aircraft type.

    Args:
        df: Raw trajectory of a single flight.
        typecode: ICAO aircraft type designator.
        callsign: ATC callsign.
        icao24: Transponder address.
        flight_id: Unique identifier for the flight; derived from callsign and date if absent.
        origin: ICAO code of the departure airport.
        destination: ICAO code of the arrival airport.

    Returns:
        A new DataFrame with :data:`CANONICAL_COLUMNS` + :data:`META_COLUMNS`, sorted by time.

    Raises:
        IngestError: if the timestamp or position columns are missing.
    """
    out = pd.DataFrame(index=df.index)
    for canonical, aliases in COLUMN_ALIASES.items():
        source = _first_present(df, aliases)
        out[canonical] = df[source] if source is not None else np.nan

    for required in ("timestamp", "lat", "lon"):
        if out[required].isna().all():
            raise IngestError(f"input has no usable '{required}' column")

    out["timestamp"] = _to_utc(out["timestamp"])
    for col in NUMERIC_COLUMNS:
        out[col] = pd.to_numeric(out[col], errors="coerce").astype("float64")

    overrides = {
        "typecode": typecode,
        "callsign": callsign,
        "icao24": icao24,
        "flight_id": flight_id,
        "origin": origin,
        "destination": destination,
    }
    for col, value in overrides.items():
        if value is not None:
            out[col] = value
        # Per-flight constants: take the first non-empty value seen in the data.
        constant = next((v for v in map(_clean_str, out[col].tolist()) if v), None)
        out[col] = constant

    if out["typecode"].iloc[0] is not None:
        out["typecode"] = str(out["typecode"].iloc[0]).upper()
    if out["icao24"].iloc[0] is not None:
        out["icao24"] = str(out["icao24"].iloc[0]).lower()
    if out["flight_id"].iloc[0] is None:
        day = out["timestamp"].min().strftime("%Y%m%d")
        ident = out["callsign"].iloc[0] or out["icao24"].iloc[0] or "flight"
        out["flight_id"] = f"{ident}_{day}"

    out = out[CANONICAL_COLUMNS + META_COLUMNS]
    return out.sort_values("timestamp", kind="stable").reset_index(drop=True)


def from_traffic(flight: Flight, **meta: str | None) -> pd.DataFrame:
    """Normalise a :class:`traffic.core.Flight` (keyword arguments as in :func:`normalise`)."""
    return normalise(flight.data, **meta)


def read_table(path: str | Path) -> pd.DataFrame:
    """Read a parquet or CSV file into a DataFrame."""
    path = Path(path)
    suffix = path.suffix.lower()
    if suffix == ".parquet":
        return pd.read_parquet(path)
    if suffix in {".csv", ".gz"}:
        return pd.read_csv(path)
    raise IngestError(f"unsupported file type: {path.name} (expected .parquet or .csv)")


def split_flights(df: pd.DataFrame) -> list[pd.DataFrame]:
    """Split a table holding several flights into one DataFrame per flight.

    Flights are separated by ``flight_id`` when present, otherwise by ``icao24`` + ``callsign``.
    """
    for keys in (["flight_id"], ["icao24", "callsign"], ["icao24"]):
        if all(k in df.columns for k in keys) and df[keys].notna().any().all():
            return [g.reset_index(drop=True) for _, g in df.groupby(keys, sort=True, dropna=False)]
    return [df]


def load_flights(path: str | Path) -> list[pd.DataFrame]:
    """Load every flight under ``path`` (a file or a directory of parquet/CSV files).

    Files whose name starts with an underscore are treated as metadata and skipped.

    Args:
        path: File or directory.

    Returns:
        One canonical DataFrame per flight, ordered by ``flight_id``.
    """
    path = Path(path)
    if path.is_dir():
        files = sorted(
            f
            for f in [*path.glob("*.parquet"), *path.glob("*.csv")]
            if not f.name.startswith("_")  # "_catalogue.csv" and similar are metadata
        )
        if not files:
            raise IngestError(f"no .parquet or .csv files found in {path}")
    elif path.exists():
        files = [path]
    else:
        raise IngestError(f"input path does not exist: {path}")

    flights: list[pd.DataFrame] = []
    for file in files:
        table = read_table(file)
        for raw in split_flights(table):
            flights.append(normalise(raw))
    flights.sort(key=lambda f: str(f["flight_id"].iloc[0]))
    logger.info("loaded %d flight(s) from %s", len(flights), path)
    return flights


def opensky_credentials_available() -> bool:
    """Return True when the OpenSky credentials are set in the environment."""
    return all(os.environ.get(var) for var in OPENSKY_ENV_VARS)


def load_opensky(
    start: str,
    stop: str,
    *,
    callsign: str | None = None,
    icao24: str | None = None,
    typecode: str | None = None,
    origin: str | None = None,
    destination: str | None = None,
) -> pd.DataFrame:  # pragma: no cover - needs network access and personal credentials
    """Download one historical flight from the OpenSky Network via ``traffic``.

    Credentials are never passed in code: ``traffic``/``pyopensky`` read
    ``OPENSKY_USERNAME`` and ``OPENSKY_PASSWORD`` from the environment.

    Args:
        start: Start of the time window, e.g. ``"2024-07-06 06:30"`` (UTC).
        stop: End of the time window (UTC).
        callsign: Callsign filter.
        icao24: Transponder address filter.
        typecode: ICAO type designator to attach (ADS-B does not transmit it).
        origin: Departure airport ICAO code, also used as a server-side filter.
        destination: Arrival airport ICAO code, also used as a server-side filter.

    Returns:
        The flight in the canonical schema.

    Raises:
        RuntimeError: if credentials are missing or no data matches the query.
    """
    if not opensky_credentials_available():
        raise RuntimeError(
            "OpenSky credentials not found. Export OPENSKY_USERNAME and OPENSKY_PASSWORD "
            "(see README, 'Optional: OpenSky Network data')."
        )
    if callsign is None and icao24 is None:
        raise ValueError("give at least a callsign or an icao24 to identify the flight")

    from traffic.data import opensky  # imported lazily: it reads the credentials

    flight = opensky.history(
        start,
        stop,
        callsign=callsign,
        icao24=icao24,
        departure_airport=origin,
        arrival_airport=destination,
        return_flight=True,
    )
    if flight is None:
        raise RuntimeError("OpenSky returned no data for this query")
    return from_traffic(
        flight,
        typecode=typecode,
        callsign=callsign,
        icao24=icao24,
        origin=origin,
        destination=destination,
    )
