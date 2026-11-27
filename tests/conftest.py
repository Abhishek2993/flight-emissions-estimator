"""Shared synthetic-trajectory builders.

Synthetic flights fly due east along the equator, where one degree of longitude is exactly
``2 * pi * R / 360`` km, so distances and speeds have closed-form answers.
"""

from __future__ import annotations

from collections.abc import Callable

import numpy as np
import pandas as pd
import pytest

from flight_emissions.geo import EARTH_RADIUS_KM, KT_TO_KMS

KM_PER_DEG_EQUATOR = 2 * np.pi * EARTH_RADIUS_KM / 360.0


def build_flight(
    altitude_ft: np.ndarray,
    step_s: float = 10.0,
    speed_kt: float | np.ndarray = 400.0,
    typecode: str | None = "A320",
    flight_id: str = "TEST01",
    start: str = "2024-01-01 12:00:00",
    vertical_rate_fpm: np.ndarray | None = None,
) -> pd.DataFrame:
    """Build a canonical trajectory flying east along the equator."""
    altitude_ft = np.asarray(altitude_ft, dtype=float)
    n = altitude_ft.size
    t = np.arange(n) * step_s
    speed = np.broadcast_to(np.asarray(speed_kt, dtype=float), (n,)).copy()
    dist_km = np.concatenate([[0.0], np.cumsum(speed[:-1] * KT_TO_KMS * np.diff(t))])
    if vertical_rate_fpm is None:
        vertical_rate_fpm = np.gradient(altitude_ft, t) * 60.0 if n > 1 else np.zeros(n)
    return pd.DataFrame(
        {
            "timestamp": pd.Timestamp(start, tz="UTC") + pd.to_timedelta(t, unit="s"),
            "lat": np.zeros(n),
            "lon": dist_km / KM_PER_DEG_EQUATOR,
            "altitude_ft": altitude_ft,
            "groundspeed_kt": speed,
            "vertical_rate_fpm": vertical_rate_fpm,
            "track": np.full(n, 90.0),
            "icao24": "abc123",
            "callsign": flight_id,
            "typecode": typecode,
            "flight_id": flight_id,
            "origin": "AAAA",
            "destination": "BBBB",
        }
    )


@pytest.fixture
def make_flight() -> Callable[..., pd.DataFrame]:
    """Fixture returning :func:`build_flight`."""
    return build_flight
