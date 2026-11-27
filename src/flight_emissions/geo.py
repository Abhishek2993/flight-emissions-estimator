"""Spherical-earth geometry helpers.

All distances use the haversine formula on a sphere of radius 6,371.0088 km (the IUGG mean
earth radius). The ellipsoidal error of a spherical model is below 0.5 %, far smaller than
the other uncertainties in this pipeline, and it keeps the maths easy to verify by hand.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import ArrayLike, NDArray

EARTH_RADIUS_KM = 6371.0088
KM_PER_NM = 1.852
FT_TO_KM = 0.0003048
KT_TO_KMS = KM_PER_NM / 3600.0  # 1 knot expressed in km/s


def haversine_km(
    lat1: ArrayLike, lon1: ArrayLike, lat2: ArrayLike, lon2: ArrayLike
) -> NDArray[np.float64]:
    """Great-circle distance between two points (or arrays of points), in km.

    ``d = 2 R asin( sqrt( sin²(Δφ/2) + cos φ1 cos φ2 sin²(Δλ/2) ) )``

    Args:
        lat1: Latitude of the first point(s) in degrees.
        lon1: Longitude of the first point(s) in degrees.
        lat2: Latitude of the second point(s) in degrees.
        lon2: Longitude of the second point(s) in degrees.

    Returns:
        Distance(s) in kilometres, broadcast to the input shape.
    """
    phi1, lam1, phi2, lam2 = (
        np.radians(np.asarray(a, dtype=float)) for a in (lat1, lon1, lat2, lon2)
    )
    dphi = phi2 - phi1
    dlam = lam2 - lam1
    a = np.sin(dphi / 2.0) ** 2 + np.cos(phi1) * np.cos(phi2) * np.sin(dlam / 2.0) ** 2
    return 2.0 * EARTH_RADIUS_KM * np.arcsin(np.sqrt(np.clip(a, 0.0, 1.0)))


def segment_lengths_km(lat: ArrayLike, lon: ArrayLike) -> NDArray[np.float64]:
    """Length of each leg between consecutive points, in km (array of ``n - 1`` values)."""
    lat = np.asarray(lat, dtype=float)
    lon = np.asarray(lon, dtype=float)
    if lat.size < 2:
        return np.zeros(0)
    return haversine_km(lat[:-1], lon[:-1], lat[1:], lon[1:])


def cumulative_distance_km(lat: ArrayLike, lon: ArrayLike) -> NDArray[np.float64]:
    """Cumulative along-track distance from the first point, in km (same length as input)."""
    legs = segment_lengths_km(lat, lon)
    return np.concatenate([[0.0], np.cumsum(legs)]) if legs.size else np.zeros(np.size(lat))


def path_length_km(lat: ArrayLike, lon: ArrayLike) -> float:
    """Total length of a polyline through the given points, in km."""
    return float(np.sum(segment_lengths_km(lat, lon)))
