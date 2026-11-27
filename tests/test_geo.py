import numpy as np
import pytest

from flight_emissions.geo import (
    EARTH_RADIUS_KM,
    cumulative_distance_km,
    haversine_km,
    path_length_km,
    segment_lengths_km,
)


def test_one_degree_on_equator():
    assert haversine_km(0, 0, 0, 1) == pytest.approx(2 * np.pi * EARTH_RADIUS_KM / 360, rel=1e-12)


def test_quarter_meridian():
    # Equator to pole is a quarter of a great circle.
    assert haversine_km(0, 0, 90, 0) == pytest.approx(np.pi * EARTH_RADIUS_KM / 2, rel=1e-12)


def test_symmetric_and_zero():
    assert haversine_km(48.0, 2.0, 43.6, 1.4) == pytest.approx(haversine_km(43.6, 1.4, 48.0, 2.0))
    assert haversine_km(10.0, 20.0, 10.0, 20.0) == 0.0


def test_known_city_pair():
    # Paris CDG -> Toulouse Blagnac is about 604 km (Δlat 5.4°, Δlon 1.2°).
    d = haversine_km(49.0097, 2.5479, 43.6293, 1.3638)
    assert d == pytest.approx(604, abs=5)


def test_antimeridian_crossing_is_short():
    assert haversine_km(0, 179.5, 0, -179.5) == pytest.approx(haversine_km(0, 0, 0, 1))


def test_path_is_never_shorter_than_great_circle():
    lat = np.array([0.0, 1.0, 0.0])
    lon = np.array([0.0, 1.0, 2.0])
    assert path_length_km(lat, lon) > haversine_km(0, 0, 0, 2)


def test_cumulative_distance_shapes():
    lat = np.zeros(4)
    lon = np.array([0.0, 1.0, 2.0, 3.0])
    legs = segment_lengths_km(lat, lon)
    cum = cumulative_distance_km(lat, lon)
    assert legs.shape == (3,)
    assert cum.shape == (4,)
    assert cum[0] == 0.0
    assert cum[-1] == pytest.approx(legs.sum())
    assert segment_lengths_km([1.0], [1.0]).size == 0
    assert cumulative_distance_km([1.0], [1.0]).tolist() == [0.0]
