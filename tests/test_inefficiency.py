import numpy as np
import pytest

from flight_emissions.clean import clean_flight
from flight_emissions.config import PipelineConfig
from flight_emissions.fuel import estimate_fuel_flow, resolve_aircraft
from flight_emissions.geo import FT_TO_KM, KT_TO_KMS, haversine_km
from flight_emissions.inefficiency import (
    acceleration_ms2,
    descent_analysis,
    detect_level_offs,
    find_descent_window,
    ideal_descent_altitude,
    route_inefficiency,
)
from flight_emissions.phases import label_phases

# ---- route inefficiency -----------------------------------------------------------------


def straight_cruise(make_flight, n=200):
    df = make_flight(np.full(n, 35000.0), speed_kt=450.0)
    df["phase"] = "cruise"
    return df


def test_straight_flight_has_no_extra_distance(make_flight):
    df = straight_cruise(make_flight)
    r = route_inefficiency(df, np.full(len(df), 0.8))
    assert r.applicable
    assert r.extra_pct == pytest.approx(0.0, abs=1e-9)
    assert r.flown_km == pytest.approx(r.great_circle_km)
    # fuel per km at constant fuel flow and speed = ff / v
    assert r.fuel_per_km_kg == pytest.approx(0.8 / (450.0 * KT_TO_KMS))


def test_out_and_back_detour(make_flight):
    df = straight_cruise(make_flight, n=3)
    df["lon"] = [0.0, 2.0, 1.0]  # fly 2 deg east, then 1 deg back: 3 deg flown for 1 deg net
    r = route_inefficiency(df, np.full(3, 1.0))
    assert r.extra_pct == pytest.approx(200.0)
    one_deg = haversine_km(0, 0, 0, 1)
    assert r.extra_km == pytest.approx(2 * one_deg)
    assert r.extra_fuel_kg == pytest.approx(r.extra_km * r.fuel_per_km_kg)


def test_ground_points_are_excluded(make_flight):
    df = straight_cruise(make_flight, n=5)
    df["lon"] = [5.0, 0.0, 1.0, 2.0, 7.0]
    df["phase"] = ["ground", "cruise", "cruise", "cruise", "ground"]
    r = route_inefficiency(df, np.ones(5))
    assert r.great_circle_km == pytest.approx(haversine_km(0, 0, 0, 2))
    assert r.extra_pct == pytest.approx(0.0, abs=1e-9)


def test_circular_flight_is_not_applicable(make_flight):
    df = straight_cruise(make_flight, n=5)
    df["lon"] = [0.0, 1.0, 2.0, 1.0, 0.0001]
    r = route_inefficiency(df, np.ones(5))
    assert not r.applicable
    assert np.isnan(r.extra_pct)
    assert "circular" in r.note


def test_short_cruise_falls_back_to_airborne_average(make_flight):
    df = straight_cruise(make_flight, n=100)  # ~230 km, all labelled climb
    df["phase"] = "climb"
    r = route_inefficiency(df, np.full(100, 1.0))
    assert r.applicable
    assert "airborne average" in r.note
    assert route_inefficiency(df.iloc[:1], np.ones(1)).applicable is False


# ---- level-off detection ---------------------------------------------------------------


def test_detects_level_off_of_known_duration_and_altitude():
    t = np.arange(0, 1200, 10.0)
    vr = np.full(t.size, -2000.0)
    vr[30:43] = 0.0  # 13 samples = 120 s level
    vr[70:74] = 50.0  # 30 s: too short
    alt = 30000 + np.concatenate([[0.0], np.cumsum(vr[:-1] / 6.0)])
    los = detect_level_offs(t, alt, vr, threshold_fpm=300, min_duration_s=60)
    assert len(los) == 1
    assert los[0].start_idx == 30 and los[0].end_idx == 42
    assert los[0].duration_s == pytest.approx(120.0)
    assert los[0].altitude_ft == pytest.approx(alt[35])


def test_continuous_descent_has_no_level_off():
    t = np.arange(0, 600, 10.0)
    assert detect_level_offs(t, 30000 - 30 * t, np.full(t.size, -1800.0)) == []


# ---- idealised descent --------------------------------------------------------------------


def test_ideal_profile_is_level_then_three_degrees():
    d = np.linspace(0, 300, 301)
    alt = ideal_descent_altitude(d, 30000.0, 3000.0, 3.0)
    needed_km = 27000 * FT_TO_KM / np.tan(np.radians(3.0))  # ~157 km
    assert alt[0] == 30000.0 and alt[-1] == pytest.approx(3000.0)
    assert alt[int(300 - needed_km) - 1] == 30000.0
    grad = np.diff(alt[-50:]) / np.diff(d[-50:]) * FT_TO_KM
    assert grad == pytest.approx(-np.tan(np.radians(3.0)))


def test_ideal_profile_constant_gradient_when_three_degrees_does_not_fit():
    d = np.linspace(0, 50, 11)
    alt = ideal_descent_altitude(d, 30000.0, 3000.0, 3.0)
    assert alt == pytest.approx(30000 - 27000 * d / 50)


def test_acceleration_of_linear_speed_ramp():
    t = np.arange(0, 300, 10.0)
    tas_kt = 250 + 0.5 * t  # 0.5 kt/s
    acc = acceleration_ms2(t, tas_kt)
    assert acc[10:-10] == pytest.approx(0.5 * 0.514444, rel=1e-3)
    assert acceleration_ms2(t[:2], tas_kt[:2]).tolist() == [0.0, 0.0]


def mission(make_flight, descent_alt):
    """Cruise at FL350 then the given descent, then a short ground roll."""
    cruise = np.full(150, 35000.0)
    taxi = np.zeros(20)
    alt = np.concatenate([cruise, descent_alt, taxi])
    speed = np.concatenate(
        [np.full(150, 300.0), np.full(descent_alt.size, 300.0), np.full(20, 15.0)]
    )
    df, _ = clean_flight(make_flight(alt, speed_kt=speed))
    df["phase"] = label_phases(df)
    return df


def three_degree_descent():
    step_ft = 300.0 * KT_TO_KMS * 10.0 * np.tan(np.radians(3.0)) / FT_TO_KM  # ft per 10 s
    return np.arange(35000.0 - step_ft, 0.0, -step_ft)


def test_continuous_descent_has_no_saving(make_flight):
    df = mission(make_flight, three_degree_descent())
    model = resolve_aircraft("A320")
    fuel = estimate_fuel_flow(df, model)
    res = descent_analysis(df, fuel, model)
    assert res.applicable
    assert res.n_level_offs == 0
    assert res.tod_altitude_ft == pytest.approx(35000.0, abs=300)
    assert abs(res.saving_pct) < 2.0
    assert res.profile is not None and len(res.profile) > 10


def test_stepped_descent_has_level_off_and_saving(make_flight):
    steep = np.linspace(35000, 15000, 51)[1:]  # 4000 ft/min
    level = np.full(36, 15000.0)  # 6 min level-off at FL150
    rest = np.linspace(15000, 0, 51)[1:]
    df = mission(make_flight, np.concatenate([steep, level, rest]))
    model = resolve_aircraft("A320")
    fuel = estimate_fuel_flow(df, model)
    res = descent_analysis(df, fuel, model)
    assert res.n_level_offs == 1
    assert res.level_offs[0].altitude_ft == pytest.approx(15000.0)
    assert res.level_time_s >= 300
    assert res.saving_kg > 0
    assert res.saving_pct > 2.0
    assert set(res.to_dict()) >= {"saving_kg", "n_level_offs", "tod_altitude_ft"}


def test_no_descent_window_when_recording_ends_in_cruise(make_flight):
    df = straight_cruise(make_flight)
    model = resolve_aircraft("A320")
    fuel = estimate_fuel_flow(df, model)
    assert find_descent_window(df, PipelineConfig()) is None
    assert not descent_analysis(df, fuel, model).applicable
