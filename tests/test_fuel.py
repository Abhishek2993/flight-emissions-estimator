import numpy as np
import pandas as pd
import pytest

from flight_emissions.clean import clean_flight
from flight_emissions.config import PipelineConfig
from flight_emissions.fuel import (
    UnsupportedAircraftError,
    check_type_plausibility,
    cumulative,
    estimate_fuel_flow,
    fuel_flow_model,
    integrate,
    integrate_by_phase,
    interval_amounts,
    resolve_aircraft,
    smoothed_vertical_rate,
    thrust_mode,
    true_airspeed_kt,
)
from flight_emissions.phases import label_phases

# ---- integration over time (closed-form answers) --------------------------------------


def test_constant_flow_integrates_to_rate_times_duration():
    t = np.arange(0, 3601, 10.0)
    assert integrate(t, np.full(t.size, 0.75)) == pytest.approx(0.75 * 3600)


def test_linear_flow_is_exact_with_trapezoid_rule():
    t = np.array([0.0, 3.0, 4.0, 10.0])  # non-uniform steps
    rate = 2.0 * t  # integral of 2t from 0 to 10 = 100
    assert integrate(t, rate) == pytest.approx(100.0)
    assert cumulative(t, rate).tolist() == pytest.approx([0.0, 9.0, 16.0, 100.0])


def test_gap_is_bridged_linearly():
    # 1 kg/s before a 600 s gap, 3 kg/s after: the gap contributes 600 * 2 kg.
    t = np.array([0.0, 10.0, 610.0, 620.0])
    rate = np.array([1.0, 1.0, 3.0, 3.0])
    assert interval_amounts(t, rate).tolist() == [10.0, 1200.0, 30.0]


def test_integrate_by_phase_uses_label_of_interval_start():
    t = np.array([0.0, 10.0, 20.0, 30.0, 40.0])
    rate = np.ones(5)
    phase = np.array(["climb", "climb", "cruise", "cruise", "descent"], dtype=object)
    out = integrate_by_phase(t, rate, phase)
    assert out["climb"] == 20.0
    assert out["cruise"] == 20.0
    assert out["descent"] == 0.0  # the last sample starts no interval
    assert out["ground"] == 0.0
    assert sum(out.values()) == pytest.approx(integrate(t, rate))


def test_integration_degenerate_inputs():
    assert integrate(np.array([5.0]), np.array([1.0])) == 0.0
    assert interval_amounts([], []).size == 0


# ---- aircraft lookup ------------------------------------------------------------------


def test_resolve_known_type():
    m = resolve_aircraft("a321")
    assert m.typecode == "A321"
    assert m.n_engines == 2
    assert m.mtow_kg == 93_500
    assert m.idle_fuel_flow_kgs == pytest.approx(2 * 0.117)
    assert not m.is_fallback


def test_resolve_synonym_and_fallback():
    assert resolve_aircraft("B762").openap_type == "B763"
    fb = resolve_aircraft(None, fallback="A320")
    assert fb.is_fallback and fb.typecode == "A320"
    assert resolve_aircraft("C172", fallback="A320").is_fallback


def test_resolve_unknown_without_fallback_raises():
    with pytest.raises(UnsupportedAircraftError):
        resolve_aircraft("C172")
    with pytest.raises(UnsupportedAircraftError):
        resolve_aircraft(None)


# ---- fuel flow ------------------------------------------------------------------------


def labelled(make_flight, alt, speed, **kw):
    df, _ = clean_flight(make_flight(np.asarray(alt, float), speed_kt=speed, **kw))
    df["phase"] = label_phases(df)
    return df


def test_cruise_fuel_flow_matches_openap_at_constant_mass(make_flight):
    df = labelled(make_flight, np.full(400, 35000.0), 450.0)
    model = resolve_aircraft("A320")
    cfg = PipelineConfig(update_mass=False)
    out = estimate_fuel_flow(df, model, cfg)
    expected = float(fuel_flow_model("A320").enroute(mass=0.85 * 78_000, tas=450, alt=35000, vs=0))
    assert out["fuel_flow_kgs"].to_numpy() == pytest.approx(expected, rel=1e-9)
    duration = (df["timestamp"].iloc[-1] - df["timestamp"].iloc[0]).total_seconds()
    assert out["fuel_cum_kg"].iloc[-1] == pytest.approx(expected * duration, rel=1e-9)


def test_mass_decreases_by_burned_fuel_and_lowers_fuel_flow(make_flight):
    df = labelled(make_flight, np.full(1000, 35000.0), 450.0)
    model = resolve_aircraft("A320")
    fixed = estimate_fuel_flow(df, model, PipelineConfig(update_mass=False))
    burn = estimate_fuel_flow(df, model, PipelineConfig(update_mass=True))
    m0 = 0.85 * model.mtow_kg
    assert burn["mass_kg"].iloc[0] == pytest.approx(m0)
    assert burn["mass_kg"].iloc[-1] == pytest.approx(m0 - fixed["fuel_cum_kg"].iloc[-1])
    assert burn["fuel_flow_kgs"].iloc[-1] < fixed["fuel_flow_kgs"].iloc[-1]
    assert burn["fuel_cum_kg"].iloc[-1] < fixed["fuel_cum_kg"].iloc[-1]


def test_heavier_aircraft_burns_more(make_flight):
    df = labelled(make_flight, np.full(200, 35000.0), 450.0)
    model = resolve_aircraft("A320")
    light = estimate_fuel_flow(df, model, PipelineConfig(mass_fraction=0.7))
    heavy = estimate_fuel_flow(df, model, PipelineConfig(mass_fraction=0.95))
    assert heavy["fuel_cum_kg"].iloc[-1] > light["fuel_cum_kg"].iloc[-1]


def test_taxi_uses_icao_idle_and_climb_burns_more_than_descent(make_flight):
    taxi = np.zeros(60)
    climb = np.linspace(0, 20000, 81)[1:]
    descent = np.linspace(20000, 0, 101)[1:]
    alt = np.concatenate([taxi, climb, descent])
    speed = np.concatenate([np.full(60, 15.0), np.full(80, 280.0), np.full(100, 280.0)])
    df = labelled(make_flight, alt, speed)
    model = resolve_aircraft("A320")
    out = estimate_fuel_flow(df, model)
    taxi_ff = out.loc[out["thrust_mode"] == "taxi", "fuel_flow_kgs"]
    assert taxi_ff.to_numpy() == pytest.approx(model.idle_fuel_flow_kgs)
    climb_ff = out.loc[df["phase"] == "climb", "fuel_flow_kgs"].mean()
    descent_ff = out.loc[df["phase"] == "descent", "fuel_flow_kgs"].mean()
    assert climb_ff > 3 * descent_ff


def test_thrust_mode_identifies_takeoff_roll():
    phase = np.array(["ground"] * 5 + ["climb"] * 3 + ["ground"] * 2, dtype=object)
    gs = np.array([10, 10, 60, 120, 150, 160, 170, 180, 90, 20], dtype=float)
    mode = thrust_mode(phase, gs, 50.0)
    assert mode.tolist() == ["taxi", "taxi", "takeoff", "takeoff", "takeoff"] + ["airborne"] * 3 + [
        "taxi",
        "taxi",
    ]
    assert set(thrust_mode(np.array(["ground"] * 3, dtype=object), np.full(3, 80.0), 50.0)) == {
        "taxi"
    }


def test_thrust_mode_identifies_landing_roll():
    phase = np.array(["descent"] * 6 + ["ground"] * 3, dtype=object)
    gs = np.array([150, 140, 135, 120, 90, 60, 40, 20, 10], dtype=float)
    alt = np.array([900, 500, 150, 0, 0, 0, 0, 0, 0], dtype=float)
    mode = thrust_mode(phase, gs, 50.0, altitude_ft=alt)
    assert mode.tolist() == ["airborne"] * 3 + ["landing"] * 3 + ["taxi"] * 3
    # A recording that ends airborne has no landing roll.
    assert "landing" not in thrust_mode(phase[:6], gs[:6], 50.0, altitude_ft=alt[:6]).tolist()


def test_true_airspeed_prefers_tas_column():
    df = pd.DataFrame({"groundspeed_kt": [400.0, 410.0], "tas_kt": [450.0, np.nan]})
    assert true_airspeed_kt(df).tolist() == [450.0, 410.0]
    assert true_airspeed_kt(df.drop(columns="tas_kt")).tolist() == [400.0, 410.0]


def test_smoothed_vertical_rate_removes_alternating_noise():
    n = 50
    df = pd.DataFrame(
        {
            "timestamp": pd.date_range("2024-01-01", periods=n, freq="10s", tz="UTC"),
            "vertical_rate_fpm": -1500.0 + 600.0 * (-1.0) ** np.arange(n),
        }
    )
    smooth = smoothed_vertical_rate(df)
    assert np.abs(smooth[5:-5] + 1500.0).max() < 250.0


def test_type_plausibility_flags_impossible_altitude_and_speed(make_flight):
    model = resolve_aircraft("B738")
    df = labelled(make_flight, np.full(100, 47000.0), 450.0)
    assert any("ceiling" in w for w in check_type_plausibility(df, model))
    slow = labelled(make_flight, np.full(100, 37000.0), 200.0)
    slow["phase"] = "cruise"
    assert any("cruise ground speed" in w for w in check_type_plausibility(slow, model))
    ok = labelled(make_flight, np.full(100, 37000.0), 450.0)
    assert check_type_plausibility(ok, model) == []
