import numpy as np
import pandas as pd
import pytest

from flight_emissions.clean import (
    clean_flight,
    detect_altitude_spikes,
    detect_position_spikes,
    gap_statistics,
    interpolate_with_gap_limit,
    merge_duplicates,
)
from flight_emissions.config import PipelineConfig


def cruise(make_flight, n=60, **kw):
    return make_flight(np.full(n, 35000.0), step_s=5.0, speed_kt=450.0, **kw)


def test_clean_straight_flight_is_lossless(make_flight):
    df, report = clean_flight(cruise(make_flight))
    assert report.n_points_removed == 0
    assert report.n_altitude_spikes == 0
    assert np.diff(df["timestamp"].astype("int64")).tolist() == [10_000_000_000] * (len(df) - 1)
    assert df["groundspeed_kt"].to_numpy() == pytest.approx(450.0)
    assert df["altitude_ft"].to_numpy() == pytest.approx(35000.0)


def test_duplicates_are_removed_and_merged(make_flight):
    raw = cruise(make_flight)
    partial = raw.iloc[[10]].copy()
    partial["altitude_ft"] = np.nan  # same timestamp, fewer fields (another message type)
    dup = pd.concat([raw, raw.iloc[[3, 4]], partial]).sample(frac=1.0, random_state=0)
    merged, n = merge_duplicates(dup)
    assert n == 3
    assert len(merged) == len(raw)
    assert merged["altitude_ft"].notna().all()
    _, report = clean_flight(dup)
    assert report.n_duplicates == 3


def test_position_spike_is_removed(make_flight):
    raw = cruise(make_flight)
    raw.loc[20, "lat"] = 2.0  # ~220 km off track in 5 s
    _, report = clean_flight(raw)
    assert report.n_position_spikes == 1


def test_position_spike_detector_keeps_good_neighbours():
    t = np.arange(10) * 10.0
    lat = np.zeros(10)
    lon = np.arange(10) * 0.01
    lat[0] = 5.0  # first point is the bad one
    lat[6] = 5.0
    mask = detect_position_spikes(t, lat, lon, max_speed_kt=800.0)
    assert mask.tolist() == [True, False, False, False, False, False, True, False, False, False]


def test_altitude_spike_is_nulled_not_dropped(make_flight):
    raw = cruise(make_flight)
    raw.loc[30, "altitude_ft"] = 5000.0
    df, report = clean_flight(raw)
    assert report.n_altitude_spikes == 1
    assert report.n_position_spikes == 0
    assert df["altitude_ft"].to_numpy() == pytest.approx(35000.0)


def test_altitude_spike_detector_ignores_steady_climb():
    t = np.arange(20) * 10.0
    alt = 2000.0 + t * 50.0  # 3000 ft/min
    assert not detect_altitude_spikes(t, alt, 10_000.0).any()


def test_implausible_values_are_rejected(make_flight):
    raw = cruise(make_flight)
    raw.loc[5, "groundspeed_kt"] = 2000.0
    raw.loc[6, "vertical_rate_fpm"] = -25000.0
    raw.loc[7, "altitude_ft"] = 90000.0
    df, report = clean_flight(raw)
    assert report.n_groundspeed_rejected == 1
    assert report.n_vertical_rate_rejected == 1
    assert report.n_altitude_out_of_bounds == 1
    assert df["groundspeed_kt"].max() < 500
    assert df["vertical_rate_fpm"].abs().max() < 100


def test_short_gap_interpolated_long_gap_left_open(make_flight):
    raw = make_flight(np.linspace(30000, 31000, 400), step_s=10.0)
    raw = raw.drop(index=range(50, 60))  # 110 s gap -> interpolated
    raw = raw.drop(index=range(200, 260))  # 610 s gap -> left open
    df, report = clean_flight(raw)
    assert report.n_short_gaps == 1
    assert report.short_gap_total_s == pytest.approx(110.0)
    assert report.n_long_gaps == 1
    assert report.long_gap_total_s == pytest.approx(610.0)
    dt = np.diff(df["timestamp"].astype("int64") / 1e9)
    assert dt.max() == pytest.approx(610.0)  # no invented points inside the long gap
    # Altitude across the short gap follows the straight climb.
    t = (df["timestamp"] - df["timestamp"].iloc[0]).dt.total_seconds()
    expected = 30000 + t / 3990.0 * 1000.0
    assert df["altitude_ft"].to_numpy() == pytest.approx(expected.to_numpy(), abs=1e-6)


def test_resample_to_custom_step(make_flight):
    raw = make_flight(np.full(121, 20000.0), step_s=1.0)
    df, _ = clean_flight(raw, PipelineConfig(resample_step_s=20))
    assert len(df) == 7


def test_interpolation_respects_gap_limit():
    t_obs = np.array([0.0, 10.0, 500.0])
    v = np.array([0.0, 10.0, 500.0])
    out = interpolate_with_gap_limit(t_obs, v, np.array([0.0, 5.0, 100.0, 500.0]), max_gap_s=60.0)
    assert out[:2].tolist() == [0.0, 5.0]
    assert np.isnan(out[2])
    assert out[3] == 500.0


def test_gap_statistics():
    t = np.array([0, 10, 20, 80, 90, 1000], dtype=float)
    assert gap_statistics(t, 30.0, 300.0) == (1, 60.0, 1, 910.0)


def test_missing_altitude_on_ground_is_filled_and_vertical_rate_derived(make_flight):
    alt = np.concatenate([np.full(10, np.nan), np.linspace(500, 3000, 20)])
    speed = np.concatenate([np.full(10, 15.0), np.full(20, 160.0)])
    raw = make_flight(alt, speed_kt=speed)
    raw["vertical_rate_fpm"] = np.nan
    df, report = clean_flight(raw)
    assert report.n_ground_altitude_filled == 10
    assert df["altitude_ft"].iloc[0] == pytest.approx(500.0)
    assert df["vertical_rate_fpm"].iloc[-5] == pytest.approx(2500 / 190 * 60, rel=1e-6)


def test_missing_groundspeed_derived_from_positions(make_flight):
    raw = cruise(make_flight)
    raw["groundspeed_kt"] = np.nan
    df, _ = clean_flight(raw)
    assert df["groundspeed_kt"].to_numpy() == pytest.approx(450.0, rel=1e-6)


def test_too_few_points(make_flight):
    df, report = clean_flight(make_flight(np.array([1000.0, 2000.0])))
    assert df.empty
    assert report.n_output == 0


def test_report_summary_mentions_counts(make_flight):
    _, report = clean_flight(cruise(make_flight))
    assert "TEST01" in report.summary()
    assert report.to_dict()["n_points_removed"] == 0
