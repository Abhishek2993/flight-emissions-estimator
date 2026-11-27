import numpy as np
import pandas as pd
import pytest

from flight_emissions.clean import clean_flight
from flight_emissions.config import PipelineConfig
from flight_emissions.phases import (
    CLIMB,
    CRUISE,
    DESCENT,
    GROUND,
    LEVEL,
    UNKNOWN,
    apply_cruise_band,
    label_phases,
    merge_short_segments,
    phase_segments,
    run_length_segments,
)


def labels(*runs: tuple[str, int]) -> np.ndarray:
    return np.array([lab for lab, n in runs for _ in range(n)], dtype=object)


def segs(arr) -> list[tuple[str, int]]:
    return [(s.label, s.end - s.start + 1) for s in run_length_segments(arr)]


T10 = np.arange(1000) * 10.0  # 10 s grid


def test_run_length_segments():
    arr = labels((CLIMB, 3), (LEVEL, 2), (CLIMB, 1))
    assert segs(arr) == [(CLIMB, 3), (LEVEL, 2), (CLIMB, 1)]
    assert run_length_segments([]) == []


def test_short_segment_between_identical_neighbours_takes_their_label():
    arr = labels((CLIMB, 30), (LEVEL, 3), (CLIMB, 30))
    out = merge_short_segments(T10[: arr.size], arr, min_duration_s=60)
    assert segs(out) == [(CLIMB, 63)]


def test_short_segment_between_different_neighbours_joins_longer_one():
    arr = labels((CLIMB, 10), (LEVEL, 2), (CRUISE, 40))
    out = merge_short_segments(T10[: arr.size], arr, min_duration_s=60)
    assert segs(out) == [(CLIMB, 10), (CRUISE, 42)]


def test_short_segment_at_edges_joins_only_neighbour():
    arr = labels((LEVEL, 2), (CLIMB, 20), (DESCENT, 1))
    out = merge_short_segments(T10[: arr.size], arr, min_duration_s=60)
    assert segs(out) == [(CLIMB, 23)]


def test_long_segments_are_untouched_and_unknown_always_merged():
    arr = labels((CLIMB, 10), (UNKNOWN, 20), (CLIMB, 10), (CRUISE, 10))
    out = merge_short_segments(T10[: arr.size], arr, min_duration_s=60)
    assert segs(out) == [(CLIMB, 40), (CRUISE, 10)]


def test_shortest_first_merge_order():
    # The 10 s descent blip merges first (into cruise), then the 30 s level run.
    arr = labels((CRUISE, 20), (DESCENT, 1), (CRUISE, 2), (LEVEL, 3), (CRUISE, 20))
    out = merge_short_segments(T10[: arr.size], arr, min_duration_s=60)
    assert segs(out) == [(CRUISE, 46)]


def test_merge_single_segment_and_duration_uses_time_not_samples():
    assert segs(merge_short_segments(T10[:5], labels((LEVEL, 5)), 600)) == [(LEVEL, 5)]
    # Three climb samples span 500 s (a data gap), so the climb survives and absorbs the
    # three level samples, which only span 30 s.
    t = np.array([0, 10, 20, 500, 510, 520], dtype=float)
    arr = labels((CLIMB, 3), (LEVEL, 3))
    out = merge_short_segments(t, arr, min_duration_s=60)
    assert segs(out) == [(CLIMB, 6)]


def test_cruise_band_relabels_only_near_top_altitude():
    alt = np.array([37000.0, 37000.0, 25000.0, 9000.0])
    arr = np.array([LEVEL, CRUISE, LEVEL, LEVEL], dtype=object)
    out = apply_cruise_band(arr, alt, PipelineConfig())
    assert out.tolist() == [CRUISE, CRUISE, LEVEL, LEVEL]
    low = apply_cruise_band(np.array([LEVEL]), np.array([8000.0]), PipelineConfig())
    assert low.tolist() == [LEVEL]


def synthetic_mission(make_flight):
    """Taxi, climb, cruise, descent, taxi on a 10 s grid with known durations."""
    taxi = np.zeros(60)
    climb = np.linspace(0, 35000, 121)[1:]  # 20 min at 1750 ft/min
    cruise = np.full(360, 35000.0)  # 60 min
    descent = np.linspace(35000, 0, 151)[1:]  # 25 min at 1400 ft/min
    alt = np.concatenate([taxi, climb, cruise, descent, taxi])
    speed = np.concatenate(
        [
            np.full(60, 15.0),
            np.full(120, 280.0),
            np.full(360, 450.0),
            np.full(150, 300.0),
            np.full(60, 15.0),
        ]
    )
    return make_flight(alt, speed_kt=speed)


def test_label_phases_on_synthetic_mission(make_flight):
    df, _ = clean_flight(synthetic_mission(make_flight))
    df["phase"] = label_phases(df)
    order = [s.label for s in run_length_segments(df["phase"].to_numpy())]
    assert order == [GROUND, CLIMB, CRUISE, DESCENT, GROUND]
    table = phase_segments(df)
    cruise_minutes = table.loc[table.phase == CRUISE, "duration_s"].sum() / 60
    assert cruise_minutes == pytest.approx(60, abs=3)
    assert table["duration_s"].sum() == pytest.approx(len(df) * 10.0)


def test_ground_override_at_elevated_airport(make_flight):
    # Taxiing at 5,400 ft (Denver): OpenAP alone would not call this ground.
    df = make_flight(np.full(60, 5400.0), speed_kt=12.0)
    out = label_phases(df)
    assert set(out) == {GROUND}


def test_label_phases_empty():
    empty = pd.DataFrame(columns=["timestamp", "altitude_ft", "groundspeed_kt"])
    assert label_phases(empty).empty
