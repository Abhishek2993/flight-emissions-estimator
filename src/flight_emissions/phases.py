"""Segment a flight into ground / climb / cruise / level / descent.

Labelling is done by OpenAP's fuzzy-logic classifier (:class:`openap.phase.FlightPhase`,
Sun et al., 2017), which looks at the mean altitude, speed and rate of climb in fixed time
windows. Two post-processing rules make it robust on real data:

1. **Ground override** - OpenAP's "ground" membership function assumes the airport is at
   sea level (it fades out between 0 and 200 ft). At Zurich (1,416 ft) or Denver (5,434 ft)
   taxiing aircraft would be labelled "level". Any sample slower than
   ``ground_speed_threshold_kt`` (default 50 kt, below any jet's flying speed) is ground.
2. **Cruise band** - OpenAP separates "cruise" from "level" by speed. Level flight within
   ``cruise_band_ft`` (default 4,000 ft) of the flight's highest altitude, and above
   10,000 ft, is relabelled cruise, so a cruise into a strong headwind stays a cruise
   while a level-off during the descent stays "level".
3. **Short-segment merging** - window-based classification flickers at phase boundaries
   (e.g. ``climb, level, climb`` for a 60 s pause in a climb). Segments shorter than
   ``min_segment_s`` are absorbed into a neighbour, shortest first, until none remain.
   Unclassified (``unknown``) segments are always absorbed.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from numpy.typing import NDArray

from flight_emissions.clean import elapsed_seconds
from flight_emissions.config import PipelineConfig

GROUND, CLIMB, CRUISE, LEVEL, DESCENT, UNKNOWN = (
    "ground",
    "climb",
    "cruise",
    "level",
    "descent",
    "unknown",
)
PHASES: tuple[str, ...] = (GROUND, CLIMB, CRUISE, LEVEL, DESCENT)
"""Phase names in the order used for tables and plots."""

OPENAP_LABELS: dict[str, str] = {
    "GND": GROUND,
    "CL": CLIMB,
    "CR": CRUISE,
    "LVL": LEVEL,
    "DE": DESCENT,
    "NA": UNKNOWN,
}


@dataclass(frozen=True)
class Segment:
    """A run of consecutive samples sharing one phase label (``end`` is inclusive)."""

    start: int
    end: int
    label: str


def openap_labels(
    t_s: NDArray[np.float64],
    altitude_ft: NDArray[np.float64],
    speed_kt: NDArray[np.float64],
    vertical_rate_fpm: NDArray[np.float64],
    window_s: int = 60,
) -> NDArray[np.str_]:
    """Raw phase labels from OpenAP's fuzzy-logic classifier, mapped to our phase names."""
    from openap.phase import FlightPhase

    fp = FlightPhase()
    fp.set_trajectory(
        np.asarray(t_s, dtype=float),
        np.asarray(altitude_ft, dtype=float),
        np.asarray(speed_kt, dtype=float),
        np.asarray(vertical_rate_fpm, dtype=float),
    )
    raw = fp.phaselabel(twindow=window_s)
    return np.array([OPENAP_LABELS.get(label, UNKNOWN) for label in raw], dtype=object)


def run_length_segments(labels: NDArray | list[str]) -> list[Segment]:
    """Split a label sequence into runs of identical labels."""
    labels = np.asarray(labels, dtype=object)
    if labels.size == 0:
        return []
    change = np.flatnonzero(labels[1:] != labels[:-1]) + 1
    starts = np.concatenate([[0], change])
    ends = np.concatenate([change - 1, [labels.size - 1]])
    return [Segment(int(s), int(e), str(labels[s])) for s, e in zip(starts, ends, strict=True)]


def _segment_duration(seg: Segment, t_s: NDArray[np.float64], step_s: float) -> float:
    """Duration of a segment: from its first sample to the first sample of the next one."""
    end_time = t_s[seg.end + 1] if seg.end + 1 < t_s.size else t_s[seg.end] + step_s
    return float(end_time - t_s[seg.start])


def merge_short_segments(
    t_s: NDArray[np.float64],
    labels: NDArray | list[str],
    min_duration_s: float,
) -> NDArray[np.str_]:
    """Absorb segments shorter than ``min_duration_s`` into their neighbours.

    Rule for the replacement label of a short segment:

    * both neighbours share a label -> that label (``climb, level, climb`` -> climb);
    * only one neighbour exists -> that neighbour;
    * otherwise -> the longer neighbour.

    The shortest offending segment is merged first and the segmentation recomputed, so
    the result does not depend on scanning direction. ``unknown`` segments are merged
    whatever their length.

    Args:
        t_s: Sample times in seconds (monotonic).
        labels: Phase label per sample.
        min_duration_s: Minimum segment duration to keep.

    Returns:
        A new label array.
    """
    t_s = np.asarray(t_s, dtype=float)
    out = np.asarray(labels, dtype=object).copy()
    step_s = float(np.median(np.diff(t_s))) if t_s.size > 1 else 0.0

    while True:
        segs = run_length_segments(out)
        if len(segs) <= 1:
            break
        durations = [_segment_duration(s, t_s, step_s) for s in segs]
        candidates = [
            i for i, s in enumerate(segs) if s.label == UNKNOWN or durations[i] < min_duration_s
        ]
        if not candidates:
            break
        # unknown first, then shortest
        i = min(candidates, key=lambda k: (segs[k].label != UNKNOWN, durations[k]))
        prev_seg = segs[i - 1] if i > 0 else None
        next_seg = segs[i + 1] if i + 1 < len(segs) else None
        if prev_seg and next_seg and prev_seg.label == next_seg.label:
            new = prev_seg.label
        elif prev_seg is None or next_seg is None:
            new = (next_seg or prev_seg).label  # type: ignore[union-attr]
        else:
            new = prev_seg.label if durations[i - 1] >= durations[i + 1] else next_seg.label
        out[segs[i].start : segs[i].end + 1] = new
    return out


CRUISE_MIN_ALTITUDE_FT = 10_000.0


def apply_cruise_band(
    labels: NDArray, altitude_ft: NDArray[np.float64], config: PipelineConfig
) -> NDArray:
    """Relabel level flight near the flight's top altitude as cruise (see module notes)."""
    labels = np.asarray(labels, dtype=object).copy()
    if altitude_ft.size == 0 or np.all(np.isnan(altitude_ft)):
        return labels
    floor = max(np.nanmax(altitude_ft) - config.cruise_band_ft, CRUISE_MIN_ALTITUDE_FT)
    labels[(labels == LEVEL) & (altitude_ft >= floor)] = CRUISE
    return labels


def label_phases(df: pd.DataFrame, config: PipelineConfig | None = None) -> pd.Series:
    """Label every sample of a cleaned trajectory with its flight phase.

    The speed passed to OpenAP is the ground speed (see the TAS simplification in
    :mod:`flight_emissions.fuel`).

    Args:
        df: Cleaned trajectory (uniform time step).
        config: Pipeline configuration.

    Returns:
        A string Series named ``phase``, aligned with ``df``.
    """
    config = config or PipelineConfig()
    if df.empty:
        return pd.Series([], dtype=object, name="phase")
    t_s = elapsed_seconds(df["timestamp"])
    labels = openap_labels(
        t_s,
        df["altitude_ft"].to_numpy(dtype=float),
        df["groundspeed_kt"].to_numpy(dtype=float),
        df["vertical_rate_fpm"].to_numpy(dtype=float),
        window_s=config.phase_window_s,
    )
    slow = df["groundspeed_kt"].to_numpy(dtype=float) < config.ground_speed_threshold_kt
    labels[slow] = GROUND
    labels = apply_cruise_band(labels, df["altitude_ft"].to_numpy(dtype=float), config)
    labels = merge_short_segments(t_s, labels, config.min_segment_s)
    return pd.Series(labels, index=df.index, name="phase")


def phase_segments(df: pd.DataFrame) -> pd.DataFrame:
    """Summarise a labelled trajectory as one row per phase segment.

    Returns:
        Columns ``phase``, ``start``, ``end`` (timestamps), ``duration_s``,
        ``altitude_start_ft`` and ``altitude_end_ft``.
    """
    rows = []
    t_s = elapsed_seconds(df["timestamp"])
    step_s = float(np.median(np.diff(t_s))) if t_s.size > 1 else 0.0
    for seg in run_length_segments(df["phase"].to_numpy()):
        rows.append(
            {
                "phase": seg.label,
                "start": df["timestamp"].iloc[seg.start],
                "end": df["timestamp"].iloc[seg.end],
                "duration_s": _segment_duration(seg, t_s, step_s),
                "altitude_start_ft": float(df["altitude_ft"].iloc[seg.start]),
                "altitude_end_ft": float(df["altitude_ft"].iloc[seg.end]),
            }
        )
    return pd.DataFrame(rows)
