"""Operational inefficiency: extra distance flown, and level-offs in the descent.

Route inefficiency
------------------
``extra % = (flown - great circle) / great circle x 100`` where *great circle* is the
haversine distance between the first and last **airborne** samples and *flown* is the sum
of the haversine legs between consecutive airborne samples. The fuel this costs is
estimated as ``extra km x cruise fuel per km`` (cruise fuel / cruise distance of the same
flight): the extra distance is assumed to be flown at cruise efficiency, which makes the
estimate conservative (extra distance flown low down, e.g. vectoring, costs more per km).

Round trips (sightseeing, training, test flights) have a great-circle distance near zero,
so the percentage is meaningless; below ``min_great_circle_km`` it is not reported.

Descent inefficiency
--------------------
The descent window runs from the **top of the final descent** (normally the end of cruise,
see :func:`find_descent_window`) to ``descent_floor_above_landing_ft`` above the landing
altitude. Inside it:

* **Level-offs** are runs where the smoothed |vertical rate| stays below
  ``level_vr_threshold_fpm`` for at least ``min_level_off_s``.
* The **actual descent fuel** is compared with an **idealised continuous descent** over
  the *same ground distance* between the *same start and end altitudes*: the aircraft
  stays at its top-of-descent altitude, then descends at a constant flight-path angle
  ``cdo_glide_angle_deg`` (default 3°) to arrive exactly at the window end. If 3° does not
  fit in the available distance, the ideal is a constant gradient over the whole window.
  The ideal flies the actual *calibrated* airspeed at each distance (converted to TAS at the
  ideal altitude and capped at the type's cruise Mach), so the comparison isolates the
  vertical profile. Both profiles use the same OpenAP fuel model, the mass at top of
  descent and the longitudinal acceleration (60 s smoothed TAS derivative): a descent is
  an energy-management manoeuvre, and a decelerating aircraft can fly a shallower path at
  idle thrust. (The main fuel estimate ignores acceleration; here both profiles are treated
  identically, so wind noise in the ground-speed derivative largely cancels.)
  ``saving = actual - ideal``. A flight that already flies a continuous descent
  gets a saving close to zero. The saving therefore captures both level-offs and
  shallow, powered descent segments flown above idle thrust.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field

import numpy as np
import pandas as pd
from numpy.typing import NDArray

from flight_emissions.clean import elapsed_seconds
from flight_emissions.config import PipelineConfig
from flight_emissions.fuel import AircraftModel, fuel_flow_model, integrate, interval_amounts
from flight_emissions.geo import (
    FT_TO_KM,
    KT_TO_KMS,
    cumulative_distance_km,
    haversine_km,
    segment_lengths_km,
)
from flight_emissions.phases import CLIMB, CRUISE, GROUND, run_length_segments

MIN_CRUISE_DISTANCE_KM = 50.0
"""Below this cruise distance, fuel per km is taken from the whole airborne part instead."""


# ---------------------------------------------------------------------------------------
# Route inefficiency
# ---------------------------------------------------------------------------------------
@dataclass
class RouteInefficiency:
    """Flown distance versus great-circle distance, and the fuel the difference costs."""

    great_circle_km: float
    flown_km: float
    extra_km: float
    extra_pct: float
    fuel_per_km_kg: float
    extra_fuel_kg: float
    applicable: bool
    note: str = ""

    def to_dict(self) -> dict[str, float | bool | str]:
        """Return the result as a flat dictionary."""
        return asdict(self)


def route_inefficiency(
    df: pd.DataFrame, fuel_flow_kgs: NDArray[np.float64], config: PipelineConfig | None = None
) -> RouteInefficiency:
    """Compare the flown distance with the great circle between the airborne end points.

    Args:
        df: Trajectory with ``lat``, ``lon``, ``timestamp`` and ``phase``.
        fuel_flow_kgs: Fuel flow per sample (kg/s), aligned with ``df``.
        config: Pipeline configuration (``min_great_circle_km``).

    Returns:
        The route inefficiency result.
    """
    config = config or PipelineConfig()
    phase = df["phase"].to_numpy()
    airborne = phase != GROUND
    if airborne.sum() < 2:
        return RouteInefficiency(*([np.nan] * 6), applicable=False, note="no airborne data")

    lat = df["lat"].to_numpy(dtype=float)
    lon = df["lon"].to_numpy(dtype=float)
    idx = np.flatnonzero(airborne)
    first, last = idx[0], idx[-1]
    gc_km = float(haversine_km(lat[first], lon[first], lat[last], lon[last]))

    legs = segment_lengths_km(lat, lon)
    amounts = interval_amounts(elapsed_seconds(df["timestamp"]), fuel_flow_kgs)
    air_leg = airborne[:-1] & airborne[1:]
    flown_km = float(legs[air_leg].sum())

    cruise_leg = air_leg & (phase[:-1] == CRUISE)
    if legs[cruise_leg].sum() >= MIN_CRUISE_DISTANCE_KM:
        fuel_per_km = float(amounts[cruise_leg].sum() / legs[cruise_leg].sum())
        basis = "cruise"
    else:
        fuel_per_km = float(amounts[air_leg].sum() / max(flown_km, 1e-9))
        basis = "airborne average (cruise too short)"

    extra_km = flown_km - gc_km
    if gc_km < config.min_great_circle_km:
        return RouteInefficiency(
            great_circle_km=gc_km,
            flown_km=flown_km,
            extra_km=np.nan,
            extra_pct=np.nan,
            fuel_per_km_kg=fuel_per_km,
            extra_fuel_kg=np.nan,
            applicable=False,
            note=(
                f"circular flight: end points {gc_km:.0f} km apart "
                f"(< {config.min_great_circle_km:.0f} km), route inefficiency not meaningful"
            ),
        )
    return RouteInefficiency(
        great_circle_km=gc_km,
        flown_km=flown_km,
        extra_km=extra_km,
        extra_pct=100.0 * extra_km / gc_km,
        fuel_per_km_kg=fuel_per_km,
        extra_fuel_kg=extra_km * fuel_per_km,
        applicable=True,
        note=f"fuel per km from {basis}",
    )


# ---------------------------------------------------------------------------------------
# Level-off detection
# ---------------------------------------------------------------------------------------
@dataclass(frozen=True)
class LevelOff:
    """A level segment inside the descent (indices are positions in the trajectory)."""

    start_idx: int
    end_idx: int
    duration_s: float
    altitude_ft: float


def detect_level_offs(
    t_s: NDArray[np.float64],
    altitude_ft: NDArray[np.float64],
    vertical_rate_fpm: NDArray[np.float64],
    threshold_fpm: float = 300.0,
    min_duration_s: float = 20.0,
) -> list[LevelOff]:
    """Find level segments: runs with ``|vertical rate| < threshold`` lasting long enough.

    The duration of a run is the time between its first and last sample.

    Args:
        t_s: Sample times (s).
        altitude_ft: Altitudes (ft).
        vertical_rate_fpm: Vertical rates (ft/min), ideally smoothed.
        threshold_fpm: Maximum |vertical rate| counted as level.
        min_duration_s: Minimum duration of a level-off.

    Returns:
        Level-offs in time order.
    """
    level = np.abs(np.asarray(vertical_rate_fpm, dtype=float)) < threshold_fpm
    out: list[LevelOff] = []
    for seg in run_length_segments(np.where(level, "level", "moving")):
        if seg.label != "level":
            continue
        duration = float(t_s[seg.end] - t_s[seg.start])
        if duration >= min_duration_s:
            alt = float(np.median(altitude_ft[seg.start : seg.end + 1]))
            out.append(LevelOff(seg.start, seg.end, duration, alt))
    return out


# ---------------------------------------------------------------------------------------
# Idealised continuous descent
# ---------------------------------------------------------------------------------------
def ideal_descent_altitude(
    distance_km: NDArray[np.float64], start_alt_ft: float, end_alt_ft: float, glide_deg: float
) -> NDArray[np.float64]:
    """Altitude of the idealised continuous descent at each along-track distance.

    Level at ``start_alt_ft`` until the point from which a ``glide_deg`` descent reaches
    ``end_alt_ft`` exactly at the last distance; a constant gradient if it does not fit.
    """
    distance_km = np.asarray(distance_km, dtype=float)
    total = float(distance_km[-1])
    drop_km = (start_alt_ft - end_alt_ft) * FT_TO_KM
    needed = drop_km / np.tan(np.radians(glide_deg))
    if needed >= total or total <= 0:
        frac = distance_km / total if total > 0 else np.zeros_like(distance_km)
        return start_alt_ft - (start_alt_ft - end_alt_ft) * frac
    tod = total - needed
    gradient_ft_per_km = np.tan(np.radians(glide_deg)) / FT_TO_KM
    return np.where(
        distance_km <= tod, start_alt_ft, start_alt_ft - (distance_km - tod) * gradient_ft_per_km
    )


def _ideal_tas_kt(
    actual_tas_kt: NDArray, actual_alt_ft: NDArray, ideal_alt_ft: NDArray, cruise_mach: float
) -> NDArray:
    """Hold the actual calibrated airspeed at the ideal altitude, capped at cruise Mach."""
    from openap import aero

    ft = aero.ft
    cas = aero.tas2cas(actual_tas_kt * aero.kts, actual_alt_ft * ft)
    tas = aero.cas2tas(cas, ideal_alt_ft * ft)
    tas_cap = aero.mach2tas(cruise_mach, ideal_alt_ft * ft)
    return np.minimum(tas, tas_cap) / aero.kts


def acceleration_ms2(
    t_s: NDArray[np.float64], tas_kt: NDArray[np.float64], window_s: float = 60.0
) -> NDArray[np.float64]:
    """Longitudinal acceleration (m/s²) from a moving-average-smoothed TAS series."""
    from openap import aero

    if t_s.size < 3:
        return np.zeros(t_s.size)
    step = float(np.median(np.diff(t_s)))
    window = max(int(round(window_s / step)), 1) | 1 if step > 0 else 1
    v = pd.Series(np.asarray(tas_kt, dtype=float) * aero.kts)
    v_smooth = v.rolling(window, center=True, min_periods=1).mean().to_numpy()
    return np.gradient(v_smooth, t_s)


def find_descent_window(df: pd.DataFrame, config: PipelineConfig) -> tuple[int, int] | None:
    """Indices of the top of the *final* descent and of the descent-analysis floor.

    * **Floor** (window end): the first sample of the final run below
      ``landing altitude + descent_floor_above_landing_ft`` that lasts until landing.
      The landing altitude is the altitude of the last airborne sample.
    * **Top of descent** (window start): the last sample within 200 ft of the highest
      altitude reached after the last climb segment that precedes the floor. For a normal
      flight this is the end of cruise; for a flight with a go-around or a sightseeing
      pattern it is the top of the final descent only.

    Returns:
        ``(i0, i1)`` or None when the data holds no descent of at least 2,000 ft above the
        floor (e.g. the recording ends in cruise).
    """
    phase = df["phase"].to_numpy()
    alt = df["altitude_ft"].to_numpy(dtype=float)
    airborne = np.flatnonzero(phase != GROUND)
    if airborne.size < 3:
        return None
    last_air = int(airborne[-1])
    floor = float(alt[last_air]) + config.descent_floor_above_landing_ft
    above = np.flatnonzero(alt[: last_air + 1] > floor)
    if above.size == 0 or above[-1] >= last_air:
        return None
    i1 = int(above[-1]) + 1
    climbs = np.flatnonzero(phase[:i1] == CLIMB)
    start = int(climbs[-1]) if climbs.size else int(airborne[0])
    segment = alt[start:i1]
    top = float(np.nanmax(segment))
    i0 = start + int(np.flatnonzero(segment >= top - 200.0)[-1])
    if alt[i0] - floor < 2_000 or i1 - i0 < 3:
        return None
    return i0, i1


@dataclass
class DescentAnalysis:
    """Level-offs in the descent and the fuel a continuous descent would have saved."""

    applicable: bool
    note: str = ""
    tod_time: pd.Timestamp | None = None
    tod_altitude_ft: float = np.nan
    end_altitude_ft: float = np.nan
    distance_km: float = np.nan
    duration_s: float = np.nan
    n_level_offs: int = 0
    level_time_s: float = 0.0
    actual_fuel_kg: float = np.nan
    ideal_fuel_kg: float = np.nan
    saving_kg: float = np.nan
    saving_pct: float = np.nan
    level_offs: list[LevelOff] = field(default_factory=list)
    profile: pd.DataFrame | None = None
    """Per-sample comparison: distance, actual/ideal altitude and fuel flow, level flag."""

    def to_dict(self) -> dict[str, object]:
        """Scalar fields as a flat dictionary (the profile and level-off list excluded)."""
        return {k: v for k, v in self.__dict__.items() if k not in ("level_offs", "profile")}


def descent_analysis(
    df: pd.DataFrame,
    fuel: pd.DataFrame,
    model: AircraftModel,
    config: PipelineConfig | None = None,
) -> DescentAnalysis:
    """Detect level-offs in the descent and estimate the saving of a continuous descent.

    Args:
        df: Labelled trajectory (``phase`` column).
        fuel: Output of :func:`flight_emissions.fuel.estimate_fuel_flow` for ``df``.
        model: Aircraft model.
        config: Pipeline configuration.

    Returns:
        The descent analysis (``applicable=False`` with a note when there is no usable
        descent, e.g. the recording ends in cruise).
    """
    config = config or PipelineConfig()
    window = find_descent_window(df, config)
    if window is None:
        return DescentAnalysis(applicable=False, note="no descent from cruise in the data")
    i0, i1 = window
    sl = slice(i0, i1 + 1)
    w = df.iloc[sl]
    t_s = elapsed_seconds(w["timestamp"])
    alt = w["altitude_ft"].to_numpy(dtype=float)
    tas = np.maximum(fuel["tas_kt"].to_numpy(dtype=float)[sl], 50.0)
    vr = fuel["vertical_rate_smooth_fpm"].to_numpy(dtype=float)[sl]
    mass = float(fuel["mass_kg"].iloc[i0])
    dist = cumulative_distance_km(w["lat"].to_numpy(), w["lon"].to_numpy())

    level_offs = detect_level_offs(
        t_s, alt, vr, config.level_vr_threshold_fpm, config.min_level_off_s
    )
    level_flag = np.zeros(alt.size, dtype=bool)
    for lo in level_offs:
        level_flag[lo.start_idx : lo.end_idx + 1] = True
    level_offs = [
        LevelOff(lo.start_idx + i0, lo.end_idx + i0, lo.duration_s, lo.altitude_ft)
        for lo in level_offs
    ]

    ff_model = fuel_flow_model(model.typecode)
    masses = np.full(alt.size, mass)
    acc = acceleration_ms2(t_s, tas)
    ff_actual = np.asarray(
        ff_model.enroute(mass=masses, tas=tas, alt=alt, vs=vr, acc=acc), dtype=float
    )
    actual_fuel = integrate(t_s, ff_actual)

    alt_ideal = ideal_descent_altitude(dist, alt[0], alt[-1], config.cdo_glide_angle_deg)
    tas_ideal = _ideal_tas_kt(tas, alt, alt_ideal, model.cruise_mach)
    leg_speed = 0.5 * (tas_ideal[:-1] + tas_ideal[1:]) * KT_TO_KMS  # km/s, no wind
    t_ideal = np.concatenate([[0.0], np.cumsum(np.diff(dist) / leg_speed)])
    vr_ideal = np.gradient(alt_ideal, t_ideal) * 60.0 if t_ideal.size > 1 else np.zeros(1)
    acc_ideal = acceleration_ms2(t_ideal, tas_ideal)
    ff_ideal = np.asarray(
        ff_model.enroute(mass=masses, tas=tas_ideal, alt=alt_ideal, vs=vr_ideal, acc=acc_ideal),
        dtype=float,
    )
    ideal_fuel = integrate(t_ideal, ff_ideal)

    saving = actual_fuel - ideal_fuel
    profile = pd.DataFrame(
        {
            "timestamp": w["timestamp"].to_numpy(),
            "distance_km": dist,
            "altitude_actual_ft": alt,
            "altitude_ideal_ft": alt_ideal,
            "fuel_flow_actual_kgs": ff_actual,
            "fuel_flow_ideal_kgs": ff_ideal,
            "level_off": level_flag,
        }
    )
    return DescentAnalysis(
        applicable=True,
        note=f"top of descent to {config.descent_floor_above_landing_ft:.0f} ft above landing",
        tod_time=w["timestamp"].iloc[0],
        tod_altitude_ft=float(alt[0]),
        end_altitude_ft=float(alt[-1]),
        distance_km=float(dist[-1]),
        duration_s=float(t_s[-1]),
        n_level_offs=len(level_offs),
        level_time_s=float(sum(lo.duration_s for lo in level_offs)),
        actual_fuel_kg=actual_fuel,
        ideal_fuel_kg=ideal_fuel,
        saving_kg=saving,
        saving_pct=100.0 * saving / actual_fuel if actual_fuel > 0 else np.nan,
        level_offs=level_offs,
        profile=profile,
    )
