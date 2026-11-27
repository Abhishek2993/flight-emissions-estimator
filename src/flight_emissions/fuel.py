"""Fuel-flow estimation with OpenAP and integration over time.

Model
-----
For every sample the fuel flow (kg/s, all engines) is chosen by what the aircraft is doing:

* **Airborne** - :meth:`openap.FuelFlow.enroute`. OpenAP solves the point-mass equation
  ``T = D(m, TAS, h) + m g sin(γ)`` with ``γ = atan(vertical rate / TAS)`` for the thrust,
  then maps thrust to fuel flow with an engine fuel model fitted to the ICAO engine
  emissions databank. Thrust is limited to the idle..max range.
* **Take-off roll** - :meth:`openap.FuelFlow.takeoff`: maximum take-off thrust for ground
  samples faster than the ground-speed threshold before the first airborne sample.
* **Landing roll** - idle (ICAO LTO convention; reverse thrust is not modelled).
* **Taxi** - the ICAO engine-databank idle fuel flow (the 7 % thrust "taxi/idle" mode of
  the ICAO landing-and-take-off cycle) times the number of engines. OpenAP's drag model is
  meaningless at taxi speed, and ICAO idle is the inventory-standard taxi value.

Simplifications (documented as assumptions in the README)
---------------------------------------------------------
* **TAS ≈ ground speed.** ADS-B gives ground speed; true airspeed needs the wind vector,
  which is not in the data. A 50 kt head- or tailwind at 450 kt is an ~11 % speed error at
  that moment; over a flight, head- and tailwind segments partly cancel. If a ``tas_kt``
  column is present (e.g. from Mode S EHS or a weather model) it is used instead.
* **Mass.** Take-off mass = ``mass_fraction`` x MTOW (default 0.85). With ``update_mass``
  the mass then decreases by the fuel already burned (two-pass fixed point:
  pass 1 at constant mass, pass 2 with ``m(t) = m0 - burned(t)``), floored at the
  operating empty weight.
* **Vertical rate** is smoothed with a 30 s centred moving average before entering the
  flight-path angle, because single-sample noise would otherwise toggle idle/climb thrust.

Integration
-----------
Fuel is integrated with the trapezoidal rule over the actual sample times:
``F = Σ ½ (ff_i + ff_{i+1}) (t_{i+1} - t_i)``. Each interval is attributed to the phase of
its starting sample. Across a long data gap the rule assumes fuel flow changes linearly
between the two sides of the gap, which is reasonable in cruise and is reported.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from functools import lru_cache
from typing import Any

import numpy as np
import pandas as pd
from numpy.typing import NDArray

from flight_emissions.clean import elapsed_seconds
from flight_emissions.config import PipelineConfig
from flight_emissions.phases import GROUND, PHASES

FT_PER_M = 3.28084


class UnsupportedAircraftError(ValueError):
    """Raised when a type code is missing or unknown to OpenAP and no fallback is set."""


@dataclass(frozen=True)
class AircraftModel:
    """Aircraft and engine parameters taken from OpenAP's database."""

    typecode: str
    """Type code as reported in the data (or the fallback, when used)."""
    openap_type: str
    """Type actually modelled by OpenAP (may be a synonym, e.g. B744 for B748)."""
    name: str
    mtow_kg: float
    oew_kg: float
    n_engines: int
    engine: str
    idle_fuel_flow_kgs: float
    """ICAO databank idle (7 % thrust) fuel flow, all engines."""
    cruise_mach: float
    ceiling_ft: float
    seats: float
    """Typical seat count: mean of OpenAP's low/high two-class figures."""
    is_fallback: bool = False

    def to_dict(self) -> dict[str, Any]:
        """Return the model parameters as a dictionary."""
        return dict(self.__dict__)


@lru_cache(maxsize=64)
def _openap_aircraft(typecode: str) -> dict[str, Any]:
    from openap import prop

    return prop.aircraft(typecode, use_synonym=True)


@lru_cache(maxsize=64)
def fuel_flow_model(typecode: str) -> Any:
    """Cached :class:`openap.FuelFlow` instance for a type code (synonyms allowed)."""
    from openap import FuelFlow

    return FuelFlow(typecode, use_synonym=True)


def resolve_aircraft(typecode: str | None, fallback: str | None = None) -> AircraftModel:
    """Look up an aircraft type in OpenAP, using OpenAP synonyms and an optional fallback.

    Args:
        typecode: ICAO type designator from the data; may be None or unknown.
        fallback: Type to use when ``typecode`` is missing or unsupported.

    Returns:
        The aircraft model.

    Raises:
        UnsupportedAircraftError: if neither the type code nor the fallback can be modelled.
    """
    from openap import prop

    candidates = [(typecode, False), (fallback, True)]
    errors = []
    for code, is_fallback in candidates:
        if not code:
            continue
        try:
            ac = _openap_aircraft(code.upper())
        except ValueError as exc:
            errors.append(str(exc))
            continue
        engine = prop.engine(ac["engine"]["default"])
        pax = ac.get("pax") or {}
        seats = float(np.mean([pax.get("low", np.nan), pax.get("high", np.nan)]))
        return AircraftModel(
            typecode=code.upper(),
            openap_type=_openap_type_name(code),
            name=ac["aircraft"],
            mtow_kg=float(ac["mtow"]),
            oew_kg=float(ac["oew"]),
            n_engines=int(ac["engine"]["number"]),
            engine=str(ac["engine"]["default"]),
            idle_fuel_flow_kgs=float(engine["ff_idl"]) * int(ac["engine"]["number"]),
            cruise_mach=float(ac["cruise"]["mach"]),
            ceiling_ft=float(ac["ceiling"]) * FT_PER_M,
            seats=seats,
            is_fallback=is_fallback,
        )
    reason = "; ".join(errors) if errors else "no type code"
    raise UnsupportedAircraftError(f"cannot model aircraft type {typecode!r}: {reason}")


def _openap_type_name(code: str) -> str:
    """Name of the OpenAP model used for ``code`` (itself, or its synonym)."""
    from openap import prop

    code = code.lower()
    if code in prop.available_aircraft(use_synonym=False):
        return code.upper()
    match = prop.aircraft_synonym.query("orig == @code")
    return str(match["new"].iloc[0]).upper() if not match.empty else code.upper()


# ---------------------------------------------------------------------------------------
# Inputs: TAS and vertical rate
# ---------------------------------------------------------------------------------------
def true_airspeed_kt(df: pd.DataFrame) -> NDArray[np.float64]:
    """True airspeed: ``tas_kt`` when available, otherwise ground speed (no wind data)."""
    gs = df["groundspeed_kt"].to_numpy(dtype=float)
    if "tas_kt" in df.columns:
        tas = df["tas_kt"].to_numpy(dtype=float)
        return np.where(np.isnan(tas), gs, tas)
    return gs


def smoothed_vertical_rate(df: pd.DataFrame, window_s: float = 30.0) -> NDArray[np.float64]:
    """Centred moving average of the vertical rate over ``window_s`` seconds."""
    vr = df["vertical_rate_fpm"].astype(float)
    t = pd.to_datetime(df["timestamp"], utc=True)
    if len(vr) < 3:
        return vr.to_numpy()
    step = float(np.median(np.diff(t.astype("int64"))) / 1e9)
    window = max(int(round(window_s / step)), 1) if step > 0 else 1
    if window % 2 == 0:
        window += 1
    return vr.rolling(window, center=True, min_periods=1).mean().to_numpy()


# ---------------------------------------------------------------------------------------
# Integration
# ---------------------------------------------------------------------------------------
def interval_amounts(t_s: NDArray[np.float64], rate: NDArray[np.float64]) -> NDArray[np.float64]:
    """Trapezoidal amount of each interval ``[t_i, t_{i+1}]`` (array of ``n - 1`` values).

    ``rate`` is a flow (e.g. kg/s); the result is the quantity (kg) of each interval.
    """
    t_s = np.asarray(t_s, dtype=float)
    rate = np.asarray(rate, dtype=float)
    if t_s.size < 2:
        return np.zeros(0)
    return 0.5 * (rate[:-1] + rate[1:]) * np.diff(t_s)


def integrate(t_s: NDArray[np.float64], rate: NDArray[np.float64]) -> float:
    """Total amount of a flow integrated over time with the trapezoidal rule."""
    return float(np.sum(interval_amounts(t_s, rate)))


def cumulative(t_s: NDArray[np.float64], rate: NDArray[np.float64]) -> NDArray[np.float64]:
    """Amount accumulated from the first sample up to each sample (starts at 0)."""
    return np.concatenate([[0.0], np.cumsum(interval_amounts(t_s, rate))])


def integrate_by_phase(
    t_s: NDArray[np.float64], rate: NDArray[np.float64], phase: NDArray | pd.Series
) -> dict[str, float]:
    """Integrate a flow and attribute each interval to the phase of its first sample.

    Returns:
        ``{phase: amount}`` for every phase in :data:`PHASES` (0 when absent) plus any
        other label present.
    """
    phase = np.asarray(phase, dtype=object)
    amounts = interval_amounts(t_s, rate)
    out = dict.fromkeys(PHASES, 0.0)
    for label in np.unique(phase[:-1]) if phase.size > 1 else []:
        out[str(label)] = float(amounts[phase[:-1] == label].sum())
    return out


# ---------------------------------------------------------------------------------------
# Fuel flow
# ---------------------------------------------------------------------------------------
LANDING_ROLL_MARGIN_FT = 100.0
"""Samples within this height of the landing altitude after touchdown are the landing roll."""


def thrust_mode(
    phase: NDArray,
    groundspeed_kt: NDArray,
    threshold_kt: float,
    altitude_ft: NDArray | None = None,
) -> NDArray:
    """Classify samples as ``taxi``, ``takeoff``, ``landing`` or ``airborne``.

    * ``takeoff`` - ground samples faster than ``threshold_kt`` before the first airborne
      sample (the take-off roll): take-off thrust.
    * ``landing`` - non-ground samples after the last sample more than
      :data:`LANDING_ROLL_MARGIN_FT` above the landing altitude, when the flight ends on
      the ground (the landing roll, still too fast for the ground-speed rule): idle, as in
      the ICAO LTO cycle. The clean-configuration drag model is not valid on the runway.
    * ``taxi`` - other ground samples: ICAO idle.
    """
    phase = np.asarray(phase, dtype=object)
    on_ground = phase == GROUND
    mode = np.where(on_ground, "taxi", "airborne").astype(object)
    airborne_idx = np.flatnonzero(~on_ground)
    if airborne_idx.size:
        before_liftoff = np.arange(phase.size) < airborne_idx[0]
        mode[on_ground & before_liftoff & (np.asarray(groundspeed_kt) >= threshold_kt)] = "takeoff"
    if altitude_ft is not None and airborne_idx.size and on_ground[-1]:
        alt = np.asarray(altitude_ft, dtype=float)
        landing_alt = alt[-1]
        high = np.flatnonzero(alt > landing_alt + LANDING_ROLL_MARGIN_FT)
        if high.size:
            after = np.arange(phase.size) > high[-1]
            mode[after & ~on_ground] = "landing"
    return mode


def _fuel_flow_at_mass(
    mass_kg: NDArray[np.float64],
    tas_kt: NDArray[np.float64],
    altitude_ft: NDArray[np.float64],
    vertical_rate_fpm: NDArray[np.float64],
    mode: NDArray,
    model: AircraftModel,
) -> NDArray[np.float64]:
    ff_model = fuel_flow_model(model.typecode)
    ff = np.full(tas_kt.size, model.idle_fuel_flow_kgs)
    air = mode == "airborne"
    if air.any():
        # On the landing roll (airborne label, ~zero altitude, low speed) the clean-config
        # drag model is undefined and returns NaN; those samples fall back to idle below.
        with np.errstate(over="ignore", invalid="ignore"):
            ff[air] = ff_model.enroute(
                mass=mass_kg[air],
                tas=np.maximum(tas_kt[air], 1.0),
                alt=np.maximum(altitude_ft[air], 0.0),
                vs=vertical_rate_fpm[air],
            )
    roll = mode == "takeoff"
    if roll.any():
        ff[roll] = ff_model.takeoff(tas=tas_kt[roll], alt=np.maximum(altitude_ft[roll], 0.0))
    return np.nan_to_num(ff, nan=model.idle_fuel_flow_kgs)


def estimate_fuel_flow(
    df: pd.DataFrame,
    model: AircraftModel,
    config: PipelineConfig | None = None,
    initial_mass_kg: float | None = None,
) -> pd.DataFrame:
    """Estimate fuel flow for every sample of a phase-labelled trajectory.

    Args:
        df: Cleaned trajectory with a ``phase`` column.
        model: Aircraft model from :func:`resolve_aircraft`.
        config: Pipeline configuration (mass fraction, mass update, thresholds).
        initial_mass_kg: Overrides ``mass_fraction x MTOW`` (used by validation).

    Returns:
        DataFrame aligned with ``df`` with columns ``tas_kt``, ``vertical_rate_smooth_fpm``,
        ``thrust_mode``, ``mass_kg``, ``fuel_flow_kgs`` and ``fuel_cum_kg``.
    """
    config = config or PipelineConfig()
    t_s = elapsed_seconds(df["timestamp"])
    tas = true_airspeed_kt(df)
    alt = df["altitude_ft"].to_numpy(dtype=float)
    vr = smoothed_vertical_rate(df)
    mode = thrust_mode(
        df["phase"].to_numpy(),
        df["groundspeed_kt"].to_numpy(),
        config.ground_speed_threshold_kt,
        altitude_ft=alt,
    )
    m0 = initial_mass_kg if initial_mass_kg is not None else config.mass_fraction * model.mtow_kg
    mass = np.full(t_s.size, m0)

    ff = _fuel_flow_at_mass(mass, tas, alt, vr, mode, model)
    if config.update_mass:
        mass = np.maximum(m0 - cumulative(t_s, ff), model.oew_kg)
        ff = _fuel_flow_at_mass(mass, tas, alt, vr, mode, model)

    return pd.DataFrame(
        {
            "tas_kt": tas,
            "vertical_rate_smooth_fpm": vr,
            "thrust_mode": mode,
            "mass_kg": mass,
            "fuel_flow_kgs": ff,
            "fuel_cum_kg": cumulative(t_s, ff),
        },
        index=df.index,
    )


def check_type_plausibility(df: pd.DataFrame, model: AircraftModel) -> list[str]:
    """Flag trajectories that look inconsistent with the declared aircraft type.

    A wrong type code cannot be detected with certainty from a trajectory, but two cheap
    checks catch gross errors: flying above the type's certified ceiling, and a cruise
    ground speed far from the type's cruise Mach number.
    """
    from openap import aero

    warnings: list[str] = []
    max_alt = float(np.nanmax(df["altitude_ft"])) if len(df) else 0.0
    if max_alt > model.ceiling_ft + 1_000:
        warnings.append(
            f"max altitude {max_alt:,.0f} ft exceeds the {model.typecode} ceiling "
            f"({model.ceiling_ft:,.0f} ft): check the type code"
        )
    if "phase" in df.columns:
        cruise = df[df["phase"] == "cruise"]
        if len(cruise) > 10:
            gs = float(cruise["groundspeed_kt"].median())
            alt_m = float(cruise["altitude_ft"].median()) / FT_PER_M
            expected = float(aero.mach2tas(model.cruise_mach, alt_m)) / aero.kts
            if abs(gs - expected) / expected > 0.35:
                warnings.append(
                    f"median cruise ground speed {gs:.0f} kt is far from the "
                    f"{model.typecode} cruise speed (~{expected:.0f} kt): check the type code"
                )
    return warnings


def fuel_by_phase(
    df: pd.DataFrame, fuel_flow_kgs: NDArray, phase: pd.Series
) -> Mapping[str, float]:
    """Fuel burned per phase in kg (convenience wrapper around :func:`integrate_by_phase`)."""
    return integrate_by_phase(elapsed_seconds(df["timestamp"]), fuel_flow_kgs, phase.to_numpy())
