"""Validation: ICAO cross-check and a fuel-model check against recorded fuel flow.

1. **ICAO Carbon Emissions Calculator cross-check** (:func:`compare_with_icao`).
   Reference values are entered by hand from the ICAO calculator into
   ``data/reference/icao_reference.csv`` (columns ``origin, destination, typecode,
   icao_co2_kg_per_pax, icao_co2_kg_per_flight, note``). Either value may be given.
   Our per-flight CO2 is converted to a per-passenger figure with
   ``CO2 / (typical seats x load factor)``. Rows without a reference value are reported as
   *awaiting ICAO value* rather than silently dropped.

2. **Flight-data-recorder check** (:func:`check_fuel_model_against_fdr`). The A320 sample
   in ``traffic`` contains *measured* fuel flow and gross weight. Running the same OpenAP
   model on it under three input scenarios separates the error sources:

   * recorded mass + TAS from recorded CAS -> error of the OpenAP model itself;
   * recorded mass + TAS = ground speed   -> adds the no-wind simplification;
   * 0.85 x MTOW + TAS = ground speed     -> the pipeline's default assumptions.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from flight_emissions.clean import elapsed_seconds
from flight_emissions.config import PipelineConfig
from flight_emissions.fuel import (
    fuel_flow_model,
    integrate,
    resolve_aircraft,
    smoothed_vertical_rate,
)

ICAO_COLUMNS = [
    "origin",
    "destination",
    "typecode",
    "icao_co2_kg_per_pax",
    "icao_co2_kg_per_flight",
    "note",
]

KNOWN_DIFFERENCE_REASONS: tuple[str, ...] = (
    "Per-passenger allocation: ICAO divides the aircraft's CO2 by passengers using "
    "route-group load factors and removes a share for belly cargo; we use typical seats "
    "x an assumed load factor and allocate everything to passengers.",
    "Aircraft mix: ICAO averages over the aircraft types operating the route; we model "
    "the specific type that flew.",
    "Distance: ICAO uses great-circle distance plus a fixed routing correction; we use the "
    "actual trajectory, including holding, vectoring and (for ELY1747) a sightseeing pattern.",
    "Mass: we assume take-off mass = 0.85 x MTOW for every flight; real take-off mass "
    "depends on payload and the fuel needed for the stage length (short flights are lighter).",
    "Model: OpenAP fuel flow has a typical error of a few percent to ~10 % (see the "
    "flight-data-recorder check); ground speed is used as true airspeed (no wind data).",
    "Scope: our figure is gate-to-gate as observed in ADS-B; taxi time outside receiver "
    "coverage is missing, while ICAO assumes a standard LTO cycle.",
)


# ---------------------------------------------------------------------------------------
# ICAO cross-check
# ---------------------------------------------------------------------------------------
def load_icao_reference(path: str | Path) -> pd.DataFrame:
    """Read the hand-entered ICAO reference table (missing columns are added empty)."""
    ref = pd.read_csv(path, dtype={"origin": str, "destination": str, "typecode": str})
    for col in ICAO_COLUMNS:
        if col not in ref.columns:
            ref[col] = np.nan
    for col in ("icao_co2_kg_per_pax", "icao_co2_kg_per_flight"):
        ref[col] = pd.to_numeric(ref[col], errors="coerce")
    for col in ("origin", "destination", "typecode"):
        ref[col] = ref[col].fillna("").str.strip().str.upper()
    ref["note"] = ref["note"].fillna("")
    return ref[ICAO_COLUMNS]


def _pct_diff(ours: float, ref: float) -> float:
    if ref is None or not np.isfinite(ref) or ref == 0:
        return np.nan
    return 100.0 * (ours - ref) / ref


def compare_with_icao(
    summary: pd.DataFrame, reference: pd.DataFrame, config: PipelineConfig | None = None
) -> pd.DataFrame:
    """Compare per-flight results with the ICAO reference values.

    Args:
        summary: Flight summary table with ``flight_id, origin, destination, typecode,
            co2_kg, seats``.
        reference: Output of :func:`load_icao_reference`.
        config: Pipeline configuration (``load_factor``).

    Returns:
        One row per reference route with our values, ICAO values, % differences and a
        ``status`` column.
    """
    config = config or PipelineConfig()
    rows = []
    for _, ref in reference.iterrows():
        match = summary[
            (summary["origin"] == ref["origin"]) & (summary["destination"] == ref["destination"])
        ]
        if ref["typecode"]:
            match = match[match["typecode"] == ref["typecode"]]
        base = {
            "origin": ref["origin"],
            "destination": ref["destination"],
            "typecode": ref["typecode"],
            "icao_co2_kg_per_flight": ref["icao_co2_kg_per_flight"],
            "icao_co2_kg_per_pax": ref["icao_co2_kg_per_pax"],
            "note": ref["note"],
        }
        if match.empty:
            rows.append({**base, "flight_id": None, "status": "no matching flight"})
            continue
        flight = match.iloc[0]
        co2 = float(flight["co2_kg"])
        pax = float(flight["seats"]) * config.load_factor
        per_pax = co2 / pax if pax > 0 else np.nan
        has_ref = np.isfinite(ref["icao_co2_kg_per_flight"]) or np.isfinite(
            ref["icao_co2_kg_per_pax"]
        )
        rows.append(
            {
                **base,
                "flight_id": flight["flight_id"],
                "our_co2_kg_per_flight": co2,
                "assumed_passengers": pax,
                "our_co2_kg_per_pax": per_pax,
                "diff_pct_per_flight": _pct_diff(co2, ref["icao_co2_kg_per_flight"]),
                "diff_pct_per_pax": _pct_diff(per_pax, ref["icao_co2_kg_per_pax"]),
                "status": "compared" if has_ref else "awaiting ICAO value",
            }
        )
    cols = [
        "origin",
        "destination",
        "typecode",
        "flight_id",
        "our_co2_kg_per_flight",
        "icao_co2_kg_per_flight",
        "diff_pct_per_flight",
        "assumed_passengers",
        "our_co2_kg_per_pax",
        "icao_co2_kg_per_pax",
        "diff_pct_per_pax",
        "status",
        "note",
    ]
    return pd.DataFrame(rows).reindex(columns=cols)


# ---------------------------------------------------------------------------------------
# Flight-data-recorder check
# ---------------------------------------------------------------------------------------
FDR_SCENARIOS: dict[str, tuple[str, str]] = {
    "recorded mass, TAS from CAS": ("recorded", "cas"),
    "recorded mass, TAS = ground speed": ("recorded", "groundspeed"),
    "0.85 x MTOW, TAS = ground speed (pipeline default)": ("assumed", "groundspeed"),
}


def check_fuel_model_against_fdr(
    fdr: pd.DataFrame,
    typecode: str = "A320",
    config: PipelineConfig | None = None,
    step_s: int = 10,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Compare OpenAP fuel flow with fuel flow measured by a flight-data recorder.

    Args:
        fdr: Columns ``timestamp, altitude_ft, groundspeed_kt, cas_kt, mass_kg,
            fuel_flow_kgh`` (see :func:`flight_emissions.sample_data.build_fdr_reference`).
        typecode: Aircraft type of the recording.
        config: Pipeline configuration (mass fraction).
        step_s: Resampling step, matching the pipeline's default.

    Returns:
        ``(summary, timeseries)``: one row per scenario with total-fuel error, per-sample
        mean absolute percentage error and correlation; and the resampled time series with
        the recorded and modelled fuel flows (kg/s).
    """
    from openap import aero

    config = config or PipelineConfig()
    model = resolve_aircraft(typecode)
    ts = (
        fdr.set_index(pd.to_datetime(fdr["timestamp"], utc=True))
        .drop(columns="timestamp")
        .resample(f"{step_s}s")
        .mean()
        .interpolate(limit=3)
        .dropna()
        .rename_axis("timestamp")
        .reset_index()
    )
    t_s = elapsed_seconds(ts["timestamp"])
    alt = ts["altitude_ft"].to_numpy()
    ts["vertical_rate_fpm"] = np.gradient(alt, t_s) * 60.0
    vr = smoothed_vertical_rate(ts)
    tas_cas = aero.cas2tas(ts["cas_kt"].to_numpy() * aero.kts, alt * aero.ft) / aero.kts
    recorded = ts["fuel_flow_kgh"].to_numpy() / 3600.0
    ff_model = fuel_flow_model(model.typecode)

    out = pd.DataFrame({"timestamp": ts["timestamp"], "altitude_ft": alt, "recorded_kgs": recorded})
    rows = []
    for name, (mass_src, speed_src) in FDR_SCENARIOS.items():
        if mass_src == "recorded":
            mass = ts["mass_kg"].to_numpy()
        else:
            m0 = config.mass_fraction * model.mtow_kg
            mass = np.full(t_s.size, m0)
        tas = tas_cas if speed_src == "cas" else ts["groundspeed_kt"].to_numpy()
        ff = np.asarray(ff_model.enroute(mass=mass, tas=tas, alt=alt, vs=vr), dtype=float)
        if mass_src == "assumed" and config.update_mass:  # same two-pass update as the pipeline
            burned = np.concatenate([[0.0], np.cumsum(0.5 * (ff[1:] + ff[:-1]) * np.diff(t_s))])
            mass = np.maximum(m0 - burned, model.oew_kg)
            ff = np.asarray(ff_model.enroute(mass=mass, tas=tas, alt=alt, vs=vr), dtype=float)
        out[name] = ff
        model_total = integrate(t_s, ff)
        recorded_total = integrate(t_s, recorded)
        valid = recorded > 0.05
        rows.append(
            {
                "scenario": name,
                "model_fuel_kg": model_total,
                "recorded_fuel_kg": recorded_total,
                "total_error_pct": 100.0 * (model_total - recorded_total) / recorded_total,
                "flow_mape_pct": float(
                    100.0 * np.mean(np.abs(ff[valid] - recorded[valid]) / recorded[valid])
                ),
                "flow_correlation": float(np.corrcoef(ff, recorded)[0, 1]),
                "mean_mass_kg": float(np.mean(mass)),
            }
        )
    return pd.DataFrame(rows), out
