"""Orchestrate the pipeline: ingest -> clean -> phases -> fuel -> emissions -> inefficiency.

:func:`process_flight` runs every stage on one flight and returns a :class:`FlightResult`.
:func:`run_pipeline` processes a directory, runs the validations and writes all outputs:

====================================  ====================================================
``flights_summary.csv``               one row per flight: fuel, emissions, inefficiency
``phase_breakdown.csv``               one row per flight x phase: time, distance, fuel,
                                      emissions
``cleaning_report.csv``               what the cleaning stage removed, per flight
``level_offs.csv``                    every level-off found in a descent
``icao_validation.csv``               cross-check against hand-entered ICAO values
``fdr_fuel_check.csv``                fuel model vs recorded A320 fuel flow
``timeseries/<flight_id>.parquet``    per-sample results (used by the dashboard)
``run_config.json``                   the configuration of the run
``summary.md``                        the Markdown tables quoted in the README
====================================  ====================================================

A flight that cannot be modelled (no usable points, unknown aircraft type) is reported
with a ``status`` explaining why, instead of stopping the run.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

from flight_emissions.clean import CleaningReport, clean_flight, elapsed_seconds
from flight_emissions.config import PipelineConfig
from flight_emissions.emissions import SPECIES, emission_rates, emissions_by_phase
from flight_emissions.fuel import (
    AircraftModel,
    UnsupportedAircraftError,
    check_type_plausibility,
    estimate_fuel_flow,
    integrate_by_phase,
    resolve_aircraft,
)
from flight_emissions.geo import haversine_km
from flight_emissions.inefficiency import (
    DescentAnalysis,
    RouteInefficiency,
    descent_analysis,
    route_inefficiency,
)
from flight_emissions.ingest import load_flights
from flight_emissions.phases import GROUND, PHASES, label_phases
from flight_emissions.validate import (
    check_fuel_model_against_fdr,
    compare_with_icao,
    load_icao_reference,
)

logger = logging.getLogger(__name__)

DEFAULT_REFERENCE = Path("data/reference/icao_reference.csv")
DEFAULT_FDR = Path("data/reference/a320_fdr_fuelflow.parquet")


@dataclass
class FlightResult:
    """Everything the pipeline computed for one flight."""

    flight_id: str
    meta: dict[str, object]
    cleaning: CleaningReport
    status: str = "ok"
    warnings: list[str] = field(default_factory=list)
    model: AircraftModel | None = None
    timeseries: pd.DataFrame | None = None
    phase_table: pd.DataFrame | None = None
    route: RouteInefficiency | None = None
    descent: DescentAnalysis | None = None

    @property
    def ok(self) -> bool:
        """True when fuel and emissions were estimated."""
        return self.status == "ok"

    def summary_row(self) -> dict[str, object]:
        """One flat row for ``flights_summary.csv``."""
        row: dict[str, object] = {"flight_id": self.flight_id, **self.meta, "status": self.status}
        row["warnings"] = " | ".join(self.warnings)
        row["long_gap_min"] = self.cleaning.long_gap_total_s / 60.0
        if not self.ok or self.timeseries is None or self.phase_table is None:
            return row
        ts, pt = self.timeseries, self.phase_table
        t_s = elapsed_seconds(ts["timestamp"])
        assert self.model is not None and self.route is not None and self.descent is not None
        row.update(
            {
                "modelled_as": self.model.openap_type,
                "seats": self.model.seats,
                "block_time_h": t_s[-1] / 3600.0 if t_s.size else 0.0,
                "airborne_time_h": pt.loc[pt.index != GROUND, "duration_s"].sum() / 3600.0,
                "takeoff_mass_kg": float(ts["mass_kg"].iloc[0]),
                "fuel_kg": float(pt["fuel_kg"].sum()),
                **{f"{sp}_kg": float(pt[f"{sp}_kg"].sum()) for sp in SPECIES},
                "great_circle_km": self.route.great_circle_km,
                "flown_km": self.route.flown_km,
                "route_extra_km": self.route.extra_km,
                "route_extra_pct": self.route.extra_pct,
                "route_extra_fuel_kg": self.route.extra_fuel_kg,
                "route_note": self.route.note,
                "descent_level_offs": self.descent.n_level_offs,
                "descent_level_time_s": self.descent.level_time_s,
                "descent_actual_fuel_kg": self.descent.actual_fuel_kg,
                "descent_ideal_fuel_kg": self.descent.ideal_fuel_kg,
                "descent_saving_kg": self.descent.saving_kg,
                "descent_saving_pct": self.descent.saving_pct,
                "descent_note": self.descent.note,
            }
        )
        return row


def _meta(df: pd.DataFrame) -> dict[str, object]:
    first = df.iloc[0]
    return {
        "callsign": first["callsign"],
        "icao24": first["icao24"],
        "typecode": first["typecode"],
        "origin": first["origin"],
        "destination": first["destination"],
        "date_utc": pd.Timestamp(first["timestamp"]).strftime("%Y-%m-%d"),
    }


def phase_table(df: pd.DataFrame, fuel: pd.DataFrame, emissions: pd.DataFrame) -> pd.DataFrame:
    """Duration, distance, fuel and emissions per phase (rows = :data:`PHASES`)."""
    t_s = elapsed_seconds(df["timestamp"])
    phase = df["phase"].to_numpy()
    dt = np.diff(t_s)
    legs = haversine_km(
        df["lat"].to_numpy()[:-1],
        df["lon"].to_numpy()[:-1],
        df["lat"].to_numpy()[1:],
        df["lon"].to_numpy()[1:],
    )
    out = pd.DataFrame(index=pd.Index(list(PHASES), name="phase"))
    out["duration_s"] = [float(dt[phase[:-1] == p].sum()) for p in PHASES]
    out["distance_km"] = [float(legs[phase[:-1] == p].sum()) for p in PHASES]
    fuel_by = integrate_by_phase(t_s, fuel["fuel_flow_kgs"].to_numpy(), phase)
    out["fuel_kg"] = [fuel_by.get(p, 0.0) for p in PHASES]
    for sp in SPECIES:
        out[f"{sp}_kg"] = emissions[sp].reindex(list(PHASES)).fillna(0.0).to_numpy()
    return out


def process_flight(raw: pd.DataFrame, config: PipelineConfig | None = None) -> FlightResult:
    """Run every stage on one flight in the canonical schema.

    Args:
        raw: One flight as returned by :func:`flight_emissions.ingest.load_flights`.
        config: Pipeline configuration.

    Returns:
        The flight's results (``status`` explains a skipped flight).
    """
    config = config or PipelineConfig()
    flight_id = str(raw["flight_id"].iloc[0])
    df, report = clean_flight(raw, config)
    result = FlightResult(flight_id=flight_id, meta=_meta(raw), cleaning=report)

    if len(df) < 10:
        result.status = "skipped: fewer than 10 usable points after cleaning"
        logger.warning("%s: %s", flight_id, result.status)
        return result
    try:
        model = resolve_aircraft(raw["typecode"].iloc[0], config.fallback_typecode)
    except UnsupportedAircraftError as exc:
        result.status = f"skipped: {exc}"
        logger.warning("%s: %s", flight_id, result.status)
        return result
    if model.is_fallback:
        result.warnings.append(
            f"type code {raw['typecode'].iloc[0]!r} missing/unsupported: "
            f"modelled as fallback {model.typecode}"
        )
    if report.long_gap_total_s > 0:
        result.warnings.append(
            f"{report.n_long_gaps} data gap(s) longer than {config.max_gap_s:.0f} s "
            f"({report.long_gap_total_s / 60:.0f} min) bridged by linear fuel-flow integration"
        )

    df["phase"] = label_phases(df, config)
    result.warnings.extend(check_type_plausibility(df, model))
    fuel = estimate_fuel_flow(df, model, config)
    rates = emission_rates(
        fuel["fuel_flow_kgs"].to_numpy(),
        fuel["tas_kt"].to_numpy(),
        df["altitude_ft"].to_numpy(),
        model,
    )
    rates.index = df.index
    emissions = emissions_by_phase(df, rates)

    result.model = model
    result.timeseries = pd.concat([df, fuel, rates], axis=1)
    result.phase_table = phase_table(df, fuel, emissions)
    result.route = route_inefficiency(df, fuel["fuel_flow_kgs"].to_numpy(), config)
    result.descent = descent_analysis(df, fuel, model, config)
    for warning in result.warnings:
        logger.warning("%s: %s", flight_id, warning)
    return result


# ---------------------------------------------------------------------------------------
# Tables
# ---------------------------------------------------------------------------------------
def summary_table(results: list[FlightResult]) -> pd.DataFrame:
    """One row per flight."""
    return pd.DataFrame([r.summary_row() for r in results])


def phase_breakdown_table(results: list[FlightResult]) -> pd.DataFrame:
    """One row per flight and phase."""
    frames = []
    for r in results:
        if r.ok and r.phase_table is not None:
            frames.append(r.phase_table.reset_index().assign(flight_id=r.flight_id))
    if not frames:
        return pd.DataFrame()
    out = pd.concat(frames, ignore_index=True)
    return out[["flight_id", *[c for c in out.columns if c != "flight_id"]]]


def cleaning_table(results: list[FlightResult]) -> pd.DataFrame:
    """One row per flight with the cleaning counts."""
    return pd.DataFrame([r.cleaning.to_dict() for r in results])


def level_off_table(results: list[FlightResult]) -> pd.DataFrame:
    """One row per detected level-off in a descent."""
    rows = []
    for r in results:
        if not (r.ok and r.descent and r.descent.applicable and r.timeseries is not None):
            continue
        ts = r.timeseries["timestamp"]
        for lo in r.descent.level_offs:
            rows.append(
                {
                    "flight_id": r.flight_id,
                    "start": ts.iloc[lo.start_idx],
                    "end": ts.iloc[lo.end_idx],
                    "duration_s": lo.duration_s,
                    "altitude_ft": lo.altitude_ft,
                }
            )
    return pd.DataFrame(rows, columns=["flight_id", "start", "end", "duration_s", "altitude_ft"])


# ---------------------------------------------------------------------------------------
# Markdown summary (the README quotes these tables verbatim)
# ---------------------------------------------------------------------------------------
def _fmt(value: object, digits: int = 0, suffix: str = "") -> str:
    if value is None or (isinstance(value, float) and not np.isfinite(value)):
        return "n/a"
    if isinstance(value, int | float | np.floating | np.integer):
        return f"{value:,.{digits}f}{suffix}"
    return str(value)


def results_markdown(summary: pd.DataFrame, icao: pd.DataFrame, fdr: pd.DataFrame) -> str:
    """Render the README results tables from the run's outputs."""
    lines = [
        "| Flight | Route | Type | Block time | Fuel (kg) | CO2 (t) | NOx (kg) "
        "| Route extra | Extra fuel (kg) | Descent level-offs | CDO saving (kg) |",
        "|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for _, r in summary.iterrows():
        if r["status"] != "ok":
            lines.append(
                f"| {r['flight_id']} | {r['origin']}-{r['destination']} | "
                f"{r['typecode']} | {r['status']} |||||||"
            )
            continue
        h, m = divmod(int(round(r["block_time_h"] * 60)), 60)
        saving = (
            f"{_fmt(r['descent_saving_kg'])} ({_fmt(r['descent_saving_pct'], 1, ' %')})"
            if np.isfinite(r["descent_saving_kg"])
            else "n/a"
        )
        route_extra = (
            _fmt(r["route_extra_pct"], 1, " %")
            if np.isfinite(r["route_extra_pct"])
            else "n/a (circular)"
        )
        lines.append(
            f"| {r['callsign']} | {r['origin']}-{r['destination']} | {r['typecode']} "
            f"| {h}h{m:02d} | {_fmt(r['fuel_kg'])} | {_fmt(r['co2_kg'] / 1000, 1)} "
            f"| {_fmt(r['nox_kg'])} | {route_extra} | {_fmt(r['route_extra_fuel_kg'])} "
            f"| {int(r['descent_level_offs'])} | {saving} |"
        )

    lines += [
        "",
        "| Route | Type | Our CO2 per flight (kg) | ICAO per flight (kg) | Diff "
        "| Our CO2 per pax (kg) | ICAO per pax (kg) | Diff | Status |",
        "|---|---|---:|---:|---:|---:|---:|---:|---|",
    ]
    for _, r in icao.iterrows():
        lines.append(
            f"| {r['origin']}-{r['destination']} | {r['typecode']} "
            f"| {_fmt(r.get('our_co2_kg_per_flight'))} | {_fmt(r['icao_co2_kg_per_flight'])} "
            f"| {_fmt(r.get('diff_pct_per_flight'), 1, ' %')} "
            f"| {_fmt(r.get('our_co2_kg_per_pax'))} | {_fmt(r['icao_co2_kg_per_pax'])} "
            f"| {_fmt(r.get('diff_pct_per_pax'), 1, ' %')} | {r['status']} |"
        )

    if not fdr.empty:
        lines += [
            "",
            "| Fuel-model scenario (A320 recorder data) | Model fuel (kg) | Recorded fuel (kg) "
            "| Total error | Per-sample MAPE | Correlation |",
            "|---|---:|---:|---:|---:|---:|",
        ]
        for _, r in fdr.iterrows():
            lines.append(
                f"| {r['scenario']} | {_fmt(r['model_fuel_kg'])} | {_fmt(r['recorded_fuel_kg'])} "
                f"| {_fmt(r['total_error_pct'], 1, ' %')} | {_fmt(r['flow_mape_pct'], 1, ' %')} "
                f"| {_fmt(r['flow_correlation'], 2)} |"
            )
    return "\n".join(lines) + "\n"


# ---------------------------------------------------------------------------------------
# Run
# ---------------------------------------------------------------------------------------
@dataclass
class RunOutputs:
    """In-memory results of :func:`run_pipeline`."""

    results: list[FlightResult]
    summary: pd.DataFrame
    phases: pd.DataFrame
    icao: pd.DataFrame
    fdr: pd.DataFrame
    figures: list[Path] = field(default_factory=list)


def run_pipeline(
    input_path: str | Path,
    out_dir: str | Path,
    config: PipelineConfig | None = None,
    reference_path: str | Path | None = DEFAULT_REFERENCE,
    fdr_path: str | Path | None = DEFAULT_FDR,
    figures_dir: str | Path | None = None,
    make_figures: bool = True,
) -> RunOutputs:
    """Process every flight under ``input_path`` and write all outputs to ``out_dir``.

    Args:
        input_path: Parquet/CSV file or directory in the canonical schema.
        out_dir: Output directory (created if needed).
        config: Pipeline configuration.
        reference_path: ICAO reference table; skipped if None or missing.
        fdr_path: Flight-data-recorder sample for the fuel-model check; skipped if None
            or missing.
        figures_dir: Where to save PNG figures (default ``out_dir/figures``).
        make_figures: Set False to skip plotting (faster, e.g. in CI).

    Returns:
        The in-memory results.
    """
    config = config or PipelineConfig()
    out_dir = Path(out_dir)
    (out_dir / "timeseries").mkdir(parents=True, exist_ok=True)

    results = [process_flight(raw, config) for raw in load_flights(input_path)]
    summary = summary_table(results)
    phases = phase_breakdown_table(results)
    summary.to_csv(out_dir / "flights_summary.csv", index=False)
    phases.to_csv(out_dir / "phase_breakdown.csv", index=False)
    cleaning_table(results).to_csv(out_dir / "cleaning_report.csv", index=False)
    level_off_table(results).to_csv(out_dir / "level_offs.csv", index=False)
    for r in results:
        if r.ok and r.timeseries is not None:
            r.timeseries.to_parquet(out_dir / "timeseries" / f"{r.flight_id}.parquet", index=False)

    icao = pd.DataFrame()
    if reference_path is not None and Path(reference_path).exists():
        ok = summary[summary["status"] == "ok"] if "status" in summary else summary
        icao = compare_with_icao(ok, load_icao_reference(reference_path), config)
        icao.to_csv(out_dir / "icao_validation.csv", index=False)

    fdr = pd.DataFrame()
    fdr_ts = pd.DataFrame()
    if fdr_path is not None and Path(fdr_path).exists():
        fdr, fdr_ts = check_fuel_model_against_fdr(pd.read_parquet(fdr_path), "A320", config)
        fdr.to_csv(out_dir / "fdr_fuel_check.csv", index=False)
        fdr_ts.to_parquet(out_dir / "fdr_timeseries.parquet", index=False)

    (out_dir / "run_config.json").write_text(json.dumps(config.to_dict(), indent=2, default=str))
    (out_dir / "summary.md").write_text(results_markdown(summary, icao, fdr))

    outputs = RunOutputs(results, summary, phases, icao, fdr)
    if make_figures:
        from flight_emissions.plotting import save_all_figures

        outputs.figures = save_all_figures(
            outputs, fdr_ts, Path(figures_dir or out_dir / "figures")
        )
    logger.info("wrote results for %d flight(s) to %s", len(results), out_dir)
    return outputs
