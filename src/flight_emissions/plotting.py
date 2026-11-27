"""Static figures (matplotlib) for the CLI run and the README.

Colours follow one fixed mapping: every phase keeps the same colour in every figure and in
the dashboard (:data:`PHASE_COLORS`). The four flight phases use the first four slots of a
colour-blind-validated categorical palette, in fixed order; ground is a neutral grey
because it is "not flying" rather than a category to compare. Map backgrounds use the
Natural Earth 1:110m country boundaries shipped with the package, so no figure needs
network access.
"""

from __future__ import annotations

import json
from functools import lru_cache
from importlib import resources
from pathlib import Path
from typing import TYPE_CHECKING

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from matplotlib.collections import LineCollection  # noqa: E402
from matplotlib.patches import Patch  # noqa: E402

from flight_emissions.clean import elapsed_seconds  # noqa: E402
from flight_emissions.phases import PHASES, run_length_segments  # noqa: E402

if TYPE_CHECKING:  # pragma: no cover
    from flight_emissions.pipeline import FlightResult, RunOutputs

PHASE_COLORS: dict[str, str] = {
    "ground": "#9c9c9c",
    "climb": "#257de0",
    "cruise": "#f47230",
    "level": "#00b894",
    "descent": "#f2a600",
    "unknown": "#d9d9d9",
}
TEXT = "#191919"
TEXT_2 = "#666666"
GRID = "#e6e6e6"
LAND_LINE = "#c8c8c8"
SERIES_1 = "#257de0"
SERIES_2 = "#f47230"
SERIES_3 = "#00b894"

STYLE = {
    "figure.facecolor": "white",
    "axes.facecolor": "white",
    "axes.edgecolor": GRID,
    "axes.labelcolor": TEXT_2,
    "axes.titlecolor": TEXT,
    "axes.titlesize": 12,
    "axes.titleweight": "bold",
    "axes.titlelocation": "left",
    "axes.labelsize": 10,
    "axes.grid": True,
    "axes.axisbelow": True,
    "grid.color": GRID,
    "grid.linewidth": 0.8,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "xtick.color": TEXT_2,
    "ytick.color": TEXT_2,
    "xtick.labelsize": 9,
    "ytick.labelsize": 9,
    "legend.frameon": False,
    "legend.fontsize": 9,
    "font.size": 10,
    "lines.linewidth": 2.0,
    "lines.solid_capstyle": "round",
    "savefig.dpi": 160,
    "savefig.bbox": "tight",
}


@lru_cache(maxsize=1)
def world_lines() -> list[np.ndarray]:
    """Country boundary polylines (lon, lat) from the bundled Natural Earth file."""
    text = (
        resources.files("flight_emissions")
        .joinpath("resources/world_boundaries_110m.geojson")
        .read_text()
    )
    feature = json.loads(text)
    return [np.asarray(line, dtype=float) for line in feature["geometry"]["coordinates"]]


def phase_legend_handles(phases: list[str] | None = None) -> list[Patch]:
    """Legend patches for the given phases (default: all)."""
    return [Patch(color=PHASE_COLORS[p], label=p) for p in (phases or list(PHASES))]


def _hours(ts: pd.Series) -> np.ndarray:
    return elapsed_seconds(ts) / 3600.0


# ---------------------------------------------------------------------------------------
# Per-flight figures
# ---------------------------------------------------------------------------------------
def draw_map(ax: plt.Axes, ts: pd.DataFrame, origin: str = "", destination: str = "") -> None:
    """Trajectory coloured by phase over country boundaries (equirectangular)."""
    lon = ts["lon"].to_numpy()
    lat = ts["lat"].to_numpy()
    pad_lon = max(np.ptp(lon) * 0.08, 0.3)
    pad_lat = max(np.ptp(lat) * 0.08, 0.3)
    xlim = (lon.min() - pad_lon, lon.max() + pad_lon)
    ylim = (lat.min() - pad_lat, lat.max() + pad_lat)
    for line in world_lines():
        ax.plot(line[:, 0], line[:, 1], color=LAND_LINE, lw=0.6, zorder=1)
    points = np.column_stack([lon, lat])
    segments = np.stack([points[:-1], points[1:]], axis=1)
    colors = [PHASE_COLORS.get(p, PHASE_COLORS["unknown"]) for p in ts["phase"].to_numpy()[:-1]]
    ax.add_collection(LineCollection(segments, colors=colors, linewidths=2.0, zorder=3))
    for (x, y), label in (((lon[0], lat[0]), origin), ((lon[-1], lat[-1]), destination)):
        ax.plot(x, y, "o", ms=6, color=TEXT, mec="white", mew=2, zorder=4)
        if label:
            ax.annotate(
                label,
                (x, y),
                xytext=(5, 5),
                textcoords="offset points",
                color=TEXT,
                fontsize=9,
                zorder=5,
            )
    ax.set_xlim(*xlim)
    ax.set_ylim(*ylim)
    ax.set_aspect(1.0 / np.cos(np.radians(np.mean(ylim))))
    ax.set_xlabel("longitude (°)")
    ax.set_ylabel("latitude (°)")
    ax.grid(False)


def shade_phases(ax: plt.Axes, hours: np.ndarray, phase: np.ndarray) -> None:
    """Shade the background of a time axis by flight phase."""
    for seg in run_length_segments(phase):
        end = hours[seg.end + 1] if seg.end + 1 < hours.size else hours[seg.end]
        ax.axvspan(
            hours[seg.start], end, color=PHASE_COLORS.get(seg.label, "#d9d9d9"), alpha=0.18, lw=0
        )


def plot_flight_overview(result: FlightResult) -> plt.Figure:
    """Map, altitude profile with phases, and fuel flow for one flight."""
    assert result.timeseries is not None
    ts = result.timeseries
    hours = _hours(ts["timestamp"])
    phase = ts["phase"].to_numpy()
    with plt.rc_context(STYLE):
        fig = plt.figure(figsize=(13, 5.6))
        grid = fig.add_gridspec(
            2, 2, width_ratios=[1.0, 1.25], height_ratios=[1.4, 1.0], wspace=0.18, hspace=0.35
        )
        ax_map = fig.add_subplot(grid[:, 0])
        draw_map(
            ax_map, ts, str(result.meta["origin"] or ""), str(result.meta["destination"] or "")
        )
        ax_map.set_title("Trajectory by phase")

        ax_alt = fig.add_subplot(grid[0, 1])
        shade_phases(ax_alt, hours, phase)
        ax_alt.plot(hours, ts["altitude_ft"], color=TEXT, lw=1.6)
        ax_alt.set_ylabel("altitude (ft)")
        ax_alt.set_title("Altitude profile")
        ax_alt.yaxis.set_major_formatter(matplotlib.ticker.StrMethodFormatter("{x:,.0f}"))
        present = [p for p in PHASES if p in set(phase)]
        ax_alt.legend(
            handles=phase_legend_handles(present),
            ncol=len(present),
            loc="upper left",
            bbox_to_anchor=(0, 1.0),
        )
        ax_alt.set_ylim(
            bottom=min(0, float(np.nanmin(ts["altitude_ft"]))),
            top=float(np.nanmax(ts["altitude_ft"])) * 1.22,
        )

        ax_ff = fig.add_subplot(grid[1, 1], sharex=ax_alt)
        shade_phases(ax_ff, hours, phase)
        ax_ff.plot(hours, ts["fuel_flow_kgs"] * 3600, color=SERIES_1, lw=1.4)
        ax_ff.set_ylabel("fuel flow (kg/h)")
        ax_ff.set_xlabel("time since first sample (h)")
        ax_ff.set_title("Estimated fuel flow (OpenAP)")
        ax_ff.yaxis.set_major_formatter(matplotlib.ticker.StrMethodFormatter("{x:,.0f}"))
        ax_ff.set_ylim(bottom=0)

        m = result.meta
        total = result.phase_table["fuel_kg"].sum() if result.phase_table is not None else np.nan
        fig.suptitle(
            f"{m['callsign']}  {m['origin']} → {m['destination']}  "
            f"({m['typecode']}, {m['date_utc']})   ·   fuel {total:,.0f} kg"
            f"   ·   CO2 {3.16 * total / 1000:,.1f} t",
            x=0.01,
            ha="left",
            fontsize=13,
            color=TEXT,
            fontweight="bold",
        )
    return fig


def plot_descent(result: FlightResult) -> plt.Figure | None:
    """Actual descent profile vs the idealised continuous descent."""
    d = result.descent
    if d is None or not d.applicable or d.profile is None:
        return None
    p = d.profile
    with plt.rc_context(STYLE):
        fig, (ax, ax_ff) = plt.subplots(
            2, 1, figsize=(9, 6), sharex=True, gridspec_kw={"height_ratios": [1.6, 1]}
        )
        ax.plot(
            p["distance_km"],
            p["altitude_ideal_ft"],
            color=SERIES_3,
            lw=2,
            label="idealised continuous descent (3°)",
        )
        ax.plot(p["distance_km"], p["altitude_actual_ft"], color=TEXT, lw=2, label="actual")
        level = p["level_off"].to_numpy()
        if level.any():
            ax.scatter(
                p.loc[level, "distance_km"],
                p.loc[level, "altitude_actual_ft"],
                s=22,
                color=SERIES_2,
                edgecolor="white",
                linewidth=1,
                zorder=4,
                label="level-off",
            )
        ax.set_ylabel("altitude (ft)")
        ax.yaxis.set_major_formatter(matplotlib.ticker.StrMethodFormatter("{x:,.0f}"))
        ax.legend(loc="upper right")
        ax.set_title(
            f"{result.meta['callsign']} descent: {d.n_level_offs} level-off(s), "
            f"estimated saving {d.saving_kg:,.0f} kg ({d.saving_pct:.1f} % of descent fuel)"
        )
        ax_ff.plot(
            p["distance_km"],
            p["fuel_flow_ideal_kgs"] * 3600,
            color=SERIES_3,
            lw=1.6,
            label=f"ideal: {d.ideal_fuel_kg:,.0f} kg",
        )
        ax_ff.plot(
            p["distance_km"],
            p["fuel_flow_actual_kgs"] * 3600,
            color=TEXT,
            lw=1.6,
            label=f"actual: {d.actual_fuel_kg:,.0f} kg",
        )
        ax_ff.set_ylabel("fuel flow (kg/h)")
        ax_ff.set_xlabel("distance from top of descent (km)")
        ax_ff.set_ylim(bottom=0)
        ax_ff.legend(loc="upper right")
    return fig


# ---------------------------------------------------------------------------------------
# Cross-flight figures
# ---------------------------------------------------------------------------------------
def _flight_labels(summary: pd.DataFrame) -> list[str]:
    return [f"{r.callsign}  {r.origin}-{r.destination}  {r.typecode}" for r in summary.itertuples()]


def plot_fuel_by_phase(phases: pd.DataFrame, summary: pd.DataFrame) -> plt.Figure:
    """Share of fuel per phase for every flight (100 % bars) with the total at the bar end."""
    ok = summary[summary["status"] == "ok"].reset_index(drop=True)
    pivot = (
        phases.pivot_table(index="flight_id", columns="phase", values="fuel_kg")
        .reindex(index=ok["flight_id"], columns=list(PHASES))
        .fillna(0.0)
    )
    share = pivot.div(pivot.sum(axis=1), axis=0) * 100
    with plt.rc_context(STYLE):
        fig, ax = plt.subplots(figsize=(11, 0.6 * len(ok) + 1.8), layout="constrained")
        y = np.arange(len(ok))[::-1]
        left = np.zeros(len(ok))
        for p in PHASES:
            ax.barh(
                y,
                share[p],
                left=left,
                height=0.55,
                color=PHASE_COLORS[p],
                label=p,
                edgecolor="white",
                linewidth=2,
            )
            left += share[p].to_numpy()
        for yi, total in zip(y, pivot.sum(axis=1), strict=True):
            ax.text(101, yi, f"{total:,.0f} kg", va="center", color=TEXT, fontsize=9)
        ax.set_yticks(y, _flight_labels(ok))
        ax.set_xlim(0, 112)
        ax.set_xticks([0, 25, 50, 75, 100], ["0 %", "25 %", "50 %", "75 %", "100 %"])
        ax.grid(axis="y", visible=False)
        ax.legend(ncol=len(PHASES), loc="lower left", bbox_to_anchor=(0, 1.0))
        ax.set_title("Share of fuel burned per flight phase (total at bar end)", pad=28)
    return fig


def plot_inefficiency(summary: pd.DataFrame) -> plt.Figure:
    """Route extra distance (%) and descent saving (kg), as two separate panels."""
    ok = summary[summary["status"] == "ok"].reset_index(drop=True)
    labels = _flight_labels(ok)
    y = np.arange(len(ok))[::-1]
    with plt.rc_context(STYLE):
        fig, (a1, a2) = plt.subplots(
            1, 2, figsize=(14, 0.6 * len(ok) + 1.8), sharey=True, layout="constrained"
        )
        route = ok["route_extra_pct"].to_numpy(dtype=float)
        a1.barh(y, np.nan_to_num(route), height=0.5, color=SERIES_1)
        for yi, v, extra in zip(y, route, ok["route_extra_fuel_kg"], strict=True):
            text = f"{v:.1f} %  (+{extra:,.0f} kg fuel)" if np.isfinite(v) else "n/a (circular)"
            a1.text(
                (v if np.isfinite(v) else 0) + 0.8, yi, text, va="center", fontsize=9, color=TEXT
            )
        a1.set_yticks(y, labels)
        a1.set_xlim(0, np.nanmax(np.append(route, 1)) * 1.6)
        a1.set_xlabel("extra distance vs great circle (%)")
        a1.set_title("Route inefficiency")
        a1.grid(axis="y", visible=False)

        saving = ok["descent_saving_kg"].to_numpy(dtype=float)
        a2.barh(y, np.nan_to_num(saving), height=0.5, color=SERIES_3)
        for yi, v, pct, n in zip(
            y, saving, ok["descent_saving_pct"], ok["descent_level_offs"], strict=True
        ):
            text = f"{v:,.0f} kg ({pct:.1f} %), {int(n)} level-off(s)" if np.isfinite(v) else "n/a"
            a2.text(max(np.nan_to_num(v), 0) + 3, yi, text, va="center", fontsize=9, color=TEXT)
        a2.set_xlim(0, np.nanmax(np.append(saving, 1)) * 1.9)
        a2.set_xlabel("estimated fuel saving of a continuous descent (kg)")
        a2.set_title("Descent inefficiency")
        a2.grid(axis="y", visible=False)
    return fig


def plot_fdr_check(fdr_ts: pd.DataFrame, fdr_summary: pd.DataFrame) -> plt.Figure:
    """Recorded vs modelled fuel flow on the A320 flight-data-recorder sample."""
    hours = _hours(fdr_ts["timestamp"])
    scenarios = list(fdr_summary["scenario"])
    with plt.rc_context(STYLE):
        fig, (ax, ax_alt) = plt.subplots(
            2, 1, figsize=(11, 6), sharex=True, gridspec_kw={"height_ratios": [1.6, 1]}
        )
        ax.plot(hours, fdr_ts["recorded_kgs"] * 3600, color=TEXT, lw=1.6, label="recorded (FDR)")
        for name, color in zip((scenarios[0], scenarios[-1]), (SERIES_1, SERIES_2), strict=True):
            err = fdr_summary.set_index("scenario").loc[name, "total_error_pct"]
            ax.plot(
                hours,
                fdr_ts[name] * 3600,
                color=color,
                lw=1.4,
                label=f"OpenAP, {name}: total {err:+.1f} %",
            )
        ax.set_ylabel("fuel flow (kg/h)")
        ax.yaxis.set_major_formatter(matplotlib.ticker.StrMethodFormatter("{x:,.0f}"))
        ax.legend(loc="upper right")
        ax.set_title("Fuel-model check against measured fuel flow (A320 flight-data recorder)")
        ax_alt.plot(hours, fdr_ts["altitude_ft"], color=TEXT_2, lw=1.4)
        ax_alt.set_ylabel("altitude (ft)")
        ax_alt.set_xlabel("time (h)")
        ax_alt.yaxis.set_major_formatter(matplotlib.ticker.StrMethodFormatter("{x:,.0f}"))
    return fig


def plot_icao(icao: pd.DataFrame) -> plt.Figure | None:
    """Our per-passenger CO2 vs the ICAO calculator, for routes with a reference value."""
    if icao.empty:
        return None
    done = icao[icao["status"] == "compared"]
    done = done[np.isfinite(done["icao_co2_kg_per_pax"].astype(float))]
    if done.empty:
        return None
    labels = [f"{r.origin}-{r.destination} {r.typecode}" for r in done.itertuples()]
    x = np.arange(len(done))
    with plt.rc_context(STYLE):
        fig, ax = plt.subplots(figsize=(1.8 * len(done) + 3, 4.5))
        ax.bar(
            x - 0.15, done["our_co2_kg_per_pax"], width=0.28, color=SERIES_1, label="this pipeline"
        )
        ax.bar(
            x + 0.15,
            done["icao_co2_kg_per_pax"],
            width=0.28,
            color=SERIES_2,
            label="ICAO calculator",
        )
        for xi, diff in zip(x, done["diff_pct_per_pax"], strict=True):
            top = max(done["our_co2_kg_per_pax"].max(), done["icao_co2_kg_per_pax"].max())
            ax.text(xi, top * 1.04, f"{diff:+.0f} %", ha="center", color=TEXT, fontsize=9)
        ax.set_xticks(x, labels)
        ax.set_ylabel("CO2 per passenger (kg)")
        ax.legend(loc="upper left")
        ax.grid(axis="x", visible=False)
        ax.set_title("Per-passenger CO2: this pipeline vs ICAO Carbon Emissions Calculator")
    return fig


# ---------------------------------------------------------------------------------------
# Batch save
# ---------------------------------------------------------------------------------------
def _save(fig: plt.Figure | None, path: Path, written: list[Path]) -> None:
    if fig is None:
        return
    fig.savefig(path)
    plt.close(fig)
    written.append(path)


def save_all_figures(outputs: RunOutputs, fdr_ts: pd.DataFrame, figures_dir: Path) -> list[Path]:
    """Write every figure of a run to ``figures_dir``; return the written paths."""
    figures_dir.mkdir(parents=True, exist_ok=True)
    written: list[Path] = []
    for r in outputs.results:
        if r.ok:
            _save(plot_flight_overview(r), figures_dir / f"{r.flight_id}_overview.png", written)
            _save(plot_descent(r), figures_dir / f"{r.flight_id}_descent.png", written)
    if not outputs.phases.empty:
        _save(
            plot_fuel_by_phase(outputs.phases, outputs.summary),
            figures_dir / "fuel_by_phase.png",
            written,
        )
        _save(plot_inefficiency(outputs.summary), figures_dir / "inefficiency.png", written)
    if not outputs.fdr.empty:
        _save(plot_fdr_check(fdr_ts, outputs.fdr), figures_dir / "fdr_fuel_check.png", written)
    _save(plot_icao(outputs.icao), figures_dir / "icao_validation.png", written)
    return written
