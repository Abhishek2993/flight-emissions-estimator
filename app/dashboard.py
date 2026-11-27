"""Interactive dashboard: ``streamlit run app/dashboard.py``.

* **Flight** - pick a flight: trajectory map coloured by phase, altitude profile with phases
  shaded, fuel flow, fuel and emissions per phase, route and descent inefficiency, and the
  data-quality report.
* **Compare** - all flights side by side, grouped by flight, aircraft type or route.
* **Validation** - ICAO cross-check and the fuel-model check against recorded fuel flow.

The sidebar exposes the two headline assumptions (take-off mass fraction, load factor) so
their effect on the results can be seen directly. Everything runs offline.
"""

from __future__ import annotations

import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from plotly.subplots import make_subplots

from flight_emissions.config import PipelineConfig
from flight_emissions.ingest import load_flights
from flight_emissions.phases import PHASES, run_length_segments
from flight_emissions.pipeline import FlightResult, process_flight, summary_table
from flight_emissions.plotting import PHASE_COLORS, SERIES_1, SERIES_2, SERIES_3, world_lines
from flight_emissions.validate import (
    KNOWN_DIFFERENCE_REASONS,
    check_fuel_model_against_fdr,
    compare_with_icao,
    load_icao_reference,
)

warnings.filterwarnings("ignore", message="Aircraft: using synonym")
ROOT = Path(__file__).resolve().parents[1]
TEXT = "#191919"
GRID = "#e6e6e6"
LAYOUT = dict(
    template="plotly_white",
    font=dict(family="Inter, system-ui, sans-serif", size=13, color=TEXT),
    margin=dict(l=10, r=10, t=40, b=10),
    hoverlabel=dict(bgcolor="white"),
)

st.set_page_config(page_title="Flight emissions estimator", page_icon="✈", layout="wide")


# ---------------------------------------------------------------------------------------
# Data (cached per input directory and assumption values)
# ---------------------------------------------------------------------------------------
@st.cache_resource(show_spinner="Running the pipeline on the selected flights...")
def run(input_dir: str, mass_fraction: float, load_factor: float) -> list[FlightResult]:
    """Process every flight in ``input_dir`` with the chosen assumptions."""
    config = PipelineConfig(mass_fraction=mass_fraction, load_factor=load_factor)
    return [process_flight(raw, config) for raw in load_flights(input_dir)]


@st.cache_data(show_spinner=False)
def fdr_check(path: str, mass_fraction: float) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Fuel-model check against the A320 flight-data-recorder sample."""
    return check_fuel_model_against_fdr(
        pd.read_parquet(path), "A320", PipelineConfig(mass_fraction=mass_fraction)
    )


# ---------------------------------------------------------------------------------------
# Figures
# ---------------------------------------------------------------------------------------
def map_figure(r: FlightResult) -> go.Figure:
    """Trajectory coloured by phase over country boundaries (equirectangular)."""
    ts = r.timeseries
    lon, lat = ts["lon"].to_numpy(), ts["lat"].to_numpy()
    pad = max(np.ptp(lon), np.ptp(lat)) * 0.08 + 0.3
    fig = go.Figure()
    xs, ys = [], []
    for line in world_lines():
        xs += [*line[:, 0], None]
        ys += [*line[:, 1], None]
    fig.add_trace(
        go.Scattergl(
            x=xs,
            y=ys,
            mode="lines",
            line=dict(color="#c8c8c8", width=1),
            hoverinfo="skip",
            showlegend=False,
        )
    )
    phase = ts["phase"].to_numpy()
    shown: set[str] = set()
    for seg in run_length_segments(phase):
        sl = slice(seg.start, min(seg.end + 2, len(ts)))  # overlap one point: no gaps
        part = ts.iloc[sl]
        fig.add_trace(
            go.Scattergl(
                x=part["lon"],
                y=part["lat"],
                mode="lines",
                line=dict(color=PHASE_COLORS.get(seg.label, "#d9d9d9"), width=3),
                name=seg.label,
                legendgroup=seg.label,
                showlegend=seg.label not in shown,
                customdata=np.column_stack([part["altitude_ft"], part["groundspeed_kt"]]),
                hovertemplate=f"{seg.label}<br>%{{customdata[0]:,.0f}} ft · "
                "%{customdata[1]:.0f} kt<extra></extra>",
            )
        )
        shown.add(seg.label)
    centre = lon.mean()
    for x, y, label in (
        (lon[0], lat[0], r.meta["origin"]),
        (lon[-1], lat[-1], r.meta["destination"]),
    ):
        side = "right" if x <= centre else "left"  # keep labels inside the plot
        fig.add_trace(
            go.Scatter(
                x=[x],
                y=[y],
                mode="markers+text",
                text=[label],
                textposition=f"top {side}",
                showlegend=False,
                hoverinfo="skip",
                marker=dict(size=9, color=TEXT, line=dict(color="white", width=2)),
            )
        )
    mid = np.radians((lat.min() + lat.max()) / 2)
    fig.update_xaxes(
        range=[lon.min() - pad, lon.max() + pad],
        title="longitude (°)",
        showgrid=False,
        zeroline=False,
    )
    fig.update_yaxes(
        range=[lat.min() - pad, lat.max() + pad],
        title="latitude (°)",
        scaleanchor="x",
        scaleratio=1 / np.cos(mid),
        showgrid=False,
        zeroline=False,
    )
    fig.update_layout(
        **LAYOUT, height=470, title="Trajectory by phase", legend=dict(orientation="h", y=-0.15)
    )
    return fig


def profile_figure(r: FlightResult) -> go.Figure:
    """Altitude (phases shaded) and fuel flow, sharing the time axis."""
    ts = r.timeseries
    t = ts["timestamp"]
    fig = make_subplots(
        rows=2,
        cols=1,
        shared_xaxes=True,
        row_heights=[0.6, 0.4],
        vertical_spacing=0.08,
        subplot_titles=("Altitude profile", "Estimated fuel flow"),
    )
    segs = run_length_segments(ts["phase"].to_numpy())
    for seg in segs:
        end = t.iloc[min(seg.end + 1, len(t) - 1)]
        for row in (1, 2):
            fig.add_vrect(
                x0=t.iloc[seg.start],
                x1=end,
                row=row,
                col=1,
                line_width=0,
                fillcolor=PHASE_COLORS.get(seg.label, "#d9d9d9"),
                opacity=0.2,
            )
    fig.add_trace(
        go.Scatter(
            x=t,
            y=ts["altitude_ft"],
            line=dict(color=TEXT, width=2),
            name="altitude",
            customdata=ts["phase"],
            hovertemplate="%{y:,.0f} ft · %{customdata}<extra></extra>",
        ),
        row=1,
        col=1,
    )
    fig.add_trace(
        go.Scatter(
            x=t,
            y=ts["fuel_flow_kgs"] * 3600,
            line=dict(color=SERIES_1, width=2),
            name="fuel flow",
            hovertemplate="%{y:,.0f} kg/h<extra></extra>",
        ),
        row=2,
        col=1,
    )
    fig.update_yaxes(title="ft", tickformat=",", row=1, col=1, gridcolor=GRID)
    fig.update_yaxes(title="kg/h", tickformat=",", row=2, col=1, rangemode="tozero", gridcolor=GRID)
    fig.update_layout(**LAYOUT, height=470, showlegend=False, hovermode="x unified")
    return fig


def phase_bar(r: FlightResult) -> go.Figure:
    """Fuel per phase (one bar per phase, the phase colour carries identity)."""
    pt = r.phase_table.loc[[p for p in PHASES if r.phase_table.loc[p, "duration_s"] > 0]]
    fig = go.Figure(
        go.Bar(
            x=pt["fuel_kg"],
            y=pt.index,
            orientation="h",
            marker=dict(color=[PHASE_COLORS[p] for p in pt.index], cornerradius=4),
            text=[f"{v:,.0f} kg" for v in pt["fuel_kg"]],
            textposition="outside",
            hovertemplate="%{y}: %{x:,.0f} kg<extra></extra>",
        )
    )
    fig.update_layout(
        **LAYOUT,
        height=260,
        title="Fuel per phase",
        xaxis=dict(title="kg", tickformat=",", gridcolor=GRID),
        yaxis=dict(autorange="reversed"),
    )
    fig.update_xaxes(range=[0, pt["fuel_kg"].max() * 1.25])
    return fig


def descent_figure(r: FlightResult) -> go.Figure | None:
    """Actual descent vs idealised continuous descent."""
    d = r.descent
    if d is None or not d.applicable:
        return None
    p = d.profile
    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=p["distance_km"],
            y=p["altitude_ideal_ft"],
            name="ideal CDO (3°)",
            line=dict(color=SERIES_3, width=2),
        )
    )
    fig.add_trace(
        go.Scatter(
            x=p["distance_km"],
            y=p["altitude_actual_ft"],
            name="actual",
            line=dict(color=TEXT, width=2),
        )
    )
    lv = p[p["level_off"]]
    if not lv.empty:
        fig.add_trace(
            go.Scatter(
                x=lv["distance_km"],
                y=lv["altitude_actual_ft"],
                mode="markers",
                name="level-off",
                marker=dict(color=SERIES_2, size=9, line=dict(color="white", width=2)),
            )
        )
    fig.update_layout(
        **LAYOUT,
        height=330,
        title="Descent: actual vs idealised continuous descent",
        xaxis=dict(title="distance from top of descent (km)", gridcolor=GRID),
        yaxis=dict(title="ft", tickformat=",", gridcolor=GRID),
        legend=dict(orientation="h", y=1.12, x=1, xanchor="right"),
        hovermode="x unified",
    )
    return fig


def padded_max(values: pd.Series) -> float:
    """Axis maximum leaving room for value labels at the bar ends."""
    top = float(np.nanmax(values.to_numpy(dtype=float))) if values.notna().any() else 1.0
    return max(top, 1e-9) * 1.3


def fmt(v: float, digits: int = 0, suffix: str = "") -> str:
    """Format a number, or 'n/a' when not finite."""
    return f"{v:,.{digits}f}{suffix}" if v is not None and np.isfinite(v) else "n/a"


# ---------------------------------------------------------------------------------------
# Layout
# ---------------------------------------------------------------------------------------
st.sidebar.title("✈ Flight emissions")
input_dir = st.sidebar.text_input("Flight data directory", str(ROOT / "data" / "sample"))
st.sidebar.markdown("**Assumptions**")
mass_fraction = st.sidebar.slider(
    "Take-off mass (fraction of MTOW)",
    0.60,
    1.00,
    0.85,
    0.01,
    help="Real take-off mass is not in ADS-B data.",
)
load_factor = st.sidebar.slider(
    "Passenger load factor", 0.50, 1.00, 0.80, 0.01, help="Used only for per-passenger CO2."
)
st.sidebar.caption(
    "Fuel flow: OpenAP · TAS ≈ ground speed (no wind data) · emissions: ICAO databank + BFFM2."
)

results = run(input_dir, mass_fraction, load_factor)
ok = [r for r in results if r.ok]
summary = summary_table(results)
if not ok:
    st.error("No flight could be processed. Check the directory and the type codes.")
    st.stop()

tab_flight, tab_compare, tab_valid = st.tabs(["Flight", "Compare", "Validation"])

with tab_flight:
    labels = {
        f"{r.meta['callsign']}  {r.meta['origin']} → {r.meta['destination']}  "
        f"({r.meta['typecode']}, {r.meta['date_utc']})": r
        for r in ok
    }
    r = labels[st.selectbox("Flight", list(labels))]
    row = r.summary_row()
    c = st.columns(5)
    c[0].metric("Fuel burned", f"{fmt(row['fuel_kg'])} kg", help="gate-to-gate as observed")
    c[1].metric("CO₂", f"{fmt(row['co2_kg'] / 1000, 1)} t")
    c[2].metric("NOₓ", f"{fmt(row['nox_kg'])} kg")
    c[3].metric(
        "Extra distance vs great circle",
        fmt(row["route_extra_pct"], 1, " %") if r.route.applicable else "n/a (circular)",
        help=r.route.note,
    )
    c[4].metric(
        "Continuous-descent saving",
        f"{fmt(row['descent_saving_kg'])} kg" if r.descent.applicable else "n/a",
        help=f"{row['descent_level_offs']} level-off(s) in the descent",
    )
    for w in r.warnings:
        st.warning(w, icon="⚠️")

    left, right = st.columns([1, 1.25])
    left.plotly_chart(map_figure(r), width="stretch")
    right.plotly_chart(profile_figure(r), width="stretch")

    left, right = st.columns([1, 1.25])
    left.plotly_chart(phase_bar(r), width="stretch")
    table = r.phase_table.copy()
    table.insert(0, "duration (min)", table.pop("duration_s") / 60)
    table = table.rename(columns=lambda col: col.replace("_kg", " (kg)").replace("_km", " (km)"))
    right.markdown("**Fuel and emissions per phase**")
    right.dataframe(table.style.format("{:,.1f}"), width="stretch")

    left, right = st.columns([1.25, 1])
    fig = descent_figure(r)
    if fig is not None:
        left.plotly_chart(fig, width="stretch")
    else:
        left.info(f"Descent analysis not available: {r.descent.note}")
    right.markdown("**Inefficiency**")
    right.dataframe(
        pd.DataFrame(
            {
                "value": [
                    fmt(r.route.great_circle_km, 0, " km"),
                    fmt(r.route.flown_km, 0, " km"),
                    fmt(r.route.extra_pct, 1, " %"),
                    fmt(r.route.extra_fuel_kg, 0, " kg"),
                    str(r.descent.n_level_offs),
                    fmt(r.descent.level_time_s, 0, " s"),
                    fmt(r.descent.actual_fuel_kg, 0, " kg"),
                    fmt(r.descent.ideal_fuel_kg, 0, " kg"),
                    fmt(r.descent.saving_kg, 0, " kg"),
                ]
            },
            index=[
                "great-circle distance",
                "flown distance",
                "extra distance",
                "extra fuel (at cruise fuel/km)",
                "level-offs in descent",
                "time level in descent",
                "descent fuel (actual)",
                "descent fuel (ideal CDO)",
                "estimated CDO saving",
            ],
        ),
        width="stretch",
    )
    with st.expander("Data quality: what the cleaning stage did"):
        st.dataframe(pd.Series(r.cleaning.to_dict(), name="value").astype(str), width="stretch")

with tab_compare:
    s = summary[summary["status"] == "ok"].copy()
    s["route"] = s["origin"] + "-" + s["destination"]
    s["label"] = s["callsign"] + " " + s["route"]
    s["co2_g_per_seat_km"] = s["co2_kg"] * 1000 / (s["seats"] * s["great_circle_km"])
    s.loc[~np.isfinite(s["route_extra_pct"]), "co2_g_per_seat_km"] = np.nan  # circular
    group = st.radio("Group by", ["flight", "aircraft type", "route"], horizontal=True)
    key = {"flight": "label", "aircraft type": "typecode", "route": "route"}[group]
    phases = (
        pd.concat([r.phase_table["fuel_kg"].rename(r.flight_id) for r in ok], axis=1)
        .T.join(s.set_index("flight_id")[[key]])
        .groupby(key)
        .sum()
    )

    left, right = st.columns(2)
    fig = go.Figure()
    for p in PHASES:
        fig.add_trace(
            go.Bar(
                y=phases.index,
                x=phases[p],
                name=p,
                orientation="h",
                marker=dict(color=PHASE_COLORS[p], line=dict(color="white", width=2)),
                hovertemplate=f"{p}: %{{x:,.0f}} kg<extra></extra>",
            )
        )
    fig.update_layout(
        **LAYOUT,
        barmode="stack",
        height=380,
        title="Fuel by phase (kg)",
        xaxis=dict(tickformat=",", gridcolor=GRID),
        legend=dict(orientation="h", y=1.1, traceorder="normal"),
        bargap=0.45,
    )
    left.plotly_chart(fig, width="stretch")

    agg = s.groupby(key).agg(
        co2_g_per_seat_km=("co2_g_per_seat_km", "mean"),
        route_extra_pct=("route_extra_pct", "mean"),
        descent_saving_kg=("descent_saving_kg", "sum"),
    )
    fig = go.Figure(
        go.Bar(
            y=agg.index,
            x=agg["co2_g_per_seat_km"],
            orientation="h",
            marker=dict(color=SERIES_1, cornerradius=4),
            text=[fmt(v, 0, " g") for v in agg["co2_g_per_seat_km"]],
            textposition="outside",
        )
    )
    fig.update_layout(
        **LAYOUT,
        height=380,
        bargap=0.45,
        xaxis=dict(gridcolor=GRID, range=[0, padded_max(agg["co2_g_per_seat_km"])]),
        title="CO₂ per seat and great-circle km (g) — circular flights excluded",
    )
    right.plotly_chart(fig, width="stretch")

    left, right = st.columns(2)
    for col, metric, title, color, unit in (
        (left, "route_extra_pct", "Extra distance vs great circle (%)", SERIES_1, " %"),
        (right, "descent_saving_kg", "Estimated continuous-descent saving (kg)", SERIES_3, " kg"),
    ):
        fig = go.Figure(
            go.Bar(
                y=agg.index,
                x=agg[metric],
                orientation="h",
                marker=dict(color=color, cornerradius=4),
                text=[fmt(v, 1, unit) for v in agg[metric]],
                textposition="outside",
            )
        )
        fig.update_layout(
            **LAYOUT,
            height=320,
            title=title,
            bargap=0.45,
            xaxis=dict(gridcolor=GRID, range=[0, padded_max(agg[metric])]),
        )
        col.plotly_chart(fig, width="stretch")

    st.markdown("**All flights**")
    cols = [
        "flight_id",
        "typecode",
        "route",
        "block_time_h",
        "fuel_kg",
        "co2_kg",
        "h2o_kg",
        "nox_kg",
        "co_kg",
        "hc_kg",
        "flown_km",
        "great_circle_km",
        "route_extra_pct",
        "descent_level_offs",
        "descent_saving_kg",
        "long_gap_min",
        "status",
    ]
    st.dataframe(s[cols].round(1), width="stretch", hide_index=True)
    skipped = summary[summary["status"] != "ok"]
    if not skipped.empty:
        st.warning(
            "Skipped flights:\n\n"
            + "\n".join(
                f"- {f}: {st_}"
                for f, st_ in zip(skipped["flight_id"], skipped["status"], strict=True)
            )
        )

with tab_valid:
    ref_path = ROOT / "data" / "reference" / "icao_reference.csv"
    st.subheader("ICAO Carbon Emissions Calculator cross-check")
    if ref_path.exists():
        icao = compare_with_icao(
            summary[summary["status"] == "ok"],
            load_icao_reference(ref_path),
            PipelineConfig(load_factor=load_factor),
        )
        st.dataframe(icao.round(1), width="stretch", hide_index=True)
        if (icao["status"] == "awaiting ICAO value").any():
            st.info(
                "Some reference values are still empty: enter them in "
                "`data/reference/icao_reference.csv` from the ICAO calculator."
            )
    st.markdown("**Why the numbers differ**")
    st.markdown("\n".join(f"- {reason}" for reason in KNOWN_DIFFERENCE_REASONS))

    fdr_path = ROOT / "data" / "reference" / "a320_fdr_fuelflow.parquet"
    if fdr_path.exists():
        st.subheader("Fuel model vs measured fuel flow (A320 flight-data recorder)")
        fdr, fdr_ts = fdr_check(str(fdr_path), mass_fraction)
        st.dataframe(fdr.round(2), width="stretch", hide_index=True)
        fig = go.Figure()
        fig.add_trace(
            go.Scatter(
                x=fdr_ts["timestamp"],
                y=fdr_ts["recorded_kgs"] * 3600,
                name="recorded",
                line=dict(color=TEXT, width=2),
            )
        )
        for name, color in zip(
            (fdr["scenario"].iloc[0], fdr["scenario"].iloc[-1]), (SERIES_1, SERIES_2), strict=True
        ):
            fig.add_trace(
                go.Scatter(
                    x=fdr_ts["timestamp"],
                    y=fdr_ts[name] * 3600,
                    name=name,
                    line=dict(color=color, width=2),
                )
            )
        fig.update_layout(
            **LAYOUT,
            height=380,
            hovermode="x unified",
            yaxis=dict(title="kg/h", tickformat=",", gridcolor=GRID),
            legend=dict(orientation="h", y=1.12),
        )
        st.plotly_chart(fig, width="stretch")
