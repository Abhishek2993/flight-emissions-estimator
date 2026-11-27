"""Clean a trajectory and resample it onto a uniform time grid.

Steps, in order (each one is counted in a :class:`CleaningReport`):

1. **Drop unusable rows** - rows without a timestamp or a position.
2. **Merge duplicates** - identical rows are dropped; rows sharing a timestamp are merged,
   keeping the first non-missing value of every column (different ADS-B message types
   carry different fields for the same instant).
3. **Value plausibility** - ground speeds, vertical rates and altitudes outside physical
   limits are set to missing (the row is kept: its position may still be good).
4. **Spike removal** - a position that implies an impossible speed *both* from the previous
   and to the next point is removed; altitudes are treated the same way with an implied
   vertical rate. Requiring both links to be bad removes isolated spikes without deleting
   the valid point that follows a spike.
5. **Resampling** - every numeric column is linearly interpolated onto a uniform grid
   (default 10 s). Gaps up to ``max_gap_s`` are bridged; grid points inside longer gaps are
   dropped rather than invented, and the gaps are reported.
6. **Derived fills** - missing ground speed is derived from consecutive positions, missing
   vertical rate from the altitude derivative, and missing altitude on the ground (surface
   position reports carry no altitude) from the nearest airborne report.
"""

from __future__ import annotations

import logging
from dataclasses import asdict, dataclass

import numpy as np
import pandas as pd
from numpy.typing import NDArray

from flight_emissions.config import PipelineConfig
from flight_emissions.geo import KT_TO_KMS, haversine_km
from flight_emissions.ingest import META_COLUMNS, NUMERIC_COLUMNS

logger = logging.getLogger(__name__)

STRING_COLUMNS = ["icao24", "callsign", "typecode", *META_COLUMNS]


@dataclass
class CleaningReport:
    """How many points each cleaning step removed or changed, and why."""

    flight_id: str
    n_input: int = 0
    n_missing_time_or_position: int = 0
    n_duplicates: int = 0
    n_groundspeed_rejected: int = 0
    n_vertical_rate_rejected: int = 0
    n_altitude_out_of_bounds: int = 0
    n_position_spikes: int = 0
    n_altitude_spikes: int = 0
    n_short_gaps: int = 0
    short_gap_total_s: float = 0.0
    n_long_gaps: int = 0
    long_gap_total_s: float = 0.0
    n_ground_altitude_filled: int = 0
    n_dropped_without_altitude: int = 0
    n_output: int = 0

    @property
    def n_points_removed(self) -> int:
        """Raw points removed before resampling (missing data, duplicates, position spikes)."""
        return self.n_missing_time_or_position + self.n_duplicates + self.n_position_spikes

    def to_dict(self) -> dict[str, float | int | str]:
        """Return the report as a flat dictionary, including derived totals."""
        out = asdict(self)
        out["n_points_removed"] = self.n_points_removed
        return out

    def summary(self) -> str:
        """One-line, human-readable summary used in the log."""
        return (
            f"{self.flight_id}: {self.n_input} raw -> {self.n_output} resampled points | "
            f"removed {self.n_missing_time_or_position} without time/position, "
            f"{self.n_duplicates} duplicates, {self.n_position_spikes} position spikes | "
            f"nulled {self.n_altitude_spikes} altitude spikes, "
            f"{self.n_altitude_out_of_bounds} out-of-range altitudes, "
            f"{self.n_groundspeed_rejected} ground speeds, "
            f"{self.n_vertical_rate_rejected} vertical rates | "
            f"{self.n_short_gaps} short gaps interpolated ({self.short_gap_total_s:.0f} s), "
            f"{self.n_long_gaps} long gaps left open ({self.long_gap_total_s:.0f} s)"
        )


def elapsed_seconds(timestamps: pd.Series) -> NDArray[np.float64]:
    """Seconds elapsed since the first timestamp of the series."""
    ts = pd.to_datetime(timestamps, utc=True)
    return (ts - ts.iloc[0]).dt.total_seconds().to_numpy(dtype=float)


# ---------------------------------------------------------------------------------------
# Step 1-2: unusable rows and duplicates
# ---------------------------------------------------------------------------------------
def drop_unusable_rows(df: pd.DataFrame) -> tuple[pd.DataFrame, int]:
    """Drop rows without a timestamp, latitude or longitude."""
    mask = df["timestamp"].notna() & df["lat"].notna() & df["lon"].notna()
    return df.loc[mask].copy(), int((~mask).sum())


def merge_duplicates(df: pd.DataFrame) -> tuple[pd.DataFrame, int]:
    """Drop exact duplicates and merge rows that share a timestamp.

    Returns:
        The de-duplicated, time-sorted frame and the number of rows removed.
    """
    n_before = len(df)
    df = df.drop_duplicates()
    df = df.groupby("timestamp", as_index=False, sort=True).first()
    return df.reset_index(drop=True), n_before - len(df)


# ---------------------------------------------------------------------------------------
# Step 3-4: plausibility and spikes
# ---------------------------------------------------------------------------------------
def _spike_flags(link_bad: NDArray[np.bool_]) -> NDArray[np.bool_]:
    """Flag points whose links to *both* neighbours are bad (end points: their only link).

    ``link_bad[k]`` describes the link between point ``k`` and ``k + 1``.
    An end point is only flagged when its single link is bad but the next link is fine,
    which identifies the end point (not its neighbour) as the culprit.
    """
    m = link_bad.size + 1
    flags = np.zeros(m, dtype=bool)
    if m < 3:
        return flags
    flags[1:-1] = link_bad[:-1] & link_bad[1:]
    flags[0] = link_bad[0] and not link_bad[1]
    flags[-1] = link_bad[-1] and not link_bad[-2]
    return flags


def detect_position_spikes(
    t_s: NDArray[np.float64],
    lat: NDArray[np.float64],
    lon: NDArray[np.float64],
    max_speed_kt: float,
    max_iter: int = 10,
) -> NDArray[np.bool_]:
    """Return a mask of position spikes (True = spike).

    The implied speed of a link is ``haversine distance / Δt``. Detection is iterated
    because removing one spike changes the links of its neighbours.
    """
    keep = np.ones(t_s.size, dtype=bool)
    for _ in range(max_iter):
        idx = np.flatnonzero(keep)
        if idx.size < 3:
            break
        dt = np.maximum(np.diff(t_s[idx]), 1.0)
        dist_km = haversine_km(lat[idx][:-1], lon[idx][:-1], lat[idx][1:], lon[idx][1:])
        speed_kt = dist_km / dt / KT_TO_KMS
        flags = _spike_flags(speed_kt > max_speed_kt)
        if not flags.any():
            break
        keep[idx[flags]] = False
    return ~keep


def detect_altitude_spikes(
    t_s: NDArray[np.float64],
    altitude_ft: NDArray[np.float64],
    max_vertical_rate_fpm: float,
    max_iter: int = 10,
) -> NDArray[np.bool_]:
    """Return a mask of altitude spikes (True = spike); missing altitudes are never flagged."""
    keep = ~np.isnan(altitude_ft)
    spikes = np.zeros(t_s.size, dtype=bool)
    for _ in range(max_iter):
        idx = np.flatnonzero(keep)
        if idx.size < 3:
            break
        dt = np.maximum(np.diff(t_s[idx]), 1.0)
        rate_fpm = np.abs(np.diff(altitude_ft[idx])) / dt * 60.0
        flags = _spike_flags(rate_fpm > max_vertical_rate_fpm)
        if not flags.any():
            break
        keep[idx[flags]] = False
        spikes[idx[flags]] = True
    return spikes


def reject_implausible_values(
    df: pd.DataFrame, config: PipelineConfig
) -> tuple[pd.DataFrame, dict[str, int]]:
    """Null out ground speeds, vertical rates and altitudes outside physical limits."""
    df = df.copy()
    gs = df["groundspeed_kt"]
    bad_gs = (gs < 0) | (gs > config.max_groundspeed_kt)
    vr = df["vertical_rate_fpm"]
    bad_vr = vr.abs() > config.max_vertical_rate_fpm
    alt = df["altitude_ft"]
    lo, hi = config.altitude_bounds_ft
    bad_alt = (alt < lo) | (alt > hi)
    df.loc[bad_gs, "groundspeed_kt"] = np.nan
    df.loc[bad_vr, "vertical_rate_fpm"] = np.nan
    df.loc[bad_alt, "altitude_ft"] = np.nan
    counts = {
        "n_groundspeed_rejected": int(bad_gs.sum()),
        "n_vertical_rate_rejected": int(bad_vr.sum()),
        "n_altitude_out_of_bounds": int(bad_alt.sum()),
    }
    return df, counts


# ---------------------------------------------------------------------------------------
# Step 5: resampling with gap handling
# ---------------------------------------------------------------------------------------
def gap_statistics(
    t_s: NDArray[np.float64], gap_min_s: float, max_gap_s: float
) -> tuple[int, float, int, float]:
    """Count and sum short (interpolated) and long (left open) gaps between observations.

    Returns:
        ``(n_short, short_total_s, n_long, long_total_s)``.
    """
    dt = np.diff(t_s)
    short = (dt > gap_min_s) & (dt <= max_gap_s)
    long_ = dt > max_gap_s
    return int(short.sum()), float(dt[short].sum()), int(long_.sum()), float(dt[long_].sum())


def interpolate_with_gap_limit(
    t_obs: NDArray[np.float64],
    values: NDArray[np.float64],
    t_grid: NDArray[np.float64],
    max_gap_s: float,
) -> NDArray[np.float64]:
    """Linearly interpolate ``values`` onto ``t_grid`` without bridging long gaps.

    A grid point gets a value only when the two valid observations around it are at most
    ``max_gap_s`` apart (or when it coincides with an observation). Outside the observed
    time span the result is NaN.
    """
    valid = ~np.isnan(values)
    tv, vv = t_obs[valid], values[valid]
    out = np.full(t_grid.size, np.nan)
    if tv.size == 0:
        return out
    if tv.size == 1:
        out[np.isclose(t_grid, tv[0])] = vv[0]
        return out
    out = np.interp(t_grid, tv, vv, left=np.nan, right=np.nan)
    nxt = np.clip(np.searchsorted(tv, t_grid, side="left"), 1, tv.size - 1)
    gap = tv[nxt] - tv[nxt - 1]
    on_obs = np.isin(t_grid, tv)
    out[(gap > max_gap_s) & ~on_obs] = np.nan
    return out


def resample_uniform(df: pd.DataFrame, step_s: float, max_gap_s: float) -> pd.DataFrame:
    """Resample the numeric columns of a clean, sorted trajectory onto a uniform grid.

    The grid starts at the first timestamp floored to ``step_s`` and runs to the last one.
    Track angles are unwrapped before interpolation so that 359° -> 1° does not pass
    through 180°. Grid points with no position (inside long gaps) are dropped.
    """
    t0 = df["timestamp"].iloc[0].floor(f"{int(step_s)}s")
    t_obs = (df["timestamp"] - t0).dt.total_seconds().to_numpy(dtype=float)
    t_grid = np.arange(0.0, t_obs[-1] + 1e-9, step_s)

    out = pd.DataFrame({"timestamp": t0 + pd.to_timedelta(t_grid, unit="s")})
    for col in NUMERIC_COLUMNS:
        values = df[col].to_numpy(dtype=float)
        if col == "track":
            valid = ~np.isnan(values)
            values = values.copy()
            values[valid] = np.degrees(np.unwrap(np.radians(values[valid])))
            out[col] = np.mod(interpolate_with_gap_limit(t_obs, values, t_grid, max_gap_s), 360.0)
        else:
            out[col] = interpolate_with_gap_limit(t_obs, values, t_grid, max_gap_s)
    for col in STRING_COLUMNS:
        out[col] = df[col].iloc[0]
    out = out.dropna(subset=["lat", "lon"]).reset_index(drop=True)
    return out


# ---------------------------------------------------------------------------------------
# Step 6: derived fills
# ---------------------------------------------------------------------------------------
def derive_groundspeed_kt(t_s: NDArray[np.float64], lat: NDArray, lon: NDArray) -> NDArray:
    """Ground speed from consecutive positions (central differences), in knots."""
    n = t_s.size
    if n < 2:
        return np.full(n, np.nan)
    legs = haversine_km(lat[:-1], lon[:-1], lat[1:], lon[1:])
    dt = np.maximum(np.diff(t_s), 1e-6)
    leg_speed = legs / dt / KT_TO_KMS
    speed = np.empty(n)
    speed[0], speed[-1] = leg_speed[0], leg_speed[-1]
    speed[1:-1] = 0.5 * (leg_speed[:-1] + leg_speed[1:])
    return speed


def derive_vertical_rate_fpm(t_s: NDArray[np.float64], altitude_ft: NDArray) -> NDArray:
    """Vertical rate as the time derivative of altitude, in ft/min."""
    if t_s.size < 2:
        return np.zeros(t_s.size)
    return np.gradient(altitude_ft, t_s) * 60.0


def fill_derived(
    df: pd.DataFrame, ground_speed_threshold_kt: float
) -> tuple[pd.DataFrame, int, int]:
    """Fill missing ground speed, ground altitude and vertical rate.

    Returns:
        The filled frame, the number of ground altitudes filled, and the number of rows
        dropped because their altitude could not be recovered.
    """
    df = df.copy()
    t_s = elapsed_seconds(df["timestamp"])

    gs_derived = derive_groundspeed_kt(t_s, df["lat"].to_numpy(), df["lon"].to_numpy())
    df["groundspeed_kt"] = df["groundspeed_kt"].fillna(pd.Series(gs_derived, index=df.index))

    # Surface position messages carry no altitude: an aircraft that is slow and has no
    # altitude is on the ground, at (approximately) the nearest reported altitude.
    alt = df["altitude_ft"].to_numpy(dtype=float).copy()
    on_ground = np.isnan(alt) & (df["groundspeed_kt"].to_numpy() < ground_speed_threshold_kt)
    valid = np.flatnonzero(~np.isnan(alt))
    n_filled = 0
    if valid.size and on_ground.any():
        missing = np.flatnonzero(on_ground)
        tv = t_s[valid]
        right = np.clip(np.searchsorted(tv, t_s[missing]), 0, valid.size - 1)
        left = np.clip(right - 1, 0, valid.size - 1)
        use_left = np.abs(t_s[missing] - tv[left]) <= np.abs(tv[right] - t_s[missing])
        alt[missing] = alt[valid[np.where(use_left, left, right)]]
        n_filled = int(missing.size)
    df["altitude_ft"] = alt

    n_before = len(df)
    df = df.dropna(subset=["altitude_ft"]).reset_index(drop=True)
    n_dropped = n_before - len(df)

    t_s = elapsed_seconds(df["timestamp"]) if len(df) else np.zeros(0)
    vr_derived = derive_vertical_rate_fpm(t_s, df["altitude_ft"].to_numpy())
    df["vertical_rate_fpm"] = df["vertical_rate_fpm"].fillna(pd.Series(vr_derived, index=df.index))
    df["track"] = df["track"].ffill().bfill()
    return df, n_filled, n_dropped


# ---------------------------------------------------------------------------------------
# Orchestration
# ---------------------------------------------------------------------------------------
def clean_flight(
    df: pd.DataFrame, config: PipelineConfig | None = None
) -> tuple[pd.DataFrame, CleaningReport]:
    """Run all cleaning steps on one canonical trajectory.

    Args:
        df: Trajectory in the canonical schema (see :mod:`flight_emissions.ingest`).
        config: Pipeline configuration; defaults are used when omitted.

    Returns:
        The cleaned, uniformly resampled trajectory and the cleaning report.
    """
    config = config or PipelineConfig()
    flight_id = str(df["flight_id"].iloc[0]) if len(df) else "unknown"
    report = CleaningReport(flight_id=flight_id, n_input=len(df))

    df, report.n_missing_time_or_position = drop_unusable_rows(df)
    df, report.n_duplicates = merge_duplicates(df)
    if len(df) < 3:
        logger.warning("%s: fewer than 3 usable points, nothing to clean", flight_id)
        report.n_output = 0
        return df.iloc[0:0], report

    df, counts = reject_implausible_values(df, config)
    for key, value in counts.items():
        setattr(report, key, value)

    t_s = elapsed_seconds(df["timestamp"])
    pos_spikes = detect_position_spikes(
        t_s, df["lat"].to_numpy(), df["lon"].to_numpy(), config.max_implied_speed_kt
    )
    report.n_position_spikes = int(pos_spikes.sum())
    df = df.loc[~pos_spikes].reset_index(drop=True)
    t_s = t_s[~pos_spikes]

    alt_spikes = detect_altitude_spikes(
        t_s, df["altitude_ft"].to_numpy(dtype=float), config.max_implied_vertical_rate_fpm
    )
    report.n_altitude_spikes = int(alt_spikes.sum())
    df.loc[alt_spikes, "altitude_ft"] = np.nan

    (
        report.n_short_gaps,
        report.short_gap_total_s,
        report.n_long_gaps,
        report.long_gap_total_s,
    ) = gap_statistics(t_s, config.gap_min_s, config.max_gap_s)

    out = resample_uniform(df, config.resample_step_s, config.max_gap_s)
    out, report.n_ground_altitude_filled, report.n_dropped_without_altitude = fill_derived(
        out, config.ground_speed_threshold_kt
    )
    report.n_output = len(out)
    logger.info(report.summary())
    return out, report
