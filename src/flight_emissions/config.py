"""Pipeline configuration.

Every tunable threshold of the pipeline lives in :class:`PipelineConfig` so that a run is
fully described by one object (and can be logged next to its results). The defaults are
chosen for commercial jet transport aircraft observed through ADS-B; each field documents
why its default value was chosen.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any


@dataclass(frozen=True)
class PipelineConfig:
    """Tunable parameters of the cleaning, phase, fuel and inefficiency stages."""

    # ---- cleaning -------------------------------------------------------------------
    resample_step_s: int = 10
    """Uniform time step after resampling. 10 s keeps a 10-hour flight at ~3,600 rows while
    still resolving a 60 s level-off with six samples."""

    gap_min_s: float = 30.0
    """Intervals between position reports longer than this are counted as gaps."""

    max_gap_s: float = 300.0
    """Gaps up to this length are linearly interpolated. Longer gaps are *not* filled:
    the grid points inside them are dropped and the gap is reported."""

    max_implied_speed_kt: float = 800.0
    """A position that implies more than this speed both from its predecessor and to its
    successor is a spike. 800 kt sits above any subsonic ground speed, even with a
    record jet-stream tailwind."""

    max_groundspeed_kt: float = 750.0
    """Reported ground speeds above this are physically implausible for subsonic aircraft."""

    max_implied_vertical_rate_fpm: float = 10_000.0
    """An altitude sample implying more than this climb/descent rate on both sides is a spike."""

    max_vertical_rate_fpm: float = 8_000.0
    """Reported vertical rates beyond this are discarded and re-derived from altitude.
    Airliners rarely exceed 6,000 ft/min, even in an emergency descent."""

    altitude_bounds_ft: tuple[float, float] = (-1_500.0, 50_000.0)
    """Altitudes outside this range (pressure altitude) are discarded."""

    # ---- flight phases --------------------------------------------------------------
    phase_window_s: int = 60
    """Averaging window of the OpenAP fuzzy-logic phase classifier."""

    ground_speed_threshold_kt: float = 50.0
    """Below this ground speed an aircraft is on the ground, whatever its altitude reads.
    Used to fix OpenAP's sea-level ground rule at elevated airports."""

    cruise_band_ft: float = 4_000.0
    """Level flight within this distance of the flight's highest altitude (and above
    10,000 ft) is cruise. OpenAP's cruise rule keys on high *speed*, so a cruise flown into
    a strong headwind (low ground speed) would otherwise be labelled "level"."""

    min_segment_s: float = 120.0
    """Phase segments shorter than this are merged into a neighbour."""

    # ---- fuel -----------------------------------------------------------------------
    mass_fraction: float = 0.85
    """ASSUMPTION: take-off mass as a fraction of maximum take-off weight (MTOW)."""

    update_mass: bool = True
    """Reduce aircraft mass by the fuel already burned (two-pass fixed-point estimate)."""

    taxi_thrust_fraction: float = 0.07
    """Thrust setting while taxiing. 7 % is the ICAO LTO-cycle idle/taxi setting."""

    fallback_typecode: str | None = None
    """Type used when a flight has no (or an unsupported) type code. ``None`` = skip the
    flight instead of guessing."""

    # ---- inefficiency ---------------------------------------------------------------
    min_great_circle_km: float = 100.0
    """Below this origin-destination distance a flight is treated as circular
    (round trip, training or sightseeing flight) and route inefficiency is not reported."""

    level_vr_threshold_fpm: float = 300.0
    """|vertical rate| below this counts as level flight in level-off detection."""

    min_level_off_s: float = 60.0
    """A level segment inside the descent must last this long to count as a level-off."""

    descent_floor_above_landing_ft: float = 3_000.0
    """The descent analysis stops this far above the landing altitude: the final-approach
    level segment used to intercept the ILS glide slope is procedural, not an inefficiency."""

    cdo_glide_angle_deg: float = 3.0
    """Flight-path angle of the idealised continuous descent (~318 ft per NM)."""

    # ---- validation -----------------------------------------------------------------
    load_factor: float = 0.80
    """ASSUMPTION: passenger load factor used to convert per-flight CO2 to per-passenger."""

    extra: dict[str, Any] = field(default_factory=dict)
    """Free-form metadata stored alongside results (e.g. the data source)."""

    def to_dict(self) -> dict[str, Any]:
        """Return the configuration as a plain dictionary (for logging and CSV output)."""
        return asdict(self)
