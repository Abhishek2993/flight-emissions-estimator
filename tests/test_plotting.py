import numpy as np
import pandas as pd

from flight_emissions.plotting import (
    PHASE_COLORS,
    plot_fdr_check,
    plot_icao,
    world_lines,
)
from flight_emissions.validate import FDR_SCENARIOS


def test_world_boundaries_are_bundled():
    lines = world_lines()
    assert len(lines) > 100
    assert all(line.shape[1] == 2 for line in lines[:10])


def test_every_phase_has_a_colour():
    assert {"ground", "climb", "cruise", "level", "descent"} <= set(PHASE_COLORS)


def test_fdr_and_icao_plots():
    n = 50
    ts = pd.DataFrame(
        {
            "timestamp": pd.date_range("2024-01-01", periods=n, freq="10s", tz="UTC"),
            "altitude_ft": np.linspace(0, 30000, n),
            "recorded_kgs": np.ones(n),
            **{name: np.ones(n) for name in FDR_SCENARIOS},
        }
    )
    summary = pd.DataFrame({"scenario": list(FDR_SCENARIOS), "total_error_pct": [1.0, 2.0, 3.0]})
    assert plot_fdr_check(ts, summary) is not None
    assert plot_icao(pd.DataFrame()) is None
    pending = pd.DataFrame({"status": ["awaiting ICAO value"], "icao_co2_kg_per_pax": [np.nan]})
    assert plot_icao(pending) is None
