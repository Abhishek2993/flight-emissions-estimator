import numpy as np
import pandas as pd
import pytest

from flight_emissions.emissions import (
    CO2_PER_KG_FUEL,
    H2O_PER_KG_FUEL,
    SPECIES,
    emission_rates,
    emissions_by_phase,
)
from flight_emissions.fuel import resolve_aircraft


def test_co2_and_h2o_are_proportional_to_fuel():
    ff = np.array([0.1, 0.5, 1.0, 2.5])
    rates = emission_rates(ff, np.full(4, 300.0), np.full(4, 20000.0), resolve_aircraft("A320"))
    assert rates["co2_gs"].to_numpy() == pytest.approx(ff * CO2_PER_KG_FUEL * 1000)
    assert rates["h2o_gs"].to_numpy() == pytest.approx(ff * H2O_PER_KG_FUEL * 1000)


def test_nox_index_rises_with_thrust_and_co_hc_fall():
    model = resolve_aircraft("A320")
    low, high = 0.12, 1.1  # kg/s: near idle vs near take-off fuel flow
    rates = emission_rates(np.array([low, high]), np.array([150.0, 150.0]), np.zeros(2), model)
    ei = rates.div([low, high], axis=0)  # g per kg of fuel
    assert ei["nox_gs"].iloc[1] > 3 * ei["nox_gs"].iloc[0]
    assert ei["co_gs"].iloc[0] > 5 * ei["co_gs"].iloc[1]
    assert ei["hc_gs"].iloc[0] > ei["hc_gs"].iloc[1]


def test_taxi_inputs_do_not_produce_nan():
    rates = emission_rates(
        np.array([0.2]), np.array([0.0]), np.array([-50.0]), resolve_aircraft("B738")
    )
    assert np.isfinite(rates.to_numpy()).all()


def test_emissions_by_phase_integrates_to_kg():
    n = 7
    df = pd.DataFrame(
        {
            "timestamp": pd.date_range("2024-01-01", periods=n, freq="10s", tz="UTC"),
            "phase": ["climb"] * 3 + ["cruise"] * 4,
        }
    )
    rates = pd.DataFrame({f"{s}_gs": np.full(n, 1000.0) for s in SPECIES})  # 1 kg/s each
    table = emissions_by_phase(df, rates)
    assert list(table.columns) == list(SPECIES)
    assert table.loc["climb", "co2"] == pytest.approx(30.0)
    assert table.loc["cruise", "co2"] == pytest.approx(30.0)
    assert table.loc["ground", "nox"] == 0.0
    assert table.index[0] == "ground"
