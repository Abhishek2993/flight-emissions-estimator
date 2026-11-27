import numpy as np
import pandas as pd
import pytest

from flight_emissions.config import PipelineConfig
from flight_emissions.fuel import fuel_flow_model
from flight_emissions.validate import (
    FDR_SCENARIOS,
    check_fuel_model_against_fdr,
    compare_with_icao,
    load_icao_reference,
)


@pytest.fixture
def summary() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "flight_id": ["F1", "F2"],
            "origin": ["LFPG", "LTFM"],
            "destination": ["LFBO", "ENGM"],
            "typecode": ["A321", "B738"],
            "co2_kg": [10_000.0, 30_000.0],
            "seats": [200.0, 175.0],
        }
    )


def test_compare_with_icao_statuses_and_differences(tmp_path, summary):
    path = tmp_path / "icao.csv"
    path.write_text(
        "origin,destination,typecode,icao_co2_kg_per_pax,icao_co2_kg_per_flight,note\n"
        "lfpg,lfbo,A321,50,8000,typed in\n"
        "LTFM,ENGM,,,,\n"
        "EGLL,KJFK,B77W,300,,\n"
    )
    ref = load_icao_reference(path)
    out = compare_with_icao(summary, ref, PipelineConfig(load_factor=0.8))
    assert out["status"].tolist() == ["compared", "awaiting ICAO value", "no matching flight"]
    first = out.iloc[0]
    assert first["assumed_passengers"] == pytest.approx(160.0)
    assert first["our_co2_kg_per_pax"] == pytest.approx(10_000 / 160)
    assert first["diff_pct_per_flight"] == pytest.approx(25.0)
    assert first["diff_pct_per_pax"] == pytest.approx(100 * (62.5 - 50) / 50)
    assert np.isnan(out.iloc[1]["diff_pct_per_pax"])
    assert out.iloc[1]["flight_id"] == "F2"  # blank typecode matches any type


def test_reference_with_missing_columns_is_padded(tmp_path):
    path = tmp_path / "icao.csv"
    path.write_text("origin,destination\nLFPG,LFBO\n")
    ref = load_icao_reference(path)
    assert list(ref.columns)[:3] == ["origin", "destination", "typecode"]
    assert np.isnan(ref.loc[0, "icao_co2_kg_per_pax"])


def synthetic_fdr(minutes=30):
    from openap import aero

    n = minutes * 60
    alt = np.full(n, 35000.0)
    cas = np.full(n, 260.0)
    tas = aero.cas2tas(cas * aero.kts, alt * aero.ft) / aero.kts
    mass = np.linspace(66_000, 65_000, n)
    ff = fuel_flow_model("A320").enroute(mass=mass, tas=tas, alt=alt, vs=0)
    return pd.DataFrame(
        {
            "timestamp": pd.date_range("2024-01-01", periods=n, freq="1s", tz="UTC"),
            "altitude_ft": alt,
            "groundspeed_kt": tas,  # no wind
            "cas_kt": cas,
            "mass_kg": mass,
            "fuel_flow_kgh": ff * 3600,
        }
    )


def test_fdr_check_reproduces_a_recording_made_by_the_model():
    summary, ts = check_fuel_model_against_fdr(synthetic_fdr())
    assert summary["scenario"].tolist() == list(FDR_SCENARIOS)
    exact = summary.iloc[0]
    assert exact["total_error_pct"] == pytest.approx(0.0, abs=0.05)
    assert exact["flow_mape_pct"] < 0.1
    # Without wind, ground speed equals TAS, so scenario 2 is exact as well.
    assert summary.iloc[1]["total_error_pct"] == pytest.approx(0.0, abs=0.05)
    # The assumed mass (0.85 x 78 t = 66.3 t, minus burn) differs from the recorded 65-66 t.
    assert summary.iloc[2]["mean_mass_kg"] > 65_000
    assert len(ts) == 180 and "recorded_kgs" in ts
