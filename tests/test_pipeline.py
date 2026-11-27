import numpy as np
import pandas as pd
import pytest

from flight_emissions.cli import main
from flight_emissions.config import PipelineConfig
from flight_emissions.pipeline import process_flight, results_markdown, run_pipeline


def mission(make_flight, typecode="A320", flight_id="SYN001", origin="AAAA", dest="BBBB"):
    taxi = np.zeros(60)
    climb = np.linspace(0, 35000, 121)[1:]
    cruise = np.full(500, 35000.0)
    descent = np.linspace(35000, 0, 151)[1:]
    alt = np.concatenate([taxi, climb, cruise, descent, taxi])
    speed = np.concatenate(
        [
            np.full(60, 15.0),
            np.full(120, 280.0),
            np.full(500, 450.0),
            np.full(150, 300.0),
            np.full(60, 15.0),
        ]
    )
    df = make_flight(alt, speed_kt=speed, typecode=typecode, flight_id=flight_id)
    df["origin"], df["destination"] = origin, dest
    return df


def test_process_flight_end_to_end(make_flight):
    r = process_flight(mission(make_flight))
    assert r.ok and r.warnings == []
    pt = r.phase_table
    assert set(pt.index) == {"ground", "climb", "cruise", "level", "descent"}
    assert pt["fuel_kg"].sum() == pytest.approx(r.timeseries["fuel_cum_kg"].iloc[-1])
    assert pt["co2_kg"].sum() == pytest.approx(3.16 * pt["fuel_kg"].sum(), rel=1e-9)
    assert pt.loc["cruise", "fuel_kg"] > pt.loc["climb", "fuel_kg"] > pt.loc["descent", "fuel_kg"]
    row = r.summary_row()
    assert row["route_extra_pct"] == pytest.approx(0.0, abs=0.01)
    assert row["descent_level_offs"] == 0
    assert row["fuel_kg"] == pytest.approx(pt["fuel_kg"].sum())


def test_missing_typecode_is_skipped_or_uses_fallback(make_flight):
    raw = mission(make_flight, typecode=None)
    skipped = process_flight(raw)
    assert not skipped.ok and "skipped" in skipped.status
    assert "fuel_kg" not in skipped.summary_row()
    fallback = process_flight(raw, PipelineConfig(fallback_typecode="A320"))
    assert fallback.ok
    assert any("fallback" in w for w in fallback.warnings)


def test_too_little_data_is_skipped(make_flight):
    r = process_flight(make_flight(np.full(5, 30000.0)))
    assert r.status.startswith("skipped")


def test_long_gap_produces_warning(make_flight):
    raw = mission(make_flight).drop(index=range(300, 400))  # 1000 s gap in cruise
    r = process_flight(raw)
    assert r.ok
    assert any("gap" in w for w in r.warnings)


@pytest.fixture
def flight_dir(tmp_path, make_flight):
    d = tmp_path / "flights"
    d.mkdir()
    mission(make_flight, flight_id="SYN001").to_parquet(d / "a.parquet")
    mission(make_flight, typecode="B738", flight_id="SYN002", origin="CCCC").to_parquet(
        d / "b.parquet"
    )
    mission(make_flight, typecode="ZZZZ", flight_id="SYN003").to_parquet(d / "c.parquet")
    ref = tmp_path / "icao.csv"
    ref.write_text(
        "origin,destination,typecode,icao_co2_kg_per_pax,icao_co2_kg_per_flight,note\n"
        "AAAA,BBBB,A320,60,,\nCCCC,BBBB,B738,,,\n"
    )
    return d, ref


def test_run_pipeline_writes_all_outputs(tmp_path, flight_dir):
    d, ref = flight_dir
    out = tmp_path / "out"
    outputs = run_pipeline(d, out, reference_path=ref, fdr_path=None, make_figures=True)
    for name in (
        "flights_summary.csv",
        "phase_breakdown.csv",
        "cleaning_report.csv",
        "level_offs.csv",
        "icao_validation.csv",
        "run_config.json",
        "summary.md",
    ):
        assert (out / name).exists(), name
    summary = pd.read_csv(out / "flights_summary.csv")
    assert summary["status"].tolist()[:2] == ["ok", "ok"]
    assert summary["status"].iloc[2].startswith("skipped")
    assert (out / "timeseries" / "SYN001.parquet").exists()
    assert outputs.icao["status"].tolist() == ["compared", "awaiting ICAO value"]
    names = {p.name for p in outputs.figures}
    assert {
        "SYN001_overview.png",
        "SYN001_descent.png",
        "fuel_by_phase.png",
        "inefficiency.png",
        "icao_validation.png",
    } <= names
    md = (out / "summary.md").read_text()
    assert "SYN001" in md and "skipped" in md


def test_results_markdown_without_validation(flight_dir, tmp_path):
    d, _ = flight_dir
    outputs = run_pipeline(
        d, tmp_path / "o", reference_path=None, fdr_path=None, make_figures=False
    )
    md = results_markdown(outputs.summary, outputs.icao, outputs.fdr)
    assert md.count("\n|") >= 4


def test_cli_run(tmp_path, flight_dir, capsys):
    d, ref = flight_dir
    code = main(
        [
            "run",
            "--input",
            str(d),
            "--out",
            str(tmp_path / "cli"),
            "--no-figures",
            "--reference",
            str(ref),
            "--fdr",
            str(tmp_path / "missing.parquet"),
            "--mass-fraction",
            "0.8",
        ]
    )
    assert code == 0
    assert "2/3 flights processed" in capsys.readouterr().out
    assert '"mass_fraction": 0.8' in (tmp_path / "cli" / "run_config.json").read_text()
