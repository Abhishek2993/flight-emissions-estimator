import numpy as np
import pandas as pd
import pytest

from flight_emissions.ingest import (
    CANONICAL_COLUMNS,
    META_COLUMNS,
    IngestError,
    load_flights,
    normalise,
    opensky_credentials_available,
    split_flights,
)


def raw_traffic_style() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "timestamp": pd.date_range("2024-07-06 06:00", periods=4, freq="10s", tz="UTC")[::-1],
            "latitude": [49.0, 49.01, 49.02, 49.03],
            "longitude": [2.5, 2.51, 2.52, 2.53],
            "altitude": pd.array([None, 1000, 1500, 2000], dtype="Int64"),
            "groundspeed": [150, 160, 170, 180],
            "vertical_speed": [0, 1500, 1500, 1500],
            "track": [10, 10, 10, 10],
            "icao24": "393322",
            "callsign": "AFR34ZG",
        }
    )


def test_normalise_maps_aliases_and_sorts():
    df = normalise(raw_traffic_style(), typecode="a321", origin="LFPG", destination="LFBO")
    assert list(df.columns) == CANONICAL_COLUMNS + META_COLUMNS
    assert df["timestamp"].is_monotonic_increasing
    assert str(df["timestamp"].dtype) == "datetime64[ns, UTC]"
    assert df["typecode"].unique().tolist() == ["A321"]
    assert df["origin"].iloc[0] == "LFPG"
    assert df["vertical_rate_fpm"].dtype == np.float64
    assert np.isnan(df["altitude_ft"]).sum() == 1  # nullable integer NA became NaN
    assert df["flight_id"].iloc[0] == "AFR34ZG_20240706"


def test_normalise_accepts_epoch_seconds():
    raw = pd.DataFrame({"time": [1_700_000_000, 1_700_000_010], "lat": [0, 0], "lon": [0, 1]})
    df = normalise(raw)
    assert df["timestamp"].iloc[0] == pd.Timestamp(1_700_000_000, unit="s", tz="UTC")
    assert df["typecode"].iloc[0] is None
    assert df["flight_id"].iloc[0] == "flight_20231114"


def test_normalise_requires_position():
    with pytest.raises(IngestError):
        normalise(pd.DataFrame({"timestamp": ["2024-01-01"], "altitude": [1000]}))


def test_split_flights_by_id():
    a = raw_traffic_style().assign(flight_id="A")
    b = raw_traffic_style().assign(flight_id="B")
    parts = split_flights(pd.concat([a, b]))
    assert [p["flight_id"].iloc[0] for p in parts] == ["A", "B"]


def test_load_flights_from_directory(tmp_path, make_flight):
    make_flight(np.full(5, 30000.0), flight_id="F2").to_parquet(tmp_path / "f2.parquet")
    make_flight(np.full(5, 30000.0), flight_id="F1").to_csv(tmp_path / "f1.csv", index=False)
    flights = load_flights(tmp_path)
    assert [f["flight_id"].iloc[0] for f in flights] == ["F1", "F2"]
    assert len(load_flights(tmp_path / "f2.parquet")) == 1


def test_load_flights_errors(tmp_path):
    with pytest.raises(IngestError):
        load_flights(tmp_path)  # empty directory
    with pytest.raises(IngestError):
        load_flights(tmp_path / "missing.parquet")
    (tmp_path / "x.txt").write_text("nope")
    with pytest.raises(IngestError):
        from flight_emissions.ingest import read_table

        read_table(tmp_path / "x.txt")


def test_opensky_credentials_from_environment(monkeypatch):
    monkeypatch.delenv("OPENSKY_USERNAME", raising=False)
    monkeypatch.delenv("OPENSKY_PASSWORD", raising=False)
    assert not opensky_credentials_available()
    monkeypatch.setenv("OPENSKY_USERNAME", "user")
    monkeypatch.setenv("OPENSKY_PASSWORD", "secret")
    assert opensky_credentials_available()


def test_merge_messages_per_second():
    from flight_emissions.sample_data import merge_messages_per_second

    raw = pd.DataFrame(
        {
            "timestamp": pd.to_datetime(
                ["2024-01-01 00:00:00.1", "2024-01-01 00:00:00.6", "2024-01-01 00:00:01.2"],
                utc=True,
            ),
            "latitude": [1.0, None, 1.1],
            "groundspeed": [None, 300.0, 301.0],
        }
    )
    out = merge_messages_per_second(raw)
    assert len(out) == 2
    assert out.loc[0, "latitude"] == 1.0 and out.loc[0, "groundspeed"] == 300.0
