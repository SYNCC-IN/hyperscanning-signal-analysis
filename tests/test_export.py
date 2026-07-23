"""Unit tests for export helpers in src.export."""

import json

import numpy as np
import pandas as pd
import pytest
import xarray as xr

from src.data_structures import MultimodalData
from src import export


@pytest.fixture
def multimodal_data_sample():
    """Create a synthetic MultimodalData instance with movie and talk events."""
    md = MultimodalData()
    md.id = "W_999"
    md.fs = 1.0
    md.modalities = ["EEG"]

    md.events = {
        "Peppa": {"name": "Peppa", "start": 10.0, "duration": 10.0},
        "Incredibles": {"name": "Incredibles", "start": 30.0, "duration": 10.0},
        "Brave": {"name": "Brave", "start": 50.0, "duration": 10.0},
        "talk_intro": {"name": "talk_intro", "start": 70.0, "duration": 5.0},
        "talk_outro": {"name": "talk_outro", "start": 80.0, "duration": 8.0},
    }

    time = np.arange(0.0, 101.0, 1.0)
    events_col = [None] * len(time)
    signal_ch = np.sin(time / 10.0)
    signal_cg = np.cos(time / 10.0)

    md.data = pd.DataFrame(
        {
            "time": time,
            "events": events_col,
            "EEG_ch_Fp1": signal_ch,
            "EEG_cg_Fp1": signal_cg,
        }
    )
    return md


def test_export_chunk_to_xarray_resets_time_and_stores_events_structure(multimodal_data_sample):
    """Chunk export should reset time at first event and persist events structure metadata."""
    da = export.export_chunk_to_xarray(
        multimodal_data=multimodal_data_sample,
        selected_events=["Brave", "Peppa", "Incredibles"],
        selected_channels=["Fp1"],
        selected_modality="EEG",
        member="ch",
        time_margin=20,
        chunk_name="passive_movies",
        verbose=False,
    )

    # The first selected event by time is Peppa at t=10, so chunk time should include -10 margin.
    time_values = np.asarray(da.coords["time"].values, dtype=float)
    assert np.isclose(time_values.min(), -10.0)
    assert np.isclose(time_values.max(), 70.0)
    assert np.any(np.isclose(time_values, 0.0))

    channel_values = list(da.coords["channel"].values)
    assert channel_values == ["Fp1"]

    assert da.attrs["task_name"] == "passive_movies"
    assert da.attrs["task_start"] == 0.0
    assert da.attrs["task_duration"] == 50.0

    ordered_events = da.attrs["task_event_names_csv"].split(",")
    assert ordered_events == ["Peppa", "Incredibles", "Brave"]

    events_structure = json.loads(da.attrs["task_events_structure"])
    assert [ev["name"] for ev in events_structure] == ["Peppa", "Incredibles", "Brave"]
    assert np.isclose(events_structure[0]["start_s"], 10.0)
    assert np.isclose(events_structure[0]["start_rel_s"], 0.0)
    assert np.isclose(events_structure[1]["start_rel_s"], 20.0)
    assert np.isclose(events_structure[2]["start_rel_s"], 40.0)


def test_export_passive_and_talk_data_exports_two_chunks_with_default_margin(
    multimodal_data_sample,
    monkeypatch,
    tmp_path,
):
    """Orchestrator should export passive_movies and talk chunks for each member/modality."""
    export_calls = []
    saved_paths = []

    def fake_create_multimodal_data(*args, **kwargs):
        return multimodal_data_sample

    def fake_export_chunk_to_xarray(**kwargs):
        export_calls.append(kwargs)
        return xr.DataArray(
            np.zeros((2, 1), dtype=float),
            coords=[np.array([0.0, 1.0]), ["Fp1"]],
            dims=["time", "channel"],
            name="signals",
        )

    def fake_to_netcdf(self, path, *args, **kwargs):
        saved_paths.append(path)

    monkeypatch.setattr(export.dataloader, "create_multimodal_data", fake_create_multimodal_data)
    monkeypatch.setattr(export, "export_chunk_to_xarray", fake_export_chunk_to_xarray)
    monkeypatch.setattr(xr.DataArray, "to_netcdf", fake_to_netcdf, raising=False)

    export.export_passive_and_talk_data(
        dyad_id_list=["W_999"],
        export_path=str(tmp_path),
        verbose=False,
    )

    # 1 modality (EEG) x 2 members (ch, cg) x 2 chunks (passive_movies, talk)
    assert len(export_calls) == 4
    assert len(saved_paths) == 4

    chunk_names = {call["chunk_name"] for call in export_calls}
    assert chunk_names == {"passive_movies", "talk"}
    assert all(call["time_margin"] == 20 for call in export_calls)

    passive_calls = [call for call in export_calls if call["chunk_name"] == "passive_movies"]
    talk_calls = [call for call in export_calls if call["chunk_name"] == "talk"]

    assert all(call["selected_events"] == ["Peppa", "Incredibles", "Brave"] for call in passive_calls)
    assert all(call["selected_events"] == ["talk_intro", "talk_outro"] for call in talk_calls)

    assert any("passive_movies.nc" in path for path in saved_paths)
    assert any("talk.nc" in path for path in saved_paths)


def test_export_passive_and_talk_data_includes_diode_modality_when_present(
    multimodal_data_sample,
    monkeypatch,
    tmp_path,
):
    """If diode column exists in data, diode chunk files should also be exported."""
    multimodal_data_sample.data["diode"] = np.linspace(0.0, 1.0, len(multimodal_data_sample.data))

    export_calls = []
    saved_paths = []

    def fake_create_multimodal_data(*args, **kwargs):
        return multimodal_data_sample

    def fake_export_chunk_to_xarray(**kwargs):
        export_calls.append(kwargs)
        return xr.DataArray(
            np.zeros((2, 1), dtype=float),
            coords=[np.array([0.0, 1.0]), ["dummy"]],
            dims=["time", "channel"],
            name="signals",
        )

    def fake_to_netcdf(self, path, *args, **kwargs):
        saved_paths.append(path)

    monkeypatch.setattr(export.dataloader, "create_multimodal_data", fake_create_multimodal_data)
    monkeypatch.setattr(export, "export_chunk_to_xarray", fake_export_chunk_to_xarray)
    monkeypatch.setattr(xr.DataArray, "to_netcdf", fake_to_netcdf, raising=False)

    export.export_passive_and_talk_data(
        dyad_id_list=["W_999"],
        export_path=str(tmp_path),
        verbose=False,
    )

    exported_modalities = {call["selected_modality"] for call in export_calls}
    assert "EEG" in exported_modalities
    assert "diode" in exported_modalities

    diode_paths = [path for path in saved_paths if "/diode/" in path]
    assert len(diode_paths) == 4  # 2 members x 2 chunks
