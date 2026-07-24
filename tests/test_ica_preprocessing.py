import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import xarray as xr

from src.ica_preprocessing import ICAPreprocessor


def test_extract_provenance_fallback_from_stem():
    preprocessor = ICAPreprocessor(export_folder=None, target_events=[])

    provenance = preprocessor._extract_provenance({}, "W_030_EEG_ch_passive_movies")

    assert provenance == {"dyad_id": "W_030", "who": "ch", "site": "W"}


def test_extract_provenance_from_attrs():
    preprocessor = ICAPreprocessor(export_folder=None, target_events=[])

    attrs = {"dyad_id": "W_030", "who": "cg", "site": "W"}

    assert preprocessor._extract_provenance(attrs, "ignored") == attrs


def test_sanitize_attrs_converts_nested_and_none_values():
    preprocessor = ICAPreprocessor(export_folder=None, target_events=[])

    attrs = {
        "none_value": None,
        "dict_value": {"a": 1},
        "list_value": [1, 2],
        "int_value": np.int64(3),
        "float_value": np.float64(4.5),
        "string_value": "ok",
        "object_value": object(),
    }

    sanitized = preprocessor._sanitize_attrs(attrs)

    assert sanitized["none_value"] == ""
    assert json.loads(sanitized["dict_value"]) == {"a": 1}
    assert json.loads(sanitized["list_value"]) == [1, 2]
    assert sanitized["int_value"] == 3
    assert sanitized["float_value"] == 4.5
    assert sanitized["string_value"] == "ok"
    assert sanitized["object_value"] == str(attrs["object_value"])


def test_plot_component_raises_for_unknown_component(tmp_path):
    ds = xr.Dataset(
        data_vars={
            "topomap_raw": (("component", "channel"), np.ones((1, 2))),
            "psd_power": (("component", "frequency"), np.ones((1, 4))),
            "fooof_aperiodic": (("component", "ap_param"), np.array([[1.0, 1.5]])),
            "fooof_peaks": (("component", "peak_idx", "peak_param"), np.ones((1, 2, 3))),
            "fooof_r_squared": (("component",), np.array([0.95])),
            "fooof_valid": (("component",), np.array([True])),
            "explained_var_ratio": (("component",), np.array([0.5])),
        },
        coords={
            "component": ["ICA000"],
            "channel": ["Fp1", "Fp2"],
            "frequency": [1.0, 2.0, 3.0, 4.0],
            "peak_idx": [0, 1],
            "peak_param": ["CF", "PW", "BW"],
            "ap_param": ["offset", "exponent"],
        },
        attrs={"file_stem": "demo", "fooof_aperiodic_mode": "fixed"},
    )
    path = tmp_path / "demo_components.nc"
    ds.to_netcdf(path)

    preprocessor = ICAPreprocessor(export_folder=Path("."), target_events=[])

    with pytest.raises(ValueError):
        preprocessor.plot_component(path, "ICA999", show=False)


def test_plot_component_grid_returns_figure(tmp_path, monkeypatch):
    ds = xr.Dataset(
        data_vars={
            "topomap_raw": (("component", "channel"), np.ones((1, 2))),
            "psd_power": (("component", "frequency"), np.ones((1, 4))),
            "fooof_aperiodic": (("component", "ap_param"), np.array([[1.0, 1.5]])),
            "fooof_peaks": (("component", "peak_idx", "peak_param"), np.ones((1, 2, 3))),
            "fooof_r_squared": (("component",), np.array([0.95])),
            "fooof_valid": (("component",), np.array([True])),
            "explained_var_ratio": (("component",), np.array([0.5])),
        },
        coords={
            "component": ["ICA000"],
            "channel": ["Fp1", "Fp2"],
            "frequency": [1.0, 2.0, 3.0, 4.0],
            "peak_idx": [0, 1],
            "peak_param": ["CF", "PW", "BW"],
            "ap_param": ["offset", "exponent"],
        },
        attrs={"file_stem": "demo", "fooof_aperiodic_mode": "fixed"},
    )
    path = tmp_path / "demo_components.nc"
    ds.to_netcdf(path)

    def fake_plot_topomap(*args, **kwargs):
        return np.ones(1), None

    monkeypatch.setattr("mne.viz.plot_topomap", fake_plot_topomap)

    preprocessor = ICAPreprocessor(export_folder=Path("."), target_events=[])
    fig = preprocessor.plot_component_grid(path, n_cols=1, show=False)

    assert fig is not None


def test_collect_features_and_clustering(tmp_path):
    components_dir = tmp_path / "ica_output" / "W_030"
    components_dir.mkdir(parents=True)

    ds = xr.Dataset(
        data_vars={
            "topomap": (("component", "channel"), np.array([[0.9, 0.1]])),
            "fooof_aperiodic": (("component", "ap_param"), np.array([[1.0, 1.5]])),
            "fooof_peaks": (("component", "peak_idx", "peak_param"), np.array([[[2.0, 0.3, 0.5], [np.nan, np.nan, np.nan]]])),
            "fooof_r_squared": (("component",), np.array([0.95])),
            "fooof_valid": (("component",), np.array([True])),
            "explained_var_ratio": (("component",), np.array([0.5])),
        },
        coords={
            "component": ["ICA000"],
            "channel": ["Fp1", "Fp2"],
            "peak_idx": [0, 1],
            "peak_param": ["CF", "PW", "BW"],
            "ap_param": ["offset", "exponent"],
        },
        attrs={"file_stem": "W_030_EEG_ch_passive_movies", "dyad_id": "W_030", "who": "ch", "site": "W"},
    )
    ds.to_netcdf(components_dir / "W_030_EEG_ch_passive_movies_components.nc")

    fake_nc = tmp_path / "W_030_EEG_ch_passive_movies.nc"
    xr.DataArray(np.ones((2, 2)), dims=["time", "channel"], attrs={"dyad_id": "W_030", "who": "ch", "site": "W"}).to_netcdf(fake_nc)

    preprocessor = ICAPreprocessor(export_folder=tmp_path, target_events=["passive_movies"])
    preprocessor.eeg_files = [fake_nc]

    features = preprocessor.collect_features(components_dir.parent, member_filter='ch')
    assert not features.empty
    assert "fooof_cf_0" in features.columns

    df_out, templates = preprocessor.cluster_components(features, n_clusters=1, features='both')
    assert 'cluster_label' in df_out.columns
    assert templates[0]['n_components'] == 1

    equiv = preprocessor.find_cross_group_equivalents({0: templates[0]}, {0: templates[0]}, features='topomap')
    assert not equiv.empty

    out_csv = tmp_path / "template.csv"
    preprocessor.export_assignment_template(df_out, out_csv)
    assert out_csv.exists()
