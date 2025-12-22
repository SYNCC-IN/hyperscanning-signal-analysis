"""
Core tests for dataloader module.

Tests the loading, filtering, and processing of EEG/ECG data from SVAROG format files.
"""
import pytest
import numpy as np
import pandas as pd
import os
import tempfile

from src.data_structures import MultimodalData
from src import dataloader


class TestLoadEegDataIntegration:
    """Integration tests for load_eeg_data function using real data files."""

    @pytest.fixture
    def sample_data_path(self):
        """Path to sample EEG data."""
        return "data/eeg/"

    @pytest.fixture
    def sample_dyad_id(self):
        """Sample dyad ID for testing."""
        return "W_001"

    @pytest.mark.integration
    def test_load_eeg_data_returns_multimodal_data(self, sample_dyad_id, sample_data_path):
        """load_eeg_data should return a MultimodalData instance."""
        if not os.path.exists(sample_data_path):
            pytest.skip("Sample data not available")
        
        result = dataloader.load_eeg_data(sample_dyad_id, sample_data_path, plot_flag=False)
        assert isinstance(result, MultimodalData)

    @pytest.mark.integration
    def test_load_eeg_data_sets_id(self, sample_dyad_id, sample_data_path):
        """load_eeg_data should set the dyad ID correctly."""
        if not os.path.exists(sample_data_path):
            pytest.skip("Sample data not available")
        
        result = dataloader.load_eeg_data(sample_dyad_id, sample_data_path, plot_flag=False)
        assert result.id == sample_dyad_id

    @pytest.mark.integration
    def test_load_eeg_data_populates_dataframe(self, sample_dyad_id, sample_data_path):
        """load_eeg_data should populate the data DataFrame."""
        if not os.path.exists(sample_data_path):
            pytest.skip("Sample data not available")
        
        result = dataloader.load_eeg_data(sample_dyad_id, sample_data_path, plot_flag=False)
        
        assert len(result.data) > 0
        assert 'time' in result.data.columns
        assert 'time_idx' in result.data.columns

    @pytest.mark.integration
    def test_load_eeg_data_creates_eeg_columns(self, sample_dyad_id, sample_data_path):
        """load_eeg_data should create EEG channel columns."""
        if not os.path.exists(sample_data_path):
            pytest.skip("Sample data not available")
        
        result = dataloader.load_eeg_data(sample_dyad_id, sample_data_path, plot_flag=False)
        
        eeg_ch_cols = [col for col in result.data.columns if col.startswith('EEG_ch_')]
        eeg_cg_cols = [col for col in result.data.columns if col.startswith('EEG_cg_')]
        
        assert len(eeg_ch_cols) > 0
        assert len(eeg_cg_cols) > 0

    @pytest.mark.integration
    def test_load_eeg_data_creates_ecg_columns(self, sample_dyad_id, sample_data_path):
        """load_eeg_data should create ECG columns."""
        if not os.path.exists(sample_data_path):
            pytest.skip("Sample data not available")
        
        result = dataloader.load_eeg_data(sample_dyad_id, sample_data_path, plot_flag=False)
        
        assert 'ECG_ch' in result.data.columns
        assert 'ECG_cg' in result.data.columns

    @pytest.mark.integration
    def test_load_eeg_data_creates_diode_column(self, sample_dyad_id, sample_data_path):
        """load_eeg_data should create diode column."""
        if not os.path.exists(sample_data_path):
            pytest.skip("Sample data not available")
        
        result = dataloader.load_eeg_data(sample_dyad_id, sample_data_path, plot_flag=False)
        assert 'diode' in result.data.columns

    @pytest.mark.integration
    def test_load_eeg_data_sets_modalities(self, sample_dyad_id, sample_data_path):
        """load_eeg_data should set EEG and ECG modalities."""
        if not os.path.exists(sample_data_path):
            pytest.skip("Sample data not available")
        
        result = dataloader.load_eeg_data(sample_dyad_id, sample_data_path, plot_flag=False)
        
        assert 'EEG' in result.modalities
        assert 'ECG' in result.modalities

    @pytest.mark.integration
    def test_load_eeg_data_detects_events(self, sample_dyad_id, sample_data_path):
        """load_eeg_data should detect experimental events."""
        if not os.path.exists(sample_data_path):
            pytest.skip("Sample data not available")
        
        result = dataloader.load_eeg_data(sample_dyad_id, sample_data_path, plot_flag=False)
        
        assert len(result.events) > 0
        # Check event structure
        for event in result.events:
            assert 'name' in event

    @pytest.mark.integration
    def test_load_eeg_data_sets_sampling_frequency(self, sample_dyad_id, sample_data_path):
        """load_eeg_data should set sampling frequency."""
        if not os.path.exists(sample_data_path):
            pytest.skip("Sample data not available")
        
        result = dataloader.load_eeg_data(sample_dyad_id, sample_data_path, plot_flag=False)
        
        assert result.eeg_fs is not None
        assert result.eeg_fs > 0
        assert result.ecg_fs is not None


class TestDesignEegFilters:
    """Test _design_eeg_filters function."""

    def test_design_filters_returns_tuple(self):
        """_design_eeg_filters should return filter coefficients tuple."""
        fs = 256
        lowcut = 4.0
        highcut = 40.0
        
        result = dataloader._design_eeg_filters(fs, lowcut, highcut)
        
        assert isinstance(result, tuple)
        assert len(result) == 4  # notch, low, high, filter_type

    def test_design_filters_iir_default(self):
        """_design_eeg_filters should use IIR filters by default."""
        fs = 256
        result = dataloader._design_eeg_filters(fs, 4.0, 40.0)
        
        filter_type = result[3]
        assert filter_type == 'iir'

    def test_design_filters_notch_coefficients(self):
        """_design_eeg_filters should return valid notch filter coefficients."""
        fs = 256
        result = dataloader._design_eeg_filters(fs, 4.0, 40.0)
        
        b_notch, a_notch = result[0]
        assert len(b_notch) > 0
        assert len(a_notch) > 0

    def test_design_filters_bandpass_coefficients(self):
        """_design_eeg_filters should return valid bandpass filter coefficients."""
        fs = 256
        result = dataloader._design_eeg_filters(fs, 4.0, 40.0)
        
        b_low, a_low = result[1]
        b_high, a_high = result[2]
        
        assert len(b_low) > 0
        assert len(b_high) > 0


class TestApplyFilters:
    """Test _apply_filters function."""

    def test_apply_filters_modifies_data_in_place(self):
        """_apply_filters should modify raw_eeg_data in place."""
        md = MultimodalData()
        md.eeg_fs = 256
        md.eeg_channel_names_ch = ['Fp1']
        md.eeg_channel_names_cg = []
        md.eeg_channel_mapping = {'Fp1': 0}
        
        # Create test signal with known frequency content
        n_samples = 1000
        t = np.arange(n_samples) / md.eeg_fs
        # Signal with 10 Hz component (should pass) and 60 Hz component (should be attenuated)
        raw_eeg_data = np.zeros((1, n_samples))
        raw_eeg_data[0, :] = np.sin(2 * np.pi * 10 * t) + np.sin(2 * np.pi * 60 * t)
        
        original_data = raw_eeg_data.copy()
        filters = dataloader._design_eeg_filters(md.eeg_fs, 4.0, 40.0)
        
        dataloader._apply_filters(md, filters, raw_eeg_data)
        
        # Data should be modified
        assert not np.array_equal(raw_eeg_data, original_data)


class TestMountEegData:
    """Test _mount_eeg_data function."""

    def test_mount_eeg_data_applies_reference(self):
        """_mount_eeg_data should apply linked-ears reference."""
        md = MultimodalData()
        md.eeg_channel_names_ch = ['Fp1', 'M1', 'M2']
        md.eeg_channel_names_cg = []
        md.eeg_channel_mapping = {'Fp1': 0, 'M1': 1, 'M2': 2}
        
        # Create test data where M1=1, M2=1, so reference = 1
        raw_eeg_data = np.array([
            [10.0, 10.0, 10.0],  # Fp1
            [1.0, 1.0, 1.0],    # M1
            [1.0, 1.0, 1.0]     # M2
        ])
        
        dataloader._mount_eeg_data(md, raw_eeg_data)
        
        # Fp1 should be re-referenced: 10 - 0.5*(1+1) = 9
        np.testing.assert_array_almost_equal(raw_eeg_data[0, :], [9.0, 9.0, 9.0])

    def test_mount_eeg_data_updates_channel_lists(self):
        """_mount_eeg_data should remove M1/M2 from channel lists."""
        md = MultimodalData()
        md.eeg_channel_names_ch = ['Fp1', 'Fp2', 'M1', 'M2']
        md.eeg_channel_names_cg = ['Fp1_cg', 'M1_cg', 'M2_cg']
        md.eeg_channel_mapping = {
            'Fp1': 0, 'Fp2': 1, 'M1': 2, 'M2': 3,
            'Fp1_cg': 4, 'M1_cg': 5, 'M2_cg': 6
        }
        
        raw_eeg_data = np.zeros((7, 100))
        
        dataloader._mount_eeg_data(md, raw_eeg_data)
        
        # M1, M2 should be removed from channel lists
        assert 'M1' not in md.eeg_channel_names_ch
        assert 'M2' not in md.eeg_channel_names_ch
        assert 'M1_cg' not in md.eeg_channel_names_cg
        assert 'M2_cg' not in md.eeg_channel_names_cg


class TestExtractEcgData:
    """Test _extract_ecg_data function."""

    def test_extract_ecg_data_creates_ecg_columns(self):
        """_extract_ecg_data should create ECG columns in DataFrame."""
        md = MultimodalData()
        md.ecg_fs = 256
        md.eeg_channel_mapping = {
            'EKG1': 0, 'EKG2': 1, 'EKG1_cg': 2, 'EKG2_cg': 3
        }
        
        n_samples = 1000
        raw_eeg_data = np.random.randn(4, n_samples) * 0.1
        
        dataloader._extract_ecg_data(md, raw_eeg_data)
        
        assert 'ECG_ch' in md.data.columns
        assert 'ECG_cg' in md.data.columns
        assert len(md.data) == n_samples

    def test_extract_ecg_data_adds_modality(self):
        """_extract_ecg_data should add ECG to modalities."""
        md = MultimodalData()
        md.ecg_fs = 256
        md.modalities = []
        md.eeg_channel_mapping = {
            'EKG1': 0, 'EKG2': 1, 'EKG1_cg': 2, 'EKG2_cg': 3
        }
        
        raw_eeg_data = np.random.randn(4, 1000) * 0.1
        
        dataloader._extract_ecg_data(md, raw_eeg_data)
        
        assert 'ECG' in md.modalities


class TestScanForEvents:
    """Test _scan_for_events function."""

    def test_scan_for_events_returns_events_list(self):
        """_scan_for_events should return a list of events."""
        # Create a simple diode signal with some pulses
        fs = 256
        duration = 10  # seconds
        n_samples = fs * duration
        diode = np.zeros(n_samples)
        
        events, thresholded = dataloader._scan_for_events(diode, fs, plot_flag=False)
        
        assert isinstance(events, list)
        assert isinstance(thresholded, np.ndarray)

    def test_scan_for_events_returns_thresholded_diode(self):
        """_scan_for_events should return thresholded diode signal."""
        fs = 256
        diode = np.random.randn(fs * 10)
        
        events, thresholded = dataloader._scan_for_events(diode, fs, plot_flag=False)
        
        # Thresholded signal should be binary (0 or 1)
        unique_values = np.unique(thresholded)
        assert all(v in [0.0, 1.0] for v in unique_values)


class TestSaveAndLoadData:
    """Test save_to_file and load_output_data functions."""

    def test_save_and_load_roundtrip(self):
        """Data saved with save_to_file should be loadable with load_output_data."""
        md = MultimodalData()
        md.id = "test_dyad"
        md.eeg_fs = 256
        md.modalities = ['EEG', 'ECG']
        md.data = pd.DataFrame({
            'time': [0.0, 0.1, 0.2],
            'EEG_ch_Fp1': [1.0, 2.0, 3.0]
        })
        
        with tempfile.TemporaryDirectory() as tmpdir:
            dataloader.save_to_file(md, tmpdir)
            
            filepath = os.path.join(tmpdir, f"{md.id}.joblib")
            assert os.path.exists(filepath)
            
            loaded = dataloader.load_output_data(filepath)
            
            assert loaded.id == md.id
            assert loaded.eeg_fs == md.eeg_fs
            assert loaded.modalities == md.modalities
            pd.testing.assert_frame_equal(loaded.data, md.data)

    def test_load_output_data_nonexistent_file(self):
        """load_output_data should handle nonexistent files gracefully."""
        result = dataloader.load_output_data("/nonexistent/path/file.joblib")
        assert result is None


class TestGetEegDataMethods:
    """Test get_eeg_data_ch and get_eeg_data_cg after loading."""

    @pytest.mark.integration
    def test_get_eeg_data_ch_returns_2d_array(self):
        """get_eeg_data_ch should return a 2D array with channels x samples."""
        if not os.path.exists("data/eeg/"):
            pytest.skip("Sample data not available")
        
        md = dataloader.load_eeg_data("W_001", "data/eeg/", plot_flag=False)
        
        ch_data = md.get_eeg_data_ch()

        assert ch_data is not None
        assert ch_data.ndim == 2
        # Number of channels should match EEG_ch_ columns
        eeg_ch_cols = [col for col in md.data.columns if col.startswith('EEG_ch_')]
        assert ch_data.shape[0] == len(eeg_ch_cols)
        assert ch_data.shape[1] == len(md.data)

    @pytest.mark.integration
    def test_get_eeg_data_cg_returns_2d_array(self):
        """get_eeg_data_cg should return a 2D array with channels x samples."""
        if not os.path.exists("data/eeg/"):
            pytest.skip("Sample data not available")
        
        md = dataloader.load_eeg_data("W_001", "data/eeg/", plot_flag=False)
        
        cg_data = md.get_eeg_data_cg()

        assert cg_data is not None
        assert cg_data.ndim == 2
        # Number of channels should match EEG_cg_ columns
        eeg_cg_cols = [col for col in md.data.columns if col.startswith('EEG_cg_')]
        assert cg_data.shape[0] == len(eeg_cg_cols)
        assert cg_data.shape[1] == len(md.data)

