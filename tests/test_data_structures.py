"""
Core tests for MultimodalData data structure.

Tests the DataFrame-based storage of multimodal signals including EEG, ECG, IBI, and diode data.
"""
import pytest
import numpy as np
import pandas as pd

from src.data_structures import (
    MultimodalData, 
    Filtration, 
    Paths, 
    Tasks, 
    ChildInfo
)


class TestMultimodalDataInit:
    """Test MultimodalData initialization."""

    def test_init_creates_empty_dataframe(self):
        """MultimodalData should initialize with an empty DataFrame."""
        md = MultimodalData()
        assert isinstance(md.data, pd.DataFrame)
        assert len(md.data) == 0

    def test_init_default_values(self):
        """MultimodalData should have correct default values."""
        md = MultimodalData()
        assert md.id is None
        assert md.eeg_fs is None
        assert md.ecg_fs is None
        assert md.ibi_fs is None
        assert md.modalities == []
        assert md.events == []
        assert md.eeg_channel_names == []
        assert md.eeg_channel_mapping == {}

    def test_init_nested_dataclasses(self):
        """MultimodalData should initialize nested dataclasses correctly."""
        md = MultimodalData()
        assert isinstance(md.filtration, Filtration)
        assert isinstance(md.paths, Paths)
        assert isinstance(md.tasks, Tasks)
        assert isinstance(md.child_info, ChildInfo)


class TestSetEegData:
    """Test set_eeg_data method."""

    def test_set_eeg_data_creates_columns(self):
        """set_eeg_data should create a column for each EEG channel."""
        md = MultimodalData()
        md.eeg_fs = 256
        
        n_channels, n_samples = 4, 1000
        eeg_data = np.random.randn(n_channels, n_samples)
        channel_mapping = {'Fp1': 0, 'Fp2': 1, 'Fp1_cg': 2, 'Fp2_cg': 3}
        
        md.set_eeg_data(eeg_data, channel_mapping)
        
        assert len(md.data) == n_samples
        assert 'EEG_ch_Fp1' in md.data.columns
        assert 'EEG_ch_Fp2' in md.data.columns
        assert 'EEG_cg_Fp1' in md.data.columns
        assert 'EEG_cg_Fp2' in md.data.columns

    def test_set_eeg_data_creates_time_columns(self):
        """set_eeg_data should create time and time_idx columns when eeg_fs is set."""
        md = MultimodalData()
        md.eeg_fs = 256
        
        eeg_data = np.random.randn(2, 512)
        channel_mapping = {'Fp1': 0, 'Fp2': 1}
        
        md.set_eeg_data(eeg_data, channel_mapping)
        
        assert 'time' in md.data.columns
        assert 'time_idx' in md.data.columns
        assert md.data['time'].iloc[0] == 0.0
        assert md.data['time_idx'].iloc[0] == 0
        # Check time at 1 second (256 samples at 256 Hz)
        assert np.isclose(md.data['time'].iloc[256], 1.0)

    def test_set_eeg_data_preserves_values(self):
        """set_eeg_data should preserve the exact values from input array."""
        md = MultimodalData()
        md.eeg_fs = 256
        
        eeg_data = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
        channel_mapping = {'Fp1': 0, 'Fp2': 1}
        
        md.set_eeg_data(eeg_data, channel_mapping)
        
        np.testing.assert_array_equal(md.data['EEG_ch_Fp1'].values, [1.0, 2.0, 3.0])
        np.testing.assert_array_equal(md.data['EEG_ch_Fp2'].values, [4.0, 5.0, 6.0])

    def test_set_eeg_data_channel_naming_convention(self):
        """Channels ending with _cg should be prefixed with EEG_cg_, others with EEG_ch_."""
        md = MultimodalData()
        md.eeg_fs = 256
        
        eeg_data = np.random.randn(4, 100)
        channel_mapping = {'Fz': 0, 'Cz': 1, 'Fz_cg': 2, 'Cz_cg': 3}
        
        md.set_eeg_data(eeg_data, channel_mapping)
        
        # Child channels
        assert 'EEG_ch_Fz' in md.data.columns
        assert 'EEG_ch_Cz' in md.data.columns
        # Caregiver channels  
        assert 'EEG_cg_Fz' in md.data.columns
        assert 'EEG_cg_Cz' in md.data.columns


class TestGetEegData:
    """Test get_eeg_data_ch and get_eeg_data_cg methods."""

    def test_get_eeg_data_ch_returns_correct_shape(self):
        """get_eeg_data_ch should return 2D array [n_channels x n_samples]."""
        md = MultimodalData()
        md.eeg_fs = 256
        
        n_samples = 100
        eeg_data = np.random.randn(4, n_samples)
        channel_mapping = {'Fp1': 0, 'Fp2': 1, 'Fp1_cg': 2, 'Fp2_cg': 3}
        
        md.set_eeg_data(eeg_data, channel_mapping)
        
        ch_data = md.get_eeg_data_ch()
        assert ch_data.shape == (2, n_samples)  # 2 child channels

    def test_get_eeg_data_cg_returns_correct_shape(self):
        """get_eeg_data_cg should return 2D array [n_channels x n_samples]."""
        md = MultimodalData()
        md.eeg_fs = 256
        
        n_samples = 100
        eeg_data = np.random.randn(4, n_samples)
        channel_mapping = {'Fp1': 0, 'Fp2': 1, 'Fp1_cg': 2, 'Fp2_cg': 3}
        
        md.set_eeg_data(eeg_data, channel_mapping)
        
        cg_data = md.get_eeg_data_cg()
        assert cg_data.shape == (2, n_samples)  # 2 caregiver channels

    def test_get_eeg_data_returns_none_when_empty(self):
        """get_eeg_data_ch/cg should return None when no EEG data is set."""
        md = MultimodalData()
        assert md.get_eeg_data_ch() is None
        assert md.get_eeg_data_cg() is None


class TestSetEcgData:
    """Test set_ecg_data method."""

    def test_set_ecg_data_creates_columns(self):
        """set_ecg_data should create ECG_ch and ECG_cg columns."""
        md = MultimodalData()
        
        ecg_ch = np.random.randn(1000)
        ecg_cg = np.random.randn(1000)
        
        md.set_ecg_data(ecg_ch, ecg_cg)
        
        assert 'ECG_ch' in md.data.columns
        assert 'ECG_cg' in md.data.columns
        assert len(md.data) == 1000

    def test_set_ecg_data_preserves_values(self):
        """set_ecg_data should preserve the exact values from input arrays."""
        md = MultimodalData()
        
        ecg_ch = np.array([1.0, 2.0, 3.0])
        ecg_cg = np.array([4.0, 5.0, 6.0])
        
        md.set_ecg_data(ecg_ch, ecg_cg)
        
        np.testing.assert_array_equal(md.data['ECG_ch'].values, ecg_ch)
        np.testing.assert_array_equal(md.data['ECG_cg'].values, ecg_cg)


class TestSetDiode:
    """Test set_diode method."""

    def test_set_diode_creates_column(self):
        """set_diode should create a diode column."""
        md = MultimodalData()
        
        diode = np.random.randn(500)
        md.set_diode(diode)
        
        assert 'diode' in md.data.columns
        assert len(md.data) == 500

    def test_set_diode_preserves_values(self):
        """set_diode should preserve the exact values."""
        md = MultimodalData()
        
        diode = np.array([0, 1, 1, 0, 1])
        md.set_diode(diode)
        
        np.testing.assert_array_equal(md.data['diode'].values, diode)


class TestSetIbiData:
    """Test set_ibi_data method."""

    def test_set_ibi_data_creates_columns(self):
        """set_ibi_data should create IBI_ch, IBI_cg, and IBI_times columns."""
        md = MultimodalData()
        
        ibi_ch = np.random.randn(100)
        ibi_cg = np.random.randn(100)
        ibi_times = np.arange(100) * 0.25  # 4 Hz
        
        md.set_ibi_data(ibi_ch, ibi_cg, ibi_times)
        
        assert 'IBI_ch' in md.data.columns
        assert 'IBI_cg' in md.data.columns
        assert 'IBI_times' in md.data.columns


class TestEnsureDataLength:
    """Test _ensure_data_length method."""

    def test_ensure_data_length_creates_dataframe(self):
        """_ensure_data_length should create DataFrame if empty."""
        md = MultimodalData()
        md._ensure_data_length(100)
        assert len(md.data) == 100

    def test_ensure_data_length_extends_dataframe(self):
        """_ensure_data_length should extend DataFrame if too short."""
        md = MultimodalData()
        md.data = pd.DataFrame({'col': [1, 2, 3]})
        md._ensure_data_length(10)
        assert len(md.data) == 10

    def test_ensure_data_length_preserves_existing_data(self):
        """_ensure_data_length should preserve existing data when extending."""
        md = MultimodalData()
        md.data = pd.DataFrame({'col': [1, 2, 3]})
        md._ensure_data_length(5)
        assert md.data['col'].iloc[0] == 1
        assert md.data['col'].iloc[2] == 3


class TestSetEventsColumn:
    """Test set_events_column method."""

    def test_set_events_column_creates_events(self):
        """set_events_column should populate events based on timing."""
        md = MultimodalData()
        md.eeg_fs = 100
        
        # Create DataFrame with time column
        md.data = pd.DataFrame({
            'time': np.arange(1000) / 100,  # 10 seconds of data
            'time_idx': np.arange(1000)
        })
        
        events = [
            {'name': 'stimulus_1', 'start': 1.0, 'duration': 2.0},
            {'name': 'stimulus_2', 'start': 5.0, 'duration': 1.0}
        ]
        
        md.set_events_column(events)
        
        assert 'events' in md.data.columns
        # Check that events are correctly placed
        assert md.data[md.data['time'] == 1.5]['events'].iloc[0] == 'stimulus_1'
        assert md.data[md.data['time'] == 5.5]['events'].iloc[0] == 'stimulus_2'


class TestAlignTimeToFirstEvent:
    """Test align_time_to_first_event method."""

    def test_align_time_shifts_correctly(self):
        """align_time_to_first_event should shift time so first event is at t=0."""
        md = MultimodalData()
        md.eeg_fs = 100
        
        md.data = pd.DataFrame({
            'time': np.arange(1000) / 100,
            'time_idx': np.arange(1000),
            'events': [None] * 100 + ['event1'] * 100 + [None] * 800
        })
        
        original_first_event_time = 1.0  # Event starts at t=1.0
        
        md.align_time_to_first_event()
        
        # First event should now be at t=0
        first_event_time = md.data[md.data['events'] == 'event1']['time'].min()
        assert np.isclose(first_event_time, 0.0)


class TestEegChannelNamesAll:
    """Test eeg_channel_names_all method."""

    def test_combines_child_and_caregiver_channels(self):
        """eeg_channel_names_all should return combined list of all channel names."""
        md = MultimodalData()
        md.eeg_channel_names_ch = ['Fp1', 'Fp2', 'Fz']
        md.eeg_channel_names_cg = ['Fp1_cg', 'Fp2_cg', 'Fz_cg']
        
        all_names = md.eeg_channel_names_all()
        
        assert all_names == ['Fp1', 'Fp2', 'Fz', 'Fp1_cg', 'Fp2_cg', 'Fz_cg']


class TestDecimateSignals:
    """Test decimate_signals method."""

    def test_decimate_creates_new_columns(self):
        """decimate_signals should create new columns with _dec suffix."""
        md = MultimodalData()
        md.eeg_fs = 256
        
        # Need enough samples for decimation
        n_samples = 1000
        eeg_data = np.random.randn(2, n_samples)
        channel_mapping = {'Fp1': 0, 'Fp2': 1}
        
        md.set_eeg_data(eeg_data, channel_mapping)
        md.decimate_signals(q=4)
        
        assert 'EEG_ch_Fp1_dec' in md.data.columns
        assert 'EEG_ch_Fp2_dec' in md.data.columns

    def test_decimate_reduces_samples(self):
        """decimate_signals should reduce sample count by factor q."""
        md = MultimodalData()
        md.eeg_fs = 256
        
        n_samples = 1000
        q = 4
        eeg_data = np.random.randn(1, n_samples)
        channel_mapping = {'Fp1': 0}
        
        md.set_eeg_data(eeg_data, channel_mapping)
        md.decimate_signals(q=q)
        
        # Count non-NaN values in decimated column
        decimated_count = md.data['EEG_ch_Fp1_dec'].notna().sum()
        expected_count = n_samples // q
        assert decimated_count == expected_count


class TestDataclasses:
    """Test supporting dataclasses."""

    def test_filtration_defaults(self):
        """Filtration dataclass should have correct defaults."""
        f = Filtration()
        assert f.notch is None
        assert f.low_pass is None
        assert f.high_pass is None
        assert f.type is None

    def test_paths_defaults(self):
        """Paths dataclass should have correct defaults."""
        p = Paths()
        assert p.eeg_directory is None
        assert p.et_directory is None
        assert p.hrv_directory is None
        assert p.output_dir is None

    def test_child_info_defaults(self):
        """ChildInfo dataclass should have correct defaults."""
        ci = ChildInfo()
        assert ci.birth_date is None
        assert ci.age_years is None
        assert ci.sex is None

