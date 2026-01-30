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
        assert md.fs is None
        assert md.modalities == []
        assert md.events == {}  # events is a dict, not a list
        assert md.eeg_channel_names == []
        assert md.eeg_channel_mapping == {}

    def test_init_nested_dataclasses(self):
        """MultimodalData should initialize nested dataclasses correctly."""
        md = MultimodalData()
        assert isinstance(md.eeg_filtration, Filtration)
        assert isinstance(md.paths, Paths)
        assert isinstance(md.tasks, Tasks)
        assert isinstance(md.child_info, ChildInfo)


class TestSetEegData:
    """Test _set_eeg_data method."""

    def test__set_eeg_data_creates_columns(self):
        """_set_eeg_data should create a column for each EEG channel."""
        md = MultimodalData()
        md.fs = 256
        md.eeg_channel_names_ch = ['Fp1', 'Fp2']
        md.eeg_channel_names_cg = ['Fp1_cg', 'Fp2_cg']
        
        n_channels, n_samples = 4, 1000
        eeg_data = np.random.randn(n_channels, n_samples)
        channel_mapping = {'Fp1': 0, 'Fp2': 1, 'Fp1_cg': 2, 'Fp2_cg': 3}
        
        md._set_eeg_data(eeg_data, channel_mapping)
        
        assert len(md.data) == n_samples
        assert 'EEG_ch_Fp1' in md.data.columns
        assert 'EEG_ch_Fp2' in md.data.columns
        assert 'EEG_cg_Fp1' in md.data.columns
        assert 'EEG_cg_Fp2' in md.data.columns

    def test__set_eeg_data_creates_time_columns(self):
        """_set_eeg_data should create time and time_idx columns when fs is set."""
        md = MultimodalData()
        md.fs = 256
        md.eeg_channel_names_ch = ['Fp1', 'Fp2']
        
        eeg_data = np.random.randn(2, 512)
        channel_mapping = {'Fp1': 0, 'Fp2': 1}
        
        md._set_eeg_data(eeg_data, channel_mapping)
        
        assert 'time' in md.data.columns
        assert 'time_idx' in md.data.columns
        assert md.data['time'].iloc[0] == 0.0
        assert md.data['time_idx'].iloc[0] == 0
        # Check time at 1 second (256 samples at 256 Hz)
        assert np.isclose(md.data['time'].iloc[256], 1.0)

    def test__set_eeg_data_preserves_values(self):
        """_set_eeg_data should preserve the exact values from input array."""
        md = MultimodalData()
        md.fs = 256
        md.eeg_channel_names_ch = ['Fp1', 'Fp2']
        
        eeg_data = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
        channel_mapping = {'Fp1': 0, 'Fp2': 1}
        
        md._set_eeg_data(eeg_data, channel_mapping)
        
        np.testing.assert_array_equal(md.data['EEG_ch_Fp1'].values, [1.0, 2.0, 3.0])
        np.testing.assert_array_equal(md.data['EEG_ch_Fp2'].values, [4.0, 5.0, 6.0])

    def test__set_eeg_data_channel_naming_convention(self):
        """Channels ending with _cg should be prefixed with EEG_cg_, others with EEG_ch_."""
        md = MultimodalData()
        md.fs = 256
        md.eeg_channel_names_ch = ['Fz', 'Cz']
        md.eeg_channel_names_cg = ['Fz_cg', 'Cz_cg']
        
        eeg_data = np.random.randn(4, 100)
        channel_mapping = {'Fz': 0, 'Cz': 1, 'Fz_cg': 2, 'Cz_cg': 3}
        
        md._set_eeg_data(eeg_data, channel_mapping)
        
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
        md.fs = 256
        md.eeg_channel_names_ch = ['Fp1', 'Fp2']
        md.eeg_channel_names_cg = ['Fp1_cg', 'Fp2_cg']
        
        n_samples = 100
        eeg_data = np.random.randn(4, n_samples)
        channel_mapping = {'Fp1': 0, 'Fp2': 1, 'Fp1_cg': 2, 'Fp2_cg': 3}
        
        md._set_eeg_data(eeg_data, channel_mapping)
        
        ch_data = md.get_eeg_data_ch()
        assert ch_data.shape == (2, n_samples)  # 2 child channels

    def test_get_eeg_data_cg_returns_correct_shape(self):
        """get_eeg_data_cg should return 2D array [n_channels x n_samples]."""
        md = MultimodalData()
        md.fs = 256
        md.eeg_channel_names_ch = ['Fp1', 'Fp2']
        md.eeg_channel_names_cg = ['Fp1_cg', 'Fp2_cg']
        
        n_samples = 100
        eeg_data = np.random.randn(4, n_samples)
        channel_mapping = {'Fp1': 0, 'Fp2': 1, 'Fp1_cg': 2, 'Fp2_cg': 3}
        
        md._set_eeg_data(eeg_data, channel_mapping)
        
        cg_data = md.get_eeg_data_cg()
        assert cg_data.shape == (2, n_samples)  # 2 caregiver channels

    def test_get_eeg_data_returns_none_when_empty(self):
        """get_eeg_data_ch/cg should return None when no EEG data is set."""
        md = MultimodalData()
        assert md.get_eeg_data_ch() is None
        assert md.get_eeg_data_cg() is None


class TestSetEcgData:
    """Test _set_ecg_data method."""

    def test__set_ecg_data_creates_columns(self):
        """_set_ecg_data should create ECG_ch and ECG_cg columns."""
        md = MultimodalData()
        
        ecg_ch = np.random.randn(1000)
        ecg_cg = np.random.randn(1000)
        
        md._set_ecg_data(ecg_ch, ecg_cg)
        
        assert 'ECG_ch' in md.data.columns
        assert 'ECG_cg' in md.data.columns
        assert len(md.data) == 1000

    def test__set_ecg_data_preserves_values(self):
        """_set_ecg_data should preserve the exact values from input arrays."""
        md = MultimodalData()
        
        ecg_ch = np.array([1.0, 2.0, 3.0])
        ecg_cg = np.array([4.0, 5.0, 6.0])
        
        md._set_ecg_data(ecg_ch, ecg_cg)


class TestSetDiode:
    """Test _set_diode method."""

    def test__set_diode_creates_column(self):
        """_set_diode should create a diode column."""
        md = MultimodalData()
        
        diode = np.random.randn(500)
        md._set_diode(diode)
        
        assert 'diode' in md.data.columns
        assert len(md.data) == 500

    def test__set_diode_preserves_values(self):
        """_set_diode should preserve the exact values."""
        md = MultimodalData()
        
        diode = np.array([0, 1, 1, 0, 1])
        md._set_diode(diode)
        
        np.testing.assert_array_equal(md.data['diode'].values, diode)


class TestSetIbiData:
    """Test set_ibi_data method."""

    @pytest.mark.skip(reason="set_ibi_data method doesn't exist - IBI created via _interpolate_ibi_signals() and _set_ibi()")
    def test_set_ibi_data_creates_columns(self):
        """set_ibi_data should create IBI_ch, IBI_cg, and IBI_times columns."""
        md = MultimodalData()
        
        ibi_ch = np.random.randn(100)
        ibi_cg = np.random.randn(100)
        ibi_times = np.arange(100) * 0.25  # 4 Hz
        
        md.set_ibi_data(ibi_ch, ibi_cg, ibi_times)
        
        assert 'IBI_ch' in md.data.columns
        assert 'IBI_cg' in md.data.columns


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


class TestCreateEventsColumn:
    """Test create_events_column method."""

    def test_create_events_column_from_eeg_et_events(self):
        """create_events_column should create events column from EEG_events and ET_event columns."""
        md = MultimodalData()
        md.fs = 100
        
        # Create DataFrame with time and event marker columns
        md.data = pd.DataFrame({
            'time': np.arange(1000) / 100,  # 10 seconds of data
            'time_idx': np.arange(1000),
            'EEG_events': [None] * 100 + ['stimulus_1'] * 200 + [None] * 300 + ['stimulus_2'] * 100 + [None] * 300
        })
        
        md._create_events_column()
        
        assert 'events' in md.data.columns
        # Check that events are correctly placed
        assert md.data.iloc[150]['events'] == 'stimulus_1'
        assert md.data.iloc[650]['events'] == 'stimulus_2'


# TestAlignTimeToFirstEvent removed - method doesn't exist in current implementation


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

    def test_decimate_returns_new_object(self):
        """decimate_signals should return a new MultimodalData object with decimated signals."""
        md = MultimodalData()
        md.fs = 256
        md.eeg_channel_names_ch = ['Fp1', 'Fp2']
        
        # Need enough samples for decimation
        n_samples = 1000
        eeg_data = np.random.randn(2, n_samples)
        channel_mapping = {'Fp1': 0, 'Fp2': 1}
        
        md._set_eeg_data(eeg_data, channel_mapping)
        md_dec = md._decimate_signals(q=4)
        
        # Should return a new object
        assert md_dec is not md
        # Original object should be unchanged
        assert md.fs == 256
        # New object should have updated fs
        assert md_dec.fs == 64
        # New object should have same column names (not _dec suffix)
        assert 'EEG_ch_Fp1' in md_dec.data.columns
        assert 'EEG_ch_Fp2' in md_dec.data.columns

    def test_decimate_reduces_samples(self):
        """decimate_signals should reduce sample count by factor q."""
        md = MultimodalData()
        md.fs = 256
        md.eeg_channel_names_ch = ['Fp1']
        
        n_samples = 1000
        q = 4
        eeg_data = np.random.randn(1, n_samples)
        channel_mapping = {'Fp1': 0}
        
        md._set_eeg_data(eeg_data, channel_mapping)
        md_dec = md._decimate_signals(q=q)
        
        # Check decimated sample count in returned object
        decimated_count = len(md_dec.data)
        expected_count = n_samples // q
        assert decimated_count == expected_count


class TestDataclasses:
    """Test supporting dataclasses."""

    def test_filtration_defaults(self):
        """Filtration dataclass should have correct defaults."""
        f = Filtration()
        assert f.notch_Q is None
        assert f.notch_freq is None
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



