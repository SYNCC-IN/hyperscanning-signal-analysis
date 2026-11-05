"""Tests for the Heidelberg import: BI-Vital HRV on the EEG grid, markers, naming."""
import numpy as np
import pandas as pd
import pytest

from src.clocksync import SyncFit
from src.heidelberg_loader import (
    COMMON_CHANNELS, base_channel_name, events_from_markers, parse_dyad_id,
)
from src.hrv_sync import HrvParams, compute_hrv_on_grid, ibi_from_peaks, rmssd_sliding

FS = 200.0


def _synthetic_table(duration_s=120.0, dropout=(60.0, 70.0), seed=0):
    """BI-Vital-like table: R-waves with alternating IBI 780/820 ms (RMSSD = 40 ms)."""
    rng = np.random.default_rng(seed)
    t = np.arange(0, duration_s, 1 / FS)
    beats, ibis, now, k = [], [], 1.0, 0
    while now < duration_s - 1:
        beats.append(now)
        ibi = 0.78 if k % 2 == 0 else 0.82
        ibis.append(ibi)
        now += ibi
        k += 1
    ecg = np.zeros_like(t)
    for beat in beats:
        ecg += 1000 * np.exp(-0.5 * ((t - beat) / 0.012) ** 2)
    ecg += rng.normal(0, 5, len(t))
    table = pd.DataFrame({'t': t, 'ecg': ecg})
    if dropout:
        table = table[(table['t'] < dropout[0]) | (table['t'] >= dropout[1])]
    return table.reset_index(drop=True)


@pytest.fixture(scope='module')
def hrv():
    fit = SyncFit(slope=0.0, intercept_s=0.0, used_mask=np.array([True]), tracks=pd.DataFrame())
    grid = np.arange(0, 120, 1 / 128)
    return grid, compute_hrv_on_grid(_synthetic_table(), fit, 'cg', grid, HrvParams())


def test_ibi_and_rmssd_values(hrv):
    grid, out = hrv
    ok = (grid > 10) & (grid < 40)                      # well inside the first continuous stretch
    assert np.nanmean(out['IBI'][ok]) == pytest.approx(800, abs=5)
    assert np.nanmean(out['RMSSD'][(grid > 5) & (grid < 25)]) == pytest.approx(40, abs=4)


def test_dropout_is_not_bridged(hrv):
    grid, out = hrv
    gap = (grid > 61.5) & (grid < 69)
    for key in ('ECG', 'IBI'):
        assert np.isnan(out[key][gap]).all(), key
    # RMSSD windows start at the grid point: no value while that start lies inside the dropout
    assert np.isnan(out['RMSSD'][(grid > 61.5) & (grid < 66.5)]).all()
    assert not np.isnan(out['IBI'][(grid > 20) & (grid < 50)]).any()


def test_info(hrv):
    _, out = hrv
    assert out['info']['mean_hr_bpm'] == pytest.approx(75, abs=2)
    assert 0.8 < out['info']['valid_ecg_fraction'] < 1.0


def test_ibi_rejects_out_of_range_and_dropouts():
    peaks = np.array([0.0, 0.8, 1.6, 4.0, 4.8])       # 2.4 s IBI is out of range
    runs = np.array([0, 0, 0, 0, 1])                  # last peak starts a new run
    ibi = ibi_from_peaks(peaks, runs, (400.0, 1500.0))
    assert np.isnan(ibi[0]) and np.isnan(ibi[3]) and np.isnan(ibi[4])
    assert ibi[1] == pytest.approx(800) and ibi[2] == pytest.approx(800)


def test_rmssd_requires_minimum_differences():
    peaks = np.cumsum(np.r_[0.0, np.tile([0.78, 0.82], 30)])
    ibi = np.r_[np.nan, np.diff(peaks) * 1000]
    grid = np.array([0.0, 40.0])
    out = rmssd_sliding(peaks, ibi, grid, window_s=30.0, min_diffs=10)
    assert out[0] == pytest.approx(40, abs=0.5)
    assert np.isnan(out[1])                            # window beyond the data


def test_events_from_markers():
    markers = [('S1', 1000), ('S10', 61000), ('S4', 100000), ('S2', 70000)]
    events = events_from_markers(markers, raw_fs_hz=1000)
    assert events['Incredibles'] == {'name': 'Incredibles', 'start': 1.0, 'duration': 60.0}
    with pytest.warns(UserWarning):
        events = events_from_markers(markers, raw_fs_hz=1000)
    assert events['Talk_1']['duration'] == 180.0       # no S40: nominal duration
    assert 'Peppa' not in events


def test_names():
    assert parse_dyad_id('H032') == 'H_032' and parse_dyad_id('H_32') == 'H_032'
    assert base_channel_name('FP2_C', '_C') == 'Fp2'
    assert base_channel_name('TP10_P', '_P') == 'TP10'
    assert len(COMMON_CHANNELS) == 19


def test_discover_heidelberg_dyads(tmp_path):
    from src.heidelberg_loader import discover_heidelberg_dyads

    eeg_dir, ecg_dir = tmp_path / 'eeg', tmp_path / 'ecg'
    eeg_dir.mkdir()
    ecg_dir.mkdir()

    def touch(folder, *names):
        for name in names:
            (folder / name).write_text('')

    touch(eeg_dir, *[f'synccin_LAKA36_46{ext}' for ext in ('.vhdr', '.eeg', '.vmrk')])   # complete (46)
    touch(ecg_dir, 'LAKA36_46-child.csv', 'LAKA36_46-adult.csv')
    touch(eeg_dir, *[f'synccin_MAAI28_7{ext}' for ext in ('.vhdr', '.eeg', '.vmrk')])    # no adult CSV (7)
    touch(ecg_dir, 'MAAI28_7-child.csv')
    touch(eeg_dir, 'synccin_XXXX01_9.vhdr', 'synccin_XXXX01_9.vmrk')                     # .eeg missing (9)
    touch(ecg_dir, 'XXXX01_9-child.csv', 'XXXX01_9-adult.csv')
    touch(eeg_dir, *[f'synccin_AAAA01_12{ext}' for ext in ('.vhdr', '.eeg', '.vmrk')])   # ambiguous (12)
    touch(eeg_dir, *[f'synccin_BBBB02_12{ext}' for ext in ('.vhdr', '.eeg', '.vmrk')])
    touch(ecg_dir, 'AAAA01_12-child.csv', 'AAAA01_12-adult.csv')
    touch(ecg_dir, 'LAKA36_50-child.csv', 'LAKA36_50-adult.csv')                         # no EEG (50)
    touch(eeg_dir, 'notes.txt', 'synccin_junk.vhdr')                                     # ignored
    touch(eeg_dir, *[f'synccin_JEAN90_21{ext}' for ext in ('.vhdr', '.eeg', '.vmrk')])   # '_' separator (21)
    touch(ecg_dir, 'JEAN90_21_adult.csv', 'JEAN90_21_child.csv')
    touch(eeg_dir, *[f'synccin_LITI65_26{ext}' for ext in ('.vhdr', '.eeg', '.vmrk')])   # mixed separators (26)
    touch(ecg_dir, 'LITI65_26-child.csv', 'LITI65_26_adult.csv')
    touch(eeg_dir, *[f'synccin_LAEL07_25{ext}' for ext in ('.vhdr', '.eeg', '.vmrk')])   # split into parts (25)
    touch(ecg_dir, 'LAEL07_25-adult-01.csv', 'LAEL07_25-adult-02.csv',
          'LAEL07_25-child-01.csv', 'LAEL07_25-child-02.csv')

    complete, problems = discover_heidelberg_dyads(eeg_dir, ecg_dir)
    assert list(complete) == ['H021', 'H026', 'H046']
    assert complete['H026']['bivital']['adult'].name == 'LITI65_26_adult.csv'
    assert complete['H026']['bivital']['child'].name == 'LITI65_26-child.csv'
    assert 'split into parts (01, 02)' in problems[25]
    dyad = complete['H046']
    assert dyad['number'] == 46 and dyad['eeg_stem'].name == 'synccin_LAKA36_46'
    assert dyad['bivital']['child'].name == 'LAKA36_46-child.csv'
    assert dyad['bivital']['adult'].name == 'LAKA36_46-adult.csv'
    assert 'adult' in problems[7]
    assert '.eeg' in problems[9]
    assert 'several EEG' in problems[12]
    assert 'no EEG' in problems[50]
    assert parse_dyad_id('H046') == 'H_046'


OLD_FORMAT_CSV = """BI-Vital 0B33, STM_FW 2.0-9-g4033137, NRF_FW 2.0, Config #3263602583, Start Time 2026-03-18 15:49:07, Uptime 00:14:14, Global Update intervall 5 ms
Updatinterval [ms], 5, 5, 10, 10, 10, 10, 10, 10, 10, 10, 10
Time [s], RR-interval [ms], ECG [Count], Acceleration_x [m/s^2], Acceleration_y [m/s^2], Acceleration_z [m/s^2], Angular-velocity_x [Deg/s], Angular-velocity_y [Deg/s], Angular-velocity_z [Deg/s], Magnetic-field_x [uT], Magnetic-field_y [uT], Magnetic-field_z [uT]
0.000,    0, 1616,   0.46,  -2.43,  10.72,    -4.75,    50.56,     4.69,   -311,   -701,   -725
0.005,     , 1641,       ,       ,       ,         ,         ,         ,       ,       ,
0.010,     , 1655,       ,       ,       ,         ,         ,         ,       ,       ,
0.055,     , 1519,  -3.13,  -3.21,   4.65,   -30.12,    14.25,     5.56,   -245,   -683,   -719
"""
NEW_FORMAT_CSV = """BI-Vital B79F, STM_FW 4.0-5-gda6196d, NRF_FW 3.0-2-g578473b, Config 0x5A415AB3, Start Time 2026-08-07 15:28:37.066, Uptime 00:05:08.376, Global Update intervall 5 ms, Drift Correction -30 ms
Updatinterval [ms], 5, 5, 5, 5, 1280, 100, 100, 100
Time [s], Heartrate [bpm], RR-interval [ms], R-Peak [ms], ECG [Count], ECG-Quality [%], Acceleration_x [m/s^2], Acceleration_y [m/s^2], Acceleration_z [m/s^2]
0.14,        ,    ,    , 3864,     ,  0.05, -0.31,  0.39
0.145,  71.0, 845, 845, 3733, 100,      ,      ,
0.15,        ,    ,    , 3559,     ,      ,      ,
"""


@pytest.mark.parametrize('text, start, rows, has_hr', [
    (OLD_FORMAT_CSV, '2026-03-18 15:49:07', 4, False),
    (NEW_FORMAT_CSV, '2026-08-07 15:28:37.066', 3, True),
])
def test_read_bivital_formats(tmp_path, text, start, rows, has_hr):
    from src.load import BIVITAL_COLUMNS, read_bivital

    path = tmp_path / 'device.csv'
    path.write_text(text)
    header, table = read_bivital(path)
    assert header['start'] == pd.Timestamp(start)
    assert list(table.columns) == BIVITAL_COLUMNS and len(table) == rows
    assert table['ecg'].notna().all()
    assert table['hr'].notna().any() == has_hr
    assert table['t'].iloc[1] == pytest.approx(0.005 if rows == 4 else 0.145)
    assert table['ax'].notna().sum() == 2 if rows == 4 else table['ax'].notna().sum() == 1


def test_read_bivital_old_format_values(tmp_path):
    from src.load import read_bivital

    path = tmp_path / 'device.csv'
    path.write_text(OLD_FORMAT_CSV)
    _, table = read_bivital(path)
    assert table['ecg'].tolist() == [1616, 1641, 1655, 1519]
    assert table['ax'].iloc[0] == pytest.approx(0.46) and table['az'].iloc[3] == pytest.approx(4.65)
    assert table['rr'].iloc[0] == 0 and np.isnan(table['rr'].iloc[1])
    assert table[['rpk', 'q']].isna().all().all()


def test_read_bivital_requires_ecg(tmp_path):
    from src.load import read_bivital

    path = tmp_path / 'device.csv'
    path.write_text('BI-Vital X, Start Time 2026-03-18 15:49:07\nUpdate\nTime [s], RR-interval [ms]\n0.0, 5\n')
    with pytest.raises(ValueError, match='ecg'):
        read_bivital(path)


def test_interpolate_to_grid_all_nan_values():
    """A signal that is absent in the recording (all NaN) gives an all-NaN column instead of an error."""
    import numpy as np
    from src.synced_export import interpolate_to_grid
    grid = np.arange(0, 5, 0.5)
    out = interpolate_to_grid(grid, np.arange(10.0), np.full(10, np.nan), 1.0)
    assert out.shape == grid.shape and np.isnan(out).all()
