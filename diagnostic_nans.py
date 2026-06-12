import numpy as np
import xarray as xr
from scipy.signal import welch
from scipy.interpolate import interp1d
import warnings

from src.passive_io_helpers import discover_cleaned_role_files

# Suppress potential alignment warnings
warnings.filterwarnings("ignore", category=RuntimeWarning)

ROOT = "/Users/admin/Documents/Hoza/PROJEKTY/SYNCC_IN_LOCAL_HOME/DATA_film_cleaned/EEG"
ROLE = 'ch'
TARGET_CH = 'Fz'
TARGET_EVENTS = ['Peppa', 'Brave', 'Incredibles']
FMAX = 20.0

def get_psd(data, fs):
    n_samples = len(data)
    if n_samples < 2: return None, None
    nperseg = min(int(round(2 * fs)), n_samples)
    if nperseg < 2: return None, None
    noverlap = nperseg // 2
    f, p = welch(data, fs=fs, nperseg=nperseg, noverlap=noverlap, detrend='constant', scaling='density')
    mask = f <= FMAX
    return f[mask], p[mask]

psd_list = []

role_files = discover_cleaned_role_files(ROOT, TARGET_EVENTS, signal_type="EEG")
subjects_scanned = len({dyad_id for _, dyad_id, _, _ in role_files})

for file_path, dyad_id, role_found, event_found in role_files:
    if role_found != ROLE:
        continue

    path = str(file_path)
    try:
        # Try default engine first, fallback to netcdf4 if available
        ds = xr.open_dataset(path)

        # Based on standard MNE-derived .nc exports
        chan_names = []
        if 'channel' in ds.coords: chan_names = ds.channel.values
        elif 'chan' in ds.coords: chan_names = ds.chan.values

        # Case-insensitive Fz search
        fz_idx = -1
        for i, c in enumerate(chan_names):
            if str(c).upper() == 'FZ':
                fz_idx = i
                break

        if fz_idx == -1:
            ds.close()
            continue

        # Extract data - often in a variable named 'data' or the only 2D variable
        # If ds has 'data', use it
        if 'data' in ds.data_vars:
            # Dimensions are likely (channel, time)
            trace = ds.data.values[fz_idx, :]
        else:
            # Just grab the first data variable
            var_name = list(ds.data_vars.keys())[0]
            trace = ds[var_name].values[fz_idx, :]

        # Sampling frequency
        fs = 500.0 # Default fallback
        if 'time' in ds.coords:
            t = ds.time.values
            if len(t) > 1:
                dt = t[1] - t[0]
                if isinstance(dt, np.timedelta64):
                    fs = 1e9 / dt.astype('float64')
                else:
                    fs = 1.0 / dt

        f, p = get_psd(trace, fs)
        if f is not None:
            psd_list.append((f, p))
        ds.close()
    except Exception:
        continue

if not psd_list:
    print(f"No Fz traces found. Scanned {subjects_scanned} subjects.")
    exit()

freq_grids = [x[0] for x in psd_list]
unique_lengths = sorted(list(set(len(g) for g in freq_grids)))
ref_idx = np.argmax([len(g) for g in freq_grids])
ref_f = freq_grids[ref_idx]

matrix = np.full((len(psd_list), len(ref_f)), np.nan)
for i, (f, p) in enumerate(psd_list):
    if len(f) == len(ref_f) and np.allclose(f, ref_f):
        matrix[i, :] = p
    else:
        f_min, f_max = max(f[0], ref_f[0]), min(f[-1], ref_f[-1])
        mask = (ref_f >= f_min) & (ref_f <= f_max)
        if np.any(mask):
            itp = interp1d(f, p, kind='linear', bounds_error=False, fill_value=np.nan)
            matrix[i, mask] = itp(ref_f[mask])

mean_psd = np.nanmean(matrix, axis=0)
n_eff = np.sum(~np.isnan(matrix), axis=0)
std_psd = np.nanstd(matrix, axis=0, ddof=1)
sem_psd = np.where(n_eff > 1, std_psd / np.sqrt(n_eff), np.nan)

print(f"Files scanned: {len(psd_list)}")
print(f"Fz traces count: {len(psd_list)}")
print(f"Ref bins count: {len(ref_f)}")
print(f"NaNs in matrix: {np.isnan(matrix).sum()}")
print(f"NaNs in mean: {np.isnan(mean_psd).sum()}")
print(f"NaNs in std: {np.isnan(std_psd).sum()}")
print(f"NaNs in sem: {np.isnan(sem_psd).sum()}")
print(f"Bins with n_eff==0: {np.sum(n_eff == 0)}")
print(f"Bins with n_eff==1: {np.sum(n_eff == 1)}")
print(f"Min n_eff: {np.min(n_eff)}")
print(f"Median n_eff: {np.median(n_eff)}")

nan_sem_indices = np.where(np.isnan(sem_psd))[0]
if len(nan_sem_indices) > 0:
    print(f"First up to 12 freqs with NaN SEM: {ref_f[nan_sem_indices[:12]]}")

if len(unique_lengths) > 1:
    print(f"Warning: Fz traces had different freq grid lengths: {unique_lengths}")
else:
    print("All Fz traces had the same freq grid length.")
