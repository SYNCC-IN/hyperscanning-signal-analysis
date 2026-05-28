import os
import re
import numpy as np
import xarray as xr
from scipy.signal import welch
from scipy.interpolate import interp1d

base_folder = '../../SYNCC_IN_LOCAL_HOME/DATA_film_cleaned/EEG'
TARGET_EVENTS = ['Peppa', 'Brave', 'Incredibles']
ROLE = 'ch'
ELECTRODE = 'Fz'
FMAX = 20.0

psd_list = []
freqs_list = []

dyad_dirs = [d for d in os.listdir(base_folder) if os.path.isdir(os.path.join(base_folder, d))]
pattern = re.compile(rf'^(W_\d+)_EEG_{ROLE}_(.+)_cleaned\.nc$')

for dyad_dir in dyad_dirs:
    dir_path = os.path.join(base_folder, dyad_dir)
    files = os.listdir(dir_path)
    for f in files:
        match = pattern.match(f)
        if not match:
            continue
        dyad, event = match.groups()
        if event not in TARGET_EVENTS:
            continue
        
        try:
            ds = xr.open_dataset(os.path.join(dir_path, f))
            if ELECTRODE not in ds.data_vars:
                ds.close()
                continue
            
            fs = ds.attrs.get('sfreq', 128.0)
            signal = ds[ELECTRODE].values
            
            # Determine event boundaries
            start_idx = 0
            end_idx = len(signal)
            if 'event_start' in ds.attrs and 'event_duration' in ds.attrs:
                start_idx = int(ds.attrs['event_start'])
                end_idx = start_idx + int(ds.attrs['event_duration'])
            elif 'event_duration_s' in ds.attrs:
                 end_idx = int(ds.attrs['event_duration_s'] * fs)
            
            segment = signal[start_idx:end_idx]
            n_samples = len(segment)
            if n_samples == 0:
                ds.close()
                continue
                
            nperseg = min(round(2 * fs), n_samples)
            if nperseg < 2:
                ds.close()
                continue
                
            f_welch, p_welch = welch(segment, fs=fs, nperseg=nperseg)
            
            # Filter fmax
            mask = f_welch <= FMAX
            psd_list.append(p_welch[mask])
            freqs_list.append(f_welch[mask])
            ds.close()
        except Exception:
            continue

if not psd_list:
    print("No traces found.")
    exit()

# Align using densest frequency grid
n_bins = [len(f) for f in freqs_list]
idx_densest = np.argmax(n_bins)
common_freqs = freqs_list[idx_densest]

stacked_psds = []
for f, p in zip(freqs_list, psd_list):
    if np.array_equal(f, common_freqs):
        stacked_psds.append(p)
    else:
        f_interp = interp1d(f, p, bounds_error=False, fill_value=np.nan)
        stacked_psds.append(f_interp(common_freqs))

stacked_matrix = np.array(stacked_psds)
n_eff = np.sum(~np.isnan(stacked_matrix), axis=0)
mean_psd = np.nanmean(stacked_matrix, axis=0)
std_psd = np.nanstd(stacked_matrix, axis=0, ddof=1)
sem_psd = std_psd / np.sqrt(n_eff)

print(f"Number of {ELECTRODE} traces: {len(psd_list)}")
print(f"Reference frequency bins: {len(common_freqs)}")
print(f"NaN count in stacked matrix: {np.isnan(stacked_matrix).sum()}")
print(f"NaN count in mean_psd: {np.isnan(mean_psd).sum()}")
print(f"NaN count in std_psd: {np.isnan(std_psd).sum()}")
print(f"NaN count in sem_psd: {np.isnan(sem_psd).sum()}")
print(f"Min n_eff: {np.min(n_eff)}")
print(f"Bins with n_eff < 2: {np.sum(n_eff < 2)}")

problematic_freqs = common_freqs[np.isnan(sem_psd)]
if len(problematic_freqs) > 0:
    print(f"First 10 problematic frequencies (SEM is NaN): {problematic_freqs[:10]}")
else:
    print("No problematic frequencies (SEM is NaN).")
