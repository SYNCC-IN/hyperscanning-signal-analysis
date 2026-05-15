import os
import sys
import importlib
from pathlib import Path
import xarray as xr
from IPython.display import Markdown, display

import warnings
import mne
from mne.preprocessing import ICA

sys.path.insert(0, os.path.abspath('..'))

from src import export
importlib.reload(export)
from src.export import load_eeg_signals, plot_loaded_eeg_signals


class ICAPreprocessor:
    def __init__(self, export_folder: Path, figures_folder: Path):
        self.export_folder = export_folder
        self.figures_folder = figures_folder
        self.figures_folder.mkdir(exist_ok=True)
        self.target_events = ['Brave', 'Peppa', 'Incredibles']
    

    def find_eeg_files(self, smoke_test: bool = True, smoke_dyads_n: int = 2):

        self.smoke_test = smoke_test
        self.smoke_dyads_n = smoke_dyads_n

        all_eeg_files = sorted([
            p for p in self.export_folder.rglob("*.nc")
            if "_EEG_" in p.name
            and any(p.stem.endswith(f"_{ev}") for ev in self.target_events)
        ])

        if not all_eeg_files:
            raise FileNotFoundError(f"No EEG NetCDF files found for events {self.target_events} under: {self.export_folder}")

        # Group files by dyad (e.g. W_030)
        files_by_dyad = {}
        for p in all_eeg_files:
            parts = p.stem.split('_')
            dyad_id = f"{parts[0]}_{parts[1]}" if len(parts) >= 2 else p.stem
            files_by_dyad.setdefault(dyad_id, []).append(p)

        all_dyads = sorted(files_by_dyad.keys())

        if self.smoke_test:
            dyads_to_process = all_dyads[:self.smoke_dyads_n]
            mode = f"SMOKE TEST (first {self.smoke_dyads_n} dyads)"
        else:
            dyads_to_process = all_dyads
            mode = "FULL ANALYSIS"

        self.eeg_files = []
        for dyad in dyads_to_process:
            self.eeg_files.extend(sorted(files_by_dyad[dyad]))

        print(f"Mode: {mode}")
        print(f"Dyads selected: {len(dyads_to_process)} / {len(all_dyads)}")
        print(f"Files selected: {len(self.eeg_files)} / {len(all_eeg_files)}")
        print("Dyads:")
        for dyad in dyads_to_process:
            print(f"  - {dyad}")


    def preprocess_with_ica(self):
        '''
        Preprocess EEG signals using ICA for blik artifact deletion.
        '''
        self.raw_signals = {}
        self.cleaned_signals = {}

        for ncdf_path in self.eeg_files:
            label = ncdf_path.stem
            display(Markdown(f"## {label}"))
            print(f"Processing: {ncdf_path.name}")
            signals, channel_names, fs, time_s, event_duration_s = load_eeg_signals(ncdf_path)

            print(signals.shape, len(channel_names), fs, time_s.shape, event_duration_s)
            self.raw_signals[label] = {'sig': signals, 'time_s': time_s, 'channel_names': channel_names, 'fs': fs, 'event_duration_s': event_duration_s}
            
            # Przygotowanie danych do MNE
            info = mne.create_info(ch_names=channel_names, sfreq=fs, ch_types=['eeg'] * len(channel_names))
            raw_signal = mne.io.RawArray(signals, info)
            raw_signal.set_montage('standard_1020')

            # Inicjalizacja i dopasowanie modelu ICA
            ica = ICA(n_components=15, random_state=42, max_iter=2000)

            # Wyciszenie ostrzeżeń Pythona i logów MNE na czas dopasowywania ICA
            with warnings.catch_warnings():
                warnings.filterwarnings("ignore", category=RuntimeWarning, module="mne")
                mne.set_log_level('ERROR')
                ica.fit(raw_signal)
                mne.set_log_level('WARNING')

            # Automatyczne szukanie komponentu odpowiadającego za mrugnięcia 
            eog_indices, eog_scores = ica.find_bads_eog(
                raw_signal, 
                ch_name=['Fp1', 'Fp2'], 
                threshold=3.0
            )
            ica.exclude = eog_indices

            print(f"Zidentyfikowano komponenty mrugnięć do usunięcia: {eog_indices}")

            # Oczyszczenie sygnału
            raw_cleaned = raw_signal.copy()
            ica.apply(raw_cleaned)

            # Powrót do macierzy NumPy
            signals_after_ica = raw_cleaned.get_data()
            self.cleaned_signals[label] = {'sig': signals_after_ica, 'time_s': time_s, 'channel_names': channel_names, 'fs': fs, 'event_duration_s': event_duration_s}
    

    def plot_signals(self, label):
        d_raw = self.raw_signals[label]
        d_clean = self.cleaned_signals[label]

        plot_loaded_eeg_signals(
            time_s=d_raw['time_s'], 
            signals=d_raw['sig'], 
            channel_names=d_raw['channel_names'], 
            event_duration_s=d_raw['event_duration_s'],
            title=f"{label} - Raw"
        )
        
        plot_loaded_eeg_signals(
            time_s=d_clean['time_s'], 
            signals=d_clean['sig'], 
            channel_names=d_clean['channel_names'], 
            event_duration_s=d_clean['event_duration_s'],
            title=f"{label} - Cleaned with ICA"
        )


    def save_cleaned_signals(self, cleaned_signals_folder: Path):
        '''
        Save cleaned signals back to NetCDF files.
        '''
        cleaned_signals_folder.mkdir(parents=True, exist_ok=True)

        for label, data in self.cleaned_signals.items():
            export_path = cleaned_signals_folder / f"{label}_cleaned.nc"
            print(f"Saving cleaned signals to: {export_path}")
            
            # Przygotowanie danych do zapisu
            sig = data['sig']
            time_s = data['time_s']
            channel_names = list(data['channel_names'])
            fs = float(data['fs'])
            event_duration_s = float(data['event_duration_s'])

            sig_transposed = sig.T 

            # Tworzenie xarray DataArray
            da = xr.DataArray(
                data=sig_transposed,
                dims=['time', 'channel'],
                coords={
                    'time': time_s, 
                    'channel': channel_names 
                },
                name="eeg_signal",
                attrs={
                    'sampling_freq': fs,
                    'event_duration_s': event_duration_s,
                    'description': 'EEG signals cleaned with ICA'
                }
            )
            
            da.to_netcdf(export_path, engine='netcdf4')