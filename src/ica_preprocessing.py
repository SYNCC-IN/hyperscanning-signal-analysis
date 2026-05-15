from pathlib import Path
import xarray as xr
import matplotlib.pyplot as plt
from IPython.display import Markdown, display

import warnings
import mne
from mne.preprocessing import ICA

from src.export import load_eeg_ncdf_as_mne_raw, plot_loaded_eeg_signals


class ICAPreprocessor:
    def __init__(self, export_folder: Path, target_events: list):
        self.export_folder = export_folder
        self.target_events = target_events
        self.eeg_files: list = []
    

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
            mode = "FULL ICA PREPROCESSING"

        self.eeg_files = []
        for dyad in dyads_to_process:
            self.eeg_files.extend(sorted(files_by_dyad[dyad]))

        print(f"Mode: {mode}")
        print(f"Dyads selected: {len(dyads_to_process)} / {len(all_dyads)}")
        print(f"Files selected: {len(self.eeg_files)} / {len(all_eeg_files)}")
        print("Dyads:")
        for dyad in dyads_to_process:
            print(f"  - {dyad}")


    def preprocess_with_ica(self, ica_n_components: int = 15, ica_max_iter: int = 2000, eog_channels: list = ['Fp1', 'Fp2'], eog_threshold: float = 3.0):
        '''
        Preprocess EEG signals using ICA for blink artifact deletion.
        '''
        if not self.eeg_files:
            raise RuntimeError(
                "No EEG files loaded. Call find_eeg_files() before preprocess_with_ica()."
            )
        self.raw_signals = {}
        self.cleaned_signals = {}

        for ncdf_path in self.eeg_files:
            label = ncdf_path.stem
            display(Markdown(f"## {label}"))
            print(f"Processing: {ncdf_path.name}")
            

            with xr.open_dataarray(ncdf_path) as da_original:
                original_attrs = da_original.attrs.copy()
                original_name = da_original.name

            raw_signal = load_eeg_ncdf_as_mne_raw(str(ncdf_path), montage = "standard_1020", scale_to_volts = 1e-6)

            signals = raw_signal.get_data() * 1e6
            channel_names = raw_signal.ch_names
            fs = raw_signal.info['sfreq']
            time_s = raw_signal.times
            event_duration_s = time_s[-1] if len(time_s) > 0 else 0.0

            self.raw_signals[label] = {
                'sig': signals, 
                'time_s': time_s, 
                'channel_names': channel_names, 
                'fs': fs, 
                'event_duration_s': event_duration_s
            }

            print(
                f"Loaded EEG data: {len(channel_names)} channels, "
                f"{raw_signal.n_times} samples at {fs} Hz, "
                f"duration {event_duration_s:.2f}s"
            )

            # Initialize ICA
            ica = ICA(n_components=ica_n_components, random_state=42, max_iter=ica_max_iter)

            with warnings.catch_warnings():
                warnings.filterwarnings("ignore", category=RuntimeWarning, module="mne")
                old_log_level = mne.set_log_level('ERROR', return_old_level=True)
                try:
                    ica.fit(raw_signal)
                finally:
                    mne.set_log_level(old_log_level)



            # Check which of the provided EOG channels actually exist in this file
            available_eog = [ch for ch in eog_channels if ch in channel_names]

            if not available_eog:
                print(f"WARNING: None of the proxy EOG channels {eog_channels} found in {label}.")
                print("Skipping automatic blink artifact removal for this file.")
                eog_indices = []
            else:
                # Use frontal EEG channels as a proxy for missing EOG.
                eog_indices, _ = ica.find_bads_eog(
                    raw_signal, 
                    ch_name=available_eog, 
                    threshold=eog_threshold
                )
            
            ica.exclude = eog_indices

            print(f"Identified blink components for removal: {eog_indices}")

            # Clean the signal
            raw_cleaned = raw_signal.copy()
            ica.apply(raw_cleaned)

            # Return to NumPy array
            signals_after_ica = raw_cleaned.get_data() * 1e6

            self.cleaned_signals[label] = {
                'sig': signals_after_ica, 
                'time_s': time_s, 
                'channel_names': channel_names, 
                'fs': fs, 
                'event_duration_s': event_duration_s,
                'original_attrs': original_attrs,
                'original_name': original_name
            }

    def plot_signals(self, label):

        if label not in self.raw_signals or label not in self.cleaned_signals:
            raise KeyError(f"No data found for {label}. Did you run preprocess_with_ica() first?")

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


    def save_cleaned_signals(self, cleaned_signals_folder: Path, save_plots: bool = True):
        '''
        Save cleaned signals back to NetCDF files, preserving and enriching original metadata.
        '''
        cleaned_signals_folder.mkdir(parents=True, exist_ok=True)

        for label, data in self.cleaned_signals.items():
            output_dir = cleaned_signals_folder / label[:5]
            output_dir.mkdir(parents=True, exist_ok=True)
            export_path = output_dir / f"{label}_cleaned.nc"
            print(f"Saving cleaned signals to: {export_path}")
            
            # Prepare data for saving
            sig = data['sig']
            time_s = data['time_s']
            channel_names = list(data['channel_names'])
            fs = float(data['fs'])
            event_duration_s = float(data['event_duration_s'])
            
            # Get original attributes to preserve all metadata, then we will enrich it
            original_attrs = data.get('original_attrs', {})
            
            # Make a copy of original attributes to modify for the cleaned version
            new_attrs = original_attrs.copy()
            
            # Enrich the description and processing history to reflect the ICA cleaning step
            old_desc = new_attrs.get('description', 'EEG signals')
            new_attrs['description'] = f"{old_desc} | Cleaned with ICA (blink artifacts removed)"
            new_attrs['processing_history'] = new_attrs.get('processing_history', '') + " -> ICA_cleaned"
            
            new_attrs['sampling_freq'] = fs
            new_attrs['event_duration_s'] = event_duration_s

            assert sig.ndim == 2 and sig.shape[0] == len(channel_names), (
                f"Expected sig shape (n_channels, n_times) with n_channels={len(channel_names)}, "
                f"got {sig.shape}"
            )
            sig_transposed = sig.T

            # Create xarray DataArray
            da = xr.DataArray(
                data=sig_transposed,
                dims=['time', 'channel'],
                coords={
                    'time': time_s, 
                    'channel': channel_names 
                },
                name=data.get('original_name', "eeg_signal"),
                attrs=new_attrs
            )
            
            da.to_netcdf(export_path, engine='netcdf4')

            if save_plots:
                plot_loaded_eeg_signals(
                    time_s=time_s, 
                    signals=sig, 
                    channel_names=channel_names, 
                    event_duration_s=event_duration_s,
                    title=f"{label} - Cleaned with ICA"
                )

                plot_export_path = output_dir / f"{label}_cleaned_ica_plot.png"
                
                # Intercept the plot and save it to a file
                plt.savefig(plot_export_path, dpi=300, bbox_inches='tight')
                plt.close()
                
                print(f"Saved plot preview to: {plot_export_path}")