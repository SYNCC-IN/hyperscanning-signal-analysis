"""ICA-based EEG preprocessing utilities.

This module provides a workflow-oriented preprocessor for EEG NetCDF exports.
It supports:
- deterministic file discovery by dyad and event,
- optional filtering by a provided valid-dyad allowlist,
- ICA cleaning with blink (EOG) artifact removal,
- optional muscle artifact removal with robust fallback,
- export of cleaned signals to NetCDF with processing provenance metadata.
"""

from pathlib import Path
from copy import deepcopy
from typing import Optional, Sequence
import numpy as np
import xarray as xr
import matplotlib.pyplot as plt
from IPython.display import Markdown, display

import warnings
import mne
from mne.preprocessing import ICA

from src.export import (
    load_eeg_ncdf_as_mne_raw,
    load_xarray_from_netcdf,
    get_export_metadata,
    plot_loaded_eeg_signals,
    _sanitize_netcdf_attrs_inplace,
)
from src.passive_io_helpers import (
    build_role_lookup,
    discover_role_files,
    pairs_from_lookup,
)


class ICAPreprocessor:
    """Run ICA preprocessing for dyad/event EEG files.

    Parameters
    ----------
    export_folder
        Root folder containing exported EEG NetCDF files.
    target_events
        Events that must be present in a complete dyad (e.g. three movie names).
    valid_dyads
        Optional allowlist of dyad IDs. If provided, only these complete dyads
        are considered during file selection.

    Notes
    -----
    Files are discovered from names matching the pattern:
    ``W_XXX_EEG_(ch|cg)_EVENT.nc``
    where ``ch`` is child and ``cg`` is caregiver.
    """

    SIGNAL_TYPE = "EEG"

    def __init__(self, export_folder: Path, target_events: list, valid_dyads: Optional[Sequence[str]] = None):
        self.export_folder = export_folder
        self.target_events = target_events
        self.eeg_files: list = []
        self.valid_dyads: Optional[list] = sorted(set(valid_dyads)) if valid_dyads is not None else None


    def set_valid_dyads(self, valid_dyads: Optional[Sequence[str]] = None):
        """Set or clear dyad allowlist used during EEG file selection.

        Parameters
        ----------
        valid_dyads
            Sequence of dyad IDs to keep, or ``None`` to disable allowlist
            filtering and use all complete dyads.
        """
        self.valid_dyads = None if valid_dyads is None else sorted(set(valid_dyads))


    def _discover_complete_role_files(self):
        """Return complete child/caregiver dyad-event tuples for target events.

        Honors ``self.valid_dyads`` (when set) via the discovery helper, so only
        allowlisted dyads are scanned.

        Returns
        -------
        list[tuple[str, str, Path, Path]]
            Sorted tuples of ``(dyad_id, event, child_path, caregiver_path)``
            only for dyad-event combinations where both roles exist.
        """
        role_files = discover_role_files(
            self.export_folder,
            self.target_events,
            signal_type=self.SIGNAL_TYPE,
            glob_pattern="*.nc",
            valid_dyads=self.valid_dyads,
        )
        return pairs_from_lookup(build_role_lookup(role_files))


    @staticmethod
    def _pick_mastoid_reference(channel_names: Sequence[str]) -> Optional[list[str]]:
        """Return detected mastoid reference channel pair, if present."""
        candidates = [
            ('M1', 'M2'),
            ('A1', 'A2'),
            ('TP9', 'TP10'),
            ('P9', 'P10'),
        ]
        by_upper = {name.upper(): name for name in channel_names}
        for left, right in candidates:
            if left in by_upper and right in by_upper:
                return [by_upper[left], by_upper[right]]
        return None


    @staticmethod
    def _expand_boolean_mask(mask: np.ndarray, pad_samples: int) -> np.ndarray:
        """Expand a boolean sample mask by +/- pad_samples."""
        if pad_samples <= 0 or mask.size == 0:
            return mask
        kernel = np.ones(2 * pad_samples + 1, dtype=int)
        expanded = np.convolve(mask.astype(int), kernel, mode='same') > 0
        return expanded


    @staticmethod
    def _mask_to_segments(mask: np.ndarray, fs: float) -> list:
        """Convert boolean sample mask to (onset_s, duration_s) segments."""
        if mask.size == 0 or not mask.any():
            return []

        edges = np.diff(mask.astype(int), prepend=0, append=0)
        starts = np.where(edges == 1)[0]
        ends = np.where(edges == -1)[0]
        segments = []
        for s, e in zip(starts, ends):
            onset = s / fs
            duration = max((e - s) / fs, 1.0 / fs)
            segments.append((onset, duration))
        return segments


    @staticmethod
    def _interpolate_masked_samples(signals: np.ndarray, mask: np.ndarray) -> np.ndarray:
        """Linearly interpolate masked samples channel-wise."""
        if signals.size == 0 or mask.size == 0 or not mask.any():
            return signals

        out = signals.copy()
        n_samples = out.shape[1]
        x = np.arange(n_samples)
        good = ~mask

        if good.sum() < 2:
            return out

        for ch in range(out.shape[0]):
            y = out[ch]
            out[ch, mask] = np.interp(x[mask], x[good], y[good])
        return out
    

    def find_eeg_files(self, smoke_test: bool = True, smoke_dyads_n: int = 2):
        """Build the list of EEG files to preprocess.

        This method selects only dyads that are complete across all
        ``self.target_events`` and, if provided, match ``self.valid_dyads``.

        Parameters
        ----------
        smoke_test
            If ``True``, limit processing to the first ``smoke_dyads_n`` dyads.
        smoke_dyads_n
            Number of dyads to keep in smoke mode.

        Raises
        ------
        FileNotFoundError
            If no usable EEG files are discovered under ``export_folder``.
        ValueError
            If filtering leaves no complete dyad-event pairs.
        """
        self.smoke_test = smoke_test
        self.smoke_dyads_n = smoke_dyads_n

        # Discovery already applies the valid_dyads allowlist (when set).
        pairs_all = self._discover_complete_role_files()
        if not pairs_all:
            allowlist_note = "" if self.valid_dyads is None else " (after valid_dyads allowlist)"
            raise FileNotFoundError(
                f"No EEG NetCDF files found for events {self.target_events}{allowlist_note} under: {self.export_folder}"
            )

        events_required = set(self.target_events)
        events_by_dyad = {}
        for dyad_id, event, _, _ in pairs_all:
            events_by_dyad.setdefault(dyad_id, set()).add(event)

        complete_dyads = sorted([
            dyad_id for dyad_id, evs in events_by_dyad.items()
            if events_required.issubset(evs)
        ])

        pairs_selected = [p for p in pairs_all if p[0] in set(complete_dyads)]
        if not pairs_selected:
            raise ValueError("No complete dyad-event EEG pairs left after valid_dyads filtering.")

        files_by_dyad = {}
        for dyad_id, event, path_ch, path_cg in pairs_selected:
            files_by_dyad.setdefault(dyad_id, {})[event] = (path_ch, path_cg)

        all_dyads = sorted(files_by_dyad.keys())

        if self.smoke_test:
            dyads_to_process = all_dyads[:self.smoke_dyads_n]
            mode = f"SMOKE TEST (first {self.smoke_dyads_n} dyads)"
        else:
            dyads_to_process = all_dyads
            mode = "FULL ICA PREPROCESSING"

        self.eeg_files = []
        for dyad in dyads_to_process:
            for event in self.target_events:
                path_ch, path_cg = files_by_dyad[dyad][event]
                self.eeg_files.extend([path_ch, path_cg])

        all_selected_files_n = len(pairs_selected) * 2

        print(f"Mode: {mode}")
        print(f"Dyads selected: {len(dyads_to_process)} / {len(all_dyads)}")
        print(f"Files selected: {len(self.eeg_files)} / {all_selected_files_n}")
        print(f"Complete dyads with all target events: {len(all_dyads)}")
        if self.valid_dyads is not None:
            print(f"Applied valid_dyads allowlist: {len(self.valid_dyads)} dyads")
        print("Dyads:")
        for dyad in dyads_to_process:
            print(f"  - {dyad}")


    def preprocess_and_save(
        self,
        cleaned_signals_folder: Path,
        ica_n_components: int = 15,
        ica_max_iter: int = 2000,
        eog_channels: list = ['Fp1', 'Fp2'],
        eog_threshold: float = 3.0,
        clean_muscle: bool = False,
        muscle_l_freq_hz: float = 30.0,
        muscle_threshold: Optional[float] = None,
        muscle_max_fraction: float = 0.4,
        protect_transients: bool = True,
        transient_zscore_threshold: float = 8.0,
        transient_min_channels: int = 6,
        transient_padding_s: float = 0.15,
        transient_max_fraction: float = 0.15,
        transient_interpolate_output: bool = True,
        save_plots: bool = True,
    ):
        """Preprocess selected EEG files with ICA and save cleaned NetCDF outputs.

        Workflow
        --------
        1. Load each EEG NetCDF file as MNE Raw.
        2. Fit ICA on high-pass filtered copy of the data.
        3. Detect EOG-related components (blink cleaning).
        4. Optionally detect muscle-related components.
        5. Exclude selected components and apply ICA to the original signal.
        6. Save cleaned signal + provenance metadata and optional plot preview.

        Parameters
        ----------
        cleaned_signals_folder
            Output directory for cleaned files (subfolders created per dyad).
        ica_n_components
            Number of ICA components to estimate.
        ica_max_iter
            Maximum number of ICA fitting iterations.
        eog_channels
            Channel names used as EOG proxies for blink component detection.
        eog_threshold
            Threshold for ``ica.find_bads_eog``.
        clean_muscle
            If ``True``, attempt to detect and remove muscle-related ICA components.
            Default is ``False`` (conservative: blink-only cleaning).
        muscle_l_freq_hz
            High-pass cutoff used before muscle-component scoring.
        muscle_threshold
            Optional threshold for ``ica.find_bads_muscle``. If ``None``, MNE
            default threshold behavior is used.
        muscle_max_fraction
            Safety guard for muscle auto-labeling. If the fraction of ICA
            components flagged as muscle is greater than this value, muscle
            indices are discarded and only EOG exclusions are applied.
        protect_transients
            If ``True``, detect large transient artifacts and exclude them from
            ICA fitting via ``bad`` annotations.
        transient_zscore_threshold
            Robust z-score threshold for transient detection.
        transient_min_channels
            Minimum number of channels that must cross threshold at a sample to
            mark it as a transient candidate.
        transient_padding_s
            Padding added before/after detected transient samples.
        transient_max_fraction
            Safety cap: if detected transient samples exceed this fraction of the
            recording, the transient mask is rejected.
        transient_interpolate_output
            If ``True``, transient samples (when accepted) are linearly
            interpolated in the final cleaned output.
        save_plots
            If ``True``, save a diagnostic plot of cleaned signals per file.

        Raises
        ------
        RuntimeError
            If no files are selected; call ``find_eeg_files`` first.

        Notes
        -----
        Muscle detection is wrapped in a robust fallback: if detection fails for
        a file, preprocessing continues with blink-only cleaning for that file.
        """
        if not self.eeg_files:
            raise RuntimeError("No EEG files loaded. Call find_eeg_files() first.")
        
        cleaned_signals_folder.mkdir(parents=True, exist_ok=True)

        for ncdf_path in self.eeg_files:
            label = ncdf_path.stem
            display(Markdown(f"## {label}"))
            print(f"Processing & Saving: {ncdf_path.name}")
            
            # Load source EEG with decoded attrs so structured metadata (metadata_json,
            # child_info, notes, etc.) can be preserved in cleaned outputs.
            da_original = load_xarray_from_netcdf(str(ncdf_path), decode_json_attrs=True)
            original_attrs = deepcopy(da_original.attrs)
            original_name = da_original.name
            original_dims = tuple(da_original.dims)
            original_time_s = np.asarray(da_original.coords['time'].values, dtype=float).copy()
            original_metadata = get_export_metadata(da_original)

            raw_signal = load_eeg_ncdf_as_mne_raw(str(ncdf_path), montage="standard_1020", scale_to_volts=1e-6)

            # Get signal data and metadata
            signals = raw_signal.get_data() * 1e6
            channel_names = raw_signal.ch_names
            fs = raw_signal.info['sfreq']
            time_s_raw = raw_signal.times

            # Preserve original exported time axis semantics (e.g., -margin .. +duration+margin).
            if len(original_time_s) == signals.shape[1]:
                time_s = original_time_s
            else:
                time_s = time_s_raw

            # Prefer source event duration metadata if available.
            event_duration_s = float(
                original_attrs.get(
                    'event_duration',
                    original_attrs.get('event_duration_s', time_s[-1] if len(time_s) > 0 else 0.0)
                )
            )

            # ICA preprocessing
            raw_for_ica = raw_signal.copy()
            # Fit and score ICA in CAR space for stable global artifact separation.
            raw_for_ica.set_eeg_reference(ref_channels='average', projection=False, verbose='ERROR')
            raw_for_ica.filter(l_freq=1.0, h_freq=None, verbose='ERROR')

            # Optional protection against high-amplitude transients.
            transient_detection_status = 'not requested'
            transient_mask = np.zeros(signals.shape[1], dtype=bool)
            transient_fraction = 0.0
            transient_segments = []
            if protect_transients:
                med = np.median(signals, axis=1, keepdims=True)
                mad = np.median(np.abs(signals - med), axis=1, keepdims=True)
                scale = 1.4826 * np.maximum(mad, 1e-12)
                robust_z = np.abs(signals - med) / scale

                min_ch = max(1, min(int(transient_min_channels), signals.shape[0]))
                candidate_mask = (robust_z >= float(transient_zscore_threshold)).sum(axis=0) >= min_ch
                pad_samples = int(round(float(transient_padding_s) * float(fs)))
                transient_mask = self._expand_boolean_mask(candidate_mask, pad_samples)
                transient_fraction = float(transient_mask.mean()) if transient_mask.size else 0.0

                if transient_mask.any() and transient_fraction <= float(transient_max_fraction):
                    transient_segments = self._mask_to_segments(transient_mask, float(fs))
                    if transient_segments:
                        ann = mne.Annotations(
                            onset=[s for s, _ in transient_segments],
                            duration=[d for _, d in transient_segments],
                            description=['bad_transient'] * len(transient_segments),
                            orig_time=raw_for_ica.info['meas_date'],
                        )
                        raw_for_ica.set_annotations(raw_for_ica.annotations + ann)
                    transient_detection_status = (
                        f'ok ({transient_fraction:.3f} fraction, {len(transient_segments)} segments)'
                    )
                elif transient_mask.any():
                    transient_mask[:] = False
                    transient_detection_status = (
                        f'rejected_too_many_flagged ({transient_fraction:.3f} > {float(transient_max_fraction):.3f})'
                    )
                    print(
                        f"  Transient mask rejected for {label}: {transient_fraction:.3f} of samples flagged "
                        f"(> {float(transient_max_fraction):.3f})."
                    )
                else:
                    transient_detection_status = 'ok (none detected)'

            ica = ICA(n_components=ica_n_components, random_state=42, max_iter=ica_max_iter)
            with warnings.catch_warnings():
                warnings.filterwarnings("ignore", category=RuntimeWarning, module="mne")
                old_log_level = mne.set_log_level('ERROR', return_old_level=True)
                try:
                    ica.fit(raw_for_ica)
                finally:
                    mne.set_log_level(old_log_level)

            available_eog = [ch for ch in eog_channels if ch in channel_names]
            if not available_eog:
                eog_indices = []
            else:
                old_log_level = mne.set_log_level('ERROR', return_old_level=True)
                try:
                    eog_indices, _ = ica.find_bads_eog(
                        raw_for_ica,
                        ch_name=available_eog,
                        threshold=eog_threshold,
                    )
                finally:
                    mne.set_log_level(old_log_level)

            # Optional muscle-component detection.
            muscle_indices = []
            muscle_detection_note = 'not requested'
            if clean_muscle:
                try:
                    raw_for_muscle = raw_for_ica.copy()
                    raw_for_muscle.filter(l_freq=muscle_l_freq_hz, h_freq=None, verbose='ERROR')

                    old_log_level = mne.set_log_level('ERROR', return_old_level=True)
                    try:
                        if muscle_threshold is None:
                            muscle_indices, _ = ica.find_bads_muscle(raw_for_muscle)
                        else:
                            muscle_indices, _ = ica.find_bads_muscle(raw_for_muscle, threshold=muscle_threshold)
                    finally:
                        mne.set_log_level(old_log_level)

                    # Guard against over-aggressive auto-labeling.
                    n_components = int(getattr(ica, 'n_components_', len(muscle_indices) or 1))
                    frac_flagged = len(set(muscle_indices)) / max(1, n_components)
                    if frac_flagged > muscle_max_fraction:
                        muscle_indices = []
                        muscle_detection_note = (
                            f'rejected_too_many_flagged '
                            f'({frac_flagged:.2f} > {muscle_max_fraction:.2f})'
                        )
                        print(
                            f"  Muscle auto-labeling rejected for {label}: "
                            f"{frac_flagged:.2f} of components flagged (> {muscle_max_fraction:.2f}). "
                            "Continuing with blink-only cleaning."
                        )
                    else:
                        muscle_detection_note = f'ok ({frac_flagged:.2f} flagged)'
                except Exception as exc:
                    # Robust fallback: keep blink cleaning only if muscle detection fails.
                    muscle_indices = []
                    muscle_detection_note = f'failed ({exc})'
                    print(f"  Muscle detection failed for {label}; continuing with blink-only cleaning. Reason: {exc}")

            all_excluded_indices = sorted(set(eog_indices).union(set(muscle_indices)))
            ica.exclude = all_excluded_indices

            print(f"  EOG ICs: {eog_indices}")
            if clean_muscle:
                print(f"  Muscle ICs: {muscle_indices}")
                print(f"  Muscle detection status: {muscle_detection_note}")
            print(f"  Final excluded ICs: {all_excluded_indices}")
            
            # Apply ICA in the same CAR space used during fitting.
            raw_cleaned = raw_signal.copy()
            raw_cleaned.set_eeg_reference(ref_channels='average', projection=False, verbose='ERROR')
            ica.apply(raw_cleaned)

            # Restore mastoid reference after ICA cleaning when channels are available.
            mastoid_ref = self._pick_mastoid_reference(channel_names)
            if mastoid_ref is not None:
                raw_cleaned.set_eeg_reference(ref_channels=mastoid_ref, projection=False, verbose='ERROR')
                reference_restore_note = f"restored_mastoid ({','.join(mastoid_ref)})"
            else:
                reference_restore_note = 'car_only_no_mastoids_found'
                print(
                    "  Could not restore mastoid reference (M1/M2-like channels not found). "
                    "Keeping CAR-referenced cleaned signal."
                )

            signals_after_ica = raw_cleaned.get_data() * 1e6

            if protect_transients and transient_interpolate_output and transient_mask.any():
                signals_after_ica = self._interpolate_masked_samples(signals_after_ica, transient_mask)

            # Saving cleaned signals to NetCDF with enriched metadata
            output_dir = cleaned_signals_folder / label[:5]
            output_dir.mkdir(parents=True, exist_ok=True)
            export_path = output_dir / f"{label}_cleaned.nc"

            new_attrs = deepcopy(original_attrs)
            old_desc = new_attrs.get('description', 'EEG signals')
            if clean_muscle:
                new_attrs['description'] = f"{old_desc} | Cleaned with ICA (blink + optional muscle artifacts removed)"
                new_attrs['processing_history'] = new_attrs.get('processing_history', '') + " -> ICA_cleaned_blink_muscle_optional"
            else:
                new_attrs['description'] = f"{old_desc} | Cleaned with ICA (blink artifacts removed)"
                new_attrs['processing_history'] = new_attrs.get('processing_history', '') + " -> ICA_cleaned_blink"
            new_attrs['sampling_freq'] = fs
            new_attrs['event_duration_s'] = event_duration_s
            new_attrs['ica_eog_channels'] = ','.join(available_eog)
            new_attrs['ica_eog_threshold'] = float(eog_threshold)
            new_attrs['ica_eog_indices'] = ','.join(str(i) for i in eog_indices)
            # NetCDF attrs should use primitive numeric/string scalars (avoid bool/None).
            new_attrs['ica_clean_muscle'] = int(clean_muscle)
            new_attrs['ica_muscle_l_freq_hz'] = float(muscle_l_freq_hz)
            new_attrs['ica_muscle_threshold'] = np.nan if muscle_threshold is None else float(muscle_threshold)
            new_attrs['ica_muscle_max_fraction'] = float(muscle_max_fraction)
            new_attrs['ica_muscle_detection_status'] = muscle_detection_note
            new_attrs['ica_muscle_indices'] = ','.join(str(i) for i in muscle_indices)
            new_attrs['ica_excluded_indices'] = ','.join(str(i) for i in all_excluded_indices)
            new_attrs['ica_reference_for_fit_apply'] = 'CAR'
            new_attrs['ica_reference_restored'] = reference_restore_note
            new_attrs['ica_protect_transients'] = int(protect_transients)
            new_attrs['ica_transient_zscore_threshold'] = float(transient_zscore_threshold)
            new_attrs['ica_transient_min_channels'] = int(transient_min_channels)
            new_attrs['ica_transient_padding_s'] = float(transient_padding_s)
            new_attrs['ica_transient_max_fraction'] = float(transient_max_fraction)
            new_attrs['ica_transient_detection_status'] = transient_detection_status
            new_attrs['ica_transient_fraction'] = float(transient_fraction)
            new_attrs['ica_transient_interpolate_output'] = int(transient_interpolate_output)

            # Keep structured metadata payload from source file and expose key blocks
            # directly so downstream code can read them without manual JSON parsing.
            if isinstance(original_metadata, dict) and original_metadata:
                new_attrs['metadata_json'] = deepcopy(original_metadata)
                if 'child_info' in original_metadata:
                    new_attrs['child_info'] = deepcopy(original_metadata['child_info'])
                if 'notes' in original_metadata:
                    new_attrs['notes'] = deepcopy(original_metadata['notes'])
                if 'event_order' in original_metadata:
                    new_attrs['event_order'] = deepcopy(original_metadata['event_order'])

            # Ensure attributes are NetCDF-safe after adding structured metadata.
            _sanitize_netcdf_attrs_inplace(new_attrs)

            # Preserve the source xarray dimension ordering whenever possible.
            if len(original_dims) == 2 and set(original_dims) == {'time', 'channel'}:
                target_dims = list(original_dims)
            else:
                target_dims = ['time', 'channel']

            if target_dims == ['time', 'channel']:
                data_out = signals_after_ica.T
                coords_out = {'time': time_s, 'channel': channel_names}
            elif target_dims == ['channel', 'time']:
                data_out = signals_after_ica
                coords_out = {'channel': channel_names, 'time': time_s}
            else:
                data_out = signals_after_ica.T
                coords_out = {'time': time_s, 'channel': channel_names}

            da = xr.DataArray(
                data=data_out,
                dims=target_dims,
                coords=coords_out,
                name=original_name,
                attrs=new_attrs,
            )
            da.to_netcdf(export_path, engine='netcdf4')
            print(f"-> Saved NetCDF to: {export_path}")

            # Save plot preview of cleaned signals if requested
            if save_plots:
                plot_loaded_eeg_signals(
                    time_s=time_s, 
                    signals=signals_after_ica, 
                    channel_names=channel_names, 
                    event_duration_s=event_duration_s,
                    title=f"{label} - Cleaned with ICA"
                )
                plot_export_path = output_dir / f"{label}_cleaned_ica_plot.png"
                plt.savefig(plot_export_path, dpi=300, bbox_inches='tight')
                plt.close()
                print(f"Saved plot preview to: {plot_export_path}")