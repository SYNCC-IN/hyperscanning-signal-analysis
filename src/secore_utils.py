from pathlib import Path
import json
import numpy as np
import xarray as xr
import matplotlib.pyplot as plt
import pandas as pd

# Define colors per event (fallback to neutral when missing)
SECORE_EVENT_COLORS = {
    'puzzle': '#fde725',
    'cleaning': '#5ec962',
    'wrong present': '#21918c',
    'surprise': '#3b528b',
    }
def _sanitize_attrs_for_netcdf(attrs):
    out = {}
    for key, value in attrs.items():
        if value is None:
            out[key] = ""
        elif isinstance(value, (str, bytes, bool, int, float, np.number)):
            out[key] = value
        elif isinstance(value, (list, tuple, dict)):
            out[key] = json.dumps(value, ensure_ascii=False, default=str)
        else:
            out[key] = str(value)
    return out


def export_h10_to_secore_ncdf(h10_xarray, dyad_id, export_root):
    sampling_freq = float(h10_xarray.attrs.get("sampling_frequency_Hz", np.nan))

    event_windows = json.loads(h10_xarray.attrs.get("event_windows_s_json", "{}"))
    events_start_s = {name: float(win["start_s"]) for name, win in event_windows.items()}
    events_duration_s = {
        name: float(win["end_s"]) - float(win["start_s"]) for name, win in event_windows.items()
    }
    event_order = [k for k, _ in sorted(events_start_s.items(), key=lambda kv: kv[1])]

    metadata_payload = {
        "notes": "",
        "child_info": {},
        "event_order": event_order,
        "secore_event_windows_s": event_windows,
    }

    time_values = h10_xarray.coords["time"].values.astype(float)

    export_plan = [
        ("Secore_IBI", "IBI", "IBI_CH", "ch", "child", "IBI"),
        ("Secore_IBI", "IBI", "IBI_CG", "cg", "caregiver", "IBI"),
        ("Secore_RMSSD", "RMSSD", "RMSSD_CH", "ch", "child", "RMSSD"),
        ("Secore_RMSSD", "RMSSD", "RMSSD_CG", "cg", "caregiver", "RMSSD"),
    ]

    saved_files = []
    for export_folder_name, modality, src_channel, who, member_folder, out_channel in export_plan:
        if src_channel not in h10_xarray.channel.values:
            raise ValueError(f"Missing channel {src_channel} in h10_xarray for {dyad_id}.")

        sig_values = h10_xarray.sel(channel=src_channel).values.astype(float)
        signals = xr.DataArray(
            data=sig_values[:, None],
            coords={"time": time_values, "channel": [out_channel]},
            dims=["time", "channel"],
            name="signals",
        )

        attrs = {
            "dyad_id": dyad_id,
            "who": who,
            "sampling_freq": sampling_freq,
            "event_name": "Secore",
            "event_start": float(time_values[0]),
            "event_duration": float(time_values[-1] - time_values[0]),
            "time_margin_s": 0.0,
            "channel_names_csv": out_channel,
            "channel_names_json": json.dumps([out_channel], ensure_ascii=True),
            "events_start_s_json": events_start_s,
            "events_duration_s_json": events_duration_s,
            "metadata_json": metadata_payload,
        }
        signals.attrs.update(_sanitize_attrs_for_netcdf(attrs))

        out_dir = Path(export_root) / export_folder_name / dyad_id / member_folder
        out_dir.mkdir(parents=True, exist_ok=True)
        out_file = out_dir / f"{dyad_id}_{modality}_{who}_Secore.nc"

        signals.to_netcdf(out_file, engine="netcdf4", format="NETCDF4_CLASSIC")
        saved_files.append(str(out_file))

    return saved_files


def save_secore_QC_figures(dyad_id, export_root):
    export_root = Path(export_root)



    # Load the four ncdf files written by the export cell above
    ibi_ch_nc = xr.open_dataarray(export_root / "Secore_IBI" / dyad_id / "child" / f"{dyad_id}_IBI_ch_Secore.nc")
    ibi_cg_nc = xr.open_dataarray(export_root / "Secore_IBI" / dyad_id / "caregiver" / f"{dyad_id}_IBI_cg_Secore.nc")
    rmssd_ch_nc = xr.open_dataarray(export_root / "Secore_RMSSD" / dyad_id / "child" / f"{dyad_id}_RMSSD_ch_Secore.nc")
    rmssd_cg_nc = xr.open_dataarray(export_root / "Secore_RMSSD" / dyad_id / "caregiver" / f"{dyad_id}_RMSSD_cg_Secore.nc")

    # Extract time axis and signal arrays
    t_nc = ibi_ch_nc.coords['time'].values
    ibi_ch_vals = ibi_ch_nc.sel(channel='IBI').values
    ibi_cg_vals = ibi_cg_nc.sel(channel='IBI').values
    rmssd_ch_vals = rmssd_ch_nc.sel(channel='RMSSD').values
    rmssd_cg_vals = rmssd_cg_nc.sel(channel='RMSSD').values

    # Reconstruct event metadata from the stored attrs
    metadata_nc = json.loads(ibi_ch_nc.attrs.get('metadata_json', '{}'))
    event_windows_s_nc = metadata_nc.get('secore_event_windows_s', {})
    event_order_nc = metadata_nc.get('event_order', sorted(event_windows_s_nc.keys()))
    # Re-build a numeric code map from the recorded event order
    event_code_map_nc = {name: i + 1 for i, name in enumerate(event_order_nc) if name != 'baseline'}

    fig, axes = plt.subplots(2, 1, sharex=True, figsize=(13, 8))

    # IBI panel
    axes[0].plot(t_nc, ibi_ch_vals, label='IBI CH', lw=1.2)
    axes[0].plot(t_nc, ibi_cg_vals, label='IBI CG', lw=1.2, alpha=0.9)
    axes[0].set_ylabel('IBI [ms]')
    axes[0].set_title('IBI and RMSSD loaded from ncdf – with event windows')

    # RMSSD panel
    axes[1].semilogy(t_nc, rmssd_ch_vals, label='RMSSD CH', lw=1.2)
    axes[1].semilogy(t_nc, rmssd_cg_vals, label='RMSSD CG', lw=1.2, alpha=0.9)
    axes[1].set_ylabel('RMSSD [ms]')
    axes[1].set_xlabel('Time [s]')

    # Shade event windows on both axes
    for event_name, window in event_windows_s_nc.items():
        if event_name == 'baseline':
            continue
        start = float(window['start_s'])
        end = float(window['end_s'])
        color = SECORE_EVENT_COLORS.get(event_name, '#bbbbbb')
        for ax in axes:
            ax.axvspan(start, end, color=color, alpha=0.18)

    # Build combined signal + event legend
    event_handles = []
    for event_name, code in sorted(event_code_map_nc.items(), key=lambda kv: kv[1]):
        color = SECORE_EVENT_COLORS.get(event_name, '#bbbbbb')
        event_handles.append(plt.Rectangle((0, 0), 1, 1, color=color, alpha=0.3, label=f'{event_name} ({code})'))

    signal_handles_0 = axes[0].get_legend_handles_labels()[0]
    axes[0].legend(handles=event_handles + signal_handles_0, loc='upper left')
    axes[1].legend(loc='upper right')

    last_event_end = max(
        (float(w['end_s']) for w in event_windows_s_nc.values() if w.get('end_s') is not None),
        default=float(t_nc[-1]),
    )
    x_max = last_event_end + 60.0
    axes[0].set_xlim(0, x_max)
    axes[1].set_xlim(0, x_max)

    plt.tight_layout()
    plt.show()

    # save the figure in the export root
    fig_dir = export_root / "SECORE_FIGS"
    fig_dir.mkdir(parents=True, exist_ok=True)
    fig.savefig(fig_dir / f"{dyad_id}_IBI_RMSSD_Secore.png", dpi=300)
    plt.close(fig)

def sec_ms_str_to_float(value):
    if pd.isna(value):
        return np.nan

    s = str(value).strip().replace("\xa0", " ").replace("\u202f", " ")
    if not s:
        return np.nan

    parts = s.split()

    # Pattern like "18 199" or "1 191 372": groups of 3 digits after the first group.
    # Interpret as total milliseconds and convert to seconds.
    if len(parts) >= 2 and all(p.lstrip('-').isdigit() for p in parts):
        if all(len(p) == 3 for p in parts[1:]):
            sign = -1 if parts[0].startswith('-') else 1
            head = parts[0].lstrip('-')
            total_ms = int(head + ''.join(parts[1:]))
            return sign * (total_ms / 1000.0)

        # Legacy pattern: "sec ms" -> sec + ms/1000
        sec = float(parts[0].replace(',', '.'))
        ms = float(parts[1].replace(',', '.'))
        return sec + ms / 1000.0

    return float(s.replace(',', '.'))