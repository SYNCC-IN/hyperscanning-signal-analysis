# %% [markdown]
# # Batch export of Heidelberg dyads to NCDF by task
# Heidelberg counterpart of `export_dyade_to_ncdf_by_task_batch.py`. Every dyad that has a complete set of raw
# files (EEG `.vhdr/.eeg/.vmrk` plus the child and adult BI-Vital CSVs) is loaded, the BI-Vital recorders are
# synchronised to the EEG clock (by-product `<dyad>_synced_signals.csv`) and the `passive_movies` / `talk` chunks
# (EEG, ECG, IBI, RMSSD at 128 Hz) are written in the Warsaw folder layout, e.g. `EEG/H_046/child/H_046_EEG_ch_talk.nc`.
#
# Raw file names (the leading part is ignored, the dyad number is the digit group after the last underscore):
#   EEG:  synccin_LAKA36_46.vhdr / .eeg / .vmrk   -> dyad 46
#   ECG:  LAKA36_46-child.csv, LAKA36_46-adult.csv -> dyad 46
import csv
import sys
import traceback
from pathlib import Path

# repository root on the path so that `src` imports as a package
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.export import export_heidelberg_by_task
from src.heidelberg_loader import discover_heidelberg_dyads, parse_dyad_id
from src.mne_bridge import check_exported_data_quality

# ---- Configuration ----
HEIDELBERG_DIR = Path('/Users/admin/Documents/Hoza/PROJEKTY/SYNCC_IN_LOCAL_HOME/HEIDELBERG')
EEG_RAW_DIR = HEIDELBERG_DIR / 'EEG' / 'RAW'
ECG_RAW_DIR = HEIDELBERG_DIR / 'EKG' / 'RAW'
EXPORT_DIR = HEIDELBERG_DIR / 'HEIDELBERG_EEG_exported_BY_TASKS'
SYNC_QC_DIR = EXPORT_DIR / 'sync_QC'          # <dyad>_synced_signals.csv, fit summary, channel rankings, offset tracks
REPORT_CSV = EXPORT_DIR / 'export_report.csv'  # one row per dyad: status and reason
DYAD_NUMBERS = None                            # None = all complete dyads; or e.g. [32, 46] to export only these
SKIP_EXISTING = False                          # True: skip dyads whose EEG passive_movies files already exist
RUN_QUALITY_CHECK = True                       # AutoReject report of the passive_movies EEG chunks (as in Warsaw)
# Bad EEG channels to interpolate before CAR, per dyad number; labels as `<channel>_ch` (child) / `<channel>_cg` (parent).
EEG_BAD_CHANNELS = {
    # 46: ['Fz_ch', 'T7_cg'],
}


def export_one_dyad(dyad_id, files, bad_channels=None):
    print(f'Exporting {dyad_id}...')
    sync_failures = export_heidelberg_by_task(
        dyad_id_list=[dyad_id],
        data_dir=None,
        export_path=EXPORT_DIR,
        dyad_files={dyad_id: files},
        sync_output_dir=SYNC_QC_DIR,
        lowcut=1.0,
        highcut=64,                 # wide, for ICLabel-based ICA cleaning downstream (as in Warsaw)
        eeg_filter_type='iir',
        target_fs_hz=128,           # = Warsaw: 1024 Hz / 8
        time_margin=20,
        EEG_bad_channels=bad_channels,
        verbose=False,
    )
    if RUN_QUALITY_CHECK:
        for member in ('ch', 'cg'):
            check_exported_data_quality(
                dyad=parse_dyad_id(dyad_id), modality='EEG', member=member, task='passive_movies',
                export_folder=str(EXPORT_DIR),
            )
    print(f'Done: {dyad_id}')
    return sync_failures.get(parse_dyad_id(dyad_id), {})


# %% [markdown]
# ## Find the dyads with complete raw data
complete, problems = discover_heidelberg_dyads(EEG_RAW_DIR, ECG_RAW_DIR)
print(f'{len(complete)} complete dyads: {[parse_dyad_id(d) for d in complete]}')
for number, reason in problems.items():
    print(f'  NOT exported, dyad {number}: {reason}')

selected = {
    dyad: files for dyad, files in complete.items()
    if DYAD_NUMBERS is None or files['number'] in DYAD_NUMBERS
}
if DYAD_NUMBERS is not None:
    for number in set(DYAD_NUMBERS) - {files['number'] for files in selected.values()}:
        print(f'  requested dyad {number} is not complete: {problems.get(number, "no files found")}')

# %% [markdown]
# ## Export (a failing dyad is logged and the batch continues)
EXPORT_DIR.mkdir(parents=True, exist_ok=True)
report = [
    {'dyad': parse_dyad_id(f'H{number:03d}'), 'status': 'skipped: incomplete raw data', 'detail': reason}
    for number, reason in problems.items()
]
for dyad_id, files in selected.items():
    dyad_label = parse_dyad_id(dyad_id)
    already = (EXPORT_DIR / 'EEG' / dyad_label / 'child' / f'{dyad_label}_EEG_ch_passive_movies.nc').exists()
    if SKIP_EXISTING and already:
        print(f'Skipping {dyad_label}: already exported')
        report.append({'dyad': dyad_label, 'status': 'skipped: already exported', 'detail': ''})
        continue
    try:
        failed_sync = export_one_dyad(dyad_id, files, EEG_BAD_CHANNELS.get(files['number']))
        if failed_sync:
            who = ', '.join(failed_sync)
            report.append({'dyad': dyad_label, 'status': f'exported, EEG only for: {who} (ECG sync failed)',
                           'detail': ' || '.join(f'{m}: {r}' for m, r in failed_sync.items())})
        else:
            report.append({'dyad': dyad_label, 'status': 'exported', 'detail': files['eeg_stem'].name})
    except Exception as error:                                   # keep going with the other dyads
        traceback.print_exc()
        report.append({'dyad': dyad_label, 'status': 'FAILED', 'detail': f'{type(error).__name__}: {error}'})

with open(REPORT_CSV, 'w', newline='') as handle:
    writer = csv.DictWriter(handle, fieldnames=['dyad', 'status', 'detail'])
    writer.writeheader()
    writer.writerows(sorted(report, key=lambda row: row['dyad']))
print(f'Report written to {REPORT_CSV}')
for row in sorted(report, key=lambda row: row['dyad']):
    print(f"  {row['dyad']}: {row['status']} {row['detail']}")
