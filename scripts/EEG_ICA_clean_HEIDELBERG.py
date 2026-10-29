# %% [markdown]
# # ICA cleaning of the exported Heidelberg EEG (passive_movies)
# Heidelberg counterpart of `EEG_ICA_clean_UNIWAW.py`. Reads the NCDF files written by
# `export_heidelberg_by_task_batch.py` and runs the three ICA stages of `ICAPreprocessor`:
#   1. FITTING      - fit ICA, save `<dyad>/<file>-ica.fif`
#   2. CLASSIFYING  - ICLabel + amplitude criteria, writes the QC figures and a CSV with an editable `exclude` column
#   -> manual QC: review the figures/CSV in ICA_QC_FIGS_and_CSV and correct the `exclude` column
#   3. APPLYING     - remove the reviewed components, write the cleaned NCDF files
import sys
from pathlib import Path

# repository root on the path so that `src` imports as a package
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.ica_preprocessing import ICAPreprocessor

# ---- Configuration ----
HEIDELBERG_DIR = Path('/Users/admin/Documents/Hoza/PROJEKTY/SYNCC_IN_LOCAL_HOME/HEIDELBERG')
EXPORT_DIR = HEIDELBERG_DIR / 'HEIDELBERG_EEG_exported_BY_TASKS'
ICA_DIR = EXPORT_DIR / 'ICA_output'
CLEANED_DIR = ICA_DIR / 'EEG_ICA_CLEANED'
TARGET_EVENTS = ['passive_movies']
DYAD_IDS = None          # list, e.g. ['H_009', 'H_046'], or None to process all dyads
SMOKE_TEST = False            # only used when DYAD_IDS is None: True = first SMOKE_DYADS_N dyads
SMOKE_DYADS_N = 2
N_COMPONENTS = 15
ICLABEL_THRESHOLD = 0.70
NEURAL_THRESHOLD = 0.5
AMPLITUDE_THRESHOLD = 150.0

proc = ICAPreprocessor(export_folder=EXPORT_DIR, target_events=TARGET_EVENTS)
proc.find_eeg_files(smoke_test=SMOKE_TEST, smoke_dyads_n=SMOKE_DYADS_N, dyad_ids=DYAD_IDS)

# %% [Stage 1] FITTING
proc.fit_and_save_ica(ICA_DIR, n_components=N_COMPONENTS)

# %% [Stage 2] CLASSIFYING
proc.classify_and_save_labels(
    ICA_DIR,
    iclabel_threshold=ICLABEL_THRESHOLD,
    neural_threshold=NEURAL_THRESHOLD,
    amplitude_threshold=AMPLITUDE_THRESHOLD,
)

# -> open the QC figures and CSV in ICA_QC_FIGS_and_CSV, check, edit `exclude`, save.
# Run Stage 3 only after this review (select the cell below and run it separately).

# %% [Stage 3] APPLYING
CLEANED_DIR.mkdir(parents=True, exist_ok=True)
proc.apply_ica_and_save(ICA_DIR, CLEANED_DIR)
