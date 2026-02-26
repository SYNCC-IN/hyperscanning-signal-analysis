# HYPERSCANNING_SIGNAL_ANALYSIS
A set of tools to analyze multimodal data recorded in hyperscanning experiments of diads in SYNCC-IN project.


In this repo, we develop Python tools to operate and analyze multimodal data, i.e.:
- EEG
- ECG
- IBI - Inter bit intervals
- ET - eye-trackers
  
Currently, the tools are tailored for the experimental setup executed at the University of Warsaw as a part of the SYNCC-IN project.

These experiments consist of three major parts: SECORE, passive MOVIE viewing, and free TALK.
The exemplary processing 'scripts warsaw_pilot_data.py' and 'warsaw_pilot_data_with_ICA.py' require that in your local repo there is a folder 'DATA' containing exemplary diade data 'W_010'

We hope that they can be adapted to the paradigms of other Partners.

## Data structure update (v2.4)

`MultimodalData.eeg_filtration` uses nested dictionaries instead of flat fields.

- `eeg_filtration.notch`: `{"Q", "freq", "a", "b", "applied"}`
- `eeg_filtration.low_pass`: `{"type", "a", "b", "applied"}`
- `eeg_filtration.high_pass`: `{"type", "a", "b", "applied"}`

Example:

```python
notch_freq = multimodal_data.eeg_filtration.notch["freq"]
high_pass_type = multimodal_data.eeg_filtration.high_pass["type"]
is_low_pass_applied = multimodal_data.eeg_filtration.low_pass["applied"]
```

For full details, see [docs/data_structure_spec.md](docs/data_structure_spec.md).

## Xarray export quickstart

```python
from src.dataloader import create_multimodal_data
from src.export import export_to_xarray

md = create_multimodal_data(
	data_base_path='./data',
	dyad_id='W030',
	load_eeg=True,
	load_et=True,
	decimate_factor=8,
)

data_xr = export_to_xarray(
	multimodal_data=md,
	selected_event='Incredibles',
	selected_channels=['Fz', 'Cz', 'Pz'],
	selected_modality='EEG',
	member='ch',
	time_margin=10,
)

print(data_xr)
print(data_xr.attrs)
data_xr.plot.line(x='time', hue='channel')

# optional: z-score per channel over time
# from scipy.stats import zscore
# data_xr.data = zscore(data_xr.data, axis=0, nan_policy='omit')
```
