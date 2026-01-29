# Hyperscanning Signal Analysis - Architecture Diagram

## System Architecture Overview

```mermaid
flowchart TB
    subgraph Input["Data Sources"]
        EEG["EEG/ECG Files<br/>(SVAROG format)<br/>*.obci, *.xml, *.tag"]
        ET["Eye-Tracking Files<br/>(CSV format)<br/>annotations, gaze, pupil, blinks"]
    end

    subgraph Load["Data Loading Layer"]
        direction LR
        CEG["create_multimodal_data()<br/>Main entry point"]
        LEG["load_eeg_data()<br/>- Read SVAROG files<br/>- Scan for events<br/>- Mount to M1/M2<br/>- Apply filters"]
        LET["load_et_data()<br/>- Read CSV files<br/>- Align timestamps<br/>- Process per task<br/>- Interpolate blinks"]
    end

    subgraph Process["Data Processing Layer"]
        direction TB
        FILT["Filtering<br/>- Bandpass (1-40 Hz)<br/>- Notch (50 Hz)<br/>- FIR/IIR options"]
        ECG["ECG Extraction<br/>- Extract EKG channels<br/>- High-pass filter<br/>- Compute IBI"]
        EVT["Event Detection<br/>- Diode signal analysis<br/>- Threshold detection<br/>- Event timing"]
        DEC["Decimation<br/>- Anti-aliasing filter<br/>- Downsample by factor q<br/>- Update fs"]
    end

    subgraph Storage["Data Structure Layer"]
        direction TB
        MMD["MultimodalData Object"]
        DF["pandas.DataFrame<br/>Unified storage for all signals"]
        META["Metadata<br/>- Sampling freq (fs)<br/>- Channel names<br/>- Events list<br/>- Filter params"]
    end

    subgraph Output["Data Access Layer"]
        direction LR
        GS["get_signals()<br/>- Filter by mode<br/>- Filter by member<br/>- Filter by events<br/>- Filter by time"]
        GE["get_events_as_marker()<br/>- Event to int mapping<br/>- Marker channel<br/>- Time alignment"]
        GD["get_data()<br/>- Extract EEG array<br/>- Get channel names<br/>- Transpose to [ch × samples]"]
        GME["export_eeg_to_mne_raw()<br/>- Create MNE Raw<br/>- Add annotations<br/>- Set montage<br/>- Add filter info"]
    end

    subgraph Analysis["Analysis Layer"]
        direction LR
        HRV["HRV Analysis<br/>- IBI signals<br/>- DTF computation"]
        EEG_A["EEG Analysis<br/>- Multi-channel DTF<br/>- Spectral analysis"]
        COMB["Combined Analysis<br/>- EEG + HRV DTF<br/>- Theta band extraction"]
    end

    %% Main data flow
    EEG --> CEG
    ET --> CEG
    CEG --> LEG
    CEG --> LET
    
    LEG --> FILT
    FILT --> ECG
    LEG --> EVT
    
    LEG --> MMD
    LET --> MMD
    ECG --> MMD
    EVT --> MMD
    
    MMD --> DF
    MMD --> META
    
    CEG --> DEC
    DEC --> MMD
    
    DF --> GS
    DF --> GE
    DF --> GD
    DF --> GME
    
    GS --> HRV
    GS --> EEG_A
    GS --> COMB
    GD --> EEG_A
    
    GME --> EXT["External Tools<br/>(MNE-Python)"]

    style Input fill:#e1f5ff
    style Load fill:#fff3e0
    style Process fill:#f3e5f5
    style Storage fill:#e8f5e9
    style Output fill:#fff9c4
    style Analysis fill:#ffe0b2
```

## Detailed Component Diagram

```mermaid
flowchart LR
    subgraph DataFrame["MultimodalData.data (pandas.DataFrame)"]
        direction TB
        TC["Time Columns<br/>- time<br/>- time_idx"]
        EEGC["EEG Channels<br/>- EEG_ch_Fp1, Fp2, ...<br/>- EEG_cg_Fp1, Fp2, ..."]
        ECGC["ECG/IBI<br/>- ECG_ch, ECG_cg<br/>- IBI_ch, IBI_cg"]
        ETC["Eye-Tracking<br/>- ET_ch_x, y, pupil<br/>- ET_cg_x, y, pupil<br/>- ET_ch_blinks<br/>- ET_cg_blinks"]
        EVTC["Events<br/>- events (string)<br/>- EEG_events<br/>- ET_event"]
        DIODE["Diode<br/>- diode"]
    end

    subgraph Methods["Key Methods"]
        direction TB
        SET["Data Setters<br/>set_eeg_data()<br/>set_ecg_data()<br/>set_ibi()<br/>set_diode()<br/>set_EEG_events_column()"]
        GET["Data Getters<br/>get_signals()<br/>get_data()<br/>get_events_as_marker()"]
        PROC["Processing<br/>decimate_signals()<br/>interpolate_ibi_signals()<br/>create_events_column()<br/>create_event_structure()"]
        EXPORT["Export<br/>export_eeg_to_mne_raw()"]
    end

    DataFrame --> Methods
    
    style DataFrame fill:#c8e6c9
    style Methods fill:#b3e5fc
```

## Signal Processing Pipeline

```mermaid
flowchart TD
    START["Raw EEG Data<br/>(n_channels × n_samples)"]
    
    START --> MOUNT["Reference Mounting<br/>Subtract 0.5×(M1 + M2)<br/>Separately for child & caregiver"]
    
    MOUNT --> DESIGN["Filter Design<br/>- Bandpass: lowcut to highcut<br/>- Notch: 50 Hz (Q=30)<br/>- Type: FIR or IIR"]
    
    DESIGN --> APPLY["Apply Filters<br/>- filtfilt for zero-phase<br/>- sosfiltfilt for IIR"]
    
    APPLY --> STORE["Store in DataFrame<br/>Each channel = column<br/>EEG_ch_* / EEG_cg_*"]
    
    STORE --> DEC_Q{Decimate?<br/>q > 1}
    
    DEC_Q -->|Yes| ANTI["Anti-aliasing Filter<br/>FIR lowpass: 0.8×(fs/2q)"]
    ANTI --> DOWN["Downsample<br/>Select every q-th sample"]
    DOWN --> UPDATE["Update fs<br/>fs_new = fs / q"]
    
    DEC_Q -->|No| READY["Ready for Analysis"]
    UPDATE --> READY
    
    READY --> ACCESS["Data Access via<br/>get_signals()"]
    
    style START fill:#ffccbc
    style READY fill:#c5e1a5
    style ACCESS fill:#fff59d
```

## Event Processing Flow

```mermaid
flowchart TD
    DIODE["Diode Signal<br/>(photo-sensor)"]
    
    DIODE --> THRESH["Threshold Detection<br/>Normalize & threshold at 0.75"]
    
    THRESH --> DERIV["Derivative Analysis<br/>Find rising/falling edges"]
    
    DERIV --> DETECT["Event Detection<br/>- Group consecutive high values<br/>- Compute start & duration<br/>- Assign names (Brave, Peppa, etc.)"]
    
    DETECT --> STORE_EEG["Store in events list<br/>Set EEG_events column"]
    
    ET_FILES["ET Annotation Files<br/>(CSV)"]
    ET_FILES --> PARSE["Parse ET Events<br/>Extract event names & timings"]
    PARSE --> STORE_ET["Set ET_event column"]
    
    STORE_EEG --> ALIGN["Align Time Columns<br/>Reset time to first event = 0"]
    STORE_ET --> ALIGN
    
    ALIGN --> MERGE["Create Unified Events<br/>Merge EEG & ET events<br/>Check consistency"]
    
    MERGE --> STRUCT["Create Event Structure<br/>events dict: name → {start, duration}"]
    
    STRUCT --> USE["Use in Analysis<br/>- Filter by event name<br/>- Select time windows<br/>- Create annotations"]
    
    style DIODE fill:#ffecb3
    style ET_FILES fill:#ffecb3
    style STRUCT fill:#c5e1a5
    style USE fill:#b2dfdb
```

## Data Access Patterns

```mermaid
flowchart TD
    USER["Researcher/Analyst"]
    
    USER --> LOAD["Load Data<br/>create_multimodal_data()"]
    
    LOAD --> MMD["MultimodalData Object<br/>with unified DataFrame"]
    
    MMD --> CHOICE{Access Pattern}
    
    CHOICE -->|"Mode-based"| MODE["get_signals()<br/>- mode='EEG'/'ECG'/'IBI'/'ET'<br/>- member='ch'/'cg'<br/>- selected_channels=[...]<br/>- selected_events=[...]<br/>- selected_times=(start, end)"]
    
    CHOICE -->|"Direct array"| DIRECT["get_data()<br/>Returns EEG data array<br/>[n_channels × n_samples]<br/>+ channel names list"]
    
    CHOICE -->|"Events as markers"| MARKER["get_events_as_marker()<br/>Returns time-aligned<br/>integer marker channel"]
    
    CHOICE -->|"MNE export"| MNE["export_eeg_to_mne_raw()<br/>- Select by event or time<br/>- Add margins<br/>- Include annotations<br/>- Set montage & filter info"]
    
    MODE --> ANALYSIS["Analysis Code<br/>- DTF computation<br/>- Spectral analysis<br/>- Visualization"]
    
    DIRECT --> ANALYSIS
    MARKER --> VIS["Visualization<br/>Plot with event backgrounds"]
    MNE --> EXT_TOOLS["External Tools<br/>- MNE source localization<br/>- Time-frequency analysis<br/>- Connectivity analysis"]
    
    ANALYSIS --> RESULTS["Research Results"]
    VIS --> RESULTS
    EXT_TOOLS --> RESULTS
    
    style USER fill:#e1bee7
    style MMD fill:#c8e6c9
    style RESULTS fill:#ffcc80
```

## File Organization

```
hyperscanning-signal-analysis/
├── data/
│   └── {dyad_id}/
│       ├── eeg/
│       │   ├── {dyad_id}.obci      # Binary EEG data
│       │   ├── {dyad_id}.xml       # Channel configuration
│       │   └── {dyad_id}.tag       # Event markers
│       └── et/
│           ├── child/
│           │   ├── 000/            # Movies task
│           │   ├── 001/            # Talk 1 task
│           │   └── 002/            # Talk 2 task
│           └── caregiver/
│               ├── 000/
│               ├── 001/
│               └── 002/
│
├── src/
│   ├── dataloader.py              # Main data loading & processing
│   ├── data_structures.py         # MultimodalData class
│   ├── eyetracker.py              # ET-specific processing
│   ├── utils.py                   # Plotting & utility functions
│   └── mtmvar.py                  # DTF computation
│
├── scripts/
│   ├── mne_export_demo.ipynb      # MNE export examples
│   ├── get_data_demo.ipynb        # Data access examples
│   ├── filter_demo.ipynb          # Filter design & testing
│   ├── decimation_test.ipynb      # Decimation testing
│   ├── EEG_ET_synch_test.ipynb    # Synchronization checks
│   ├── Example_DataLoader_usage.ipynb
│   └── warsaw_pilot_data.py       # Analysis pipeline
│
├── tests/
│   ├── test_dataloader.py
│   └── test_data_structures.py
│
└── docs/
    ├── data_structure_spec.md
    └── architecture_diagram.md     # This file
```

## Key Design Principles

1. **Unified Storage**: All signals (EEG, ECG, IBI, ET) stored in single DataFrame
2. **Common Sampling**: All signals resampled to common `fs` (typically 1024 Hz or decimated)
3. **Time Alignment**: Time column aligned so first movie event starts at t=0
4. **Flexible Access**: Multiple methods to retrieve data by mode, member, event, or time
5. **Immutable Decimation**: `decimate_signals()` returns new object, preserves original
6. **Event Integration**: Events from both EEG (diode) and ET (annotations) merged and validated
7. **MNE Compatibility**: Export to MNE format with proper annotations and filter info
8. **Modular Processing**: Separate functions for loading, filtering, and processing each modality

## Signal Flow Summary

```
Raw Files → Load → Mount/Reference → Filter → Extract ECG/IBI → 
Store in DataFrame → Decimate (optional) → Align Events → 
Access via get_signals()/get_data() → Analysis (DTF, HRV, etc.) → Results
```

## Common Usage Patterns

### Pattern 1: Load and Extract EEG for Event
```python
# Load data
mmd = create_multimodal_data(data_base_path, dyad_id)

# Get EEG for specific event
time, channels, data = mmd.get_signals(
    mode='EEG', 
    member='ch',
    selected_channels=['Fz', 'Cz', 'Pz'],
    selected_events=['Brave']
)
```

### Pattern 2: Export to MNE with Time Selection
```python
# Export specific time window
raw, times = export_eeg_to_mne_raw(
    mmd, 
    who='cg',
    times=(100.0, 300.0)  # 100s to 300s
)
raw.plot()
```

### Pattern 3: Export Event with Margins
```python
# Export event with margins for baseline
raw, times = export_eeg_to_mne_raw(
    mmd, 
    who='ch',
    event='Incredibles',
    margin_around_event=10.0  # 10s before and after
)
# MNE annotations will include event timing
raw.annotations
```

### Pattern 4: Direct Array Access
```python
# Get EEG as numpy array
eeg_data, channel_names = get_data(
    df=mmd.data,
    who='ch'
)
# eeg_data shape: [n_channels × n_samples]
# channel_names: ['Fp1', 'Fp2', 'F7', ...]
```

### Pattern 5: Multi-modal Analysis
```python
# Get synchronized signals
eeg_time, eeg_ch, eeg_data = mmd.get_signals('EEG', 'ch', ...)
ibi_time, ibi_ch, ibi_data = mmd.get_signals('IBI', 'ch', ...)
et_time, et_cols, et_data = mmd.get_signals('ET', 'ch', ...)
```

---

*This architecture supports hyperscanning (dual-EEG) analysis with synchronized eye-tracking and cardiac signals for child-caregiver dyad studies.*
