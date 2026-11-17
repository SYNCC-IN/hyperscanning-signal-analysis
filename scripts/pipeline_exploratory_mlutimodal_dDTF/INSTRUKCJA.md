# Instrukcja: pipeline sprzężeń mózg–mózg i serce–serce (dDTF / GPDC)

Pipeline liczy skierowane sprzężenia między dzieckiem a opiekunem podczas oglądania trzech filmów (Peppa,
Incredibles, Brave). Dla każdej diady i filmu dopasowuje jeden model MVAR do kilku sygnałów: obwiedni rytmu EEG
w wybranym ROI oraz surowego IBI (serce). Wyniki porównuje z pseudo-diadami (dziecko z jednej diady, opiekun z innej),
a następnie testuje permutacyjnie. Ustawienia są zgodne z manuskryptem *Multimodal_MVAR_sym* i z symulacjami
(`simulations_multimodal_MVAR/`).

Wszystkie ustawienia są w jednym pliku: [`pipeline_config.json`](pipeline_config.json) (sekcja `"shared"` dla
wszystkich etapów + sekcja każdego etapu).

---

## 1. Przygotowanie

```bash
git clone https://github.com/SYNCC-IN/hyperscanning-signal-analysis.git
cd hyperscanning-signal-analysis
git checkout pipeline_manuscript_alignment
python -m venv .venv
.venv/bin/pip install -r requirements.txt
```

Potrzebny jest dostęp do danych projektu SYNCC-IN na Google Drive (folder *WP4 – Joint study / UniWAW Data
collection*), zamontowanego lokalnie (np. Google Drive for desktop).

Wszystkie polecenia uruchamiamy z katalogu głównego repozytorium.

---

## 2. Gdzie są ścieżki do danych

### 2.1. Dane wejściowe (EEG i IBI)

Ścieżka bazowa to `DRIVE_ROOT` w sekcji `"shared"` pliku `pipeline_config.json`. Wskazuje folder
*UniWAW Data collection* na Google Drive. Z niej etapy 01 i 02 budują:

| dane | ścieżka |
|---|---|
| EEG po ICA (ciągłe fragmenty `passive_movies`) | `<DRIVE_ROOT>/UNIWAW_EEG_exported_BY_TASKS/ICA_output/EEG_ICA_CLEANED/<dyad_id>/` |
| IBI (serce) | `<DRIVE_ROOT>/UNIWAW_EEG_exported_BY_TASKS/IBI/<dyad_id>/` |

Wpisana w repozytorium wartość `DRIVE_ROOT` to ścieżka na komputerze autora. **Nie trzeba jej zmieniać w
repozytorium.** Swoją ścieżkę podaje się na jeden z dwóch sposobów (pierwszy wygrywa):

1. zmienna środowiskowa:
   ```bash
   export SYNCC_DRIVE_ROOT="/ścieżka/do/SYNCC-IN/WP4          - Joint study/UniWAW Data collection"
   ```
   (uwaga: w nazwie folderu *WP4* jest kilka spacji, trzeba je zachować);
2. plik `scripts/pipeline_exploratory_mlutimodal_dDTF/pipeline_config.local.json` (git go ignoruje):
   ```json
   {"shared": {"DRIVE_ROOT": "/ścieżka/do/SYNCC-IN/WP4          - Joint study/UniWAW Data collection"}}
   ```

### 2.2. Indywidualne pasma (wymagane dla węzłów EEG)

Obwiednia EEG liczona jest w **indywidualnym** paśmie rytmu każdej osoby. Pasma pochodzą z osobnego pipeline'u
spektralnego [`scripts/pipeline_Exploratory_spectral_analysis/`](../pipeline_Exploratory_spectral_analysis/),
który trzeba uruchomić wcześniej (kroki 01–04). Wynik:

```
Exploratory_spectral_analysis/04_band_assignment/band_assignments.csv
```

Uwaga: w tym pipeline ścieżka do EEG jest wpisana na sztywno na górze skryptów
`01_compute_psd.py` i `02_individual_specparam.py` (zmienna `DATA_DIR`) – trzeba ją zmienić na swoją
(`<DRIVE_ROOT>/UNIWAW_EEG_exported_BY_TASKS/ICA_output/EEG_ICA_CLEANED`).

### 2.3. Lista diad

`"shared".included_dyads` w `pipeline_config.json` – ręcznie wybrana lista diad (po kontroli jakości z etapu 01).
Etap 01 drukuje i zapisuje propozycję listy na podstawie kompletności danych.

### 2.4. Gdzie lądują wyniki

W repozytorium, w katalogu `Interbrain_ffDTF_analysis/` (nazwa z `ANALYSIS_ROOT_NAME`; katalog jest ignorowany przez
git), po jednym podkatalogu na etap:

| etap | katalog | najważniejsze pliki |
|---|---|---|
| 01 | `01_coverage/` | `coverage.csv` – kompletność danych, okna filmów |
| 02 | `02_envelopes/` | `<dyad>_<film>.nc` – sygnały wejściowe modelu (2,5 Hz) |
| 03 | `03_mvar/` | diagnostyka dopasowania, `mvar_order_gate.html` |
| 04 | `04_connectivity/<estymator>/` | opisowe kostki łączności, `connectivity_gate.html` |
| 05 | `05_surrogate/` | tabele par `<estymator>/<film>_pairing_table.nc`, Δ i z: `stage05_delta_table_<null>.csv` |
| 06 | `06_group/` | (opcjonalnie) bayesowskie wielkości efektów |
| 07 | `07_permutation/` | **wyniki główne**: `<estymator>/stage07_tests.csv`, `permutation_gate.html` |

Pliki `*.html` to „bramki” – raporty do obejrzenia w przeglądarce po każdym etapie.

---

## 3. Uruchomienie

```bash
.venv/bin/python scripts/pipeline_exploratory_mlutimodal_dDTF/stage01_coverage.py
.venv/bin/python scripts/pipeline_exploratory_mlutimodal_dDTF/stage02_envelopes.py
.venv/bin/python scripts/pipeline_exploratory_mlutimodal_dDTF/stage03_mvar_order.py
.venv/bin/python scripts/pipeline_exploratory_mlutimodal_dDTF/stage04_connectivity.py
.venv/bin/python scripts/pipeline_exploratory_mlutimodal_dDTF/stage05_surrogate.py
.venv/bin/python scripts/pipeline_exploratory_mlutimodal_dDTF/stage07_permutation_tests.py
.venv/bin/python scripts/pipeline_exploratory_mlutimodal_dDTF/stage06_group_model.py   # opcjonalny, ok. 15 min
```

- Etapy 01–02 czytają dane z Google Drive; od etapu 03 wszystko liczy się z plików w `Interbrain_ffDTF_analysis/`.
- Etapy 03 i 04 są opisowe (diagnostyka, rysunki). Wnioskowanie to etapy 05 → 07.
- Przed syntetycznym sprawdzeniem end-to-end (ok. 5 min) trzeba wygenerować referencyjne wyniki pełnych symulacji S1 i S3 dla dDTF oraz GPDC:
  `.venv/bin/python simulations_multimodal_MVAR/dDTF/run_all.py full 1 3`
  Wyniki znajdują się w ignorowanym przez git katalogu `simulations_multimodal_MVAR/dDTF/Results/`; etap 00c sprawdza,
  czy wymagane pliki CSV istnieją, zanim rozpocznie kosztowne symulowane studia.
- Sprawdzenie, że kod działa poprawnie (dane syntetyczne o znanej prawdzie):
  ```bash
  .venv/bin/python scripts/pipeline_exploratory_mlutimodal_dDTF/stage00c_end_to_end.py 20 50
  .venv/bin/python -m pytest tests/test_pipeline_alignment.py
  ```

### Ustawienia analizy (nie zmieniać bez powodu – są zwalidowane w symulacjach)

W `"shared"`: `TARGET_SFREQ` 2,5 Hz, okna `WIN_LEN_S` 10 s z nakładaniem `OVERLAP_FRAC` 0,5, `FIT_METHOD` `ols`,
`DETREND_TYPE` `none`, rząd modelu `COMMON_MODEL_ORDER` 4, pasmo sprzężeń `COUPLING_BAND_HZ` 0,15–0,5 Hz,
`BOX_COX_LAMBDA` 0,25, estymatory `ESTIMATORS` GPDC i dDTF (GPDC główny dla wniosków o pojedynczych krawędziach).
W `stage07_permutation_tests`: liczba permutacji, ziarno, rodzina testów (`FAMILY_EDGES`, domyślnie wszystkie
krawędzie między dzieckiem a opiekunem) i kontrasty filmowe (`FILM_CONTRASTS`). **Rodzinę i kontrasty ustala się
przed obejrzeniem wyników.**

---

## 4. Gdzie są definicje ROI i elektrod

### 4.1. Węzły modelu – `"shared".nodes`

To jest główne miejsce, w którym decyduje się, **jakie sygnały wchodzą do modelu**. Obecnie:

```json
"nodes": [
  {"name": "child:ROI", "role": "child",     "signal": "roi_envelope", "roi_channels": ["P7"], "roi_label": "temporo-parietal"},
  {"name": "cg:ROI",    "role": "caregiver", "signal": "roi_envelope", "roi_channels": ["P7"], "roi_label": "temporo-parietal"},
  {"name": "child:HRV", "role": "child",     "signal": "raw_ibi"},
  {"name": "cg:HRV",    "role": "caregiver", "signal": "raw_ibi"}
]
```

Pola węzła:

| pole | znaczenie |
|---|---|
| `name` | nazwa węzła; **konwencja `child:<etykieta>` / `cg:<etykieta>`** (część po dwukropku to typ sygnału, używana w testach struktury krawędzi) |
| `role` | `child` albo `caregiver` |
| `signal` | `roi_envelope` (obwiednia rytmu EEG) albo `raw_ibi` (surowe IBI, serce) |
| `roi_channels` | **elektrody** ROI (nazwy z systemu 10-20, jak w plikach EEG); obwiednie z tych elektrod są uśredniane (`ROI_REDUCTION`: `average_envelopes`) |
| `roi_label` | **z którego ROI brać indywidualne pasmo** w `band_assignments.csv`; musi być jedną z etykiet niżej |

Pasmo rytmu: `"stage02_envelopes".BAND` (`fast` = szybszy rytm, okolice alfa; `slow` = wolniejszy).

### 4.2. Definicje ROI dla pasm indywidualnych

Słownik ROI → elektrody jest w pipeline spektralnym,
[`04_band_assignment.py`](../pipeline_Exploratory_spectral_analysis/04_band_assignment.py) (zmienna `ROI_CHANNELS`):

| `roi_label` | elektrody, na których wyznaczono pasmo |
|---|---|
| `frontal-midline` | Fz |
| `sensorimotor` | C3, C4 |
| `central-midline` | Cz |
| `parietal` | P3, Pz, P4 |
| `occipital` | O1, O2 |
| `lateral-temporal` | T7, T8 |
| `temporo-parietal` | P7, P8 |

Pasmo wyznaczone na elektrodach ROI (np. P7+P8) może być stosowane do obwiedni z innego podzbioru elektrod węzła
(obecnie `roi_channels: ["P7"]`). Nowy ROI (nowa etykieta) wymaga dopisania go w tym słowniku i ponownego
uruchomienia kroków 01–04 pipeline'u spektralnego.

### 4.3. ROI w etapie 01

`"shared".ROI_LABEL` / `ROI_CHANNELS` (i `"stage01_coverage".ROI_CHANNELS`) służą **tylko** kontroli
kompletności w etapie 01 (czy elektrody są w plikach). Wpisz tu sumę elektrod wszystkich węzłów EEG.

---

## 5. Zmiana zestawu sygnałów

**Zasada ogólna.** Zmieniamy `"shared".nodes` i `"shared".edge_topology`, potem uruchamiamy pipeline **od etapu 02**
(etap 01 tylko wtedy, gdy zmieniają się elektrody). Pozostałe etapy same dostosowują się do liczby węzłów.

- `edge_topology` to lista nazwanych krawędzi (źródło → cel) z etykietą klasy (`H2_primary`, `exploratory`, ...).
  Każda nazwa węzła musi tu istnieć w `nodes` (etapy to sprawdzają). Krawędzie spoza listy też są liczone; lista
  służy do nazwania i wyróżnienia krawędzi w raportach.
- Testowana rodzina (max-T) to domyślnie **wszystkie krawędzie między osobami** (dziecko → opiekun i odwrotnie).
  Krawędzie wewnątrz jednej osoby są tylko opisywane – pseudo-diady nie dają dla nich testu.
- Wyniki każdego wariantu najlepiej trzymać osobno: zmień `"shared".ANALYSIS_ROOT_NAME` (np. `Interbrain_roi_only`),
  wtedy wariant zapisze wszystko do nowego katalogu i nie nadpisze wyników analizy podstawowej.

Warianty a–c poniżej przeszły test na danych syntetycznych (etapy 05, 07 i 06). Etapy 01–02 nie są od nich
zależne inaczej niż przez listę węzłów, ale na danych rzeczywistych nie były w tych wariantach uruchamiane.

### a) Tylko ROI mózgowe (1 ROI na osobę, bez serca)

```json
"nodes": [
  {"name": "child:ROI", "role": "child",     "signal": "roi_envelope", "roi_channels": ["P7"], "roi_label": "temporo-parietal"},
  {"name": "cg:ROI",    "role": "caregiver", "signal": "roi_envelope", "roi_channels": ["P7"], "roi_label": "temporo-parietal"}
],
"edge_topology": [
  {"source": "cg:ROI",    "target": "child:ROI", "class": "H2_primary"},
  {"source": "child:ROI", "target": "cg:ROI",    "class": "H2_reverse"}
]
```

Model 2-zmienny, rodzina testów = 2 krawędzie, jedna para do testu kierunku. Testy struktury krawędzi
(kierunek / źródło / cel w bloku czynnikowym) nie mają sensu i są pomijane automatycznie.

### b) Kilka ROI mózgowych

Każdy ROI to osobny węzeł dla każdej osoby, np. skroniowo-ciemieniowy i czołowy:

```json
"nodes": [
  {"name": "child:TP", "role": "child",     "signal": "roi_envelope", "roi_channels": ["P7", "P8"], "roi_label": "temporo-parietal"},
  {"name": "cg:TP",    "role": "caregiver", "signal": "roi_envelope", "roi_channels": ["P7", "P8"], "roi_label": "temporo-parietal"},
  {"name": "child:FM", "role": "child",     "signal": "roi_envelope", "roi_channels": ["Fz"],       "roi_label": "frontal-midline"},
  {"name": "cg:FM",    "role": "caregiver", "signal": "roi_envelope", "roi_channels": ["Fz"],       "roi_label": "frontal-midline"}
]
```

i `edge_topology` z krawędziami między tymi węzłami. Można też łączyć kilka ROI z sercem (np. 2 ROI + HRV = 6
węzłów). Na co uważać:

- **Liczba węzłów rośnie szybko w kosztach.** Przy k węzłach i rzędzie 4 każde równanie modelu ma 4·k regresorów,
  a na film przypada ok. 230 wierszy danych (11 okien po 25 próbek, minus rząd modelu w każdym oknie). Zwalidowano 4 węzły; 6 jest jeszcze rozsądne, 8 już raczej
  nie. Sprawdź w etapie 05 liczbę niestabilnych dopasowań (bramka `surrogate_gate.html`): rosnąca liczba
  odrzuconych diad to sygnał, że model jest za duży.
- **Rodzina testów rośnie** (dla m ROI na osobę: 2·m² krawędzi między osobami), co zmniejsza moc po korekcie max-T.
  Jeśli hipoteza dotyczy kilku krawędzi, zawęź `"stage07_permutation_tests".FAMILY_EDGES` (lista nazw krawędzi,
  np. `["cg:TP->child:TP", "child:TP->cg:TP"]`) – **przed** obejrzeniem wyników.
- Testy struktury krawędzi w bloku czynnikowym działają tylko przy dokładnie dwóch typach węzłów na osobę
  (np. TP i FM, albo ROI i HRV); przy innych układach są pomijane.

### c) Same sygnały sercowe

```json
"nodes": [
  {"name": "child:HRV", "role": "child",     "signal": "raw_ibi"},
  {"name": "cg:HRV",    "role": "caregiver", "signal": "raw_ibi"}
],
"edge_topology": [
  {"source": "cg:HRV",    "target": "child:HRV", "class": "H4_primary"},
  {"source": "child:HRV", "target": "cg:HRV",    "class": "H4_reverse"}
]
```

Uwagi:
- **Pliki EEG są nadal potrzebne**: z nich etapy 01–02 biorą znaczniki początku i końca filmów (synchronizacja).
  Etap 02 wczytuje też `band_assignments.csv` na starcie (plik musi istnieć, choć nie jest używany).
- Sygnał sercowy to surowe IBI (opcja A manuskryptu). Istotna krawędź mówi, że rytm serca jednej osoby poprawia
  predykcję rytmu drugiej; nie mówi, jakim mechanizmem.

---

## 6. Jak czytać wyniki (skrót z manuskryptu, Box 2)

- Wyniki główne: `07_permutation/permutation_gate.html` – dDTF i GPDC obok siebie; pogrubione = istotne po korekcie
  max-T. Δ i z z etapu 05 to wielkości efektu.
- **GPDC** dla wniosków o pojedynczych krawędziach. **dDTF** pokazuje, czy osoby są sprzężone – przy sprzężeniu w
  jedną stronę dDTF często pokazuje też krawędź odwrotną; kierunek odczytuj z testu 3.
- Krawędzie wewnątrz jednej osoby nie są testowane.
- Wspólny czynnik niezwiązany z filmem (np. oddech, którego nie rejestrowano) może udawać sprzężenie między osobami.
- Kowariaty (wiek) – tylko eksploracyjnie (`ADD_AGE_COVARIATE`).
- Raportuj: liczbę stabilnych dopasowań i wykluczonych diad na film, p surowe i skorygowane, Δ/z, zakres nulla,
  rodzinę testów i estymator. Szczegóły i wyniki pierwszego przebiegu:
  [`REPORT_pipeline_alignment_2026-11-16.md`](REPORT_pipeline_alignment_2026-11-16.md).
