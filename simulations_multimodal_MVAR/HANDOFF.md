# Handoff: simulations for the multimodal dDTF methods paper (stan: 2026-11-01)

## Cel
Artykuł metodologiczny (NeuroImage, Original research / methodological) o multimodalnym dDTF w hyperscanningu
(obwiednie EEG + IBI/HRV, dyady opiekun–dziecko, SYNCC-IN). Cały materiał dowodowy to dane syntetyczne i symulacje.
Pasma indywidualne (specparam) i dane rzeczywiste idą do osobnych artykułów; oddech tylko w Dyskusji.
Draft: `docs/draft_v02_ddtf_multimodal_hyperscanning_simulations.md` (sekcje `[RESULT]` czekają na wyniki; v0.1: `docs/draft_methods_multimodal_dDTF_hyperscanning.md`).

## UWAGA: po przeniesieniu katalogu do `simulations_multimodal_MVAR/dDTF/`
Skrypty `S01…S11` mają `PROJECT_ROOT = Path(__file__).resolve().parents[1]`, co teraz wskazuje na `simulations_multimodal_MVAR`, a nie na korzeń repo
(importy `src.*` przestaną działać). Trzeba zmienić na `parents[2]`. `run_all.py` i README nadal mówią o `simulations/…` — zaktualizować ścieżki.
`git status` pokazuje stare `simulations/*` jako usunięte (przeniesione).

## Kod
- `src/sim_generators.py`, `sim_cardiac.py`, `sim_estimation.py`, `sim_inference.py`, `sim_harness.py`
  (`sim_estimation.edge_matrix` przyjmuje też `estimator` jako callable — konkurenci w S11).
- `simulations_multimodal_MVAR/dDTF/S01…S11_*.py` (numeracja = Tabela 2), `run_all.py`, `Results/S_xx/` (pełny przebieg: 200 replikacji; S02, S09–S11: 100; ≈3,5 h na 4 rdzeniach, S04 ≈ 1,7 h).
- Testy: `tests/test_sim_estimation.py`, `tests/test_ddtf_theory.py` (7 przechodzi). Inne testy repo nie zbierają się na Pythonie 3.10 (`StrEnum`) — problem środowiska VM.
- Konwencje: macierze `[target, source]`; węzły poziomu 1: 0 child ROI, 1 cg ROI, 2 child HRV, 3 cg HRV; wspólne p=4 (SC tylko w S08); Box-Cox 0,25; pasmo 0,15–0,5 Hz [TBD]; progi kryteriów `[proposed]`.
- `SIM_NREP=<n>` nadpisuje liczbę replikacji (S03 dodatkowo `SIM_NDYADS`, `SIM_SEG`).

## Wyniki pełnych przebiegów (skrót)
- Kalibracja (S03): mediana z ≈ 0, sd z 0,94–0,99, FPR dyady 0,04–0,05, FPR badania 0,05–0,08 (lekki nadmiar, rośnie z N).
- Moc (S04): jedna grupa ≈1 od sprzężenia 0,2; kontrast grup N=10 tylko 0,3–0,56, N≥25 ≈1.
- S01: krawędzie zaplanowane wykrywane 100%. S06: artefakt gładkości do 0,20 przed korektą, ≈0 po korekcie surogatami.
- S07: null łączony daje fałszywy kontrast grup (FPR 0,22–0,70 przy różnym wzmocnieniu bodźca), `within_group` 0,01–0,05; null z innego filmu zawyża FPR (0,15–0,24).
- S08: stabilne wobec estymatora, rzędu, okna, pasma; brak Box-Cox → FPR 0,23 (Box-Cox obowiązkowy).
- S02/S11: obwiednia niewrażliwa na różnicę częstości; widzi sprzężenie amplitudowe, nie fazowe; PLV odwrotnie (faza 100%, amplituda 6%). Ścieżka fazowa S02 zależy od siatki (wartość bez sprzężenia spada z różnicą częstości) — nie interpretować jako porównania z obwiednią.
- S09: pulling wykrywany w 100% przy każdej różnicy częstości (K=2π·0,15 Hz); sprzężenie amplitudowe 0,25–0,5.
- S10: przewidywanie "A widzi tylko linear, A′ tylko amplitudę" NIE potwierdzone — rozdzielenie częściowe, zależne od siły (A: amp_to_eeg 0,25–0,97; A′: linear 0,24–1,0; B nieco czystszy).

## Główny otwarty problem: przeciek kierunkowy / kaskady
Kaskada 1→0→2: krawędź nieobecna 1→2 wykrywana w 100% badań (dyad 0,32–0,36), odwrotna 0→1 w ok. 68% (S08). Dotyczy też DTF/ffDTF i konkurentów w S11.
Diagnoza (BŁĘDU W IMPLEMENTACJI NIE MA):
- dDTF = ffDTF · |κ|, κ z S⁻¹ = AᴴV⁻¹A; zgodne z niezależnym obliczeniem z S=HVHᴴ (1e-12) i z repo.
- Dla prawdziwych współczynników kaskady κ₁₂ = 3e-16, dDTF 1→2 = 0 (ffDTF 1→2 = 0,0044 ≠ 0).
- Dla estymaty |κ| maleje jak 1/√n (0,26 / 0,09 / 0,05 przy 150 / 1500 / 15000 próbek); dDTF 1→2 ≈ 13% wartości bezpośredniej.
- Test surogatów: w surogacie ffDTF 1→2 ≈ 0 (0,00005 vs 0,0014 real), więc szum κ × duży przepływ całkowity daje ~28× większą wartość niż null → wykrycie; dłuższe nagrania pogarszają (600 s: dyad 0,99).
- GPDC w tym samym teście: krawędź nieobecna 0% (60 s i 300 s), odwrotna 0% / 23%; moc krawędzi bezpośredniej 100%.
Skrypty diagnostyczne były jednorazowe (na VM); odtworzyć z `run_study` + `estimator="GPDC"` (kaskada: krawędzie (1,0,1,.4),(0,2,1,.4), N=25, 60 i 300 s).

## Następne kroki
1. Naprawić ścieżki po przeniesieniu (patrz UWAGA), uruchomić `run_all.py quick`.
2. Dodać GPDC jako pełny wariant w S05, S08, S11; porównać kalibrację, moc, przeciek z dDTF; zdecydować: dDTF, GPDC, albo dDTF (siła) + GPDC (kierunek).
3. Jeśli dDTF zostaje: alternatywny null/statystyka (np. koherencja cząstkowa z korektą obciążenia).
4. Naprawić/zastąpić ścieżkę fazową w S02.
5. Zaktualizować draft v0.2: wypełnić `[RESULT]`; "direct" → sformułowanie o sprzężeniu międzyosobowym; Tabela 1 i S10 (częściowe rozdzielenie); within-group i film-matched null jako wynik S07; N≥25 dla kontrastu grup; Box-Cox obowiązkowy; p=4 (nie SC); odczyt signed-rank; S10 na poziomie obwiedni; usunąć "załamanie" ścieżki fazowej.
6. Zafiksować (zarejestrować) progi kryteriów przed finalną interpretacją.
