# Hallucináció-ellenőrzés — Önlabor Beszámoló

Ellenőrzés dátuma: 2026-05-16  
Ellenőrzött fájlok: `Beszámoló/Sablon/dolgozat_szovege.tex`, `Beszámoló/Sablon/hivatkozasok.bib`  
Referencia: `src/model.py`, `src/train.py`, `models/*.pt`, `results/training_history_*.json`, `data/test.csv`

---

## 1. Irodalmi hivatkozások

| # | Hivatkozás | Ítélet | Megjegyzés |
|---|-----------|--------|-----------|
| A | Black & Scholes (1973) | **OK** | |
| B | Hutchinson, Lo & Poggio (1994) | **FIGYELMEZTETÉS** | DOI hibás: `tb05149.x` → helyes: `tb00081.x` |
| C | Garcia & Gençay (2000) | **OK** | |
| D | Culkin & Das (2017) | **OK** | |
| E | Liu, Oosterlee & Bohte (2019) | **OK** | |
| F | Ruf & Wang (2020) | **FIGYELMEZTETÉS** | Oldalszám: pp. 1–45 → helyes: pp. 1–46 |
| G | Della Corte et al. (2023) | **OK** | arXiv:2307.07657 |
| H | Raissi et al. (2019) | **OK** | |
| I | Liao et al. (2024) | **OK** | arXiv:2412.12213 |
| J | Srivastava et al. (2015) | **OK** | |
| K | Huang et al. (2017) | **OK** | |
| L | McKay et al. (1979) | **OK** | |

---

## 2. Modell paraméterszámok

Számítás módja: `sum(p.numel() for p in model.parameters())` az aktuális `src/model.py` alapján.

| Modell | Dolgozat | Tényleges | Eltérés | Ítélet |
|--------|----------|-----------|---------|--------|
| MLPPricer | 31,001 | **30,901** | −100 | **HIBA** |
| DeepMLPPricer | 267,521 | **267,265** | −256 | **HIBA** |
| ResNetPricer | 398,593 | **399,873** | +1,280 | **HIBA** |
| GELUResNetPricer | 398,593 | **398,337** | −256 | **HIBA** |
| DenseMLPPricer | 101,894 | **101,381** | −513 | **HIBA** |
| HighwayPricer | 528,129 | **527,873** | −256 | **HIBA** |
| FINNPricer | 403,202 | **402,882** | −320 | **HIBA** |

Mind a 7 modell paraméterszáma téves. Az eltérések 256-os (hidden dim) többszöresei — valószínűleg a kód a dolgozat megírása után módosult, vagy a számok manuálisan lettek megbecsülve.

---

## 3. Numerikus eredmények

Forrás: `models/*.pt` checkpointok + `data/test.csv` (N = 150,000).

### 3a. Best Epoch

| Modell | Dolgozat | Tényleges | Ítélet |
|--------|----------|-----------|--------|
| MLPPricer | 68 | **123** | **HIBA** |
| DeepMLPPricer | 4 | **27** | **HIBA** |
| ResNetPricer | 80 | **28** | **HIBA** |
| GELUResNetPricer | 40 | **31** | **HIBA** |
| DenseMLPPricer | 18 | **5** | **HIBA** |
| HighwayPricer | 22 | **9** | **HIBA** |
| FINNPricer | 17 | **21** | FIGYELMEZTETÉS |

Helyes értékek forrása: `results/training_history_<ModelName>.json` → `best_epoch` mező.

### 3b. RMSE (teszthalmaz)

| Modell | Dolgozat | Tényleges | Ítélet |
|--------|----------|-----------|--------|
| MLPPricer | 0.004612 | **0.004602** | OK (~0.2%) |
| DeepMLPPricer | 0.016575 | **0.006061** | **KRITIKUS HIBA** (173% eltérés) |
| ResNetPricer | 0.004788 | **0.006217** | **HIBA** (30%) |
| GELUResNetPricer | 0.009135 | **0.009293** | FIGYELMEZTETÉS (1.7%) |
| DenseMLPPricer | 0.015172 | **0.018042** | **HIBA** (19%) |
| HighwayPricer | 0.015281 | **0.014884** | FIGYELMEZTETÉS (2.6%) |
| FINNPricer | 0.009653 | **0.009143** | FIGYELMEZTETÉS (5.3%) |

> **Kritikus:** A DeepMLPPricer a dolgozatban a legrosszabb modellként szerepel (RMSE=0.016575), miközben valójában a második legjobb (RMSE=0.006061), csak az MLPPricer előzi meg. A rangsor megfordul.

### 3c. MAE (teszthalmaz)

| Modell | Dolgozat | Tényleges | Ítélet |
|--------|----------|-----------|--------|
| MLPPricer | 0.002780 | **0.002788** | OK |
| DeepMLPPricer | 0.013233 | **0.004351** | **KRITIKUS HIBA** |
| ResNetPricer | 0.003274 | **0.004772** | **HIBA** |
| GELUResNetPricer | 0.006027 | **0.006248** | FIGYELMEZTETÉS |
| DenseMLPPricer | 0.010694 | **0.013094** | **HIBA** |
| HighwayPricer | 0.011005 | **0.010581** | FIGYELMEZTETÉS |
| FINNPricer | 0.006625 | **0.006134** | FIGYELMEZTETÉS |

### 3d. R² (teszthalmaz)

| Modell | Dolgozat | Tényleges | Ítélet |
|--------|----------|-----------|--------|
| MLPPricer | 0.999407 | **0.999449** | OK |
| DeepMLPPricer | 0.992338 | **0.999044** | **KRITIKUS HIBA** |
| ResNetPricer | 0.999361 | **0.998994** | FIGYELMEZTETÉS |
| GELUResNetPricer | 0.997673 | **0.997752** | OK |
| DenseMLPPricer | 0.993581 | **0.991528** | FIGYELMEZTETÉS |
| HighwayPricer | 0.993488 | **0.994234** | FIGYELMEZTETÉS |
| FINNPricer | 0.997401 | **0.997824** | FIGYELMEZTETÉS |

### 3e. Val MSE ×10⁻⁵

Forrás: `results/training_history_*.json` → `best_val_loss`.

| Modell | Dolgozat | Tényleges | Ítélet |
|--------|----------|-----------|--------|
| MLPPricer | 2.13 | **2.06** | OK |
| DeepMLPPricer | 27.40 | **3.63** | **KRITIKUS HIBA** |
| ResNetPricer | 2.28 | **3.85** | **HIBA** |
| GELUResNetPricer | 8.28 | **8.50** | OK |
| DenseMLPPricer | 22.80 | **32.28** | **HIBA** |
| HighwayPricer | 23.20 | **21.95** | OK |
| FINNPricer | 9.23 | **8.24** | FIGYELMEZTETÉS |

### 3f. ResNetPricer+PI — hiányzó modell

A dolgozat tartalmaz egy `ResNetPricer+PI` sort teljes eredménytáblával, de:
- Nincs checkpoint: `models/` mappában nem szerepel
- Nincs training history: `results/` mappában nem szerepel

**Az adatok nem reprodukálhatók** — ha a modellt valóban betanítottad, a checkpoint nem lett elmentve; ha nem, a számok forrása ismeretlen.

---

## 4. Matematikai képletek és hiperparaméterek

| Állítás | Ítélet |
|---------|--------|
| Black-Scholes call képlet (d₁, d₂) | **OK** |
| Put-call paritás: P = C − S + K·e^(−rT) | **OK** |
| Homogenitás: C(λS, λK,...) = λC | **OK** |
| GELU: x·Φ(x) | **OK** |
| Physics loss: ReLU(−δ) + ReLU(δ−1) | **OK** (egyezik `train.py:253–255`) |
| Highway gate: H·T + x·(1−T) | **OK** |
| Optimizer: Adam, lr=1e-3, weight\_decay=1e-4 | **OK** |
| Batch size: 4096, max\_epochs: 200, patience: 10 | **OK** |
| LR scheduler: factor=0.5, patience=5, min\_lr=1e-6 | **OK** |
| Adathalmaz split: 70/15/15 (700k/150k/150k) | **OK** |
| Adattartományok (S, moneyness, T, r, σ) | **OK** |
| Latin Hypercube Sampling (LHS) | **OK** |
| Seed: 42 | **OK** |

---

## Összefoglalás és teendők

| Kategória | OK | Figyelmeztetés | Hiba | Kritikus |
|-----------|----|----------------|------|---------|
| Irodalmi hivatkozások (12) | 10 | 2 | 0 | 0 |
| Paraméterszámok (7) | 0 | 0 | 7 | 0 |
| Best epoch (7) | 0 | 1 | 6 | 0 |
| RMSE / MAE / R² / Val MSE | 2 | 8 | 5 | 3 |
| Képletek & hiperparaméterek | mind OK | — | — | — |

### Prioritizált teendők

1. **[KRITIKUS]** DeepMLPPricer összes eredményét javítani: RMSE=0.006061, MAE=0.004351, R²=0.999044, Val MSE=3.63×10⁻⁵, best_epoch=27. A modell valójában a második legjobb, nem a legrosszabb — a szöveges értékelést is átírni.

2. **[KRITIKUS]** ResNetPricer+PI: vagy újra betanítani és elmenteni a checkpointot, vagy a sort törölni a táblázatokból.

3. **[HIBA]** Mind a 7 paraméterszámot frissíteni a fenti táblázat alapján.

4. **[HIBA]** Összes best_epoch értéket javítani a `results/training_history_*.json` fájlok alapján.

5. **[HIBA]** ResNetPricer és DenseMLPPricer RMSE/MAE értékek javítása.

6. **[FIGYELMEZTETÉS]** Hutchinson et al. (1994) DOI: `10.1111/j.1540-6261.1994.tb00081.x`

7. **[FIGYELMEZTETÉS]** Ruf & Wang (2020) oldalszám: pp. 1–46
