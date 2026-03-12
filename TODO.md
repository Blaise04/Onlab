# Fejlesztési javaslatok

Az implementált javaslatok kerüljenek ki ebből a listából.

| # | Azonosító | Fájl | Státusz |
|---|-----------|------|---------|
| 1 | `seed-fix` | `src/data_generator.py:140` | [x] |
| 2 | `moneyness-sampling` | `src/data_generator.py` | [x] |
| 3 | `normalize-teljes` | `src/data_generator.py:117-119` | [x] |
| 4 | `unit-tesztek` | `tests/` (nem létezik) | [ ] |
| 5 | `model` | `src/model.py` (nem létezik) | [ ] |
| 6 | `train` | `src/train.py` (nem létezik) | [ ] |
| 7 | `evaluate` | `src/evaluate.py` (nem létezik) | [ ] |
| 8 | `eda` | `notebooks/eda.ipynb` (nem létezik) | [ ] |

---

### 1. `seed-fix`
**Fájl:** `src/data_generator.py:140`
**Probléma:** `save_dataset` shuffle-je `random_state=0`-ra van fix-elve, holott a CLI `--seed` értékét kellene használni.
**Megoldás:** `df.sample(frac=1, random_state=0)` → `df.sample(frac=1, random_state=seed)`, és a `seed` paraméter felvétele a függvény szignatúrájába.

---

### 2. `moneyness-sampling`
**Fájl:** `src/data_generator.py` – `DEFAULT_PARAMS`, `generate_dataset`
**Probléma:** S ([10, 150]) és K ([7, 650]) egymástól független mintavételezése sok irreális mélyen OTM/ITM kombinációt generál.
**Megoldás:** S/K (moneyness) ∈ [0.7, 1.3] tartományban mintavételezni, majd K-ból visszaszámolni S-t — ahogy a `docs/bs_szintetikus_adatok.md` §7 javasolja.

---

### 3. `normalize-teljes`
**Fájl:** `src/data_generator.py:117-119`
**Probléma:** `--normalize` csak `moneyness` és `call_price_norm` oszlopokat ad, nem normalizálja az összes inputot [0,1]-re.
**Megoldás:** Min-max scaler az összes inputra (T, r, sigma, q), vagy külön `--scale-inputs` flag.

---

### 4. `unit-tesztek`
**Fájl:** `tests/test_black_scholes.py` (nem létezik)
**Probléma:** Nincs teszt a BS formulákra — egy numerikus hiba néma hibát okozhat.
**Megoldás:** put-call paritás, ATM call ár ismert értékkel, görög szimmetriák (delta ∈ [0,1], gamma ≥ 0).

---

### 5. `model`
**Fájl:** `src/model.py` (nem létezik)
**Probléma:** Neurális háló architektúra teljesen hiányzik.
**Megoldás:** MLP PyTorch-ban (4-5 hidden layer, ReLU), Culkin & Das (2017) alapján.

---

### 6. `train`
**Fájl:** `src/train.py` (nem létezik)
**Probléma:** Training loop hiányzik.
**Megoldás:** CSV betöltés, DataLoader, Adam optimizer, MSE loss, epoch loop, checkpoint mentés.

---

### 7. `evaluate`
**Fájl:** `src/evaluate.py` (nem létezik)
**Probléma:** Kiértékelési script hiányzik.
**Megoldás:** MAE, RMSE, R² a test seten, összehasonlítás az analitikus BS árral.

---

### 8. `eda`
**Fájl:** `notebooks/eda.ipynb` (nem létezik)
**Probléma:** Az adathalmaz eloszlásáról nincs vizualizáció.
**Megoldás:** Jupyter notebook: input hisztogramok, call/put ár eloszlás, moneyness vs. call ár scatter.
