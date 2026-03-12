# Fejlesztési javaslatok

Az implementált javaslatok kerüljenek ki ebből a listából.

| # | Azonosító | Fájl | Státusz |
|---|-----------|------|---------|
| 1 | `seed-fix` | `src/data_generator.py:140` | [x] |
| 2 | `moneyness-sampling` | `src/data_generator.py` | [x] |
| 3 | `normalize-teljes` | `src/data_generator.py:117-119` | [x] |
| 4 | `unit-tesztek` | `tests/` (nem létezik) | [x] |
| 5 | `model` | `src/model.py` (nem létezik) | [ ] |
| 6 | `train` | `src/train.py` (nem létezik) | [ ] |
| 7 | `evaluate` | `src/evaluate.py` (nem létezik) | [ ] |
| 8 | `eda` | `notebooks/eda.ipynb` (nem létezik) | [ ] |
| 9 | `put-call-validacio` | `src/data_generator.py` | [ ] |
| 10 | `relativ-hiba-metrika` | `src/evaluate.py` | [ ] |
| 11 | `baseline-összehasonlitas` | `src/model.py` | [ ] |
| 12 | `requirements` | `requirements.txt` (nem létezik) | [ ] |
| 13 | `config` | `config.yaml` (nem létezik) | [ ] |
| 14 | `pipeline-notebook` | `notebooks/pipeline.ipynb` (nem létezik) | [ ] |

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

---

### 9. `put-call-validacio`
**Fájl:** `src/data_generator.py`
**Probléma:** A generált adatokon nincs put-call paritás ellenőrzés — néma numerikus hiba esetén hibás adatokkal tanítunk.
**Megoldás:** Generálás után ellenőrizni, hogy `C - P ≈ S·e^(-qT) - K·e^(-rT)` minden sorra teljesül (tolerancia: 1e-6).

---

### 10. `relativ-hiba-metrika`
**Fájl:** `src/evaluate.py`
**Probléma:** MSE/MAE abszolút hibát mér, ami eltorzul a mélyen ITM/OTM opciók esetén.
**Megoldás:** Relatív hiba: `|pred - BS| / BS` — pénzügyileg relevánsabb metrika, érdemes a fő kiértékelési mérőszámnak választani.

---

### 11. `baseline-összehasonlitas`
**Fájl:** `src/model.py`
**Probléma:** Nincs baseline, amihez a modell teljesítményét viszonyítani lehet.
**Megoldás:** Egy egyszerű sekély MLP (1-2 réteg) baseline-ként, a mélyebb architektúrával összehasonlítva.

---

### 12. `requirements`
**Fájl:** `requirements.txt` (nem létezik)
**Probléma:** A projekt függőségei nincsenek rögzítve — másik gépen nem reprodukálható a környezet.
**Megoldás:** `requirements.txt` létrehozása: `numpy`, `scipy`, `pandas`, `torch`, `pyarrow`, `jupyter`, stb. verzióval.

---

### 13. `config`
**Fájl:** `config.yaml` (nem létezik)
**Probléma:** A hiperparaméterek (learning rate, batch size, rétegek száma stb.) szét vannak szórva a kódban.
**Megoldás:** Központi `config.yaml` fájl, amit a train/evaluate scriptek betöltenek — kód módosítás nélkül kísérletezhető.

---

### 14. `pipeline-notebook`
**Fájl:** `notebooks/pipeline.ipynb` (nem létezik)
**Probléma:** A teljes pipeline (generálás → tanítás → kiértékelés) nincs egy helyen bemutatva.
**Megoldás:** Jupyter notebook, ami végigvezet a teljes folyamaton — prezentációhoz és beadáshoz hasznos.
