# Fejlesztési javaslatok

Az implementált javaslatok kerüljenek ki ebből a listából.

| # | Azonosító | Fájl | Státusz |
|---|-----------|------|---------|
| 1 | `seed-fix` | `src/data_generator.py:140` | [x] |
| 2 | `moneyness-sampling` | `src/data_generator.py` | [x] |
| 3 | `normalize-teljes` | `src/data_generator.py:117-119` | [x] |
| 4 | `unit-tesztek` | `tests/` | [x] |
| 5 | `model` | `src/model.py` | [x] |
| 6 | `train` | `src/train.py` | [x] |
| 7 | `evaluate` | `src/evaluate.py` | [x] |
| 8 | `requirements` | `requirements.txt` | [x] |
| 9 | `eda` | `notebooks/eda.ipynb` (nem létezik) | [ ] |
| 10 | `put-call-validacio` | `src/data_generator.py` | [ ] |
| 11 | `relativ-hiba-metrika` | `src/evaluate.py` | [ ] |
| 12 | `baseline-összehasonlitas` | `src/model.py` | [ ] |
| 13 | `config` | `config.yaml` (nem létezik) | [ ] |
| 14 | `pipeline-notebook` | `notebooks/pipeline.ipynb` (nem létezik) | [ ] |
| 15 | `put-call-loss` | `src/train.py` | [ ] |
| 16 | `greek-supervised` | `src/train.py`, `src/model.py` | [ ] |
| 17 | `heston-data` | `src/data_generator.py` | [ ] |
| 18 | `attention-block` | `src/model.py` | [ ] |
| 19 | `fourier-embedding` | `src/model.py` | [ ] |
| 20 | `mc-dropout` | `src/evaluate.py` | [ ] |
| 21 | `onecycle-lr` | `src/train.py` | [ ] |
| 22 | `put-call-augment` | `src/train.py` | [x] |
| 23 | `szegmentalt-metrikak` | `src/evaluate.py` | [x] |
| 24 | `greek-autograd` | `src/evaluate.py` | [ ] |
| 25 | `optuna-sweep` | `src/train.py` (új fájl) | [ ] |

---

### 9. `eda`
**Fájl:** `notebooks/eda.ipynb` (nem létezik)
**Probléma:** Az adathalmaz eloszlásáról nincs vizualizáció.
**Megoldás:** Jupyter notebook: input hisztogramok, call/put ár eloszlás, moneyness vs. call ár scatter.

---

### 10. `put-call-validacio`
**Fájl:** `src/data_generator.py`
**Probléma:** A generált adatokon nincs put-call paritás ellenőrzés — néma numerikus hiba esetén hibás adatokkal tanítunk.
**Megoldás:** Generálás után ellenőrizni, hogy `C - P ≈ S·e^(-qT) - K·e^(-rT)` minden sorra teljesül (tolerancia: 1e-6).

---

### 11. `relativ-hiba-metrika`
**Fájl:** `src/evaluate.py`
**Probléma:** MSE/MAE abszolút hibát mér, ami eltorzul a mélyen ITM/OTM opciók esetén.
**Megoldás:** Relatív hiba: `|pred - BS| / BS` — pénzügyileg relevánsabb metrika, érdemes a fő kiértékelési mérőszámnak választani.

---

### 12. `baseline-összehasonlitas`
**Fájl:** `src/model.py`
**Probléma:** Nincs baseline, amihez a modell teljesítményét viszonyítani lehet.
**Megoldás:** Egy egyszerű sekély MLP (1-2 réteg) baseline-ként, a mélyebb architektúrával összehasonlítva.

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

---

### 15. `put-call-loss`
**Fájl:** `src/train.py`
**Probléma:** A háló csak MSE-t minimalizál — a put-call paritást (`C - P = S·e^(-qT) - K·e^(-rT)`) nem kényszeríti ki semmi, így az ár fizikailag inkonzisztens lehet.
**Megoldás:** Extra loss tag: `loss = MSE(price) + λ · MSE(C - P - paritás)`. A λ hiperparaméter szabályozza az egyensúlyt.

---

### 16. `greek-supervised`
**Fájl:** `src/train.py`, `src/model.py`
**Probléma:** A görögöket (delta, gamma) az adathalmaz tartalmazza, de a tanítás során nem használjuk fel őket.
**Megoldás:** Fizika-informált tanítás (PINN-szerű): `loss = MSE(price) + λ · MSE(delta)`. A delta analitikusan ismert BS-ből, így extra adatgenerálás nélkül implementálható.

---

### 17. `heston-data`
**Fájl:** `src/data_generator.py`
**Probléma:** A jelenlegi generátor flat vol surface-t feltételez (Black-Scholes), ami nem modellezi a valós piaci volatility smile-t / skew-t.
**Megoldás:** Heston vagy SABR sztochasztikus volatilitás modell implementálása alternatív adatforrásként — a háló jobb valós piaci általánosítást tanul.

---

### 18. `attention-block`
**Fájl:** `src/model.py`
**Probléma:** Az MLP és ResNet architektúrák nem modellezik explicitül a feature-ök közötti nem-lineáris interakciókat (pl. moneyness × sigma együttes hatása).
**Megoldás:** Kis 2-fejes self-attention réteg beillesztése a ResNet blokkokba — a Transformer-stílusú attention mechanizmus ezt natively tanítja.

---

### 19. `fourier-embedding`
**Fájl:** `src/model.py`
**Probléma:** Az 5 bemeneti feature lineáris beágyazása nem hatékony periodicikus vagy magas-frekvenciájú struktúrák tanításához.
**Megoldás:** Random Fourier Features (Rahimi & Recht 2007): az inputot `[sin(Wx+b), cos(Wx+b)]` térbe emelve a háló mélyebb nemlinearitásokat tanul alacsonyabb paraméterszámmal.

---

### 20. `mc-dropout`
**Fájl:** `src/evaluate.py`
**Probléma:** A modellek pontbecslést adnak — nincs konfidencia intervallum az áron, ami kockázatkezeléshez elengedhetetlen lenne.
**Megoldás:** Monte Carlo Dropout: inference közben bekapcsolt Dropout, T forward pass átlaga és szórása adja a predikciót és annak bizonytalanságát.

---

### 21. `onecycle-lr`
**Fájl:** `src/train.py`
**Probléma:** A `ReduceLROnPlateau` reaktív — csak akkor csökkenti a tanulási rátát, ha már stagnál. Lassabb konvergencia és suboptimális minimum.
**Megoldás:** `torch.optim.lr_scheduler.OneCycleLR` — szinuszosan változó LR egy epoch-cikluson belül, Smith (2018) alapján. Gyorsabb konvergencia, jobb generalizáció.

---

### 22. `put-call-augment`
**Fájl:** `src/train.py`
**Probléma:** Az adathalmaz csak call árakat tartalmaz tanításhoz — a put adatok kihasználatlanok maradnak.
**Megoldás:** Put-call paritás alapján minden call mintához ingyen generálható egy put minta: `P = C - S·e^(-qT) + K·e^(-rT)`. Ez megduplázza az effektív adathalmazt extra generálás nélkül.

---

### 23. `szegmentalt-metrikak`
**Fájl:** `src/evaluate.py`
**Probléma:** Az átlagos RMSE elrejti, hogy a háló mélyen OTM opciókon (`S/K < 0.8`) tipikusan sokkal rosszabbul teljesít, mint ATM-en.
**Megoldás:** Moneyness-szegmentált kiértékelés: OTM (`S/K < 0.9`), ATM (`0.9 ≤ S/K ≤ 1.1`), ITM (`S/K > 1.1`) külön metrikákkal — realisztikus teljesítménykép.

---

### 24. `greek-autograd`
**Fájl:** `src/evaluate.py`
**Probléma:** A görög-közelítések minőségét nem értékeljük — a háló delta-ja eltérhet az analitikus BS deltától.
**Megoldás:** `torch.autograd.grad` segítségével a betanított hálóból analitikusan számolható `∂price/∂moneyness` (közelítő delta), majd összehasonlítható az analitikus BS deltával — extra tanítás nélkül.

---

### 25. `optuna-sweep`
**Fájl:** új `src/sweep.py`
**Probléma:** A hiperparaméterek (hidden_dim, n_layers, lr, dropout, batch_size) manuálisan vannak rögzítve — lehet, hogy jobb kombináció létezik.
**Megoldás:** Optuna-alapú Bayesian hiperparaméter keresés, 50-100 trial-lal. A legjobb trial automatikusan checkpoint-olva. Publikálható kísérletezési eredményt ad minimális extra kóddal.