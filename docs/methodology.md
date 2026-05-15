# Módszertan

## 1. Adathalmaz generálása

### 1.1 Paramétertartományok

Az adathalmaz szintetikus Black-Scholes árakból áll, amelyeket az alábbi paramétertartományokon generálunk:

| Paraméter | Leírás | Minimum | Maximum |
|-----------|--------|---------|---------|
| `S` | Alaptermék azonnali ára | 10.0 | 150.0 |
| `moneyness` (S/K) | Pénzben-lét aránya | 0.5 | 1.5 |
| `T` | Lejáratig hátralévő idő (év) | 0.005 | 2.0 |
| `r` | Kockázatmentes kamatláb | 0.00 | 0.05 |
| `σ` | Volatilitás (éves) | 0.05 | 0.90 |

A kötési árat (K) a `moneyness = S/K` összefüggésből számítjuk vissza: `K = S / moneyness`. Az osztalékhozamot `q = 0`-nak feltételezzük.

### 1.2 Mintavételezési módszerek

A generátor három mintavételezési módszert támogat:

- **uniform** – egyenletes véletlen mintavételezés az egyes paraméterekből függetlenül
- **lhs** (Latin Hypercube Sampling) – a paramétértér egyenletesebb lefedettségét biztosítja; a `scipy.stats.qmc.LatinHypercube` implementációt alkalmazza
- **grid** – egyenletes rácspontok az összes paraméteren (`itertools.product`), n-ből visszaszámított lépésszámmal

A kísérletek során az **lhs** módszert alkalmaztuk, mivel jobb térfogat-lefedettséget nyújt azonos mintaszám mellett.

### 1.3 Adathalmaz mérete és szétválasztása

Az összes minta száma: **1 000 000**. A szétválasztás arányai:

| Halmaz | Arány | Minták száma |
|--------|-------|-------------|
| Tanítóhalmaz | 70% | 700 000 |
| Validációs halmaz | 15% | 150 000 |
| Teszthalmaz | 15% | 150 000 |

A szétválasztás véletlenszerű keveréssel (shuffle) és rögzített maggal (`seed = 42`) történik a reprodukálhatóság érdekében.

### 1.4 Célváltozók és normalizálás

A nyers call- és put-árakat kötési árral (K) normalizáljuk:

```
call_price_norm = C / K
put_price_norm  = P / K
```

A put-árat put-call paritásból számítjuk:

```
P = C - S + K · e^(-rT)
```

Normalizált formában (`q = 0` esetén):

```
put_norm = call_norm - moneyness + e^(-rT)
```

A bemeneti jellemzőket is `[0, 1]` intervallumra skálázzuk a paraméteres minimum–maximum értékek alapján, ezzel megkönnyítve a hálózat tanulását.

### 1.5 Opcionális kiegészítők

- **Görögök** (`include_greeks=True`): delta, gamma, vega, theta, rho kiszámítható és hozzáfűzhető az adathalmazhoz.
- **Gauss-zaj** (`noise_std > 0`): normális eloszlású zaj adható az árakhoz; a negatívvá váló értékeket 0-ra klampolják.

### 1.6 Tárolási formátum

Az adathalmazokat Parquet formátumban tároljuk (`data/train.parquet`, `data/val.parquet`, `data/test.parquet`), amelyek gyors betöltést és kis méretű tárolást biztosítanak.

---

## 2. Black-Scholes implementáció

Az `src/black_scholes.py` modulban implementált képletek adják az adathalmaz igaz értékeit, és egyben a kiértékelési referenciapontot.

### 2.1 Segédváltozók

```
d₁ = [ln(S/K) + (r + 0.5·σ²)·T] / (σ·√T)
d₂ = d₁ - σ·√T
```

### 2.2 Call- és put-ár

```
C = S·Φ(d₁) − K·e^(−rT)·Φ(d₂)
P = C − S + K·e^(−rT)          [put-call paritás]
```

ahol Φ a standard normális kumulatív eloszlásfüggvény (`scipy.stats.norm.cdf`).

Lejárat közelében (`T ≈ 0`) a belső értéket adjuk vissza: `max(S − K·e^(−rT), 0)`.

### 2.3 Görögök

| Görög | Képlet |
|-------|--------|
| Δ (delta) | `Φ(d₁)` |
| Γ (gamma) | `φ(d₁) / (S·σ·√T)` |
| ν (vega) | `S·φ(d₁)·√T · 0.01` (1%-pontos σ változásra) |
| Θ (theta) | `[−S·φ(d₁)·σ/(2√T) − r·K·e^(−rT)·Φ(d₂)] / 365` (napi) |
| ρ (rho) | `K·T·e^(−rT)·Φ(d₂) · 0.01` (1%-pontos r változásra) |

ahol φ a standard normális sűrűségfüggvény. Az implementáció vektorizált NumPy műveleteket alkalmaz, a nullával való osztást `np.errstate(divide='ignore', invalid='ignore')` segítségével kezeli.

---

## 3. Modell architektúrák

Minden modell bemenete 4 normalizált jellemző: `[moneyness_norm, T_norm, r_norm, sigma_norm]`, kimenete egyetlen normalizált call-ár (`call_price_norm`). Az implementáció `src/model.py`-ban található.

### 3.1 MLPPricer — alaplap (Culkin & Das, 2017)

```
Input(4) → Linear(4→100) → ReLU
         → Linear(100→100) → ReLU   (×3 további réteg)
         → Linear(100→1)
```

- Rejtett rétegek: 4 × 100 neuron
- Aktiváció: ReLU
- Paraméterszám: ~30 900
- Szerepe: reprodukálni a Culkin & Das (2017) eredeti architektúráját; referenciamodell

### 3.2 DeepMLPPricer (Della Corte et al., 2023)

```
Input(4) → Linear(4→256)
→ [LayerNorm(256) → ReLU → Dropout(0.1) → Linear(256→256)] × 4
→ LayerNorm(256) → Linear(256→1)
```

- Rejtett rétegek: 4 × 256 neuron (pre-LN elrendezés)
- Normalizáció: LayerNorm
- Dropout: 0.1
- Paraméterszám: ~265 000

### 3.3 ResNetPricer (Della Corte et al., 2023)

```
Input(4) → Linear(4→256) → BatchNorm1d → ReLU
→ [ResidualBlock(256)] × 3
→ Linear(256→1)

ResidualBlock:
  x → Linear(256→256) → BatchNorm1d → ReLU
    → Dropout(0.1) → Linear(256→256) → BatchNorm1d
    → + x   (reziduális kapcsolat)
```

- Blokkok száma: 3
- Normalizáció: BatchNorm1d
- Paraméterszám: ~400 000

### 3.4 GELUResNetPricer

ResNetPricer GELU-alapú változata:

```
Input(4) → Linear(4→256) → GELU
→ [GELUResidualBlock(256)] × 3
→ LayerNorm(256) → Linear(256→1)

GELUResidualBlock (pre-LN):
  x → LayerNorm → Linear → GELU → Dropout(0.1) → Linear → + x
```

- Aktiváció: GELU (simább gradiens ReLU-hoz képest)
- Normalizáció: LayerNorm (pre-LN stílus)
- Paraméterszám: ~400 000

### 3.5 DenseMLPPricer (Huang et al., 2017 – DenseNet-stílus)

```
h₁ = GELU(W₁ · x)                        [bemenet: 4]
h₂ = GELU(W₂ · [x, h₁])                  [bemenet: 132]
h₃ = GELU(W₃ · [x, h₁, h₂])             [bemenet: 260]
h₄ = GELU(W₄ · [x, h₁, h₂, h₃])        [bemenet: 388]
out = W_out · [x, h₁, h₂, h₃, h₄]       [bemenet: 516]
```

- Minden réteg az összes korábbi kimenet összefűzését kapja bemenetként
- Rejtett dimenzió: 128/réteg
- Paraméterszám: ~102 000

### 3.6 HighwayPricer (Srivastava et al., 2015)

```
Input(4) → Linear(4→256) → GELU
→ [HighwayBlock(256)] × 4
→ Linear(256→1)

HighwayBlock:
  H = GELU(W_H · x)             [transzformáció]
  T = σ(W_T · x − 1)            [kapu; −1 bias inicializáció]
  out = H · T + x · (1 − T)     [tanulható skip arány]
```

- Kapu: tanulható transzport-arány (0…1)
- Blokkok száma: 4
- Paraméterszám: ~528 000

### 3.7 FINNPricer — Finance-Informed Neural Network (Liu et al., 2019)

```
Közelítő ág:
  Input(4) → Linear(4→64) → GELU → Linear(64→64) → GELU → Linear(64→1)

Korrekciós ág:
  Input(4) → Linear(4→256) → GELU
           → [GELUResidualBlock(256)] × 3
           → LayerNorm(256) → Linear(256→1)

Kimenet: közelítő + korrekciós
```

- Kettős ágstruktúra: könnyű közelítő + nehéz korrekciós ág
- Paraméterszám: ~402 000

---

## 4. Tanítás és validáció

### 4.1 Hiperparaméterek

| Paraméter | Érték |
|-----------|-------|
| Max. epochszám | 200 |
| Batch méret | 4096 |
| Tanulási ráta (kezdeti) | 1×10⁻³ |
| L2 regularizáció (weight decay) | 1×10⁻⁴ |
| Early stopping türelem | 10 epoch |
| Véletlen mag (seed) | 42 |

### 4.2 Optimizer

**Adam** (`torch.optim.Adam`) adaptív tanulási rátával és impulzus becsléssel.

### 4.3 Veszteségfüggvény

**MSE** (Mean Squared Error, `nn.MSELoss`):

```
L = (1/N) · Σ (ŷᵢ − yᵢ)²
```

ahol `yᵢ` a normalizált BS call-ár, `ŷᵢ` a háló kimenete.

#### Opcionális: Physics-Informed Loss

`--physics-loss` kapcsolóval aktiválható; a delta-korlátot is büntetőtagként veszi figyelembe:

```
L_total = L_MSE + λ · L_delta
```

ahol `L_delta` a delta-becslés `[0, 1]` intervallumtól való eltérése, λ = 0.1 (alapértelmezett).

### 4.4 Tanulási ráta ütemező

**ReduceLROnPlateau** (`torch.optim.lr_scheduler.ReduceLROnPlateau`):

| Paraméter | Érték |
|-----------|-------|
| Mód | `min` (loss minimalizálás) |
| Csökkentési faktor | 0.5 (felezés) |
| Türelem | 5 epoch |
| Minimális LR | 1×10⁻⁶ |

Ha a validációs veszteség 5 epokon át nem javul, a tanulási rátát megfelezi az ütemező.

### 4.5 Korai megállás (Early Stopping)

Ha a validációs veszteség 10 epokon át nem csökken, a tanítás leáll. Minden epochban, ha a validációs veszteség új minimumot ér el, a modell állapotát (`state_dict`) mentjük.

### 4.6 Tanítási ciklus

Minden epochban:

1. A tanítóhalmaz véletlenszerűen keverésre kerül (shuffle=True).
2. Batch-enkénti forward pass, loss számítás, `loss.backward()`, `optimizer.step()`.
3. A validációs halmazon (shuffle=False) kiértékelés `torch.no_grad()` kontextusban.
4. Az LR-ütemező lépése a validációs loss alapján.
5. Ha javulás: checkpoint mentése; ha nincs javulás `patience` epokon át: megállás.

### 4.7 Put-call paritás augmentáció (opcionális)

`--augment-put` kapcsolóval az adathalmaz megduplázódik: a put-példányok egy bináris `is_put` jellemzőt kapnak (0 = call, 1 = put), a modell bemeneti dimenziója 4-ről 5-re nő.

### 4.8 Eszközválasztás

A kód automatikusan a legjobb elérhető eszközt választja: CUDA (NVIDIA GPU) → MPS (Apple Silicon) → CPU.

---

## 5. Eredmények generálása és kiértékelés

### 5.1 Metrikák

Az `src/evaluate.py` modul az alábbi metrikákat számítja a teszthalmaz normalizált árain:

| Metrika | Képlet | Leírás |
|---------|--------|--------|
| RMSE | `√(mean((ŷ−y)²))` | Négyzetgyök átlagos négyzetes hiba |
| MAE | `mean(|ŷ−y|)` | Átlagos abszolút hiba |
| MAPE | `100 · mean(|ŷ−y| / (|y|+ε))` | Átlagos abszolút százalékos hiba |
| Max. hiba | `max(|ŷ−y|)` | Legrosszabb eset |
| R² | `1 − SS_res/SS_tot` | Determinációs együttható |

### 5.2 Szegmentált kiértékelés

A teszthalmaz moneyness szerint három szegmensre bontva is kiértékelődik:

| Szegmens | Feltétel (S/K) |
|----------|---------------|
| OTM (Out-of-the-Money) | `S/K < 0.9` |
| ATM (At-the-Money) | `0.9 ≤ S/K ≤ 1.1` |
| ITM (In-the-Money) | `S/K > 1.1` |

Minden szegmensre külön RMSE, MAE, MAPE, max. hiba és R² értéket számítunk, mivel az opciós árazás nehézségi foka szegmensenként eltér (az ATM opciók árazása a legnehezebb).

### 5.3 Batch inferencia

A kiértékelés `torch.no_grad()` kontextusban, 4096-os batch mérettel fut, hogy a memóriahasználat kezelt maradjon nagy teszt halmazon is.

### 5.4 Checkpoint formátum

A mentett `.pt` checkpointok tartalmazzák:

- `model_class` – osztálynév (pl. `"MLPPricer"`)
- `model_kwargs` – konstruktor argumentumok (rétegméretek stb.)
- `state_dict` – modell súlyok
- `feature_cols` – bemeneti jellemzőoszlopok listája
- `target_col` – célváltozó neve
- `history` – epoch-szintű train/val loss és LR napló
- `best_epoch` – legjobb validációs veszteség eposza
- `val_loss` – legjobb validációs MSE érték

### 5.5 Tanítási görbék

A `plot_training_curves.py` szkript a `results/training_history_*.csv` fájlokból összehasonlító loss-görbéket generál (train vs. val, modellenkénti szubplotok és aggregált összehasonlítás).

### 5.6 Modellek összehasonlítása

A `compare_models()` függvény több `.pt` checkpointot tölt be, és táblázatos formában hasonlítja össze a teljesítménymetrikákat (RMSE, MAE, MAPE, R²) mind a teljes teszthalmaz, mind a három moneyness-szegmens szerint.

---

## 6. Kísérleti eredmények összefoglalása

A 7 modell betanítása után mért validációs MSE-értékek:

| Sorrend | Modell | Val MSE | Legjobb epoch | Paraméterszám |
|---------|--------|---------|--------------|--------------|
| 1. | MLPPricer | 2.0607×10⁻⁵ | 123 | 30 900 |
| 2. | DeepMLPPricer | 3.6347×10⁻⁵ | 27 | 265 000 |
| 3. | ResNetPricer | 3.8462×10⁻⁵ | 28 | 400 000 |
| 4. | FINNPricer | 8.2406×10⁻⁵ | 21 | 402 000 |
| 5. | GELUResNetPricer | 8.4989×10⁻⁵ | 31 | 400 000 |
| 6. | HighwayPricer | 2.1954×10⁻⁴ | 9 | 528 000 |
| 7. | DenseMLPPricer | 3.2283×10⁻⁴ | 5 | 102 000 |

**Főbb megfigyelések:**
- Az MLPPricer a legegyszerűbb architektúra, mégis a legjobb eredményt érte el: 134 epochon át tanult, míg a komplex modellek 5–31 epochnál megálltak.
- A paraméterszám nem korrelál egyenesen a teljesítménnyel (pl. 528k HighwayPricer > 30k MLPPricer, de rosszabb eredményt hoz).
- A BatchNorm (ResNetPricer) hatékonyabbnak bizonyult, mint a LayerNorm (GELUResNetPricer) ezen a feladaton.
- A finance-informed (FINN) kettős ágstruktúra ígéretes megközelítés, de szintetikus adatokon az egyszerű MLP kompetitív marad.
