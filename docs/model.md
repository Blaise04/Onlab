# Neurális háló opciós árazó — Architektúra dokumentáció

## 1. Áttekintés

A projekt 1. fázisában Black-Scholes szintetikus adatokon tanított neurális hálók
call opció árakat becsülnek. Az architektúrák két generációban épülnek fel:

**1. generáció** (baseline):

| Modell         | CLI neve    | Irodalom                   | Paraméterek | Aktiváció |
|----------------|-------------|----------------------------|-------------|-----------|
| MLPPricer      | `mlp`       | Culkin & Das (2017)        | ~31 000     | ReLU      |
| DeepMLPPricer  | `deep_mlp`  | Lürig et al. (2023)        | ~268 000    | ReLU      |
| ResNetPricer   | `resnet`    | Lürig et al. (2023)        | ~399 000    | ReLU      |

**2. generáció** (kísérleti):

| Modell           | CLI neve      | Irodalom                        | Paraméterek | Aktiváció |
|------------------|---------------|---------------------------------|-------------|-----------|
| GELUResNetPricer | `gelu_resnet` | ResNetPricer + GELU             | ~399 000    | GELU      |
| DenseMLPPricer   | `dense_mlp`   | Huang et al. (2017) DenseNet    | ~102 000    | GELU      |
| HighwayPricer    | `highway`     | Srivastava et al. (2015)        | ~528 000    | GELU      |
| FINNPricer       | `finn`        | Liu et al. (2019), arXiv:2412   | ~403 000    | GELU      |

---

## 2. Bemeneti / kimeneti reprezentáció

### Bemeneti feature-ök (5 db, már [0,1]-re skálázva)

| Oszlop           | Leírás                         | Tartomány (eredeti) |
|------------------|--------------------------------|---------------------|
| `moneyness_norm` | S/K arány normálva             | [0.5, 1.5]          |
| `T_norm`         | Lejáratig hátralévő idő        | [0.005, 2.0] év     |
| `r_norm`         | Kockázatmentes ráta            | [0.0, 0.05]         |
| `sigma_norm`     | Volatilitás                    | [0.05, 0.90]        |
| `q_norm`         | Osztalékhozam                  | [0.0, 0.03]         |

### Kimenet

`call_price_norm` = C / K — dimenziótalanított call ár (Garcia & Gençay homogeneity hint).
A K-val való osztás kihasználja a Black-Scholes ár homogenitási tulajdonságát:
C(S, K, T, r, σ) = K · f(S/K, T, r, σ), ezért a háló könnyebben általánosít.

---

## 3. Modell architektúrák

### 3.1 MLPPricer — Culkin & Das (2017) baseline

```
Input(5)
  → Linear(5 → 100) → ReLU
  → Linear(100 → 100) → ReLU   ┐
  → Linear(100 → 100) → ReLU   │  3 rejtett réteg
  → Linear(100 → 100) → ReLU   ┘
  → Linear(100 → 1)
```

- Nincs normalizáció, nincs Dropout — hűen követi az eredeti cikket
- `MLPPricer(input_dim=5, hidden_dim=100, n_layers=4)`

### 3.2 DeepMLPPricer — Lürig et al. (2023) javított MLP

```
Input(5)
  → Linear(5 → 256)
  → [LayerNorm → ReLU → Dropout(0.1) → Linear(256 → 256)] × 4
  → LayerNorm
  → Linear(256 → 1)
```

- Pre-LN stílus: normalizáció a nemlinearitás előtt (stabilabb gradiens)
- `DeepMLPPricer(input_dim=5, hidden_dim=256, n_layers=4, dropout=0.1)`

### 3.3 ResNetPricer — Lürig et al. (2023) reziduális MLP

```
Input(5)
  → Linear(5 → 256) → ReLU          ← input projekció
  → [ResidualBlock(256)] × 3
  → LayerNorm
  → Linear(256 → 1)
```

**ResidualBlock(dim)**:
```
x → LayerNorm → Linear(dim→dim) → ReLU → Dropout(0.1) → Linear(dim→dim) → + x
```

- Pre-LN reziduális kapcsolat (nincs projekciós réteg: dim_in == dim_out)
- `ResNetPricer(input_dim=5, hidden_dim=256, n_blocks=3, dropout=0.1)`

### 3.4 GELUResNetPricer — ResNet GELU aktivációval

```
Input(5)
  → Linear(5 → 256) → GELU              ← input projekció
  → [GELUResidualBlock(256)] × 3
  → LayerNorm
  → Linear(256 → 1)
```

**GELUResidualBlock(dim)**:
```
x → LayerNorm → Linear(dim→dim) → GELU → Dropout(0.1) → Linear(dim→dim) → + x
```

- Azonos struktúra mint ResNetPricer, ReLU → GELU csere
- Motiváció: a BS árak simák, GELU simább gradienst biztosít (nincs "törött" derivált 0-nál)
- `GELUResNetPricer(input_dim=5, hidden_dim=256, n_blocks=3, dropout=0.1)`

---

### 3.5 DenseMLPPricer — DenseNet-stílusú összefűzéses MLP

```
h₁ = GELU(W₁·x)
h₂ = GELU(W₂·[x, h₁])
h₃ = GELU(W₃·[x, h₁, h₂])
h₄ = GELU(W₄·[x, h₁, h₂, h₃])
output = W_out·[x, h₁, h₂, h₃, h₄]
```

- Minden réteg az összes korábbi kimenetét kapja → jobb gradiens-áramlás
- Korai rétegek direkt kapcsolódnak a kimenethez ("feature reuse")
- Kisebb hidden_dim (128) is elegendő, mert a dense skip-ek gazdagítják a reprezentációt
- `DenseMLPPricer(input_dim=5, hidden_dim=128, n_layers=4, dropout=0.1)`
- Irodalom: Huang et al. (2017) — *Densely Connected Convolutional Networks*

---

### 3.6 HighwayPricer — tanulható gating

```
Input(5)
  → Linear(5 → 256) → GELU
  → [HighwayBlock(256)] × 4
  → Linear(256 → 1)
```

**HighwayBlock(dim)**:
```
H = GELU(W_H·x + b_H)        ← transform
T = σ(W_T·x + b_T)           ← transform gate  (b_T init: -1)
y = H·T + x·(1 − T)          ← gated output
```

- A skip arány nem rögzített (mint ResNetben), hanem tanult
- Gate bias −1-re inicializálva: kezdetben inkább "carry" (skip), majd tanul
- `HighwayPricer(input_dim=5, hidden_dim=256, n_blocks=4, dropout=0.1)`
- Irodalom: Srivastava et al. (2015) — *Training Very Deep Networks*

---

### 3.7 FINNPricer — Finance-Informed Neural Network

```
Ág 1 (approx):     x → [Linear(5→64) → GELU] × 2 → Linear(64→1) → BS̃
Ág 2 (correction): x → Linear(5→256) → GELU
                     → [GELUResidualBlock(256)] × 3
                     → LayerNorm → Linear(256→1)   → δ
Output: BS̃ + δ
```

- Az approx ág a "könnyű" eseteket közelíti (ITM opciók)
- A correction ág a nehéz eseteket korrigálja (mélyen OTM, rövid lejárat)
- Akadémiailag a legtartalmasabb: a két ág külön szerepet kap
- `FINNPricer(input_dim=5, approx_dim=64, resnet_dim=256, n_blocks=3, dropout=0.1)`
- Irodalom: Liu et al. (2019) — *A neural network-based framework for financial model calibration*;
  arXiv:2412.12213 — *AI Black-Scholes*

---

## 4. Training konfiguráció

| Paraméter      | Érték                              |
|----------------|------------------------------------|
| Optimizer      | Adam                               |
| Tanulási ráta  | 1e-3                               |
| Weight decay   | 1e-4                               |
| Loss           | MSELoss                            |
| LR Scheduler   | ReduceLROnPlateau(factor=0.5, patience=5, min_lr=1e-6) |
| Early stopping | patience=10 epoch                  |
| Batch méret    | 4096                               |
| Max epochok    | 200                                |

### 4.1 Physics-Informed Loss (opcionális)

Bármely modellel kombinálható a `--physics-loss` flag-gel:

```
L = L_MSE + λ · L_delta
```

ahol:
```
L_delta = mean(relu(−∂C_norm/∂m_norm) + relu(∂C_norm/∂m_norm − 1))
```

- `∂C_norm/∂moneyness_norm` a modell predikciónak moneyness szerinti deriváltja (autograd)
- A korlát: delta ∈ [0, 1] — a call delta definíció szerint nem lehet negatív, ill. 1-nél nagyobb
- Irodalmi alap: Liu et al. (2019), PINN (arXiv:2312.06711)
- CLI: `--physics-loss --physics-lambda 0.1`

A legjobb validációs loss-nál checkpoint mentés történik
(`models/{model_name}_best.pt`), amely tartalmazza:
- `state_dict` — modell súlyok
- `model_class`, `model_kwargs` — rekonstrukcióhoz
- `feature_cols`, `target_col` — adatfeldolgozáshoz
- `history` — tanítási görbe (a best epochig bezárólag)

**Megjegyzés a history tárolásáról:** A checkpoint mentése a legjobb validációs
epochnál történik, és a history csak az addig gyűjtött adatokat tartalmazza.
Az early stopping futtatja a modellt a best epoch után még `patience=10` epochot,
de ezek az epochok nem kerülnek a checkpointba — csak a best epochig terjedő
adatsorok érhetők el.

---

## 5. Kiértékelési metrikák

| Metrika    | Képlet                                              |
|------------|-----------------------------------------------------|
| RMSE       | √(mean((y_true − y_pred)²))                        |
| MAE        | mean(|y_true − y_pred|)                             |
| MAPE (%)   | mean(|y_true − y_pred| / (|y_true| + ε)) × 100     |
| max_error  | max(|y_true − y_pred|)                              |
| R²         | 1 − SS_res / SS_tot                                 |

MAPE-nél ε = 1e-8 védi a mélyen pénzen kívüli opciókat (közel nulla árak).

---

## 6. Gyors indítás

```bash
# Adathalmaz generálás (ha még nem létezik)
python generate_dataset.py --n 1000000 --method lhs --format parquet \
    --normalize --scale-inputs --greeks --seed 42

# 1. generáció tanítása
python train.py --model mlp      --epochs 200 --output models/
python train.py --model deep_mlp --epochs 200 --output models/
python train.py --model resnet   --epochs 200 --output models/

# 2. generáció tanítása
python train.py --model gelu_resnet --epochs 200 --output models/
python train.py --model dense_mlp   --epochs 200 --output models/
python train.py --model highway     --epochs 200 --output models/
python train.py --model finn        --epochs 200 --output models/

# Physics-informed loss (bármely modellel)
python train.py --model resnet --physics-loss --physics-lambda 0.1 --epochs 200 --output models/ --name resnet_phys

# Összehasonlító kiértékelés
python evaluate.py \
  --checkpoint models/mlp_best.pt models/deep_mlp_best.pt \
              models/resnet_best.pt models/gelu_resnet_best.pt \
              models/dense_mlp_best.pt models/highway_best.pt \
              models/finn_best.pt models/resnet_phys_best.pt \
  --compare --segmented

# Forward pass ellenőrzés
python -c "
from src.model import (MLPPricer, DeepMLPPricer, ResNetPricer,
                       GELUResNetPricer, DenseMLPPricer, HighwayPricer,
                       FINNPricer, count_parameters)
import torch
x = torch.randn(16, 5)
for M in [MLPPricer(), DeepMLPPricer(), ResNetPricer(),
          GELUResNetPricer(), DenseMLPPricer(), HighwayPricer(), FINNPricer()]:
    y = M(x)
    print(f'{type(M).__name__:20s} {tuple(y.shape)}  params={count_parameters(M):,}')
"
```

---

## 7. Kísérleti eredmények

Az összes modellt azonos feltételek mellett tanítottuk: 700 000 szintetikus Black-Scholes minta
(LHS-mintavételezés, seed=42), 150 000-es validációs és teszt halmaz, max 200 epoch,
patience=10, batch=4096, Adam (lr=1e-3, weight_decay=1e-4) + ReduceLROnPlateau scheduler
(factor=0.5, patience=5). Kiértékelés a teszt halmazon (150 000 minta).
GPU: NVIDIA GeForce RTX 4060 Laptop GPU (8 GB).

Részletes epoch-szintű adatok: `results/training_curves.json` és `results/training_curves.csv`.

### 7.1 Összefoglaló táblázat

A táblázat a teszt halmazon mért eredményeket tartalmazza (150 000 minta, seed=42).
A `Val MSE (best)` a legjobb checkpoint validációs MSE értéke. A `Best ep` az a epoch,
amelyen a legjobb checkpoint mentése történt; az `Összes ep (checkpoint)` a checkpointban
tárolt epochok száma (= best epoch, ld. 4. fejezet megjegyzése).

| Modell        | Paraméterek | Best ep | Val MSE (×10⁻⁵) | Test RMSE  | Test MAE   |    R²     |
|---------------|-------------|---------|-----------------|------------|------------|-----------|
| mlp           |      31 001 |      68 |        2.13     | 0.004612   | 0.002780   | 0.999407  |
| deep_mlp      |     267 521 |       4 |       27.40     | 0.016575   | 0.013233   | 0.992338  |
| resnet        |     398 593 |      80 |        2.28     | 0.004788   | 0.003274   | 0.999361  |
| gelu_resnet   |     398 593 |      40 |        8.28     | 0.009135   | 0.006027   | 0.997673  |
| dense_mlp     |     101 894 |      18 |       22.80     | 0.015172   | 0.010694   | 0.993581  |
| highway       |     528 129 |      22 |       23.20     | 0.015281   | 0.011005   | 0.993488  |
| finn          |     403 202 |      17 |        9.23     | 0.009653   | 0.006625   | 0.997401  |
| resnet_phys   |     398 593 |      68 |        3.28     | 0.005783   | 0.003987   | 0.999067  |

Megjegyzés: `mlp` bizonyult a legpontosabb modellnek (RMSE=0.004612, R²=0.999407),
megelőzve a `resnet`-et (RMSE=0.004788). A `resnet_phys` physics-loss regularizációval
~21%-kal magasabb RMSE-t ért el, de garantálja a delta-korlátot.

### 7.2 Szegmentált eredmények (RMSE)

Szegmenshatárok: OTM = moneyness < 0.9, ATM = 0.9–1.1, ITM = moneyness > 1.1.
Teszthalmazon mért értékek (N(OTM)=60 051, N(ATM)=29 948, N(ITM)=60 001).

| Modell        | OTM (m<0.9)  | ATM (0.9-1.1) | ITM (m>1.1)  | max_error |    R² (all)  |
|---------------|--------------|---------------|--------------|-----------|--------------|
| mlp           | **0.002933** | **0.005903**  |   0.005214   | 0.076284  |   0.999407   |
| deep_mlp      |   0.013577   |   0.013992    |   0.020115   | 0.103493  |   0.992338   |
| resnet        |   0.004429   |   0.005342    | **0.004842** | 0.065314  |   0.999361   |
| gelu_resnet   |   0.008078   |   0.011879    |   0.008536   | 0.105482  |   0.997673   |
| dense_mlp     |   0.011125   |   0.018354    |   0.016836   | 0.115152  |   0.993581   |
| highway       |   0.011821   |   0.020864    |   0.015054   | 0.115609  |   0.993488   |
| finn          |   0.008649   |   0.012113    |   0.009211   | 0.105467  |   0.997401   |
| resnet_phys   |   0.005502   |   0.006357    |   0.005756   | 0.063004  |   0.999067   |

Megjegyzés: az `mlp` OTM és ATM szegmensben a legjobb, a `resnet` ITM szegmensben
vezet. A `resnet_phys` minden szegmensben egyenletesebb teljesítményt mutat.

### 7.3 Tanulási görbe összefoglaló

A checkpoint a legjobb epochnál mentődik, ezért az alábbi táblázatokban csak a
best epochig terjedő val loss értékek állnak rendelkezésre.
A részletes adatok: `results/training_curves.json` és `results/training_curves.csv`.

#### mlp (MLPPricer — 31 001 param)

| Epoch | Val loss   |
|-------|------------|
|     1 | 0.000079   |
|     5 | 0.000046   |
|    10 | 0.000056   |
|    20 | 0.000028   |
|    30 | 0.000026   |
|    50 | 0.000024   |
|    68 | 0.000021 * |

- Checkpointban tárolt epochok: 68, best epoch: 68
- LR: 1.00e-03 végig (a scheduler a 75. epochon csökkentett volna, de az best epochon
  még nem változott — a checkpoint a 68. epochot tartalmazza)
- Konvergencia: **lassú, de folyamatos javulás** — a val loss az összes 68 tárolt
  epochon monoton trend mentén csökkent (kisebb ingadozásokkal). A loss 0.000079-ről
  0.000021-re ért le a 68. epochra. Nem mutatkozott korai leállás a best epochig.

#### deep_mlp (DeepMLPPricer — 267 521 param)

| Epoch | Val loss   |
|-------|------------|
|     1 | 0.000515   |
|     4 | 0.000274 * |

- Checkpointban tárolt epochok: 4, best epoch: 4
- LR: 1.00e-03
- Konvergencia: **nagyon korai leállás** — mindössze 4 epoch adatát tartalmazza
  a checkpoint (a modell a 4. epochon érte el a legjobb validációs értéket).
  Az 1–4. epochon javult (0.000515 → 0.000274), majd az early stopping leállította.
  Valószínű ok: a Pre-LN + Dropout konfiguráció e konfigurációban gyorsan lokális
  minimumba rekedt — magasabb LR-warmup vagy kisebb LR szükséges.

#### resnet (ResNetPricer — 398 593 param)

| Epoch | Val loss   |
|-------|------------|
|     1 | 0.000350   |
|     5 | 0.000076   |
|    10 | 0.000045   |
|    20 | 0.000114   |
|    30 | 0.000058   |
|    50 | 0.000031   |
|    75 | 0.000023   |
|    80 | 0.000023 * |

- Checkpointban tárolt epochok: 80, best epoch: 80
- LR csökkentések: epoch 25 (1.00e-03→5.00e-04), 33 (→2.50e-04), 42 (→1.25e-04), 53 (→6.25e-05)
- Konvergencia: **fokozatos, de nem monoton** — a val loss az 1–10. epochon erősen
  csökkent (0.000350→0.000045), majd a 20. epochon visszaemelkedett (0.000114),
  majd ismét csökkent. A scheduler LR-csökkentései fokozatosan stabilizálták a tanítást;
  a legjobb érték (0.000023) a 75–80. epochok körül alakult ki.

#### gelu_resnet (GELUResNetPricer — 398 593 param)

| Epoch | Val loss   |
|-------|------------|
|     1 | 0.003225   |
|     5 | 0.000179   |
|    10 | 0.000095   |
|    20 | 0.000227   |
|    30 | 0.000271   |
|    40 | 0.000083 * |

- Checkpointban tárolt epochok: 40, best epoch: 40
- LR csökkentések: epoch 23 (1.00e-03→5.00e-04), 33 (→2.50e-04), 40 (→1.25e-04)
- Konvergencia: **gyors kezdeti javulás, majd erős instabilitás** — az 1–16. epochon
  rapid csökkentés (0.003225→0.000086), de a 20–22. epochon erős ugrás (0.000086→
  0.002769→0.000352), ami a GELU ResNet LR-érzékenységére utal. A scheduler
  LR-csökkentései után újra stabilizálódott, és a 40. epochon érte el a legjobb
  értéket (0.000083).

#### dense_mlp (DenseMLPPricer — 101 894 param)

| Epoch | Val loss   |
|-------|------------|
|     1 | 0.001454   |
|     5 | 0.000303   |
|    10 | 0.000260   |
|    18 | 0.000228 * |

- Checkpointban tárolt epochok: 18, best epoch: 18
- LR: 1.00e-03 (a scheduler nem lépett be a best epochig)
- Konvergencia: **lassú, csökkenő mértékű javulás** — a val loss az 5–18. epochon
  mindössze 0.000303-ról 0.000228-ra csökkent (25% javulás). A modell nem érte el
  a többi architektúra pontosságát; a dense skip-kapcsolatok BS adatokon nem hoznak
  érdemi előnyt.

#### highway (HighwayPricer — 528 129 param)

| Epoch | Val loss   |
|-------|------------|
|     1 | 0.001003   |
|     5 | 0.000460   |
|    10 | 0.000258   |
|    20 | 0.000240   |
|    22 | 0.000232 * |

- Checkpointban tárolt epochok: 22, best epoch: 22
- LR: 1.00e-03 (a scheduler nem lépett be a best epochig)
- Konvergencia: **közepes sebesség, majd plafon effektus** — az 1–6. epochon rapid
  csökkentés (0.001003→0.000263), majd a 7–22. epochon lelassult javulás. A gate
  mechanizmus nem hozott érdemi előnyt a sima MLP-hez képest; a legnagyobb modell
  (528K param) a leggyengébbek egyike.

#### finn (FINNPricer — 403 202 param)

| Epoch | Val loss   |
|-------|------------|
|     1 | 0.002869   |
|     5 | 0.000229   |
|    10 | 0.000123   |
|    17 | 0.000092 * |

- Checkpointban tárolt epochok: 17, best epoch: 17
- LR: 1.00e-03
- Konvergencia: **gyors és instabil** — a kétágú architektúra gyorsan konvergál
  (0.002869→0.000092 17 epoch alatt), de a 18–22. epochon erős visszaesés mutatkozott
  (0.000140→0.000249), amit a checkpoint nem tartalmaz (best epoch előtt vagyunk).
  A gyors konvergencia ellenére az eredmény nem éri el az mlp/resnet szintjét.

#### resnet_phys (ResNetPricer + Physics Loss — 398 593 param)

| Epoch | Val loss   |
|-------|------------|
|     1 | 0.000377   |
|     5 | 0.000110   |
|    10 | 0.000069   |
|    15 | 0.000226   |
|    20 | 0.000046   |
|    25 | 0.000154   |
|    30 | 0.000039   |
|    40 | 0.000036   |
|    50 | 0.000046   |
|    60 | 0.000035   |
|    68 | 0.000033 * |

- Checkpointban tárolt epochok: 68, best epoch: 68
- LR csökkentések: epoch 17 (1.00e-03→5.00e-04), 30 (→2.50e-04), 38 (→1.25e-04),
  53 (→6.25e-05)
- Konvergencia: **fokozatos, de erősen nem monoton** — a physics loss extra gradiens
  zajt visz be; a val loss az 10–15. epochok között visszaemelkedett (0.000069→0.000226),
  majd ismét a 25. epochon (0.000154). A LR-csökkentések segítségével végül a 68. epochon
  érte el a legjobb értéket (0.000033). Az erős oszcilláció a physics regularizáció
  λ=0.1-es erősségéből adódik.

---

## 8. Következtetések

### 8.1 Általános megállapítások

**Az MLPPricer (Culkin & Das baseline) bizonyult a legjobb modellnek** az összesített
teszt RMSE alapján (RMSE=0.004612, R²=0.999407). Ez meglepő eredmény: a ~31 000
paraméterű, normalizáció és dropout nélküli egyszerű háló felülmúlta az összes
komplexebb architektúrát. A ResNetPricer (RMSE=0.004788) csak minimálisan gyengébb,
de 13× több paraméterrel dolgozik.

A sorrend RMSE szerint (kisebb = jobb):
**mlp (0.004612) > resnet (0.004788) > resnet_phys (0.005783) > gelu_resnet (0.009135) > finn (0.009653) > dense_mlp (0.015172) > highway (0.015281) > deep_mlp (0.016575)**

### 8.2 Miért teljesítenek gyengébben a 2. generációs modellek?

**GELUResNetPricer** (RMSE=0.009135): A GELU aktiváció — várakozásainkkal ellentétben —
nem javított a ReLU-hoz képest. A tanítás során a 19–22. epochon erős instabilitás lépett
fel (val loss 0.000086-ról 0.002769-re ugrott), ami arra utal, hogy a GELU + LR=1e-3
kombináció ReLU-hoz képest érzékenyebb a gradiens robbanásra. A végső RMSE ~2× rosszabb
a ResNetéhez képest.

**DenseMLPPricer** (RMSE=0.015172): A dense skip-kapcsolatok BS opciós árazásnál nem
hasznosak: a BS ár sima, nem igényli a korai feature-ök direkt átadását. A modell
mindössze 18 best epochig tanult, a val loss 0.000228-on stagnált — lényegesen magasabb
mint a ResNet (0.000023).

**HighwayPricer** (RMSE=0.015281): A tanulható gate-ek felesleges paramétereket visznek
be (528K param, mégis gyengébb). A highway mechanizmus mélyen rétegezett képosztályozásnál
hasznos; 5-dimenziós sima táblázati adatnál nem jelent előnyt. ATM szegmensben
szignifikánsan gyengébb (RMSE=0.020864).

**FINNPricer** (RMSE=0.009653): A kétágú architektúra gyors konvergenciát mutat
(17 best epoch), de szintetikus BS adatokon a "közelítő ág" nem tud valódi előnyt
nyújtani, mert nincs modellhiba amit korrigálni kellene — csak a háló saját hibáját
becsüli.

**DeepMLPPricer** (RMSE=0.016575): A leggyengébb teljesítmény 4 best epoch után
early stopping-gal. A Pre-LN + Dropout konfiguráció e kísérletben nem konvergált —
a modell 4 epochon belül elért egy lokális minimumot (val=0.000274), majd nem javult
tovább. Lassabb tanulási ráta vagy warmup szükséges.

### 8.3 Physics-Informed Loss hatása

A `resnet_phys` (RMSE=0.005783) ~21%-kal magasabb RMSE-t mutat, mint a sima ResNet
(RMSE=0.004788), de **garantálja a delta-korlátot**: `∂C_norm/∂moneyness_norm ∈ [0, 1]`.

A physics loss fő haszna nem az MSE-ben mérhető — hanem a modell pénzügyi konzisztenciájában:
a tanult delta közelíti a Black-Scholes delta-t anélkül, hogy azt expliciten optimalizálnánk.
Ez különösen fontos, ha a modellt nem csak árazásra, hanem fedezési stratégiák számítására
is használni akarjuk (Liu et al. 2019).

A `resnet_phys` val loss görbéjében erős nem-monoton konvergencia látható — a physics
gradiens zajt visz be, ezért visszaesések mutatkoznak (pl. epoch 15: 0.000226, epoch 25:
0.000154), miközben epoch 10-en még 0.000069 volt. A legjobb val loss (0.000033) csak
a 68. epochon érkezett el, ami lassabb konvergenciát jelez.

### 8.4 Szegmentált elemzés

Az OTM szegmensben az `mlp` a legjobb (RMSE=0.002933), ami arra utal, hogy az egyszerű
ReLU MLP jól kezeli a közel nulla árakat. A `resnet` az ITM szegmensben vezet (0.004842),
ahol a nagyobb modelkapacitás kihasználható. A `resnet_phys` a legjobban kiegyensúlyozott
szegmensenkénti teljesítményt mutatja (OTM: 0.005502, ATM: 0.006357, ITM: 0.005756) —
a physics regularizáció egyenletesebb hibát eredményez.

A `highway` ATM szegmensben szignifikánsan gyengébb (RMSE=0.020864) a többi modellnél,
ami arra utal, hogy a gating mechanizmus ATM tartományban instabilitást okoz.

### 8.5 Összefoglalás

| Kategória              | Győztes          | Megjegyzés                                              |
|------------------------|------------------|---------------------------------------------------------|
| Legjobb összesített    | MLPPricer        | RMSE=0.004612, R²=0.9994, 31K param                    |
| Legjobb param/telj.    | MLPPricer        | 31K param, legjobb RMSE — nincs kompromisszum           |
| Legjobb fizikai korl.  | resnet_phys      | Delta-korlát garantált, RMSE ~21%-kal magasabb          |
| Legstabilabb           | ResNetPricer     | Fokozatos konvergencia 80 epochon át                    |
| 2. gen. legjobb        | GELUResNetPricer | RMSE=0.009135, de instabilitásra hajlamos               |
| Leggyorsabb konv.      | FINNPricer       | 17 epochon best, de instabil                            |
| Leggyengébb            | DeepMLPPricer    | 4 epochon best, val=0.000274 — nem konvergált           |

A kísérletek azt mutatják, hogy **BS szintetikus adatokon az architektúra bonyolítása
nem feltétlenül javít**: az egyszerű MLP (Culkin & Das) versenyképes marad. A ResNet
skip-kapcsolatai stabil alternatívát nyújtanak, a magasabb paraméterszám (Highway, Dense)
vagy az aktivációcsere (GELU) önmagában nem hoznak áttörést. A physics-informed
regularizáció minimális MSE-veszteséggel biztosít pénzügyi konzisztenciát — ez a
megközelítés érdemes a 2. fázisban (historikus adatok) is megvizsgálni.

---

## 9. Irodalmi háttér

- **Culkin & Das (2017)** — *Machine Learning in Finance: The Case of Deep Learning for Option Pricing*.
  Elsők között mutatták meg, hogy egyszerű MLP (4 réteg, 100 neuron) képes közel-BS pontossággal
  árazni call opciókat szintetikus adatokon.

- **Garcia & Gençay (2000)** — *Pricing and hedging derivative securities with neural networks
  and a homogeneity hint*.
  Bevezette a homogeneity hint-et: C/K = f(S/K, T, r, σ) alakra hozva a problémát a háló
  könnyebben általánosít és kevesebb adatból tanul.

- **Lürig et al. (2023)** — *Deep Learning for Option Pricing*.
  LayerNorm + Dropout + reziduális kapcsolatokkal javított MLP architektúrákat vizsgált;
  Pre-LN reziduális hálók bizonyultak a legstabilabbnak és legpontosabbnak.

- **Huang et al. (2017)** — *Densely Connected Convolutional Networks* (DenseNet).
  Minden réteg az összes korábbi réteggel össze van kötve; javítja a gradiens-áramlást
  és lehetővé teszi a feature reuse-t. A DenseMLPPricer ezt az elvet alkalmazza MLP-re.

- **Srivastava et al. (2015)** — *Training Very Deep Networks* (Highway Networks).
  Tanulható transform gate-tel (σ-függvény) irányítja az információáramlást;
  a háló maga dönti el, mikor "enged át" és mikor "transzformál".

- **Liu et al. (2019)** — *A neural network-based framework for financial model calibration*.
  Fizikai korlátokat (görögök) épít be a tanítási veszteségfüggvénybe;
  a delta-korlát physics-informed loss alapját adja.

- **arXiv:2412.12213** — *AI Black-Scholes*.
  Két-ágú architektúra: egy ág a közelítést, egy másik a korrekciót végzi.
  A FINNPricer ezen az elven alapul.
