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
| `moneyness_norm` | S/K arány normálva             | [0.5, 2.0]          |
| `T_norm`         | Lejáratig hátralévő idő        | [7/365, 2.0] év     |
| `r_norm`         | Kockázatmentes ráta            | [0.0, 0.10]         |
| `sigma_norm`     | Volatilitás                    | [0.05, 0.60]        |
| `q_norm`         | Osztalékhozam                  | [0.0, 0.05]         |

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
- `history` — tanítási görbe

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
python train.py --model resnet --physics-loss --physics-lambda 0.1 --epochs 200 --output models/

# Smoke test (gyors, 10 epoch)
python train.py --model gelu_resnet --epochs 10 --output models/

# Összehasonlító kiértékelés
python evaluate.py \
  --checkpoint models/mlp_best.pt models/resnet_best.pt \
              models/gelu_resnet_best.pt models/dense_mlp_best.pt \
              models/highway_best.pt models/finn_best.pt \
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

## 7. Irodalmi háttér

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
