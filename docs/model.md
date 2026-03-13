# Neurális háló opciós árazó — Architektúra dokumentáció

## 1. Áttekintés

A projekt 1. fázisában Black-Scholes szintetikus adatokon tanított neurális hálók
call opció árakat becsülnek. Három architektúrát valósítunk meg, az irodalom alapján:

| Modell         | Irodalom                   | Paraméterek |
|----------------|----------------------------|-------------|
| MLPPricer      | Culkin & Das (2017)        | ~30 900     |
| DeepMLPPricer  | Lürig et al. (2023)        | ~265 000    |
| ResNetPricer   | Lürig et al. (2023)        | ~400 000    |

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

# Tanítás
python train.py --model mlp      --epochs 200 --output models/
python train.py --model deep_mlp --epochs 200 --output models/
python train.py --model resnet   --epochs 200 --output models/

# Smoke test (gyors, 10 epoch)
python train.py --model mlp --epochs 10 --output models/

# Kiértékelés
python evaluate.py --checkpoint models/mlp_best.pt
python evaluate.py --checkpoint models/mlp_best.pt models/deep_mlp_best.pt \
    models/resnet_best.pt --compare

# Forward pass ellenőrzés
python -c "
from src.model import MLPPricer, DeepMLPPricer, ResNetPricer, count_parameters
import torch
x = torch.randn(16, 5)
for M in [MLPPricer(), DeepMLPPricer(), ResNetPricer()]:
    y = M(x)
    print(type(M).__name__, y.shape, count_parameters(M))
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
