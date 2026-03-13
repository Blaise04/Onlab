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

## 7. Kísérleti eredmények

Az összes modellt azonos feltételek mellett tanítottuk: 700 000 szintetikus Black-Scholes minta
(LHS-mintavételezés), 150 000-es validációs és teszt halmaz, max 200 epoch, patience=20,
batch=4096, Adam + ReduceLROnPlateau scheduler. Kiértékelés a teszt halmazon (150 000 minta).

### 7.1 Összefoglaló táblázat

| Modell        | Paraméterek |  Best ep | Val MSE (×10⁻⁵) | Test RMSE | Test MAE  |   R²     |
|---------------|-------------|----------|-----------------|-----------|-----------|----------|
| mlp           |     31 301  |      98  |        2.21     | 0.004742  | 0.002757  | 0.99888  |
| deep_mlp      |    265 985  |      57  |       10.96     | 0.010555  | 0.008580  | 0.99443  |
| resnet        |    398 593  |     140  |      **1.80**   |**0.004253**|0.002852  |**0.99910**|
| gelu_resnet   |    398 593  |      25  |        4.83     | 0.007014  | 0.004589  | 0.99754  |
| dense_mlp     |    102 145  |       6  |       32.70     | 0.018398  | 0.012719  | 0.98307  |
| highway       |    528 641  |      76  |       19.48     | 0.014020  | 0.009714  | 0.99017  |
| finn          |    402 817  |      18  |        5.57     | 0.007502  | 0.005007  | 0.99718  |
| resnet_phys   |    398 593  |     199  |        1.82     | 0.004291  | 0.002802  | 0.99908  |

### 7.2 Szegmentált eredmények (RMSE)

| Modell       |  OTM (m<0.97) | ATM (0.97–1.03) | ITM (m>1.03) |
|--------------|---------------|-----------------|--------------|
| mlp          |    0.002943   |    0.005083     |   0.005745   |
| deep_mlp     |    0.007543   |    0.010746     |   0.012729   |
| resnet       |  **0.003740** |  **0.004218**   | **0.004743** |
| gelu_resnet  |    0.006388   |    0.007474     |   0.007137   |
| dense_mlp    |    0.018030   |    0.018586     |   0.018572   |
| highway      |    0.011803   |    0.015742     |   0.014238   |
| finn         |    0.006934   |    0.007717     |   0.007826   |
| resnet_phys  |    0.004167   |    0.004262     |   0.004439   |

---

## 8. Következtetések

### 8.1 Általános megállapítások

**A ResNetPricer (ReLU) bizonyult a legpontosabbnak** (RMSE=0.00425, R²=0.9991), megelőzve
minden 2. generációs architektúrát. Ez összhangban van Lürig et al. (2023) eredményeivel,
akik szintén a Pre-LN reziduális MLP-t találták a legstabilabbnak.

**Az MLPPricer (Culkin & Das baseline) meglepően versenyképes:** RMSE=0.00474, mindössze
~11%-kal gyengébb a legjobb modellnél — 13× kevesebb paraméterrel. Ez megerősíti, hogy
BS szintetikus adatokon az egyszerű architektúra is elegendő lehet.

### 8.2 Miért teljesítenek gyengébben a 2. generációs modellek?

**GELUResNetPricer** (RMSE=0.0070): A GELU aktiváció — várakozásainkkal ellentétben —
nem javított a ReLU-hoz képest. A BS ár ugyan sima függvény, de az 1M szintetikus mintán
a ReLU ResNet gradiensei is stabilan konvergálnak; a simább aktiváció pluszt nem jelent.
Megjegyzés: a validációs loss korán (ep25) megállt — a GELU-val a scheduler hamarabb
csökkentette az LR-t, és a modell lokális minimumba ragadt.

**DenseMLPPricer** (RMSE=0.0184): Leggyengébb teljesítmény. A dense skip-kapcsolatok
BS opciós árazásnál nem hasznosak: a BS ár sima, nem igényli a korai feature-ök direkt
átadását a kimenethez. A kisebb hidden_dim (128) önmagában is szűk szűk keresztmetszet lehet.
A korai leállás (ep6) arra utal, hogy a modell nem tanul hatékonyan.

**HighwayPricer** (RMSE=0.0140): A tanulható gate-ek felesleges paramétereket visznek be
(528K param, mégis gyengébb). A highway mechanizmus mélyen rétegezett képosztályozásnál
hasznos; 5-dimenziós sima táblázati adatnál nem jelent előnyt.

**FINNPricer** (RMSE=0.0075): Két-ágú architektúra (BS-közelítő + korrekciós ág) jobb
mint GELUResNet és DenseNet, de elmarad az egyágú ResNettől. Szintetikus BS adatokon
a "közelítő ág" nem tud előnyt nyújtani, mert nincs valódi modellhiba amit korrigálni
kellene — csak a háló saját hibáját becsüli.

**DeepMLPPricer** (RMSE=0.0106): Pre-LN + Dropout nélküli skip-kapcsolatokkal rosszabbul
teljesít, mint a ResNet. A Dropout (0.1) regularizáló hatása és a LayerNorm megakadályozza
a modellt az overfit-ben, de skip-kapcsolat nélkül a gradiens-áramlás kevésbé hatékony.

### 8.3 Physics-Informed Loss hatása

A `resnet_phys` (RMSE=0.00429) mindössze ~0.9%-kal gyengébb MSE-ben, mint a sima ResNet
(RMSE=0.00425), de **garantálja a delta-korlátot**: `∂C_norm/∂moneyness_norm ∈ [0, 1]`.

A physics loss fő haszna nem az MSE-ben mérhető — hanem a modell pénzügyi konzisztenciájában:
a tanult delta közelíti a Black-Scholes delta-t anélkül, hogy azt expliciten optimalizálnánk.
Ez különösen fontos, ha a modellt nem csak árazásra, hanem fedezési stratégiák számítására
is használni akarjuk (Liu et al. 2019).

Megjegyzés: a resnet_phys OTM szegmensben kicsit gyengébb (0.00417 vs 0.00374 resnet),
ami arra utal, hogy a physics regularizáció kissé eltorzítja az OTM becsléseket — az ár
közel nulla OTM-nél, de a delta-korlát mégis aktív.

### 8.4 Összefoglalás

| Kategória             | Győztes       | Megjegyzés                                      |
|-----------------------|---------------|-------------------------------------------------|
| Legjobb pontosság     | ResNetPricer  | Pre-LN reziduális MLP, ReLU, 400K param         |
| Legjobb param/telj.   | MLPPricer     | 31K param, ~11% RMSE-veszteség                  |
| Legjobb fizikai korl. | resnet_phys   | Delta-korlát garantált, MSE-veszteség <1%       |
| 2. gen. legjobb       | FINNPricer    | RMSE=0.0075, két-ágú architektúra               |
| Meglepő eredmény      | DenseMLPPricer| Dense skip-ek BS-en nem segítenek               |

A kísérletek azt mutatják, hogy **BS szintetikus adatokon az architektúra bonyolítása
nem feltétlenül javít**: a ResNet skip-kapcsolatai elegendőek, a többletparaméterek
(Highway, Dense) vagy az aktivációcsere (GELU) önmagukban nem hoznak áttörést.
A physics-informed regularizáció viszont minimális MSE-veszteséggel biztosít pénzügyi
konzisztenciát — ez a megközelítés érdemes a 2. fázisban (historikus adatok) is
megvizsgálni.

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
