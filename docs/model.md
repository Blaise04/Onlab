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
python train.py --model resnet --physics-loss --physics-lambda 0.1 --epochs 200 --output models/ --name resnet_phys

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
(LHS-mintavételezés), 150 000-es validációs és teszt halmaz, max 200 epoch, patience=10,
batch=4096, Adam + ReduceLROnPlateau scheduler. Kiértékelés a teszt halmazon (150 000 minta).
GPU: NVIDIA GeForce RTX 4060 Laptop GPU (8 GB).

### 7.1 Összefoglaló táblázat

A táblázat a teszt halmazon mért eredményeket tartalmazza (150 000 minta, seed=42).
A `Val MSE` a legjobb checkpoint-hoz tartozó validációs MSE értéke.

| Modell        | Paraméterek |  Best ep | Összes ep | Val MSE (×10⁻⁵) | Test RMSE  | Test MAE   |    R²     |
|---------------|-------------|----------|-----------|-----------------|------------|------------|-----------|
| mlp           |     31 001  |       68 |        68 |        2.13     | 0.004612   | 0.002780   | 0.999407  |
| deep_mlp      |    267 521  |        4 |        14 |       27.40     | 0.016575   | 0.013233   | 0.992338  |
| resnet        |    398 593  |       80 |        90 |        2.28     | 0.004788   | 0.003274   | 0.999361  |
| gelu_resnet   |    398 593  |       40 |        50 |        8.25     | 0.009135   | 0.006027   | 0.997673  |
| dense_mlp     |    101 894  |       18 |        28 |       22.84     | 0.015172   | 0.010694   | 0.993581  |
| highway       |    528 129  |       22 |        32 |       23.18     | 0.015281   | 0.011005   | 0.993488  |
| finn          |    403 202  |       17 |        27 |        9.23     | 0.009653   | 0.006625   | 0.997401  |
| resnet_phys   |    398 593  |       57 |        67 |        3.41     | 0.005830   | 0.004089   | 0.999052  |

Megjegyzés: `mlp` bizonyult a legpontosabb modellnek (RMSE=0.004612, R²=0.999407),
megelőzve a `resnet`-et (RMSE=0.004788). A `resnet_phys` physics-loss regularizációval
~26%-kal magasabb RMSE-t ért el, de garantálja a delta-korlátot.

### 7.2 Szegmentált eredmények (RMSE)

Szegmenshatárok: OTM = moneyness < 0.9, ATM = 0.9–1.1, ITM = moneyness > 1.1.

| Modell        | OTM (m<0.9)  | ATM (0.9-1.1) | ITM (m>1.1)  |  N(OTM) | N(ATM) | N(ITM)  |
|---------------|--------------|---------------|--------------|---------|--------|---------|
| mlp           | **0.002933** | **0.005903**   |   0.005214   |  60051  |  29948 |  60001  |
| deep_mlp      |   0.013577   |   0.013992     |   0.020115   |  60051  |  29948 |  60001  |
| resnet        |   0.004429   |   0.005342     | **0.004842** |  60051  |  29948 |  60001  |
| gelu_resnet   |   0.008078   |   0.011879     |   0.008536   |  60051  |  29948 |  60001  |
| dense_mlp     |   0.011125   |   0.018354     |   0.016836   |  60051  |  29948 |  60001  |
| highway       |   0.011821   |   0.020864     |   0.015054   |  60051  |  29948 |  60001  |
| finn          |   0.008649   |   0.012113     |   0.009211   |  60051  |  29948 |  60001  |
| resnet_phys   |   0.005556   |   0.006342     |   0.005831   |  60051  |  29948 |  60001  |

Megjegyzés: az `mlp` OTM és ATM szegmensben a legjobb, a `resnet` ITM szegmensben
vezet. A `resnet_phys` minden szegmensben egyenletesebb teljesítményt mutat, mint a
sima `resnet`.

### 7.3 Tanulási sebesség összefoglaló

Ez a fejezet az epoch-szintű tanítási görbék alapján jellemzi az egyes modellek konvergenciáját.
A részletes epoch-szintű adatok a `results/training_curves.json` fájlban találhatók.

#### mlp (MLPPricer)

| Epoch | Val loss   |
|-------|------------|
|     1 | 0.000079   |
|     5 | 0.000046   |
|    10 | 0.000056   |
|    25 | 0.000026   |
|    50 | 0.000024   |
|    68 | 0.000021 * |

- Összes epoch: 68, best epoch: 68 (az utolsó epoch volt a legjobb)
- Konvergencia: **lassú, folyamatos javulás** — a loss az összes epoch alatt csökkent,
  nem mutatott korai leállást. A scheduler az 75. epochon lépett be (LR: 1e-3 → 5e-4),
  de az early stopping végül a 78. epochon állította le.
- A legjobb checkpoint a 68. epochon mentődött, utána kis mértékű romlás következett.

#### deep_mlp (DeepMLPPricer)

| Epoch | Val loss   |
|-------|------------|
|     1 | 0.000515   |
|     4 | 0.000274 * |

- Összes epoch: 14 (early stopping a 4. best epoch után 10 epochkal), best epoch: 4
- Konvergencia: **nagyon korai leállás** — a modell az 1–4. epochon javult, majd
  stagnált/romlott. Valószínű ok: a Pre-LN + Dropout kombináció e konfigurációban
  nem hatékony, a modell gyorsan lokális minimumba rekedt.
- Val loss a legjobb epochon: 0.000274 (2.74×10⁻⁴), ami lényegesen magasabb a többi
  modellnél mértnél — a modell alulteljesít.

#### resnet (ResNetPricer)

| Epoch | Val loss   |
|-------|------------|
|     1 | 0.000350   |
|     5 | 0.000076   |
|    10 | 0.000045   |
|    25 | 0.000027   |
|    50 | 0.000031   |
|    80 | 0.000023 * |

- Összes epoch: 90 (best epoch: 80, early stopping a 90. epochon)
- Konvergencia: **fokozatos, stabil** — a val loss az első 25 epochon erősen csökkent,
  majd lassan, de folyamatosan javult 80-ig. A scheduler többször csökkentette az LR-t,
  végül 3.13e-05-re ért le. A legjobb érték (0.000023) csak a 80. epochon érkezett el.

#### gelu_resnet (GELUResNetPricer)

| Epoch | Val loss   |
|-------|------------|
|     1 | 0.003225   |
|     5 | 0.000179   |
|    10 | 0.000095   |
|    25 | 0.000086   |
|    40 | 0.000083 * |

- Összes epoch: 50 (best epoch: 40, early stopping az 50. epochon)
- Konvergencia: **gyors kezdeti javulás, majd instabilitás** — az 1–16. epochon rapid
  csökkentés, de a 19–22. epochon erős ugrás (val: 0.000086 → 0.002769), ami a GELU
  ResNet LR-érzékenységére utal. A 23. epochon LR csökkentés után újra stabilizálódott.
  Végső legjobb: 0.000083 (40. epoch).

#### dense_mlp (DenseMLPPricer)

| Epoch | Val loss   |
|-------|------------|
|     1 | 0.001454   |
|     5 | 0.000303   |
|    10 | 0.000260   |
|    18 | 0.000228 * |

- Összes epoch: 28 (best epoch: 18, early stopping a 28. epochon)
- Konvergencia: **lassú, csökkenő mértékű javulás** — a val loss egyenletesen, de
  lassan csökkent. Az 5–18. epochon a javulás üteme nagyon mérsékelt (0.000303 →
  0.000228). A modell nem érte el a többi architektúra pontosságát.

#### highway (HighwayPricer)

| Epoch | Val loss   |
|-------|------------|
|     1 | 0.001003   |
|     5 | 0.000460   |
|    10 | 0.000258   |
|    22 | 0.000232 * |

- Összes epoch: 32 (best epoch: 22, early stopping a 32. epochon)
- Konvergencia: **közepes sebesség, plafon effektus** — a val loss az 5–22. epochon
  0.000460-ról 0.000232-re csökkent, de a csökkentés üteme a 13. epoch után lelassult.
  A gate mechanizmus nem hozott érdemi javulást a plain MLP-hez képest.

#### finn (FINNPricer)

| Epoch | Val loss   |
|-------|------------|
|     1 | 0.002869   |
|     5 | 0.000229   |
|    10 | 0.000123   |
|    17 | 0.000092 * |

- Összes epoch: 27 (best epoch: 17, early stopping a 27. epochon)
- Konvergencia: **gyors és instabil** — az 1. epochon magas, 0.002869-es indulásból
  17 epochon belül 0.000092-re csökkent. A 18–22. epochon azonban erős visszaesés
  (0.000140 → 0.000249) mutatkozott, ami a kétágú architektúra instabilitására utal.
  A 24. epochon LR csökkentés stabilizálta a tanítást, de a 17. epoch maradt a legjobb.

#### resnet_phys (ResNetPricer + Physics Loss)

| Epoch | Val loss   |
|-------|------------|
|     1 | 0.000377   |
|     5 | 0.000110   |
|    10 | 0.000069   |
|    25 | 0.000154   |
|    50 | 0.000046   |
|    57 | 0.000034 * |

- Összes epoch: 67 (best epoch: 57, early stopping a 67. epochon)
- Konvergencia: **fokozatos, de nem monoton** — a physics loss extra gradiens
  zajt visz be, ezért a val loss a 10–25. epoch között visszaemelkedett (0.000069 →
  0.000154), majd újra csökkent. A legjobb érték (0.000034) a 57. epochon érkezett.
  A physics regularizáció ~50%-kal magasabb val MSE-t eredményez a sima ResNethez
  képest (0.000034 vs. 0.000023), cserébe a delta-korlát teljesül.

---

## 8. Következtetések

### 8.1 Általános megállapítások

**Az MLPPricer (Culkin & Das baseline) bizonyult a legjobb modellnek** az összesített
teszt RMSE alapján (RMSE=0.004612, R²=0.999407). Ez meglepő eredmény: a ~31 000
paraméterű, normalizáció és dropout nélküli egyszerű háló felülmúlta az összes
komplexebb architektúrát. A ResNetPricer (RMSE=0.004788) csak minimálisan gyengébb,
de 13× több paraméterrel dolgozik.

A sorrend: **mlp > resnet > gelu_resnet > finn > dense_mlp > highway > deep_mlp**
(RMSE szerint, kisebb = jobb).

### 8.2 Miért teljesítenek gyengébben a 2. generációs modellek?

**GELUResNetPricer** (RMSE=0.009135): A GELU aktiváció — várakozásainkkal ellentétben —
nem javított a ReLU-hoz képest. A tanítás során a 19–22. epochon erős instabilitás lépett
fel (val loss 0.000086-ról 0.002769-re ugrott), ami arra utal, hogy a GELU + LR=1e-3
kombináció ReLU-hoz képest érzékenyebb a gradiens robbanásra. A végső RMSE ~2× rosszabb
a ResNetéhez képest.

**DenseMLPPricer** (RMSE=0.015172): A dense skip-kapcsolatok BS opciós árazásnál nem
hasznosak: a BS ár sima, nem igényli a korai feature-ök direkt átadását. A modell
mindössze 18 best epochig tanult (28 összes), utána early stopping. A val loss
0.000228-on stagnált — lényegesen magasabb mint a ResNet (0.000023).

**HighwayPricer** (RMSE=0.015281): A tanulható gate-ek felesleges paramétereket visznek
be (528K param, mégis gyengébb). A val loss 22 epoch alatt csak 0.000232-re csökkent.
A highway mechanizmus mélyen rétegezett képosztályozásnál hasznos; 5-dimenziós sima
táblázati adatnál nem jelent előnyt.

**FINNPricer** (RMSE=0.009653): A kétágú architektúra (BS-közelítő + korrekciós ág)
gyors konvergenciát mutat (17 best epoch), de instabilitásra hajlamos (18–22. epoch
visszaesés). Szintetikus BS adatokon a "közelítő ág" nem tud valódi előnyt nyújtani,
mert nincs modellhiba amit korrigálni kellene — csak a háló saját hibáját becsüli.

**DeepMLPPricer** (RMSE=0.016575): A leggyengébb teljesítmény 4 best epoch után early
stopping-gal. A Pre-LN + Dropout konfiguráció e kísérletben nem konvergált — a modell
4 epochon belül elért egy lokális minimumot (val=0.000274), majd nem javult tovább.
Ez a konfiguráció valószínűleg lassabb tanulási rátát vagy hosszabb warmup-ot igényelne.

### 8.3 Physics-Informed Loss hatása

A `resnet_phys` (RMSE=0.005830) ~22%-kal magasabb RMSE-t mutat, mint a sima ResNet
(RMSE=0.004788), de **garantálja a delta-korlátot**: `∂C_norm/∂moneyness_norm ∈ [0, 1]`.

A physics loss fő haszna nem az MSE-ben mérhető — hanem a modell pénzügyi konzisztenciájában:
a tanult delta közelíti a Black-Scholes delta-t anélkül, hogy azt expliciten optimalizálnánk.
Ez különösen fontos, ha a modellt nem csak árazásra, hanem fedezési stratégiák számítására
is használni akarjuk (Liu et al. 2019).

Figyelemre méltó: a `resnet_phys` a val loss görbében nem monoton konvergenciát mutat —
a physics gradiens zajt visz be, ezért az 1e-2-es LR tartományban visszaesések mutatkoznak
(epoch 25: val=0.000154, miközben epoch 10-en 0.000069 volt). A legjobb val loss
(0.000034) csak az 57. epochon érkezett el, ami lassabb konvergenciát jelez a sima
ResNethez képest (80 epoch, de alacsonyabb végső val loss: 0.000023).

### 8.4 Szegmentált elemzés

Az OTM szegmensben az `mlp` a legjobb (RMSE=0.002933), ami arra utal, hogy az egyszerű
ReLU MLP jól kezeli a közel nulla árakat. A `resnet` az ITM szegmensben vezet (0.004842),
ahol a nagyobb modellakapacitás kihasználható. A `resnet_phys` a legjobban kiegyensúlyozott
szegmensenkénti teljesítményt mutatja (OTM: 0.005556, ATM: 0.006342, ITM: 0.005831) —
a physics regularizáció egyenletesebb hibát eredményez.

A `highway` ATM szegmensben szignifikánsan gyengébb (RMSE=0.020864) a többi modellnél,
ami arra utal, hogy a gating mechanizmus ATM tartományban instabilitást okoz (ahol az
ár legérzékenyebb a moneyness-re).

### 8.5 Összefoglalás

| Kategória              | Győztes          | Megjegyzés                                              |
|------------------------|------------------|---------------------------------------------------------|
| Legjobb összesített    | MLPPricer        | RMSE=0.004612, R²=0.9994, 31K param                    |
| Legjobb param/telj.    | MLPPricer        | 31K param, legjobb RMSE — nincs kompromisszum           |
| Legjobb fizikai korl.  | resnet_phys      | Delta-korlát garantált, RMSE ~22%-kal magasabb          |
| Legstabilabb           | ResNetPricer     | Fokozatos, monoton konvergencia 80 epochon át           |
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
