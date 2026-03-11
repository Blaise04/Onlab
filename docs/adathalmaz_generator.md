# Black-Scholes szintetikus adathalmaz generátor

Dokumentáció a `generate_dataset.py` CLI szkripthez és a mögöttes `src/` modulokhoz.

---

## Áttekintés

A generátor véletlen paraméter-kombinációkhoz Black-Scholes zárt képlettel számítja ki az opció árakat és görögöket, majd train/val/test halmazokra bontva menti az eredményt. A cél neurális hálók betanításához alkalmas, reprodukálható szintetikus adathalmaz előállítása.

### Fájlstruktúra

```
src/
  black_scholes.py     # BS képlet és görögök (tiszta matematika)
  data_generator.py    # mintavételezés, DataFrame összerakás, mentés
generate_dataset.py    # CLI belépési pont
data/                  # generált kimeneti fájlok (train/val/test)
```

---

## Gyors indítás

```bash
# Alapértelmezett: 100 000 minta, LHS, CSV kimenet
python generate_dataset.py

# Ajánlott konfiguráció neurális hálóhoz
python generate_dataset.py --n 300000 --method lhs --greeks --normalize --output data/
```

---

## CLI argumentumok

| Argumentum | Típus | Default | Leírás |
|------------|-------|---------|--------|
| `--n` | int | 100 000 | Generálandó minták száma |
| `--method` | str | `lhs` | Mintavételezési módszer: `uniform`, `lhs`, `grid` |
| `--output` | str | `data/` | Kimeneti mappa |
| `--format` | str | `csv` | Fájlformátum: `csv` vagy `parquet` |
| `--greeks` | flag | ki | Görögök (delta, gamma, vega, theta, rho) számítása |
| `--normalize` | flag | ki | Moneyness (S/K) és normált call ár (call/K) hozzáadása |
| `--noise` | float | 0.0 | Gauss-zaj szórása az opció árakhoz |
| `--seed` | int | 42 | Véletlenszám mag (reprodukálhatóság) |

---

## Paramétertartományok

Az összes bemeneti paraméter az alábbi intervallumokból kerül mintavételezésre:

| Változó | Minimum | Maximum | Leírás |
|---------|---------|---------|--------|
| S | 10.0 | 150.0 | Mögöttes eszköz árfolyama |
| K | 7.0 | 650.0 | Kötési ár |
| T | 0.005 | 2.0 | Lejáratig hátralévő idő (év) |
| r | 0.00 | 0.05 | Kockázatmentes kamatláb |
| sigma | 0.05 | 0.90 | Volatilitás |
| q | 0.00 | 0.03 | Folyamatos osztalékhozam |

A tartományok forrása: `docs/bs_szintetikus_adatok.md` (7. fejezet).

---

## Mintavételezési módszerek

### `uniform`
`np.random.uniform` alapú, független egyenletes eloszlású mintavételezés minden dimenzióban. Egyszerű, de nagy mintaszámnál klaszteresedhet (Culkin & Das, 2017 stílus).

### `lhs` (ajánlott)
Latin Hypercube Sampling (`scipy.stats.qmc.LatinHypercube`). Az egyes dimenziókban garantáltan egyenletes rácsfelbontást biztosít, ezért jobb lefedettséget nyújt azonos mintaszám mellett. Neurális hálók betanításához ez az ajánlott módszer.

### `grid`
Descartes-szorzat alapú rácsmintázat (`itertools.product`). Az `--n` értékéből visszaszámolt dimenziónkénti lépésszámmal dolgozik (közelítőleg `n^(1/6)` lépés dimenziónként). Determinisztikus, jól reprodukálható, de exponenciálisan nő a dimenzióval (Tidy Finance stílus).

---

## Kimeneti oszlopok

### Alap (mindig jelen van)

| Oszlop | Leírás |
|--------|--------|
| `S` | Mögöttes eszköz árfolyama |
| `K` | Kötési ár |
| `T` | Lejáratig hátralévő idő (év) |
| `r` | Kockázatmentes kamatláb |
| `sigma` | Volatilitás |
| `q` | Osztalékhozam |
| `call_price` | Black-Scholes call opció ára |
| `put_price` | Black-Scholes put opció ára (put-call paritásból) |

### Görögök (`--greeks` flag)

| Oszlop | Görög | Leírás |
|--------|-------|--------|
| `delta` | Δ | ∂C/∂S – érzékenység az árfolyamra |
| `gamma` | Γ | ∂²C/∂S² – delta görbülete |
| `vega` | ν | ∂C/∂σ · 0.01 – 1%-pontos volatilitás-változásra |
| `theta` | Θ | ∂C/∂T / 365 – napi időérték-csökkenés |
| `rho` | ρ | ∂C/∂r · 0.01 – 1%-pontos kamatláb-változásra |

### Normalizáció (`--normalize` flag)

| Oszlop | Leírás |
|--------|--------|
| `moneyness` | S / K – dimenziótlan pénzesség |
| `call_price_norm` | call_price / K – normált call ár |

---

## Kimeneti fájlok

A generátor automatikusan 70% / 15% / 15% arányban bontja szét az adathalmazt:

```
data/
  train.csv   # 70% – betanítás
  val.csv     # 15% – validáció (hiperparaméter-hangolás)
  test.csv    # 15% – végső kiértékelés
```

Parquet formátumhoz (`--format parquet`) szükséges a `pyarrow` csomag.

---

## A Black-Scholes képlet

```
d1 = [ ln(S/K) + (r - q + σ²/2) · T ] / (σ · √T)
d2 = d1 - σ · √T

C  = S · e^(-qT) · N(d1) - K · e^(-rT) · N(d2)
P  = C - S · e^(-qT) + K · e^(-rT)
```

ahol N(·) a standard normális eloszlás CDF-je.

**Peremeset (T ≈ 0):** intrinsic value visszaadása, görögök = 0.

**Ellenőrzés:** ATM eset (S = K = 100, T = 1, r = 0.05, σ = 0.2, q = 0) → C ≈ **10.4506**

---

## Python API

A CLI helyett közvetlenül is használható:

```python
from src.black_scholes import bs_call, bs_put, bs_delta
from src.data_generator import generate_dataset, save_dataset

# Egyedi számítás
price = bs_call(S=100, K=100, T=1, r=0.05, sigma=0.2)

# Vektorizált számítás (numpy tömb)
import numpy as np
S = np.array([90, 100, 110])
prices = bs_call(S, K=100, T=1, r=0.05, sigma=0.2)

# Adathalmaz generálás
df = generate_dataset(n=50000, method='lhs', include_greeks=True, normalize=True, seed=42)

# Mentés
save_dataset(df, output_path='data/', format='csv')
```

---

## Függőségek

```
numpy       # vektorizált számítás
scipy       # LHS mintavételezés (scipy.stats.qmc)
pandas      # DataFrame, CSV export
pyarrow     # parquet mentéshez (opcionális)
```

Telepítés:
```bash
pip install numpy scipy pandas pyarrow
```

---

## Kapcsolódó dokumentumok

- `docs/bs_szintetikus_adatok.md` – A paramétertartományok és módszertan irodalmi háttere