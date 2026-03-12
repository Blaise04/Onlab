# Black-Scholes neurális hálóhoz szintetikus adathalmaz paraméterei

## 1. Bevezetés

A Black-Scholes (BS) modell zárt formulát ad európai call és put opciók elméleti árának kiszámítására. Neurális háló betanításához szintetikus adathalmazt kell generálni: a bemeneti paraméterek (részvényár, kötési ár, volatilitás, stb.) különböző kombinációihoz kiszámítjuk a BS-képlettel az opció árát, amelyet a háló kimenetként tanul meg közelíteni.

A szintetikus adathalmaz előnye, hogy:
- korlátlan mennyiségű, zajmentes minta generálható,
- a paramétértér tetszőlegesen lefedható,
- a modell pontossága közvetlenül ellenőrizhető az analitikus megoldással szemben.

---

## 2. Bemeneti paraméterek

| Paraméter | Szimbólum | Leírás | Ajánlott tartomány | Eloszlás |
|---|---|---|---|---|
| Részvényár | S | Az alaptermék jelenlegi piaci ára | $10 – $150 (vagy S/K ∈ [0.7, 1.3]) | Uniform vagy log-normális (GBM) |
| Kötési ár | K | Az opció lehívási ára | $7 – $650 (Culkin) / $20 – $90 (Tidy Finance) | Uniform |
| Lejáratig hátralévő idő | T (τ) | Években mérve | 0.005 – 2.0 év (≈ 2 nap – 2 év) | Uniform |
| Kockázatmentes kamatláb | r | Éves kamatláb | 0% – 5% | Uniform |
| Osztalékhozam | q | Éves osztalékhozam | 0% – 3% | Uniform (opcionális paraméter) |
| Volatilitás | σ | Éves implikált/historikus volatilitás | 5% – 90% (Culkin) / 10% – 80% (Tidy Finance) | Uniform |

### Megjegyzések a bemeneti paraméterekhez

**Moneyness (S/K arány):**
- Reális tartomány: m ∈ [0.7, 1.3] — kizárja a mélyen out-of-the-money (OTM) és in-the-money (ITM) eseteket
- Kiterjesztett tartomány: m ∈ [0.3, 2.0] — szélesebb lefedettséghez

**Részvényár eloszlása:**
- Fizikailag a részvényár log-normális eloszlást követ (Geometriai Brown-mozgás, GBM)
- Szintetikus adatnál a legtöbb tanulmány egyszerűség kedvéért uniform eloszlást alkalmaz

**Lejárati idő:**
- Szűkített tartomány (τ ∈ [0.5, 1.0]) pontosabb neurális háló közelítést eredményez
- A neurális hálók a paramétértartomány **határain kevésbé pontosak**, ezért a szélső értékek (nagyon rövid vagy nagyon hosszú lejárat) kerülendők

---

## 3. Kimeneti paraméterek

| Kimenet | Szimbólum | Leírás | Megjegyzés |
|---|---|---|---|
| Call opció ára | C | Európai call opció Black-Scholes ára | Legelterjedtebb kimenet |
| Put opció ára | P | Európai put opció ára | Put-call paritással levezethető C-ből |
| Delta | Δ | ∂C/∂S — az opció ára változása az alaptermék árának egységnyi változására | Görög, ritkábban |
| Gamma | Γ | ∂²C/∂S² — a delta konvexitása | Görög, ritkábban |
| Vega | ν | ∂C/∂σ — érzékenység a volatilitásra | Görög, ritkábban |
| Theta | Θ | ∂C/∂T — időérték-csökkenés | Görög, ritkábban |
| Rho | ρ | ∂C/∂r — kamatérzékenység | Görög, ritkábban |

A legtöbb tanulmány kizárólag a **call opció árát** (C) használja kimenetként. A görögök tanítása külön modellt vagy multi-output architektúrát igényel.

---

## 4. Adatgenerálási módszerek

### 4.1 Kombinatorikus rács (Grid)
Minden paraméternek diszkrét értékkészletet adunk, és az összes lehetséges kombinációt kiszámítjuk.

- **Előny:** teljes lefedettség, reprodukálható
- **Hátrány:** exponenciálisan nő a méret (pl. 5 paraméter × 10 érték = 100 000 sor)
- **Példa (Tidy Finance):** ~1,5 millió megfigyelés

```python
import numpy as np
from itertools import product

S_vals = np.linspace(40, 60, 10)
K_vals = np.linspace(20, 90, 10)
r_vals = np.arange(0.00, 0.06, 0.01)
T_vals = np.arange(0.25, 2.25, 1/12)
sigma_vals = np.arange(0.10, 0.90, 0.10)

grid = list(product(S_vals, K_vals, r_vals, T_vals, sigma_vals))
```

### 4.2 Egyenletes véletlen mintavételezés (Uniform Random Sampling)
Minden paraméterhez véletlenszerűen, egyenletes eloszlásból veszünk mintát.

- **Előny:** egyszerű, gyors, n tetszőlegesen választható
- **Hátrány:** nem garantált az egyenletes lefedettség (klaszterezés lehetséges)
- **Példa (Culkin & Das):** 300 000 megfigyelés

```python
n = 300_000
S = np.random.uniform(10, 150, n)
K = np.random.uniform(7, 650, n)
T = np.random.uniform(0.005, 2.0, n)
r = np.random.uniform(0.00, 0.05, n)
sigma = np.random.uniform(0.05, 0.90, n)
```

### 4.3 Latin Hypercube Sampling (LHS)
A paramétérteret egyenlő valószínűségű cellákra osztja, és mindegyikből pontosan egy mintát vesz.

- **Előny:** egyenletesebb lefedettség uniform-nál, kevesebb mintával is jobb
- **Hátrány:** implementáció bonyolultabb
- **Ajánlott:** egyre inkább preferált a szakirodalomban

```python
from scipy.stats import qmc

sampler = qmc.LatinHypercube(d=5)
sample = sampler.random(n=300_000)
l_bounds = [10, 7, 0.005, 0.00, 0.05]
u_bounds = [150, 650, 2.0, 0.05, 0.90]
params = qmc.scale(sample, l_bounds, u_bounds)
```

### 4.4 Geometriai Brown-mozgás (GBM) szimulált árutak
A részvényár szimulált árutakból (path) kerül kinyerésre, ami log-normális eloszlást eredményez.

- **Előny:** realisztikus részvényár-eloszlás
- **Hátrány:** bonyolultabb, más paraméterek még mindig uniform-ok
- **Használat:** FINN / AI Black-Scholes tanulmány (arxiv: 2412.12213)

---

## 5. Normalizációs javaslatok

### 5.1 Lineáris homogenitás kihasználása (Culkin & Das)
A Black-Scholes call ár lineárisan homogén S és K-ban:

```
C(S, K, T, r, σ) = K · C(S/K, 1, T, r, σ)
```

Ezért elegendő K=1-re normalizálni: az S/K (moneyness) arány lesz a bemenet S helyett.

**Előny:** a dimenzionalitás csökken (6 → 5 effektív paraméter), a háló gyorsabban konvergál.

### 5.2 Standard normalizáció
Minden bemeneti paramétert [0, 1] vagy [-1, 1] intervallumra normalizálni (min-max vagy z-score).

### 5.3 Kimeneti normalizáció
- A call ár normalizálható C/S (az alaptermék árához viszonyítva) vagy C/K (kötési árhoz viszonyítva) formában.

---

## 6. Összehasonlító táblázat

| Tanulmány | Adathalmaz mérete | S tartomány | K tartomány | T tartomány | r tartomány | σ tartomány | Eloszlás | Zaj |
|---|---|---|---|---|---|---|---|---|
| Culkin & Das (2017) | 300 000 | $10–$50 | $7–$650 | 1 nap–3 év | 1%–3% | 5%–90% | Nem normális | Nincs |
| Tidy Finance (Python) | ~1 500 000 | $40–$60 | $20–$90 | 3 hó–2 év | 0%–5% | 10%–80% | Grid | N(0, 0.15) |
| PINN (2312.06711) | — | [0, 160] | 40 (fix) | [0, 1] év | 0.05 (fix) | 0.2 (fix) | Folytonos | Nincs |
| FINN / AI BS (2412.12213) | — | GBM szimulált | Uniform | Uniform | 0% | — | Log-normális + Uniform | Nincs |

---

## 7. Javasolt konfiguráció saját kísérlethez

Az alábbi beállítás kiindulópontként ajánlott:

| Paraméter | Tartomány | Típus | Eloszlás |
|---|---|---|---|
| S | [10, 150] | mintavételezett | Uniform |
| moneyness (S/K) | [0.7, 1.3] | mintavételezett | Uniform |
| K | — | levezetett: K = S / moneyness | — |
| T | [0.005, 2.0] év | mintavételezett | Uniform |
| r | [0.00, 0.05] | mintavételezett | Uniform |
| σ | [0.05, 0.90] | mintavételezett | Uniform |
| q | [0.00, 0.03] | mintavételezett | Uniform |
| **Kimenet: C, P** | BS-képlettel számítva | — | — |

**Megjegyzés a K generáláshoz:** S és moneyness mintavételezésével, majd K = S / moneyness levezetésével garantálható, hogy minden minta realisztikus moneyness tartományban legyen. A K közvetlen mintavételezése S-től függetlenül sok irreális mélyen OTM/ITM kombinációt eredményezne.

- **Adathalmaz mérete:** 100 000 – 500 000 megfigyelés
- **Mintavételezés:** Latin Hypercube Sampling (LHS)
- **Normalizáció:** moneyness alapból jelen van + kimenet K-val osztva (`--normalize`)
- **Train/val/test:** 70% / 15% / 15%

---

## 8. Hivatkozások

1. Culkin, R. & Das, S. R. (2017). *Machine Learning in Finance: The Case of Deep Learning for Option Pricing*. [PDF](https://srdas.github.io/Papers/BlackScholesNN.pdf)

2. Tidy Finance. *Option Pricing via Machine Learning with Python*. [Link](https://www.tidy-finance.org/python/option-pricing-via-machine-learning.html)

3. Benth, F. E. et al. (2023). *Physics Informed Neural Network for Option Pricing*. arXiv:2312.06711. [Link](https://arxiv.org/html/2312.06711v1)

4. Liao, W. et al. (2024). *The AI Black-Scholes: Finance-Informed Neural Network for Option Pricing*. arXiv:2412.12213. [Link](https://arxiv.org/html/2412.12213v1)

5. Liu, S. et al. (2019). *Pricing options and computing implied volatilities using neural networks*. arXiv:1901.08943. [PDF](https://arxiv.org/pdf/1901.08943)

6. Stanford CS230 (2019). *Option Pricing with Deep Learning*. [PDF](https://cs230.stanford.edu/projects_fall_2019/reports/26260984.pdf)
