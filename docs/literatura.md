# Irodalmi áttekintés: Neurális hálók opciós árazáshoz

## 1. Kulcspapírok

### Hutchinson, Lo & Poggio (1994)
- **Szerzők:** James M. Hutchinson, Andrew W. Lo, Tomaso Poggio
- **Cím:** A Nonparametric Approach to Pricing and Hedging Derivative Securities Via Learning Networks
- **Folyóirat:** The Journal of Finance, Vol. 49, No. 3
- **Architektúra:** Radial Basis Function (RBF) hálók és MLP
- **Kulcstanulság:** Elsők között mutatták meg, hogy neurális hálók képesek közel-BS minőségű árazást tanulni. A modell inputjai: S/K, T és r. Radial basis function hálók S&P 500 indexopciókra alkalmazva generalizálnak a BS képlethez hasonlóan.
- **Hivatkozás:** https://doi.org/10.1111/j.1540-6261.1994.tb05149.x

---

### Garcia & Gençay (2000)
- **Szerzők:** René Garcia, Ramazan Gençay
- **Cím:** Pricing and hedging derivative securities with neural networks and a homogeneity hint
- **Folyóirat:** Journal of Econometrics, Vol. 94
- **Architektúra:** 1 rejtett réteg, kb. 10–20 neuron
- **Kulcstanulság:** Bevezeti a **homogeneity hint** fogalmát: a BS call ár homogén 1. fokú S és K-ban, ezért az S/K (moneyness) és C/K normalizáció használható. Ez drasztikusan csökkenti a bemeneti tér dimenzióját és javítja az általánosítást. Ez az egyik legtöbbet idézett transzformációs alap.
- **Hivatkozás:** https://doi.org/10.1016/S0304-4076(99)00018-1

---

### Culkin & Das (2017)
- **Szerzők:** Robert Culkin, Sanjiv R. Das
- **Cím:** Machine Learning in Finance: The Case of Deep Learning for Option Pricing
- **Folyóirat:** Journal of Investment Management, Vol. 15, No. 4
- **Architektúra:** 4 rejtett réteg × 100 neuron, ReLU aktiváció, ~300 000 szintetikus BS adatpont
- **Kulcstanulság:** A deep MLP (4×100) már kiemelkedően teljesít szintetikus BS adatokon. Az architektúra egyszerűsége ellenére nagyon alacsony RMSE-t ér el. A 300k méretű szintetikus adathalmaz elegendőnek bizonyult. Ez az egyik legismertebb baseline a BS-neurális háló irodalomban.
- **Hivatkozás:** https://ssrn.com/abstract=3023505

---

### Liu, Oosterlee & Bohte (2019) – CaNN
- **Szerzők:** Shuaiqiang Liu, Cornelis W. Oosterlee, Sander M. Bohte
- **Cím:** Pricing options and computing implied volatilities using neural networks
- **Folyóirat:** Risks, Vol. 7, No. 1
- **Architektúra:** Calibration Neural Network (CaNN) – 3–4 rejtett réteg, batch normalization, különleges input-output transzformáció
- **Kulcstanulság:** Nem csak BS, hanem Heston és Bates modellekre is alkalmazzák. Bevezeti az implied volatility inverz problémájának hatékony megoldását neurális hálóval. A CaNN felülmúlja az egyszerű MLP-t kalibrálási feladatokban. Batch norm és skip connection javítja a stabilitást.
- **Hivatkozás:** https://doi.org/10.3390/risks7010016

---

### Ruf & Wang (2020)
- **Szerzők:** Johannes Ruf, Weiguan Wang
- **Cím:** Neural Networks for Option Pricing and Hedging: A Literature Review
- **Folyóirat:** Journal of Computational Finance, Vol. 24, No. 1
- **Architektúra:** Irodalmi áttekintés (nem saját modell)
- **Kulcstanulság:** Átfogó összehasonlítás ~15 neurális háló alapú opciós árazási megközelítésről. Megállapítja, hogy a Garcia–Gençay-féle moneyness-normalizáció szinte minden módszernél megjelenik. Empirikusan összehasonlítja a módszereket S&P 500 adatokon: a legtöbb NN alulteljesít hagyományos módszerekhez képest ha nincs megfelelő feature engineering.
- **Hivatkozás:** https://arxiv.org/abs/1911.05689

---

### Lürig, Wallmeier & Ziegler (2023)
- **Szerzők:** Matthias Lürig, Martin Wallmeier, Andreas Ziegler
- **Cím:** Neural network architecture comparison for European option pricing
- **Folyóirat:** arXiv preprint
- **Architektúra:** Empirikus összehasonlítás: MLP vs. ResNet vs. Highway Network
- **Kulcstanulság:** A Highway Network és ResNet skip connection-jei javítanak a gradiens propagáción mély hálóknál, de sekélyebb hálóknál (3–5 réteg) az egyszerű MLP is versenyképes. Batch normalization helyett Layer normalization stabilabb viselkedést mutat. Az optimális architektúra: 4–5 réteg, 64–256 neuron rétegenkénti, residual kapcsolatokkal. A tanulmány szintetikus és valós S&P 500 adatokon is elvégzi a kísérleteket.
- **Hivatkozás:** https://arxiv.org/abs/2312.xxxxx *(pontos arXiv ID szükséges)*

---

### PINN megközelítések (2021–2023)
- **Képviselő cikk:** Becker, Cheridito & Jentzen (2019), valamint Raissi et al. (2019) általános PINN keretrendszer
- **Cím:** Physics-Informed Neural Networks for Option Pricing
- **Architektúra:** MLP + Black-Scholes PDE veszteségfüggvény-tag (fizikai korlát)
- **Kulcstanulság:** A BS PDE (∂C/∂t + ½σ²S²∂²C/∂S² + rS∂C/∂S − rC = 0) regularizációs tagként adható a veszteségfüggvényhez. Ez kevesebb adatból is pontos becslést tesz lehetővé, és a modell fizikailag konzisztens marad. Hátránya: a PDE-tag számítása lassabb és érzékenyebb hiperparaméterekre.
- **Hivatkozás:** https://arxiv.org/abs/1711.10561 (általános PINN)

---

### FINN – Financially Informed Neural Network (2024)
- **Cím:** FINN: Financially Informed Neural Networks for Option Pricing
- **Architektúra:** Hibrid: neurális háló + pénzügyi korlátok (pl. put-call paritás, monotonitás)
- **Kulcstanulság:** Soft constraint-ek (büntetőtagok) formájában beépíti a no-arbitrage feltételeket. Eredmény: kevesebb adatból pontosabb és arbitrázsmentes árazás. A megközelítés különösen hasznos hiányos adatokkal (pl. illiquid opciók) dolgozva.
- **Hivatkozás:** *(2024-es preprint, részletes hivatkozás szükséges)*

---

## 2. Input/output transzformációk

### Homogeneity hint (Garcia & Gençay, 2000)

A Black-Scholes call ár homogén 1. fokú az S és K paraméterekben:

```
C(λS, λK, T, r, σ) = λ · C(S, K, T, r, σ)
```

Ebből következik, hogy:

```
C/K = f(S/K, T, r, σ)
```

**Miért fontos?**
- Az S és K helyett egyetlen `m = S/K` (moneyness) input elegendő
- A kimeneti C/K dimenziótalanítja az árat a kötési árhöz képest
- Ez csökkenti a bemeneti tér dimenzióját és javítja a generalizációt
- A neurális háló nem kell hogy tanulja az abszolút árszint-skálázást

**Implementáció a projektben:**
- Input: `[S/K, T, r, sigma]` (4 feature)
- Output: `C/K`
- A nyers C ár visszaszámítható: `C = (C/K) * K`

---

## 3. Architektúra összehasonlítás

| Architektúra | Rétegek | Neuronok/réteg | Skip conn. | Normalizáció | Eredmény (Lürig et al.) |
|---|---|---|---|---|---|
| **MLP** | 4–5 | 64–256 | Nincs | Batch/Layer | Baseline, jó sekélyen |
| **ResNet** | 6–10 | 64–256 | Rétegenkénti összeg | Batch Norm | Mély hálóknál jobb gradiens |
| **Highway Network** | 4–8 | 64–256 | Tanult gate | Layer Norm | Legjobb stabilizáció |

**Főbb megfigyelések (Lürig et al. 2023):**
- 3–5 réteges MLP az egyszerű BS feladaton kellő pontosságú
- Residual kapcsolatok mélyebb hálóknál (6+) szükségesek a gradiens eltűnés ellen
- Layer Normalization stabilabb mint Batch Normalization kis batch méreteken
- A moneyness-input (S/K, C/K) minden architektúrán javít

---

## 4. Javasolt architektúra (saját implementáció)

### Alap: Culkin & Das (2017) kibővítve Lürig et al. (2023) alapján

| Paraméter | Érték | Indoklás |
|---|---|---|
| **Architektúra típus** | MLP (esetleg ResNet) | Egyszerű, jól dokumentált baseline |
| **Input dim** | 4 | S/K, T, r, σ |
| **Output dim** | 1 | C/K |
| **Rejtett rétegek** | 4 | Culkin & Das alapján |
| **Neuronok/réteg** | 100–256 | 100 (Culkin), 256 (Lürig) |
| **Aktiváció** | ReLU | Standard, jól működik |
| **Normalizáció** | Layer Norm vagy BatchNorm | Layer Norm stabilabb |
| **Dropout** | 0.1–0.2 | Regularizáció |
| **Optimizer** | Adam | lr=1e-3, weight decay=1e-4 |
| **Veszteség** | MSE | Esetleg Huber loss |
| **Adathalmaz méret** | ~300 000 | Culkin & Das alapján |
| **Train/val/test split** | 70/15/15% | – |

### Hálódiagram (egyszerűsítve)

```
Input (4) → FC(256) → LayerNorm → ReLU → Dropout
          → FC(256) → LayerNorm → ReLU → Dropout
          → FC(256) → LayerNorm → ReLU → Dropout
          → FC(256) → LayerNorm → ReLU → Dropout
          → FC(1)   → Output (C/K)
```

### Kiterjesztés (2. fázis)

Ha a görögök (delta, vega, gamma, theta, rho) is outputok:
- Output dim: 6 (ár + 5 görög)
- Alternatíva: görögök automatikus differenciálással (autograd) számíthatók az árból

---

## 5. Összefoglalás

A projekt 1. fázisához a **Culkin & Das (2017) architektúra** (4×100 MLP) megfelelő kiindulópont, amelyet a **Garcia & Gençay (2000) homogeneity hint** transzformációval (S/K input, C/K output) kombinálva kell alkalmazni. A Lürig et al. (2023) eredményei alapján érdemes Layer Normalization-t hozzáadni a stabilitás érdekében.

A PINN/FINN megközelítések a 2. fázisban (historikus adatok, arbitrázs-mentesség) relevánsak.
