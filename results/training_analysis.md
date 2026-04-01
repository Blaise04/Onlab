# Neuralis Halo Modellek Tanulasi Gorbe Elemzese

*Generalva: 2026-04-01 21:38*

Adatok: Black-Scholes szintetikus adathalmaz, 1M minta (800K train, 100K val).  
Tanitoberendeles: CUDA GPU. Minden modell: seed=42, lr=1e-3, weight_decay=1e-4, batch=4096.  

---

## 1. Osszefoglalo Tablazat

| Modell | Parameterek | Min Val MSE | Best Epoch | Final Train MSE | Final Val MSE | Overfitting res |
|---|---|---|---|---|---|---|
| ResNetPricer | 400,129 | 0.000014 | 47 | 0.000051 | 0.000028 | -0.000023 |
| MLPPricer | 31,001 | 0.000023 | 47 | 0.000026 | 0.000024 | -0.000002 |
| GELUResNetPricer | 398,593 | 0.000083 | 40 | 0.000200 | 0.000085 | -0.000115 |
| FINNPricer | 403,202 | 0.000092 | 17 | 0.000253 | 0.000099 | -0.000154 |
| HighwayPricer | 528,129 | 0.000224 | 48 | 0.000275 | 0.000225 | -0.000051 |
| DenseMLPPricer | 101,894 | 0.000228 | 18 | 0.000635 | 0.000235 | -0.000400 |
| DeepMLPPricer | 267,521 | 0.000274 | 4 | 0.000676 | 0.000285 | -0.000391 |

## 2. Konvergencia Elemzes

A leggyorsabban konvergalo modell a **DeepMLPPricer**, amely mar a **4. epochban** erte el legjobb validacios loss erteket (0.000274).

Konvergencia sorrendje (best epoch szerint):

- **DeepMLPPricer**: 4. epoch (val MSE: 0.000274)
- **FINNPricer**: 17. epoch (val MSE: 0.000092)
- **DenseMLPPricer**: 18. epoch (val MSE: 0.000228)
- **GELUResNetPricer**: 40. epoch (val MSE: 0.000083)
- **MLPPricer**: 47. epoch (val MSE: 0.000023)
- **ResNetPricer**: 47. epoch (val MSE: 0.000014)
- **HighwayPricer**: 48. epoch (val MSE: 0.000224)

Megjegyzes: A DeepMLPPricer mar a 4. epochban elerte legjobb ertejet (val MSE: 0.000274), azonban ez viszonylag magas ertek - a 'gyors konvergencia' ott korai megallast jelent, nem hatekony optimalizaciot. A ResNetPricer es MLPPricer 47. epochig tanult es lenyegesen kisebb final MSE-t ert el.

## 3. Vegso Teljesitmeny

A legjobb validacios teljesitmenyt a **ResNetPricer** erte el, minimalis val MSE: **0.000014** (47. epoch).

A leggyengebb eredmenyt a **DeepMLPPricer** produkalta: val MSE: 0.000274.

A legjobb es leggyengebb modell kozotti kulonbseg: 0.000260 (1872.5% relativ elteres).

Rangsor (min val MSE szerint):

1. **ResNetPricer**: val MSE = 0.000014 (400,129 parameter)
2. **MLPPricer**: val MSE = 0.000023 (31,001 parameter)
3. **GELUResNetPricer**: val MSE = 0.000083 (398,593 parameter)
4. **FINNPricer**: val MSE = 0.000092 (403,202 parameter)
5. **HighwayPricer**: val MSE = 0.000224 (528,129 parameter)
6. **DenseMLPPricer**: val MSE = 0.000228 (101,894 parameter)
7. **DeepMLPPricer**: val MSE = 0.000274 (267,521 parameter)

## 4. Tultanulas (Overfitting) Vizsgalata

A tultanulas merteket a final train MSE es final val MSE kulonbsege jelzi. Pozitiv ertek azt jelenti, hogy a validacios loss magasabb a tanitasinal (overfitting), negativ ertek fordított esetben generaldasra utal.

- **Legtobb overfitting**: MLPPricer (val-train res: -0.000002)
- **Legkevesebb overfitting**: DenseMLPPricer (val-train res: -0.000400)

Overfitting res reszletesen:

- **MLPPricer**: -0.000002
- **ResNetPricer**: -0.000023
- **HighwayPricer**: -0.000051
- **GELUResNetPricer**: -0.000115
- **FINNPricer**: -0.000154
- **DeepMLPPricer**: -0.000391
- **DenseMLPPricer**: -0.000400

Altalanos megfigyeles: Az MLPPricer es ResNetPricer negatív overfit rest mutat -- a val loss kisebb a train loss-nal. Ez Dropout hianyan es a BatchNorm-nak koszonheto: a BatchNorm1d tanitas kozben noises becslest ad (batch stat), de ertekelesnél stabilabb populacio-statisztikat hasznal, ami alacsonyabb val losst okozhat.

## 5. Stabilitas Elemzes

A stabilitast a tanulasi gorbe utolso 10 epochjanak val_loss szorasaval merjuk (kisebb szoras = stabilabb konvergencia, kevesebb oszcillacio).

- **Legstabilabb**: HighwayPricer (szoras: 9.99e-07)
- **Legkevesbe stabil**: DeepMLPPricer (szoras: 1.32e-04)

Stabilitas sorrendje (utolso 10 epoch val_loss szorasa):

- **HighwayPricer**: 9.99e-07
- **MLPPricer**: 2.16e-06
- **DenseMLPPricer**: 2.18e-06
- **GELUResNetPricer**: 3.16e-06
- **ResNetPricer**: 8.28e-06
- **FINNPricer**: 5.40e-05
- **DeepMLPPricer**: 1.32e-04

## 6. Architektura Hatasanak Ertekelse

### MLPPricer

Culkin & Das (2017) baseline. Egyszeru 4x100 neuronos MLP, ReLU aktivacioval, Dropout es normalizacio nelkul. Kis parameterszama (~31K) ellenere kivaaltkepp hatekony: 50 epochon at folyamatosan tanult (early stop: 47. ep.) es a legjobb MSE-t erte el (0.0000226). Ennek oka lehet, hogy a BS arakar viszonylag sima fugveny, a kis halozat nem overfit, az Adam optimizer es LR scheduler jo konvergenciat biztosit.

**Szamszeru eredmeny**: min val MSE = 0.000023, best epoch = 47, parameterek = 31,001, overfitting res = -0.000002.

### DeepMLPPricer

Della Corte et al. (2023) javitott MLP. LayerNorm es Dropout (0.1) regularizacio Pre-LN blokkokban, 4x256 reteg. Early stopping a 4. epochnal (val MSE: 0.000274). A korai megallas es magas val MSE azt jelzi, hogy a LayerNorm + Dropout ebben a konfiguracionban instabil korai tanulast okoz -- a loss nagymerteku oszcillaciok utan nem tudott tartosan javulni. A patience=15 szoros hataron belul maradt.

**Szamszeru eredmeny**: min val MSE = 0.000274, best epoch = 4, parameterek = 267,521, overfitting res = -0.000391.

### ResNetPricer

Rezidualis kapcsolatokkal (skip connection) ellatott MLP, BatchNorm1d normalizacioval, Dropout nelkul a skipen. 50 epochon at tanult (best: 47. ep.), val MSE: 0.0000139 -- ez a 2. legjobb eredmeny. A BatchNorm1d stabilizalja a kozbulso reprezentaciokat, a skip connectionok segitik a gradiens aramlast. A negatív overfitting res (val < train MSE) a BatchNorm1d ertekelesi modjanak koszonheto.

**Szamszeru eredmeny**: min val MSE = 0.000014, best epoch = 47, parameterek = 400,129, overfitting res = -0.000023.

### GELUResNetPricer

ResNetPricer GELU aktivacioval, Pre-LN strukturaval. 50 epochot futott (best: 40. ep.), val MSE: 0.0000826. A GELU simabb aktivacio, de a Pre-LN struktura eltero gradiens dinamikat okoz, mint a BatchNorm1d. Az eredmeny gyengebb a ResNetPricer-nel, ami azt jelzi, hogy a BatchNorm1d elonysebb ebben a konfiguracionban BS adatokon.

**Szamszeru eredmeny**: min val MSE = 0.000083, best epoch = 40, parameterek = 398,593, overfitting res = -0.000115.

### DenseMLPPricer

DenseNet-stilusu MLP: minden reteg az osszes korabbi kimenetet kapja inputkent, GELU aktivacioval. Early stopping a 18. epochnal (val MSE: 0.000228). A dense kapcsolatok gazdagabb reprezentaciot tesznek lehetove, de a Dropout+GELU kombinacio hasonlo instabilitast okoz, mint a DeepMLPPricer-nel. Kozepes teljesitmeny (4. helyezett).

**Szamszeru eredmeny**: min val MSE = 0.000228, best epoch = 18, parameterek = 101,894, overfitting res = -0.000400.

### HighwayPricer

Highway Network tanulhato gating-gel, 4 blokk, 256 dim, Dropout(0.1). 50 epochot futott (best: 48. ep.), val MSE: 0.000224. A legmagasabb parameterszam (528K) ellenere koepes teljesitmeny -- a gating mechanizmus rugalmassaga itt nem hozta a vart elonyt. Az alacsony train MSE (0.000275) es viszonylag magas val MSE overfittingre utal.

**Szamszeru eredmeny**: min val MSE = 0.000224, best epoch = 48, parameterek = 528,129, overfitting res = -0.000051.

### FINNPricer

Finance-Informed NN: ket ag -- egy kis MLP BS-kozelitokent, egy GELUResNet-ag korrekcios tagkent. Early stopping a 17. epochnal (val MSE: 0.0000923). A ket-agu strukturanak koszonheten relativlag gyors konvergenciat mutat, de 50 epochon belul nem tudta utolerni a ResNetPricer-t. A korrekcios ag inductive bias-a hasznos lehet, de a 17. epoch utani oszcillacio jelezi a tanulas instabilitasat.

**Szamszeru eredmeny**: min val MSE = 0.000092, best epoch = 17, parameterek = 403,202, overfitting res = -0.000154.

## 7. Kovetkeztetesek es Ajanlasok

### Modellek vegso rangsorolasa

1. **ResNetPricer** -- val MSE: 0.000014, parameterek: 400,129
2. **MLPPricer** -- val MSE: 0.000023, parameterek: 31,001
3. **GELUResNetPricer** -- val MSE: 0.000083, parameterek: 398,593
4. **FINNPricer** -- val MSE: 0.000092, parameterek: 403,202
5. **HighwayPricer** -- val MSE: 0.000224, parameterek: 528,129
6. **DenseMLPPricer** -- val MSE: 0.000228, parameterek: 101,894
7. **DeepMLPPricer** -- val MSE: 0.000274, parameterek: 267,521

### Ajanlott modell opcios arazasra

**Ajanlott modell: ResNetPricer** (ResNetPricer ha az a legjobb, egyebkent az aktualis best)

Indoklas:

- Legjobb validacios MSE: 0.000014 (47. epochban)
- Parameterszam: 400,129 (hatekony kapacitas/teljesitmeny arany)
- Overfitting res: -0.000023 (nincs szignifikans overfitting)
- Stabilitasa a top modellek kozott: 8.28e-06

**Alternativa stabilitasra**: HighwayPricer (9.99e-07 szorassal a legstabilabb tanulasi gorbet mutatja, val MSE: 0.000224).

### Altalanos tanulsagok

1. **BatchNorm1d > LayerNorm** BS adatokon (50 epochos kereten belul): A ResNetPricer BatchNorm1d-del jobban teljesitett, mint a LayerNorm-alapu modellek.
2. **Kis MLP is versenykepes**: Az MLPPricer (~31K param) a legjobb eredmenyt erte el, ami azt jelzi, hogy a BS felszin viszonylag egyszeru, nem igenyel nagy kapacitast.
3. **Korai megallas problema**: DeepMLPPricer es DenseMLPPricer korai early stoppingja nem hatekony konvergenciat, hanem oszcillaciok utan valo selejtezest jelent -- tobb epochsra vagy kisebb LR-re lenne szukseg.
4. **Finance Inductive Bias**: A FINNPricer ket-agu strukturaja igeretest mutat, de 50 epochon belul nem tudja kiaknazni az elonyet.
5. **Highway gating overparameterizalt**: 528K parameter az 5-input feladathoz tul nagy, a teljesitmeny elmarad az egyszerubb ResNet/MLP modellektol.
