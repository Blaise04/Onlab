# Tanulási görbék elemzése

Az elemzés alapja: `results/plots/` — 8 egyedi ábra + 1 összesítő.

---

## 1. MLPPricer — a legstabilabb tanuló

A train_loss és val_loss görbék szinte teljesen egymáson fekszenek végig, overfitting nincs.
Az LR egyszer sem csökkent (a scheduler nem lépett be), mégis 68 epochon át lassan, egyenletesen
javult. Ez arra utal, hogy ez az architektúra természetesen simán illeszkedik a BS tájképre.

---

## 2. ResNetPricer — az LR scheduler hatása jól látható

A zöld lr vonal lépcsőzetes mintát mutat: 2-3 jól látható csökkentés ~25. és ~35. epoch körül.
Minden LR-esés után a val_loss markánsan ugrik lefelé. Ez azt jelenti, hogy a ResNet **szüksége
van** a schedulerre a konvergenciához — az MLP-vel ellentétben nem talál jó irányt önmagától.

---

## 3. GELUResNetPricer — látható instabilitás ~20-23. epoch körül

A val_loss görbén nagy tüske látható (felugrik ~10⁻³-ra, majd visszaesik). Ez a típusos
GELU-s gradiens-instabilitás: a sima aktiváció bizonyos paramétertérben hajlamos hirtelen nagy
gradiensekre. Az ReLU ResNet-nél ilyen tüske nincs.

---

## 4. DeepMLPPricer — leggyorsabb overfitting

Csak 4 epoch, és az összes modell közül itt a legnagyobb a train_loss / val_loss közötti rés.
A val_loss már 2. epoch után platóra állt, míg a train_loss tovább csökkent. A Dropout(0.1) +
LayerNorm sem védte meg a modellt — valószínűleg a skip-kapcsolat hiánya és a 256-os szélesség
együtt okoz gyors memorálást.

---

## 5. DenseMLPPricer, HighwayPricer, FINNPricer — korai leállás, scheduler soha nem lépett be

Mindhárom 17-22 epoch között megállt. Az LR görbéjük lapos — a `ReduceLROnPlateau` (patience=5)
nem tudott elegendő epochot várni ahhoz, hogy csökkentse az LR-t és kinyissa a modellt a mélyebb
minimumok felé. Ha a patience nagyobb lett volna (pl. 20), valószínűleg jobban konvergáltak volna.

---

## 6. ResNetPricer (physics) — a fizikai loss zajt visz be

A val_loss görbéje sokkal oszcillálóbb mint a sima ResNet-é, különösen 10-30. epoch között
láthatók tüskék. A `L_delta` gradiens folyamatosan "ütközik" az MSE gradiensével, ami
instabilitást okoz — de végül hasonló szintre konvergál.

---

## 7. Összesítő ábra tanulságai

Az MLP (kék) a végén **egyértelműen a legtöbb modell alá kerül**. A ResNet (zöld) szorosan
követi. A többi modell a ~10⁻⁴ tartományban rekedt — egy teljes nagyságrenddel rosszabb
mint az MLP végső értéke (~2×10⁻⁵).

---

## Legfontosabb tanulság

A 2. generációs modellek nem azért rosszabbak, mert gyengébb architektúrák, hanem mert
**az early stopping + patience=10 kombináció túl korán megvágta őket**, mielőtt a scheduler
beindulhatott volna. Az MLP és ResNet hosszabb futásából profitált.
