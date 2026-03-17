<objective>
Tekintsd át a projekt összes dokumentációját a `docs/` mappában, és javítsd azokat, ha hibát, pontatlanságot, hiányosságot vagy elavult információt találsz. A cél az, hogy a dokumentáció pontosan tükrözze a jelenlegi implementáció állapotát.
</objective>

<context>
Ez egy egyetemi Önlab projekt, amely Black-Scholes modell alapú neurális hálós opcióárazást valósít meg.
A projekt nyelve **magyar** — minden dokumentáció, commit üzenet és javítás magyarul legyen.

Dokumentációs fájlok a `docs/` mappában:
- `docs/bs_szintetikus_adatok.md` – adathalmaz paraméterek és sampling stratégia
- `docs/adathalmaz_generator.md` – az adatgeneráló pipeline leírása
- `docs/model.md` – neurális háló architektúrák és tanítási eredmények
- `docs/literatura.md` – irodalomjegyzék

Forráskód (a dokumentáció ezeket írja le):
- `src/black_scholes.py` – BS zárt képlet és görögök
- `src/data_generator.py` – adathalmaz generálás
- `src/model.py` – modell architektúrák (MLP, DeepMLP, ResNet, stb.)
- `src/train.py` – tanítási pipeline
- `src/evaluate.py` – kiértékelő pipeline
- `generate_dataset.py` – CLI belépési pont

Olvasd el a `CLAUDE.md`-t a projekt konvenciókért.
</context>

<review_process>
1. Olvasd el az összes forráskód fájlt (`src/`, `generate_dataset.py`, `train.py`, `evaluate.py`) hogy megértsd a jelenlegi implementáció állapotát.
2. Olvasd el az összes dokumentációs fájlt (`docs/`).
3. Minden dokumentációs fájlnál ellenőrizd:
   - **Pontosság**: A leírt paraméterek, értékek, viselkedések egyeznek-e a kóddal?
   - **Teljességt**: Hiányzik-e valami fontos funkció, paraméter, architektúra?
   - **Konzisztencia**: A dokumentumok egymással összhangban vannak-e?
   - **Elírások, nyelvtani hibák**: Gépelési hibák, magyartalanságok.
   - **Elavult információ**: Van-e olyan rész, ami már nem érvényes a jelenlegi kódban?
4. Javítsd a talált hibákat a fájlokban.
</review_process>

<constraints>
- Csak valódi hibákat javíts — ne írj át stílusból, ha a tartalom helyes.
- Ne add hozzá azokat az implementált funkciókat, amelyek nincsenek dokumentálva, ha azok triviálisak — csak ha a hiány félrevezető.
- Minden módosítás legyen magyar nyelvű.
- Ne változtasd meg a dokumentumok struktúráját, hacsak az nem szükséges a javításhoz.
</constraints>

<output>
Módosítsd közvetlenül a hibás `docs/*.md` fájlokat. Ne hozz létre új fájlokat.

A javítások elvégzése után adj egy rövid összefoglalót:
- Melyik fájlokban találtál hibát?
- Milyen jellegű hibák voltak (pontatlanság / hiány / elírás / elavult)?
- Mit javítottál?
</output>

<verification>
Mielőtt befejezed, ellenőrizd:
- Minden `docs/` fájlt átnéztél.
- A javítások a jelenlegi forráskóddal összhangban vannak.
- Nem kerültek be magyar helyett angol mondatok.
</verification>
