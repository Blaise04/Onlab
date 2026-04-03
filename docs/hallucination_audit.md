# AI Hallucináció Audit Riport

**Dátum:** 2026-04-03  
**Auditor:** Claude Sonnet 4.6 (claude-sonnet-4-6)  
**Projekt:** Opciós árazási neurális háló — Black-Scholes szintetikus adatokon

---

## Összefoglaló

Az audit kiterjedt az összes dokumentációs fájlra (`docs/`), a teljes `src/` forráskódra, a gyökér szintű `.md` fájlokra (`CLAUDE.md`, `DATA.md`, `TODO.md`), a `results/training_analysis.md` fájlra, a `tests/` könyvtárra, és az utolsó 30 git commit üzenetére.

**Vizsgált területek száma:** 13 fájl + git log  
**Talált problémák összesen:** 8 (ebből 2 biztos hiba, 3 valószínű hiba, 3 gyanús/minor eltérés)  
**Általános értékelés:** A projekt kódbázisa (matematika, API-k, PyTorch-hívások) megbízható és jól tesztelt. A legtöbb hallucináció az irodalmi hivatkozásokban és a dokumentáció-kód konzisztenciájában mutatkozik, nem a tényleges számításokban. Egy korábbi súlyos hallucinációt (Lürig et al. 2023) már javítottak.

---

## Talált problémák

---

### [BIZTOS HIBA] Liu et al. (2019) hibás cím az összes hivatkozásban

- **Helyszín:** `docs/model.md:169`, `docs/model.md:596`, `docs/literatura.md:35-41`
- **Eredeti szöveg:**
  > `Liu et al. (2019) — *A neural network-based framework for financial model calibration*`
- **Probléma:** A Liu, Oosterlee & Bohte (2019) cikk valódi címe: *"Pricing options and computing implied volatilities using neural networks"* (Risks, Vol. 7, No. 1, arXiv:1901.08943). Ez a cím szerepel helyesen a `docs/literatura.md` elején (35–41. sor), de a `docs/model.md`-ben és a `docs/literatura.md` 596. sorában (9. irodalmi háttér szakasz) egy **másik, tévesen rendelt cím** jelenik meg. A "calibration framework" cím nem tartozik ehhez a cikkhez.
- **Javasolt javítás:** Minden előfordulásban: `Liu et al. (2019) — *Pricing options and computing implied volatilities using neural networks*`
- **Érintett sorok:**
  - `docs/model.md:169`: `Liu et al. (2019) — *A neural network-based framework for financial model calibration*`
  - `docs/model.md:596`: `Liu et al. (2019) — *A neural network-based framework for financial model calibration*`

---

### [BIZTOS HIBA] `docs/literatura.md` — Input dim eltérés a javasolt architektúrában

- **Helyszín:** `docs/literatura.md:135`, `docs/literatura.md:150`
- **Eredeti szöveg:**
  ```
  | **Input dim** | 4 | S/K, T, r, σ |
  ...
  Input (4) → FC(256) → LayerNorm → ReLU → Dropout
  ```
- **Probléma:** A projekt 5 bemeneti feature-t használ: `S/K, T, r, σ, q` (az osztalékhozam `q` is bemenet). Ez következetesen dokumentált az összes többi helyen (`docs/model.md`, `DATA.md`, `CLAUDE.md`, a tényleges kód `src/model.py`-ban és `src/train.py`-ban). A `literatura.md` javasolt architektúra táblázatában és a hálódiagramban `input_dim=4` szerepel, ami helytelen — a `q` (osztalékhozam) kihagyása a dokumentációból téveszti meg az olvasót.
- **Javasolt javítás:** `| **Input dim** | 5 | S/K, T, r, σ, q |` és a diagramban `Input (5)`.

---

### [VALÓSZÍNŰ HIBA] FINN cikk azonosítása a `docs/literatura.md`-ben

- **Helyszín:** `docs/literatura.md:74-78`
- **Eredeti szöveg:**
  ```
  ### FINN – Financially Informed Neural Network (2024)
  - **Cím:** FINN: Financially Informed Neural Networks for Option Pricing
  - **Hivatkozás:** *(2024-es preprint, részletes hivatkozás szükséges)*
  ```
- **Probléma:** A "FINN: Financially Informed Neural Networks for Option Pricing" cím nem egy azonosított, létező cikk hivatkozása — a dokumentáció maga is jelöli, hogy `részletes hivatkozás szükséges`. Miközben az `arXiv:2412.12213` ("AI Black-Scholes") cikk más helyen hivatkozva van, a FINN szakasz egy valószínűleg nem létező vagy félreértett cikket nevez meg önálló hivatkozásként. A cím alapján a projekt saját `FINNPricer` modelljének neve lett visszavetítve egy irodalmi forrásba.
- **Javasolt javítás:** A FINN szakaszt egyértelműen az arXiv:2412.12213 cikkre kell visszavezetni (amelynek valódi címe: *"The AI Black-Scholes: Finance-Informed Neural Network for Option Pricing"* — Liao et al. 2024), vagy törlendő mint önálló hivatkozás.

---

### [VALÓSZÍNŰ HIBA] `docs/literatura.md` PINN hivatkozás — Becker, Cheridito & Jentzen (2019) és arXiv:1711.10561 összekapcsolása

- **Helyszín:** `docs/literatura.md:66-70`
- **Eredeti szöveg:**
  ```
  - **Képviselő cikk:** Becker, Cheridito & Jentzen (2019), valamint Raissi et al. (2019) általános PINN keretrendszer
  - **Hivatkozás:** https://arxiv.org/abs/1711.10561 (általános PINN)
  ```
- **Probléma:** Az arXiv:1711.10561 a Raissi, Perdikaris & Karniadakis (2017) "Physics-informed neural networks: A deep learning framework for solving forward and inverse problems involving nonlinear partial differential equations" cikk, amelynek szerzői **nem** Becker, Cheridito & Jentzen — ők egy teljesen különböző kutatócsoportot alkotnak. A két hivatkozás össze van keverve. Becker, Cheridito & Jentzen (2019) cikke más (pl. "Deep optimal stopping").
- **Javasolt javítás:** Szétválasztani a két hivatkozást: Raissi et al. (2017/2019) az arXiv:1711.10561-hez, Becker et al. hivatkozása külön, pontos adatokkal vagy eltávolítva.

---

### [VALÓSZÍNŰ HIBA] `results/training_analysis.md` — eltérő kísérleti beállítások a `docs/model.md`-hez képest

- **Helyszín:** `results/training_analysis.md:6`, vs. `docs/model.md:283-284`
- **Eredeti szöveg (training_analysis.md):**
  > `Adatok: Black-Scholes szintetikus adathalmaz, 1M minta (800K train, 100K val).`
- **Eredeti szöveg (docs/model.md):**
  > `Az összes modellt azonos feltételek mellett tanítottuk: 700 000 szintetikus Black-Scholes minta [...], 150 000-es validációs és teszt halmaz`
- **Probléma:** A két dokumentum eltérő adatfelosztást ír le ugyanazokról a kísérletekről. A `training_analysis.md` 800K/100K felosztást állít, a `docs/model.md` 700K/150K/150K felosztást. A `DATA.md` és a tényleges kód (`src/data_generator.py`) a 70%/15%/15% = 700K/150K/150K felosztást erősíti meg. A `training_analysis.md` adat valószínűleg hibásan lett előállítva vagy korábbi verziót tükröz.
- **Javasolt javítás:** A `results/training_analysis.md` fejléc-sorát javítani: `1M minta (700K train, 150K val, 150K test)`.

---

### [GYANÚS] `docs/literatura.md` — Culkin & Das részvényár-tartomány eltérés

- **Helyszín:** `docs/bs_szintetikus_adatok.md:146`
- **Eredeti szöveg:**
  > `| Culkin & Das (2017) | 300 000 | $10–$50 | $7–$650 | ...`
- **Probléma:** A `docs/literatura.md:29` és más helyek $10–$150-es S tartományt írnak Culkin & Das-hoz (`bs_szintetikus_adatok.md` kísérlet konfiguráció fejezetéből ered), de az összehasonlító táblázatban `$10–$50` szerepel az eredeti Culkin & Das adataiként. Ez utóbbi valószínűbb az eredeti papírból (korlátozott S tartomány), de belső inkonzisztenciát alkot. Nehéz ellenőrizni az eredeti paper hozzáférés nélkül.
- **Értékelés:** GYANÚS — belső inkonzisztencia, az eredeti paper verifikálása szükséges.

---

### [GYANÚS] `docs/model.md` — `training_curves_analysis.md` hivatkozás nem létező fájlra mutat

- **Helyszín:** git log — `c71c4c4` commit ("Docs: tanulási görbék vizuális elemzése (training_curves_analysis.md)")
- **Probléma:** A git státusz szerint `docs/training_curves_analysis.md` **törölve** lett (a git status `D docs/training_curves_analysis.md`-t mutat). A `docs/model.md:287` hivatkozza: `results/training_curves.json` és `results/training_curves.csv`. A `training_curves_analysis.md` fájlt létrehozták, majd eltávolították, de a commit üzenet és egyes belső hivatkozások nem reflektálják ezt.
- **Értékelés:** GYANÚS — nem hallucináció, hanem dokumentum-konzisztencia probléma; a törölt fájlra vonatkozó commit üzenet félrevezető.

---

### [GYANÚS] `docs/literatura.md` — Culkin & Das "Nem normális" eloszlás megjelölése

- **Helyszín:** `docs/bs_szintetikus_adatok.md:146`
- **Eredeti szöveg:**
  > `| Culkin & Das (2017) | 300 000 | ... | Nem normális | Nincs |`
- **Probléma:** A "Nem normális" eloszlás megjelölés az összehasonlító táblázatban értelmezhetetlen — az összes többi sor "Uniform", "Grid" vagy "Log-normális + Uniform" értéket tartalmaz. Valószínűleg "Uniform" kellett volna (ahogy a szövegben máshol is szerepel Culkin & Das esetére), vagy a forrás nem volt egyértelmű.
- **Értékelés:** GYANÚS — félrevezető megjelölés, valószínűleg elírás.

---

## Rendben lévő területek

### Matematika és Black-Scholes képletek

- **`src/black_scholes.py`**: Az összes képlet (d1, d2, call, put, delta, gamma, vega, theta, rho) matematikailag helyes. A d1 képletben `(r - q + σ²/2)` helyesen szerepel. A put-call paritás implementációja (`C - S·e^(-qT) + K·e^(-rT)`) helyes. A görögök definíciói pontosak.
- **`tests/test_black_scholes.py`**: 35 unit teszt, amelyek ellenőrzik az ismert értékeket (ATM: 10.4506), a peremfeltételeket, a put-call paritást és a linearitást — mind helyes.
- **ATM értékellenőrzés**: Az `S=K=100, T=1, r=0.05, σ=0.2, q=0` esetére hivatkozott `C ≈ 10.4506` helyesen ellenőrizhető és a kód ezt valóban visszaadja.

### PyTorch API-k

- **`src/model.py`**: Minden PyTorch osztály és metódus (`nn.Linear`, `nn.BatchNorm1d`, `nn.LayerNorm`, `nn.GELU`, `nn.Sequential`, `nn.ModuleList`) létező és helyesen használt API. A `count_parameters` segédfüggvény helyes implementáció.
- **`src/train.py`**: `torch.optim.Adam`, `torch.optim.lr_scheduler.ReduceLROnPlateau`, `torch.autograd.grad` — mind létező és helyesen használt.
- **`src/evaluate.py`**: `torch.load`, `model.eval()`, `torch.no_grad()` — helyes PyTorch mintázatok.

### NumPy / SciPy API-k

- **`src/data_generator.py`**: `scipy.stats.qmc.LatinHypercube`, `qmc.scale`, `np.random.default_rng` — mind létező API-k, helyesen használva.

### Modell architektúrák kód-dokumentáció konzisztenciája

- **MLPPricer**: A dokumentáció (4 réteg × 100 neuron) és a kód pontosan megegyezik.
- **ResNetPricer**: A dokumentáció (BatchNorm1d, skip connection) és a kód megegyezik (a korábbi Lürig-hallucináció javítása után).
- **HighwayPricer**: A Highway Network gate-bias inicializálás `-1.0`-ra dokumentálva és implementálva — helyes, megfelel Srivastava et al. (2015) javaslatának.
- **DenseMLPPricer**: A dense kapcsolatok implementációja megegyezik a dokumentáció diagramjával.

### Ismert, korábban javított hallucináció

- **Lürig et al. (2023)**: Korábban hamis hivatkozásként azonosított és javított commit `0ace1fd`-ben. A jelenlegi `docs/literatura.md` helyesen a Della Corte et al. (2023) cikket tartalmazza — ez a javítás helyes és következetes az egész projektben.

### Teljesítményállítások

- A `docs/model.md` 7. fejezetének számszerű eredményei (RMSE, R², MSE értékek) konzisztensek a `results/model_comparison.csv` és `results/training_analysis.md` adatokkal (a különböző felosztástól eltekintve). A számok egymással koherensek, nem tartalmaz légből kapott teljesítmény-állítást.

### Adathalmaz dokumentáció

- **`DATA.md`**: Az 1M minta, LHS módszer, seed=42, parquet formátum és 70/15/15 felosztás konzisztens a kóddal.
- **Paramétertartományok**: Az összes tartomány (`S`, `moneyness`, `T`, `r`, `sigma`, `q`) azonos a dokumentációban és a kódban (`DEFAULT_PARAMS` szótár).

### Git commit üzenetek

- Az utolsó 30 commit üzenet tartalmilag pontos, tükrözi a tényleges módosításokat. Nincs hamis állítás a commitokban.

---

## Javaslatok

1. **Liu et al. (2019) cím egységesítése**: A `docs/model.md` 9. irodalmi háttér szakaszában és a FINNPricer dokumentációjában javítani kell a cím-hibát.

2. **FINN irodalmi hivatkozás tisztázása**: A `docs/literatura.md` FINN szakaszát vagy az arXiv:2412.12213 cikkre kell visszavezetni (Liao et al. 2024), vagy egyértelműen jelölni, hogy ez a projektben saját fejlesztés, nem önálló irodalmi forrás.

3. **PINN hivatkozások szétválasztása**: A Becker et al. és Raissi et al. szerzőségét külön kezelni a `docs/literatura.md`-ben.

4. **training_analysis.md adatfelosztás javítása**: A 800K/100K helyett 700K/150K/150K felosztást feltüntetni.

5. **input_dim javítása a literatura.md-ben**: A javasolt architektúra táblázatban és diagramban 4 helyett 5 bemeneti feature-t feltüntetni.

6. **Hallucináció-megelőzési folyamat**: Minden új irodalmi hivatkozás hozzáadásakor ellenőrizni a DOI/arXiv linket a cím és szerzők egyezéséért. Az ilyen auditok rendszeres futtatása (különösen új dokumentáció előtt) megelőzheti a Lürig-típusú eseményeket.

---

*Riport vége — 2026-04-03*
