# CLAUDE.md

Ez a fájl útmutatást nyújt a Claude Code (claude.ai/code) számára, amikor ezzel a repository-val dolgozik.

## Projektről

Ez egy egyetemi önálló laboratóriumi projekt (Önlab) a 6. félévre. A projekt PyCharm-mal van konfigurálva (`.idea/`), tehát Python-alapú.

### Könyvtárstruktúra

- `docs/` – kutatási dokumentumok és tervek
  - `bs_szintetikus_adatok.md` – Black-Scholes szintetikus adathalmaz paraméterei neurális hálóhoz
- `src/` – Python forráskód
  - `black_scholes.py` – BS zárt képlet és görögök (vektorizált numpy)
  - `data_generator.py` – adathalmaz generálás, mintavételezés, mentés
- `generate_dataset.py` – CLI belépési pont
- `data/` – generált adathalmazok (train/val/test CSV vagy parquet)

### Adathalmaz generálás

```bash
python generate_dataset.py --n 300000 --method lhs --output data/ --format csv
python generate_dataset.py --n 100000 --method uniform --greeks --normalize
```

Argumentumok: `--n`, `--method` (uniform/lhs/grid), `--output`, `--format` (csv/parquet), `--greeks`, `--normalize`, `--noise`, `--seed`

### Függőségek

- `numpy`, `scipy`, `pandas` – kötelező
- `pyarrow` – parquet mentéshez (opcionális)

## Git konvenciók

A változtatásokat mindig tiszta és olvasható commit üzenetekkel kell commitolni.