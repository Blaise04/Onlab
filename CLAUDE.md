# CLAUDE.md

## Nyelv
Minden kommunikáció, commit üzenet és dokumentáció **magyarul** legyen.

## Projekt
Opciós árazási neurális háló (Black-Scholes szintetikus adatokon).
- Bemenet: S/K (moneyness), T, r, σ (4 feature, q=0 feltételezés)
- Kimenet: call (és put-call paritásból put) normalizált ár
- Adathalmaz: `data/` — 1M minta, `generate_dataset.py` állítja elő

## Struktúra
- `src/` — core logika (data_generator, model, train, evaluate, black_scholes)
- `train.py` / `evaluate.py` — CLI belépőpontok
- `tests/` — unit tesztek
- `TODO.md` — nyitott fejlesztési javaslatok listája (25 tétel)

## Futtatás
```bash
python generate_dataset.py   # adathalmaz generálás
python train.py              # modell tanítás
python evaluate.py           # kiértékelés
python -m pytest tests/      # tesztek
```
