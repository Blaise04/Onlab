# Adathalmaz dokumentáció

## Generálás

- **Dátum**: 2026-03-21
- **Parancs**: `python generate_dataset.py --n 1000000 --method lhs --normalize --format parquet --seed 42`
- **Futási idő**: 0.78 másodperc
- **Seed**: 42

## Mintavételezési módszer

**Latin Hypercube Sampling (LHS)**

Az LHS az egységkockát egyenlő valószínűségű cellákra osztja dimenzióként, majd minden cellából pontosan egy mintát vesz. Ez garantálja, hogy a paramétertér egyenletesen lefedett legyen, ellentétben az egyenletes véletlenszerű mintavételezéssel (ahol csomósodás is előfordulhat). 1 millió mintánál az LHS számítási költsége még elfogadható.

## Paramétertartományok

| Paraméter | Min | Max | Leírás |
|-----------|-----|-----|--------|
| S (részvényár) | 10.0 | 150.0 | Mögöttes eszköz ára |
| moneyness (S/K) | 0.5 | 1.5 | Pénzességi arány |
| T (lejáratig hátralévő idő) | 0.005 | 2.0 | Évben kifejezve |
| r (kockázatmentes kamatláb) | 0.00 | 0.05 | Éves |
| sigma (volatilitás) | 0.05 | 0.90 | Éves implicit volatilitás |
| q (osztalékhozam) | 0.00 | 0.03 | Folytonos osztalékhozam |

## Oszlopok

### Input feature-ök (nyers)
- `moneyness` – S/K arány
- `S` – részvényár
- `K` – kötési ár (K = S / moneyness)
- `T` – lejáratig hátralévő idő
- `r` – kockázatmentes kamatláb
- `sigma` – volatilitás
- `q` – osztalékhozam

### Célváltozók
- `call_price` – Black-Scholes call ár
- `put_price` – Black-Scholes put ár
- `call_price_norm` – Normált call ár (call_price / K)

## Fájlok

| Fájl | Sorok | Méret |
|------|-------|-------|
| `data/train.parquet` | 700 000 | 57 MB |
| `data/val.parquet` | 150 000 | 15 MB |
| `data/test.parquet` | 150 000 | 15 MB |

**Felosztás**: 70% train / 15% val / 15% test (véletlenszerű shuffle, seed=42)

## Sanity check

ATM opció (S=K=100, T=1, r=0.05, sigma=0.2): call ár ≈ 10.45
Generált eredmény: 10.4506

## Döntések indoklása

- **LHS vs uniform**: Az LHS szisztematikusan lefedi a paraméterteret, elkerüli a klaszteresedést.
- **LHS vs grid**: A 6 dimenziós rács 1M pontnál ~4 pont/dimenzió lenne, ami nagyon durva. Az LHS folytonos és jobb.
- **Parquet vs CSV**: 1M sornál a Parquet ~5-10x kisebb fájlméretet és gyorsabb I/O-t biztosít.
- **Normalizáció**: A `call_price_norm` (call/K) dimenziómentes, más K értékekre is általánosítható modellt tesz lehetővé.
