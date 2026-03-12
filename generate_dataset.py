"""
CLI belépési pont – Black-Scholes szintetikus adathalmaz generálás.

Használat:
    python generate_dataset.py --n 300000 --method lhs --output data/ --format csv
"""

import argparse
import time

from src.data_generator import generate_dataset, save_dataset


def parse_args():
    p = argparse.ArgumentParser(
        description='Black-Scholes szintetikus adathalmaz generátor'
    )
    p.add_argument('--n',         type=int,   default=100_000,
                   help='Minták száma (default: 100000)')
    p.add_argument('--method',    type=str,   default='lhs',
                   choices=['uniform', 'lhs', 'grid'],
                   help='Mintavételezési módszer (default: lhs)')
    p.add_argument('--output',    type=str,   default='data/',
                   help='Kimeneti mappa (default: data/)')
    p.add_argument('--format',    type=str,   default='csv',
                   choices=['csv', 'parquet'],
                   help='Kimeneti fájlformátum (default: csv)')
    p.add_argument('--greeks',    action='store_true',
                   help='Görögök (delta, gamma, vega, theta, rho) számítása')
    p.add_argument('--normalize', action='store_true',
                   help='Moneyness normalizáció (S/K és call/K hozzáadása)')
    p.add_argument('--noise',     type=float, default=0.0,
                   help='Gauss-zaj szórása az opció árhoz (default: 0.0)')
    p.add_argument('--seed',      type=int,   default=42,
                   help='Véletlenszám mag (default: 42)')
    return p.parse_args()


def main():
    args = parse_args()

    print("Black-Scholes adathalmaz generálás")
    print(f"  Minták:        {args.n:,}")
    print(f"  Módszer:       {args.method}")
    print(f"  Görögök:       {args.greeks}")
    print(f"  Normalizáció:  {args.normalize}")
    print(f"  Zaj (std):     {args.noise}")
    print(f"  Seed:          {args.seed}")
    print()

    t0 = time.time()
    df = generate_dataset(
        n=args.n,
        method=args.method,
        include_greeks=args.greeks,
        normalize=args.normalize,
        noise_std=args.noise,
        seed=args.seed,
    )
    elapsed = time.time() - t0

    print(f"Generálás kész: {len(df):,} sor, {elapsed:.2f}s")
    print(f"Oszlopok: {list(df.columns)}")
    print()

    print(f"Mentés: {args.output} ({args.format})")
    save_dataset(df, output_path=args.output, format=args.format, seed=args.seed)

    print()
    print("Sanity check (ATM: S=K=100, T=1, r=0.05, sigma=0.2):")
    from src.black_scholes import bs_call as _call
    atm_price = float(_call(S=100, K=100, T=1, r=0.05, sigma=0.2))
    print(f"  Call ár = {atm_price:.4f}  (várható: ~10.45)")

    print()
    print("Első 3 sor:")
    print(df.head(3).to_string(index=False))


if __name__ == '__main__':
    main()
