"""Parquet fájl előnézet – terminálban futtatható."""
import sys
import pandas as pd

def preview(path: str, rows: int = 20):
    df = pd.read_parquet(path)
    print(f"Fájl: {path}")
    print(f"Méret: {len(df):,} sor × {len(df.columns)} oszlop")
    print(f"Oszlopok: {list(df.columns)}")
    print(f"\nElső {rows} sor:")
    print(df.head(rows).to_string(index=True))
    print(f"\nLeíró statisztika:")
    print(df.describe().to_string())

if __name__ == "__main__":
    path = sys.argv[1] if len(sys.argv) > 1 else "data/train.parquet"
    rows = int(sys.argv[2]) if len(sys.argv) > 2 else 20
    preview(path, rows)
