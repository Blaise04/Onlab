"""
Black-Scholes szintetikus adathalmaz generátor.
"""

import itertools
import os

import numpy as np
import pandas as pd
from scipy.stats import qmc

from src.black_scholes import (
    bs_call, bs_put, bs_delta, bs_gamma, bs_vega, bs_theta, bs_rho,
)

DEFAULT_PARAMS = {
    'S':     (10.0,  150.0),
    'K':     (7.0,   650.0),
    'T':     (0.005,   2.0),
    'r':     (0.00,   0.05),
    'sigma': (0.05,   0.90),
    'q':     (0.00,   0.03),
}

PARAM_NAMES = ['S', 'K', 'T', 'r', 'sigma', 'q']


def _sample_uniform(n: int, rng: np.random.Generator) -> np.ndarray:
    """Egyenletes véletlen mintavételezés."""
    bounds = np.array([DEFAULT_PARAMS[p] for p in PARAM_NAMES])
    low, high = bounds[:, 0], bounds[:, 1]
    return rng.uniform(low, high, size=(n, len(PARAM_NAMES)))


def _sample_lhs(n: int, seed: int) -> np.ndarray:
    """Latin Hypercube mintavételezés (jobb lefedettség)."""
    bounds = np.array([DEFAULT_PARAMS[p] for p in PARAM_NAMES])
    low, high = bounds[:, 0], bounds[:, 1]
    sampler = qmc.LatinHypercube(d=len(PARAM_NAMES), seed=seed)
    unit_sample = sampler.random(n)
    return qmc.scale(unit_sample, low, high)


def _sample_grid(n: int) -> np.ndarray:
    """Rács mintavételezés (Tidy Finance stílus).

    Az n-ből visszaszámolja a dimenziónkénti lépések számát úgy,
    hogy az összes kombináció közelítőleg n legyen.
    """
    steps_per_dim = max(2, int(round(n ** (1 / len(PARAM_NAMES)))))
    axes = [
        np.linspace(DEFAULT_PARAMS[p][0], DEFAULT_PARAMS[p][1], steps_per_dim)
        for p in PARAM_NAMES
    ]
    grid = np.array(list(itertools.product(*axes)))
    return grid


def generate_dataset(
    n: int = 100_000,
    method: str = 'lhs',
    include_greeks: bool = False,
    normalize: bool = False,
    noise_std: float = 0.0,
    seed: int = 42,
) -> pd.DataFrame:
    """Szintetikus Black-Scholes adathalmaz generálása.

    Paraméterek
    -----------
    n             : minták száma (grid esetén közelítő)
    method        : 'uniform', 'lhs', vagy 'grid'
    include_greeks: delta, gamma, vega, theta, rho hozzáadása
    normalize     : moneyness (S/K) és normált call ár (call/K) hozzáadása
    noise_std     : Gauss-zaj szórása a call/put árhoz (0 = nincs)
    seed          : véletlenszám mag

    Visszaadja
    ----------
    pd.DataFrame az összes feature-rel és célváltozóval
    """
    rng = np.random.default_rng(seed)

    if method == 'uniform':
        samples = _sample_uniform(n, rng)
    elif method == 'lhs':
        samples = _sample_lhs(n, seed)
    elif method == 'grid':
        samples = _sample_grid(n)
    else:
        raise ValueError(f"Ismeretlen mintavételezési módszer: '{method}'. Válasszon: uniform, lhs, grid")

    S, K, T, r, sigma, q = (samples[:, i] for i in range(len(PARAM_NAMES)))

    call_price = bs_call(S, K, T, r, sigma, q)
    put_price  = bs_put(S, K, T, r, sigma, q)

    if noise_std > 0.0:
        call_price += rng.normal(0.0, noise_std, size=call_price.shape)
        put_price  += rng.normal(0.0, noise_std, size=put_price.shape)
        call_price = np.maximum(call_price, 0.0)
        put_price  = np.maximum(put_price,  0.0)

    data = {
        'S': S, 'K': K, 'T': T, 'r': r, 'sigma': sigma, 'q': q,
        'call_price': call_price,
        'put_price':  put_price,
    }

    if include_greeks:
        data['delta'] = bs_delta(S, K, T, r, sigma, q)
        data['gamma'] = bs_gamma(S, K, T, r, sigma, q)
        data['vega']  = bs_vega(S, K, T, r, sigma, q)
        data['theta'] = bs_theta(S, K, T, r, sigma, q)
        data['rho']   = bs_rho(S, K, T, r, sigma, q)

    if normalize:
        data['moneyness']       = S / K
        data['call_price_norm'] = call_price / K

    df = pd.DataFrame(data)
    return df


def save_dataset(
    df: pd.DataFrame,
    output_path: str,
    format: str = 'csv',
) -> None:
    """Adathalmaz mentése train/val/test szétválasztással (70/15/15%).

    Paraméterek
    -----------
    df          : a teljes adathalmaz DataFrame
    output_path : kimeneti mappa
    format      : 'csv' vagy 'parquet'
    """
    os.makedirs(output_path, exist_ok=True)

    df = df.sample(frac=1, random_state=0).reset_index(drop=True)
    n = len(df)
    n_train = int(0.70 * n)
    n_val   = int(0.15 * n)

    splits = {
        'train': df.iloc[:n_train],
        'val':   df.iloc[n_train:n_train + n_val],
        'test':  df.iloc[n_train + n_val:],
    }

    for split_name, split_df in splits.items():
        fname = f"{split_name}.{format}"
        fpath = os.path.join(output_path, fname)
        if format == 'csv':
            split_df.to_csv(fpath, index=False)
        elif format == 'parquet':
            split_df.to_parquet(fpath, index=False)
        else:
            raise ValueError(f"Ismeretlen formátum: '{format}'. Válasszon: csv, parquet")
        print(f"  Mentve: {fpath}  ({len(split_df):,} sor)")
