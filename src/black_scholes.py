"""
Black-Scholes árképletek és görögök – vektorizált numpy implementáció.

Feltételezés: q=0 (nincs osztalékhozam) — az eredeti 4-paraméteres BS képlet.
Bemenet: S, K, T, r, sigma
"""

import numpy as np
from scipy.stats import norm


def _d1_d2(S, K, T, r, sigma):
    """d1 és d2 segédváltozók kiszámítása (q=0 esetén)."""
    with np.errstate(divide='ignore', invalid='ignore'):
        d1 = (np.log(S / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    return d1, d2


def bs_call(S, K, T, r, sigma):
    """Black-Scholes call opció ára (q=0).

    Paraméterek
    -----------
    S     : mögöttes eszköz árfolyama
    K     : kötési ár
    T     : lejáratig hátralévő idő (évben)
    r     : kockázatmentes kamatláb
    sigma : volatilitás

    Visszaadja
    ----------
    call ár (skalár vagy numpy tömb)
    """
    S, K, T, r, sigma = (np.asarray(x, dtype=float) for x in (S, K, T, r, sigma))
    intrinsic = np.maximum(S - K * np.exp(-r * T), 0.0)
    # T ≈ 0 esetén intrinsic value
    near_expiry = T < 1e-8
    d1, d2 = _d1_d2(S, K, T, r, sigma)
    price = S * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2)
    return np.where(near_expiry, intrinsic, price)


def bs_put(S, K, T, r, sigma):
    """Black-Scholes put opció ára (put-call paritásból, q=0)."""
    S, K, T, r, sigma = (np.asarray(x, dtype=float) for x in (S, K, T, r, sigma))
    call = bs_call(S, K, T, r, sigma)
    return call - S + K * np.exp(-r * T)


def bs_delta(S, K, T, r, sigma):
    """Call delta (∂C/∂S), q=0."""
    S, K, T, r, sigma = (np.asarray(x, dtype=float) for x in (S, K, T, r, sigma))
    near_expiry = T < 1e-8
    d1, _ = _d1_d2(S, K, T, r, sigma)
    delta = norm.cdf(d1)
    intrinsic_delta = np.where(S > K, 1.0, 0.0)
    return np.where(near_expiry, intrinsic_delta, delta)


def bs_gamma(S, K, T, r, sigma):
    """Gamma (∂²C/∂S²), q=0."""
    S, K, T, r, sigma = (np.asarray(x, dtype=float) for x in (S, K, T, r, sigma))
    near_expiry = T < 1e-8
    d1, _ = _d1_d2(S, K, T, r, sigma)
    gamma = norm.pdf(d1) / (S * sigma * np.sqrt(T))
    return np.where(near_expiry, 0.0, gamma)


def bs_vega(S, K, T, r, sigma):
    """Vega (∂C/∂σ), 1%-pontos változásra normálva, q=0."""
    S, K, T, r, sigma = (np.asarray(x, dtype=float) for x in (S, K, T, r, sigma))
    near_expiry = T < 1e-8
    d1, _ = _d1_d2(S, K, T, r, sigma)
    vega = S * norm.pdf(d1) * np.sqrt(T) * 0.01
    return np.where(near_expiry, 0.0, vega)


def bs_theta(S, K, T, r, sigma):
    """Call theta (∂C/∂T), naptári napra normálva (/365), q=0."""
    S, K, T, r, sigma = (np.asarray(x, dtype=float) for x in (S, K, T, r, sigma))
    near_expiry = T < 1e-8
    d1, d2 = _d1_d2(S, K, T, r, sigma)
    term1 = -S * norm.pdf(d1) * sigma / (2 * np.sqrt(T))
    term2 = -r * K * np.exp(-r * T) * norm.cdf(d2)
    theta = (term1 + term2) / 365.0
    return np.where(near_expiry, 0.0, theta)


def bs_rho(S, K, T, r, sigma):
    """Call rho (∂C/∂r), 1%-pontos változásra normálva, q=0."""
    S, K, T, r, sigma = (np.asarray(x, dtype=float) for x in (S, K, T, r, sigma))
    near_expiry = T < 1e-8
    _, d2 = _d1_d2(S, K, T, r, sigma)
    rho = K * T * np.exp(-r * T) * norm.cdf(d2) * 0.01
    return np.where(near_expiry, 0.0, rho)
