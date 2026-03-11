"""
Black-Scholes árképletek és görögök – vektorizált numpy implementáció.
"""

import numpy as np
from scipy.stats import norm


def _d1_d2(S, K, T, r, sigma, q=0.0):
    """d1 és d2 segédváltozók kiszámítása."""
    with np.errstate(divide='ignore', invalid='ignore'):
        d1 = (np.log(S / K) + (r - q + 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    return d1, d2


def bs_call(S, K, T, r, sigma, q=0.0):
    """Black-Scholes call opció ára.

    Paraméterek
    -----------
    S     : mögöttes eszköz árfolyama
    K     : kötési ár
    T     : lejáratig hátralévő idő (évben)
    r     : kockázatmentes kamatláb
    sigma : volatilitás
    q     : folyamatos osztalékhozam (default 0)

    Visszaadja
    ----------
    call ár (skalár vagy numpy tömb)
    """
    S, K, T, r, sigma, q = (np.asarray(x, dtype=float) for x in (S, K, T, r, sigma, q))
    intrinsic = np.maximum(S * np.exp(-q * T) - K * np.exp(-r * T), 0.0)
    # T ≈ 0 esetén intrinsic value
    near_expiry = T < 1e-8
    d1, d2 = _d1_d2(S, K, T, r, sigma, q)
    price = S * np.exp(-q * T) * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2)
    return np.where(near_expiry, intrinsic, price)


def bs_put(S, K, T, r, sigma, q=0.0):
    """Black-Scholes put opció ára (put-call paritásból)."""
    S, K, T, r, sigma, q = (np.asarray(x, dtype=float) for x in (S, K, T, r, sigma, q))
    call = bs_call(S, K, T, r, sigma, q)
    return call - S * np.exp(-q * T) + K * np.exp(-r * T)


def bs_delta(S, K, T, r, sigma, q=0.0):
    """Call delta (∂C/∂S)."""
    S, K, T, r, sigma, q = (np.asarray(x, dtype=float) for x in (S, K, T, r, sigma, q))
    near_expiry = T < 1e-8
    d1, _ = _d1_d2(S, K, T, r, sigma, q)
    delta = np.exp(-q * T) * norm.cdf(d1)
    intrinsic_delta = np.where(S > K, np.exp(-q * T), 0.0)
    return np.where(near_expiry, intrinsic_delta, delta)


def bs_gamma(S, K, T, r, sigma, q=0.0):
    """Gamma (∂²C/∂S²)."""
    S, K, T, r, sigma, q = (np.asarray(x, dtype=float) for x in (S, K, T, r, sigma, q))
    near_expiry = T < 1e-8
    d1, _ = _d1_d2(S, K, T, r, sigma, q)
    gamma = np.exp(-q * T) * norm.pdf(d1) / (S * sigma * np.sqrt(T))
    return np.where(near_expiry, 0.0, gamma)


def bs_vega(S, K, T, r, sigma, q=0.0):
    """Vega (∂C/∂σ), 1%-pontos változásra normálva."""
    S, K, T, r, sigma, q = (np.asarray(x, dtype=float) for x in (S, K, T, r, sigma, q))
    near_expiry = T < 1e-8
    d1, _ = _d1_d2(S, K, T, r, sigma, q)
    vega = S * np.exp(-q * T) * norm.pdf(d1) * np.sqrt(T) * 0.01
    return np.where(near_expiry, 0.0, vega)


def bs_theta(S, K, T, r, sigma, q=0.0):
    """Call theta (∂C/∂T), naptári napra normálva (/365)."""
    S, K, T, r, sigma, q = (np.asarray(x, dtype=float) for x in (S, K, T, r, sigma, q))
    near_expiry = T < 1e-8
    d1, d2 = _d1_d2(S, K, T, r, sigma, q)
    term1 = -S * np.exp(-q * T) * norm.pdf(d1) * sigma / (2 * np.sqrt(T))
    term2 = -r * K * np.exp(-r * T) * norm.cdf(d2)
    term3 = q * S * np.exp(-q * T) * norm.cdf(d1)
    theta = (term1 + term2 + term3) / 365.0
    return np.where(near_expiry, 0.0, theta)


def bs_rho(S, K, T, r, sigma, q=0.0):
    """Call rho (∂C/∂r), 1%-pontos változásra normálva."""
    S, K, T, r, sigma, q = (np.asarray(x, dtype=float) for x in (S, K, T, r, sigma, q))
    near_expiry = T < 1e-8
    _, d2 = _d1_d2(S, K, T, r, sigma, q)
    rho = K * T * np.exp(-r * T) * norm.cdf(d2) * 0.01
    return np.where(near_expiry, 0.0, rho)
