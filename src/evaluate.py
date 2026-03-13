"""
Modell kiértékelés és összehasonlítás neurális háló opciós árazóhoz.
"""

from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from src.model import get_model
from src.train import (OptionDataset, AugmentedOptionDataset,
                       DEFAULT_FEATURE_COLS, DEFAULT_TARGET_COL)

def _dataset_cls(feature_cols: list):
    """OptionDataset vagy AugmentedOptionDataset a feature_cols alapján."""
    return AugmentedOptionDataset if 'is_put' in feature_cols else OptionDataset


# Moneyness szegmens határok (S/K alapján)
MONEYNESS_SEGMENTS = {
    'OTM': (0.0,  0.9),
    'ATM': (0.9,  1.1),
    'ITM': (1.1, np.inf),
}


# ---------------------------------------------------------------------------
# Modell betöltés
# ---------------------------------------------------------------------------

def load_model(checkpoint_path: str, device: str = 'cpu') -> Tuple[nn.Module, dict]:
    """
    Checkpoint-ból rekonstruálja és betölti a modellt.

    Args:
        checkpoint_path: .pt fájl elérési útja
        device         : 'cpu' | 'cuda' | 'mps'

    Returns:
        (model, metadata) tuple, ahol metadata tartalmazza a history-t,
        feature_cols-t, target_col-t stb.
    """
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)

    model_class  = checkpoint['model_class']
    model_kwargs = checkpoint['model_kwargs']

    model = get_model(model_class, **model_kwargs)
    model.load_state_dict(checkpoint['state_dict'])
    model = model.to(device)
    model.eval()

    meta = {
        'feature_cols': checkpoint.get('feature_cols', DEFAULT_FEATURE_COLS),
        'target_col':   checkpoint.get('target_col',   DEFAULT_TARGET_COL),
        'history':      checkpoint.get('history', {}),
        'best_epoch':   checkpoint.get('best_epoch', None),
        'val_loss':     checkpoint.get('val_loss', None),
        'model_class':  model_class,
        'model_kwargs': model_kwargs,
    }
    return model, meta


# ---------------------------------------------------------------------------
# Metrikák
# ---------------------------------------------------------------------------

def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    """
    Kiértékelési metrikák számítása (pure numpy, sklearn nélkül).

    Metrikák:
        RMSE      : négyzetgyök átlagos négyzetes hiba
        MAE       : átlagos abszolút hiba
        MAPE      : átlagos abszolút százalékos hiba (ε=1e-8 védelem)
        max_error : maximális abszolút hiba
        R²        : determinációs együttható

    Args:
        y_true: valós értékek, shape (N,)
        y_pred: becsült értékek, shape (N,)

    Returns:
        dict metrikákkal
    """
    y_true = y_true.ravel()
    y_pred = y_pred.ravel()
    residuals = y_true - y_pred

    rmse      = float(np.sqrt(np.mean(residuals ** 2)))
    mae       = float(np.mean(np.abs(residuals)))
    mape      = float(np.mean(np.abs(residuals) / (np.abs(y_true) + 1e-8))) * 100.0
    max_error = float(np.max(np.abs(residuals)))

    ss_res = np.sum(residuals ** 2)
    ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else float('nan')

    return {
        'RMSE':      rmse,
        'MAE':       mae,
        'MAPE (%)':  mape,
        'max_error': max_error,
        'R2':        float(r2),
    }


# ---------------------------------------------------------------------------
# Modell kiértékelés
# ---------------------------------------------------------------------------

def evaluate_model(
    checkpoint_path: str,
    test_path: str,
    batch_size: int = 4096,
    device: str = 'cpu',
    verbose: bool = True,
) -> dict:
    """
    Checkpoint alapján kiértékeli a modellt a teszt halmazon.

    Returns:
        dict: metrikák + y_true, y_pred tömbök
    """
    model, meta = load_model(checkpoint_path, device=device)
    feature_cols = meta['feature_cols']
    target_col   = meta['target_col']

    test_ds = _dataset_cls(feature_cols)(test_path, feature_cols, target_col, device=device)
    loader  = DataLoader(test_ds, batch_size=batch_size, shuffle=False)

    y_preds = []
    with torch.no_grad():
        for X_batch, _ in loader:
            y_preds.append(model(X_batch).cpu().numpy())

    y_pred = np.concatenate(y_preds, axis=0).ravel()
    y_true = test_ds.y.cpu().numpy().ravel()

    metrics = compute_metrics(y_true, y_pred)

    if verbose:
        print(f"\nKiértékelés: {checkpoint_path}")
        print(f"  Modell:  {meta['model_class']}  ({meta['model_kwargs']})")
        print(f"  Teszt:   {len(test_ds):,} sor")
        if meta['best_epoch']:
            print(f"  Legjobb epoch: {meta['best_epoch']}, val loss: {meta['val_loss']:.6f}")
        print()
        print("  Metrikák:")
        _print_metrics(metrics)

    return {'metrics': metrics, 'y_true': y_true, 'y_pred': y_pred, 'meta': meta}


def _print_metrics(metrics: dict, indent: int = 4):
    """Metrikák szép kiírása."""
    pad = " " * indent
    for k, v in metrics.items():
        print(f"{pad}{k:<15} {v:.6f}")


# ---------------------------------------------------------------------------
# Szegmentált metrikák
# ---------------------------------------------------------------------------

def compute_segmented_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    moneyness: np.ndarray,
) -> Dict[str, dict]:
    """
    Moneyness-szegmentált metrikák: OTM / ATM / ITM + összesített.

    Args:
        y_true    : valós árak, shape (N,)
        y_pred    : becsült árak, shape (N,)
        moneyness : S/K értékek, shape (N,)

    Returns:
        dict[szegmens_név -> metrika_dict], tartalmaz 'all' kulcsot is
    """
    y_true    = y_true.ravel()
    y_pred    = y_pred.ravel()
    moneyness = moneyness.ravel()

    all_metrics = compute_metrics(y_true, y_pred)
    all_metrics['n'] = len(y_true)
    results: Dict[str, dict] = {'all': all_metrics}
    for seg_name, (lo, hi) in MONEYNESS_SEGMENTS.items():
        mask = (moneyness >= lo) & (moneyness < hi)
        n = int(mask.sum())
        if n == 0:
            results[seg_name] = {k: float('nan') for k in all_metrics}
            results[seg_name]['n'] = 0
        else:
            m = compute_metrics(y_true[mask], y_pred[mask])
            m['n'] = n
            results[seg_name] = m
    return results


def evaluate_model_segmented(
    checkpoint_path: str,
    test_path: str,
    batch_size: int = 4096,
    device: str = 'cpu',
    verbose: bool = True,
) -> dict:
    """
    Moneyness-szegmentált kiértékelés a teszt halmazon.

    A parquet fájlból a 'moneyness' oszlopot (S/K) olvassa be a szegmentáláshoz.
    A modell inference-e azonos az evaluate_model-lel.

    Returns:
        dict: segmented_metrics (OTM/ATM/ITM/all) + y_true, y_pred, moneyness, meta
    """
    model, meta = load_model(checkpoint_path, device=device)
    feature_cols = meta['feature_cols']
    target_col   = meta['target_col']

    test_ds = _dataset_cls(feature_cols)(test_path, feature_cols, target_col, device=device)
    loader  = DataLoader(test_ds, batch_size=batch_size, shuffle=False)

    y_preds = []
    with torch.no_grad():
        for X_batch, _ in loader:
            y_preds.append(model(X_batch).cpu().numpy())

    y_pred    = np.concatenate(y_preds, axis=0).ravel()
    y_true    = test_ds.y.cpu().numpy().ravel()
    moneyness = pd.read_parquet(test_path, columns=['moneyness'])['moneyness'].to_numpy()
    # AugmentedOptionDataset esetén a dataset 2x akkora (call + put),
    # a moneyness értékek mindkét félben azonosak → megduplázás
    if len(moneyness) < len(y_true):
        moneyness = np.tile(moneyness, len(y_true) // len(moneyness))

    seg_metrics = compute_segmented_metrics(y_true, y_pred, moneyness)

    if verbose:
        print(f"\nSzegmentált kiértékelés: {checkpoint_path}")
        print(f"  Modell: {meta['model_class']}  ({meta['model_kwargs']})")
        print(f"  Teszt:  {len(test_ds):,} sor\n")
        _print_segmented_metrics(seg_metrics)

    return {
        'segmented_metrics': seg_metrics,
        'y_true':    y_true,
        'y_pred':    y_pred,
        'moneyness': moneyness,
        'meta':      meta,
    }


def _print_segmented_metrics(seg_metrics: Dict[str, dict]):
    """Szegmentált metrikák táblázatos kiírása.

    MAPE-t kizárja: normalizált OTM árak közel nullához torzítják.
    """
    skip        = {'n', 'MAPE (%)'}
    metric_keys = [k for k in seg_metrics['all'].keys() if k not in skip]
    segments    = ['OTM', 'ATM', 'ITM', 'all']
    col0, col   = 8, 14

    header = "Szegmens".ljust(col0) + "N".rjust(10)
    header += "".join(k.rjust(col) for k in metric_keys)
    sep    = "-" * len(header)

    print(header)
    print(sep)
    for seg in segments:
        m   = seg_metrics[seg]
        row = seg.ljust(col0) + str(m['n']).rjust(10)
        row += "".join(f"{m[k]:.6f}".rjust(col) for k in metric_keys)
        print(row)
    print(sep)
    print("  (MAPE kihagyva: normalizált OTM árak közel nullához torzítják)")


# ---------------------------------------------------------------------------
# Modellek összehasonlítása
# ---------------------------------------------------------------------------

def compare_models(
    checkpoint_paths: List[str],
    test_path: str,
    batch_size: int = 4096,
    device: str = 'cpu',
):
    """
    Több modell összehasonlítása táblázatos formában.

    Args:
        checkpoint_paths: .pt fájlok listája
        test_path       : teszt parquet elérési útja
        batch_size      : batch méret inference-hez
        device          : számítási eszköz
    """
    results = []
    for cp in checkpoint_paths:
        res = evaluate_model(cp, test_path, batch_size=batch_size,
                             device=device, verbose=False)
        name = res['meta']['model_class']
        results.append((name, res['metrics']))

    # Fejléc
    metric_names = list(results[0][1].keys())
    col_model = 18
    col_metric = 14

    header  = "Modell".ljust(col_model)
    header += "".join(m.rjust(col_metric) for m in metric_names)
    sep = "-" * len(header)

    print("\nModellek összehasonlítása (teszt halmaz)")
    print(sep)
    print(header)
    print(sep)
    for name, metrics in results:
        row = name.ljust(col_model)
        row += "".join(f"{v:.6f}".rjust(col_metric) for v in metrics.values())
        print(row)
    print(sep)
