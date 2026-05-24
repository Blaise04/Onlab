"""
Tanulasi gorbe grafikonok generalasa a CSV fajlokbol.

Futtatasa:
    python results/plot_training_curves.py
"""

import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import pandas as pd

from src.model import (
    MLPPricer, DeepMLPPricer, ResNetPricer,
    GELUResNetPricer, DenseMLPPricer, HighwayPricer, FINNPricer,
    count_parameters,
)

RESULTS_DIR = os.path.dirname(os.path.abspath(__file__))
PLOTS_DIR   = os.path.join(RESULTS_DIR, 'plots')
os.makedirs(PLOTS_DIR, exist_ok=True)

MODEL_NAMES = [
    'MLPPricer', 'DeepMLPPricer', 'ResNetPricer',
    'GELUResNetPricer', 'DenseMLPPricer', 'HighwayPricer', 'FINNPricer',
]
COLORS  = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b', '#e377c2']
MARKERS = ['o', 's', '^', 'D', 'v', 'P', '*']

# Modell osztályok default kwargs-szal (paraméterszám dinamikus kiszámításához)
MODEL_CLASSES = {
    'MLPPricer':       MLPPricer,
    'DeepMLPPricer':   DeepMLPPricer,
    'ResNetPricer':    ResNetPricer,
    'GELUResNetPricer': GELUResNetPricer,
    'DenseMLPPricer':  DenseMLPPricer,
    'HighwayPricer':   HighwayPricer,
    'FINNPricer':      FINNPricer,
}

# CSV betöltés — hiányzó fájlok csendesen kihagyva
results = {}
for name in MODEL_NAMES:
    csv_path = os.path.join(RESULTS_DIR, f'training_history_{name}.csv')
    if os.path.exists(csv_path):
        results[name] = pd.read_csv(csv_path)

# -----------------------------------------------------------------------
# 1. Egyedi tanulasi gorbek
# -----------------------------------------------------------------------
for name, df in results.items():
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(df['epoch'], df['train_loss'], label='Train MSE', color='steelblue', linewidth=2)
    ax.plot(df['epoch'], df['val_loss'],   label='Val MSE',   color='tomato',    linewidth=2)
    ax.set_xlabel('Epoch', fontsize=12)
    ax.set_ylabel('MSE Loss', fontsize=12)
    ax.set_title(f'{name} Tanulasi Gorbe', fontsize=14)
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_yscale('log')
    fig.savefig(os.path.join(PLOTS_DIR, f'training_curve_{name}.png'), dpi=150, bbox_inches='tight')
    plt.close(fig)

# -----------------------------------------------------------------------
# 2. Osszehasonlito gorbe (minden modell val MSE egy abran)
# -----------------------------------------------------------------------
fig, ax = plt.subplots(figsize=(13, 7))
for i, (name, df) in enumerate(results.items()):
    ax.plot(df['epoch'], df['val_loss'],
            label=f"{name} (min={df['val_loss'].min():.5f})",
            color=COLORS[i % len(COLORS)], linewidth=2)
ax.set_xlabel('Epoch', fontsize=12)
ax.set_ylabel('Val MSE', fontsize=12)
ax.legend()
ax.grid(True, alpha=0.3)
ax.set_yscale('log')
fig.savefig(os.path.join(PLOTS_DIR, 'training_curve_comparison.png'), dpi=150, bbox_inches='tight')
plt.close(fig)

# -----------------------------------------------------------------------
# 3. Validacios MSE bar chart (csokkeno sorrendben)
# -----------------------------------------------------------------------
if results:
    best_val = {name: df['val_loss'].min() for name, df in results.items()}
    sorted_names = sorted(best_val, key=best_val.get)  # csokkeno: legkisebb elol
    sorted_vals  = [best_val[n] for n in sorted_names]
    bar_colors   = [COLORS[MODEL_NAMES.index(n) % len(COLORS)] for n in sorted_names]

    fig, ax = plt.subplots(figsize=(12, 6))
    bars = ax.bar(sorted_names, sorted_vals, color=bar_colors, edgecolor='black', linewidth=0.7)
    ax.set_yscale('log')
    ax.set_xlabel('Modell', fontsize=12)
    ax.set_ylabel('Legjobb Val MSE (log-skala)', fontsize=12)
    ax.grid(True, axis='y', alpha=0.3)
    plt.xticks(rotation=20, ha='right')

    # Ertekfeliratok az oszlopokon
    for bar, val in zip(bars, sorted_vals):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            val * 1.15,
            f'{val:.2e}',
            ha='center', va='bottom', fontsize=9, fontweight='bold',
        )

    fig.savefig(os.path.join(PLOTS_DIR, 'val_mse_bar_chart.png'), dpi=150, bbox_inches='tight')
    plt.close(fig)

# -----------------------------------------------------------------------
# 4. Parameterszam vs. val MSE scatter plot
# -----------------------------------------------------------------------
if results:
    param_counts = {}
    for name, cls in MODEL_CLASSES.items():
        if name in results:
            model_instance = cls()
            param_counts[name] = count_parameters(model_instance)

    if param_counts:
        fig, ax = plt.subplots(figsize=(10, 7))
        for i, name in enumerate(param_counts):
            x_val = param_counts[name]
            y_val = best_val[name]
            color = COLORS[MODEL_NAMES.index(name) % len(COLORS)]
            marker = MARKERS[MODEL_NAMES.index(name) % len(MARKERS)]
            ax.scatter(x_val, y_val, color=color, marker=marker, s=120, zorder=5, label=name)
            ax.annotate(
                name,
                (x_val, y_val),
                textcoords='offset points',
                xytext=(8, 4),
                fontsize=9,
            )
        ax.set_xscale('log')
        ax.set_yscale('log')
        ax.set_xlabel('Paraméterszám (log-skála)', fontsize=12)
        ax.set_ylabel('Legjobb Val MSE (log-skála)', fontsize=12)
        ax.set_title('Paraméterszám vs. validációs MSE', fontsize=14)
        ax.legend(fontsize=9, loc='upper right')
        ax.grid(True, alpha=0.3, which='both')
        fig.savefig(os.path.join(PLOTS_DIR, 'params_vs_val_mse.png'), dpi=150, bbox_inches='tight')
        plt.close(fig)

print('Grafikonok generalva.')
