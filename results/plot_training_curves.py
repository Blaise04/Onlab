"""
Tanulasi gorbe grafikonok generalasa a CSV fajlokbol.

Futtatasa:
    python results/plot_training_curves.py
"""

import os
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import pandas as pd

RESULTS_DIR = os.path.dirname(os.path.abspath(__file__))
PLOTS_DIR   = os.path.join(RESULTS_DIR, 'plots')
os.makedirs(PLOTS_DIR, exist_ok=True)

MODEL_NAMES = [
    'MLPPricer', 'DeepMLPPricer', 'ResNetPricer',
    'GELUResNetPricer', 'DenseMLPPricer', 'HighwayPricer', 'FINNPricer',
]
COLORS  = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b', '#e377c2']
MARKERS = ['o', 's', '^', 'D', 'v', 'P', '*']

results = {}
for name in MODEL_NAMES:
    csv_path = os.path.join(RESULTS_DIR, f'training_history_{name}.csv')
    if os.path.exists(csv_path):
        results[name] = pd.read_csv(csv_path)

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

fig, ax = plt.subplots(figsize=(13, 7))
for i, (name, df) in enumerate(results.items()):
    ax.plot(df['epoch'], df['val_loss'],
            label=f"{name} (min={df['val_loss'].min():.5f})",
            color=COLORS[i % len(COLORS)], linewidth=2)
ax.set_yscale('log')
    ax.legend()
fig.savefig(os.path.join(PLOTS_DIR, 'training_curve_comparison.png'), dpi=150, bbox_inches='tight')
plt.close(fig)
print('Grafikonok generalva.')
