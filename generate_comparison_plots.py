"""
Összehasonlító és v2 tanulási görbe plotok generálása.

Kimenti:
    results/plots/comparison/  — eredeti vs. v2 összehasonlítók
    results/plots/v2/          — csak a v2 futtatások

Futtatás:
    python generate_comparison_plots.py
"""

import os
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import pandas as pd
import numpy as np

RESULTS_DIR  = 'results'
PLOTS_COMP   = os.path.join(RESULTS_DIR, 'plots', 'comparison')
PLOTS_V2     = os.path.join(RESULTS_DIR, 'plots', 'v2')
os.makedirs(PLOTS_COMP, exist_ok=True)
os.makedirs(PLOTS_V2,   exist_ok=True)

# Modellek (MLPPricer-nek nincs v2, de az összehasonlítóban referencia)
ALL_MODELS = [
    'MLPPricer', 'DeepMLPPricer', 'ResNetPricer',
    'GELUResNetPricer', 'DenseMLPPricer', 'HighwayPricer', 'FINNPricer',
]
V2_MODELS = [m for m in ALL_MODELS if m != 'MLPPricer']

COLORS = {
    'MLPPricer':       '#1f77b4',
    'DeepMLPPricer':   '#ff7f0e',
    'ResNetPricer':    '#2ca02c',
    'GELUResNetPricer':'#d62728',
    'DenseMLPPricer':  '#9467bd',
    'HighwayPricer':   '#8c564b',
    'FINNPricer':      '#e377c2',
}
MARKERS = ['o', 's', '^', 'D', 'v', 'P', '*']


def load_csv(model_name: str, suffix: str = '') -> pd.DataFrame | None:
    path = os.path.join(RESULTS_DIR, f'training_history_{model_name}{suffix}.csv')
    return pd.read_csv(path) if os.path.exists(path) else None


def best_epoch_line(ax, df: pd.DataFrame, color: str, linestyle: str = '-'):
    best_idx = df['val_loss'].idxmin()
    best_ep  = df.loc[best_idx, 'epoch']
    best_val = df.loc[best_idx, 'val_loss']
    ax.axvline(best_ep, color=color, linestyle=linestyle, linewidth=1.0, alpha=0.5)
    return best_ep, best_val


# -----------------------------------------------------------------------
# 1. Összehasonlító: eredeti vs. v2 — egyedi görbék modellenként
# -----------------------------------------------------------------------
print("1. Összehasonlító egyedi görbék (eredeti vs. v2)...")
for model in V2_MODELS:
    df_orig = load_csv(model)
    df_v2   = load_csv(model, '_v2')
    if df_orig is None or df_v2 is None:
        continue

    color = COLORS[model]
    fig, axes = plt.subplots(1, 2, figsize=(14, 5), sharey=True)
    fig.suptitle(f'{model} — eredeti (patience=10) vs. v2 (patience=30)', fontsize=13)

    for ax, df, label, ls in [
        (axes[0], df_orig, 'Eredeti (patience=10)', '--'),
        (axes[1], df_v2,   'v2 (patience=30)',      '-'),
    ]:
        ax.plot(df['epoch'], df['train_loss'], color='steelblue',
                linewidth=1.8, linestyle=ls, label='Train MSE')
        ax.plot(df['epoch'], df['val_loss'],   color='tomato',
                linewidth=1.8, linestyle=ls, label='Val MSE')
        best_ep, best_val = best_epoch_line(ax, df, 'gray', ':')
        ax.set_title(f'{label}\nbest epoch={best_ep}, val={best_val:.2e}', fontsize=10)
        ax.set_xlabel('Epoch')
        ax.set_yscale('log')
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=9)

    axes[0].set_ylabel('MSE Loss (log)')
    fig.tight_layout()
    out = os.path.join(PLOTS_COMP, f'compare_{model}.png')
    fig.savefig(out, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"   {out}")


# -----------------------------------------------------------------------
# 2. Összehasonlító: eredeti vs. v2 — val MSE egy ábrán (overlay)
#    Minden modellnél: eredeti=szaggatott, v2=folytonos
# -----------------------------------------------------------------------
print("2. Összehasonlító összes modell val MSE overlay...")
fig, ax = plt.subplots(figsize=(14, 7))
for i, model in enumerate(ALL_MODELS):
    color = COLORS[model]
    df_orig = load_csv(model)
    df_v2   = load_csv(model, '_v2')
    if df_orig is not None:
        ax.plot(df_orig['epoch'], df_orig['val_loss'],
                color=color, linewidth=1.8, linestyle='--', alpha=0.7,
                label=f'{model} eredeti')
    if df_v2 is not None:
        ax.plot(df_v2['epoch'], df_v2['val_loss'],
                color=color, linewidth=2.2, linestyle='-',
                label=f'{model} v2')

ax.set_xlabel('Epoch', fontsize=12)
ax.set_ylabel('Val MSE (log)', fontsize=12)
ax.set_title('Validációs MSE — eredeti (szaggatott) vs. v2 (folytonos)', fontsize=13)
ax.set_yscale('log')
ax.grid(True, alpha=0.3)
ax.legend(fontsize=8, ncol=2, loc='upper right')
fig.tight_layout()
out = os.path.join(PLOTS_COMP, 'compare_all_val_mse_overlay.png')
fig.savefig(out, dpi=150, bbox_inches='tight')
plt.close(fig)
print(f"   {out}")


# -----------------------------------------------------------------------
# 3. Összehasonlító: grouped bar chart — legjobb val MSE eredeti vs. v2
# -----------------------------------------------------------------------
print("3. Grouped bar chart (best val MSE: eredeti vs. v2)...")
models_with_both = [m for m in ALL_MODELS
                    if load_csv(m) is not None and (load_csv(m, '_v2') is not None or m == 'MLPPricer')]

orig_vals, v2_vals, labels = [], [], []
for model in ALL_MODELS:
    df_orig = load_csv(model)
    df_v2   = load_csv(model, '_v2')
    if df_orig is None:
        continue
    labels.append(model.replace('Pricer', ''))
    orig_vals.append(df_orig['val_loss'].min())
    v2_vals.append(df_v2['val_loss'].min() if df_v2 is not None else None)

x      = np.arange(len(labels))
width  = 0.35
fig, ax = plt.subplots(figsize=(13, 6))
bars1 = ax.bar(x - width/2, orig_vals, width, label='Eredeti (patience=10)',
               color='steelblue', edgecolor='black', linewidth=0.6, alpha=0.85)
bars2_vals = [v if v is not None else 0 for v in v2_vals]
bars2_vis  = [v is not None for v in v2_vals]
bars2 = ax.bar(x + width/2, bars2_vals, width, label='v2 (patience=30)',
               color='tomato', edgecolor='black', linewidth=0.6, alpha=0.85)
for bar, vis in zip(bars2, bars2_vis):
    if not vis:
        bar.set_visible(False)

ax.set_yscale('log')
ax.set_xlabel('Modell', fontsize=12)
ax.set_ylabel('Legjobb Val MSE (log-skála)', fontsize=12)
ax.set_title('Legjobb validációs MSE — eredeti vs. v2 futtatás', fontsize=13)
ax.set_xticks(x)
ax.set_xticklabels(labels, rotation=15, ha='right')
ax.legend(fontsize=10)
ax.grid(True, axis='y', alpha=0.3)

for bar, val in zip(bars1, orig_vals):
    ax.text(bar.get_x() + bar.get_width()/2, val * 1.2,
            f'{val:.1e}', ha='center', va='bottom', fontsize=7.5, fontweight='bold')
for bar, val, vis in zip(bars2, v2_vals, bars2_vis):
    if vis:
        ax.text(bar.get_x() + bar.get_width()/2, val * 1.2,
                f'{val:.1e}', ha='center', va='bottom', fontsize=7.5, fontweight='bold')

fig.tight_layout()
out = os.path.join(PLOTS_COMP, 'compare_best_val_mse_bar.png')
fig.savefig(out, dpi=150, bbox_inches='tight')
plt.close(fig)
print(f"   {out}")


# -----------------------------------------------------------------------
# 4. Csak v2: egyedi tanulási görbék
# -----------------------------------------------------------------------
print("4. V2 egyedi tanulási görbék...")
for model in V2_MODELS:
    df = load_csv(model, '_v2')
    if df is None:
        continue
    best_ep  = int(df.loc[df['val_loss'].idxmin(), 'epoch'])
    best_val = df['val_loss'].min()

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(df['epoch'], df['train_loss'], color='steelblue', linewidth=2, label='Train MSE')
    ax.plot(df['epoch'], df['val_loss'],   color='tomato',    linewidth=2, label='Val MSE')
    ax.axvline(best_ep, color='gray', linestyle=':', linewidth=1.2,
               label=f'Best epoch={best_ep} ({best_val:.2e})')
    ax.set_xlabel('Epoch', fontsize=12)
    ax.set_ylabel('MSE Loss (log)', fontsize=12)
    ax.set_title(f'{model} — v2 tanulási görbe (patience=30)', fontsize=13)
    ax.set_yscale('log')
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    out = os.path.join(PLOTS_V2, f'v2_training_curve_{model}.png')
    fig.savefig(out, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"   {out}")


# -----------------------------------------------------------------------
# 5. Csak v2: összes modell val MSE egy ábrán
# -----------------------------------------------------------------------
print("5. V2 összes modell val MSE overlay...")
fig, ax = plt.subplots(figsize=(13, 7))
# MLPPricer referenciaként (eredeti, nincs v2)
df_mlp = load_csv('MLPPricer')
if df_mlp is not None:
    ax.plot(df_mlp['epoch'], df_mlp['val_loss'],
            color=COLORS['MLPPricer'], linewidth=2, linestyle='--',
            label='MLPPricer (referencia, patience=10)')

for i, model in enumerate(V2_MODELS):
    df = load_csv(model, '_v2')
    if df is None:
        continue
    ax.plot(df['epoch'], df['val_loss'],
            color=COLORS[model], linewidth=2,
            label=f"{model} (min={df['val_loss'].min():.2e})")

ax.set_xlabel('Epoch', fontsize=12)
ax.set_ylabel('Val MSE (log)', fontsize=12)
ax.set_title('V2 futtatások validációs MSE-je (MLPPricer referenciával)', fontsize=13)
ax.set_yscale('log')
ax.grid(True, alpha=0.3)
ax.legend(fontsize=9, loc='upper right')
fig.tight_layout()
out = os.path.join(PLOTS_V2, 'v2_all_val_mse_overlay.png')
fig.savefig(out, dpi=150, bbox_inches='tight')
plt.close(fig)
print(f"   {out}")


# -----------------------------------------------------------------------
# 6. Csak v2: bar chart (legjobb val MSE, csökkenő sorrend)
#    MLPPricer is benne van referenciaként
# -----------------------------------------------------------------------
print("6. V2 bar chart (best val MSE)...")
bar_data = {}
df_mlp = load_csv('MLPPricer')
if df_mlp is not None:
    bar_data['MLPPricer'] = df_mlp['val_loss'].min()
for model in V2_MODELS:
    df = load_csv(model, '_v2')
    if df is not None:
        bar_data[model] = df['val_loss'].min()

sorted_models = sorted(bar_data, key=bar_data.get)
sorted_vals   = [bar_data[m] for m in sorted_models]
bar_colors    = [COLORS[m] for m in sorted_models]
labels_short  = [m.replace('Pricer', '') for m in sorted_models]

# MLPPricer szaggatott szegéllyel (referencia, nem v2)
edge_colors = ['black' if m != 'MLPPricer' else 'navy' for m in sorted_models]
line_widths = [0.6 if m != 'MLPPricer' else 2.0 for m in sorted_models]
hatches     = ['' if m != 'MLPPricer' else '///' for m in sorted_models]

fig, ax = plt.subplots(figsize=(12, 6))
bars = ax.bar(labels_short, sorted_vals, color=bar_colors,
              edgecolor=edge_colors, linewidth=line_widths)
for bar, hatch in zip(bars, hatches):
    bar.set_hatch(hatch)

ax.set_yscale('log')
ax.set_xlabel('Modell', fontsize=12)
ax.set_ylabel('Legjobb Val MSE (log-skála)', fontsize=12)
ax.set_title('V2 futtatások legjobb validációs MSE-je\n(MLPPricer: eredeti referencia, csíkozva)', fontsize=12)
ax.grid(True, axis='y', alpha=0.3)
plt.xticks(rotation=20, ha='right')

for bar, val in zip(bars, sorted_vals):
    ax.text(bar.get_x() + bar.get_width()/2, val * 1.2,
            f'{val:.2e}', ha='center', va='bottom', fontsize=9, fontweight='bold')

mlp_patch = mpatches.Patch(facecolor=COLORS['MLPPricer'], hatch='///',
                            edgecolor='navy', label='MLPPricer (eredeti, referencia)')
ax.legend(handles=[mlp_patch], fontsize=9)
fig.tight_layout()
out = os.path.join(PLOTS_V2, 'v2_best_val_mse_bar.png')
fig.savefig(out, dpi=150, bbox_inches='tight')
plt.close(fig)
print(f"   {out}")


print("\nKész. Mentett mappák:")
print(f"  {PLOTS_COMP}")
print(f"  {PLOTS_V2}")
