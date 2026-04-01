"""
Osszes modell beapontitasa, epoch-szintu CSV naplozas, grafikonok es elemzes.

Futtatas:
    python run_experiments.py
    python run_experiments.py --epochs 50 --patience 15
"""

import argparse
import csv
import os
import random
import time

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from src.model import get_model, count_parameters
from src.train import OptionDataset, DEFAULT_FEATURE_COLS, DEFAULT_TARGET_COL

# ---------------------------------------------------------------------------
# Konstansok
# ---------------------------------------------------------------------------

RESULTS_DIR = "results"
PLOTS_DIR   = os.path.join(RESULTS_DIR, "plots")

# Modellek konfiguracioja (architekturak valtozatlanul hagyva)
MODEL_CONFIGS = [
    ('mlp',         {'input_dim': 5, 'hidden_dim': 100, 'n_layers': 4},                 'MLPPricer'),
    ('deep_mlp',    {'input_dim': 5, 'hidden_dim': 256, 'n_layers': 4, 'dropout': 0.1}, 'DeepMLPPricer'),
    ('resnet',      {'input_dim': 5, 'hidden_dim': 256, 'n_blocks': 3, 'dropout': 0.1}, 'ResNetPricer'),
    ('gelu_resnet', {'input_dim': 5, 'hidden_dim': 256, 'n_blocks': 3, 'dropout': 0.1}, 'GELUResNetPricer'),
    ('dense_mlp',   {'input_dim': 5, 'hidden_dim': 128, 'n_layers': 4, 'dropout': 0.1}, 'DenseMLPPricer'),
    ('highway',     {'input_dim': 5, 'hidden_dim': 256, 'n_blocks': 4, 'dropout': 0.1}, 'HighwayPricer'),
    ('finn',        {'input_dim': 5, 'approx_dim': 64, 'resnet_dim': 256,
                     'n_blocks': 3, 'dropout': 0.1},                                    'FINNPricer'),
]


# ---------------------------------------------------------------------------
# Seed beallitas
# ---------------------------------------------------------------------------

def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


# ---------------------------------------------------------------------------
# Epoch-szintu tanitas MAE naplozassal
# ---------------------------------------------------------------------------

def train_with_history(
    model_key: str,
    model_kwargs: dict,
    class_name: str,
    train_path: str,
    val_path: str,
    results_dir: str,
    batch_size: int = 4096,
    max_epochs: int = 50,
    lr: float = 1e-3,
    weight_decay: float = 1e-4,
    patience: int = 15,
    device: str = 'auto',
    seed: int = 42,
) -> dict:
    """
    Betanit egy modellt, epoch-szintu metrikakat ment CSV-be.

    Visszater:
        dict: tanitasi history es osszefoglalo adatok
    """
    set_seed(seed)

    # Eszkoz kivalasztas
    if device == 'auto':
        if torch.cuda.is_available():
            device = 'cuda'
        elif torch.backends.mps.is_available():
            device = 'mps'
        else:
            device = 'cpu'
    device = torch.device(device)

    # Adatok betoltese
    print(f"  Adatok betoltese...")
    train_ds = OptionDataset(train_path, DEFAULT_FEATURE_COLS, DEFAULT_TARGET_COL, device=str(device))
    val_ds   = OptionDataset(val_path,   DEFAULT_FEATURE_COLS, DEFAULT_TARGET_COL, device=str(device))

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_loader   = DataLoader(val_ds,   batch_size=batch_size, shuffle=False)

    # Modell inicializalasa (architektura erintetlen)
    model = get_model(model_key, **model_kwargs).to(device)
    params = count_parameters(model)
    print(f"  Parameterek: {params:,}")
    print(f"  Eszkoz: {device}")

    # Optimizer es criterion (src/train.py-val azonos beallitasok)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    criterion = nn.MSELoss()
    mae_criterion = nn.L1Loss()
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=5, min_lr=1e-6
    )

    # CSV fajl megnyitasa
    csv_path = os.path.join(results_dir, f"training_history_{class_name}.csv")
    csv_file = open(csv_path, 'w', newline='', encoding='utf-8')
    writer = csv.writer(csv_file)
    writer.writerow(['epoch', 'train_loss', 'val_loss', 'train_mae', 'val_mae'])

    history = {
        'train_loss': [], 'val_loss': [],
        'train_mae': [], 'val_mae': [],
    }
    best_val_loss = float('inf')
    best_epoch = 0
    epochs_no_improve = 0

    print(f"  Tanitas ({max_epochs} epoch, patience={patience})...")

    for epoch in range(1, max_epochs + 1):
        t0 = time.time()

        # --- Tanitas ---
        model.train()
        train_loss_sum = 0.0
        train_mae_sum  = 0.0
        for X_batch, y_batch in train_loader:
            optimizer.zero_grad()
            pred = model(X_batch)
            loss = criterion(pred, y_batch)
            loss.backward()
            optimizer.step()
            n = len(X_batch)
            train_loss_sum += loss.item() * n
            train_mae_sum  += mae_criterion(pred, y_batch).item() * n

        train_loss = train_loss_sum / len(train_ds)
        train_mae  = train_mae_sum  / len(train_ds)

        # --- Validacio ---
        model.eval()
        val_loss_sum = 0.0
        val_mae_sum  = 0.0
        with torch.no_grad():
            for X_batch, y_batch in val_loader:
                pred = model(X_batch)
                n = len(X_batch)
                val_loss_sum += criterion(pred, y_batch).item() * n
                val_mae_sum  += mae_criterion(pred, y_batch).item() * n

        val_loss = val_loss_sum / len(val_ds)
        val_mae  = val_mae_sum  / len(val_ds)

        # History mentes
        history['train_loss'].append(train_loss)
        history['val_loss'].append(val_loss)
        history['train_mae'].append(train_mae)
        history['val_mae'].append(val_mae)

        # CSV sor irasa
        writer.writerow([epoch, train_loss, val_loss, train_mae, val_mae])
        csv_file.flush()

        scheduler.step(val_loss)

        # Early stopping es checkpoint
        improved = val_loss < best_val_loss
        if improved:
            best_val_loss = val_loss
            best_epoch = epoch
            epochs_no_improve = 0
            # Modell mentese results/ konyvtarba
            torch.save({
                'state_dict':   model.state_dict(),
                'model_class':  model_key,
                'model_kwargs': model_kwargs,
                'feature_cols': DEFAULT_FEATURE_COLS,
                'target_col':   DEFAULT_TARGET_COL,
                'best_epoch':   epoch,
                'val_loss':     val_loss,
            }, os.path.join(results_dir, f"model_{class_name}.pt"))
        else:
            epochs_no_improve += 1

        elapsed = time.time() - t0
        print(
            f"    Epoch {epoch:4d}/{max_epochs} | "
            f"Train MSE: {train_loss:.6f} | Val MSE: {val_loss:.6f} | "
            f"Val MAE: {val_mae:.6f} | {elapsed:.1f}s"
            + (" *" if improved else "")
        )

        if epochs_no_improve >= patience:
            print(f"    Early stopping: {patience} epoch ota nem javult.")
            break

    csv_file.close()
    print(f"  CSV mentve: {csv_path}")
    print(f"  Legjobb epoch: {best_epoch}, val MSE: {best_val_loss:.6f}")

    return {
        'class_name':      class_name,
        'params':          params,
        'best_epoch':      best_epoch,
        'final_train_loss': history['train_loss'][-1],
        'final_val_loss':   history['val_loss'][-1],
        'min_val_loss':     best_val_loss,
        'history':          history,
        'total_epochs':     len(history['train_loss']),
    }


# ---------------------------------------------------------------------------
# Osszehasonlito tablazat generalasa
# ---------------------------------------------------------------------------

def save_comparison_table(results: list, results_dir: str):
    """results/model_comparison.csv letrehozasa."""
    rows = []
    for r in results:
        rows.append({
            'modell_neve':       r['class_name'],
            'final_train_loss':  r['final_train_loss'],
            'final_val_loss':    r['final_val_loss'],
            'min_val_loss':      r['min_val_loss'],
            'best_epoch':        r['best_epoch'],
            'parametereк_szama': r['params'],
        })
    df = pd.DataFrame(rows)
    path = os.path.join(results_dir, 'model_comparison.csv')
    df.to_csv(path, index=False, encoding='utf-8')
    print(f"Osszehasonlito tablazat mentve: {path}")
    return df


# ---------------------------------------------------------------------------
# Grafikonok generalasa
# ---------------------------------------------------------------------------

def generate_plots(results: list, plots_dir: str, results_dir: str):
    """Tanulasi gorbe grafikonok generalasa matplotlib segitsegevel."""
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    os.makedirs(plots_dir, exist_ok=True)

    colors  = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728',
                '#9467bd', '#8c564b', '#e377c2']
    markers = ['o', 's', '^', 'D', 'v', 'P', '*']

    # 1. Egyedi tanulasi gorbe grafikon minden modellhez
    for r in results:
        name = r['class_name']
        hist = r['history']
        epochs = list(range(1, len(hist['train_loss']) + 1))

        fig, ax = plt.subplots(figsize=(10, 6))
        ax.plot(epochs, hist['train_loss'], label='Train MSE', color='steelblue',
                linewidth=2, marker='o', markersize=3)
        ax.plot(epochs, hist['val_loss'],   label='Val MSE',   color='tomato',
                linewidth=2, marker='s', markersize=3)
        ax.set_xlabel('Epoch', fontsize=12)
        ax.set_ylabel('MSE Loss', fontsize=12)
        ax.set_title(f'{name} Tanulasi Gorbe', fontsize=14, fontweight='bold')
        ax.legend(fontsize=11)
        ax.grid(True, alpha=0.3)
        ax.set_yscale('log')

        best_ep = r['best_epoch']
        best_loss = r['min_val_loss']
        ax.axvline(x=best_ep, color='gray', linestyle='--', alpha=0.7)
        n_ep = len(epochs)
        offset_x = best_ep + max(1, n_ep * 0.05)
        ax.annotate(
            f'Best: {best_loss:.5f}\n(ep. {best_ep})',
            xy=(best_ep, best_loss),
            xytext=(offset_x, best_loss * 1.5),
            arrowprops=dict(arrowstyle='->', color='gray'),
            fontsize=9, color='gray'
        )

        path = os.path.join(plots_dir, f'training_curve_{name}.png')
        fig.savefig(path, dpi=150, bbox_inches='tight')
        plt.close(fig)
        print(f"  Grafikon mentve: {path}")

    # 2. Osszehasonlito val_loss grafikon
    fig, ax = plt.subplots(figsize=(13, 7))
    for i, r in enumerate(results):
        name = r['class_name']
        hist = r['history']
        epochs = list(range(1, len(hist['val_loss']) + 1))
        ax.plot(
            epochs, hist['val_loss'],
            label=f"{name} (min={r['min_val_loss']:.5f})",
            color=colors[i % len(colors)],
            marker=markers[i % len(markers)],
            markersize=4, linewidth=2,
            markevery=max(1, len(epochs) // 15)
        )

    ax.set_xlabel('Epoch', fontsize=12)
    ax.set_ylabel('Validacios MSE Loss', fontsize=12)
    ax.set_title('Modellek Osszehasonlitasa -- Validacios Loss', fontsize=14, fontweight='bold')
    ax.legend(fontsize=9, loc='upper right')
    ax.grid(True, alpha=0.3)
    ax.set_yscale('log')

    path = os.path.join(plots_dir, 'training_curve_comparison.png')
    fig.savefig(path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"  Osszehasonlito grafikon mentve: {path}")

    # 3. MAE osszehasonlito grafikon
    fig, ax = plt.subplots(figsize=(13, 7))
    for i, r in enumerate(results):
        name = r['class_name']
        hist = r['history']
        if not hist.get('val_mae'):
            continue
        epochs = list(range(1, len(hist['val_mae']) + 1))
        min_mae = min(hist['val_mae'])
        ax.plot(
            epochs, hist['val_mae'],
            label=f"{name} (min={min_mae:.5f})",
            color=colors[i % len(colors)],
            marker=markers[i % len(markers)],
            markersize=4, linewidth=2,
            markevery=max(1, len(epochs) // 15)
        )

    ax.set_xlabel('Epoch', fontsize=12)
    ax.set_ylabel('Validacios MAE', fontsize=12)
    ax.set_title('Modellek Osszehasonlitasa -- Validacios MAE', fontsize=14, fontweight='bold')
    ax.legend(fontsize=9, loc='upper right')
    ax.grid(True, alpha=0.3)

    path = os.path.join(plots_dir, 'mae_comparison.png')
    fig.savefig(path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"  MAE osszehasonlito grafikon mentve: {path}")

    save_plot_script(results_dir)


# ---------------------------------------------------------------------------
# Plot script mentese
# ---------------------------------------------------------------------------

def save_plot_script(results_dir: str):
    """results/plot_training_curves.py -- onallo grafikon generalo."""
    lines = [
        '"""',
        'Tanulasi gorbe grafikonok generalasa a CSV fajlokbol.',
        '',
        'Futtatasa:',
        '    python results/plot_training_curves.py',
        '"""',
        '',
        'import os',
        'import matplotlib',
        "matplotlib.use('Agg')",
        'import matplotlib.pyplot as plt',
        'import pandas as pd',
        '',
        'RESULTS_DIR = os.path.dirname(os.path.abspath(__file__))',
        "PLOTS_DIR   = os.path.join(RESULTS_DIR, 'plots')",
        'os.makedirs(PLOTS_DIR, exist_ok=True)',
        '',
        'MODEL_NAMES = [',
        "    'MLPPricer', 'DeepMLPPricer', 'ResNetPricer',",
        "    'GELUResNetPricer', 'DenseMLPPricer', 'HighwayPricer', 'FINNPricer',",
        ']',
        '',
        "COLORS  = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b', '#e377c2']",
        "MARKERS = ['o', 's', '^', 'D', 'v', 'P', '*']",
        '',
        'results = {}',
        'for name in MODEL_NAMES:',
        "    csv_path = os.path.join(RESULTS_DIR, f'training_history_{name}.csv')",
        '    if os.path.exists(csv_path):',
        '        results[name] = pd.read_csv(csv_path)',
        "        print(f'Betoltve: {csv_path} ({len(results[name])} epoch)')",
        '    else:',
        "        print(f'HIANYZIK: {csv_path}')",
        '',
        '# Egyedi tanulasi gorbe',
        'for name, df in results.items():',
        '    fig, ax = plt.subplots(figsize=(10, 6))',
        "    ax.plot(df['epoch'], df['train_loss'], label='Train MSE', color='steelblue', linewidth=2)",
        "    ax.plot(df['epoch'], df['val_loss'],   label='Val MSE',   color='tomato',    linewidth=2)",
        "    ax.set_xlabel('Epoch', fontsize=12)",
        "    ax.set_ylabel('MSE Loss', fontsize=12)",
        "    ax.set_title(f'{name} Tanulasi Gorbe', fontsize=14, fontweight='bold')",
        '    ax.legend(fontsize=11)',
        '    ax.grid(True, alpha=0.3)',
        "    ax.set_yscale('log')",
        "    fig.savefig(os.path.join(PLOTS_DIR, f'training_curve_{name}.png'), dpi=150, bbox_inches='tight')",
        '    plt.close(fig)',
        '',
        '# Osszehasonlito val_loss',
        'fig, ax = plt.subplots(figsize=(13, 7))',
        'for i, (name, df) in enumerate(results.items()):',
        "    ax.plot(df['epoch'], df['val_loss'],",
        "            label=f\"{name} (min={df['val_loss'].min():.5f})\",",
        '            color=COLORS[i % len(COLORS)], linewidth=2)',
        "ax.set_xlabel('Epoch', fontsize=12)",
        "ax.set_ylabel('Validacios MSE Loss', fontsize=12)",
        "ax.set_title('Modellek Osszehasonlitasa -- Validacios Loss', fontsize=14, fontweight='bold')",
        "ax.legend(fontsize=9, loc='upper right')",
        '    ax.grid(True, alpha=0.3)',
        "ax.set_yscale('log')",
        "fig.savefig(os.path.join(PLOTS_DIR, 'training_curve_comparison.png'), dpi=150, bbox_inches='tight')",
        'plt.close(fig)',
        '',
        "print('Grafikonok generalva:', PLOTS_DIR)",
    ]
    path = os.path.join(results_dir, 'plot_training_curves.py')
    with open(path, 'w', encoding='utf-8') as f:
        f.write('\n'.join(lines) + '\n')
    print(f"  Plot script mentve: {path}")


# ---------------------------------------------------------------------------
# Elemzes generalasa
# ---------------------------------------------------------------------------

def generate_analysis(results: list, results_dir: str):
    """training_analysis.md elkeszitese a tanitasi eredmenyek alapjan."""

    sorted_by_val = sorted(results, key=lambda r: r['min_val_loss'])
    best  = sorted_by_val[0]
    worst = sorted_by_val[-1]
    fastest = min(results, key=lambda r: r['best_epoch'])

    for r in results:
        r['overfit_gap'] = r['final_val_loss'] - r['final_train_loss']

    most_overfit  = max(results, key=lambda r: r['overfit_gap'])
    least_overfit = min(results, key=lambda r: r['overfit_gap'])

    for r in results:
        hist = r['history']['val_loss']
        tail = hist[-10:] if len(hist) >= 10 else hist
        r['tail_std'] = float(np.std(tail))

    most_stable  = min(results, key=lambda r: r['tail_std'])
    least_stable = max(results, key=lambda r: r['tail_std'])

    lines = []
    lines.append("# Neuralis Halo Modellek Tanulasi Gorbe Elemzese\n\n")
    lines.append(f"*Generalva: {time.strftime('%Y-%m-%d %H:%M')}*\n\n")

    lines.append("## 1. Osszefoglalo Tablazat\n\n")
    lines.append("| Modell | Parameterek | Min Val MSE | Best Epoch | Final Train MSE | Final Val MSE | Overfitting res |\n")
    lines.append("|---|---|---|---|---|---|---|\n")
    for r in sorted_by_val:
        lines.append(
            f"| {r['class_name']} | {r['params']:,} | {r['min_val_loss']:.6f} | "
            f"{r['best_epoch']} | {r['final_train_loss']:.6f} | "
            f"{r['final_val_loss']:.6f} | {r['overfit_gap']:+.6f} |\n"
        )
    lines.append("\n")

    lines.append("## 2. Konvergencia Elemzes\n\n")
    lines.append(
        f"A leggyorsabban konvergalo modell a **{fastest['class_name']}**, "
        f"amely mar a **{fastest['best_epoch']}. epochban** erte el legjobb validacios loss erteket "
        f"({fastest['min_val_loss']:.6f}).\n\n"
    )
    lines.append("Konvergencia sorrendje (best epoch szerint):\n\n")
    for r in sorted(results, key=lambda r: r['best_epoch']):
        lines.append(f"- **{r['class_name']}**: {r['best_epoch']}. epoch (val MSE: {r['min_val_loss']:.6f})\n")
    lines.append("\n")

    lines.append("## 3. Vegso Teljesitmeny\n\n")
    lines.append(
        f"A legjobb validacios teljesitmenyt a **{best['class_name']}** erte el, "
        f"minimalis val MSE: **{best['min_val_loss']:.6f}** ({best['best_epoch']}. epoch).\n\n"
    )
    lines.append(
        f"A leggyengebb eredmenyt a **{worst['class_name']}** produkalta: "
        f"val MSE: {worst['min_val_loss']:.6f}.\n\n"
    )
    rel_diff = (worst['min_val_loss'] / best['min_val_loss'] - 1) * 100
    lines.append(
        f"A legjobb es leggyengebb modell kozotti kulonbseg: "
        f"{worst['min_val_loss'] - best['min_val_loss']:.6f} "
        f"({rel_diff:.1f}% relativ elteres).\n\n"
    )

    lines.append("## 4. Tultanulas (Overfitting) Vizsgalata\n\n")
    lines.append(
        "A tultanulas merteket a final train MSE es final val MSE kulonbsege jelzi. "
        "Pozitiv ertek (val > train) overfittingre utal.\n\n"
    )
    lines.append(
        f"- **Legtobb overfitting**: {most_overfit['class_name']} "
        f"(res: {most_overfit['overfit_gap']:+.6f})\n"
    )
    lines.append(
        f"- **Legkevesebb overfitting**: {least_overfit['class_name']} "
        f"(res: {least_overfit['overfit_gap']:+.6f})\n\n"
    )
    lines.append("Overfitting res reszletesen:\n\n")
    for r in sorted(results, key=lambda r: r['overfit_gap'], reverse=True):
        lines.append(f"- {r['class_name']}: {r['overfit_gap']:+.6f}\n")
    lines.append("\n")

    lines.append("## 5. Stabilitas Elemzes\n\n")
    lines.append(
        "A stabilitast a tanulasi gorbe utolso 10 epochjanak szorasaval merjuk "
        "(kisebb szoras = stabilabb konvergencia).\n\n"
    )
    lines.append(
        f"- **Legstabilabb**: {most_stable['class_name']} "
        f"(szoras: {most_stable['tail_std']:.2e})\n"
    )
    lines.append(
        f"- **Legkevesbe stabil**: {least_stable['class_name']} "
        f"(szoras: {least_stable['tail_std']:.2e})\n\n"
    )
    lines.append("Stabilitas sorrendje:\n\n")
    for r in sorted(results, key=lambda r: r['tail_std']):
        lines.append(f"- {r['class_name']}: {r['tail_std']:.2e}\n")
    lines.append("\n")

    lines.append("## 6. Architektura Hatasanak Ertekelse\n\n")

    arch_info = [
        ('MLPPricer',
         "Culkin & Das (2017) baseline. Egyszeru 4x100 neuronos MLP, ReLU aktivacioval. "
         "Kis parameterszam (~30 900) miatt korlatozott kapacitasu, de gyorsan tanul."),
        ('DeepMLPPricer',
         "Della Corte et al. (2023) javitott MLP. LayerNorm es Dropout regularizacio "
         "stabilizalja a tanulast, 4x256 reteg melyebb reprezentaciot tesz lehetove."),
        ('ResNetPricer',
         "Rezidualis kapcsolatokkal (skip connection) ellatott MLP, BatchNorm1d normalizacioval. "
         "A skip connectionok megkonnyitik a gradiens aramlast mely halokban."),
        ('GELUResNetPricer',
         "ResNetPricer GELU aktivacioval -- a GELU simabb, differencialhatobb nemlinearitas, "
         "ami BS arak sima felulethez jobban illeszkedhet. Pre-LN struktura."),
        ('DenseMLPPricer',
         "DenseNet-stilusu MLP: minden reteg az osszes korabbi kimenetet kapja inputkent. "
         "Ez gazdagabb reprezentaciot biztosit, GELU aktivacio es Dropout regularizacioval."),
        ('HighwayPricer',
         "Highway Network: tanulhato gating mechanizmus (transform gate T) donti el, "
         "mennyit 'enged at' az eredeti jel kontra a transzformalt. Gate bias -1.0 init."),
        ('FINNPricer',
         "Finance-Informed NN: ket ag -- egy kis MLP BS-kozelitokent, egy ResNet-ag "
         "korrekcios tagkent. Az osszeadas inductive bias-t visz be a haloba."),
    ]

    for name, desc in arch_info:
        r = next((x for x in results if x['class_name'] == name), None)
        if r is None:
            continue
        lines.append(f"### {name}\n\n")
        lines.append(f"{desc}\n\n")
        lines.append(
            f"Eredmeny: min val MSE = {r['min_val_loss']:.6f}, "
            f"best epoch = {r['best_epoch']}, "
            f"parameterek = {r['params']:,}.\n\n"
        )

    lines.append("## 7. Kovetkeztetesek es Ajanlasok\n\n")
    lines.append("### Modellek rangsorolasa (min val MSE alapjan)\n\n")
    for rank, r in enumerate(sorted_by_val, 1):
        lines.append(
            f"{rank}. **{r['class_name']}** -- val MSE: {r['min_val_loss']:.6f}, "
            f"parameterek: {r['params']:,}\n"
        )
    lines.append("\n")

    lines.append("### Ajanlott modell\n\n")
    lines.append(
        f"**Ajanlott modell: {best['class_name']}**\n\n"
        f"Indoklas: A {best['class_name']} erte el a legjobb validacios MSE-t "
        f"({best['min_val_loss']:.6f}), amelyet a {best['best_epoch']}. epochban ert el. "
        f"Parameterszam: {best['params']:,}, overfitting res: {best['overfit_gap']:+.6f}.\n\n"
    )

    if most_stable['class_name'] != best['class_name']:
        lines.append(
            f"Megjegyzes: Ha a stabilitas prioritas, akkor a **{most_stable['class_name']}** "
            f"javasolt ({most_stable['tail_std']:.2e} szorassal a legstabilabb tanulasi gorbet mutatja).\n\n"
        )

    lines.append(
        "### Altalanos megfigyelesek\n\n"
        "- A rezidualis kapcsolatok (ResNet, GELU ResNet) altalaban gyorsabb konvergenciat\n"
        "  hoznak, mint a sima MLP-k.\n"
        "- A Highway Network tanulhato gating mechanizmusa rugalmasabb, de tobb parametert igenyel.\n"
        "- A FINNPricer ket-agu strukturaja penzugyi inductive bias-t visz be,\n"
        "  ami BS-adatokon elonyes lehet.\n"
        "- A Dropout es LayerNorm regularizacio altalaban csokkenti az overfitting-et.\n"
        "- A BatchNorm1d a ResNetPricer-ben batch-szintu normalizaciot vegez,\n"
        "  ami erzekeny a batch meretre.\n"
    )

    path = os.path.join(results_dir, 'training_analysis.md')
    with open(path, 'w', encoding='utf-8') as f:
        f.writelines(lines)
    print(f"Elemzes mentve: {path}")


# ---------------------------------------------------------------------------
# Foprogrram
# ---------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser(description='Osszes modell beapontitasa es elemzes')
    p.add_argument('--train',        type=str,   default='data/train.parquet')
    p.add_argument('--val',          type=str,   default='data/val.parquet')
    p.add_argument('--results',      type=str,   default='results')
    p.add_argument('--epochs',       type=int,   default=50)
    p.add_argument('--patience',     type=int,   default=15)
    p.add_argument('--batch-size',   type=int,   default=4096)
    p.add_argument('--lr',           type=float, default=1e-3)
    p.add_argument('--weight-decay', type=float, default=1e-4)
    p.add_argument('--device',       type=str,   default='auto')
    p.add_argument('--seed',         type=int,   default=42)
    return p.parse_args()


def main():
    args = parse_args()

    os.makedirs(args.results, exist_ok=True)
    os.makedirs(os.path.join(args.results, 'plots'), exist_ok=True)

    print("=" * 70)
    print("OPCIOS ARAZO MODELLEK OSSZEHASONLITASA")
    print("=" * 70)
    print(f"  Epochok:    {args.epochs} (patience={args.patience})")
    print(f"  Batch:      {args.batch_size}")
    print(f"  LR:         {args.lr}")
    print(f"  Seed:       {args.seed}")
    print(f"  Eredmenyek: {args.results}/")
    print()

    all_results = []
    total_start = time.time()

    for i, (model_key, model_kwargs, class_name) in enumerate(MODEL_CONFIGS, 1):
        print(f"\n[{i}/{len(MODEL_CONFIGS)}] {class_name}")
        print("-" * 70)

        result = train_with_history(
            model_key=model_key,
            model_kwargs=model_kwargs,
            class_name=class_name,
            train_path=args.train,
            val_path=args.val,
            results_dir=args.results,
            batch_size=args.batch_size,
            max_epochs=args.epochs,
            lr=args.lr,
            weight_decay=args.weight_decay,
            patience=args.patience,
            device=args.device,
            seed=args.seed,
        )
        all_results.append(result)

    total_elapsed = time.time() - total_start
    print(f"\nOsszes tanitasi ido: {total_elapsed/60:.1f} perc")

    print("\n" + "=" * 70)
    print("EREDMENYEK MENTESE")
    print("=" * 70)

    df_compare = save_comparison_table(all_results, args.results)
    print()
    print(df_compare.to_string(index=False))
    print()

    print("Grafikonok generalasa...")
    generate_plots(all_results, os.path.join(args.results, 'plots'), args.results)
    print()

    print("Elemzes generalasa...")
    generate_analysis(all_results, args.results)
    print()

    print("=" * 70)
    print("KESZ -- minden fajl a results/ konyvtarban")
    print("=" * 70)


if __name__ == '__main__':
    main()
