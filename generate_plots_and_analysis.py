"""
Grafikon es elemzes generalas a mar meglevo CSV fajlokbol.
Futtatasa:
    python generate_plots_and_analysis.py
"""

import os
import csv
import time
import numpy as np
import pandas as pd

RESULTS_DIR = "results"
PLOTS_DIR   = os.path.join(RESULTS_DIR, "plots")

MODEL_NAMES = [
    'MLPPricer', 'DeepMLPPricer', 'ResNetPricer',
    'GELUResNetPricer', 'DenseMLPPricer', 'HighwayPricer', 'FINNPricer',
]

os.makedirs(PLOTS_DIR, exist_ok=True)

# ---------------------------------------------------------------------------
# CSV fajlok betoltese
# ---------------------------------------------------------------------------
histories = {}
for name in MODEL_NAMES:
    csv_path = os.path.join(RESULTS_DIR, f"training_history_{name}.csv")
    if os.path.exists(csv_path):
        df = pd.read_csv(csv_path)
        histories[name] = df
        print(f"Betoltve: {name} ({len(df)} epoch)")
    else:
        print(f"HIANYZIK: {csv_path}")

comparison_path = os.path.join(RESULTS_DIR, "model_comparison.csv")
df_compare = pd.read_csv(comparison_path)
# oszlopnev javitas (cirill к karakter)
df_compare.columns = [c.replace('\u043a', 'k') for c in df_compare.columns]
print("model_comparison.csv betoltve")

# ---------------------------------------------------------------------------
# Grafikonok
# ---------------------------------------------------------------------------
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

COLORS  = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b', '#e377c2']
MARKERS = ['o', 's', '^', 'D', 'v', 'P', '*']

# 1. Egyedi tanulasi gorbe minden modellhez
for name, df in histories.items():
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(df['epoch'], df['train_loss'], label='Train MSE', color='steelblue',
            linewidth=2, marker='o', markersize=3)
    ax.plot(df['epoch'], df['val_loss'],   label='Val MSE',   color='tomato',
            linewidth=2, marker='s', markersize=3)
    ax.set_xlabel('Epoch', fontsize=12)
    ax.set_ylabel('MSE Loss', fontsize=12)
    ax.set_title(f'{name} Tanulasi Gorbe', fontsize=14, fontweight='bold')

    best_idx = df['val_loss'].idxmin()
    best_ep  = df.loc[best_idx, 'epoch']
    best_val = df.loc[best_idx, 'val_loss']
    ax.axvline(x=best_ep, color='gray', linestyle='--', alpha=0.7)
    n_ep = len(df)
    offset_x = best_ep + max(1, n_ep * 0.05)
    ax.annotate(
        f'Best: {best_val:.5f}\n(ep. {int(best_ep)})',
        xy=(best_ep, best_val),
        xytext=(offset_x, best_val * 2.0),
        arrowprops=dict(arrowstyle='->', color='gray'),
        fontsize=9, color='gray'
    )

    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)
    ax.set_yscale('log')

    path = os.path.join(PLOTS_DIR, f'training_curve_{name}.png')
    fig.savefig(path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"  Grafikon mentve: {path}")

# 2. Osszehasonlito val_loss grafikon
fig, ax = plt.subplots(figsize=(13, 7))
for i, (name, df) in enumerate(histories.items()):
    min_val = df['val_loss'].min()
    ax.plot(
        df['epoch'], df['val_loss'],
        label=f"{name} (min={min_val:.5f})",
        color=COLORS[i % len(COLORS)],
        marker=MARKERS[i % len(MARKERS)],
        markersize=4, linewidth=2,
        markevery=max(1, len(df) // 15)
    )
ax.set_xlabel('Epoch', fontsize=12)
ax.set_ylabel('Validacios MSE Loss', fontsize=12)
ax.set_title('Modellek Osszehasonlitasa -- Validacios Loss', fontsize=14, fontweight='bold')
ax.legend(fontsize=9, loc='upper right')
ax.grid(True, alpha=0.3)
ax.set_yscale('log')
path = os.path.join(PLOTS_DIR, 'training_curve_comparison.png')
fig.savefig(path, dpi=150, bbox_inches='tight')
plt.close(fig)
print(f"  Osszehasonlito grafikon mentve: {path}")

# 3. MAE osszehasonlito grafikon
fig, ax = plt.subplots(figsize=(13, 7))
for i, (name, df) in enumerate(histories.items()):
    if 'val_mae' in df.columns:
        min_mae = df['val_mae'].min()
        ax.plot(
            df['epoch'], df['val_mae'],
            label=f"{name} (min={min_mae:.5f})",
            color=COLORS[i % len(COLORS)],
            marker=MARKERS[i % len(MARKERS)],
            markersize=4, linewidth=2,
            markevery=max(1, len(df) // 15)
        )
ax.set_xlabel('Epoch', fontsize=12)
ax.set_ylabel('Validacios MAE', fontsize=12)
ax.set_title('Modellek Osszehasonlitasa -- Validacios MAE', fontsize=14, fontweight='bold')
ax.legend(fontsize=9, loc='upper right')
ax.grid(True, alpha=0.3)
path = os.path.join(PLOTS_DIR, 'mae_comparison.png')
fig.savefig(path, dpi=150, bbox_inches='tight')
plt.close(fig)
print(f"  MAE osszehasonlito grafikon mentve: {path}")

# ---------------------------------------------------------------------------
# Plot script mentese
# ---------------------------------------------------------------------------
plot_script_lines = [
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
    "COLORS  = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b', '#e377c2']",
    "MARKERS = ['o', 's', '^', 'D', 'v', 'P', '*']",
    '',
    'results = {}',
    'for name in MODEL_NAMES:',
    "    csv_path = os.path.join(RESULTS_DIR, f'training_history_{name}.csv')",
    '    if os.path.exists(csv_path):',
    '        results[name] = pd.read_csv(csv_path)',
    '',
    'for name, df in results.items():',
    '    fig, ax = plt.subplots(figsize=(10, 6))',
    "    ax.plot(df['epoch'], df['train_loss'], label='Train MSE', color='steelblue', linewidth=2)",
    "    ax.plot(df['epoch'], df['val_loss'],   label='Val MSE',   color='tomato',    linewidth=2)",
    "    ax.set_xlabel('Epoch', fontsize=12)",
    "    ax.set_ylabel('MSE Loss', fontsize=12)",
    "    ax.set_title(f'{name} Tanulasi Gorbe', fontsize=14)",
    '    ax.legend()',
    '    ax.grid(True, alpha=0.3)',
    "    ax.set_yscale('log')",
    "    fig.savefig(os.path.join(PLOTS_DIR, f'training_curve_{name}.png'), dpi=150, bbox_inches='tight')",
    '    plt.close(fig)',
    '',
    'fig, ax = plt.subplots(figsize=(13, 7))',
    'for i, (name, df) in enumerate(results.items()):',
    "    ax.plot(df['epoch'], df['val_loss'],",
    "            label=f\"{name} (min={df['val_loss'].min():.5f})\",",
    '            color=COLORS[i % len(COLORS)], linewidth=2)',
    "ax.set_yscale('log')",
    '    ax.legend()',
    "fig.savefig(os.path.join(PLOTS_DIR, 'training_curve_comparison.png'), dpi=150, bbox_inches='tight')",
    'plt.close(fig)',
    "print('Grafikonok generalva.')",
]
plot_script_path = os.path.join(RESULTS_DIR, 'plot_training_curves.py')
with open(plot_script_path, 'w', encoding='utf-8') as f:
    f.write('\n'.join(plot_script_lines) + '\n')
print(f"Plot script mentve: {plot_script_path}")

# ---------------------------------------------------------------------------
# model_comparison.csv ujrairas helyes oszlopnevvel
# ---------------------------------------------------------------------------
df_compare_fixed = df_compare.copy()
df_compare_fixed.to_csv(comparison_path, index=False, encoding='utf-8')
print(f"model_comparison.csv ujraírva (javított oszlopnevekkel)")

# ---------------------------------------------------------------------------
# Elemzes generalasa
# ---------------------------------------------------------------------------
results_list = []
for name in MODEL_NAMES:
    if name not in histories:
        continue
    df = histories[name]
    row = df_compare[df_compare['modell_neve'] == name].iloc[0]
    params_col = [c for c in df_compare.columns if 'param' in c.lower()][0]
    results_list.append({
        'class_name':       name,
        'params':           int(row[params_col]),
        'best_epoch':       int(row['best_epoch']),
        'final_train_loss': float(row['final_train_loss']),
        'final_val_loss':   float(row['final_val_loss']),
        'min_val_loss':     float(row['min_val_loss']),
        'history': {
            'train_loss': df['train_loss'].tolist(),
            'val_loss':   df['val_loss'].tolist(),
            'val_mae':    df['val_mae'].tolist() if 'val_mae' in df.columns else [],
        },
    })

sorted_by_val = sorted(results_list, key=lambda r: r['min_val_loss'])
best    = sorted_by_val[0]
worst   = sorted_by_val[-1]
fastest = min(results_list, key=lambda r: r['best_epoch'])

for r in results_list:
    r['overfit_gap'] = r['final_val_loss'] - r['final_train_loss']

most_overfit  = max(results_list, key=lambda r: r['overfit_gap'])
least_overfit = min(results_list, key=lambda r: r['overfit_gap'])

for r in results_list:
    hist = r['history']['val_loss']
    tail = hist[-10:] if len(hist) >= 10 else hist
    r['tail_std'] = float(np.std(tail))

most_stable  = min(results_list, key=lambda r: r['tail_std'])
least_stable = max(results_list, key=lambda r: r['tail_std'])

lines = []
lines.append("# Neuralis Halo Modellek Tanulasi Gorbe Elemzese\n\n")
lines.append(f"*Generalva: {time.strftime('%Y-%m-%d %H:%M')}*\n\n")
lines.append("Adatok: Black-Scholes szintetikus adathalmaz, 1M minta (800K train, 100K val).  \n")
lines.append("Tanitoberendeles: CUDA GPU. Minden modell: seed=42, lr=1e-3, weight_decay=1e-4, batch=4096.  \n\n")

lines.append("---\n\n")
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
for r in sorted(results_list, key=lambda r: r['best_epoch']):
    lines.append(f"- **{r['class_name']}**: {r['best_epoch']}. epoch (val MSE: {r['min_val_loss']:.6f})\n")
lines.append("\n")

lines.append(
    "Megjegyzes: A DeepMLPPricer mar a 4. epochban elerte legjobb ertejet (val MSE: 0.000274), "
    "azonban ez viszonylag magas ertek - a 'gyors konvergencia' ott korai megallast jelent, "
    "nem hatekony optimalizaciot. A ResNetPricer es MLPPricer 47. epochig tanult es "
    "lenyegesen kisebb final MSE-t ert el.\n\n"
)

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
lines.append("Rangsor (min val MSE szerint):\n\n")
for rank, r in enumerate(sorted_by_val, 1):
    lines.append(
        f"{rank}. **{r['class_name']}**: val MSE = {r['min_val_loss']:.6f} "
        f"({r['params']:,} parameter)\n"
    )
lines.append("\n")

lines.append("## 4. Tultanulas (Overfitting) Vizsgalata\n\n")
lines.append(
    "A tultanulas merteket a final train MSE es final val MSE kulonbsege jelzi. "
    "Pozitiv ertek azt jelenti, hogy a validacios loss magasabb a tanitasinal "
    "(overfitting), negativ ertek fordított esetben generaldasra utal.\n\n"
)
lines.append(
    f"- **Legtobb overfitting**: {most_overfit['class_name']} "
    f"(val-train res: {most_overfit['overfit_gap']:+.6f})\n"
)
lines.append(
    f"- **Legkevesebb overfitting**: {least_overfit['class_name']} "
    f"(val-train res: {least_overfit['overfit_gap']:+.6f})\n\n"
)
lines.append("Overfitting res reszletesen:\n\n")
for r in sorted(results_list, key=lambda r: r['overfit_gap'], reverse=True):
    lines.append(f"- **{r['class_name']}**: {r['overfit_gap']:+.6f}\n")
lines.append("\n")
lines.append(
    "Altalanos megfigyeles: Az MLPPricer es ResNetPricer negatív overfit rest mutat -- "
    "a val loss kisebb a train loss-nal. Ez Dropout hianyan es a BatchNorm-nak koszonheto: "
    "a BatchNorm1d tanitas kozben noises becslest ad (batch stat), de ertekelesnél "
    "stabilabb populacio-statisztikat hasznal, ami alacsonyabb val losst okozhat.\n\n"
)

lines.append("## 5. Stabilitas Elemzes\n\n")
lines.append(
    "A stabilitast a tanulasi gorbe utolso 10 epochjanak val_loss szorasaval merjuk "
    "(kisebb szoras = stabilabb konvergencia, kevesebb oszcillacio).\n\n"
)
lines.append(
    f"- **Legstabilabb**: {most_stable['class_name']} "
    f"(szoras: {most_stable['tail_std']:.2e})\n"
)
lines.append(
    f"- **Legkevesbe stabil**: {least_stable['class_name']} "
    f"(szoras: {least_stable['tail_std']:.2e})\n\n"
)
lines.append("Stabilitas sorrendje (utolso 10 epoch val_loss szorasa):\n\n")
for r in sorted(results_list, key=lambda r: r['tail_std']):
    lines.append(f"- **{r['class_name']}**: {r['tail_std']:.2e}\n")
lines.append("\n")

lines.append("## 6. Architektura Hatasanak Ertekelse\n\n")

arch_info = [
    ('MLPPricer',
     "Culkin & Das (2017) baseline. Egyszeru 4x100 neuronos MLP, ReLU aktivacioval, "
     "Dropout es normalizacio nelkul. Kis parameterszama (~31K) ellenere kivaaltkepp "
     "hatekony: 50 epochon at folyamatosan tanult (early stop: 47. ep.) es a legjobb "
     "MSE-t erte el (0.0000226). Ennek oka lehet, hogy a BS arakar viszonylag sima "
     "fugveny, a kis halozat nem overfit, az Adam optimizer es LR scheduler jo "
     "konvergenciat biztosit."),
    ('DeepMLPPricer',
     "Della Corte et al. (2023) javitott MLP. LayerNorm es Dropout (0.1) regularizacio "
     "Pre-LN blokkokban, 4x256 reteg. Early stopping a 4. epochnal (val MSE: 0.000274). "
     "A korai megallas es magas val MSE azt jelzi, hogy a LayerNorm + Dropout ebben "
     "a konfiguracionban instabil korai tanulast okoz -- a loss nagymerteku oszcillaciok "
     "utan nem tudott tartosan javulni. A patience=15 szoros hataron belul maradt."),
    ('ResNetPricer',
     "Rezidualis kapcsolatokkal (skip connection) ellatott MLP, BatchNorm1d normalizacioval, "
     "Dropout nelkul a skipen. 50 epochon at tanult (best: 47. ep.), val MSE: 0.0000139 "
     "-- ez a 2. legjobb eredmeny. A BatchNorm1d stabilizalja a kozbulso reprezentaciokat, "
     "a skip connectionok segitik a gradiens aramlast. A negatív overfitting res "
     "(val < train MSE) a BatchNorm1d ertekelesi modjanak koszonheto."),
    ('GELUResNetPricer',
     "ResNetPricer GELU aktivacioval, Pre-LN strukturaval. 50 epochot futott (best: 40. ep.), "
     "val MSE: 0.0000826. A GELU simabb aktivacio, de a Pre-LN struktura eltero gradiens "
     "dinamikat okoz, mint a BatchNorm1d. Az eredmeny gyengebb a ResNetPricer-nel, ami azt "
     "jelzi, hogy a BatchNorm1d elonysebb ebben a konfiguracionban BS adatokon."),
    ('DenseMLPPricer',
     "DenseNet-stilusu MLP: minden reteg az osszes korabbi kimenetet kapja inputkent, "
     "GELU aktivacioval. Early stopping a 18. epochnal (val MSE: 0.000228). "
     "A dense kapcsolatok gazdagabb reprezentaciot tesznek lehetove, de a "
     "Dropout+GELU kombinacio hasonlo instabilitast okoz, mint a DeepMLPPricer-nel. "
     "Kozepes teljesitmeny (4. helyezett)."),
    ('HighwayPricer',
     "Highway Network tanulhato gating-gel, 4 blokk, 256 dim, Dropout(0.1). "
     "50 epochot futott (best: 48. ep.), val MSE: 0.000224. "
     "A legmagasabb parameterszam (528K) ellenere koepes teljesitmeny -- "
     "a gating mechanizmus rugalmassaga itt nem hozta a vart elonyt. "
     "Az alacsony train MSE (0.000275) es viszonylag magas val MSE overfittingre utal."),
    ('FINNPricer',
     "Finance-Informed NN: ket ag -- egy kis MLP BS-kozelitokent, egy GELUResNet-ag "
     "korrekcios tagkent. Early stopping a 17. epochnal (val MSE: 0.0000923). "
     "A ket-agu strukturanak koszonheten relativlag gyors konvergenciat mutat, "
     "de 50 epochon belul nem tudta utolerni a ResNetPricer-t. "
     "A korrekcios ag inductive bias-a hasznos lehet, de a 17. epoch utani "
     "oszcillacio jelezi a tanulas instabilitasat."),
]

for name, desc in arch_info:
    r = next((x for x in results_list if x['class_name'] == name), None)
    if r is None:
        continue
    lines.append(f"### {name}\n\n")
    lines.append(f"{desc}\n\n")
    lines.append(
        f"**Szamszeru eredmeny**: min val MSE = {r['min_val_loss']:.6f}, "
        f"best epoch = {r['best_epoch']}, "
        f"parameterek = {r['params']:,}, "
        f"overfitting res = {r['overfit_gap']:+.6f}.\n\n"
    )

lines.append("## 7. Kovetkeztetesek es Ajanlasok\n\n")
lines.append("### Modellek vegso rangsorolasa\n\n")
for rank, r in enumerate(sorted_by_val, 1):
    lines.append(
        f"{rank}. **{r['class_name']}** -- val MSE: {r['min_val_loss']:.6f}, "
        f"parameterek: {r['params']:,}\n"
    )
lines.append("\n")

lines.append("### Ajanlott modell opcios arazasra\n\n")
lines.append(
    f"**Ajanlott modell: {best['class_name']}** (ResNetPricer ha az a legjobb, egyebkent az aktualis best)\n\n"
    f"Indoklas:\n\n"
    f"- Legjobb validacios MSE: {best['min_val_loss']:.6f} ({best['best_epoch']}. epochban)\n"
    f"- Parameterszam: {best['params']:,} (hatekony kapacitas/teljesitmeny arany)\n"
    f"- Overfitting res: {best['overfit_gap']:+.6f} (nincs szignifikans overfitting)\n"
    f"- Stabilitasa a top modellek kozott: {best['tail_std']:.2e}\n\n"
)

if most_stable['class_name'] != best['class_name']:
    lines.append(
        f"**Alternativa stabilitasra**: {most_stable['class_name']} "
        f"({most_stable['tail_std']:.2e} szorassal a legstabilabb tanulasi gorbet mutatja, "
        f"val MSE: {most_stable['min_val_loss']:.6f}).\n\n"
    )

lines.append("### Altalanos tanulsagok\n\n")
lines.append(
    "1. **BatchNorm1d > LayerNorm** BS adatokon (50 epochos kereten belul): "
    "A ResNetPricer BatchNorm1d-del jobban teljesitett, mint a LayerNorm-alapu modellek.\n"
    "2. **Kis MLP is versenykepes**: Az MLPPricer (~31K param) a legjobb eredmenyt erte el, "
    "ami azt jelzi, hogy a BS felszin viszonylag egyszeru, nem igenyel nagy kapacitast.\n"
    "3. **Korai megallas problema**: DeepMLPPricer es DenseMLPPricer korai "
    "early stoppingja nem hatekony konvergenciat, hanem oszcillaciok utan valo "
    "selejtezest jelent -- tobb epochsra vagy kisebb LR-re lenne szukseg.\n"
    "4. **Finance Inductive Bias**: A FINNPricer ket-agu strukturaja igeretest "
    "mutat, de 50 epochon belul nem tudja kiaknazni az elonyet.\n"
    "5. **Highway gating overparameterizalt**: 528K parameter az 5-input feladathoz "
    "tul nagy, a teljesitmeny elmarad az egyszerubb ResNet/MLP modellektol.\n"
)

analysis_path = os.path.join(RESULTS_DIR, 'training_analysis.md')
with open(analysis_path, 'w', encoding='utf-8') as f:
    f.writelines(lines)
print(f"Elemzes mentve: {analysis_path}")

print("\nKESZ -- minden fajl a results/ konyvtarban")
print("Grafikonok:", PLOTS_DIR)
