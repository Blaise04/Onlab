import pandas as pd
import matplotlib.pyplot as plt
import os

df = pd.read_csv('results/training_curves.csv')
os.makedirs('results/plots', exist_ok=True)

MODEL_DISPLAY = {
    'mlp':          'MLPPricer',
    'deep_mlp':     'DeepMLPPricer',
    'resnet':       'ResNetPricer',
    'gelu_resnet':  'GELUResNetPricer',
    'dense_mlp':    'DenseMLPPricer',
    'highway':      'HighwayPricer',
    'finn':         'FINNPricer',
    'resnet_phys':  'ResNetPricer (physics)',
}

# Egyedi ábrák
for model_name in df['model'].unique():
    mdf = df[df['model'] == model_name].copy()
    display_name = MODEL_DISPLAY.get(model_name, model_name)
    best_rows = mdf[mdf['is_best'] == 1]
    best_epoch = best_rows['epoch'].values[0] if len(best_rows) > 0 else None

    fig, ax1 = plt.subplots(figsize=(10, 6))

    l1, = ax1.plot(mdf['epoch'], mdf['train_loss'], color='#1f77b4', label='train_loss')
    l2, = ax1.plot(mdf['epoch'], mdf['val_loss'],   color='#ff7f0e', label='val_loss')
    ax1.set_yscale('log')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss (log scale)')
    ax1.grid(True, alpha=0.3)

    ax2 = ax1.twinx()
    lr_col = mdf['lr'].ffill()
    l3, = ax2.plot(mdf['epoch'], lr_col, color='#2ca02c', linestyle='--', alpha=0.7, label='lr')
    ax2.set_ylabel('Learning Rate', color='#2ca02c')
    ax2.tick_params(axis='y', labelcolor='#2ca02c')

    lines = [l1, l2, l3]
    labels = [l.get_label() for l in lines]
    if best_epoch is not None:
        vl = ax1.axvline(x=best_epoch, color='red', linestyle=':', alpha=0.8, label=f'best epoch ({best_epoch})')
        lines.append(vl)
        labels.append(f'best epoch ({best_epoch})')

    ax1.legend(lines, labels, loc='upper right')
    ax1.set_title(f'{display_name} — Training Curves')
    plt.tight_layout()
    plt.savefig(f'results/plots/{model_name}_training_curves.png', dpi=150)
    plt.close()
    print(f"Mentve: results/plots/{model_name}_training_curves.png")

# Összesítő ábra
fig, ax = plt.subplots(figsize=(12, 7))
colors = plt.cm.tab10.colors

for i, model_name in enumerate(df['model'].unique()):
    mdf = df[df['model'] == model_name]
    display_name = MODEL_DISPLAY.get(model_name, model_name)
    color = colors[i % len(colors)]
    ax.plot(mdf['epoch'], mdf['val_loss'], color=color, label=display_name)
    best_rows = mdf[mdf['is_best'] == 1]
    if len(best_rows) > 0:
        ax.scatter(best_rows['epoch'], best_rows['val_loss'], color=color, s=40, zorder=5)

ax.set_yscale('log')
ax.set_xlabel('Epoch')
ax.set_ylabel('Validation Loss (log scale)')
ax.set_title('Validation Loss Comparison — All Models')
ax.legend(loc='upper right')
ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('results/plots/all_models_val_loss_comparison.png', dpi=150)
plt.close()
print("Mentve: results/plots/all_models_val_loss_comparison.png")
