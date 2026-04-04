"""
Egységes osszehasonlito tanitas -- mind a 7 neuralis halo architektura.

Minden modell azonos hiperparameterekkel tanitodik (fair osszehasonlitas),
az epoch-szintu adatok CSV es JSON formatumban kerulnek a results/ mappaba.

Mentett fajlok (modellenként):
    results/training_history_{ModelClassName}.csv   -- epoch, train_loss, val_loss, lr
    results/training_history_{ModelClassName}.json  -- teljes history dict

Használat:
    python train_all.py
"""

import json
import os
import random
import time

import numpy as np
import pandas as pd
import torch

from src.model import get_model, count_parameters
from src.train import train_model, DEFAULT_FEATURE_COLS, DEFAULT_TARGET_COL


# ---------------------------------------------------------------------------
# Veletlen szam mag rogzites (masolva a train.py-bol)
# ---------------------------------------------------------------------------

def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


# ---------------------------------------------------------------------------
# Egységes hiperparaméterek (fair összehasonlítás)  [FONTOS: ne módosítsd]
# ---------------------------------------------------------------------------

SEED         = 42
EPOCHS       = 200
BATCH_SIZE   = 4096
LR           = 1e-3
WEIGHT_DECAY = 1e-4
PATIENCE     = 10
AUGMENT_PUT  = False
PHYSICS_LOSS = False
DEVICE       = 'auto'
TRAIN_PATH   = 'data/train.parquet'
VAL_PATH     = 'data/val.parquet'
OUTPUT_DIR   = 'models/'

# Modell neve -> Python osztálynév leképezés
MODEL_CLASS_NAMES = {
    'mlp':         'MLPPricer',
    'deep_mlp':    'DeepMLPPricer',
    'resnet':      'ResNetPricer',
    'gelu_resnet': 'GELUResNetPricer',
    'dense_mlp':   'DenseMLPPricer',
    'highway':     'HighwayPricer',
    'finn':        'FINNPricer',
}

# Modell-specifikus kwargs (másolva a train.py build_model_kwargs() defaults dict-jéből,
# augment_put=False -> input_dim=4)
INPUT_DIM = 4
MODEL_DEFAULTS = {
    'mlp':         {'input_dim': INPUT_DIM, 'hidden_dim': 100, 'n_layers': 4},
    'deep_mlp':    {'input_dim': INPUT_DIM, 'hidden_dim': 256, 'n_layers': 4, 'dropout': 0.1},
    'resnet':      {'input_dim': INPUT_DIM, 'hidden_dim': 256, 'n_blocks': 3, 'dropout': 0.1},
    'gelu_resnet': {'input_dim': INPUT_DIM, 'hidden_dim': 256, 'n_blocks': 3, 'dropout': 0.1},
    'dense_mlp':   {'input_dim': INPUT_DIM, 'hidden_dim': 128, 'n_layers': 4, 'dropout': 0.1},
    'highway':     {'input_dim': INPUT_DIM, 'hidden_dim': 256, 'n_blocks': 4, 'dropout': 0.1},
    'finn':        {'input_dim': INPUT_DIM, 'approx_dim': 64, 'resnet_dim': 256,
                    'n_blocks': 3, 'dropout': 0.1},
}

# Tanítandó modellek sorrendben
ALL_MODELS = ['mlp', 'deep_mlp', 'resnet', 'gelu_resnet', 'dense_mlp', 'highway', 'finn']


# ---------------------------------------------------------------------------
# Epoch adatok mentése
# ---------------------------------------------------------------------------

def save_history(history: dict, model_class_name: str):
    """
    Epoch-szintű tanítási history mentése CSV és JSON formátumban.

    Args:
        history          : train_model() által visszaadott history dict
        model_class_name : pl. 'MLPPricer', 'ResNetPricer', stb.
    """
    os.makedirs('results', exist_ok=True)

    # CSV: epoch, train_loss, val_loss, lr sorok
    n_epochs = len(history['train_loss'])
    rows = []
    for i in range(n_epochs):
        rows.append({
            'epoch':      i + 1,
            'train_loss': history['train_loss'][i],
            'val_loss':   history['val_loss'][i],
            'lr':         history['lr'][i],
        })
    df = pd.DataFrame(rows, columns=['epoch', 'train_loss', 'val_loss', 'lr'])
    csv_path = os.path.join('results', f'training_history_{model_class_name}.csv')
    df.to_csv(csv_path, index=False)
    print(f"  CSV mentve:  {csv_path}")

    # JSON: teljes history dict (best_epoch, best_val_loss is benne van)
    json_path = os.path.join('results', f'training_history_{model_class_name}.json')
    with open(json_path, 'w', encoding='utf-8') as f:
        json.dump(history, f, indent=2, ensure_ascii=False)
    print(f"  JSON mentve: {json_path}")


# ---------------------------------------------------------------------------
# Fő szkript
# ---------------------------------------------------------------------------

def main():
    # Seed beállítása a szkript legelején
    set_seed(SEED)

    print("=" * 70)
    print("Összehasonlító neurális háló tanítás — mind a 7 architektúra")
    print("=" * 70)
    print(f"  Seed:           {SEED}")
    print(f"  Epochok:        {EPOCHS}")
    print(f"  Batch méret:    {BATCH_SIZE}")
    print(f"  LR:             {LR}")
    print(f"  Weight decay:   {WEIGHT_DECAY}")
    print(f"  Patience:       {PATIENCE}")
    print(f"  Augment put:    {AUGMENT_PUT}")
    print(f"  Physics loss:   {PHYSICS_LOSS}")
    print(f"  Eszköz:         {DEVICE}")
    print(f"  Tanítóhalmaz:   {TRAIN_PATH}")
    print(f"  Val. halmaz:    {VAL_PATH}")
    print(f"  Kimeneti mappa: {OUTPUT_DIR}")
    print()

    # Osszefoglalo lista: (modell_neve, osztálynev, best_epoch, best_val_loss,
    #                      param_szam, eltelt_masodperc)
    osszefoglalo = []
    teljes_kezdet = time.time()

    for modell_neve in ALL_MODELS:
        model_class_name = MODEL_CLASS_NAMES[modell_neve]
        model_kwargs = MODEL_DEFAULTS[modell_neve].copy()

        print()
        print("=" * 70)
        print(f"  Modell: {modell_neve}  ({model_class_name})")
        print(f"  Kwargs: {model_kwargs}")
        print("=" * 70)

        # Seed újrainicializálás minden modell előtt -> reprodukálható összehasonlítás
        set_seed(SEED)

        model = get_model(modell_neve, **model_kwargs)
        param_szam = count_parameters(model)
        print(f"  Tanítható paraméterek: {param_szam:,}")
        print()

        modell_kezdet = time.time()

        history = train_model(
            model=model,
            train_path=TRAIN_PATH,
            val_path=VAL_PATH,
            output_dir=OUTPUT_DIR,
            model_name=modell_neve,
            model_class=modell_neve,
            model_kwargs=model_kwargs,
            feature_cols=DEFAULT_FEATURE_COLS,
            target_col=DEFAULT_TARGET_COL,
            batch_size=BATCH_SIZE,
            max_epochs=EPOCHS,
            lr=LR,
            weight_decay=WEIGHT_DECAY,
            patience=PATIENCE,
            device=DEVICE,
            augment_put=AUGMENT_PUT,
            physics_loss=PHYSICS_LOSS,
        )

        eltelt = time.time() - modell_kezdet

        # Epoch adatok mentése
        print()
        print(f"  Eredmények mentése ({model_class_name})...")
        save_history(history, model_class_name)

        best_epoch    = history.get('best_epoch', -1)
        best_val_loss = history.get('best_val_loss', float('nan'))
        osszefoglalo.append(
            (modell_neve, model_class_name, best_epoch, best_val_loss, param_szam, eltelt)
        )

        print()
        print(f"  [{modell_neve}] Kész — legjobb epoch: {best_epoch}, "
              f"val loss: {best_val_loss:.6f}, idő: {eltelt:.1f}s")

    # ---------------------------------------------------------------------------
    # Összesítő táblázat a konzolra
    # ---------------------------------------------------------------------------
    teljes_ido = time.time() - teljes_kezdet

    print()
    print("=" * 70)
    print("OSSZESITO EREDMENYEK")
    print("=" * 70)
    fejlec = (
        f"{'Modell':<14} "
        f"{'Osztalynev':<20} "
        f"{'Param':>10} "
        f"{'Best epoch':>10} "
        f"{'Best val loss':>14} "
        f"{'Ido (s)':>9}"
    )
    print(fejlec)
    print("-" * 80)
    for (mnev, mcls, bepoch, bvl, nparam, ido) in osszefoglalo:
        sor = (
            f"{mnev:<14} "
            f"{mcls:<20} "
            f"{nparam:>10,} "
            f"{bepoch:>10} "
            f"{bvl:>14.6f} "
            f"{ido:>9.1f}"
        )
        print(sor)
    print("-" * 80)
    print(f"Összes eltelt idő: {teljes_ido:.1f}s  ({teljes_ido / 60:.1f} perc)")
    print()

    # Legjobb modell kiemelese
    legjobb = min(osszefoglalo, key=lambda x: x[3])
    print(f"Legjobb modell: {legjobb[0]} ({legjobb[1]})  --  val loss: {legjobb[3]:.6f}")
    print()
    print("Mentett fajlok (results/ mappa):")
    for (_, mcls, _, _, _, _) in osszefoglalo:
        print(f"  results/training_history_{mcls}.csv")
        print(f"  results/training_history_{mcls}.json")
    print()
    print("Tanítás sikeresen befejezve.")


if __name__ == '__main__':
    main()
