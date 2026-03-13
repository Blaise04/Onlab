"""
Az összes modell egymás utáni betanítása és eredmények naplózása.

Használat:
    python train_all.py
    python train_all.py --epochs 200 --patience 20 --output models/
"""

import argparse
import json
import os
import random
import time

import numpy as np
import torch

from src.model import get_model, count_parameters
from src.train import (train_model, DEFAULT_FEATURE_COLS, DEFAULT_TARGET_COL)


# Modellek konfigurációja: (cli_neve, model_kwargs, checkpoint_neve)
MODEL_CONFIGS = [
    ('mlp',         {'input_dim': 5, 'hidden_dim': 100, 'n_layers': 4},                          'mlp'),
    ('deep_mlp',    {'input_dim': 5, 'hidden_dim': 256, 'n_layers': 4, 'dropout': 0.1},          'deep_mlp'),
    ('resnet',      {'input_dim': 5, 'hidden_dim': 256, 'n_blocks': 3, 'dropout': 0.1},          'resnet'),
    ('gelu_resnet', {'input_dim': 5, 'hidden_dim': 256, 'n_blocks': 3, 'dropout': 0.1},          'gelu_resnet'),
    ('dense_mlp',   {'input_dim': 5, 'hidden_dim': 128, 'n_layers': 4, 'dropout': 0.1},          'dense_mlp'),
    ('highway',     {'input_dim': 5, 'hidden_dim': 256, 'n_blocks': 4, 'dropout': 0.1},          'highway'),
    ('finn',        {'input_dim': 5, 'approx_dim': 64, 'resnet_dim': 256,
                     'n_blocks': 3, 'dropout': 0.1},                                             'finn'),
    # Physics-informed resnet (azonos architektúra, physics loss-szal)
    ('resnet',      {'input_dim': 5, 'hidden_dim': 256, 'n_blocks': 3, 'dropout': 0.1},          'resnet_phys'),
]


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def parse_args():
    p = argparse.ArgumentParser(description='Összes modell betanítása')
    p.add_argument('--train',          type=str,   default='data/train.parquet')
    p.add_argument('--val',            type=str,   default='data/val.parquet')
    p.add_argument('--output',         type=str,   default='models/')
    p.add_argument('--epochs',         type=int,   default=200)
    p.add_argument('--patience',       type=int,   default=20)
    p.add_argument('--batch-size',     type=int,   default=4096)
    p.add_argument('--lr',             type=float, default=1e-3)
    p.add_argument('--weight-decay',   type=float, default=1e-4)
    p.add_argument('--device',         type=str,   default='auto')
    p.add_argument('--seed',           type=int,   default=42)
    p.add_argument('--physics-lambda', type=float, default=0.1)
    p.add_argument('--skip-existing',  action='store_true',
                   help='Már létező checkpointokat kihagyja')
    return p.parse_args()


def main():
    args = parse_args()
    set_seed(args.seed)
    os.makedirs(args.output, exist_ok=True)

    summary = {}
    total_start = time.time()

    print("=" * 70)
    print("ÖSSZES MODELL BETANÍTÁSA")
    print("=" * 70)
    print(f"  Epochok:  {args.epochs} (patience={args.patience})")
    print(f"  Batch:    {args.batch_size}")
    print(f"  LR:       {args.lr}")
    print(f"  Eszköz:   {args.device}")
    print(f"  Seed:     {args.seed}")
    print()

    for i, (model_name, model_kwargs, checkpoint_name) in enumerate(MODEL_CONFIGS, 1):
        checkpoint_path = os.path.join(args.output, f"{checkpoint_name}_best.pt")
        physics = (checkpoint_name == 'resnet_phys')

        print(f"\n[{i}/{len(MODEL_CONFIGS)}] {checkpoint_name.upper()}"
              + (" [physics-loss]" if physics else ""))
        print("-" * 70)

        if args.skip_existing and os.path.exists(checkpoint_path):
            print(f"  Kihagyva: {checkpoint_path} már létezik.")
            # Beolvassuk a meglévő eredményt
            ck = torch.load(checkpoint_path, map_location='cpu', weights_only=False)
            summary[checkpoint_name] = {
                'best_epoch': ck.get('best_epoch'),
                'val_loss':   ck.get('val_loss'),
                'params':     sum(p.numel() for p in get_model(model_name, **model_kwargs).parameters()),
            }
            continue

        set_seed(args.seed)  # reprodukálhatóság minden modellnél
        model = get_model(model_name, **model_kwargs)
        params = count_parameters(model)
        print(f"  Paraméterek: {params:,}")

        t0 = time.time()
        history = train_model(
            model=model,
            train_path=args.train,
            val_path=args.val,
            output_dir=args.output,
            model_name=checkpoint_name,
            model_class=model_name,
            model_kwargs=model_kwargs,
            feature_cols=DEFAULT_FEATURE_COLS,
            target_col=DEFAULT_TARGET_COL,
            batch_size=args.batch_size,
            max_epochs=args.epochs,
            lr=args.lr,
            weight_decay=args.weight_decay,
            patience=args.patience,
            device=args.device,
            augment_put=False,
            physics_loss=physics,
            physics_lambda=args.physics_lambda,
        )
        elapsed = time.time() - t0

        summary[checkpoint_name] = {
            'best_epoch': history['best_epoch'],
            'val_loss':   history['best_val_loss'],
            'params':     params,
            'train_time_s': elapsed,
        }
        print(f"  Tanítási idő: {elapsed:.0f}s")

    # Összefoglaló
    total_elapsed = time.time() - total_start
    print("\n" + "=" * 70)
    print("TANÍTÁS KÉSZ")
    print("=" * 70)
    print(f"Összes idő: {total_elapsed/60:.1f} perc\n")

    col = 18
    header = f"{'Modell':<{col}} {'Params':>10} {'Best ep':>8} {'Val MSE':>12} {'Idő (s)':>10}"
    print(header)
    print("-" * len(header))
    for name, info in summary.items():
        t_str = f"{info.get('train_time_s', 0):.0f}" if 'train_time_s' in info else "—"
        print(f"{name:<{col}} {info['params']:>10,} {info['best_epoch']:>8} "
              f"{info['val_loss']:>12.6f} {t_str:>10}")

    # JSON mentés
    results_path = os.path.join(args.output, 'train_summary.json')
    with open(results_path, 'w', encoding='utf-8') as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    print(f"\nÖsszefoglaló mentve: {results_path}")


if __name__ == '__main__':
    main()
