"""
CLI belépési pont – neurális háló opciós árazó tanítás.

Használat:
    python train.py --model resnet --epochs 200 --output models/
    python train.py --model mlp --hidden-dim 100 --output models/
    python train.py --model deep_mlp --batch-size 4096 --output models/
    python train.py --model gelu_resnet --epochs 200 --output models/
    python train.py --model dense_mlp --epochs 200 --output models/
    python train.py --model highway --epochs 200 --output models/
    python train.py --model finn --epochs 200 --output models/
    python train.py --model resnet --physics-loss --physics-lambda 0.1 --epochs 200
"""

import argparse
import random

import numpy as np
import torch

from src.model import get_model, count_parameters
from src.train import (train_model, DEFAULT_FEATURE_COLS, AUGMENTED_FEATURE_COLS,
                       DEFAULT_TARGET_COL)


def parse_args():
    p = argparse.ArgumentParser(
        description='Neurális háló opciós árazó tanítás'
    )
    p.add_argument('--model',        type=str, default='mlp',
                   choices=['mlp', 'deep_mlp', 'resnet',
                            'gelu_resnet', 'dense_mlp', 'highway', 'finn'],
                   help='Modell architektúra (default: mlp)')
    p.add_argument('--train',        type=str, default='data/train.parquet',
                   help='Tanítóhalmaz parquet elérési útja')
    p.add_argument('--val',          type=str, default='data/val.parquet',
                   help='Validációs halmaz parquet elérési útja')
    p.add_argument('--output',       type=str, default='models/',
                   help='Kimeneti mappa checkpointhoz (default: models/)')
    p.add_argument('--batch-size',   type=int, default=4096,
                   help='Mini-batch méret (default: 4096)')
    p.add_argument('--epochs',       type=int, default=200,
                   help='Maximális epoch szám (default: 200)')
    p.add_argument('--lr',           type=float, default=1e-3,
                   help='Tanulási ráta (default: 0.001)')
    p.add_argument('--weight-decay', type=float, default=1e-4,
                   help='L2 regularizáció (default: 0.0001)')
    p.add_argument('--patience',     type=int, default=10,
                   help='Early stopping türelem epochban (default: 10)')
    p.add_argument('--hidden-dim',   type=int, default=None,
                   help='Rejtett réteg mérete (default: modell-specifikus)')
    p.add_argument('--n-layers',     type=int, default=None,
                   help='Rejtett rétegek / blokkok száma (default: modell-specifikus)')
    p.add_argument('--device',       type=str, default='auto',
                   help='Számítási eszköz: auto | cpu | cuda | mps (default: auto)')
    p.add_argument('--seed',         type=int, default=42,
                   help='Véletlenszám mag (default: 42)')
    p.add_argument('--augment-put',  action='store_true',
                   help='Put-call paritással megduplázza az adathalmazt (input_dim=6)')
    p.add_argument('--physics-loss', action='store_true',
                   help='Physics-informed loss: delta korlát (∂C/∂moneyness ∈ [0,1])')
    p.add_argument('--physics-lambda', type=float, default=0.1,
                   help='Physics loss súlya λ: L = L_MSE + λ·L_delta (default: 0.1)')
    p.add_argument('--name',         type=str, default=None,
                   help='Checkpoint neve (default: modell neve, pl. resnet_phys)')
    return p.parse_args()


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def build_model_kwargs(args) -> dict:
    """Modell kwargs összeállítása az argumentumokból."""
    input_dim = 6 if args.augment_put else 5
    defaults = {
        'mlp':         {'input_dim': input_dim, 'hidden_dim': 100, 'n_layers': 4},
        'deep_mlp':    {'input_dim': input_dim, 'hidden_dim': 256, 'n_layers': 4, 'dropout': 0.1},
        'resnet':      {'input_dim': input_dim, 'hidden_dim': 256, 'n_blocks': 3, 'dropout': 0.1},
        'gelu_resnet': {'input_dim': input_dim, 'hidden_dim': 256, 'n_blocks': 3, 'dropout': 0.1},
        'dense_mlp':   {'input_dim': input_dim, 'hidden_dim': 128, 'n_layers': 4, 'dropout': 0.1},
        'highway':     {'input_dim': input_dim, 'hidden_dim': 256, 'n_blocks': 4, 'dropout': 0.1},
        'finn':        {'input_dim': input_dim, 'approx_dim': 64, 'resnet_dim': 256,
                        'n_blocks': 3, 'dropout': 0.1},
    }
    kwargs = defaults[args.model].copy()
    if args.hidden_dim is not None:
        kwargs['hidden_dim'] = args.hidden_dim
    if args.n_layers is not None:
        if args.model in ('resnet', 'gelu_resnet', 'highway', 'finn'):
            kwargs['n_blocks'] = args.n_layers
        else:
            kwargs['n_layers'] = args.n_layers
    return kwargs


def main():
    args = parse_args()
    set_seed(args.seed)

    print("Neurális háló opciós árazó tanítás")
    print(f"  Modell:        {args.model}")
    print(f"  Tanítóhalmaz:  {args.train}")
    print(f"  Val. halmaz:   {args.val}")
    print(f"  Kimeneti mappa:{args.output}")
    print(f"  Batch méret:   {args.batch_size}")
    print(f"  Max epochok:   {args.epochs}")
    print(f"  Tanulási ráta: {args.lr}")
    print(f"  Weight decay:  {args.weight_decay}")
    print(f"  Patience:      {args.patience}")
    print(f"  Seed:          {args.seed}")
    print(f"  Augment put:   {args.augment_put}")
    print(f"  Physics loss:  {args.physics_loss}"
          + (f" (lambda={args.physics_lambda})" if args.physics_loss else ""))
    print()

    model_kwargs = build_model_kwargs(args)
    print(f"  Modell kwargs: {model_kwargs}")
    print()

    model = get_model(args.model, **model_kwargs)
    print(f"  Paraméterek: {count_parameters(model):,}")
    print()

    feature_cols = AUGMENTED_FEATURE_COLS if args.augment_put else DEFAULT_FEATURE_COLS

    checkpoint_name = args.name if args.name else args.model

    history = train_model(
        model=model,
        train_path=args.train,
        val_path=args.val,
        output_dir=args.output,
        model_name=checkpoint_name,
        model_class=args.model,
        model_kwargs=model_kwargs,
        feature_cols=feature_cols,
        target_col=DEFAULT_TARGET_COL,
        batch_size=args.batch_size,
        max_epochs=args.epochs,
        lr=args.lr,
        weight_decay=args.weight_decay,
        patience=args.patience,
        device=args.device,
        augment_put=args.augment_put,
        physics_loss=args.physics_loss,
        physics_lambda=args.physics_lambda,
    )

    print()
    print(f"Tanítás kész. Legjobb epoch: {history['best_epoch']}, "
          f"val loss: {history['best_val_loss']:.6f}")


if __name__ == '__main__':
    main()
