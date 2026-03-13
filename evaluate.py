"""
CLI belépési pont – neurális háló opciós árazó kiértékelés.

Használat:
    python evaluate.py --checkpoint models/mlp_best.pt
    python evaluate.py --checkpoint models/mlp_best.pt --segmented
    python evaluate.py --checkpoint models/mlp_best.pt models/deep_mlp_best.pt models/resnet_best.pt --compare
"""

import argparse

from src.evaluate import evaluate_model, evaluate_model_segmented, compare_models


def parse_args():
    p = argparse.ArgumentParser(
        description='Neurális háló opciós árazó kiértékelés'
    )
    p.add_argument('--checkpoint',  type=str, nargs='+', required=True,
                   help='Checkpoint .pt fájl(ok) elérési útja')
    p.add_argument('--test',        type=str, default='data/test.parquet',
                   help='Teszt halmaz parquet elérési útja (default: data/test.parquet)')
    p.add_argument('--compare',     action='store_true',
                   help='Modellek összehasonlítása táblázatos formában')
    p.add_argument('--segmented',   action='store_true',
                   help='Moneyness-szegmentált kiértékelés (OTM/ATM/ITM)')
    p.add_argument('--batch-size',  type=int, default=4096,
                   help='Batch méret inference-hez (default: 4096)')
    p.add_argument('--device',      type=str, default='cpu',
                   help='Számítási eszköz: cpu | cuda | mps (default: cpu)')
    return p.parse_args()


def main():
    args = parse_args()

    if args.compare or len(args.checkpoint) > 1:
        compare_models(
            checkpoint_paths=args.checkpoint,
            test_path=args.test,
            batch_size=args.batch_size,
            device=args.device,
        )
    elif args.segmented:
        evaluate_model_segmented(
            checkpoint_path=args.checkpoint[0],
            test_path=args.test,
            batch_size=args.batch_size,
            device=args.device,
            verbose=True,
        )
    else:
        evaluate_model(
            checkpoint_path=args.checkpoint[0],
            test_path=args.test,
            batch_size=args.batch_size,
            device=args.device,
            verbose=True,
        )


if __name__ == '__main__':
    main()
