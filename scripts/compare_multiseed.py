"""Run Stage 0.2F paired multi-seed stability evaluation."""

from __future__ import annotations

import argparse

from autogfm.config import SyntheticMarketConfig, TrainingConfig
from autogfm.experiments import run_stage0_2f_multiseed_experiments


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-seed", type=int, default=42)
    parser.add_argument(
        "--model-seeds",
        type=int,
        nargs="+",
        default=[42, 43, 44, 45, 46],
    )
    parser.add_argument(
        "--graph-permutation-seed",
        type=int,
        default=1042,
    )
    parser.add_argument("--epochs", type=int, default=150)
    parser.add_argument("--patience", type=int, default=12)
    parser.add_argument("--min-delta", type=float, default=1e-4)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--hidden-dim", type=int, default=32)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--train-ratio", type=float, default=0.65)
    parser.add_argument("--validation-ratio", type=float, default=0.15)
    parser.add_argument(
        "--output-dir",
        type=str,
        default="outputs/stage0_2f",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    market_config = SyntheticMarketConfig(seed=args.data_seed)
    training_template = TrainingConfig(
        hidden_dim=args.hidden_dim,
        batch_size=args.batch_size,
        epochs=args.epochs,
        learning_rate=args.learning_rate,
        seed=args.model_seeds[0],
    )

    run_stage0_2f_multiseed_experiments(
        market_config=market_config,
        training_template=training_template,
        model_seeds=args.model_seeds,
        graph_permutation_seed=args.graph_permutation_seed,
        output_dir=args.output_dir,
        patience=args.patience,
        min_delta=args.min_delta,
        train_ratio=args.train_ratio,
        validation_ratio=args.validation_ratio,
    )


if __name__ == "__main__":
    main()
