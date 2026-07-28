"""Run Stage 0.2E safe graph experiments."""

from __future__ import annotations

import argparse

from autogfm.config import SyntheticMarketConfig, TrainingConfig
from autogfm.experiments import run_stage0_2e_safe_graph_experiments


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=60)
    parser.add_argument("--patience", type=int, default=8)
    parser.add_argument("--min-delta", type=float, default=1e-4)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--hidden-dim", type=int, default=32)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--train-ratio", type=float, default=0.65)
    parser.add_argument("--validation-ratio", type=float, default=0.15)
    parser.add_argument(
        "--output-dir",
        type=str,
        default="outputs/stage0_2e",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    market_config = SyntheticMarketConfig(seed=args.seed)
    training_config = TrainingConfig(
        hidden_dim=args.hidden_dim,
        batch_size=args.batch_size,
        epochs=args.epochs,
        learning_rate=args.learning_rate,
        seed=args.seed,
    )

    run_stage0_2e_safe_graph_experiments(
        market_config=market_config,
        training_config=training_config,
        output_dir=args.output_dir,
        patience=args.patience,
        min_delta=args.min_delta,
        train_ratio=args.train_ratio,
        validation_ratio=args.validation_ratio,
    )


if __name__ == "__main__":
    main()
