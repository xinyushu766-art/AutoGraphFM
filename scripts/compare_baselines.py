"""Run the controlled Stage 0.2 baseline comparison."""

from __future__ import annotations

import argparse

from autogfm.config import SyntheticMarketConfig, TrainingConfig
from autogfm.experiments import run_stage0_2_experiments


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--hidden-dim", type=int, default=32)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--output-dir",
        type=str,
        default="outputs/stage0_2",
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

    run_stage0_2_experiments(
        market_config=market_config,
        training_config=training_config,
        output_dir=args.output_dir,
    )


if __name__ == "__main__":
    main()
