"""Train the Stage 0.1 baseline on synthetic automotive data."""

from __future__ import annotations

import argparse
import random

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, Subset

from autogfm.config import SyntheticMarketConfig
from autogfm.data import SyntheticAutomotiveDataset
from autogfm.models import SimpleGraphTemporalForecaster


def set_seed(seed: int) -> None:
    """Set deterministic seeds used by this small experiment."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def temporal_split(
    dataset: SyntheticAutomotiveDataset,
    train_ratio: float = 0.8,
) -> tuple[Subset, Subset]:
    """Split sliding windows in chronological order, never randomly."""
    split_index = int(len(dataset) * train_ratio)
    train_ids = list(range(split_index))
    validation_ids = list(range(split_index, len(dataset)))
    return Subset(dataset, train_ids), Subset(dataset, validation_ids)


@torch.no_grad()
def evaluate(
    model: nn.Module,
    loader: DataLoader,
    adjacency: torch.Tensor,
    device: torch.device,
) -> float:
    """Return mean squared error over the validation period."""
    model.eval()
    loss_fn = nn.MSELoss()
    losses: list[float] = []

    for x, y in loader:
        x = x.to(device)
        y = y.to(device)
        prediction = model(x, adjacency)
        losses.append(float(loss_fn(prediction, y).item()))

    return float(np.mean(losses))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--hidden-dim", type=int, default=32)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = SyntheticMarketConfig()
    set_seed(config.seed)

    dataset = SyntheticAutomotiveDataset(config)
    train_set, validation_set = temporal_split(dataset)

    train_loader = DataLoader(
        train_set,
        batch_size=args.batch_size,
        shuffle=False,
    )
    validation_loader = DataLoader(
        validation_set,
        batch_size=args.batch_size,
        shuffle=False,
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    adjacency = dataset.adjacency.to(device)

    model = SimpleGraphTemporalForecaster(
        num_features=config.num_features,
        hidden_dim=args.hidden_dim,
        horizon=config.horizon,
    ).to(device)

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.learning_rate,
        weight_decay=1e-4,
    )
    loss_fn = nn.MSELoss()

    for epoch in range(1, args.epochs + 1):
        model.train()
        train_losses: list[float] = []

        for x, y in train_loader:
            x = x.to(device)
            y = y.to(device)

            optimizer.zero_grad(set_to_none=True)
            prediction = model(x, adjacency)
            loss = loss_fn(prediction, y)
            loss.backward()
            optimizer.step()

            train_losses.append(float(loss.item()))

        validation_loss = evaluate(
            model,
            validation_loader,
            adjacency,
            device,
        )
        print(
            f"epoch={epoch:03d} "
            f"train_mse={np.mean(train_losses):.6f} "
            f"validation_mse={validation_loss:.6f}"
        )


if __name__ == "__main__":
    main()
