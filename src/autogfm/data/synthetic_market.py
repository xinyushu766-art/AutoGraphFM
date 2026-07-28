"""Synthetic automotive market with graph-coupled demand.

The generator creates three features for every vehicle and month:

0. log-sales;
1. price index;
2. review sentiment.

The target is future log-sales. A static relation graph is constructed from
segment membership and price similarity.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch
from torch import Tensor
from torch.utils.data import Dataset

from autogfm.config import SyntheticMarketConfig


@dataclass(frozen=True)
class SyntheticMarket:
    """Full synthetic market before sliding-window extraction."""

    features: Tensor
    adjacency: Tensor
    segments: Tensor


def _build_relation_graph(
    segments: np.ndarray,
    base_prices: np.ndarray,
) -> np.ndarray:
    """Create sparse weighted edges from segment and price similarity."""
    num_vehicles = len(segments)
    graph = np.zeros((num_vehicles, num_vehicles), dtype=np.float32)

    price_scale = float(np.std(base_prices) + 1e-6)
    for i in range(num_vehicles):
        for j in range(num_vehicles):
            if i == j:
                continue

            segment_factor = 1.0 if segments[i] == segments[j] else 0.25
            price_similarity = np.exp(
                -abs(base_prices[i] - base_prices[j]) / price_scale
            )
            graph[i, j] = segment_factor * price_similarity

    max_neighbors = min(4, num_vehicles - 1)
    sparse_graph = np.zeros_like(graph)

    for i in range(num_vehicles):
        neighbor_ids = np.argpartition(graph[i], -max_neighbors)[-max_neighbors:]
        sparse_graph[i, neighbor_ids] = graph[i, neighbor_ids]

    return np.maximum(sparse_graph, sparse_graph.T)


def generate_synthetic_market(config: SyntheticMarketConfig) -> SyntheticMarket:
    """Generate one graph-coupled multivariate market time series."""
    config.validate()
    rng = np.random.default_rng(config.seed)

    num_vehicles = config.num_vehicles
    num_steps = config.num_steps

    segments = rng.integers(low=0, high=4, size=num_vehicles)
    base_prices = rng.uniform(0.8, 1.2, size=num_vehicles)
    base_demand = rng.uniform(800.0, 5000.0, size=num_vehicles)
    trend = rng.uniform(-2.0, 8.0, size=num_vehicles)
    seasonal_phase = rng.uniform(0.0, 2.0 * np.pi, size=num_vehicles)

    adjacency = _build_relation_graph(segments, base_prices)
    row_sum = adjacency.sum(axis=1, keepdims=True) + 1e-6
    neighbor_mix = adjacency / row_sum

    sales = np.zeros((num_steps, num_vehicles), dtype=np.float32)
    prices = np.zeros((num_steps, num_vehicles), dtype=np.float32)
    sentiment = np.zeros((num_steps, num_vehicles), dtype=np.float32)

    previous_sales = base_demand.copy()

    for time_index in range(num_steps):
        seasonal = 0.12 * np.sin(
            2.0 * np.pi * time_index / 12.0 + seasonal_phase
        )
        price_noise = rng.normal(0.0, 0.015, size=num_vehicles)
        prices[time_index] = np.clip(base_prices + price_noise, 0.65, 1.35)

        sentiment[time_index] = np.clip(
            0.55
            + 0.10 * seasonal
            + rng.normal(0.0, 0.04, size=num_vehicles),
            0.0,
            1.0,
        )

        neighbor_demand = neighbor_mix @ previous_sales
        own_persistence = 0.62 * previous_sales
        relation_context = 0.18 * neighbor_demand
        structural_demand = (
            0.20
            * base_demand
            * (1.0 + seasonal)
            * (1.15 - prices[time_index])
            * (0.70 + sentiment[time_index])
        )
        innovation = rng.normal(0.0, 90.0, size=num_vehicles)

        current_sales = (
            own_persistence
            + relation_context
            + structural_demand
            + trend
            + innovation
        )
        current_sales = np.clip(current_sales, 0.0, None)

        sales[time_index] = current_sales.astype(np.float32)
        previous_sales = current_sales

    log_sales = np.log1p(sales)
    features = np.stack([log_sales, prices, sentiment], axis=-1)

    return SyntheticMarket(
        features=torch.tensor(features, dtype=torch.float32),
        adjacency=torch.tensor(adjacency, dtype=torch.float32),
        segments=torch.tensor(segments, dtype=torch.long),
    )


class SyntheticAutomotiveDataset(Dataset[tuple[Tensor, Tensor]]):
    """Sliding-window samples from one synthetic market."""

    def __init__(self, config: SyntheticMarketConfig) -> None:
        self.config = config
        self.market = generate_synthetic_market(config)

        last_start = config.num_steps - config.history - config.horizon
        self.start_indices = list(range(last_start + 1))

    @property
    def adjacency(self) -> Tensor:
        """Return the shared static adjacency matrix."""
        return self.market.adjacency

    def __len__(self) -> int:
        return len(self.start_indices)

    def __getitem__(self, index: int) -> tuple[Tensor, Tensor]:
        start = self.start_indices[index]
        history_end = start + self.config.history
        horizon_end = history_end + self.config.horizon

        x = self.market.features[start:history_end]
        y = self.market.features[history_end:horizon_end, :, 0]
        return x, y
