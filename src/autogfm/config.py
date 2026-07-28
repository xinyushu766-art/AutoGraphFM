"""Configuration objects for reproducible Stage 0 experiments."""

from dataclasses import dataclass


@dataclass(frozen=True)
class SyntheticMarketConfig:
    """Controls the synthetic automotive market generator."""

    num_steps: int = 120
    num_vehicles: int = 16
    num_features: int = 3
    history: int = 12
    horizon: int = 3
    seed: int = 42

    def validate(self) -> None:
        """Fail early when the market configuration is invalid."""
        if self.num_steps <= self.history + self.horizon:
            raise ValueError("num_steps must be larger than history + horizon.")
        if self.num_vehicles < 2:
            raise ValueError("num_vehicles must be at least 2.")
        if self.num_features != 3:
            raise ValueError(
                "Stage 0 defines exactly three features: "
                "sales, price_index, and sentiment."
            )


@dataclass(frozen=True)
class TrainingConfig:
    """Controls model training and chronological validation."""

    hidden_dim: int = 32
    batch_size: int = 16
    epochs: int = 30
    learning_rate: float = 1e-3
    weight_decay: float = 1e-4
    train_ratio: float = 0.8
    seed: int = 42

    def validate(self) -> None:
        """Fail early when the training configuration is invalid."""
        if self.hidden_dim <= 0:
            raise ValueError("hidden_dim must be positive.")
        if self.batch_size <= 0:
            raise ValueError("batch_size must be positive.")
        if self.epochs <= 0:
            raise ValueError("epochs must be positive.")
        if self.learning_rate <= 0:
            raise ValueError("learning_rate must be positive.")
        if not 0.0 < self.train_ratio < 1.0:
            raise ValueError("train_ratio must be between 0 and 1.")
