"""Q network for deployable Learned-Q BRiG-AFA."""

import torch
from torch import nn


class QNetwork(nn.Module):
    """Predict final risk-to-go for acquiring a candidate feature.

    Inputs are deployable at decision time: zero-filled observed features, the
    observation mask, candidate feature one-hot vector, and remaining budget.
    """

    def __init__(
        self,
        n_features: int,
        hidden_dim: int = 128,
        n_hidden: int = 2,
        activation: str = "relu",
        norm: str = "none",
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        if activation not in {"relu", "gelu"}:
            raise ValueError("activation must be 'relu' or 'gelu'")
        if norm not in {"none", "layernorm"}:
            raise ValueError("norm must be 'none' or 'layernorm'")
        if n_hidden < 1:
            raise ValueError("n_hidden must be at least 1")
        if hidden_dim < 1:
            raise ValueError("hidden_dim must be at least 1")
        if dropout < 0.0:
            raise ValueError("dropout must be non-negative")

        self.architecture = {
            "n_features": int(n_features),
            "hidden_dim": int(hidden_dim),
            "n_hidden": int(n_hidden),
            "activation": str(activation),
            "norm": str(norm),
            "dropout": float(dropout),
        }
        input_dim = n_features * 3 + 1
        layers = []
        activation_cls = nn.ReLU if activation == "relu" else nn.GELU
        for layer_idx in range(n_hidden):
            layers.append(nn.Linear(input_dim if layer_idx == 0 else hidden_dim, hidden_dim))
            if norm == "layernorm":
                layers.append(nn.LayerNorm(hidden_dim))
            layers.append(activation_cls())
            if dropout > 0.0:
                layers.append(nn.Dropout(float(dropout)))
        layers.append(nn.Linear(hidden_dim, 1))
        self.net = nn.Sequential(*layers)

    def forward(
        self,
        x_obs: torch.Tensor,
        mask: torch.Tensor,
        candidate_onehot: torch.Tensor,
        remaining_budget: torch.Tensor,
    ) -> torch.Tensor:
        if remaining_budget.dim() == 1:
            remaining_budget = remaining_budget.unsqueeze(1)
        inputs = torch.cat([x_obs, mask.float(), candidate_onehot.float(), remaining_budget.float()], dim=-1)
        return self.net(inputs).squeeze(-1)

def q_architecture_from_checkpoint(checkpoint: dict) -> dict:
    architecture = checkpoint.get("q_architecture", {}) or {}
    return {
        "hidden_dim": int(architecture.get("hidden_dim", 128)),
        "n_hidden": int(architecture.get("n_hidden", 2)),
        "activation": str(architecture.get("activation", "relu")),
        "norm": str(architecture.get("norm", "none")),
        "dropout": float(architecture.get("dropout", 0.0)),
    }

