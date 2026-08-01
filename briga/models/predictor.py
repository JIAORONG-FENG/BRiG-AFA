"""Partial-input predictor models."""

from typing import Dict, Mapping, Optional, Tuple

import torch
from torch import nn


DEFAULT_PREDICTOR_ARCHITECTURE = {
    "hidden_dim": 128,
    "n_hidden": 2,
    "dropout": 0.0,
    "activation": "relu",
    "norm": "none",
}


def predictor_architecture_from_checkpoint(checkpoint: Dict[str, object]) -> Dict[str, object]:
    """Read predictor architecture metadata, falling back to legacy defaults."""
    architecture = dict(DEFAULT_PREDICTOR_ARCHITECTURE)
    nested = checkpoint.get("architecture")
    if isinstance(nested, Mapping):
        for key in architecture:
            if key in nested:
                architecture[key] = nested[key]
    for key in architecture:
        if key in checkpoint:
            architecture[key] = checkpoint[key]
    architecture["hidden_dim"] = int(architecture["hidden_dim"])
    architecture["n_hidden"] = int(architecture["n_hidden"])
    architecture["dropout"] = float(architecture["dropout"])
    architecture["activation"] = str(architecture["activation"])
    architecture["norm"] = str(architecture["norm"])
    return architecture


def _load_torch_checkpoint(path: str, device: torch.device) -> Dict[str, object]:
    """Load a local project-generated predictor checkpoint.

    Newer PyTorch versions support weights_only=True for safer loading, but
    some of our local checkpoints were saved as full checkpoint dictionaries
    and cannot be read in weights-only mode. These checkpoints are generated
    inside this project, so falling back to weights_only=False is acceptable.
    """
    try:
        checkpoint = torch.load(path, map_location=device, weights_only=True)
    except TypeError:
        checkpoint = torch.load(path, map_location=device)
    except Exception as exc:
        message = str(exc).lower()
        if (
            "weights only load failed" in message
            or "weights_only" in message
            or "weightsunpickler" in message
            or "unsupported operand" in message
        ):
            checkpoint = torch.load(path, map_location=device, weights_only=False)
        else:
            raise

    if not isinstance(checkpoint, dict):
        raise NotImplementedError("Unsupported predictor checkpoint format at {}".format(path))
    return checkpoint


def _state_dict_from_checkpoint(checkpoint: Dict[str, object]) -> Mapping[str, torch.Tensor]:
    state_dict = checkpoint.get("model_state_dict")
    if state_dict is None:
        state_dict = checkpoint.get("state_dict")
    if not isinstance(state_dict, Mapping):
        raise NotImplementedError("Predictor checkpoint is missing a model state dict")
    return state_dict


def _infer_predictor_dimensions(state_dict: Mapping[str, torch.Tensor]) -> Tuple[int, int, int, int]:
    linear_keys = sorted(
        [
            key
            for key, value in state_dict.items()
            if key.startswith("net.") and key.endswith(".weight") and value.dim() == 2
        ],
        key=lambda key: int(key.split(".")[1]),
    )
    if len(linear_keys) < 2:
        raise NotImplementedError("Cannot infer predictor architecture from checkpoint state dict")

    first_weight = state_dict[linear_keys[0]]
    last_weight = state_dict[linear_keys[-1]]
    if first_weight.dim() != 2 or last_weight.dim() != 2:
        raise NotImplementedError("Unsupported predictor linear weight shapes in checkpoint")
    if int(first_weight.size(1)) % 2 != 0:
        raise NotImplementedError("Cannot infer n_features from predictor input dimension")

    n_features = int(first_weight.size(1) // 2)
    hidden_dim = int(first_weight.size(0))
    n_classes = int(last_weight.size(0))
    n_hidden = len(linear_keys) - 1
    return n_features, n_classes, hidden_dim, n_hidden


def load_partial_input_predictor_checkpoint(
    checkpoint_path: str,
    device: Optional[torch.device] = None,
) -> "PartialInputMLP":
    """Load a partial-input predictor from legacy or metadata-rich checkpoints."""
    device_obj = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
    checkpoint = _load_torch_checkpoint(checkpoint_path, device_obj)
    state_dict = _state_dict_from_checkpoint(checkpoint)
    inferred_n_features, inferred_n_classes, inferred_hidden_dim, inferred_n_hidden = (
        _infer_predictor_dimensions(state_dict)
    )

    architecture = predictor_architecture_from_checkpoint(checkpoint)
    nested = checkpoint.get("architecture")
    has_nested_hidden_dim = isinstance(nested, Mapping) and "hidden_dim" in nested
    has_nested_n_hidden = isinstance(nested, Mapping) and "n_hidden" in nested
    if "hidden_dim" not in checkpoint and not has_nested_hidden_dim:
        architecture["hidden_dim"] = inferred_hidden_dim
    if "n_hidden" not in checkpoint and not has_nested_n_hidden:
        architecture["n_hidden"] = inferred_n_hidden

    n_features = int(checkpoint.get("n_features", inferred_n_features))
    n_classes = int(checkpoint.get("n_classes", inferred_n_classes))
    if n_features != inferred_n_features:
        raise ValueError(
            "Checkpoint metadata n_features={} does not match state dict n_features={}".format(
                n_features, inferred_n_features
            )
        )
    if n_classes != inferred_n_classes:
        raise ValueError(
            "Checkpoint metadata n_classes={} does not match state dict n_classes={}".format(
                n_classes, inferred_n_classes
            )
        )

    model = PartialInputMLP(n_features=n_features, n_classes=n_classes, **architecture).to(device_obj)
    model.load_state_dict(state_dict)
    model.eval()
    for param in model.parameters():
        param.requires_grad_(False)
    return model


class PartialInputMLP(nn.Module):
    """MLP that consumes zero-filled observed values and an observation mask."""

    def __init__(
        self,
        n_features: int,
        n_classes: int,
        hidden_dim: int = 128,
        n_hidden: int = 2,
        dropout: float = 0.0,
        activation: str = "relu",
        norm: str = "none",
    ) -> None:
        super().__init__()
        if int(n_hidden) < 1:
            raise ValueError("n_hidden must be at least 1")
        if activation not in {"relu", "gelu"}:
            raise ValueError("activation must be 'relu' or 'gelu'")
        if norm not in {"none", "layernorm"}:
            raise ValueError("norm must be 'none' or 'layernorm'")

        self.n_features = int(n_features)
        self.n_classes = int(n_classes)
        self.hidden_dim = int(hidden_dim)
        self.n_hidden = int(n_hidden)
        self.dropout = float(dropout)
        self.activation = activation
        self.norm = norm

        layers = []
        input_dim = self.n_features * 2
        for layer_idx in range(self.n_hidden):
            layers.append(nn.Linear(input_dim if layer_idx == 0 else self.hidden_dim, self.hidden_dim))
            if self.norm == "layernorm":
                layers.append(nn.LayerNorm(self.hidden_dim))
            layers.append(nn.ReLU() if self.activation == "relu" else nn.GELU())
            if self.dropout > 0.0:
                layers.append(nn.Dropout(self.dropout))
        layers.append(nn.Linear(self.hidden_dim, self.n_classes))
        self.net = nn.Sequential(*layers)

    def architecture_metadata(self) -> Dict[str, object]:
        return {
            "n_features": self.n_features,
            "n_classes": self.n_classes,
            "hidden_dim": self.hidden_dim,
            "n_hidden": self.n_hidden,
            "dropout": self.dropout,
            "activation": self.activation,
            "norm": self.norm,
        }

    def forward(self, x_obs: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        inputs = torch.cat([x_obs, mask.float()], dim=-1)
        return self.net(inputs)
