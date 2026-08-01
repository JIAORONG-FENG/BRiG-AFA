"""Deployable Learned-Q BRiG-AFA policy."""

from typing import Dict, Union

import torch

from briga.models.q_network import QNetwork


QModelOrDict = Union[QNetwork, Dict[int, QNetwork], Dict[str, QNetwork]]


def _model_for_budget(model_q: QModelOrDict, remaining_budget: int, device: torch.device) -> QNetwork:
    if isinstance(model_q, dict):
        model = model_q.get(int(remaining_budget))
        if model is None:
            model = model_q[str(int(remaining_budget))]
    else:
        model = model_q
    model = model.to(device)
    model.eval()
    return model


def learned_briga_policy(model_q: QModelOrDict, x: torch.Tensor, budget: int, device: str = None) -> torch.Tensor:
    """Greedily acquire features with the lowest learned risk-to-go.

    Inference uses no labels, no regime metadata, and no unobserved candidate
    values. Candidate values are revealed only after the feature is selected.
    """
    device_obj = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
    if isinstance(model_q, dict):
        for model in model_q.values():
            model.to(device_obj)
            model.eval()
    else:
        model_q = model_q.to(device_obj)
        model_q.eval()

    x_device = x.to(device_obj)
    n_samples, n_features = x_device.shape
    mask = torch.zeros(n_samples, n_features, device=device_obj)
    selected_steps = []

    with torch.no_grad():
        for step in range(int(budget)):
            remaining_budget = int(budget) - step
            q_model = _model_for_budget(model_q, remaining_budget, device_obj)
            x_obs = x_device * mask
            q_values = torch.full((n_samples, n_features), float("inf"), device=device_obj)
            for candidate in range(n_features):
                available = mask[:, candidate] == 0
                if not bool(available.any()):
                    continue
                count = int(available.sum().item())
                candidate_onehot = torch.zeros(count, n_features, device=device_obj)
                candidate_onehot[:, candidate] = 1.0
                remaining = torch.full((count,), remaining_budget / float(n_features), device=device_obj)
                q_values[available, candidate] = q_model(
                    x_obs[available], mask[available], candidate_onehot, remaining
                )
            selected = q_values.argmin(dim=1)
            mask[torch.arange(n_samples, device=device_obj), selected] = 1.0
            selected_steps.append(selected.cpu())

    return torch.stack(selected_steps, dim=1).long()

def myopic_q_policy(model_q: QModelOrDict, x: torch.Tensor, budget: int, device: str = None) -> torch.Tensor:
    """Greedily acquire features using the one-step Q model at every step.

    This is deployable: it uses only observed zero-filled features, the current
    mask, candidate id, and the one-step remaining-budget scalar.
    """
    device_obj = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
    if isinstance(model_q, dict):
        for model in model_q.values():
            model.to(device_obj)
            model.eval()
    else:
        model_q = model_q.to(device_obj)
        model_q.eval()

    x_device = x.to(device_obj)
    n_samples, n_features = x_device.shape
    mask = torch.zeros(n_samples, n_features, device=device_obj)
    selected_steps = []

    with torch.no_grad():
        for _ in range(int(budget)):
            q_model = _model_for_budget(model_q, 1, device_obj)
            x_obs = x_device * mask
            q_values = torch.full((n_samples, n_features), float("inf"), device=device_obj)
            for candidate in range(n_features):
                available = mask[:, candidate] == 0
                if not bool(available.any()):
                    continue
                count = int(available.sum().item())
                candidate_onehot = torch.zeros(count, n_features, device=device_obj)
                candidate_onehot[:, candidate] = 1.0
                remaining = torch.full((count,), 1.0 / float(n_features), device=device_obj)
                q_values[available, candidate] = q_model(
                    x_obs[available], mask[available], candidate_onehot, remaining
                )
            selected = q_values.argmin(dim=1)
            mask[torch.arange(n_samples, device=device_obj), selected] = 1.0
            selected_steps.append(selected.cpu())

    return torch.stack(selected_steps, dim=1).long()

