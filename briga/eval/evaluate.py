"""Evaluate fixed-budget feature acquisition policies."""

from typing import Dict, Iterable, List

import pandas as pd
import torch
from torch import nn
import torch.nn.functional as F

from briga.eval.baselines import (
    compute_global_mi_order,
    context_first_policy,
    global_mi_policy,
    regime_oracle_policy,
    random_policy,
)


def indices_to_mask(indices: torch.Tensor, n_features: int) -> torch.Tensor:
    mask = torch.zeros(indices.size(0), n_features, dtype=torch.float32)
    mask.scatter_(1, indices.long(), 1.0)
    return mask


@torch.no_grad()
def evaluate_selected_features(
    model: nn.Module,
    x: torch.Tensor,
    y: torch.Tensor,
    selected: torch.Tensor,
    batch_size: int = 512,
    device: str = None,
) -> Dict[str, float]:
    device_obj = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
    model = model.to(device_obj)
    model.eval()
    total_loss = 0.0
    total_correct = 0
    total = 0

    for start in range(0, x.size(0), batch_size):
        end = start + batch_size
        xb = x[start:end].to(device_obj)
        yb = y[start:end].to(device_obj)
        mask = indices_to_mask(selected[start:end], x.size(1)).to(device_obj)
        logits = model(xb * mask, mask)
        total_loss += F.cross_entropy(logits, yb, reduction="sum").item()
        total_correct += (logits.argmax(dim=1) == yb).sum().item()
        total += yb.numel()

    return {"accuracy": total_correct / total, "cross_entropy": total_loss / total}


def evaluate_baselines(
    model: nn.Module,
    val_tensors: tuple,
    test_tensors: tuple,
    metadata: Dict[str, object],
    budgets: Iterable[int] = (1, 2, 3, 5, 8),
    batch_size: int = 512,
    device: str = None,
    random_seeds: int = 10,
    mi_tensors: tuple = None,
) -> pd.DataFrame:
    test_x, test_y = test_tensors
    test_regime = metadata["test_regime"]
    mi_x, mi_y = mi_tensors if mi_tensors is not None else val_tensors
    feature_order = compute_global_mi_order(mi_x, mi_y)
    rows: List[Dict[str, object]] = []

    for k in budgets:
        random_metrics = []
        for seed in range(random_seeds):
            selected = random_policy(test_x, k, seed=1000 + 100 * int(k) + seed)
            random_metrics.append(evaluate_selected_features(model, test_x, test_y, selected, batch_size, device))
        random_row = {
            "policy": "random",
            "budget": int(k),
            "accuracy": sum(metric["accuracy"] for metric in random_metrics) / random_seeds,
            "cross_entropy": sum(metric["cross_entropy"] for metric in random_metrics) / random_seeds,
        }
        rows.append(random_row)
        print(
            "policy=random k={} accuracy={:.4f} cross_entropy={:.4f}".format(
                k, random_row["accuracy"], random_row["cross_entropy"]
            )
        )

        policies = {
            "global_mi": global_mi_policy(test_x, k, feature_order),
            "context_first": context_first_policy(test_x, k, metadata),
            "regime_oracle": regime_oracle_policy(test_x, test_regime, k, metadata),
        }
        for name, selected in policies.items():
            metrics = evaluate_selected_features(model, test_x, test_y, selected, batch_size, device)
            row = {"policy": name, "budget": int(k)}
            row.update(metrics)
            rows.append(row)
            print(
                "policy={} k={} accuracy={:.4f} cross_entropy={:.4f}".format(
                    name, k, metrics["accuracy"], metrics["cross_entropy"]
                )
            )

    return pd.DataFrame(rows)
