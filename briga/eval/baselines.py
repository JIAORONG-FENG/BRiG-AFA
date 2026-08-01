"""Fixed-budget acquisition baselines for the synthetic task."""

from typing import Dict, List, Tuple

import numpy as np
import torch
from sklearn.feature_selection import mutual_info_classif


def random_policy(x: torch.Tensor, k: int, seed: int = 0) -> torch.Tensor:
    rng = np.random.RandomState(seed)
    n_samples, n_features = x.shape
    selected = np.zeros((n_samples, k), dtype=np.int64)
    for i in range(n_samples):
        selected[i] = rng.choice(n_features, size=k, replace=False)
    return torch.from_numpy(selected)


def compute_global_mi_order(x_val: torch.Tensor, y_val: torch.Tensor) -> List[int]:
    """Rank features by mutual information with the label.

    For strict benchmark use, pass training tensors. Validation tensors are
    acceptable only for quick diagnostics.
    """
    scores = mutual_info_classif(x_val.numpy(), y_val.numpy(), random_state=0)
    return list(np.argsort(-scores))


def global_mi_policy(x: torch.Tensor, k: int, feature_order: List[int]) -> torch.Tensor:
    selected = torch.tensor(feature_order[:k], dtype=torch.long)
    return selected.unsqueeze(0).repeat(x.size(0), 1)


def _metadata_feature_groups(metadata: Dict[str, object]) -> Tuple[int, List[int], List[int], List[int]]:
    context_feature = int(metadata["context_feature"])
    regime0 = list(metadata["regime0_informative"])
    regime1 = list(metadata["regime1_informative"])
    distractors = list(metadata["distractor_features"])
    return context_feature, regime0, regime1, distractors


def _dedupe_order(candidates: List[int], n_features: int) -> List[int]:
    seen = set()
    order = []
    for idx in candidates + list(range(n_features)):
        if idx not in seen:
            seen.add(idx)
            order.append(idx)
    return order


def context_first_policy(x: torch.Tensor, k: int, metadata: Dict[str, object]) -> torch.Tensor:
    """Select the context feature first, then infer the regime from its value."""
    context_feature, regime0, regime1, distractors = _metadata_feature_groups(metadata)
    n_features = x.size(1)

    rows = []
    inferred_regimes = (x[:, context_feature] > 0).long()
    for regime in inferred_regimes.tolist():
        primary = regime0 if int(regime) == 0 else regime1
        secondary = regime1 if int(regime) == 0 else regime0
        order = _dedupe_order([context_feature] + primary + secondary + distractors, n_features)
        rows.append(order[:k])
    return torch.tensor(rows, dtype=torch.long)


def regime_oracle_policy(x: torch.Tensor, regimes: torch.Tensor, k: int, metadata: Dict[str, object]) -> torch.Tensor:
    """Diagnostic upper bound that receives the true regime for free."""
    context_feature, regime0, regime1, distractors = _metadata_feature_groups(metadata)
    n_features = x.size(1)

    rows = []
    for regime in regimes.tolist():
        primary = regime0 if int(regime) == 0 else regime1
        secondary = regime1 if int(regime) == 0 else regime0
        order = _dedupe_order(primary + [context_feature] + secondary + distractors, n_features)
        rows.append(order[:k])
    return torch.tensor(rows, dtype=torch.long)
