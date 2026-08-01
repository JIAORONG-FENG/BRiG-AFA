"""Train a minimal deployable Learned-Q BRiG model."""

import os
from typing import Dict, Iterable, List, Optional, Tuple

import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm

from briga.models.q_network import QNetwork


def _one_hot(indices: torch.Tensor, n_features: int) -> torch.Tensor:
    return torch.nn.functional.one_hot(indices.long(), num_classes=n_features).float()


def _set_row_mask(mask: torch.Tensor, row: int, order: list, count: int) -> None:
    if count <= 0:
        return
    mask[row, torch.tensor(order[:count], dtype=torch.long, device=mask.device)] = 1.0


def _sample_state_masks_synthetic_guided(x: torch.Tensor, remaining_budget: int) -> torch.Tensor:
    """Sample states using the original synthetic context-regime structure.

    This sampler is retained for diagnostics and backwards comparisons. It is
    not the generic BRiG training method because it uses feature 0 and fixed
    feature groups as task-specific structure.
    """
    batch_size, n_features = x.shape
    device = x.device
    max_observed = n_features - int(remaining_budget)
    mask = torch.zeros(batch_size, n_features, device=device)
    mask_types = torch.randint(0, 6, (batch_size,), device=device)
    regimes = (x[:, 0] > 0).long()

    for i in range(batch_size):
        mask_type = int(mask_types[i].item())
        regime = int(regimes[i].item())
        prefix_len = int(torch.randint(0, max_observed + 1, (1,), device=device).item())

        if mask_type == 0:
            continue
        if mask_type == 1:
            if max_observed >= 1:
                mask[i, 0] = 1.0
            continue
        if mask_type == 2:
            n_observed = prefix_len
            if n_observed > 0:
                mask[i, torch.randperm(n_features, device=device)[:n_observed]] = 1.0
            continue

        if regime == 0:
            context_first = [0, 1, 2, 3, 4, 5, 6] + list(range(7, n_features))
            oracle_like = [1, 2, 3, 0, 4, 5, 6] + list(range(7, n_features))
        else:
            context_first = [0, 4, 5, 6, 1, 2, 3] + list(range(7, n_features))
            oracle_like = [4, 5, 6, 0, 1, 2, 3] + list(range(7, n_features))
        global_like = [5, 0, 4, 1, 2, 6, 3] + list(range(7, n_features))

        if mask_type == 3:
            _set_row_mask(mask, i, context_first, prefix_len)
        elif mask_type == 4:
            _set_row_mask(mask, i, oracle_like, prefix_len)
        else:
            _set_row_mask(mask, i, global_like, prefix_len)

    return mask


@torch.no_grad()
def _rollin_mask_from_q_models(
    q_models: Dict[int, QNetwork],
    x: torch.Tensor,
    prefix_len: int,
) -> torch.Tensor:
    """Build a mask by greedily rolling in with already trained Q models."""
    batch_size, n_features = x.shape
    device = x.device
    mask = torch.zeros(batch_size, n_features, device=device)
    if prefix_len <= 0 or not q_models:
        return mask

    available_budgets = sorted(int(budget) for budget in q_models.keys())
    for step in range(prefix_len):
        remaining_hint = min(max(available_budgets), max(available_budgets[0], prefix_len - step))
        q_model = q_models[remaining_hint]
        q_model.eval()
        values = torch.full((batch_size, n_features), float("inf"), device=device)
        x_obs = x * mask
        remaining = torch.full((batch_size,), float(remaining_hint) / float(n_features), device=device)

        for candidate in range(n_features):
            available = mask[:, candidate] == 0
            if not bool(available.any()):
                continue
            count = int(available.sum().item())
            candidate_onehot = torch.zeros(count, n_features, device=device)
            candidate_onehot[:, candidate] = 1.0
            values[available, candidate] = q_model(
                x_obs[available],
                mask[available],
                candidate_onehot,
                remaining[available],
            )

        selected = values.argmin(dim=1)
        mask[torch.arange(batch_size, device=device), selected] = 1.0
    return mask


def _sample_state_masks_generic(
    x: torch.Tensor,
    remaining_budget: int,
    feature_order: Optional[List[int]] = None,
    q_models: Optional[Dict[int, QNetwork]] = None,
) -> torch.Tensor:
    """Sample generic current-state masks without labels or task metadata.

    The generic sampler mixes empty masks, random masks, random prefix masks,
    optional static/global-order prefix masks, and optional learned roll-in
    masks. It does not inspect labels, metadata, regimes, or hard-coded feature
    groups.
    """
    batch_size, n_features = x.shape
    device = x.device
    max_observed = n_features - int(remaining_budget)
    mask = torch.zeros(batch_size, n_features, device=device)
    n_types = 3 + int(feature_order is not None) + int(bool(q_models))
    mask_types = torch.randint(0, n_types, (batch_size,), device=device)

    feature_order_tensor = None
    if feature_order is not None:
        deduped = []
        seen = set()
        for idx in list(feature_order) + list(range(n_features)):
            idx = int(idx)
            if idx not in seen:
                seen.add(idx)
                deduped.append(idx)
        feature_order_tensor = torch.tensor(deduped, dtype=torch.long, device=device)

    for i in range(batch_size):
        mask_type = int(mask_types[i].item())
        prefix_len = int(torch.randint(0, max_observed + 1, (1,), device=device).item())

        if mask_type == 0:
            continue

        if mask_type == 1:
            if prefix_len > 0:
                mask[i, torch.randperm(n_features, device=device)[:prefix_len]] = 1.0
            continue

        if mask_type == 2:
            if prefix_len > 0:
                order = torch.rand(n_features, device=device).argsort()
                mask[i, order[:prefix_len]] = 1.0
            continue

        if feature_order_tensor is not None and mask_type == 3:
            if prefix_len > 0:
                mask[i, feature_order_tensor[:prefix_len]] = 1.0
            continue

        if q_models:
            rollin = _rollin_mask_from_q_models(q_models, x[i : i + 1], prefix_len)
            mask[i] = rollin[0]

    return mask


def _sample_state_masks(
    x: torch.Tensor,
    remaining_budget: int,
    state_sampling_mode: str,
    feature_order: Optional[List[int]] = None,
    q_models: Optional[Dict[int, QNetwork]] = None,
) -> torch.Tensor:
    if state_sampling_mode == "generic":
        return _sample_state_masks_generic(
            x=x,
            remaining_budget=remaining_budget,
            feature_order=feature_order,
            q_models=q_models,
        )
    if state_sampling_mode == "synthetic_guided":
        return _sample_state_masks_synthetic_guided(x, remaining_budget)
    raise ValueError("state_sampling_mode must be 'generic' or 'synthetic_guided'")

def _expand_all_candidates(
    x: torch.Tensor,
    y: torch.Tensor,
    mask: torch.Tensor,
    remaining_budget: int,
) -> Tuple[
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
]:
    batch_size, n_features = x.shape
    x_rows = []
    y_rows = []
    mask_rows = []
    candidate_rows = []
    next_mask_rows = []
    state_rows = []
    row_indices = torch.arange(batch_size, dtype=torch.long, device=x.device)

    for candidate in range(n_features):
        available = mask[:, candidate] == 0
        if not bool(available.any()):
            continue
        candidate_idx = torch.full((int(available.sum().item()),), candidate, dtype=torch.long, device=x.device)
        current_mask = mask[available]
        next_mask = current_mask.clone()
        next_mask[:, candidate] = 1.0

        x_rows.append(x[available])
        y_rows.append(y[available])
        mask_rows.append(current_mask)
        candidate_rows.append(candidate_idx)
        next_mask_rows.append(next_mask)
        state_rows.append(row_indices[available])

    x_exp = torch.cat(x_rows, dim=0)
    y_exp = torch.cat(y_rows, dim=0)
    mask_exp = torch.cat(mask_rows, dim=0)
    candidate_exp = torch.cat(candidate_rows, dim=0)
    next_mask_exp = torch.cat(next_mask_rows, dim=0)
    state_exp = torch.cat(state_rows, dim=0)
    remaining = torch.full((x_exp.size(0),), float(remaining_budget) / float(n_features), device=x.device)
    return x_exp, y_exp, mask_exp, candidate_exp, next_mask_exp, remaining, state_exp


def _validate_risk_type(risk_type: str) -> str:
    risk_type = str(risk_type)
    if risk_type not in {"ce", "ptrue", "ce_ptrue"}:
        raise ValueError("risk_type must be one of 'ce', 'ptrue', or 'ce_ptrue'")
    return risk_type


def _normalize_train_budgets(budgets: Iterable[int], n_features: int) -> List[int]:
    """Validate budget-specific Bellman training budgets.

    Training Q_r requires Q_{r-1}. Therefore, train budgets must be
    consecutive 1..B. Evaluation may still report sparse budgets such as
    1,2,4,8,12, but training must include all intermediate budgets.
    """
    train_budgets = sorted(set(int(b) for b in budgets))
    if not train_budgets:
        raise ValueError("budgets must contain at least one budget")
    if train_budgets[0] != 1:
        raise ValueError("BRiG-AFA train budgets must start at 1")
    if train_budgets[-1] > int(n_features):
        raise ValueError(
            "max train budget {} exceeds n_features {}".format(train_budgets[-1], int(n_features))
        )
    expected = list(range(1, train_budgets[-1] + 1))
    if train_budgets != expected:
        raise ValueError(
            "BRiG-AFA requires consecutive train budgets 1..B. Got {}. "
            "Use tuple(range(1, max(eval_budgets)+1)) for training.".format(train_budgets)
        )
    return train_budgets


def _candidate_ranking_loss(
    q_values: torch.Tensor,
    target_risk: torch.Tensor,
    state_ids: torch.Tensor,
) -> torch.Tensor:
    losses = []
    for state_id in torch.unique(state_ids, sorted=True):
        group = state_ids == state_id
        group_scores = -q_values[group].unsqueeze(0)
        best_action = target_risk[group].argmin().view(1)
        losses.append(torch.nn.functional.cross_entropy(group_scores, best_action))
    if not losses:
        return q_values.new_tensor(0.0)
    return torch.stack(losses).mean()


@torch.no_grad()
def _terminal_risk(
    predictor: nn.Module,
    x: torch.Tensor,
    y: torch.Tensor,
    mask: torch.Tensor,
    risk_type: str = "ce",
    lambda_ptrue: float = 0.5,
) -> torch.Tensor:
    logits = predictor(x * mask, mask)
    risk_type = _validate_risk_type(risk_type)
    probabilities = torch.nn.functional.softmax(logits, dim=1)
    p_true = probabilities.gather(1, y.long().unsqueeze(1)).squeeze(1)
    ptrue_risk = 1.0 - p_true
    if risk_type == "ptrue":
        return ptrue_risk

    ce = torch.nn.functional.cross_entropy(logits, y, reduction="none")
    if risk_type == "ce_ptrue":
        return ce + float(lambda_ptrue) * ptrue_risk
    return ce


@torch.no_grad()
def _target_min_q(
    q_prev: QNetwork,
    x: torch.Tensor,
    next_mask: torch.Tensor,
    remaining_budget: int,
) -> torch.Tensor:
    batch_size, n_features = x.shape
    values = torch.full((batch_size, n_features), float("inf"), device=x.device)
    x_obs = x * next_mask
    remaining = torch.full((batch_size,), float(remaining_budget) / float(n_features), device=x.device)

    for candidate in range(n_features):
        available = next_mask[:, candidate] == 0
        if not bool(available.any()):
            continue
        candidate_idx = torch.full((int(available.sum().item()),), candidate, dtype=torch.long, device=x.device)
        candidate_onehot = _one_hot(candidate_idx, n_features)
        values[available, candidate] = q_prev(
            x_obs[available],
            next_mask[available],
            candidate_onehot,
            remaining[available],
        )
    return values.min(dim=1).values


@torch.no_grad()
def _rollout_terminal_ce_from_mask(
    q_models: Dict[int, QNetwork],
    predictor: nn.Module,
    x: torch.Tensor,
    y: torch.Tensor,
    initial_mask: torch.Tensor,
    remaining_steps: int,
    risk_type: str = "ce",
    lambda_ptrue: float = 0.5,
) -> torch.Tensor:
    """Greedily roll out budget-specific Q models, then score terminal risk."""
    mask = initial_mask.clone()
    batch_size, n_features = x.shape

    for steps_left in range(int(remaining_steps), 0, -1):
        q_model = q_models[int(steps_left)]
        q_model.eval()
        values = torch.full((batch_size, n_features), float("inf"), device=x.device)
        x_obs = x * mask
        remaining = torch.full((batch_size,), float(steps_left) / float(n_features), device=x.device)

        for candidate in range(n_features):
            available = mask[:, candidate] == 0
            if not bool(available.any()):
                continue
            candidate_idx = torch.full((int(available.sum().item()),), candidate, dtype=torch.long, device=x.device)
            candidate_onehot = _one_hot(candidate_idx, n_features)
            values[available, candidate] = q_model(
                x_obs[available],
                mask[available],
                candidate_onehot,
                remaining[available],
            )

        best_candidate = values.argmin(dim=1)
        mask[torch.arange(batch_size, device=x.device), best_candidate] = 1.0

    return _terminal_risk(predictor, x, y, mask, risk_type=risk_type, lambda_ptrue=lambda_ptrue)


def _train_one_budget_level(
    q_model: QNetwork,
    q_prev: QNetwork,
    q_rollout_models: Dict[int, QNetwork],
    predictor: nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    n_features: int,
    remaining_budget: int,
    epochs: int,
    device: torch.device,
    empty_mask_weight: float = 1.0,
    use_empty_rollout_target: bool = True,
    state_sampling_mode: str = "generic",
    feature_order: Optional[List[int]] = None,
    use_learned_rollin: bool = False,
    risk_type: str = "ce",
    lambda_ptrue: float = 0.5,
    ranking_loss_weight: float = 0.0,
) -> None:
    loss_fn = nn.MSELoss()
    for epoch in range(1, epochs + 1):
        q_model.train()
        total_loss = 0.0
        total = 0
        desc = "train q_briga r={} epoch {}/{}".format(remaining_budget, epoch, epochs)
        progress = tqdm(loader, desc=desc, leave=False)
        for xb, yb in progress:
            xb = xb.to(device)
            yb = yb.to(device)
            mask = _sample_state_masks(
                xb,
                remaining_budget,
                state_sampling_mode=state_sampling_mode,
                feature_order=feature_order,
                q_models=q_rollout_models if use_learned_rollin else None,
            )
            (
                x_exp,
                y_exp,
                mask_exp,
                candidate_exp,
                next_mask_exp,
                remaining,
                state_exp,
            ) = _expand_all_candidates(xb, yb, mask, remaining_budget)
            candidate_onehot = _one_hot(candidate_exp, n_features)

            with torch.no_grad():
                if remaining_budget == 1:
                    target = _terminal_risk(
                        predictor,
                        x_exp,
                        y_exp,
                        next_mask_exp,
                        risk_type=risk_type,
                        lambda_ptrue=lambda_ptrue,
                    )
                else:
                    target = _target_min_q(q_prev, x_exp, next_mask_exp, remaining_budget - 1)

            pred = q_model(x_exp * mask_exp, mask_exp, candidate_onehot, remaining)
            loss = loss_fn(pred, target)
            if ranking_loss_weight > 0.0:
                ranking_loss = _candidate_ranking_loss(pred, target, state_exp)
                loss = loss + float(ranking_loss_weight) * ranking_loss

            if bool(use_empty_rollout_target) and remaining_budget > 1:
                empty_mask = torch.zeros_like(mask)
                (
                    empty_x_exp,
                    empty_y_exp,
                    empty_mask_exp,
                    empty_candidate_exp,
                    empty_next_mask_exp,
                    empty_remaining,
                    _,
                ) = _expand_all_candidates(xb, yb, empty_mask, remaining_budget)
                empty_candidate_onehot = _one_hot(empty_candidate_exp, n_features)
                with torch.no_grad():
                    empty_target = _rollout_terminal_ce_from_mask(
                        q_rollout_models,
                        predictor,
                        empty_x_exp,
                        empty_y_exp,
                        empty_next_mask_exp,
                        remaining_budget - 1,
                        risk_type=risk_type,
                        lambda_ptrue=lambda_ptrue,
                    )
                empty_pred = q_model(
                    empty_x_exp * empty_mask_exp,
                    empty_mask_exp,
                    empty_candidate_onehot,
                    empty_remaining,
                )
                empty_loss = loss_fn(empty_pred, empty_target)
                loss = loss + empty_mask_weight * empty_loss

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item() * x_exp.size(0)
            total += x_exp.size(0)
            progress.set_postfix(loss=total_loss / total)

        print("r={} epoch {:02d} q_loss={:.6f}".format(remaining_budget, epoch, total_loss / total))


def train_q_briga(
    predictor: nn.Module,
    train_tensors: Tuple[torch.Tensor, torch.Tensor],
    n_features: int,
    checkpoint_path: str = "checkpoints/q_briga.pt",
    budgets: Iterable[int] = (1, 2, 3, 4, 5),
    epochs: int = 6,
    batch_size: int = 256,
    lr: float = 1e-3,
    seed: int = 0,
    device: str = None,
    state_sampling_mode: str = "generic",
    feature_order: Optional[List[int]] = None,
    use_learned_rollin: bool = False,
    risk_type: str = "ce",
    lambda_ptrue: float = 0.5,
    q_hidden_dim: int = 128,
    q_n_hidden: int = 2,
    q_activation: str = "relu",
    q_norm: str = "none",
    q_dropout: float = 0.0,
    ranking_loss_weight: float = 0.0,
    use_empty_rollout_target: bool = True,
    empty_mask_weight: float = 1.0,
    method_version: str = "budget_specific_bellman_v1",
    dataset_name: str = "unknown",
    input_transform: str = "unknown",
    eval_budgets: Optional[Iterable[int]] = None,
    predictor_checkpoint: str = "",
    low_budget_threshold: int = 0,
    low_budget_epochs: Optional[int] = None,
) -> Dict[int, QNetwork]:
    torch.manual_seed(seed)
    risk_type = _validate_risk_type(risk_type)
    if ranking_loss_weight < 0.0:
        raise ValueError("ranking_loss_weight must be non-negative")
    if low_budget_epochs is not None and int(low_budget_epochs) < 1:
        raise ValueError("low_budget_epochs must be positive when provided")
    if int(low_budget_threshold) < 0:
        raise ValueError("low_budget_threshold must be non-negative")
    if float(empty_mask_weight) < 0.0:
        raise ValueError("empty_mask_weight must be non-negative")
    if state_sampling_mode not in {"generic", "synthetic_guided"}:
        raise ValueError("state_sampling_mode must be 'generic' or 'synthetic_guided'")

    train_budgets = _normalize_train_budgets(budgets, n_features)
    reported_eval_budgets = (
        [int(b) for b in eval_budgets]
        if eval_budgets is not None
        else list(train_budgets)
    )
    for b in reported_eval_budgets:
        if int(b) < 1 or int(b) > int(train_budgets[-1]):
            raise ValueError(
                "eval budget {} must be within trained budget range 1..{}".format(
                    int(b), int(train_budgets[-1])
                )
            )

    device_obj = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
    predictor = predictor.to(device_obj)
    predictor.eval()
    for param in predictor.parameters():
        param.requires_grad_(False)

    train_x, train_y = train_tensors
    loader = DataLoader(TensorDataset(train_x, train_y), batch_size=batch_size, shuffle=True)
    effective_epochs = max(int(epochs), 8)
    q_models: Dict[int, QNetwork] = {}

    for remaining_budget in train_budgets:
        q_model = QNetwork(
            n_features=n_features,
            hidden_dim=int(q_hidden_dim),
            n_hidden=int(q_n_hidden),
            activation=q_activation,
            norm=q_norm,
            dropout=float(q_dropout),
        ).to(device_obj)
        optimizer = torch.optim.Adam(q_model.parameters(), lr=lr)
        q_prev = q_models.get(remaining_budget - 1)
        if remaining_budget > 1 and q_prev is None:
            raise RuntimeError("missing Q_{} while fitting Q_{}".format(remaining_budget - 1, remaining_budget))
        if q_prev is not None:
            q_prev.eval()
            for param in q_prev.parameters():
                param.requires_grad_(False)

        _train_one_budget_level(
            q_model=q_model,
            q_prev=q_prev,
            q_rollout_models=q_models,
            predictor=predictor,
            loader=loader,
            optimizer=optimizer,
            n_features=n_features,
            remaining_budget=remaining_budget,
            epochs=(
                int(low_budget_epochs)
                if low_budget_epochs is not None and remaining_budget <= int(low_budget_threshold)
                else effective_epochs
            ),
            device=device_obj,
            state_sampling_mode=state_sampling_mode,
            feature_order=feature_order,
            use_learned_rollin=use_learned_rollin,
            risk_type=risk_type,
            lambda_ptrue=float(lambda_ptrue),
            ranking_loss_weight=float(ranking_loss_weight),
            empty_mask_weight=float(empty_mask_weight),
            use_empty_rollout_target=bool(use_empty_rollout_target),
        )

        q_model.eval()
        for param in q_model.parameters():
            param.requires_grad_(False)
        q_models[remaining_budget] = q_model

    checkpoint_dir = os.path.dirname(checkpoint_path)
    if checkpoint_dir:
        os.makedirs(checkpoint_dir, exist_ok=True)
    torch.save(
        {
            "n_features": n_features,
            "risk_type": risk_type,
            "lambda_ptrue": float(lambda_ptrue),
            "q_architecture": {
                "hidden_dim": int(q_hidden_dim),
                "n_hidden": int(q_n_hidden),
                "activation": str(q_activation),
                "norm": str(q_norm),
                "dropout": float(q_dropout),
            },
            "method_version": str(method_version),
            "dataset_name": str(dataset_name),
            "input_transform": str(input_transform),
            "predictor_checkpoint": str(predictor_checkpoint),
            "train_budgets": list(train_budgets),
            "eval_budgets": list(reported_eval_budgets),
            "state_sampling_mode": str(state_sampling_mode),
            "feature_order_used": feature_order is not None,
            "use_learned_rollin": bool(use_learned_rollin),
            "use_empty_rollout_target": bool(use_empty_rollout_target),
            "empty_mask_weight": float(empty_mask_weight),
            "ranking_loss_weight": float(ranking_loss_weight),
            "low_budget_threshold": int(low_budget_threshold),
            "low_budget_epochs": None if low_budget_epochs is None else int(low_budget_epochs),
            "epochs_requested": int(epochs),
            "epochs_effective_default": int(effective_epochs),
            "batch_size": int(batch_size),
            "lr": float(lr),
            "seed": int(seed),
            "budget_state_dicts": {str(budget): model.state_dict() for budget, model in q_models.items()},
        },
        checkpoint_path,
    )
    print("saved q_briga checkpoint to {}".format(checkpoint_path))
    return q_models
