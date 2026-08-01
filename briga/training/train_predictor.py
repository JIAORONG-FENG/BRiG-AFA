"""Training utilities for the synthetic partial-input predictor."""

import os
from typing import Dict, Iterable, Tuple

import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm

from briga.models.predictor import PartialInputMLP


def sample_random_masks(
    batch_size: int,
    n_features: int,
    device: torch.device,
    min_budget: int = 1,
) -> torch.Tensor:
    if int(min_budget) < 0 or int(min_budget) > int(n_features):
        raise ValueError("min_budget must be in [0, n_features]")
    budgets = torch.randint(int(min_budget), n_features + 1, (batch_size,), device=device)
    scores = torch.rand(batch_size, n_features, device=device)
    order = scores.argsort(dim=1)
    ranks = order.argsort(dim=1)
    return (ranks < budgets.unsqueeze(1)).float()


def sample_budget_aware_masks(
    batch_size: int,
    n_features: int,
    device: torch.device,
    eval_budgets: Iterable[int],
    min_budget: int = 1,
) -> torch.Tensor:
    """Sample masks with observed counts biased toward evaluation budgets."""
    valid = sorted(
        set(
            int(b)
            for b in eval_budgets
            if int(min_budget) <= int(b) <= int(n_features)
        )
    )
    if not valid:
        return sample_random_masks(batch_size, n_features, device, min_budget=min_budget)

    choices = torch.tensor(valid, dtype=torch.long, device=device)
    budgets = choices[torch.randint(0, choices.numel(), (batch_size,), device=device)]
    scores = torch.rand(batch_size, n_features, device=device)
    order = scores.argsort(dim=1)
    ranks = order.argsort(dim=1)
    return (ranks < budgets.unsqueeze(1)).float()


def _set_mask_rows(mask: torch.Tensor, rows: torch.Tensor, features: torch.Tensor) -> None:
    if rows.numel() == 0 or features.numel() == 0:
        return
    mask[rows.unsqueeze(1), features.unsqueeze(0)] = 1.0


def sample_synthetic_structured_masks(x: torch.Tensor) -> torch.Tensor:
    """Sample structured masks for the synthetic context-regime task.

    This is intentionally tied to ``synthetic_context.py``: feature 0 is the
    context, features 1-3 are informative in regime 0, and features 4-6 are
    informative in regime 1.
    """
    batch_size, n_features = x.shape
    device = x.device
    mask = torch.zeros(batch_size, n_features, device=device)
    regimes = (x[:, 0] > 0).long()
    mask_types = torch.randint(0, 7, (batch_size,), device=device)

    context = torch.tensor([0], device=device, dtype=torch.long)
    regime0 = torch.tensor([1, 2, 3], device=device, dtype=torch.long)
    regime1 = torch.tensor([4, 5, 6], device=device, dtype=torch.long)
    all_informative = torch.tensor([0, 1, 2, 3, 4, 5, 6], device=device, dtype=torch.long)
    all_features = torch.arange(n_features, device=device, dtype=torch.long)

    _set_mask_rows(mask, (mask_types == 0).nonzero(as_tuple=False).view(-1), context)

    for regime_value, informative in [(0, regime0), (1, regime1)]:
        rows = ((mask_types == 1) & (regimes == regime_value)).nonzero(as_tuple=False).view(-1)
        _set_mask_rows(mask, rows, informative)

        rows = ((mask_types == 2) & (regimes == regime_value)).nonzero(as_tuple=False).view(-1)
        _set_mask_rows(mask, rows, torch.cat([context, informative]))

    _set_mask_rows(mask, (mask_types == 3).nonzero(as_tuple=False).view(-1), all_informative)
    _set_mask_rows(mask, (mask_types == 4).nonzero(as_tuple=False).view(-1), all_features)

    prefix_budgets = torch.randint(1, n_features + 1, (batch_size,), device=device)
    for i in range(batch_size):
        if mask_types[i].item() == 5:
            if regimes[i].item() == 0:
                order = [0, 1, 2, 3, 4, 5, 6] + list(range(7, n_features))
            else:
                order = [0, 4, 5, 6, 1, 2, 3] + list(range(7, n_features))
            mask[i, torch.tensor(order[: prefix_budgets[i].item()], device=device, dtype=torch.long)] = 1.0
        elif mask_types[i].item() == 6:
            if regimes[i].item() == 0:
                order = [1, 2, 3, 0, 4, 5, 6] + list(range(7, n_features))
            else:
                order = [4, 5, 6, 0, 1, 2, 3] + list(range(7, n_features))
            mask[i, torch.tensor(order[: prefix_budgets[i].item()], device=device, dtype=torch.long)] = 1.0

    return mask


def sample_mixed_masks(
    x: torch.Tensor,
    structured_fraction: float = 0.5,
    min_budget: int = 1,
    eval_budgets: Iterable[int] = (),
    budget_aware_fraction: float = 0.0,
) -> torch.Tensor:
    random_mask = sample_random_masks(x.size(0), x.size(1), x.device, min_budget=min_budget)

    if float(budget_aware_fraction) > 0.0:
        budget_mask = sample_budget_aware_masks(
            x.size(0),
            x.size(1),
            x.device,
            eval_budgets=eval_budgets,
            min_budget=min_budget,
        )
        use_budget = torch.rand(x.size(0), 1, device=x.device) < float(budget_aware_fraction)
        random_mask = torch.where(use_budget, budget_mask, random_mask)

    if structured_fraction <= 0.0:
        return random_mask
    structured_mask = sample_synthetic_structured_masks(x)
    use_structured = torch.rand(x.size(0), 1, device=x.device) < structured_fraction
    return torch.where(use_structured, structured_mask, random_mask)


def accuracy_from_logits(logits: torch.Tensor, y: torch.Tensor) -> float:
    return (logits.argmax(dim=1) == y).float().mean().item()


@torch.no_grad()
def evaluate_predictor(
    model: nn.Module,
    x: torch.Tensor,
    y: torch.Tensor,
    batch_size: int,
    device: torch.device,
    min_budget: int = 1,
) -> Tuple[float, float]:
    model.eval()
    loader = DataLoader(TensorDataset(x, y), batch_size=batch_size, shuffle=False)
    criterion = nn.CrossEntropyLoss(reduction="sum")
    total_loss = 0.0
    total_correct = 0
    total = 0
    for xb, yb in loader:
        xb = xb.to(device)
        yb = yb.to(device)
        mask = sample_random_masks(xb.size(0), xb.size(1), device, min_budget=min_budget)
        logits = model(xb * mask, mask)
        total_loss += criterion(logits, yb).item()
        total_correct += (logits.argmax(dim=1) == yb).sum().item()
        total += yb.numel()
    return total_correct / total, total_loss / total


def train_partial_input_predictor(
    train_tensors: Tuple[torch.Tensor, torch.Tensor],
    val_tensors: Tuple[torch.Tensor, torch.Tensor],
    n_features: int,
    n_classes: int,
    checkpoint_path: str = "checkpoints/predictor.pt",
    epochs: int = 20,
    batch_size: int = 256,
    lr: float = 1e-3,
    seed: int = 0,
    device: str = None,
    structured_fraction: float = 0.5,
    hidden_dim: int = 128,
    n_hidden: int = 2,
    dropout: float = 0.0,
    activation: str = "relu",
    norm: str = "none",
    weight_decay: float = 0.0,
    min_budget: int = 1,
    eval_budgets: Iterable[int] = (),
    budget_aware_fraction: float = 0.0,
) -> Tuple[PartialInputMLP, Dict[str, float]]:
    torch.manual_seed(seed)
    device_obj = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
    model = PartialInputMLP(
        n_features=n_features,
        n_classes=n_classes,
        hidden_dim=hidden_dim,
        n_hidden=n_hidden,
        dropout=dropout,
        activation=activation,
        norm=norm,
    ).to(device_obj)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    criterion = nn.CrossEntropyLoss()
    train_x, train_y = train_tensors
    val_x, val_y = val_tensors
    loader = DataLoader(TensorDataset(train_x, train_y), batch_size=batch_size, shuffle=True)

    for epoch in range(1, epochs + 1):
        model.train()
        total_correct = 0
        total_loss = 0.0
        total = 0
        progress = tqdm(loader, desc="train predictor epoch {}/{}".format(epoch, epochs), leave=False)
        for xb, yb in progress:
            xb = xb.to(device_obj)
            yb = yb.to(device_obj)
            mask = sample_mixed_masks(
                xb,
                structured_fraction=structured_fraction,
                min_budget=min_budget,
                eval_budgets=eval_budgets,
                budget_aware_fraction=budget_aware_fraction,
            )
            logits = model(xb * mask, mask)
            loss = criterion(logits, yb)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item() * yb.numel()
            total_correct += (logits.argmax(dim=1) == yb).sum().item()
            total += yb.numel()
            progress.set_postfix(loss=total_loss / total, acc=total_correct / total)

        val_acc, val_loss = evaluate_predictor(model, val_x, val_y, batch_size, device_obj, min_budget=min_budget)
        print(
            "epoch {:02d} train_acc={:.4f} train_loss={:.4f} val_acc={:.4f} val_loss={:.4f}".format(
                epoch, total_correct / total, total_loss / total, val_acc, val_loss
            )
        )

    checkpoint = {"model_state_dict": model.state_dict()}
    checkpoint.update(model.architecture_metadata())
    checkpoint.update(
        {
            "structured_fraction": float(structured_fraction),
            "min_budget": int(min_budget),
            "eval_budgets": [int(b) for b in eval_budgets],
            "budget_aware_fraction": float(budget_aware_fraction),
            "epochs": int(epochs),
            "batch_size": int(batch_size),
            "lr": float(lr),
            "weight_decay": float(weight_decay),
            "seed": int(seed),
        }
    )

    os.makedirs(os.path.dirname(checkpoint_path), exist_ok=True)
    torch.save(checkpoint, checkpoint_path)
    print("saved checkpoint to {}".format(checkpoint_path))
    metrics = {"val_acc_random_masks": val_acc, "val_loss_random_masks": val_loss}
    return model, metrics
