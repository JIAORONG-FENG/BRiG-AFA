"""Analyze Fashion-MNIST selected-pixel acquisition behavior from checkpoints.

This script does not train models or run a new experiment. It loads an existing
selected-pixel Fashion-MNIST dataset split plus trained predictor/Q checkpoints,
computes policy selections on the test set, and writes diagnostic frequency
summaries and k=4 heatmaps.
"""

import argparse
import os
import sys
from typing import Dict, List

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from briga.data.fashion_mnist_selected import (  # noqa: E402
    SELECTED_PIXEL_INDICES,
    make_fashion_mnist_selected_data,
)
from briga.eval.baselines import compute_global_mi_order, global_mi_policy  # noqa: E402
from briga.eval.learned_briga import learned_briga_policy, myopic_q_policy  # noqa: E402
from briga.models.predictor import PartialInputMLP, predictor_architecture_from_checkpoint  # noqa: E402
from briga.models.q_network import QNetwork, q_architecture_from_checkpoint  # noqa: E402


DEFAULT_PREDICTOR_CKPT = "checkpoints/fashion_mnist_selected_predictor_seed7.pt"
DEFAULT_Q_CKPT = "checkpoints/q_briga_fashion_mnist_selected_seed7.pt"
DEFAULT_OUTPUT_CSV = "results/fashion_mnist_selected_acquisition_frequency_seed7.csv"
DEFAULT_FIGURE_DIR = "results/paper_figures"
DEFAULT_BUDGETS = (2, 4, 8)


def load_predictor(checkpoint_path: str, device: torch.device) -> PartialInputMLP:
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError("Predictor checkpoint not found: {}".format(checkpoint_path))
    checkpoint = torch.load(checkpoint_path, map_location=device)
    required = {"model_state_dict", "n_features", "n_classes"}
    missing = required.difference(checkpoint.keys())
    if missing:
        raise NotImplementedError(
            "Unsupported predictor checkpoint format at {}. Missing keys: {}. "
            "Expected train_partial_input_predictor format with model_state_dict, "
            "n_features, and n_classes.".format(checkpoint_path, sorted(missing))
        )
    architecture = predictor_architecture_from_checkpoint(checkpoint)
    model = PartialInputMLP(
        n_features=int(checkpoint["n_features"]),
        n_classes=int(checkpoint["n_classes"]),
        **architecture,
    ).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    return model


def load_q_models(checkpoint_path: str, device: torch.device) -> Dict[int, QNetwork]:
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError("Q checkpoint not found: {}".format(checkpoint_path))
    checkpoint = torch.load(checkpoint_path, map_location=device)
    required = {"n_features", "budget_state_dicts"}
    missing = required.difference(checkpoint.keys())
    if missing:
        raise NotImplementedError(
            "Unsupported Q checkpoint format at {}. Missing keys: {}. Expected "
            "train_q_briga format with n_features and budget_state_dicts.".format(
                checkpoint_path, sorted(missing)
            )
        )

    n_features = int(checkpoint["n_features"])
    state_dicts = checkpoint["budget_state_dicts"]
    if not isinstance(state_dicts, dict):
        raise NotImplementedError(
            "Unsupported Q checkpoint format at {}. budget_state_dicts must be a dict.".format(
                checkpoint_path
            )
        )

    q_models: Dict[int, QNetwork] = {}
    for budget_key, state_dict in state_dicts.items():
        budget = int(budget_key)
        model = QNetwork(n_features=n_features, **q_architecture_from_checkpoint(checkpoint)).to(device)
        model.load_state_dict(state_dict)
        model.eval()
        for param in model.parameters():
            param.requires_grad_(False)
        q_models[budget] = model
    return q_models


def summarize_selection(
    selected: torch.Tensor,
    policy: str,
    budget: int,
    selected_pixel_indices: List[int],
) -> List[Dict[str, object]]:
    selected_np = selected.cpu().numpy()
    n_samples = int(selected_np.shape[0])
    rows = []
    for feature_idx, pixel_idx in enumerate(selected_pixel_indices):
        positions = np.argwhere(selected_np == feature_idx)
        selected_count = int(positions.shape[0])
        frequency = float(selected_count / n_samples)
        mean_rank = float(np.mean(positions[:, 1] + 1)) if selected_count > 0 else float("nan")
        rows.append(
            {
                "policy": policy,
                "budget": int(budget),
                "feature_index": int(feature_idx),
                "pixel_index": int(pixel_idx),
                "pixel_row": int(pixel_idx // 28),
                "pixel_col": int(pixel_idx % 28),
                "selected_count": selected_count,
                "selection_frequency": frequency,
                "mean_acquisition_rank_if_selected": mean_rank,
            }
        )
    rows.sort(key=lambda row: (-float(row["selection_frequency"]), int(row["feature_index"])))
    for rank, row in enumerate(rows, start=1):
        row["frequency_rank"] = int(rank)
    rows.sort(key=lambda row: (row["policy"], row["budget"], row["feature_index"]))
    return rows


def save_heatmap(
    rows: pd.DataFrame,
    policy: str,
    budget: int,
    output_path: str,
) -> None:
    heatmap = np.zeros((28, 28), dtype=np.float32)
    subset = rows[(rows["policy"] == policy) & (rows["budget"] == budget)]
    for _, row in subset.iterrows():
        heatmap[int(row["pixel_row"]), int(row["pixel_col"])] = float(row["selection_frequency"])

    plt.figure(figsize=(4.2, 4.0))
    image = plt.imshow(heatmap, cmap="viridis", vmin=0.0, vmax=1.0)
    plt.colorbar(image, fraction=0.046, pad=0.04, label="Selection frequency")
    plt.title("{} k={}".format(policy, budget))
    plt.xlabel("Pixel column")
    plt.ylabel("Pixel row")
    plt.tight_layout()
    plt.savefig(output_path, dpi=200)
    plt.close()


def print_top_pixels(rows: pd.DataFrame, top_n: int) -> None:
    for (policy, budget), group in rows.groupby(["policy", "budget"]):
        top = group.sort_values(["selection_frequency", "feature_index"], ascending=[False, True]).head(top_n)
        pixels = [int(value) for value in top["pixel_index"].tolist()]
        print("policy={} budget={} top_pixels={}".format(policy, int(budget), pixels))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--predictor-checkpoint", type=str, default=DEFAULT_PREDICTOR_CKPT)
    parser.add_argument("--q-checkpoint", type=str, default=DEFAULT_Q_CKPT)
    parser.add_argument("--budgets", type=int, nargs="+", default=list(DEFAULT_BUDGETS))
    parser.add_argument("--output-csv", type=str, default=DEFAULT_OUTPUT_CSV)
    parser.add_argument("--figure-dir", type=str, default=DEFAULT_FIGURE_DIR)
    parser.add_argument("--top-n", type=int, default=8)
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--download", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    device = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))

    data = make_fashion_mnist_selected_data(seed=int(args.seed), download=bool(args.download))
    val_x, val_y = data["val_tensors"]
    test_x, _ = data["test_tensors"]
    selected_pixel_indices = [int(idx) for idx in data["metadata"].get("selected_pixel_indices", SELECTED_PIXEL_INDICES)]

    predictor = load_predictor(args.predictor_checkpoint, device)
    q_models = load_q_models(args.q_checkpoint, device)
    if int(data["n_features"]) != predictor.net[0].in_features // 2:
        raise ValueError("Dataset feature count does not match predictor checkpoint.")

    feature_order = compute_global_mi_order(val_x, val_y)
    budgets = [int(budget) for budget in args.budgets]
    rows: List[Dict[str, object]] = []

    for budget in budgets:
        if budget > int(data["n_features"]):
            raise ValueError("budget {} exceeds n_features {}".format(budget, data["n_features"]))
        if budget not in q_models:
            raise NotImplementedError(
                "Q checkpoint does not include budget {}. Available budgets: {}".format(
                    budget, sorted(q_models.keys())
                )
            )

        selections = {
            "global_mi": global_mi_policy(test_x, budget, feature_order),
            "myopic_q": myopic_q_policy(q_models, test_x, budget, device=str(device)),
            "learned_briga": learned_briga_policy(q_models, test_x, budget, device=str(device)),
        }
        for policy, selected in selections.items():
            rows.extend(summarize_selection(selected, policy, budget, selected_pixel_indices))

    output_dir = os.path.dirname(args.output_csv)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
    os.makedirs(args.figure_dir, exist_ok=True)

    frequency_df = pd.DataFrame(rows)
    frequency_df.to_csv(args.output_csv, index=False)

    heatmap_paths = {
        "learned_briga": os.path.join(args.figure_dir, "fashion_mnist_selected_learned_briga_k4_heatmap.png"),
        "myopic_q": os.path.join(args.figure_dir, "fashion_mnist_selected_myopic_q_k4_heatmap.png"),
        "global_mi": os.path.join(args.figure_dir, "fashion_mnist_selected_global_mi_k4_heatmap.png"),
    }
    if 4 in budgets:
        for policy, output_path in heatmap_paths.items():
            save_heatmap(frequency_df, policy, 4, output_path)
            print("saved heatmap:", output_path)
    else:
        print("budget k=4 not requested; heatmaps not written")

    print("saved acquisition frequency CSV:", args.output_csv)
    print_top_pixels(frequency_df, int(args.top_n))


if __name__ == "__main__":
    main()
