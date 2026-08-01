"""Run a lightweight selected-pixel Fashion-MNIST BRiG-AFA probe."""

import argparse
import math
import os
import sys
from typing import Dict, List

import pandas as pd
import torch

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from briga.data.fashion_mnist_selected import make_fashion_mnist_selected_data
from briga.eval.baselines import compute_global_mi_order, global_mi_policy, random_policy
from briga.eval.evaluate import evaluate_selected_features
from briga.eval.learned_briga import learned_briga_policy, myopic_q_policy
from briga.training.train_predictor import train_partial_input_predictor
from briga.training.train_q_briga import train_q_briga


DATASET_NAME = "fashion_mnist_selected"
DEFAULT_CACHE_DIR = os.path.expanduser(
    os.environ.get("BRIGA_DATA_CACHE", "~/.cache/briga_afa/torchvision")
)
DEFAULT_EVAL_BUDGETS = (1, 2, 4, 8, 12, 16, 20)


def all_features_policy(x: torch.Tensor) -> torch.Tensor:
    order = torch.arange(x.size(1), dtype=torch.long)
    return order.unsqueeze(0).repeat(x.size(0), 1)


def empty_features_policy(x: torch.Tensor) -> torch.Tensor:
    return torch.empty(x.size(0), 0, dtype=torch.long)


def label_balance(y: torch.Tensor) -> Dict[int, float]:
    values, counts = torch.unique(y, return_counts=True)
    total = float(y.numel())
    return {int(v.item()): float(c.item()) / total for v, c in zip(values, counts)}


def average_random_policy_metrics(
    predictor,
    x: torch.Tensor,
    y: torch.Tensor,
    budget: int,
    seed: int,
    random_repeats: int,
    batch_size: int,
) -> Dict[str, float]:
    metrics = []
    for repeat in range(int(random_repeats)):
        selected = random_policy(
            x,
            budget,
            seed=12000 + 1000 * int(seed) + 100 * int(budget) + repeat,
        )
        metrics.append(
            evaluate_selected_features(
                predictor,
                x,
                y,
                selected,
                batch_size=batch_size,
            )
        )

    return {
        "accuracy": sum(item["accuracy"] for item in metrics) / float(random_repeats),
        "cross_entropy": sum(item["cross_entropy"] for item in metrics) / float(random_repeats),
    }


def add_metric_row(
    rows: List[Dict[str, object]],
    seed: int,
    policy: str,
    budget: int,
    metrics: Dict[str, float],
    predictor_info: Dict[str, float],
    n_train: int,
    n_val: int,
    n_test: int,
    n_features: int,
    n_classes: int,
) -> None:
    rows.append(
        {
            "dataset": DATASET_NAME,
            "seed": int(seed),
            "policy": policy,
            "budget": int(budget),
            "accuracy": float(metrics["accuracy"]),
            "cross_entropy": float(metrics["cross_entropy"]),
            "predictor_val_acc_random_masks": float(
                predictor_info.get("val_acc_random_masks", float("nan"))
            ),
            "predictor_val_loss_random_masks": float(
                predictor_info.get("val_loss_random_masks", float("nan"))
            ),
            "n_train": int(n_train),
            "n_val": int(n_val),
            "n_test": int(n_test),
            "n_features": int(n_features),
            "n_classes": int(n_classes),
        }
    )


def standard_error(values: pd.Series) -> float:
    count = int(values.count())
    if count <= 1:
        return 0.0
    return float(values.std(ddof=1) / math.sqrt(count))


def make_summary(raw_df: pd.DataFrame) -> pd.DataFrame:
    grouped = raw_df.groupby(["policy", "budget"], as_index=False)
    summary = grouped.agg(
        dataset=("dataset", "first"),
        accuracy_mean=("accuracy", "mean"),
        accuracy_se=("accuracy", standard_error),
        cross_entropy_mean=("cross_entropy", "mean"),
        cross_entropy_se=("cross_entropy", standard_error),
        n_seeds=("seed", "nunique"),
        n_features=("n_features", "first"),
        n_classes=("n_classes", "first"),
    )

    policy_order = {
        "empty_mask": 0,
        "full_features": 1,
        "random": 2,
        "global_mi": 3,
        "myopic_q": 4,
        "learned_briga": 5,
    }
    summary["policy_order"] = summary["policy"].map(policy_order).fillna(99).astype(int)
    summary = summary.sort_values(["budget", "policy_order"]).drop(columns=["policy_order"])
    return summary


def make_comparisons(summary_df: pd.DataFrame) -> pd.DataFrame:
    available = summary_df[summary_df["policy"].isin(["learned_briga", "myopic_q", "global_mi"])]
    rows = []

    for budget, group in available.groupby("budget"):
        by_policy = group.set_index("policy")
        if "learned_briga" not in by_policy.index:
            continue

        learned_acc = float(by_policy.loc["learned_briga", "accuracy_mean"])
        learned_ce = float(by_policy.loc["learned_briga", "cross_entropy_mean"])
        myopic_acc = (
            float(by_policy.loc["myopic_q", "accuracy_mean"])
            if "myopic_q" in by_policy.index
            else float("nan")
        )
        global_acc = (
            float(by_policy.loc["global_mi", "accuracy_mean"])
            if "global_mi" in by_policy.index
            else float("nan")
        )
        myopic_ce = (
            float(by_policy.loc["myopic_q", "cross_entropy_mean"])
            if "myopic_q" in by_policy.index
            else float("nan")
        )
        global_ce = (
            float(by_policy.loc["global_mi", "cross_entropy_mean"])
            if "global_mi" in by_policy.index
            else float("nan")
        )

        rows.append(
            {
                "budget": int(budget),
                "learned_minus_myopic_accuracy": learned_acc - myopic_acc,
                "learned_minus_global_mi_accuracy": learned_acc - global_acc,
                "learned_minus_myopic_ce": learned_ce - myopic_ce,
                "learned_minus_global_ce": learned_ce - global_ce,
            }
        )

    return pd.DataFrame(rows)


def run_one_seed(args: argparse.Namespace, seed: int) -> List[Dict[str, object]]:
    print("\n" + "=" * 90)
    print("Fashion-MNIST selected-pixel seed:", seed)
    print("=" * 90)

    torch.manual_seed(seed)
    data = make_fashion_mnist_selected_data(
        seed=seed,
        cache_dir=DEFAULT_CACHE_DIR,
        download=True,
    )

    train_x, train_y = data["train_tensors"]
    val_x, val_y = data["val_tensors"]
    test_x, test_y = data["test_tensors"]
    metadata = data["metadata"]

    n_train = int(train_x.size(0))
    n_val = int(val_x.size(0))
    n_test = int(test_x.size(0))
    n_features = int(data["n_features"])
    n_classes = int(data["n_classes"])

    print("cache_dir:", metadata.get("cache_dir"))
    print("selected_pixel_indices:", metadata.get("selected_pixel_indices"))
    print("n_train:", n_train)
    print("n_val:", n_val)
    print("n_test:", n_test)
    print("n_features:", n_features)
    print("n_classes:", n_classes)
    print("train label balance:", label_balance(train_y))
    print("val label balance:", label_balance(val_y))
    print("test label balance:", label_balance(test_y))

    os.makedirs("checkpoints", exist_ok=True)
    predictor_ckpt = "checkpoints/fashion_mnist_selected_predictor_seed{}.pt".format(seed)
    q_ckpt = "checkpoints/q_briga_fashion_mnist_selected_seed{}.pt".format(seed)

    predictor, predictor_info = train_partial_input_predictor(
        train_tensors=data["train_tensors"],
        val_tensors=data["val_tensors"],
        n_features=n_features,
        n_classes=n_classes,
        checkpoint_path=predictor_ckpt,
        epochs=int(args.predictor_epochs),
        batch_size=int(args.batch_size),
        lr=float(args.lr),
        seed=seed,
        structured_fraction=0.0,
    )
    print("predictor_info:", predictor_info)

    feature_order = compute_global_mi_order(val_x, val_y)
    print("global_mi_order_first:", [int(v) for v in feature_order[: min(16, len(feature_order))]])

    eval_budgets = tuple(int(b) for b in args.budgets)
    if max(eval_budgets) > n_features:
        raise ValueError("budgets must not exceed n_features={}".format(n_features))
    train_budgets = tuple(range(1, max(eval_budgets) + 1))

    q_models = train_q_briga(
        predictor=predictor,
        train_tensors=data["train_tensors"],
        n_features=n_features,
        checkpoint_path=q_ckpt,
        budgets=train_budgets,
        epochs=int(args.q_epochs),
        batch_size=int(args.batch_size),
        lr=float(args.lr),
        seed=seed,
        state_sampling_mode="generic",
    )

    rows: List[Dict[str, object]] = []

    empty_metrics = evaluate_selected_features(
        predictor,
        test_x,
        test_y,
        empty_features_policy(test_x),
        batch_size=int(args.batch_size),
    )
    add_metric_row(
        rows,
        seed,
        "empty_mask",
        0,
        empty_metrics,
        predictor_info,
        n_train,
        n_val,
        n_test,
        n_features,
        n_classes,
    )

    full_metrics = evaluate_selected_features(
        predictor,
        test_x,
        test_y,
        all_features_policy(test_x),
        batch_size=int(args.batch_size),
    )
    add_metric_row(
        rows,
        seed,
        "full_features",
        n_features,
        full_metrics,
        predictor_info,
        n_train,
        n_val,
        n_test,
        n_features,
        n_classes,
    )

    for budget in eval_budgets:
        print("--- seed {} budget {} ---".format(seed, budget))

        random_metrics = average_random_policy_metrics(
            predictor,
            test_x,
            test_y,
            budget,
            seed,
            int(args.random_repeats),
            batch_size=int(args.batch_size),
        )
        add_metric_row(
            rows,
            seed,
            "random",
            budget,
            random_metrics,
            predictor_info,
            n_train,
            n_val,
            n_test,
            n_features,
            n_classes,
        )
        print(
            "random        k={} acc={:.4f} ce={:.4f}".format(
                budget,
                random_metrics["accuracy"],
                random_metrics["cross_entropy"],
            )
        )

        global_selected = global_mi_policy(test_x, budget, feature_order)
        global_metrics = evaluate_selected_features(
            predictor,
            test_x,
            test_y,
            global_selected,
            batch_size=int(args.batch_size),
        )
        add_metric_row(
            rows,
            seed,
            "global_mi",
            budget,
            global_metrics,
            predictor_info,
            n_train,
            n_val,
            n_test,
            n_features,
            n_classes,
        )
        print(
            "global_mi     k={} acc={:.4f} ce={:.4f}".format(
                budget,
                global_metrics["accuracy"],
                global_metrics["cross_entropy"],
            )
        )

        myopic_selected = myopic_q_policy(q_models, test_x, budget)
        myopic_metrics = evaluate_selected_features(
            predictor,
            test_x,
            test_y,
            myopic_selected,
            batch_size=int(args.batch_size),
        )
        add_metric_row(
            rows,
            seed,
            "myopic_q",
            budget,
            myopic_metrics,
            predictor_info,
            n_train,
            n_val,
            n_test,
            n_features,
            n_classes,
        )
        print(
            "myopic_q      k={} acc={:.4f} ce={:.4f}".format(
                budget,
                myopic_metrics["accuracy"],
                myopic_metrics["cross_entropy"],
            )
        )

        briga_selected = learned_briga_policy(q_models, test_x, budget)
        briga_metrics = evaluate_selected_features(
            predictor,
            test_x,
            test_y,
            briga_selected,
            batch_size=int(args.batch_size),
        )
        add_metric_row(
            rows,
            seed,
            "learned_briga",
            budget,
            briga_metrics,
            predictor_info,
            n_train,
            n_val,
            n_test,
            n_features,
            n_classes,
        )
        print(
            "learned_briga k={} acc={:.4f} ce={:.4f}".format(
                budget,
                briga_metrics["accuracy"],
                briga_metrics["cross_entropy"],
            )
        )

    return rows


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seeds", type=int, nargs="+", default=[1, 3, 5, 7, 9])
    parser.add_argument("--budgets", type=int, nargs="+", default=list(DEFAULT_EVAL_BUDGETS))
    parser.add_argument("--predictor-epochs", type=int, default=8)
    parser.add_argument("--q-epochs", type=int, default=4)
    parser.add_argument("--batch-size", type=int, default=2048)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--random-repeats", type=int, default=10)
    parser.add_argument(
        "--raw-path",
        type=str,
        default="results/fashion_mnist_selected_probe_raw.csv",
    )
    parser.add_argument(
        "--summary-path",
        type=str,
        default="results/fashion_mnist_selected_probe_summary.csv",
    )
    parser.add_argument(
        "--comparison-path",
        type=str,
        default="results/fashion_mnist_selected_probe_key_comparisons.csv",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    for path in [args.raw_path, args.summary_path, args.comparison_path]:
        directory = os.path.dirname(path)
        if directory:
            os.makedirs(directory, exist_ok=True)

    print("seeds:", args.seeds)
    print("budgets:", args.budgets)
    print("batch_size:", args.batch_size)
    print("predictor_epochs:", args.predictor_epochs)
    print("q_epochs:", args.q_epochs)

    rows: List[Dict[str, object]] = []
    for seed in args.seeds:
        rows.extend(run_one_seed(args, int(seed)))
        pd.DataFrame(rows).to_csv(args.raw_path, index=False)

    raw_df = pd.DataFrame(rows)
    summary_df = make_summary(raw_df)
    comparison_df = make_comparisons(summary_df)

    raw_df.to_csv(args.raw_path, index=False)
    summary_df.to_csv(args.summary_path, index=False)
    comparison_df.to_csv(args.comparison_path, index=False)

    print("\nsaved raw results to {}".format(args.raw_path))
    print("saved summary results to {}".format(args.summary_path))
    print("saved key comparisons to {}".format(args.comparison_path))
    print("\n=== fashion_mnist_selected_probe_summary.csv ===")
    print(summary_df.to_csv(index=False), end="")
    print("\n=== fashion_mnist_selected_probe_key_comparisons.csv ===")
    print(comparison_df.to_csv(index=False), end="")


if __name__ == "__main__":
    main()
