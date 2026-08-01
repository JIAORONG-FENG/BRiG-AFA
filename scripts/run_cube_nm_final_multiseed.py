"""Final multi-seed CUBE-NM-style controlled synthetic BRiG experiment."""

import argparse
import os
import sys
from typing import Dict, Iterable, List

import pandas as pd
import torch

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from briga.data.synthetic_cube_nm import make_synthetic_cube_nm_data
from briga.eval.baselines import compute_global_mi_order, global_mi_policy, random_policy
from briga.eval.evaluate import evaluate_selected_features
from briga.eval.learned_briga import learned_briga_policy, myopic_q_policy
from briga.training.train_predictor import train_partial_input_predictor
from briga.training.train_q_briga import train_q_briga


DATASET_NAME = "cube_nm_final"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seeds", type=int, nargs="+", default=[1, 3, 5, 7, 9])
    parser.add_argument("--budgets", type=int, nargs="+", default=[1, 2, 3, 5, 8])
    parser.add_argument("--n-train", type=int, default=6000)
    parser.add_argument("--n-val", type=int, default=2000)
    parser.add_argument("--n-test", type=int, default=2000)
    parser.add_argument("--noise-std", type=float, default=0.25)
    parser.add_argument("--predictor-epochs", type=int, default=12)
    parser.add_argument("--q-epochs", type=int, default=6)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--random-repeats", type=int, default=10)
    parser.add_argument("--raw-path", type=str, default="results/cube_nm_final_multiseed_raw.csv")
    parser.add_argument("--summary-path", type=str, default="results/cube_nm_final_multiseed_summary.csv")
    parser.add_argument("--trajectory-path", type=str, default="results/cube_nm_final_trajectory_summary.csv")
    return parser.parse_args()


def _dedupe_order(order: List[int], n_features: int) -> List[int]:
    seen = set()
    result = []
    for idx in order + list(range(n_features)):
        idx = int(idx)
        if idx not in seen:
            seen.add(idx)
            result.append(idx)
    return result


def cube_context_first_policy(x: torch.Tensor, k: int, metadata: Dict[str, object]) -> torch.Tensor:
    context = int(metadata["context_feature"])
    global_features = list(metadata["global_informative"])
    branch0 = list(metadata["regime0_informative"])
    branch1 = list(metadata["regime1_informative"])
    distractors = list(metadata["distractor_features"])
    n_features = x.size(1)

    rows = []
    inferred_branch = (x[:, context] > 0).long()
    for branch in inferred_branch.tolist():
        primary = branch1 if int(branch) == 1 else branch0
        secondary = branch0 if int(branch) == 1 else branch1
        order = _dedupe_order([context] + primary + global_features + secondary + distractors, n_features)
        rows.append(order[:k])
    return torch.tensor(rows, dtype=torch.long)


def cube_regime_oracle_policy(
    x: torch.Tensor,
    regimes: torch.Tensor,
    k: int,
    metadata: Dict[str, object],
) -> torch.Tensor:
    context = int(metadata["context_feature"])
    global_features = list(metadata["global_informative"])
    branch0 = list(metadata["regime0_informative"])
    branch1 = list(metadata["regime1_informative"])
    distractors = list(metadata["distractor_features"])
    n_features = x.size(1)

    rows = []
    for branch in regimes.tolist():
        primary = branch1 if int(branch) == 1 else branch0
        secondary = branch0 if int(branch) == 1 else branch1
        order = _dedupe_order(primary + global_features + [context] + secondary + distractors, n_features)
        rows.append(order[:k])
    return torch.tensor(rows, dtype=torch.long)


def context_first_rate(selected: torch.Tensor) -> float:
    if selected.size(1) == 0:
        return 0.0
    return float((selected[:, 0] == 0).float().mean().item())


def relevant_feature_rate(selected: torch.Tensor, regimes: torch.Tensor, metadata: Dict[str, object]) -> float:
    if selected.numel() == 0:
        return 0.0
    global_features = set(int(i) for i in metadata["global_informative"])
    branch0 = set(int(i) for i in metadata["regime0_informative"])
    branch1 = set(int(i) for i in metadata["regime1_informative"])
    relevant = 0
    for row, regime in zip(selected.tolist(), regimes.tolist()):
        branch_features = branch1 if int(regime) == 1 else branch0
        useful = global_features | branch_features
        relevant += sum(1 for idx in row if int(idx) in useful)
    return float(relevant / selected.numel())


def add_trajectory_rows(
    rows: List[Dict[str, object]],
    policy: str,
    budget: int,
    selected: torch.Tensor,
    regimes: torch.Tensor,
    n_features: int,
    seed: int,
) -> None:
    for step in range(selected.size(1)):
        step_values = selected[:, step]
        for feature in range(n_features):
            rows.append(
                {
                    "policy": policy,
                    "budget": int(budget),
                    "scope": "all",
                    "regime": "all",
                    "step": step + 1,
                    "feature": feature,
                    "frequency": float((step_values == feature).float().mean().item()),
                    "seed": int(seed),
                }
            )

    for regime_value in [0, 1]:
        regime_mask = regimes == regime_value
        if not bool(regime_mask.any()):
            continue
        regime_selected = selected[regime_mask]
        for feature in range(n_features):
            rows.append(
                {
                    "policy": policy,
                    "budget": int(budget),
                    "scope": "selected_any_step",
                    "regime": int(regime_value),
                    "step": "any",
                    "feature": feature,
                    "frequency": float((regime_selected == feature).float().mean().item()),
                    "seed": int(seed),
                }
            )


def evaluate_policy(
    predictor,
    x: torch.Tensor,
    y: torch.Tensor,
    regimes: torch.Tensor,
    selected: torch.Tensor,
    policy: str,
    budget: int,
    metadata: Dict[str, object],
) -> Dict[str, object]:
    metrics = evaluate_selected_features(predictor, x, y, selected)
    return {
        "policy": policy,
        "budget": int(budget),
        "accuracy": metrics["accuracy"],
        "cross_entropy": metrics["cross_entropy"],
        "context_first_rate": context_first_rate(selected),
        "relevant_feature_rate": relevant_feature_rate(selected, regimes, metadata),
    }


def average_random_policy(
    predictor,
    x: torch.Tensor,
    y: torch.Tensor,
    regimes: torch.Tensor,
    budget: int,
    metadata: Dict[str, object],
    random_repeats: int,
    trajectory_rows: List[Dict[str, object]],
    seed: int,
) -> Dict[str, object]:
    metric_rows = []
    selected_rows = []
    for repeat in range(int(random_repeats)):
        selected = random_policy(x, budget, seed=5000 + 1000 * int(seed) + 100 * int(budget) + repeat)
        selected_rows.append(selected)
        metric_rows.append(evaluate_policy(predictor, x, y, regimes, selected, "random", budget, metadata))

    stacked_selected = torch.cat(selected_rows, dim=0)
    stacked_regimes = regimes.repeat(int(random_repeats))
    add_trajectory_rows(trajectory_rows, "random", budget, stacked_selected, stacked_regimes, x.size(1), seed)

    return {
        "policy": "random",
        "budget": int(budget),
        "accuracy": sum(row["accuracy"] for row in metric_rows) / float(random_repeats),
        "cross_entropy": sum(row["cross_entropy"] for row in metric_rows) / float(random_repeats),
        "context_first_rate": sum(row["context_first_rate"] for row in metric_rows) / float(random_repeats),
        "relevant_feature_rate": sum(row["relevant_feature_rate"] for row in metric_rows) / float(random_repeats),
    }


def add_common_columns(
    row: Dict[str, object],
    seed: int,
    n_train: int,
    n_val: int,
    n_test: int,
    n_features: int,
    n_classes: int,
    noise_std: float,
) -> Dict[str, object]:
    result = {
        "dataset": DATASET_NAME,
        "seed": int(seed),
        **row,
        "n_train": int(n_train),
        "n_val": int(n_val),
        "n_test": int(n_test),
        "n_features": int(n_features),
        "n_classes": int(n_classes),
        "noise_std": float(noise_std),
    }
    return result


def standard_error(series: pd.Series) -> float:
    if len(series) <= 1:
        return 0.0
    return float(series.std(ddof=1) / (len(series) ** 0.5))


def summarize(raw_df: pd.DataFrame) -> pd.DataFrame:
    grouped = raw_df.groupby(["dataset", "policy", "budget"], as_index=False)
    return grouped.agg(
        accuracy_mean=("accuracy", "mean"),
        accuracy_se=("accuracy", standard_error),
        cross_entropy_mean=("cross_entropy", "mean"),
        cross_entropy_se=("cross_entropy", standard_error),
        context_first_rate_mean=("context_first_rate", "mean"),
        context_first_rate_se=("context_first_rate", standard_error),
        relevant_feature_rate_mean=("relevant_feature_rate", "mean"),
        relevant_feature_rate_se=("relevant_feature_rate", standard_error),
        n_seeds=("seed", "nunique"),
    ).sort_values(["budget", "policy"])


def print_metric(row: Dict[str, object]) -> None:
    print(
        "seed={} policy={} k={} accuracy={:.4f} cross_entropy={:.4f} "
        "context_first_rate={:.4f} relevant_feature_rate={:.4f}".format(
            row["seed"],
            row["policy"],
            row["budget"],
            row["accuracy"],
            row["cross_entropy"],
            row["context_first_rate"],
            row["relevant_feature_rate"],
        )
    )


def main() -> None:
    args = parse_args()
    budgets = sorted(set(int(budget) for budget in args.budgets))
    max_budget = max(budgets)

    raw_rows: List[Dict[str, object]] = []
    trajectory_rows: List[Dict[str, object]] = []

    for seed in args.seeds:
        seed = int(seed)
        print("=== seed {} ===".format(seed))
        torch.manual_seed(seed)
        data = make_synthetic_cube_nm_data(
            n_train=args.n_train,
            n_val=args.n_val,
            n_test=args.n_test,
            seed=seed,
            noise_std=args.noise_std,
        )
        metadata = data["metadata"]
        n_features = int(data["n_features"])
        n_classes = int(data["n_classes"])

        predictor, _ = train_partial_input_predictor(
            train_tensors=data["train_tensors"],
            val_tensors=data["val_tensors"],
            n_features=n_features,
            n_classes=n_classes,
            checkpoint_path="checkpoints/cube_nm_final_predictor_seed{}.pt".format(seed),
            epochs=args.predictor_epochs,
            batch_size=args.batch_size,
            lr=1e-3,
            seed=seed,
            structured_fraction=0.0,
        )

        train_budgets = tuple(range(1, max_budget + 1))
        q_models = train_q_briga(
            predictor=predictor,
            train_tensors=data["train_tensors"],
            n_features=n_features,
            checkpoint_path="checkpoints/q_briga_cube_nm_final_seed{}.pt".format(seed),
            budgets=train_budgets,
            epochs=args.q_epochs,
            batch_size=args.batch_size,
            lr=1e-3,
            seed=seed,
            state_sampling_mode="generic",
        )

        val_x, val_y = data["val_tensors"]
        test_x, test_y = data["test_tensors"]
        test_regime = metadata["test_regime"]
        feature_order = compute_global_mi_order(val_x, val_y)

        empty_selected = torch.empty((test_x.size(0), 0), dtype=torch.long)
        full_selected = torch.arange(n_features, dtype=torch.long).unsqueeze(0).repeat(test_x.size(0), 1)

        for policy, budget, selected in [
            ("empty_mask", 0, empty_selected),
            ("full_features", n_features, full_selected),
        ]:
            row = evaluate_policy(predictor, test_x, test_y, test_regime, selected, policy, budget, metadata)
            raw_row = add_common_columns(
                row, seed, args.n_train, args.n_val, args.n_test, n_features, n_classes, args.noise_std
            )
            raw_rows.append(raw_row)
            print_metric(raw_row)

        for budget in budgets:
            row = average_random_policy(
                predictor=predictor,
                x=test_x,
                y=test_y,
                regimes=test_regime,
                budget=budget,
                metadata=metadata,
                random_repeats=args.random_repeats,
                trajectory_rows=trajectory_rows,
                seed=seed,
            )
            raw_row = add_common_columns(
                row, seed, args.n_train, args.n_val, args.n_test, n_features, n_classes, args.noise_std
            )
            raw_rows.append(raw_row)
            print_metric(raw_row)

            selected_by_policy = {
                "global_mi": global_mi_policy(test_x, budget, feature_order),
                "myopic_q": myopic_q_policy(q_models, test_x, budget),
                "learned_briga": learned_briga_policy(q_models, test_x, budget),
                "context_first": cube_context_first_policy(test_x, budget, metadata),
                "regime_oracle": cube_regime_oracle_policy(test_x, test_regime, budget, metadata),
            }
            for policy, selected in selected_by_policy.items():
                row = evaluate_policy(predictor, test_x, test_y, test_regime, selected, policy, budget, metadata)
                raw_row = add_common_columns(
                    row, seed, args.n_train, args.n_val, args.n_test, n_features, n_classes, args.noise_std
                )
                raw_rows.append(raw_row)
                add_trajectory_rows(trajectory_rows, policy, budget, selected, test_regime, n_features, seed)
                print_metric(raw_row)

    raw_df = pd.DataFrame(raw_rows)
    summary_df = summarize(raw_df)
    trajectory_df = pd.DataFrame(trajectory_rows)

    for path in [args.raw_path, args.summary_path, args.trajectory_path]:
        os.makedirs(os.path.dirname(path), exist_ok=True)
    raw_df.to_csv(args.raw_path, index=False)
    summary_df.to_csv(args.summary_path, index=False)
    trajectory_df.to_csv(args.trajectory_path, index=False)

    print("saved raw results to {}".format(args.raw_path))
    print("saved summary results to {}".format(args.summary_path))
    print("saved trajectory summary to {}".format(args.trajectory_path))
    print("=== cube_nm_final_multiseed_summary.csv ===")
    print(summary_df.to_csv(index=False), end="")


if __name__ == "__main__":
    main()
