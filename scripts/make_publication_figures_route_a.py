#!/usr/bin/env python
"""Generate publication-style Route-A figures and compact result tables."""

from __future__ import annotations

import math
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results"
OUTPUT = RESULTS / "publication_route_a"
FIGURES = OUTPUT / "figures"
TABLES = OUTPUT / "tables"

LABELS = {
    "random": "Random",
    "global_mi": "Global MI",
    "myopic_q": "Myopic Q",
    "learned_briga": "BRiG-AFA",
    "context_first": "Context-first reference",
    "regime_oracle": "Regime oracle",
}

# Okabe-Ito-inspired palette plus distinct line styles and markers.
STYLES = {
    "random": dict(color="#7F7F7F", marker="o", linestyle=":"),
    "global_mi": dict(color="#E69F00", marker="s", linestyle="--"),
    "myopic_q": dict(color="#0072B2", marker="^", linestyle="-."),
    "learned_briga": dict(color="#CC79A7", marker="D", linestyle="-"),
    "context_first": dict(color="#009E73", marker="v", linestyle="--"),
    "regime_oracle": dict(color="#000000", marker="P", linestyle=":"),
}


def configure_style() -> None:
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 8.5,
            "axes.labelsize": 9,
            "axes.titlesize": 9.5,
            "legend.fontsize": 7.5,
            "xtick.labelsize": 8,
            "ytick.labelsize": 8,
            "axes.linewidth": 0.8,
            "lines.linewidth": 1.8,
            "lines.markersize": 5,
            "figure.dpi": 150,
            "savefig.dpi": 300,
            "savefig.bbox": "tight",
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )


def save_figure(fig: plt.Figure, stem: str) -> None:
    FIGURES.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIGURES / f"{stem}.pdf")
    fig.savefig(FIGURES / f"{stem}.png")
    plt.close(fig)


def paired_difference(
    raw: pd.DataFrame,
    lhs: str = "learned_briga",
    rhs: str = "myopic_q",
    metric: str = "accuracy",
) -> pd.DataFrame:
    pivot = raw.pivot_table(index=["seed", "budget"], columns="policy", values=metric)
    pivot = pivot.dropna(subset=[lhs, rhs]).copy()
    pivot["difference"] = pivot[lhs] - pivot[rhs]
    grouped = pivot.groupby("budget")["difference"]
    result = grouped.agg(["mean", "count"]).reset_index()
    result["se"] = grouped.std(ddof=1).to_numpy() / np.sqrt(result["count"])
    result["se"] = result["se"].fillna(0.0)
    return result


def plot_policy_curves(
    ax: plt.Axes,
    summary: pd.DataFrame,
    policies: list[str],
    metric: str,
    se_metric: str,
    ylabel: str,
) -> None:
    for policy in policies:
        frame = summary[summary["policy"] == policy].sort_values("budget")
        if frame.empty:
            continue
        style = STYLES[policy]
        ax.errorbar(
            frame["budget"],
            frame[metric],
            yerr=frame[se_metric] if se_metric in frame else None,
            capsize=2.5,
            capthick=0.8,
            elinewidth=0.9,
            markeredgecolor="white",
            markeredgewidth=0.45,
            label=LABELS[policy],
            **style,
        )
    ax.set_xlabel("Acquisition budget")
    ax.set_ylabel(ylabel)
    ax.grid(axis="y", color="#D9D9D9", linewidth=0.6, alpha=0.7)
    ax.spines[["top", "right"]].set_visible(False)


def plot_paired_gain(ax: plt.Axes, raw: pd.DataFrame, title: str) -> pd.DataFrame:
    gain = paired_difference(raw)
    style = STYLES["learned_briga"]
    ax.axhline(0.0, color="#555555", linewidth=0.9)
    ax.errorbar(
        gain["budget"],
        100.0 * gain["mean"],
        yerr=100.0 * gain["se"],
        capsize=3,
        capthick=0.9,
        elinewidth=1.0,
        markeredgecolor="white",
        markeredgewidth=0.45,
        **style,
    )
    ax.set_xlabel("Acquisition budget")
    ax.set_ylabel("BRiG-AFA $-$ Myopic Q (pp)")
    ax.set_title(title)
    ax.grid(axis="y", color="#D9D9D9", linewidth=0.6, alpha=0.7)
    ax.spines[["top", "right"]].set_visible(False)
    return gain


def make_main_performance_figures() -> tuple[pd.DataFrame, pd.DataFrame]:
    cube_summary = pd.read_csv(RESULTS / "cube_nm_final_multiseed_summary.csv")
    cube_raw = pd.read_csv(RESULTS / "cube_nm_final_multiseed_raw.csv")
    fashion_summary = pd.read_csv(RESULTS / "fashion_mnist_selected_5seed_summary.csv")
    fashion_raw = pd.read_csv(RESULTS / "fashion_mnist_selected_5seed_raw.csv")
    policies = ["random", "global_mi", "myopic_q", "learned_briga"]

    fig, axes = plt.subplots(1, 2, figsize=(7.15, 2.75), constrained_layout=True)
    plot_policy_curves(axes[0], cube_summary, policies, "accuracy_mean", "accuracy_se", "Accuracy")
    axes[0].set_title("(a) Predictive performance")
    axes[0].set_xticks([1, 2, 3, 5, 8])
    axes[0].legend(frameon=False, ncol=2, loc="lower right")
    cube_gain = plot_paired_gain(axes[1], cube_raw, "(b) Paired gain over Myopic Q")
    axes[1].set_xticks([1, 2, 3, 5, 8])
    fig.suptitle("CUBE-NM: controlled context-dependent acquisition", fontsize=10.5)
    save_figure(fig, "fig2_cube_nm_main")

    fig, axes = plt.subplots(1, 2, figsize=(7.15, 2.75), constrained_layout=True)
    plot_policy_curves(
        axes[0], fashion_summary, policies, "accuracy_mean", "accuracy_se", "Test accuracy"
    )
    axes[0].set_title("(a) Predictive performance")
    axes[0].set_xticks([1, 2, 4, 8, 12, 16, 20])
    axes[0].legend(frameon=False, ncol=2, loc="lower right")
    fashion_gain = plot_paired_gain(axes[1], fashion_raw, "(b) Paired gain over Myopic Q")
    axes[1].set_xticks([1, 2, 4, 8, 12, 16, 20])
    fig.suptitle("Fashion-MNIST: selected-pixel acquisition", fontsize=10.5)
    save_figure(fig, "fig3_fashion_mnist_main")
    return cube_gain, fashion_gain


def make_heatmap_panel() -> None:
    frequency = pd.read_csv(RESULTS / "fashion_mnist_selected_acquisition_frequency_seed7.csv")
    policies = ["global_mi", "myopic_q", "learned_briga"]
    titles = ["(a) Global MI", "(b) Myopic Q", "(c) BRiG-AFA"]

    fig, axes = plt.subplots(1, 3, figsize=(7.15, 2.5), constrained_layout=True)
    image = None
    for ax, policy, title in zip(axes, policies, titles):
        frame = frequency[(frequency["policy"] == policy) & (frequency["budget"] == 4)]
        heatmap = np.zeros((28, 28), dtype=float)
        for _, row in frame.iterrows():
            heatmap[int(row["pixel_row"]), int(row["pixel_col"])] = float(row["selection_frequency"])
        image = ax.imshow(heatmap, cmap="viridis", vmin=0.0, vmax=1.0, interpolation="nearest")
        ax.set_title(title)
        ax.set_xlabel("Pixel column")
        ax.set_ylabel("Pixel row")
        ax.set_xticks([0, 9, 18, 27])
        ax.set_yticks([0, 9, 18, 27])
        ax.tick_params(length=2)
    colorbar = fig.colorbar(image, ax=axes, shrink=0.82, pad=0.02)
    colorbar.set_label("Selection frequency")
    fig.suptitle("Instance-wise acquisition patterns (budget 4; illustrative seed 7)", fontsize=10.5)
    save_figure(fig, "fig4_fashion_mnist_acquisition_maps")


def make_appendix_figures() -> None:
    cube = pd.read_csv(RESULTS / "cube_nm_final_multiseed_summary.csv")
    fashion = pd.read_csv(RESULTS / "fashion_mnist_selected_5seed_summary.csv")
    mini = pd.read_csv(RESULTS / "miniboone_tabular_3seed_summary.csv")
    policies = ["random", "global_mi", "myopic_q", "learned_briga"]

    fig, axes = plt.subplots(1, 2, figsize=(7.15, 2.7), constrained_layout=True)
    plot_policy_curves(
        axes[0], cube, policies, "cross_entropy_mean", "cross_entropy_se", "Test cross-entropy"
    )
    axes[0].set_title("(a) CUBE-NM")
    axes[0].set_xticks([1, 2, 3, 5, 8])
    plot_policy_curves(
        axes[1], fashion, policies, "cross_entropy_mean", "cross_entropy_se", "Test cross-entropy"
    )
    axes[1].set_title("(b) Fashion-MNIST")
    axes[1].set_xticks([1, 2, 4, 8, 12, 16, 20])
    axes[1].legend(frameon=False, ncol=2, loc="upper right")
    save_figure(fig, "figA1_cross_entropy")

    fig, axes = plt.subplots(1, 2, figsize=(7.15, 2.7), constrained_layout=True)
    plot_policy_curves(axes[0], mini, policies, "accuracy_mean", "accuracy_se", "Accuracy")
    full = mini[mini["policy"] == "full_features"]
    if not full.empty:
        axes[0].axhline(
            float(full.iloc[0]["accuracy_mean"]), color="#333333", linestyle="--", linewidth=1.0,
            label="Full features",
        )
    axes[0].set_title("(a) Accuracy")
    axes[0].set_xticks([1, 2, 4, 8, 16])
    plot_policy_curves(
        axes[1], mini, policies, "cross_entropy_mean", "cross_entropy_se", "Test cross-entropy"
    )
    if not full.empty:
        axes[1].axhline(
            float(full.iloc[0]["cross_entropy_mean"]), color="#333333", linestyle="--", linewidth=1.0,
            label="Full features",
        )
    axes[1].set_title("(b) Cross-entropy")
    axes[1].set_xticks([1, 2, 4, 8, 16])
    handles, labels = axes[1].get_legend_handles_labels()
    fig.legend(handles, labels, frameon=False, ncol=5, loc="lower center", bbox_to_anchor=(0.5, -0.03))
    save_figure(fig, "figA2_miniboone")

    references = ["context_first", "regime_oracle"]
    fig, ax = plt.subplots(figsize=(3.6, 2.7), constrained_layout=True)
    plot_policy_curves(
        ax,
        cube,
        ["learned_briga", *references],
        "accuracy_mean",
        "accuracy_se",
        "Test accuracy",
    )
    ax.set_xticks([1, 2, 3, 5, 8])
    ax.set_title("CUBE-NM diagnostic references")
    ax.legend(frameon=False, loc="lower right")
    save_figure(fig, "figA3_cube_nm_references")


def make_tables(cube_gain: pd.DataFrame, fashion_gain: pd.DataFrame) -> None:
    TABLES.mkdir(parents=True, exist_ok=True)
    rows = []
    for dataset, gain in [("CUBE-NM", cube_gain), ("Fashion-MNIST", fashion_gain)]:
        for _, row in gain.iterrows():
            rows.append(
                {
                    "dataset": dataset,
                    "budget": int(row["budget"]),
                    "briga_minus_myopic_accuracy_pp": round(100.0 * float(row["mean"]), 2),
                    "paired_se_pp": round(100.0 * float(row["se"]), 2),
                    "n_seeds": int(row["count"]),
                }
            )
    pd.DataFrame(rows).to_csv(TABLES / "table_main_paired_gains.csv", index=False)

    per_seed = pd.read_csv(RESULTS / "fashion_mnist_selected_5seed_budget_mean_per_seed.csv")
    pivot = per_seed.pivot(index="seed", columns="policy", values="mean_accuracy_k2_16")
    difference = pivot["learned_briga"] - pivot["myopic_q"]
    summary = pd.DataFrame(
        [
            {
                "dataset": "Fashion-MNIST",
                "budget_summary": "mean over k={2,4,8,12,16}",
                "briga_accuracy": round(float(pivot["learned_briga"].mean()), 4),
                "myopic_accuracy": round(float(pivot["myopic_q"].mean()), 4),
                "paired_gain_pp": round(100.0 * float(difference.mean()), 2),
                "paired_se_pp": round(100.0 * float(difference.std(ddof=1) / math.sqrt(len(difference))), 2),
                "n_seeds": int(len(difference)),
            }
        ]
    )
    summary.to_csv(TABLES / "table_fashion_aggregate.csv", index=False)


def main() -> None:
    configure_style()
    cube_gain, fashion_gain = make_main_performance_figures()
    make_heatmap_panel()
    make_appendix_figures()
    make_tables(cube_gain, fashion_gain)
    print(f"Wrote publication assets to {OUTPUT}")


if __name__ == "__main__":
    main()
