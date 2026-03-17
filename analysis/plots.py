"""Visualization: boxplots, histograms, learning curves."""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from config.settings import METRICS_DIR


def plot_metric_comparison(
    prng_values: np.ndarray,
    qrng_values: np.ndarray,
    metric_name: str,
    save_dir: Path = METRICS_DIR,
) -> None:
    """Side-by-side boxplot and histogram for a single metric."""
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    # Boxplot
    axes[0].boxplot(
        [prng_values, qrng_values],
        labels=["PRNG (LCG)", "QRNG"],
        patch_artist=True,
        boxprops=[dict(facecolor="#e74c3c", alpha=0.5), dict(facecolor="#2ecc71", alpha=0.5)],
    )
    axes[0].set_title(f"{metric_name} — Boxplot")
    axes[0].set_ylabel(metric_name)

    # Histogram
    bins = 30
    axes[1].hist(prng_values, bins=bins, alpha=0.5, color="#e74c3c", label="PRNG (LCG)", density=True)
    axes[1].hist(qrng_values, bins=bins, alpha=0.5, color="#2ecc71", label="QRNG", density=True)
    axes[1].set_title(f"{metric_name} — Distribution")
    axes[1].set_xlabel(metric_name)
    axes[1].set_ylabel("Density")
    axes[1].legend()

    plt.tight_layout()
    save_dir.mkdir(parents=True, exist_ok=True)
    plt.savefig(save_dir / f"comparison_{metric_name}.png", dpi=150)
    plt.close()


def plot_training_curves(
    prng_log_path: Path,
    qrng_log_path: Path,
    save_dir: Path = METRICS_DIR,
    window: int = 100,
) -> None:
    """Plot reward and success rate learning curves for both agents."""
    prng_log = pd.read_csv(prng_log_path)
    qrng_log = pd.read_csv(qrng_log_path)

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Reward over episodes (smoothed)
    for df, color, label in [
        (prng_log, "#e74c3c", "PRNG (LCG)"),
        (qrng_log, "#2ecc71", "QRNG"),
    ]:
        smoothed = df["reward"].rolling(window=window, min_periods=1).mean()
        axes[0].plot(df["episode"], smoothed, color=color, alpha=0.8, label=label)

    axes[0].set_xlabel("Episode")
    axes[0].set_ylabel(f"Reward (rolling avg, w={window})")
    axes[0].set_title("Training Reward")
    axes[0].legend()

    # Success rate over episodes (smoothed)
    for df, color, label in [
        (prng_log, "#e74c3c", "PRNG (LCG)"),
        (qrng_log, "#2ecc71", "QRNG"),
    ]:
        smoothed = df["success"].rolling(window=window, min_periods=1).mean() * 100
        axes[1].plot(df["episode"], smoothed, color=color, alpha=0.8, label=label)

    axes[1].set_xlabel("Episode")
    axes[1].set_ylabel(f"Success Rate % (rolling avg, w={window})")
    axes[1].set_title("Training Success Rate")
    axes[1].legend()

    plt.tight_layout()
    save_dir.mkdir(parents=True, exist_ok=True)
    plt.savefig(save_dir / "training_curves.png", dpi=150)
    plt.close()
    print(f"Training curves saved to {save_dir / 'training_curves.png'}")


def plot_evaluation_summary(save_dir: Path = METRICS_DIR) -> None:
    """Bar charts comparing evaluation metrics across all 4 conditions."""
    conditions = [
        "a2c_prng_on_prng", "a2c_prng_on_qrng",
        "a2c_qrng_on_qrng", "a2c_qrng_on_prng",
    ]
    labels = [
        "PRNG→PRNG\n(intra)", "PRNG→QRNG\n(cross)",
        "QRNG→QRNG\n(intra)", "QRNG→PRNG\n(cross)",
    ]
    colors = ["#e74c3c", "#c0392b", "#2ecc71", "#27ae60"]

    dfs = {}
    for cond in conditions:
        path = save_dir / f"eval_{cond}.csv"
        if path.exists():
            dfs[cond] = pd.read_csv(path)

    if len(dfs) < 4:
        print("Not all evaluation CSVs found. Skipping summary plot.")
        return

    fig, axes = plt.subplots(1, 3, figsize=(16, 5))

    for ax, metric, title in [
        (axes[0], "success", "Success Rate (%)"),
        (axes[1], "reward", "Mean Reward"),
        (axes[2], "steps", "Mean Steps"),
    ]:
        values = []
        for cond in conditions:
            if metric == "success":
                values.append(dfs[cond][metric].mean() * 100)
            else:
                values.append(dfs[cond][metric].mean())

        ax.bar(range(4), values, color=colors, alpha=0.8)
        ax.set_xticks(range(4))
        ax.set_xticklabels(labels, fontsize=8)
        ax.set_title(title)

        for i, v in enumerate(values):
            ax.text(i, v + 0.5, f"{v:.1f}", ha="center", fontsize=8)

    plt.tight_layout()
    plt.savefig(save_dir / "evaluation_summary.png", dpi=150)
    plt.close()
    print(f"Evaluation summary saved to {save_dir / 'evaluation_summary.png'}")
