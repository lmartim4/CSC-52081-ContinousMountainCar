"""Plotting utilities for training curves and comparisons."""

import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path


def smooth(values, window=50):
    """Simple moving average smoothing."""
    if len(values) < window:
        return values
    cumsum = np.cumsum(np.insert(values, 0, 0))
    return (cumsum[window:] - cumsum[:-window]) / window


def plot_training_curves(rewards_dict, title="Training Curves", window=50,
                         save_path=None):
    """Plot training reward curves for multiple runs.

    Args:
        rewards_dict: {label: list_of_episode_rewards}
        title: Plot title
        window: Smoothing window
        save_path: If given, save figure to this path
    """
    fig, ax = plt.subplots(figsize=(10, 5))

    for label, rewards in rewards_dict.items():
        smoothed = smooth(rewards, window)
        ax.plot(smoothed, label=label, alpha=0.9)
        ax.fill_between(
            range(len(smoothed)),
            smooth(rewards, window * 2) if len(rewards) > window * 2 else smoothed,
            alpha=0.1,
        )

    ax.set_xlabel("Episode")
    ax.set_ylabel("Reward")
    ax.set_title(title)
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()

    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
    return fig


def plot_comparison(results_dict, metric="mean_reward", save_path=None):
    """Bar chart comparing agents on a given metric.

    Args:
        results_dict: {agent_name: eval_results_dict}
        metric: Key to plot from eval results
        save_path: If given, save figure
    """
    fig, ax = plt.subplots(figsize=(8, 5))

    names = list(results_dict.keys())
    values = [r[metric] for r in results_dict.values()]
    errors = [r.get(f"std_{metric.replace('mean_', '')}", 0) for r in results_dict.values()]

    bars = ax.bar(names, values, yerr=errors, capsize=5, color=plt.cm.Set2.colors[:len(names)])
    ax.set_ylabel(metric.replace("_", " ").title())
    ax.set_title(f"Agent Comparison — {metric.replace('_', ' ').title()}")
    ax.grid(True, axis="y", alpha=0.3)

    for bar, val in zip(bars, values):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height(),
                f"{val:.1f}", ha="center", va="bottom", fontsize=10)

    plt.tight_layout()
    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
    return fig
