import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import cm
import numpy as np
from typing import Dict, Optional
import os

# ====== Feature Inportance ====== #
def plot_evaluation_metrics(
    metrics: Dict[str, float],
    model_name: Optional[str] = None,
    save_dir: Optional[str] = None,
    show: bool = True
) -> None:
    """
    Plots a bar chart of evaluation metrics.

    Args:
        metrics: dict mapping metric names to scores, e.g.
                 {"accuracy": 0.82, "precision": 0.79, "recall": 0.88, "f1 score": 0.83}
        title: optional title for the plot.
        save_path: if given, full path (including filename) where to save the .png.
    """
    names = list(metrics.keys())
    values = [metrics[n] for n in names]

    x = np.arange(len(names))
    width = 0.6

    # select a color palette
    cmap = cm.get_cmap("tab20b")    
    colors = cmap(np.linspace(0, 1, len(names)))
    fig, ax = plt.subplots(figsize=(6, 4))
    bars = ax.bar(x, values, width, color=colors)

    for bar in bars:
        h = bar.get_height()
        ax.text(
            bar.get_x() + bar.get_width()/2,
            h + 0.01,
            f"{h:.2f}",
            ha="center",
            va="bottom",
            fontsize=9
        )

    ax.set_xticks(x)
    ax.set_xticklabels([n.capitalize() for n in names], rotation=30, ha="right")
    ax.set_ylim(0, 1)
    ax.set_ylabel("Score")
    ax.set_title(f"Evaluation metrics {model_name}")
    ax.grid(axis="y", linestyle="--", alpha=0.5)
    plt.tight_layout()

    if save_dir:
        os.makedirs(os.path.dirname(save_dir), exist_ok=True)
        save_path = os.path.join(save_dir, f"evaluation_metrics_{model_name}.png")
        plt.savefig(save_path, dpi=300)
    
    if show:
        plt.show()


# ====== Plot train vs validation losses ======
def plot_train_val_losses(history: dict,
                          model_name: Optional[str] = None,
                          path: Optional[str] = None,
                          show: bool = False) -> None:
    epochs_range = range(1, len(history["training_losses"]) + 1)

    plt.figure(figsize=(12, 6))
    
    # Subplot 1: training losses.
    plt.subplot(1, 2, 1)
    plt.plot(epochs_range, history["training_losses"], label='Training losses', marker='o', linestyle='--', color='royalblue')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title(f'{model_name} training Loss')
    plt.legend()
    plt.grid(which='both', linestyle='--', linewidth=0.5)

    # Subplot 2: validation loss.
    plt.subplot(1, 2, 2)
    plt.plot(epochs_range, history["validation_losses"], label='Validation losses', marker='*', linestyle='--', color='purple')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title(f'{model_name} Validation Loss')
    plt.legend()
    plt.grid(which='both', linestyle='--', linewidth=0.5)

    plt.tight_layout()

    if path:
        save_dir = os.path.dirname(path)
        if save_dir and not os.path.exists(save_dir):
            os.makedirs(save_dir)
        save_path = os.path.join(path,f"{model_name}_train_&_validation.png")
        plt.savefig(save_path)
    
    if show:
        plt.show()

# ====== Plot train vs validation losses ======
def plot_train_val_stats(history: dict,
                        model_name: Optional[str] = None,
                        path: Optional[str] = None,
                        show: bool = False) -> None:
    epochs_range = range(1, len(history["training_losses"]) + 1)

    plt.figure(figsize=(14, 8))
    
    # Subplot 1: training losses.
    plt.subplot(1, 3, 1)
    plt.plot(epochs_range, history["training_losses"], label='Training losses', marker='o', linestyle='--', color='royalblue')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title(f'{model_name} training Loss')
    plt.legend()
    plt.grid(which='both', linestyle='--', linewidth=0.5)

    # Subplot 2: validation loss.
    plt.subplot(1, 3, 2)
    plt.plot(epochs_range, history["validation_losses"], label='Validation losses', marker='*', linestyle='--', color='purple')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title(f'{model_name} Validation Loss')
    plt.legend()
    plt.grid(which='both', linestyle='--', linewidth=0.5)

    # Subplot 3: f1 score.
    plt.subplot(1, 3, 3)
    plt.plot(epochs_range, history["f1_score"], label='f1 score', marker='d', linestyle='--', color='forestgreen')
    plt.xlabel('Epoch')
    plt.ylabel('score')
    plt.title(f'{model_name} Validation f1 score')
    plt.legend()
    plt.grid(which='both', linestyle='--', linewidth=0.5)

    plt.tight_layout()

    if path:
        save_dir = os.path.dirname(path)
        if save_dir and not os.path.exists(save_dir):
            os.makedirs(save_dir)
        save_path = os.path.join(path,f"{model_name}_train_&_validation_(f1_&_loss).png")
        plt.savefig(save_path)
    
    if show:
        plt.show()



