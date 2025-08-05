import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import cm
import numpy as np
from typing import Dict, Optional, Union
import os

# ====== Feature Inportance ====== #
def plot_ft_importance(feature_importances,
                       feature_names: list[str],
                       top_features: int = 10,
                       save_dir: Optional[str] = None,
                       model_name: Optional[str] = None):
    # ** 1. Config for font CJK ** #
    plt.rcParams['font.family'] = ['Noto Sans CJK JP'] 
    plt.rcParams['axes.unicode_minus'] = False 

    # ** 2.create a pandas series for features importance and sort in not ascending form ** #
    importances = pd.Series(feature_importances, index=feature_names)
    importances = importances.sort_values(ascending=True)
    top = importances.tail(top_features)

    print(f"\n Top {top_features} feature importances:")
    print(top[::-1])
    
    # ** 3. plot the feature importances ** #
    plt.figure(figsize=(8,5))
    ax = top.plot(kind="barh")
    ax.set_title(f"Top {top_features} feature importances ({model_name}):")
    ax.set_xlabel("Importance")
    ax.set_ylabel("Features")
    plt.tight_layout()
    plt.grid(axis="both", linestyle="--", alpha=0.7)

    if save_dir:
        os.makedirs(save_dir, exist_ok=True)
        save_path = os.path.join(save_dir, f"feature_importance_{model_name}.png")
        plt.savefig(save_path)
        print(f"Feature importances fig saved successfully to --> {save_dir}")
        
    plt.show()

# ====== Feature Inportance ====== #
def plot_evaluation_metrics(
    metrics: Dict[str, float],
    model_name: Optional[str] = None,
    save_dir: Optional[str] = None
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
    plt.show()



