import os
import random
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch


"""
Utility functions for training, evaluation, and reproducibility.
"""


# Reproducibility

def set_seed(seed=42, deterministic=False):
    """Set random seeds."""

    random.seed(seed)
    np.random.seed(seed)

    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    if deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

    else:
        torch.backends.cudnn.benchmark = True


# Device helpers

def get_device(device_arg="cuda"):
    """Return torch device."""

    if device_arg == "cuda" and torch.cuda.is_available():
        return torch.device("cuda")

    return torch.device("cpu")


def get_device_info():
    """Return device information."""

    if torch.cuda.is_available():
        return (
            f"GPU: "
            f"{torch.cuda.get_device_name(0)} "
            f"(CUDA {torch.version.cuda})"
        )

    return "CPU"


# Directory helpers

def ensure_dir(path):
    """Create directory if it does not exist."""

    Path(path).mkdir(
        parents=True,
        exist_ok=True
    )

    return path


def ensure_dirs(paths):
    """Create multiple directories."""

    for path in paths:
        ensure_dir(path)


def get_checkpoint_dir(base_dir="checkpoints"):
    """Return checkpoint directory."""

    return ensure_dir(base_dir)


# Checkpoint helpers

def save_checkpoint(
    model,
    path,
    optimizer=None,
    epoch=None,
    metrics=None,
    verbose=True
):
    """Save model checkpoint."""

    checkpoint = {
        "model_state_dict": model.state_dict(),
    }

    if optimizer is not None:
        checkpoint["optimizer_state_dict"] = (
            optimizer.state_dict()
        )

    if epoch is not None:
        checkpoint["epoch"] = epoch

    if metrics is not None:
        checkpoint["metrics"] = metrics

    ensure_dir(os.path.dirname(path))

    torch.save(checkpoint, path)

    if verbose:
        print(f"Checkpoint saved to {path}")


def load_checkpoint(
    model,
    path,
    device=None,
    optimizer=None,
    strict=True
):
    """Load model checkpoint."""

    if device is None:
        device = get_device()

    checkpoint = torch.load(
        path,
        map_location=device
    )

    model.load_state_dict(
        checkpoint["model_state_dict"],
        strict=strict
    )

    if (
        optimizer is not None
        and "optimizer_state_dict" in checkpoint
    ):
        optimizer.load_state_dict(
            checkpoint["optimizer_state_dict"]
        )

    epoch = checkpoint.get("epoch", None)
    metrics = checkpoint.get("metrics", None)

    print(f"Checkpoint loaded from {path}")

    return model, epoch, metrics


def load_model_weights(
    model,
    path,
    device=None,
    strict=True
):
    """Load model weights only."""

    if device is None:
        device = get_device()

    weights = torch.load(
        path,
        map_location=device
    )

    if "model_state_dict" in weights:
        weights = weights["model_state_dict"]

    model.load_state_dict(
        weights,
        strict=strict
    )

    print(f"Model weights loaded from {path}")

    return model


# Plotting

def plot_training_curves(
    train_losses,
    val_metrics,
    epochs=None,
    save_path=None,
    show=True
):
    """Plot training curves."""

    if epochs is None:
        epochs = list(
            range(1, len(train_losses) + 1)
        )

    n_metrics = len(val_metrics)

    fig, axes = plt.subplots(
        1,
        n_metrics + 1,
        figsize=(5 * (n_metrics + 1), 4)
    )

    if n_metrics == 0:
        axes = [axes]

    elif n_metrics + 1 == 1:
        axes = [axes]

    # Training loss
    axes[0].plot(
        epochs,
        train_losses,
        marker="o",
        linewidth=2
    )

    axes[0].set_title("Training Loss")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Loss")
    axes[0].grid(True, alpha=0.3)

    # Validation metrics
    for idx, (metric_name, values) in enumerate(
        val_metrics.items()
    ):

        ax = axes[idx + 1]

        ax.plot(
            epochs,
            values,
            marker="s",
            linewidth=2
        )

        ax.set_title(
            f"Validation {metric_name.capitalize()}"
        )

        ax.set_xlabel("Epoch")
        ax.set_ylabel(metric_name.capitalize())
        ax.grid(True, alpha=0.3)

    plt.tight_layout()

    if save_path:
        ensure_dir(os.path.dirname(save_path))

        plt.savefig(
            save_path,
            dpi=300,
            bbox_inches="tight"
        )

        print(f"Plot saved to {save_path}")

    if show:
        plt.show()

    else:
        plt.close()


def plot_bar_metrics(
    metrics_dict,
    title="Evaluation Metrics",
    save_path=None,
    show=True,
    ylim=(0, 1)
):
    """Plot metric bar chart."""

    metrics = list(metrics_dict.keys())
    values = list(metrics_dict.values())

    plt.figure(figsize=(8, 5))

    plt.bar(metrics, values)

    plt.ylabel("Score")
    plt.title(title)
    plt.ylim(ylim)

    plt.grid(
        True,
        alpha=0.3,
        axis="y"
    )

    plt.tight_layout()

    if save_path:
        ensure_dir(os.path.dirname(save_path))

        plt.savefig(
            save_path,
            dpi=300,
            bbox_inches="tight"
        )

        print(f"Plot saved to {save_path}")

    if show:
        plt.show()

    else:
        plt.close()


def plot_comparison_bars(
    results_dict,
    metric_name="Accuracy",
    title=None,
    save_path=None,
    show=True
):
    """Plot model comparison bars."""

    models = list(results_dict.keys())
    values = list(results_dict.values())

    if title is None:
        title = f"Model Comparison - {metric_name}"

    plt.figure(figsize=(8, 5))

    plt.bar(models, values)

    plt.ylabel(metric_name)
    plt.title(title)

    plt.ylim(
        0,
        max(1.0, max(values) * 1.1)
    )

    plt.grid(
        True,
        alpha=0.3,
        axis="y"
    )

    plt.xticks(
        rotation=45,
        ha="right"
    )

    plt.tight_layout()

    if save_path:
        ensure_dir(os.path.dirname(save_path))

        plt.savefig(
            save_path,
            dpi=300,
            bbox_inches="tight"
        )

        print(f"Plot saved to {save_path}")

    if show:
        plt.show()

    else:
        plt.close()


# Training helpers

class AverageMeter:
    """Track running average."""

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def get_class_weights(dataset, device=None):
    """Compute balanced class weights."""

    from sklearn.utils.class_weight import (
        compute_class_weight
    )

    labels = [label for _, label in dataset]

    unique_labels = np.unique(labels)

    weights = compute_class_weight(
        "balanced",
        classes=unique_labels,
        y=labels
    )

    if device is None:
        device = get_device()

    return torch.tensor(
        weights,
        dtype=torch.float
    ).to(device)