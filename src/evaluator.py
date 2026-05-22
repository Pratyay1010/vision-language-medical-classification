import numpy as np
import torch
import torch.nn.functional as F

from sklearn.metrics import (
    accuracy_score,
    f1_score,
    roc_auc_score,
    top_k_accuracy_score,
)


# Basic metrics

def compute_accuracy(y_true, y_pred):
    """Compute accuracy."""
    return accuracy_score(y_true, y_pred)


def compute_topk_accuracy(y_true, y_prob, k=3):
    """Compute Top-K accuracy."""
    try:
        return top_k_accuracy_score(y_true, y_prob, k=k)

    except Exception:
        y_true = np.array(y_true)
        y_prob = np.array(y_prob)

        topk_preds = np.argsort(-y_prob, axis=1)[:, :k]

        return np.mean([
            y_true[i] in topk_preds[i]
            for i in range(len(y_true))
        ])


def compute_f1_score(y_true, y_pred, average="weighted"):
    """Compute F1 score."""
    return f1_score(y_true, y_pred, average=average)


def compute_auc_score(y_true, y_prob, multi_class="ovr"):
    """Compute AUC-ROC."""
    try:
        return roc_auc_score(
            y_true,
            y_prob,
            multi_class=multi_class
        )

    except Exception:
        return float("nan")


# Prediction collection

class PredictionCollector:
    """Collect predictions during evaluation."""

    def __init__(self):
        self.y_true = []
        self.y_pred = []
        self.y_prob = []

    def update(self, labels, predictions, probabilities=None):
        """Add batch predictions."""

        self.y_true.extend(
            labels.cpu().numpy()
            if torch.is_tensor(labels)
            else labels
        )

        self.y_pred.extend(
            predictions.cpu().numpy()
            if torch.is_tensor(predictions)
            else predictions
        )

        if probabilities is not None:
            probs = (
                probabilities.cpu().numpy()
                if torch.is_tensor(probabilities)
                else probabilities
            )

            self.y_prob.extend(probs)

    def reset(self):
        """Clear stored predictions."""
        self.y_true = []
        self.y_pred = []
        self.y_prob = []

    @property
    def arrays(self):
        """Return predictions as numpy arrays."""
        return (
            np.array(self.y_true),
            np.array(self.y_pred),
            np.array(self.y_prob)
            if self.y_prob else None
        )

    def compute_metrics(self, top_k=None):
        """Compute evaluation metrics."""

        y_true, y_pred, y_prob = self.arrays

        metrics = {
            "accuracy": compute_accuracy(y_true, y_pred),
            "f1_score": compute_f1_score(y_true, y_pred),
        }

        if top_k is not None and y_prob is not None:
            metrics[f"top_{top_k}_accuracy"] = (
                compute_topk_accuracy(
                    y_true,
                    y_prob,
                    k=top_k
                )
            )

        if y_prob is not None:
            metrics["auc"] = compute_auc_score(
                y_true,
                y_prob
            )

        return metrics


# Evaluation functions

def evaluate_classification(
    model,
    dataloader,
    device,
    top_k=None,
    return_predictions=False
):
    """Evaluate classification model."""

    model.eval()

    collector = PredictionCollector()

    with torch.no_grad():

        for images, labels in dataloader:

            images = images.to(device)
            labels = labels.to(device)

            outputs = model(images)

            probs = F.softmax(outputs, dim=1)
            preds = probs.argmax(dim=1)

            collector.update(labels, preds, probs)

    metrics = collector.compute_metrics(top_k=top_k)

    if return_predictions:
        return metrics, collector.arrays

    return metrics


def evaluate_clip_zero_shot(
    model,
    text_features,
    dataloader,
    device,
    top_k=None
):
    """Evaluate CLIP zero-shot classification."""

    model.eval()

    collector = PredictionCollector()

    with torch.no_grad():

        for images, labels in dataloader:

            images = images.to(device)
            labels = labels.to(device)

            # Encode images
            image_features = model.encode_image(images)

            image_features = (
                image_features
                / image_features.norm(dim=-1, keepdim=True)
            )

            # Compute similarity
            similarity = (
                100.0 * image_features @ text_features.T
            )

            probs = F.softmax(similarity, dim=-1)
            preds = probs.argmax(dim=-1)

            collector.update(labels, preds, probs)

    return collector.compute_metrics(top_k=top_k)


def evaluate_prompt_learner(
    model,
    prompt_learner,
    dataloader,
    device,
    model_type="coop"
):
    """Evaluate prompt learning models."""

    model.eval()
    prompt_learner.eval()

    collector = PredictionCollector()

    with torch.no_grad():

        for images, labels in dataloader:

            images = images.to(device)
            labels = labels.to(device)

            # Encode images
            image_features = model.encode_image(images)

            image_features = (
                image_features
                / image_features.norm(dim=-1, keepdim=True)
            )

            if model_type.lower() == "maple":

                text_features = prompt_learner(image_features)

                logits = 100.0 * torch.bmm(
                    image_features.unsqueeze(1),
                    text_features.permute(0, 2, 1)
                ).squeeze(1)

            else:

                text_features = (
                    prompt_learner(image_features)
                    if model_type.lower() == "cocoop"
                    and hasattr(prompt_learner, "meta_net")
                    else prompt_learner()
                )

                logits = (
                    100.0 * image_features @ text_features.T
                )

            probs = F.softmax(logits, dim=-1)
            preds = probs.argmax(dim=-1)

            collector.update(labels, preds, probs)

    return collector.compute_metrics(top_k=None)


# Print helpers

def print_metrics(metrics, prefix="", decimals=4):
    """Print metrics dictionary."""

    if prefix:
        print(f"{prefix} ", end="")

    metric_strings = []

    for name, value in metrics.items():

        if not np.isnan(value):
            metric_strings.append(
                f"{name}={value:.{decimals}f}"
            )

        else:
            metric_strings.append(f"{name}=N/A")

    print(" | ".join(metric_strings))


def format_metrics_string(metrics, decimals=4):
    """Format metrics as string."""

    return ", ".join([
        (
            f"{name}={value:.{decimals}f}"
            if not np.isnan(value)
            else f"{name}=N/A"
        )
        for name, value in metrics.items()
    ])