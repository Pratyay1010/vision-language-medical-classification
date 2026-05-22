import os
import argparse
import numpy as np
import torch
import torch.nn as nn
from torchvision import models, datasets
from torch.utils.data import DataLoader
import timm
import matplotlib.pyplot as plt

# Reusable imports from src
from src.datasets import get_pacemaker_transforms
from src.evaluator import evaluate_classification

# Plot results
def plot_results(results, save_path="pacemaker_vit_results.png"):
    metrics = ["Top-1 Acc", "Top-3 Acc", "F1-score", "AUC-ROC"]
    x = np.arange(len(results))
    bar_width = 0.2
    plt.figure(figsize=(10,6))

    for i, metric in enumerate(metrics):
        values = [results[m][i] if results[m][i] is not None else 0 for m in results]
        bars = plt.bar(x + i*bar_width, values, width=bar_width, label=metric)
        # overlay values
        for bar, val in zip(bars, values):
            plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01, f"{val:.2f}",
                     ha='center', va='bottom', fontsize=9)

    plt.xticks(x + 1.5*bar_width, [m.upper() for m in results])
    plt.ylabel("Score")
    plt.ylim(0, 1)
    plt.title("Pacemaker Dataset Evaluation Metrics")
    plt.legend()
    plt.tight_layout()
    plt.savefig(save_path)
    print(f"Saved plot as {save_path}")
    plt.show()

# Main
def main(args):
    device = torch.device(args.device if torch.cuda.is_available() or args.device=="cpu" else "cpu")
    print(f"Using device: {device}")

    _, test_tfms = get_pacemaker_transforms(224)

    test_dataset = datasets.ImageFolder(root=os.path.join(args.data_dir, "Test"), transform=test_tfms)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, num_workers=2)
    num_classes = len(test_dataset.classes)

    models_cfg = {
        "imagenet": models.vit_b_16(weights="IMAGENET1K_V1"),
        "clip": timm.create_model("vit_base_patch16_clip_224.openai", pretrained=True, num_classes=num_classes),
        "dino": timm.create_model("vit_small_patch16_224.dino", pretrained=True)
    }

    results = {}
    for name, model in models_cfg.items():
        if name == "imagenet":
            in_features = model.heads.head.in_features
            model.heads.head = nn.Linear(in_features, num_classes)
        elif name == "clip":
            in_features = model.head.in_features
            model.head = nn.Linear(in_features, num_classes)
        elif name == "dino":
            embed_dim = model.embed_dim
            model.head = nn.Linear(embed_dim, num_classes)

        weight_file = f"vit_{name}_pacemaker.pth"
        model.load_state_dict(torch.load(weight_file, map_location=device))
        model = model.to(device)

        metrics = evaluate_classification(model, test_loader, device, top_k=3)
        acc1 = metrics["accuracy"]
        acc3 = metrics["top_3_accuracy"]
        f1 = metrics["f1_score"]
        auc = metrics.get("auc", None)
        results[name] = (acc1, acc3, f1, auc)

    for k, (acc1, acc3, f1, auc) in results.items():
        print(f"{k.upper()} → Top-1 Acc={acc1:.4f}, Top-3 Acc={acc3:.4f}, F1={f1:.4f}, AUC={auc if auc else 'N/A'}")

    plot_results(results)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", type=str, required=True, help="Path to Pacemaker dataset root")
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--device", type=str, default="cuda", help="Device to use: 'cuda' or 'cpu'")
    args = parser.parse_args()
    main(args)