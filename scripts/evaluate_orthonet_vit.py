import os
import argparse
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import models
import timm
import matplotlib.pyplot as plt

# Reusable imports from src
from src.datasets import OrthonetDataset, get_imagenet_transforms
from src.evaluator import evaluate_classification

# Main
def main(args):
    device = torch.device(args.device if torch.cuda.is_available() or args.device=="cpu" else "cpu")
    print(f"Using device: {device}")

    _, test_tfms = get_imagenet_transforms(224)

    test_dataset = OrthonetDataset(csv_file=os.path.join(args.csv_dir, "test.csv"),
                                   root_dir=args.image_dir,
                                   transform=test_tfms)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False, num_workers=1)
    num_classes = len(test_dataset.label2idx)

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

        model.load_state_dict(torch.load(f"vit_{name}_orthonet.pth", map_location=device))
        model = model.to(device)
        
        metrics = evaluate_classification(model, test_loader, device)
        acc = metrics["accuracy"]
        f1 = metrics["f1_score"]
        auc = metrics.get("auc", None)
        results[name] = (acc, f1, auc)

    # Print results
    for k, (acc, f1, auc) in results.items():
        print(f"{k.upper()} → Accuracy: {acc:.4f}, F1: {f1:.4f}, AUC: {auc if auc is not None else 'N/A'}")

    # Save bar plot
    metrics_list = ['Accuracy', 'F1', 'AUC']
    fig, ax = plt.subplots(figsize=(8, 6))
    bar_width = 0.2
    x = np.arange(len(models_cfg))

    for i, metric in enumerate(metrics_list):
        values = [results[m][i] if results[m][i] is not None else 0 for m in results]
        ax.bar(x + i*bar_width, values, width=bar_width, label=metric)

    ax.set_xticks(x + bar_width)
    ax.set_xticklabels([m.upper() for m in results])
    ax.set_ylim(0, 1)
    ax.set_ylabel("Score")
    ax.set_title("Model Evaluation Metrics")
    ax.legend()
    plt.tight_layout()
    plt.savefig("orthonet_vit_results.png")
    print("Saved plot as orthonet_vit_results.png")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv_dir", type=str, required=True, help="Path to CSV directory")
    parser.add_argument("--image_dir", type=str, required=True, help="Path to image directory")
    parser.add_argument("--device", type=str, default="cuda", help="Device to use: 'cuda' or 'cpu'")
    args = parser.parse_args()
    main(args)