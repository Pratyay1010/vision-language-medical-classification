import os
import argparse
import numpy as np
import matplotlib.pyplot as plt
import torch
from torch.utils.data import DataLoader
from torchvision import datasets
import clip

# Reusable imports from src
from src.evaluator import evaluate_clip_zero_shot

# Zero-Shot Classification
def zero_shot_classify(test_loader, model, text_features, device):
    metrics = evaluate_clip_zero_shot(model, text_features, test_loader, device, top_k=3)
    acc = metrics["accuracy"]
    f1  = metrics["f1_score"]
    top3_acc = metrics["top_3_accuracy"]
    auc = metrics.get("auc", 0.0)
    if np.isnan(auc):
        auc = 0.0

    print(f"\nZero-Shot CLIP on Pacemaker → "
          f"Top-1={acc:.4f}, Top-3={top3_acc:.4f}, "
          f"F1={f1:.4f}, AUC={auc if auc else 'N/A'}")

    return {
        "Top-1 Accuracy": acc,
        "Top-3 Accuracy": top3_acc,
        "F1-score": f1,
        "AUC-ROC": auc
    }

# Plot results
def plot_zero_shot_results(results, save_path="pacemaker_zero_shot_results.png"):
    metrics = list(results.keys())
    values = list(results.values())

    plt.figure(figsize=(8,5))
    bars = plt.bar(metrics, values, color=["skyblue", "lightgreen", "salmon", "orange"])
    plt.ylabel("Score")
    plt.title("Zero-Shot CLIP on Pacemaker Dataset")
    plt.ylim(0, 1)

    # Overlay values on bars
    for bar, val in zip(bars, values):
        plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                 f"{val:.2f}", ha='center', va='bottom', fontsize=10)

    plt.tight_layout()
    plt.savefig(save_path)
    print(f"Saved plot as {save_path}")
    plt.show()

# Main
def main(args):
    # Device selection
    device = torch.device(args.device if torch.cuda.is_available() or args.device=="cpu" else "cpu")
    print(f"Using device: {device}")

    model, preprocess = clip.load("ViT-B/32", device=device, jit=False)

    # Load Pacemaker test set
    test_dataset = datasets.ImageFolder(root=os.path.join(args.data_dir, "Test"),
                                        transform=preprocess)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size,
                             shuffle=False, num_workers=2)

    # Generate prompts dynamically for all classes
    class_names = test_dataset.classes
    prompts = [f"an X-ray of a {cls.replace('_', ' ')} pacemaker implant"
               for cls in class_names]

    # Encode text prompts
    with torch.no_grad():
        text_tokens = clip.tokenize(prompts).to(device)
        text_features = model.encode_text(text_tokens)
        text_features /= text_features.norm(dim=-1, keepdim=True)

    # Run zero-shot classification
    results = zero_shot_classify(test_loader, model, text_features, device)
    plot_zero_shot_results(results)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", type=str, required=True, help="Path to Pacemaker dataset root")
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--device", type=str, default="cuda", help="Device to use: 'cuda' or 'cpu'")
    args = parser.parse_args()
    main(args)