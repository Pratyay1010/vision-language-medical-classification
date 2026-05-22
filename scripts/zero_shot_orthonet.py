import os
import argparse
import numpy as np
import matplotlib.pyplot as plt
import torch
from torch.utils.data import DataLoader
import clip

# Reusable imports from src
from src.datasets import OrthonetDataset
from src.evaluator import evaluate_clip_zero_shot

# Zero-Shot Classification
def zero_shot_classify(model, text_features, test_loader, device):
    metrics = evaluate_clip_zero_shot(model, text_features, test_loader, device)
    acc = metrics["accuracy"]
    f1  = metrics["f1_score"]
    auc = metrics.get("auc", 0.0)
    if np.isnan(auc):
        auc = 0.0

    print(f"\nZero-Shot CLIP on Orthonet → Acc={acc:.4f}, F1={f1:.4f}, AUC={auc if auc else 'N/A'}")
    return {"Top-1 Accuracy": acc, "F1-score": f1, "AUC-ROC": auc}

# Plot
def plot_zero_shot_results(results, save_path=None):
    metrics, values = list(results.keys()), list(results.values())
    plt.figure(figsize=(7,5))
    bars = plt.bar(metrics, values, color=["skyblue", "salmon", "orange"])
    plt.ylabel("Score")
    plt.title("Zero-Shot CLIP on Orthonet Dataset")
    plt.ylim(0, 1)

    # Overlay metric values on top
    for bar, value in zip(bars, values):
        plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02, f"{value:.2f}", 
                 ha='center', va='bottom', fontsize=10)

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path)
        print(f"Saved plot as {save_path}")
    plt.show()

# Main
def main(args):
    device = torch.device(args.device if torch.cuda.is_available() or args.device=="cpu" else "cpu")
    print(f"Using device: {device}")

    model, preprocess = clip.load("ViT-B/32", device=device, jit=False)

    class_names = [
        'Hip_SmithAndNephew_Polarstem_NilCol',
        'Knee_SmithAndNephew_GenesisII',
        'Hip_Stryker_Exeter',
        'Knee_Depuy_Synthes_Sigma',
        'Hip_DepuySynthes_Corail_Collar',
        'Hip_DepuySynthes_Corail_NilCol',
        'Hip_SmithAndNephew_Anthology',
        'Hip_JRIOrtho_FurlongEvolution_Collar',
        'Knee_SmithAndNephew_Legion2',
        'Hip_Stryker_AccoladeII',
        'Hip_JRIOrtho_FurlongEvolution_NilCol',
        'Knee_ZimmerBiomet_Oxford'
    ]
    prompts = [f"an X-ray of a {cls.replace('_', ' ')} implant" for cls in class_names]

    with torch.no_grad():
        text_tokens = clip.tokenize(prompts).to(device)
        text_features = model.encode_text(text_tokens)
        text_features /= text_features.norm(dim=-1, keepdim=True)

    # Dataset + DataLoader using reusable OrthonetDataset
    test_dataset = OrthonetDataset(csv_file=os.path.join(args.csv_dir, "test.csv"),
                                   root_dir=args.image_dir,
                                   transform=preprocess)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, num_workers=2)

    # Run zero-shot
    results = zero_shot_classify(model, text_features, test_loader, device)
    plot_zero_shot_results(results, save_path="orthonet_zero_shot_results.png")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv_dir", type=str, required=True, help="Path to CSV directory")
    parser.add_argument("--image_dir", type=str, required=True, help="Path to image directory")
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--device", type=str, default="cuda", help="Device to use: 'cuda' or 'cpu'")
    args = parser.parse_args()
    main(args)