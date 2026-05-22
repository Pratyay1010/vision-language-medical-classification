import os
import argparse
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import models, datasets
from torch.utils.data import DataLoader
from tqdm import tqdm
import timm
import matplotlib.pyplot as plt

# Reusable imports from src
from src.datasets import get_pacemaker_transforms, split_dataset_by_labels
from src.evaluator import evaluate_classification

# Evaluation
def evaluate_model(model, loader, device):
    metrics = evaluate_classification(model, loader, device, top_k=3)
    return metrics["accuracy"], metrics["top_3_accuracy"]

# Training
def train_model(model, train_loader, val_loader, criterion, optimizer, device, epochs, save_path, plot_path):
    train_acc_list, val_top1_list, val_top3_list = [], [], []

    for epoch in range(epochs):
        model.train()
        running_loss, correct, total = 0.0, 0, 0
        for images, labels in tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs}"):
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            _, predicted = torch.max(outputs, 1)
            correct += (predicted == labels).sum().item()
            total += labels.size(0)

        train_acc = 100 * correct / total
        val_top1, val_top3 = evaluate_model(model, val_loader, device)
        train_acc_list.append(train_acc)
        val_top1_list.append(val_top1*100)
        val_top3_list.append(val_top3*100)

        print(f"Epoch {epoch+1}: Loss={running_loss/len(train_loader):.4f}, "
              f"Train Acc={train_acc:.2f}%, Val Top-1={val_top1*100:.2f}%, Val Top-3={val_top3*100:.2f}%")

    # Save model
    torch.save(model.state_dict(), save_path)
    print(f"Model saved to {save_path}")

    # Plot accuracies
    plt.figure(figsize=(8,5))
    epochs_range = range(1, epochs+1)
    plt.plot(epochs_range, train_acc_list, label="Train Acc", marker='o')
    plt.plot(epochs_range, val_top1_list, label="Val Top-1", marker='s')
    plt.plot(epochs_range, val_top3_list, label="Val Top-3", marker='^')
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy (%)")
    plt.title(f"Training and Validation Accuracy ({os.path.basename(save_path).split('.')[0]})")
    plt.ylim(0, 100)
    plt.grid(True)
    plt.legend()
    for x, y in zip(epochs_range, train_acc_list):
        plt.text(x, y+1, f"{y:.1f}", ha='center', fontsize=8)
    for x, y in zip(epochs_range, val_top1_list):
        plt.text(x, y+1, f"{y:.1f}", ha='center', fontsize=8)
    for x, y in zip(epochs_range, val_top3_list):
        plt.text(x, y+1, f"{y:.1f}", ha='center', fontsize=8)
    plt.tight_layout()
    plt.savefig(plot_path)
    print(f"Accuracy plot saved to {plot_path}")
    plt.close()

# Main
def main(args):
    device = torch.device(args.device if torch.cuda.is_available() or args.device=="cpu" else "cpu")
    print(f"Using device: {device}")

    train_tfms, test_tfms = get_pacemaker_transforms(224)

    # Load dataset
    full_train = datasets.ImageFolder(root=os.path.join(args.data_dir, "Train"), transform=train_tfms)
    train_subset, val_subset = split_dataset_by_labels(full_train, test_size=0.2, random_state=42)

    train_loader = DataLoader(train_subset, batch_size=args.batch_size, shuffle=True, num_workers=2)
    val_loader   = DataLoader(val_subset, batch_size=args.batch_size, shuffle=False, num_workers=2)

    num_classes = len(full_train.classes)

    # Train all three models
    models_cfg = {
        "imagenet": models.vit_b_16(weights="IMAGENET1K_V1"),
        "clip": timm.create_model("vit_base_patch16_clip_224.openai", pretrained=True, num_classes=num_classes),
        "dino": timm.create_model("vit_small_patch16_224.dino", pretrained=True)
    }

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

        model = model.to(device)
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.AdamW(model.parameters(), lr=args.lr)
        save_path = f"vit_{name}_pacemaker.pth"
        plot_path = f"vit_{name}_pacemaker.png"

        print(f"\nTraining {name.upper()} model...")
        train_model(model, train_loader, val_loader, criterion, optimizer, device, args.epochs, save_path, plot_path)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", type=str, required=True, help="Path to Pacemaker dataset root")
    parser.add_argument("--epochs", type=int, default=15)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=3e-5)
    parser.add_argument("--device", type=str, default="cuda", help="Device to use: 'cuda' or 'cpu'")
    args = parser.parse_args()
    main(args)