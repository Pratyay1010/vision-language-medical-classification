import os
import argparse
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import models
import timm
from tqdm import tqdm
import matplotlib.pyplot as plt

# Reusable imports from src
from src.datasets import OrthonetDataset, get_imagenet_transforms
from src.evaluator import evaluate_classification
from src.utils import get_class_weights

# Training / Evaluation
def train_model(model, train_loader, test_loader, criterion, optimizer, device, epochs, save_path, plot_path):
    train_acc_list, test_acc_list = [], []

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
        test_acc = evaluate_classification(model, test_loader, device)["accuracy"] * 100
        train_acc_list.append(train_acc)
        test_acc_list.append(test_acc)

        print(f"Epoch {epoch+1}: Loss={running_loss/len(train_loader):.4f}, "
              f"Train Acc={train_acc:.2f}%, Test Acc={test_acc:.2f}%")

    # Save model
    torch.save(model.state_dict(), save_path)
    print(f"Model saved to {save_path}")

    # Plot training/test accuracy
    plt.figure(figsize=(8,5))
    plt.plot(range(1, epochs+1), train_acc_list, label="Train Acc", marker='o')
    plt.plot(range(1, epochs+1), test_acc_list, label="Test Acc", marker='s')
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy (%)")
    plt.title(f"Training and Test Accuracy ({os.path.basename(save_path).split('.')[0]})")
    plt.ylim(0, 100)
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    for x, y in enumerate(train_acc_list, start=1):
        plt.text(x, y+1, f"{y:.1f}", ha='center', fontsize=8)
    for x, y in enumerate(test_acc_list, start=1):
        plt.text(x, y+1, f"{y:.1f}", ha='center', fontsize=8)
    plt.savefig(plot_path)
    print(f"Accuracy plot saved to {plot_path}")
    plt.close()

# Main
def main(args):
    device = torch.device(args.device if torch.cuda.is_available() or args.device=="cpu" else "cpu")
    print(f"Using device: {device}")

    train_tfms, test_tfms = get_imagenet_transforms(224)

    train_dataset = OrthonetDataset(csv_file=os.path.join(args.csv_dir, "train.csv"),
                                    root_dir=args.image_dir,
                                    transform=train_tfms)
    test_dataset = OrthonetDataset(csv_file=os.path.join(args.csv_dir, "test.csv"),
                                   root_dir=args.image_dir,
                                   transform=test_tfms)
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True, num_workers=1)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False, num_workers=1)

    num_classes = len(train_dataset.label2idx)

    # Compute class weights using src.utils helper
    class_weights = get_class_weights(train_dataset, device)
    criterion = nn.CrossEntropyLoss(weight=class_weights)

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
        optimizer = optim.AdamW(model.parameters(), lr=args.lr)
        save_path = f"vit_{name}_orthonet.pth"
        plot_path = f"vit_{name}_orthonet.png"
        print(f"\nTraining {name.upper()} model...")
        train_model(model, train_loader, test_loader, criterion, optimizer, device, args.epochs, save_path, plot_path)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv_dir", type=str, required=True, help="Path to CSV directory")
    parser.add_argument("--image_dir", type=str, required=True, help="Path to image directory")
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--device", type=str, default="cuda", help="Device to use: 'cuda' or 'cpu'")
    args = parser.parse_args()
    main(args)