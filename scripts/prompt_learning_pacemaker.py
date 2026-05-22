import os
import argparse
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import clip
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

# Reusable imports from src
from src.datasets import PacemakerDataset, get_pacemaker_transforms, split_dataset_by_labels
from src.evaluator import PredictionCollector

# Argument parser
parser = argparse.ArgumentParser(description="Train CoOp / CoCoOp / MaPLe on Pacemaker Dataset")
parser.add_argument("--train_dir", type=str, required=True)
parser.add_argument("--test_dir", type=str, required=True)
parser.add_argument("--batch_size", type=int, default=32)
parser.add_argument("--epochs", type=int, default=30)
parser.add_argument("--lr", type=float, default=5e-5)
parser.add_argument("--n_ctx", type=int, default=16)
parser.add_argument("--model_type", type=str, choices=["coop", "cocoop", "maple"], default="coop")
parser.add_argument("--save_path", type=str, default="best_model.pt")
args = parser.parse_args()

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Transforms & DataLoaders
train_tfms, test_tfms = get_pacemaker_transforms(224)

train_dataset_full = PacemakerDataset(args.train_dir, transform=train_tfms)
test_dataset = PacemakerDataset(args.test_dir, transform=test_tfms)
classnames = train_dataset_full.classes

train_dataset, val_dataset = split_dataset_by_labels(train_dataset_full, test_size=0.2, random_state=42)

train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=4)
val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=4)
test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, num_workers=4)

# Load CLIP model
model, preprocess = clip.load("ViT-B/32", device=device, jit=False)
model.float()
for p in model.parameters():
    p.requires_grad = False
model.eval()

# Prompt Learners
class CoOpPromptLearner(nn.Module):
    def __init__(self, clip_model, classnames, n_ctx=16, device="cuda"):
        super().__init__()
        self.model = clip_model
        self.n_ctx = n_ctx
        self.ctx_dim = clip_model.token_embedding.embedding_dim
        self.device = device
        self.dtype = next(clip_model.parameters()).dtype
        self.ctx = nn.Parameter(torch.randn(n_ctx, self.ctx_dim, dtype=self.dtype, device=device) * 0.02)
        tokenized = clip.tokenize([c.replace("_", " ") for c in classnames]).to(device)
        self.register_buffer("tokenized", tokenized)

    def forward(self):
        token_embeddings = self.model.token_embedding(self.tokenized).to(self.ctx.dtype)
        token_embeddings[:, 1:1+self.n_ctx, :] = self.ctx.unsqueeze(0).expand(token_embeddings.size(0), -1, -1)
        x = token_embeddings + self.model.positional_embedding.to(self.ctx.dtype)
        x = x.permute(1, 0, 2)
        x = self.model.transformer(x)
        x = x.permute(1, 0, 2)
        x = self.model.ln_final(x[:, 0, :])
        x = x @ self.model.text_projection.to(self.ctx.dtype)
        return x / x.norm(dim=-1, keepdim=True)

class CoCoOpPromptLearner(CoOpPromptLearner):
    def forward(self, image_features=None):
        text_features = super().forward()
        return text_features

class MaPLePromptLearner(CoOpPromptLearner):
    def forward(self, image_features=None):
        text_features = super().forward()
        if image_features is not None:
            text_features = text_features + 0.1 * image_features.mean(dim=0, keepdim=True)
        return text_features

# Initialize prompt learner
if args.model_type == "coop":
    prompt_learner = CoOpPromptLearner(model, classnames, n_ctx=args.n_ctx, device=device).to(device)
elif args.model_type == "cocoop":
    prompt_learner = CoCoOpPromptLearner(model, classnames, n_ctx=args.n_ctx, device=device).to(device)
else:
    prompt_learner = MaPLePromptLearner(model, classnames, n_ctx=args.n_ctx, device=device).to(device)

optimizer = optim.AdamW(prompt_learner.parameters(), lr=args.lr)
loss_fn = nn.CrossEntropyLoss()

# Evaluation function
def evaluate(loader):
    model.eval()
    prompt_learner.eval()
    collector = PredictionCollector()
    with torch.no_grad():
        text_features = prompt_learner().to(next(model.parameters()).dtype)
        for imgs, labels in loader:
            imgs, labels = imgs.to(device), labels.to(device)
            img_features = model.encode_image(imgs).to(text_features.dtype)
            img_features = img_features / img_features.norm(dim=-1, keepdim=True)
            if args.model_type in ["cocoop", "maple"]:
                text_features = prompt_learner(img_features)
            logits = 100.0 * img_features @ text_features.T
            probs = logits.softmax(dim=-1)
            preds = probs.argmax(dim=-1)
            collector.update(labels, preds, probs)
            
    metrics = collector.compute_metrics(top_k=3)
    return (
        metrics["accuracy"], 
        metrics["top_3_accuracy"], 
        metrics["f1_score"], 
        metrics.get("auc", float("nan"))
    )

# Training loop
history = {
    "train_loss": [],
    "val_top1": [],
    "val_top3": [],
    "val_f1": [],
    "val_auc": []
}

best_val = 0
for epoch in range(1, args.epochs + 1):
    prompt_learner.train()
    total_loss = 0
    for imgs, labels in tqdm(train_loader, desc=f"Epoch {epoch}"):
        imgs, labels = imgs.to(device), labels.to(device)
        optimizer.zero_grad()
        img_features = model.encode_image(imgs).to(prompt_learner.ctx.dtype)
        img_features = img_features / img_features.norm(dim=-1, keepdim=True)
        if args.model_type in ["cocoop", "maple"]:
            text_features = prompt_learner(img_features)
        else:
            text_features = prompt_learner()
        logits = 100.0 * img_features @ text_features.T
        loss = loss_fn(logits, labels)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()

    avg_loss = total_loss / len(train_loader)
    val_top1, val_top3, val_f1, val_auc = evaluate(val_loader)

    # Save metrics
    history["train_loss"].append(avg_loss)
    history["val_top1"].append(val_top1)
    history["val_top3"].append(val_top3)
    history["val_f1"].append(val_f1)
    history["val_auc"].append(val_auc)

    print(f"Epoch {epoch} → Loss={avg_loss:.4f}, Val Top1={val_top1:.4f}, Top3={val_top3:.4f}, F1={val_f1:.4f}, AUC={val_auc:.4f}")
    if val_top1 > best_val:
        best_val = val_top1
        torch.save(prompt_learner.state_dict(), args.save_path)
        print(f"Saved best model to {args.save_path}")

# Test evaluation
print("\nFinal Test Evaluation:")
test_top1, test_top3, test_f1, test_auc = evaluate(test_loader)
print(f"Test → Top-1={test_top1:.4f}, Top-3={test_top3:.4f}, F1={test_f1:.4f}, AUC={test_auc:.4f}")

epochs_range = range(1, args.epochs + 1)

plt.figure(figsize=(14,4))

# Train Loss
plt.subplot(1,3,1)
plt.plot(epochs_range, history["train_loss"], marker='o', color='skyblue')
plt.title("Train Loss")
plt.xlabel("Epochs")
plt.ylabel("Loss")
plt.grid(True)

# Validation Top-1 Accuracy
plt.subplot(1,3,2)
plt.plot(epochs_range, history["val_top1"], marker='o', color='green', label="Top-1")
plt.plot(epochs_range, history["val_top3"], marker='x', color='orange', label="Top-3")
plt.title("Validation Accuracy")
plt.xlabel("Epochs")
plt.ylabel("Accuracy")
plt.legend()
plt.grid(True)

# Validation F1 & AUC
plt.subplot(1,3,3)
plt.plot(epochs_range, history["val_f1"], marker='o', color='purple', label="F1 Score")
plt.plot(epochs_range, history["val_auc"], marker='x', color='red', label="AUC")
plt.title("Validation F1 & AUC")
plt.xlabel("Epochs")
plt.ylabel("Score")
plt.legend()
plt.grid(True)

plt.tight_layout()
plt.savefig("prompt_learning_history.png", dpi=300, bbox_inches="tight")
plt.show()