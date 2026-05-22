import os
import argparse
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
from torchvision import datasets
import clip
import matplotlib.pyplot as plt
from tqdm import tqdm

# Reusable imports from src
from src.datasets import MappedImageFolder
from src.evaluator import PredictionCollector

# Parse arguments
parser = argparse.ArgumentParser(description="Stage 1: Train CoOp / CoCoOp / MaPLe")
parser.add_argument("--model", type=str, required=True, choices=["CoOp", "CoCoOp", "MaPLe"],
                    help="Which prompt learner to train")
parser.add_argument("--train_dir", type=str, required=True, help="Path to training dataset")
parser.add_argument("--test_dir", type=str, required=True, help="Path to test dataset")
parser.add_argument("--batch_size", type=int, default=32)
parser.add_argument("--epochs", type=int, default=30)
parser.add_argument("--lr", type=float, default=5e-5)
args = parser.parse_args()

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
MANUFACTURERS = ["BIO", "BOS", "MDT", "SOR", "STJ"]

# Dataset & Preprocess
model_clip, preprocess = clip.load("ViT-B/32", device=device, jit=False)
model_clip.float()
for p in model_clip.parameters():
    p.requires_grad = False
model_clip.eval()

train_raw = datasets.ImageFolder(args.train_dir, transform=preprocess)
test_raw = datasets.ImageFolder(args.test_dir, transform=preprocess)

# Using modular MappedImageFolder imported from src.datasets
train_dataset_full = MappedImageFolder(train_raw, MANUFACTURERS)
test_dataset = MappedImageFolder(test_raw, MANUFACTURERS)

val_ratio = 0.2
train_size = int((1 - val_ratio) * len(train_dataset_full))
val_size = len(train_dataset_full) - train_size
train_dataset, val_dataset = random_split(train_dataset_full, [train_size, val_size])

train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=2)
val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=2)
test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, num_workers=2)

# Prompt Learners
class CoOpPromptLearner(nn.Module):
    def __init__(self, clip_model, classnames, n_ctx=8, context_init="a photo of a", device="cuda"):
        super().__init__()
        self.model = clip_model
        self.classnames = classnames
        self.n_ctx = n_ctx
        self.device = device
        self.dtype = next(clip_model.parameters()).dtype

        ctx_tokens = clip.tokenize([context_init]).to(device)
        with torch.no_grad():
            embed = clip_model.token_embedding(ctx_tokens).to(self.dtype)
        self.ctx = nn.Parameter(embed[0, 1:1+n_ctx, :].clone())

        prompt_texts = [f"{context_init} {name}" for name in classnames]
        self.register_buffer("tokenized_prompts", clip.tokenize(prompt_texts).to(device))

    def forward(self):
        token_embedding = self.model.token_embedding(self.tokenized_prompts).to(self.ctx.dtype)
        token_embedding[:, 1:1+self.n_ctx, :] = self.ctx.unsqueeze(0).expand(token_embedding.shape[0], -1, -1)
        x = token_embedding + self.model.positional_embedding.to(self.ctx.dtype)
        x = x.permute(1, 0, 2)
        x = self.model.transformer(x)
        x = x.permute(1, 0, 2)
        x = self.model.ln_final(x[:, 0, :])
        x = x @ self.model.text_projection
        return x / x.norm(dim=-1, keepdim=True)

class CoCoOpPromptLearner(nn.Module):
    def __init__(self, clip_model, classnames, n_ctx=8, hidden_dim=128, context_init="a photo of a", device="cuda"):
        super().__init__()
        self.clip_model = clip_model
        self.classnames = classnames
        self.n_ctx = n_ctx
        self.device = device
        self.dtype = next(clip_model.parameters()).dtype

        ctx_tokens = clip.tokenize([context_init]).to(device)
        with torch.no_grad():
            embed = clip_model.token_embedding(ctx_tokens).to(self.dtype)
        self.ctx = nn.Parameter(embed[0, 1:1 + n_ctx, :].clone())

        prompt_texts = [f"{context_init} {name}" for name in classnames]
        self.register_buffer("tokenized_prompts", clip.tokenize(prompt_texts).to(device))

    def forward(self, image_features=None):
        token_embeddings = self.clip_model.token_embedding(self.tokenized_prompts).to(self.ctx.dtype)
        token_embeddings[:, 1:1 + self.n_ctx, :] = self.ctx.unsqueeze(0).expand(token_embeddings.size(0), -1, -1)
        x = token_embeddings + self.clip_model.positional_embedding.to(self.ctx.dtype)
        x = x.permute(1, 0, 2)
        x = self.clip_model.transformer(x)
        x = x.permute(1, 0, 2)
        x = self.clip_model.ln_final(x[:, 0, :])
        x = x @ self.clip_model.text_projection
        return x / x.norm(dim=-1, keepdim=True)

class MaPLePromptLearner(nn.Module):
    def __init__(self, clip_model, classnames, n_ctx=8, n_aspects=3, hidden_dim=128, context_init="a photo of a", device="cuda"):
        super().__init__()
        self.model = clip_model
        self.classnames = classnames
        self.n_ctx = n_ctx
        self.n_aspects = n_aspects
        self.device = device
        self.dtype = next(clip_model.parameters()).dtype
        self.num_classes = len(classnames)

        ctx_tokens = clip.tokenize([context_init]).to(device)
        with torch.no_grad():
            embed = clip_model.token_embedding(ctx_tokens).to(self.dtype)
        self.ctx = nn.Parameter(torch.randn(n_aspects, n_ctx, embed.shape[-1], device=device) * 0.02)

        prompt_texts = [f"{context_init} {name}" for name in classnames]
        self.register_buffer("tokenized_prompts", clip.tokenize(prompt_texts).to(device))

        self.meta_nets = nn.ModuleList([
            nn.Sequential(
                nn.Linear(clip_model.visual.output_dim, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, n_ctx * self.ctx.shape[-1])
            ) for _ in range(n_aspects)
        ])

        self.positional_embedding = clip_model.positional_embedding.to(self.dtype)

    def forward(self, img_features=None):
        batch_size = img_features.shape[0] if img_features is not None else 1
        num_classes = self.num_classes

        token_embeddings = self.model.token_embedding(self.tokenized_prompts).to(self.dtype)
        seq_len = token_embeddings.shape[1]

        token_embeddings = token_embeddings.unsqueeze(0).expand(batch_size, num_classes, seq_len, -1).clone()
        combined_prompts = torch.zeros(batch_size, num_classes, self.model.text_projection.shape[1], device=self.device, dtype=self.dtype)

        for aspect_idx in range(self.n_aspects):
            ctx = self.ctx[aspect_idx].unsqueeze(0).expand(batch_size, -1, -1)
            if img_features is not None:
                delta = self.meta_nets[aspect_idx](img_features).view(batch_size, self.n_ctx, -1)
                ctx = ctx + delta

            prompts = token_embeddings.clone()
            prompts[:, :, 1:1 + self.n_ctx, :] = ctx.unsqueeze(1) + prompts[:, :, 1:1 + self.n_ctx, :]
            prompts = prompts + self.positional_embedding.unsqueeze(0).unsqueeze(0)

            x = prompts.view(batch_size * num_classes, seq_len, -1).permute(1, 0, 2)
            x = self.model.transformer(x)
            x = x.permute(1, 0, 2)
            x = self.model.ln_final(x[:, 0, :])
            x = x @ self.model.text_projection
            x = x / x.norm(dim=-1, keepdim=True)
            x = x.view(batch_size, num_classes, -1)

            combined_prompts += x
        combined_prompts /= self.n_aspects
        return combined_prompts

# Initialize learner
if args.model == "CoOp":
    prompt_learner = CoOpPromptLearner(model_clip, MANUFACTURERS).to(device)
elif args.model == "CoCoOp":
    prompt_learner = CoCoOpPromptLearner(model_clip, MANUFACTURERS).to(device)
else:
    prompt_learner = MaPLePromptLearner(model_clip, MANUFACTURERS).to(device)

optimizer = optim.AdamW(prompt_learner.parameters(), lr=args.lr)
loss_fn = nn.CrossEntropyLoss()

# Evaluation function
def evaluate(loader):
    model_clip.eval()
    prompt_learner.eval()
    collector = PredictionCollector()
    with torch.no_grad():
        for imgs, labels in loader:
            imgs, labels = imgs.to(device), labels.to(device)
            img_feats = model_clip.encode_image(imgs)
            img_feats = img_feats / img_feats.norm(dim=-1, keepdim=True)
            img_feats = img_feats.to(prompt_learner.ctx.dtype if hasattr(prompt_learner, "ctx") else torch.float32)

            if args.model == "MaPLe":
                text_feats = prompt_learner(img_feats)
                logits = 100.0 * torch.bmm(img_feats.unsqueeze(1), text_feats.permute(0, 2, 1)).squeeze(1)
            else:
                text_feats = prompt_learner()
                logits = 100.0 * img_feats @ text_feats.T

            probs = logits.softmax(dim=-1)
            preds = probs.argmax(dim=-1)
            collector.update(labels, preds, probs)
        torch.cuda.empty_cache()
        
    metrics = collector.compute_metrics()
    return metrics["accuracy"], metrics["f1_score"], metrics.get("auc", float("nan"))

# Training loop
train_losses, val_accs, val_f1s = [], [], []
best_val = 0

for epoch in range(1, args.epochs + 1):
    prompt_learner.train()
    total_loss = 0
    for imgs, labels in tqdm(train_loader, desc=f"Epoch {epoch}/{args.epochs}"):
        imgs, labels = imgs.to(device), labels.to(device)
        optimizer.zero_grad()
        img_feats = model_clip.encode_image(imgs)
        img_feats = img_feats / img_feats.norm(dim=-1, keepdim=True)
        if args.model == "MaPLe":
            text_feats = prompt_learner(img_feats)
            logits = 100.0 * torch.bmm(img_feats.unsqueeze(1), text_feats.permute(0, 2, 1)).squeeze(1)
        else:
            text_feats = prompt_learner()
            logits = 100.0 * img_feats @ text_feats.T
        loss = loss_fn(logits, labels)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    avg_loss = total_loss / len(train_loader)
    val_acc, val_f1, val_auc = evaluate(val_loader)
    train_losses.append(avg_loss)
    val_accs.append(val_acc)
    val_f1s.append(val_f1)
    print(f"Epoch {epoch} → Loss={avg_loss:.4f}, Val Acc={val_acc:.4f}, F1={val_f1:.4f}, AUC={val_auc:.4f}")
    if val_acc > best_val:
        best_val = val_acc
        torch.save(prompt_learner.state_dict(), f"stage1_{args.model}.pt")

# Test & Plot
print("\nEvaluating on test set...")
test_acc, test_f1, test_auc = evaluate(test_loader)
print(f"Final Test → Acc={test_acc:.4f}, F1={test_f1:.4f}, AUC={test_auc:.4f}")

epochs_range = list(range(1, args.epochs + 1))

plt.figure(figsize=(12,4))

# Train Loss
plt.subplot(1,3,1)
plt.plot(epochs_range, train_losses, marker='o', color='skyblue')
plt.title("Train Loss")
plt.xlabel("Epochs")
plt.ylabel("Loss")
plt.grid(True)

# Validation Accuracy
plt.subplot(1,3,2)
plt.plot(epochs_range, val_accs, marker='o', color='green')
plt.title("Validation Accuracy")
plt.xlabel("Epochs")
plt.ylabel("Accuracy")
plt.grid(True)

# Validation F1 Score
plt.subplot(1,3,3)
plt.plot(epochs_range, val_f1s, marker='o', color='orange')
plt.title("Validation F1 Score")
plt.xlabel("Epochs")
plt.ylabel("F1 Score")
plt.grid(True)

plt.tight_layout()

save_path = "prompt_learning_stage1_history.png"
plt.savefig(save_path, dpi=300, bbox_inches="tight")

plt.show()