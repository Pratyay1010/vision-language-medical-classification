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
from src.datasets import OrthonetDataset, get_imagenet_transforms, get_orthonet_augmented_transforms
from src.evaluator import PredictionCollector
from src.utils import get_class_weights

# Setup
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model, preprocess = clip.load("ViT-B/32", device=device, jit=False)
model.float(); [p.requires_grad_(False) for p in model.parameters()]
print(f"CLIP backbone frozen on {device}.")

# Transforms
# Reusing the modular transforms defined in src.datasets
train_tfms, _ = get_imagenet_transforms(224)
test_tfms, _ = get_orthonet_augmented_transforms(224)

# Utils
def evaluate_clip(model, prompt_learner, loader, device, use_img_cond=False):
    model.eval(); prompt_learner.eval()
    collector = PredictionCollector()
    with torch.no_grad():
        for imgs, labels in loader:
            imgs, labels = imgs.to(device), labels.to(device)
            img_features = model.encode_image(imgs).float()
            img_features /= img_features.norm(dim=-1, keepdim=True)
            if use_img_cond:
                text_features = prompt_learner(img_features)
                logits = 100.0 * torch.einsum("bd,bcd->bc", img_features, text_features)
            else:
                text_features = prompt_learner()
                logits = 100.0 * img_features @ text_features.T
            probs = logits.softmax(dim=-1)
            preds = probs.argmax(dim=-1)
            collector.update(labels, preds, probs)
            
    metrics = collector.compute_metrics()
    return {
        "Top-1 Accuracy": metrics["accuracy"], 
        "F1-score": metrics["f1_score"], 
        "AUC-ROC": metrics.get("auc", float("nan"))
    }

def plot_results(history, title, save_path=None):
    metrics, values = list(history.keys()), list(history.values())
    plt.figure(figsize=(7,5))
    plt.bar(metrics, values, color=["skyblue", "salmon", "orange"])
    plt.ylabel("Score"); plt.title(title); plt.ylim(0, 1)
    if save_path: plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.show()

# Prompt Learners
class CoOpPromptLearner(nn.Module):
    def __init__(self, clip_model, classnames, n_ctx=16, device="cuda"):
        super().__init__()
        self.model, self.n_ctx = clip_model, n_ctx
        self.ctx_dim = clip_model.token_embedding.embedding_dim
        self.ctx = nn.Parameter(torch.randn(self.n_ctx, self.ctx_dim, device=device) * 0.02)
        tokenized = clip.tokenize([c.replace("_"," ") for c in classnames]).to(device)
        self.register_buffer("tokenized", tokenized)
    def forward(self):
        token_embeddings = self.model.token_embedding(self.tokenized)
        token_embeddings[:,1:1+self.n_ctx,:] = self.ctx
        x = token_embeddings + self.model.positional_embedding
        x = self.model.transformer(x.permute(1,0,2)).permute(1,0,2)
        x = self.model.ln_final(x[:,0,:]) @ self.model.text_projection
        return x / x.norm(dim=-1, keepdim=True)

class CoCoOpPromptLearner(nn.Module):
    def __init__(self, clip_model, classnames, n_ctx=2, meta_hidden_dim=64, device="cuda"):
        super().__init__()
        self.model, self.n_ctx, self.num_classes = clip_model, n_ctx, len(classnames)
        self.ctx_dim = clip_model.token_embedding.embedding_dim
        self.ctx = nn.Parameter(torch.randn(self.n_ctx, self.ctx_dim, device=device) * 0.02)
        tokenized = clip.tokenize([c.replace("_"," ") for c in classnames]).to(device)
        self.register_buffer("tokenized", tokenized)
        self.register_buffer("token_embeddings", clip_model.token_embedding(tokenized))
        self.register_buffer("positional_embedding", clip_model.positional_embedding)
        img_feat_dim = clip_model.visual.output_dim
        self.meta_net = nn.Sequential(nn.Linear(img_feat_dim, meta_hidden_dim), nn.ReLU(),
                                      nn.Linear(meta_hidden_dim, self.n_ctx*self.ctx_dim)).to(device)
    def forward(self, img_features):
        B = img_features.size(0); text_features = []
        delta_ctx = self.meta_net(img_features).view(B,self.n_ctx,self.ctx_dim)
        for i in range(self.num_classes):
            token_emb = self.token_embeddings[i].unsqueeze(0).expand(B,-1,-1).clone()
            token_emb[:,1:1+self.n_ctx,:] += self.ctx.unsqueeze(0)+delta_ctx
            x = token_emb + self.positional_embedding.unsqueeze(0)
            x = self.model.transformer(x.permute(1,0,2)).permute(1,0,2)
            x = self.model.ln_final(x[:,0,:]) @ self.model.text_projection
            text_features.append(x/x.norm(dim=-1,keepdim=True))
        return torch.stack(text_features,dim=1)

class MaPLePromptLearner(nn.Module):
    def __init__(self, clip_model, classnames, n_ctx=2, meta_hidden_dim=64, device="cuda"):
        super().__init__()
        self.model, self.n_ctx, self.num_classes = clip_model, n_ctx, len(classnames)
        self.ctx_dim = clip_model.token_embedding.embedding_dim
        self.ctx_shared = nn.Parameter(torch.randn(self.n_ctx,self.ctx_dim,device=device)*0.02)
        self.ctx_image  = nn.Parameter(torch.zeros(self.n_ctx,self.ctx_dim,device=device))
        tokenized = clip.tokenize([c.replace("_"," ") for c in classnames]).to(device)
        self.register_buffer("token_embeddings", clip_model.token_embedding(tokenized))
        self.register_buffer("positional_embedding", clip_model.positional_embedding)
        img_feat_dim = clip_model.visual.output_dim
        self.meta_net = nn.Sequential(nn.Linear(img_feat_dim,meta_hidden_dim),nn.ReLU(),
                                      nn.Linear(meta_hidden_dim,self.n_ctx*self.ctx_dim)).to(device)
    def forward(self, img_features):
        B = img_features.size(0); delta_img = self.meta_net(img_features).view(B,self.n_ctx,self.ctx_dim)
        text_features = []
        for i in range(self.num_classes):
            token_emb = self.token_embeddings[i].unsqueeze(0).expand(B,-1,-1).clone()
            delta_ctx = self.ctx_shared.unsqueeze(0)+self.ctx_image.unsqueeze(0)+delta_img
            token_emb[:,1:1+self.n_ctx,:] += delta_ctx
            x = token_emb + self.positional_embedding.unsqueeze(0)
            x = self.model.transformer(x.permute(1,0,2)).permute(1,0,2)
            x = self.model.ln_final(x[:,0,:]) @ self.model.text_projection
            text_features.append(x/x.norm(dim=-1,keepdim=True))
        return torch.stack(text_features,dim=1)
    
# Training Function
def train_prompt_learner(name, learner, use_img_cond=False, epochs=50, lr=5e-4):
    class_weights = get_class_weights(train_dataset, device)
    loss_fn = nn.CrossEntropyLoss(weight=class_weights)
    optimizer = optim.AdamW(learner.parameters(), lr=lr)
    best_acc, history = 0, None
    for epoch in range(1, epochs+1):
        learner.train(); total_loss=0
        for imgs, labels in tqdm(train_loader, desc=f"{name} Epoch {epoch}"):
            imgs, labels = imgs.to(device), labels.to(device)
            optimizer.zero_grad()
            img_features = model.encode_image(imgs).float()
            img_features /= img_features.norm(dim=-1, keepdim=True)
            if use_img_cond:
                text_features = learner(img_features)
                logits = 100.0*torch.einsum("bd,bcd->bc", img_features,text_features)
            else:
                text_features = learner()
                logits = 100.0*img_features @ text_features.T
            loss = loss_fn(logits, labels); loss.backward(); optimizer.step()
            total_loss+=loss.item()
        history = evaluate_clip(model, learner, val_loader, device, use_img_cond)
        print(f"{name} Epoch {epoch}: {history}")
        if history["Top-1 Accuracy"]>best_acc:
            best_acc=history["Top-1 Accuracy"]; torch.save(learner.state_dict(), f"best_{name.lower()}.pt")
    plot_results(history, f"{name} Validation Performance", f"{name.lower()}_results.png")

# Main with Argparse
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train CLIP Prompt Learners on Orthonet")
    parser.add_argument("--method", type=str, default="coop", choices=["CoOp", "CoCoOp", "MaPLe"],
                        help="Prompt learning method")
    parser.add_argument("--epochs", type=int, default=50, help="Number of training epochs")
    parser.add_argument("--lr", type=float, default=5e-4, help="Learning rate")
    parser.add_argument("--batch_size", type=int, default=32, help="Batch size")
    parser.add_argument("--csv_dir", type=str, required=True, help="Path to CSV directory")
    parser.add_argument("--img_dir", type=str, required=True, help="Path to image directory")
    args = parser.parse_args()

    # Load Datasets using modular OrthonetDataset
    train_dataset = OrthonetDataset(csv_file=f"{args.csv_dir}/train.csv", root_dir=args.img_dir, transform=train_tfms)
    test_dataset  = OrthonetDataset(csv_file=f"{args.csv_dir}/test.csv",  root_dir=args.img_dir, transform=test_tfms)
    train_loader  = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True,  num_workers=4)
    val_loader    = DataLoader(test_dataset,  batch_size=args.batch_size, shuffle=False, num_workers=4)

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

    # Initialize learner
    if args.method.lower() == "coop":
        learner = CoOpPromptLearner(model, class_names).to(device)
        use_img_cond = False
    elif args.method.lower() == "cocoop":
        learner = CoCoOpPromptLearner(model, class_names).to(device)
        use_img_cond = True
    elif args.method.lower() == "maple":
        learner = MaPLePromptLearner(model, class_names).to(device)
        use_img_cond = True

    train_prompt_learner(args.method.upper(), learner, use_img_cond=use_img_cond, epochs=args.epochs, lr=args.lr)