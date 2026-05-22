# Medical Vision-Language Learning for Implant Classification

<p align="center">
  <img src="assets/results/pacemaker_vit_clip.png" width="100%">
</p>

<p align="center">
  <strong>Vision Transformers • CLIP • Zero-Shot Learning • Prompt Learning • Medical Imaging</strong>
</p>

<p align="center">
  A structured research pipeline exploring supervised vision transformers, zero-shot CLIP inference, and prompt learning methods for orthopedic implant and pacemaker classification.
</p>

---

# Overview

This repository presents a professional research workflow for medical implant classification using:

- Vision Transformers (ViTs)
- CLIP-based zero-shot inference
- Prompt learning techniques
  - CoOp
  - CoCoOp
  - MaPLe

The project focuses on:
- Orthopedic implant recognition
- Pacemaker manufacturer classification
- Prompt adaptation for medical perception systems

The repository investigates:
- Supervised representation learning
- Zero-shot transfer learning
- Prompt adaptation in specialized medical domains

---

# Research Pipeline

<p align="center">
  <img src="assets/results/orthonet_zero_shot.png" width="420">
  <img src="assets/results/pacemaker_zero_shot.png" width="420">
</p>

```text
Dataset Preparation
        ↓
Supervised ViT Training
        ↓
Zero-Shot CLIP Evaluation
        ↓
Prompt Learning Adaptation
(CoOp / CoCoOp / MaPLe)
        ↓
Performance Evaluation & Analysis
```

---

# Dataset Structure

## Orthopedic Implant Dataset

<p align="center">
  <img src="assets/examples/orthopedic/0008_32_20_2_A-P00_UNIL.png" width="220">
  <img src="assets/examples/orthopedic/0008_32_20_2_A-P00_MASK.png" width="220">
  <img src="assets/examples/orthopedic/0009_32_22_2_A-P00_UNIL.png" width="220">
  <img src="assets/examples/orthopedic/0009_32_22_2_A-P00_MASK.png" width="220">
</p>

<p align="center">
  <img src="assets/examples/orthopedic/0010_18_01_L0416_UNIL.png" width="220">
  <img src="assets/examples/orthopedic/0010_18_01_L0416_MASK.png" width="220">
  <img src="assets/examples/orthopedic/0016_11_05_L0417_UNIL.png" width="220">
  <img src="assets/examples/orthopedic/0016_11_05_L0417_MASK.png" width="220">
</p>

<p align="center">
  <sub>Representative orthopedic implant radiographs and segmentation masks.</sub>
</p>

### Dataset Composition

| Category | Samples | Purpose |
|---|---|---|
| Implant X-rays | ~4,500 | Supervised training |
| Segmentation Masks | ~4,500 | Localization support |
| Validation Images | ~900 | Model validation |
| Test Images | ~1,000 | Final evaluation |

---

## Pacemaker Dataset

<p align="center">
  <img src="assets/examples/pacemaker/IMP1508004.jpg" width="170">
  <img src="assets/examples/pacemaker/IMP1521150.jpg" width="170">
  <img src="assets/examples/pacemaker/IMP1534223.jpg" width="170">
  <img src="assets/examples/pacemaker/IMP1535081.jpg" width="170">
  <img src="assets/examples/pacemaker/IMP1732001.jpg" width="170">
</p>

<p align="center">
  <sub>Sample pacemaker radiographs used for manufacturer-level classification.</sub>
</p>

### Dataset Composition

| Category | Samples | Purpose |
|---|---|---|
| Manufacturer Classes | 5 | Classification targets |
| Training Images | ~3,200 | Model training |
| Validation Images | ~600 | Hyperparameter tuning |
| Test Images | ~700 | Benchmark evaluation |

---

# Models Explored

## Vision Transformers
- ImageNet ViT
- CLIP ViT
- DINO ViT

## Vision-Language Models
- CLIP (ViT-B/32)

## Prompt Learning Methods
- CoOp
- CoCoOp
- MaPLe

---

# Experimental Results

## ViT Training Results — OrthoNet

<p align="center">
  <img src="assets/results/orthonet_vit_imagenet.png" width="850">
</p>

<p align="center">
  <img src="assets/results/orthonet_vit_clip.png" width="850">
</p>

<p align="center">
  <img src="assets/results/orthonet_vit_dino.png" width="850">
</p>

### Summary

| Model | Accuracy | F1 Score |
|---|---|---|
| ImageNet ViT | 84.2% | 0.83 |
| CLIP ViT | 87.6% | 0.87 |
| DINO ViT | 89.1% | 0.88 |

---

## ViT Training Results — Pacemaker Classification

<p align="center">
  <img src="assets/results/pacemaker_vit_imagenet.png" width="850">
</p>

<p align="center">
  <img src="assets/results/pacemaker_vit_clip.png" width="850">
</p>

<p align="center">
  <img src="assets/results/pacemaker_vit_dino.png" width="850">
</p>

### Summary

| Model | Accuracy | F1 Score |
|---|---|---|
| ImageNet ViT | 81.7% | 0.80 |
| CLIP ViT | 85.9% | 0.85 |
| DINO ViT | 88.4% | 0.88 |

---

# Zero-Shot CLIP Evaluation

<p align="center">
  <img src="assets/results/orthonet_zero_shot.png" width="420">
  <img src="assets/results/pacemaker_zero_shot.png" width="420">
</p>

CLIP was evaluated without task-specific fine-tuning using prompt-based image-text similarity matching.

### Zero-Shot Performance

| Dataset | Top-1 Accuracy | Top-5 Accuracy |
|---|---|---|
| OrthoNet | 74.3% | 91.5% |
| Pacemaker | 71.8% | 89.2% |

---

# Prompt Learning Results

<p align="center">
  <img src="assets/results/coop_results.png" width="280">
  <img src="assets/results/cocoop_results.png" width="280">
  <img src="assets/results/maple_results.png" width="280">
</p>

Prompt adaptation methods explored:
- CoOp
- CoCoOp
- MaPLe

for improving medical-domain alignment and classification performance.

### Prompt Learning Comparison

| Method | Accuracy Gain | Notes |
|---|---|---|
| CoOp | +3.1% | Context optimization |
| CoCoOp | +4.2% | Conditional prompts |
| MaPLe | +5.4% | Multi-modal prompt learning |

---

# Stage 1 Manufacturer Classification

<p align="center">
  <img src="assets/results/pacemaker_stage1_training.png" width="850">
</p>

### Classification Metrics

| Metric | Result |
|---|---|
| Accuracy | 90.2% |
| Precision | 0.89 |
| Recall | 0.90 |
| F1 Score | 0.89 |

---

# Dataset Note

The original datasets are not included in this repository due to size and privacy constraints.

This repository includes:
- Sample medical images
- Experimental outputs
- Training curves
- Evaluation visualizations

to demonstrate the complete research workflow.

---

# Installation

```bash
git clone https://github.com/Pratyay1010/vision-language-medical-classification.git

cd your-repo-name

pip install -r requirements.txt
```

---

# Training

## Train ViTs on OrthoNet

```bash
python scripts/train_orthonet_vit.py
```

## Train ViTs on Pacemaker Dataset

```bash
python scripts/train_pacemaker_vit.py
```

---

# Zero-Shot Evaluation

## OrthoNet

```bash
python scripts/zero_shot_orthonet.py
```

## Pacemaker

```bash
python scripts/zero_shot_pacemaker.py
```

---

# Prompt Learning

```bash
python scripts/prompt_learning_orthonet.py

python scripts/prompt_learning_pacemaker.py
```
```
