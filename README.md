# Medical Vision-Language Learning for Implant Classification

<p align="center">
  <img src="assets/results/pacemaker_vit_clip.png" width="100%">
</p>

<p align="center">
  <strong>Vision Transformers • CLIP • Zero-Shot Learning • Prompt Learning • Medical Imaging</strong>
</p>

<p align="center">
  Vision-language learning pipelines for orthopedic implant and pacemaker classification using supervised ViTs, zero-shot CLIP, and prompt adaptation methods.
</p>



# Overview

This repository explores medical vision-language learning approaches for:

- orthopedic implant recognition
- pacemaker classification
- prompt-based medical perception

The project investigates:
- supervised Vision Transformers
- CLIP-based zero-shot inference
- prompt learning adaptation in specialized medical domains

Implemented methods include:
- ImageNet ViT
- CLIP ViT
- DINO ViT
- CoOp
- CoCoOp
- MaPLe



# Research Pipeline

<p align="center">
  <img src="assets/results/orthonet_zero_shot.png" width="420">
  <img src="assets/results/maple_results.png" width="420">
</p>

```text
Supervised ViTs
        ↓
Zero-Shot CLIP
        ↓
Prompt Learning Adaptation
```



# Dataset Examples

## Orthopedic Implant Samples

<p align="center">
  <img src="assets/examples/orthopedic/0008_32_20_2_A-P00_UNIL.png" width="260">
  <img src="assets/examples/orthopedic/0008_32_20_2_A-P00_MASK.png" width="260">
</p>

<p align="center">
  <sub>X-ray image and corresponding segmentation mask.</sub>
</p>



## Pacemaker Samples

<p align="center">
  <img src="assets/examples/pacemaker/IMP1508004.jpg" width="180">
  <img src="assets/examples/pacemaker/IMP1521150.jpg" width="180">
  <img src="assets/examples/pacemaker/IMP1534223.jpg" width="180">
</p>

<p align="center">
  <sub>Representative pacemaker radiographs used for manufacturer-level classification.</sub>
</p>



# Models Explored

## Vision Transformers
- ImageNet ViT
- CLIP ViT
- DINO ViT

## Vision-Language Models
- CLIP (ViT-B/32)

## Prompt Learning
- CoOp
- CoCoOp
- MaPLe



# Key Results

<p align="center">
  <img src="assets/results/pacemaker_vit_clip.png" width="850">
</p>

<br>

<p align="center">
  <img src="assets/results/orthonet_zero_shot.png" width="420">
  <img src="assets/results/maple_results.png" width="420">
</p>



# Dataset Note

The original datasets are not included in this repository due to size constraints.

This repository contains:
- sample medical images
- experiment outputs
- training visualizations
- evaluation results

to demonstrate the complete research pipeline.



# Installation

```bash
git clone https://github.com/Pratyay1010/vision-language-medical-classification.git

cd vision-language-medical-classification

pip install -r requirements.txt
```



# Training

## Train ViTs on OrthoNet

```bash
python scripts/train_orthonet_vit.py
```

## Train ViTs on Pacemaker Dataset

```bash
python scripts/train_pacemaker_vit.py
```



# Zero-Shot Evaluation

```bash
python scripts/zero_shot_orthonet.py

python scripts/zero_shot_pacemaker.py
```



# Prompt Learning

```bash
python scripts/prompt_learning_orthonet.py

python scripts/prompt_learning_pacemaker.py
```



# Features

- Modular research-oriented codebase
- Reusable datasets and evaluation utilities
- Vision-language learning workflows
- Zero-shot CLIP evaluation
- Prompt learning experimentation
- Medical imaging focus
