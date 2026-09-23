# ResNet-50 Benchmark Results: Food-101 Classification

**Member 2:** Matheesha Weerakoon  
**Module:** SE4050 — Deep Learning (2026)  
**Assigned Architecture:** Deep Residual Network (ResNet-50)  
**Learning Paradigm:** Two-Stage Transfer Learning & Selective Fine-Tuning  

---

## 1. Executive Performance Summary

The deep residual learning network (ResNet-50) was evaluated on the **Food-101 dataset** (101 categories, 101,000 images) following the SE4050 Controlled Experimental Protocol. Final evaluations were executed on the **strictly unseen held-out test split of 25,250 images**.

| Metric | Phase 1 (Feature Extraction) | Phase 2 (Stage 5 Fine-Tuned) | Final Unseen Test Set | Baseline Comparison (Custom CNN) |
| :--- | :---: | :---: | :---: | :---: |
| **Top-1 Accuracy** | 61.31% (Val) | 69.47% (Val) | **73.54%** | **+11.85%** (vs 61.69%) |
| **Top-5 Accuracy** | 84.65% (Val) | 88.65% (Val) | **92.02%** | **+9.90%** (vs 82.12%) |
| **Cross-Entropy Loss** | 1.6720 (Val) | 1.3744 (Val) | **1.1110** | **-0.3331** (vs 1.4441) |
| **Macro Precision** | — | — | **0.7411** | — |
| **Macro Recall** | — | — | **0.7354** | — |
| **Macro F1-Score** | — | — | **0.7355** | Balanced across 101 classes |
| **Inference Latency (T4 GPU)**| — | — | **10.98 ms / image** | Real-time (>91 FPS) |
| **Model Size on Disk** | 90.80 MB | 90.80 MB | **98.4 MB** | Pretrained weights |

---

## 2. Model Architecture & Complexity

- **Base Architecture:** ResNet-50 (He et al., 2016) pretrained on ImageNet ($1.2\times 10^6$ images, 1,000 classes).
- **Backbone Depth:** 50 parameterized layers (48 convolution layers, 1 max-pooling, 1 average-pooling).
- **Residual Blocks:** Bottleneck blocks with $1\times 1 \to 3\times 3 \to 1\times 1$ convolutions and identity/projection shortcut connections ($y = \mathcal{F}(x) + x$).
- **Custom Classification Head:**
  - `GlobalAveragePooling2D()`: Compresses $(7, 7, 2048)$ spatial feature map to a $2048$-dimensional vector without flattening parameters.
  - `BatchNormalization()`: Stabilizes post-pooling distribution across mini-batches.
  - `Dropout(0.3)`: Injects stochastic regularization during training.
  - `Dense(101, activation='softmax')`: Produces calibrated categorical probabilities across 101 food classes.
- **Parameter Breakdown:**
  - **Total Parameters:** $23,802,853$ ($90.80\text{ MB}$)
  - **Trainable Parameters:** $9,130,085$ ($34.83\text{ MB}$)
  - **Non-Trainable Parameters:** $14,672,768$ ($55.97\text{ MB}$)

---

## 3. Two-Stage Training Methodology

### Phase 1: Feature Extraction (Epochs 1–6)
- **Strategy:** Backbone completely frozen (`base_model.trainable = False`). Only the custom classification head was trained.
- **Optimizer:** Adam ($\beta_1=0.9, \beta_2=0.999$, initial $\text{lr} = 10^{-3}$).
- **Objective:** Learn a non-destructive mapping from generic ImageNet feature space to Food-101 class logits.
- **Convergence:** Rapid convergence within 6 epochs: Training accuracy reached $58.38\%$, Validation accuracy reached **$61.31\%$**, and Top-5 validation accuracy reached **$84.65\%$**.

### Phase 2: Selective Fine-Tuning (Epochs 7–20)
- **Strategy:** Unfroze Stage 5 residual blocks from layer `conv5_block1_out` onwards ($9,130,085$ trainable parameters) while keeping Stages 1–4 frozen to preserve low-level edge, texture, and shape primitives.
- **Batch Normalization Freezing:** All `BatchNormalization` layers inside the pretrained backbone remained strictly frozen (`layer.trainable = False`) to prevent small-batch statistics from corrupting running mean and variance estimates.
- **Optimizer:** Adam with reduced learning rate ($\text{lr} = 10^{-5}$).
- **Learning Rate Annealing:** At Epoch 18, `ReduceLROnPlateau(factor=0.2, patience=3)` reduced $\text{lr}$ to $2\times 10^{-6}$, achieving final validation accuracy of **$69.47\%$** and validation loss of **$1.3744$**.

---

## 4. Standardized Output Artifacts Catalog

This directory contains the 7 standardized files required for multi-model cross-comparison:

1. **[`metrics.json`](file:///home/matheeshawsl/sliit/DS/assigment/food-classification-deep-learning/results/resnet50/metrics.json)**: Machine-readable scalar metrics conforming to the assignment schema.
2. **[`classification_report.json`](file:///home/matheeshawsl/sliit/DS/assigment/food-classification-deep-learning/results/resnet50/classification_report.json)**: Per-class precision, recall, F1-score, and support for all 101 food categories.
3. **[`confusion_matrix.png`](file:///home/matheeshawsl/sliit/DS/assigment/food-classification-deep-learning/results/resnet50/confusion_matrix.png)**: High-resolution ($300\text{ DPI}$) normalized $101 \times 101$ heatmap on the unseen test set.
4. **[`training_curves.png`](file:///home/matheeshawsl/sliit/DS/assigment/food-classification-deep-learning/results/resnet50/training_curves.png)**: Publication-grade dual-panel plot showing cross-entropy loss and accuracy progression across both phases, with vertical divider at Epoch 6.
5. **[`history.csv`](file:///home/matheeshawsl/sliit/DS/assigment/food-classification-deep-learning/results/resnet50/history.csv)**: Epoch-by-epoch tracking of `loss`, `accuracy`, `top_5_accuracy`, `val_loss`, `val_accuracy`, `val_top_5_accuracy`, and `learning_rate`.
6. **[`config.yaml`](file:///home/matheeshawsl/sliit/DS/assigment/food-classification-deep-learning/results/resnet50/config.yaml)**: Comprehensive declaration of data splits, preprocessing pipelines, regularizers, and optimizer schedules.
7. **[`model_summary.txt`](file:///home/matheeshawsl/sliit/DS/assigment/food-classification-deep-learning/results/resnet50/model_summary.txt)**: Full Keras layer summary detailing output tensor shapes and parameter allocations.
