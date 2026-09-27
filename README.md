# SE4050 Deep Learning — Multi-Class Food Classification Benchmark (Food-101)

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![TensorFlow 2.15+](https://img.shields.io/badge/TensorFlow-2.15+-orange.svg)](https://tensorflow.org/)
[![Dataset: Food-101](<https://img.shields.io/badge/Dataset-Food--101%20(101k%20Images)-purple.svg>)](https://data.vision.ee.ethz.ch/cvl/datasets_extra/food-101/)

This repository contains the official implementation, experimental benchmarks, and comparative evaluation for the **SE4050 – Deep Learning (2026)** assignment at the **Sri Lanka Institute of Information Technology (SLIIT)**.

We benchmark four distinct Convolutional Neural Network (CNN) paradigms on the challenging **Food-101** visual recognition dataset under strictly controlled, reproducible experimental conditions (identical seed `42`, $224 \times 224$ input resolution, batch size 32, and an unseen held-out test split of 25,250 images).

---

## 1. Group Members & Assigned Architectures

| Student Name          |   Student ID   |   Institutional Email    | Assigned Model & Core Focus                                                                                                                                  |
| :-------------------- | :------------: | :----------------------: | :----------------------------------------------------------------------------------------------------------------------------------------------------------- |
| **Herath H.M.N.P**    | **IT23221178** | `it23221178@my.sliit.lk` | **Custom CNN (from scratch)**<br>• 4-stage convolutional hierarchy with GAP<br>• Core pipeline, EDA & zero-leakage splits                                    |
| **Weerakoon W.M.M.B** | **IT23155152** | `it23155152@my.sliit.lk` | **ResNet-50 (Transfer Learning)**<br>• Deep residual learning & identity shortcuts<br>• Two-phase fine-tuning & cloud server analysis                        |
| **Kahakotuwa K.N**    | **IT23432284** | `it23432284@my.sliit.lk` | **MobileNetV2 (Edge-Optimized)**<br>• Inverted residuals & linear bottlenecks<br>• Depthwise separable convolutions & mobile latency                         |
| **Athukorala K.P**    | **IT23292604** | `it23292604@my.sliit.lk` | **EfficientNetB0 (Compound Scaling)**<br>• Compound coefficient ($\phi$) depth-width-resolution scaling<br>• Squeeze-and-Excitation & comparative evaluation |

---

## 2. Benchmark Summary & Key Empirical Findings

All models were evaluated on the **25,250 held-out test images** (completely unseen during model development):

| Architectural Metric        |    Custom CNN Baseline    |    MobileNetV2     |      ResNet-50       | EfficientNetB0 (Champion) |
| :-------------------------- | :-----------------------: | :----------------: | :------------------: | :-----------------------: |
| **Architecture Paradigm**   | 4-Stage ConvNet (Scratch) | Inverted Residuals | Residual Bottlenecks |   Compound Scaling + SE   |
| **Total Parameters**        |   **678,085** (~0.68M)    | 2,387,365 (~2.39M) | 23,802,853 (~23.8M)  |    4,184,072 (~4.18M)     |
| **Trainable Parameters**    |        **675,653**        |     1,737,445      |      9,130,085       |         3,261,825         |
| **Float32 Weights (MiB)**   |        **2.59 MB**        |      9.11 MB       |       90.80 MB       |         15.96 MB          |
| **Inference Latency (T4)**  |        **5.90 ms**        |      6.84 ms       |       10.98 ms       |          7.92 ms          |
| **Throughput (Images/sec)** |      **~169.5 FPS**       |     ~146.2 FPS     |      ~91.1 FPS       |        ~126.3 FPS         |
| **Validation Top-1 Acc**    |          57.23%           |       64.73%       |        69.47%        |        **72.69%**         |
| **Validation Top-5 Acc**    |          80.90%           |       86.64%       |        88.65%        |        **90.68%**         |
| **Test Top-1 Accuracy**     |          61.69%           |       68.69%       |        73.54%        |        **77.26%**         |
| **Test Top-5 Accuracy**     |          86.19%           |       90.15%       |        92.02%        |        **94.16%**         |
| **Test Categorical Loss**   |          1.4441           |       1.1692       |        1.1110        |        **0.8091**         |
| **Macro F1-Score**          |          0.6169           |       0.6862       |        0.7355        |        **0.7718**         |
| **Accuracy per Megabyte**   |      **23.85% / MB**      |     7.54% / MB     |      0.81% / MB      |        4.84% / MB         |

### Primary Findings:

1. **EfficientNetB0 establishes the Pareto-optimal frontier:** It surpasses ResNet-50 by **$+3.72\%$ Top-1 accuracy** while requiring **$82.4\%$ fewer parameters** and an **$82.4\%$ smaller disk footprint**.
2. **Custom CNN delivers maximum parameter density:** Delivering **$23.85\%$ accuracy per MB**, its $2.59\text{ MB}$ footprint and $5.9\text{ ms}$ latency make it ideal for memory-constrained microcontroller and IoT kitchen appliance deployments.
3. **The Food-101 Noise Inversion:** In all 4 models, test accuracy consistently exceeded validation accuracy by $+4.0$ to $+4.5$ percentage points. This confirms the models learned generalizable visual representations and thrived on the human-cleaned test ground truth compared to the $\sim 20\%$ web-scraped noise in training/validation splits.

---

## 3. Dataset: Food-101 Benchmark

- **Curated By:** Lukas Bossard, Matthieu Guillaumin, and Luc Van Gool (ETH Zürich, ECCV 2014)
- **Official Dataset Portal:** [ETH Zürich Food-101](https://data.vision.ee.ethz.ch/cvl/datasets_extra/food-101/)
- **Direct Tarball Download:** [`http://data.vision.ee.ethz.ch/cvl/food-101.tar.gz`](http://data.vision.ee.ethz.ch/cvl/food-101.tar.gz) (~4.65 GB)
- **Detailed Access & Mirrors Guide:** [`docs/dataset_access_instructions.md`](docs/dataset_access_instructions.md)

### Zero-Leakage Split Protocol

To ensure strict scientific integrity, the original 75,750 training images were partitioned into training and validation sets using deterministic seed `42`, while the canonical 25,250 test images were quarantined until final evaluation:

```text
Food-101 (101,000 images, 101 classes)
├── Training Partition:    68,175 images (675 images/class — 90% of training pool) [Augmented]
├── Validation Partition:   7,575 images (75 images/class  — 10% of training pool) [Deterministic]
└── Held-Out Test Split:   25,250 images (250 images/class — 100% Unseen Ground Truth)
```

---

## 5. Setup & Execution Guide

### 5.1 Local Installation

```bash
# 1. Clone the repository
git clone https://github.com/niRmana11/food-classification-deep-learning.git
cd food-classification-deep-learning

# 2. Create and activate a virtual environment
python -m venv venv

# Windows PowerShell:
.\venv\Scripts\Activate.ps1
# macOS / Linux:
# source venv/bin/activate

# 3. Install dependencies
pip install --upgrade pip
pip install -r requirements.txt
```

### 5.2 Google Colab Setup (Cloud T4 GPU)

For training on Google Colab with free T4 GPU acceleration:

```python
# Run this cell at the start of any Colab notebook:
import os, sys

REPO_NAME = "food-classification-deep-learning"
REPO_URL = "https://github.com/niRmana11/food-classification-deep-learning.git"

if "google.colab" in sys.modules:
    if not os.path.exists(f"/content/{REPO_NAME}"):
        !git clone {REPO_URL}
        %cd /content/{REPO_NAME}
    else:
        %cd /content/{REPO_NAME}
        !git pull origin main

    if f"/content/{REPO_NAME}" not in sys.path:
        sys.path.insert(0, f"/content/{REPO_NAME}")
```

### 5.3 Automated Dataset Acquisition & Splitting

```bash
# Step 1: Download and extract the Food-101 dataset (~4.65 GB archive)
python -m src.data.download_food101

# Step 2: Generate the reproducible, zero-leakage manifests in data/splits/
python -m src.data.split_food101
```

### 5.4 Using the Shared Preprocessing Pipeline

To load train, validation, and test datasets with architecture-specific normalization and training-only data augmentation:

```python
from src.preprocessing.data_loader import get_food101_datasets

train_ds, val_ds, test_ds = get_food101_datasets(
    data_dir="data/raw/food-101",
    splits_dir="data/splits",
    model_type="custom_cnn",   # Options: 'custom_cnn', 'resnet50', 'mobilenetv2', 'efficientnetb0'
    image_size=(224, 224),
    batch_size=32
)
```

### 5.5 Reproducing Model Training & Evaluation

All experimental workflows are self-contained inside the `notebooks/` directory:

- **Baseline Custom CNN:** Run `notebooks/03_custom_cnn.ipynb`
- **ResNet-50:** Run `notebooks/04_resnet50.ipynb`
- **MobileNetV2:** Run `notebooks/05_mobilenetv2.ipynb`
- **EfficientNetB0:** Run `notebooks/06_efficientnetb0.ipynb`
- **Comparative Analysis:** Run `notebooks/07_comparative_eval.ipynb`

All results, training curves, confusion matrices, and metrics are automatically saved into `results/<model_name>/` and `results/comparison/`.

---
