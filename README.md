# SE4050 Deep Learning 2026 — Food Classification

![Project Status](<https://img.shields.io/badge/status-dataset%20confirmed%20(Food--101)-green>)
![Task](<https://img.shields.io/badge/task-multiclass%20image%20classification%20(101%20classes)-blue>)
![Models](https://img.shields.io/badge/models-Custom%20CNN%20%7C%20ResNet50%20%7C%20MobileNetV2%20%7C%20EfficientNetB0-orange)
![Dataset](<https://img.shields.io/badge/dataset-Food--101%20(101k%20images)-purple>)

---

## 1. Project Overview & Research Question

This project investigates multi-class visual food recognition using deep convolutional neural networks. We implement, evaluate, and critically compare four distinct architectures under strictly controlled, fair experimental conditions to satisfy the requirements of module **SE4050 – Deep Learning (2026)** at **SLIIT**.

### Central Research Question

> _"How do different CNN architectures—a baseline Custom CNN, ResNet50, MobileNetV2, and EfficientNetB0—compare for multi-class food image classification in terms of predictive performance, generalization, computational efficiency, model complexity, training stability, and classification error patterns?"_

### Hypotheses Under Investigation

1. **Transfer Learning vs. Scratch Baseline:** Pretrained backbones (ResNet50, MobileNetV2, EfficientNetB0) will generalize significantly better on high-variance food textures than the Custom CNN trained from scratch.
2. **Efficiency vs. Performance:** MobileNetV2 will provide the lowest inference latency and smallest memory footprint, making it ideal for mobile/edge food-logging applications with minimal top-1 accuracy degradation.
3. **Compound Scaling Efficiency:** EfficientNetB0 will achieve competitive or superior accuracy compared to ResNet50 while demanding substantially fewer parameters and lower training FLOPs.
4. **Error Analysis & Confusion Clusters:** Visually proximate categories with fine-grained intra-class overlap (e.g., _steak_ vs. _filet mignon_, or _apple pie_ vs. _bread pudding_) will constitute the primary classification error modes.

---

## 2. Team Structure & Work Allocation

| Member       | Name          | Role             | Primary Technical Responsibility                                                             | Target Model / Workstream                         |
| :----------- | :------------ | :--------------- | :------------------------------------------------------------------------------------------- | :------------------------------------------------ |
| **Member 1** | **Nirmana**   | **Group Leader** | Repo setup, Food-101 pipeline, Leakage-free split, EDA, Custom CNN architecture, Integration | **Custom CNN** (Designed from scratch with GAP)   |
| **Member 2** | **Matheesha** | Team Member      | ResNet50 architecture, residual feature extraction, transfer learning & fine-tuning          | **ResNet50** (Deep Residual Learning)             |
| **Member 3** | **Kanushka**  | Team Member      | MobileNetV2 architecture, inverted residual blocks, latency & edge benchmarking              | **MobileNetV2** (Lightweight / Edge-optimized)    |
| **Member 4** | **Kaveesha**  | Team Member      | EfficientNetB0 architecture, compound scaling analysis, comparative results aggregation      | **EfficientNetB0** (Compound Coefficient Scaling) |

---

## 3. Dataset: Food-101

- **Dataset Name:** Food-101
- **Original Source:** ETH Zürich Computer Vision Lab
- **Authors / Creators:** Lukas Bossard, Matthieu Guillaumin, Luc Van Gool (ECCV 2014)
- **Official URL:** [https://data.vision.ee.ethz.ch/cvl/datasets_extra/food-101/](https://data.vision.ee.ethz.ch/cvl/datasets_extra/food-101/)
- **Total Images:** 101,000 real-world RGB images
- **Number of Classes:** 101 distinct food categories
- **Class Balance:** Exactly 1,000 images per class at the global level
- **Data Properties:** Extreme intra-class variance in presentation, background, lighting, and viewing angles. Training set intentionally retains ~20% natural label noise as documented by the original authors.

### Canonical Data Split Protocol (Zero Data Leakage)

To strictly satisfy assignment constraints and prevent data leakage:

```text
Food-101 (101,000 images)
│
├── Original Training Split (75,750 images — 750 / class)
│   ├── ~90%  → Training Set (~68,175 images — ~675 / class) [Augmentation Applied]
│   └── ~10%  → Validation Set (~7,575 images — ~75 / class) [Deterministic Preprocessing]
│
└── Original Test Split (25,250 images — 250 / class)
    └── FINAL UNTOUCHED TEST SET (Completely unseen during all model and hyperparameter development)
```

> **STRICT PROTOCOL:** All data exploration, feature extraction, preprocessing decisions, model architectures, and hyperparameter tuning must rely **exclusively** on the training and validation splits. The 25,250 test images remain unseen and are evaluated **only once** during the final comparative assessment.

_Note: Raw images are NOT committed to GitHub. See `data/README.md` for reproducible download and extraction instructions._

---

## 4. Controlled Experimental Protocol

The SLIIT marking rubric requires all models to be evaluated under fair and comparable experimental conditions:

- **Input Dimensions:** 224 × 224 pixels across all 4 architectures.
- **Random Seed:** Synchronized to `42` for all data sampling, validation splitting, and weight initializations.
- **Data Augmentation:** Applied **only** to the training split (random horizontal flip, slight rotation $\pm 10\%$, subtle zoom $\pm 10\%$). Validation and test splits are strictly deterministic.
- **Model-Specific Preprocessing:**
  - _Custom CNN:_ Direct scaling to $[0, 1]$ via `Rescaling(1./255)`.
  - _ResNet50:_ Caffe-style zero-centered BGR scaling via `tf.keras.applications.resnet50.preprocess_input`.
  - _MobileNetV2:_ Normalized to $[-1, 1]$ via `tf.keras.applications.mobilenet_v2.preprocess_input`.
  - _EfficientNetB0:_ Native pass-through via `tf.keras.applications.efficientnet.preprocess_input`.
- **Hardware Feasibility Benchmark:** Before launching full 25-epoch runs, a small benchmark is executed on Google Colab (T4 GPU) to verify GPU memory footprint, throughput, and lock the batch size (candidate: 32).

---

## 5. Evaluation Metrics & Critical Comparison

Evaluation goes well beyond simple accuracy to satisfy the 30% Critical Analysis and 10% Model Comparison rubrics:

### 1. Classification Performance

- **Top-1 Accuracy** & **Top-5 Accuracy**
- **Precision, Recall, and F1-Score** (Macro and Weighted averages)
- **High-Resolution Confusion Matrix** (identifying inter-class confusion patterns)
- **Classification Error Analysis** (top 5 most confused class pairs)

### 2. Computational & Complexity Metrics

- **Total Parameters vs. Trainable Parameters**
- **Model Storage Footprint (MB)** on disk
- **Training Time** (total seconds and seconds per epoch)
- **Inference Latency** (average milliseconds per image over 1,000 test samples)
- **Peak GPU Memory Allocation (VRAM MB)** during training

---

## 6. Standardized Experiment Artifacts

Every member must log and commit reproducible experiment outputs under their respective directory in `results/<model_name>/`:

```text
results/<model_name>/
├── config.yaml                   # Exact hyperparameters and run metadata
├── history.csv                   # Per-epoch loss, accuracy, val_loss, val_accuracy
├── model_summary.txt             # Architecture layer breakdown and parameter counts
├── training_curves.png           # Dual-plot training vs. validation loss & accuracy
├── confusion_matrix.png          # Normalized 101-class confusion matrix plot
├── classification_report.json    # Complete per-class precision, recall, and F1 metrics
└── metrics.json                  # Top-1 acc, Top-5 acc, latency, disk size, training time
```

---

## 7. Repository Structure

```text
food-classification-deep-learning/
│
├── README.md                     # Comprehensive project guide and reproduction manual
├── requirements.txt              # Pinned Python package dependencies
├── .gitignore                    # Prevents dataset, weight, and cache commits
│
├── configs/
│   └── config.yaml               # Shared global configuration (seed, split, hyperparams)
│
├── data/
│   ├── README.md                 # Dataset acquisition and extraction guide
│   ├── raw/                      # Downloaded Food-101 files (git-ignored)
│   ├── processed/                # Preprocessed images / cached tf.data (git-ignored)
│   └── splits/                   # Deterministic train/val/test file manifests (.txt / .json)
│
├── notebooks/
│   ├── 01_food101_eda.ipynb      # Complete EDA: class counts, dimensions, quality, noise
│   ├── 02_gpu_benchmark.ipynb    # GPU memory, throughput, and batch size feasibility check
│   ├── 03_custom_cnn.ipynb       # Custom CNN baseline implementation & training (Nirmana)
│   ├── 04_resnet50.ipynb         # ResNet50 transfer learning & fine-tuning (Matheesha)
│   ├── 05_mobilenetv2.ipynb      # MobileNetV2 transfer learning & fine-tuning (Kanushka)
│   ├── 06_efficientnetb0.ipynb   # EfficientNetB0 transfer learning & tuning (Kaveesha)
│   └── 07_comparative_eval.ipynb # Cross-model evaluation, radar charts & critical analysis
│
├── src/
│   ├── __init__.py
│   ├── data/
│   │   ├── __init__.py
│   │   ├── download_food101.py   # Script to download and verify Food-101 archive
│   │   └── split_food101.py      # Canonical 90/10 train-val split generator (seed locked)
│   ├── preprocessing/
│   │   ├── __init__.py
│   │   ├── augmentations.py      # Training-only augmentation pipeline
│   │   └── data_loader.py        # Model-specific preprocessing and tf.data generators
│   ├── models/
│   │   ├── __init__.py
│   │   ├── custom_cnn.py         # Baseline CNN (4 Conv blocks + GAP + Dropout + Softmax)
│   │   ├── resnet50.py           # ResNet50 transfer learning builder
│   │   ├── mobilenetv2.py        # MobileNetV2 transfer learning builder
│   │   └── efficientnetb0.py     # EfficientNetB0 transfer learning builder
│   ├── training/
│   │   ├── __init__.py
│   │   ├── trainer.py            # Standardized model compilation, fit loop, and callbacks
│   │   └── callbacks.py          # EarlyStopping, ReduceLROnPlateau, CSVLogger
│   ├── evaluation/
│   │   ├── __init__.py
│   │   ├── metrics.py            # Classification report, top-k accuracy calculation
│   │   ├── benchmark_latency.py  # Inference speed and throughput profiler
│   │   └── visualize.py          # Confusion matrix and training curve plotting utilities
│   └── utils/
│       ├── __init__.py
│       ├── config_parser.py      # YAML configuration loader
│       └── seed.py               # Deterministic seed locker (numpy, python, tf)
│
├── results/
│   ├── custom_cnn/               # Member 1 evaluation artifacts
│   ├── resnet50/                 # Member 2 evaluation artifacts
│   ├── mobilenetv2/              # Member 3 evaluation artifacts
│   └── efficientnetb0/           # Member 4 evaluation artifacts
│
├── report/
│   └── README.md                 # Academic report drafting guidelines (SLIIT 10 sections)
└── presentation/
    └── README.md                 # 10-minute YouTube presentation script and slides
```

---

## 8. Complete Setup & Execution Guide (Local & Google Colab)

This section provides reproducible, step-by-step instructions for all team members on both local development machines and Google Colab cloud GPU environments.

### 8.1 End-to-End Execution Flow
```text
1. Clone Repository & Setup Virtual Environment
                      │
2. Download & Extract Food-101 Dataset (~5GB)
   (via: python -m src.data.download_food101)
                      │
3. Read Canonical Split Manifests
   (data/splits/train.txt, val.txt, test.txt)
                      │
4. Load & Preprocess Batches via Shared Factory
   (from src.preprocessing.data_loader import get_food101_datasets)
                      │
5. Train Model & Log Metrics on Colab T4 GPU
   (Custom CNN | ResNet50 | MobileNetV2 | EfficientNetB0)
                      │
6. Save Standard Artifacts into results/<model_name>/
                      │
7. Submit Pull Request to main for Group Leader Review
```

---

### 8.2 Environment Setup

#### Option A: Local Development (Windows / macOS / Linux)
Recommended for code editing, EDA, and local debugging:
```powershell
# 1. Clone the repository
git clone https://github.com/niRmana11/food-classification-deep-learning.git
cd food-classification-deep-learning

# 2. Create and activate a virtual environment
python -m venv venv
.\venv\Scripts\Activate.ps1    # On Windows PowerShell
# source venv/bin/activate     # On macOS / Linux

# 3. Upgrade pip and install dependencies
pip install --upgrade pip
pip install -r requirements.txt
```

#### Option B: Google Colab Setup (Cloud T4 GPU)
Recommended for model training. Add this self-healing cell at the top of every Colab notebook:
```python
# Cell 1: Environment Setup & Automatic Git Sync
import os, sys

REPO_NAME = "food-classification-deep-learning"
REPO_URL = "https://github.com/niRmana11/food-classification-deep-learning.git"

if 'google.colab' in sys.modules:
    print("[INFO] Running in Google Colab environment.")
    if not os.path.exists(f"/content/{REPO_NAME}"):
        !git clone {REPO_URL}
        %cd /content/{REPO_NAME}
    else:
        %cd /content/{REPO_NAME}
        !git pull origin main
    
    if f"/content/{REPO_NAME}" not in sys.path:
        sys.path.insert(0, f"/content/{REPO_NAME}")
    
    import tensorflow as tf
    gpus = tf.config.list_physical_devices('GPU')
    if gpus:
        print(f"[SUCCESS] GPU detected: {gpus[0].name}")
        !nvidia-smi --query-gpu=name,memory.total,memory.free --format=csv
    else:
        print("[WARNING] No GPU detected! Go to: Runtime -> Change runtime type -> T4 GPU")
```

---

### 8.3 Dataset Acquisition & Verification

The Food-101 archive is ~5GB containing 101,000 images. **Never commit images to GitHub.**

1. **Download & Extract Dataset:**
   Run this single command inside the project root:
   ```bash
   python -m src.data.download_food101
   ```
   - Automatically downloads `food-101.tar.gz` from ETH Zürich.
   - Extracts images into `data/raw/food-101/images/`.
   - Safely skips downloading if images are already present.

2. **Canonical Split Manifests:**
   The deterministic split manifests are stored in `data/splits/`:
   - `train.txt`: 68,175 image paths (exactly 675 images per class — 90% of training pool).
   - `val.txt`: 7,575 image paths (exactly 75 images per class — 10% of training pool).
   - `test.txt`: 25,250 image paths (unseen final evaluation set).
   - `classes.txt`: Alphabetical list of all 101 classes.
   - `split_summary.json`: Metadata summary and validation statistics.

   *(To regenerate these manifests with locked seed 42, run: `python -m src.data.split_food101`).*

---

### 8.4 Using the Shared Preprocessing Pipeline

All team members must load datasets using the central factory function in `src/preprocessing/data_loader.py` to maintain experimental integrity:

```python
from src.preprocessing.data_loader import get_food101_datasets

# Available model_type options: 'custom_cnn', 'resnet50', 'mobilenetv2', 'efficientnetb0'
train_ds, val_ds, test_ds = get_food101_datasets(
    data_dir="data/raw/food-101",
    splits_dir="data/splits",
    model_type="resnet50",     # Specify your assigned model architecture!
    image_size=(224, 224),     # Standardized input resolution
    batch_size=32              # Locked by GPU feasibility benchmark
)
```

#### Pipeline Capabilities:
- **Model-Specific Scaling:**
  - `custom_cnn`: Scales pixel values to $[0.0, 1.0]$ via `Rescaling(1./255)`.
  - `resnet50`: Zero-centered BGR mean subtraction via Caffe formula.
  - `mobilenetv2`: Normalizes pixel values into $[-1.0, 1.0]$.
  - `efficientnetb0`: Native pass-through (handled internally by EfficientNet).
- **Training-Only Augmentation:** Applied exclusively on `train_ds` (random horizontal flip, $\pm 5\%$ rotation, $\pm 10\%$ zoom). `val_ds` and `test_ds` remain strictly deterministic.
- **Hardware Optimization:** Uses multithreaded I/O parsing and `.prefetch(tf.data.AUTOTUNE)`.

---

### 8.5 GPU Feasibility Benchmark Results

The pipeline was validated on a Google Colab NVIDIA T4 GPU (16GB VRAM) running ResNet50 (the heaviest backbone):
- **Peak VRAM Consumed:** 4,217 MB / 15,360 MB (27.5% capacity)
- **Data Throughput:** 77.7 images/second
- **Estimated Full Epoch:** ~18.7 minutes
- **Out-of-Memory (OOM) Status:** Verified 100% safe. `batch_size: 32` is officially locked.

---

### 8.6 Team Member Training Workflow

Each member independently develops and trains their assigned architecture:

```powershell
# 1. Ensure you are on latest main
git checkout main
git pull origin main

# 2. Switch to your feature branch
git checkout -b feature/<member>-<model>
# Example: git checkout -b feature/matheesha-resnet50

# 3. Develop model architecture in src/models/<model>.py
# 4. Train in Colab using notebooks/0<number>_<model>.ipynb

# 5. Export and save results into results/<model>/
#    (history.csv, model_summary.txt, training_curves.png, confusion_matrix.png, metrics.json)

# 6. Commit changes and push
git add src/models/ notebooks/ results/
git commit -m "feat: implement <model_name> training pipeline and log baseline artifacts"
git push -u origin feature/<member>-<model>

# 7. Open a Pull Request (PR) on GitHub for Leader Review
```

---

## 9. Git Workflow & Collaboration Rules

### Branch Naming Convention

- **Member 1 (Nirmana):** `feature/nirmana-custom-cnn`
- **Member 2 (Matheesha):** `feature/matheesha-resnet50`
- **Member 3 (Kanushka):** `feature/kanushka-mobilenetv2`
- **Member 4 (Kaveesha):** `feature/kaveesha-efficientnetb0`
- **Bug Fixes / Documentation:** `fix/<short-description>` or `docs/<short-description>`

### Pull Request & Review Protocol

1. Create and switch to your feature branch before writing code.
2. Commit with meaningful conventional messages (`feat:`, `fix:`, `docs:`, `exp:`, `chore:`).
3. Push branch to GitHub and open a Pull Request (PR) into `main`.
4. Group Leader (Nirmana) reviews the PR for code cleanliness, compliance with the shared configuration, and absence of data leakage before merging.
5. **Direct pushing to `main` is strictly prohibited.**

### Contribution Traceability (Assignment Mandate)

- Every member must make **regular, weekly commits and pushes** from their own GitHub account.
- Last-minute or trivial bulk commits are penalized according to the marking scheme.

---

## 10. Academic Submission Checklist

- [ ] Unseen test set evaluated only once after all hyperparameter decisions are finalized.
- [ ] No raw image datasets or bulky weights (`.h5`, `.keras`, `.pt`) committed to GitHub.
- [ ] Complete `requirements.txt` tested across local machines and Google Colab.
- [ ] Complete evaluation outputs committed to `results/` for all 4 architectures.
- [ ] Turnitin similarity report generated (file named with Group Leader registration number).
- [ ] Source code submitted to GradeScope.
- [ ] 10-minute presentation video recorded, uploaded to YouTube, and link tested.
- [ ] CourseWeb submission archive (`<leader_reg_num>.zip`) prepared containing `Members.txt`, `Report.pdf`, `Turnitin_report.pdf`, and `Submission.txt`.

