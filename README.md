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

## 8. Git Workflow & Collaboration Rules

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

## 9. Academic Submission Checklist

- [ ] Unseen test set evaluated only once after all hyperparameter decisions are finalized.
- [ ] No raw image datasets or bulky weights (`.h5`, `.keras`, `.pt`) committed to GitHub.
- [ ] Complete `requirements.txt` tested across local machines and Google Colab.
- [ ] Complete evaluation outputs committed to `results/` for all 4 architectures.
- [ ] Turnitin similarity report generated (file named with Group Leader registration number).
- [ ] Source code submitted to GradeScope.
- [ ] 10-minute presentation video recorded, uploaded to YouTube, and link tested.
- [ ] CourseWeb submission archive (`<leader_reg_num>.zip`) prepared containing `Members.txt`, `Report.pdf`, `Turnitin_report.pdf`, and `Submission.txt`.
