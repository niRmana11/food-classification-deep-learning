# SE4050 Deep Learning 2026 — Master Project Plan & Execution Guide

## Multi-Class Food Classification Benchmark (Food-101)

- **Module:** SE4050 – Deep Learning (2026)
- **Institution:** Sri Lanka Institute of Information Technology (SLIIT) — BSc (Hons) in Information Technology
- **Project Category:** Supervised Deep Learning (Multi-Class Visual Classification)
- **Dataset:** Food-101 (101,000 images, 101 classes, ETH Zürich Computer Vision Lab)
- **Target Deadline:** 30 September 2026
- **Current Status:** **Phase 0 (Foundation & Pipeline) Complete** — Transitioning to Phase 1 (Model Implementation & Experiments)

---

## 1. Executive Summary & Research Framework

### 1.1 Project Overview

This project benchmarks four distinct Convolutional Neural Network (CNN) architectures on the challenging **Food-101** visual recognition dataset. Visual food classification is notoriously complex due to:

1. High intra-class variance (dishes vary widely in presentation, lighting, and ingredients).
2. Low inter-class variance (visually similar items like _steak_ vs. _filet mignon_, or _apple pie_ vs. _bread pudding_).
3. Realistic label noise (the Food-101 training split intentionally contains ~20% natural web-crawled noise).

### 1.2 Central Research Question

> _"How do different CNN architectures—a baseline Custom CNN from scratch, ResNet50, MobileNetV2, and EfficientNetB0—compare for multi-class food image classification in terms of predictive accuracy, generalization, computational efficiency, model complexity, training stability, and fine-grained classification error patterns?"_

### 1.3 Core Hypotheses Under Investigation

1. **Transfer Learning Superiority:** Pretrained ImageNet representations (ResNet50, MobileNetV2, EfficientNetB0) will converge faster and achieve substantially higher top-1 accuracy than the Custom CNN trained from scratch on high-variance food textures.
2. **Edge Efficiency (MobileNetV2):** Utilizing depthwise separable convolutions and inverted residual bottlenecks, MobileNetV2 will deliver the lowest parameter footprint and fastest inference latency with minimal degradation in top-1 accuracy.
3. **Compound Scaling Efficiency (EfficientNetB0):** EfficientNetB0’s compound coefficient scaling (balancing network depth, width, and resolution) will achieve accuracy comparable or superior to ResNet50 while demanding substantially fewer parameters and lower training FLOPs.
4. **Classification Error Clustering:** Visually proximate categories with fine-grained ingredient overlap will dominate the classification error space across all four architectures.

---

## 2. SLIIT Assignment Rubric Mapping

To secure maximum marks (Grade A), our workflow directly satisfies each rubric component:

| Rubric Component                     | Weight  | Target Standard in This Plan                                                                                                                                                                            | Lead Responsible |
| :----------------------------------- | :-----: | :------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------ | :--------------- |
| **Selection of Dataset**             |   5%    | Authentic, complex, 101-class real-world benchmark (Food-101). Not from course lectures or generic tutorials. Full academic citations provided.                                                         | All Members      |
| **Description of Dataset**           |   5%    | Comprehensive EDA covering source, size, class balance verification (1,000/class), image dimensions, pixel ranges, intra-class variance, and intentional label noise.                                   | Nirmana          |
| **Data Preprocessing**               |   10%   | Fully automated pipeline: 90/10 train-val split, unseen test set isolation (zero data leakage), training-only data augmentation, and architecture-specific scaling.                                     | Nirmana          |
| **Appropriate Learning Algorithms**  |   10%   | Four fundamentally distinct architectures (Custom CNN baseline, Deep Residual ResNet50, Lightweight Inverted Residual MobileNetV2, Compound Scaled EfficientNetB0) with technical justifications.       | All Members      |
| **Implementation & Reproducibility** |   5%    | Clean modular code (`src/`), reproducible scripts, locked seed (`42`), GPU benchmark verification, and active, traceable GitHub branches/PRs from all 4 members.                                        | Nirmana          |
| **Results & Fair Comparison**        |   5%    | Controlled conditions: 224×224 resolution, batch size 32, identical seed, full train/val/test metrics, learning curves, and multi-class confusion matrices.                                             | All Members      |
| **Critical Analysis & Discussion**   | **30%** | Deep technical breakdown: overfitting/underfitting dynamics, convergence curves, confusion cluster analysis, parameter efficiency, inference latency, memory cost, and practical deployment trade-offs. | All Members      |
| **Professionalism of Report**        |   10%   | 10 standard SLIIT report sections, rigorous academic writing, high-resolution figures, formatted tables, and full IEEE/Harvard citations.                                                               | All Members      |
| **Viva Voce Demonstration**          | **20%** | Live demonstration of the complete system, mastery of individual model code, deep understanding of comparative trade-offs, and 10-minute presentation video.                                            | All Members      |

---

## 3. The Completed Foundation (What Has Been Built)

As Group Leader, Nirmana has established the core engineering foundation before onboarding the team. All team members inherit these pre-tested components:

```text
[COMPLETED INFRASTRUCTURE]
├── Repository Scaffold & .gitignore  --> Prevents heavy datasets, weights, and caches from entering Git
├── Project Configuration (config.yaml)--> Standardizes seed=42, 224x224 input, batch_size=32
├── Dataset Downloader (src/data/)     --> Automated download and verification of Food-101 (~5GB)
├── Zero-Leakage Splits (data/splits/) --> Canonical manifests: train (68,175), val (7,575), test (25,250)
├── Shared Preprocessing (src/prep/)  --> Multithreaded tf.data loader with model-specific normalization
└── GPU Benchmark (notebooks/02)      --> Colab T4 verified: 4,217 MB VRAM (27.5%), 77.7 img/s, 0 OOM errors
```

---

## 4. Controlled Experimental Protocol (Fair Comparison Rules)

To ensure the comparison across all 4 models is scientifically valid:

- **Input Resolution:** Uniform bilinear resizing to **224 × 224 × 3** across all 4 models.
- **Batch Size:** **32** (strictly locked following the Google Colab T4 feasibility benchmark).
- **Random Seed:** **42** for all sampling, data shuffling, and initializations.
- **Data Augmentation Policy:**
  - Applied **only** to the training split: random horizontal flip, slight rotation ($\pm 5\%$), and subtle zoom ($\pm 10\%$).
  - Validation and test splits remain **strictly deterministic** (no random transformations).
- **Model-Specific Preprocessing:**
  - _Custom CNN:_ Direct rescale to $[0.0, 1.0]$ via `Rescaling(1./255)`.
  - _ResNet50:_ Caffe-style zero-centered BGR mean subtraction via `tf.keras.applications.resnet50.preprocess_input`.
  - _MobileNetV2:_ Normalized to $[-1.0, 1.0]$ via `tf.keras.applications.mobilenet_v2.preprocess_input`.
  - _EfficientNetB0:_ Native pass-through via `tf.keras.applications.efficientnet.preprocess_input`.
- **Golden Rule on Test Data:**
  - All model development, architectural iterations, and early-stopping adjustments are conducted **strictly on `train_ds` and `val_ds`**.
  - The 25,250 test images in `test_ds` remain completely untouched and are evaluated **only once** for the final benchmark report.

---

## 5. Team Work Allocation & Detailed Member Work Packages

```
┌──────────────────────────────────────────────────────────────────────────────────┐
│                             TEAM RESPONSIBILITY MATRIX                           │
├───────────────────┬──────────────┬──────────────────┬────────────────────────────┤
│ Member            │ Role         │ Model Ownership  │ Technical Focus            │
├───────────────────┼──────────────┼──────────────────┼────────────────────────────┤
│ Member 1: Nirmana │ Group Leader │ Custom CNN       │ Baseline from scratch,     │
│                   │              │ (Scratch + GAP)  │ EDA, Pipeline, Integration │
├───────────────────┼──────────────┼──────────────────┼────────────────────────────┤
│ Member 2: Matheesha│ Team Member  │ ResNet50         │ Residual learning, deep    │
│                   │              │ (Transfer + Fine)│ feature extraction         │
├───────────────────┼──────────────┼──────────────────┼────────────────────────────┤
│ Member 3: Kanushka│ Team Member  │ MobileNetV2      │ Edge efficiency, inverted  │
│                   │              │ (Transfer + Fine)│ residuals, latency profiling│
├───────────────────┼──────────────┼──────────────────┼────────────────────────────┤
│ Member 4: Kaveesha│ Team Member  │ EfficientNetB0   │ Compound scaling, model    │
│                   │              │ (Transfer + Fine)│ comparison aggregator      │
└───────────────────┴──────────────┴──────────────────┴────────────────────────────┘
```

---

### Member 1 (Group Leader): Nirmana

- **Assigned Model:** **Custom CNN** (Trained from Scratch)
- **Key Code Files:** `src/models/custom_cnn.py`, `notebooks/01_food101_eda.ipynb`, `notebooks/03_custom_cnn.ipynb`
- **Results Folder:** `results/custom_cnn/`

#### Specific Responsibilities & Step-by-Step Execution:

1. **Complete the EDA Notebook (`notebooks/01_food101_eda.ipynb`):**
   - Verify class balance: confirm exactly 1,000 images per class globally.
   - Plot class distribution histograms and inspect dimension/aspect-ratio variations.
   - Generate sample visual grids illustrating intra-class variance and intentional web noise.
   - Document data quality findings for the report (5% Dataset Description mark).
2. **Design the Custom CNN Architecture (`src/models/custom_cnn.py`):**
   - Implement a modular 4-stage convolutional backbone:
     `Input(224,224,3) -> [Conv2D(32) -> BatchNorm -> ReLU -> MaxPool] -> [Conv2D(64) -> BatchNorm -> ReLU -> MaxPool] -> [Conv2D(128) -> BatchNorm -> ReLU -> MaxPool] -> [Conv2D(256) -> BatchNorm -> ReLU -> MaxPool] -> GlobalAveragePooling2D -> Dropout(0.4) -> Dense(101, softmax)`
   - _Technical Justification:_ Use `GlobalAveragePooling2D` instead of `Flatten` to drastically reduce parameter count, prevent dense layer overfitting, and improve spatial translation invariance.
3. **Train & Evaluate Baseline:**
   - Train on Colab T4 GPU with Adam optimizer ($lr=0.001$), EarlyStopping (patience=5), and ReduceLROnPlateau.
   - Evaluate on validation data; once converged, evaluate on the final unseen test set.
   - Save all 7 standardized artifacts into `results/custom_cnn/`.
4. **Leadership & Integration:**
   - Review incoming Pull Requests from members, ensure no merge conflicts, and verify clean branch contributions.

---

### Member 2: Matheesha

- **Assigned Model:** **ResNet50** (Deep Residual Network)
- **Key Code Files:** `src/models/resnet50.py`, `notebooks/04_resnet50.ipynb`
- **Results Folder:** `results/resnet50/`

#### Specific Responsibilities & Step-by-Step Execution:

1. **Branch Setup:** Create and work strictly on `feature/matheesha-resnet50`.
2. **Model Architecture Implementation (`src/models/resnet50.py`):**
   - Load `tf.keras.applications.ResNet50(weights='imagenet', include_top=False, input_shape=(224, 224, 3))`.
   - Add classification head: `GlobalAveragePooling2D() -> BatchNormalization() -> Dropout(0.3) -> Dense(101, activation='softmax')`.
3. **Two-Stage Training Strategy:**
   - **Phase 1 (Feature Extraction):** Freeze all base ResNet50 layers (`base_model.trainable = False`). Train classification head with Adam ($lr=0.001$) for ~5–8 epochs until convergence.
   - **Phase 2 (Fine-Tuning):** Unfreeze top residual blocks (e.g., layers from `conv5_block1_out` onward). Train end-to-end with a reduced learning rate ($lr=10^{-5}$) to prevent catastrophic forgetting.
4. **Logging & Critical Analysis:**
   - Log learning curves showing Phase 1 vs. Phase 2 transitions.
   - Measure training time per epoch, total parameters, and inference latency.
   - Export all 7 standardized artifacts into `results/resnet50/` and open a PR.

---

### Member 3: Kanushka

- **Assigned Model:** **MobileNetV2** (Lightweight Inverted Residual CNN)
- **Key Code Files:** `src/models/mobilenetv2.py`, `notebooks/05_mobilenetv2.ipynb`
- **Results Folder:** `results/mobilenetv2/`

#### Specific Responsibilities & Step-by-Step Execution:

1. **Branch Setup:** Create and work strictly on `feature/kanushka-mobilenetv2`.
2. **Model Architecture Implementation (`src/models/mobilenetv2.py`):**
   - Load `tf.keras.applications.MobileNetV2(weights='imagenet', include_top=False, input_shape=(224, 224, 3))`.
   - Add classification head: `GlobalAveragePooling2D() -> Dropout(0.2) -> Dense(101, activation='softmax')`.
3. **Training & Edge Latency Profiling:**
   - **Phase 1:** Train head with frozen backbone ($lr=0.001$) using EarlyStopping.
   - **Phase 2:** Unfreeze top inverted residual blocks (e.g., from layer index 120 onward) with reduced $lr=10^{-5}$.
   - **Latency Profiling:** Profile inference latency (milliseconds per image) across 1,000 test images to prove MobileNetV2’s suitability for mobile food-logging applications.
4. **Logging & Critical Analysis:**
   - Compare parameter count (~2.3M–3.5M params) against ResNet50 (~25M params).
   - Export all 7 standardized artifacts into `results/mobilenetv2/` and open a PR.

---

### Member 4: Kaveesha

- **Assigned Model:** **EfficientNetB0** & **Cross-Model Results Aggregator**
- **Key Code Files:** `src/models/efficientnetb0.py`, `notebooks/06_efficientnetb0.ipynb`, `notebooks/07_comparative_eval.ipynb`
- **Results Folder:** `results/efficientnetb0/`

#### Specific Responsibilities & Step-by-Step Execution:

1. **Branch Setup:** Create and work strictly on `feature/kaveesha-efficientnetb0`.
2. **Model Architecture Implementation (`src/models/efficientnetb0.py`):**
   - Load `tf.keras.applications.EfficientNetB0(weights='imagenet', include_top=False, input_shape=(224, 224, 3))`.
   - Add classification head: `GlobalAveragePooling2D() -> BatchNormalization() -> Dropout(0.3) -> Dense(101, activation='softmax')`.
3. **Training & Fine-Tuning:**
   - **Phase 1:** Feature extraction with frozen base ($lr=0.001$).
   - **Phase 2:** Unfreeze top layers with fine-tuning learning rate ($lr=10^{-5}$).
4. **Cross-Model Aggregator Notebook (`notebooks/07_comparative_eval.ipynb`):**
   - Collect the `metrics.json` and `history.csv` files from all 4 models.
   - Generate comparative bar charts: Top-1 Accuracy, Top-5 Accuracy, Parameter Counts, Inference Latency, and Model Disk Size.
   - Generate comparative learning curve overlays (Train/Val Loss for all 4 models on a single dual plot).
   - Generate the master 4-model comparison table for the report (Section 7).

---

## 6. Practical Developer Workflow (How to Work with Colab & Git)

Every team member must follow this exact sequence to ensure zero lost work and 100% reproducible results:

```powershell
# ==============================================================================
# 1. LOCAL SETUP (On your laptop in VS Code / Terminal)
# ==============================================================================
git checkout main
git pull origin main
git checkout -b feature/<your-name>-<your-model>
# Example: git checkout -b feature/matheesha-resnet50
```

```python
# ==============================================================================
# 2. GOOGLE COLAB SETUP (At the top of your Colab Notebook)
# ==============================================================================
# Cell 1: Connect to GitHub and pull your branch
import os, sys

REPO_NAME = "food-classification-deep-learning"
BRANCH_NAME = "feature/<your-name>-<your-model>"  # <-- Your branch name

if not os.path.exists(f"/content/{REPO_NAME}"):
    !git clone -b {BRANCH_NAME} https://github.com/niRmana11/food-classification-deep-learning.git
    %cd /content/{REPO_NAME}
else:
    %cd /content/{REPO_NAME}
    !git pull origin {BRANCH_NAME}

if f"/content/{REPO_NAME}" not in sys.path:
    sys.path.insert(0, f"/content/{REPO_NAME}")

# Cell 2: Download Food-101 if not already present (~2 min in Colab)
!python -m src.data.download_food101

# Cell 3: Load the shared data pipeline (Zero Boilerplate!)
from src.preprocessing.data_loader import get_food101_datasets

train_ds, val_ds, test_ds = get_food101_datasets(
    data_dir="data/raw/food-101",
    splits_dir="data/splits",
    model_type="<your_model>",  # 'custom_cnn', 'resnet50', 'mobilenetv2', or 'efficientnetb0'
    image_size=(224, 224),
    batch_size=32
)
```

```powershell
# ==============================================================================
# 3. SAVING RESULTS & OPENING PULL REQUESTS
# ==============================================================================
# Save your completed notebook into notebooks/ and artifacts into results/<model>/
git add src/models/ notebooks/ results/
git commit -m "feat: implement <model> training, fine-tuning, and evaluation artifacts"
git push origin feature/<your-name>-<your-model>

# Go to GitHub repository -> Open Pull Request into 'main'
# Tag Nirmana (@niRmana11) for review
```

---

## 7. Standardized Output Artifacts Specification

Every model's directory in `results/<model_name>/` must contain these 7 standardized files to allow direct comparison:

```text
results/<model_name>/
├── config.yaml                   # Snapshot of image size, batch size, epochs, optimizers
├── history.csv                   # Epoch-by-epoch loss, accuracy, val_loss, val_accuracy
├── model_summary.txt             # Full layer breakdown, total parameters, trainable parameters
├── training_curves.png           # High-res dual plot: Loss Curve (left), Accuracy Curve (right)
├── confusion_matrix.png          # High-res 101-class normalized confusion matrix
├── classification_report.json    # Complete precision, recall, and F1-score per class
└── metrics.json                  # Final dictionary of scalar metrics (schema below)
```

### `metrics.json` Standard Schema:

```json
{
  "model_name": "ResNet50",
  "total_parameters": 25636710,
  "trainable_parameters": 23561957,
  "model_size_mb": 98.4,
  "training_time_seconds": 5420.5,
  "avg_epoch_time_seconds": 451.7,
  "inference_latency_ms_per_image": 14.2,
  "val_top1_accuracy": 0.742,
  "val_top5_accuracy": 0.918,
  "test_top1_accuracy": 0.738,
  "test_top5_accuracy": 0.915,
  "test_macro_precision": 0.741,
  "test_macro_recall": 0.738,
  "test_macro_f1": 0.736
}
```

---

## 8. Two-Week Master Execution Timeline (Sept 19 – Sept 30, 2026)

| Date              | Phase                        | Nirmana (Leader)                                            | Matheesha                                             | Kanushka                                                 | Kaveesha                                                       |
| :---------------- | :--------------------------- | :---------------------------------------------------------- | :---------------------------------------------------- | :------------------------------------------------------- | :------------------------------------------------------------- |
| **Sept 19 (Sat)** | **Onboarding & Setup**       | Push foundation, invite team, assign branches               | Clone repo, set up Colab, test data loader            | Clone repo, set up Colab, test data loader               | Clone repo, set up Colab, test data loader                     |
| **Sept 20 (Sun)** | **EDA & Baselines**          | Run EDA notebook (`01_eda`), generate visual plots          | Implement ResNet50 base (`src/models/resnet50.py`)    | Implement MobileNetV2 base (`src/models/mobilenetv2.py`) | Implement EfficientNetB0 base (`src/models/efficientnetb0.py`) |
| **Sept 21 (Mon)** | **Phase 1 Training**         | Train Custom CNN baseline (GAP architecture)                | Train ResNet50 Phase 1 (frozen feature extractor)     | Train MobileNetV2 Phase 1 (frozen feature extractor)     | Train EfficientNetB0 Phase 1 (frozen feature extractor)        |
| **Sept 22 (Tue)** | **Phase 2 Fine-Tuning**      | Custom CNN tuning (Dropout, learning rate)                  | Fine-tune ResNet50 top residual blocks ($lr=10^{-5}$) | Fine-tune MobileNetV2 top blocks ($lr=10^{-5}$)          | Fine-tune EfficientNetB0 top blocks ($lr=10^{-5}$)             |
| **Sept 23 (Wed)** | **Evaluation & Latency**     | Generate Custom CNN confusion matrix & curves               | Generate ResNet50 confusion matrix & latency          | Generate MobileNetV2 confusion matrix & latency          | Generate EfficientNetB0 confusion matrix & latency             |
| **Sept 24 (Thu)** | **PR Review & Integration**  | Review & merge all 3 member PRs into `main`                 | Push artifacts to `results/resnet50/`, open PR        | Push artifacts to `results/mobilenetv2/`, open PR        | Push artifacts to `results/efficientnetb0/`, open PR           |
| **Sept 25 (Fri)** | **Comparative Aggregation**  | Verify test set evaluation across all 4 models              | Draft Section 6 (ResNet50 Architecture & Results)     | Draft Section 6 (MobileNetV2 Architecture & Results)     | Run `07_comparative_eval.ipynb`, compile master tables         |
| **Sept 26 (Sat)** | **Critical Analysis Sprint** | Lead Section 8 writing (Critical Analysis 30%)              | Analyze Overfitting vs. Underfitting dynamics         | Analyze Latency, Complexity vs. Edge Viability           | Analyze Compound Scaling vs. Residual trade-offs               |
| **Sept 27 (Sun)** | **Report Compilation**       | Assemble Sections 1–5, compile unified PDF                  | Review references, format architecture tables         | Document parameter & memory comparison                   | Format confusion cluster analysis figures                      |
| **Sept 28 (Mon)** | **Report Finalization**      | Run Turnitin similarity check, refine academic text         | Proofread Sections 6–7                                | Proofread Section 8                                      | Proofread Section 9–10                                         |
| **Sept 29 (Tue)** | **Video Presentation**       | Lead 10-minute presentation video recording                 | Record ResNet50 segment (2.5 mins)                    | Record MobileNetV2 segment (2.5 mins)                    | Record EfficientNetB0 segment (2.5 mins)                       |
| **Sept 30 (Wed)** | **Final Submission**         | Package `<leader_id>.zip`, upload to CourseWeb & GradeScope | Final Viva practice session                           | Final Viva practice session                              | Final Viva practice session                                    |

---

## 9. Final Report Structure (Targeting 100% on the 10 SLIIT Sections)

The final submission requires a single, professionally formatted academic report (`Report.pdf`):

| Section | Title                                        | Target Length | Core Content                                                                                                                                                        | Primary Contributor               |
| :-----: | :------------------------------------------- | :-----------: | :------------------------------------------------------------------------------------------------------------------------------------------------------------------ | :-------------------------------- |
|  **1**  | **Introduction & Problem Definition**        |   1.5 pages   | Food classification context, research question, hypotheses, real-world utility                                                                                      | Nirmana                           |
|  **2**  | **Background & Related Work**                |   2.0 pages   | CNN history, residual learning, depthwise convolutions, compound scaling, food benchmarks                                                                           | Kanushka & Matheesha              |
|  **3**  | **Dataset Description & EDA**                |   2.5 pages   | Food-101 source, 101 classes, balance verification, resolution histograms, label noise                                                                              | Nirmana                           |
|  **4**  | **Data Preprocessing & Feature Engineering** |   2.0 pages   | 90/10 split protocol, zero data leakage proof, train-only augmentations, model scalers                                                                              | Nirmana                           |
|  **5**  | **Experimental Design & Setup**              |   2.0 pages   | Hardware specs (Colab T4), feasibility benchmark results, seed 42, metrics definitions                                                                              | Kaveesha                          |
|  **6**  | **Model Architectures**                      |   3.5 pages   | In-depth breakdown of Custom CNN, ResNet50, MobileNetV2, EfficientNetB0 with layer diagrams                                                                         | All 4 Members (Individual models) |
|  **7**  | **Results & Model Comparison**               |   3.0 pages   | Master comparison table (Top-1, Top-5, F1, latency, params), learning curves, bar charts                                                                            | Kaveesha & All Members            |
|  **8**  | **Critical Analysis & Discussion (30%)**     | **5.0 pages** | **Heart of the Report:** Overfitting/underfitting, convergence stability, confusion clusters, error modes, computational cost vs. accuracy trade-off, failure cases | **All Members (Collaborative)**   |
|  **9**  | **Conclusion & Future Work**                 |   1.5 pages   | Summary of key findings, validation of hypotheses, limitations, edge deployment feasibility                                                                         | Nirmana                           |
| **10**  | **References**                               |   1.5 pages   | Full IEEE/Harvard academic citations (Food-101 paper, ResNet, MobileNet, EfficientNet)                                                                              | Matheesha                         |

---

## 10. Final Submission Checklist & Deliverables

Before deadline submission on **30 September 2026**, verify each item:

1. **CourseWeb Submission:** Single ZIP file named `<leader_registration_number>.zip` containing:
   - `Members.txt` (Student names, registration numbers, institutional email addresses).
   - `Report.pdf` (Final compiled academic report).
   - `Turnitin_report.pdf` (Similarity index report generated under leader registration number).
   - `Submission.txt` (GitHub project URL + YouTube 10-minute presentation link).
2. **GradeScope Submission:** Full source code directory submitted according to module instructions.
3. **GitHub Repository Compliance:**
   - Active, meaningful weekly commit history from **all 4 group members**.
   - No bulky dataset archives or model weights (`.h5`, `.pt`, `.keras`) committed.
   - Clean, reviewed Pull Requests merged into `main`.
4. **Viva Voce Examination Preparation:**
   - Every member must be prepared to explain their individual model architecture, code, and justify their hyperparameter choices during the live demonstration.
