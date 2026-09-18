# SE4050 Deep Learning 2026 — Master Project Plan

## Multi-Class Food Classification (Food-101 Benchmark)

- **Module:** SE4050 – Deep Learning (2026)
- **Institution:** SLIIT — BSc (Hons) in Information Technology
- **Project Category:** Supervised Deep Learning (Multi-Class Image Classification)
- **Submission Deadline:** 30 September 2026
- **Current Status:** Dataset Confirmed — Food-101 (Phase 1: Pipeline & EDA Active)

---

## 1. Project Objective & Research Formulation

### Central Research Question

> _"How do different CNN architectures—Custom CNN, ResNet50, MobileNetV2, and EfficientNetB0—compare for multi-class food image classification in terms of predictive performance, generalization, computational efficiency, model complexity, training stability, and classification error patterns?"_

### Hypotheses to Test

1. **Transfer Learning Superiority:** Pretrained ImageNet representations in ResNet50, MobileNetV2, and EfficientNetB0 will achieve substantially higher generalization and convergence speed compared to a Custom CNN trained from scratch.
2. **Edge Efficiency (MobileNetV2):** MobileNetV2's depthwise separable convolutions will yield the lowest parameter count and fastest inference latency with an acceptable trade-off in top-1 accuracy.
3. **Compound Scaling (EfficientNetB0):** EfficientNetB0 will reach parity or superiority in top-1 accuracy against ResNet50 with fewer parameters and lower training FLOPs.
4. **Fine-Grained Confusions:** Semantic and visual proximity in food classes (e.g., _steak_ vs. _filet mignon_, _apple pie_ vs. _bread pudding_) will constitute the primary bottleneck in classification error.

---

## 2. Team Work Allocation & Individual Ownership

| Member                | Name          | Primary Ownership                                                                   | Architecture / Workstream                            | Deliverables                                                                                                           |
| :-------------------- | :------------ | :---------------------------------------------------------------------------------- | :--------------------------------------------------- | :--------------------------------------------------------------------------------------------------------------------- |
| **Member 1 (Leader)** | **Nirmana**   | Repository scaffolding, Food-101 pipeline, Data leakage prevention, EDA, Custom CNN | **Custom CNN** (scratch baseline with GAP)           | `src/data/`, `src/preprocessing/`, `notebooks/01_food101_eda.ipynb`, `src/models/custom_cnn.py`, `results/custom_cnn/` |
| **Member 2**          | **Matheesha** | Residual learning study, feature extraction vs. fine-tuning, training dynamics      | **ResNet50** (deep residual networks)                | `src/models/resnet50.py`, `notebooks/04_resnet50.ipynb`, `results/resnet50/`                                           |
| **Member 3**          | **Kanushka**  | Inverted residual bottlenecks, edge performance & latency profiling                 | **MobileNetV2** (lightweight edge model)             | `src/models/mobilenetv2.py`, `notebooks/05_mobilenetv2.ipynb`, `results/mobilenetv2/`                                  |
| **Member 4**          | **Kaveesha**  | Compound scaling analysis, multi-model evaluation aggregation                       | **EfficientNetB0** (balanced depth/width/resolution) | `src/models/efficientnetb0.py`, `notebooks/06_efficientnetb0.ipynb`, `results/efficientnetb0/`, comparison scripts     |

---

## 3. Dataset Specification: Food-101

- **Source:** ETH Zürich Computer Vision Lab (Bossard et al., ECCV 2014)
- **URL:** [https://data.vision.ee.ethz.ch/cvl/datasets_extra/food-101/](https://data.vision.ee.ethz.ch/cvl/datasets_extra/food-101/)
- **Total Images:** 101,000 RGB images across 101 classes.
- **Global Balance:** Exactly 1,000 images per class.
- **Noise Property:** ~20% intentionally retained training label noise (real-world web crawl). Test set is verified and clean.

### Strict Data Partitioning Protocol (Preventing Data Leakage)

```text
Food-101 (101,000 images)
│
├── Original Training Split (75,750 images — 750 / class)
│   ├── ~90%  → Training Set (~68,175 images — ~675 / class) [Augmentation Applied]
│   └── ~10%  → Validation Set (~7,575 images — ~75 / class) [Deterministic Preprocessing]
│
└── Original Test Split (25,250 images — 250 / class)
    └── FINAL UNTOUCHED TEST SET (Completely unseen during all model development)
```

> **MANDATE:** All preprocessing transformations, architecture refinements, and hyperparameter selections are conducted **strictly on training and validation splits**. The test set is evaluated once at final benchmark.

---

## 4. Controlled Experimental Protocol

- **Target Image Size:** 224 × 224 pixels.
- **Batch Size:** 32 (standardized via initial GPU memory benchmark).
- **Random Seed:** Synchronized to `42`.
- **Augmentation (Training only):** Random horizontal flip, subtle rotation ($\pm 10\%$), subtle zoom ($\pm 10\%$).
- **Model Preprocessing:**
  - Custom CNN: Rescale to $[0, 1]$
  - ResNet50: Caffe BGR mean-centering (`tf.keras.applications.resnet50.preprocess_input`)
  - MobileNetV2: Scaled to $[-1, 1]$ (`tf.keras.applications.mobilenet_v2.preprocess_input`)
  - EfficientNetB0: Native pass-through (`tf.keras.applications.efficientnet.preprocess_input`)

---

## 5. Experiment Artifacts & Logging Protocol

Every model output must be preserved in `results/<model_name>/`:

- `config.yaml`
- `history.csv`
- `model_summary.txt`
- `training_curves.png`
- `confusion_matrix.png`
- `classification_report.json`
- `metrics.json` (Top-1, Top-5, latency ms/image, disk size MB, training time)

---

## 6. Two-Week Execution Timeline (Sept 17 – Sept 30, 2026)

| Phase       | Target Dates | Key Milestones                                                                                          | Responsible    |
| :---------- | :----------- | :------------------------------------------------------------------------------------------------------ | :------------- |
| **Phase 0** | Sept 17–18   | Scaffolding, `.gitignore`, dependencies, configuration, documentation update                            | Nirmana        |
| **Phase 1** | Sept 19–20   | Food-101 split generator, EDA notebook, Data-loader with model preprocessing, Colab T4 benchmark        | Nirmana        |
| **Phase 2** | Sept 21–24   | Independent model training (Custom CNN, ResNet50, MobileNetV2, EfficientNetB0 baselines + 1 tuning run) | All 4 Members  |
| **Phase 3** | Sept 25–26   | Final unseen test evaluation, metric aggregation, latency profiling, confusion matrix analysis          | All 4 Members  |
| **Phase 4** | Sept 27–28   | Draft 10-section report focusing heavily on the 30% Critical Analysis and Error Modes                   | All 4 Members  |
| **Phase 5** | Sept 29–30   | Record 10-minute presentation video, Turnitin check, package GradeScope & CourseWeb ZIP                 | Nirmana & Team |

---

## 7. Decision Log

| Date       | Decision                                               | Rationale                                                              | Status        |
| :--------- | :----------------------------------------------------- | :--------------------------------------------------------------------- | :------------ |
| 2026-09-18 | Selected **Food-101**                                  | High real-world authenticity, 101 classes, standard academic benchmark | **Locked**    |
| 2026-09-18 | 90/10 split on original 75,750 train; test kept unseen | Prevents data leakage; complies with SLIIT assignment guidelines       | **Locked**    |
| 2026-09-18 | Standardized image dimension 224×224                   | Universal compatibility across all 4 architectures                     | **Locked**    |
| 2026-09-18 | Architecture-specific preprocessing backbones          | Preserves pretrained weight integrity                                  | **Locked**    |
| 2026-09-18 | T4 GPU feasibility benchmark before full training      | Optimizes batch size and eliminates out-of-memory crashes on Colab     | **Scheduled** |
