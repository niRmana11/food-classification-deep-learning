# SE4050 Deep Learning 2026 — Food Classification

![Project Status](https://img.shields.io/badge/status-in%20progress-yellow)
![Task](https://img.shields.io/badge/task-multiclass%20image%20classification-blue)
![Models](https://img.shields.io/badge/models-4-green)

## Project

A supervised deep-learning project for **multi-class food image classification**, implementing and comparing four distinct CNN architectures.

### Selected models

1. **Custom CNN**
2. **ResNet50**
3. **MobileNetV2**
4. **EfficientNetB0**

### Team

| Member           | Name      | Primary responsibility                               |
| ---------------- | --------- | ---------------------------------------------------- |
| 1 / Group Leader | Nirmana   | Dataset, EDA, preprocessing, Custom CNN, integration |
| 2                | Matheesha | ResNet50                                             |
| 3                | Kanushka  | MobileNetV2                                          |
| 4                | Kaveesha  | EfficientNetB0, results aggregation                  |

---

## Assignment

**Module:** SE4050 – Deep Learning  
**Institution:** BSc (Hons) in Information Technology  
**Academic year:** 2026  
**Submission deadline:** 30 September 2026

The assignment requires a supervised-learning group to implement and compare at least four distinct deep-learning models/architectures and evaluate them under fair and comparable conditions.

---

## Objective

The project aims to investigate how different CNN architectures perform on multi-class food image classification, considering:

- Predictive performance
- Generalization
- Overfitting/underfitting
- Convergence/training behaviour
- Classification errors
- Computational efficiency
- Model complexity
- Practical limitations

The project will not rely on accuracy alone.

---

## Repository Structure

```text
food-classification-deep-learning/
│
├── README.md
├── requirements.txt
├── .gitignore
│
├── configs/
├── notebooks/
├── src/
│   ├── data/
│   ├── preprocessing/
│   ├── models/
│   ├── training/
│   └── evaluation/
│
├── results/
│   ├── custom_cnn/
│   ├── resnet50/
│   ├── mobilenetv2/
│   └── efficientnetb0/
│
├── report/
└── presentation/
```

The exact structure may be adjusted as development progresses.

---

## Dataset

**Dataset: TBD**

The selected dataset must be:

- Authentic and real-world
- Sufficiently complex
- Multi-class
- Suitable for deep learning
- Publicly accessible
- Properly licensed/citable
- Not taken from a module lecture, laboratory exercise or end-to-end tutorial

### Dataset documentation

Once selected, add:

- Dataset name
- Original source
- Creator
- License
- URL
- Dataset version
- Number of images
- Number of classes
- Class distribution
- Download/access instructions
- Known limitations

The dataset itself should generally **not** be committed to this repository. Store access/download instructions instead.

---

## Development and Training

### Local machines

The team has:

- GTX 1050 laptop
- MX130 laptop
- i5 8th-generation integrated-graphics laptop
- MacBook Air M1

These machines are mainly used for development, EDA, debugging and small experiments.

### Cloud

Full training can use free cloud notebook environments such as:

- Google Colab
- Kaggle Notebooks

Cloud hardware availability and usage limits can vary, so all final experiment configurations and outputs must be saved.

---

## Experimental Principles

All models should be evaluated using a fair and documented protocol.

Where practical, use the same:

- Dataset split
- Class mapping
- Image preprocessing
- Augmentation policy
- Test set
- Evaluation metrics
- Experimental procedure
- Random seed strategy

The test set must remain unseen until final evaluation.

---

## Evaluation

Primary classification metrics:

- Accuracy
- Precision
- Recall
- F1-score
- Confusion matrix
- ROC-AUC where appropriate for the multi-class setting

Additional comparison:

- Training curves
- Parameter count
- Training time
- Inference time where feasible
- Model size where feasible

---

## Model Strategy

### Custom CNN

A CNN designed and trained by the team as a baseline.

### ResNet50

A residual CNN using transfer learning initially, with fine-tuning if appropriate.

### MobileNetV2

A lightweight CNN using transfer learning, with particular attention to efficiency and model complexity.

### EfficientNetB0

An efficient modern CNN using transfer learning, with analysis of predictive performance and computational trade-offs.

Exact architecture and hyperparameter decisions will be documented in `configs/` and the final report.

---

## Git Workflow

### Branch naming

```text
feature/<member>-<task>
fix/<short-description>
docs/<short-description>
exp/<short-description>
```

Examples:

```text
feature/nirmana-custom-cnn
feature/matheesha-resnet50
feature/kanushka-mobilenetv2
feature/kaveesha-efficientnetb0
docs/dataset-eda
fix/preprocessing-leakage
```

### Workflow

```text
main
  ↓
create feature branch
  ↓
implement/test
  ↓
commit
  ↓
push branch
  ↓
Pull Request
  ↓
review
  ↓
merge into main
```

Avoid substantial direct commits to `main`.

### Commit messages

Use:

```text
<type>: <short description>
```

Examples:

```text
feat: implement ResNet50 training pipeline
exp: run MobileNetV2 baseline
fix: prevent test data leakage
docs: add dataset documentation
chore: update training dependencies
```

Avoid messages such as:

```text
update
changes
done
final
stuff
```

---

## GitHub Issues and Project Board

Use Issues to track work.

Suggested labels:

- `dataset`
- `preprocessing`
- `model`
- `experiment`
- `evaluation`
- `report`
- `documentation`
- `bug`
- `urgent`

Recommended Project columns:

```text
Backlog → To Do → In Progress → Review → Done
```

---

## Collaboration Rules

Because the group is working remotely:

### WhatsApp

Use for:

- Short updates
- Coordination
- Questions
- Urgent problems
- Meeting scheduling

### GitHub

Use as the source of truth for:

- Code
- Results
- Configurations
- Documentation
- Issues
- Pull Requests
- Decisions

Important decisions made in WhatsApp should be recorded in the repository.

---

## Experiment Records

Every meaningful experiment should record:

- Experiment ID
- Model
- Dataset/split version
- Image size
- Batch size
- Learning rate
- Optimizer
- Epochs
- Random seed
- Transfer-learning/fine-tuning status
- Validation metrics
- Test metrics after finalization
- Training time
- Notes

Example:

```text
resnet50_baseline_v01
mobilenetv2_baseline_v01
efficientnetb0_finetune_v02
customcnn_baseline_v01
```

Do not overwrite previous experiment results.

---

## Results

Final results will be stored under:

```text
results/
├── custom_cnn/
├── resnet50/
├── mobilenetv2/
└── efficientnetb0/
```

Each model should retain sufficient evidence to reproduce the reported result.

---

## Report

The final report follows the assignment structure:

1. Introduction and Problem Definition
2. Background and Related Work
3. Dataset Description and Exploratory Data Analysis
4. Data Preprocessing and Feature Engineering
5. Experimental Design
6. Model Architectures
7. Results and Model Comparison
8. Critical Analysis and Discussion
9. Conclusion
10. References

The critical-analysis section is especially important because it represents 30% of the supervised-learning rubric.

---

## Important Academic/Submission Rules

Before final submission, verify:

- Dataset source/license is acknowledged
- Pretrained models are acknowledged
- Libraries and external resources are acknowledged
- Relevant literature is cited
- AI-assisted content is acknowledged where required
- All four models are documented
- Test data was not used for tuning
- All members made meaningful GitHub contributions
- Each member pushed at least once per week
- The repository contains setup/dependency information
- Submission files follow the assignment instructions

---

## Team Completion Checklist

- [ ] Dataset selected
- [ ] Dataset documented
- [ ] EDA completed
- [ ] Preprocessing completed
- [ ] Custom CNN completed
- [ ] ResNet50 completed
- [ ] MobileNetV2 completed
- [ ] EfficientNetB0 completed
- [ ] Final experiments completed
- [ ] Evaluation completed
- [ ] Error analysis completed
- [ ] Computational comparison completed
- [ ] Critical analysis completed
- [ ] Report completed
- [ ] YouTube video completed
- [ ] GitHub finalized
- [ ] GradeScope code submitted
- [ ] CourseWeb ZIP finalized
- [ ] Final submission checked

---

## Project Principle

> **Work independently, integrate centrally, document everything, and compare fairly.**

Each member owns a primary workstream, but all members must understand the complete project for the viva.
