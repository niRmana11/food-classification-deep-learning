# Section 5: Experimental Design and Setup

**Author:** Kaveesha Athukorala (Member 4)  
**Academic Module:** SE4050 — Deep Learning (2026)  
**Institution:** Sri Lanka Institute of Information Technology (SLIIT)  

---

## 5.1 Design Philosophy: A Controlled Comparison

The central research question of this study is *comparative* — it asks how four architectures differ, not how high an accuracy any one of them can reach. That distinction dictates the entire experimental design. A benchmark in which each model is separately tuned to its own best configuration would answer a different question and would be **confounded**: any observed accuracy difference could then be attributed to the architecture, to the learning-rate schedule, to the input resolution, or to the augmentation policy, with no way to separate the causes.

This study therefore adopts a **single-factor controlled design**. Every experimental variable except the network backbone is held fixed across all four runs, so that the backbone is the only plausible explanation for a difference in outcome:

$$\text{Performance}_{i} = f(\underbrace{\text{Architecture}_{i}}_{\text{the independent variable}}, \; \underbrace{\mathcal{D}, R, B, S, A, \mathcal{O}}_{\text{held constant}})$$

where $\mathcal{D}$ is the dataset partition, $R$ the input resolution, $B$ the mini-batch size, $S$ the random seed, $A$ the augmentation policy and $\mathcal{O}$ the optimizer family.

The deliberate cost of this design is that no model is presented at its individually optimal configuration. The reported accuracies are therefore **lower bounds on each architecture's potential**, and are meaningful only relative to one another. Section 5.7 records where this control could not be perfectly maintained.

---

## 5.2 Hardware and Software Environment

All four models were trained and evaluated on the **same class of accelerator**, so that wall-clock timings and memory measurements remain comparable.

### Table 5.1: Execution Environment

| Component | Specification |
| :--- | :--- |
| **Accelerator** | NVIDIA Tesla T4 (Google Colab), 15,360 MiB usable GDDR6 |
| **Tensor cores** | 320 (Turing, FP16/FP32 mixed-precision capable) |
| **Host runtime** | Google Colaboratory, standard GPU runtime |
| **Deep-learning framework** | TensorFlow $\ge$ 2.15 with Keras $\ge$ 3.0 (pinned in `requirements.txt`) |
| **Numerical stack** | NumPy $<$ 2.0, pandas $\ge$ 2.0, scikit-learn $\ge$ 1.3 |
| **Numerical precision** | Full `float32` throughout (mixed precision deliberately not enabled) |
| **Version control** | GitHub, one feature branch per member, reviewed pull requests into `main` |

Mixed-precision training was **not** enabled. Although it would have reduced epoch times on the T4's tensor cores, FP16 accumulation introduces architecture-dependent numerical behaviour, and lightweight models with narrow bottleneck layers (MobileNetV2, EfficientNetB0) are more susceptible to underflow in FP16 than wide residual stacks. Enabling it would therefore have introduced a confound precisely along the axis under study.

The practical constraint of the environment is that Colab enforces session time limits and can reclaim a runtime without warning. This shaped the training schedules described in Section 5.5 and is the direct cause of the epoch-budget limitation acknowledged in Section 5.7.

### 5.2.1 Feasibility Verification

Before any member began training, the shared pipeline was profiled on the heaviest candidate model (ResNet-50) in `notebooks/02_gpu_benchmark.ipynb`. The full results appear in **Table 4.3**; the decisive figures are a peak occupancy of **4,217 MiB of 15,360 MiB (27.5%)** at batch size 32 and a sustained input throughput of **77.7 images/second** with zero out-of-memory events.

This established a $> 11$ GiB headroom margin and made $B = 32$ safe for every architecture without further per-model tuning. The measurement was confirmed in production: EfficientNetB0's full training run logged a peak occupancy of **3,288 MiB**, comfortably inside the profiled envelope.

---

## 5.3 Reproducibility and Seed Control

A single seed, $S = 42$, is declared in `configs/config.yaml` and applied at every point where randomness enters the experiment:

| Stochastic process | Control mechanism |
| :--- | :--- |
| Train/validation partition | Deterministic split written once to `data/splits/*.txt`; **manifests are version-controlled**, so every member consumes byte-identical partitions |
| Manifest pre-shuffle | `random.Random(42).shuffle(samples)` — declusters the class-sorted manifest |
| Mini-batch shuffling | `dataset.shuffle(buffer_size=2048, seed=42)` |
| Classification-head initialisation | `tf.keras.utils.set_random_seed(42)` before model construction |
| Augmentation sampling | Keras preprocessing layers, seeded by the global seed |

Committing the split manifests rather than the splitting *code* is the stronger guarantee: it removes any dependence on library version, platform or iteration order, and it is what allows four members working on four machines to train against provably identical data.

**Honest limitation.** Seeding does not make GPU training bit-wise deterministic. Several cuDNN kernels — notably the backward pass of convolution — use non-deterministic atomic accumulation whose floating-point summation order varies between runs. Achieving exact reproducibility would require `tf.config.experimental.enable_op_determinism()`, at a substantial throughput cost that the Colab session budget did not permit. Re-running any experiment in this report should therefore reproduce the reported metrics to approximately $\pm 0.3$ percentage points, not exactly. This is disclosed rather than glossed over, and it does not affect the comparative conclusions, whose margins (3.72 to 15.57 pp) are an order of magnitude larger than the run-to-run variance.

---

## 5.4 Controlled and Varied Factors

### Table 5.2: Factors Held Constant Across All Four Experiments

| Factor | Locked value | Rationale |
| :--- | :--- | :--- |
| Input resolution | $224 \times 224 \times 3$, bilinear | ImageNet-native resolution; equalises the information available to every backbone |
| Mini-batch size | 32 | Fixes gradient-noise scale; verified feasible in Section 5.2.1 |
| Random seed | 42 | See Section 5.3 |
| Train / validation / test split | 68,175 / 7,575 / 25,250 | Identical manifests for all members |
| Augmentation policy | Horizontal flip, $\pm 5\%$ rotation, $\pm 10\%$ zoom — **training split only** | Identical regularisation pressure; val/test left deterministic |
| Optimizer family | Adam ($\beta_1 = 0.9$, $\beta_2 = 0.999$) | Removes optimizer choice as a confound |
| Initial learning rate | $10^{-3}$ | Identical starting point for every run |
| Loss function | Sparse categorical cross-entropy | The shared loader emits integer labels |
| Numerical precision | `float32` | See Section 5.2 |

### Table 5.3: Factors Deliberately Varied

| Factor | Custom CNN | ResNet-50 | MobileNetV2 | EfficientNetB0 |
| :--- | :--- | :--- | :--- | :--- |
| **Backbone** (the independent variable) | 4-stage scratch CNN | Deep residual | Inverted residual | Compound-scaled MBConv |
| **Input normalization** | $[0, 1]$ rescale | Caffe BGR mean-subtraction | $[-1, 1]$ | Native pass-through |
| **Head dropout rate** | 0.4 | 0.3 | 0.2 | 0.3 |

Input normalization *must* vary: each pretrained backbone was trained under a specific input distribution, and feeding it any other would discard the ImageNet prior the experiment exists to evaluate. It is a dependent consequence of the architecture choice, not a free parameter.

The dropout rates differ because each member tuned regularisation to their own head width on the validation split. This is a genuine, if minor, deviation from perfect control and is recorded as such in Section 5.7.

---

## 5.5 Training Protocol

### 5.5.1 Two-Phase Transfer Learning

The three pretrained architectures follow a common two-phase schedule. Training a randomly initialised head jointly with a pretrained backbone would push large early gradients through weights that are already near a good solution, destroying the ImageNet representation in the first few steps — the failure mode known as **catastrophic forgetting**.

$$\textbf{Phase 1 (feature extraction): } \theta_{\text{backbone}} \text{ frozen}, \quad \eta = 10^{-3}$$
$$\textbf{Phase 2 (fine-tuning): } \theta_{\text{top blocks}} \text{ unfrozen}, \quad \eta = 10^{-5}$$

Two rules are enforced identically across all three transfer models:

1. **Learning-rate reduction of two orders of magnitude at the phase boundary**, bounding the update magnitude applied to pretrained weights.
2. **Backbone BatchNormalization layers remain frozen throughout Phase 2.** A frozen `BatchNormalization` layer runs in inference mode and retains the population statistics $(\mu_{\text{pop}}, \sigma^{2}_{\text{pop}})$ estimated over ImageNet's 1.28M images, instead of overwriting them with the far noisier statistics of a 32-image mini-batch.

The Custom CNN has no pretrained weights and therefore trains in a single phase at $\eta = 10^{-3}$, serving as the from-scratch control condition.

### 5.5.2 Callbacks

| Callback | Configuration | Purpose |
| :--- | :--- | :--- |
| `EarlyStopping` | `monitor='val_loss'`, `restore_best_weights=True`, `patience` 4–5 (5 declared in `configs/config.yaml`) | Terminates on validation plateau and restores the best-generalising weights, not the last |
| `ReduceLROnPlateau` | `monitor='val_loss'`, `factor=0.2`, `patience` 2–3 | Escapes plateaus without a full restart |
| `ModelCheckpoint` | Best `val_loss`, written to mounted Google Drive | Survives Colab session termination |

Every callback monitors **`val_loss`, never `val_accuracy`**, and never any test quantity. Loss is the continuous, better-calibrated signal: accuracy is a step function of the argmax and can remain flat across epochs during which the model's confidence is still improving materially.

Checkpointing to Drive rather than to the ephemeral `/content` filesystem is what made a 3.66-hour run survivable — a disconnection costs one epoch rather than the entire experiment.

---

## 5.6 Evaluation Metrics

All splits in this study are **exactly class-balanced** — 675 training, 75 validation and 250 test images for each of the 101 classes, verifiable from `data/splits/`. This is a consequence of Food-101's curated construction, and it has a useful analytical implication established in Section 5.6.3.

### 5.6.1 Predictive Accuracy

Let $N$ be the number of evaluation images, $y_i$ the true label of image $i$, and $\mathbf{p}_i \in \mathbb{R}^{101}$ the predicted softmax distribution.

**Top-1 accuracy** — the fraction of images whose single highest-scoring class is correct:

$$\text{Top-1} = \frac{1}{N} \sum_{i=1}^{N} \mathbb{1}\!\left[\arg\max_{k} \; p_{i,k} = y_i\right]$$

**Top-5 accuracy** — the fraction whose true class appears anywhere in the five highest-scoring predictions:

$$\text{Top-5} = \frac{1}{N} \sum_{i=1}^{N} \mathbb{1}\!\left[y_i \in \text{Top}_5(\mathbf{p}_i)\right]$$

Top-5 is reported alongside Top-1 because it is the metric that matches the deployment reality of this task. Several Food-101 category pairs are genuinely ambiguous at $224 \times 224$ — `steak` against `filet_mignon` is the clearest case — so a production interface would surface a short ranked candidate list for user confirmation rather than a single hard label. Top-5 measures the quality of that list; Top-1 alone would score a model as wrong for ranking the correct class second out of 101.

### 5.6.2 Per-Class Quality

For each class $k$, with $TP_k$, $FP_k$ and $FN_k$ the true positives, false positives and false negatives:

$$\text{Precision}_k = \frac{TP_k}{TP_k + FP_k}, \qquad \text{Recall}_k = \frac{TP_k}{TP_k + FN_k}$$

$$F1_k = 2 \cdot \frac{\text{Precision}_k \cdot \text{Recall}_k}{\text{Precision}_k + \text{Recall}_k}$$

Precision and recall answer different questions and both are needed. Recall asks *"of all the real `apple_pie` images, how many did the model find?"*; precision asks *"of everything the model called `apple_pie`, how much really was?"*. A model can score well on one while failing badly on the other, and the $F1$ harmonic mean — which is dominated by the smaller of the two — reports the weaker.

### 5.6.3 Aggregation and an Exact Identity

Macro-averaging takes the unweighted mean over classes; weighted-averaging weights each class by its support $n_k$:

$$\text{Macro-}F1 = \frac{1}{101} \sum_{k=1}^{101} F1_k, \qquad \text{Weighted-}F1 = \frac{\sum_{k} n_k F1_k}{\sum_{k} n_k}$$

Because every test class has identical support $n_k = 250$, these two collapse to the same quantity:

$$n_k = n \;\; \forall k \;\; \Longrightarrow \;\; \text{Weighted-}F1 = \frac{n \sum_k F1_k}{101 \, n} = \text{Macro-}F1$$

A second identity follows from the same balance. Since $\sum_k TP_k$ is the total number of correct predictions:

$$\text{Macro-Recall} = \frac{1}{101}\sum_{k=1}^{101} \frac{TP_k}{250} = \frac{\sum_k TP_k}{101 \times 250} = \frac{\sum_k TP_k}{N} = \text{Top-1 Accuracy}$$

This is not a coincidence in the results and should not be read as one: for EfficientNetB0 the macro-recall of 0.7726 equals its Top-1 accuracy of 77.26% exactly, and the same holds for every model in the benchmark. **Macro-precision carries no such identity** — its denominators are prediction counts, which are not balanced — so the gap between macro-precision and macro-recall is informative, indicating whether a model over- or under-predicts particular classes.

Macro-averaging is therefore the headline aggregate in this report: it gives each of the 101 classes equal weight, so a model cannot disguise poor performance on hard categories behind strong performance on easy ones.

### 5.6.4 Confusion Matrices

Each model's $101 \times 101$ confusion matrix is **row-normalised**:

$$\tilde{C}_{ij} = \frac{C_{ij}}{\sum_{j'} C_{ij'}}$$

so that row $i$ is the predicted-class distribution for true class $i$ and the diagonal entry $\tilde{C}_{ii}$ is exactly $\text{Recall}_i$. Row normalisation, rather than raw counts, is what makes error *rates* comparable between classes, and it is the basis of the confusion-cluster analysis in Section 8.

### 5.6.5 Computational Metrics

| Metric | Definition | Reported in |
| :--- | :--- | :--- |
| Total parameters | All weights, trainable and frozen | `metrics.json` |
| Trainable parameters | Weights receiving gradients in Phase 2 | `metrics.json` |
| Float32 weight footprint | $\text{total params} \times 4$ bytes | Recomputed uniformly in Section 7 |
| Training wall-clock | End-to-end seconds, both phases | `metrics.json` |
| Inference latency | Milliseconds per image | **Withheld — see below** |

Two composite ratios quantify the accuracy-versus-cost trade-off that motivates the whole benchmark:

$$\mathcal{E}_{\text{param}} = \frac{\text{Test Top-1 (\%)}}{\text{Total parameters (millions)}}, \qquad \mathcal{E}_{\text{mem}} = \frac{\text{Test Top-1 (\%)}}{\text{Weight footprint (MiB)}}$$

Both must be read alongside absolute accuracy, never instead of it. A trivially small network can post an excellent $\mathcal{E}_{\text{param}}$ while remaining unusable — the Custom CNN's ratio of 90.98 is the highest in the benchmark at an unusable 61.69% Top-1 — so an efficiency ratio is meaningful only above an absolute accuracy threshold.

**On weight footprint.** The `model_size_mb` field was recorded on inconsistent bases by the four components (EfficientNetB0 logged 41.47 MB for a `.keras` checkpoint that also serialises Adam moment tensors, against 15.96 MiB of actual weights; ResNet-50 logged 98.40 against 90.80 MiB). Section 7 therefore recomputes every footprint as $\text{total params} \times 4$ bytes — the basis Keras itself prints in `model.summary()`.

**On latency.** The four logged latency values were captured under differing protocols and are not mutually comparable: MobileNetV2, the smallest pretrained model in the study, reports 167.29 ms/image against ResNet-50's 10.98 ms/image, which is physically impossible and identifies the figures as an instrumentation artefact. Latency is consequently **excluded** from all comparative tables and figures. `notebooks/07_comparative_eval.ipynb` contains a re-profiling routine under a single documented protocol — fixed batch size, discarded warm-up iterations, mean over 1,000 test images, one GPU session — pending access to all four trained checkpoints.

---

## 5.7 Evaluation Protocol and Threats to Validity

### 5.7.1 The Test-Set Protocol

The 25,250-image test split was held in strict isolation. All architecture selection, hyperparameter choice, early-stopping decisions and diagnostic inspection were carried out **exclusively on the training and validation splits**. Each model's test set was evaluated **once**, after its development was complete, and no model was revised afterwards.

This discipline is what makes the reported test figures an unbiased estimate of generalisation. Repeatedly consulting a test set turns it into a second validation set: the experimenter becomes the optimisation algorithm, and the reported number silently becomes optimistic.

### 5.7.2 Acknowledged Threats

An honest experimental design states where control was imperfect.

1. **Unequal epoch budgets.** ResNet-50, MobileNetV2 and the Custom CNN each ran 20 epochs; EfficientNetB0 ran 18. More significantly, EfficientNetB0's validation loss was **still decreasing on its final epoch** and `EarlyStopping` never fired — it stopped on budget, not on convergence. Its reported 77.26% is therefore a lower bound, and the accuracy gap to ResNet-50 is if anything understated.

2. **Unequal phase-transition points.** Phase 2 began at epoch 7 for ResNet-50 but at epoch 9 for MobileNetV2 and EfficientNetB0. Each member selected the transition from their own Phase 1 validation plateau, which is methodologically defensible but means the fine-tuning budgets were not identical.

3. **Differing head dropout rates** (0.2 to 0.4) and minor divergence in callback patience values, as recorded in Table 5.3 and Section 5.5.2.

4. **Label-noise asymmetry between splits.** The Food-101 training partition retains roughly 20% web-crawl label noise by design, whereas the test partition was manually cleaned (Bossard et al., 2014). Because the validation split is carved from the noisy training partition, validation metrics are systematically pessimistic relative to test metrics. This is not a flaw in the protocol but a property of the dataset, and it is visible as a **consistent +3.96 to +4.57 pp test-over-validation gap across all four models** — a shared effect that could not arise from four independent per-model errors.

5. **Non-deterministic GPU kernels**, as discussed in Section 5.3.

None of these threatens the headline comparative conclusions, whose effect sizes substantially exceed the uncertainty each introduces. They are recorded so that a reader can calibrate the precision of the reported figures, and they define the highest-value directions for future work in Section 9.
