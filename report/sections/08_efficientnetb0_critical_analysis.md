# Section 8: Critical Analysis & Discussion — EfficientNetB0 Component

**Author:** Kaveesha Athukorala (Member 4)  
**Assigned Architecture:** EfficientNetB0 (Compound-Scaled MBConv Network)  
**Academic Module:** SE4050 — Deep Learning (Rubric Weight: 30%)  

---

## 8.4 EfficientNetB0 In-Depth Critical Analysis: Compound Scaling vs. Residual Depth

### 8.4.1 Convergence Trajectory and Learning Dynamics

The complete epoch-level record is preserved in [`results/efficientnetb0/history_phase1.csv`](../../results/efficientnetb0/history_phase1.csv) and [`results/efficientnetb0/history_phase2.csv`](../../results/efficientnetb0/history_phase2.csv), plotted in [`results/efficientnetb0/training_curves.png`](../../results/efficientnetb0/training_curves.png).

```
Loss Progression:
  Phase 1 (Epochs 1–8):   Train Loss: 2.2273 -> 1.4784  |  Val Loss: 1.5335 -> 1.3555
  Phase 2 (Epochs 9–18):  Train Loss: 1.2992 -> 0.9484  |  Val Loss: 1.2530 -> 1.0809
  Unseen Test Loss:       0.8091

Accuracy Progression:
  Phase 1 (Epochs 1–8):   Train Acc:  47.06% -> 61.37%  |  Val Acc:  61.99% -> 66.02% (Val Top-5: 87.41%)
  Phase 2 (Epochs 9–18):  Train Acc:  65.49% -> 74.16%  |  Val Acc:  68.61% -> 72.69% (Val Top-5: 90.68%)
  Unseen Test Accuracy:   77.26% (Top-1)  |  94.16% (Top-5)  |  Macro F1: 0.7718
```

#### 1. Phase 1: Immediate Convergence from a Frozen Backbone

With the entire 4.05M-parameter backbone frozen, gradient flow was restricted to just **131,941 head parameters (3.2% of the network)**. The most striking observation is how much of the final performance was reached before a single backbone weight was updated: validation accuracy hit **61.99% after one epoch** and **66.02% by Epoch 8** — already exceeding the Custom CNN's *final* test accuracy of 61.69% (Section 8.1) by 4.33 percentage points.

This is direct empirical evidence for **Hypothesis 1 (Transfer Learning Superiority)**. A linear softmax classifier fitted on frozen ImageNet features outperforms an entire convolutional network trained from scratch for 30+ epochs on the same 68,175 training images. The ImageNet representation already encodes the texture, colour-opponency and material-appearance primitives that food discrimination requires; Food-101's 68K images are simply insufficient to learn them from random initialisation.

Neither callback fired during Phase 1. Validation loss declined near-monotonically ($1.5335 \rightarrow 1.3532$ at Epoch 7, with a negligible $+0.0023$ uptick at Epoch 8), so `ReduceLROnPlateau` (patience 2) never triggered and $\eta$ remained at $10^{-3}$ throughout. The frozen-feature ceiling was approached but not breached: the final two epochs yielded only $+0.12$ pp of validation accuracy in total, confirming that further head-only training was exhausted and that unfreezing was the correct next step.

#### 2. The Asymmetric Regularization Effect (Validation Outperforming Training)

Throughout the whole of Phase 1, **validation accuracy exceeded training accuracy at every single epoch** — most dramatically at Epoch 1 (61.99% validation vs. 47.06% training, a 14.93 pp inversion). This apparent paradox has two mechanical causes, both of which are active only during training:

1. **Inverted Dropout ($p = 0.30$).** During training, 30% of the 1,280 pooled channels are stochastically zeroed and survivors rescaled by $\tfrac{1}{1-p} \approx 1.43$. At evaluation time Dropout is disabled and the network predicts with 100% of its channel capacity. EfficientNetB0's rate of $0.30$ is higher than MobileNetV2's $0.20$, which is why our inversion (14.93 pp) is larger than the one Kanushka reports in Section 8.3.2.
2. **Stochastic Training Augmentation.** Training batches pass through `RandomFlip("horizontal")`, `RandomRotation(0.05)` and `RandomZoom(0.10)`; validation and test batches are strictly deterministic (Section 4). The training metric is therefore computed on a *harder*, perturbed distribution than the validation metric.

The inversion narrows monotonically as training proceeds (14.93 pp $\rightarrow$ 4.65 pp by Epoch 8) and reverses during Phase 2, which is the expected signature of a correctly regularised model transitioning from underfitting toward its capacity limit.

#### 3. Phase 2: Fine-Tuning Unlocks the Domain Gap

Unfreezing `block6`, `block7` and `top_conv` at $\eta = 10^{-5}$ raised the trainable parameter count from 131,941 to **3,261,825 (a $24.7\times$ increase)** and produced an immediate, large discontinuity in the curves:

| Transition | Train Loss | Val Loss | Train Acc | Val Acc |
| :--- | ---: | ---: | ---: | ---: |
| Epoch 8 (end of Phase 1) | 1.4784 | 1.3555 | 61.37% | 66.02% |
| Epoch 9 (first fine-tune epoch) | 1.2992 | 1.2530 | 65.49% | **68.61%** |
| **Single-epoch delta** | **−0.1792** | **−0.1025** | **+4.12 pp** | **+2.59 pp** |

A single fine-tuning epoch delivered more validation gain (+2.59 pp) than the final *four* epochs of Phase 1 combined (+1.19 pp). This quantifies the **domain gap** between ImageNet object categories and fine-grained food photography: the generic features of Stages 1–5 transfer perfectly, but the class-specific abstractions concentrated in Stage 6 (48.9% of all parameters) must be re-learned for the food domain.

Critically, **no catastrophic forgetting occurred**. Validation loss fell monotonically across all ten Phase 2 epochs ($1.2530 \rightarrow 1.0809$) with no oscillation or divergence. Two protocol decisions are responsible, and both can be verified from the artifacts rather than merely asserted:

- **The two-order-of-magnitude learning-rate reduction** ($10^{-3} \rightarrow 10^{-5}$) kept update magnitudes small relative to the pretrained weight scale.
- **The BatchNorm freeze** preserved the ImageNet population statistics. This is confirmed arithmetically: the logged `trainable_parameters` of $3{,}261{,}825$ equals the non-BatchNorm parameter sum of the unfrozen stages ($2{,}008{,}300 + 711{,}984 + 409{,}600$) plus the head ($131{,}941$) **exactly**, proving that all $51{,}712$ BatchNorm parameters inside `block6`/`block7`/`top_bn` were correctly excluded from the gradient path (Section 6.4.4).

#### 4. An Honest Limitation: The Model Was Budget-Limited, Not Converged

The most important caveat in this analysis is that **EfficientNetB0 had not finished learning when training stopped**. The evidence is unambiguous:

- Validation loss improved on **every one of the ten** Phase 2 epochs, including the last (Epoch 17 $\rightarrow$ 18: $1.0924 \rightarrow 1.0809$, a $-0.0115$ improvement).
- `EarlyStopping(patience=5)` never triggered in either phase; training terminated because the configured `epochs_max` of 10 was reached, not because performance plateaued.
- The train–validation generalization gap at termination was only **1.47 pp** (74.16% vs. 72.69%) with a loss gap of just $0.1325$ — the signature of a model still in its **underfitting** regime with substantial headroom, not one beginning to overfit.

The reported 77.26% therefore represents a **lower bound** on this architecture's achievable Food-101 performance under our protocol. The published EfficientNetB0 literature reaches ~83–88% on Food-101 with longer schedules and progressive unfreezing. This ceiling was imposed by the Google Colab T4 session budget (3.66 hours of GPU wall-clock already consumed), not by the architecture. We state this explicitly rather than presenting the result as converged, and it is the single clearest avenue for future work (Section 9).

---

### 8.4.2 Test-Set Generalization and the Food-101 Label-Noise Asymmetry

Evaluated **once** on the 25,250 held-out test images, in accordance with the Golden Rule of Section 4:

| Metric | Validation (7,575 imgs) | Test (25,250 imgs) | Delta |
| :--- | ---: | ---: | ---: |
| **Top-1 Accuracy** | 72.69% | **77.26%** | **+4.57 pp** |
| **Top-5 Accuracy** | 90.68% | **94.16%** | **+3.48 pp** |
| **Cross-Entropy Loss** | 1.0809 | **0.8091** | **−0.2718** |
| **Macro Precision** | — | **0.7742** | — |
| **Macro Recall** | — | **0.7726** | — |
| **Macro F1-Score** | — | **0.7718** | — |

Test performance exceeding validation performance is counterintuitive and demands explanation rather than celebration. Three candidate causes were considered:

1. **Sampling luck.** Rejected. The test split contains 25,250 images ($3.3\times$ the validation split); a 4.57 pp gap is far outside plausible sampling variance at that size.
2. **Data leakage.** Rejected. Splits are generated from canonical manifests in `data/splits/` derived from the official Food-101 `train.txt`/`test.txt` partition; the test set is disjoint by construction and was evaluated exactly once.
3. **Label-noise asymmetry.** **Accepted.** As documented by Bossard et al. (2014), the Food-101 *training* split was deliberately left uncleaned and contains approximately **20% web-crawl label noise**, while the *test* split was **manually curated and verified**. Our validation split is carved from the noisy training partition, so it inherits that ~20% noise; the test split does not. A model that has correctly learned the true class concept will be penalised on noisy validation labels and rewarded on clean test labels.

The decisive evidence is that **this same inversion appears in all four architectures** — Custom CNN (+4.46 pp), MobileNetV2 (+3.96 pp), ResNet-50 (+4.07 pp) and EfficientNetB0 (+4.57 pp). A property shared by four independently trained architectures with different capacities and optimisation trajectories is a property of the **dataset**, not of any one model. This is a cross-cutting finding for Section 7 and should be stated once at benchmark level rather than four times.

A secondary implication concerns the interaction between model capacity and label noise. An over-parameterised network can memorise noisy training labels, which depresses clean-test performance. EfficientNetB0's small effective capacity (3.26M trainable parameters) combined with its early stopping point means it never entered the memorisation regime — it was still fitting broad culinary structure when training ended. Its **94.16% Top-5 accuracy**, the highest in the benchmark, indicates that even when the Top-1 prediction is wrong, the correct class is almost always among the model's leading hypotheses.

---

### 8.4.3 Compound Scaling vs. Residual Depth Scaling: The Central Trade-Off

This subsection addresses the comparative question assigned to this component: does **compound scaling** (EfficientNetB0) deliver a better accuracy-per-resource return than **residual depth scaling** (ResNet-50)?

Table 8.4 consolidates all four benchmarked architectures under the identical controlled protocol of Section 4; [`report/figures/fig8_accuracy_vs_parameters.png`](../figures/fig8_accuracy_vs_parameters.png) plots the same data as accuracy against complexity. Both are generated by [`notebooks/07_comparative_eval.ipynb`](../../notebooks/07_comparative_eval.ipynb).

| Metric | Custom CNN | MobileNetV2 | ResNet-50 | **EfficientNetB0** |
| :--- | ---: | ---: | ---: | ---: |
| **Scaling Philosophy** | — (baseline) | Width (depthwise) | Depth (residual) | **Compound (d·w·r)** |
| **Total Parameters** | 678,085 | 2,387,365 | 23,802,853 | **4,184,072** |
| **Trainable Parameters** | 675,653 | 1,737,445 | 9,130,085 | **3,261,825** |
| **Float32 Weight Size** | 2.59 MiB | 9.11 MiB | 90.80 MiB | **15.96 MiB** |
| **Training Wall-Clock** | 14,339.8 s | 11,862.0 s | 16,481.0 s | **13,158.6 s** |
| **Validation Top-1** | 57.23% | 64.73% | 69.47% | **72.69%** |
| **Test Top-1** | 61.69% | 68.69% | 73.54% | **77.26%** |
| **Test Top-5** | — | 90.15% | 92.02% | **94.16%** |
| **Test Macro F1** | 0.6169 | 0.6862 | 0.7355 | **0.7718** |
| **Test Loss** | 1.4441 | 1.1692 | 1.1110 | **0.8091** |
| **$\mathcal{E}_{\text{param}}$ (Top-1 % / M-params)** | 90.98 | 28.77 | 3.09 | **18.46** |
| **$\mathcal{E}_{\text{mem}}$ (Top-1 % / MiB)** | 23.85 | 7.54 | 0.81 | **4.84** |

#### 1. EfficientNetB0 Strictly Dominates ResNet-50

The comparison against ResNet-50 is not a trade-off at all — it is **Pareto domination**. EfficientNetB0 is superior on every axis measured:

$$\text{Accuracy: } 77.26\% \text{ vs. } 73.54\% \;\Rightarrow\; \mathbf{+3.72 \text{ pp}}$$
$$\text{Parameters: } 4.18\text{M vs. } 23.80\text{M} \;\Rightarrow\; \mathbf{5.69\times \text{ fewer}}$$
$$\text{Weight footprint: } 15.96\text{ MiB vs. } 90.80\text{ MiB} \;\Rightarrow\; \mathbf{5.69\times \text{ smaller}}$$
$$\text{Training time: } 13{,}159\text{ s vs. } 16{,}481\text{ s} \;\Rightarrow\; \mathbf{20.2\% \text{ faster}}$$

Expressed as the accuracy-to-parameter efficiency ratio $\mathcal{E}_{\text{param}} = \frac{\text{Test Top-1 (\%)}}{\text{Params (M)}}$ defined in Section 8.3.2:

$$\frac{\mathcal{E}_{\text{param}}(\text{EfficientNetB0})}{\mathcal{E}_{\text{param}}(\text{ResNet-50})} = \frac{18.46}{3.09} \approx \mathbf{5.98\times}$$

**This is the strongest single result in the benchmark and it confirms Hypothesis 3 without qualification.** ResNet-50 spends 19.6M additional parameters to arrive at an accuracy 3.72 pp *lower*. Those parameters are not merely inefficient; they are actively counter-productive under Food-101's noise conditions.

#### 2. Why Depth-Only Scaling Loses

Three mechanisms explain the gap, and they map directly onto the theory in Section 6.4.1:

- **Unbalanced scaling wastes capacity.** ResNet-50 scales depth aggressively (50 layers) while holding input resolution fixed at $224 \times 224$ and channel width at ImageNet defaults. Per Equation (4), depth contributes only *linearly* to FLOPs while width and resolution contribute *quadratically* — so a depth-scaled network buys the least representational return per unit of parameter cost. EfficientNetB0's NAS-derived baseline balances all three axes at $\phi = 0$, extracting more accuracy from a far smaller budget.
- **Squeeze-and-Excitation is decisive for fine-grained discrimination.** ResNet-50's bottleneck blocks weight every channel identically. EfficientNetB0's SE modules (Section 6.4.2) compute *input-conditional* channel gates, allowing the network to amplify texture-sensitive channels for `steak` and colour-sensitive channels for `beet_salad` within the same forward pass. For 101 classes sharing overlapping low-level statistics, this input-adaptive feature selection is worth more than 19M additional static parameters.
- **Capacity interacts badly with 20% label noise.** ResNet-50's 9.13M trainable parameters provide ample capacity to memorise mislabelled training examples. Its larger validation–test inversion behaviour and lower clean-test accuracy are consistent with partial noise memorisation, whereas EfficientNetB0's tighter capacity acts as an implicit regulariser.

#### 3. The Genuine Trade-Off: Against MobileNetV2

The honest trade-off in this benchmark is not against ResNet-50 but against MobileNetV2, and here EfficientNetB0 does **not** dominate:

$$\text{EfficientNetB0 buys } \mathbf{+8.57 \text{ pp}} \text{ of Top-1 accuracy } (68.69\% \rightarrow 77.26\%)$$
$$\text{at a cost of } \mathbf{1.75\times} \text{ more parameters and } \mathbf{1.75\times} \text{ more weight memory.}$$

MobileNetV2 retains the higher raw parameter-efficiency ratio ($\mathcal{E}_{\text{param}} = 28.77$ vs. $18.46$), so the choice is genuinely application-dependent:

- Where the binding constraint is **binary size or SRAM residency** — a bundled mobile app, a microcontroller, an INT8-quantised NPU — MobileNetV2's 9.11 MiB remains the correct engineering answer (Section 8.3.4).
- Where the binding constraint is **accuracy at acceptable cost** — a server-side food-logging API, a nutrition-estimation backend, a clinical dietary-assessment tool where an 8.57 pp error-rate difference has real consequences — EfficientNetB0 is the correct answer, and its 15.96 MiB footprint still fits comfortably in settings where ResNet-50's 90.80 MiB does not.

The Custom CNN's headline $\mathcal{E}_{\text{param}}$ of 90.98 is the highest in the table and must be interpreted carefully: it is an artifact of dividing by a very small denominator. Its absolute 61.69% ceiling is unusable for 101-way fine-grained classification, which demonstrates that efficiency ratios are only meaningful **above an absolute accuracy threshold**, never in isolation.

#### 4. A Measurement Caveat on Inference Latency

The latency column is deliberately omitted from Table 8.4. Our instrumentation recorded **two** figures for EfficientNetB0 — **13.43 ms/image** under batched inference (batch size 32) and **342.26 ms/image** at batch size 1 — a $25\times$ spread arising purely from GPU kernel-launch overhead and the absence of batch parallelism, not from any architectural property.

Because the other three components each logged a single latency figure without recording which protocol produced it, the four numbers in the `metrics.json` files are **not mutually comparable**. The clearest symptom is that MobileNetV2 — the smallest pretrained model in the benchmark — reports 167.29 ms/image against ResNet-50's 10.98 ms/image, which is physically implausible and is certainly a protocol artifact rather than a real result. Presenting those values side by side would appear to *refute* Hypothesis 2 (Edge Efficiency) on the strength of a measurement error.

Section 7 therefore re-profiles all four models under a single documented protocol (identical batch size, GPU warm-up iterations discarded, mean over 1,000 test images) before publishing any latency comparison. This is flagged here rather than silently corrected, because the discrepancy is visible in the committed artifacts.

---

### 8.4.4 Fine-Grained Error Patterns and Confusion Cluster Analysis

Per-class results are recorded in [`results/efficientnetb0/classification_report.json`](../../results/efficientnetb0/classification_report.json), with the cluster summary in [`results/efficientnetb0/error_analysis.json`](../../results/efficientnetb0/error_analysis.json) and the normalized 101-class matrix in [`results/efficientnetb0/confusion_matrix.png`](../../results/efficientnetb0/confusion_matrix.png).

#### 1. Highest-Performing Classes

| Class | Precision | Recall | F1-Score | Discriminative Visual Signature |
| :--- | ---: | ---: | ---: | :--- |
| **`edamame`** | 0.9841 | 0.9920 | **0.9880** | Saturated green, repetitive pod geometry, near-zero intra-class variance |
| **`macarons`** | 0.9710 | 0.9360 | **0.9532** | Rigid circular sandwich structure, artificial high-chroma colour palette |
| **`oysters`** | 0.9320 | 0.9320 | **0.9320** | Irregular shell texture on crushed ice, high structural saliency |
| **`hot_and_sour_soup`** | 0.9344 | 0.9120 | **0.9231** | Distinctive broth chroma with suspended tofu strips and chilli-oil sheen |
| **`miso_soup`** | 0.9221 | 0.9000 | **0.9109** | Uniform cloudy broth, characteristic lacquer bowl, consistent framing |
| **`spaghetti_carbonara`** | 0.8937 | 0.9080 | **0.9008** | Long-strand pasta geometry with pale egg emulsion |

These categories share **low intra-class variance**, **rigid geometric structure**, and **distinctive global colour statistics** — exactly the properties that Squeeze-and-Excitation channel gating exploits most effectively.

#### 2. Lowest-Performing Classes and Their Confusion Partners

| Class | Precision | Recall | F1-Score | Dominant Confusion Partner (rate) |
| :--- | ---: | ---: | ---: | :--- |
| **`steak`** | 0.5530 | 0.4800 | **0.5139** | `filet_mignon` (16.8%), `prime_rib` (11.2%) |
| **`apple_pie`** | 0.5261 | 0.5240 | **0.5251** | `bread_pudding` (9.2%) |
| **`ravioli`** | 0.5528 | 0.5440 | **0.5484** | `lasagna`, `gnocchi` |
| **`pork_chop`** | 0.5502 | 0.5480 | **0.5491** | `steak`, `filet_mignon` |
| **`foie_gras`** | 0.5789 | 0.5280 | **0.5523** | `scallops`, `filet_mignon` |
| **`bread_pudding`** | 0.5340 | 0.6280 | **0.5772** | `apple_pie`, `french_toast` |
| **`huevos_rancheros`** | 0.6737 | 0.5120 | **0.5818** | `breakfast_burrito`, `nachos` |
| **`scallops`** | 0.6432 | 0.5480 | **0.5918** | `foie_gras`, `grilled_salmon` |

#### 3. Confusion Symmetry Proves Genuine Visual Ambiguity

The `steak` $\leftrightarrow$ `filet_mignon` pair is confused in **both directions** — `steak` misclassified as `filet_mignon` 16.8% of the time, and `filet_mignon` misclassified as `steak` 11.2% of the time. A one-directional error would indicate a class-prior bias or a decision-boundary artifact that additional training could correct. **Bidirectional, near-symmetric confusion instead indicates that the two classes are genuinely overlapping in the input space**: a seared beef medallion photographed at $224 \times 224$ carries no reliable signal distinguishing a tenderloin cut from a sirloin cut. This is a limit of the task specification, not a deficiency of the model, and no amount of additional training on this data would resolve it.

The complete set of leading confusions partitions cleanly into four semantic clusters:

1. **The Seared-Protein Cluster** — `steak` / `filet_mignon` / `prime_rib` / `pork_chop` / `foie_gras`. Browned Maillard surfaces, dark reduction sauces, similar plating conventions.
2. **The Baked-Dessert Cluster** — `apple_pie` / `bread_pudding` / `french_toast`. Golden-brown crust, caramelised sugar, dusted icing sugar.
3. **The Raw-Preparation Cluster** — `beef_tartare` $\rightarrow$ `tuna_tartare` (12.0%), `ceviche`. Identical ring-moulded presentation; the discriminating cue is protein hue alone.
4. **The Soft-Texture / Grain Cluster** — `chocolate_cake` $\rightarrow$ `chocolate_mousse` (11.2%), `frozen_yogurt` $\rightarrow$ `ice_cream` (9.2%), `fried_rice` $\rightarrow$ `risotto` (9.2%).

#### 4. Cross-Architectural Validation of Hypothesis 4

Ranking all 101 classes by mean F1 across the four architectures — plotted in [`report/figures/fig9_hardest_classes.png`](../figures/fig9_hardest_classes.png) and tabulated in [`results/comparison/hardest_classes.csv`](../../results/comparison/hardest_classes.csv) — shows that **the identical clusters dominate the error space of every architecture**, at uniformly higher absolute performance:

| Class | MobileNetV2 F1 | EfficientNetB0 F1 | Improvement |
| :--- | ---: | ---: | ---: |
| `steak` | 0.3491 | **0.5139** | +0.1648 |
| `apple_pie` | 0.3923 | **0.5251** | +0.1328 |
| `pork_chop` | 0.4101 | **0.5491** | +0.1390 |
| `foie_gras` | 0.4271 | **0.5523** | +0.1252 |
| `ravioli` | 0.4377 | **0.5484** | +0.1107 |

`steak` is the single worst class for **both** architectures; `apple_pie`, `pork_chop`, `foie_gras` and `ravioli` appear in the bottom five of both. **Hypothesis 4 is therefore confirmed:** the error space is governed by intrinsic visual proximity between fine-grained culinary categories, not by architectural choice. Compound scaling raises the entire performance curve — every hard class improves by 0.11–0.16 F1 — but it does not change *which* classes are hard.

The practical corollary is that further accuracy on Food-101 will not come from a better backbone. It requires either higher input resolution (to resolve meat-fibre and crumb-structure texture that $224 \times 224$ discards), or task reformulation — hierarchical classification that predicts a coarse superclass before a fine-grained leaf, or accepting Top-3 prediction as the product-level output.

---

### 8.4.5 Synthesis and Deployment Recommendation

1. **EfficientNetB0 is the recommended production architecture for server-side deployment.** It delivers the benchmark's best Top-1 (77.26%), Top-5 (94.16%), macro F1 (0.7718) and test loss (0.8091) while consuming 5.69× fewer parameters and 20.2% less training time than ResNet-50. There is no axis on which ResNet-50 is preferable, so **residual depth scaling is not justified for this task at this compute budget**.

2. **Architecture selection should be constraint-driven, not accuracy-driven.** The benchmark yields a clean decision rule: choose **MobileNetV2** when binary size or on-device SRAM residency binds; choose **EfficientNetB0** when accuracy binds and ~16 MB is affordable; choose **ResNet-50** in neither case.

3. **The reported result is a floor, not a ceiling.** Validation loss was still decreasing at termination and `EarlyStopping` never fired. Extending Phase 2 beyond 10 epochs, adopting progressive unfreezing of Stage 5, or scaling to EfficientNetB1–B3 at higher input resolution (per the compound-scaling law of Equation 4, at a predictable $2^{\phi}$ compute cost) are the highest-value next steps.

4. **Top-5 output is the correct product interface.** Given that the dominant errors are bidirectional confusions between genuinely ambiguous categories (`steak` $\leftrightarrow$ `filet_mignon`), no single-label interface can be reliable. At **94.16% Top-5 accuracy**, presenting five ranked candidates for user confirmation converts an architectural limitation into a one-tap interaction.

5. **Two artifact-level corrections are applied in Section 7.** The `model_size_mb` field was logged on differing bases across the four components — EfficientNetB0's 41.47 records an optimizer-inclusive `.keras` checkpoint against 15.96 MiB of actual weights, ResNet-50's 98.40 against 90.80 MiB — so [`notebooks/07_comparative_eval.ipynb`](../../notebooks/07_comparative_eval.ipynb) recomputes every footprint uniformly as $\text{total params} \times 4$ bytes. The four inference-latency figures were likewise captured under differing protocols and are withheld pending a single-protocol re-profiling. Both corrections are made in the open rather than silently, to avoid publishing misleading cross-architecture claims.
