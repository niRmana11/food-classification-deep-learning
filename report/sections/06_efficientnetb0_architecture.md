# Section 6: Model Architectures — EfficientNetB0 Component

**Author:** Kaveesha Athukorala (Member 4)  
**Assigned Architecture:** EfficientNetB0 (Compound-Scaled MBConv Network)  
**Academic Module:** SE4050 — Deep Learning  

---

## 6.4 EfficientNetB0 (Compound Scaling with Squeeze-and-Excitation MBConv Blocks)

### 6.4.1 Theoretical Foundation: The Compound Scaling Principle

Prior to Tan & Le (2019), convolutional network scaling was performed along a **single dimension at a time**. Practitioners scaled one of three axes in isolation:

1. **Depth ($d$):** stacking additional layers (the ResNet family: ResNet-18 $\rightarrow$ ResNet-50 $\rightarrow$ ResNet-152).
2. **Width ($w$):** widening the channel dimension of each layer (WideResNet, MobileNet width multiplier $\alpha$).
3. **Resolution ($r$):** feeding higher-resolution input images (e.g., $224 \times 224 \rightarrow 331 \times 331$).

Each of these scaling strategies improves accuracy, but each **saturates independently**. Tan & Le empirically demonstrated that scaling depth alone beyond $d \approx 6.0$ yields diminishing accuracy returns, because very deep networks become difficult to optimize despite skip connections; scaling width alone saturates because wide-but-shallow networks capture rich fine-grained features but fail to learn high-level semantic abstractions; and scaling resolution alone saturates because additional input pixels cannot be exploited by a receptive field that has not grown correspondingly.

The central insight of EfficientNet is that these three dimensions are **not independent**. Higher input resolution demands greater depth (to enlarge the receptive field enough to cover the larger image) and greater width (to encode the finer-grained patterns that additional pixels expose). Optimal scaling therefore requires increasing all three **in a fixed ratio simultaneously**.

Formally, the authors define a **compound coefficient** $\phi$ that uniformly scales all three dimensions:

$$\text{depth: } d = \alpha^{\phi}, \qquad \text{width: } w = \beta^{\phi}, \qquad \text{resolution: } r = \gamma^{\phi}$$

subject to the constraint:

$$\alpha \cdot \beta^{2} \cdot \gamma^{2} \approx 2, \qquad \text{where } \alpha \ge 1,\; \beta \ge 1,\; \gamma \ge 1$$

The rationale for the constraint follows directly from the computational cost of convolution. The FLOP count of a standard convolutional layer is **linear** in depth but **quadratic** in both width and resolution:

$$\text{FLOPs} \;\propto\; d \cdot w^{2} \cdot r^{2}$$

Substituting the scaling definitions from Equation (1):

$$\text{FLOPs}(\phi) \;\propto\; \alpha^{\phi} \cdot \left(\beta^{\phi}\right)^{2} \cdot \left(\gamma^{\phi}\right)^{2} = \left(\alpha \cdot \beta^{2} \cdot \gamma^{2}\right)^{\phi} \approx 2^{\phi}$$

Equation (4) is the key result: under the constraint of Equation (2), **total computational cost increases by a predictable factor of exactly $2^{\phi}$** for any chosen $\phi$. This converts model scaling from an expensive architecture search into a single-parameter decision with a known compute budget.

A small grid search over the baseline network under the fixed budget $\phi = 1$ yielded the published constants:

$$\alpha = 1.2 \;(\text{depth}), \qquad \beta = 1.1 \;(\text{width}), \qquad \gamma = 1.15 \;(\text{resolution})$$

**EfficientNet-B0 is the baseline network at $\phi = 0$**, i.e. $d = w = r = 1.0$. It is the architecture from which B1 through B7 are derived by increasing $\phi$. Selecting B0 for this benchmark is therefore deliberate: it isolates the contribution of the **NAS-derived block topology and the compound-scaling design philosophy** at the smallest possible compute budget, which is the fairest comparison point against ResNet-50 (a depth-scaled network) and MobileNetV2 (a width-scaled network) under our locked $224 \times 224$ / batch-32 protocol.

---

### 6.4.2 The MBConv Block: Inverted Residuals with Squeeze-and-Excitation

The B0 baseline was itself discovered by **Neural Architecture Search (NAS)** optimizing a multi-objective reward that balances accuracy against target FLOPs. The resulting primitive is the **MBConv block** (Mobile Inverted Bottleneck Convolution), which extends MobileNetV2's inverted residual (Section 6.3.2) with two additions: **Squeeze-and-Excitation channel attention** and the **Swish** activation function.

#### 1. Squeeze-and-Excitation (SE) Channel Attention

A standard convolution treats every output channel as equally important. Squeeze-and-Excitation (Hu et al., 2018) allows the network to **recalibrate channel importance conditioned on the input image itself**. Given an intermediate feature map $\mathbf{U} \in \mathbb{R}^{H \times W \times C}$:

**Squeeze** — global spatial information is compressed into a per-channel descriptor by global average pooling:

$$z_{c} = \mathbf{F}_{sq}(\mathbf{u}_{c}) = \frac{1}{H \times W} \sum_{i=1}^{H} \sum_{j=1}^{W} u_{c}(i, j), \qquad c \in \{1, \dots, C\}$$

**Excitation** — the descriptor $\mathbf{z} \in \mathbb{R}^{C}$ is passed through a two-layer bottleneck MLP with reduction ratio $r = 0.25$, producing per-channel gating weights in $(0, 1)$:

$$\mathbf{s} = \mathbf{F}_{ex}(\mathbf{z}, \mathbf{W}) = \sigma\!\left(\mathbf{W}_{2} \cdot \delta\!\left(\mathbf{W}_{1} \mathbf{z}\right)\right)$$

where $\delta$ denotes Swish, $\sigma$ denotes the sigmoid function, $\mathbf{W}_{1} \in \mathbb{R}^{rC \times C}$ and $\mathbf{W}_{2} \in \mathbb{R}^{C \times rC}$.

**Scale** — each channel of the original feature map is rescaled by its learned gate:

$$\tilde{\mathbf{u}}_{c} = s_{c} \cdot \mathbf{u}_{c}$$

For fine-grained food classification this mechanism is directly relevant. When the input is an image of `caesar_salad`, the SE gate can amplify channels responsive to leaf texture and shredded-cheese granularity while suppressing channels tuned to plate rim and tablecloth. The network learns an **input-dependent feature selection policy** rather than a single fixed filter bank, which is precisely what is required when 101 classes share overlapping low-level statistics.

#### 2. Swish Activation

EfficientNet replaces ReLU with the smooth, non-monotonic **Swish** function:

$$\text{Swish}(x) = x \cdot \sigma(x) = \frac{x}{1 + e^{-x}}$$

Unlike ReLU, Swish is differentiable everywhere and permits small negative activations to propagate ($\text{Swish}(x) \to 0^{-}$ as $x \to -\infty$ rather than being hard-clipped to exactly zero). This preserves a weak gradient signal through nominally "dead" units and empirically improves convergence in deep stacks.

#### 3. Complete MBConv Block Structure

```
                 Input Feature Map x (h x w x k)
                          │                    │
                          ▼                    │
                ┌──────────────────┐           │
                │ 1x1 Conv (Expand)│  t = 6    │
                │ BatchNorm + Swish│           │
                └─────────┬────────┘           │
                          ▼                    │
                ┌──────────────────┐           │
                │ kxk Depthwise    │  k ∈ {3,5}│
                │ BatchNorm + Swish│           │
                └─────────┬────────┘           │  Identity Shortcut
                          ▼                    │  (stride 1 and k == k')
          ┌───────────────────────────┐        │
          │  SQUEEZE-AND-EXCITATION   │        │
          │  GlobalAvgPool -> (C,)    │        │
          │  Dense(0.25C) + Swish     │        │
          │  Dense(C) + Sigmoid       │        │
          │  Channel-wise Multiply    │        │
          └─────────────┬─────────────┘        │
                        ▼                      │
                ┌──────────────────┐           │
                │ 1x1 Conv (Project)│          │
                │ BatchNorm         │          │
                │ (Linear — no act.)│          │
                └─────────┬────────┘           │
                          ▼                    │
                          └────────►(+)◄───────┘
                                     │
                                     ▼
                             Output (h x w x k')
```

The `1x1 Conv (Project)` layer carries **no activation**, preserving MobileNetV2's Linear Bottleneck principle (Section 6.3.2): applying a non-linearity in the narrow bottleneck would irrecoverably collapse the low-dimensional manifold.

---

### 6.4.3 EfficientNet-B0 Structural Breakdown

The B0 backbone comprises **16 MBConv blocks** organised into 7 stages, bracketed by a $3 \times 3$ convolutional stem and a $1 \times 1$ pointwise head expansion. Keras exposes these stages under the layer-name prefixes `block1` through `block7`:

| Stage | Keras Prefix | Input Resolution | Operator | Expansion ($t$) | Kernel | Output Channels | Blocks ($n$) | Stride |
| :---: | :--- | :---: | :--- | :---: | :---: | :---: | :---: | :---: |
| **Stem** | `stem_conv` | $224 \times 224 \times 3$ | Conv2D | — | $3 \times 3$ | 32 | 1 | 2 |
| **Stage 1** | `block1` | $112 \times 112 \times 32$ | MBConv1 + SE | 1 | $3 \times 3$ | 16 | 1 | 1 |
| **Stage 2** | `block2` | $112 \times 112 \times 16$ | MBConv6 + SE | 6 | $3 \times 3$ | 24 | 2 | 2 |
| **Stage 3** | `block3` | $56 \times 56 \times 24$ | MBConv6 + SE | 6 | $5 \times 5$ | 40 | 2 | 2 |
| **Stage 4** | `block4` | $28 \times 28 \times 40$ | MBConv6 + SE | 6 | $3 \times 3$ | 80 | 3 | 2 |
| **Stage 5** | `block5` | $14 \times 14 \times 80$ | MBConv6 + SE | 6 | $5 \times 5$ | 112 | 3 | 1 |
| **Stage 6** | `block6` | $14 \times 14 \times 112$ | MBConv6 + SE | 6 | $5 \times 5$ | 192 | 4 | 2 |
| **Stage 7** | `block7` | $7 \times 7 \times 192$ | MBConv6 + SE | 6 | $3 \times 3$ | 320 | 1 | 1 |
| **Head Ext** | `top_conv` | $7 \times 7 \times 320$ | Conv2D | — | $1 \times 1$ | 1280 | 1 | 1 |

Two structural properties are worth noting against the other three benchmarked architectures:

- **Mixed kernel sizes.** Stages 3, 5 and 6 use $5 \times 5$ depthwise kernels rather than the uniform $3 \times 3$ used throughout MobileNetV2. NAS selected larger receptive fields specifically at mid-to-high abstraction levels, where broader spatial context distinguishes coarse dish layout.
- **Extreme back-loading of capacity.** As the parameter audit in Section 6.4.4 shows, Stage 6 alone holds $2{,}044{,}396$ parameters — **48.9% of the entire 4.18M-parameter network** — while Stages 1–3 together hold only $67{,}650$ parameters (1.6%). This concentration is what makes selective fine-tuning of the top stages so effective, and it directly motivates our Phase 2 unfreezing policy.

---

### 6.4.4 Classification Head Design and Parameter Footprint

The native 1,000-class ImageNet classifier was discarded (`include_top=False`), exposing the $7 \times 7 \times 1280$ feature map produced by `top_conv`. A task-specific head was appended, implemented in [`src/models/efficientnetb0.py`](../../src/models/efficientnetb0.py):

1. **Global Average Pooling (`GlobalAveragePooling2D`)** — collapses $(7 \times 7 \times 1280)$ to a $(1280,)$ embedding by spatial mean:

   $$\bar{z}_{c} = \frac{1}{7 \times 7} \sum_{i=1}^{7} \sum_{j=1}^{7} \mathbf{F}_{i,j,c}, \qquad c \in \{1, \dots, 1280\}$$

   A `Flatten` alternative would produce a $62{,}720$-dimensional vector and a $6.33$M-parameter Dense layer — **larger than the entire backbone** — and would overfit almost immediately. This choice deliberately mirrors the Custom CNN baseline head (Section 6.1) so that the architectural comparison isolates backbone quality rather than head design.

2. **Head Batch Normalization (`BatchNormalization`)** — re-centres and re-scales the pooled ImageNet feature distribution for the new 101-class objective. This layer is **not part of the backbone** and remains trainable in both phases, which is what allows the head to adapt even while every backbone BatchNorm is frozen.

3. **Dropout (`Dropout(0.3)`)** — zeroes 30% of pooled channels per training step, scaling survivors by $\tfrac{1}{1-p} \approx 1.43$. The rate of $0.3$ matches ResNet-50 (Section 6.2) rather than MobileNetV2's $0.2$, reflecting EfficientNetB0's larger head-facing embedding.

4. **Dense Softmax (`Dense(101, softmax)`)** — projects the 1,280-dimensional embedding to 101 class posteriors:

   $$P(y = k \mid \mathbf{x}) = \frac{\exp(z_{k})}{\sum_{j=1}^{101} \exp(z_{j})}, \qquad k \in \{1, \dots, 101\}$$

#### Comprehensive Parameter Audit

Parameter counts below are extracted directly from [`results/efficientnetb0/model_summary.txt`](../../results/efficientnetb0/model_summary.txt):

| Component | Keras Scope | Output Shape | Parameters | Trainable (Phase 1) | Trainable (Phase 2) |
| :--- | :--- | :---: | ---: | ---: | ---: |
| **Input + Internal Preprocessing** | `input_image`, `rescaling`, `normalization` | $(224, 224, 3)$ | 7 | 0 | 0 |
| **Convolutional Stem** | `stem_conv`, `stem_bn` | $(112, 112, 32)$ | 992 | 0 (Frozen) | 0 (Frozen) |
| **Stage 1 (MBConv1)** | `block1` | $(112, 112, 16)$ | 1,544 | 0 (Frozen) | 0 (Frozen) |
| **Stage 2 (MBConv6)** | `block2` | $(56, 56, 24)$ | 17,770 | 0 (Frozen) | 0 (Frozen) |
| **Stage 3 (MBConv6)** | `block3` | $(28, 28, 40)$ | 48,336 | 0 (Frozen) | 0 (Frozen) |
| **Stage 4 (MBConv6)** | `block4` | $(14, 14, 80)$ | 248,210 | 0 (Frozen) | 0 (Frozen) |
| **Stage 5 (MBConv6)** | `block5` | $(14, 14, 112)$ | 551,116 | 0 (Frozen) | 0 (Frozen) |
| **Stage 6 (MBConv6)** | `block6` | $(7, 7, 192)$ | 2,044,396 | 0 (Frozen) | **2,008,300** (36,096 BN frozen) |
| **Stage 7 (MBConv6)** | `block7` | $(7, 7, 320)$ | 722,480 | 0 (Frozen) | **711,984** (10,496 BN frozen) |
| **Head Expansion** | `top_conv` | $(7, 7, 1280)$ | 409,600 | 0 (Frozen) | **409,600** |
| **Head Expansion BN** | `top_bn` | $(7, 7, 1280)$ | 5,120 | 0 (Frozen) | 0 (Frozen) |
| **Global Pooling** | `global_avg_pool` | $(1280,)$ | 0 | 0 | 0 |
| **Head Normalization** | `bn_gap` | $(1280,)$ | 5,120 | **2,560** | **2,560** |
| **Regularization** | `dropout_head` | $(1280,)$ | 0 | 0 | 0 |
| **Classification Head** | `food101_predictions` | $(101,)$ | 129,381 | **129,381** | **129,381** |
| **TOTALS** | — | — | **4,184,072** | **131,941 (3.2%)** | **3,261,825 (78.0%)** |

The Phase 2 trainable total reconciles exactly:

$$\underbrace{2{,}008{,}300}_{\texttt{block6}} + \underbrace{711{,}984}_{\texttt{block7}} + \underbrace{409{,}600}_{\texttt{top\_conv}} + \underbrace{2{,}560 + 129{,}381}_{\text{head}} = \mathbf{3{,}261{,}825}$$

matching the value logged in [`results/efficientnetb0/metrics.json`](../../results/efficientnetb0/metrics.json). This arithmetic agreement is a **verification that the BatchNorm freeze was applied correctly**: the $51{,}712$ BatchNorm parameters residing inside the unfrozen stages remained excluded from the trainable set.

The float32 weight footprint is **15.96 MiB** — $5.69\times$ smaller than ResNet-50 (90.80 MiB) at higher accuracy, and only $1.75\times$ larger than MobileNetV2 (9.11 MiB).

> **Reporting note.** The `model_size_mb: 41.47` field in `metrics.json` records the on-disk `.keras` checkpoint, which additionally serialises the two Adam moment tensors for all 3.26M trainable parameters. The four components also logged this field on differing bases (ResNet-50 recorded 98.40 against 90.80 MiB of weights). Section 7 therefore recomputes the footprint uniformly as $\text{total params} \times 4\ \text{bytes}$ — exactly what Keras prints in `model.summary()` — for all four architectures, and that uniform figure is used throughout this report.

---

### 6.4.5 Two-Phase Transfer Learning Protocol

The training protocol is defined in [`results/efficientnetb0/config.yaml`](../../results/efficientnetb0/config.yaml) and executed in [`notebooks/06_efficientnetb0.ipynb`](../../notebooks/06_efficientnetb0.ipynb).

```
PHASE 1: Feature Extraction (Epochs 1–8)
┌──────────────────────────────────────────────┐
│  EfficientNetB0 Backbone (stem, block1–7,    │ ===> STRICTLY FROZEN
│  top_conv)  —  4,052,131 params              │      No Gradient Updates
└──────────────────────┬───────────────────────┘
                       ▼
┌──────────────────────────────────────────────┐
│  Head: GAP + BatchNorm + Dropout(0.3)        │ ===> TRAINABLE
│  + Dense(101, softmax)  —  131,941 params    │      Adam (lr = 1e-3)
└──────────────────────────────────────────────┘

PHASE 2: Selective Fine-Tuning (Epochs 9–18)
┌──────────────────────────────────────────────┐
│  stem, block1–block5  (867,968 params)       │ ===> STRICTLY FROZEN
│  Generic edge / colour / texture filters     │      Preserves primitives
├──────────────────────────────────────────────┤
│  block6, block7, top_conv                    │ ===> UNFROZEN & TRAINABLE
│  Class-specific semantic abstractions        │      Adam (lr = 1e-5)
├──────────────────────────────────────────────┤
│  ALL backbone BatchNormalization layers      │ ===> STRICTLY FROZEN
│  ImageNet running μ / σ² preserved           │      (51,712 params)
├──────────────────────────────────────────────┤
│  Classification head (incl. bn_gap)          │ ===> TRAINABLE (lr = 1e-5)
└──────────────────────────────────────────────┘
```

#### Phase 1 — Feature Extraction (8 epochs)

- **Objective:** fit the randomly initialised 101-class head without allowing large early gradients to corrupt the pretrained backbone.
- **Configuration:** `base_model.trainable = False`; Adam with $\eta = 10^{-3}$, $\beta_{1} = 0.9$, $\beta_{2} = 0.999$; `sparse_categorical_crossentropy` (the shared loader emits integer class indices, not one-hot vectors).
- **Callbacks:** `EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)` and `ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=2)`. Neither triggered — validation loss improved essentially monotonically across all 8 epochs, so the learning rate remained at $10^{-3}$ throughout.

#### Phase 2 — Selective Top-Stage Fine-Tuning (10 epochs)

- **Selection by name, not by index.** Layers are unfrozen by matching the prefixes `("block6", "block7", "top_conv")` rather than by an arbitrary slice index such as `layers[120:]`. The selection is therefore architecturally meaningful ("the final two MBConv stages plus the head expansion") and remains reproducible across Keras versions that may renumber layers.
- **Rationale for this cut point.** Stages 6 and 7 plus `top_conv` account for $3{,}176{,}476$ of $4{,}184{,}072$ parameters (**75.9%** of the network) yet occupy only the final $7 \times 7$ spatial resolution, where representations are most class-specific and therefore most in need of domain adaptation. Stages 1–5 encode Gabor-like edges, colour-opponency and generic texture that transfer from ImageNet to food photography unchanged.
- **The BatchNorm freeze rule.** Every backbone `BatchNormalization` layer is held frozen so that it executes in inference mode and continues to use the ImageNet population statistics $(\mu_{\text{pop}}, \sigma^{2}_{\text{pop}})$ estimated over 1.28M images, rather than overwriting them with noisy statistics from 32-image mini-batches:

  ```python
  base_model.trainable = True  # unfreeze everything, then re-freeze selectively
  for layer in base_model.layers:
      trainable = layer.name.startswith(FINETUNE_BLOCKS)
      if isinstance(layer, tf.keras.layers.BatchNormalization) and freeze_batchnorm:
          trainable = False
      layer.trainable = trainable
  ```

- **Mandatory recompilation.** Keras caches the trainable-weight list at compile time, so `compile_model()` is invoked again after the unfreezing step with the reduced learning rate. Omitting this recompilation causes the `trainable` changes to be **silently ignored** and Phase 2 to degenerate into a second Phase 1 — a failure mode that would be invisible in the loss curves.
- **Learning rate reduction.** $\eta$ is lowered by two orders of magnitude from $10^{-3}$ to $10^{-5}$. With 3.26M newly trainable parameters receiving gradients for the first time, retaining $10^{-3}$ would produce update magnitudes comparable to the pretrained weights themselves, destroying the ImageNet representation within a single epoch (catastrophic forgetting).
- **Duration:** 10 epochs (Epochs 9–18), for a combined 18-epoch schedule. Total wall-clock training time on the Google Colab NVIDIA T4 was **13,158.63 s (3.66 hours)**: 5,590.40 s for Phase 1 (699 s/epoch) and 7,568.23 s for Phase 2 (757 s/epoch), at a peak GPU memory occupancy of 3,288 MB.
