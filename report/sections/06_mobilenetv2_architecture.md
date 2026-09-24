# Section 6: Model Architectures — MobileNetV2 Component

**Author:** Kanushka Kahakotuwa (Member 3)  
**Assigned Architecture:** MobileNetV2 (Inverted Residual and Linear Bottleneck CNN)  
**Academic Module:** SE4050 — Deep Learning  

---

## 6.3 MobileNetV2 (Inverted Residual and Linear Bottleneck Network)

### 6.3.1 Theoretical Foundation: Depthwise Separable Convolutions

Standard convolutional layers in deep neural networks jointly compute spatial features and cross-channel correlations within a single parameterized tensor operation. For an input feature map $\mathbf{X} \in \mathbb{R}^{D_F \times D_F \times M}$ (where $D_F$ denotes spatial height and width, and $M$ represents input channel depth), a standard convolutional layer parameterized by kernel weights $\mathbf{W} \in \mathbb{R}^{D_K \times D_K \times M \times N}$ produces an output feature map $\mathbf{Y} \in \mathbb{R}^{D_F \times D_F \times N}$ through the discrete cross-correlation:

$$\mathbf{Y}_{k, l, n} = \sum_{i=1}^{D_K} \sum_{j=1}^{D_K} \sum_{m=1}^{M} \mathbf{W}_{i, j, m, n} \cdot \mathbf{X}_{k+i-1, l+j-1, m}$$

The total computational cost in terms of multiply-accumulate (MAC) operations for standard convolution is:

$$\mathcal{C}_{\text{standard}} = D_K \cdot D_K \cdot M \cdot N \cdot D_F \cdot D_F$$

To break the cubic dependency on spatial dimensions and channel dimensions, Howard et al. (2017) and Sandler et al. (2018) formulated **Depthwise Separable Convolutions**, which factorize the standard convolution into two decoupled, computationally lightweight operations:

1. **Depthwise Convolution (Spatial Filtering):** Applies a single convolutional filter per input channel independently, learning pure spatial representations without inter-channel communication:
   
   $$\mathbf{\hat{Y}}_{k, l, m} = \sum_{i=1}^{D_K} \sum_{j=1}^{D_K} \mathbf{W}^{\text{dw}}_{i, j, m} \cdot \mathbf{X}_{k+i-1, l+j-1, m}$$

   The computational cost of the depthwise phase is:
   
   $$\mathcal{C}_{\text{depthwise}} = D_K \cdot D_K \cdot M \cdot D_F \cdot D_F$$

2. **Pointwise Convolution (Channel Combining):** Applies a $1 \times 1$ convolution across all channels to linearly combine spatial features into a new channel space of depth $N$:
   
   $$\mathbf{Y}_{k, l, n} = \sum_{m=1}^{M} \mathbf{W}^{\text{pw}}_{m, n} \cdot \mathbf{\hat{Y}}_{k, l, m}$$

   The computational cost of the pointwise phase is:
   
   $$\mathcal{C}_{\text{pointwise}} = M \cdot N \cdot D_F \cdot D_F$$

Combining both operations, the total computational complexity of Depthwise Separable Convolution is:

$$\mathcal{C}_{\text{separable}} = D_K \cdot D_K \cdot M \cdot D_F \cdot D_F + M \cdot N \cdot D_F \cdot D_F$$

Dividing Equation (5) by Equation (2) yields the exact theoretical computational reduction ratio:

$$\frac{\mathcal{C}_{\text{separable}}}{\mathcal{C}_{\text{standard}}} = \frac{D_K^2 \cdot M \cdot D_F^2 + M \cdot N \cdot D_F^2}{D_K^2 \cdot M \cdot N \cdot D_F^2} = \frac{1}{N} + \frac{1}{D_K^2}$$

For standard $3 \times 3$ convolution kernels ($D_K = 3$) and typical channel dimensions ($N \ge 64$), the term $\frac{1}{N} \ll \frac{1}{9}$, yielding an approximate computational reduction factor of:

$$\frac{\mathcal{C}_{\text{separable}}}{\mathcal{C}_{\text{standard}}} \approx \frac{1}{9} \approx 0.111$$

This demonstrates a theoretical **$88\%$ to $89\%$ reduction in computational FLOPs and parameter count** compared to standard convolutions with virtually negligible degradation in feature extraction capacity.

---

### 6.3.2 Inverted Residuals and the Linear Bottleneck Principle

While MobileNetV1 demonstrated the parameter efficiency of depthwise separable convolutions, it suffered from optimization instability and representational bottlenecks when scaled down. MobileNetV2 (Sandler et al., 2018) introduced two architectural breakthroughs: **Inverted Residual Blocks** and **Linear Bottlenecks**.

#### 1. Inverted Residual Topology (Narrow $\rightarrow$ Wide $\rightarrow$ Narrow)
In traditional residual architectures such as ResNet-50 (He et al., 2016), residual bottleneck blocks follow a **Wide $\rightarrow$ Narrow $\rightarrow$ Wide** topology: high-dimensional channels are compressed via $1 \times 1$ convolution, processed via $3 \times 3$ convolution, and expanded back to high dimensions before the shortcut addition.

In contrast, MobileNetV2 inverts this paradigm into a **Narrow $\rightarrow$ Wide $\rightarrow$ Narrow** structure:
- **Input:** Low-dimensional bottleneck representation ($k$ channels).
- **Expansion ($1 \times 1$ Conv):** Expands the channel depth by an expansion factor $t=6$ to $t \cdot k$ channels, projecting the representation into a higher-dimensional space where expressive non-linear transformations can occur.
- **Spatial Filtering ($3 \times 3$ Depthwise Conv):** Applies spatial feature extraction across the expanded $t \cdot k$ channels.
- **Projection ($1 \times 1$ Conv):** Compresses channel depth back down to the low-dimensional bottleneck space ($k'$ channels).
- **Residual Shortcut Connection:** Added directly between the low-dimensional bottleneck endpoints (when input and output dimensions match and stride $s=1$).

```
Classical ResNet Bottleneck:         MobileNetV2 Inverted Residual:
   [High Channels (e.g. 256)]           [Low Channels (e.g. 32)]
              │                                    │
              ▼                                    ▼
       1x1 Conv (Compress)                 1x1 Conv (Expand, t=6)
              ▼                                    ▼
       3x3 Conv (Spatial)                  3x3 Depthwise Conv (Spatial)
              ▼                                    ▼
       1x1 Conv (Expand)                   1x1 Conv (Linear Projection)
              ▼                                    ▼
   [High Channels (e.g. 256)]           [Low Channels (e.g. 32)]
              │                                    │
       (+) Shortcut                         (+) Shortcut
```

#### 2. The Linear Bottleneck Principle (Manifold Preservation)
A central theoretical insight of MobileNetV2 is that the "manifold of interest" in real-world visual data can be embedded within low-dimensional subspaces. However, applying a non-linear activation (such as standard ReLU) to low-dimensional representations unavoidably destroys information:

$$\text{ReLU}(z) = \max(0, z)$$

Whenever an activation $z < 0$, ReLU zeroes out the coordinate, collapsing the manifold structure. In high-dimensional spaces (e.g., after the $1 \times 1$ expansion to $t \cdot k$ channels), zeroing out negative coordinates retains sufficient dimensional degrees of freedom. However, if non-linear activations are applied after projecting back down to the narrow bottleneck, significant visual information is permanently lost.

Therefore, MobileNetV2 enforces a **strictly Linear Bottleneck**: no non-linear activation is applied after the final $1 \times 1$ projection layer. Non-linearities ($\text{ReLU6}$) are applied exclusively in the expanded high-dimensional hidden representations.

#### 3. Bounded ReLU6 Activation for Edge Deployment
To ensure numerical stability on low-power mobile microprocessors and integer-quantized edge accelerators (such as Google Edge TPU or ARM Cortex-M), MobileNetV2 uses the bounded **$\text{ReLU6}$** activation function:

$$\text{ReLU6}(z) = \min(\max(0, z), 6)$$

Equation (9) clamps positive activations to a maximum upper bound of $6.0$. When quantized to low-precision fixed-point representations (such as 8-bit integer `int8`), $\text{ReLU6}$ prevents dynamic range overflow and preserves high resolution for fractional values, guaranteeing seamless hardware quantization without accuracy loss.

---

### 6.3.3 Complete Architecture Structural Breakdown

The structural implementation of the MobileNetV2 Inverted Residual Block depends on the stride parameter $s$:

```
Case 1: Stride s = 1 (Input channels = Output channels)
                  Input Feature Map x (h x w x k)
                           │                    │
                           ▼                    │
                 ┌──────────────────┐           │
                 │ 1x1 Conv2D (t*k) │           │
                 │ Batch Normalization          │
                 │ ReLU6            │           │
                 └─────────┬────────┘           │
                           ▼                    │
                 ┌──────────────────┐           │
                 │ 3x3 Depthwise    │           │
                 │ Batch Normalization          │
                 │ ReLU6            │           │
                 └─────────┬────────┘           │  Identity Shortcut
                           ▼                    │  (Residual Connection)
                 ┌──────────────────┐           │
                 │ 1x1 Linear Conv  │           │
                 │ Batch Normalization          │
                 │ (No Activation)  │           │
                 └─────────┬────────┘           │
                           ▼                    │
                           └─────────►(+)◄──────┘
                                       │
                                       ▼
                               Output Feature Map

Case 2: Stride s = 2 (Spatial Downsampling)
                  Input Feature Map x (h x w x k)
                               │
                               ▼
                 ┌──────────────────┐
                 │ 1x1 Conv2D (t*k) │
                 │ Batch Normalization
                 │ ReLU6            │
                 └─────────┬────────┘
                           ▼
                 ┌──────────────────┐
                 │ 3x3 Depthwise    │ (Stride = 2, Spatial Halving)
                 │ Batch Normalization
                 │ ReLU6            │
                 └─────────┬────────┘
                           ▼
                 ┌──────────────────┐
                 │ 1x1 Linear Conv  │
                 │ Batch Normalization
                 │ (No Activation)  │
                 └─────────┬────────┘
                           │  (No Shortcut — spatial dimensions mismatched)
                           ▼
                   Output (h/2 x w/2 x k')
```

The MobileNetV2 backbone consists of 17 sequential Inverted Residual Bottleneck blocks arranged across 7 macro-stages, preceded by an initial standard $3 \times 3$ convolutional stem and followed by a $1 \times 1$ pointwise expansion layer:

| Stage | Input Resolution | Operator | Expansion Factor ($t$) | Output Channels ($c$) | Block Count ($n$) | Stride ($s$) |
| :---: | :---: | :--- | :---: | :---: | :---: | :---: |
| **Stem** | $224 \times 224 \times 3$ | Conv2D $3 \times 3$ | — | 32 | 1 | 2 |
| **Stage 1** | $112 \times 112 \times 32$ | Inverted Residual | 1 | 16 | 1 | 1 |
| **Stage 2** | $112 \times 112 \times 16$ | Inverted Residual | 6 | 24 | 2 | 2 |
| **Stage 3** | $56 \times 56 \times 24$ | Inverted Residual | 6 | 32 | 3 | 2 |
| **Stage 4** | $28 \times 28 \times 32$ | Inverted Residual | 6 | 64 | 4 | 2 |
| **Stage 5** | $14 \times 14 \times 64$ | Inverted Residual | 6 | 96 | 3 | 1 |
| **Stage 6** | $14 \times 14 \times 96$ | Inverted Residual | 6 | 160 | 3 | 2 |
| **Stage 7** | $7 \times 7 \times 160$ | Inverted Residual | 6 | 320 | 1 | 1 |
| **Head Ext**| $7 \times 7 \times 320$ | Conv2D $1 \times 1$ | — | 1280 | 1 | 1 |

---

### 6.3.4 Custom Classification Head Design & Parameter Footprint

To adapt the pretrained MobileNetV2 backbone to the 101-class Food-101 classification benchmark, we stripped the native 1,000-class ImageNet classification layer and appended a task-specific classification head designed for high generalization and regularization:

1. **Global Average Pooling (`GlobalAveragePooling2D`):** Collapses the final spatial tensor of shape $(7 \times 7 \times 1280)$ into a 1D feature vector of shape $(1280,)$ by computing spatial spatial means across each channel:
   
   $$\bar{z}_c = \frac{1}{7 \times 7} \sum_{i=1}^{7} \sum_{j=1}^{7} \mathbf{F}_{i, j, c}, \quad c \in \{1, \dots, 1280\}$$

   This eliminates spatial positioning dependencies, minimizes total parameters compared to flattening, and introduces structural invariance to spatial food plate translations.

2. **Dropout Regularization (`Dropout(0.2)`):** During training, $20\%$ of channel activations are randomly zeroed out with rate $p = 0.20$. This prevents co-adaptation of high-level food texture features and mitigates overfitting on complex dishes.

3. **Dense Softmax Output Layer (`Dense(101, activation='softmax')`):** Projects the 1,280-dimensional feature embedding into 101 normalized class posterior probabilities:
   
   $$P(y = k \mid \mathbf{x}) = \frac{\exp(z_k)}{\sum_{j=1}^{101} \exp(z_j)}, \quad k \in \{1, \dots, 101\}$$

#### Comprehensive Parameter Breakdown
The resulting parameter footprint of our functional Keras implementation is summarized below:

| Component | Layer Specification | Output Shape | Parameters | Trainable (Phase 1) | Trainable (Phase 2) |
| :--- | :--- | :---: | :---: | :---: | :---: |
| **Input Tensor** | `Input(shape=(224, 224, 3))` | $(224, 224, 3)$ | 0 | 0 | 0 |
| **MobileNetV2 Backbone**| Layers 1 to 119 (Stages 1–5) | $(14, 14, 96)$ | 1,180,544 | 0 (Frozen) | 0 (Frozen) |
| **Fine-Tuning Blocks** | Layers 120 to 154 (Stages 6–7) | $(7, 7, 1280)$ | 1,077,440 | 0 (Frozen) | 1,061,760 |
| **Batch Normalization**| Throughout backbone | Multiple | 34,112 | 0 (Frozen) | 0 (Frozen) |
| **Global Pooling** | `GlobalAveragePooling2D()` | $(1280,)$ | 0 | 0 | 0 |
| **Regularization** | `Dropout(rate=0.2)` | $(1280,)$ | 0 | 0 | 0 |
| **Classification Head**| `Dense(units=101, softmax)` | $(101,)$ | 129,381 | 129,381 | 129,381 |
| **TOTALS** | — | — | **2,387,365** | **129,381 (5.4%)** | **1,737,445 (72.8%)** |

The total uncompressed weight footprint of MobileNetV2 is **9.11 MB**, making it **$10.7\times$ smaller than ResNet-50 (98.4 MB)** and easily deployable within embedded mobile storage constraints.

---

### 6.3.5 Two-Stage Transfer Learning and Progressive Fine-Tuning Protocol

Training deep convolutional models directly from scratch on 101-class fine-grained food datasets requires millions of annotated samples to learn low-level edge and texture filters. To maximize transfer efficiency while preserving optimization stability, we executed a disciplined **Two-Phase Transfer Learning Protocol**:

```
PHASE 1: Feature Extraction (Epochs 1–8)
┌──────────────────────────────────────────────┐
│  MobileNetV2 Backbone (Layers 1–154)         │ ===> STRICTLY FROZEN
│  ImageNet Pretrained Weights (2.26M params)  │      No Gradient Updates
└──────────────────────┬───────────────────────┘
                       ▼
┌──────────────────────────────────────────────┐
│  Custom Head (GAP + Dropout 0.2 + Dense 101) │ ===> TRAINABLE
│  Randomly Initialized (129K params)          │      Adam Optimizer (lr = 1e-3)
└──────────────────────────────────────────────┘

PHASE 2: Domain-Specific Fine-Tuning (Epochs 9–20)
┌──────────────────────────────────────────────┐
│  Early Backbone Layers 1–119 (Low-level)     │ ===> STRICTLY FROZEN
│  General Edge & Color Filters                │      Preserves Primitive Features
├──────────────────────────────────────────────┤
│  Top Residual Blocks 14–16 (Layers 120–154)  │ ===> UNFROZEN & TRAINABLE
│  High-Level Semantic Food Texture Blocks     │      Adam Optimizer (lr = 1e-5)
├──────────────────────────────────────────────┤
│  All Batch Normalization Layers              │ ===> STRICTLY FROZEN
│  Running Mean & Variance Statistics Locked   │      Prevents Internal Covariate Shift
├──────────────────────────────────────────────┤
│  Custom Classification Head (101 classes)    │ ===> TRAINABLE (lr = 1e-5)
└──────────────────────────────────────────────┘
```

#### Phase 1: Feature Extraction
- **Objective:** Train the newly initialized 101-class classification head without corrupting the generic spatial feature detectors in the pretrained backbone.
- **Frozen Backbone:** All 154 layers of MobileNetV2 are locked (`trainable = False`), restricting gradient backpropagation to the 129,381 parameters of the dense head.
- **Optimizer & Learning Rate:** Adam optimizer initialized with $\eta = 10^{-3}$, $\beta_1 = 0.9$, $\beta_2 = 0.999$, and $\epsilon = 10^{-7}$.
- **Callbacks:** `EarlyStopping(patience=4, restore_best_weights=True)` and `ReduceLROnPlateau(factor=0.2, patience=2, min_lr=1e-5)`.
- **Duration:** 8 full epochs.

#### Phase 2: High-Level Inverted Residual Fine-Tuning
- **Objective:** Adapt the high-level semantic representation of MobileNetV2 to the complex visual textures, culinary presentations, and ingredient groupings of Food-101.
- **Selective Layer Unfreezing:** The network is split at layer index 120 (corresponding to Inverted Residual Blocks 14, 15, and 16 in Stages 6 and 7). Layers 1 to 119 remain frozen, preserving generalizable low-level filters (Gabor-like edge detectors, color gradients).
- **The Critical BatchNorm Freeze Rule:** As established by Radford et al. and He et al., unfreezing Batch Normalization layers during fine-tuning with small mini-batches ($\text{batch size} = 32$) causes severe internal covariate shift: mini-batch sample statistics overwrite the well-calibrated population statistics ($\mu_{\text{pop}}, \sigma^2_{\text{pop}}$) learned across 1.28 million ImageNet images. We strictly enforced:
  
  ```python
  for layer in model.layers[120:]:
      if isinstance(layer, tf.keras.layers.BatchNormalization):
          layer.trainable = False
      else:
          layer.trainable = True
  ```

- **Gentle Learning Rate ($\eta = 10^{-5}$):** To prevent catastrophic forgetting of pretrained weights, the learning rate was decreased by two orders of magnitude to $10^{-5}$.
- **Duration:** 12 additional fine-tuning epochs (Epochs 9–20), completing the full 20-epoch training schedule.
