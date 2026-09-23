# Section 6: Model Architectures — ResNet-50 Component

**Author:** Matheesha Weerakoon (Member 2)  
**Assigned Architecture:** ResNet-50 (Deep Residual Network)  
**Academic Module:** SE4050 — Deep Learning  

---

## 6.2 Deep Residual Network (ResNet-50)

### 6.2.1 The Degradation Problem and Residual Learning Theory
In traditional feedforward convolutional neural networks (e.g., AlexNet, VGG), network depth is directly correlated with representational capacity. However, as demonstrated by He et al. (2016), stacking additional convolutional layers leads to the **degradation problem**: beyond a certain depth, network accuracy saturates and then degrades rapidly. This degradation is not caused by overfitting (as training error also increases), nor by vanishing/exploding gradients (which are mitigated by Batch Normalization and normalized weight initialization). Instead, it reflects an optimization difficulty: deeper networks struggle to learn simple identity mappings through consecutive non-linear transformations.

To resolve this, He et al. formulated the **Residual Learning Framework**. Given an underlying mapping $\mathcal{H}(x)$ to be fit by several stacked layers, the network is parameterized to learn the residual mapping:

$$\mathcal{F}(x) = \mathcal{H}(x) - x$$

The original mapping is cast into:

$$\mathcal{H}(x) = \mathcal{F}(x, \{W_i\}) + x$$

where $x$ and $\mathcal{H}(x)$ denote the input and output vectors of the residual block, and $\mathcal{F}(x, \{W_i\})$ represents the residual function to be learned. The operation $\mathcal{F} + x$ is implemented via feedforward connections with shortcut (skip) connections that bypass one or more parameterized layers.

From an optimization perspective, if an identity mapping were optimal, driving the residual weights $\mathcal{F}(x)$ to zero is substantially easier than learning an identity transformation across multiple stacked weight layers with non-linear activation functions:

$$\frac{\partial \mathcal{E}}{\partial x} = \frac{\partial \mathcal{E}}{\partial \mathcal{H}} \left( \frac{\partial \mathcal{F}}{\partial x} + 1 \right)$$

Equation (3) demonstrates that the gradient $\frac{\partial \mathcal{E}}{\partial x}$ always contains the term $1$, allowing backpropagated error gradients to flow unobstructed back to earlier layers regardless of network depth, effectively eliminating gradient attenuation.

---

### 6.2.2 Bottleneck Residual Block Architecture
In ResNet-50, residual learning is implemented through a 3-layer **Bottleneck Block** rather than a 2-layer basic block. Each bottleneck block consists of a sequence of three convolutions:

1. **$1 \times 1$ Convolution (Dimensionality Reduction):** Compresses channel depth by a factor of 4 (e.g., from 256 to 64 channels), reducing the input dimension for the subsequent spatial convolution.
2. **$3 \times 3$ Convolution (Spatial Feature Extraction):** Performs spatial feature learning on the low-dimensional channel representation.
3. **$1 \times 1$ Convolution (Dimensionality Restoration):** Restores channel depth back to the high-dimensional representation (e.g., from 64 to 256 channels).

```
                      Input x (e.g., 256 channels)
                          │                    │
                          ▼                    │
                  ┌──────────────┐             │
                  │ 1x1 Conv, 64 │             │
                  │ BatchNorm    │             │
                  │ ReLU         │             │
                  └──────┬───────┘             │  Identity Shortcut
                         ▼                     │  (or 1x1 Projection
                  ┌──────────────┐             │   Shortcut W_s * x)
                  │ 3x3 Conv, 64 │             │
                  │ BatchNorm    │             │
                  │ ReLU         │             │
                  └──────┬───────┘             │
                         ▼                     │
                  ┌──────────────┐             │
                  │1x1 Conv, 256 │             │
                  │ BatchNorm    │             │
                  └──────┬───────┘             │
                         ▼                     │
                         └──────────►(+)◄──────┘
                                      │
                                      ▼
                                     ReLU
                                      │
                                    Output
```

When input and output spatial dimensions match, an **identity shortcut** ($x$) is utilized with zero parameter overhead. When spatial resolution is downsampled (stride = 2) or channel dimensions expand, a **projection shortcut** with a $1 \times 1$ convolution and Batch Normalization is applied:

$$y = \mathcal{F}(x, \{W_i\}) + W_s x$$

where $W_s$ represents a trainable linear projection matrix matching channel dimensions.

---

### 6.2.3 Full Network Topology & Layer Hierarchy
ResNet-50 comprises 5 architectural stages with a total of 50 parameterized layers:

| Stage | Output Dimension | Layers / Operations | Repetitions |
| :---: | :---: | :--- | :---: |
| **Input** | $224 \times 224 \times 3$ | RGB Image Input | — |
| **Stage 1 (Stem)** | $112 \times 112 \times 64$ | $7 \times 7$ Conv, stride 2, BatchNorm, ReLU | 1 |
| | $56 \times 56 \times 64$ | $3 \times 3$ Max Pooling, stride 2 | 1 |
| **Stage 2** | $56 \times 56 \times 256$ | $\begin{bmatrix} 1 \times 1, 64 \\ 3 \times 3, 64 \\ 1 \times 1, 256 \end{bmatrix}$ Bottleneck Blocks | $3\times$ |
| **Stage 3** | $28 \times 28 \times 512$ | $\begin{bmatrix} 1 \times 1, 128 \\ 3 \times 3, 128 \\ 1 \times 1, 512 \end{bmatrix}$ Bottleneck Blocks | $4\times$ |
| **Stage 4** | $14 \times 14 \times 1024$ | $\begin{bmatrix} 1 \times 1, 256 \\ 3 \times 3, 256 \\ 1 \times 1, 1024 \end{bmatrix}$ Bottleneck Blocks | $6\times$ |
| **Stage 5** | $7 \times 7 \times 2048$ | $\begin{bmatrix} 1 \times 1, 512 \\ 3 \times 3, 512 \\ 1 \times 1, 2048 \end{bmatrix}$ Bottleneck Blocks | $3\times$ |
| **Classification Head** | $2048$ | GlobalAveragePooling2D, BatchNorm, Dropout(0.3) | 1 |
| **Output** | $101$ | Dense(101, activation='softmax') | 1 |

---

### 6.2.4 Custom Classification Head Design
In our implementation (`src/models/resnet50.py`), the original 1000-class ImageNet fully connected layer was removed and replaced by a domain-adapted classification head:

1. **Global Average Pooling (GAP):**  
   Instead of flattening the final tensor ($7 \times 7 \times 2048 = 100,352$ parameters), GAP computes the spatial mean across each feature map:
   $$z_c = \frac{1}{H \times W} \sum_{i=1}^{H} \sum_{j=1}^{W} X_{i,j,c}$$
   This reduces the dimensionality from $(7, 7, 2048)$ directly to a $2048$-dimensional vector. As demonstrated by Lin et al. (2013), GAP enforces structural spatial invariance and eliminates over $100\text{ million}$ potential weights, acting as a powerful native regularizer against overfitting.
2. **Post-Pooling Batch Normalization:**  
   Standardizes the $2048$-dimensional representation across mini-batches, centering inputs to the final linear classifier.
3. **Dropout Regularization ($p = 0.3$):**  
   Randomly deactivates $30\%$ of activations during training passes, preventing co-adaptation of features across food texture representations.
4. **Softmax Output Layer:**  
   $101$ units with normalized exponential activation yielding calibrated posterior probabilities:
   $$P(y = k \mid x) = \frac{e^{z_k}}{\sum_{j=1}^{101} e^{z_j}}$$

---

### 6.2.5 Two-Stage Transfer Learning & Selective Fine-Tuning Protocol
Training ResNet-50 from random initialization on 68,175 food images risks catastrophic overfitting given the model's 23.8M parameters. We adopted a principled two-stage transfer learning protocol:

#### Phase 1: Feature Extraction (Frozen Backbone)
- **Configuration:** The entire ResNet-50 backbone was frozen (`base_model.trainable = False`), locking 23,587,712 parameters.
- **Trainable Units:** Only the classification head ($215,141$ parameters) was trained.
- **Optimizer:** Adam with standard initial learning rate $\alpha = 10^{-3}$, $\beta_1 = 0.9, \beta_2 = 0.999$.
- **Purpose:** Adapt the random classification head weights to the pretrained feature representations without disrupting learned convolutional filters.

#### Phase 2: Selective Fine-Tuning (Unfrozen Stage 5)
- **Configuration:** Backbone layers were selectively unfrozen from `conv5_block1_out` onwards ($3$ bottleneck residual blocks in Stage 5), exposing $9,130,085$ trainable parameters.
- **Preservation of Early Stages:** Stages 1 through 4 remained frozen to safeguard universal low-level visual primitives (Gabor-like edge detectors, color gradients, and primitive geometric textures).
- **Batch Normalization Invariance:** All `BatchNormalization` layers within the backbone remained frozen (`layer.trainable = False`). In fine-tuning with batch sizes of 32, computing mini-batch statistics corrupts the pretrained running mean $\mu$ and running variance $\sigma^2$, leading to rapid feature drift. Freezing BatchNorm ensures the network acts as a linear affine scaling operation:
  $$\hat{x} = \frac{x - \mu_{\text{pretrained}}}{\sqrt{\sigma^2_{\text{pretrained}} + \epsilon}} \cdot \gamma + \beta$$
- **Optimizer Schedule:** Adam with a conservative learning rate $\alpha = 10^{-5}$ (reduced to $2 \times 10^{-6}$ at Epoch 18 via `ReduceLROnPlateau`), preventing large gradient updates from destroying pretrained representations.
