# Section 2: Background and Related Work

**Authors:** Matheesha Weerakoon (Member 2) & Kanushka (Member 3)  
**Academic Module:** SE4050 — Deep Learning (2026)  
**Institution:** Sri Lanka Institute of Information Technology (SLIIT)  

---

## 2.1 Visual Food Classification and Dietary Healthcare

Automated visual food recognition has emerged as a distinct, highly active subfield of computer vision. Unlike standard object categorization benchmarks (e.g., PASCAL VOC, ImageNet) which focus on rigid manufactured objects (vehicles, tools) or well-defined biological organisms (mammals, birds), food items represent non-rigid, highly deformable entities characterized by high intra-class variance and low inter-class separability [1].

Early computational approaches to food recognition relied predominantly on handcrafted visual feature extractors. Researchers extracted Scale-Invariant Feature Transform (SIFT) descriptors, Speeded-Up Robust Features (SURF), Local Binary Patterns (LBP), Color Histograms, and Histograms of Oriented Gradients (HOG) to quantify visual texture, color distribution, and edge orientations. These low-level features were subsequently aggregated using Bag-of-Visual-Words (BoVW) or Fisher Vectors and classified using Support Vector Machines (SVMs) or Random Forests [1].

In their seminal benchmark paper introducing the Food-101 dataset, Bossard, Guillaumin, and Van Gool (2014) employed Random Forests to mine discriminative visual components, achieving a top-1 classification accuracy of **50.76%** on clean test imagery [1]. Their experiments highlighted the severe limitations of handcrafted features: static filter banks cannot adapt to the complex spatial compositions, liquid glazes, melting textures, and varied culinary presentations inherent to food imagery.

The advent of modern deep learning and Convolutional Neural Networks (CNNs) fundamentally altered this landscape. Deep CNNs learn hierarchical representations directly from raw pixels through gradient descent, replacing handcrafted heuristics with end-to-end differentiable feature extractors [10]. Subsequent studies demonstrated that deep convolutional models could surpass 70% top-1 accuracy on Food-101, validating the hypothesis that multi-layered non-linear feature hierarchies are essential for capturing fine-grained culinary abstractions [5], [11].

---

## 2.2 Evolution of Deep Convolutional Architectures

The architectural evolution of CNNs over the past decade reflects a continuous progression toward resolving the fundamental trade-off between representational capacity, gradient stability, and computational efficiency.

### 2.2.1 Deep Feedforward Networks and Plain CNNs
The success of AlexNet in the 2012 ImageNet challenge demonstrated that deep convolutional networks could be trained effectively using GPU acceleration, Rectified Linear Units (ReLU), and Dropout [10]. Simonyan and Zisserman (2014) introduced the VGG architecture, establishing that replacing large convolutional kernels (e.g., $7 \times 7$ or $11 \times 11$) with stacks of small $3 \times 3$ kernels substantially increases non-linear decision capacity while reducing total parameter count [12]. Two consecutive $3 \times 3$ convolutions possess an effective receptive field of $5 \times 5$, while three consecutive $3 \times 3$ convolutions cover $7 \times 7$, but require $3 \times (3^2 \cdot C^2) = 27C^2$ parameters compared to $49C^2$ for a single $7 \times 7$ layer.

However, standard deep feedforward architectures encounter severe optimization barriers as depth increases. In our benchmark, the **Custom CNN** serves as an experimental baseline representing this plain convolutional paradigm. By utilizing five consecutive convolutional blocks with batch normalization, max-pooling, and global average pooling, it establishes the empirical ceiling achievable from random weight initialization without pretrained inductive priors.

### 2.2.2 Multi-Branch Convolutions (Inception)
Szegedy et al. (2015) introduced the Inception (GoogLeNet) architecture, which addressed the challenge of variable scale in visual objects by processing feature maps through multiple parallel kernel sizes ($1 \times 1$, $3 \times 3$, $5 \times 5$) within the same layer [15]. By incorporating $1 \times 1$ convolutions as dimensionality reduction bottlenecks before compute-heavy spatial convolutions, Inception demonstrated that structural efficiency could reduce parameter complexity compared to standard dense layers.

### 2.2.3 Deep Residual Learning (ResNet)
As networks grew beyond 20 layers, researchers observed the **degradation problem**: beyond a certain threshold, network accuracy saturates and then degrades rapidly on both training and test data [2]. He et al. (2016) demonstrated that this degradation is not caused by overfitting or vanishing gradients, but rather by the mathematical difficulty of optimizing deep stacks of non-linear transformations to learn simple identity functions [2].

To resolve this, He et al. introduced the **Residual Learning Framework**. Instead of requiring stacked layers to directly fit an underlying mapping $\mathcal{H}(x)$, the network is explicitly parameterized to learn the residual mapping:

$$\mathcal{F}(x) = \mathcal{H}(x) - x$$

The original function is recovered via feedforward skip (shortcut) connections:

$$\mathcal{H}(x) = \mathcal{F}(x, \{W_i\}) + x$$

The residual formulation provides an unobstructed gradient highway:

$$\frac{\partial \mathcal{E}}{\partial x} = \frac{\partial \mathcal{E}}{\partial \mathcal{H}} \left( \frac{\partial \mathcal{F}}{\partial x} + 1 \right)$$

Because the backpropagated gradient always retains the $+1$ additive term, gradients can propagate directly across dozens of layers without vanishing, allowing architectures such as **ResNet-50** to be optimized reliably [2], [3]. In ResNet-50, residual learning is implemented via 3-layer **Bottleneck Blocks** ($1 \times 1 \to 3 \times 3 \to 1 \times 1$ convolutions), reducing computational complexity while expanding channel depth to 2,048 dimensions [2].

### 2.2.4 Inverted Residuals and Mobile Architectures (MobileNetV2)
While ResNet-50 resolved gradient degradation, its 23.8M parameters and ~3.8 GFLOPs per forward pass present substantial computational barriers for battery-powered mobile devices and embedded edge hardware. 

Howard et al. (2017) pioneered **Depthwise Separable Convolutions** in MobileNetV1, factorizing standard convolution into:
1. A **Depthwise Convolution** that applies a single $3 \times 3$ filter per input channel (spatial filtering).
2. A **Pointwise ($1 \times 1$) Convolution** that linearly combines the outputs across channels (channel mixing) [16].

This factorization reduces computational cost by a factor of:

$$\text{Reduction} = \frac{D_K \cdot D_K \cdot M \cdot D_F \cdot D_F + M \cdot N \cdot D_F \cdot D_F}{D_K \cdot D_K \cdot M \cdot N \cdot D_F \cdot D_F} = \frac{1}{N} + \frac{1}{D_K^2}$$

For standard $3 \times 3$ filters ($D_K = 3$), depthwise separable convolution achieves an 8- to 9-fold reduction in computation with only a minor reduction in accuracy [16].

Sandler et al. (2018) extended this paradigm in **MobileNetV2** by introducing the **Inverted Residual Block with Linear Bottleneck** [4]. Contrary to standard ResNet residual blocks (which compress channels at the input and expand at the output), MobileNetV2:
1. Expands low-dimensional input features to a higher-dimensional manifold using a $1 \times 1$ expansion convolution (expansion factor $t = 6$).
2. Applies a $3 \times 3$ depthwise convolution in the expanded space.
3. Projects back to a low-dimensional representation using a $1 \times 1$ linear bottleneck convolution without non-linear activation [4].

Sandler et al. demonstrated that non-linear activation functions (such as ReLU) destroy information when applied to low-dimensional manifolds; maintaining a linear bottleneck preserves representational richness while keeping parameter footprints small (~2.39M parameters) [4].

### 2.2.5 Compound Scaling and Neural Architecture Search (EfficientNet)
Historically, scaling CNNs for higher accuracy was conducted along isolated, arbitrary dimensions:
- Increasing depth (e.g., ResNet-50 to ResNet-152) [2].
- Increasing channel width (e.g., WideResNet) [5].
- Increasing input image resolution (e.g., $224 \times 224$ to $448 \times 448$).

Tan and Le (2019) demonstrated that scaling these three dimensions independently yields diminishing returns. They formulated the **Compound Scaling Method**, which uses a fixed compound coefficient $\phi$ to scale depth ($d$), width ($w$), and resolution ($r$) simultaneously in a principled ratio:

$$\text{depth: } d = \alpha^\phi, \quad \text{width: } w = \beta^\phi, \quad \text{resolution: } r = \gamma^\phi$$

$$\text{subject to: } \alpha \cdot \beta^2 \cdot \gamma^2 \approx 2, \quad \alpha \ge 1, \beta \ge 1, \gamma \ge 1$$

where $\alpha, \beta, \gamma$ are constant coefficients determined through a localized grid search [5]. Because network FLOPs scale proportional to $d$, $w^2$, and $r^2$, doubling the compute budget ($\phi \to \phi + 1$) doubles total FLOPs predictably.

Applying this compound scaling to an automated Neural Architecture Search (NAS) baseline produced the **EfficientNet** family [5]. The baseline model, **EfficientNetB0**, leverages Mobile Inverted Bottleneck Convolutions (MBConv) augmented with **Squeeze-and-Excitation (SE)** channel attention, enabling dynamic channel-wise feature recalibration at minimal parameter cost (~4.18M parameters) [5].

---

## 2.3 Inductive Transfer Learning in Computer Vision

### 2.3.1 Theoretical Foundations of Transfer Learning
Formally, transfer learning involves a source domain $\mathcal{D}_S$ and source task $\mathcal{T}_S$, and a target domain $\mathcal{D}_T$ and target task $\mathcal{T}_T$ [11]. A domain is defined by a feature space $\mathcal{X}$ and marginal probability distribution $P(X)$, such that $\mathcal{D} = \{\mathcal{X}, P(X)\}$. A task consists of a label space $\mathcal{Y}$ and an objective predictive function $f(\cdot)$, such that $\mathcal{T} = \{\mathcal{Y}, f(\cdot)\}$.

In our study:
- **Source Domain & Task ($\mathcal{D}_S, \mathcal{T}_S$):** ImageNet Large Scale Visual Recognition Challenge (ILSVRC) comprising $1.28 \times 10^6$ natural images across $K_S = 1,000$ object categories [10].
- **Target Domain & Task ($\mathcal{D}_T, \mathcal{T}_T$):** Food-101 comprising $1.01 \times 10^5$ culinary photographs across $K_T = 101$ fine-grained food classes [1].

Transfer learning aims to improve the learning of the target predictive function $f_T(\cdot)$ using the knowledge acquired from $\mathcal{D}_S$ and $\mathcal{T}_S$, where $\mathcal{D}_S \ne \mathcal{D}_T$ and $\mathcal{T}_S \ne \mathcal{T}_T$ [11].

### 2.3.2 Hierarchical Feature Transfer and Representation Generalization
Deep CNNs trained on ImageNet learn a hierarchical visual decomposition:
1. **Initial Layers (Stages 1–2):** Extract universal visual primitives—Gabor-like directional edge filters, color transitions, frequency gradients, and localized corner detectors. These representations are domain-agnostic and transfer across diverse visual tasks.
2. **Intermediate Layers (Stages 3–4):** Form compound textural motifs, surface curvature detectors, repetitive boundary groupings, and component geometries.
3. **Deepest Layers (Stage 5 / Final Blocks):** Encode class-specific semantic assemblies and task-dependent object configurations [11].

When transferring to fine-grained culinary recognition, the lowest-level edge and color representations are directly applicable. However, high-level features must adapt from rigid object geometries (e.g., cars, dogs) to fluid, plated culinary textures (e.g., noodle broths, grill marks, caramel glazes) [1].

### 2.3.3 Two-Stage Transfer Learning Protocol
To transfer representations without corrupting learned weights, contemporary deep learning employs a two-stage protocol:
1. **Phase 1: Feature Extraction:** The pretrained convolutional backbone is frozen (`trainable = False`). Only the newly initialized classification head is trained with a standard learning rate ($\alpha \approx 10^{-3}$). This aligns the randomly initialized classification weights with the existing feature space without causing gradient shocks to the backbone.
2. **Phase 2: Fine-Tuning:** The top convolutional stages of the backbone are selectively unfrozen and trained end-to-end using a substantially reduced learning rate ($\alpha \approx 10^{-5}$). This allows high-level feature filters to specialize in target domain textures while preserving generalized mid-level representations.

---

## 2.4 Modern Regularization and Structural Innovations

Training deep networks on complex multi-class datasets requires modern architectural regularizers to ensure stability and generalization:

### 2.4.1 Global Average Pooling (GAP)
In early CNNs (AlexNet, VGG), spatial feature maps from the final convolutional layer were flattened into a 1D vector and passed to dense fully connected layers [12]. In VGG-16, this dense transition accounted for over 100 million weights ($7 \times 7 \times 512 \to 4096 = 102.7\text{M}$ weights), creating a severe parameter bottleneck prone to overfitting.

Lin, Chen, and Yan (2014) introduced **Global Average Pooling** in Network In Network [6]. GAP replaces dense flattening by computing the spatial average of each feature map directly:

$$z_c = \frac{1}{H \times W} \sum_{i=1}^{H} \sum_{j=1}^{W} X_{i,j,c}$$

where $X \in \mathbb{R}^{H \times W \times C}$ is the final convolutional activation tensor. GAP provides three decisive advantages:
1. It eliminates parameter overhead at the convolutional-dense interface, drastically reducing total model size.
2. It enforces structural correspondence between feature maps and category logits.
3. It imparts spatial translation invariance, operating as a structural regularizer against overfitting [6].

All four architectures in our benchmark utilize Global Average Pooling before final classification.

### 2.4.2 Batch Normalization
Ioffe and Szegedy (2015) introduced **Batch Normalization (BatchNorm)** to accelerate deep network training by stabilizing internal activation distributions [8]. For a mini-batch $\mathcal{B} = \{x_1, \dots, x_m\}$, BatchNorm normalizes each dimension across the batch:

$$\mu_{\mathcal{B}} = \frac{1}{m} \sum_{i=1}^m x_i, \quad \sigma_{\mathcal{B}}^2 = \frac{1}{m} \sum_{i=1}^m (x_i - \mu_{\mathcal{B}})^2$$

$$\hat{x}_i = \frac{x_i - \mu_{\mathcal{B}}}{\sqrt{\sigma_{\mathcal{B}}^2 + \epsilon}}$$

$$y_i = \gamma \hat{x}_i + \beta$$

where $\gamma$ and $\beta$ are learnable scale and shift parameters [8]. BatchNorm smooths the optimization landscape, permits substantially higher learning rates, and reduces sensitivity to weight initialization.

### 2.4.3 Dropout Regularization
Srivastava et al. (2014) formulated **Dropout** as an effective technique for mitigating complex co-adaptations between neurons [9]. During each training iteration, individual activations are independently zeroed with probability $p$:

$$r_j \sim \text{Bernoulli}(1 - p)$$

$$\tilde{y} = r \odot y$$

During evaluation, Dropout is deactivated, and weights are scaled by $1 - p$. This approximates geometric model averaging across $2^N$ sub-networks, ensuring individual neurons learn robust features that remain predictive even when other activations are absent [9]. In our benchmark, dropout rates between $0.2$ and $0.4$ are systematically applied across the classification heads of all four candidate models.
