# Section 5: Experimental Design and Setup

**Authors:** Kaveesha (Member 4) & Matheesha Weerakoon (Member 2)  
**Assigned Workstream:** Controlled Experimental Protocol, Hardware Profiling, Optimization Framework  
**Academic Module:** SE4050 — Deep Learning (2026)  
**Institution:** Sri Lanka Institute of Information Technology (SLIIT)  

---

## 5.1 Overview and Scientific Objectives of the Experimental Protocol

To perform a scientifically valid comparative evaluation across four distinct Convolutional Neural Network (CNN) architectures—a scratch-trained Custom CNN, ResNet-50, MobileNetV2, and EfficientNetB0—it is essential to isolate architectural inductive biases from confounding experimental variables. Variations in input resolution, mini-batch dimensions, stochastic data augmentations, optimizer hyperparameter configurations, learning rate schedules, and data partitioning can significantly alter empirical convergence behavior and final classification metrics [11].

Consequently, this benchmark implements a strictly standardized experimental protocol governed by the following core objectives:

1. **Strict Factor Isolation:** All four models are trained and evaluated under identical input resolutions ($224 \times 224 \times 3$), uniform mini-batch sizes ($B = 32$), locked pseudo-random number seeds ($\text{seed} = 42$), and identical train, validation, and test data splits.
2. **Zero-Leakage Generalization Assessment:** The held-out test partition consisting of 25,250 unperturbed images is quarantined during all phases of model exploration, hyperparameter tuning, and early-stopping decisions. It is evaluated exactly once per model using the final optimized checkpoint weights.
3. **Hardware-Constrained Feasibility:** Training pipelines and memory allocations are calibrated specifically for single-GPU Google Colab Tesla T4 hardware environments, ensuring zero Out-Of-Memory (OOM) runtime terminations while maximizing pipeline throughput via asynchronous prefetching.
4. **Principled Transfer Learning Regimes:** For the three pretrained models, a disciplined two-phase training protocol (Feature Extraction followed by targeted Fine-Tuning) is deployed with learning rate decay callbacks to preserve generic low-level feature extractors while adapting high-level representations to food domain semantics.

---

## 5.2 Controlled Experimental Protocol and Fair Comparison Rules

Table 5.1 delineates the standardized control parameters enforced uniformly across all four architectural implementations.

### Table 5.1: Standardized Experimental Control Parameters

| Experimental Parameter | Standardized Specification | Justification & Methodological Control |
| :--- | :--- | :--- |
| **Input Spatial Resolution** | $224 \times 224 \times 3$ (RGB) | Standard canonical input dimension for ImageNet backbones; eliminates resolution-induced accuracy disparities. |
| **Mini-Batch Size ($B$)** | 32 samples per batch | Balances stochastic gradient variance with GPU memory limits on 16 GB Tesla T4 hardware. |
| **Global Random Seed** | `42` | Enforced across Python `random`, NumPy `np.random`, and TensorFlow `tf.random` for full reproducibility. |
| **Dataset Splits** | Train: 68,175 (67.5%)<br>Val: 7,575 (7.5%)<br>Test: 25,250 (25.0%) | Deterministic class-balanced partitioning (675 train / 75 val / 250 test per class) with pre-shuffled manifests. |
| **Data Augmentation Policy** | RandomFlip, Rotation ($\pm 5\%$), Zoom ($\pm 10\%$) | Applied strictly on the training partition; validation and test splits evaluate strictly in canonical, unperturbed state. |
| **Loss Function** | Sparse Categorical Cross-Entropy | Direct optimization over integer class indices $\mathcal{Y} \in \{0, \dots, 100\}$ without one-hot memory overhead. |
| **Primary Optimizer** | Adam ($\beta_1 = 0.9, \beta_2 = 0.999, \epsilon = 10^{-7}$) | Adaptive moment estimation providing rapid convergence across sparse and non-convex gradient surfaces [7]. |
| **Evaluation Cadence** | End of every epoch | Validation loss and accuracy evaluated over the full 7,575 validation split to guide learning rate schedules. |
| **Test Split Isolation** | Evaluated exactly once | 25,250 test images evaluated only after final model weights are finalized, guaranteeing zero validation leakage. |

### 5.2.1 Standardized Input Tensor Formulation
All raw Food-101 JPEG images vary from $193 \times 286$ to $512 \times 512$ pixels (mean aspect ratio $\approx 1.06$, as detailed in Section 3). In accordance with the controlled protocol, raw images are dynamically resized to:

$$X \in \mathbb{R}^{B \times 224 \times 224 \times 3}$$

using bilinear interpolation. Bilinear interpolation computes output pixel intensities via a distance-weighted average of the four nearest pixel neighbors in the original coordinate grid, preserving fine-grained edges and ingredient textures while maintaining computational efficiency during GPU streaming.

### 5.2.2 Architecture-Specific Normalization Pipelines
While input spatial geometry is strictly identical, each architecture requires a specific numerical input distribution corresponding to its original pretraining objective [10]. Applying mismatched normalization undermines pretrained convolutional weights and causes severe convergence instability. The preprocessing pipeline (`src/preprocessing/data_loader.py`) implements four mutually isolated scaling transformations:

1. **Custom CNN (Linear Rescaling):**
   Trained from scratch with randomly initialized weights (Glorot Uniform). Pixel values are linearly scaled from integer byte ranges $[0, 255]$ into the continuous interval $[0.0, 1.0]$:
   
   $$x_{\text{custom}} = \frac{x}{255.0}$$

2. **ResNet-50 (Caffe-Style Zero-Centered BGR Normalization):**
   ResNet-50 was originally trained on ImageNet using Caffe, which converts RGB channels to BGR and subtracts the empirical ImageNet channel means without scaling:
   
   $$x_{\text{resnet}} = \text{BGR}(x) - \mu_{\text{ImageNet}}, \quad \text{where } \mu_{\text{ImageNet}} = [103.939, 116.779, 123.680]$$

3. **MobileNetV2 (Inverted Residual Min-Max Normalization):**
   MobileNetV2 utilizes inverted residual bottlenecks with bounded linear activations, designed for input tensors mapped to the symmetric dynamic range $[-1.0, 1.0]$:
   
   $$x_{\text{mobilenet}} = \frac{x}{127.5} - 1.0$$

4. **EfficientNetB0 (Native Pass-Through Normalization):**
   EfficientNet architectures incorporate internal normalization layers within their model graph (`Normalization` or `Rescaling` built directly into the Keras functional model definition). The input tensor is passed through in the range $[0.0, 255.0]$:
   
   $$x_{\text{efficientnet}} = x$$

### 5.2.3 Training-Only Stochastic Data Augmentation
To mitigate overfitting on the 68,175 training samples while addressing the extensive intra-class variation of food presentation, stochastic geometric transformations are applied exclusively during training passes. The augmentation pipeline is instantiated via Keras preprocessing layers:

$$\mathcal{T}_{\text{aug}}(x) = \text{RandomZoom}(0.10) \circ \text{RandomRotation}(0.05) \circ \text{RandomFlip}(\text{"horizontal"})(x)$$

- **Horizontal Reflection (`RandomFlip("horizontal")`):** Simulates viewpoint invariance across left-right dining orientations. Vertical flips are deliberately omitted because food is presented against gravitational surfaces; inverted vertical dishes do not reflect natural dining photography.
- **Subtle Rotation (`RandomRotation(0.05)`):** Applies random rotational perturbations within $\pm 18^\circ$ ($\pm 5\%$ of $360^\circ$). This accounts for handheld smartphone tilt during meal capture without introducing severe corner clipping or black padding artifacts.
- **Subtle Zoom (`RandomZoom(0.10)`):** Applies random zoom variations within $\pm 10\%$, encouraging scale-invariant feature extraction across varying camera-to-plate distances.

Crucially, the validation ($N = 7,575$) and test ($N = 25,250$) pipelines omit $\mathcal{T}_{\text{aug}}$ entirely. They execute deterministic bilinear resizing and architecture-specific normalization, guaranteeing uncorrupted and reproducible benchmark measurements.

---

## 5.3 Computational Environment and Hardware Feasibility Benchmarks

All model development, training runs, and inference profiling were executed within the Google Colaboratory cloud computing infrastructure.

### 5.3.1 Hardware and Software Infrastructure
The computational host specifications utilized throughout this research comprise:

- **Graphics Processing Unit (GPU):** NVIDIA Tesla T4 (Turing Architecture, TU104 core)
  - Dedicated VRAM: 15,360 MiB (15.0 GiB) GDDR6 with 256-bit memory bus
  - Theoretical Bandwidth: 320 GB/s
  - Hardware Acceleration: 2,560 CUDA Cores, 320 Turing Tensor Cores
  - Driver & Compute Capability: NVIDIA Driver 535.104.05, CUDA Compute Capability 7.5
- **Host Central Processing Unit (CPU):** Intel(R) Xeon(R) CPU @ 2.20GHz (2 Virtual Cores, 4 Threads)
- **System Memory (Host RAM):** 12.7 GB available system memory
- **Operating System & Environment:** Linux Ubuntu 22.04.3 LTS (x86_64 kernel 6.1.85+)
- **Deep Learning Framework:** TensorFlow 2.17.0 / Keras 3.4.1, cuDNN 8.9.7, Python 3.10.12

### 5.3.2 Pre-Training Feasibility and Memory Headroom Profiling
Prior to launching full-scale model training, an empirical feasibility profiling benchmark was executed (`notebooks/02_benchmark_dataloader.ipynb`) to verify pipeline throughput and memory stability under continuous GPU allocation.

```
================================================================================
COLAB T4 FEASIBILITY PROFILING RESULTS (Mini-Batch Size = 32)
================================================================================
Allocated GPU Memory (Weights + Activations):   4,217.5 MiB  (27.46% of Total)
Unallocated VRAM Safety Headroom:              11,142.5 MiB  (72.54% of Total)
Pipeline Throughput (Prefetched tf.data):         77.72 images / second
Time per Mini-Batch (Forward + Backward Pass):    411.7 milliseconds
Out-of-Memory (OOM) Exceptions Detected:          0 (Zero)
================================================================================
```

The empirical consumption of 4,217.5 MiB demonstrates that a batch size of $B = 32$ operates comfortably within the safe operational envelope of the Tesla T4 GPU, providing over 11 GiB of unallocated headroom. This buffer proved vital for accommodating the dynamic memory allocations required during Phase 2 fine-tuning when deep residual gradient graphs and optimizer momentum buffers expand across all 50 layers.

### 5.3.3 High-Throughput Asynchronous Data Pipeline (`tf.data`)
Given the substantial volume of Food-101 (68,175 training images totaling ~5 GB on disk), standard sequential Python I/O creates severe CPU bottlenecks where the GPU idles while waiting for disk reads and JPEG decoding. To ensure full GPU saturation, our data loader (`src/preprocessing/data_loader.py`) implements a multi-threaded asynchronous pipeline utilizing the `tf.data` API:

```
  Disk (JPEG Files)
         │
         ▼  [ parallel_read / interleave: num_parallel_calls = AUTOTUNE ]
  Raw Byte Streams
         │
         ▼  [ tf.io.decode_jpeg & tf.image.resize (224, 224) ]
  Decoded Image Tensors
         │
         ▼  [ Pre-Shuffled Manifest + tf.data.Dataset.shuffle(buffer_size = 2048) ]
  Stochastic Mini-Batches (B = 32)
         │
         ▼  [ tf.data.Dataset.prefetch(buffer_size = tf.data.AUTOTUNE) ]
  GPU Memory Buffer (Zero Latency GPU Ingestion)
```

The pipeline enforces three architectural optimizations:
1. **Vectorized Bilinear Resizing:** Image decoding and spatial resizing are fused into parallel map operations executing across CPU threads (`num_parallel_calls = tf.data.AUTOTUNE`).
2. **Pre-Shuffled Index Manifests:** To resolve the severe non-i.i.d. streaming shuffle buffer failure discovered during initial trials (detailed in Section 8.2), dataset manifests are pre-shuffled at the Python filesystem level (`seed=42`) before being ingested into TensorFlow's 2,048-element streaming buffer.
3. **Double-Buffered Asynchronous Prefetching (`prefetch(AUTOTUNE)`):** While the GPU computes forward and backward passes on mini-batch $k$, the host CPU concurrently prepares, augments, and batches mini-batch $k+1$ in background host memory, eliminating GPU starvation.

---

## 5.4 Optimization Strategies and Two-Phase Transfer Learning Protocol

### 5.4.1 Loss Function Formulation
Because Food-101 is formulated as a single-label multi-class visual recognition problem across $K = 101$ mutually exclusive culinary categories, all networks are optimized using **Sparse Categorical Cross-Entropy**. For a mini-batch of $N$ samples, where $y_i \in \{0, 1, \dots, K-1\}$ represents the ground-truth class index and $\hat{y}_{i,k} \in [0, 1]$ represents the predicted posterior softmax probability for category $k$, the objective function is formulated as:

$$\mathcal{L}(\theta) = -\frac{1}{N} \sum_{i=1}^N \ln \hat{y}_{i, y_i} = -\frac{1}{N} \sum_{i=1}^N \sum_{k=0}^{K-1} \mathbb{I}(y_i = k) \ln \left( \frac{\exp(z_{i,k})}{\sum_{j=0}^{K-1} \exp(z_{i,j})} \right)$$

where $z_{i} \in \mathbb{R}^K$ denotes the unnormalized logit vector output by the final dense projection layer, and $\mathbb{I}(\cdot)$ is the indicator function. The sparse formulation is mathematically equivalent to standard categorical cross-entropy over one-hot target vectors, but eliminates the memory overhead of maintaining sparse 101-dimensional vectors for 68,175 samples.

### 5.4.2 Primary Optimizer Dynamics (Adam)
All models are optimized using the **Adam (Adaptive Moment Estimation)** algorithm [7]. Adam computes individual adaptive learning rates for different parameters from estimates of first and second raw moments of the gradients:

$$m_t = \beta_1 m_{t-1} + (1 - \beta_1) g_t$$

$$v_t = \beta_2 v_{t-1} + (1 - \beta_2) g_t^2$$

$$\hat{m}_t = \frac{m_t}{1 - \beta_1^t}, \quad \hat{v}_t = \frac{v_t}{1 - \beta_2^t}$$

$$\theta_t = \theta_{t-1} - \frac{\eta}{\sqrt{\hat{v}_t} + \epsilon} \hat{m}_t$$

Standard hyperparameter defaults are enforced across all experiments: first-moment decay $\beta_1 = 0.9$, second-moment decay $\beta_2 = 0.999$, and numerical stability constant $\epsilon = 10^{-7}$.

### 5.4.3 Two-Phase Transfer Learning Protocol
Training deep pretrained backbones directly on a new target domain with randomly initialized classification heads presents a major risk: large gradient backpropagation from an untrained head can corrupt pretrained convolutional feature extractors, a failure mode known as **catastrophic forgetting** [11].

To prevent this, ResNet-50, MobileNetV2, and EfficientNetB0 are trained via a disciplined **Two-Phase Transfer Learning Protocol**:

```
[ PHASE 1: WARMUP / FEATURE EXTRACTION ]
  - Base Pretrained Backbone: FROZEN (trainable = False)
  - Custom Classification Head: TRAINABLE (Dense 101 + Dropout + BatchNorm)
  - Learning Rate: eta_1 = 1e-3 (0.001)
  - Goal: Warm up classification head to food domain without corrupting base weights
  - Duration: 6 to 8 epochs until validation loss stabilizes

                          │
                          ▼  (Weights warm, gradients stable)

[ PHASE 2: TARGETED FINE-TUNING ]
  - Top Convolutional / Residual Blocks: UNFROZEN (trainable = True)
  - Early Feature Extraction Stages: FROZEN (preserve low-level edges/textures)
  - Batch Normalization Layers: FROZEN (keep pretrained running statistics)
  - Learning Rate: eta_2 = 1e-5 (0.00001)  [100x reduction]
  - Goal: Jointly adapt high-level semantic representations to culinary concepts
  - Duration: 10 to 14 epochs until convergence with EarlyStopping
```

#### Detailed Phase Breakdown:
1. **Phase 1 (Feature Extraction Warmup):**
   The convolutional base is instantiated with ImageNet weights and set to `base_model.trainable = False`. Only the newly appended top classification head (consisting of `GlobalAveragePooling2D`, optional `BatchNormalization`, `Dropout` ($p \in [0.2, 0.4]$), and a 101-unit `Dense(softmax)` layer) contains trainable parameters. The model is trained using an initial learning rate $\eta_1 = 10^{-3}$ ($0.001$) for 6 to 8 epochs. This allows the dense projection weights to align with the pre-extracted ImageNet feature space without disturbing the convolutional filter weights.

2. **Phase 2 (Targeted Fine-Tuning):**
   Following Phase 1 convergence, designated top architectural blocks are unfrozen to allow domain-specific feature adaptation:
   - **ResNet-50:** Stage 5 residual blocks unfrozen (`conv5_block1_out` onward), exposing 9,130,085 trainable parameters while keeping Stages 1–4 frozen.
   - **MobileNetV2:** Top inverted residual blocks unfrozen (layer 120 onward), exposing 1,737,445 trainable parameters.
   - **EfficientNetB0:** Top compound stages unfrozen (Block 6, Block 7, and top Conv), exposing 3,261,825 trainable parameters.
   - **Custom CNN:** Trained end-to-end from scratch across all 20 epochs with $\eta = 10^{-3}$ and adaptive decay, as it contains no pretrained weights.

   Crucially, during Phase 2, the learning rate is scaled down by a factor of 100 to $\eta_2 = 10^{-5}$ ($0.00001$). This minute step size guarantees that gradient updates make subtle adjustments to high-level filter weights without destabilizing learned feature representations. Furthermore, all `BatchNormalization` layers within the base models are explicitly maintained in non-trainable inference mode (`training=False`) during fine-tuning, preventing updates to ImageNet mean and variance tracking statistics.

### 5.4.4 Dynamic Regularization and Convergence Callbacks
To guarantee objective stopping criteria and prevent overfitting, each training session integrates three automated Keras callbacks:

1. **`EarlyStopping`:** Monitors validation loss (`val_loss`) with a patience of 5 epochs. If validation loss fails to achieve a relative improvement for 5 consecutive epochs, training terminates early. The parameter `restore_best_weights = True` automatically restores model parameters from the epoch with the lowest observed validation loss.
2. **`ReduceLROnPlateau`:** Dynamically monitors `val_loss`. If the validation loss plateaus for 2 consecutive epochs (`patience = 2`), the optimizer learning rate is decayed by a factor of $\gamma = 0.2$ (or 0.5):
   
   $$\eta_{t+1} = \eta_t \cdot \gamma, \quad \text{subject to } \eta_{t+1} \ge \eta_{\min} = 10^{-6}$$
   
   This enables the optimizer to escape saddle points and settle into narrower, flatter local minima.
3. **`ModelCheckpoint`:** Serializes the complete model state (`.keras` format) whenever validation loss achieves a new global minimum, ensuring that intermediate weight progress is preserved against cloud runtime disconnections.

---

## 5.5 Quantitative Evaluation Metrics and Mathematical Formulations

To provide a comprehensive, multi-dimensional assessment of model performance, our evaluation framework employs both classification performance metrics and computational efficiency metrics.

### 5.5.1 Top-1 and Top-5 Classification Accuracy
In visual recognition benchmarks with a large number of classes ($K = 101$), relying solely on Top-1 accuracy can obscure meaningful performance distinctions, particularly when candidate categories share subtle culinary similarities. Both Top-1 and Top-5 accuracy are formally computed over the test set:

- **Top-1 Accuracy:** Measures the proportion of test images for which the highest-probability prediction matches the ground-truth label:
  
  $$\text{Top-1 Accuracy} = \frac{1}{N_{\text{test}}} \sum_{i=1}^{N_{\text{test}}} \mathbb{I}\left( \arg\max_{k \in \{0, \dots, K-1\}} \hat{y}_{i,k} = y_i \right)$$

- **Top-5 Accuracy:** Measures the proportion of test images for which the true category is included within the model's top 5 most confident predictions:
  
  $$\text{Top-5 Accuracy} = \frac{1}{N_{\text{test}}} \sum_{i=1}^{N_{\text{test}}} \mathbb{I}\left( y_i \in \text{argtop}_5 (\hat{y}_{i}) \right)$$

Top-5 accuracy is particularly relevant for real-world mobile dietary logging systems, where an application presents a ranked list of five candidate dishes for user confirmation.

### 5.5.2 Multi-Class Precision, Recall, and F1-Score
To evaluate per-category performance across all $K = 101$ classes, class-specific confusion counts are accumulated: True Positives ($TP_k$), False Positives ($FP_k$), and False Negatives ($FN_k$).

- **Per-Class Precision ($\text{Precision}_k$):** The proportion of images predicted as class $k$ that actually belong to class $k$:
  
  $$\text{Precision}_k = \frac{TP_k}{TP_k + FP_k}$$

- **Per-Class Recall ($\text{Recall}_k$):** The proportion of ground-truth images of class $k$ that were correctly identified:
  
  $$\text{Recall}_k = \frac{TP_k}{TP_k + FN_k}$$

- **Per-Class F1-Score ($F1_k$):** The harmonic mean of precision and recall for class $k$:
  
  $$F1_k = 2 \cdot \frac{\text{Precision}_k \cdot \text{Recall}_k}{\text{Precision}_k + \text{Recall}_k} = \frac{2 \cdot TP_k}{2 \cdot TP_k + FP_k + FN_k}$$

### 5.5.3 Macro-Averaged Aggregation
Because Food-101's test partition contains exactly 250 images per class ($N_k = 250, \sum N_k = 25,250$), class distributions are balanced. To evaluate models without allowing high-performing classes to mask poor performance in difficult categories, we report **Macro-Averaged** metrics, which assign equal weight to each of the 101 categories:

$$\text{Macro Precision} = \frac{1}{K} \sum_{k=0}^{K-1} \text{Precision}_k$$

$$\text{Macro Recall} = \frac{1}{K} \sum_{k=0}^{K-1} \text{Recall}_k$$

$$\text{Macro F1-Score} = \frac{1}{K} \sum_{k=0}^{K-1} F1_k$$

### 5.5.4 Computational Complexity and Latency Profiling
To evaluate the viability of each architecture for practical deployment across cloud and edge platforms, four computational metrics are systematically profiled:

1. **Parameter Complexity ($P_{\text{total}}$ and $P_{\text{trainable}}$):** The total number of parameters and the subset of weights updated during Phase 2 fine-tuning.
2. **Model Storage Footprint ($\text{Size}_{\text{MB}}$):** The physical disk footprint of the serialized single-precision (Float32) weight tensor file in megabytes (MB) and mebibytes (MiB).
3. **Training Wall-Clock Duration ($T_{\text{train}}$):** Cumulative execution time in seconds and hours required to complete both Phase 1 and Phase 2 training on the Colab Tesla T4 GPU.
4. **Inference Latency ($L_{\text{inf}}$):** Average wall-clock inference duration per image measured in milliseconds (ms). Latency profiling is conducted on the Tesla T4 GPU across 1,000 unseen test samples using a standardized warmup protocol (50 warmup iterations) to eliminate initial CUDA kernel initialization overhead:
   
   $$L_{\text{inf}} = \frac{1}{M} \sum_{m=1}^M \left( t_{\text{end}}^{(m)} - t_{\text{start}}^{(m)} \right) \times 1000 \quad [\text{ms/image}]$$

5. **Generalization Gap ($\Delta_{\text{gen}}$):** The absolute divergence between final training accuracy and validation accuracy:
   
   $$\Delta_{\text{gen}} = |\text{Accuracy}_{\text{train}} - \text{Accuracy}_{\text{val}}|$$
   
   A small generalization gap ($\Delta_{\text{gen}} < 5\%$) signifies balanced generalization, whereas large gaps ($\Delta_{\text{gen}} > 10\%$) indicate overfitting.

---

## 5.6 Summary of Experimental Setup Across All Four Models

Table 5.2 consolidates the final hyperparameter and training configurations implemented for each of the four benchmarked models.

### Table 5.2: Complete Architectural and Experimental Configuration Matrix

| Parameter / Configuration | Custom CNN (Baseline) | ResNet-50 | MobileNetV2 | EfficientNetB0 |
| :--- | :--- | :--- | :--- | :--- |
| **Model Category** | Plain 4-Stage CNN | Deep Residual Network | Inverted Residual CNN | Compound Scaled CNN |
| **Pretraining Source** | None (Random Init) | ImageNet-1k | ImageNet-1k | ImageNet-1k |
| **Input Shape** | $224 \times 224 \times 3$ | $224 \times 224 \times 3$ | $224 \times 224 \times 3$ | $224 \times 224 \times 3$ |
| **Batch Size ($B$)** | 32 | 32 | 32 | 32 |
| **Loss Function** | Sparse Categorical CE | Sparse Categorical CE | Sparse Categorical CE | Sparse Categorical CE |
| **Phase 1 Epochs** | — (Single Phase) | 6 epochs ($\eta = 10^{-3}$) | 6 epochs ($\eta = 10^{-3}$) | 8 epochs ($\eta = 10^{-3}$) |
| **Phase 2 Epochs** | 20 epochs ($\eta = 10^{-3}$) | 14 epochs ($\eta = 10^{-5}$) | 14 epochs ($\eta = 10^{-5}$) | 10 epochs ($\eta = 10^{-5}$) |
| **Total Completed Epochs** | 20 | 20 | 20 | 18 (Early Stopped) |
| **Unfrozen Fine-Tuning Depth** | All layers (scratch) | Stage 5 (`conv5_block1_out`) | Top blocks (Layer 120+) | Blocks 6, 7 & Top Conv |
| **Classification Head** | GAP $\to$ Drop(0.4) $\to$ Dense | GAP $\to$ BN $\to$ Drop(0.3) $\to$ Dense | GAP $\to$ Drop(0.2) $\to$ Dense | GAP $\to$ BN $\to$ Drop(0.3) $\to$ Dense |
| **Total Parameters** | 678,085 | 23,802,853 | 2,387,365 | 4,184,072 |
| **Trainable Parameters (P2)** | 675,653 | 9,130,085 | 1,737,445 | 3,261,825 |
| **Float32 Weights Size** | 2.59 MiB | 90.80 MiB | 9.11 MiB | 15.96 MiB |
| **Preprocessing Scaling** | $x / 255.0 \to [0, 1]$ | Caffe BGR Mean Subtraction | $(x / 127.5) - 1 \to [-1, 1]$ | Pass-Through ($[0, 255]$) |
| **Target Hardware** | Colab Tesla T4 GPU | Colab Tesla T4 GPU | Colab Tesla T4 GPU | Colab Tesla T4 GPU |
