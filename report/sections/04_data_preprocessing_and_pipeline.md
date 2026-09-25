# Section 4: Data Preprocessing and Feature Engineering

**Author:** Nirmana (Group Leader / Member 1)  
**Assigned Workstream:** Repository Scaffolding, Data Pipeline, EDA, Custom CNN, Integration  
**Academic Module:** SE4050 — Deep Learning (2026)  
**Institution:** Sri Lanka Institute of Information Technology (SLIIT)  

---

## 4.1 Overview and Methodological Rationale

In deep learning for visual recognition, the data preparation pipeline is not merely an auxiliary processing step; it directly dictates the mathematical validity of the experimental results. Convolutional neural networks require input tensors of identical spatial dimensionality, consistent numerical dynamic ranges, and strict statistical isolation between training, optimization, and evaluation phases.

The preprocessing framework engineered for this project was designed around three central objectives:
1. **Mathematical Isolation (Zero Data Leakage):** Eliminating all potential pathways through which information from the test partition could influence data scaling, augmentation policies, or hyperparameter selection.
2. **Architectural Compatibility:** Accommodating the heterogeneous input specifications of four distinct backbones (Custom CNN, ResNet-50, MobileNetV2, and EfficientNetB0) without forcing suboptimal, one-size-fits-all normalization.
3. **Computational Scalability:** Streaming 101,000 images (~5.0 GB) through asynchronous, non-blocking GPU memory buffers via TensorFlow’s `tf.data` API, avoiding RAM exhaustion across varied hardware environments.

---

## 4.2 Dataset Partitioning and Prevention of Data Leakage

> **SLIIT Rubric Focus (10%):** *Correctly separates training, validation, and test data and demonstrates effective prevention of data leakage.*

Data leakage occurs when information from outside the training dataset is inadvertently utilized to train or configure a machine learning model. In computer vision, leakage commonly manifests in two forms:
- **Partition Overlap:** Shuffling an entire dataset globally prior to splitting, which can place near-duplicate burst shots of the same meal into both training and test sets.
- **Optimization Leakage:** Computing dataset-wide statistics (e.g., mean and standard deviation) across the pooled dataset, or using test set performance to inform early stopping and learning rate schedules.

### 4.2.1 Canonical Partitioning Protocol
To guarantee strict statistical isolation, we preserved the original partitioning established by the dataset creators (Bossard et al. [1]). The official 25,250 test images were quarantined as an untouched evaluation pool. The remaining 75,750 training images were partitioned into training ($\mathcal{D}_{\text{train}}$) and validation ($\mathcal{D}_{\text{val}}$) subsets using a stratified 90/10 ratio.

The partitioning was implemented deterministically in `src/data/split_food101.py` with a fixed pseudo-random seed $S = 42$:

$$\mathcal{D}_{\text{total}} = \mathcal{D}_{\text{train}} \cup \mathcal{D}_{\text{val}} \cup \mathcal{D}_{\text{test}}$$

where:

$$|\mathcal{D}_{\text{train}}| = 68,175 \quad (675 \text{ images/class})$$
$$|\mathcal{D}_{\text{val}}| = 7,575 \quad (75 \text{ images/class})$$
$$|\mathcal{D}_{\text{test}}| = 25,250 \quad (250 \text{ images/class})$$

$$\mathcal{D}_{\text{train}} \cap \mathcal{D}_{\text{val}} = \emptyset, \quad \mathcal{D}_{\text{train}} \cap \mathcal{D}_{\text{test}} = \emptyset, \quad \mathcal{D}_{\text{val}} \cap \mathcal{D}_{\text{test}} = \emptyset$$

### Table 4.1: Dataset Partitioning Protocol and Leakage Isolation Guarantees

| Dataset Partition | Sample Count | Images / Class | Operational Role in Study | Leakage Isolation Mechanism |
| :--- | :---: | :---: | :--- | :--- |
| **Training ($\mathcal{D}_{\text{train}}$)** | 68,175 | 675 | Parameter optimization via backpropagation | Stochastic data augmentations active; isolated from val/test |
| **Validation ($\mathcal{D}_{\text{val}}$)** | 7,575 | 75 | Hyperparameter tuning, learning rate decay, Early Stopping | Deterministic preprocessing; zero backpropagation |
| **Final Test ($\mathcal{D}_{\text{test}}$)** | 25,250 | 250 | Single-pass comparative benchmark | **Strictly Unseen;** evaluated only once after freezing final weights |

The resulting file paths were committed as lightweight plain-text manifests (`data/splits/train.txt`, `val.txt`, `test.txt`). This guarantees that all four models across all four student workstations ingest the identical image instances, ensuring fair experimental comparison.

---

## 4.3 Spatial Standardization and Resizing Strategy

As established during Exploratory Data Analysis (Section 3.3), raw Food-101 photographs exhibit variable resolutions ranging from $193 \times 286$ to $512 \times 512$ pixels with varying aspect ratios. 

### 4.3.1 Bilinear Interpolation to $224 \times 224$
To feed deep convolutional backbones, every input image is mapped to a standardized tensor dimension:

$$x \in \mathbb{R}^{224 \times 224 \times 3}$$

Resizing is performed dynamically on decoded JPEG byte streams using **Bilinear Interpolation**:

$$I(x, y) = \sum_{i=1}^2 \sum_{j=1}^2 w_{i,j} \cdot I(x_i, y_j)$$

where $w_{i,j}$ represents the bilinear distance weights calculated from the four nearest neighboring pixel coordinates.

### 4.3.2 Methodological Justification for $224 \times 224$
1. **Backbone Standard:** ResNet-50, MobileNetV2, and EfficientNetB0 were originally parameterized and pretrained on ImageNet at $224 \times 224$. Preserving this canonical dimension maintains the receptive field geometry of pretrained convolutional kernels.
2. **Computational Tractability:** Operating at higher resolutions (e.g., $384 \times 384$ or $512 \times 512$) quadruples the spatial activation memory in early layers ($H \times W \times C$), causing Out-of-Memory (OOM) failures under cloud T4 GPU constraints (16GB VRAM) when training with batch size 32.
3. **Aspect Ratio Preservation:** Our EDA demonstrated that 82% of Food-101 images possess aspect ratios between 1.0 and 1.33. Direct bilinear resizing introduces negligible visual distortion while eliminating the computational overhead of zero-padding or complex bounding-box cropping.

---

## 4.4 Data Augmentation Policy

> **SLIIT Rubric Focus (10%):** *Applies appropriate and well-justified augmentation techniques with effective prevention of data leakage.*

Data augmentation artificially expands the training distribution, forcing the network to learn invariant feature detectors rather than memorizing spatial orientations. However, naive augmentation can alter the semantic identity of the object (e.g., flipping text digits turns a '6' into a '9'). In food recognition, transformations must strictly respect the physical reality of culinary presentation.

### 4.4.1 Domain-Justified Augmentation Transformations
The augmentation pipeline was implemented in `src/preprocessing/augmentations.py` using TensorFlow Keras layers:

1. **Random Horizontal Flip (`RandomFlip("horizontal")`):**
   - *Justification:* Plated food exhibits bilateral reflectional invariance. A bowl of ramen, a pizza, or a plate of dumplings viewed from the left or right retains identical semantic identity.
   - *Exclusion of Vertical Flips:* Vertical flipping was deliberately omitted. Food is governed by gravity; plates rest on horizontal surfaces. An upside-down bowl of soup or inverted cake constitutes an unrealistic visual anomaly that degrades feature learning.
2. **Subtle Random Rotation (`RandomRotation(factor=0.05)`):**
   - *Justification:* Applies continuous rotations bounded within $\theta \in [-18^\circ, +18^\circ]$ ($0.05 \times 360^\circ$). This simulates natural angular shifts when users photograph a table from different sitting positions, without clipping dish edges.
3. **Subtle Random Zoom (`RandomZoom(height_factor=(-0.10, 0.10))`):**
   - *Justification:* Applies scaling between 90% and 110%, simulating variations in camera-to-plate distance without cropping out discriminative ingredients.

### 4.4.2 Strict Enforcement of Training-Only Execution
To uphold mathematical rigor, data augmentation is executed **exclusively during the training phase**:

$$\tilde{x} = \begin{cases} 
\mathcal{T}_{\text{aug}}(x), & \text{if } \text{is\_training} = \text{True} \\
x, & \text{if } \text{is\_training} = \text{False} 
\end{cases}$$

Validation ($\mathcal{D}_{\text{val}}$) and Test ($\mathcal{D}_{\text{test}}$) sets are processed with strictly deterministic transforms. Introducing stochastic variations into validation data introduces variance into loss measurements, destabilizing Early Stopping and learning rate scheduling.

---

## 4.5 Model-Specific Normalization Framework

A pervasive error in multi-model deep learning benchmarks is applying a uniform scaling rule (e.g., dividing pixel values by 255.0) across all networks. Pretrained models possess fixed inductive biases learned from ImageNet, and their weights depend heavily on the specific color space and numerical range utilized during pretraining.

To ensure fair experimental conditions without compromising pretrained weight integrity, our data loader (`src/preprocessing/data_loader.py`) implements a model-specific normalization dispatcher:

### Table 4.2: Model-Specific Mathematical Normalization Strategies

| Architecture | Input Value Range | Color Channel Ordering | Normalization Equation / Implementation | Technical Justification |
| :--- | :---: | :---: | :--- | :--- |
| **Custom CNN** | $[0.0, 1.0]$ | RGB | $x_{\text{norm}} = \frac{x}{255.0}$ | Standard min-max normalization for scratch-trained models; stabilizes initial weight gradients. |
| **ResNet-50** | $[-123.68, 151.06]$ | **BGR** | $x_{\text{norm}} = x_{\text{BGR}} - \mu_{\text{Caffe}}$<br>where $\mu = [103.939, 116.779, 123.68]$ | Matches original Caffe implementation [2]. Zero-centers pixel values around ImageNet mean; reverses channels to BGR. |
| **MobileNetV2** | $[-1.0, 1.0]$ | RGB | $x_{\text{norm}} = \frac{x}{127.5} - 1.0$ | Scales data into $[-1, 1]$, centered at zero. Optimal for lightweight networks with linear bottlenecks and ReLU6. |
| **EfficientNetB0**| $[0.0, 255.0]$ | RGB | Native Pass-Through<br>(`efficientnet.preprocess_input`) | EfficientNet includes an internal `Rescaling` layer in its computational graph; external division corrupts weights. |

---

## 4.6 Asynchronous Input Pipeline Architecture (`tf.data`)

Loading 101,000 uncompressed JPEG images sequentially from disk during training introduces severe I/O bottlenecks, leaving the GPU idle between mini-batches. To maximize hardware utilization, our pipeline in `src/preprocessing/data_loader.py` leverages TensorFlow's asynchronous `tf.data` API.

### Figure 4.1: End-to-End Preprocessing Pipeline Flowchart

```
 [ Disk Manifest: train.txt ] (e.g., 'pizza/1005649')
              │
              ▼
   [ tf.io.read_file ]        (Parallel binary disk read)
              │
              ▼
   [ tf.image.decode_jpeg ]   (Decompresses raw bytes to RGB tensor)
              │
              ▼
   [ tf.image.resize ]        (Bilinear interpolation to 224 x 224)
              │
              ▼
   [ Model-Specific Scaler ]  (Custom [0,1] | ResNet Caffe BGR | MobileNet [-1,1])
              │
              ▼
   [ tf.data.Dataset.batch ]  (Mini-batch aggregation: Batch Size = 32)
              │
              ▼
   [ Training Augmentation ]  (RandomFlip, Rotation, Zoom applied on batch)
              │
              ▼
   [ .prefetch(AUTOTUNE) ]    (Asynchronous background prefetching into GPU VRAM)
              │
              ▼
       [ GPU Model.fit ]      (Zero I/O starvation)
```

*Figure 4.1: Flow diagram of the high-throughput streaming data pipeline, illustrating thread-parallel parsing, deterministic resizing, backbone-specific scaling, train-only augmentation, and memory-safe prefetching.*

### 4.6.1 Pipeline Optimization Primitives:
1. **Parallel Interleaved Mapping (`num_parallel_calls=tf.data.AUTOTUNE`):** Image loading, decoding, and bilinear resizing are distributed across all available CPU threads asynchronously.
2. **Stochastic Shuffling (`shuffle_buffer=2048`):** Prevents class-correlated gradient spikes by drawing samples randomly from an active buffer of 2,048 image paths before batching.
3. **Double-Buffered Memory Prefetching (`prefetch(buffer_size=tf.data.AUTOTUNE)`):** While the GPU processes batch $N$ on its tensor cores, the CPU concurrently prepares and augments batch $N+1$ in host memory, virtually eliminating pipeline latency.

---

## 4.7 Empirical Validation: GPU Memory & Throughput Profiling

Before authorizing all four team members to launch full-scale model training, the completed pipeline was empirically benchmarked in `notebooks/02_gpu_benchmark.ipynb` using Google Colab's NVIDIA T4 GPU (16 GB VRAM).

### Table 4.3: Empirical Pipeline Feasibility Profiling Results

| Metric | Measured Value | Operational Interpretation |
| :--- | :---: | :--- |
| **Host Accelerator** | NVIDIA T4 GPU (16 GB GDDR6) | Standard cloud execution environment |
| **Selected Mini-Batch Size** | 32 | Optimal compromise between gradient variance and memory cost |
| **Spatial Input Shape** | $224 \times 224 \times 3$ | Standardized input across all 4 architectures |
| **Peak VRAM Consumed** | **4,217 MB / 15,360 MB (27.5%)** | **Safe:** Leaves >11 GB buffer; eliminates Out-of-Memory (OOM) risks |
| **Pipeline Throughput** | **77.7 images / second** | High data throughput; zero GPU starvation |
| **Estimated Epoch Duration** | ~18.7 minutes / epoch | Tractable training cycles under Colab session time limits |

The profiling confirmed that a batch size of 32 consumes only **27.5%** of available GPU memory on the heaviest candidate model (ResNet-50), officially locking `batch_size: 32` and `image_size: [224, 224]` as the standardized parameters for all subsequent experiments.

---

## References for Section 4

```text
[1] L. Bossard, M. Guillaumin, and L. Van Gool, "Food-101—mining discriminative components with random forests," in European Conference on Computer Vision (ECCV). Springer, 2014, pp. 446-461.
[2] K. He, X. Zhang, S. Ren, and J. Sun, "Deep residual learning for image recognition," in IEEE CVPR, 2016, pp. 770-778.
[3] M. Sandler, A. Howard, M. Zhu, A. Zhmoginov, and L.-C. Chen, "MobileNetV2: Inverted residuals and linear bottlenecks," in IEEE CVPR, 2018, pp. 4510-4520.
[4] M. Tan and Q. V. Le, "EfficientNet: Rethinking model scaling for convolutional neural networks," in ICML, 2019, pp. 6105-6114.
```
