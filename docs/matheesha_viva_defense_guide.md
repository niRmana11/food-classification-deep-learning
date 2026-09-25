# SE4050 Deep Learning — Viva Voce Defense Guide (20% Marks)

**Candidate:** Matheesha Weerakoon (Member 2)  
**Assigned Architecture:** ResNet-50 (Transfer Learning & Selective Fine-Tuning)  
**Dataset:** Food-101 (101 classes, 101,000 images)  
**Final Test Performance:** **73.54% Top-1 Accuracy**, **92.02% Top-5 Accuracy**, **10.98 ms Latency**  

---

## 🎓 Master Question & Answer Defense Reference

### Q1: What is the "Degradation Problem" and why does ResNet solve it?
**Examiner Focus:** Understanding why we can't just keep adding convolutional layers.  
**Your Answer:**  
> *"In plain deep networks (like deep VGG), as depth exceeds 20–30 layers, accuracy saturates and then degrades rapidly on both training and test sets. This is not caused by overfitting (since training error also rises) or vanishing gradients (which are mitigated by BatchNorm). Rather, it is an optimization degradation problem: deeper stacks of non-linear layers struggle to learn identity mappings.  
> ResNet solves this by reformulating the mapping into $\mathcal{H}(x) = \mathcal{F}(x) + x$. By adding the identity skip connection $x$, the network only needs to learn the residual mapping $\mathcal{F}(x) = \mathcal{H}(x) - x$. If an identity mapping is optimal, driving the residual weights toward zero is much easier for backpropagation than learning identity across non-linear activations."*

---

### Q2: What is the mathematical formulation of backpropagation in a residual connection?
**Examiner Focus:** Mathematical understanding of gradient flow (Lecture 2 & 4).  
**Your Answer:**  
> *"Given the output of a residual unit $y = \mathcal{F}(x, \{W_i\}) + x$, the gradient of the loss $\mathcal{E}$ with respect to the input $x$ is:
> $$\frac{\partial \mathcal{E}}{\partial x} = \frac{\partial \mathcal{E}}{\partial y} \cdot \left( \frac{\partial \mathcal{F}}{\partial x} + 1 \right)$$
> The critical insight is the $+1$ term. Even if the weight gradient $\frac{\partial \mathcal{F}}{\partial x}$ becomes arbitrarily small (approaching zero), the gradient $\frac{\partial \mathcal{E}}{\partial y} \times 1$ flows directly back to earlier layers without diminishing. This provides an uninterrupted gradient highway back to the earliest convolution layers."*

---

### Q3: What is the difference between an Identity Shortcut and a Projection Shortcut?
**Examiner Focus:** Layer dimensionality mechanics.  
**Your Answer:**  
> *"When the input $x$ and residual output $\mathcal{F}(x)$ have the exact same spatial dimensions $(H, W)$ and channel depth $C$, we use an **Identity Shortcut**: $y = \mathcal{F}(x) + x$, which introduces zero additional parameters and zero FLOP overhead.  
> However, when downsampling occurs (stride $= 2$) or when channel dimensions increase (e.g., from 64 to 256), the dimensions of $\mathcal{F}(x)$ and $x$ do not match for element-wise addition. In that case, we apply a **Projection Shortcut**: $y = \mathcal{F}(x) + W_s x$, where $W_s$ is implemented as a $1\times 1$ convolution with stride 2 and Batch Normalization to linearly project $x$ into the matching dimension."*

---

### Q4: Why does ResNet-50 use a 3-layer "Bottleneck Block" instead of the 2-layer basic block in ResNet-18/34?
**Examiner Focus:** Computational efficiency & $1\times 1$ convolutions.  
**Your Answer:**  
> *"In deeper architectures like ResNet-50, stacking two $3\times 3$ convolutions at high channel counts (e.g. 512 channels) is computationally prohibitive.  
> ResNet-50 uses a 3-layer Bottleneck Block:
> 1. A **$1\times 1$ convolution** to compress channel dimensionality by a factor of 4 (e.g. from 256 down to 64 channels).
> 2. A **$3\times 3$ convolution** that performs spatial feature extraction on the low-dimensional 64-channel space.
> 3. A second **$1\times 1$ convolution** to restore dimensionality back up to 256 channels.  
> This bottleneck design reduces floating-point operations (FLOPs) and parameter counts dramatically, allowing 50 layers to have fewer parameters than a 34-layer network using only $3\times 3$ blocks."*

---

### Q5: Why did you use Global Average Pooling (GAP) instead of a Flatten layer?
**Examiner Focus:** Classification head design and regularization (Lecture 5).  
**Your Answer:**  
> *"The output of Stage 5 in ResNet-50 is a $(7 \times 7 \times 2048)$ tensor.  
> If we used `Flatten()`, it would create a $100,352$-dimensional vector. Connecting that to our 101 classes would require over **10.1 million weights** in a single Dense layer, creating a massive parameter bottleneck prone to severe overfitting.  
> Instead, `GlobalAveragePooling2D()` computes the spatial average across each $7 \times 7$ feature slice, compressing the tensor directly to a $2048$-dimensional vector with **zero parameters**. As shown by Lin et al. (Network in Network), GAP enforces spatial translation invariance and acts as a structural regularizer."*

---

### Q6: Why did you keep Batch Normalization layers FROZEN during Phase 2 fine-tuning?
**Examiner Focus:** Advanced fine-tuning subtleties.  
**Your Answer:**  
> *"During fine-tuning with a mini-batch size of 32, computing mini-batch mean and variance inside the pretrained backbone causes noisy statistics that can corrupt the well-calibrated running statistics learned on 1.2 million ImageNet images.  
> In `src/models/resnet50.py`, when unfreezing Stage 5, we explicitly check:
> ```python
> if isinstance(layer, layers.BatchNormalization):
>     layer.trainable = False
> ```
> This keeps BatchNorm operating in inference mode using pretrained $\mu$ and $\sigma^2$ as fixed linear affine transformations, which stabilizes gradient updates and prevents catastrophic feature drift."*

---

### Q7: Why unfreeze only Stage 5 (`conv5_block1_out`) instead of the entire network?
**Examiner Focus:** Transfer learning theory (hierarchical feature representation).  
**Your Answer:**  
> *"Deep CNNs learn hierarchical visual representations:
> - **Stages 1 & 2:** Generic low-level visual primitives (edges, color blobs, simple corners).
> - **Stages 3 & 4:** Mid-level motifs (textures, boundary curves, repetitive patterns).
> - **Stage 5:** High-level semantic parts and task-specific object representations.  
> Low-level edge and texture detectors are universal across both ImageNet and Food-101. Unfreezing them would risk overfitting and destroying generalized filters. By unfreezing only Stage 5 (9.1M trainable parameters out of 23.8M), we adapt high-level visual food concepts (e.g. food plating, surface glazes, grill marks) while preserving foundational low-level filters."*

---

### Q8: Explain the Non-i.i.d. Shuffle Bug you discovered and how you resolved it.
**Examiner Focus:** Real-world problem-solving & data engineering.  
**Your Answer:**  
> *"In our initial runs, validation accuracy stalled at ~20%. Diagnosing the pipeline revealed that `data/splits/train.txt` was sorted sequentially by category (750 images of apple pie, followed by 750 images of ribs, etc.).  
> Our in-memory shuffle buffer was 2,048. Since $2048 / 675 \approx 3$, any batch of 32 images was drawn from only 2 to 3 adjacent classes. The gradients were severely non-i.i.d., causing catastrophic sequential forgetting: as the model learned ribs, it forgot apple pie.  
> I fixed this in `data_loader.py` by deterministically pre-shuffling the sample manifest in Python with `seed=42` before creating the TensorFlow dataset. This guaranteed that every 32-sample batch contained samples from ~28–32 distinct classes. Immediately, validation accuracy jumped from 20% to 61.31% in Phase 1 and reached 73.54% on the test set."*

---

### Q9: Why was Validation Accuracy higher than Training Accuracy in Phase 1?
**Examiner Focus:** Regularization and evaluation protocol.  
**Your Answer:**  
> *"This occurs because our training and evaluation pipelines treat regularization differently:
> 1. **Dropout (p=0.3):** Active only during training (30% of signals dropped). Inactive during validation, giving the validation pass full network capacity.
> 2. **Data Augmentation:** Random rotations, flips, and zooms are applied exclusively to training images, making the training distribution deliberately harder than the canonical validation distribution."*

---

### Q10: Why was your Test Accuracy (73.54%) higher than your Validation Accuracy (69.47%)?
**Examiner Focus:** Data split integrity & test evaluation protocol.  
**Your Answer:**  
> *"Three reasons:
> 1. **Test Set Scale:** The test split is substantially larger (25,250 images) than the validation split (7,575 images), providing lower variance and tighter confidence bounds.
> 2. **Evaluation with Fully Restored Best Weights:** At the end of Phase 2, `ReduceLROnPlateau` decayed $\text{lr}$ to $2\times 10^{-6}$ at Epoch 18, and `restore_best_weights=True` restored the optimal weight checkpoint before test evaluation.
> 3. **No Dropout or Augmentations:** The test evaluation runs in deterministic inference mode with zero dropout noise, allowing full exploitation of the fine-tuned Stage 5 features."*

---

### Q11: Why Sparse Categorical Cross-Entropy instead of Categorical Cross-Entropy?
**Examiner Focus:** Loss function selection & memory efficiency.  
**Your Answer:**  
> *"Both compute the exact same mathematical loss:
> $$\mathcal{L} = -\sum_{k=1}^{C} y_k \log(\hat{y}_k)$$
> However, `CategoricalCrossentropy` requires targets to be one-hot encoded vectors of shape $(N, 101)$. For 68,175 training images, one-hot matrices waste GPU memory on zeros. `SparseCategoricalCrossentropy` accepts integer class labels $\{0, 1, \dots, 100\}$, reducing memory footprint and allowing integer indexing directly inside the GPU CUDA kernel."*

---

### Q12: What does your Confusion Matrix tell you about ResNet-50's weaknesses?
**Examiner Focus:** Critical error analysis (30% rubric).  
**Your Answer:**  
> *"From our normalized $101\times 101$ confusion matrix:
> - **Top Strengths:** Highly structured items like `edamame` (98.2% F1) and `macarons` (94.0% F1) are near-perfect due to distinctive geometric shapes and unique color gamuts.
> - **Top Confusion Clusters:** The lowest F1 scores are on cooked meat cuts—`steak` (42.7% F1) and `pork_chop` (47.8% F1). At $224\times 224$ resolution, both present as brown seared protein with grill marks, often served with identical sides like mashed potatoes. The fine muscle fiber differences are beyond the resolution threshold. Similarly, `apple_pie` (49.3% F1) is confused with `bread_pudding` because the filling is occluded by the golden-brown crust."*

---

### Q13: How did you profile inference latency and is ResNet-50 viable for production?
**Examiner Focus:** Deployment engineering.  
**Your Answer:**  
> *"We profiled latency on an NVIDIA T4 GPU across 1,024 test images using `time.perf_counter()` after warming up GPU memory caches. ResNet-50 averaged **10.98 ms per image**, corresponding to a throughput of **~91 frames per second**.  
> This proves ResNet-50 is highly viable for **cloud-hosted REST APIs** (e.g. food delivery or calorie logging backends). However, for mobile on-device deployment where thermal throttling and battery consumption are constraints, its 98 MB size and 3.8 GFLOPs motivate using MobileNetV2."*
