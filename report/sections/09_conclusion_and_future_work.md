# Section 9: Conclusion and Future Work

**Authors:** Nirmana (Group Leader / Member 1), Matheesha Weerakoon (Member 2), Kanushka (Member 3), Kaveesha (Member 4)  
**Academic Module:** SE4050 — Deep Learning (2026)  
**Institution:** Sri Lanka Institute of Information Technology (SLIIT)  

---

## 9.1 Executive Summary and Synthesis of Research Findings

This research project presented a comprehensive empirical benchmark of four distinct Convolutional Neural Network (CNN) architectures on the challenging **Food-101** visual recognition dataset (101,000 images across 101 fine-grained culinary categories). Visual food classification is characterized by high intra-class variance, low inter-class separability, non-rigid deformable geometries, and natural web label noise. By enforcing a strictly standardized experimental protocol—uniform $224 \times 224 \times 3$ resolution, fixed mini-batch size of 32, locked pseudo-random seed of 42, and an uncorrupted held-out test split of 25,250 images—the benchmark isolated architectural design principles from confounding experimental factors.

The quantitative evaluation established a definitive performance hierarchy across the 25,250 test images:

1. **EfficientNetB0 (Benchmark Winner):** Attained the highest classification performance across all evaluated metrics: **$77.26\%$ Top-1 Accuracy**, **$94.16\%$ Top-5 Accuracy**, a Macro F1-Score of **$0.7718$**, and a minimal test loss of **$0.8091$**. Its compound scaling methodology (jointly optimizing network depth, width, and resolution) allowed it to surpass ResNet-50 by **$+3.72\%$** while utilizing only **$17.6\%$ of ResNet-50's parameter budget** (4.18M vs. 23.80M parameters) and an exceptionally compact **$15.96\text{ MiB}$** weight footprint.
2. **ResNet-50 (Deep Residual Baseline):** Achieved **$73.54\%$ Top-1 Accuracy** and **$92.02\%$ Top-5 Accuracy**. Its deep 50-layer bottleneck residual architecture demonstrated strong feature representation capacity, but exhibited the largest storage requirement (**$90.80\text{ MiB}$**), the highest computational training cost (16,481 seconds), and the largest generalization gap ($\Delta_{\text{gen}} = 13.06\text{ pp}$), indicating diminishing returns from expanding channel depth to 2,048 dimensions via dense convolutions.
3. **MobileNetV2 (Lightweight Edge Baseline):** Achieved **$68.69\%$ Top-1 Accuracy** and **$90.15\%$ Top-5 Accuracy** with only **$2,387,365$ parameters** and a **$9.11\text{ MiB}$** storage footprint. Crossing the $90\%$ Top-5 threshold confirms that depthwise separable convolutions and inverted residual bottlenecks provide sufficient discriminative power for interactive mobile food-logging applications.
4. **Custom CNN (Scratch-Trained Baseline):** Attained **$61.69\%$ Top-1 Accuracy** and **$80.90\%$ Top-5 Accuracy** from random initialization using only 678,085 parameters and a $2.59\text{ MiB}$ footprint. Outperforming the original Food-101 Random Forest baseline ($50.76\%$ [1]) by **$+10.93\%$** demonstrates the effectiveness of our 4-stage convolutional design with Global Average Pooling.

Crucially, the experimental findings provided conclusive empirical validation for all four core research hypotheses:
- **Hypothesis 1 (Transfer Learning Superiority):** Fully confirmed. Pretrained ImageNet representations outperformed the scratch-trained Custom CNN by $+7.00\%$ to $+15.57\%$ Top-1 accuracy, with immediate first-epoch convergence ($>40\%$).
- **Hypothesis 2 (Edge Efficiency — MobileNetV2):** Fully confirmed. MobileNetV2 delivered $28.77\%$ Top-1 accuracy per million parameters, outperforming ResNet-50's parameter efficiency by a factor of 9.3.
- **Hypothesis 3 (Compound Scaling — EfficientNetB0):** Fully confirmed. EfficientNetB0 dominated the empirical Pareto frontier, establishing that balanced scaling of depth, width, and resolution is mathematically superior to increasing depth alone.
- **Hypothesis 4 (Structured Error Clustering):** Fully confirmed. Cross-model F1 analysis demonstrated that classification errors were heavily concentrated within identical semantic confusion clusters (`steak` vs. `pork_chop`, `apple_pie` vs. `bread_pudding`, `ravioli` vs. `dumplings`), proving that error distributions are dictated by inherent visual ambiguity rather than individual model architectures.

---

## 9.2 Practical Architectural Selection Framework (Engineering Decision Matrix)

The empirical findings from this benchmark provide a practical engineering decision framework for selecting deep learning architectures based on deployment constraints, hardware limitations, and target application requirements.

### Table 9.1: Architectural Selection and Deployment Decision Matrix

| Deployment Tier & Environment | Primary Operational Constraints | Recommended Architecture | Test Top-1 / Top-5 | Storage & Memory Footprint | Architectural Justification |
| :--- | :--- | :---: | :---: | :---: | :--- |
| **Tier 1: Cloud-Based Dietary Assessment & Clinical APIs** | Maximize classification accuracy; batch throughput; cloud GPU servers available. | **EfficientNetB0** | **77.26%** / **94.16%** | 4.18M params<br>15.96 MiB weights | Benchmark winner. Delivers the highest Top-1 and Top-5 accuracy with minimal test loss ($0.8091$), low generalization gap ($1.47\text{ pp}$), and modest cloud memory requirements. |
| **Tier 2: Mobile & Edge On-Device Food Logging (iOS / Android)** | Strict memory budget ($< 15\text{ MB}$); low battery consumption; cellular app downloads. | **MobileNetV2** | **68.69%** / **90.15%** | 2.39M params<br>9.11 MiB weights | Ultra-compact weight footprint ($9.11\text{ MiB}$) easily complies with app-store limits. Top-5 accuracy exceeding $90\%$ is well-suited for interactive 5-item recommendation prompts. |
| **Tier 3: Enterprise GPU Datacenters with TensorRT** | High-concurrency batch serving; highly optimized GEMM acceleration; legacy support. | **ResNet-50** | **73.54%** / **92.02%** | 23.80M params<br>90.80 MiB weights | Standard dense $3 \times 3$ convolutions map directly onto cuDNN Tensor Core routines, achieving rapid batched GPU throughput ($10.98\text{ ms/img}$) and broad enterprise toolchain compatibility. |
| **Tier 4: Ultra-Constrained Embedded IoT / Smart Canteens** | Microcontrollers or edge sensors (RAM $< 4\text{ MB}$); no external pretraining dependencies. | **Custom CNN** | **61.69%** / **80.90%** | 0.68M params<br>2.59 MiB weights | Extremely compact parameter budget ($678\text{K}$) and minimal disk footprint ($2.59\text{ MiB}$); operates with the lowest forward-pass latency ($5.90\text{ ms}$) without pretraining license constraints. |

---

## 9.3 Critical Research Limitations

While the benchmark established rigorous comparative insights, several technical and domain limitations must be acknowledged:

### 1. The Single-Label Multi-Class Assumption
The Food-101 benchmark is formulated as a single-label multi-class classification problem, where every image is mapped to exactly one category $y \in \{0, \dots, 100\}$. In authentic dining environments, meal photographs frequently contain multiple food categories simultaneously (e.g., a plate containing a grilled pork chop accompanied by french fries, green beans, and dipping sauces). Forcing a neural network to output a single global label for multi-item plates introduces artificial label ambiguity and penalizes models that correctly detect secondary dish components.

### 2. Intrinsic Web-Crawled Label Noise
Food-101 was constructed from uncurated web images (*foodspotting.com*), and its training partition deliberately contains approximately **20% natural noise** [1]. Visual auditing confirmed instances of mislabeled dishes (e.g., bread pudding labeled as apple pie), extreme close-up crops showing only sauce glazes, non-food dining clutter, and artistic filters. While this tests model robustness under noisy real-world conditions, it establishes an artificial upper bound on achievable accuracy and contributes to the validation loss divergence observed during Phase 2 fine-tuning.

### 3. Spatial Resolution and Texture Downsampling
All images were downsampled to a standardized spatial resolution of $224 \times 224$ pixels. While standard for ImageNet backbones, this downsampling discards fine-grained textural cues—such as herb flakes, spice grains, sesame seeds, rice grain boundaries, and thin pastry layers—that are critical for differentiating closely related culinary dishes (e.g., distinguishing `steak` from `filet_mignon` or `dumplings` from `gyoza`).

### 4. Absence of Volumetric and Nutritional Integration
In clinical healthcare and automated dietary assessment, identifying the food category represents only the initial step. A complete dietary monitoring system must estimate portion size, 3D food volume, and ingredient composition to compute accurate caloric and macronutrient values. Standard 2D RGB classification lacks geometric depth information, making direct volumetric inference impossible without external reference markers or multi-view depth capture.

---

## 9.4 Future Work and Research Directions

Building upon the findings and limitations identified in this study, four promising avenues for future research are proposed:

### 1. Vision Transformers (ViTs) and Modern Hybrid Architectures
Recent advances in visual recognition have demonstrated that self-attention mechanisms in Vision Transformers (ViTs) and hierarchical architectures (such as Swin Transformer and ConvNeXt) can model long-range spatial dependencies more effectively than standard convolutions. Future research should evaluate whether window-based self-attention can better separate primary food items from complex dining table backgrounds, improving performance on amorphous categories such as `lasagna` and `pad_thai`.

### 2. Multi-Task Learning: Simultaneous Detection, Segmentation, and Volume Estimation
To overcome the single-label limitation, future systems should formulate food recognition as a multi-task problem combining **instance segmentation** (e.g., Mask R-CNN or YOLOv8/YOLOv11) with **monocular depth estimation**. Segmenting each discrete food component on a plate enables simultaneous multi-item classification and pixel-area calculation, which, when fused with depth maps, allows 3D volume reconstruction and automated caloric estimation.

### 3. Domain-Specific Self-Supervised Pretraining (SSL)
Current transfer learning paradigms rely on backbones pretrained on ImageNet-1k, which primarily contains generic manufactured objects and wildlife. Pretraining deep networks directly on large-scale, uncurated food repositories (e.g., Recipe1M+ or Instagram food imagery) using modern self-supervised learning algorithms (such as DINOv2 or SimCLR) would allow models to develop rich, domain-specific visual representations tailored to culinary textures and food presentation styles prior to supervised fine-tuning.

### 4. Edge Quantization and Neural Processing Unit (NPU) Compilation
To facilitate deployment on battery-powered mobile devices and embedded edge sensors, future work should explore Post-Training Quantization (PTQ) and Quantization-Aware Training (QAT). Converting Float32 model weights to 8-bit integers (INT8) via TensorFlow Lite and ONNX Runtime typically reduces memory consumption by **$75\%$** with minimal degradation in top-1 accuracy ($<1\%$). Furthermore, compiling quantized graphs for dedicated hardware accelerators (e.g., Apple Neural Engine, Google Coral Edge TPU) would resolve the CUDA kernel launch overhead observed in MobileNetV2, enabling real-time, on-device food recognition at under 10 ms per frame.

---

## 9.5 Concluding Remarks

The SE4050 multi-class food classification benchmark successfully executed an end-to-end deep learning engineering lifecycle. By establishing an automated data pipeline, verifying dataset integrity through rigorous exploratory analysis, enforcing strict experimental controls, implementing four diverse CNN architectures, and conducting granular cross-model critical evaluations, our group achieved all project milestones ahead of schedule.

The empirical findings clearly identify **EfficientNetB0** as the superior architecture for high-accuracy cloud-based dietary recognition, while **MobileNetV2** represents the optimal solution for memory-constrained edge deployment. The codebase, model weights, and standardized results artifacts have been fully integrated into the version-controlled repository, providing a fully reproducible benchmark that satisfies all academic requirements of the SLIIT SE4050 Deep Learning curriculum.
