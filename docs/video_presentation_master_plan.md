# SE4050 Deep Learning — Group Video Presentation Master Plan & Storyboard

**Academic Module:** SE4050 — Deep Learning (2026)  
**Institution:** Sri Lanka Institute of Information Technology (SLIIT)  
**Assessment Component:** Viva Voce Demonstration & Presentation Video (20% Total Module Marks)  
**Target Duration:** 10:00 – 12:00 Minutes (Strictly Enforced: ~2.5 to 3.0 Minutes per Member)  
**Target Delivery Format:** 1080p 60fps Screen Recording + Voiceover / Picture-in-Picture Webcam  
**Repository Anchor:** [`docs/video_presentation_master_plan.md`](file:///c:/Users/USER/OneDrive/Documents/Projects/food-classification-deep-learning/docs/video_presentation_master_plan.md)  

---

## 1. Executive Summary & Production Objectives

This document establishes the official storyboard, scene-by-scene script guidelines, visual asset mapping, and speaking responsibilities for the 10-minute group video submission. 

### Core Goals for Maximum Rubric Marks (20% Weight):
1. **Equal Team Engagement:** Each of the 4 members presents a distinct, rigorous technical segment of approximately 2.5 minutes with seamless verbal handoffs.
2. **End-to-End Scientific Storytelling:** Move logically from Problem Formulation $\to$ Pipeline & Baseline $\to$ Deep Residuals $\to$ Mobile Edge Viability $\to$ Compound Scaling & Master Cross-Model Pareto Frontier.
3. **Hard Empirical Evidence:** Show zero generic placeholder slides. Every claim is supported by actual repository assets: figures (`report/figures/fig1` to `fig9`), confusion matrices, loss curves, and verified benchmark tables.
4. **Code & Architecture Transparency:** Demonstrate real modular Python code (`src/`), configuration parameters (`config.yaml`), and reproducible Jupyter notebooks with execution outputs.

---

## 2. Member Responsibility & Time Budget Allocation

| Member | Assigned Role | Presentation Focus & Rubric Coverage | Allotted Time | Word Count Budget (~135-145 wpm) |
| :--- | :--- | :--- | :---: | :---: |
| **Member 1 (Nirmana)**<br>*Group Leader* | **System Architecture, EDA & Custom CNN** | • Project overview, real-world challenge of fine-grained food classification.<br>• Dataset provenance (Food-101) & EDA findings.<br>• Zero-leakage data pipeline & augmentation strategy.<br>• Custom CNN design from scratch (GAP vs. Flattening) & baseline test results. | **00:00 – 02:45**<br>*(2 min 45 sec)* | 370 – 400 words |
| **Member 2 (Matheesha)** | **Deep Residual Learning & Transfer Strategy (ResNet-50)** | • The degradation problem and residual identity shortcuts ($F(x) + x$).<br>• Two-phase transfer learning (frozen vs. Stage 5 fine-tuning).<br>• Algorithmic post-mortem: The non-i.i.d. shuffle buffer bug and mathematical fix.<br>• ResNet-50 test metrics, cloud server viability, and meat cluster error analysis. | **02:45 – 05:15**<br>*(2 min 30 sec)* | 340 – 370 words |
| **Member 3 (Kanushka)** | **Mobile Efficiency & Edge Viability (MobileNetV2)** | • Inverted residual blocks & linear bottlenecks (preserving manifold information).<br>• Depthwise separable convolutions (computation reduction: $\frac{1}{N} + \frac{1}{D_k^2}$).<br>• Edge deployment profiling: latency (6.84 ms) vs. memory (9.11 MB).<br>• MobileNetV2 fine-tuning dynamics and frozen BatchNorm justification. | **05:15 – 07:45**<br>*(2 min 30 sec)* | 340 – 370 words |
| **Member 4 (Kaveesha)** | **Compound Scaling (EfficientNetB0) & Comparative Evaluation** | • Compound scaling principle (balancing depth, width, resolution with $\phi$).<br>• Squeeze-and-Excitation channel attention mechanics.<br>• Cross-Model Master Benchmark: Fair comparison across all 4 architectures.<br>• Pareto efficiency frontier, hardest classes analysis, and final production recommendations. | **07:45 – 10:30**<br>*(2 min 45 sec)* | 370 – 400 words |

---

## 3. Master Slide-by-Slide Storyboard & Scene Plan

```mermaid
timeline
    title 10-Minute Video Presentation Timeline
    section Member 1 (Nirmana)
        00:00 - 00:30 : Title & Problem Statement
        00:30 - 01:15 : Food-101 Dataset & EDA
        01:15 - 02:00 : Pipeline & Augmentations
        02:00 - 02:45 : Custom CNN & Baseline
    section Member 2 (Matheesha)
        02:45 - 03:30 : ResNet-50 Theory & Bottlenecks
        03:30 - 04:15 : Non-i.i.d. Shuffle Bug Fix
        04:15 - 05:15 : 2-Phase Tuning & Results
    section Member 3 (Kanushka)
        05:15 - 06:00 : MobileNetV2 Inverted Residuals
        06:00 - 06:45 : Depthwise Separable Math
        06:45 - 07:45 : Latency & Edge Viability
    section Member 4 (Kaveesha)
        07:45 - 08:30 : EfficientNetB0 Compound Scaling
        08:30 - 09:30 : Cross-Model Benchmark & Pareto Frontier
        09:30 - 10:30 : Hardest Classes & Conclusion
```

---

### Scene 1: Introduction, Problem Formulation & Industrial Motivation
- **Presenter:** Member 1 (Nirmana)
- **Time:** `00:00 – 00:30` (30 seconds)
- **Visual Assets on Screen:**
  - Slide 1: Group Title Slide (Project Title, SE4050, SLIIT logo, Group Member names and Student IDs).
  - Quick cut / callout: High intra-class variance vs. low inter-class variance diagram (e.g., comparing different presentations of pizza vs. comparing steak to pork chop).
- **Key Talking Points:**
  - Formal greeting, introduction of the group and individual members.
  - Definition of the research problem: Supervised multi-class visual recognition of food imagery on the challenging Food-101 benchmark.
  - Industrial significance: Dietary intake tracking, automated calorie estimation, automated checkout in smart restaurants, and clinical diabetes management.
  - Core technical challenge: High intra-class variance (presentation, cooking style), extreme inter-class visual similarity (amorphous textures), and $\sim 20\%$ natural web-crawled label noise.

---

### Scene 2: Dataset Exploration (Food-101) & EDA Insights
- **Presenter:** Member 1 (Nirmana)
- **Time:** `00:30 – 01:15` (45 seconds)
- **Visual Assets on Screen:**
  - Slide 2: Split screen showing:
    - Left: [`report/figures/fig1_class_distribution.png`](file:///c:/Users/USER/OneDrive/Documents/Projects/food-classification-deep-learning/report/figures/fig1_class_distribution.png) (101 classes, strictly balanced 1,000 images per class).
    - Top-Right: [`report/figures/fig2_image_dimensions.png`](file:///c:/Users/USER/OneDrive/Documents/Projects/food-classification-deep-learning/report/figures/fig2_image_dimensions.png) (Max dimension 512, variable aspect ratios).
    - Bottom-Right: [`report/figures/fig4_intra_class_variance.png`](file:///c:/Users/USER/OneDrive/Documents/Projects/food-classification-deep-learning/report/figures/fig4_intra_class_variance.png) and [`report/figures/fig5_inter_class_similarity.png`](file:///c:/Users/USER/OneDrive/Documents/Projects/food-classification-deep-learning/report/figures/fig5_inter_class_similarity.png).
- **Key Talking Points:**
  - Food-101 benchmark provenance: 101,000 RGB images curated by Bossard et al. (ETH Zürich).
  - Verification of class balance: Exactly 1,000 images per class eliminating majority-class bias.
  - Image geometry: Aspect ratios vary from portrait to landscape with maximum dimension 512 pixels; requires standardized bilinear interpolation to $224 \times 224 \times 3$.
  - High intra-class diversity vs inter-class confusion: Show how `steak` visually overlaps with `pork chop` and `filet mignon`.

---

### Scene 3: Preprocessing Pipeline & Data Integrity
- **Presenter:** Member 1 (Nirmana)
- **Time:** `01:15 – 02:00` (45 seconds)
- **Visual Assets on Screen:**
  - Slide 3: Pipeline Architecture Diagram + Code snippet of `src/preprocessing/data_loader.py` and `augmentations.py`.
  - Terminal/Notebook view: Verification of zero-leakage splits from `data/splits/split_summary.json` (Train: 68,175; Val: 7,575; Test: 25,250).
  - GPU benchmark snippet: Colab T4 utilization (4,217 MB peak VRAM, 77.7 img/s throughput, batch size 32).
- **Key Talking Points:**
  - Strict zero-leakage split protocol: Deterministic seed 42, 90/10 train-val split of original training data, while the 25,250 test images remain completely isolated until final evaluation.
  - Multithreaded `tf.data` pipeline: Parallel disk I/O, `AUTOTUNE` prefetching, and RAM caching.
  - Training-only data augmentations: Random horizontal flipping, random rotation ($\pm 10\%$), and random zoom ($\pm 10\%$) applied exclusively to training batches. Validation and test batches remain unperturbed.
  - Architecture-specific normalization: Standard $[0, 1]$ for Custom CNN, Caffe BGR zero-centering for ResNet-50, and $[-1, 1]$ scaling for MobileNetV2.

---

### Scene 4: Custom CNN Baseline — Design & Empirical Results
- **Presenter:** Member 1 (Nirmana)
- **Time:** `02:00 – 02:45` (45 seconds)
- **Visual Assets on Screen:**
  - Slide 4: Custom CNN Layer Topology Diagram (`src/models/custom_cnn.py`).
  - Right Side: [`results/custom_cnn/training_curves.png`](file:///c:/Users/USER/OneDrive/Documents/Projects/food-classification-deep-learning/results/custom_cnn/training_curves.png) and summary metric callouts.
  - Highlighted callout box: "GAP vs Flattening: 65k vs 12.8M parameters ($99.5\%$ reduction)".
- **Key Talking Points:**
  - Architecture specification: 4-stage convolutional hierarchy (32 $\to$ 64 $\to$ 128 $\to$ 256 filters) with Batch Normalization, ReLU, and MaxPooling.
  - Global Average Pooling (GAP) innovation: Replacing a traditional $14 \times 14 \times 256$ Flatten layer eliminated $12.8\text{ million}$ dense parameters, saving $99.5\%$ parameter overhead and preventing catastrophic dense overfitting.
  - Baseline performance: Trained completely from scratch to **61.69% Test Top-1 Accuracy**, **86.19% Top-5 Accuracy**, at only **678k parameters** and **5.90 ms latency** (170 FPS).
  - The test vs. val generalization paradox: Test accuracy ($61.69\%$) exceeded validation ($57.23\%$) because test data was human-cleaned, whereas training contained $\sim 20\%$ web noise.
- **Verbal Handover to Member 2:**
  > *"To establish whether deeper representation learning and ImageNet pretraining can overcome the visual ambiguities of fine-grained food classification, I will now hand over to Matheesha to present our ResNet-50 deep residual learning investigation."*

---

### Scene 5: Deep Residual Learning & ResNet-50 Architecture
- **Presenter:** Member 2 (Matheesha)
- **Time:** `02:45 – 03:30` (45 seconds)
- **Visual Assets on Screen:**
  - Slide 5: Residual Learning & Bottleneck Architecture Diagram.
  - Visual: Equation $\mathcal{H}(x) = \mathcal{F}(x) + x$ with diagram of the identity shortcut and 3-layer Bottleneck ($1\times 1 \to 3\times 3 \to 1\times 1$).
  - ResNet-50 Stage 1–5 Macro Diagram showing frozen vs unfrozen layers.
- **Key Talking Points:**
  - The Degradation Problem: Plain deep networks saturate and degrade due to optimization difficulties in learning identity mappings.
  - Mathematical formulation of ResNet: Reformulating layers to learn residual mappings $\mathcal{F}(x) = \mathcal{H}(x) - x$. The $+1$ gradient term in $\frac{\partial \mathcal{E}}{\partial x} = \frac{\partial \mathcal{E}}{\partial y} (\frac{\partial \mathcal{F}}{\partial x} + 1)$ ensures uninterrupted gradient flow back to early layers.
  - The 3-layer Bottleneck Block: Channel compression ($1\times 1$), spatial convolution ($3\times 3$), and restoration ($1\times 1$) reducing FLOPs across 50 layers.
  - Transfer learning configuration: Pretrained on ImageNet ($1.28\text{M}$ images); classification head customized with GAP, Batch Normalization, Dropout ($p=0.3$), and Dense(101).

---

### Scene 6: Algorithmic Post-Mortem: The Non-i.i.d. Shuffle Buffer Bug Fix
- **Presenter:** Member 2 (Matheesha)
- **Time:** `03:30 – 04:15` (45 seconds)
- **Visual Assets on Screen:**
  - Slide 6: Visual diagram of Sequential Manifest vs. Shuffle Window.
  - Code diff or snippet: `random.Random(42).shuffle(samples)` in `src/preprocessing/data_loader.py`.
  - Diagnostic comparison chart: Early failure curve ($20\%$ validation accuracy stall) vs. resolved curve ($61.31\%$ in Phase 1).
- **Key Talking Points:**
  - The failure symptom: ResNet-50 initially stalled at $\sim 20\%$ validation accuracy despite varying learning rates and optimizers.
  - Mathematical root cause: Manifest `train.txt` was ordered alphabetically by class (675 samples per class). With a streaming buffer size of 2,048, each mini-batch of 32 images sampled from at most 3 adjacent classes.
  - The non-i.i.d. gradient crisis: Biased mini-batch gradients led to catastrophic sequential forgetting within every single epoch.
  - The deterministic resolution: Pre-shuffling file paths in Python prior to tensor slicing restored independent and identically distributed (i.i.d.) mini-batches ($P(\text{class } k \in \mathcal{B}_t) \approx \text{Uniform}(0, 100)$), immediately unblocking convergence from $20\%$ to $61.31\%$.

---

### Scene 7: ResNet-50 Two-Phase Tuning, Results & Error Analysis
- **Presenter:** Member 2 (Matheesha)
- **Time:** `04:15 – 05:15` (60 seconds)
- **Visual Assets on Screen:**
  - Slide 7: ResNet-50 Phase 1 vs Phase 2 Learning Curves ([`results/resnet50/training_curves.png`](file:///c:/Users/USER/OneDrive/Documents/Projects/food-classification-deep-learning/results/resnet50/training_curves.png)).
  - Confusion Matrix excerpt: Top classes (`edamame` $98.2\%$, `macarons` $94.0\%$) vs Hardest classes (`steak` $42.7\%$ vs `pork chop` $47.8\%$).
  - Results Callout: **73.54% Top-1 Accuracy**, **92.02% Top-5 Accuracy**, **10.98 ms Latency**, **90.8 MB Model Size**.
- **Key Talking Points:**
  - Two-phase training dynamics:
    - Phase 1 (Frozen Backbone, $\eta = 10^{-3}$, Epochs 1–6): Feature extraction reached $61.31\%$ validation accuracy.
    - Phase 2 (Unfreezing Stage 5 residual blocks, $\eta = 10^{-5}$, Epochs 7–20): Domain-specific fine-tuning pushed validation accuracy to $69.47\%$. Frozen BatchNorm prevented covariate collapse.
  - Final test evaluation: Achieved **$73.54\%$ Top-1 Accuracy** and **$92.02\%$ Top-5 Accuracy** on unseen test data with test loss $1.1110$.
  - Error breakdown: Excels on distinct geometric structures (`edamame`, `macarons`), but struggles on seared animal proteins (`steak` vs `pork chop`) where cross-hatch grill marks and warm lighting obscure muscle grain at $224 \times 224$ resolution.
  - Production trade-off: Exceptional accuracy for cloud servers, but 90.8 MB storage and 10.98 ms latency present thermal and memory hurdles for edge devices.
- **Verbal Handover to Member 3:**
  > *"While ResNet-50 provides high classification capacity, mobile and edge deployments demand extreme parameter and memory efficiency. Kanushka will now demonstrate how MobileNetV2 tackles this challenge through inverted residuals and depthwise separable convolutions."*

---

### Scene 8: MobileNetV2 Architecture: Inverted Residuals & Linear Bottlenecks
- **Presenter:** Member 3 (Kanushka)
- **Time:** `05:15 – 06:00` (45 seconds)
- **Visual Assets on Screen:**
  - Slide 8: MobileNetV2 Inverted Residual vs. Classical ResNet Bottleneck comparison diagram.
  - Visual: Thick-thin-thick (ResNet) vs. Thin-thick-thin (MobileNetV2) memory flow.
  - Mathematical annotation: Expansion factor $t=6$, ReLU6 activation, and Linear Bottleneck output.
- **Key Talking Points:**
  - Inverted residual paradigm: Unlike ResNet which compresses channels before convolution ($256 \to 64 \to 256$), MobileNetV2 expands low-dimensional representations to higher dimensions ($32 \to 192 \to 32$, expansion factor $t=6$) for expressive spatial filtering.
  - Linear Bottleneck rationale: Non-linearities like ReLU destroy manifold information in low-dimensional subspaces. MobileNetV2 removes non-linear activation from the final projection layer, preserving information density.
  - Memory-efficient shortcut connections: Residual identity additions are placed directly between thin bottlenecks, drastically minimizing peak RAM residency during inference.

---

### Scene 9: Depthwise Separable Convolutions & Mathematical Efficiency
- **Presenter:** Member 3 (Kanushka)
- **Time:** `06:00 – 06:45` (45 seconds)
- **Visual Assets on Screen:**
  - Slide 9: Depthwise Separable Convolution breakdown:
    - Step 1: Depthwise Convolution ($3\times 3$ spatial filtering per channel).
    - Step 2: Pointwise Convolution ($1\times 1$ channel combination).
  - Mathematical ratio comparison: $\frac{\text{Cost}_{\text{DWS}}}{\text{Cost}_{\text{Standard}}} = \frac{1}{N} + \frac{1}{D_k^2} = \frac{1}{N} + \frac{1}{9} \approx 88\text{–}90\%$ computational savings.
- **Key Talking Points:**
  - Decoupling spatial filtering from channel projection: Standard convolutions perform both simultaneously at heavy computational cost.
  - The Depthwise step: Applies a single $3\times 3$ kernel per input channel ($D_k \times D_k \times M \times D_F^2$).
  - The Pointwise step: Computes linear combinations across channels via $1\times 1$ convolutions ($M \times N \times D_F^2$).
  - Theoretical computational savings: Reduces FLOPs by approximately $8\times$ to $9\times$ compared to standard convolutions with minimal loss in empirical representational capability.

---

### Scene 10: MobileNetV2 Results, Latency Profiling & Edge Feasibility
- **Presenter:** Member 3 (Kanushka)
- **Time:** `06:45 – 07:45` (60 seconds)
- **Visual Assets on Screen:**
  - Slide 10: MobileNetV2 Phase 1 vs Phase 2 Learning Curves ([`results/mobilenetv2/training_curves.png`](file:///c:/Users/USER/OneDrive/Documents/Projects/food-classification-deep-learning/results/mobilenetv2/training_curves.png)).
  - Edge comparison graphic: 9.11 MB footprint (vs 90.8 MB ResNet) | 6.84 ms latency | 2.39M parameters.
  - Test metrics badge: **68.69% Top-1 Accuracy**, **90.15% Top-5 Accuracy**, **Macro F1: 0.6862**.
- **Key Talking Points:**
  - Two-phase training execution:
    - Phase 1 (Frozen Backbone, Epochs 1–8): Reached $57.03\%$ validation accuracy.
    - Phase 2 (Unfreezing top Inverted Residual Blocks 14–16, $\eta = 10^{-5}$, Epochs 9–20): Unlocked fine-grained food texture learning, elevating validation accuracy to $64.73\%$.
  - Test generalization: Delivered **$68.69\%$ Top-1 Accuracy** and **$90.15\%$ Top-5 Accuracy** on unseen test data with test loss $1.1692$.
  - Edge device validation: A model footprint of only **$9.11\text{ MB}$** ($10\times$ smaller than ResNet-50) and an inference latency of **$6.84\text{ ms}$** (~$146\text{ FPS}$) confirm that MobileNetV2 is perfectly suited for on-device smartphone applications with strict thermal and memory limits.
- **Verbal Handover to Member 4:**
  > *"Having evaluated deep residuals and mobile inverted bottlenecks, the question arises: can we achieve both superior accuracy and high efficiency through principled compound scaling? Kaveesha will now present our EfficientNetB0 findings and synthesize the master cross-model comparative evaluation."*

---

### Scene 11: Compound Scaling Principle & EfficientNetB0 Architecture
- **Presenter:** Member 4 (Kaveesha)
- **Time:** `07:45 – 08:30` (45 seconds)
- **Visual Assets on Screen:**
  - Slide 11: Compound Scaling 3D Diagram (Depth $d$, Width $w$, Resolution $r$ scaled simultaneously under $\alpha \cdot \beta^2 \cdot \gamma^2 \approx 2$).
  - MBConv Block with Squeeze-and-Excitation (SE) channel attention module diagram.
- **Key Talking Points:**
  - The limitation of arbitrary single-dimension scaling: Scaling only network depth (like ResNet) causes vanishing gradients; scaling only width leads to shallow representations; scaling only resolution yields diminishing accuracy returns.
  - The Compound Scaling Principle (Tan & Le, 2019): Uniformly balances depth ($d = \alpha^\phi$), width ($w = \beta^\phi$), and input resolution ($r = \gamma^\phi$) using a compound coefficient $\phi$.
  - Squeeze-and-Excitation (SE) optimization: Dynamically recalibrates channel-wise feature responses by modeling channel interdependencies via global pooling, squeeze ($1\times 1$), and sigmoid gating.
  - EfficientNetB0 results: **77.26% Test Top-1 Accuracy**, **94.16% Top-5 Accuracy**, **0.8091 Test Loss** at only **4.18M parameters** and **15.96 MB footprint**.

---

### Scene 12: Cross-Model Master Benchmark & The Pareto Frontier
- **Presenter:** Member 4 (Kaveesha)
- **Time:** `08:30 – 09:30` (60 seconds)
- **Visual Assets on Screen:**
  - Slide 12: Master Cross-Model Comparison Table + Two Core Figures side-by-side:
    - Left: [`report/figures/fig6_comparative_learning_curves.png`](file:///c:/Users/USER/OneDrive/Documents/Projects/food-classification-deep-learning/report/figures/fig6_comparative_learning_curves.png) (All 4 training & validation curves).
    - Right: [`report/figures/fig8_accuracy_vs_parameters.png`](file:///c:/Users/USER/OneDrive/Documents/Projects/food-classification-deep-learning/report/figures/fig8_accuracy_vs_parameters.png) (Accuracy vs Parameters / Pareto Frontier).
- **Key Talking Points:**
  - Strict fair comparison protocol: All 4 models evaluated under identical conditions ($224 \times 224$ input, batch size 32, deterministic seed 42, $25,250$ unseen test images).
  - The Master Metrics Table:
    - Custom CNN (Scratch): 678k params | 2.59 MB | 61.69% Top-1 | 86.19% Top-5 | 5.90 ms
    - MobileNetV2: 2.39M params | 9.11 MB | 68.69% Top-1 | 90.15% Top-5 | 6.84 ms
    - ResNet-50: 23.8M params | 90.8 MB | 73.54% Top-1 | 92.02% Top-5 | 10.98 ms
    - EfficientNetB0: 4.18M params | 15.96 MB | **77.26% Top-1** | **94.16% Top-5** | 7.92 ms
  - The Pareto Frontier insight: EfficientNetB0 dominates the accuracy-efficiency curve, beating ResNet-50 by **$+3.72\%$ Top-1 accuracy** while requiring **$82.4\%$ fewer parameters** and **$82.4\%$ less disk storage**.
  - Parameter Density Metric: Custom CNN achieves an extraordinary **$23.85\%$ accuracy per MB**, validating its role in extreme micro-edge deployments.

---

### Scene 13: Universal Hardest Classes, Deployment Decision Matrix & Conclusion
- **Presenter:** Member 4 (Kaveesha)
- **Time:** `09:30 – 10:30` (60 seconds)
- **Visual Assets on Screen:**
  - Slide 13: Top: [`report/figures/fig9_hardest_classes.png`](file:///c:/Users/USER/OneDrive/Documents/Projects/food-classification-deep-learning/report/figures/fig9_hardest_classes.png) (Cross-model performance across hardest categories).
  - Bottom: Deployment Recommendation Matrix (Cloud vs Mobile vs Micro-Edge).
  - Final Slide: GitHub Repository QR code, acknowledgments, and team sign-off.
- **Key Talking Points:**
  - Universal fine-grained failure modes across all 4 architectures:
    - Opaque wrappers (`breakfast_burrito`): Worst performer across all models because tortilla wrap conceals internal ingredients.
    - Seared proteins (`steak`, `pork_chop`, `filet_mignon`): Persistent visual confusion due to shared sear marks and warm restaurant lighting.
    - Baked crusts (`apple_pie` vs `bread_pudding`): Pastry crusts mask internal fillings.
  - Concrete Deployment Decision Framework:
    - **Ultra-Constrained Edge / IoT (< 5 MB RAM):** Deploy **Custom CNN** (2.59 MB, 5.9 ms, 86.2% Top-5).
    - **Mobile Smartphone Applications (Offline):** Deploy **MobileNetV2** (9.11 MB, 6.8 ms, 90.2% Top-5).
    - **Cloud Enterprise APIs & Production Health Apps:** Deploy **EfficientNetB0** (15.96 MB, 77.26% Top-1, 94.16% Top-5).
  - Final conclusion: Thank the examiners, invite questions for the live Viva Voce defense.

---

## 4. Cross-Model Benchmark Data Reference Table

This table must be displayed during Member 4's presentation (Scene 12) and referenced throughout the video:

| Architectural Metric | Custom CNN Baseline | MobileNetV2 | ResNet-50 | EfficientNetB0 (Champion) |
| :--- | :---: | :---: | :---: | :---: |
| **Model Type / Paradigm** | 4-Stage ConvNet (Scratch) | Inverted Residuals | Residual Bottlenecks | Compound Scaling + SE |
| **Total Parameters** | **678,085** (~0.68M) | 2,387,365 (~2.39M) | 23,802,853 (~23.8M) | 4,184,072 (~4.18M) |
| **Trainable Parameters** | **675,653** | 1,737,445 | 9,130,085 | 3,261,825 |
| **Float32 Weights on Disk** | **2.59 MB** | 9.11 MB | 90.80 MB | 15.96 MB |
| **Inference Latency (T4 GPU)**| **5.90 ms** | 6.84 ms | 10.98 ms | 7.92 ms |
| **Throughput (Images/sec)** | **~169.5 FPS** | ~146.2 FPS | ~91.1 FPS | ~126.3 FPS |
| **Validation Top-1 Accuracy**| 57.23% | 64.73% | 69.47% | **72.69%** |
| **Validation Top-5 Accuracy**| 80.90% | 86.64% | 88.65% | **90.68%** |
| **Unseen Test Top-1 Accuracy**| 61.69% | 68.69% | 73.54% | **77.26%** |
| **Unseen Test Top-5 Accuracy**| 86.19% | 90.15% | 92.02% | **94.16%** |
| **Unseen Test Loss** | 1.4441 | 1.1692 | 1.1110 | **0.8091** |
| **Macro F1-Score** | 0.6169 | 0.6862 | 0.7355 | **0.7718** |
| **Accuracy per Megabyte** | **23.85% / MB** | 7.54% / MB | 0.81% / MB | 4.84% / MB |

---

## 5. Technical Recording & Production Standards

### Visual Standards
- **Resolution & Aspect Ratio:** 1920×1080 (16:9 Full HD) at 60 fps.
- **Slide Theme:** Clean academic dark-mode or crisp white slate with high-contrast text. Consistent typography (e.g., Inter, Montserrat, or Roboto).
- **Embedded Visuals:** Use the high-resolution `.png` files directly from `report/figures/` and `results/`. Avoid stretched, pixelated, or low-DPI screen grabs.
- **Picture-in-Picture Webcam:** Position presenter webcam in the upper-right corner ($320 \times 180$ px), framed from the shoulders up with professional lighting.

### Audio & Pacing Standards
- **Pacing:** Maintain a deliberate, professional cadence of **130 to 145 words per minute**.
- **Audio Equalization:** Use software noise suppression (e.g., OBS Noise Suppression filter, Krisp, or Audacity) to eliminate room echo, keyboard clicks, and fan noise.
- **Audio Leveling:** Normalize master audio to $-1.0\text{ dB}$ peak and $-16\text{ LUFS}$ integrated loudness across all 4 presenters.

---

## 6. How Members & AI Agents Should Use This Plan to Generate Scripts

When generating spoken scripts for any specific team member, adhere to the following template:

```markdown
### Script Prompting Structure for Team Members & AI Assistants:
1. Identify the Speaker: (e.g., "Generate word-for-word spoken script for Member 2 (Matheesha)").
2. Strict Time Bound: Exactly 2 minutes 30 seconds (~350 words).
3. Section Inclusions:
   - Opening transition reception (acknowledging previous member).
   - Core technical exposition with mathematical principles stated naturally.
   - Screen cue directions: [SHOW SLIDE 5: RESIDUAL BOTTLENECK], [SHOW CODE SNIPPET: SHUFFLE BUG].
   - Exact empirical numbers stated out loud (e.g., "73.54% top-1 accuracy on 25,250 unseen test images").
   - Explicit handover cue to next member.
```

---

## 7. Submission Checklist & Deliverables

- [ ] **Slide Deck Finalized:** 13–15 slides covering Scenes 1 through 13.
- [ ] **Individual Audio/Video Clips Recorded:**
  - [ ] Member 1: 00:00 – 02:45
  - [ ] Member 2: 02:45 – 05:15
  - [ ] Member 3: 05:15 – 07:45
  - [ ] Member 4: 07:45 – 10:30
- [ ] **Video Editing & Assembly:** Stitched with smooth 0.5-second cross-dissolves, synchronized slides, and unified audio levels.
- [ ] **Duration Verification:** Total video length strictly between **10:00 and 11:30 minutes** (under the 12:00 hard ceiling).
- [ ] **YouTube Upload:** Uploaded as **Unlisted** (or Public) in 1080p, URL tested in incognito mode.
- [ ] **Submission.txt Prepared:** GitHub repository URL + YouTube video link placed in the root folder as mandated by the master plan.
