# SE4050 Deep Learning — Member 3 (Kanushka) Presentation Script & Guide

**Presenter:** Kanushka Kahakotuwa (Member 3)  
**Assigned Architecture:** MobileNetV2 (Inverted Residual and Linear Bottleneck CNN)  
**Allotted Video Time:** `05:15 – 07:45` (Strictly **2 Minutes 30 Seconds** / 150 Seconds)  
**Target Word Count:** **355 Words** (Delivered at an optimal ~142 words/minute)  
**Associated Slides:** Slides 8, 9, and 10 in the Group Master Presentation Deck  

---

## 1. Scene-by-Scene Visual & Slide Layout

| Scene & Time | Slide Title | Visual Assets to Show on Screen | Speaking Goal |
| :--- | :--- | :--- | :--- |
| **Scene 8**<br>`05:15 – 06:00`<br>*(45 sec)* | **Slide 8:** MobileNetV2 Architecture: Inverted Residuals & Linear Bottlenecks | • Side-by-side diagram: Classical ResNet (Wide $\to$ Narrow $\to$ Wide) vs. MobileNetV2 Inverted Residual (Narrow $\to$ Wide $\to$ Narrow with expansion factor $t=6$).<br>• Diagram illustrating low-dimensional manifold preservation (Linear Bottleneck without ReLU). | Acknowledge Matheesha; explain inverted residual flow and why linear bottlenecks prevent information loss. |
| **Scene 9**<br>`06:00 – 06:45`<br>*(45 sec)* | **Slide 9:** Depthwise Separable Convolutions & Mathematical Efficiency | • Diagram factorizing Standard Conv into: (1) $3\times3$ Depthwise Conv (spatial) + (2) $1\times1$ Pointwise Conv (channel).<br>• Mathematical equation: $\frac{\text{Cost}_{\text{DWS}}}{\text{Cost}_{\text{Standard}}} = \frac{1}{N} + \frac{1}{D_k^2} \approx \frac{1}{9}$ ($88\text{--}89\%$ FLOP reduction). | Explain decoupling of spatial filtering from channel projection and derive the 88% compute reduction. |
| **Scene 10**<br>`06:45 – 07:45`<br>*(60 sec)* | **Slide 10:** Empirical Results, Latency Profiling & Edge Feasibility | • High-res dual learning curves (`results/mobilenetv2/training_curves.png`).<br>• Metric Badges: **68.69% Top-1**, **90.15% Top-5**, **Macro F1: 0.6862**, **9.11 MB Footprint**, **6.84 ms GPU Latency**.<br>• Comparison graphic: MobileNetV2 (9.11 MB) vs. ResNet-50 (90.8 MB). | Detail 2-phase training, state exact test benchmark numbers, prove edge viability, and hand over to Kaveesha. |

---

## 2. Word-for-Word Spoken Presentation Script

> **Delivery Tip:** Speak with confidence, clear articulation, and steady cadence. Do not rush. Hit the screen cues precisely as you transition slides.

---

### [05:15 – 06:00] Scene 8: Inverted Residuals & Linear Bottlenecks (Slide 8)

**[ON SCREEN: SLIDE 8 — INVERTED RESIDUALS & LINEAR BOTTLENECKS DIAGRAM]**  
**[WEBCAM: Upper-right corner, looking directly into the camera]**

> *"Thank you, Matheesha.*
>
> *While ResNet-50 achieves strong accuracy, deploying deep networks on mobile devices requires fundamentally rethinking memory bandwidth and parameter efficiency. For Member 3, I investigated **MobileNetV2**.*
>
> *Unlike classical residual blocks that compress channels before convolution, MobileNetV2 introduces an **Inverted Residual** structure. It takes a thin, low-dimensional bottleneck, expands it by a factor of six into high-dimensional space for expressive spatial filtering, and then projects it back down to a narrow bottleneck.*
>
> *Crucially, we enforce a **Linear Bottleneck**: non-linear activations like ReLU are strictly omitted from the final projection layer. In low-dimensional spaces, non-linearities irrevocably collapse the visual manifold of interest. Removing ReLU preserves essential food texture representations while shortcut connections between thin bottlenecks dramatically reduce peak memory residency during inference."*

---

### [06:00 – 06:45] Scene 9: Depthwise Separable Convolutions & Math (Slide 9)

**[TRANSITION TO SLIDE 9 — DEPTHWISE SEPARABLE CONVOLUTION MATH]**

> *"The computational engine driving MobileNetV2 is **Depthwise Separable Convolution**, which decouples spatial filtering from channel correlation.*
>
> *A standard convolution performs both simultaneously, incurring heavy compute. MobileNetV2 splits this into two steps:*
> 1. *First, a **depthwise convolution** applies a lightweight three-by-three spatial filter to each input channel independently.*
> 2. *Second, a **pointwise one-by-one convolution** linearly combines these spatial features across channels.*
>
> *Mathematically, this reduces computational cost by a factor of one over N plus one over D-k squared. For three-by-three kernels, this delivers an **eighty-eight to eighty-nine percent reduction in FLOPs** compared to standard convolutions, slashing total network parameters to just **2.39 million**."*

---

### [06:45 – 07:45] Scene 10: Training Dynamics, Test Results & Handover (Slide 10)

**[TRANSITION TO SLIDE 10 — MOBILENETV2 RESULTS, CURVES & EDGE PROFILE]**

> *"We trained MobileNetV2 in two disciplined phases:*
> - *In **Phase 1**, training only the classification head with a frozen backbone reached **57.03%** validation accuracy.*
> - *In **Phase 2**, unfreezing top inverted residual blocks fourteen through sixteen under a delicate learning rate of ten-to-the-minus-five pushed validation accuracy to **64.73%**. Crucially, we kept Batch Normalization frozen to preserve ImageNet population statistics.*
>
> *On the final, unseen Food-101 test set of twenty-five thousand two hundred and fifty images, MobileNetV2 achieved **68.69% Top-1 Accuracy** and **90.15% Top-5 Accuracy**, with an impressive test loss of **1.1692** and macro F1-score of **0.6862**.*
>
> *With an uncompressed disk footprint of only **9.11 megabytes**—ten times lighter than ResNet-50—and an inference latency of **6.84 milliseconds** on GPU, MobileNetV2 proves exceptionally viable for on-device mobile nutrition apps.*
>
> *I will now hand over to Kaveesha to present our compound scaling findings with EfficientNetB0 and synthesize the master comparative evaluation."*

**[END OF KANUSHKA'S SEGMENT — EXACT TIMER: 07:45]**

---

## 3. Rehearsal & Pronunciation Quick-Reference

To ensure your spoken delivery sounds completely natural and authoritative, practice these specific technical phrases:

| Written Term | How to Say It Naturally |
| :--- | :--- |
| **$\mathcal{F}(x) + x$** | *"F of x plus x"* |
| **$t=6$** | *"expansion factor of six"* |
| **$3 \times 3$** | *"three-by-three"* |
| **$1 \times 1$** | *"one-by-one"* |
| **$\frac{1}{N} + \frac{1}{D_k^2}$** | *"one over N plus one over D-k squared"* |
| **$\approx 88\text{--}89\%$** | *"eighty-eight to eighty-nine percent"* |
| **2,387,365** | *"two-point-three-nine million"* |
| **$\eta = 10^{-5}$** | *"ten to the minus five"* |
| **9.11 MB** | *"nine-point-one-one megabytes"* |
| **6.84 ms** | *"six-point-eight-four milliseconds"* |
| **25,250** | *"twenty-five thousand two hundred and fifty"* |
| **68.69%** | *"sixty-eight point six nine percent"* |
| **90.15%** | *"ninety point one five percent"* |

---

## 4. Top 5 Viva Voce Examination Questions & Answers (MobileNetV2)

The examiners will test your deep individual understanding during the live Viva Voce defense. Here are the exact technical answers to the most common questions:

### Q1: Why did you freeze Batch Normalization layers during Phase 2 fine-tuning?
> **Answer:** *"In fine-tuning, mini-batch sizes are typically small (batch size 32). If Batch Normalization layers are trainable, the batch statistics ($\mu_{\text{batch}}, \sigma^2_{\text{batch}}$) computed on small food batches overwrite the well-calibrated population statistics ($\mu_{\text{pop}}, \sigma^2_{\text{pop}}$) learned across 1.28 million ImageNet samples. Freezing Batch Normalization prevents internal covariate collapse, avoids gradient explosion, and ensures training stability under small learning rates."*

### Q2: Why is the bottleneck in MobileNetV2 'linear' instead of using a ReLU activation?
> **Answer:** *"According to the Manifold of Interest hypothesis, meaningful visual representations lie on low-dimensional manifolds embedded within high-dimensional space. Non-linear activations like ReLU set all negative coordinates to zero, which destroys information in low-dimensional spaces. By omitting non-linear activations from the final $1 \times 1$ projection layer, MobileNetV2 preserves the full geometric structure of the feature manifold."*

### Q3: How do Depthwise Separable Convolutions achieve ~88% computational savings?
> **Answer:** *"Standard convolution computes spatial filtering and channel mixing simultaneously with cost $D_K^2 \cdot M \cdot N \cdot D_F^2$. Depthwise separable convolution splits this into depthwise convolution ($D_K^2 \cdot M \cdot D_F^2$) followed by pointwise convolution ($M \cdot N \cdot D_F^2$). Dividing the separable cost by the standard cost yields $\frac{1}{N} + \frac{1}{D_K^2}$. For $3 \times 3$ kernels ($D_K = 3$) and $N \ge 64$, $\frac{1}{N} \approx 0$ and $\frac{1}{9} \approx 0.111$, representing an $88.9\%$ reduction in multiply-accumulate operations."*

### Q4: Why did MobileNetV2 achieve higher accuracy on the test set (68.69%) than the validation set (64.73%)?
> **Answer:** *"This is due to the composition of the Food-101 dataset. As documented by Bossard et al., the Food-101 training and validation splits were web-crawled and intentionally contain approximately $20\%$ label noise. However, the $25,250$ test images were manually inspected and cleaned. Evaluating our well-regularized model on clean test ground truth naturally yields higher predictive accuracy than on noisy validation labels."*

### Q5: What makes MobileNetV2 superior to ResNet-50 for edge deployment despite being ~4.8% lower in top-1 accuracy?
> **Answer:** *"Edge deployment is constrained by on-chip SRAM cache and battery life. ResNet-50 has 23.8 million parameters ($90.8\text{ MB}$ weights), exceeding typical mobile on-chip cache ($8\text{--}32\text{ MB}$) and forcing continuous off-chip DRAM reads that cause thermal throttling. MobileNetV2 has only 2.39 million parameters ($9.11\text{ MB}$ weights), fitting entirely into on-chip cache. In terms of efficiency, MobileNetV2 delivers $28.77\%$ Top-1 accuracy per million parameters—over $9\times$ higher parameter efficiency than ResNet-50 ($3.09\%$/M-param)—while reaching over $90\%$ Top-5 accuracy."*

---

## 5. Pre-Recording Technical Checklist

- [ ] **Slide Visuals Ready:** Slides 8, 9, and 10 prepared in PowerPoint / Google Slides / Canva with diagrams matching the report.
- [ ] **OBS / Screen Recorder Configured:** 1920×1080 resolution, 60 fps, microphone input selected with noise filter active.
- [ ] **Camera Position:** Webcam in upper-right corner ($320 \times 180\text{ px}$), eye-level framing, good lighting on face.
- [ ] **Stopwatch / Timer Open:** Set phone timer to **2 minutes 30 seconds**.
- [ ] **Rehearsal:** Read through the script 2–3 times aloud to ensure your pacing falls naturally between **2:25 and 2:32**.
