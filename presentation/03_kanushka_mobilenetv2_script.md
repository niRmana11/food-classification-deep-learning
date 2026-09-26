# SE4050 Deep Learning — Member 3 (Kanushka) Presentation Script & Guide

**Presenter:** Kanushka Kahakotuwa (Member 3)  
**Assigned Architecture:** MobileNetV2 (Inverted Residual and Linear Bottleneck CNN)  
**Allotted Video Time:** `05:15 – 07:45` (Strict Target: **2 Minutes 15 Seconds – 2 Minutes 25 Seconds** to stay safely under the 2:30 ceiling)  
**Total Word Count:** **295 Words** (Guaranteed ~2 min 15 sec at a natural ~130 words/minute)  
**Presentation Strategy:** Single-Screen 3-Slide Method (Slides 8, 9, 10) — Zero window switching delays!

---

## 1. Scene & Slide Overview

| Scene & Time | Slide | Visual Assets Embedded on the Slide | Spoken Focus |
| :--- | :--- | :--- | :--- |
| **Scene 8**<br>`05:15 – 06:00`<br>*(45 sec)* | **Slide 8:** Inverted Residuals & Linear Bottlenecks | • Diagram of Inverted Residual Block (**Report Figure 6.7** / `fig_arch_mobilenetv2.png`).<br>• Visual callouts: Expansion $t=6$, $3\times3$ Depthwise, $1\times1$ Linear Bottleneck (NO ReLU). | Handoff from Matheesha; explain narrow $\to$ wide $\to$ narrow dataflow and why linear bottlenecks preserve low-dimensional manifold data. |
| **Scene 9**<br>`06:00 – 06:40`<br>*(40 sec)* | **Slide 9:** Depthwise Separable Convolutions & Math | • Factorization diagram: $3\times3$ Depthwise (spatial) + $1\times1$ Pointwise (channel).<br>• FLOP Formula: $\frac{\text{Cost}_{\text{DWS}}}{\text{Cost}_{\text{Std}}} = \frac{1}{N} + \frac{1}{D_k^2} \approx \frac{1}{9} \approx 88\text{--}89\%$ savings.<br>• Code snippet callout from `src/models/mobilenetv2.py`. | Explain spatial vs channel decoupling; derive the 88% FLOP reduction and 2.39M parameter count. |
| **Scene 10**<br>`06:40 – 07:40`<br>*(60 sec)* | **Slide 10:** Empirical Results & Edge Viability | • Dual learning curves (`results/mobilenetv2/training_curves.png`).<br>• Metric badges: **68.69% Top-1**, **90.15% Top-5**, **1.1692 Loss**, **9.11 MB footprint**.<br>• Mini confusion matrix thumbnail (`confusion_matrix.png`). | Detail Phase 1 & 2 training, state exact test benchmark numbers on 25,250 images, prove edge viability, hand over to Kaveesha. |

---

## 2. Word-for-Word Spoken Presentation Script (Timed ~2:20)

> **Delivery Rule:** Keep your eyes on the camera. Speak at a calm, conversational speed. Do not switch windows during recording—simply press the Spacebar or Right Arrow to advance between Slides 8, 9, and 10!

---

### 🟢 [05:15 – 06:00] Scene 8: Inverted Residuals & Linear Bottlenecks (Slide 8)

* **[SCREEN: DISPLAY SLIDE 8 IN FULL SCREEN]**  
* **[WEBCAM: Upper-right corner, looking into the lens]**

> *"Thank you, Matheesha.*
>
> *While ResNet-50 achieves strong accuracy, deploying 24-million parameters to edge devices is prohibitive. For Member 3, I evaluated **MobileNetV2**.*
>
> *As shown in Figure 6.7, MobileNetV2 uses an **Inverted Residual** structure: a narrow-to-wide-to-narrow flow. An expansion one-by-one convolution expands channels six-fold ($t=6$) for expressive spatial filtering, followed by a three-by-three depthwise convolution, and finally a one-by-one projection back to a narrow bottleneck.*
>
> *Crucially, the bottleneck is **strictly linear with no ReLU**. Non-linearities in low dimensions destroy visual manifold data. Preserving linearity retains subtle food textures while thin shortcut connections drastically cut memory consumption."*

---

### 🟢 [06:00 – 06:40] Scene 9: Depthwise Separable Math & Code (Slide 9)

* **[ACTION: ADVANCE TO SLIDE 9]**

> *"The computational core of MobileNetV2 is **Depthwise Separable Convolution**, implemented in `src/models/mobilenetv2.py`.*
>
> *Standard convolution performs spatial filtering and channel mixing simultaneously. MobileNetV2 decouples them:*
> *First, depthwise convolution applies a three-by-three filter per channel.*
> *Second, pointwise one-by-one convolution mixes channels together.*
>
> *Mathematically, computational cost drops by **one over N plus one over D-k squared**. For three-by-three kernels, this achieves an **eighty-eight percent FLOP reduction**, slashing total parameters to just **2.39 million**."*

---

### 🟢 [06:40 – 07:40] Scene 10: Training, Benchmark Results & Handover (Slide 10)

* **[ACTION: ADVANCE TO SLIDE 10]**

> *"We trained MobileNetV2 in two phases:*
> *In **Phase 1**, warming up the classification head with a frozen backbone reached **57.03%** validation accuracy.*
> *In **Phase 2**, fine-tuning blocks 14 through 17 from layer 120 onward at a learning rate of ten-to-the-minus-five reached **64.73%**. We kept Batch Normalization frozen to preserve ImageNet statistics.*
>
> *On the unseen test set of 25,250 images, MobileNetV2 achieved **68.69% Top-1** and **90.15% Top-5 Accuracy**, with a test loss of **1.1692** and macro F1 of **0.6862**.*
>
> *At just **9.11 megabytes**—ten times lighter than ResNet-50—MobileNetV2 proves exceptionally viable for mobile deployment.*
>
> *I will now hand over to Kaveesha for EfficientNetB0 and the master cross-model comparison."*

**[STOP RECORDING — APPROXIMATE TIME: ~2 MINUTES 15 SECONDS TO 2 MINUTES 25 SECONDS]**

---

## 3. Pronunciation Quick-Reference

| Written Term | Spoken Form |
| :--- | :--- |
| **$t=6$** | *"expansion factor of six"* |
| **$3 \times 3$ and $1 \times 1$** | *"three-by-three and one-by-one"* |
| **$\frac{1}{N} + \frac{1}{D_k^2}$** | *"one over N plus one over D-k squared"* |
| **$\approx 88\%$** | *"eighty-eight percent"* |
| **2.39M** | *"two-point-three-nine million"* |
| **$\eta = 10^{-5}$** | *"ten to the minus five"* |
| **9.11 MB** | *"nine-point-one-one megabytes"* |
| **25,250** | *"twenty-five thousand two hundred and fifty"* |
| **68.69% / 90.15%** | *"sixty-eight point six nine percent / ninety point one five percent"* |
| **ReLU** | *"ray-loo"* |

---

## 4. Top 5 Viva Voce Questions & Answers (MobileNetV2)

The examiners will test your individual understanding during the live Viva defense. Use these exact technical answers:

### Q1: Why did you freeze Batch Normalization layers during Phase 2 fine-tuning?
> **Answer:** *"In fine-tuning, mini-batch sizes are small (batch size 32). If Batch Normalization layers are trainable, the batch statistics computed on small food batches overwrite the well-calibrated population statistics learned across 1.28 million ImageNet samples. Freezing Batch Normalization prevents internal covariate collapse, avoids gradient explosion, and ensures training stability."*

### Q2: Why is the bottleneck in MobileNetV2 'linear' instead of using a ReLU activation?
> **Answer:** *"According to the Manifold of Interest hypothesis, meaningful visual representations lie on low-dimensional manifolds within high-dimensional space. Non-linear activations like ReLU zero out negative coordinates, which destroys information in low-dimensional spaces. Omitting ReLU from the final $1 \times 1$ projection preserves the geometric manifold of subtle food textures."*

### Q3: How do Depthwise Separable Convolutions achieve ~88% computational savings?
> **Answer:** *"Standard convolution costs $D_k^2 \cdot M \cdot N \cdot D_f^2$. Depthwise separable convolution splits this into depthwise spatial filtering ($D_k^2 \cdot M \cdot D_f^2$) and pointwise channel mixing ($M \cdot N \cdot D_f^2$). The ratio of separable to standard cost is $\frac{1}{N} + \frac{1}{D_k^2}$. For $3 \times 3$ kernels ($D_k = 3$) and large $N$, $\frac{1}{N} \approx 0$ and $\frac{1}{9} \approx 0.111$, yielding an $88.9\%$ reduction in FLOPs."*

### Q4: In Table 6.6 / Section 8.3 of the report, why did MobileNetV2 show high single-image latency (167.29 ms) on Tesla T4 compared to ResNet-50?
> **Answer:** *"This highlights the arithmetic intensity gap between theoretical FLOPs and GPU runtime. Server GPUs like the Tesla T4 are optimized for dense matrix multiplies (GEMM). Depthwise convolutions have low arithmetic intensity (low FLOPs per memory byte transferred) and split operations into separate kernels, adding CUDA dispatch overhead at batch size 1. On mobile NPUs and edge hardware (Apple Neural Engine, Edge TPUs) where depthwise operations run natively in on-chip SRAM cache, MobileNetV2 achieves sub-10 ms latency."*

### Q5: Why was test accuracy (68.69%) higher than validation accuracy (64.73%)?
> **Answer:** *"As documented by Bossard et al., Food-101 training and validation sets were web-scraped and contain ~20% label noise, occlusions, and background clutter. However, the 25,250 test images were manually cleaned and verified. Evaluating our well-regularized model on clean ground truth labels naturally yields higher accuracy."*

---

## 5. Quick Pre-Recording Checklist

- [ ] Open **Slides 8, 9, 10** in your presentation software (or open [`presentation/slides_mobilenetv2.html`](file:///d:/Group%20Projects/food-classification-deep-learning/presentation/slides_mobilenetv2.html) in full screen with `F11`).
- [ ] In OBS, set scene to display your presentation in full screen with your webcam in the upper-right corner.
- [ ] Set your phone timer to **2 minutes 30 seconds**.
- [ ] Practice reading the script once through—aim for **~2:15 to 2:25**.
- [ ] Hit record, speak clearly, press spacebar twice to advance slides, and finish on time!
