# SE4050 Deep Learning — Member 3 (Kanushka) Presentation Script & Guide

**Presenter:** Kanushka Kahakotuwa (Member 3)  
**Assigned Architecture:** MobileNetV2 (Inverted Residual and Linear Bottleneck CNN)  
**Allotted Video Time:** `05:15 – 07:45` (Strictly **2 Minutes 30 Seconds** / 150 Seconds)  
**Target Word Count:** **355 Words** (Delivered at an optimal ~140–142 words/minute)  
**Multi-Window Demonstration:** Slides 8, 9, 10 + VS Code Repo (`src/models/mobilenetv2.py`) + Google Colab (`notebooks/05_mobilenetv2.ipynb`) + Report Artifacts  

---

## 1. Multi-Screen Visual Navigation & Timing Map

This timeline outlines how to seamlessly demonstrate your slides, actual codebase, Google Colab training logs, and test results within your allotted 2 minutes and 30 seconds:

| Scene & Time | Primary Screen View | Secondary / Live Demo View | Spoken Objective |
| :--- | :--- | :--- | :--- |
| **Scene 8**<br>`05:15 – 06:00`<br>*(45 sec)* | **Slide 8:** Inverted Residuals & Linear Bottlenecks Diagram | *Quick glance at report:* Section 6.3 / Figure 6.7 | Acknowledge Matheesha; explain inverted residual flow ($t=6$) and why linear bottlenecks preserve low-dimensional manifold information. |
| **Scene 9**<br>`06:00 – 06:45`<br>*(45 sec)* | **VS Code / GitHub Repo:** [`src/models/mobilenetv2.py`](file:///d:/Group%20Projects/food-classification-deep-learning/src/models/mobilenetv2.py) | **Slide 9:** Depthwise Separable Factorization & Math | Walk through modular Python code; show Depthwise + Pointwise layers; derive the 88%–89% FLOP reduction ($\frac{1}{N} + \frac{1}{D_k^2}$). |
| **Scene 10**<br>`06:45 – 07:45`<br>*(60 sec)* | **Google Colab Notebook:** [`notebooks/05_mobilenetv2.ipynb`](file:///d:/Group%20Projects/food-classification-deep-learning/notebooks/05_mobilenetv2.ipynb) | **Slide 10 / Report:** [`results/mobilenetv2/training_curves.png`](file:///d:/Group%20Projects/food-classification-deep-learning/results/mobilenetv2/training_curves.png) & metrics | Demonstrate Phase 1/2 training cells, frozen BatchNorm, final test evaluation on 25,250 images, highlight 9.11 MB edge footprint, and hand over to Kaveesha. |

---

## 2. Word-for-Word Spoken Presentation Script with Live Actions

> **Delivery Tip:** Keep your webcam visible in the upper-right corner ($320 \times 180\text{ px}$). Use `Alt + Tab` smoothly between windows. Maintain a steady, confident cadence of ~140 words per minute.

---

### [05:15 – 06:00] Scene 8: Inverted Residuals & Linear Bottlenecks (Slide 8)

* **[ACTION: DISPLAY SLIDE 8 ON FULL SCREEN (OR REPORT FIGURE 6.7)]**  
* **[WEBCAM: Upper-right corner, looking directly into the camera lens]**

> *"Thank you, Matheesha.*
>
> *While ResNet-50 achieves strong classification accuracy, deploying 24-million-parameter networks to battery-constrained mobile and IoT hardware is computationally prohibitive. For Member 3, I engineered and evaluated **MobileNetV2**.*
>
> *Unlike classical residual networks that compress channels before convolution, MobileNetV2 introduces an **Inverted Residual** structure. It takes a thin, low-dimensional bottleneck, expands it by an expansion factor of six into high-dimensional space for expressive spatial filtering, and then projects it back down to a narrow bottleneck.*
>
> *Crucially, we enforce a **Linear Bottleneck**: non-linear activations like ReLU are strictly omitted from the final projection layer. In low-dimensional spaces, non-linearities irrevocably destroy the underlying manifold of interest. Removing ReLU preserves essential food texture representations while shortcut connections between thin bottlenecks dramatically reduce peak memory residency during inference."*

---

### [06:00 – 06:45] Scene 9: Code Walkthrough & Depthwise Separable Math (VS Code / Slide 9)

* **[ACTION: SWITCH TO VS CODE SHOWING `src/models/mobilenetv2.py` (LINES 30–75) OR SLIDE 9]**  
* **[ACTION: HIGHLIGHT THE DEPTHWISE & POINTWISE CONVOLUTION IMPLEMENTATION]**

> *"Looking at our implementation in `src/models/mobilenetv2.py`, the core computational engine driving MobileNetV2 is **Depthwise Separable Convolution**, which completely decouples spatial filtering from channel correlation.*
>
> *A standard convolution performs both simultaneously, incurring heavy compute. MobileNetV2 factorizes this into two disciplined steps:*
> 1. *First, a **depthwise convolution** applies a lightweight three-by-three spatial filter to each input channel independently.*
> 2. *Second, a **pointwise one-by-one convolution** linearly combines these spatial features across channels.*
>
> *As documented in Section 6.3 of our report, this factorizes computational cost by roughly **one over N plus one over D-k squared**. For standard three-by-three kernels, this delivers an **eighty-eight to eighty-nine percent reduction in FLOPs** compared to standard convolutions, slashing total network parameters to just **2.39 million**."*

---

### [06:45 – 07:45] Scene 10: Colab Live Execution, Results & Handover (Colab / Slide 10)

* **[ACTION: SWITCH TO GOOGLE COLAB `notebooks/05_mobilenetv2.ipynb` — SCROLL TO STEP 5 & STEP 8]**  
* **[ACTION: SHOW PHASE 1 & 2 CELLS, THEN DISPLAY FINAL TEST EVALUATION & `training_curves.png`]**

> *"Here in our Google Colab training environment, you can see our standardized two-phase protocol:*
> - *In **Phase 1**, training only the classification head with a frozen backbone reached **57.03%** validation accuracy.*
> - *In **Phase 2**, unfreezing top inverted residual blocks fourteen through seventeen—from layer 120 onward—under a learning rate of ten-to-the-minus-five propelled validation accuracy to **64.73%**. Crucially, we kept Batch Normalization layers frozen to safeguard ImageNet population statistics.*
>
> *On the quarantined, unseen Food-101 test set of twenty-five thousand two hundred and fifty images, MobileNetV2 achieved **68.69% Top-1 Accuracy** and **90.15% Top-5 Accuracy**, with an impressive test loss of **1.1692** and macro F1 of **0.6862**.*
>
> *With an uncompressed disk footprint of only **9.11 megabytes**—ten times lighter than ResNet-50—and superior parameter efficiency of **28.77% accuracy per million parameters**, MobileNetV2 proves exceptionally viable for on-device mobile nutrition apps.*
>
> *I will now hand over to Kaveesha to present our compound scaling findings with EfficientNetB0 and synthesize our master comparative evaluation."*

**[END OF KANUSHKA'S SEGMENT — EXACT TIMER: 07:45]**

---

## 3. Pronunciation & Technical Delivery Guide

Practice these exact scientific phrases aloud to ensure fluent, confident delivery during the recording:

| Written Term | Spoken Form |
| :--- | :--- |
| **$t=6$** | *"expansion factor of six"* |
| **$3 \times 3$ and $1 \times 1$** | *"three-by-three and one-by-one"* |
| **$\frac{1}{N} + \frac{1}{D_k^2}$** | *"one over N plus one over D-k squared"* |
| **$\approx 88\text{--}89\%$** | *"eighty-eight to eighty-nine percent"* |
| **2,387,365** | *"two-point-three-nine million"* |
| **$\eta = 10^{-5}$** | *"ten to the minus five"* |
| **9.11 MB** | *"nine-point-one-one megabytes"* |
| **6.84 ms / 167.29 ms** | *"six-point-eight-four milliseconds / one hundred sixty-seven point two nine milliseconds"* |
| **25,250** | *"twenty-five thousand two hundred and fifty"* |
| **68.69% / 90.15%** | *"sixty-eight point six nine percent / ninety point one five percent"* |
| **ReLU6** | *"ray-loo six"* |
| **NPU / GEMM** | *"N-P-U / jem"* |

---

## 4. Top 5 Viva Voce Examination Questions & Answers (MobileNetV2)

The examiners will evaluate your individual technical depth using your report (Section 6.3 and 8.3). Here are the model answers:

### Q1: Why did you freeze Batch Normalization layers during Phase 2 fine-tuning?
> **Answer:** *"In transfer learning fine-tuning, mini-batch sizes are small (batch size 32). If Batch Normalization layers are left trainable, the mean and variance computed on small food batches overwrite the robust ImageNet population statistics ($\mu_{\text{pop}}, \sigma^2_{\text{pop}}$) calibrated across 1.28 million images. Freezing Batch Normalization prevents internal covariate collapse, avoids gradient explosion, and ensures training stability when using low learning rates ($\eta = 10^{-5}$)."*

### Q2: Why is the bottleneck in MobileNetV2 'linear' instead of using a ReLU activation?
> **Answer:** *"According to the Manifold of Interest hypothesis, meaningful visual representations lie on low-dimensional manifolds embedded within high-dimensional spaces. Non-linear activations like ReLU set all negative coordinates to zero, which destroys manifold information in narrow channels. By omitting non-linear activations from the final $1 \times 1$ projection layer, MobileNetV2 preserves the full linear geometry of extracted food features."*

### Q3: How do Depthwise Separable Convolutions achieve ~88% computational savings?
> **Answer:** *"Standard convolution costs $D_k^2 \cdot M \cdot N \cdot D_f^2$. Depthwise separable convolution decouples this into depthwise spatial filtering ($D_k^2 \cdot M \cdot D_f^2$) and pointwise channel mixing ($M \cdot N \cdot D_f^2$). Dividing the separable cost by standard cost yields $\frac{1}{N} + \frac{1}{D_k^2}$. For $3 \times 3$ kernels ($D_k = 3$) and large $N \ge 64$, $\frac{1}{N} \approx 0$ and $\frac{1}{9} \approx 0.111$, representing an $88.9\%$ reduction in multiply-accumulate operations."*

### Q4: In Table 6.6 and Section 8.3 of the report, why did MobileNetV2 show high single-image latency (167.29 ms) on Tesla T4 compared to ResNet-50?
> **Answer:** *"This reveals the hardware execution disconnect between theoretical FLOPs and GPU runtime. Server GPUs like the Tesla T4 are optimized for high arithmetic intensity dense matrix multiplies (GEMM). Depthwise convolutions have low arithmetic intensity (low FLOPs per memory byte transferred) and split operations into separate depthwise and pointwise kernels, doubling CUDA kernel dispatch overhead at batch size 1. However, on mobile NPUs and edge hardware (such as Apple Neural Engine or Google Edge TPU) where depthwise operations execute in on-chip SRAM cache, MobileNetV2 runs natively in under 10 ms."*

### Q5: Why did MobileNetV2 achieve higher accuracy on the test set (68.69%) than the validation set (64.73%)?
> **Answer:** *"This is due to the structure of the Food-101 benchmark. As documented by Bossard et al., training and validation sets were web-scraped and intentionally contain ~20% label noise, occlusions, and background clutter. However, the 25,250 test images were manually inspected and verified. When evaluating our well-regularized network on clean ground truth labels, accuracy naturally increases."*

---

## 5. Live Recording Step-by-Step Checklist

- [ ] **Desktop Windows Pre-arranged:**
  - **Window 1 (Slides / Report):** Slide 8 (or report Figure 6.7) open on full screen.
  - **Window 2 (VS Code):** [`src/models/mobilenetv2.py`](file:///d:/Group%20Projects/food-classification-deep-learning/src/models/mobilenetv2.py) open, font size zoomed to 16–18pt.
  - **Window 3 (Google Colab):** [`notebooks/05_mobilenetv2.ipynb`](file:///d:/Group%20Projects/food-classification-deep-learning/notebooks/05_mobilenetv2.ipynb) scrolled to Phase 1/2 training and Step 8 test evaluation output.
- [ ] **OBS Studio Settings:**
  - Video capture: Entire Desktop Display (1080p, 60fps) to ensure smooth `Alt + Tab` transitions.
  - Webcam source: Picture-in-picture in upper-right corner ($320 \times 180\text{ px}$).
  - Audio filters: RNNoise suppression and Limiter (-1.0 dB).
- [ ] **Timer:** Phone or desk timer set to exactly **2:30**.
- [ ] **Rehearsal:** Read through the script out loud 2 times while practicing the `Alt + Tab` window transitions. Verify that your delivery lands between **2:25 and 2:32**.
- [ ] **Handoffs:** Practice receiving cleanly from Matheesha and handing over seamlessly to Kaveesha at 07:45.
