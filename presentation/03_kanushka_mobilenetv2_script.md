# SE4050 Deep Learning — Member 3 (Kanushka) Presentation Script & Guide

**Presenter:** Kanushka Kahakotuwa (Member 3)  
**Assigned Architecture:** MobileNetV2 (Inverted Residual and Linear Bottleneck CNN)  
**Allotted Video Time:** `05:15 – 07:45` (Strictly **2 Minutes 30 Seconds** / 150 Seconds)  
**Word Count Target:** **335 Words** (~135–140 words/minute — leaves 10–15 seconds for smooth window switching)  
**Presentation Strategy:** **Live Multi-Evidence Demonstration** (Architecture Diagram $\to$ VS Code Repo $\to$ Google Colab Notebook)

---

## 1. The Pre-Positioned Window Setup (Do This BEFORE Pressing Record!)

To prevent wasting time searching or scrolling during the recording, have these **3 exact windows open side-by-side or stacked in your Alt-Tab order**:

* **Window 1 (Architecture):** Report Page 24 **Figure 6.7** (or Slide 8) maximized in full screen.
* **Window 2 (Code Evidence):** VS Code open to [`src/models/mobilenetv2.py`](file:///d:/Group%20Projects/food-classification-deep-learning/src/models/mobilenetv2.py), pre-scrolled to **lines 70–95** (where `MobileNetV2` base, GAP, and classification head are defined).
* **Window 3 (Execution Evidence):** Google Colab open to [`notebooks/05_mobilenetv2.ipynb`](file:///d:/Group%20Projects/food-classification-deep-learning/notebooks/05_mobilenetv2.ipynb), pre-scrolled directly to **Step 8 (Test Evaluation output cell showing `68.69%` and `training_curves.png`)**.

> 💡 **The Secret to 2:30:** You only press `Alt + Tab` **TWICE** in the entire 2.5 minutes! Never scroll searching for code while recording.

---

## 2. Multi-Screen Timeline & Evidence Cue Sheet

| Video Time | Target Seconds | Active Application On Screen | Live Cursor Action | Spoken Content |
| :--- | :---: | :--- | :--- | :--- |
| **05:15 – 06:00** | **45 sec** | **Window 1:** Report Figure 6.7 / Slide 8 | Point cursor to Box 1 ($1\times1$ expand), Box 2 ($3\times3$ Depthwise), Box 3 (Linear $1\times1$ NO ReLU), and top shortcut arrow. | Inverted residual concept ($t=6$), memory efficiency, and manifold preservation. |
| **06:00 – 06:45** | **45 sec** | **Window 2:** VS Code (`src/models/mobilenetv2.py`) | *Press `Alt+Tab` once.* Highlight lines 70–85. Point to FLOP math formula $\frac{1}{N} + \frac{1}{D_k^2}$ in the docstring. | Depthwise Separable factorization, 88% compute reduction, and parameter drop to 2.39M. |
| **06:45 – 07:45** | **60 sec** | **Window 3:** Google Colab (`05_mobilenetv2.ipynb`) | *Press `Alt+Tab` once.* Point to Step 5 (Phase 1 & 2 logs), training curve plot, and Step 8 test metrics (`68.69%`, `90.15%`). | 2-phase training, frozen BatchNorm, 25,250 test evaluation, 9.11 MB edge footprint, handover to Kaveesha. |

---

## 3. Word-for-Word Spoken Script with Live Evidence Cues

---

### 📍 [05:15 – 06:00] Scene 8: Architecture & Manifold Theory (Report Figure 6.7 / Slide 8)

* **[SCREEN: WINDOW 1 — FULL SCREEN REPORT FIGURE 6.7 OR SLIDE 8]**  
* **[WEBCAM: Upper-right corner, looking into the camera lens]**

> *"Thank you, Matheesha.*
>
> *While ResNet-50 achieves strong classification accuracy, deploying 24-million parameters to edge devices is computationally prohibitive. For Member 3, I evaluated **MobileNetV2**.*
>
> *As shown in **Figure 6.7**, MobileNetV2 inverts classical residuals through a **narrow-to-wide-to-narrow** flow:*
> 1. *First, an expansion one-by-one convolution **[POINT CURSOR TO FIRST BOX]** expands channel depth six-fold ($t=6$) for rich spatial filtering.*
> 2. *Next, a three-by-three depthwise convolution **[POINT CURSOR TO MIDDLE BOX]** performs lightweight spatial filtering per channel.*
> 3. *Finally, a pointwise one-by-one convolution **[POINT CURSOR TO THIRD BOX]** projects features back down to a compact bottleneck.*
>
> *Crucially, this bottleneck is **strictly linear with no ReLU**. Non-linearities in low dimensions destroy visual manifold data. Preserving linearity retains delicate food textures, while shortcut connections **[TRACE TOP SHORTCUT ARROW]** between thin bottlenecks drastically minimize memory residency during inference."*

---

### 📍 [06:00 – 06:45] Scene 9: Code Evidence & Depthwise Separable Math (VS Code)

* **[ACTION: PRESS ALT+TAB ONCE $\to$ WINDOW 2: VS CODE `src/models/mobilenetv2.py` PRE-SCROLLED]**  
* **[ACTION: HIGHLIGHT LINES 70–85 AND THE TOP DOCSTRING FORMULA]**

> *"Looking directly at our implementation in `src/models/mobilenetv2.py`, the core computational engine is **Depthwise Separable Convolution**, which completely decouples spatial filtering from channel mixing.*
>
> *Standard convolution performs both simultaneously, incurring heavy compute. MobileNetV2 factorizes this into two disciplined steps:*
> - *First, a **depthwise convolution** applies a three-by-three kernel to each channel independently.*
> - *Second, a **pointwise one-by-one convolution** linearly combines these spatial features across channels.*
>
> *As documented in Section 6.3 and shown in our code docstring **[POINT CURSOR TO MATH]**, this factorizes computational cost by **one over N plus one over D-k squared**. For three-by-three kernels, this achieves an **eighty-eight to eighty-nine percent FLOP reduction**, slashing total network parameters to just **2.39 million**."*

---

### 📍 [06:45 – 07:45] Scene 10: Colab Live Execution, Results & Handover (Google Colab)

* **[ACTION: PRESS ALT+TAB ONCE $\to$ WINDOW 3: GOOGLE COLAB `05_mobilenetv2.ipynb` PRE-SCROLLED]**  
* **[ACTION: POINT CURSOR TO STEP 5/6 TRAINING LOGS, THEN DOWN TO STEP 8 TEST EVALUATION]**

> *"Here in our Google Colab training notebook, you can see our standardized two-phase protocol:*
> - *In **Phase 1** **[POINT TO PHASE 1 CELLS]**, training only the classification head with a frozen backbone reached **57.03%** validation accuracy.*
> - *In **Phase 2** **[POINT TO PHASE 2 CODE & PLOT]**, unfreezing top blocks fourteen through seventeen—from layer 120 onward—under a learning rate of ten-to-the-minus-five propelled validation accuracy to **64.73%**. Crucially, we kept Batch Normalization layers frozen to safeguard ImageNet statistics.*
>
> *On the final, unseen Food-101 test set of 25,250 images **[POINT TO STEP 8 EVALUATION LOGS]**, MobileNetV2 achieved **68.69% Top-1 Accuracy** and **90.15% Top-5 Accuracy**, with a test loss of **1.1692** and macro F1 of **0.6862**.*
>
> *With an uncompressed disk footprint of only **9.11 megabytes**—ten times lighter than ResNet-50—and superior parameter efficiency of **28.77% accuracy per million parameters**, MobileNetV2 proves exceptionally viable for on-device mobile nutrition apps.*
>
> *I will now hand over to Kaveesha for EfficientNetB0 and the master cross-model comparison."*

**[END OF KANUSHKA'S SEGMENT — EXACT TIMER: 07:45]**

---

## 4. Rehearsal Pronunciation Quick-Reference

| Written Term | Spoken Form |
| :--- | :--- |
| **$t=6$** | *"expansion factor of six"* |
| **$3 \times 3$ and $1 \times 1$** | *"three-by-three and one-by-one"* |
| **$\frac{1}{N} + \frac{1}{D_k^2}$** | *"one over N plus one over D-k squared"* |
| **$\approx 88\text{--}89\%$** | *"eighty-eight to eighty-nine percent"* |
| **2,387,365** | *"two-point-three-nine million"* |
| **$\eta = 10^{-5}$** | *"ten to the minus five"* |
| **9.11 MB** | *"nine-point-one-one megabytes"* |
| **25,250** | *"twenty-five thousand two hundred and fifty"* |
| **68.69% / 90.15%** | *"sixty-eight point six nine percent / ninety point one five percent"* |
| **ReLU** | *"ray-loo"* |

---

## 5. Top 5 Viva Voce Questions & Answers (MobileNetV2)

### Q1: Why did you freeze Batch Normalization layers during Phase 2 fine-tuning?
> **Answer:** *"In fine-tuning, mini-batch sizes are small (batch size 32). If Batch Normalization layers are trainable, the batch statistics computed on small food batches overwrite the well-calibrated population statistics learned across 1.28 million ImageNet samples. Freezing Batch Normalization prevents internal covariate collapse, avoids gradient explosion, and ensures training stability under small learning rates."*

### Q2: Why is the bottleneck in MobileNetV2 'linear' instead of using a ReLU activation?
> **Answer:** *"According to the Manifold of Interest hypothesis, meaningful visual representations lie on low-dimensional manifolds embedded within high-dimensional space. Non-linear activations like ReLU set all negative coordinates to zero, which destroys information in low-dimensional spaces. By omitting non-linear activations from the final $1 \times 1$ projection layer, MobileNetV2 preserves the full geometric structure of the feature manifold."*

### Q3: How do Depthwise Separable Convolutions achieve ~88% computational savings?
> **Answer:** *"Standard convolution computes spatial filtering and channel mixing simultaneously with cost $D_k^2 \cdot M \cdot N \cdot D_f^2$. Depthwise separable convolution splits this into depthwise convolution ($D_k^2 \cdot M \cdot D_f^2$) followed by pointwise convolution ($M \cdot N \cdot D_f^2$). Dividing the separable cost by the standard cost yields $\frac{1}{N} + \frac{1}{D_k^2}$. For $3 \times 3$ kernels ($D_k = 3$) and $N \ge 64$, $\frac{1}{N} \approx 0$ and $\frac{1}{9} \approx 0.111$, representing an $88.9\%$ reduction in multiply-accumulate operations."*

### Q4: In Table 6.6 / Section 8.3 of the report, why did MobileNetV2 show high single-image latency (167.29 ms) on Tesla T4 compared to ResNet-50?
> **Answer:** *"This reflects the arithmetic intensity gap between theoretical FLOPs and GPU execution. Server GPUs like the Tesla T4 are optimized for dense matrix multiplies (GEMM). Depthwise convolutions have low arithmetic intensity (low FLOPs per memory byte transferred) and split operations into separate kernels, adding CUDA kernel dispatch overhead at batch size 1. However, on mobile NPUs and edge hardware (Apple Neural Engine, Edge TPUs) where depthwise operations run natively in on-chip SRAM cache, MobileNetV2 achieves sub-10 ms latency."*

### Q5: Why did MobileNetV2 achieve higher accuracy on the test set (68.69%) than the validation set (64.73%)?
> **Answer:** *"As documented by Bossard et al., Food-101 training and validation sets were web-scraped and intentionally contain ~20% label noise, occlusions, and background clutter. However, the 25,250 test images were manually inspected and cleaned. Evaluating our well-regularized model on clean ground truth labels naturally yields higher accuracy."*

---

## 6. Pre-Recording Checklist for the 2:30 Run

- [ ] **Arrange the 3 windows:**
  1. Open Figure 6.7 / Slide 8 full screen.
  2. Open VS Code to `src/models/mobilenetv2.py` (lines 70–95 visible).
  3. Open Google Colab `05_mobilenetv2.ipynb` (pre-scrolled to Step 8 test evaluation output).
- [ ] Test pressing `Alt + Tab` twice to ensure Windows cycles directly through `Window 1 -> Window 2 -> Window 3`.
- [ ] In OBS Studio, configure **Desktop Display Capture** + **Webcam PIP** (upper right corner, $320 \times 180\text{ px}$).
- [ ] Start your phone stopwatch at 0:00.
- [ ] Read through the script while doing the 2 `Alt+Tab` presses and cursor movements. You will land between **2:20 and 2:30** cleanly!
