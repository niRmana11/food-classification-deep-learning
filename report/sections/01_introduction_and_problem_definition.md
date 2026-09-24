# Section 1: Introduction and Problem Definition

**Author:** Nirmana (Group Leader / Member 1)  
**Assigned Workstream:** Repository Scaffolding, Data Pipeline, EDA, Custom CNN, Integration  
**Academic Module:** SE4050 — Deep Learning (2026)  
**Institution:** Sri Lanka Institute of Information Technology (SLIIT)  

---

## 1.1 Context and Real-World Motivation

Food image classification is an important application of computer vision in which a machine learning model identifies the food category represented in an image. Recent developments in deep learning, particularly convolutional neural networks (CNNs), have enabled increasingly sophisticated visual recognition systems that can operate on complex real-world imagery. Automated food recognition has significant practical applications in computational dietary assessment, clinical healthcare, mobile food logging, automated retail, food service environments, and culinary information retrieval systems.

One critical application is **computational dietary assessment**. Conventional dietary monitoring depends heavily on users manually recording the food and beverages they consume. This manual logging process is time-consuming and prone to human reporting errors through incomplete self-reporting or inaccurate dish identification [1]. An automated food recognition system provides a foundational stage for image-based dietary assessment by identifying food categories directly from user photographs. Once a food item has been recognized, nutritional data (such as estimated calories, carbohydrates, and macronutrient ratios) can be mapped to the predicted category. Technology-assisted dietary assessment has therefore been widely investigated as a mechanism for reducing patient burden and improving dietary compliance [1].

Food recognition also has direct applications in **mobile health and digital food diaries**. Rather than requiring users to manually search extensive food databases, a mobile application can analyze meal photographs and instantly suggest candidate categories. Such functionality accelerates dietary tracking and supports preventative health monitoring for chronic conditions such as obesity and Type-2 diabetes.

Another valuable commercial application lies in **automated food service and retail environments**. In cafeterias, smart canteens, and self-service dining facilities, vision-based food recognition can identify plated dishes to support automated cashierless checkout. Furthermore, automated food tagging assists commercial delivery and discovery platforms (e.g., Yelp, UberEats) in categorizing unstructured food photographs for culinary search and personalized recommendation.

Despite these compelling applications, reliable food recognition remains a fundamentally challenging computer vision problem. Food does not generally possess the rigid geometric structure found in manufactured objects. Visually distinct dishes may differ only through subtle characteristics such as ingredient composition, cooking methods, surface texture, or plating style. These domain characteristics make food classification an exceptionally rich problem for evaluating different deep learning architectures and their ability to extract discriminative representations under uncontrolled visual conditions.

---

## 1.2 Technical Problem Definition and Domain Complexity

The task investigated in this study is formulated as a supervised multi-class image classification problem. Given an input food photograph $x \in \mathcal{X}$, where $\mathcal{X} \subset \mathbb{R}^{H \times W \times C}$ denotes the space of RGB images with height $H$, width $W$, and $C=3$ color channels, the objective is to learn a parameterized mapping function $f_\theta: \mathcal{X} \to \mathcal{Y}$ that assigns the image to one of $K = 101$ mutually exclusive food categories $\mathcal{Y} = \{1, 2, \dots, K\}$.

The classifier can be expressed as:

$$\hat{y} = f(x; \theta)$$

where $x$ represents the input image tensor, $\theta$ denotes the trainable model parameters, and $\hat{y} \in \mathcal{Y}$ represents the predicted food class label.

For a multi-class neural network classifier, the final dense layer produces an unnormalized logit vector $z = [z_1, z_2, \dots, z_K]^T \in \mathbb{R}^K$. Applying the softmax activation function yields a posterior class probability distribution:

$$P(Y = k \mid x; \theta) = \frac{\exp(z_k)}{\sum_{j=1}^K \exp(z_j)}, \quad \text{for } k \in \{1, \dots, K\}$$

The predicted category is then determined via the maximum a posteriori (MAP) decision rule:

$$\hat{y} = \arg\max_{k \in \{1, \dots, K\}} P(Y = k \mid x; \theta)$$

The primary scientific challenge is to learn representations that remain robustly discriminative despite substantial variations within and between categories. Specifically, food recognition is characterized by four primary complexities:

### 1. High Intra-Class Variation
Food categories do not exhibit fixed canonical geometries. Images belonging to the same semantic category differ considerably depending on culinary ingredients, cooking techniques, toppings, portion sizes, plating vessels, lighting conditions, camera viewpoints, and dining backgrounds. For example, two images labeled as `pizza` may feature completely distinct toppings (e.g., Margherita vs. deep-dish meat pizza), slice geometries, crust thicknesses, and visual compositions. Similar high variance occurs in categories such as `salad`, `fried_rice`, and `soup`. Consequently, models must learn deep semantic abstractions rather than memorizing superficial color or shape patterns.

### 2. Fine-Grained Inter-Class Similarity
Food classification frequently requires distinguishing between categories that share near-identical color palettes, surface textures, and ingredients. Within Food-101, notable confusion pairs include `dumplings` versus `gyoza`, `steak` versus `filet_mignon`, and `apple_pie` versus `bread_pudding`. In these scenarios, inter-class differences are subtle, requiring the network to capture fine-grained localized and contextual cues. This makes the domain more closely related to fine-grained visual classification than standard generic object categorization.

### 3. Non-Rigid and Deformable Visual Structures
Unlike industrial manufactured objects that possess rigid bounding envelopes, food items are inherently amorphous and non-rigid. Individual food components frequently blend or overlap, while sauces, garnishes, tableware, and cutlery partially occlude the primary dish. Variations in presentation substantially alter the visual appearance of the same category, obligating the classifier to learn invariant spatial and texture hierarchies rather than relying on sharp object boundaries.

### 4. Real-World Image and Label Noise
Food-101 was curated from unconstrained web sources (*foodspotting.com*), and its training images were intentionally uncleaned. Consequently, the training data contains natural visual noise, mislabeled items, multi-dish plates, and ambient dining clutter. This introduces an authentic machine learning challenge: models must possess sufficient architectural regularization to generalize without memorizing noisy training instances.

These characteristics collectively provide an ideal testbed for evaluating architectures with fundamentally distinct design philosophies. This study therefore systematically compares a scratch-trained **Custom CNN** baseline against three prominent pretrained architectures: **ResNet-50**, **MobileNetV2**, and **EfficientNetB0**.

---

## 1.3 Central Research Question

This study is guided by the following central research question:

> *"How do fundamentally distinct deep convolutional neural network architectures—specifically a scratch-trained Custom CNN, ResNet-50, MobileNetV2, and EfficientNetB0—compare for multi-class food image classification in terms of predictive performance, generalization, computational efficiency, model complexity, inference latency, and fine-grained classification error patterns?"*

The research question intentionally evaluates dimensions beyond simple top-1 accuracy. For practical deployment in mobile health and real-time retail systems, a model must be critically judged on its generalization capacity to unseen data, parameter efficiency, memory footprint, inference speed, and robustness across fine-grained confusion clusters.

---

## 1.4 Research Objectives and Hypotheses

The primary objective of this study is to implement and comparatively benchmark four CNN architectures under a strictly standardized experimental protocol. The comparison centers on predictive accuracy, generalization dynamics, computational cost, parameter complexity, training stability, and classification error topologies.

Four scientific hypotheses are formulated to guide the experimental investigation:

* **Hypothesis 1 (H1 — Transfer Learning Generalization):**  
  The pretrained backbones (ResNet-50, MobileNetV2, and EfficientNetB0) will leverage visual representations learned from large-scale natural imagery (ImageNet) to achieve faster convergence, reduced training loss, and significantly higher generalization accuracy than the Custom CNN trained from random initialization on complex food textures.

* **Hypothesis 2 (H2 — Efficiency and Accuracy Trade-Off):**  
  MobileNetV2 will exhibit a superior computational efficiency profile due to its inverted residual blocks and depthwise separable convolutions. The experiment investigates whether this lightweight footprint can deliver rapid inference and low memory consumption while maintaining competitive classification performance.

* **Hypothesis 3 (H3 — EfficientNetB0 and Compound Scaling):**  
  EfficientNetB0 will achieve a balanced trade-off between predictive accuracy and computational cost. The investigation tests whether its compound scaling methodology (jointly scaling depth, width, and resolution) provides classification performance competitive with deeper networks (ResNet-50) while operating with substantially fewer parameters and FLOPs.

* **Hypothesis 4 (H4 — Structured Classification Errors):**  
  Classification errors will not be uniformly distributed across the 101 classes, but will concentrate heavily in semantic clusters exhibiting culinary proximity, shared ingredients, or similar preparation styles. Confusion matrix analysis and class-level metrics will be evaluated to test this structured error hypothesis.

---

## 1.5 Dataset Selection and Academic Justification

The benchmark selected for this research is the **Food-101** dataset, introduced by Bossard, Guillaumin, and Van Gool at the *European Conference on Computer Vision (ECCV) 2014* [2]. Food-101 represents an established benchmark for fine-grained culinary recognition, consisting of 101 food categories and exactly 101,000 images.

Each class contains exactly 1,000 images. The original dataset partition provides 750 training images and 250 test images per category:
* **Original Training Set:** 75,750 images (750 per category)
* **Original Test Set:** 25,250 images (250 per category)
* **Total Volume:** 101,000 images across 101 categories

---

### Table 1.1: Comprehensive Overview of the Food-101 Benchmark Dataset

| Attribute | Specification | Academic / Technical Justification |
| :--- | :--- | :--- |
| **Dataset Name** | Food-101 | Standard academic benchmark in culinary vision literature |
| **Original Source** | ETH Zürich Computer Vision Lab (CVL) | High credibility, publicly accessible, fully citable [2] |
| **Creators** | Lukas Bossard, Matthieu Guillaumin, Luc Van Gool | Introduced at ECCV 2014 |
| **Official URL** | `https://data.vision.ee.ethz.ch/cvl/datasets_extra/food-101/` | Open-access research archive |
| **Total Images** | 101,000 RGB photographs | Sufficient scale for deep neural network training |
| **Number of Classes** | 101 distinct culinary categories | Covers extensive international diversity |
| **Class Distribution** | Uniform (1,000 images per category) | Eliminates class-imbalance bias and loss skew |
| **Original Split** | 75,750 Training / 25,250 Test | Official partition established by authors |
| **Study Split** | 68,175 Train / 7,575 Val / 25,250 Test | Canonical 90/10 split on training set; test kept unseen |
| **Input Format** | Variable-resolution JPEG images | Reflects uncurated smartphone camera uploads |
| **Color Channels** | 3 (Standard RGB) | Standard format for 2D convolutional networks |

---

Food-101 was selected based on three critical academic criteria:

1. **Authentic Academic Benchmark:** Food-101 is a recognized computer vision benchmark rather than a synthetic, lecture-based, or simplified tutorial dataset (e.g., MNIST, CIFAR-10). It provides an authentic standard for evaluating model behavior.
2. **Sufficient Real-World Complexity:** The dataset exhibits high natural variance across illumination, camera viewpoints, dining backgrounds, food textures, and dish plating. This complexity allows meaningful comparative analysis of model generalization and error distributions.
3. **Strict Data Partitioning and Leakage Prevention:** The dataset includes a predefined, standardized test partition. In this study, the original 25,250 test images are strictly isolated and remain unseen during all training and hyperparameter tuning. The original 75,750 training portion was partitioned into canonical training and validation subsets:

$$\mathcal{D}_{\text{train}} = 68,175 \text{ images } (90\%), \quad \mathcal{D}_{\text{val}} = 7,575 \text{ images } (10\%)$$

while the final test split remains:

$$\mathcal{D}_{\text{test}} = 25,250 \text{ images } (\text{Untouched})$$

A fixed random seed of `42` was enforced to ensure the data partitioning is 100% reproducible across all team members.

The historical Food-101 literature provides valuable context regarding task difficulty. In the original 2014 study, Bossard et al. reported a **50.76%** accuracy using their Random Forest Discriminative Components (RFDC) method, and **56.40%** using an early CNN architecture [2]. These published benchmarks serve as comparative baselines for our experimental findings.

---

## 1.6 Key Technical Contributions of This Study

This investigation delivers five core engineering and experimental contributions:

### 1.6.1 Canonical Data Partitioning Protocol (Zero Data Leakage)
A deterministic partitioning script (`src/data/split_food101.py`) was implemented to enforce synchronized data usage across all four models. By isolating the 25,250 test images and generating stratified manifests (`data/splits/train.txt`, `val.txt`, `test.txt`) with seed `42`, we guarantee that experimental differences reflect model architecture rather than data sampling disparities.

### 1.6.2 Unified Preprocessing and Augmentation Framework
A standardized, multithreaded input pipeline (`src/preprocessing/data_loader.py`) was engineered using `tf.data`. Images are bilinearly interpolated to $224 \times 224 \times 3$. Training-only augmentation (random horizontal flipping, $\pm 5\%$ rotation, $\pm 10\%$ zoom) is applied strictly to $\mathcal{D}_{\text{train}}$, while validation and test sets are processed deterministically. Model-specific scaling (e.g., Caffe BGR centering for ResNet-50 vs. $[-1, 1]$ normalization for MobileNetV2) is handled within the common factory.

### 1.6.3 GPU Feasibility Profiling
Prior to full-scale training, an empirical feasibility benchmark was executed on Google Colab using an NVIDIA T4 GPU (16 GB VRAM). Profiling confirmed that a batch size of 32 consumes **4,217 MB** of VRAM (27.5% capacity) with an input throughput of **77.7 images/second**, verifying that training would operate without out-of-memory (OOM) failures under cloud constraints.

### 1.6.4 Scratch-Trained Custom CNN Baseline
A custom 4-stage convolutional neural network was designed, compiled, and trained from scratch (`src/models/custom_cnn.py`). Incorporating Batch Normalization, dual Dropout (0.4/0.3), and Global Average Pooling (GAP), the model maintains a compact footprint of **678,085 parameters**. When evaluated on the unseen test set, the Custom CNN achieved **61.69% Test Top-1 Accuracy** and **86.19% Test Top-5 Accuracy**, substantially outperforming the historical Random Forest baseline (50.76% [2]).

### 1.6.5 Multi-Dimensional Comparative Evaluation
This study departs from one-dimensional accuracy metrics by profiling models across:
* **Predictive Performance:** Top-1 Accuracy, Top-5 Accuracy, Precision, Recall, Macro F1-score.
* **Error Structure:** 101-class Normalized Confusion Matrices and inter-class error pairs.
* **Computational Efficiency:** Total parameters, trainable parameters, storage size (MB), training wall-clock time, and inference latency (ms/image).

---

### Figure 1.1: End-to-End Experimental Workflow and System Architecture

```
                      [ Food-101 Dataset: 101,000 Images ]
                                       │
                  ┌────────────────────┴────────────────────┐
                  ▼                                         ▼
      [ Original Train: 75,750 ]                [ Original Test: 25,250 ]
                  │                                  (Strictly Isolated)
          ┌───────┴───────┐                                 │
          ▼               ▼                                 │
     Train (90%)      Val (10%)                             │
     (68,175 imgs)   (7,575 imgs)                           │
          │               │                                 │
          └───────┬───────┘                                 │
                  ▼                                         │
     [ Shared tf.data Pipeline ]                            │
     - Bilinear Resizing: 224 x 224 x 3                     │
     - Train-Only Augmentation (Flip, Rot, Zoom)            │
     - Architecture-Specific Mathematical Scalers           │
                  │                                         │
        ┌─────────┼─────────┬─────────┐                     │
        ▼         ▼         ▼         ▼                     │
    [Custom   [ResNet-50] [MobileNet [Efficient             │
      CNN]                  V2]       NetB0]                │
     (Scratch) (Transfer) (Transfer) (Transfer)             │
        │         │         │         │                     │
        └─────────┼─────────┴─────────┘                     │
                  ▼                                         │
      Validation Performance                                │
      & Hyperparameter Tuning Check                         │
                  │                                         │
                  └────────────────────┬────────────────────┘
                                       ▼
                       [ Final Unseen Test Benchmark ]
                       - Top-1 and Top-5 Test Accuracy
                       - Macro Precision, Recall, F1
                       - 101x101 Confusion Matrix
                       - Inference Latency (ms/image)
                       - Parameter Count & Model Size (MB)
```

*Figure 1.1: High-level architectural flowchart of the experimental methodology, illustrating data ingestion, canonical split isolation, shared multithreaded preprocessing, independent model training on Google Colab T4 GPU, and centralized multi-dimensional evaluation on the unseen test set.*

---

## 1.7 Organization of the Report

The remainder of this report is organized as follows:
* **Section 2 (Background & Related Work):** Reviews the theoretical foundations of convolutional neural networks, residual architectures, inverted bottlenecks, compound scaling, and computational dietary assessment literature.
* **Section 3 (Dataset Description & Exploratory Data Analysis):** Analyzes class distributions, spatial resolution histograms, visual diversity, intra-class variance, and data quality audits.
* **Section 4 (Data Preprocessing & Feature Engineering):** Formulates the data partitioning protocol, leakage prevention guarantees, training augmentation policies, and backbone-specific input scalers.
* **Section 5 (Experimental Design & Setup):** Specifies the hardware environment, loss formulations, optimization schedules, and evaluation metrics.
* **Section 6 (Model Architectures):** Details the mathematical formulations and layer configurations of the Custom CNN, ResNet-50, MobileNetV2, and EfficientNetB0.
* **Section 7 (Results & Model Comparison):** Reports empirical results across predictive metrics, learning curves, and computational efficiency tables.
* **Section 8 (Critical Analysis & Discussion):** Explores generalization gaps, convergence dynamics, confusion clusters, error topologies, and mobile edge deployment trade-offs.
* **Section 9 (Conclusion & Future Work):** Synthesizes findings, evaluates research hypotheses, and outlines future research directions.
* **Section 10 (References):** Provides formal academic citations conforming to IEEE standards.

---

## References for Section 1

```text
[1] F. Zhu et al., "Technology-assisted dietary assessment," Proceedings of the Nutrition Society, vol. 74, no. 4, pp. 408-414, 2015.
[2] L. Bossard, M. Guillaumin, and L. Van Gool, "Food-101—mining discriminative components with random forests," in European Conference on Computer Vision (ECCV). Springer, 2014, pp. 446-461.
[3] K. He, X. Zhang, S. Ren, and J. Sun, "Deep residual learning for image recognition," in IEEE CVPR, 2016, pp. 770-778.
[4] M. Sandler, A. Howard, M. Zhu, A. Zhmoginov, and L.-C. Chen, "MobileNetV2: Inverted residuals and linear bottlenecks," in IEEE CVPR, 2018, pp. 4510-4520.
[5] M. Tan and Q. V. Le, "EfficientNet: Rethinking model scaling for convolutional neural networks," in ICML, 2019, pp. 6105-6114.
```
