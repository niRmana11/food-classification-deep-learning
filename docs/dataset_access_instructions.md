# Food-101 Dataset Access Instructions & Benchmark Documentation

## 1. Dataset Overview & Academic Provenance

The **Food-101** visual recognition dataset is an authentic, large-scale multi-class benchmark for fine-grained food classification. It was collected, curated, and published by the **Computer Vision Laboratory (CVL) at ETH Zürich** and introduced at the European Conference on Computer Vision (ECCV) 2014 by Lukas Bossard, Matthieu Guillaumin, and Luc Van Gool [1].

### Key Specifications:

- **Total Images:** 101,000 RGB images
- **Number of Categories:** 101 distinct culinary categories
- **Distribution:** Exactly 1,000 images per class (perfectly class-balanced)
- **Image Resolution:** Longest edge rescaled to 512 pixels; variable aspect ratios
- **Label Noise Characteristic:** The training images intentionally contain $\sim 20\%$ natural web-crawled noise (mislabelled dishes, occluded toppings), while the held-out test split was manually verified and cleaned by human annotators.

---

## 2. Official Dataset Links & Alternative Mirrors

The Food-101 dataset is publicly available through the following official sources and mirrors:

| Source / Provider                | Link / Access URL                                                                               |    Archive Type & Size    | Description                                                                                        |
| :------------------------------- | :---------------------------------------------------------------------------------------------- | :-----------------------: | :------------------------------------------------------------------------------------------------- |
| **ETH Zürich (Official Portal)** | [ETH Zürich Food-101 Project Page](https://data.vision.ee.ethz.ch/cvl/datasets_extra/food-101/) |        Web Portal         | Official hosting portal containing dataset description, benchmarks, and original research details. |
| **ETH Zürich (Direct Tarball)**  | [Download `food-101.tar.gz`](http://data.vision.ee.ethz.ch/cvl/food-101.tar.gz)                 |   `.tar.gz` (~4.65 GB)    | Primary direct download link used by our automated download pipeline.                              |
| **Kaggle Datasets Mirror**       | [Kaggle: Food-101 Dataset by Dan Becker](https://www.kaggle.com/datasets/dansbecker/food-101)   | Cloud / Archive (~5.0 GB) | High-speed cloud mirror ideal for Kaggle Notebooks and fast local downloads.                       |
| **Hugging Face Datasets**        | [Hugging Face: `ethz/food101`](https://huggingface.co/datasets/ethz/food101)                    |  Parquet / Arrow Streams  | Standard Hugging Face hub mirror for fast streaming and exploration.                               |
| **TensorFlow Datasets (TFDS)**   | [TFDS Catalog: `food101`](https://www.tensorflow.org/datasets/catalog/food101)                  |     TFRecord Builder      | Google's standardized TensorFlow dataset catalog module.                                           |

---

## 3. Automated Download via Repository Pipeline

This repository includes a fully automated downloader with resume-check, stream progress visualization (`tqdm`), and extraction verification in [`src/data/download_food101.py`](file:///c:/Users/USER/OneDrive/Documents/Projects/food-classification-deep-learning/src/data/download_food101.py).

### Execution Command:

From the root of the repository, execute:

```bash
# Using python module execution
python -m src.data.download_food101
```

Or directly via script path:

```bash
python src/data/download_food101.py
```

### What the Script Does Automatically:

1. Verifies if `data/raw/food-101/images/` already exists; if present, skips downloading immediately.
2. If absent, downloads `http://data.vision.ee.ethz.ch/cvl/food-101.tar.gz` (~4.65 GB) into `data/raw/` with a live progress bar.
3. Automatically unpacks the archive into `data/raw/food-101/`.
4. Validates the existence of all 101 category image directories.

> **Colab Note:** When executing inside Google Colab (as demonstrated in `notebooks/01_food101_eda.ipynb` through `06_efficientnetb0.ipynb`), downloading directly over Google's cloud backbone takes approximately **1 to 2 minutes**.

---

## 4. Manual Download & Setup Instructions

If you prefer to download the archive manually through a browser or download manager:

1. Download the archive from ETH Zürich:
   - URL: `http://data.vision.ee.ethz.ch/cvl/food-101.tar.gz`
2. Place the downloaded `food-101.tar.gz` file into:
   ```
   food-classification-deep-learning/data/raw/
   ```
3. Extract the archive into `data/raw/`:
   - **Linux / macOS:**
     ```bash
     cd data/raw
     tar -xzf food-101.tar.gz
     ```
   - **Windows (PowerShell):**
     ```powershell
     cd data/raw
     tar -xzf food-101.tar.gz
     ```
   - **Windows (GUI):** Right-click `food-101.tar.gz` $\to$ Extract here using 7-Zip or WinRAR.

---

## 5. Expected Directory Layout

Once downloaded and extracted, the repository directory layout must match the following structure:

```text
food-classification-deep-learning/
├── data/
│   ├── raw/
│   │   ├── food-101.tar.gz           # Original archive (~4.65 GB, can be deleted after extract)
│   │   └── food-101/
│   │       ├── images/               # 101 category subdirectories
│   │       │   ├── apple_pie/        # 1,000 images (e.g., 1005649.jpg)
│   │       │   ├── baby_back_ribs/   # 1,000 images
│   │       │   ├── baklava/          # 1,000 images
│   │       │   └── ... (98 more classes)
│   │       ├── license_agreement.txt # Original ETH Zürich license
│   │       ├── README.txt            # ETH Zürich dataset notes
│   │       └── meta/                 # Original text metadata
│   │           ├── classes.txt       # List of 101 class identifiers
│   │           ├── labels.txt        # Display names
│   │           ├── train.json        # Original train JSON
│   │           ├── train.txt         # Original train image list (75,750 images)
│   │           ├── test.json         # Original test JSON
│   │           └── test.txt          # Original test image list (25,250 images)
│   └── splits/                       # Zero-leakage reproducible splits (generated by split_food101.py)
│       ├── classes.txt               # Canonical 101 class names
│       ├── train.txt                 # 68,175 training samples (90% of original train)
│       ├── val.txt                   # 7,575 validation samples (10% of original train)
│       ├── test.txt                  # 25,250 untouched test samples
│       └── split_summary.json        # Manifest metadata with locked seed: 42
```

---

## 6. Generating the Zero-Leakage Dataset Splits

To prevent test-set leakage and guarantee reproducible experiments across all four architectures, we partition the original 75,750 training images into **68,175 training** and **7,575 validation** samples (90/10 split) while preserving the canonical **25,250 test** images completely untouched:

```bash
# Generate deterministic splits using global seed 42
python -m src.data.split_food101
```

Verification of the generated manifests is persisted in [`data/splits/split_summary.json`](file:///c:/Users/USER/OneDrive/Documents/Projects/food-classification-deep-learning/data/splits/split_summary.json):

```json
{
  "dataset": "Food-101",
  "num_classes": 101,
  "total_images": 101000,
  "seed": 42,
  "train_images": 68175,
  "val_images": 7575,
  "test_images": 25250
}
```

---

## 7. Python Code Snippet: Loading Datasets in Code

Our unified data loader [`src/preprocessing/data_loader.py`](file:///c:/Users/USER/OneDrive/Documents/Projects/food-classification-deep-learning/src/preprocessing/data_loader.py) ingests these splits with multithreaded decoding, model-specific normalization, and training-only data augmentation:

```python
from src.preprocessing.data_loader import get_food101_datasets

# Load train, validation, and test datasets with strict isolation
train_ds, val_ds, test_ds = get_food101_datasets(
    data_dir="data/raw/food-101",
    splits_dir="data/splits",
    model_type="custom_cnn",    # Options: "custom_cnn", "resnet50", "mobilenetv2", "efficientnetb0"
    image_size=(224, 224),
    batch_size=32
)

# Verify batch shapes
for images, labels in train_ds.take(1):
    print(f"Batch Image Tensor: {images.shape}")  # (32, 224, 224, 3)
    print(f"Batch Label Tensor: {labels.shape}")  # (32, 101) (one-hot encoded)
```

---

## 8. System & Storage Requirements

- **Disk Space:**
  - Compressed Archive (`food-101.tar.gz`): $\sim 4.65\text{ GB}$
  - Uncompressed Image Directory: $\sim 5.10\text{ GB}$
  - Total Recommended Free Storage: **$\ge 12.0\text{ GB}$** (during download and extraction)
- **RAM:** Minimum $8\text{ GB}$ (16 GB recommended for high-throughput disk-to-RAM prefetching).
- **GPU:** Dedicated GPU with $\ge 4\text{ GB}$ VRAM (NVIDIA T4 on Colab or local GTX 1650 / RTX series). Batch size 32 requires approximately $4.2\text{ GB}$ peak VRAM.

---

## 9. Academic License & Citation

The Food-101 dataset is released under the **Creative Commons Attribution-NonCommercial-ShareAlike 3.0 (CC BY-NC-SA 3.0)** license. It is intended strictly for academic research and educational evaluation.

If referencing this dataset, please cite the original ECCV 2014 paper:

```bibtex
@inproceedings{bossard14,
  author    = {Lukas Bossard and Matthieu Guillaumin and Luc {Van Gool}},
  title     = {Food-101 -- Mining Discriminative Components with Random Forests},
  booktitle = {Proceedings of the European Conference on Computer Vision (ECCV)},
  year      = {2014},
  pages     = {446--461},
  doi       = {10.1007/978-3-319-10599-4_29}
}
```
