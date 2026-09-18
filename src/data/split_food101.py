"""
Canonical Food-101 Stratified Split Generator (Zero Data Leakage Protocol)
"""

import json
import os
from collections import Counter
from pathlib import Path
import yaml
from sklearn.model_selection import train_test_split


def load_config(config_path: str = "configs/config.yaml") -> dict:
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


def generate_canonical_splits(
    food101_dir: str = "data/raw/food-101",
    output_dir: str = "data/splits",
    val_ratio: float = 0.10,
    seed: int = 42
):
    """
    Generates deterministic train, validation, and test split manifests:
    - 75,750 original training images -> 90% train (68,175) and 10% val (7,575)
    - 25,250 original test images -> untouched final test set (25,250)
    """
    base_path = Path(food101_dir)
    meta_path = base_path / "meta"
    out_path = Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    train_meta = meta_path / "train.txt"
    test_meta = meta_path / "test.txt"
    classes_meta = meta_path / "classes.txt"

    if not (train_meta.exists() and test_meta.exists() and classes_meta.exists()):
        raise FileNotFoundError(
            f"Could not find Food-101 meta files in {meta_path.resolve()}. "
            f"Ensure the dataset is extracted with the 'meta/' directory present."
        )

    # Read class labels
    with open(classes_meta, "r") as f:
        classes = [line.strip() for line in f if line.strip()]
    print(f"[INFO] Loaded {len(classes)} classes from {classes_meta.name}")

    # Read official test set (UNTOUCHED)
    with open(test_meta, "r") as f:
        test_samples = [line.strip() for line in f if line.strip()]
    print(f"[INFO] Preserving {len(test_samples)} test samples as the UNSEEN test set.")

    # Read official train set (to be split into train and validation)
    with open(train_meta, "r") as f:
        orig_train_samples = [line.strip() for line in f if line.strip()]
    print(f"[INFO] Loaded {len(orig_train_samples)} original training samples.")

    # Extract class label from each path (format: "class_name/image_id")
    train_labels = [path.split("/")[0] for path in orig_train_samples]

    # Perform stratified train/validation split
    train_samples, val_samples = train_test_split(
        orig_train_samples,
        test_size=val_ratio,
        stratify=train_labels,
        random_state=seed
    )

    # Sort lists for strict reproducibility
    train_samples.sort()
    val_samples.sort()
    test_samples.sort()

    print(f"[INFO] Stratified Split Generated:")
    print(f"       -> Train:      {len(train_samples):,} images ({len(train_samples)/len(orig_train_samples)*100:.1f}%)")
    print(f"       -> Validation: {len(val_samples):,} images ({len(val_samples)/len(orig_train_samples)*100:.1f}%)")
    print(f"       -> Final Test: {len(test_samples):,} images (Untouched)")

    # Verify class balance across splits
    train_class_counts = Counter([s.split("/")[0] for s in train_samples])
    val_class_counts = Counter([s.split("/")[0] for s in val_samples])
    test_class_counts = Counter([s.split("/")[0] for s in test_samples])

    min_train = min(train_class_counts.values())
    max_train = max(train_class_counts.values())
    min_val = min(val_class_counts.values())
    max_val = max(val_class_counts.values())

    print(f"[VERIFICATION] Images per class in Train: {min_train} to {max_train}")
    print(f"[VERIFICATION] Images per class in Val:   {min_val} to {max_val}")

    # Write manifests
    def write_manifest(file_path: Path, items: list):
        with open(file_path, "w") as f:
            for item in items:
                f.write(f"{item}\n")

    write_manifest(out_path / "train.txt", train_samples)
    write_manifest(out_path / "val.txt", val_samples)
    write_manifest(out_path / "test.txt", test_samples)
    write_manifest(out_path / "classes.txt", classes)

    # Write summary metadata for report evidence
    summary = {
        "dataset": "Food-101",
        "num_classes": len(classes),
        "total_images": len(train_samples) + len(val_samples) + len(test_samples),
        "seed": seed,
        "splits": {
            "train": {
                "count": len(train_samples),
                "percentage_of_train_split": round((1.0 - val_ratio) * 100, 2),
                "images_per_class_min": min_train,
                "images_per_class_max": max_train
            },
            "validation": {
                "count": len(val_samples),
                "percentage_of_train_split": round(val_ratio * 100, 2),
                "images_per_class_min": min_val,
                "images_per_class_max": max_val
            },
            "test": {
                "count": len(test_samples),
                "images_per_class": len(test_samples) // len(classes),
                "note": "Original test set preserved untouched to prevent data leakage"
            }
        }
    }

    with open(out_path / "split_summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    print(f"[SUCCESS] All manifests and split_summary.json generated in: {out_path.resolve()}")


if __name__ == "__main__":
    config = load_config()
    food101_dir = config["dataset"]["raw_dir"]
    splits_dir = config["dataset"]["splits_dir"]
    val_ratio = config["dataset"]["validation_split_ratio"]
    seed = config["project"]["seed"]

    generate_canonical_splits(
        food101_dir=food101_dir,
        output_dir=splits_dir,
        val_ratio=val_ratio,
        seed=seed
    )
