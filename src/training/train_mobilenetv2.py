"""
MobileNetV2 Training Pipeline (Phase 1 Feature Extraction & Phase 2 Fine-Tuning)
Author: Kanushka (Member 3)
Module: SE4050 - Deep Learning (2026)

Usage:
  # Phase 1 only (frozen base, classifier head training):
  python -m src.training.train_mobilenetv2 --phase 1 --epochs 8

  # Phase 2 only (requires best_phase1.keras):
  python -m src.training.train_mobilenetv2 --phase 2 --epochs 12

  # Both phases end-to-end:
  python -m src.training.train_mobilenetv2 --phase all

  # Quick pipeline verification dry-run:
  python -m src.training.train_mobilenetv2 --dry-run
"""

import argparse
import json
import os
import sys
import time
from pathlib import Path

try:
    import yaml
except ImportError:
    yaml = None

import pandas as pd


def parse_args():
    parser = argparse.ArgumentParser(description="Train MobileNetV2 on Food-101")
    parser.add_argument(
        "--phase",
        choices=["1", "2", "all"],
        default="all",
        help="Training phase: '1' (Feature extraction), '2' (Fine-tuning), or 'all'"
    )
    parser.add_argument("--epochs", type=int, default=None, help="Override epoch count")
    parser.add_argument("--batch-size", type=int, default=32, help="Batch size (default: 32)")
    parser.add_argument("--dry-run", action="store_true", help="Run 1 step per epoch for pipeline smoke testing")
    parser.add_argument("--config", type=str, default="configs/mobilenetv2.yaml", help="Path to config file")
    return parser.parse_args()


def load_yaml_config(path: str) -> dict:
    if not os.path.exists(path):
        return {}
    if yaml is not None:
        with open(path, "r") as f:
            return yaml.safe_load(f)
    return {}


def train():
    args = parse_args()
    cfg = load_yaml_config(args.config)

    # Ensure output directories exist
    results_dir = Path("results/mobilenetv2")
    results_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 70)
    print("  SE4050 FOOD-101 — MOBILENETV2 TRAINING PIPELINE (MEMBER 3: KANUSHKA)")
    print("=" * 70)
    print(f"[INFO] Target Phase: {args.phase.upper()}")
    print(f"[INFO] Dry Run Mode: {args.dry_run}")
    print(f"[INFO] Batch Size:   {args.batch_size}")

    import tensorflow as tf
    from src.models.mobilenetv2 import (
        build_mobilenetv2_model,
        setup_fine_tuning,
        get_model_parameter_stats,
    )
    from src.preprocessing.data_loader import get_food101_datasets

    # 1. Load Data
    print("\n[STEP 1] Loading Food-101 datasets with MobileNetV2 normalization [-1, 1]...")
    train_ds, val_ds, test_ds = get_food101_datasets(
        data_dir="data/raw/food-101",
        splits_dir="data/splits",
        model_type="mobilenetv2",
        image_size=(224, 224),
        batch_size=args.batch_size,
    )

    steps_per_epoch = 2 if args.dry_run else None
    val_steps = 2 if args.dry_run else None

    # 2. Phase 1: Feature Extraction
    history_phase1 = None
    if args.phase in ["1", "all"]:
        p1_epochs = args.epochs or cfg.get("phase1_feature_extraction", {}).get("epochs", 8)
        if args.dry_run:
            p1_epochs = 1

        print(f"\n[STEP 2] Phase 1: Training Classification Head ({p1_epochs} epochs)...")
        model = build_mobilenetv2_model(
            num_classes=101,
            input_shape=(224, 224, 3),
            dropout_rate=0.2,
            trainable_base=False,
            learning_rate=0.001,
        )

        stats = get_model_parameter_stats(model)
        print(f"[INFO] Model parameters: Total={stats['total_parameters']:,}, Trainable={stats['trainable_parameters']:,}")

        # Save model summary
        with open(results_dir / "model_summary.txt", "w") as f:
            model.summary(print_fn=lambda x: f.write(x + "\n"))

        cb_phase1 = [
            tf.keras.callbacks.ModelCheckpoint(
                str(results_dir / "best_phase1.keras"),
                monitor="val_loss",
                save_best_only=True,
                verbose=1,
            ),
            tf.keras.callbacks.EarlyStopping(
                monitor="val_loss",
                patience=4,
                restore_best_weights=True,
                verbose=1,
            ),
            tf.keras.callbacks.ReduceLROnPlateau(
                monitor="val_loss",
                factor=0.2,
                patience=2,
                min_lr=1e-5,
                verbose=1,
            ),
        ]

        t0 = time.time()
        history_phase1 = model.fit(
            train_ds,
            validation_data=val_ds,
            epochs=p1_epochs,
            steps_per_epoch=steps_per_epoch,
            validation_steps=val_steps,
            callbacks=cb_phase1,
        )
        duration_p1 = time.time() - t0
        print(f"[SUCCESS] Phase 1 completed in {duration_p1:.1f}s.")

    # 3. Phase 2: Fine-Tuning
    history_phase2 = None
    if args.phase in ["2", "all"]:
        p2_epochs = args.epochs or cfg.get("phase2_fine_tuning", {}).get("epochs", 12)
        if args.dry_run:
            p2_epochs = 1

        print(f"\n[STEP 3] Phase 2: Fine-Tuning Top Residual Blocks ({p2_epochs} epochs)...")
        if args.phase == "2":
            p1_checkpoint = results_dir / "best_phase1.keras"
            if p1_checkpoint.exists():
                print(f"[INFO] Loading weights from Phase 1 checkpoint: {p1_checkpoint}")
                model = tf.keras.models.load_model(str(p1_checkpoint))
            else:
                print("[WARNING] Phase 1 checkpoint not found. Instantiating fresh model.")
                model = build_mobilenetv2_model(num_classes=101, trainable_base=False)

        model = setup_fine_tuning(model, fine_tune_at_layer=120, learning_rate=1e-5)
        stats_p2 = get_model_parameter_stats(model)
        print(f"[INFO] Updated parameters: Trainable={stats_p2['trainable_parameters']:,}")

        cb_phase2 = [
            tf.keras.callbacks.ModelCheckpoint(
                str(results_dir / "best_model.keras"),
                monitor="val_loss",
                save_best_only=True,
                verbose=1,
            ),
            tf.keras.callbacks.EarlyStopping(
                monitor="val_loss",
                patience=5,
                restore_best_weights=True,
                verbose=1,
            ),
            tf.keras.callbacks.ReduceLROnPlateau(
                monitor="val_loss",
                factor=0.2,
                patience=2,
                min_lr=1e-7,
                verbose=1,
            ),
        ]

        initial_epoch = len(history_phase1.epoch) if history_phase1 else 0
        total_target_epochs = initial_epoch + p2_epochs

        t0 = time.time()
        history_phase2 = model.fit(
            train_ds,
            validation_data=val_ds,
            epochs=total_target_epochs,
            initial_epoch=initial_epoch,
            steps_per_epoch=steps_per_epoch,
            validation_steps=val_steps,
            callbacks=cb_phase2,
        )
        duration_p2 = time.time() - t0
        print(f"[SUCCESS] Phase 2 completed in {duration_p2:.1f}s.")

    # 4. Save History & Output Status
    print("\n[STEP 4] Exporting training metrics history...")
    combined_hist = {}
    if history_phase1:
        for k in history_phase1.history.keys():
            combined_hist[k] = list(history_phase1.history[k])
    if history_phase2:
        for k in history_phase2.history.keys():
            if k in combined_hist:
                combined_hist[k].extend(list(history_phase2.history[k]))
            else:
                combined_hist[k] = list(history_phase2.history[k])

    if combined_hist:
        df = pd.DataFrame(combined_hist)
        df.index.name = "epoch"
        df.to_csv(results_dir / "history.csv")
        print(f"[INFO] Saved {results_dir / 'history.csv'}")

    print("\n" + "=" * 70)
    print("  MOBILENETV2 TRAINING RUN COMPLETE")
    print("=" * 70)


if __name__ == "__main__":
    train()
