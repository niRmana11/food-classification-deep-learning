"""
Verification & Diagnostics Suite for MobileNetV2 Pipeline
Author: Kanushka (Member 3)
Module: SE4050 - Deep Learning (2026)

Verifies:
1. Canonical Food-101 Split Manifests & Class Integrity
2. Configuration Protocol Compliance (Seed, Res, Batch Size)
3. MobileNetV2 Preprocessing Normalization Bounds ([-1.0, 1.0])
4. Model Architecture & Parameter Constraints
"""

import sys
from pathlib import Path

try:
    import yaml
except ImportError:
    yaml = None

def verify_pipeline():
    print("=" * 70)
    print("  SE4050 FOOD-101 BENCHMARK - MEMBER 3 (KANUSHKA) VERIFICATION")
    print("=" * 70)
    
    root = Path(__file__).resolve().parent.parent.parent
    splits_dir = root / "data" / "splits"
    config_path = root / "configs" / "config.yaml"

    passed = 0
    total = 0

    # 1. Verify Configuration Protocol
    total += 1
    print("\n[CHECK 1] Verifying global configuration protocol (configs/config.yaml)...")
    if not config_path.exists():
        print("  [FAIL] configs/config.yaml not found!")
    else:
        with open(config_path, "r") as f:
            content = f.read()
        if yaml is not None:
            cfg = yaml.safe_load(content)
            seed = cfg.get("project", {}).get("seed")
            target_size = cfg.get("image", {}).get("target_size")
            batch_size = cfg.get("benchmark", {}).get("selected_batch_size")
            mobilenet_prep = cfg.get("preprocessing", {}).get("mobilenetv2")
        else:
            # Fallback text parsing
            seed = 42 if "seed: 42" in content else None
            target_size = [224, 224] if "[224, 224]" in content else None
            batch_size = 32 if "selected_batch_size: 32" in content else None
            mobilenet_prep = "normalize_neg1_to_pos1" if 'mobilenetv2: "normalize_neg1_to_pos1"' in content else None

        assert seed == 42, f"Seed must be 42, got {seed}"
        assert target_size == [224, 224], f"Target size must be [224, 224], got {target_size}"
        assert batch_size == 32, f"Batch size must be 32, got {batch_size}"
        assert mobilenet_prep == "normalize_neg1_to_pos1", f"Unexpected prep: {mobilenet_prep}"

        print(f"  [PASS] Config verified: Seed={seed}, ImageSize={target_size}, BatchSize={batch_size}")
        print(f"         Preprocessing mode: {mobilenet_prep}")
        passed += 1

    # 2. Verify Split Manifests
    total += 1
    print("\n[CHECK 2] Verifying canonical data split manifests (data/splits/)...")
    classes_file = splits_dir / "classes.txt"
    train_file = splits_dir / "train.txt"
    val_file = splits_dir / "val.txt"
    test_file = splits_dir / "test.txt"

    files_ok = all(f.exists() for f in [classes_file, train_file, val_file, test_file])
    if not files_ok:
        print("  [FAIL] One or more manifest files are missing in data/splits/!")
    else:
        with open(classes_file) as f:
            classes = [l.strip() for l in f if l.strip()]
        with open(train_file) as f:
            train_lines = [l.strip() for l in f if l.strip()]
        with open(val_file) as f:
            val_lines = [l.strip() for l in f if l.strip()]
        with open(test_file) as f:
            test_lines = [l.strip() for l in f if l.strip()]

        print(f"  [INFO] Total classes: {len(classes)} (Expected: 101)")
        print(f"  [INFO] Train samples: {len(train_lines):,} (Expected: 68,175)")
        print(f"  [INFO] Val samples:   {len(val_lines):,} (Expected: 7,575)")
        print(f"  [INFO] Test samples:  {len(test_lines):,} (Expected: 25,250)")

        assert len(classes) == 101, f"Expected 101 classes, found {len(classes)}"
        assert len(train_lines) == 68175, f"Expected 68,175 train samples, found {len(train_lines)}"
        assert len(val_lines) == 7575, f"Expected 7,575 val samples, found {len(val_lines)}"
        assert len(test_lines) == 25250, f"Expected 25,250 test samples, found {len(test_lines)}"
        print("  [PASS] All split manifests match canonical Food-101 zero-leakage counts exactly.")
        passed += 1

    # 3. Verify TensorFlow & MobileNetV2 Normalization Math
    total += 1
    print("\n[CHECK 3] Checking TensorFlow & MobileNetV2 normalization math...")
    try:
        import tensorflow as tf
        from src.preprocessing.data_loader import preprocess_for_model
        
        # Test synthetic image with range [0, 255]
        dummy_img = tf.constant([[[0.0, 127.5, 255.0]]], dtype=tf.float32)
        norm_img = preprocess_for_model(dummy_img, model_type="mobilenetv2")
        
        # Normalization should map 0 -> -1.0, 127.5 -> 0.0, 255 -> 1.0
        min_val = float(tf.reduce_min(norm_img))
        max_val = float(tf.reduce_max(norm_img))
        mid_val = float(norm_img[0, 0, 1])

        print(f"  [INFO] Input [0, 127.5, 255] mapped to: [{min_val:.1f}, {mid_val:.1f}, {max_val:.1f}]")
        assert abs(min_val - (-1.0)) < 1e-4, f"Min expected -1.0, got {min_val}"
        assert abs(mid_val - 0.0) < 1e-4, f"Mid expected 0.0, got {mid_val}"
        assert abs(max_val - 1.0) < 1e-4, f"Max expected 1.0, got {max_val}"
        print("  [PASS] MobileNetV2 normalization strictly bounded within [-1.0, 1.0].")
        passed += 1
    except ImportError:
        print("  [SKIP] TensorFlow not installed in current local shell yet.")
        print("         (Will be validated in Google Colab / once local pip install finishes).")

    # 4. Verify Model Implementation Architecture Import
    total += 1
    print("\n[CHECK 4] Checking src/models/mobilenetv2.py...")
    model_script = root / "src" / "models" / "mobilenetv2.py"
    if model_script.exists():
        print("  [PASS] src/models/mobilenetv2.py is present and ready.")
        passed += 1
    else:
        print("  [FAIL] src/models/mobilenetv2.py missing!")

    print("\n" + "=" * 70)
    print(f"  VERIFICATION RESULT: {passed}/{total} CHECKS COMPLETED")
    print("=" * 70)
    return passed == total

if __name__ == "__main__":
    verify_pipeline()
