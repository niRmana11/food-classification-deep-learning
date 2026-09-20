"""
Unified Food-101 Data Pipeline & Preprocessing Factory
"""

import os
from pathlib import Path
from typing import Tuple, List
import tensorflow as tf

from src.preprocessing.augmentations import get_training_augmentation_pipeline


# Model specific preprocessing functions
def preprocess_for_model(image: tf.Tensor, model_type: str) -> tf.Tensor:
    """
    Applies the exact mathematical preprocessing required by each architecture backbone.
    """
    model_type = model_type.lower()
    
    if model_type == "custom_cnn":
        # Custom CNN trained from scratch expects pixel values scaled to [0, 1]
        return tf.cast(image, tf.float32) / 255.0
    
    elif model_type == "resnet50":
        # ResNet50 expects Caffe-style zero-centered BGR format
        return tf.keras.applications.resnet50.preprocess_input(image)
    
    elif model_type == "mobilenetv2":
        # MobileNetV2 expects normalization scaled to [-1.0, 1.0]
        return tf.keras.applications.mobilenet_v2.preprocess_input(image)
    
    elif model_type == "efficientnetb0":
        # EfficientNet has built-in rescaling layers. expects float inputs in [0, 255]
        return tf.keras.applications.efficientnet.preprocess_input(image)
    
    else:
        raise ValueError(
            f"Unsupported model_type: '{model_type}'. "
            f"Allowed options: 'custom_cnn', 'resnet50', 'mobilenetv2', 'efficientnetb0'"
        )


def _load_manifest_file(manifest_path: Path) -> List[str]:
    with open(manifest_path, "r") as f:
        return [line.strip() for line in f if line.strip()]


def create_food101_dataset(
    images_dir: str,
    manifest_file: str,
    class_names: List[str],
    image_size: Tuple[int, int] = (224, 224),
    batch_size: int = 32,
    is_training: bool = False,
    model_type: str = "custom_cnn",
    shuffle_buffer: int = 2048
) -> tf.data.Dataset:
    """
    Constructs an optimized tf.data.Dataset from a split manifest file.
    """
    images_base = Path(images_dir)
    samples = _load_manifest_file(Path(manifest_file))

    if is_training:
        # Deterministically pre-shuffle manifest samples with seed 42 so all 101 classes
        # are uniformly distributed across training batches (prevents sequential forgetting
        # caused by class-sorted manifests).
        import random
        random.Random(42).shuffle(samples)
    
    class_to_idx = {name: idx for idx, name in enumerate(class_names)}

    # Construct full image file paths and integer class labels
    file_paths = []
    labels = []
    for sample in samples:
        # sample is formatted as: 'class_name/image_id'
        full_path = str(images_base / f"{sample}.jpg")
        class_name = sample.split("/")[0]
        file_paths.append(full_path)
        labels.append(class_to_idx[class_name])

    dataset = tf.data.Dataset.from_tensor_slices((file_paths, labels))

    if is_training:
        dataset = dataset.shuffle(buffer_size=shuffle_buffer, seed=42)

    def _parse_and_preprocess(file_path, label):
        # Read image from disk
        raw_bytes = tf.io.read_file(file_path)
        img = tf.image.decode_jpeg(raw_bytes, channels=3)
        # Resize to common resolution
        img = tf.image.resize(img, image_size, method="bilinear")
        # Model-specific normalization
        img = preprocess_for_model(img, model_type=model_type)
        return img, label

    # Map with multithreaded parsing
    dataset = dataset.map(_parse_and_preprocess, num_parallel_calls=tf.data.AUTOTUNE)

    # Apply batching
    dataset = dataset.batch(batch_size, drop_remainder=False)

    # Apply data augmentation ONLY on training batches
    if is_training:
        augmentation_layer = get_training_augmentation_pipeline()
        dataset = dataset.map(
            lambda x, y: (augmentation_layer(x, training=True), y),
            num_parallel_calls=tf.data.AUTOTUNE
        )

    # Prefetch in GPU memory for high throughput
    dataset = dataset.prefetch(buffer_size=tf.data.AUTOTUNE)
    return dataset


def get_food101_datasets(
    data_dir: str = "data/raw/food-101",
    splits_dir: str = "data/splits",
    model_type: str = "custom_cnn",
    image_size: Tuple[int, int] = (224, 224),
    batch_size: int = 32
) -> Tuple[tf.data.Dataset, tf.data.Dataset, tf.data.Dataset]:
    """
    Public factory function returning (train_ds, val_ds, test_ds)
    with strict isolation and model-specific preprocessing.
    """
    images_dir = Path(data_dir) / "images"
    splits_path = Path(splits_dir)
    
    classes_file = splits_path / "classes.txt"
    class_names = _load_manifest_file(classes_file)

    train_ds = create_food101_dataset(
        images_dir=str(images_dir),
        manifest_file=str(splits_path / "train.txt"),
        class_names=class_names,
        image_size=image_size,
        batch_size=batch_size,
        is_training=True,
        model_type=model_type
    )

    val_ds = create_food101_dataset(
        images_dir=str(images_dir),
        manifest_file=str(splits_path / "val.txt"),
        class_names=class_names,
        image_size=image_size,
        batch_size=batch_size,
        is_training=False,
        model_type=model_type
    )

    test_ds = create_food101_dataset(
        images_dir=str(images_dir),
        manifest_file=str(splits_path / "test.txt"),
        class_names=class_names,
        image_size=image_size,
        batch_size=batch_size,
        is_training=False,
        model_type=model_type
    )

    return train_ds, val_ds, test_ds
