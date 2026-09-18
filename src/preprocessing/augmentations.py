"""
Training-Only Image Augmentation Pipeline for Food-101
Food images preserve semantic identity under horizontal flips,
subtle rotations and slight zoom/crop. Vertical flips are avoided.
"""

import tensorflow as tf
from tensorflow.keras import layers


def get_training_augmentation_pipeline(
    rotation_factor: float = 0.05,
    zoom_factor: float = 0.10
) -> tf.keras.Sequential:
    """
    Returns a Keras Sequential model applying data augmentation.
    Applied exclusively during training. acts as identity during inference/eval.
    """
    return tf.keras.Sequential([
        layers.RandomFlip("horizontal"),
        layers.RandomRotation(rotation_factor),
        layers.RandomZoom(height_factor=(-zoom_factor, zoom_factor)),
    ], name="food101_train_augmentation")
