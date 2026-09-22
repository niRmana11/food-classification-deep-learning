"""
Custom CNN Architecture (Trained from Scratch Baseline)

Key Design Features:
- 4-Stage Hierarchical Feature Extraction (32 -> 64 -> 128 -> 256 filters)
- Batch Normalization after every convolution to stabilize internal covariate shift
- GlobalAveragePooling2D instead of Flatten to mitigate dense overfitting
- Dual Dropout regularization (0.4 and 0.3) for food texture generalization
"""

import tensorflow as tf
from tensorflow.keras import layers, models
from typing import Tuple


def build_custom_cnn(
    input_shape: Tuple[int, int, int] = (224, 224, 3),
    num_classes: int = 101,
    learning_rate: float = 0.001
) -> tf.keras.Model:
    """
    Constructs and compiles the baseline Custom CNN model.
    """
    model = models.Sequential(name="Custom_Food101_CNN")

    # Block 1 - Low level edge and color feature extraction
    model.add(layers.Input(shape=input_shape, name="input_image"))
    
    model.add(layers.Conv2D(32, (3, 3), padding="same", use_bias=False, name="conv1_1"))
    model.add(layers.BatchNormalization(name="bn1_1"))
    model.add(layers.ReLU(name="relu1_1"))
    
    model.add(layers.Conv2D(32, (3, 3), padding="same", use_bias=False, name="conv1_2"))
    model.add(layers.BatchNormalization(name="bn1_2"))
    model.add(layers.ReLU(name="relu1_2"))
    model.add(layers.MaxPooling2D(pool_size=(2, 2), name="pool1"))

    # Block 2 - Simple texture and corner patterns
    model.add(layers.Conv2D(64, (3, 3), padding="same", use_bias=False, name="conv2_1"))
    model.add(layers.BatchNormalization(name="bn2_1"))
    model.add(layers.ReLU(name="relu2_1"))
    
    model.add(layers.Conv2D(64, (3, 3), padding="same", use_bias=False, name="conv2_2"))
    model.add(layers.BatchNormalization(name="bn2_2"))
    model.add(layers.ReLU(name="relu2_2"))
    model.add(layers.MaxPooling2D(pool_size=(2, 2), name="pool2"))


    # Block 3 - Complex food texture and ingredient patterns
    model.add(layers.Conv2D(128, (3, 3), padding="same", use_bias=False, name="conv3_1"))
    model.add(layers.BatchNormalization(name="bn3_1"))
    model.add(layers.ReLU(name="relu3_1"))
    
    model.add(layers.Conv2D(128, (3, 3), padding="same", use_bias=False, name="conv3_2"))
    model.add(layers.BatchNormalization(name="bn3_2"))
    model.add(layers.ReLU(name="relu3_2"))
    model.add(layers.MaxPooling2D(pool_size=(2, 2), name="pool3"))

    # Block 4 - High-level semantic food representation
    model.add(layers.Conv2D(256, (3, 3), padding="same", use_bias=False, name="conv4_1"))
    model.add(layers.BatchNormalization(name="bn4_1"))
    model.add(layers.ReLU(name="relu4_1"))
    model.add(layers.MaxPooling2D(pool_size=(2, 2), name="pool4"))


    # Classification Head (Global Average Pooling + Dropout Regularization)
    model.add(layers.GlobalAveragePooling2D(name="global_avg_pool"))
    model.add(layers.BatchNormalization(name="bn_gap"))
    model.add(layers.Dropout(0.4, name="dropout_1"))
    
    model.add(layers.Dense(256, activation="relu", name="fc1"))
    model.add(layers.BatchNormalization(name="bn_fc1"))
    model.add(layers.Dropout(0.3, name="dropout_2"))
    
    model.add(layers.Dense(num_classes, activation="softmax", name="food101_predictions"))

    # Model Compilation
    optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)
    model.compile(
        optimizer=optimizer,
        loss="sparse_categorical_crossentropy",
        metrics=[
            "accuracy",
            tf.keras.metrics.SparseTopKCategoricalAccuracy(k=5, name="top_5_accuracy")
        ]
    )


    return model


if __name__ == "__main__":
    cnn = build_custom_cnn()
    cnn.summary()
