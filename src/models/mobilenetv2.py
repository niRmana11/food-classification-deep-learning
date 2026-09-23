"""
MobileNetV2 Architecture Implementation & Factory for Food-101 Classification
Author: Kanushka (Member 3)
Module: SE4050 - Deep Learning (2026)

Architecture Highlights & Technical Justifications:
---------------------------------------------------
1. Depthwise Separable Convolutions:
   Splits standard convolution into a 3x3 depthwise convolution (spatial filtering)
   followed by a 1x1 pointwise convolution (feature combination). This achieves
   approximately an 8x-9x reduction in computational complexity compared to standard
   convolutions with minimal drop in top-1 accuracy:
   Computation Ratio: (D_K * D_K * M * D_F * D_F + M * N * D_F * D_F) / (D_K * D_K * M * N * D_F * D_F)
                    = 1/N + 1/(D_K^2) ~= 1/101 + 1/9 ~= 0.12

2. Inverted Residual Bottlenecks:
   Unlike traditional residual blocks (which compress -> filter -> expand), MobileNetV2
   uses an inverted residual structure: expand (1x1 conv) -> filter (3x3 depthwise) ->
   project/compress (1x1 linear bottleneck). Shortcuts are placed between low-dimensional
   bottleneck representations.

3. Linear Bottlenecks:
   Non-linear activations (like ReLU) in low-dimensional spaces discard crucial
   manifold information. MobileNetV2 uses linear (no activation) outputs at the
   end of each bottleneck block to preserve representation capacity.

4. Lightweight Footprint for Edge & Mobile Deployment:
   - Base Parameters: ~2,257,984 (~2.26M)
   - Classification Head (GAP + Dense 101): ~131,493 (~0.13M)
   - Total Parameters: ~2,389,477 (~2.39M)
   - ResNet50 comparison: ResNet50 has ~25.6M parameters (~10.7x larger).
   - EfficientNetB0 comparison: EfficientNetB0 has ~5.3M parameters (~2.2x larger).
"""

from typing import Tuple, Dict, Any, Optional
import tensorflow as tf
from tensorflow.keras import layers, models


def build_mobilenetv2_model(
    num_classes: int = 101,
    input_shape: Tuple[int, int, int] = (224, 224, 3),
    dropout_rate: float = 0.2,
    trainable_base: bool = False,
    learning_rate: float = 0.001,
    weights: str = "imagenet"
) -> tf.keras.Model:
    """
    Builds the MobileNetV2 transfer-learning model for Food-101.
    
    Architecture:
      Input(224, 224, 3)
         -> Pretrained MobileNetV2 Backbone (without top)
         -> GlobalAveragePooling2D
         -> Dropout(dropout_rate)
         -> Dense(num_classes, activation='softmax', name='food101_predictions')

    Args:
        num_classes: Number of target categories (101 for Food-101).
        input_shape: Input image dimensions (224, 224, 3).
        dropout_rate: Dropout fraction before dense classifier head (default 0.2).
        trainable_base: Whether backbone weights are trainable initially (default False).
        learning_rate: Initial learning rate for Adam optimizer.
        weights: Pretrained weights ('imagenet' or None).

    Returns:
        Compiled tf.keras.Model instance.
    """
    # 1. Base Pretrained Backbone
    base_model = tf.keras.applications.MobileNetV2(
        input_shape=input_shape,
        include_top=False,
        weights=weights
    )
    base_model.trainable = trainable_base

    # 2. Functional Model Definition
    inputs = tf.keras.Input(shape=input_shape, name="input_image")
    
    # Pass through MobileNetV2 backbone
    # Note: training=False keeps BatchNormalization layers in inference mode even during fine-tuning
    x = base_model(inputs, training=False)
    
    # 3. Classification Head
    # GlobalAveragePooling2D replaces Flatten: reduces spatial dimensions (7x7x1280 -> 1280)
    # drastically cutting parameter count and avoiding dense overfitting.
    x = layers.GlobalAveragePooling2D(name="global_avg_pool")(x)
    
    # Regularization via Dropout
    if dropout_rate > 0.0:
        x = layers.Dropout(dropout_rate, name=f"dropout_{dropout_rate}")(x)
        
    outputs = layers.Dense(
        num_classes,
        activation="softmax",
        name="food101_predictions"
    )(x)

    model = models.Model(inputs=inputs, outputs=outputs, name="MobileNetV2_Food101")

    # 4. Compile with Categorical Crossentropy / Sparse Categorical Crossentropy
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
        loss="sparse_categorical_crossentropy",
        metrics=[
            "accuracy",
            tf.keras.metrics.SparseTopKCategoricalAccuracy(k=5, name="top_5_accuracy")
        ]
    )

    return model


def setup_fine_tuning(
    model: tf.keras.Model,
    fine_tune_at_layer: int = 120,
    learning_rate: float = 1e-5
) -> tf.keras.Model:
    """
    Configures Phase 2 Fine-Tuning:
    Unfreezes the top inverted residual bottleneck blocks of the MobileNetV2 backbone
    (from `fine_tune_at_layer` onward) and recompiles with a low learning rate.

    Args:
        model: MobileNetV2 Model created with build_mobilenetv2_model.
        fine_tune_at_layer: Layer index from which layers become trainable.
                            MobileNetV2 has 154 layers; layer 120 unfreezes the
                            top two inverted residual blocks (blocks 14-16).
        learning_rate: Reduced learning rate (typically 1e-5) to prevent
                       catastrophic forgetting of pretrained visual representations.

    Returns:
        Recompiled model configured for fine-tuning.
    """
    # Locate backbone layer
    base_model = None
    for layer in model.layers:
        if isinstance(layer, tf.keras.Model) or "mobilenetv2" in layer.name.lower():
            base_model = layer
            break

    if base_model is None:
        raise ValueError("Could not find MobileNetV2 base model layer inside the model.")

    # Unfreeze the base model
    base_model.trainable = True

    # Freeze all layers before fine_tune_at_layer
    total_layers = len(base_model.layers)
    for i, layer in enumerate(base_model.layers[:fine_tune_at_layer]):
        layer.trainable = False

    for i, layer in enumerate(base_model.layers[fine_tune_at_layer:], start=fine_tune_at_layer):
        # Keep BatchNormalization layers frozen to maintain running mean/std stability
        if isinstance(layer, layers.BatchNormalization):
            layer.trainable = False
        else:
            layer.trainable = True

    trainable_count = sum(1 for layer in base_model.layers if layer.trainable)
    print(
        f"[INFO] MobileNetV2 fine-tuning enabled from layer {fine_tune_at_layer}/{total_layers}. "
        f"Trainable base layers: {trainable_count}."
    )

    # Recompile with smaller learning rate
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
        loss="sparse_categorical_crossentropy",
        metrics=[
            "accuracy",
            tf.keras.metrics.SparseTopKCategoricalAccuracy(k=5, name="top_5_accuracy")
        ]
    )

    return model


def get_model_parameter_stats(model: tf.keras.Model) -> Dict[str, Any]:
    """
    Calculates parameter breakdown and estimated memory footprint.
    """
    total_params = model.count_params()
    trainable_params = sum(
        tf.keras.backend.count_params(w) for w in model.trainable_weights
    )
    non_trainable_params = total_params - trainable_params
    
    # 4 bytes per 32-bit float parameter
    size_mb = (total_params * 4) / (1024 * 1024)

    return {
        "model_name": "MobileNetV2",
        "total_parameters": total_params,
        "trainable_parameters": trainable_params,
        "non_trainable_parameters": non_trainable_params,
        "estimated_weights_size_mb": round(size_mb, 2)
    }


if __name__ == "__main__":
    print("[TEST] Building MobileNetV2 for Food-101 (Phase 1: Feature Extraction)...")
    model = build_mobilenetv2_model(num_classes=101, trainable_base=False)
    phase1_stats = get_model_parameter_stats(model)
    print("[PHASE 1 STATS]:", phase1_stats)

    print("\n[TEST] Configuring MobileNetV2 (Phase 2: Fine-Tuning from layer 120)...")
    setup_fine_tuning(model, fine_tune_at_layer=120, learning_rate=1e-5)
    phase2_stats = get_model_parameter_stats(model)
    print("[PHASE 2 STATS]:", phase2_stats)
