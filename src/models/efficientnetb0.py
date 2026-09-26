"""
EfficientNetB0 Transfer Learning Architecture (Two-Phase Fine-Tuning)

Key Design Features:
- ImageNet-pretrained compound-scaled backbone (Tan & Le, 2019). B0 is the NAS-derived
  baseline at phi=0 from which B1-B7 are scaled in depth, width and resolution together.
- MBConv blocks: inverted residual bottleneck + depthwise convolution +
  Squeeze-and-Excitation channel attention, Swish activation, ~4.0M backbone parameters.
- Phase 1: frozen backbone feature extraction. Only the classification head trains.
- Phase 2: selective unfreezing of block6 / block7 / top_conv at a 100x reduced
  learning rate, adapting only the most class-specific abstractions to the food domain.
- BatchNormalization stays frozen during Phase 2 to preserve ImageNet running statistics.
- GlobalAveragePooling2D instead of Flatten to mitigate dense-layer overfitting
  (matches the Custom CNN baseline head for a fair architectural comparison).
"""

import tensorflow as tf
from tensorflow.keras import layers, models
from typing import Tuple, Sequence


# Backbone stages adapted during Phase 2 fine-tuning. Blocks 1-5 encode generic
# edge/texture features that transfer unchanged; blocks 6-7 and top_conv encode the
# abstract, class-specific features that must be re-learned for food categories.
FINETUNE_BLOCKS: Tuple[str, ...] = ("block6", "block7", "top_conv")


def compile_model(
    model: tf.keras.Model,
    learning_rate: float = 0.001
) -> tf.keras.Model:
    """
    Applies the locked experimental protocol compilation settings.

    Uses the sparse loss/metric variants because the shared data loader emits integer
    class indices rather than one-hot vectors (see src/preprocessing/data_loader.py).

    NOTE: Keras caches the trainable-weight list at compile time, so this MUST be
    called again after any change to layer.trainable, or the change is silently ignored.
    """
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
        loss="sparse_categorical_crossentropy",
        metrics=[
            "accuracy",
            tf.keras.metrics.SparseTopKCategoricalAccuracy(k=5, name="top_5_accuracy")
        ]
    )
    return model


def build_efficientnetb0(
    input_shape: Tuple[int, int, int] = (224, 224, 3),
    num_classes: int = 101,
    learning_rate: float = 0.001,
    dropout_rate: float = 0.3
) -> Tuple[tf.keras.Model, tf.keras.Model]:
    """
    Constructs and compiles the EfficientNetB0 classifier in its Phase 1
    (frozen feature extraction) configuration.

    Returns:
        (model, base_model) - base_model is returned so the notebook can selectively
        unfreeze its top blocks for Phase 2 fine-tuning.
    """
    inputs = layers.Input(shape=input_shape, name="input_image")

    # Pretrained backbone. include_top=False drops the 1000-class ImageNet classifier
    # and exposes the 7x7x1280 feature map produced by top_conv.
    # NOTE: Keras EfficientNet contains internal Rescaling/Normalization layers, so it
    # expects raw float pixels in [0, 255]. The shared loader routes 'efficientnetb0'
    # through efficientnet.preprocess_input, which is a deliberate pass-through.
    base_model = tf.keras.applications.EfficientNetB0(
        include_top=False,
        weights="imagenet",
        input_tensor=inputs
    )
    base_model.trainable = False  # Phase 1: freeze the entire backbone

    # Classification Head (Global Average Pooling + Dropout Regularization)
    # 7x7x1280 -> 1280 by spatial averaging. Flatten would produce a 62,720-dim vector
    # and a ~6.3M parameter Dense layer, exceeding the backbone itself and overfitting.
    x = layers.GlobalAveragePooling2D(name="global_avg_pool")(base_model.output)

    # Re-centres the pooled ImageNet feature distribution for the new 101-class head.
    # This head BatchNorm is NOT part of the backbone and stays trainable throughout.
    x = layers.BatchNormalization(name="bn_gap")(x)
    x = layers.Dropout(dropout_rate, name="dropout_head")(x)

    outputs = layers.Dense(num_classes, activation="softmax", name="food101_predictions")(x)

    model = models.Model(inputs=inputs, outputs=outputs, name="EfficientNetB0_Food101")

    return compile_model(model, learning_rate=learning_rate), base_model


def unfreeze_top_blocks(
    base_model: tf.keras.Model,
    finetune_blocks: Sequence[str] = FINETUNE_BLOCKS,
    freeze_batchnorm: bool = True
) -> int:
    """
    Prepares the backbone for Phase 2 by unfreezing only the top MBConv stages.

    Blocks are selected by name rather than by layer count so the choice is
    reproducible and architecturally meaningful ("the final two MBConv stages")
    rather than an arbitrary slice index.

    Args:
        freeze_batchnorm: keep every backbone BatchNormalization layer frozen. Leave
            this True. A frozen BatchNormalization layer runs in inference mode, so its
            ImageNet running statistics are preserved instead of being overwritten by
            noisy 32-image batch statistics.

    Returns:
        The number of backbone layers made trainable.
    """
    base_model.trainable = True  # unfreezes everything; re-frozen selectively below
    block_prefixes = tuple(finetune_blocks)
    unfrozen_count = 0

    for layer in base_model.layers:
        is_target_block = layer.name.startswith(block_prefixes)
        is_batchnorm = isinstance(layer, layers.BatchNormalization)

        if is_target_block and not (freeze_batchnorm and is_batchnorm):
            layer.trainable = True
            unfrozen_count += 1
        else:
            layer.trainable = False

    return unfrozen_count


if __name__ == "__main__":
    effnet, backbone = build_efficientnetb0()
    effnet.summary()

    print(f"\n[PHASE 1] Trainable weight tensors: {len(effnet.trainable_weights)}")
    n_unfrozen = unfreeze_top_blocks(backbone)
    compile_model(effnet, learning_rate=1e-5)
    print(f"[PHASE 2] Unfroze {n_unfrozen} backbone layers "
          f"({', '.join(FINETUNE_BLOCKS)}), BatchNormalization kept frozen.")
    print(f"[PHASE 2] Trainable weight tensors: {len(effnet.trainable_weights)}")
