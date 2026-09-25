"""
ResNet50 Architecture Builder & Transfer Learning Factory for Food-101

Implements:
1. Phase 1: Feature extraction with frozen ImageNet pretrained ResNet50 backbone.
2. Phase 2: Fine-tuning unfreezing Stage 5 residual blocks (conv5_block1_out onward).
3. Custom classification head: GlobalAveragePooling2D -> BatchNorm -> Dropout(0.3) -> Dense(101).
"""

from typing import Tuple
import tensorflow as tf
from tensorflow.keras import layers, models, regularizers


def build_resnet50_model(
    num_classes: int = 101,
    input_shape: Tuple[int, int, int] = (224, 224, 3),
    dropout_rate: float = 0.3,
    l2_reg: float = 1e-4,
    weights: str = "imagenet"
) -> Tuple[tf.keras.Model, tf.keras.Model]:
    """
    Constructs the ResNet50 transfer learning model for Phase 1 (Feature Extraction).
    
    Args:
        num_classes: Number of food target categories (default: 101).
        input_shape: Input image dimensions (default: 224x224x3).
        dropout_rate: Dropout probability in classification head (default: 0.3).
        l2_reg: L2 weight regularization penalty factor.
        weights: Pretrained weights to load (default: 'imagenet').
        
    Returns:
        model: The full end-to-end Keras Model.
        base_model: The ResNet50 backbone instance.
    """
    # 1. Instantiate pretrained backbone without ImageNet 1000-class dense head
    base_model = tf.keras.applications.ResNet50(
        weights=weights,
        include_top=False,
        input_shape=input_shape
    )
    
    # 2. Freeze all backbone layers for initial feature extraction
    base_model.trainable = False
    
    # 3. Build classification head
    inputs = layers.Input(shape=input_shape, name="input_image")
    
    # Run backbone in inference mode so batch normalization statistics from ImageNet
    # are preserved during initial head training
    x = base_model(inputs, training=False)
    
    # Global Average Pooling flattens 7x7 spatial feature map into 2048-d vector
    # without exploding parameter counts
    x = layers.GlobalAveragePooling2D(name="global_avg_pool")(x)
    
    # Batch Normalization stabilizes activations before the dense classifier
    x = layers.BatchNormalization(name="head_batch_norm")(x)
    
    # Dropout prevents overfitting on the dense classification layer
    x = layers.Dropout(dropout_rate, name="head_dropout")(x)
    
    # Softmax output layer
    outputs = layers.Dense(
        num_classes,
        activation="softmax",
        kernel_regularizer=regularizers.l2(l2_reg),
        name="predictions"
    )(x)
    
    model = models.Model(inputs=inputs, outputs=outputs, name="ResNet50_Food101")
    return model, base_model


def unfreeze_resnet50_for_finetuning(
    model: tf.keras.Model,
    base_model: tf.keras.Model,
    fine_tune_from_layer: str = "conv5_block1_out"
) -> int:
    """
    Unfreezes top residual blocks of ResNet50 for Phase 2 Fine-Tuning.
    
    Layers prior to `fine_tune_from_layer` remain frozen to preserve general low-level
    visual primitives (edges, textures, color gradients). Layers from `fine_tune_from_layer`
    onward are set to trainable.
    
    Args:
        model: The full ResNet50 model.
        base_model: The ResNet50 backbone.
        fine_tune_from_layer: Name of the layer from which unfreezing begins.
                              Default: 'conv5_block1_out' (unfreezes stage 5 residual blocks).
                              
    Returns:
        trainable_layer_count: Number of backbone layers set to trainable.
    """
    base_model.trainable = True
    
    unfreeze = False
    trainable_count = 0
    
    for layer in base_model.layers:
        if layer.name == fine_tune_from_layer:
            unfreeze = True
        
        if unfreeze:
            # BatchNormalization layers inside base model are kept frozen during fine-tuning
            # to prevent noisy updates on mini-batches from corrupting running mean/variance
            if isinstance(layer, layers.BatchNormalization):
                layer.trainable = False
            else:
                layer.trainable = True
                trainable_count += 1
        else:
            layer.trainable = False
            
    return trainable_count


def compile_resnet50_model(
    model: tf.keras.Model,
    learning_rate: float = 0.001
) -> None:
    """
    Compiles the model with Adam optimizer, sparse categorical crossentropy,
    and multi-metric tracking (accuracy & top-5 accuracy).
    """
    optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)
    loss = tf.keras.losses.SparseCategoricalCrossentropy()
    metrics = [
        "accuracy",
        tf.keras.metrics.SparseTopKCategoricalAccuracy(k=5, name="top_5_accuracy")
    ]
    
    model.compile(
        optimizer=optimizer,
        loss=loss,
        metrics=metrics
    )
