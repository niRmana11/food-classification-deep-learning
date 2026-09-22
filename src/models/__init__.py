"""
Model Architecture Definitions for Food Classification
"""

from src.models.resnet50 import (
    build_resnet50_model,
    unfreeze_resnet50_for_finetuning,
    compile_resnet50_model,
)

__all__ = [
    "build_resnet50_model",
    "unfreeze_resnet50_for_finetuning",
    "compile_resnet50_model",
]
