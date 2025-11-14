"""
Training pipeline for grape disease classification.

This script implements a two-stage training strategy based on transfer learning
with MobileNetV2:

1. Stage 1:
   - Load a pretrained MobileNetV2 backbone.
   - Freeze the backbone.
   - Train a custom classification head.

2. Stage 2:
   - Unfreeze the last portion of the backbone.
   - Fine-tune the model with a lower learning rate.

The script saves:
- Stage 1 model to the models directory.
- Final fine-tuned model to the models directory.
- Class names to a NumPy file for inference compatibility.
- Training curves to the evaluation directory.

All paths are resolved dynamically relative to this file so the script works
reliably regardless of the current working directory.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Dict, List, Tuple

import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input


# ---------------------------------------------------------------------------
# Project paths
# ---------------------------------------------------------------------------

CURRENT_DIR = Path(__file__).resolve().parent
BASE_DIR = CURRENT_DIR.parent
DATASET_DIR = BASE_DIR / "dataset"
MODELS_DIR = BASE_DIR / "models"
EVAL_DIR = BASE_DIR / "evaluation"

MODEL_STAGE1_PATH = MODELS_DIR / "grape_model_stage1.h5"
MODEL_FINAL_PATH = MODELS_DIR / "grape_model_finetuned_final.h5"
CLASS_NAMES_PATH = MODELS_DIR / "class_names.npy"
TRAINING_PLOT_PATH = EVAL_DIR / "training_plot_combined.png"


# ---------------------------------------------------------------------------
# Training configuration
# ---------------------------------------------------------------------------

IMAGE_SIZE: Tuple[int, int] = (224, 224)
BATCH_SIZE: int = 32
VALIDATION_SPLIT: float = 0.2
SEED: int = 42

EPOCHS_STAGE1: int = 10
EPOCHS_STAGE2: int = 10

INITIAL_LEARNING_RATE: float = 1e-4
FINE_TUNING_LEARNING_RATE: float = 1e-5
DROPOUT_RATE: float = 0.5
FINE_TUNE_FRACTION: float = 0.75


# ---------------------------------------------------------------------------
# Utility helpers
# ---------------------------------------------------------------------------

def ensure_directories() -> None:
    """
    Create required output directories if they do not already exist.
    """
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    EVAL_DIR.mkdir(parents=True, exist_ok=True)


def load_datasets() -> Tuple[tf.data.Dataset, tf.data.Dataset, np.ndarray]:
    """
    Load the training and validation datasets from the dataset directory.

    The dataset directory must follow this structure:

        dataset/
            class_a/
            class_b/
            class_c/
            ...

    Returns:
        A tuple containing:
            - training dataset
            - validation dataset
            - class names as a NumPy array
    """
    if not DATASET_DIR.exists():
        raise FileNotFoundError(
            f"Dataset directory not found: {DATASET_DIR}"
        )

    train_dataset = tf.keras.utils.image_dataset_from_directory(
        DATASET_DIR,
        validation_split=VALIDATION_SPLIT,
        subset="training",
        seed=SEED,
        image_size=IMAGE_SIZE,
        batch_size=BATCH_SIZE,
        label_mode="int",
    )

    validation_dataset = tf.keras.utils.image_dataset_from_directory(
        DATASET_DIR,
        validation_split=VALIDATION_SPLIT,
        subset="validation",
        seed=SEED,
        image_size=IMAGE_SIZE,
        batch_size=BATCH_SIZE,
        label_mode="int",
    )

    class_names = np.array(train_dataset.class_names)

    autotune = tf.data.AUTOTUNE
    train_dataset = train_dataset.prefetch(buffer_size=autotune)
    validation_dataset = validation_dataset.prefetch(buffer_size=autotune)

    return train_dataset, validation_dataset, class_names


def build_model(num_classes: int) -> Tuple[tf.keras.Model, tf.keras.Model]:
    """
    Build the transfer-learning model using MobileNetV2 as the backbone.

    The architecture is:
        Input
        -> Data augmentation
        -> Preprocessing
        -> MobileNetV2 feature extractor
        -> GlobalAveragePooling2D
        -> BatchNormalization
        -> Dense(128, relu)
        -> Dropout
        -> Dense(num_classes, softmax)

    Args:
        num_classes: Number of output classes.

    Returns:
        A tuple containing:
            - The compiled model architecture
            - The MobileNetV2 backbone
    """
    base_model = MobileNetV2(
        weights="imagenet",
        include_top=False,
        input_shape=(IMAGE_SIZE[0], IMAGE_SIZE[1], 3),
    )
    base_model.trainable = False

    data_augmentation = tf.keras.Sequential(
        [
            layers.RandomFlip("horizontal"),
            layers.RandomRotation(0.15),
            layers.RandomZoom(0.1),
            layers.RandomContrast(0.1),
        ],
        name="data_augmentation",
    )

    inputs = layers.Input(shape=(IMAGE_SIZE[0], IMAGE_SIZE[1], 3))
    x = data_augmentation(inputs)
    x = preprocess_input(x)
    x = base_model(x, training=False)
    x = layers.GlobalAveragePooling2D(name="global_average_pooling")(x)
    x = layers.BatchNormalization(name="batch_normalization")(x)
    x = layers.Dense(128, activation="relu", name="dense_128")(x)
    x = layers.Dropout(DROPOUT_RATE, name="dropout")(x)
    outputs = layers.Dense(num_classes, activation="softmax", name="predictions")(x)

    model = models.Model(inputs, outputs, name="grape_mobilenetv2_classifier")
    return model, base_model


def plot_training_curves(history: tf.keras.callbacks.History) -> None:
    """
    Plot training and validation accuracy/loss curves and save them to disk.

    Args:
        history: Keras History object returned by model.fit().
    """
    accuracy = history.history.get("accuracy", [])
    val_accuracy = history.history.get("val_accuracy", [])
    loss = history.history.get("loss", [])
    val_loss = history.history.get("val_loss", [])

    epochs = range(1, len(accuracy) + 1)

    plt.figure(figsize=(12, 5))

    plt.subplot(1, 2, 1)
    plt.plot(epochs, accuracy, label="Training Accuracy")
    plt.plot(epochs, val_accuracy, label="Validation Accuracy")
    plt.title("Training and Validation Accuracy")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.legend()
    plt.grid(True, alpha=0.3)

    plt.subplot(1, 2, 2)
    plt.plot(epochs, loss, label="Training Loss")
    plt.plot(epochs, val_loss, label="Validation Loss")
    plt.title("Training and Validation Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()
    plt.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(TRAINING_PLOT_PATH, dpi=300, bbox_inches="tight")
    plt.close()

    print(f"Training curves saved to: {TRAINING_PLOT_PATH}")


def merge_histories(
    stage1_history: tf.keras.callbacks.History,
    stage2_history: tf.keras.callbacks.History,
) -> tf.keras.callbacks.History:
    """
    Merge two training history objects into a single history-like object.

    This is useful for plotting a single combined curve after two-stage training.

    Args:
        stage1_history: History from the first training stage.
        stage2_history: History from the fine-tuning stage.

    Returns:
        A history-like object with concatenated metric values.
    """
    combined_history: Dict[str, List[float]] = {}

    all_keys = set(stage1_history.history.keys()) | set(stage2_history.history.keys())
    for key in all_keys:
        combined_history[key] = (
            list(stage1_history.history.get(key, []))
            + list(stage2_history.history.get(key, []))
        )

    class CombinedHistory:
        def __init__(self, history: Dict[str, List[float]]) -> None:
            self.history = history

    return CombinedHistory(combined_history)  # type: ignore[return-value]


# ---------------------------------------------------------------------------
# Main training pipeline
# ---------------------------------------------------------------------------

def train_model() -> None:
    """
    Train the grape classification model using transfer learning and fine-tuning.
    """
    ensure_directories()

    print("Loading datasets...")
    train_dataset, validation_dataset, class_names = load_datasets()
    num_classes = len(class_names)

    print(f"Detected classes: {list(class_names)}")
    print(f"Number of classes: {num_classes}")

    print("Building model...")
    model, base_model = build_model(num_classes)

    # Stage 1: train the classification head while keeping the backbone frozen.
    print("\nStarting stage 1 training...")
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=INITIAL_LEARNING_RATE),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )

    stage1_history = model.fit(
        train_dataset,
        validation_data=validation_dataset,
        epochs=EPOCHS_STAGE1,
        verbose=1,
    )

    model.save(MODEL_STAGE1_PATH)
    print(f"Stage 1 model saved to: {MODEL_STAGE1_PATH}")

    # Stage 2: fine-tune the upper layers of the backbone.
    print("\nStarting stage 2 fine-tuning...")
    base_model.trainable = True

    fine_tune_at = int(len(base_model.layers) * FINE_TUNE_FRACTION)
    for layer in base_model.layers[:fine_tune_at]:
        layer.trainable = False

    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=FINE_TUNING_LEARNING_RATE),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )

    stage2_history = model.fit(
        train_dataset,
        validation_data=validation_dataset,
        epochs=EPOCHS_STAGE2,
        verbose=1,
    )

    model.save(MODEL_FINAL_PATH)
    print(f"Final fine-tuned model saved to: {MODEL_FINAL_PATH}")

    np.save(CLASS_NAMES_PATH, class_names)
    print(f"Class names saved to: {CLASS_NAMES_PATH}")

    combined_history = merge_histories(stage1_history, stage2_history)
    plot_training_curves(combined_history)

    print("\nTraining completed successfully.")


if __name__ == "__main__":
    train_model()
