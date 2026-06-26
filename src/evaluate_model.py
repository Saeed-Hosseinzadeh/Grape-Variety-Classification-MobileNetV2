"""
Model evaluation script for the grape disease classifier.

This module loads the final fine‑tuned model and evaluates it on the
validation dataset. It produces a classification report and generates
a confusion matrix visualization.

Outputs:
    evaluation/confusion_matrix.png
"""

from pathlib import Path

import numpy as np
import seaborn as sns
import tensorflow as tf
import matplotlib.pyplot as plt

from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix


# ---------------------------------------------------------------------
# Project paths
# ---------------------------------------------------------------------

CURRENT_DIR = Path(__file__).resolve().parent
BASE_DIR = CURRENT_DIR.parent

DATASET_DIR = BASE_DIR / "dataset"
MODELS_DIR = BASE_DIR / "models"
EVAL_DIR = BASE_DIR / "evaluation"

MODEL_PATH = MODELS_DIR / "grape_model_finetuned_final.h5"
CLASS_NAMES_PATH = MODELS_DIR / "class_names.npy"
CONFUSION_MATRIX_PATH = EVAL_DIR / "confusion_matrix.png"

IMAGE_SIZE = (224, 224)
BATCH_SIZE = 32
VALIDATION_SPLIT = 0.2
SEED = 42


def load_validation_dataset():
    """
    Load validation dataset from the dataset directory.
    """

    dataset = tf.keras.utils.image_dataset_from_directory(
        DATASET_DIR,
        validation_split=VALIDATION_SPLIT,
        subset="validation",
        seed=SEED,
        image_size=IMAGE_SIZE,
        batch_size=BATCH_SIZE,
        label_mode="int",
    )

    return dataset


def evaluate_model():
    """
    Evaluate the trained CNN model using the validation dataset.
    """

    EVAL_DIR.mkdir(exist_ok=True)

    print("Loading model...")
    model = tf.keras.models.load_model(MODEL_PATH)

    print("Loading class names...")
    class_names = np.load(CLASS_NAMES_PATH)

    print("Loading validation dataset...")
    dataset = load_validation_dataset()

    y_true = []
    y_pred = []

    print("Running predictions...")

    for images, labels in dataset:
        predictions = model.predict(images, verbose=0)
        predicted_labels = np.argmax(predictions, axis=1)

        y_true.extend(labels.numpy())
        y_pred.extend(predicted_labels)

    y_true = np.array(y_true)
    y_pred = np.array(y_pred)

    print("\nClassification Report\n")
    print(
        classification_report(
            y_true,
            y_pred,
            target_names=class_names,
        )
    )

    cm = confusion_matrix(y_true, y_pred)

    plt.figure(figsize=(8, 6))
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=class_names,
        yticklabels=class_names,
    )

    plt.title("Confusion Matrix")
    plt.xlabel("Predicted Label")
    plt.ylabel("True Label")

    plt.tight_layout()
    plt.savefig(CONFUSION_MATRIX_PATH, dpi=300)
    plt.close()

    print(f"\nConfusion matrix saved to: {CONFUSION_MATRIX_PATH}")


if __name__ == "__main__":
    evaluate_model()
