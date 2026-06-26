"""
Generate a visual architecture diagram of the trained model.

This script loads the final trained CNN model and produces a visual
diagram showing all layers and connections.

Output:
    evaluation/model_architecture.png
"""

from pathlib import Path

import tensorflow as tf
from tensorflow.keras.utils import plot_model


CURRENT_DIR = Path(__file__).resolve().parent
BASE_DIR = CURRENT_DIR.parent

MODELS_DIR = BASE_DIR / "models"
EVAL_DIR = BASE_DIR / "evaluation"

MODEL_PATH = MODELS_DIR / "grape_model_finetuned_final.h5"
OUTPUT_PATH = EVAL_DIR / "model_architecture.png"


def visualize_model():
    """
    Generate and save the model architecture diagram.
    """

    EVAL_DIR.mkdir(exist_ok=True)

    print("Loading model...")
    model = tf.keras.models.load_model(MODEL_PATH)

    print("Generating architecture diagram...")

    plot_model(
        model,
        to_file=OUTPUT_PATH,
        show_shapes=True,
        show_layer_names=True,
        dpi=300,
        expand_nested=True,
    )

    print(f"Model architecture saved to: {OUTPUT_PATH}")


if __name__ == "__main__":
    visualize_model()
