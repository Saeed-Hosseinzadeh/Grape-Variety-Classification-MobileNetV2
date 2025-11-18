"""
Inference module for grape disease classification.

This script loads the final fine‑tuned MobileNetV2 model and performs
single-image inference. It can be used either as:

1. A command-line tool
2. A reusable Python function inside other modules (e.g., Streamlit app)

Example CLI usage:
    python src/predict.py path/to/image.jpg

The script dynamically resolves model paths relative to the project
structure, ensuring it works regardless of the current working directory.

Project structure (relevant parts):

project_root/
│
├── models/
│   ├── grape_model_finetuned_final.h5
│   └── class_names.npy
│
├── src/
│   └── predict.py
│
└── app.py
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Tuple

import numpy as np
import tensorflow as tf
from PIL import Image
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input


# ----------------------------------------------------------------------
# Project path configuration
# ----------------------------------------------------------------------

CURRENT_DIR = Path(__file__).resolve().parent
BASE_DIR = CURRENT_DIR.parent
MODELS_DIR = BASE_DIR / "models"

MODEL_PATH = MODELS_DIR / "grape_model_finetuned_final.h5"
CLASS_NAMES_PATH = MODELS_DIR / "class_names.npy"

IMAGE_SIZE: Tuple[int, int] = (224, 224)


# ----------------------------------------------------------------------
# Model loading
# ----------------------------------------------------------------------

def load_model_and_classes() -> Tuple[tf.keras.Model, np.ndarray]:
    """
    Load the trained CNN model and class labels.

    Returns
    -------
    Tuple[tf.keras.Model, np.ndarray]
        Loaded Keras model and class name array.

    Raises
    ------
    FileNotFoundError
        If the model or class names file cannot be found.
    """

    if not MODEL_PATH.exists():
        raise FileNotFoundError(f"Model file not found: {MODEL_PATH}")

    if not CLASS_NAMES_PATH.exists():
        raise FileNotFoundError(f"Class names file not found: {CLASS_NAMES_PATH}")

    model = tf.keras.models.load_model(MODEL_PATH)
    class_names = np.load(CLASS_NAMES_PATH)

    return model, class_names


# ----------------------------------------------------------------------
# Image preprocessing
# ----------------------------------------------------------------------

def preprocess_image(image_path: Path) -> np.ndarray:
    """
    Load and preprocess an image for MobileNetV2 inference.

    Steps performed:
    1. Load image from disk
    2. Convert to RGB
    3. Resize to model input size
    4. Convert to NumPy array
    5. Apply MobileNetV2 preprocessing
    6. Expand dimensions to create batch

    Parameters
    ----------
    image_path : Path
        Path to the input image.

    Returns
    -------
    np.ndarray
        Preprocessed image tensor ready for model prediction.
    """

    if not image_path.exists():
        raise FileNotFoundError(f"Image not found: {image_path}")

    image = Image.open(image_path).convert("RGB")
    image = image.resize(IMAGE_SIZE)

    image_array = np.array(image, dtype=np.float32)
    image_array = preprocess_input(image_array)

    image_array = np.expand_dims(image_array, axis=0)

    return image_array


# ----------------------------------------------------------------------
# Prediction logic
# ----------------------------------------------------------------------

def predict_image(image_path: Path) -> Tuple[str, float]:
    """
    Predict the class of a given image.

    Parameters
    ----------
    image_path : Path
        Path to the image file.

    Returns
    -------
    Tuple[str, float]
        Predicted class label and confidence score.
    """

    model, class_names = load_model_and_classes()

    processed_image = preprocess_image(image_path)

    predictions = model.predict(processed_image, verbose=0)

    predicted_index = int(np.argmax(predictions[0]))
    confidence = float(predictions[0][predicted_index])

    predicted_class = class_names[predicted_index]

    return predicted_class, confidence


# ----------------------------------------------------------------------
# CLI interface
# ----------------------------------------------------------------------

def main() -> None:
    """
    Command-line interface for running inference.

    Example
    -------
    python src/predict.py test_grape_1.jpeg
    """

    parser = argparse.ArgumentParser(
        description="Predict grape disease from an input image."
    )

    parser.add_argument(
        "image",
        type=str,
        help="Path to the input image",
    )

    args = parser.parse_args()

    image_path = Path(args.image)

    predicted_class, confidence = predict_image(image_path)

    print("\nPrediction Result")
    print("-----------------")
    print(f"Image: {image_path}")
    print(f"Predicted class: {predicted_class}")
    print(f"Confidence: {confidence:.4f}")


if __name__ == "__main__":
    main()
