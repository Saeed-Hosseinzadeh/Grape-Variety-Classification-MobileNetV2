"""
Grad-CAM visualization for the grape disease classifier.

This script generates a Grad-CAM heatmap highlighting the regions of an
input image that most influenced the model's prediction.

Output:
    evaluation/gradcam_result.png
"""

from pathlib import Path

import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

from PIL import Image
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input


# ---------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------

CURRENT_DIR = Path(__file__).resolve().parent
BASE_DIR = CURRENT_DIR.parent

MODELS_DIR = BASE_DIR / "models"
EVAL_DIR = BASE_DIR / "evaluation"

MODEL_PATH = MODELS_DIR / "grape_model_finetuned_final.h5"

INPUT_IMAGE = BASE_DIR / "assets" / "test_grape_1.jpeg"
OUTPUT_PATH = EVAL_DIR / "gradcam_result.png"

IMAGE_SIZE = (224, 224)


def preprocess_image(image_path: Path):
    """
    Load and preprocess an image for model inference.
    """

    image = Image.open(image_path).convert("RGB")
    image = image.resize(IMAGE_SIZE)

    img_array = np.array(image)
    img_array = preprocess_input(img_array)

    img_array = np.expand_dims(img_array, axis=0)

    return image, img_array


def find_last_conv_layer(model):
    """
    Automatically locate the last convolutional layer in the model.
    """

    for layer in reversed(model.layers):
        if isinstance(layer, tf.keras.layers.Conv2D):
            return layer.name

    raise ValueError("No convolutional layer found in the model.")


def make_gradcam_heatmap(img_array, model, last_conv_layer_name):
    """
    Generate Grad-CAM heatmap.
    """

    grad_model = tf.keras.models.Model(
        model.inputs,
        [model.get_layer(last_conv_layer_name).output, model.output],
    )

    with tf.GradientTape() as tape:
        conv_outputs, predictions = grad_model(img_array)
        class_index = tf.argmax(predictions[0])
        loss = predictions[:, class_index]

    grads = tape.gradient(loss, conv_outputs)

    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))

    conv_outputs = conv_outputs[0]

    heatmap = conv_outputs @ pooled_grads[..., tf.newaxis]
    heatmap = tf.squeeze(heatmap)

    heatmap = tf.maximum(heatmap, 0) / tf.math.reduce_max(heatmap)

    return heatmap.numpy()


def overlay_heatmap(image, heatmap):
    """
    Overlay Grad-CAM heatmap on the original image.
    """

    heatmap = np.uint8(255 * heatmap)

    heatmap = Image.fromarray(heatmap).resize(image.size)
    heatmap = np.array(heatmap)

    plt.figure(figsize=(6, 6))

    plt.imshow(image)
    plt.imshow(heatmap, cmap="jet", alpha=0.4)

    plt.axis("off")
    plt.title("Grad-CAM Visualization")

    plt.savefig(OUTPUT_PATH, bbox_inches="tight", dpi=300)
    plt.close()


def run_gradcam():
    """
    Generate Grad-CAM visualization for the input image.
    """

    EVAL_DIR.mkdir(exist_ok=True)

    print("Loading model...")
    model = tf.keras.models.load_model(MODEL_PATH)

    print("Finding last convolution layer...")
    last_conv_layer = find_last_conv_layer(model)

    print(f"Using layer: {last_conv_layer}")

    image, img_array = preprocess_image(INPUT_IMAGE)

    heatmap = make_gradcam_heatmap(img_array, model, last_conv_layer)

    overlay_heatmap(image, heatmap)

    print(f"Grad-CAM result saved to: {OUTPUT_PATH}")


if __name__ == "__main__":
    run_gradcam()
