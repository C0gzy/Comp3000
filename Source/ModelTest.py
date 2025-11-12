# Simple script to load the trained coral bleaching model and run a single-image prediction

import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

MODEL_PATH = "coral_bleaching_model.h5"
IMG_SIZE = 224
CLASS_NAMES = ["Healthy Coral", "Bleached Coral"]
THRESHOLD = 0.5  # probability threshold for binary classification


def load_and_preprocess_image(image_path: str) -> tf.Tensor:
    """Load an image from disk, resize, normalise, and add batch dimension."""
    image = tf.io.read_file(image_path)
    image = tf.image.decode_image(image, channels=3, expand_animations=False)
    image = tf.image.resize(image, [IMG_SIZE, IMG_SIZE])
    image = tf.cast(image, tf.float32) / 255.0
    image = tf.expand_dims(image, axis=0)  # shape: (1, IMG_SIZE, IMG_SIZE, 3)
    return image


def predict_image(image_path: str) -> None:
    print(f"Loading model from '{MODEL_PATH}'...")
    model = tf.keras.models.load_model(MODEL_PATH)

    print(f"Preprocessing image '{image_path}'...")
    image = load_and_preprocess_image(image_path)

    print("Running prediction...")
    probability = float(model.predict(image, verbose=0)[0][0])
    predicted_label = CLASS_NAMES[int(probability >= THRESHOLD)]

    print(f"\nPrediction results for '{image_path}':")
    print(f"- Probability of bleaching: {probability:.4f}")
    print(f"- Predicted class: {predicted_label}")

    # Optional: display the image
    plt.imshow(tf.squeeze(image))
    plt.title(f"{predicted_label} (p={probability:.2f})")
    plt.axis("off")
    plt.show()


if __name__ == "__main__":
    # Example usage: point to any image within the dataset
    SAMPLE_IMAGE = "../CoralDataSet/test/CORAL_BL/KawaihaeShallow2015IMG_5205_2340.PNG"
    predict_image(SAMPLE_IMAGE)