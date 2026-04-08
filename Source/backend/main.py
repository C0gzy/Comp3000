from flask import request, jsonify, Flask
from flask_cors import CORS
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import os

IMG_SIZE = 224
CLASS_NAMES = ["Healthy Coral", "Bleached Coral"]
THRESHOLD = 0.5  # probability threshold for binary classification
MODEL_PATH = "CurrentModel.h5"

app = Flask(__name__)

# CORS: allow frontend (Next.js default port) to call this API
CORS(app, origins=["http://localhost:3000", "http://127.0.0.1:3000"], methods=["GET", "POST", "OPTIONS"], allow_headers=["Content-Type"])

# Required for /classify route - saves uploaded images here
app.config["UPLOAD_FOLDER"] = os.path.join(os.path.dirname(__file__), "uploads")
os.makedirs(app.config["UPLOAD_FOLDER"], exist_ok=True)
print("Starting backend...")

def load_and_preprocess_image(image_path: str) -> tf.Tensor:
    #Load an image from disk, resize, normalise, and add batch dimension.
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

    return jsonify({'probability': probability, 'predicted_label': predicted_label})

@app.route('/classify', methods=['POST'])
def classify():
    print("Classifying image...")
    if 'image' not in request.files:
        return jsonify({'error': 'No image provided'}), 400
    image = request.files['image']
    image_path = os.path.join(app.config['UPLOAD_FOLDER'], image.filename)
    image.save(image_path)
    return predict_image(image_path)


if __name__ == "__main__":
    # Port 5001: macOS uses 5000 for AirPlay Receiver
    app.run(debug=True, port=5001)