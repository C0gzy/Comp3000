from flask import request, jsonify, Flask
from flask_cors import CORS
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import os
import threading
from werkzeug.utils import secure_filename

IMG_SIZE = 224
CLASS_NAMES = ["Healthy Coral", "Bleached Coral"]
THRESHOLD = 0.5  # probability threshold for binary classification
MODEL_PATH = "CurrentModel.h5"

app = Flask(__name__)

# CORS for the frontend to call the API
CORS(
    app,
    origins=["http://localhost:3000", "http://127.0.0.1:3000"],
    methods=["GET", "POST", "DELETE", "OPTIONS"],
    allow_headers=["Content-Type"],
)

jobs_lock = threading.Lock()
jobs: dict[int, dict] = {}
next_job_id = 1

# Required for /classify route - saves uploaded images here
app.config["UPLOAD_FOLDER"] = os.path.join(os.path.dirname(__file__), "uploads")
os.makedirs(app.config["UPLOAD_FOLDER"], exist_ok=True)
print("Starting backend...")

def load_and_preprocess_image(image_path: str) -> tf.Tensor:
    #Load an image , resize, normalise, and add batch dimension.
    image = tf.io.read_file(image_path)
    image = tf.image.decode_image(image, channels=3, expand_animations=False)
    image = tf.image.resize(image, [IMG_SIZE, IMG_SIZE])
    image = tf.cast(image, tf.float32) / 255.0
    image = tf.expand_dims(image, axis=0)  # shape: (1, IMG_SIZE, IMG_SIZE, 3)
    return image

def predict_image_dict(image_path: str) -> dict:
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

    return {"probability": probability, "predicted_label": predicted_label}


def predict_image(image_path: str):
    return jsonify(predict_image_dict(image_path))


def run_classification_job(job_id: int, image_path: str) -> None:
    try:
        with jobs_lock:
            job = jobs.get(job_id)
            if not job or job.get("cancelled"):
                return
            job["status"] = "processing"

        result = predict_image_dict(image_path)

        with jobs_lock:
            job = jobs.get(job_id)
            if not job or job.get("cancelled"):
                return
            job["result"] = result
            job["progress"] = 100
            job["status"] = "completed"
    except Exception as e:
        with jobs_lock:
            job = jobs.get(job_id)
            if job and not job.get("cancelled"):
                job["status"] = "error"
                job["error"] = str(e)
                job["progress"] = 100


def start_classification_job(job_id: int, image_path: str) -> None:
    thread = threading.Thread(target=run_classification_job, args=(job_id, image_path), daemon=True)
    thread.start()


@app.route("/V1/api/health", methods=["GET"])
def v1_health():
    return jsonify({"status": "success"}), 200


@app.route("/V1/api/status/<int:job_id>", methods=["GET"])
def v1_job_status(job_id: int):
    with jobs_lock:
        job = jobs.get(job_id)
    if not job:
        return jsonify({"error": "Job not found"}), 404
    return jsonify({"progress": f"{job['progress']}%"}), 200


@app.route("/V1/api/classify", methods=["POST"])
def v1_classify():
    if "image" not in request.files:
        return jsonify({"error": "No image provided"}), 400
    image = request.files["image"]
    if not image.filename:
        return jsonify({"error": "No filename"}), 400

    filename = secure_filename(image.filename) or "upload.bin"
    global next_job_id
    with jobs_lock:
        job_id = next_job_id
        next_job_id += 1
        jobs[job_id] = {
            "progress": 0,
            "status": "queued",
            "result": None,
            "error": None,
            "cancelled": False,
        }

    image_path = os.path.join(app.config["UPLOAD_FOLDER"], f"{job_id}_{filename}")
    image.save(image_path)
    start_classification_job(job_id, image_path)
    return jsonify({"job_id": job_id}), 200


@app.route("/V1/api/classify/<int:job_id>", methods=["DELETE"])
def v1_delete_job(job_id: int):
    with jobs_lock:
        if job_id not in jobs:
            return jsonify({"error": "Job not found"}), 404
        jobs[job_id]["cancelled"] = True
        del jobs[job_id]
    return jsonify({"status": "success"}), 200

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
    app.run(debug=True, port=5001)