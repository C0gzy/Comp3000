from flask import request, jsonify, Flask
from flask_cors import CORS
import os
import numpy as np
import threading
import base64
import tempfile
from io import BytesIO
from PIL import Image, ImageDraw
from werkzeug.utils import secure_filename
IMG_SIZE = 224
CHUNK_SIZE = 224
HEALTHY_LABEL = "Healthy Coral"
BLEACHED_LABEL = "Bleached Coral"
UNSURE_LABEL = "Unsure"
CLASS_NAMES = [HEALTHY_LABEL, BLEACHED_LABEL]
THRESHOLD = 0.5  # probability threshold for binary classification
BASE_DIR = os.path.dirname(__file__)
PRIMARY_MODEL_PATH = os.path.join(BASE_DIR, "PrimaryModel.tflite")
SECONDARY_MODEL_PATH = os.path.join(BASE_DIR, "SecondaryModel.tflite")
UNSURE_LOW = 0.4
UNSURE_HIGH = 0.6
SECONDARY_BONUS_WEIGHT = 0.2

app = Flask(__name__)

# CORS for the frontend to call the API
CORS(
    app,
    origins=[
        "http://localhost:3000",
        "http://127.0.0.1:3000",
        "https://comp3000.vercel.app",
    ],
    methods=["GET", "POST", "DELETE", "OPTIONS"],
    allow_headers=["Content-Type"],
)

jobs_lock = threading.Lock()
jobs: dict[int, dict] = {}
next_job_id = 1

# Required for /classify route - saves uploaded images here
app.config["UPLOAD_FOLDER"] = os.path.join(tempfile.gettempdir(), "coral-backend-uploads")
os.makedirs(app.config["UPLOAD_FOLDER"], exist_ok=True)
print("Starting backend...")


def load_model(model_path: str):
    from ai_edge_litert.interpreter import Interpreter

    interpreter = Interpreter(model_path=model_path)
    interpreter.allocate_tensors()
    return {
        "interpreter": interpreter,
        "input_details": interpreter.get_input_details()[0],
        "output_details": interpreter.get_output_details()[0],
    }


def load_ensemble_models():
    return load_model(PRIMARY_MODEL_PATH), load_model(SECONDARY_MODEL_PATH)


def load_and_preprocess_image(image_path: str) -> np.ndarray:
    image = Image.open(image_path)
    return preprocess_pil_image(image)

# Preprocess a PIL image for the model
def preprocess_pil_image(image: Image.Image) -> np.ndarray:
    image = image.convert("RGB").resize((IMG_SIZE, IMG_SIZE))
    image_array = np.asarray(image, dtype=np.float32) / 255.0
    return np.expand_dims(image_array, axis=0)

# This classifies the ensemble scores
def classify_ensemble_scores(primary_probability: float, secondary_probability: float):
    # If the primary probability is greater than the threshold, return the bleached label
    if primary_probability >= THRESHOLD:
        return {
            "probability": primary_probability,
            "predicted_label": BLEACHED_LABEL,
            "secondary_bonus": 0.0,
            "decision_reason": "primary_bleached",
        }

    # Calculate the secondary bonus
    secondary_bonus = secondary_probability * SECONDARY_BONUS_WEIGHT
    adjusted_probability = min(1.0, primary_probability + secondary_bonus)

    # If the secondary probability is between the unsure low and high thresholds return the unsure label
    if UNSURE_LOW <= secondary_probability <= UNSURE_HIGH:
        predicted_label = UNSURE_LABEL
        decision_reason = "secondary_unsure_primary_no"
    # If the adjusted probability is greater than the unsure low threshold return the unsure label
    elif adjusted_probability >= UNSURE_LOW:
        predicted_label = UNSURE_LABEL
        decision_reason = "bonus_near_threshold_primary_no"
    # If the adjusted probability is less than the unsure low threshold return the healthy label
    else:
        predicted_label = HEALTHY_LABEL
        decision_reason = "primary_no_low_support"

    return {
        "probability": adjusted_probability,
        "predicted_label": predicted_label,
        "secondary_bonus": secondary_bonus,
        "decision_reason": decision_reason,
    }


# This gets the probability of the model
def model_probability(model: dict, image: np.ndarray):
    if hasattr(model, "predict"):
        return float(model.predict(image, verbose=0)[0][0])

    interpreter = model["interpreter"]
    input_details = model["input_details"]
    output_details = model["output_details"]
    input_data = image.astype(input_details["dtype"])

    interpreter.set_tensor(input_details["index"], input_data)
    interpreter.invoke()
    output = interpreter.get_tensor(output_details["index"])

    scale, zero_point = output_details.get("quantization", (0.0, 0))
    if scale:
        output = (output.astype(np.float32) - zero_point) * scale

    return float(np.ravel(output)[0])


# This predicts the preprocessed image
def predict_preprocessed_image(primary_model, secondary_model, image: np.ndarray):
    primary_probability = model_probability(primary_model, image)
    secondary_probability = model_probability(secondary_model, image)
    prediction = classify_ensemble_scores(primary_probability, secondary_probability)
    prediction.update({
        "primary_probability": primary_probability,
        "secondary_probability": secondary_probability,
    })
    return prediction


# This predicts the image
def predict_image_dict(image_path: str):
    print(f"Loading primary model from '{PRIMARY_MODEL_PATH}'...")
    print(f"Loading secondary model from '{SECONDARY_MODEL_PATH}'...")
    primary_model, secondary_model = load_ensemble_models()

    print(f"Preprocessing image '{image_path}'...")
    image = load_and_preprocess_image(image_path)
    print("Running prediction...")
    result = predict_preprocessed_image(primary_model, secondary_model, image)

    print(f"\nPrediction results for '{image_path}':")
    print(f"- Probability of bleaching: {result['probability']:.4f}")
    print(f"- Predicted class: {result['predicted_label']}")

    return result


def predict_image(image_path: str):
    return jsonify(predict_image_dict(image_path))


def get_image_chunks(image: Image.Image):
    width, height = image.size
    chunks = []
    for y in range(0, height, CHUNK_SIZE):
        for x in range(0, width, CHUNK_SIZE):
            chunk_width = min(CHUNK_SIZE, width - x)
            chunk_height = min(CHUNK_SIZE, height - y)
            crop = image.crop((x, y, x + chunk_width, y + chunk_height))
            chunks.append({
                "x": x,
                "y": y,
                "width": chunk_width,
                "height": chunk_height,
                "image": crop,
            })
    return chunks

# This is the colour of what each chunk of the image will be
def overlay_colour(probability: float, predicted_label: str | None = None):
    if predicted_label == BLEACHED_LABEL or probability >= THRESHOLD:
        # Red
        return (239, 68, 68, 90)
    if predicted_label == UNSURE_LABEL or 0.4 <= probability <= 0.6:
        # Orange
        return (251, 146, 60, 90)
    # Green
    return (34, 197, 94, 90)

# This draws the grid on the main image and colours each chuck of the image
def build_result_image(original_image: Image.Image, chunks: list[dict]):
    base = original_image.convert("RGBA")
    overlay = Image.new("RGBA", base.size, (0, 0, 0, 0))
    draw = ImageDraw.Draw(overlay)

    # for loop through each chunk and draw the grid and colour the chunk
    for chunk in chunks:
        colour = overlay_colour(chunk["probability"], chunk.get("predicted_label"))
        outline = colour[:3] + (220,)
        x = chunk["x"]
        y = chunk["y"]
        width = chunk["width"]
        height = chunk["height"]
        draw.rectangle(
            (x, y, x + width - 1, y + height - 1),
            fill=colour,
            outline=outline,
            width=2,
        )

    output = Image.alpha_composite(base, overlay).convert("RGB")
    buffer = BytesIO()
    output.save(buffer, format="PNG")
    # Encode the image to base64
    encoded = base64.b64encode(buffer.getvalue()).decode("utf-8")
    return f"data:image/png;base64,{encoded}"

# This gets the status of each job
def job_status_response(job: dict):
    completed_chunks = job.get("completed_chunks", 0)
    total_chunks = job.get("total_chunks", 0)
    response = {
        "status": job["status"],
        "completed_chunks": completed_chunks,
        "total_chunks": total_chunks,
        "remaining_chunks": max(total_chunks - completed_chunks, 0),
        "progress": f"{completed_chunks}/{total_chunks}",
    }
    if job.get("error"):
        response["error"] = job["error"]
    if job["status"] == "completed" and job.get("result"):
        response.update(job["result"])
    return response


def run_classification_job(job_id: int, image_path: str):
    try:
        with jobs_lock:
            job = jobs.get(job_id)
            if not job or job.get("cancelled"):
                return
            job["status"] = "processing"

        print(f"Loading primary model from '{PRIMARY_MODEL_PATH}'...")
        print(f"Loading secondary model from '{SECONDARY_MODEL_PATH}'...")
        primary_model, secondary_model = load_ensemble_models()
        original_image = Image.open(image_path).convert("RGB")
        image_chunks = get_image_chunks(original_image)

        with jobs_lock:
            job = jobs.get(job_id)
            if not job or job.get("cancelled"):
                return
            job["total_chunks"] = len(image_chunks)

        bleached_chunks = 0
        healthy_chunks = 0
        unsure_chunks = 0
        chunk_results = []
        for index, chunk in enumerate(image_chunks, start=1):
            with jobs_lock:
                job = jobs.get(job_id)
                if not job or job.get("cancelled"):
                    return

            image = preprocess_pil_image(chunk["image"])
            prediction = predict_preprocessed_image(primary_model, secondary_model, image)
            chunk_result = {
                "x": chunk["x"],
                "y": chunk["y"],
                "width": chunk["width"],
                "height": chunk["height"],
                "probability": prediction["probability"],
                "predicted_label": prediction["predicted_label"],
                "primary_probability": prediction["primary_probability"],
                "secondary_probability": prediction["secondary_probability"],
                "secondary_bonus": prediction["secondary_bonus"],
                "decision_reason": prediction["decision_reason"],
            }
            if chunk_result["predicted_label"] == BLEACHED_LABEL:
                bleached_chunks += 1
            elif chunk_result["predicted_label"] == HEALTHY_LABEL:
                healthy_chunks += 1
            elif chunk_result["predicted_label"] == UNSURE_LABEL:
                unsure_chunks += 1

            chunk_results.append(chunk_result)

            with jobs_lock:
                job = jobs.get(job_id)
                if not job or job.get("cancelled"):
                    return
                job["completed_chunks"] = index

        result_image = build_result_image(original_image, chunk_results)

        with jobs_lock:
            job = jobs.get(job_id)
            if not job or job.get("cancelled"):
                return
            job["result"] = {
                "result_image": result_image,
                "chunks": chunk_results,
                "summary": {
                    "bleached_chunks": bleached_chunks,
                    "healthy_chunks": healthy_chunks,
                    "unsure_chunks": unsure_chunks,
                    "total_chunks": len(chunk_results),
                    "primary_model": os.path.basename(PRIMARY_MODEL_PATH),
                    "secondary_model": os.path.basename(SECONDARY_MODEL_PATH),
                },
            }
            job["status"] = "completed"
    except Exception as e:
        with jobs_lock:
            job = jobs.get(job_id)
            if job and not job.get("cancelled"):
                job["status"] = "error"
                job["error"] = str(e)


def start_classification_job(job_id: int, image_path: str):
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
    return jsonify(job_status_response(job)), 200


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
            "completed_chunks": 0,
            "total_chunks": 0,
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