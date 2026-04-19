import io
import os
import pytest
import numpy as np
from unittest.mock import patch, MagicMock
import tensorflow as tf

from main import (
    app,
    load_and_preprocess_image,
    predict_image,
    run_classification_job,
    IMG_SIZE,
    CLASS_NAMES,
    THRESHOLD,
)

# This fixture creates a test client for the Flask app.
@pytest.fixture
def client():
    app.config["TESTING"] = True
    with app.test_client() as client:
        yield client


@pytest.fixture(autouse=True)
def reset_v1_jobs():
    import main

    with main.jobs_lock:
        main.jobs.clear()
        main.next_job_id = 1
    yield


# This fixture creates a tiny valid PNG on disk and returns its path.
@pytest.fixture
def sample_image_path(tmp_path):
    """Create a tiny valid PNG on disk and return its path."""
    img = np.random.randint(0, 255, (32, 32, 3), dtype=np.uint8)
    path = str(tmp_path / "sample.png")
    tf.io.write_file(path, tf.io.encode_png(img))
    return path


# This test checks if the image size is 224x224.
def test_img_size():
    assert IMG_SIZE == 224


# This test checks if the class names are correct.
def test_class_names():
    assert CLASS_NAMES == ["Healthy Coral", "Bleached Coral"]


# This test checks if the threshold is 0.5.
def test_threshold():
    assert THRESHOLD == 0.5


# This test checks if the preprocess returns the correct shape.
def test_preprocess_returns_correct_shape(sample_image_path):
    tensor = load_and_preprocess_image(sample_image_path)
    assert tensor.shape == (1, IMG_SIZE, IMG_SIZE, 3)


# This test checks if the preprocess values are normalised.
def test_preprocess_values_normalised(sample_image_path):
    tensor = load_and_preprocess_image(sample_image_path)
    assert tf.reduce_min(tensor).numpy() >= 0.0
    assert tf.reduce_max(tensor).numpy() <= 1.0


# This test checks if the preprocess dtype is correct.
def test_preprocess_dtype(sample_image_path):
    tensor = load_and_preprocess_image(sample_image_path)
    assert tensor.dtype == tf.float32


# This test checks if the predict image returns the correct label.
@patch("main.tf.keras.models.load_model")
def test_predict_returns_healthy(mock_load, sample_image_path):
    mock_model = MagicMock()
    mock_model.predict.return_value = np.array([[0.2]])
    mock_load.return_value = mock_model

    with app.test_request_context():
        response = predict_image(sample_image_path)
        data = response.get_json()

    assert data["predicted_label"] == "Healthy Coral"
    assert 0.0 <= data["probability"] <= 1.0


# This test checks if the predict image returns the correct label.
@patch("main.tf.keras.models.load_model")
def test_predict_returns_bleached(mock_load, sample_image_path):
    mock_model = MagicMock()
    mock_model.predict.return_value = np.array([[0.8]])
    mock_load.return_value = mock_model

    with app.test_request_context():
        response = predict_image(sample_image_path)
        data = response.get_json()

    assert data["predicted_label"] == "Bleached Coral"
    assert 0.0 <= data["probability"] <= 1.0


# This test checks if the predict image returns the correct label.
@patch("main.tf.keras.models.load_model")
def test_predict_at_threshold_boundary(mock_load, sample_image_path):
    """Probability == THRESHOLD should be classified as Bleached."""
    mock_model = MagicMock()
    mock_model.predict.return_value = np.array([[THRESHOLD]])
    mock_load.return_value = mock_model

    with app.test_request_context():
        response = predict_image(sample_image_path)
        data = response.get_json()

    assert data["predicted_label"] == "Bleached Coral"


# This test checks if the predict image returns the correct label.
@patch("main.tf.keras.models.load_model")
def test_predict_just_below_threshold(mock_load, sample_image_path):
    mock_model = MagicMock()
    mock_model.predict.return_value = np.array([[THRESHOLD - 0.01]])
    mock_load.return_value = mock_model

    with app.test_request_context():
        response = predict_image(sample_image_path)
        data = response.get_json()

    assert data["predicted_label"] == "Healthy Coral"


# This test checks if the classify endpoint returns the correct error.
def test_classify_no_image(client):
    response = client.post("/classify")
    assert response.status_code == 400
    assert response.get_json()["error"] == "No image provided"


# This test checks if the classify endpoint returns the correct label.
@patch("main.tf.keras.models.load_model")
def test_classify_with_image(mock_load, client, sample_image_path):
    mock_model = MagicMock()
    mock_model.predict.return_value = np.array([[0.3]])
    mock_load.return_value = mock_model

    with open(sample_image_path, "rb") as f:
        data = {"image": (io.BytesIO(f.read()), "test_coral.png")}
        response = client.post(
            "/classify",
            data=data,
            content_type="multipart/form-data",
        )

    assert response.status_code == 200
    body = response.get_json()
    assert "probability" in body
    assert "predicted_label" in body
    assert body["predicted_label"] == "Healthy Coral"


# This test checks if the classify endpoint returns the correct label.
@patch("main.tf.keras.models.load_model")
def test_classify_returns_bleached(mock_load, client, sample_image_path):
    mock_model = MagicMock()
    mock_model.predict.return_value = np.array([[0.9]])
    mock_load.return_value = mock_model

    with open(sample_image_path, "rb") as f:
        data = {"image": (io.BytesIO(f.read()), "test_coral.png")}
        response = client.post(
            "/classify",
            data=data,
            content_type="multipart/form-data",
        )

    assert response.status_code == 200
    assert response.get_json()["predicted_label"] == "Bleached Coral"


# This test checks if the classify endpoint returns the correct error.
def test_classify_wrong_method(client):
    response = client.get("/classify")
    assert response.status_code == 405


# This test checks if the upload folder exists.
def test_upload_folder_exists():
    assert os.path.isdir(app.config["UPLOAD_FOLDER"])


def test_v1_health(client):
    response = client.get("/V1/api/health")
    assert response.status_code == 200
    assert response.get_json() == {"status": "success"}


def test_v1_status_not_found(client):
    response = client.get("/V1/api/status/99999")
    assert response.status_code == 404


@patch("main.predict_image_dict")
def test_v1_classify_and_status(mock_predict, client, sample_image_path):
    mock_predict.return_value = {"probability": 0.3, "predicted_label": "Healthy Coral"}
    with patch(
        "main.start_classification_job",
        side_effect=lambda jid, p: run_classification_job(jid, p),
    ):
        with open(sample_image_path, "rb") as f:
            data = {"image": (io.BytesIO(f.read()), "test_coral.png")}
            response = client.post(
                "/V1/api/classify",
                data=data,
                content_type="multipart/form-data",
            )
    assert response.status_code == 200
    body = response.get_json()
    assert "job_id" in body
    job_id = body["job_id"]
    status_resp = client.get(f"/V1/api/status/{job_id}")
    assert status_resp.status_code == 200
    assert status_resp.get_json()["progress"] == "100%"


@patch("main.predict_image_dict")
def test_v1_delete_job(mock_predict, client, sample_image_path):
    mock_predict.return_value = {"probability": 0.3, "predicted_label": "Healthy Coral"}
    with patch(
        "main.start_classification_job",
        side_effect=lambda jid, p: run_classification_job(jid, p),
    ):
        with open(sample_image_path, "rb") as f:
            data = {"image": (io.BytesIO(f.read()), "test_coral.png")}
            response = client.post(
                "/V1/api/classify",
                data=data,
                content_type="multipart/form-data",
            )
    job_id = response.get_json()["job_id"]
    del_resp = client.delete(f"/V1/api/classify/{job_id}")
    assert del_resp.status_code == 200
    assert del_resp.get_json() == {"status": "success"}
    assert client.get(f"/V1/api/status/{job_id}").status_code == 404


def test_v1_delete_unknown_job(client):
    assert client.delete("/V1/api/classify/99999").status_code == 404


def test_v1_classify_no_image(client):
    response = client.post("/V1/api/classify")
    assert response.status_code == 400
