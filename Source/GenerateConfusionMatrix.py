import warnings
warnings.filterwarnings('ignore', category=UserWarning, module='google.protobuf')

import os
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, classification_report
from tqdm import tqdm

MODEL_PATH = "./ModelHistory/V20251116_132020.h5"
IMG_SIZE = 224
BATCH_SIZE = 32
THRESHOLD = 0.5
CLASS_NAMES = ['Healthy', 'Bleached']

TEST_CORAL_DIR = "../CoralDataSet/test/CORAL"
TEST_BLEACHED_DIR = "../CoralDataSet/test/CORAL_BL"


def load_image_paths(image_dir):
    paths = []
    for f in tqdm(os.listdir(image_dir), desc=f"Scanning {os.path.basename(image_dir)}", unit="file"):
        if f.lower().endswith((".jpg", ".jpeg", ".png")):
            paths.append(os.path.join(image_dir, f))
    return paths


def load_and_preprocess_image(image_path, label):
    img = tf.io.read_file(image_path)
    img = tf.image.decode_image(img, channels=3, expand_animations=False)
    img = tf.image.resize(img, [IMG_SIZE, IMG_SIZE])
    img = tf.cast(img, tf.float32) / 255.0
    return img, label


def main():
    print(f"Loading model from '{MODEL_PATH}'...")
    model = tf.keras.models.load_model(MODEL_PATH)

    coral_paths = load_image_paths(TEST_CORAL_DIR)
    bleached_paths = load_image_paths(TEST_BLEACHED_DIR)

    all_paths = coral_paths + bleached_paths
    all_labels = [0] * len(coral_paths) + [1] * len(bleached_paths)

    print(f"\nTest set: {len(coral_paths)} healthy, {len(bleached_paths)} bleached ({len(all_paths)} total)")

    dataset = tf.data.Dataset.from_tensor_slices((all_paths, all_labels))
    dataset = dataset.map(load_and_preprocess_image, num_parallel_calls=tf.data.AUTOTUNE)
    dataset = dataset.batch(BATCH_SIZE).prefetch(tf.data.AUTOTUNE)

    print("Running predictions...")
    probabilities = model.predict(dataset, verbose=1).flatten()
    predictions = (probabilities >= THRESHOLD).astype(int)
    true_labels = np.array(all_labels)

    cm = confusion_matrix(true_labels, predictions)

    print(f"\n{classification_report(true_labels, predictions, target_names=CLASS_NAMES)}")

    fig, ax = plt.subplots(figsize=(6, 5))
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=CLASS_NAMES)
    disp.plot(cmap='Blues', ax=ax, values_format='d')
    ax.set_title(f'Confusion Matrix (threshold={THRESHOLD})')
    fig.tight_layout()

    output_path = "confusion_matrix.png"
    fig.savefig(output_path, dpi=150)
    print(f"\nSaved to '{output_path}'")
    plt.show()


if __name__ == "__main__":
    main()
