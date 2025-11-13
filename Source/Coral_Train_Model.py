import warnings
# Suppress protobuf version warning (harmless compatibility warning)
warnings.filterwarnings('ignore', category=UserWarning, module='google.protobuf')

import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import os
import asyncio
from tqdm import tqdm

print("Starting Coral Train Model...")

# Configuration
IMG_SIZE = 224  # Target width/height in pixels (matches dataset patches)
BATCH_SIZE = 32  # Number of images per training step
EPOCHS = 25  # Max number of times the model sees the full training set
LEARNING_RATE = 0.0005  # Initial step size for the Adam optimizer
USE_CLASS_WEIGHTS = True  # Enable class weighting to handle imbalance
OPTIMIZE_THRESHOLD = True  # Find optimal classification threshold

async def load_images(image_dir):
    images = []
    files = os.listdir(image_dir)
    for file in tqdm(files, desc=f"Loading images from {os.path.basename(image_dir)}", unit="file"):
        if file.lower().endswith((".jpg", ".jpeg", ".png")):
            images.append(os.path.join(image_dir, file))
    return images

def load_and_preprocess_image(image_path, label):
    #Load and preprocess a single image
    img = tf.io.read_file(image_path)
    img = tf.image.decode_image(img, channels=3, expand_animations=False)
    img = tf.image.resize(img, [IMG_SIZE, IMG_SIZE])
    img = tf.cast(img, tf.float32) / 255.0  # Normalize to [0, 1]
    return img, label

def create_dataset(image_paths, labels, shuffle=True, batch_size=BATCH_SIZE):
    #Create a TensorFlow dataset from image paths and labels
    dataset = tf.data.Dataset.from_tensor_slices((image_paths, labels))
    if shuffle:
        dataset = dataset.shuffle(buffer_size=len(image_paths))
    dataset = dataset.map(load_and_preprocess_image, num_parallel_calls=tf.data.AUTOTUNE)
    dataset = dataset.batch(batch_size)
    dataset = dataset.prefetch(tf.data.AUTOTUNE)
    return dataset

def create_model():
    #Create a CNN model for binary classification
    model = tf.keras.Sequential([
        # First convolutional block
        tf.keras.layers.Conv2D(32, (3, 3), activation='relu', input_shape=(IMG_SIZE, IMG_SIZE, 3)),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.MaxPooling2D(2, 2),
        tf.keras.layers.Dropout(0.25),
        
        # Second convolutional block
        tf.keras.layers.Conv2D(64, (3, 3), activation='relu'),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.MaxPooling2D(2, 2),
        tf.keras.layers.Dropout(0.25),
        
        # Third convolutional block
        tf.keras.layers.Conv2D(128, (3, 3), activation='relu'),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.MaxPooling2D(2, 2),
        tf.keras.layers.Dropout(0.25),
        
        # Fourth convolutional block
        tf.keras.layers.Conv2D(256, (3, 3), activation='relu'),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.MaxPooling2D(2, 2),
        tf.keras.layers.Dropout(0.25),
        
        # Flatten and dense layers
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(512, activation='relu'),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.Dropout(0.5),
        tf.keras.layers.Dense(256, activation='relu'),
        tf.keras.layers.Dropout(0.5),
        
        # Output layer (binary classification)
        tf.keras.layers.Dense(1, activation='sigmoid')
    ])
    
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=LEARNING_RATE),
        loss='binary_crossentropy',
        metrics=['accuracy', tf.keras.metrics.Precision(), tf.keras.metrics.Recall()]
    )
    
    return model

def calculate_class_weights(labels):
    #Calculate class weights to handle imbalanced datasets
    from sklearn.utils.class_weight import compute_class_weight
    
    unique_labels = np.unique(labels)
    class_weights = compute_class_weight(
        'balanced',
        classes=unique_labels,
        y=labels
    )
    # Convert to dictionary format for TensorFlow
    class_weight_dict = dict(zip(unique_labels, class_weights))
    print(f"\nClass weights: {class_weight_dict}")
    print(f"  Class 0 (Healthy): {class_weight_dict[0]:.4f}")
    print(f"  Class 1 (Bleached): {class_weight_dict[1]:.4f}")
    return class_weight_dict

def evaluate_with_threshold(model, dataset, labels, threshold=0.5):
    #Evaluate model with a custom classification threshold
    # Get predictions (probabilities)
    predictions = model.predict(dataset, verbose=0)
    
    # Apply threshold
    binary_predictions = (predictions >= threshold).astype(int).flatten()
    
    # Calculate metrics
    from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
    
    accuracy = accuracy_score(labels, binary_predictions)
    precision = precision_score(labels, binary_predictions, zero_division=0)
    recall = recall_score(labels, binary_predictions, zero_division=0)
    f1 = f1_score(labels, binary_predictions, zero_division=0)
    cm = confusion_matrix(labels, binary_predictions)
    
    return {
        'threshold': threshold,
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'confusion_matrix': cm,
        'predictions': predictions.flatten()
    }

def find_optimal_threshold(model, dataset, labels, metric='f1'):
    #Find optimal threshold that maximizes specified metric (f1, recall, or precision)
    print(f"\nFinding optimal threshold to maximize {metric}...")
    
    thresholds = np.arange(0.1, 0.95, 0.05)
    results = []
    
    for threshold in thresholds:
        metrics = evaluate_with_threshold(model, dataset, labels, threshold)
        results.append({
            'threshold': threshold,
            'f1': metrics['f1'],
            'precision': metrics['precision'],
            'recall': metrics['recall'],
            'accuracy': metrics['accuracy']
        })
    
    # Find best threshold based on metric
    if metric == 'f1':
        best = max(results, key=lambda x: x['f1'])
    elif metric == 'recall':
        best = max(results, key=lambda x: x['recall'])
    elif metric == 'precision':
        best = max(results, key=lambda x: x['precision'])
    else:
        best = max(results, key=lambda x: x['f1'])
    
    print(f"\nOptimal threshold: {best['threshold']:.3f}")
    print(f"  F1 Score: {best['f1']:.4f}")
    print(f"  Precision: {best['precision']:.4f}")
    print(f"  Recall: {best['recall']:.4f}")
    print(f"  Accuracy: {best['accuracy']:.4f}")
    
    return best['threshold']

async def main():
    print("Loading images...")
    print(f"TensorFlow has access to the following devices:\n{tf.config.list_physical_devices()}")
    
    # Load training images
    train_coral_images = await load_images("../CoralDataSetAugmented/train/CORAL")
    train_bleached_images = await load_images("../CoralDataSetAugmented/train/CORAL_BL")
    
    # Load validation images
    val_coral_images = await load_images("../CoralDataSet/val/CORAL")
    val_bleached_images = await load_images("../CoralDataSet/val/CORAL_BL")
    
    # Load test images
    test_coral_images = await load_images("../CoralDataSet/test/CORAL")
    test_bleached_images = await load_images("../CoralDataSet/test/CORAL_BL")
    
    print(f"\nDataset sizes:")
    print(f"Train - Healthy: {len(train_coral_images)}, Bleached: {len(train_bleached_images)}")
    print(f"Val   - Healthy: {len(val_coral_images)}, Bleached: {len(val_bleached_images)}")
    print(f"Test  - Healthy: {len(test_coral_images)}, Bleached: {len(test_bleached_images)}")
    
    # Create labels: 0 = healthy coral, 1 = bleached coral
    train_images = train_coral_images + train_bleached_images
    train_labels = [0] * len(train_coral_images) + [1] * len(train_bleached_images)
    
    val_images = val_coral_images + val_bleached_images
    val_labels = [0] * len(val_coral_images) + [1] * len(val_bleached_images)
    
    test_images = test_coral_images + test_bleached_images
    test_labels = [0] * len(test_coral_images) + [1] * len(test_bleached_images)
    
    # Calculate class weights if enabled
    class_weight_dict = None
    if USE_CLASS_WEIGHTS:
        class_weight_dict = calculate_class_weights(train_labels)
    
    # Create TensorFlow datasets
    print("\nCreating datasets...")
    train_dataset = create_dataset(train_images, train_labels, shuffle=True, batch_size=BATCH_SIZE)
    val_dataset = create_dataset(val_images, val_labels, shuffle=False, batch_size=BATCH_SIZE)
    test_dataset = create_dataset(test_images, test_labels, shuffle=False, batch_size=BATCH_SIZE)
    
    # Create the model
    print("\nCreating model...")
    model = create_model()
    model.summary()
    
    # Callbacks
    callbacks = [
        tf.keras.callbacks.EarlyStopping(
            monitor='val_loss',
            patience=10,  # Stop training if validation loss doesn't improve for 5 consecutive epochs
            restore_best_weights=True,
            verbose=1
        ),
        tf.keras.callbacks.ModelCheckpoint(
            'best_coral_model.h5',
            monitor='val_loss',
            save_best_only=True,
            save_weights_only=False,
            verbose=1
        ),
        tf.keras.callbacks.ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.5,
            patience=2,  # Reduce learning rate if validation loss doesn't improve for 2 consecutive epochs
            min_lr=1e-7,
            verbose=1
        )
    ]
    
    # Train the model

    print("Training model...")

    history = model.fit(
        train_dataset,
        epochs=EPOCHS,
        validation_data=val_dataset,
        callbacks=callbacks,
        class_weight=class_weight_dict,  # Apply class weights if enabled
        verbose=1
    )
    
    # Evaluate on test set with default threshold (0.5)

    print("Evaluating on test set (threshold=0.5)...")

    test_results = model.evaluate(test_dataset, verbose=1)
    print(f"\nTest Results (threshold=0.5):")
    print(f"Loss: {test_results[0]:.4f}")
    print(f"Accuracy: {test_results[1]:.4f}")
    print(f"Precision: {test_results[2]:.4f}")
    print(f"Recall: {test_results[3]:.4f}")
    
    # Find optimal threshold if enabled
    optimal_threshold = 0.5
    if OPTIMIZE_THRESHOLD:
    
        print("Threshold Optimization")

        optimal_threshold = find_optimal_threshold(model, val_dataset, val_labels, metric='f1')
        
        # Evaluate with optimal threshold
    
        print(f"Evaluating on test set (threshold={optimal_threshold:.3f})...")

        optimal_results = evaluate_with_threshold(model, test_dataset, test_labels, optimal_threshold)
        print(f"\nTest Results (optimal threshold={optimal_threshold:.3f}):")
        print(f"Accuracy: {optimal_results['accuracy']:.4f}")
        print(f"Precision: {optimal_results['precision']:.4f}")
        print(f"Recall: {optimal_results['recall']:.4f}")
        print(f"F1 Score: {optimal_results['f1']:.4f}")
        print(f"\nConfusion Matrix:")
        print(f"  True Negatives: {optimal_results['confusion_matrix'][0][0]}")
        print(f"  False Positives: {optimal_results['confusion_matrix'][0][1]}")
        print(f"  False Negatives: {optimal_results['confusion_matrix'][1][0]}")
        print(f"  True Positives: {optimal_results['confusion_matrix'][1][1]}")
    
    # Save final model
    model.save('coral_bleaching_model.h5')
    print("\nModel saved as 'coral_bleaching_model.h5'")
    
    # Plot training history
    plt.figure(figsize=(12, 4))
    
    plt.subplot(1, 2, 1)
    plt.plot(history.history['accuracy'], label='Train Accuracy')
    plt.plot(history.history['val_accuracy'], label='Val Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.title('Model Accuracy')
    
    plt.subplot(1, 2, 2)
    plt.plot(history.history['loss'], label='Train Loss')
    plt.plot(history.history['val_loss'], label='Val Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.title('Model Loss')
    
    plt.tight_layout()
    plt.savefig('training_history.png')
    print("Training history plot saved as 'training_history.png'")






if __name__ == "__main__":
    asyncio.run(main())
