"""
Face Mask Detection - Training Script
Trains the MobileNetV2-based mask detector on the prepared dataset.

Usage:
    python train.py                    # Train with default settings (20 epochs)
    python train.py --epochs 10        # Custom epoch count
    python train.py --batch-size 16    # Custom batch size
"""

import os
import argparse
import numpy as np
import matplotlib
matplotlib.use("Agg")  # Non-interactive backend for saving plots
import matplotlib.pyplot as plt

from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.utils import to_categorical
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from imutils import paths

from model import build_mask_detector

# Suppress TF info logs
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

import tensorflow as tf
import cv2


# ─── Configuration ──────────────────────────────────────────────────────────────
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATASET_DIR = os.path.join(BASE_DIR, "dataset")
MODELS_DIR = os.path.join(BASE_DIR, "models")
MODEL_PATH = os.path.join(MODELS_DIR, "mask_detector.h5")
PLOT_PATH = os.path.join(MODELS_DIR, "training_plot.png")

IMAGE_SIZE = 224
INIT_LR = 1e-4


def load_dataset():
    """Load images and labels from the dataset directory."""
    print("\n📂 Loading dataset...")

    image_paths = list(paths.list_images(DATASET_DIR))
    data = []
    labels = []

    for img_path in image_paths:
        # Extract label from parent directory name
        label = img_path.split(os.path.sep)[-2]

        # Load and preprocess image
        image = cv2.imread(img_path)
        if image is None:
            continue
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = cv2.resize(image, (IMAGE_SIZE, IMAGE_SIZE))

        data.append(image)
        labels.append(label)

    # Convert to numpy arrays
    data = np.array(data, dtype="float32")
    labels = np.array(labels)

    print(f"   Loaded {len(data)} images")
    print(f"   Classes: {np.unique(labels)}")

    return data, labels


def preprocess_data(data, labels):
    """Preprocess images and encode labels."""
    # Normalize pixel values to [0, 1]
    data = data / 255.0

    # Encode labels: with_mask=0, without_mask=1
    le = LabelEncoder()
    labels_encoded = le.fit_transform(labels)
    labels_onehot = to_categorical(labels_encoded)

    print(f"\n🏷️  Label mapping: {dict(zip(le.classes_, le.transform(le.classes_)))}")

    return data, labels_onehot, le


def create_data_augmentor():
    """Create an image data augmentation generator."""
    return ImageDataGenerator(
        rotation_range=20,
        zoom_range=0.15,
        width_shift_range=0.2,
        height_shift_range=0.2,
        shear_range=0.15,
        horizontal_flip=True,
        fill_mode="nearest",
    )


def train(epochs=20, batch_size=32):
    """Main training pipeline."""
    print("=" * 60)
    print("  🎭 Face Mask Detection - Model Training")
    print("=" * 60)

    # Ensure models directory exists
    os.makedirs(MODELS_DIR, exist_ok=True)

    # 1. Load dataset
    data, labels = load_dataset()

    if len(data) == 0:
        print("\n❌ No images found in dataset/ directory!")
        print("   Run `python prepare_dataset.py` first.")
        return

    # 2. Preprocess
    data, labels_onehot, label_encoder = preprocess_data(data, labels)

    # 3. Split into train/test
    print("\n📊 Splitting dataset (80% train / 20% test)...")
    X_train, X_test, y_train, y_test = train_test_split(
        data, labels_onehot, test_size=0.20, stratify=labels_onehot, random_state=42
    )
    print(f"   Training:   {len(X_train)} images")
    print(f"   Testing:    {len(X_test)} images")

    # 4. Data augmentation
    augmentor = create_data_augmentor()

    # 5. Build model
    print("\n🧠 Building model...")
    model = build_mask_detector()
    model.compile(
        loss="binary_crossentropy",
        optimizer=Adam(learning_rate=INIT_LR),
        metrics=["accuracy"],
    )

    # 6. Train!
    print(f"\n🚀 Training for {epochs} epochs (batch size: {batch_size})...")
    print("-" * 60)

    history = model.fit(
        augmentor.flow(X_train, y_train, batch_size=batch_size),
        steps_per_epoch=len(X_train) // batch_size,
        validation_data=(X_test, y_test),
        validation_steps=len(X_test) // batch_size,
        epochs=epochs,
    )

    # 7. Evaluate
    print("\n" + "=" * 60)
    print("  📈 Evaluation Results")
    print("=" * 60)

    predictions = model.predict(X_test, batch_size=batch_size)
    pred_indices = np.argmax(predictions, axis=1)

    print(classification_report(
        np.argmax(y_test, axis=1),
        pred_indices,
        target_names=label_encoder.classes_,
    ))

    # 8. Save model
    print(f"\n💾 Saving model to {MODEL_PATH}...")
    model.save(MODEL_PATH)
    print("✅ Model saved!")

    # 9. Plot training curves
    _plot_training(history, epochs)

    print(f"\n🎉 Training complete!")
    print(f"   Model: {MODEL_PATH}")
    print(f"   Plot:  {PLOT_PATH}")


def _plot_training(history, epochs):
    """Plot and save training accuracy/loss curves."""
    plt.style.use("ggplot")
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    N = epochs

    # Accuracy plot
    ax1.plot(np.arange(0, N), history.history["accuracy"], label="Train Accuracy",
             color="#4CAF50", linewidth=2)
    ax1.plot(np.arange(0, N), history.history["val_accuracy"], label="Val Accuracy",
             color="#2196F3", linewidth=2, linestyle="--")
    ax1.set_title("Training & Validation Accuracy", fontsize=14, fontweight="bold")
    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("Accuracy")
    ax1.legend(loc="lower right")
    ax1.grid(True, alpha=0.3)

    # Loss plot
    ax2.plot(np.arange(0, N), history.history["loss"], label="Train Loss",
             color="#F44336", linewidth=2)
    ax2.plot(np.arange(0, N), history.history["val_loss"], label="Val Loss",
             color="#FF9800", linewidth=2, linestyle="--")
    ax2.set_title("Training & Validation Loss", fontsize=14, fontweight="bold")
    ax2.set_xlabel("Epoch")
    ax2.set_ylabel("Loss")
    ax2.legend(loc="upper right")
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(PLOT_PATH, dpi=150, bbox_inches="tight")
    print(f"\n📊 Training plot saved to {PLOT_PATH}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train the face mask detector model")
    parser.add_argument("--epochs", type=int, default=20, help="Number of training epochs")
    parser.add_argument("--batch-size", type=int, default=32, help="Training batch size")
    args = parser.parse_args()

    train(epochs=args.epochs, batch_size=args.batch_size)
