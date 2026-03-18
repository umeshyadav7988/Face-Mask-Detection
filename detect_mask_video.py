"""
Face Mask Detection - Real-Time Video Detection
Uses OpenCV DNN face detector + trained Keras model for real-time mask detection.

Usage:
    python detect_mask_video.py                        # Webcam detection
    python detect_mask_video.py --confidence 0.6       # Custom confidence threshold
"""

import os
import argparse
import cv2
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input


# ─── Configuration ──────────────────────────────────────────────────────────────
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "models", "mask_detector.h5")

# OpenCV DNN Face Detector paths
FACE_PROTO = os.path.join(BASE_DIR, "face_detector", "deploy.prototxt")
FACE_MODEL = os.path.join(BASE_DIR, "face_detector", "res10_300x300_ssd_iter_140000.caffemodel")

# Haar Cascade fallback
HAAR_CASCADE = os.path.join(BASE_DIR, "haarcascade_frontalface_default.xml")

# Colors
COLOR_MASK = (0, 200, 0)       # Green for mask
COLOR_NO_MASK = (0, 0, 255)    # Red for no mask
COLOR_BG = (0, 0, 0)           # Black for label background


def download_face_detector():
    """Download the Caffe-based face detector model files."""
    import urllib.request

    face_dir = os.path.join(BASE_DIR, "face_detector")
    os.makedirs(face_dir, exist_ok=True)

    proto_url = (
        "https://raw.githubusercontent.com/opencv/opencv/master/samples/"
        "dnn/face_detector/deploy.prototxt"
    )
    model_url = (
        "https://github.com/opencv/opencv_3rdparty/raw/dnn_samples_face_detector_20170830/"
        "res10_300x300_ssd_iter_140000.caffemodel"
    )

    if not os.path.exists(FACE_PROTO):
        print("📥 Downloading face detector prototxt...")
        urllib.request.urlretrieve(proto_url, FACE_PROTO)

    if not os.path.exists(FACE_MODEL):
        print("📥 Downloading face detector model (10 MB)...")
        urllib.request.urlretrieve(model_url, FACE_MODEL)

    print("✅ Face detector files ready!")


def load_face_detector():
    """Load the face detection model (DNN or Haar Cascade fallback)."""
    if os.path.exists(FACE_PROTO) and os.path.exists(FACE_MODEL):
        print("🔍 Using OpenCV DNN face detector (SSD)")
        face_net = cv2.dnn.readNet(FACE_PROTO, FACE_MODEL)
        return face_net, "dnn"
    elif os.path.exists(HAAR_CASCADE):
        print("🔍 Using Haar Cascade face detector (fallback)")
        face_cascade = cv2.CascadeClassifier(HAAR_CASCADE)
        return face_cascade, "haar"
    else:
        print("⚠️  No face detector found. Downloading DNN model...")
        download_face_detector()
        face_net = cv2.dnn.readNet(FACE_PROTO, FACE_MODEL)
        return face_net, "dnn"


def detect_faces_dnn(frame, face_net, confidence_threshold=0.5):
    """Detect faces using OpenCV DNN."""
    (h, w) = frame.shape[:2]
    blob = cv2.dnn.blobFromImage(frame, 1.0, (300, 300), (104.0, 177.0, 123.0))
    face_net.setInput(blob)
    detections = face_net.forward()

    faces = []
    locs = []

    for i in range(detections.shape[2]):
        confidence = detections[0, 0, i, 2]
        if confidence < confidence_threshold:
            continue

        box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
        (startX, startY, endX, endY) = box.astype("int")

        # Ensure bounding box is within frame
        startX = max(0, startX)
        startY = max(0, startY)
        endX = min(w, endX)
        endY = min(h, endY)

        face = frame[startY:endY, startX:endX]
        if face.size == 0:
            continue

        face = cv2.cvtColor(face, cv2.COLOR_BGR2RGB)
        face = cv2.resize(face, (224, 224))
        face = img_to_array(face)
        face = preprocess_input(face)

        faces.append(face)
        locs.append((startX, startY, endX, endY))

    return faces, locs


def detect_faces_haar(frame, face_cascade):
    """Detect faces using Haar Cascade."""
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    rects = face_cascade.detectMultiScale(
        gray, scaleFactor=1.1, minNeighbors=5, minSize=(60, 60)
    )

    faces = []
    locs = []

    for (x, y, w, h) in rects:
        face = frame[y:y+h, x:x+w]
        face = cv2.cvtColor(face, cv2.COLOR_BGR2RGB)
        face = cv2.resize(face, (224, 224))
        face = img_to_array(face)
        face = preprocess_input(face)

        faces.append(face)
        locs.append((x, y, x + w, y + h))

    return faces, locs


def predict_mask(faces, mask_net):
    """Run mask prediction on detected faces."""
    if len(faces) == 0:
        return []

    faces = np.array(faces, dtype="float32")
    preds = mask_net.predict(faces, verbose=0)
    return preds


def draw_results(frame, locs, preds):
    """Draw bounding boxes and labels on the frame."""
    for (box, pred) in zip(locs, preds):
        (startX, startY, endX, endY) = box
        (mask_prob, no_mask_prob) = pred

        # Determine class and color
        if mask_prob > no_mask_prob:
            label = "Mask"
            color = COLOR_MASK
            confidence = mask_prob
        else:
            label = "No Mask"
            color = COLOR_NO_MASK
            confidence = no_mask_prob

        label_text = f"{label}: {confidence * 100:.1f}%"

        # Draw bounding box
        cv2.rectangle(frame, (startX, startY), (endX, endY), color, 2)

        # Draw label background
        label_size = cv2.getTextSize(label_text, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)[0]
        label_y = max(startY - 10, label_size[1] + 10)
        cv2.rectangle(
            frame,
            (startX, label_y - label_size[1] - 8),
            (startX + label_size[0] + 8, label_y + 4),
            color, -1,
        )
        cv2.putText(
            frame, label_text,
            (startX + 4, label_y - 2),
            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2,
        )

    return frame


def run_detection(confidence=0.5):
    """Run real-time face mask detection on webcam feed."""
    print("=" * 60)
    print("  🎭 Face Mask Detection - Real-Time Video")
    print("=" * 60)

    # Check if model exists
    if not os.path.exists(MODEL_PATH):
        print(f"\n❌ Trained model not found at {MODEL_PATH}")
        print("   Run `python train.py` first to train the model.")
        return

    # Load models
    print("\n📦 Loading models...")
    face_detector, detector_type = load_face_detector()
    mask_net = load_model(MODEL_PATH)
    print("✅ Models loaded!")

    # Start video capture
    print("\n📹 Starting webcam...")
    print("   Press 'q' to quit")
    print("-" * 60)

    cap = cv2.VideoCapture(0)

    if not cap.isOpened():
        print("❌ Could not access webcam!")
        print("   Make sure your webcam is connected and not in use.")
        return

    # Set camera resolution
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

    fps_counter = 0
    fps = 0
    import time
    prev_time = time.time()

    while True:
        ret, frame = cap.read()
        if not ret:
            print("❌ Failed to read frame from webcam")
            break

        # Detect faces
        if detector_type == "dnn":
            faces, locs = detect_faces_dnn(frame, face_detector, confidence)
        else:
            faces, locs = detect_faces_haar(frame, face_detector)

        # Predict masks
        if len(faces) > 0:
            preds = predict_mask(faces, mask_net)
            frame = draw_results(frame, locs, preds)

        # Calculate FPS
        fps_counter += 1
        current_time = time.time()
        if current_time - prev_time >= 1.0:
            fps = fps_counter
            fps_counter = 0
            prev_time = current_time

        # Draw FPS and info bar
        cv2.rectangle(frame, (0, 0), (250, 35), (0, 0, 0), -1)
        cv2.putText(
            frame, f"FPS: {fps} | Faces: {len(locs)}",
            (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 255, 255), 2,
        )

        # Show frame
        cv2.imshow("Face Mask Detection", frame)

        # Quit on 'q'
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()
    print("\n👋 Detection stopped!")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Real-time face mask detection")
    parser.add_argument(
        "--confidence", type=float, default=0.5,
        help="Minimum confidence for face detection (default: 0.5)",
    )
    args = parser.parse_args()

    run_detection(confidence=args.confidence)
