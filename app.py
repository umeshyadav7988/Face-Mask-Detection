"""
Face Mask Detection - Flask Web Application
Live webcam face mask detection with a stunning web UI.

Usage:
    python3 app.py                    # Start on default port 5000
    python3 app.py --port 8080        # Custom port
"""

import os
import io
import base64
import argparse

import cv2
import numpy as np
from flask import Flask, render_template, request, jsonify, Response
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input

# ─── Configuration ──────────────────────────────────────────────────────────────
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "models", "mask_detector.h5")
FACE_PROTO = os.path.join(BASE_DIR, "face_detector", "deploy.prototxt")
FACE_MODEL = os.path.join(BASE_DIR, "face_detector", "res10_300x300_ssd_iter_140000.caffemodel")
HAAR_CASCADE = os.path.join(BASE_DIR, "haarcascade_frontalface_default.xml")

app = Flask(
    __name__,
    template_folder=os.path.join(BASE_DIR, "templates"),
    static_folder=os.path.join(BASE_DIR, "static"),
)

# Global models (loaded on startup)
face_detector = None
detector_type = None
mask_model = None


def load_models():
    """Load face detector and mask classifier models."""
    global face_detector, detector_type, mask_model

    print("📦 Loading models...")

    # Load mask detection model
    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError(
            f"Trained model not found at {MODEL_PATH}. Run `python3 train.py` first."
        )
    mask_model = load_model(MODEL_PATH)

    # Load face detector
    if os.path.exists(FACE_PROTO) and os.path.exists(FACE_MODEL):
        face_detector = cv2.dnn.readNet(FACE_PROTO, FACE_MODEL)
        detector_type = "dnn"
        print("   ✅ Face detector: OpenCV DNN (SSD)")
    elif os.path.exists(HAAR_CASCADE):
        face_detector = cv2.CascadeClassifier(HAAR_CASCADE)
        detector_type = "haar"
        print("   ✅ Face detector: Haar Cascade")
    else:
        from detect_mask_video import download_face_detector
        download_face_detector()
        face_detector = cv2.dnn.readNet(FACE_PROTO, FACE_MODEL)
        detector_type = "dnn"

    print("   ✅ Mask classifier: MobileNetV2")
    print("✅ All models loaded!")


def detect_and_annotate(image_bytes):
    """
    Detect faces in the image and classify mask/no-mask.
    Returns the annotated image (base64) and detection results.
    """
    nparr = np.frombuffer(image_bytes, np.uint8)
    frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

    if frame is None:
        return None, []

    (h, w) = frame.shape[:2]
    results = []

    if detector_type == "dnn":
        blob = cv2.dnn.blobFromImage(frame, 1.0, (300, 300), (104.0, 177.0, 123.0))
        face_detector.setInput(blob)
        detections = face_detector.forward()

        for i in range(detections.shape[2]):
            confidence = detections[0, 0, i, 2]
            if confidence < 0.5:
                continue

            box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
            (startX, startY, endX, endY) = box.astype("int")
            startX, startY = max(0, startX), max(0, startY)
            endX, endY = min(w, endX), min(h, endY)

            face = frame[startY:endY, startX:endX]
            if face.size == 0:
                continue

            face_rgb = cv2.cvtColor(face, cv2.COLOR_BGR2RGB)
            face_rgb = cv2.resize(face_rgb, (224, 224))
            face_rgb = img_to_array(face_rgb)
            face_rgb = preprocess_input(face_rgb)
            face_rgb = np.expand_dims(face_rgb, axis=0)

            pred = mask_model.predict(face_rgb, verbose=0)[0]
            mask_prob, no_mask_prob = pred

            if mask_prob > no_mask_prob:
                label = "Mask"
                color = (0, 200, 0)
                conf = mask_prob
            else:
                label = "No Mask"
                color = (0, 0, 255)
                conf = no_mask_prob

            results.append({
                "label": label,
                "confidence": float(conf * 100),
                "box": [int(startX), int(startY), int(endX), int(endY)],
            })

            cv2.rectangle(frame, (startX, startY), (endX, endY), color, 3)
            label_text = f"{label}: {conf * 100:.1f}%"
            label_size = cv2.getTextSize(label_text, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)[0]
            y_pos = max(startY - 12, label_size[1] + 12)
            cv2.rectangle(
                frame,
                (startX, y_pos - label_size[1] - 10),
                (startX + label_size[0] + 10, y_pos + 6),
                color, -1,
            )
            cv2.putText(
                frame, label_text,
                (startX + 5, y_pos),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2,
            )

    else:  # Haar Cascade
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        rects = face_detector.detectMultiScale(gray, 1.1, 5, minSize=(60, 60))

        for (x, y, fw, fh) in rects:
            face = frame[y:y+fh, x:x+fw]
            face_rgb = cv2.cvtColor(face, cv2.COLOR_BGR2RGB)
            face_rgb = cv2.resize(face_rgb, (224, 224))
            face_rgb = img_to_array(face_rgb)
            face_rgb = preprocess_input(face_rgb)
            face_rgb = np.expand_dims(face_rgb, axis=0)

            pred = mask_model.predict(face_rgb, verbose=0)[0]
            mask_prob, no_mask_prob = pred

            if mask_prob > no_mask_prob:
                label = "Mask"
                color = (0, 200, 0)
                conf = mask_prob
            else:
                label = "No Mask"
                color = (0, 0, 255)
                conf = no_mask_prob

            results.append({
                "label": label,
                "confidence": float(conf * 100),
                "box": [int(x), int(y), int(x + fw), int(y + fh)],
            })

            cv2.rectangle(frame, (x, y), (x + fw, y + fh), color, 3)
            label_text = f"{label}: {conf * 100:.1f}%"
            cv2.rectangle(frame, (x, y - 30), (x + len(label_text) * 14, y), color, -1)
            cv2.putText(
                frame, label_text, (x + 5, y - 8),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2,
            )

    # Encode result image to base64
    _, buffer = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 90])
    result_b64 = base64.b64encode(buffer).decode("utf-8")

    return result_b64, results


# ─── Routes ─────────────────────────────────────────────────────────────────────

@app.route("/")
def index():
    return render_template("index.html")


@app.route("/detect", methods=["POST"])
def detect():
    """Accept a webcam frame (base64 or file) and return detection results."""
    # Handle base64 frame from webcam
    if request.is_json:
        data = request.get_json()
        if "frame" not in data:
            return jsonify({"error": "No frame data"}), 400

        # Decode base64 frame
        frame_data = data["frame"]
        # Strip the data URL prefix if present
        if "," in frame_data:
            frame_data = frame_data.split(",")[1]

        image_bytes = base64.b64decode(frame_data)

    # Handle file upload
    elif "image" in request.files:
        file = request.files["image"]
        if file.filename == "":
            return jsonify({"error": "No file selected"}), 400
        image_bytes = file.read()

    else:
        return jsonify({"error": "No image data provided"}), 400

    result_image, detections = detect_and_annotate(image_bytes)

    if result_image is None:
        return jsonify({"error": "Could not process image"}), 400

    return jsonify({
        "image": result_image,
        "detections": detections,
        "total_faces": len(detections),
        "with_mask": sum(1 for d in detections if d["label"] == "Mask"),
        "without_mask": sum(1 for d in detections if d["label"] == "No Mask"),
    })


@app.route("/health")
def health():
    return jsonify({"status": "ok", "models_loaded": mask_model is not None})


# ─── Main ───────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Face Mask Detection Web App")
    parser.add_argument("--port", type=int, default=5000, help="Port to run on")
    parser.add_argument("--debug", action="store_true", help="Enable debug mode")
    args = parser.parse_args()

    load_models()

    print(f"\n🌐 Starting web server at http://localhost:{args.port}")
    print("   Open this URL in your browser to use the web demo")
    print("-" * 60)

    app.run(host="0.0.0.0", port=args.port, debug=args.debug)
