# 🎭 Face Mask Detection

**AI-powered face mask detection** using OpenCV and TensorFlow/Keras.

Detects faces in images or live webcam feed and classifies whether each person is wearing a mask or not — with bounding boxes and confidence scores.

![Python](https://img.shields.io/badge/Python-3.8+-blue?logo=python)
![TensorFlow](https://img.shields.io/badge/TensorFlow-2.10+-orange?logo=tensorflow)
![OpenCV](https://img.shields.io/badge/OpenCV-4.7+-green?logo=opencv)

---

## ✨ Features

| Feature | Description |
|---------|-------------|
| 🧠 **Deep Learning Model** | MobileNetV2 transfer learning for fast, accurate classification |
| 📹 **Real-Time Detection** | Live webcam face mask detection with FPS counter |
| 🌐 **Web Demo** | Beautiful Flask web app with drag-and-drop image upload |
| 📊 **Training Pipeline** | Data augmentation, train/test split, accuracy plots |
| 🎨 **Premium UI** | Dark glassmorphism theme with smooth animations |

---

## 🚀 Quick Start

### 1. Install Dependencies

```bash
cd FaceMaskDetection
pip install -r requirements.txt
```

### 2. Prepare Dataset

```bash
python prepare_dataset.py
```

This downloads a curated face mask dataset. If the download fails, it auto-generates a synthetic dataset for testing.

### 3. Train the Model

```bash
# Full training (20 epochs)
python train.py

# Quick test (2 epochs)
python train.py --epochs 2

# Custom batch size
python train.py --epochs 10 --batch-size 16
```

After training, you'll find:
- `models/mask_detector.h5` — Trained model
- `models/training_plot.png` — Accuracy & loss curves

### 4. Real-Time Webcam Detection

```bash
python detect_mask_video.py
```

- 🟩 **Green box** = Wearing mask
- 🟥 **Red box** = No mask
- Press **`q`** to quit

### 5. Web Demo

```bash
python app.py
```

Open **http://localhost:5000** in your browser, upload a photo, and see the results!

---

## 📁 Project Structure

```
FaceMaskDetection/
├── dataset/                  # Training images (auto-generated)
│   ├── with_mask/
│   └── without_mask/
├── models/                   # Trained models & plots
├── face_detector/            # OpenCV DNN face detector (auto-downloaded)
├── static/css/style.css      # Web UI styles
├── static/js/app.js          # Web UI logic
├── templates/index.html      # Web UI template
├── model.py                  # CNN model architecture
├── prepare_dataset.py        # Dataset preparation
├── train.py                  # Model training
├── detect_mask_video.py      # Real-time webcam detection
├── app.py                    # Flask web server
├── requirements.txt          # Python dependencies
└── README.md                 # This file
```

---

## 🧠 Model Architecture

```
MobileNetV2 (ImageNet, frozen)
    ↓
AveragePooling2D (7×7)
    ↓
Flatten
    ↓
Dense(128, ReLU)
    ↓
Dropout(0.5)
    ↓
Dense(2, Softmax) → [Mask, No Mask]
```

**Input**: 224×224 RGB images  
**Output**: 2-class probability (mask / no mask)

---

## 🛠️ Tech Stack

- **TensorFlow / Keras** — Deep learning framework
- **OpenCV** — Face detection (DNN SSD + Haar Cascade fallback)
- **MobileNetV2** — Lightweight CNN backbone
- **Flask** — Web application framework
- **scikit-learn** — Train/test splitting & metrics
- **Pillow** — Image processing
- **Matplotlib** — Training visualization

---

## 📋 How It Works

1. **Face Detection** — OpenCV's DNN-based SSD detector locates faces in each frame/image
2. **Preprocessing** — Each face is cropped, resized to 224×224, and normalized
3. **Classification** — The MobileNetV2 model predicts mask/no-mask probability
4. **Visualization** — Bounding boxes and confidence scores are drawn on the output

---

## 📜 License

This project is open source and available under the [MIT License](LICENSE).

---

Built with ❤️ using TensorFlow/Keras & OpenCV
