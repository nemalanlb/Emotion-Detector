<<<<<<< HEAD
# Real-Time Facial Emotion Detection Using OpenCV and Deep Learning

Implements the system described in the project report: a CNN trained on
FER-2013 classifies emotions (Angry, Disgust, Fear, Happy, Sad, Surprise,
Neutral) from faces detected live via a webcam using OpenCV.

## Files

| File | Purpose |
|---|---|
| `preprocess.py` | Module 2 — loads `fer2013.csv`, resizes/normalizes images, splits train/val/test |
| `model.py` | CNN architecture (Conv2D → ReLU → MaxPool ×3 → Dense → Dropout → Softmax) |
| `train_model.py` | Module 3 — trains the CNN, plots accuracy/loss curves, saves `emotion_model.h5` |
| `realtime_detection.py` | Modules 4 & 5 — webcam capture, Haar Cascade face detection, live emotion prediction + bounding box display |
| `requirements.txt` | Python dependencies |

## Setup (Windows, VS Code)

1. **Create a virtual environment** (open a terminal in VS Code):
   ```
   python -m venv venv
   venv\Scripts\activate
   ```

2. **Install dependencies:**
   ```
   pip install -r requirements.txt
   ```

3. **Download the dataset.**
   Get `fer2013.csv` from Kaggle:
   https://www.kaggle.com/datasets/msambare/fer2013
   Place it in the same folder as these scripts.

## Train the model

```
python train_model.py
```

This trains for up to 50 epochs (with early stopping), evaluates on the
test split, saves the best model as `emotion_model.h5`, and writes a
`training_history.png` chart of accuracy/loss.

Training on CPU can take a while (FER-2013 has ~35,887 images) — a GPU
speeds this up significantly if available. You can lower `EPOCHS` in
`train_model.py` for a quicker run while testing the pipeline.

## Run real-time detection

Once `emotion_model.h5` exists:

```
python realtime_detection.py
```

This opens your webcam, detects faces, and overlays the predicted
emotion + confidence score on each face in real time. Press **ESC** to
quit.

## Notes / troubleshooting

- If the webcam window doesn't open, make sure no other application
  (Zoom, Teams, etc.) is using the camera, and that you granted camera
  permission to Python/VS Code in Windows settings.
- If accuracy is low, this is expected for a from-scratch CNN on
  FER-2013 (typical from-scratch accuracy is roughly 60–70%; the
  report's "Future Enhancements" section suggests transfer learning
  with MobileNetV2/EfficientNet/ResNet to improve this).
- Good, even lighting on your face improves detection accuracy, per
  the report's stated limitations.
  
---
title: Emotion Detector
emoji: 🎭
colorFrom: blue
colorTo: purple
sdk: streamlit
sdk_version: 1.60.0
app_file: app.py
pinned: false
---