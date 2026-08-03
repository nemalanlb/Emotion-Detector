"""
Module 4 & 5: Real-Time Detection + Result Display
------------------------------------------------------
Opens the webcam, detects faces with an OpenCV Haar Cascade, feeds each
detected face into the trained CNN, and draws a bounding box + predicted
emotion label (with confidence score) on the live video feed.

Usage:
    python realtime_detection.py

Press ESC to quit.
"""

import cv2
import numpy as np
from tensorflow.keras.models import load_model

MODEL_PATH = "emotion_model.h5"
IMG_SIZE = 48

# Order must match flow_from_directory's alphabetical class indices:
# angry, disgust, fear, happy, neutral, sad, surprise
EMOTIONS = ["Angry", "Disgust", "Fear", "Happy", "Neutral", "Sad", "Surprise"]

# Colors (BGR) per emotion for the bounding box, purely cosmetic
EMOTION_COLORS = {
    "Angry": (0, 0, 255),
    "Disgust": (0, 128, 0),
    "Fear": (128, 0, 128),
    "Happy": (0, 255, 255),
    "Sad": (255, 0, 0),
    "Surprise": (0, 165, 255),
    "Neutral": (200, 200, 200),
}


def load_face_detector():
    cascade_path = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
    face_cascade = cv2.CascadeClassifier(cascade_path)
    if face_cascade.empty():
        raise IOError(f"Could not load Haar Cascade from {cascade_path}")
    return face_cascade


def preprocess_face(gray_frame, x, y, w, h):
    """Crop, resize, and normalize a detected face for the CNN."""
    face = gray_frame[y:y + h, x:x + w]
    face = cv2.resize(face, (IMG_SIZE, IMG_SIZE))
    face = face.astype("float32") / 255.0
    face = np.expand_dims(face, axis=(0, -1))  # shape -> (1, 48, 48, 1)
    return face


def main():
    print("Loading trained model...")
    model = load_model(MODEL_PATH)

    face_cascade = load_face_detector()

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        raise IOError("Cannot open webcam. Check that it is connected and not in use.")

    print("Starting webcam feed. Press ESC to quit.")

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Failed to grab frame from webcam.")
            break

        frame = cv2.flip(frame, 1)  # mirror for a natural selfie-view
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        faces = face_cascade.detectMultiScale(
            gray, scaleFactor=1.1, minNeighbors=5, minSize=(60, 60)
        )

        for (x, y, w, h) in faces:
            face_input = preprocess_face(gray, x, y, w, h)

            predictions = model.predict(face_input, verbose=0)[0]
            emotion_idx = int(np.argmax(predictions))
            emotion_label = EMOTIONS[emotion_idx]
            confidence = float(predictions[emotion_idx]) * 100

            color = EMOTION_COLORS.get(emotion_label, (255, 255, 255))
            cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)

            text = f"{emotion_label} ({confidence:.1f}%)"
            cv2.putText(
                frame, text, (x, y - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2,
            )

        cv2.imshow("Real-Time Facial Emotion Detection", frame)

        # ESC key to exit
        if cv2.waitKey(1) & 0xFF == 27:
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
