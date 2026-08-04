import streamlit as st
import cv2
import numpy as np
from tensorflow.keras.models import load_model

MODEL_PATH = "emotion_model.h5"
IMG_SIZE = 48
EMOTIONS = ["Angry", "Disgust", "Fear", "Happy", "Neutral", "Sad", "Surprise"]

@st.cache_resource
def load_assets():
    model = load_model(MODEL_PATH)
    cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
    return model, cascade

model, face_cascade = load_assets()

st.title("Real-Time Facial Emotion Detection")
img_file = st.camera_input("Take a photo")

if img_file is not None:
    file_bytes = np.frombuffer(img_file.getvalue(), np.uint8)
    frame = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(60, 60))

    for (x, y, w, h) in faces:
        face = gray[y:y+h, x:x+w]
        face = cv2.resize(face, (IMG_SIZE, IMG_SIZE)).astype("float32") / 255.0
        face = np.expand_dims(face, axis=(0, -1))
        preds = model.predict(face, verbose=0)[0]
        idx = int(np.argmax(preds))
        label = f"{EMOTIONS[idx]} ({preds[idx]*100:.1f}%)"
        cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
        cv2.putText(frame, label, (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

    st.image(frame, channels="BGR")