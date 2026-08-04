import streamlit as st
import cv2
import numpy as np
from tensorflow.keras.models import load_model
from streamlit_webrtc import webrtc_streamer, VideoProcessorBase
import av

MODEL_PATH = "emotion_model.h5"
IMG_SIZE = 48
EMOTIONS = ["Angry", "Disgust", "Fear", "Happy", "Neutral", "Sad", "Surprise"]
EMOTION_COLORS = {
    "Angry": (0, 0, 255), "Disgust": (0, 128, 0), "Fear": (128, 0, 128),
    "Happy": (0, 255, 255), "Sad": (255, 0, 0), "Surprise": (0, 165, 255),
    "Neutral": (200, 200, 200),
}

@st.cache_resource
def load_assets():
    model = load_model(MODEL_PATH)
    cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
    return model, cascade

model, face_cascade = load_assets()

class EmotionProcessor(VideoProcessorBase):
    def recv(self, frame):
        img = frame.to_ndarray(format="bgr24")
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(60, 60))

        for (x, y, w, h) in faces:
            face = gray[y:y+h, x:x+w]
            face = cv2.resize(face, (IMG_SIZE, IMG_SIZE)).astype("float32") / 255.0
            face = np.expand_dims(face, axis=(0, -1))
            preds = model.predict(face, verbose=0)[0]
            idx = int(np.argmax(preds))
            label, conf = EMOTIONS[idx], preds[idx] * 100
            color = EMOTION_COLORS.get(label, (255, 255, 255))
            cv2.rectangle(img, (x, y), (x+w, y+h), color, 2)
            cv2.putText(img, f"{label} ({conf:.1f}%)", (x, y-10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)

        return av.VideoFrame.from_ndarray(img, format="bgr24")

st.title("Real-Time Facial Emotion Detection")
webrtc_streamer(
    key="emotion-detect",
    video_processor_factory=EmotionProcessor,
    rtc_configuration={"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]},
)