#pip install fer==22.4.0

import streamlit as st
import cv2
import numpy as np
from fer import FER
import av
from streamlit_webrtc import webrtc_streamer, VideoProcessorBase


# =========================
# Emotion Recognition Class
# =========================
class FacialEmotionRecognition:
   
    def __init__(self):
        #self.fer = FER()
        self.fer = FER(mtcnn=False)

        self.label_colors = {
            'Neutral': (245, 114, 66),
            'Happy': (5, 245, 5),
            'Surprise': (18, 255, 215),
            'Sad': (245, 5, 49),
            'Angry': (82, 50, 168),
            'Disgust': (5, 245, 141),
            'Fear': (205, 245, 5)
        }

    def prediction_label(self, image_np):
        output_image = image_np.copy()

        # Using detect_emotions to return a list with a dictionary
        results_raw = self.fer.detect_emotions(image_np)
        #print("results_raw: ", results_raw)
        if not results_raw:
            return output_image, None

        results = results_raw[0]

        x, y, w, h = results['box']

        # Sorting emotions by the highest probability
        # results['emotions'] is a dictionary
        # with emotion as key and a number as a probability
        emotions = sorted(
            results['emotions'].items(),
            key=lambda x: x[1],
            reverse=True
        )

        class_pred = emotions[0][0].title()
        class_prob = "{:.2%}".format(emotions[0][1])

        color = self.label_colors.get(class_pred, (255, 255, 255))

        cv2.rectangle(
            output_image,
            (x, y),
            (x + w, y + h),
            color,
            2
        )

        cv2.putText(
            output_image,
            f"{class_pred}: {class_prob}",
            (x, y - 10),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            color,
            2,
            cv2.LINE_AA
        )

        return output_image, dict(emotions)


# =========================
# Streamlit UI
# =========================
st.title("🎥 Live Facial Emotion Recognition")

st.write("Webcam-based emotion detection using FER + Streamlit WebRTC")

# Creating an instance of FacialEmotionRecognition class
fer_app = FacialEmotionRecognition()


# =====================
# Video Processor Class
# =====================
class VideoProcessor(VideoProcessorBase):
    def __init__(self):
        self.fer_app = FacialEmotionRecognition()

    # Defining a function called every time a new webcam frame arrives
    def recv(self, frame: av.VideoFrame) -> av.VideoFrame:
        # Converting from WebRTC to OpenCV format
        img = frame.to_ndarray(format="bgr24")

        annotated, _ = self.fer_app.prediction_label(img)

        # Converting back to WebRTC format
        return av.VideoFrame.from_ndarray(annotated, format="bgr24")


# =========================
# WebRTC Stream
# =========================
webrtc_streamer(
    key="emotion",
    video_processor_factory=VideoProcessor,
)