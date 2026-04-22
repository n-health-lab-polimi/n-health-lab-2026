# We use WebRTC protocol to stream video from the webcam
import streamlit as st
from ultralytics import YOLO
import av
from streamlit_webrtc import webrtc_streamer

# Load the YOLO model (this will be cached to avoid reloading on every frame)
@st.cache_resource
def load_model():
    return YOLO('yolo26n-pose.pt')


model = load_model()

# Callback function to process video frames (called once per frame)
# Encoding a frame as a PyAV VideoFrame
def pose_callback(frame: av.VideoFrame):
    # Decoding the frame to a Numpy array
    # YOLO expects Numpy array; OpenCV uses BGR format
    img = frame.to_ndarray(format="bgr24")

    # Run YOLO pose
    results = model(img, verbose=False)

    # Draw keypoints + skeleton lines
    annotated = results[0].plot()

    # We need to encode the processed image since WebRTC expects a VideoFrame
    return av.VideoFrame.from_ndarray(annotated, format="bgr24")


st.title("YOLO Pose Estimation")

# Function to embed a live WebRTC video stream in Streamlit
# Streamlit uses keys to remember widget state
webrtc_streamer(
    key="pose-stream",
    video_frame_callback=pose_callback,
    media_stream_constraints={
        "video": True,
        "audio": False,
    },
)