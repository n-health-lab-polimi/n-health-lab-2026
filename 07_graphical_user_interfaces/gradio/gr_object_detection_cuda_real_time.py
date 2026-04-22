import torch
from ultralytics import YOLO
from fastrtc import Stream, VideoStreamHandler
import gradio as gr

# Load Model
device = "cuda" if torch.cuda.is_available() else "cpu"
model = YOLO("yolo11n.pt").to(device)

def detect_objects(frame):
    # Performance critical settings:
    # 1. imgsz=640. Try with 320 for massive speedup 
    # 2. stream=False (we only want the immediate result for this one frame)
    # results is a list of results, one for each image passed to the model
    results = model(frame, imgsz=640, verbose=False, device=device)
    # Using plot function to draw bounding box, labels and confidence scores
    return results[0].plot()

# VideoStreamHandler with skip_frames=True
# This prevents the "buildup" of frames that may cause a lag.
handler = VideoStreamHandler(
    detect_objects, 
    skip_frames=True  # Drops incoming frames while the GPU is busy
)

stream = Stream(
    handler=handler,
    modality="video",
    mode="send-receive",
)

# To remove "Use with API", "Built with Gradio", and "Settings
# at the bottom of gradio interface

css = """
footer {display: none !important;}
"""

if __name__ == "__main__":
    stream.ui.css = css
    stream.ui.launch()