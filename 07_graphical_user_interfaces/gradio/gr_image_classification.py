import torch
import gradio as gr
import requests
from PIL import Image
from torchvision import transforms

#PIL: Python Image Library to provide Python with image editing capabilities

model = torch.hub.load('pytorch/vision:v0.6.0', 'resnet18', pretrained=True).eval()

# Download human-readable labels for ImageNet.
response = requests.get("https://git.io/JJkYN")
labels = response.text.split("\n")

def predict(inp):
  # Convert image to a PyTorch tensor and add a batch dimension
  inp = transforms.ToTensor()(inp).unsqueeze(0)
  
  # Perform inference and compute class probabilities
  # No autograd needed for inference
  with torch.no_grad():
    prediction = torch.nn.functional.softmax(model(inp)[0], dim=0)
    confidences = {labels[i]: float(prediction[i]) for i in range(1000)}
  return confidences



gr.Interface(fn=predict,
             inputs=gr.Image(type="pil"),
             outputs=gr.Label(num_top_classes=3),
             examples=["lion.jpg", "cheetah.jpg"], css="footer {visibility: hidden}").launch()
