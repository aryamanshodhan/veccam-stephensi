import streamlit as st
from PIL import Image, ImageOps
import numpy as np
import torch
from torchvision import transforms
import torch.nn.functional as F
from util_functions import pad_image_to_square
import cv2

@st.cache_resource
def load_model(): 
    """
    Load PyTorch model from disk and move it to the appropriate device.

    Returns:
        model (torch.nn.Module): The loaded PyTorch model.
    """
    model = torch.load("models/species_best_0610.pt", map_location="cpu")
    st.write("species_best_0610.pt loaded successfully!")
    return model

@st.cache_resource
def load_yolo_model():
    """
    Loads a custom YOLOv5 model from a local path and sends it to the CPU.

    Returns:
        yolo: A TorchHub model object representing the YOLOv5 model.
    """
    yolo = torch.hub.load("ultralytics/yolov5", "custom", path="model/yolo_best_0610.pt", force_reload=True)
    yolo.to("cpu")
    return yolo

st.write("""
         # VectorCAM Stephensi Detector 06/10/2024
         """
         )

device = torch.device("cpu")

with st.spinner("Models are loading..."):
    st.write("#### Models:")
    model = load_model()

file = st.file_uploader("Upload the image to be classified", type=["jpg", "png"])

# Main Code Block

if file is None:
    st.text("#### Please upload an image file!")
else:
    image = Image.open(file)

    # Open the image
    image_disp = image.copy()

    # Resize the image
    max_size = (400, 400)
    image_disp.thumbnail(max_size)
    st.write("### Uploaded Image")
    st.image(image_disp, use_column_width= False)