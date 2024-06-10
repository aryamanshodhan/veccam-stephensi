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
    yolo = torch.hub.load("ultralytics/yolov5", "custom", path="models/yolo_best_0610.pt", force_reload=True)
    yolo.to("cpu")
    return yolo

def yolo_crop(image):
    """Apply YOLO object detection on an image and crop it around the detected mosquito.

    Args:
        image (PIL.Image.Image): Input image to crop.

    Returns:
        PIL.Image.Image: Cropped image centered around the detected mosquito.

    Raises:
        TypeError: If the input image is not a PIL image.

    Note:
        This function requires the `load_yolo` function to be defined and available in the current namespace.
        The YOLO model used by `load_yolo` must be able to detect mosquitoes in the input image.
    """

    yolo = load_yolo_model()
    results = yolo(image)
    try: 
       # crop the image
        xmin = int(results.xyxy[0].numpy()[0][0])
        ymin = int(results.xyxy[0].numpy()[0][1])
        xmax = int(results.xyxy[0].numpy()[0][2])
        ymax = int(results.xyxy[0].numpy()[0][3])
        conf0=results.xyxy[0].numpy()[0][4]
        class0=results.xyxy[0].numpy()[0][-1]
        im_crop = image.crop((xmin, ymin, xmax , ymax))
        print("Image cropped successfully!")
        print('Genus',class0)
        return class0,conf0,im_crop

    except:
       st.write("No mosquito detected")
    return image

# Main Code Block

st.write("""
         # VectorCAM Stephensi Detector 06/10/2024
         """
         )

device = torch.device("cpu")

with st.spinner("Models are loading..."):
    st.write("#### Models:")
    model = load_model()
    yolo = load_yolo_model()

file = st.file_uploader("Upload the image to be classified", type=["jpg", "png"])

transforms = transforms.Compose([
    transforms.Resize([300,300]),
    transforms.ToTensor(),
])

species_all = ["Anopheles Stephensi, Not Anopheles Stephensi"]

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

    ### YOLO CROP
    genus,conf,yolo_cropped_image = yolo_crop(image)
    st.write("### Shape of the cropped image is", yolo_cropped_image.size)