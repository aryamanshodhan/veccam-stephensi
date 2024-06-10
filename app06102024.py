import streamlit as st
from PIL import Image, ImageOps
import numpy as np
import torch
from torchvision import transforms
import torch.nn.functional as F
from util_functions import pad_image_to_square
from ultralytics import YOLO
import cv2

# import ssl
# ssl._create_default_https_context = ssl._create_unverified_context

if torch.cuda.is_available():
    device = torch.device('cuda')
else:
    device = torch.device('cpu')

@st.cache_resource
def load_model(): 
    """
    Load PyTorch model from disk and move it to the appropriate device.

    Returns:
        model (torch.nn.Module): The loaded PyTorch model.
    """
    model = torch.load("models/species_best_0610.pt", map_location=device)
    st.write("species_best_0610.pt loaded successfully!")
    return model

@st.cache_resource
def load_yolo_model():
    """
    Loads a custom YOLOv5 model from a local path and sends it to the CPU.

    Returns:
        yolo: A TorchHub model object representing the YOLOv5 model.
    """
    yolo = torch.hub.load('ultralytics/yolov5', 'custom', path='models/yolo_best_0610.pt', force_reload=True)
    yolo.to(device)
    return yolo
    # yolo = YOLO("models/YOLO_08_30.pt")
    # yolo.to('cpu')
    # st.write("yolo_best_0610.pt loaded successfully!")
    # return yolo

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

def preprocess_image(image):
    t = transforms.Compose([
        transforms.Resize([300,300]),
        transforms.ToTensor(),
    ])
    image = t(image)
    return image

def upload_predict(upload_image, model):
    """
    Perform image classification on a given image using a pre-trained model.

    Args:
    - upload_image: A PIL Image object representing the image to be classified.
    - model: A PyTorch model object that has been trained on image classification.

    Returns:
    - pred_class: An integer representing the predicted class label of the image.
    - probab_value: A float representing the predicted class probability of the image.
    """
    inputs = preprocess_image(upload_image)
    img_tensor = inputs.unsqueeze(0)

    # Run the model
    output = model(img_tensor)
    # st.write(output.detach().numpy())
    # # get softmax of output

    # #output = F.softmax(output, dim=1)

    # probab, pred = torch.max(output, 1)
    # print(output, pred, probab, probab.item())
    # pred_class = pred.item()
    # probab_value = probab.item()

    
    # return pred_class, probab_value

# Main Code Block

st.write("""
         # VectorCAM Stephensi Detector 06/10/2024
         """
         )

with st.spinner("Models are loading..."):
    st.write("#### Models:")
    model = load_model()

file = st.file_uploader("Upload the image to be classified", type=["jpg", "png"])

species_all = ["Not Anopheles Stephensi, Anopheles Stephensi"]

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

    ### PAD IMAGE
    image = pad_image_to_square(yolo_cropped_image)
    st.write("### Cropped and Padded Image")
    image_disp = image.copy()
    image_disp.thumbnail(max_size)
    st.image(image_disp, use_column_width= False)

    ### CLASSIFY IMAGE
    upload_predict(image, model)
    # st.write("### Species: ",species_all[label])
    # st.write(f"#### Confidence : {score*100:.2f} % ")