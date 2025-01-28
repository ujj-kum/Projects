import os
import streamlit as st
import torch
from torchvision import transforms
import numpy as np
from PIL import Image

# This set of transformation was used while training model
img_height = 180
img_width = 180

data_transforms = transforms.Compose([
    transforms.Resize(size=(img_height, img_width)),
    transforms.ToTensor(),
    transforms.Lambda(lambda x: x / 255.0)  # Normalize the image to [0, 1]
])


# Function to predict the class of input image
def predict_class(image, model):
    model.eval()
    image = data_transforms(image)
    # Add an extra dimension for batch dimension
    image = image.unsqueeze(0)
    
    prediction = model(image)

    return prediction


# Load model 
model_dir = os.path.join(os.path.dirname(__file__), 'best_model_weights.pth')
model_weights = torch.load(f=model_dir, map_location=torch.device('cpu'))

model_dir = os.path.join(os.path.dirname(__file__), 'best_model.pth')
model = torch.load(f=model_dir, map_location=torch.device('cpu'))

class_names = ['daisy', 'dandelion', 'rose', 'sunflower', 'tulip']


# Title of UI
st.title("Flower Classifier")

# Upload image
file = st.file_uploader(label="Upload an image of a flower", type=["jpg", "png"])

if file is None:
    st.text("Waiting for upload....")
else:
    slot = st.empty()
    slot.text("Running Inference....")

    # Open uploaded image
    test_image = Image.open(file)

    # Displaying uploaded image with caption and width
    st.image(image=test_image, caption="Input Image", width=400)

    # Generating predictions
    pred = predict_class(image=test_image, model=model)
    max_value, index = torch.max(input=pred, dim=1)
    result = class_names[index]

    # Generating output message
    output = "The image is a "+ result
    
    slot.text("Done")

    # Displaying output
    slot.success(output)



