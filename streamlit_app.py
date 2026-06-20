import streamlit as st
import torch
import torchvision.transforms as transforms
from PIL import Image
import os

# Internal project dependencies 
from data_loader import get_dataset_path
from model import CIFAKEClassifier

# --- STREAMLIT UI CONFIGURATION ---
st.set_page_config(page_title="CIFAKE Image Detector", page_icon="🤖", layout="centered")

st.title("🤖 AI vs. Real Image Prediction")
st.subheader("Detect images generated via Stable Diffusion / Midjourney using Deep Learning")

# --- DATASET & MODEL CONFIGURATION ---
@st.cache_resource
def initialize_environment_and_model():
    """
    Downloads dataset metadata smoothly in the cloud background and 
    instantiates the PyTorch network structure.
    """
    # Triggers dataset download path check via kagglehub
    _ = get_dataset_path() 
    
    # Instantiate the network blueprint
    model = CIFAKEClassifier()
    model.eval() # Set model execution to Evaluation Mode
    return model

model = initialize_environment_and_model()

# --- PREPROCESSING UTILITY ---
def preprocess_image(image):
    """
    Transforms uploaded image to fit the 32x32 size and normalization values 
    expected by models trained on the CIFAKE/CIFAR-10 data ecosystem.
    """
    transform = transforms.Compose([
        transforms.Resize((32, 32)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.4914, 0.4822, 0.4465], std=[0.2023, 0.1994, 0.2010])
    ])
    # Convert image to RGB format, apply transformations, and add a batch dimension
    return transform(image.convert("RGB")).unsqueeze(0)

# --- USER FILE UPLOAD ---
uploaded_file = st.file_uploader("Upload an image file (JPG, JPEG, PNG)...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Display the uploaded image cleanly
    image = Image.open(uploaded_file)
    st.image(image, caption="Uploaded Image File", use_column_width=True)
    
    st.write("🔄 *Analyzing structural pixel distributions...*")
    
    # Transform raw image to model inputs
    input_tensor = preprocess_image(image)
    
    # Run prediction through network
    with torch.no_grad():
        outputs = model(input_tensor)
        probabilities = torch.softmax(outputs, dim=1)[0]
        prediction_idx = torch.argmax(probabilities).item()
    
    # Label array matching output vectors (Index 0: FAKE, Index 1: REAL)
    classes = ["AI-Generated (FAKE)", "Authentic Camera Photo (REAL)"]
    verdict = classes[prediction_idx]
    confidence = probabilities[prediction_idx].item() * 100
    
    # Present findings dynamically with Streamlit alert blocks
    st.markdown("---")
    if prediction_idx == 0:
        st.error(f"🚨 **Verdict:** The AI predicts this is **{verdict}**.")
    else:
        st.success(f"✅ **Verdict:** The AI predicts this is **{verdict}**.")
        
    
else:
    st.write("💡 *Please upload a graphic file to run the computer vision analysis.*")
