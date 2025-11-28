"""
app_simple.py
--------------
Streamlit app for Bird vs Drone classification and visualization.
✅ Classifies uploaded image
✅ Displays Confusion Matrix & Accuracy/Loss Graphs
✅ Works with trained MobileNetV3 model
"""

import streamlit as st
import torch
from torchvision import models, transforms
from PIL import Image
from pathlib import Path
import os

# ---------------- CONFIG ----------------
st.set_page_config(page_title="🕊️ Bird vs Drone Classifier", layout="centered")
MODEL_PATH = Path("artifacts/bird_vs_drone.pt")
CONF_MATRIX_PATH = Path("artifacts/confusion_matrix.png")
ACC_LOSS_PATH = Path("artifacts/accuracy_loss_graphs.png")
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ---------------- LOAD MODEL ----------------
@st.cache_resource
def load_model():
    model = models.mobilenet_v3_small(weights=None)
    in_features = model.classifier[-1].in_features
    model.classifier[-1] = torch.nn.Linear(in_features, 1)
    model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
    model.to(DEVICE)
    model.eval()
    return model

st.title("🕊️ Bird vs Drone Classifier")
st.caption("Upload an image to classify it as a **Bird** or a **Drone**. "
           "Below you can also view model performance metrics.")

model = load_model()

# ---------------- TRANSFORM ----------------
transform = transforms.Compose([
    transforms.Resize((160, 160)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])

# ---------------- SIDEBAR ----------------
st.sidebar.header("📊 Model Evaluation")
st.sidebar.write("Click below to view evaluation metrics and graphs.")

if st.sidebar.button("Show Evaluation Results"):
    if CONF_MATRIX_PATH.exists():
        st.image(str(CONF_MATRIX_PATH), caption="Confusion Matrix", use_container_width=True)
    else:
        st.warning("⚠️ Confusion matrix not found. Run `evaluate_model.py` first.")
    if ACC_LOSS_PATH.exists():
        st.image(str(ACC_LOSS_PATH), caption="Accuracy & Loss Graphs", use_container_width=True)
    else:
        st.warning("⚠️ Accuracy/Loss graphs not found. Re-run evaluation after saving training logs.")

st.markdown("---")

# ---------------- IMAGE UPLOAD ----------------
uploaded_file = st.file_uploader("📤 Upload an image", type=["jpg", "jpeg", "png"])

if uploaded_file:
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Uploaded Image", use_container_width=True)

    img_tensor = transform(image).unsqueeze(0).to(DEVICE)

    with torch.no_grad():
        logits = model(img_tensor)
        prob = torch.sigmoid(logits).item()

    label = "Drone" if prob >= 0.5 else "Bird"
    confidence = prob * 100 if label == "Drone" else (100 - prob * 100)

    st.subheader(f"Prediction: **{label}**")
    st.metric(label="Confidence", value=f"{confidence:.2f}%")

    if label == "Bird":
        st.success("🕊️ It's a Bird! Safe skies!")
    else:
        st.error("🚁 It's a Drone! Alert the security team!")

# ---------------- FOOTER ----------------
st.markdown("---")
st.caption("Developed for Capstone Project | Computer Vision • Deep Learning • Streamlit Deployment")

