"""
app_ui.py
---------
Streamlit UI for Bird vs Drone classification.
✅ Simple, elegant interface
✅ Upload image → get prediction + confidence
✅ Works with trained MobileNetV3 model
"""

import streamlit as st
import torch
from torchvision import models, transforms
from PIL import Image
from pathlib import Path

# ---------------- CONFIG ----------------
st.set_page_config(page_title="🕊️ Bird vs Drone Classifier", layout="centered")
st.title("🕊️ Bird vs Drone Classifier")
st.caption("Upload an image to let the model identify if it’s a **Bird** or a **Drone**.")

MODEL_PATH = Path("artifacts/bird_vs_drone.pt")
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

model = load_model()

# ---------------- IMAGE TRANSFORMS ----------------
transform = transforms.Compose([
    transforms.Resize((160, 160)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])

# ---------------- MAIN UI ----------------
uploaded_file = st.file_uploader("📤 Upload an image", type=["jpg", "jpeg", "png"])

if uploaded_file:
    # Load and show uploaded image
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Uploaded Image", use_container_width=True)

    # Preprocess and predict
    img_tensor = transform(image).unsqueeze(0).to(DEVICE)

    with torch.no_grad():
        logits = model(img_tensor)
        prob = torch.sigmoid(logits).item()

    label = "Drone" if prob >= 0.5 else "Bird"
    confidence = prob * 100 if label == "Drone" else (100 - prob * 100)

    # Display prediction
    st.markdown("---")
    st.subheader(f"🧠 Prediction: **{label}**")
    st.metric(label="Confidence", value=f"{confidence:.2f}%")

    # Optional visual feedback
    if label == "Bird":
        st.success("🕊️ It's a Bird! Safe skies.")
    else:
        st.error("🚁 It's a Drone! Security alert!")

# ---------------- FOOTER ----------------
st.markdown("---")
st.caption("Developed for Capstone Project | Deep Learning • Computer Vision • Streamlit Deployment")

