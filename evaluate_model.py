"""
evaluate_model.py
-----------------
Evaluates the Bird vs Drone classifier.
✅ Confusion Matrix
✅ Classification Report
✅ Accuracy & Loss Graphs
"""

import torch
import torch.nn as nn
from torchvision import datasets, transforms, models
from sklearn.metrics import confusion_matrix, classification_report
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import json
from pathlib import Path

# ---------------- CONFIG ----------------
DATA_DIR = Path("classification_dataset")
TEST_DIR = DATA_DIR / "test"
MODEL_PATH = Path("artifacts/bird_vs_drone.pt")
SAVE_PATH = Path("artifacts")
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

IMG_SIZE = 160
BATCH_SIZE = 32

# ---------------- LOAD DATA ----------------
test_tf = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485,0.456,0.406],
                         std=[0.229,0.224,0.225])
])
test_ds = datasets.ImageFolder(TEST_DIR, transform=test_tf)
test_loader = torch.utils.data.DataLoader(test_ds, batch_size=BATCH_SIZE, shuffle=False)
class_names = test_ds.classes
print(f"📁 Found test classes: {class_names}")

# ---------------- LOAD MODEL ----------------
model = models.mobilenet_v3_small(weights=None)
in_features = model.classifier[-1].in_features
model.classifier[-1] = nn.Linear(in_features, 1)
model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
model.to(DEVICE)
model.eval()
print("✅ Model loaded successfully!")

# ---------------- EVALUATE ----------------
y_true, y_pred = [], []
with torch.no_grad():
    for x, y in test_loader:
        x = x.to(DEVICE)
        y = y.to(DEVICE).float().view(-1, 1)
        logits = model(x)
        preds = (torch.sigmoid(logits) >= 0.5).long()
        y_true += y.cpu().numpy().flatten().tolist()
        y_pred += preds.cpu().numpy().flatten().tolist()

# ---------------- REPORTS ----------------
print("\n📊 Classification Report:")
report_text = classification_report(y_true, y_pred, target_names=class_names, digits=4)
print(report_text)

# Save report to JSON
report_dict = classification_report(y_true, y_pred, target_names=class_names, output_dict=True)
with open(SAVE_PATH / "classification_report.json", "w") as f:
    json.dump(report_dict, f, indent=2)

# ---------------- CONFUSION MATRIX ----------------
cm = confusion_matrix(y_true, y_pred)
plt.figure(figsize=(5,4))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
            xticklabels=class_names, yticklabels=class_names)
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title("Confusion Matrix - Bird vs Drone")
plt.tight_layout()
plt.savefig(SAVE_PATH / "confusion_matrix.png")
plt.show()

# ---------------- PLOT ACCURACY & LOSS GRAPHS ----------------
log_file = SAVE_PATH / "training_log.json"
if log_file.exists():
    with open(log_file, "r") as f:
        logs = json.load(f)
        train_losses = logs.get("train_losses", [])
        val_accuracies = logs.get("val_accuracies", [])

        epochs = list(range(1, len(train_losses)+1))

        plt.figure(figsize=(10,4))

        # Accuracy plot
        plt.subplot(1, 2, 1)
        plt.plot(epochs, val_accuracies, marker='o', label="Validation Accuracy", color='blue')
        plt.title("Validation Accuracy per Epoch")
        plt.xlabel("Epoch")
        plt.ylabel("Accuracy")
        plt.grid(True)
        plt.legend()

        # Loss plot
        plt.subplot(1, 2, 2)
        plt.plot(epochs, train_losses, marker='o', label="Training Loss", color='red')
        plt.title("Training Loss per Epoch")
        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        plt.grid(True)
        plt.legend()

        plt.tight_layout()
        plt.savefig(SAVE_PATH / "accuracy_loss_graphs.png")
        plt.show()

        print("✅ Accuracy & Loss graphs saved to artifacts/accuracy_loss_graphs.png")
else:
    print("\n⚠️ No training log found — skipping accuracy/loss graphs.")

