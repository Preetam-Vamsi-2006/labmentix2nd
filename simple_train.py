"""
simple_train.py
---------------
Fast & clean Bird vs Drone classifier using MobileNetV3-Small.
✅ Fixed BCE loss shape mismatch (y.view(-1,1))
✅ Trains ~6 epochs (~10 min CPU)
✅ Saves best model for Streamlit deployment
"""

import torch
import torch.nn as nn
from torchvision import datasets, transforms, models
from torch.utils.data import DataLoader
from sklearn.metrics import accuracy_score
from tqdm import tqdm
from pathlib import Path
import json

# ---------------- CONFIG ----------------
DATA_DIR = Path("classification_dataset")
TRAIN_DIR = DATA_DIR / "train"
VAL_DIR   = DATA_DIR / "valid"
TEST_DIR  = DATA_DIR / "test"
SAVE_PATH = Path("artifacts")
SAVE_PATH.mkdir(exist_ok=True)

IMG_SIZE = 160
EPOCHS = 6
BATCH_SIZE = 32
LR = 0.0005
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"🖥️ Using device: {DEVICE}")

# ---------------- DATASET ----------------
train_tf = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(10),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485,0.456,0.406],
                         std=[0.229,0.224,0.225])
])
val_tf = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485,0.456,0.406],
                         std=[0.229,0.224,0.225])
])

train_ds = datasets.ImageFolder(TRAIN_DIR, transform=train_tf)
val_ds   = datasets.ImageFolder(VAL_DIR, transform=val_tf)
test_ds  = datasets.ImageFolder(TEST_DIR, transform=val_tf)

train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)
val_loader   = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False)
test_loader  = DataLoader(test_ds, batch_size=BATCH_SIZE, shuffle=False)

# ---------------- MODEL ----------------
print("\n🚀 Loading MobileNetV3-Small...")
model = models.mobilenet_v3_small(weights=models.MobileNet_V3_Small_Weights.DEFAULT)
for p in model.features.parameters():
    p.requires_grad = False
in_features = model.classifier[-1].in_features
model.classifier[-1] = nn.Linear(in_features, 1)  # Binary output
model = model.to(DEVICE)

criterion = nn.BCEWithLogitsLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=LR)

# ---------------- TRAINING ----------------
best_acc = 0
print("\n🏋️ Training started...\n")

for epoch in range(EPOCHS):
    model.train()
    total, correct = 0, 0
    for x, y in tqdm(train_loader, desc=f"Epoch {epoch+1}/{EPOCHS}", leave=False):
        x = x.to(DEVICE)
        y = y.to(DEVICE).float().view(-1, 1)  # ✅ fix label shape
        optimizer.zero_grad()
        logits = model(x)
        loss = criterion(logits, y)
        loss.backward()
        optimizer.step()
        preds = (torch.sigmoid(logits) >= 0.5).long()
        correct += (preds == y.long()).sum().item()
        total += y.size(0)
    train_acc = correct / total

    # ----- Validation -----
    model.eval()
    val_y_true, val_y_pred = [], []
    with torch.no_grad():
        for x, y in val_loader:
            x = x.to(DEVICE)
            y = y.to(DEVICE).float().view(-1, 1)
            preds = (torch.sigmoid(model(x)) >= 0.5).long()
            val_y_true += y.cpu().numpy().tolist()
            val_y_pred += preds.cpu().numpy().tolist()
    val_acc = accuracy_score(val_y_true, val_y_pred)
    print(f"Epoch {epoch+1}: TrainAcc={train_acc:.4f}, ValAcc={val_acc:.4f}")

    if val_acc > best_acc:
        best_acc = val_acc
        torch.save(model.state_dict(), SAVE_PATH / "bird_vs_drone.pt")

print("\n✅ Training complete! Best model saved to artifacts/bird_vs_drone.pt")

# ---------------- TESTING ----------------
print("\n🧪 Evaluating on test data...")
model.load_state_dict(torch.load(SAVE_PATH / "bird_vs_drone.pt", map_location=DEVICE))
model.eval()

y_true, y_pred = [], []
with torch.no_grad():
    for x, y in test_loader:
        x = x.to(DEVICE)
        y = y.to(DEVICE).float().view(-1, 1)
        preds = (torch.sigmoid(model(x)) >= 0.5).long()
        y_true += y.cpu().numpy().tolist()
        y_pred += preds.cpu().numpy().tolist()

test_acc = accuracy_score(y_true, y_pred)
print(f"📊 Test Accuracy: {test_acc:.4f}")

with open(SAVE_PATH / "results.json", "w") as f:
    json.dump({"test_accuracy": test_acc}, f, indent=2)

print("🎯 All done! Ready for Streamlit deployment.")
