import torch
import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from models.custom_cnn import CustomCNN
from models.transfer_models import build_model

def count_params(model):
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return total, trainable

def test_model(model, name):
    x = torch.randn(4, 3, 224, 224)  # dummy batch
    y = model(x)
    total, trainable = count_params(model)
    print(f"\n🧠 Model: {name}")
    print(f" - Output shape: {y.shape}")
    print(f" - Total params: {total:,}")
    print(f" - Trainable params: {trainable:,}")

if __name__ == "__main__":
    # Test Custom CNN
    custom_model = CustomCNN()
    test_model(custom_model, "Custom CNN")

    # Test Transfer Learning Models
    for model_name in ["resnet50", "mobilenet_v3", "efficientnet_b0"]:
        model = build_model(model_name, pretrained=False, freeze_backbone=True)
        test_model(model, model_name)

