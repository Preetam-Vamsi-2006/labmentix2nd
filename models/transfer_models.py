import torch.nn as nn
import torchvision.models as models

def build_model(model_name: str, pretrained=True, freeze_backbone=True, out_features=1):
    """
    Build a transfer learning model for binary classification.
    Supports: resnet50, mobilenet_v3, efficientnet_b0
    """
    model_name = model_name.lower()

    if model_name == "resnet50":
        model = models.resnet50(weights=models.ResNet50_Weights.DEFAULT if pretrained else None)
        in_features = model.fc.in_features
        model.fc = nn.Linear(in_features, out_features)
        if freeze_backbone:
            for param in list(model.parameters())[:-2]:
                param.requires_grad = False

    elif model_name == "mobilenet_v3":
        model = models.mobilenet_v3_small(weights=models.MobileNet_V3_Small_Weights.DEFAULT if pretrained else None)
        in_features = model.classifier[-1].in_features
        model.classifier[-1] = nn.Linear(in_features, out_features)
        if freeze_backbone:
            for param in model.features.parameters():
                param.requires_grad = False

    elif model_name == "efficientnet_b0":
        model = models.efficientnet_b0(weights=models.EfficientNet_B0_Weights.DEFAULT if pretrained else None)
        in_features = model.classifier[-1].in_features
        model.classifier[-1] = nn.Linear(in_features, out_features)
        if freeze_backbone:
            for param in model.features.parameters():
                param.requires_grad = False

    else:
        raise ValueError(f"❌ Unknown model name: {model_name}")

    return model

