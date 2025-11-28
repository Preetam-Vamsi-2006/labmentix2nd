import torch
import torch.nn as nn

class CustomCNN(nn.Module):
    """
    Custom CNN for Bird vs Drone binary classification.
    """
    def __init__(self, dropout_p1: float = 0.4, dropout_p2: float = 0.3):
        super().__init__()
        self.features = nn.Sequential(
            # Input: 3x224x224 → Output: 32x112x112
            nn.Conv2d(3, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2),

            # 32x112x112 → 64x56x56
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2),

            # 64x56x56 → 128x28x28
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(2),

            # 128x28x28 → 256x14x14
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.MaxPool2d(2),

            # Global average pooling
            nn.AdaptiveAvgPool2d(1)
        )

        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Dropout(dropout_p1),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(dropout_p2),
            nn.Linear(128, 1)
        )

    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return x.squeeze(1)

