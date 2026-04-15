"""
EfficientNet-V2-S classifier for binary knee OA classification.
"""

import torch
import torch.nn as nn
from torchvision.models import efficientnet_v2_s


class Classifier(nn.Module):
    """
    EfficientNet-V2-S with a single-output classifier head.

    Binary classification via BCEWithLogitsLoss:
      output > 0 (sigmoid > 0.5) -> KL234 (positive class)
      output <= 0 (sigmoid <= 0.5) -> KL01 (negative class)
    """

    def __init__(self, dropout=0.2):
        super().__init__()
        self.net = efficientnet_v2_s(weights=None)
        in_features = self.net.classifier[1].in_features
        self.net.classifier = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(in_features, 1),
        )

    def forward(self, x):
        return self.net(x).squeeze(-1)


def load_model(model_path, device):
    """Load a trained model checkpoint."""
    model = Classifier().to(device)
    ckpt = torch.load(model_path, map_location=device, weights_only=False)

    if isinstance(ckpt, dict):
        if "state_dict" in ckpt:
            model.load_state_dict(ckpt["state_dict"])
        elif "model" in ckpt:
            model.load_state_dict(ckpt["model"])
        else:
            model.load_state_dict(ckpt)
    else:
        model.load_state_dict(ckpt)

    model.eval()
    model.float()
    return model
