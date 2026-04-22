from pathlib import Path
from collections import OrderedDict

import torch
import torch.nn as nn
from torchvision import models

from labels import LABELS

MODEL_PATH = Path("best_nih_densenet121.pth")


def build_model(num_classes: int):
    model = models.densenet121(weights=None)
    in_features = model.classifier.in_features
    model.classifier = nn.Linear(in_features, num_classes)
    return model


def clean_state_dict(state_dict):
    cleaned = OrderedDict()
    for key, value in state_dict.items():
        new_key = key

        if new_key.startswith("model."):
            new_key = new_key[len("model."):]

        cleaned[new_key] = value
    return cleaned


def load_trained_model(device: str = "cpu"):
    model = build_model(len(LABELS))

    checkpoint = torch.load(MODEL_PATH, map_location=device)

    if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
        state_dict = checkpoint["model_state_dict"]
    elif isinstance(checkpoint, dict):
        state_dict = checkpoint
    else:
        raise ValueError("Unsupported checkpoint format.")

    state_dict = clean_state_dict(state_dict)

    model.load_state_dict(state_dict, strict=True)
    model.to(device)
    model.eval()
    return model