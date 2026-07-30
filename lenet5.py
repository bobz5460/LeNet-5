"""Classic LeNet-5 and its portable, versioned inference description."""
from __future__ import annotations

from typing import Any

import torch
from torch import nn

ARCHITECTURE_VERSION = "lenet5-tanh-32x32-v1"
LARGE_ARCHITECTURE_VERSION = "lenet-large-tanh-32x32-v1"


class LeNet5(nn.Module):
    """LeNet-5 for a 32×32 single-channel input using Tanh activations."""

    def __init__(self, num_classes: int) -> None:
        super().__init__()
        self.num_classes = num_classes
        self.features = nn.Sequential(
            nn.Conv2d(1, 6, kernel_size=5), nn.Tanh(), nn.AvgPool2d(2, 2),
            nn.Conv2d(6, 16, kernel_size=5), nn.Tanh(), nn.AvgPool2d(2, 2),
            nn.Conv2d(16, 120, kernel_size=5), nn.Tanh(),
        )
        self.classifier = nn.Sequential(nn.Flatten(), nn.Linear(120, 84), nn.Tanh(), nn.Linear(84, num_classes))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.classifier(self.features(x))


class LeNetLarge(nn.Module):
    """A wider LeNet variant using the original Tanh and average-pooling methods."""

    def __init__(self, num_classes: int) -> None:
        super().__init__()
        self.num_classes = num_classes
        self.features = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=5), nn.Tanh(), nn.AvgPool2d(2, 2),
            nn.Conv2d(16, 48, kernel_size=5), nn.Tanh(), nn.AvgPool2d(2, 2),
            nn.Conv2d(48, 192, kernel_size=5), nn.Tanh(),
        )
        self.classifier = nn.Sequential(nn.Flatten(), nn.Linear(192, 256), nn.Tanh(), nn.Linear(256, num_classes))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.classifier(self.features(x))


def architecture_metadata(num_classes: int, variant: str = "lenet5") -> dict[str, Any]:
    """Every operation needed to reproduce inference outside PyTorch."""
    if variant == "lenet5":
        channels, hidden, architecture_id = (6, 16, 120), 84, ARCHITECTURE_VERSION
    elif variant == "large":
        channels, hidden, architecture_id = (16, 48, 192), 256, LARGE_ARCHITECTURE_VERSION
    else:
        raise ValueError(f"Unknown LeNet variant: {variant}")
    c1, c2, c3 = channels
    conv = lambda name, weight, bias, ic, oc: {"name": name, "op": "conv2d", "weight_key": weight, "bias_key": bias, "weight_layout": "OIHW", "in_channels": ic, "out_channels": oc, "kernel": [5, 5], "stride": [1, 1], "padding": [0, 0], "dilation": [1, 1], "groups": 1}
    pool = lambda name: {"name": name, "op": "avg_pool2d", "kernel": [2, 2], "stride": [2, 2], "padding": [0, 0], "ceil_mode": False, "count_include_pad": True}
    return {
        "id": architecture_id,
        "input": {"layout": "NCHW", "dtype": "float32", "shape": [1, 1, 32, 32]},
        "convolution_semantics": "cross_correlation",
        "layers": [
            conv("features.0", "features.0.weight", "features.0.bias", 1, c1), {"name": "features.1", "op": "tanh"}, pool("features.2"),
            conv("features.3", "features.3.weight", "features.3.bias", c1, c2), {"name": "features.4", "op": "tanh"}, pool("features.5"),
            conv("features.6", "features.6.weight", "features.6.bias", c2, c3), {"name": "features.7", "op": "tanh"},
            {"name": "classifier.0", "op": "flatten", "start_dim": 1, "end_dim": -1},
            {"name": "classifier.1", "op": "linear", "weight_key": "classifier.1.weight", "bias_key": "classifier.1.bias", "weight_layout": "OI", "in_features": c3, "out_features": hidden},
            {"name": "classifier.2", "op": "tanh"},
            {"name": "classifier.3", "op": "linear", "weight_key": "classifier.3.weight", "bias_key": "classifier.3.bias", "weight_layout": "OI", "in_features": hidden, "out_features": num_classes},
        ],
        "output": {"type": "logits", "shape": [1, num_classes], "postprocess": "softmax over class dimension for probabilities"},
    }


def model_from_architecture(architecture_id: str, num_classes: int) -> nn.Module:
    if architecture_id == ARCHITECTURE_VERSION:
        return LeNet5(num_classes)
    if architecture_id == LARGE_ARCHITECTURE_VERSION:
        return LeNetLarge(num_classes)
    raise ValueError(f"Unsupported architecture: {architecture_id}")
