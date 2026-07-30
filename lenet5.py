"""Classic LeNet-5 and its portable, versioned inference description."""
from __future__ import annotations

from typing import Any

import torch
from torch import nn

ARCHITECTURE_VERSION = "lenet5-tanh-32x32-v1"


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


def architecture_metadata(num_classes: int) -> dict[str, Any]:
    """Every operation needed to reproduce inference outside PyTorch."""
    conv = lambda name, weight, bias, ic, oc: {"name": name, "op": "conv2d", "weight_key": weight, "bias_key": bias, "weight_layout": "OIHW", "in_channels": ic, "out_channels": oc, "kernel": [5, 5], "stride": [1, 1], "padding": [0, 0], "dilation": [1, 1], "groups": 1}
    pool = lambda name: {"name": name, "op": "avg_pool2d", "kernel": [2, 2], "stride": [2, 2], "padding": [0, 0], "ceil_mode": False, "count_include_pad": True}
    return {
        "id": ARCHITECTURE_VERSION,
        "input": {"layout": "NCHW", "dtype": "float32", "shape": [1, 1, 32, 32]},
        "convolution_semantics": "cross_correlation",
        "layers": [
            conv("features.0", "features.0.weight", "features.0.bias", 1, 6), {"name": "features.1", "op": "tanh"}, pool("features.2"),
            conv("features.3", "features.3.weight", "features.3.bias", 6, 16), {"name": "features.4", "op": "tanh"}, pool("features.5"),
            conv("features.6", "features.6.weight", "features.6.bias", 16, 120), {"name": "features.7", "op": "tanh"},
            {"name": "classifier.0", "op": "flatten", "start_dim": 1, "end_dim": -1},
            {"name": "classifier.1", "op": "linear", "weight_key": "classifier.1.weight", "bias_key": "classifier.1.bias", "weight_layout": "OI", "in_features": 120, "out_features": 84},
            {"name": "classifier.2", "op": "tanh"},
            {"name": "classifier.3", "op": "linear", "weight_key": "classifier.3.weight", "bias_key": "classifier.3.bias", "weight_layout": "OI", "in_features": 84, "out_features": num_classes},
        ],
        "output": {"type": "logits", "shape": [1, num_classes], "postprocess": "softmax over class dimension for probabilities"},
    }
