"""Configurable LeNet-family models and portable inference descriptions."""
from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

import torch
from torch import nn

ARCHITECTURE_VERSION = "lenet5-tanh-32x32-v1"
LARGE_ARCHITECTURE_VERSION = "lenet-large-tanh-32x32-v1"
MAX_ARCHITECTURE_VERSION = "lenet-max-tanh-32x32-v1"
CONFIGURABLE_ARCHITECTURE_VERSION = "lenet-configurable-32x32-v2"

ACTIVATIONS = ("tanh", "relu", "gelu", "sigmoid", "leaky_relu", "elu", "silu")
POOLINGS = ("avg", "max")
MODEL_PRESETS = {
    "lenet5": ((6, 16, 120), 84),
    "large": ((16, 48, 192), 256),
    "max": ((32, 96, 384), 512),
}


@dataclass(frozen=True)
class LeNetConfig:
    """All architecture choices supported by the portable export format."""

    preset: str = "lenet5"
    channels: tuple[int, int, int] = (6, 16, 120)
    hidden_dim: int = 84
    activation: str = "tanh"
    pooling: str = "avg"
    leaky_relu_slope: float = 0.01

    def __post_init__(self) -> None:
        if self.preset not in MODEL_PRESETS:
            raise ValueError(f"Unknown LeNet preset: {self.preset}")
        if len(self.channels) != 3 or any(channel < 1 for channel in self.channels):
            raise ValueError("channels must contain three positive values")
        if self.hidden_dim < 1:
            raise ValueError("hidden_dim must be positive")
        if self.activation not in ACTIVATIONS:
            raise ValueError(f"Unsupported activation: {self.activation}")
        if self.pooling not in POOLINGS:
            raise ValueError(f"Unsupported pooling: {self.pooling}")

    def export(self) -> dict[str, Any]:
        result = asdict(self)
        result["channels"] = list(self.channels)
        return result


def make_config(preset: str, *, activation: str = "tanh", pooling: str = "avg", channels: tuple[int, int, int] | None = None, hidden_dim: int | None = None, leaky_relu_slope: float = 0.01) -> LeNetConfig:
    default_channels, default_hidden = MODEL_PRESETS[preset]
    return LeNetConfig(preset, channels or default_channels, hidden_dim or default_hidden, activation, pooling, leaky_relu_slope)


def activation_layer(config: LeNetConfig) -> nn.Module:
    layers: dict[str, nn.Module] = {
        "tanh": nn.Tanh(), "relu": nn.ReLU(), "gelu": nn.GELU(), "sigmoid": nn.Sigmoid(),
        "leaky_relu": nn.LeakyReLU(config.leaky_relu_slope), "elu": nn.ELU(), "silu": nn.SiLU(),
    }
    return layers[config.activation]


def pooling_layer(config: LeNetConfig) -> nn.Module:
    return nn.AvgPool2d(2, 2) if config.pooling == "avg" else nn.MaxPool2d(2, 2)


class ConfigurableLeNet(nn.Module):
    """LeNet topology with selectable width, activation, and pooling method."""

    def __init__(self, num_classes: int, config: LeNetConfig) -> None:
        super().__init__()
        self.num_classes, self.config = num_classes, config
        c1, c2, c3 = config.channels
        self.features = nn.Sequential(
            nn.Conv2d(1, c1, kernel_size=5), activation_layer(config), pooling_layer(config),
            nn.Conv2d(c1, c2, kernel_size=5), activation_layer(config), pooling_layer(config),
            nn.Conv2d(c2, c3, kernel_size=5), activation_layer(config),
        )
        self.classifier = nn.Sequential(nn.Flatten(), nn.Linear(c3, config.hidden_dim), activation_layer(config), nn.Linear(config.hidden_dim, num_classes))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.classifier(self.features(x))


class LeNet5(ConfigurableLeNet):
    def __init__(self, num_classes: int, **options: Any) -> None:
        super().__init__(num_classes, make_config("lenet5", **options))


class LeNetLarge(ConfigurableLeNet):
    def __init__(self, num_classes: int, **options: Any) -> None:
        super().__init__(num_classes, make_config("large", **options))


class LeNetMax(ConfigurableLeNet):
    """Largest built-in LeNet preset (32→96→384 channels, 512 hidden units)."""

    def __init__(self, num_classes: int, **options: Any) -> None:
        super().__init__(num_classes, make_config("max", **options))


def _architecture_id(config: LeNetConfig) -> str:
    defaults = make_config(config.preset)
    if config == defaults:
        return {"lenet5": ARCHITECTURE_VERSION, "large": LARGE_ARCHITECTURE_VERSION, "max": MAX_ARCHITECTURE_VERSION}[config.preset]
    return CONFIGURABLE_ARCHITECTURE_VERSION


def architecture_metadata(num_classes: int, variant: str = "lenet5", config: LeNetConfig | None = None) -> dict[str, Any]:
    """Every selected operation and parameter needed for non-PyTorch inference."""
    config = config or make_config(variant)
    c1, c2, c3 = config.channels
    conv = lambda name, ic, oc: {"name": name, "op": "conv2d", "weight_key": f"{name}.weight", "bias_key": f"{name}.bias", "weight_layout": "OIHW", "in_channels": ic, "out_channels": oc, "kernel": [5, 5], "stride": [1, 1], "padding": [0, 0], "dilation": [1, 1], "groups": 1}
    activation = {"name": "", "op": config.activation}
    if config.activation == "leaky_relu": activation["negative_slope"] = config.leaky_relu_slope
    pool_op = "avg_pool2d" if config.pooling == "avg" else "max_pool2d"
    pool = lambda name: {"name": name, "op": pool_op, "kernel": [2, 2], "stride": [2, 2], "padding": [0, 0], "ceil_mode": False, **({"count_include_pad": True} if config.pooling == "avg" else {})}
    def act(name: str) -> dict[str, Any]: return {**activation, "name": name}
    return {
        "id": _architecture_id(config), "config": config.export(),
        "input": {"layout": "NCHW", "dtype": "float32", "shape": [1, 1, 32, 32]},
        "convolution_semantics": "cross_correlation",
        "layers": [
            conv("features.0", 1, c1), act("features.1"), pool("features.2"),
            conv("features.3", c1, c2), act("features.4"), pool("features.5"),
            conv("features.6", c2, c3), act("features.7"),
            {"name": "classifier.0", "op": "flatten", "start_dim": 1, "end_dim": -1},
            {"name": "classifier.1", "op": "linear", "weight_key": "classifier.1.weight", "bias_key": "classifier.1.bias", "weight_layout": "OI", "in_features": c3, "out_features": config.hidden_dim},
            act("classifier.2"),
            {"name": "classifier.3", "op": "linear", "weight_key": "classifier.3.weight", "bias_key": "classifier.3.bias", "weight_layout": "OI", "in_features": config.hidden_dim, "out_features": num_classes},
        ],
        "output": {"type": "logits", "shape": [1, num_classes], "postprocess": "softmax over class dimension for probabilities"},
    }


def model_from_architecture(architecture: str | dict[str, Any], num_classes: int) -> ConfigurableLeNet:
    if isinstance(architecture, dict):
        architecture_id = architecture.get("id")
        config_data = architecture.get("config")
        if config_data:
            config = LeNetConfig(**{**config_data, "channels": tuple(config_data["channels"])})
            return ConfigurableLeNet(num_classes, config)
    else:
        architecture_id = architecture
    presets = {ARCHITECTURE_VERSION: "lenet5", LARGE_ARCHITECTURE_VERSION: "large", MAX_ARCHITECTURE_VERSION: "max"}
    if architecture_id in presets:
        return {"lenet5": LeNet5, "large": LeNetLarge, "max": LeNetMax}[presets[architecture_id]](num_classes)
    raise ValueError(f"Unsupported architecture: {architecture_id}")
