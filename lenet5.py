"""Configurable LeNet-family models and portable inference descriptions."""
from __future__ import annotations

from dataclasses import asdict, dataclass
import math
from typing import Any

import torch
from torch import nn

ARCHITECTURE_VERSION = "lenet5-tanh-32x32-v1"
LARGE_ARCHITECTURE_VERSION = "lenet-large-tanh-32x32-v1"
MAX_ARCHITECTURE_VERSION = "lenet-max-tanh-32x32-v1"
CONFIGURABLE_ARCHITECTURE_VERSION = "lenet-configurable-32x32-v3"
REGULARIZED_ARCHITECTURE_VERSION = "lenet-configurable-32x32-v4"

ACTIVATIONS = ("tanh", "relu", "gelu", "sigmoid", "leaky_relu", "elu", "silu", "clamp")
GELU_APPROXIMATIONS = ("none", "tanh")
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
    gelu_approximate: str = "none"
    activation_clamp_min: float | None = None
    activation_clamp_max: float | None = None
    batch_norm: bool = False
    dropout: float = 0.0

    def __post_init__(self) -> None:
        if self.preset not in MODEL_PRESETS:
            raise ValueError(f"Unknown LeNet preset: {self.preset}")
        if len(self.channels) != 3 or any(channel < 1 for channel in self.channels):
            raise ValueError("channels must contain three positive values")
        if self.hidden_dim < 1:
            raise ValueError("hidden_dim must be positive")
        if self.activation not in ACTIVATIONS:
            raise ValueError(f"Unsupported activation: {self.activation}")
        if self.gelu_approximate not in GELU_APPROXIMATIONS:
            raise ValueError(f"Unsupported GELU approximation: {self.gelu_approximate}")
        if self.activation == "clamp" and self.activation_clamp_min is None and self.activation_clamp_max is None:
            raise ValueError("clamp activation requires --activation-clamp-min and/or --activation-clamp-max")
        if any(bound is not None and not math.isfinite(bound) for bound in (self.activation_clamp_min, self.activation_clamp_max)):
            raise ValueError("activation clamp bounds must be finite")
        if self.activation_clamp_min is not None and self.activation_clamp_max is not None and self.activation_clamp_min > self.activation_clamp_max:
            raise ValueError("activation clamp minimum must not exceed its maximum")
        if self.pooling not in POOLINGS:
            raise ValueError(f"Unsupported pooling: {self.pooling}")
        if not 0 <= self.dropout < 1:
            raise ValueError("dropout must be in [0, 1)")

    def export(self) -> dict[str, Any]:
        result = asdict(self)
        result["channels"] = list(self.channels)
        return result


def make_config(preset: str, *, activation: str = "tanh", pooling: str = "avg", channels: tuple[int, int, int] | None = None, hidden_dim: int | None = None, leaky_relu_slope: float = 0.01, gelu_approximate: str = "none", activation_clamp_min: float | None = None, activation_clamp_max: float | None = None, batch_norm: bool = False, dropout: float = 0.0) -> LeNetConfig:
    default_channels, default_hidden = MODEL_PRESETS[preset]
    return LeNetConfig(preset, channels or default_channels, hidden_dim or default_hidden, activation, pooling, leaky_relu_slope, gelu_approximate, activation_clamp_min, activation_clamp_max, batch_norm, dropout)


def activation_layer(config: LeNetConfig) -> nn.Module:
    layers: dict[str, nn.Module] = {
        "tanh": nn.Tanh(), "relu": nn.ReLU(), "gelu": nn.GELU(), "sigmoid": nn.Sigmoid(),
        "leaky_relu": nn.LeakyReLU(config.leaky_relu_slope), "elu": nn.ELU(), "silu": nn.SiLU(),
    }
    if config.activation == "clamp":
        return ActivationClamp(config.activation_clamp_min, config.activation_clamp_max)
    if config.activation == "gelu":
        return nn.GELU(approximate=config.gelu_approximate)
    return layers[config.activation]


class ActivationClamp(nn.Module):
    """Clamp activation values while accepting either one-sided bound."""

    def __init__(self, min_value: float | None, max_value: float | None) -> None:
        super().__init__()
        self.min_value, self.max_value = min_value, max_value

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.clamp(x, min=self.min_value, max=self.max_value)


def activation_layers(config: LeNetConfig) -> list[nn.Module]:
    layers = [activation_layer(config)]
    if config.activation != "clamp" and (config.activation_clamp_min is not None or config.activation_clamp_max is not None):
        layers.append(ActivationClamp(config.activation_clamp_min, config.activation_clamp_max))
    return layers


def pooling_layer(config: LeNetConfig) -> nn.Module:
    return nn.AvgPool2d(2, 2) if config.pooling == "avg" else nn.MaxPool2d(2, 2)


class ConfigurableLeNet(nn.Module):
    """LeNet topology with selectable width, activation, and pooling method."""

    def __init__(self, num_classes: int, config: LeNetConfig) -> None:
        super().__init__()
        self.num_classes, self.config = num_classes, config
        c1, c2, c3 = config.channels
        # With batch_norm/dropout disabled this is byte-for-byte the original preset topology.
        features: list[nn.Module] = []
        for in_channels, out_channels, pool in ((1, c1, True), (c1, c2, True), (c2, c3, False)):
            features.append(nn.Conv2d(in_channels, out_channels, kernel_size=5))
            if config.batch_norm:
                features.append(nn.BatchNorm2d(out_channels))
            features.extend(activation_layers(config))
            if pool:
                features.append(pooling_layer(config))
        classifier: list[nn.Module] = [nn.Flatten(), nn.Linear(c3, config.hidden_dim)]
        if config.batch_norm:
            classifier.append(nn.BatchNorm1d(config.hidden_dim))
        classifier.extend(activation_layers(config))
        if config.dropout:
            classifier.append(nn.Dropout(config.dropout))
        classifier.append(nn.Linear(config.hidden_dim, num_classes))
        self.features, self.classifier = nn.Sequential(*features), nn.Sequential(*classifier)

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
    return REGULARIZED_ARCHITECTURE_VERSION if config.batch_norm or config.dropout else CONFIGURABLE_ARCHITECTURE_VERSION


def architecture_metadata(num_classes: int, variant: str = "lenet5", config: LeNetConfig | None = None) -> dict[str, Any]:
    """Every selected operation and parameter needed for non-PyTorch inference."""
    config = config or make_config(variant)
    c1, c2, c3 = config.channels
    conv = lambda name, ic, oc: {"name": name, "op": "conv2d", "weight_key": f"{name}.weight", "bias_key": f"{name}.bias", "weight_layout": "OIHW", "in_channels": ic, "out_channels": oc, "kernel": [5, 5], "stride": [1, 1], "padding": [0, 0], "dilation": [1, 1], "groups": 1}
    batch_norm = lambda name, features: {"name": name, "op": "batch_norm2d" if name.startswith("features") else "batch_norm1d", "weight_key": f"{name}.weight", "bias_key": f"{name}.bias", "running_mean_key": f"{name}.running_mean", "running_var_key": f"{name}.running_var", "num_features": features, "eps": 1e-5, "momentum": 0.1}
    activation = {"name": "", "op": config.activation}
    if config.activation == "leaky_relu": activation["negative_slope"] = config.leaky_relu_slope
    if config.activation == "gelu": activation["approximate"] = config.gelu_approximate
    pool_op = "avg_pool2d" if config.pooling == "avg" else "max_pool2d"
    pool = lambda name: {"name": name, "op": pool_op, "kernel": [2, 2], "stride": [2, 2], "padding": [0, 0], "ceil_mode": False, **({"count_include_pad": True} if config.pooling == "avg" else {})}
    def act(name: str) -> dict[str, Any]: return {**activation, "name": name}
    def clamp(name: str) -> dict[str, Any]:
        return {"name": name, "op": "clamp", **({"min": config.activation_clamp_min} if config.activation_clamp_min is not None else {}), **({"max": config.activation_clamp_max} if config.activation_clamp_max is not None else {})}
    layers: list[dict[str, Any]] = []
    feature_index = 0
    for in_channels, out_channels, has_pool in ((1, c1, True), (c1, c2, True), (c2, c3, False)):
        name = f"features.{feature_index}"; layers.append(conv(name, in_channels, out_channels)); feature_index += 1
        if config.batch_norm:
            layers.append(batch_norm(f"features.{feature_index}", out_channels)); feature_index += 1
        layers.append(act(f"features.{feature_index}")); feature_index += 1
        if config.activation != "clamp" and (config.activation_clamp_min is not None or config.activation_clamp_max is not None):
            layers.append(clamp(f"features.{feature_index}")); feature_index += 1
        if has_pool:
            layers.append(pool(f"features.{feature_index}")); feature_index += 1
    classifier_index = 0
    layers.extend([
        {"name": f"classifier.{classifier_index}", "op": "flatten", "start_dim": 1, "end_dim": -1},
        {"name": f"classifier.{classifier_index + 1}", "op": "linear", "weight_key": f"classifier.{classifier_index + 1}.weight", "bias_key": f"classifier.{classifier_index + 1}.bias", "weight_layout": "OI", "in_features": c3, "out_features": config.hidden_dim},
    ])
    classifier_index += 2
    if config.batch_norm:
        layers.append(batch_norm(f"classifier.{classifier_index}", config.hidden_dim)); classifier_index += 1
    layers.append(act(f"classifier.{classifier_index}")); classifier_index += 1
    if config.activation != "clamp" and (config.activation_clamp_min is not None or config.activation_clamp_max is not None):
        layers.append(clamp(f"classifier.{classifier_index}")); classifier_index += 1
    if config.dropout:
        layers.append({"name": f"classifier.{classifier_index}", "op": "dropout", "p": config.dropout, "inference_behavior": "identity"}); classifier_index += 1
    layers.append({"name": f"classifier.{classifier_index}", "op": "linear", "weight_key": f"classifier.{classifier_index}.weight", "bias_key": f"classifier.{classifier_index}.bias", "weight_layout": "OI", "in_features": config.hidden_dim, "out_features": num_classes})
    return {
        "id": _architecture_id(config), "config": config.export(),
        "input": {"layout": "NCHW", "dtype": "float32", "shape": [1, 1, 32, 32]},
        "convolution_semantics": "cross_correlation",
        "layers": layers,
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
