"""Pillow-only implementation of an export manifest's preprocessing contract."""
from __future__ import annotations

import numpy as np
import torch
from PIL import Image, ImageOps


def preprocess(image: Image.Image, metadata: dict) -> tuple[torch.Tensor, Image.Image]:
    result = image.convert("L")
    for operation in metadata["operations"]:
        op = operation["op"]
        if op == "invert": result = ImageOps.invert(result)
        elif op == "resize": result = result.resize(tuple(operation["size"]), Image.Resampling.BILINEAR)
        elif op == "pad": result = ImageOps.expand(result, border=(operation["left"], operation["top"], operation["right"], operation["bottom"]), fill=operation["fill"])
        elif op == "to_tensor": continue
        elif op == "normalize":
            array = np.asarray(result, dtype=np.float32)[None] / 255.0
            return torch.from_numpy((array - operation["mean"][0]) / operation["std"][0]), result
        else: raise ValueError(f"Unsupported preprocessing operation: {op}")
    raise ValueError("Manifest preprocessing must end with normalize")
