"""Pillow-only implementation of an export manifest's preprocessing contract."""
from __future__ import annotations

import numpy as np
import torch
from PIL import Image, ImageOps


def preprocess(image: Image.Image, metadata: dict) -> tuple[torch.Tensor, Image.Image]:
    result = image.convert("L")
    # Older EMNIST exports applied transpose then horizontal flip to raw files.
    # Since a browser drawing is already upright, its equivalent model input is
    # only the horizontal flip (the old model was trained on mirrored glyphs).
    legacy_emnist = any(
        operation.get("reason") == "correct EMNIST storage orientation" and "apply_to" not in operation
        for operation in metadata["operations"]
    )
    for operation in metadata["operations"]:
        # EMNIST files are stored sideways; browser drawings are already upright.
        if operation.get("apply_to") == "dataset":
            continue
        if legacy_emnist and operation["op"] == "transpose":
            continue
        op = operation["op"]
        if op == "invert": result = ImageOps.invert(result)
        elif op == "transpose": result = result.transpose(Image.Transpose.TRANSPOSE)
        elif op == "flip_horizontal": result = result.transpose(Image.Transpose.FLIP_LEFT_RIGHT)
        elif op == "resize": result = result.resize(tuple(operation["size"]), Image.Resampling.BILINEAR)
        elif op == "pad": result = ImageOps.expand(result, border=(operation["left"], operation["top"], operation["right"], operation["bottom"]), fill=operation["fill"])
        elif op == "to_tensor": continue
        elif op == "normalize":
            array = np.asarray(result, dtype=np.float32)[None] / 255.0
            return torch.from_numpy((array - operation["mean"][0]) / operation["std"][0]), result
        else: raise ValueError(f"Unsupported preprocessing operation: {op}")
    raise ValueError("Manifest preprocessing must end with normalize")
