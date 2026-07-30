"""Write/read a self-describing LeNet-5 checkpoint and JSON manifest."""
from __future__ import annotations

import argparse, json
from pathlib import Path
from typing import Any
import numpy as np
import torch
from lenet5 import ARCHITECTURE_VERSION, LARGE_ARCHITECTURE_VERSION, LeNet5, LeNetLarge, architecture_metadata, model_from_architecture


def build_bundle(model: LeNet5 | LeNetLarge, dataset: str, classes: list[str], preprocessing: dict, training: dict | None = None) -> dict[str, Any]:
    variant = "large" if isinstance(model, LeNetLarge) else "lenet5"
    return {"format_version": 1, "architecture": architecture_metadata(len(classes), variant), "dataset": dataset, "classes": classes, "preprocessing": preprocessing, "training": training or {}, "state_dict": model.state_dict()}


def manifest(bundle: dict[str, Any], weights_file: str | None = None) -> dict[str, Any]:
    result = {k: v for k, v in bundle.items() if k != "state_dict"}
    if weights_file:
        result["weights"] = {"file": weights_file, "format": "npz", "tensors": {key: {"shape": list(value.shape), "dtype": "float32"} for key, value in bundle["state_dict"].items()}}
    return result


def save_bundle(bundle: dict[str, Any], path: str | Path):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True); torch.save(bundle, path)
    weights_path = path.with_suffix(".weights.npz")
    np.savez(weights_path, **{key: value.detach().cpu().numpy().astype(np.float32) for key, value in bundle["state_dict"].items()})
    json_path = path.with_suffix(".json"); json_path.write_text(json.dumps(manifest(bundle, weights_path.name), indent=2) + "\n", encoding="utf-8")
    return path, json_path


def load_model(path: str | Path, device="cpu"):
    bundle = torch.load(path, map_location=device, weights_only=False)
    architecture_id = bundle.get("architecture", {}).get("id")
    if bundle.get("format_version") != 1 or architecture_id not in {ARCHITECTURE_VERSION, LARGE_ARCHITECTURE_VERSION}: raise ValueError("Not a supported self-describing LeNet export")
    model = model_from_architecture(architecture_id, len(bundle["classes"])).to(device); model.load_state_dict(bundle["state_dict"]); model.eval()
    return model, bundle


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Regenerate and validate an export JSON manifest"); parser.add_argument("model", type=Path); args = parser.parse_args()
    _, bundle = load_model(args.model); output = args.model.with_suffix(".json"); weights = args.model.with_suffix(".weights.npz")
    if not weights.is_file(): raise FileNotFoundError(f"Portable weight file missing: {weights}")
    output.write_text(json.dumps(manifest(bundle, weights.name), indent=2) + "\n", encoding="utf-8"); print(f"Validated {args.model}; wrote {output}")
