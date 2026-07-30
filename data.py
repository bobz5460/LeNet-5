"""Datasets and the exact shared image preprocessing pipeline."""
from __future__ import annotations

from pathlib import Path
import string

from PIL import Image, ImageOps
from torch.utils.data import Dataset
from torchvision import datasets, transforms

IMAGE_SUFFIXES = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}
LETTERS = tuple("ABCDEFGHIJKLMNOPQRSTUVWXYZ")
EMNIST_BYCLASS_CLASSES = tuple(string.digits + string.ascii_uppercase + string.ascii_lowercase)


def preprocessing_metadata() -> dict:
    return {"source_color": "grayscale", "pixel_range_before_normalization": [0.0, 1.0], "operations": [
        {"op": "resize", "size": [28, 28], "interpolation": "bilinear"},
        {"op": "pad", "left": 2, "top": 2, "right": 2, "bottom": 2, "fill": 0},
        {"op": "to_tensor", "layout": "CHW", "dtype": "float32"},
        {"op": "normalize", "mean": [0.1307], "std": [0.3081], "formula": "(x - mean) / std"},
    ]}


def image_transform():
    return transforms.Compose([transforms.Resize((28, 28), interpolation=transforms.InterpolationMode.BILINEAR), transforms.Pad(2, fill=0), transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))])


def emnist_transform():
    """Correct EMNIST's stored orientation, then use the shared 32×32 pipeline."""
    return transforms.Compose([
        transforms.Lambda(lambda image: image.transpose(Image.Transpose.TRANSPOSE)),
        transforms.Resize((28, 28), interpolation=transforms.InterpolationMode.BILINEAR),
        transforms.Pad(2, fill=0),
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,)),
    ])


def nist_transform():
    """NIST scans are dark ink on a light page; invert to MNIST polarity."""
    return transforms.Compose([transforms.Lambda(ImageOps.invert), transforms.Resize((28, 28), interpolation=transforms.InterpolationMode.BILINEAR), transforms.Pad(2, fill=0), transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))])


def mnist_datasets(root: str | Path):
    transform = image_transform()
    return datasets.MNIST(str(root), train=True, download=True, transform=transform), datasets.MNIST(str(root), train=False, download=True, transform=transform)


def emnist_byclass_datasets(root: str | Path):
    """EMNIST ByClass: 814,255 handwritten digits and upper/lower-case letters."""
    transform = emnist_transform()
    return (
        datasets.EMNIST(str(root), split="byclass", train=True, download=True, transform=transform),
        datasets.EMNIST(str(root), split="byclass", train=False, download=True, transform=transform),
    )


def emnist_preprocessing_metadata() -> dict:
    metadata = preprocessing_metadata()
    metadata["operations"].insert(0, {"op": "transpose", "apply_to": "dataset", "reason": "correct EMNIST storage orientation"})
    return metadata


def _letter_from_path(path: Path, root: Path) -> str | None:
    for part in reversed(path.relative_to(root).parts[:-1]):
        if len(part) == 1 and part.upper() in LETTERS:
            return part.upper()
        try:
            decimal = int(part)
        except ValueError:
            decimal = -1
        if 65 <= decimal <= 90:  # explicit decimal ASCII directories, e.g. 65
            return chr(decimal)
        try:
            hexadecimal = int(part, 16)
        except ValueError:
            continue
        if 65 <= hexadecimal <= 90:  # SD19 by_class directories, e.g. 41
            return chr(hexadecimal)
    return None


class NIST19Letters(Dataset):
    """A-Z image subset from an unpacked NIST SD19 v1 directory tree."""
    def __init__(self, root: str | Path, transform=None) -> None:
        self.root = Path(root)
        if not self.root.is_dir():
            raise FileNotFoundError(f"NIST directory does not exist: {self.root}")
        self.transform, self.classes = transform or nist_transform(), LETTERS
        self.class_to_idx = {c: i for i, c in enumerate(self.classes)}
        self.samples = sorted((path, self.class_to_idx[label]) for path in self.root.rglob("*") if path.is_file() and path.suffix.lower() in IMAGE_SUFFIXES and (label := _letter_from_path(path, self.root)) is not None)
        if not self.samples:
            raise RuntimeError("No A-Z image files found. Point --nist-root at an unpacked SD19 by_class tree.")

    def __len__(self): return len(self.samples)

    def __getitem__(self, index):
        path, target = self.samples[index]
        with Image.open(path) as image:
            return self.transform(image.convert("L")), target
