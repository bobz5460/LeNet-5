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
MNIST_NORMALIZATION = (0.1307, 0.3081)


def normalize_foreground(image: Image.Image, *, canvas_size: int = 28, foreground_size: int = 20,
                         threshold: int = 20) -> Image.Image:
    """Crop bright ink, preserve aspect ratio, and center it on a fixed canvas.

    MNIST glyphs occupy roughly a 20x20 box in a 28x28 image.  Matching that
    geometry is important for photos and canvas drawings, whose blank margins
    otherwise make a high MNIST test score misleading.
    """
    if not 0 < foreground_size <= canvas_size:
        raise ValueError("foreground_size must be in (0, canvas_size]")
    image = image.convert("L")
    mask = image.point(lambda value: 255 if value > threshold else 0)
    box = mask.getbbox()
    if box is None:
        return Image.new("L", (canvas_size, canvas_size), 0)
    glyph = image.crop(box)
    width, height = glyph.size
    scale = foreground_size / max(width, height)
    resized = glyph.resize((max(1, round(width * scale)), max(1, round(height * scale))), Image.Resampling.BILINEAR)
    result = Image.new("L", (canvas_size, canvas_size), 0)
    result.paste(resized, ((canvas_size - resized.width) // 2, (canvas_size - resized.height) // 2))
    return result


def preprocessing_metadata(mean: float = MNIST_NORMALIZATION[0], std: float = MNIST_NORMALIZATION[1], *,
                           foreground_normalization: bool = True, foreground_size: int = 20,
                           foreground_threshold: int = 20) -> dict:
    operations = [
        *([{"op": "normalize_foreground", "canvas_size": 28, "foreground_size": foreground_size,
            "threshold": foreground_threshold, "reason": "match MNIST glyph scale and centering"}] if foreground_normalization else []),
        {"op": "resize", "size": [28, 28], "interpolation": "bilinear"},
        {"op": "pad", "left": 2, "top": 2, "right": 2, "bottom": 2, "fill": 0},
        {"op": "to_tensor", "layout": "CHW", "dtype": "float32"},
        {"op": "normalize", "mean": [mean], "std": [std], "formula": "(x - mean) / std"},
    ]
    return {"source_color": "grayscale", "pixel_range_before_normalization": [0.0, 1.0], "operations": operations}


def _geometric_augmentation(enabled: bool, rotation_degrees: float, translate: float,
                            scale_min: float, scale_max: float, shear_degrees: float):
    """Return an opt-in handwriting augmentation, before resizing and padding."""
    if not enabled:
        return []
    if not 0 <= translate < 1:
        raise ValueError("augmentation translate must be in [0, 1)")
    if scale_min <= 0 or scale_max < scale_min:
        raise ValueError("augmentation scale range must be positive and ordered")
    return [transforms.RandomAffine(
        degrees=rotation_degrees, translate=(translate, translate),
        scale=(scale_min, scale_max), shear=shear_degrees, fill=0,
        interpolation=transforms.InterpolationMode.BILINEAR,
    )]


def image_transform(mean: float = MNIST_NORMALIZATION[0], std: float = MNIST_NORMALIZATION[1], *,
                    augment: bool = False, rotation_degrees: float = 0, translate: float = 0,
                    scale_min: float = 1, scale_max: float = 1, shear_degrees: float = 0,
                    foreground_normalization: bool = True, foreground_size: int = 20,
                    foreground_threshold: int = 20):
    return transforms.Compose([
        *([transforms.Lambda(lambda image: normalize_foreground(image, foreground_size=foreground_size, threshold=foreground_threshold))] if foreground_normalization else []),
        *_geometric_augmentation(augment, rotation_degrees, translate, scale_min, scale_max, shear_degrees),
        transforms.Resize((28, 28), interpolation=transforms.InterpolationMode.BILINEAR),
        transforms.Pad(2, fill=0), transforms.ToTensor(), transforms.Normalize((mean,), (std,)),
    ])


def emnist_transform(mean: float = MNIST_NORMALIZATION[0], std: float = MNIST_NORMALIZATION[1], *,
                     augment: bool = False, rotation_degrees: float = 0, translate: float = 0,
                     scale_min: float = 1, scale_max: float = 1, shear_degrees: float = 0,
                     foreground_normalization: bool = True, foreground_size: int = 20,
                     foreground_threshold: int = 20):
    """Correct EMNIST's stored orientation, then use the shared 32×32 pipeline."""
    return transforms.Compose([
        transforms.Lambda(lambda image: image.transpose(Image.Transpose.TRANSPOSE)),
        *([transforms.Lambda(lambda image: normalize_foreground(image, foreground_size=foreground_size, threshold=foreground_threshold))] if foreground_normalization else []),
        *_geometric_augmentation(augment, rotation_degrees, translate, scale_min, scale_max, shear_degrees),
        transforms.Resize((28, 28), interpolation=transforms.InterpolationMode.BILINEAR),
        transforms.Pad(2, fill=0),
        transforms.ToTensor(),
        transforms.Normalize((mean,), (std,)),
    ])


def nist_transform(mean: float = MNIST_NORMALIZATION[0], std: float = MNIST_NORMALIZATION[1], *,
                   augment: bool = False, rotation_degrees: float = 0, translate: float = 0,
                   scale_min: float = 1, scale_max: float = 1, shear_degrees: float = 0,
                   foreground_normalization: bool = True, foreground_size: int = 20,
                   foreground_threshold: int = 20):
    """NIST scans are dark ink on a light page; invert to MNIST polarity."""
    return transforms.Compose([transforms.Lambda(ImageOps.invert), *([transforms.Lambda(lambda image: normalize_foreground(image, foreground_size=foreground_size, threshold=foreground_threshold))] if foreground_normalization else []), *_geometric_augmentation(augment, rotation_degrees, translate, scale_min, scale_max, shear_degrees), transforms.Resize((28, 28), interpolation=transforms.InterpolationMode.BILINEAR), transforms.Pad(2, fill=0), transforms.ToTensor(), transforms.Normalize((mean,), (std,))])


def mnist_datasets(root: str | Path, **transform_options):
    return datasets.MNIST(str(root), train=True, download=True, transform=image_transform(**transform_options)), datasets.MNIST(str(root), train=False, download=True, transform=image_transform(**transform_options))


def emnist_byclass_datasets(root: str | Path, **transform_options):
    """EMNIST ByClass: 814,255 handwritten digits and upper/lower-case letters."""
    return (
        datasets.EMNIST(str(root), split="byclass", train=True, download=True, transform=emnist_transform(**transform_options)),
        datasets.EMNIST(str(root), split="byclass", train=False, download=True, transform=emnist_transform(**transform_options)),
    )


def emnist_preprocessing_metadata(mean: float = MNIST_NORMALIZATION[0], std: float = MNIST_NORMALIZATION[1], **options) -> dict:
    metadata = preprocessing_metadata(mean, std, **options)
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

    def raw_item(self, index):
        """Return an independent raw image so train/validation splits can differ in augmentation."""
        path, target = self.samples[index]
        with Image.open(path) as image:
            return image.convert("L").copy(), target

    def __getitem__(self, index):
        image, target = self.raw_item(index)
        return self.transform(image), target
