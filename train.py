"""Train and evaluate LeNet-5 on MNIST or Google's Quick, Draw! objects.

Example: python train.py --epochs 5 --output checkpoints/lenet_mnist.pt
"""

from __future__ import annotations

import argparse
import os
import random
import resource
import urllib.request
from pathlib import Path

import torch
import numpy as np
from torch import nn, optim
from torch.utils.data import DataLoader, Dataset
from PIL import Image
from torchvision import datasets, transforms

from lenet import LeNet5
from classes import QUICKDRAW_CLASSES


class ResilientMNIST(datasets.MNIST):
    """MNIST dataset with a working fallback mirror.

    Older torchvision releases still try yann.lecun.com, which no longer
    serves the dataset. The Google-hosted mirror uses the same filenames and
    format expected by torchvision.
    """

    mirrors = [
        "https://storage.googleapis.com/cvdf-datasets/mnist/",
        "https://ossci-datasets.s3.amazonaws.com/mnist/",
    ]


class QuickDrawDataset(Dataset):
    """Ten object classes from Quick, Draw!'s 28x28 NumPy dataset.

    Quick, Draw! has no official train/test split, so each class is split
    deterministically into 90% training and 10% testing examples.
    """

    base_url = "https://storage.googleapis.com/quickdraw_dataset/full/numpy_bitmap/"

    def __init__(self, root: str, train: bool, transform=None, limit_per_class: int = 10_000) -> None:
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)
        self.train = train
        self.transform = transform
        self.arrays: list[np.ndarray] = []
        self.examples: list[tuple[int, int]] = []
        for label, category in enumerate(QUICKDRAW_CLASSES):
            path = self.root / f"{category}.npy"
            try:
                array = np.load(path, mmap_mode="r")
                count = int(array.shape[0])
            except (OSError, ValueError):
                if path.exists():
                    path.unlink()
                url = self.base_url + path.name
                print(f"Downloading Quick, Draw! class {category} from {url}")
                try:
                    urllib.request.urlretrieve(url, path)
                except Exception as exc:
                    raise RuntimeError(f"Could not download {url}. Check your internet connection or download the .npy files manually into {self.root}/") from exc
                array = np.load(path, mmap_mode="r")
                count = int(array.shape[0])
            self.arrays.append(array)
            split = int(count * 0.9)
            start, end = (0, min(split, limit_per_class)) if train else (split, count)
            self.examples.extend((label, index) for index in range(start, end))

    def __len__(self) -> int:
        return len(self.examples)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, int]:
        label, row = self.examples[index]
        pixels = self.arrays[label][row]
        # Quick, Draw! numpy_bitmap files store each 28x28 image flattened
        # into 784 pixels.
        image = Image.fromarray(pixels.reshape(28, 28), mode="L")
        if self.transform is not None:
            image = self.transform(image)
        return image, label


def set_seed(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _limit_worker_threads(_worker_id: int) -> None:
    """Prevent each data-loader worker from spawning its own CPU thread pool."""
    torch.set_num_threads(1)


def make_loaders(dataset_name: str, data_dir: str, batch_size: int, workers: int, quickdraw_limit: int) -> tuple[DataLoader, DataLoader]:
    train_transform = transforms.Compose([
        # Rotate digits by up to 10 degrees and shift them by up to 50%.
        # This is intentionally applied only to the training split.
        transforms.RandomAffine(degrees=0, translate=(0, 0)),
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,)),
    ])
    test_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,)),
    ])
    if dataset_name == "quickdraw":
        train_set = QuickDrawDataset(Path(data_dir) / "quickdraw", train=True, transform=train_transform, limit_per_class=quickdraw_limit)
        test_set = QuickDrawDataset(Path(data_dir) / "quickdraw", train=False, transform=test_transform, limit_per_class=quickdraw_limit)
    else:
        train_set = ResilientMNIST(data_dir, train=True, download=True, transform=train_transform)
        test_set = ResilientMNIST(data_dir, train=False, download=True, transform=test_transform)
    loader_options = {
        "batch_size": batch_size,
        "num_workers": workers,
        "pin_memory": torch.cuda.is_available(),
        "persistent_workers": workers > 0,
        "worker_init_fn": _limit_worker_threads,
    }
    if workers > 0:
        loader_options["prefetch_factor"] = 4
    return (
        DataLoader(train_set, shuffle=True, **loader_options),
        DataLoader(test_set, shuffle=False, **loader_options),
    )


def train_one_epoch(model: nn.Module, loader: DataLoader, loss_fn: nn.Module, optimizer: optim.Optimizer, device: torch.device) -> tuple[float, float]:
    model.train()
    total_loss = total_correct = total_items = 0
    for images, labels in loader:
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)
        optimizer.zero_grad(set_to_none=True)
        logits = model(images)
        loss = loss_fn(logits, labels)
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * labels.size(0)
        total_correct += (logits.argmax(1) == labels).sum().item()
        total_items += labels.size(0)
    return total_loss / total_items, total_correct / total_items


@torch.no_grad()
def evaluate(model: nn.Module, loader: DataLoader, loss_fn: nn.Module, device: torch.device) -> tuple[float, float]:
    model.eval()
    total_loss = total_correct = total_items = 0
    for images, labels in loader:
        logits = model(images.to(device, non_blocking=True))
        labels = labels.to(device, non_blocking=True)
        total_loss += loss_fn(logits, labels).item() * labels.size(0)
        total_correct += (logits.argmax(1) == labels).sum().item()
        total_items += labels.size(0)
    return total_loss / total_items, total_correct / total_items


def main() -> None:
    parser = argparse.ArgumentParser(description="Train LeNet-5 on MNIST or Quick, Draw! objects")
    parser.add_argument("--dataset", choices=("mnist", "quickdraw"), default="mnist")
    parser.add_argument("--data-dir", default="data", help="where MNIST is downloaded")
    parser.add_argument("--output", default="checkpoints/lenet_mnist.pt")
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=2048, help="larger batches improve GPU utilization (default: 2048)")
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--workers", type=int, default=None, help="data-loader worker processes (auto-capped for system resources)")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--quickdraw-limit", type=int, default=10_000, help="training examples per class for Quick, Draw! (0 = all)")
    args = parser.parse_args()

    cpu_cores = os.cpu_count() or 1
    torch.set_num_threads(cpu_cores)
    torch.set_num_interop_threads(min(cpu_cores, 4))
    set_seed(args.seed)
    # Each worker needs several multiprocessing pipes/file descriptors. A
    # worker per core is unsafe on large servers even though model operators
    # can still use every CPU core.
    soft_fd_limit = resource.getrlimit(resource.RLIMIT_NOFILE)[0]
    requested_workers = min(cpu_cores, 16) if args.workers is None else max(0, args.workers)
    fd_safe_workers = max(0, (soft_fd_limit - 64) // 16)
    workers = min(requested_workers, fd_safe_workers, 16)
    if workers != requested_workers:
        print(f"Using {workers} data-loader workers (requested {requested_workers}; file-descriptor safe limit is {soft_fd_limit})")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type == "cuda":
        torch.backends.cudnn.benchmark = True
        print(f"CUDA enabled: {torch.cuda.get_device_name(0)}")
    quickdraw_limit = args.quickdraw_limit or 10**9
    train_loader, test_loader = make_loaders(args.dataset, args.data_dir, args.batch_size, workers, quickdraw_limit)
    model = LeNet5().to(device)
    loss_fn = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=2, gamma=0.5)

    print(f"Training on {device} ({len(train_loader.dataset):,} train / {len(test_loader.dataset):,} test images)")
    best_accuracy = 0.0
    class_names = QUICKDRAW_CLASSES if args.dataset == "quickdraw" else tuple(str(i) for i in range(10))
    for epoch in range(1, args.epochs + 1):
        train_loss, train_accuracy = train_one_epoch(model, train_loader, loss_fn, optimizer, device)
        test_loss, test_accuracy = evaluate(model, test_loader, loss_fn, device)
        scheduler.step()
        print(f"epoch {epoch:2d}/{args.epochs}: train loss {train_loss:.4f}, train acc {train_accuracy:.2%} | test loss {test_loss:.4f}, test acc {test_accuracy:.2%}")
        if test_accuracy >= best_accuracy:
            best_accuracy = test_accuracy
            output = Path(args.output)
            output.parent.mkdir(parents=True, exist_ok=True)
            torch.save({
                "format_version": 2,
                "model_state_dict": model.state_dict(),
                "model_name": "LeNet-5",
                "architecture": {
                    "input_shape": [1, 28, 28],
                    "layers": [
                        {"type": "Conv2d", "in_channels": 1, "out_channels": 6, "kernel_size": 5, "activation": "Tanh"},
                        {"type": "AvgPool2d", "kernel_size": 2, "stride": 2},
                        {"type": "Conv2d", "in_channels": 6, "out_channels": 16, "kernel_size": 5, "activation": "Tanh"},
                        {"type": "AvgPool2d", "kernel_size": 2, "stride": 2},
                        {"type": "Linear", "in_features": 256, "out_features": 120, "activation": "Tanh"},
                        {"type": "Linear", "in_features": 120, "out_features": 84, "activation": "Tanh"},
                        {"type": "Linear", "in_features": 84, "out_features": 10, "activation": "None"},
                    ],
                },
                "dataset": args.dataset,
                "classes": list(class_names),
                "normalization": {"mean": [0.1307], "std": [0.3081]},
                "epoch": epoch,
                "accuracy": test_accuracy,
                "loss": test_loss,
                "optimizer_state_dict": optimizer.state_dict(),
                "scheduler_state_dict": scheduler.state_dict(),
            }, output)
    print(f"Saved best checkpoint to {args.output} ({best_accuracy:.2%} test accuracy)")


if __name__ == "__main__":
    main()
