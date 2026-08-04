"""Train and export configurable LeNet-family handwriting classifiers."""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import random

import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset, Subset, WeightedRandomSampler, random_split

from data import (EMNIST_BYCLASS_CLASSES, LETTERS, MNIST_NORMALIZATION, NIST19Letters,
                  emnist_byclass_datasets, emnist_preprocessing_metadata, emnist_transform,
                  image_transform, mnist_datasets, nist_transform, preprocessing_metadata)
from export_model import build_bundle, save_bundle
from lenet5 import ACTIVATIONS, POOLINGS, MODEL_PRESETS, ConfigurableLeNet, make_config


def evaluate(model, loader, device, non_blocking, amp_enabled, num_classes):
    model.eval(); correct = total = 0; confusion = torch.zeros((num_classes, num_classes), dtype=torch.long)
    with torch.no_grad():
        for x, y in loader:
            x = x.to(device, non_blocking=non_blocking)
            with torch.autocast(device_type="cuda", dtype=torch.float16, enabled=amp_enabled): prediction = model(x).argmax(1).cpu()
            correct += (prediction == y).sum().item(); total += y.numel()
            confusion += torch.bincount(y * num_classes + prediction, minlength=num_classes ** 2).reshape(num_classes, num_classes)
    return correct / total, confusion


def cache_dataset(dataset, device, workers, stage_batch_size):
    print(f"Caching {len(dataset):,} samples in {'GPU VRAM' if str(device).startswith('cuda') else 'system RAM'}…")
    loader = DataLoader(dataset, stage_batch_size, num_workers=workers, pin_memory=str(device).startswith("cuda"),
                        persistent_workers=workers > 0, prefetch_factor=4 if workers > 0 else None)
    images, labels = [], []
    for x, y in loader:
        images.append(x.to(device, non_blocking=True)); labels.append(y.to(device, non_blocking=True))
    return torch.cat(images), torch.cat(labels)


def evaluate_cached(model, images, labels, batch_size, device, amp_enabled, num_classes):
    model.eval(); correct = 0; confusion = torch.zeros((num_classes, num_classes), dtype=torch.long)
    with torch.no_grad():
        for start in range(0, len(labels), batch_size):
            x = images[start:start + batch_size].to(device, non_blocking=str(device).startswith("cuda"))
            y = labels[start:start + batch_size].to(device, non_blocking=str(device).startswith("cuda"))
            with torch.autocast(device_type="cuda", dtype=torch.float16, enabled=amp_enabled): prediction = model(x).argmax(1)
            correct += (prediction == y).sum().item()
            confusion += torch.bincount((y * num_classes + prediction).cpu(), minlength=num_classes ** 2).reshape(num_classes, num_classes)
    return correct / len(labels), confusion


def augment_cached_batch(x: torch.Tensor, args, background: float) -> torch.Tensor:
    """Apply independently sampled affine transforms directly to a cached batch.

    Images are already normalized, so offset by the normalized background before
    grid sampling and restore it afterwards.  This preserves black padding rather
    than treating normalized zero as the background.
    """
    if not args.augment:
        return x
    count, _, height, width = x.shape
    device, dtype = x.device, x.dtype
    degrees = (torch.rand(count, device=device, dtype=dtype) * 2 - 1) * args.rotation_degrees
    radians = torch.deg2rad(degrees)
    scale = torch.empty(count, device=device, dtype=dtype).uniform_(args.scale_min, args.scale_max)
    shear = torch.deg2rad((torch.rand(count, device=device, dtype=dtype) * 2 - 1) * args.shear_degrees)
    # A forward transform is scale * rotation * horizontal shear. affine_grid
    # needs its inverse because it maps each output coordinate back to input.
    cosine, sine, tangent = torch.cos(radians), torch.sin(radians), torch.tan(shear)
    forward = torch.stack((
        torch.stack((scale * (cosine - sine * tangent), -scale * sine), dim=1),
        torch.stack((scale * (sine + cosine * tangent), scale * cosine), dim=1),
    ), dim=1)
    inverse = torch.linalg.inv(forward)
    translate = torch.stack((
        (torch.rand(count, device=device, dtype=dtype) * 2 - 1) * args.translate * 2,
        (torch.rand(count, device=device, dtype=dtype) * 2 - 1) * args.translate * 2,
    ), dim=1)
    theta = torch.cat((inverse, -(inverse @ translate.unsqueeze(2))), dim=2)
    grid = torch.nn.functional.affine_grid(theta, x.shape, align_corners=False)
    return torch.nn.functional.grid_sample(x - background, grid, mode="bilinear", padding_mode="zeros", align_corners=False) + background


class TransformedNISTSubset(Dataset):
    """A split with its own transform; random_split otherwise shares one dataset transform."""
    def __init__(self, dataset: NIST19Letters, indices, transform):
        self.dataset, self.indices, self.transform = dataset, list(indices), transform
        self.targets = [dataset.samples[index][1] for index in self.indices]

    def __len__(self): return len(self.indices)

    def __getitem__(self, index):
        image, target = self.dataset.raw_item(self.indices[index])
        return self.transform(image), target


def set_transform(dataset: Dataset, transform) -> None:
    """Set a transform through a Subset without changing its samples or split."""
    while isinstance(dataset, Subset):
        dataset = dataset.dataset
    dataset.transform = transform


def labels_for(dataset: Dataset) -> torch.Tensor:
    while isinstance(dataset, Subset):
        return labels_for(dataset.dataset)[torch.as_tensor(dataset.indices)]
    if hasattr(dataset, "targets"):
        return torch.as_tensor(dataset.targets, dtype=torch.long)
    if hasattr(dataset, "samples"):
        return torch.tensor([target for _, target in dataset.samples], dtype=torch.long)
    raise TypeError("dataset does not expose labels required for class balancing")


def normalization_stats(dataset: Dataset, workers: int, batch_size: int) -> tuple[float, float]:
    """Compute pixel-weighted statistics after resize/pad, using training data only."""
    print("Computing training-set normalization statistics…")
    loader = DataLoader(dataset, batch_size, num_workers=workers, pin_memory=False,
                        persistent_workers=workers > 0, prefetch_factor=2 if workers > 0 else None)
    pixel_sum = pixel_square_sum = 0.0; count = 0
    for images, _ in loader:
        pixel_sum += images.sum().item(); pixel_square_sum += images.square().sum().item(); count += images.numel()
    mean = pixel_sum / count
    return mean, max(pixel_square_sum / count - mean * mean, 1e-12) ** 0.5


def choose_cache_mode(requested: str, train_count: int, val_count: int, cuda: bool) -> str:
    """Use VRAM caching automatically only when there is ample free memory.

    Caching removes per-batch PIL/DataLoader work, which otherwise dominates tiny
    LeNet batches.  The conservative limit also leaves room for the temporary
    tensors used while concatenating the cache and for other GPU workloads.
    """
    if requested != "auto":
        return requested
    if not cuda:
        return "none"
    free_bytes, _ = torch.cuda.mem_get_info()
    image_bytes = (train_count + val_count) * 32 * 32 * torch.empty((), dtype=torch.float32).element_size()
    if image_bytes <= free_bytes // 4:
        print(f"Auto-enabling GPU dataset cache ({image_bytes / 2**20:.0f} MiB of image tensors; {free_bytes / 2**30:.1f} GiB free).")
        return "cuda"
    print(f"Not caching dataset on GPU: needs {image_bytes / 2**30:.1f} GiB of image tensors; only {free_bytes / 2**30:.1f} GiB is free.")
    return "none"


def augmentation_options(args) -> dict:
    return {"augment": args.augment, "rotation_degrees": args.rotation_degrees,
            "translate": args.translate, "scale_min": args.scale_min,
            "scale_max": args.scale_max, "shear_degrees": args.shear_degrees}


def foreground_options(args) -> dict:
    return {"foreground_normalization": args.foreground_normalization,
            "foreground_size": args.foreground_size, "foreground_threshold": args.foreground_threshold}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("dataset", choices=("mnist", "nist19", "emnist-byclass")); parser.add_argument("--data-root", default="data")
    parser.add_argument("--model", choices=tuple(MODEL_PRESETS), default=None, help="Network preset; defaults to a regularized large model")
    parser.add_argument("--activation", choices=ACTIVATIONS, default=None); parser.add_argument("--pooling", choices=POOLINGS, default=None)
    parser.add_argument("--channels"); parser.add_argument("--hidden-dim", type=int); parser.add_argument("--leaky-relu-slope", type=float, default=0.01)
    parser.add_argument("--gelu-approximate", choices=("none", "tanh"), default="none", help="GELU implementation; tanh is a faster approximation")
    parser.add_argument("--activation-clamp-min", type=float, help="Optional lower bound applied after each activation")
    parser.add_argument("--activation-clamp-max", type=float, help="Optional upper bound applied after each activation")
    parser.add_argument("--batch-norm", action=argparse.BooleanOptionalAction, default=None, help="Insert BatchNorm after conv/hidden layers")
    parser.add_argument("--dropout", type=float, default=None, help="Classifier dropout probability (0 preserves LeNet)")
    parser.add_argument("--nist-root", type=Path); parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--output-file", type=Path, help="Checkpoint filename within --output-dir (defaults to the model and dataset name)")
    parser.add_argument("--epochs", type=int, default=10); parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--optimizer", choices=("adam", "adamw"), default="adamw"); parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4); parser.add_argument("--label-smoothing", type=float, default=0.05)
    parser.add_argument("--scheduler", choices=("none", "cosine", "plateau"), default="cosine"); parser.add_argument("--min-learning-rate", type=float, default=1e-5)
    parser.add_argument("--plateau-patience", type=int, default=5); parser.add_argument("--plateau-factor", type=float, default=0.5)
    parser.add_argument("--class-balancing", choices=("none", "loss", "sampler"), default=None); parser.add_argument("--class-weight-power", type=float, default=1.0)
    parser.add_argument("--normalization", choices=("mnist", "dataset"), default="dataset", help="MNIST constants or statistics computed from this training split")
    parser.add_argument("--augment", action=argparse.BooleanOptionalAction, default=True); parser.add_argument("--rotation-degrees", type=float, default=12)
    parser.add_argument("--translate", type=float, default=0.1); parser.add_argument("--scale-min", type=float, default=0.9); parser.add_argument("--scale-max", type=float, default=1.1); parser.add_argument("--shear-degrees", type=float, default=10)
    parser.add_argument("--foreground-normalization", action=argparse.BooleanOptionalAction, default=True, help="crop/scale/recenter ink to MNIST-like geometry")
    parser.add_argument("--foreground-size", type=int, default=20); parser.add_argument("--foreground-threshold", type=int, default=20)
    parser.add_argument("--report-confusion-matrix", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--val-fraction", type=float, default=0.1); parser.add_argument("--seed", type=int, default=42); parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--workers", type=int, default=min(4, os.cpu_count() or 1)); parser.add_argument("--prefetch-factor", type=int, default=4)
    parser.add_argument("--amp", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--compile", action=argparse.BooleanOptionalAction, default=None, help="compile automatically on CUDA to reduce small-batch overhead")
    parser.add_argument("--cpu-threads", type=int, default=0)
    parser.add_argument("--cache-dataset", choices=("auto", "none", "ram", "cuda"), default="auto", help="cache preprocessed data; auto uses GPU VRAM only when there is ample free space")
    parser.add_argument("--cache-batch-size", type=int, default=4096, help="samples per staging transfer while caching (independent of --batch-size)")
    args = parser.parse_args(); torch.manual_seed(args.seed); random.seed(args.seed)
    if not 0 <= args.label_smoothing < 1: parser.error("--label-smoothing must be in [0, 1)")
    if args.weight_decay < 0 or args.class_weight_power < 0: parser.error("weight decay and class-weight power must be non-negative")
    if args.cache_batch_size < 1: parser.error("--cache-batch-size must be positive")
    if not 0 < args.foreground_size <= 28 or not 0 <= args.foreground_threshold < 256: parser.error("--foreground-size must be in [1, 28] and --foreground-threshold in [0, 255]")
    if args.cpu_threads > 0: torch.set_num_threads(args.cpu_threads)
    cuda = args.device.startswith("cuda")
    if cuda and not torch.cuda.is_available(): parser.error("--device cuda was requested, but CUDA is unavailable to PyTorch")
    if cuda: torch.backends.cudnn.benchmark = True; torch.set_float32_matmul_precision("high")
    elif args.cache_dataset == "cuda": parser.error("--cache-dataset cuda requires --device cuda")
    compile_model = cuda if args.compile is None else args.compile
    # Initially omit normalization so the optional statistics pass sees [0, 1] pixels.
    raw_options = {"mean": 0.0, "std": 1.0, **foreground_options(args)}
    if args.dataset == "mnist":
        train_set, val_set = mnist_datasets(args.data_root, **raw_options); classes = [str(i) for i in range(10)]; transform_factory = image_transform
    elif args.dataset == "emnist-byclass":
        train_set, val_set = emnist_byclass_datasets(args.data_root, **raw_options); classes = list(EMNIST_BYCLASS_CLASSES); transform_factory = emnist_transform
    else:
        if args.nist_root is None: parser.error("nist19 requires --nist-root")
        full = NIST19Letters(args.nist_root); n_val = max(1, round(len(full) * args.val_fraction)); train_split, val_split = random_split(full, [len(full) - n_val, n_val], generator=torch.Generator().manual_seed(args.seed)); train_set = TransformedNISTSubset(full, train_split.indices, nist_transform(**raw_options)); val_set = TransformedNISTSubset(full, val_split.indices, nist_transform(**raw_options)); classes = list(LETTERS); transform_factory = nist_transform
    mean, std = MNIST_NORMALIZATION if args.normalization == "mnist" else normalization_stats(train_set, args.workers, min(args.batch_size, 2048))
    args.cache_dataset = choose_cache_mode(args.cache_dataset, len(train_set), len(val_set), cuda)
    cached = args.cache_dataset != "none"
    common_transform = {"mean": mean, "std": std, **foreground_options(args)}
    # Cached images must be deterministic.  Random affine augmentation is then
    # sampled on the model device below, once for every training batch.
    cache_transform_options = {**augmentation_options(args), "augment": False} if cached else augmentation_options(args)
    set_transform(train_set, transform_factory(**common_transform, **cache_transform_options))
    set_transform(val_set, transform_factory(**common_transform))
    prep = emnist_preprocessing_metadata(mean, std, **foreground_options(args)) if args.dataset == "emnist-byclass" else preprocessing_metadata(mean, std, **foreground_options(args))
    if args.dataset == "nist19": prep["operations"].insert(0, {"op": "invert", "reason": "NIST dark-ink scan to MNIST-style bright foreground"})
    balance = args.class_balancing if args.class_balancing is not None else ("loss" if args.dataset == "emnist-byclass" else "none")
    train_labels_cpu = labels_for(train_set); counts = torch.bincount(train_labels_cpu, minlength=len(classes)).float()
    class_weights = (counts.sum() / (len(classes) * counts)).pow(args.class_weight_power) if balance != "none" else None
    sample_weights = class_weights[train_labels_cpu] if balance == "sampler" else None
    if cached:
        cache_device = args.device if args.cache_dataset == "cuda" else "cpu"
        cache_workers = min(args.workers, 8)
        train_images, train_labels = cache_dataset(train_set, cache_device, cache_workers, args.cache_batch_size)
        val_images, val_labels = cache_dataset(val_set, cache_device, cache_workers, args.cache_batch_size)
    else:
        loader_args = {"num_workers": args.workers, "pin_memory": cuda}
        if args.workers > 0: loader_args.update({"persistent_workers": True, "prefetch_factor": args.prefetch_factor})
        sampler = WeightedRandomSampler(sample_weights, len(sample_weights), replacement=True, generator=torch.Generator().manual_seed(args.seed)) if sample_weights is not None else None
        train_loader = DataLoader(train_set, args.batch_size, shuffle=sampler is None, sampler=sampler, **loader_args); val_loader = DataLoader(val_set, args.batch_size, **loader_args)
    model_name = args.model or "large"
    modern_default = model_name != "lenet5"
    activation = args.activation or ("gelu" if modern_default else "tanh")
    pooling = args.pooling or ("max" if modern_default else "avg")
    batch_norm = args.batch_norm if args.batch_norm is not None else modern_default
    dropout = args.dropout if args.dropout is not None else (0.15 if modern_default else 0.0)
    try:
        channels = tuple(int(value) for value in args.channels.split(",")) if args.channels else None
        if channels is not None and len(channels) != 3: raise ValueError
    except ValueError: parser.error("--channels must be three comma-separated positive integers, e.g. 24,72,288")
    try: config = make_config(model_name, activation=activation, pooling=pooling, channels=channels, hidden_dim=args.hidden_dim, leaky_relu_slope=args.leaky_relu_slope, gelu_approximate=args.gelu_approximate, activation_clamp_min=args.activation_clamp_min, activation_clamp_max=args.activation_clamp_max, batch_norm=batch_norm, dropout=dropout)
    except ValueError as error: parser.error(str(error))
    model = ConfigurableLeNet(len(classes), config).to(args.device)
    if compile_model: model = torch.compile(model, mode="reduce-overhead")
    optimizer = (torch.optim.AdamW if args.optimizer == "adamw" else torch.optim.Adam)(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
    loss_fn = nn.CrossEntropyLoss(weight=class_weights.to(args.device) if balance == "loss" else None, label_smoothing=args.label_smoothing)
    scheduler = (torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, args.epochs, eta_min=args.min_learning_rate) if args.scheduler == "cosine" else torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="max", factor=args.plateau_factor, patience=args.plateau_patience, min_lr=args.min_learning_rate) if args.scheduler == "plateau" else None)
    scaler = torch.cuda.amp.GradScaler(enabled=cuda and args.amp); best_state, best_accuracy = None, -1.0
    for epoch in range(1, args.epochs + 1):
        model.train(); total_loss = total = 0
        if cached:
            index_device = train_images.device
            indices = torch.multinomial(sample_weights.to(index_device), len(train_labels), replacement=True) if sample_weights is not None else torch.randperm(len(train_labels), device=index_device)
            batches = ((train_images[index].to(args.device, non_blocking=cuda), train_labels[index].to(args.device, non_blocking=cuda)) for index in indices.split(args.batch_size))
        else: batches = ((x.to(args.device, non_blocking=cuda), y.to(args.device, non_blocking=cuda)) for x, y in train_loader)
        for x, y in batches:
            if cached: x = augment_cached_batch(x, args, -mean / std)
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(device_type="cuda", dtype=torch.float16, enabled=cuda and args.amp): loss = loss_fn(model(x), y)
            scaler.scale(loss).backward(); scaler.step(optimizer); scaler.update(); total_loss += loss.item() * y.size(0); total += y.size(0)
        accuracy, _ = evaluate_cached(model, val_images, val_labels, args.batch_size, args.device, cuda and args.amp, len(classes)) if cached else evaluate(model, val_loader, args.device, cuda, cuda and args.amp, len(classes))
        if scheduler: scheduler.step(accuracy) if args.scheduler == "plateau" else scheduler.step()
        print(f"epoch {epoch:03d}/{args.epochs}: loss={total_loss/total:.4f}, validation_accuracy={accuracy:.2%}, learning_rate={optimizer.param_groups[0]['lr']:.2e}")
        if accuracy > best_accuracy:
            best_accuracy = accuracy; state_model = model._orig_mod if compile_model else model; best_state = {k: v.detach().cpu().clone() for k, v in state_model.state_dict().items()}
    if compile_model: model = model._orig_mod
    model.load_state_dict(best_state)
    final_accuracy, confusion = evaluate_cached(model, val_images, val_labels, args.batch_size, args.device, cuda and args.amp, len(classes)) if cached else evaluate(model, val_loader, args.device, cuda, cuda and args.amp, len(classes))
    per_class = [{"index": i, "label": label, "support": int(confusion[i].sum()), "correct": int(confusion[i, i]), "accuracy": (confusion[i, i].item() / confusion[i].sum().item() if confusion[i].sum() else None)} for i, label in enumerate(classes)]
    augmentation = augmentation_options(args)
    training = {"epochs": args.epochs, "optimizer": args.optimizer, "learning_rate": args.learning_rate, "weight_decay": args.weight_decay, "scheduler": args.scheduler, "label_smoothing": args.label_smoothing, "class_balancing": balance, "class_weight_power": args.class_weight_power, "normalization": {"method": args.normalization, "mean": mean, "std": std}, "augmentation": augmentation, "seed": args.seed, "best_validation_accuracy": best_accuracy, "final_validation_accuracy": final_accuracy, "per_class_accuracy": per_class, "batch_size": args.batch_size, "workers": args.workers, "amp": cuda and args.amp, "dataset_cache": args.cache_dataset, "dataset_cached_on_cuda": args.cache_dataset == "cuda", "model": model_name, "model_config": config.export()}
    model_tag = f"lenet_{model_name}" if model_name != "lenet5" else "lenet5"
    if args.output_file is not None and args.output_file.parent != Path("."):
        parser.error("--output-file must be a filename, not a path; use --output-dir to choose its directory")
    output_name = args.output_file.name if args.output_file is not None else f"{model_tag}_{args.dataset}.pt"
    if not output_name.endswith(".pt"):
        output_name += ".pt"
    model_path, json_path = save_bundle(build_bundle(model, args.dataset, classes, prep, training), args.output_dir / output_name)
    if args.report_confusion_matrix:
        metrics_path = model_path.with_suffix(".metrics.json"); metrics_path.write_text(json.dumps({"validation_accuracy": final_accuracy, "classes": classes, "confusion_matrix": confusion.tolist(), "per_class": per_class}, indent=2) + "\n", encoding="utf-8")
        print(f"Saved validation metrics: {metrics_path}")
    print(f"Saved best model: {model_path}\nSaved inference manifest: {json_path}")


if __name__ == "__main__": main()
