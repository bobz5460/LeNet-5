"""Train and export a LeNet-5 model for MNIST or NIST SD19 letters."""
from __future__ import annotations

import argparse
import os
from pathlib import Path
import random

import torch
from torch import nn
from torch.utils.data import DataLoader, random_split

from data import LETTERS, NIST19Letters, mnist_datasets, preprocessing_metadata
from export_model import build_bundle, save_bundle
from lenet5 import LeNet5


def evaluate(model, loader, device, non_blocking, amp_enabled):
    model.eval(); correct = total = 0
    with torch.no_grad():
        for x, y in loader:
            x = x.to(device, non_blocking=non_blocking)
            with torch.autocast(device_type="cuda", dtype=torch.float16, enabled=amp_enabled): prediction = model(x).argmax(1).cpu()
            correct += (prediction == y).sum().item(); total += y.numel()
    return correct / total


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("dataset", choices=("mnist", "nist19")); parser.add_argument("--data-root", default="data")
    parser.add_argument("--nist-root", type=Path); parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--epochs", type=int, default=10); parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--learning-rate", type=float, default=1e-3); parser.add_argument("--val-fraction", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=42); parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--workers", type=int, default=min(4, os.cpu_count() or 1), help="DataLoader worker processes (default: 4 or CPU count)")
    parser.add_argument("--prefetch-factor", type=int, default=4, help="Batches each worker keeps ready")
    parser.add_argument("--amp", action=argparse.BooleanOptionalAction, default=True, help="Use CUDA mixed precision when available")
    parser.add_argument("--compile", action="store_true", help="Use torch.compile (beneficial mainly for longer runs)")
    parser.add_argument("--cpu-threads", type=int, default=0, help="PyTorch CPU compute threads; 0 leaves PyTorch default")
    args = parser.parse_args(); torch.manual_seed(args.seed); random.seed(args.seed)
    if args.cpu_threads > 0: torch.set_num_threads(args.cpu_threads)
    cuda = args.device.startswith("cuda")
    if cuda and not torch.cuda.is_available(): parser.error("--device cuda was requested, but CUDA is unavailable to PyTorch")
    if cuda:
        torch.backends.cudnn.benchmark = True
        torch.set_float32_matmul_precision("high")
    if args.dataset == "mnist":
        train_set, val_set = mnist_datasets(args.data_root); classes = [str(i) for i in range(10)]; prep = preprocessing_metadata()
    else:
        if args.nist_root is None: parser.error("nist19 requires --nist-root")
        full = NIST19Letters(args.nist_root); n_val = max(1, round(len(full) * args.val_fraction)); train_set, val_set = random_split(full, [len(full) - n_val, n_val], generator=torch.Generator().manual_seed(args.seed))
        classes = list(LETTERS); prep = preprocessing_metadata(); prep["operations"].insert(0, {"op": "invert", "reason": "NIST dark-ink scan to MNIST-style bright foreground"})
    loader_args = {"num_workers": args.workers, "pin_memory": cuda}
    if args.workers > 0: loader_args.update({"persistent_workers": True, "prefetch_factor": args.prefetch_factor})
    train_loader = DataLoader(train_set, args.batch_size, shuffle=True, **loader_args)
    val_loader = DataLoader(val_set, args.batch_size, **loader_args)
    model = LeNet5(len(classes)).to(args.device)
    if args.compile: model = torch.compile(model)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate); loss_fn = nn.CrossEntropyLoss(); scaler = torch.cuda.amp.GradScaler(enabled=cuda and args.amp); best_state, best_accuracy = None, -1.0
    for epoch in range(1, args.epochs + 1):
        model.train(); total_loss = total = 0
        for x, y in train_loader:
            x, y = x.to(args.device, non_blocking=cuda), y.to(args.device, non_blocking=cuda); optimizer.zero_grad(set_to_none=True)
            with torch.autocast(device_type="cuda", dtype=torch.float16, enabled=cuda and args.amp): loss = loss_fn(model(x), y)
            scaler.scale(loss).backward(); scaler.step(optimizer); scaler.update()
            total_loss += loss.item() * y.size(0); total += y.size(0)
        accuracy = evaluate(model, val_loader, args.device, cuda, cuda and args.amp); print(f"epoch {epoch:03d}/{args.epochs}: loss={total_loss/total:.4f}, validation_accuracy={accuracy:.2%}")
        if accuracy > best_accuracy:
            best_accuracy = accuracy
            state_model = model._orig_mod if args.compile else model
            best_state = {k: v.detach().cpu().clone() for k, v in state_model.state_dict().items()}
    if args.compile: model = model._orig_mod
    model.load_state_dict(best_state)
    training = {"epochs": args.epochs, "optimizer": "Adam", "learning_rate": args.learning_rate, "seed": args.seed, "best_validation_accuracy": best_accuracy, "batch_size": args.batch_size, "workers": args.workers, "amp": cuda and args.amp}
    model_path, json_path = save_bundle(build_bundle(model, args.dataset, classes, prep, training), args.output_dir / f"lenet5_{args.dataset}.pt")
    print(f"Saved best model: {model_path}\nSaved inference manifest: {json_path}")


if __name__ == "__main__": main()
