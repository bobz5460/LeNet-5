#!/usr/bin/env python3
"""Download and prepare every dataset used by this project."""
from __future__ import annotations

import argparse
from pathlib import Path
import tempfile
from urllib.request import Request, urlopen
import zipfile

NIST_SD19_URL = "https://s3.amazonaws.com/nist-srd/SD19/by_class.zip"


def download(url: str, destination: Path) -> None:
    """Atomically download a file, showing its byte progress when available."""
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.is_file():
        return
    request = Request(url, headers={"User-Agent": "LeNet-5 dataset downloader"})
    with urlopen(request) as response, tempfile.NamedTemporaryFile(dir=destination.parent, delete=False) as temporary:
        temporary_path = Path(temporary.name)
        total = int(response.headers.get("Content-Length", 0))
        received = 0
        try:
            while chunk := response.read(1024 * 1024):
                temporary.write(chunk)
                received += len(chunk)
                if total:
                    print(f"\rDownloading {destination.name}: {received / total:.1%}", end="", flush=True)
            print()
            temporary_path.replace(destination)
        except BaseException:
            temporary_path.unlink(missing_ok=True)
            raise


def download_nist19(data_root: Path) -> None:
    nist_dir = data_root / "nist19"
    archive = nist_dir / "by_class.zip"
    ready_file = nist_dir / "by_class" / "41" / "hsf_0" / "hsf_0_00000.png"
    if ready_file.is_file():
        print("NIST SD19 is already extracted.")
        return
    if archive.is_file() and not zipfile.is_zipfile(archive):
        print("Removing invalid NIST SD19 archive.")
        archive.unlink()
    if not archive.is_file():
        print("Downloading NIST SD19 (~984 MB archive; ~7 GB unpacked)...")
        download(NIST_SD19_URL, archive)
    print("Extracting NIST SD19...")
    with zipfile.ZipFile(archive) as zip_file:
        zip_file.extractall(nist_dir)


def download_torchvision_datasets(data_root: Path) -> None:
    try:
        from torchvision import datasets
    except ImportError as error:
        raise SystemExit("Install the project requirements first: python3 -m pip install -r requirements.txt") from error

    print("Downloading/preparing MNIST...")
    datasets.MNIST(str(data_root), train=True, download=True)
    datasets.MNIST(str(data_root), train=False, download=True)
    print("Downloading/preparing EMNIST ByClass (digits, A-Z, and a-z)...")
    datasets.EMNIST(str(data_root), split="byclass", train=True, download=True)
    datasets.EMNIST(str(data_root), split="byclass", train=False, download=True)


def main() -> None:
    parser = argparse.ArgumentParser(description="Download MNIST, EMNIST ByClass, and NIST SD19")
    parser.add_argument("--data-root", type=Path, default=Path("data"))
    parser.add_argument("--skip-nist19", action="store_true", help="Do not download the ~7 GB unpacked NIST SD19 dataset")
    args = parser.parse_args()
    data_root = args.data_root.resolve()
    download_torchvision_datasets(data_root)
    if not args.skip_nist19:
        download_nist19(data_root)
    print(f"Done. Datasets are ready under {data_root}")


if __name__ == "__main__":
    main()
