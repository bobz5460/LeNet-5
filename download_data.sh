#!/usr/bin/env bash
# Download MNIST and NIST SD19 v1 into this repository's ignored data/ folder.
set -euo pipefail

project_dir=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
data_dir="$project_dir/data"
nist_dir="$data_dir/nist19"
mnist_raw="$data_dir/MNIST/raw"
nist_zip="$nist_dir/by_class.zip"
nist_url="https://s3.amazonaws.com/nist-srd/SD19/by_class.zip"
mnist_url="https://ossci-datasets.s3.amazonaws.com/mnist"

for command in curl unzip gzip; do
  command -v "$command" >/dev/null || { echo "Missing required command: $command" >&2; exit 1; }
done

mkdir -p "$nist_dir" "$mnist_raw"

if [[ ! -f "$nist_zip" ]] || ! unzip -tq "$nist_zip" >/dev/null 2>&1; then
  echo "Downloading NIST SD19 by_class (~984 MB archive; ~7 GB unpacked)..."
  curl --fail --location --continue-at - --output "$nist_zip" "$nist_url"
else
  echo "NIST SD19 archive is already present and valid."
fi

if [[ ! -f "$nist_dir/by_class/41/hsf_0/hsf_0_00000.png" ]]; then
  echo "Extracting NIST SD19..."
  unzip -q -o "$nist_zip" -d "$nist_dir"
else
  echo "NIST SD19 is already extracted."
fi

download_mnist_file() {
  local file=$1
  local uncompressed=${file%.gz}
  if [[ -f "$mnist_raw/$uncompressed" ]]; then
    echo "MNIST $uncompressed is already ready."
    return
  fi
  echo "Downloading MNIST $file..."
  curl --fail --location --output "$mnist_raw/$file" "$mnist_url/$file"
  gzip -dkf "$mnist_raw/$file"
}

download_mnist_file train-images-idx3-ubyte.gz
download_mnist_file train-labels-idx1-ubyte.gz
download_mnist_file t10k-images-idx3-ubyte.gz
download_mnist_file t10k-labels-idx1-ubyte.gz

echo "Done. MNIST: $data_dir/MNIST | NIST SD19: $nist_dir/by_class"
