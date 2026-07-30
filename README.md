# LeNet-5 training and testing

Train 32×32 grayscale LeNet classifiers for MNIST digits, NIST Special Database 19 (handwritten letters), or EMNIST ByClass (digits plus letters), then test exports with a drawing application.

## Install

Use Python 3.10+ and install the PyTorch build appropriate for your machine from pytorch.org, then:

```bash
pip install -r requirements.txt
```

## Train

Download the datasets first (NIST SD19 makes this approximately 8 GB of free disk space):

```bash
python3 download_data.py
```

Use `python3 download_data.py --skip-nist19` when only training MNIST or EMNIST ByClass.

Then train a model:

```bash
python3 train.py mnist --output-dir exports/mnist --epochs 10
python3 train.py nist19 --nist-root /path/to/by_class --output-dir exports/nist19 --epochs 20
python3 train.py emnist-byclass --output-dir exports/emnist-byclass --epochs 30
```

`emnist-byclass` downloads EMNIST ByClass automatically. It contains 814,255 examples across 62 labels: `0–9`, `A–Z`, and `a–z`. It uses the larger LeNet variant by default (16→48→192 convolution channels and a 256-unit hidden layer), while preserving LeNet's Tanh activations, average pooling, convolutional layout, and Adam/cross-entropy training method. Select `--model lenet5` or `--model large` explicitly to override the default. Its default export is `exports/emnist-byclass/lenet_large_emnist-byclass.pt`.

## Faster training

The trainer uses CUDA mixed precision, pinned memory, persistent data-loader workers, prefetching, cuDNN tuning, and TF32 automatically when CUDA is available. For a long GPU run, start with:

```bash
python3 train.py nist19 --nist-root data/nist19/by_class --output-dir exports/nist19 --device cuda --batch-size 2048 --workers 8 --compile --cache-dataset cuda
python3 train.py emnist-byclass --output-dir exports/emnist-byclass --epochs 30 --device cuda --batch-size 2048 --workers 8 --compile --cache-dataset cuda
```

Lower `--batch-size` if CUDA runs out of memory. On an L40S, `--cache-dataset cuda` stages the transformed dataset once, then trains entirely from GPU memory; it removes filesystem/PIL work from every epoch. `--workers 8` matches this machine's eight logical CPUs; tune it downward if disk contention makes it slower. Classic LeNet-5 is deliberately very small, so a modern GPU may still appear lightly utilized even at maximum useful throughput.

MNIST downloads automatically to `data/`. For NIST SD19 v1, unpack its `by_class` tree first. Image folders can be letters (`A`), decimal ASCII codes (`65`), or hexadecimal ASCII codes (`41`), as in the NIST distribution. Only A–Z are used.

## Web UI

```bash
python3 gui.py exports/mnist/lenet5_mnist.pt
```

This starts a local web app at `http://127.0.0.1:8000` and opens it in your browser. Draw, select **Predict**, and inspect every class probability, the prepared 32×32 model input, all preprocessing settings, architecture metadata, class mapping, training information, and the weight-tensor inventory. Use `--port 8080` or `--no-browser` if needed.

## Portable export contract

Each `.pt` checkpoint has `format_version`, `architecture`, `preprocessing`, `classes`, and `state_dict`. It exports an adjacent JSON manifest and a portable `<model>.weights.npz` tensor archive for non-PyTorch consumers. The manifest specifies exact layer order, cross-correlation convolution, Tanh activations, average-pooling parameters, NCHW input, preprocessing, labels, and weight layouts (`OIHW` for conv, `OI` for linear). It is the contract for non-PyTorch inference. Regenerate/validate its manifest with `python3 export_model.py model.pt`.

## Tests

```bash
python3 -m unittest discover -s tests -v
```
