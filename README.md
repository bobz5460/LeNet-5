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

By default, the checkpoint name is based on the selected model and dataset. Supply
`--output-file` to choose it yourself; the adjacent manifest, weights archive, and
metrics file use the same base name:

```bash
python3 train.py mnist --output-dir exports/mnist --output-file my-digits-model.pt --epochs 10
```

`emnist-byclass` downloads EMNIST ByClass automatically. It contains 814,255 examples across 62 labels: `0–9`, `A–Z`, and `a–z`. Training now defaults to the wider `large` model for every dataset, with GELU, max pooling, batch normalization, dropout, label smoothing, and affine augmentation. This is a better baseline for handwriting collected outside the benchmark. Select `--model lenet5` for the original 6→16→120 Tanh/average-pooling topology, or select `--model large`/`--model max` explicitly.

EMNIST files are transposed in storage. The trainer corrects that source-only detail; browser drawings stay upright. Retrain any EMNIST export made before this correction so it is not based on mirrored characters.

## Model configuration

Three LeNet-style presets are available: `lenet5` (6→16→120 channels), `large` (16→48→192), and `max` (32→96→384 with a 512-unit hidden layer). `lenet5` with its defaults is the original LeNet-5 topology: Tanh, average pooling, and no normalization or dropout. The wider presets default to GELU, max pooling, batch normalization, and 0.15 classifier dropout; all presets can use `tanh`, `relu`, `gelu`, `sigmoid`, `leaky_relu`, `elu`, `silu`, or `clamp`, plus `avg` or `max` pooling. Use `--gelu-approximate tanh` for the faster GELU approximation. `--activation-clamp-min` and `--activation-clamp-max` optionally clamp the output of every non-clamp activation; `--activation clamp` requires one or both bounds. You can override convolution widths, hidden layer, batch normalization, and classifier dropout:

```bash
python3 train.py emnist-byclass --model max --activation gelu --pooling max --channels 40,120,480 --hidden-dim 640 --batch-norm --dropout 0.2 --output-dir exports/emnist-max --epochs 50

# GELU with its outputs clipped to [-1, 1]
python3 train.py mnist --model large --activation gelu --gelu-approximate tanh \
  --activation-clamp-min -1 --activation-clamp-max 1 --output-dir exports/gelu-clamped
```

Every exported model includes its resolved configuration and explicit activation/pooling operations in its manifest, so `gui.py` and `export_model.py` can reload any supported combination.

Training defaults to AdamW (`1e-3`, `1e-4` weight decay) and cosine learning-rate decay. All optimization choices are configurable: `--optimizer adam`, `--weight-decay 0`, `--scheduler none`, or `--scheduler plateau`. Cross-entropy label smoothing is opt-in with `--label-smoothing`.

## Accuracy-focused EMNIST training

The input pipeline also crops bright ink, preserves its aspect ratio, and centers it in a 20×20 box before the usual 28×28→32×32 conversion. This makes browser drawings and scanned custom images much closer to MNIST geometry. The exact operation is embedded in new exports, so the web UI applies it identically. Retrain old exports to receive this improvement.

EMNIST uses training-only affine augmentation by default (rotation, translation, scale, and shear), dataset-specific normalization measured from the training split, inverse-frequency weighted loss, AdamW, label smoothing, and cosine decay. Validation and browser inference never use augmentation. Every setting can be disabled or tuned:

```bash
python3 train.py emnist-byclass --model max --activation gelu --pooling max \
  --batch-norm --dropout 0.2 --epochs 50 --batch-size 1024 \
  --rotation-degrees 15 --translate 0.12 --scale-min 0.85 --scale-max 1.15 --shear-degrees 12 \
  --class-balancing loss --class-weight-power 0.5 \
  --optimizer adamw --learning-rate 1e-3 --weight-decay 1e-4 --scheduler cosine \
  --output-dir exports/emnist-max
```

Use `--no-augment`, `--normalization mnist`, `--class-balancing none`, `--scheduler none`, `--no-batch-norm`, and `--dropout 0` for ablations or legacy-style runs. `--class-balancing sampler` uses weighted sampling instead of weighted loss. Training writes `<model>.metrics.json`, which contains the validation confusion matrix and per-class accuracy; the same per-class summary is retained in the checkpoint manifest.

Caching preserves augmentation: the trainer caches the deterministic preprocessed images, then creates a fresh affine transform for each cached batch directly on the model device. On CUDA, caching now defaults to `auto`: it keeps data in VRAM when there is ample free memory, eliminating DataLoader and image-processing stalls that are especially noticeable with small batches. Cache transfers stage 4,096 samples at a time by default, independently of the training `--batch-size`, so choosing a small training batch does not turn the initial GPU load into thousands of tiny transfers. Use `--cache-batch-size` to tune that staging size, `--cache-dataset cuda` to force VRAM caching, `--cache-dataset ram` to keep it in system RAM and stream each batch to the GPU, or `--cache-dataset none` to disable caching. With 50 GB VRAM, CUDA caching is generally the faster option; RAM caching is useful when the dataset is too large for VRAM. CUDA training also compiles by default with a low-overhead mode; use `--no-compile` if its startup cost is not worthwhile for a short run.

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
