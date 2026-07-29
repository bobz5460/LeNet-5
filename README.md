# LeNet-5 MNIST / Quick, Draw!

A compact PyTorch implementation of the classic LeNet-5 convolutional network, plus a local browser UI for drawing and testing a trained model.

## Setup

```bash
uv sync
```

`uv` creates the environment and installs the dependencies. PyTorch uses CUDA automatically when a CUDA-capable installation and GPU are available; otherwise it falls back to the CPU.

## Train

MNIST is downloaded automatically into `data/` the first time. The trainer uses a current fallback mirror because some torchvision versions still reference the retired Yann LeCun download URL:

```bash
uv run python train.py --epochs 5
```

The best model is saved to `checkpoints/lenet_mnist.pt`. Use `--epochs`, `--batch-size`, `--lr`, or `--data-dir` to customize training. CUDA is selected automatically when available. PyTorch operations use all detected CPU cores; data-loader workers are automatically capped to avoid exhausting system file descriptors. Override with `--workers N` if needed.

Training applies random rotations up to ±10° and translations of up to 50% horizontally and vertically. Test images are not augmented.

### Train on Quick, Draw! objects

Quick, Draw! files for airplane, apple, banana, bicycle, cat, dog, fish, flower, house, and star are downloaded automatically into `data/quickdraw/`:

```bash
uv run python train.py --dataset quickdraw --epochs 5 --output checkpoints/lenet_quickdraw.pt
```

By default this uses 10,000 training drawings per class and all remaining drawings for testing. Use `--quickdraw-limit 0` to use every available training drawing. Quick, Draw! files are much larger than MNIST, so the first download may take a while.

## Test with the web UI

```bash
uv run python webui.py
```

Open the printed local URL in a browser. By default it loads the MNIST checkpoint. Use another port with `--port 8080` if needed.

Draw a white-on-black digit with the mouse, then click **Predict**. To use another checkpoint:

```bash
uv run python webui.py --checkpoint path/to/model.pt
```

For a Quick, Draw! checkpoint, use:

```bash
uv run python webui.py --dataset quickdraw --checkpoint checkpoints/lenet_quickdraw.pt
```

The network is the classic 28x28 LeNet-5 layout: 1→6 convolution, average pooling, 6→16 convolution, average pooling, then fully connected layers 256→120→84→10.
