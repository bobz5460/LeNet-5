# LeNet-5 MNIST / Quick, Draw!

A compact PyTorch implementation of the classic LeNet-5 convolutional network, plus a Tkinter GUI for drawing digits and testing a trained model.

## Setup

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Train

MNIST is downloaded automatically into `data/` the first time. The trainer uses a current fallback mirror because some torchvision versions still reference the retired Yann LeCun download URL:

```bash
python train.py --epochs 5
```

The best model is saved to `checkpoints/lenet_mnist.pt`. Use `--epochs`, `--batch-size`, `--lr`, or `--data-dir` to customize training. CUDA is selected automatically when available.

Training applies random rotations up to ±10° and translations of up to 50% horizontally and vertically. Test images are not augmented.

### Train on Quick, Draw! digits

Quick, Draw! digit files are downloaded automatically into `data/quickdraw/`:

```bash
python train.py --dataset quickdraw --epochs 5 --output checkpoints/lenet_quickdraw.pt
```

By default this uses 10,000 training drawings per digit and all remaining drawings for testing. Use `--quickdraw-limit 0` to use every available training drawing. Quick, Draw! files are much larger than MNIST, so the first download may take a while.

## Test with the GUI

```bash
python gui.py
```

Draw a white-on-black digit with the mouse, then click **Predict**. To use another checkpoint:

```bash
python gui.py --checkpoint path/to/model.pt
```

For a Quick, Draw! checkpoint, use:

```bash
python gui.py --dataset quickdraw --checkpoint checkpoints/lenet_quickdraw.pt
```

The network is the classic 28x28 LeNet-5 layout: 1→6 convolution, average pooling, 6→16 convolution, average pooling, then fully connected layers 256→120→84→10.
