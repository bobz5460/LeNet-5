"""A simple drawing interface for testing a trained MNIST LeNet-5."""

from __future__ import annotations

import argparse
import tkinter as tk
from tkinter import messagebox

import numpy as np
import torch
from PIL import Image, ImageDraw

from lenet import load_checkpoint


class DigitApp:
    def __init__(self, root: tk.Tk, checkpoint: str, dataset: str) -> None:
        self.root = root
        self.root.title(f"LeNet-5 · {dataset.title()} Digit Tester")
        self.root.configure(bg="#10151c")
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        try:
            self.model = load_checkpoint(checkpoint, self.device)
        except (FileNotFoundError, RuntimeError, OSError) as exc:
            messagebox.showerror("Could not load model", f"{checkpoint}\n\nTrain a model first or pass --checkpoint.\n\n{exc}")
            root.destroy()
            return

        self.canvas_size = 280
        self.canvas = tk.Canvas(root, width=self.canvas_size, height=self.canvas_size, bg="black", highlightthickness=0, cursor="crosshair")
        self.canvas.pack(padx=24, pady=(24, 12))
        self.canvas.bind("<B1-Motion>", self.draw)
        self.canvas.bind("<Button-1>", self.draw)
        self.canvas.bind("<ButtonRelease-1>", self.end_stroke)
        self.image = Image.new("L", (self.canvas_size, self.canvas_size), 0)
        self.drawer = ImageDraw.Draw(self.image)
        self.last_point: tuple[int, int] | None = None

        self.result = tk.StringVar(value="Draw a digit")
        tk.Label(root, textvariable=self.result, bg="#10151c", fg="white", font=("TkDefaultFont", 20, "bold")).pack(pady=(0, 14))
        buttons = tk.Frame(root, bg="#10151c")
        buttons.pack(pady=(0, 20))
        tk.Button(buttons, text="Predict", command=self.predict, width=12).pack(side=tk.LEFT, padx=5)
        tk.Button(buttons, text="Clear", command=self.clear, width=12).pack(side=tk.LEFT, padx=5)
        tk.Label(root, text="Model: " + checkpoint, bg="#10151c", fg="#9aa7b5", wraplength=320).pack(padx=12, pady=(0, 18))

    def draw(self, event: tk.Event) -> None:
        # A 14px brush on a 280px canvas becomes roughly 1.4px at MNIST's
        # 28x28 resolution, closer to the original dataset's stroke width.
        radius = 7
        x, y = event.x, event.y
        if self.last_point is None:
            self.canvas.create_oval(x - radius, y - radius, x + radius, y + radius, fill="white", outline="white")
            self.drawer.ellipse((x - radius, y - radius, x + radius, y + radius), fill=255)
        else:
            previous_x, previous_y = self.last_point
            self.canvas.create_line(previous_x, previous_y, x, y, fill="white", width=radius * 2, capstyle=tk.ROUND)
            self.drawer.line((previous_x, previous_y, x, y), fill=255, width=radius * 2)
            self.drawer.ellipse((x - radius, y - radius, x + radius, y + radius), fill=255)
        self.last_point = (x, y)

    def end_stroke(self, _event: tk.Event) -> None:
        self.last_point = None

    def clear(self) -> None:
        self.canvas.delete("all")
        self.image = Image.new("L", (self.canvas_size, self.canvas_size), 0)
        self.drawer = ImageDraw.Draw(self.image)
        self.last_point = None
        self.result.set("Draw a digit")

    def predict(self) -> None:
        small = self.image.resize((28, 28), Image.Resampling.LANCZOS)
        tensor = torch.from_numpy(np.array(small, dtype="float32") / 255.0)
        tensor = (tensor - 0.1307) / 0.3081
        with torch.no_grad():
            probabilities = self.model(tensor.unsqueeze(0).unsqueeze(0).to(self.device)).softmax(1)[0]
        digit = int(probabilities.argmax())
        self.result.set(f"Prediction: {digit}   ({probabilities[digit].item():.1%} confidence)")


def main() -> None:
    parser = argparse.ArgumentParser(description="Draw a digit and classify it with LeNet-5")
    parser.add_argument("--checkpoint", default="checkpoints/lenet_mnist.pt")
    parser.add_argument("--dataset", choices=("mnist", "quickdraw"), default="mnist")
    args = parser.parse_args()
    root = tk.Tk()
    DigitApp(root, args.checkpoint, args.dataset)
    root.mainloop()


if __name__ == "__main__":
    main()
