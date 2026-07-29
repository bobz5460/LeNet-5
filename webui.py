"""Local browser UI for drawing and classifying an image with LeNet-5."""

from __future__ import annotations

import argparse
import base64
import io

import numpy as np
import torch
from flask import Flask, jsonify, render_template_string, request
from PIL import Image

from classes import QUICKDRAW_CLASSES
from lenet import load_checkpoint


PAGE = """<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1">
  <title>LeNet-5 · {{ dataset_label }}</title>
  <style>
    :root { color-scheme: dark; font-family: system-ui, sans-serif; }
    body { margin: 0; min-height: 100vh; display: grid; place-items: center; background: #10151c; color: #f5f7fa; }
    main { width: min(92vw, 390px); text-align: center; }
    h1 { font-size: 1.45rem; margin: 0 0 .35rem; }
    p { color: #9aa7b5; margin: 0 0 1.2rem; }
    canvas { display: block; width: 280px; height: 280px; margin: 0 auto 1rem; background: #000; border: 2px solid #586575; border-radius: 8px; cursor: crosshair; touch-action: none; }
    button { border: 0; border-radius: 6px; padding: .7rem 1.25rem; margin: 0 .2rem; font-size: 1rem; cursor: pointer; }
    #predict { background: #4d8df7; color: white; } #clear { background: #2b3440; color: white; }
    #result { min-height: 2rem; margin-top: 1.25rem; font-size: 1.35rem; font-weight: 700; }
    #confidences { margin-top: 1rem; text-align: left; }
    .confidence { display: grid; grid-template-columns: 5.5rem 1fr 3.5rem; gap: .5rem; align-items: center; margin: .35rem 0; font-size: .88rem; }
    .bar { height: .55rem; background: #2b3440; border-radius: 99px; overflow: hidden; }
    .fill { height: 100%; background: #4d8df7; border-radius: 99px; }
    #model { font-size: .8rem; overflow-wrap: anywhere; }
  </style>
</head>
<body><main>
  <h1>LeNet-5 · {{ dataset_label }}</h1>
  <p>Draw in the box, then click Predict.</p>
  <canvas id="canvas" width="280" height="280"></canvas>
  <div><button id="predict">Predict</button><button id="clear">Clear</button></div>
  <div id="result">Draw a sample</div>
  <div id="confidences"></div>
  <p id="model">{{ checkpoint }}</p>
</main>
<script>
const canvas = document.getElementById('canvas'), ctx = canvas.getContext('2d');
ctx.fillStyle = 'black'; ctx.fillRect(0, 0, canvas.width, canvas.height);
let drawing = false, last = null;
function point(e) { const r = canvas.getBoundingClientRect(); return [(e.clientX-r.left)*canvas.width/r.width, (e.clientY-r.top)*canvas.height/r.height]; }
function stroke(e) {
  const p = point(e); ctx.strokeStyle='white'; ctx.lineWidth=14; ctx.lineCap='round'; ctx.lineJoin='round';
  ctx.beginPath(); if (last) { ctx.moveTo(last[0], last[1]); } else { ctx.moveTo(p[0], p[1]); }
  ctx.lineTo(p[0], p[1]); ctx.stroke(); last=p;
}
canvas.addEventListener('pointerdown', e => { drawing=true; last=null; canvas.setPointerCapture(e.pointerId); stroke(e); });
canvas.addEventListener('pointermove', e => { if (drawing) stroke(e); });
canvas.addEventListener('pointerup', () => { drawing=false; last=null; });
canvas.addEventListener('pointercancel', () => { drawing=false; last=null; });
document.getElementById('clear').onclick = () => { ctx.fillStyle='black'; ctx.fillRect(0,0,canvas.width,canvas.height); document.getElementById('result').textContent='Draw a sample'; document.getElementById('confidences').innerHTML=''; };
document.getElementById('predict').onclick = async () => {
  const result = document.getElementById('result'); result.textContent='Predicting…';
  try {
    const response = await fetch('/predict', {method:'POST', headers:{'Content-Type':'application/json'}, body:JSON.stringify({image:canvas.toDataURL('image/png')})});
    const data = await response.json(); if (!response.ok) throw new Error(data.error || 'Prediction failed');
    result.textContent = `Prediction: ${data.label} (${(data.confidence*100).toFixed(1)}% confidence)`;
    document.getElementById('confidences').innerHTML = data.confidences.map(item =>
      `<div class="confidence"><span>${item.label}</span><div class="bar"><div class="fill" style="width:${item.confidence*100}%"></div></div><span>${(item.confidence*100).toFixed(1)}%</span></div>`
    ).join('');
  } catch (error) { result.textContent = error.message; }
};
</script></body></html>"""


def create_app(model: torch.nn.Module, device: torch.device, checkpoint: str, dataset: str) -> Flask:
    app = Flask(__name__)
    labels = QUICKDRAW_CLASSES if dataset == "quickdraw" else tuple(str(i) for i in range(10))

    @app.get("/")
    def index():
        return render_template_string(PAGE, dataset_label="Quick, Draw!" if dataset == "quickdraw" else "MNIST", checkpoint=checkpoint)

    @app.post("/predict")
    def predict():
        try:
            encoded = request.get_json(force=True)["image"].split(",", 1)[1]
            image = Image.open(io.BytesIO(base64.b64decode(encoded))).convert("L").resize((28, 28), Image.Resampling.LANCZOS)
            pixels = torch.from_numpy(np.asarray(image, dtype="float32") / 255.0)
            pixels = ((pixels - 0.1307) / 0.3081).unsqueeze(0).unsqueeze(0).to(device)
            with torch.no_grad():
                probabilities = model(pixels).softmax(1)[0]
            prediction = int(probabilities.argmax())
            confidence_values = [float(value) for value in probabilities]
            return jsonify(
                label=labels[prediction],
                confidence=confidence_values[prediction],
                confidences=[
                    {"label": label, "confidence": confidence}
                    for label, confidence in zip(labels, confidence_values)
                ],
            )
        except Exception as exc:
            return jsonify(error=str(exc)), 400

    return app


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the LeNet-5 browser UI")
    parser.add_argument("--checkpoint", default="checkpoints/lenet_mnist.pt")
    parser.add_argument("--dataset", choices=("mnist", "quickdraw"), default="mnist")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=5000)
    args = parser.parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = load_checkpoint(args.checkpoint, device)
    print(f"Web UI running at http://{args.host}:{args.port} (device: {device})")
    create_app(model, device, args.checkpoint, args.dataset).run(host=args.host, port=args.port, threaded=True)


if __name__ == "__main__":
    main()
