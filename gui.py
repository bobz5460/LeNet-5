"""Dependency-free local web UI for a self-describing LeNet-5 export."""
from __future__ import annotations

import argparse
import base64
import io
import json
import webbrowser
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import torch
from PIL import Image

from export_model import load_model, manifest
from webui_preprocess import preprocess


PAGE = r'''<!doctype html><html><head><meta charset="utf-8"><title>LeNet-5 tester</title>
<style>body{font:15px system-ui;margin:0;background:#10131a;color:#e8edf7}main{max-width:1180px;margin:auto;padding:24px}h1{margin-top:0}.grid{display:grid;grid-template-columns:310px 1fr;gap:24px}.card{background:#1b2130;border:1px solid #313c54;border-radius:10px;padding:16px}canvas{background:#000;border-radius:6px;touch-action:none;width:280px;height:280px;cursor:crosshair}button{padding:9px 14px;margin:10px 5px 0 0;border:0;border-radius:6px;background:#55a6ff;color:#061323;font-weight:bold}pre{white-space:pre-wrap;overflow:auto;max-height:600px;background:#0d111a;padding:12px;border-radius:6px}table{border-collapse:collapse;width:100%}td,th{padding:5px;border-bottom:1px solid #313c54;text-align:left}.bar{height:10px;background:#55a6ff;border-radius:3px}</style></head><body><main>
<h1>LeNet-5 handwriting tester</h1><div class="grid"><section class="card"><canvas id="draw" width="280" height="280"></canvas><br><button id="predict">Predict</button><button id="clear">Clear</button><h2 id="answer">Draw a character</h2><img id="prepared" width="160" height="160" alt="Model input preview"></section>
<section><div class="card"><h2>All class probabilities</h2><table><thead><tr><th>Class</th><th>Probability</th><th></th></tr></thead><tbody id="scores"></tbody></table></div><div class="card"><h2>Model information</h2><pre id="info">Loading…</pre></div></section></div></main>
<script>
const c=document.querySelector('#draw'),x=c.getContext('2d');let drawing=false,last,background='black',ink='white';
function clear(){x.fillStyle=background;x.fillRect(0,0,280,280);c.style.background=background;document.querySelector('#answer').textContent='Draw a character';document.querySelector('#scores').innerHTML='';document.querySelector('#prepared').removeAttribute('src')}clear();
function point(e){let r=c.getBoundingClientRect(),p=e.touches?e.touches[0]:e;return [(p.clientX-r.left)*280/r.width,(p.clientY-r.top)*280/r.height]}
function paint(e){if(!drawing)return;e.preventDefault();let p=point(e);x.strokeStyle=ink;x.lineWidth=20;x.lineCap='round';x.beginPath();x.moveTo(...last);x.lineTo(...p);x.stroke();last=p}
c.addEventListener('pointerdown',e=>{drawing=true;last=point(e);c.setPointerCapture(e.pointerId);paint(e)});c.addEventListener('pointermove',paint);c.addEventListener('pointerup',()=>drawing=false);document.querySelector('#clear').onclick=clear;
fetch('/api/info').then(r=>r.json()).then(d=>{document.querySelector('#info').textContent=JSON.stringify(d,null,2);if(d.preprocessing.operations.some(o=>o.op==='invert')){background='white';ink='black';clear()}});
document.querySelector('#predict').onclick=async()=>{let r=await fetch('/api/predict',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({image:c.toDataURL('image/png')})});let d=await r.json();if(d.error){document.querySelector('#answer').textContent=d.error;return}document.querySelector('#answer').textContent=`Prediction: ${d.prediction.label} (${(d.prediction.probability*100).toFixed(2)}%)`;document.querySelector('#prepared').src='data:image/png;base64,'+d.prepared_image;document.querySelector('#scores').innerHTML=d.scores.map(s=>`<tr><td>${s.label}</td><td>${(s.probability*100).toFixed(3)}%</td><td><div class="bar" style="width:${s.probability*100}%"></div></td></tr>`).join('')};
</script></body></html>'''


def public_info(bundle: dict) -> dict:
    """Full non-weight manifest plus an explicit tensor inventory."""
    result = manifest(bundle)
    result["weight_tensors"] = {key: {"shape": list(value.shape), "dtype": str(value.dtype), "layout": "OIHW" if value.ndim == 4 else "OI" if value.ndim == 2 else "O"} for key, value in bundle["state_dict"].items()}
    return result


class AppServer(ThreadingHTTPServer):
    def __init__(self, address, model_path):
        super().__init__(address, Handler); self.model, self.bundle = load_model(model_path); self.info = public_info(self.bundle)


class Handler(BaseHTTPRequestHandler):
    def log_message(self, *_): pass

    def send_json(self, data, status=200):
        body = json.dumps(data).encode(); self.send_response(status); self.send_header("Content-Type", "application/json"); self.send_header("Content-Length", str(len(body))); self.end_headers(); self.wfile.write(body)

    def do_GET(self):
        if self.path == "/":
            body = PAGE.encode(); self.send_response(200); self.send_header("Content-Type", "text/html; charset=utf-8"); self.send_header("Content-Length", str(len(body))); self.end_headers(); self.wfile.write(body)
        elif self.path == "/api/info": self.send_json(self.server.info)
        else: self.send_json({"error": "Not found"}, 404)

    def do_POST(self):
        if self.path != "/api/predict": self.send_json({"error": "Not found"}, 404); return
        try:
            length = int(self.headers["Content-Length"]); encoded = json.loads(self.rfile.read(length))["image"].split(",", 1)[1]
            image = Image.open(io.BytesIO(base64.b64decode(encoded))).convert("L")
            input_tensor, preview = preprocess(image, self.server.bundle["preprocessing"])
            with torch.no_grad(): probabilities = torch.softmax(self.server.model(input_tensor.unsqueeze(0)), 1)[0].tolist()
            classes = self.server.bundle["classes"]; scores = [{"index": i, "label": label, "probability": probabilities[i]} for i, label in enumerate(classes)]
            encoded_preview = io.BytesIO(); preview.save(encoded_preview, format="PNG")
            best = max(scores, key=lambda item: item["probability"]); self.send_json({"prediction": best, "scores": scores, "prepared_image": base64.b64encode(encoded_preview.getvalue()).decode()})
        except Exception as error: self.send_json({"error": str(error)}, 400)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Serve a local LeNet-5 web UI"); parser.add_argument("model", type=Path); parser.add_argument("--port", type=int, default=8000); parser.add_argument("--no-browser", action="store_true"); args = parser.parse_args()
    server = AppServer(("127.0.0.1", args.port), args.model); url = f"http://127.0.0.1:{args.port}"
    print(f"LeNet-5 web UI: {url} (Ctrl+C to stop)")
    if not args.no_browser: webbrowser.open(url)
    try: server.serve_forever()
    except KeyboardInterrupt: print("\nStopped.")
