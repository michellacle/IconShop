#!/usr/bin/env python3
"""SVG generation web server for IconShop.

Usage:
    python3 scripts/svg_server.py [--port 8081] [--weight proj_log/FIGR_SVG/step_100000]
"""

import argparse
import glob
import json
import os
import subprocess
import sys
import threading
import time
from datetime import datetime
from http.server import HTTPServer, BaseHTTPRequestHandler
from socketserver import ThreadingMixIn

# Global state
MODEL_WEIGHT = None
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PENDING = {}  # request_id -> status dict
PENDING_LOCK = threading.Lock()
REQUEST_COUNTER = 0
CHECKPOINTS = []


def discover_checkpoints():
    """Find all epoch_* and step_* directories under proj_log/FIGR_SVG/."""
    base = os.path.join(PROJECT_ROOT, "proj_log", "FIGR_SVG")
    if not os.path.isdir(base):
        return []
    results = []
    for entry in sorted(os.listdir(base)):
        full = os.path.join(base, entry)
        if os.path.isdir(full) and os.path.isfile(os.path.join(full, "model.safetensors")):
            results.append(entry)
    return results


HTML_PAGE = r"""
<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>IconShop SVG Generator</title>
<style>
  * { box-sizing: border-box; margin: 0; padding: 0; }
  body {
    font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
    background: #0d1117; color: #c9d1d9; padding: 1.5rem 2rem;
    min-height: 100vh;
  }
  h1 { font-size: 1.4rem; margin-bottom: 0.2rem; }
  .subtitle { color: #8b949e; font-size: 0.8rem; margin-bottom: 1.2rem; }
  .controls {
    display: grid; grid-template-columns: 1fr 1fr; gap: 0.8rem;
    margin-bottom: 1rem;
  }
  .control-group { display: flex; flex-direction: column; gap: 0.25rem; }
  .control-group.full { grid-column: 1 / -1; }
  label { font-size: 0.75rem; color: #8b949e; font-weight: 600; text-transform: uppercase; letter-spacing: 0.05em; }
  input[type="text"], select, input[type="number"] {
    padding: 0.5rem 0.75rem; font-size: 0.9rem;
    background: #161b22; border: 1px solid #30363d; border-radius: 6px;
    color: #c9d1d9; outline: none;
  }
  input[type="text"]:focus, select:focus, input[type="number"]:focus { border-color: #58a6ff; }
  select { cursor: pointer; }
  input[type="range"] {
    -webkit-appearance: none; width: 100%; height: 6px;
    background: #30363d; border-radius: 3px; outline: none; margin-top: 0.3rem;
  }
  input[type="range"]::-webkit-slider-thumb {
    -webkit-appearance: none; width: 16px; height: 16px;
    background: #58a6ff; border-radius: 50%; cursor: pointer;
  }
  .range-row { display: flex; align-items: center; gap: 0.5rem; }
  .range-val { font-size: 0.85rem; color: #58a6ff; min-width: 2.5rem; text-align: right; font-weight: 600; }
  .input-row { display: flex; gap: 0.5rem; margin-bottom: 1rem; }
  .input-row input { flex: 1; }
  button {
    padding: 0.55rem 1.4rem; font-size: 0.95rem; font-weight: 600;
    background: #238636; color: #fff; border: none; border-radius: 6px;
    cursor: pointer; white-space: nowrap;
  }
  button:hover { background: #2ea043; }
  button:disabled { background: #30363d; color: #8b949e; cursor: not-allowed; }
  #status {
    margin-bottom: 0.8rem; min-height: 1.4rem; font-size: 0.85rem;
    color: #8b949e;
  }
  #status.error { color: #f85149; }
  #status.done { color: #3fb950; }
  #svg-container {
    background: #fff; border-radius: 8px; padding: 1.5rem;
    display: flex; flex-wrap: wrap; gap: 1rem; align-items: center; justify-content: center;
    min-height: 250px;
  }
  #svg-container svg { max-width: 180px; max-height: 180px; }
  .placeholder { color: #8b949e; }
  .svg-label { font-size: 0.7rem; color: #666; text-align: center; margin-top: 0.25rem; }
  .svg-card { display: flex; flex-direction: column; align-items: center; }
</style>
</head>
<body>
<h1>IconShop SVG Generator</h1>
<p class="subtitle">Model: FIGR_SVG | <span id="cp-count"></span> checkpoints available</p>

<div class="controls">
  <div class="control-group">
    <label for="checkpoint">Checkpoint</label>
    <select id="checkpoint"></select>
  </div>
  <div class="control-group">
    <label for="samples">Samples</label>
    <select id="samples">
      <option value="1" selected>1</option>
      <option value="2">2</option>
      <option value="3">3</option>
      <option value="4">4</option>
    </select>
  </div>
  <div class="control-group">
    <label for="top-p">Top-p (nucleus)</label>
    <div class="range-row">
      <input type="range" id="top-p" min="0.1" max="1.0" step="0.05" value="0.5">
      <span class="range-val" id="top-p-val">0.50</span>
    </div>
  </div>
  <div class="control-group">
    <label for="top-k">Top-k (0 = disabled)</label>
    <input type="number" id="top-k" min="0" max="500" value="0" step="1">
  </div>
  <div class="control-group">
    <label for="temperature">Temperature</label>
    <div class="range-row">
      <input type="range" id="temperature" min="0.1" max="2.0" step="0.05" value="1.0">
      <span class="range-val" id="temperature-val">1.00</span>
    </div>
  </div>
  <div class="control-group">
    <label for="pix-len">Max sequence length</label>
    <select id="pix-len">
      <option value="256">256 (fast)</option>
      <option value="512" selected>512 (default)</option>
      <option value="1024">1024 (complex)</option>
      <option value="2048">2048 (very complex)</option>
    </select>
  </div>
</div>

<div class="input-row">
  <input type="text" id="prompt" placeholder="Enter a prompt (e.g. Aruba, cat, star, rocket)" autofocus>
  <button id="generate" onclick="startGenerate()">Generate</button>
</div>

<div id="status"></div>
<div id="svg-container">
  <span class="placeholder">Your SVG will appear here</span>
</div>

<script>
const promptEl = document.getElementById('prompt');
const btnEl = document.getElementById('generate');
const statusEl = document.getElementById('status');
const svgEl = document.getElementById('svg-container');
const cpSelect = document.getElementById('checkpoint');
const topP = document.getElementById('top-p');
const topPVal = document.getElementById('top-p-val');
const topK = document.getElementById('top-k');
const temperature = document.getElementById('temperature');
const temperatureVal = document.getElementById('temperature-val');
const samples = document.getElementById('samples');

let pollTimer = null;

// Populate checkpoints from server
fetch('/api/checkpoints')
  .then(r => r.json())
  .then(cps => {
    (cps.checkpoints || []).forEach(cp => {
      const opt = document.createElement('option');
      opt.value = cp; opt.textContent = cp;
      cpSelect.appendChild(opt);
    });
    document.getElementById('cp-count').textContent = (cps.checkpoints || []).length;
  });

topP.addEventListener('input', () => {
  topPVal.textContent = parseFloat(topP.value).toFixed(2);
});

temperature.addEventListener('input', () => {
  temperatureVal.textContent = parseFloat(temperature.value).toFixed(2);
});

promptEl.addEventListener('keydown', e => { if (e.key === 'Enter') startGenerate(); });

function setStatus(msg, cls) {
  statusEl.textContent = msg;
  statusEl.className = cls || '';
}

function getOptions() {
  return {
    prompt: promptEl.value.trim(),
    checkpoint: cpSelect.value,
    samples: parseInt(samples.value),
    top_p: parseFloat(topP.value),
    top_k: parseInt(topK.value),
    temperature: parseFloat(temperature.value),
    pix_len: parseInt(document.getElementById('pix-len').value),
  };
}

function startGenerate() {
  const opts = getOptions();
  if (!opts.prompt) return;

  btnEl.disabled = true;
  svgEl.innerHTML = '<span class="placeholder">Generating...</span>';
  setStatus('Submitting request...', '');

  fetch('/generate', {
    method: 'POST',
    headers: {'Content-Type': 'application/json'},
    body: JSON.stringify(opts)
  })
  .then(r => r.json())
  .then(data => {
    if (data.error) {
      setStatus(data.error, 'error');
      btnEl.disabled = false;
      return;
    }
    const rid = data.request_id;
    setStatus('Processing... (model load ~5s, then inference)', '');
    pollResult(rid);
  })
  .catch(err => {
    setStatus('Request failed: ' + err, 'error');
    btnEl.disabled = false;
  });
}

function pollResult(rid) {
  fetch('/result/' + rid)
    .then(r => r.json())
    .then(data => {
      if (data.status === 'running') {
        setStatus('Processing... ' + (data.elapsed || 0) + 's elapsed', '');
        pollTimer = setTimeout(() => pollResult(rid), 1000);
      } else if (data.status === 'done') {
        clearTimeout(pollTimer);
        btnEl.disabled = false;
        const elapsed = data.elapsed || '?';
        // Render all SVGs
        svgEl.innerHTML = '';
        (data.svg_list || []).forEach((svg, i) => {
          const card = document.createElement('div');
          card.className = 'svg-card';
          card.innerHTML = svg + '<div class="svg-label">variant ' + (i+1) + '</div>';
          svgEl.appendChild(card);
        });
        setStatus('Done in ' + elapsed + 's (' + (data.svg_list || []).length + ' SVG(s))', 'done');
      } else if (data.status === 'error') {
        clearTimeout(pollTimer);
        btnEl.disabled = false;
        setStatus('Error: ' + (data.message || 'unknown'), 'error');
      } else if (data.status === 'expired') {
        clearTimeout(pollTimer);
        btnEl.disabled = false;
        setStatus('Request expired (server restarted?)', 'error');
      }
    })
    .catch(() => {
      clearTimeout(pollTimer);
      btnEl.disabled = false;
      setStatus('Lost connection to server', 'error');
    });
}
</script>
</body>
</html>
"""


class ThreadingHTTPServer(ThreadingMixIn, HTTPServer):
    daemon_threads = True
    allow_reuse_address = True


class Handler(BaseHTTPRequestHandler):
    def log_message(self, format, *args):
        print(f"[{datetime.now().strftime('%H:%M:%S')}] {format % args}", file=sys.stderr, flush=True)

    def do_GET(self):
        if self.path == '/' or self.path == '/index.html':
            self.send_response(200)
            self.send_header('Content-Type', 'text/html; charset=utf-8')
            self.end_headers()
            self.wfile.write(HTML_PAGE.encode())
        elif self.path == '/api/checkpoints':
            self._json(200, {"checkpoints": CHECKPOINTS})
        elif self.path.startswith('/result/'):
            rid = self.path.split('/result/')[-1]
            with PENDING_LOCK:
                entry = PENDING.get(rid)
            if entry is None:
                self._json(404, {"status": "expired"})
            elif entry["status"] == "running":
                elapsed = int(time.time() - entry["start"])
                self._json(200, {"status": "running", "elapsed": elapsed})
            else:
                self._json(200, entry)
        else:
            self._json(404, {"error": "not found"})

    def do_POST(self):
        global REQUEST_COUNTER
        if self.path == '/generate':
            length = int(self.headers.get('Content-Length', 0))
            body = self.rfile.read(length)
            try:
                data = json.loads(body)
            except Exception:
                self._json(400, {"error": "invalid JSON"})
                return

            prompt = data.get("prompt", "").strip()
            if not prompt:
                self._json(400, {"error": "prompt is required"})
                return

            # Options with defaults
            checkpoint = data.get("checkpoint", MODEL_WEIGHT or "step_100000")
            n_samples = min(int(data.get("samples", 1)), 4)
            top_p = min(max(float(data.get("top_p", 0.5)), 0.0), 1.0)
            top_k = max(int(data.get("top_k", 0)), 0)
            temperature = min(max(float(data.get("temperature", 1.0)), 0.1), 2.0)
            pix_len = max(int(data.get("pix_len", 512)), 128)

            REQUEST_COUNTER += 1
            rid = f"req-{REQUEST_COUNTER}"
            entry = {
                "status": "running", "start": time.time(),
                "prompt": prompt, "checkpoint": checkpoint,
                "samples": n_samples, "top_p": top_p, "top_k": top_k,
                "temperature": temperature, "pix_len": pix_len,
            }
            with PENDING_LOCK:
                PENDING[rid] = entry

            t = threading.Thread(target=run_generation, args=(rid, entry), daemon=True)
            t.start()
            self._json(202, {"request_id": rid, "status": "running"})
        else:
            self._json(404, {"error": "not found"})

    def _json(self, code, obj):
        body = json.dumps(obj).encode()
        self.send_response(code)
        self.send_header('Content-Type', 'application/json')
        self.send_header('Content-Length', str(len(body)))
        self.end_headers()
        self.wfile.write(body)


def run_generation(rid, entry):
    prompt = entry["prompt"]
    checkpoint = entry["checkpoint"]
    n_samples = entry["samples"]
    top_p = entry["top_p"]
    top_k = entry["top_k"]
    temperature = entry.get("temperature", 1.0)
    pix_len = entry.get("pix_len", 512)

    weight_path = os.path.join(PROJECT_ROOT, "proj_log", "FIGR_SVG", checkpoint)
    if not os.path.isdir(weight_path):
        with PENDING_LOCK:
            PENDING[rid] = {"status": "error", "message": f"Checkpoint not found: {checkpoint}", "start": entry["start"]}
        return

    cmd = [
        sys.executable, os.path.join(PROJECT_ROOT, "cli.py"),
        "--weight", weight_path,
        "--prompt", prompt,
        "-n", str(n_samples),
        "--top-p", str(top_p),
        "--top-k", str(top_k),
        "--temperature", str(temperature),
        "--pix-len", str(pix_len),
    ]

    try:
        start = time.time()
        proc = subprocess.run(
            cmd, capture_output=True, text=True, timeout=180,
            cwd=PROJECT_ROOT,
        )
        elapsed = round(time.time() - start, 1)
        stdout = proc.stdout.strip()
        stderr = proc.stderr.strip()

        if proc.returncode == 0 and stdout:
            svg_lines = [l.strip() for l in stdout.split('\n') if l.strip().startswith('<')]
            if svg_lines:
                with PENDING_LOCK:
                    PENDING[rid].update({
                        "status": "done", "svg_list": svg_lines,
                        "elapsed": elapsed,
                    })
            else:
                with PENDING_LOCK:
                    PENDING[rid].update({
                        "status": "error",
                        "message": "No SVG output produced" + (f" (stderr: {stderr[:200]})" if stderr else ""),
                        "elapsed": elapsed,
                    })
        else:
            msg = stderr or f"exit code {proc.returncode}"
            with PENDING_LOCK:
                PENDING[rid].update({
                    "status": "error", "message": msg[:500], "elapsed": elapsed,
                })

    except subprocess.TimeoutExpired:
        with PENDING_LOCK:
            PENDING[rid].update({"status": "error", "message": "Timeout after 180s", "elapsed": 180})
    except Exception as e:
        with PENDING_LOCK:
            PENDING[rid].update({"status": "error", "message": str(e)[:500]})


def main():
    parser = argparse.ArgumentParser(description="IconShop SVG web server")
    parser.add_argument("--port", type=int, default=8081)
    parser.add_argument("--host", type=str, default="0.0.0.0")
    parser.add_argument("--weight", type=str, default="proj_log/FIGR_SVG/step_100000",
                        help="Default checkpoint directory (relative to project root)")
    args = parser.parse_args()

    global MODEL_WEIGHT
    MODEL_WEIGHT = args.weight

    # Discover checkpoints
    global CHECKPOINTS
    CHECKPOINTS = discover_checkpoints()
    if not CHECKPOINTS:
        print("Warning: no checkpoints found under proj_log/FIGR_SVG/", file=sys.stderr)

    # Validate default weight
    weight_path = os.path.join(PROJECT_ROOT, args.weight)
    if not os.path.isdir(weight_path):
        print(f"Error: default weight not found: {weight_path}", file=sys.stderr)
        sys.exit(1)

    print(f"Serving on http://0.0.0.0:{args.port}/")
    print(f"Default weight: {args.weight}")
    print(f"Checkpoints: {len(CHECKPOINTS)} found")
    print("Press Ctrl+C to stop.\n")

    server = ThreadingHTTPServer((args.host, args.port), Handler)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nShutting down.")
        server.shutdown()


if __name__ == "__main__":
    main()
