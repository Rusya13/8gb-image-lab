"""The local web page: http://127.0.0.1:8188

    python -m app.server [--port 8188] [--model qwen-turbo]

Serves one page (app/static/index.html) and a small JSON API. Listens on 127.0.0.1 only;
nothing leaves your PC except the first-run model download from Hugging Face.
"""

from __future__ import annotations

import argparse
import base64
import io
import json
import threading
import time
import traceback
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

from . import monitor
from .engine import MODELS, Engine, versions

ROOT = Path(__file__).resolve().parent
OUT = ROOT.parent / "outputs"
MAX_BODY = 40 * 2**20
SIZES = [(512, 512), (768, 768), (1024, 1024), (1344, 768), (768, 1344), (1344, 1344)]

engine = Engine(log=lambda m: print(m, flush=True))
job = {"running": False, "started": 0.0, "what": "", "last": None, "error": ""}
history: list[dict] = []
job_lock = threading.Lock()


def _models() -> list[dict]:
    return [{"key": m.key, "title": m.title, "steps": m.steps, "can_edit": m.can_edit, "licence": m.licence,
             "download_gb": m.download_gb} for m in MODELS.values()]


def _status() -> dict:
    return {**monitor.snapshot(), "progress": engine.progress, "loaded": engine.key,
            "running": job["running"], "what": job["what"],
            "elapsed": round(time.time() - job["started"], 1) if job["running"] else 0,
            "last": job["last"], "error": job["error"], "history": history[-12:]}


def _run(req: dict) -> None:
    """One job at a time: load the model if needed, then draw."""
    try:
        from PIL import Image

        key = req.get("model", "qwen-turbo")
        if key not in MODELS:
            raise ValueError(f"unknown model {key}")
        load = None
        if engine.key != key:
            job["what"] = f"loading {MODELS[key].title} (the first time this converts the model: minutes)"
            load = engine.load(key)
        image = None
        if req.get("image"):
            raw = base64.b64decode(req["image"].split(",", 1)[-1])
            image = Image.open(io.BytesIO(raw))
            image.load()
        width, height = int(req.get("width", 1024)), int(req.get("height", 1024))
        if not (256 <= width <= 2048 and 256 <= height <= 2048) or width % 32 or height % 32:
            raise ValueError("width and height must be multiples of 32 between 256 and 2048")
        job["what"] = "editing" if image is not None else "drawing"
        picture, stats = engine.draw(str(req.get("prompt", ""))[:4000], width, height,
                                     int(req.get("seed", 0)), image=image)
        if picture is None:
            raise RuntimeError(stats["error"])
        OUT.mkdir(exist_ok=True)
        name = f"{time.strftime('%Y%m%d-%H%M%S')}-{key}-{width}x{height}.png"
        picture.save(OUT / name)
        stats.update(file=name, load_s=load["load_s"] if load else 0.0, title=MODELS[key].title)
        job["last"] = stats
        history.append({k: stats[k] for k in ("title", "width", "height", "steps", "seconds", "vram_peak_mb",
                                               "ram_peak_mb", "shared_growth_mb", "edit", "file", "load_s")})
        job["error"] = ""
    except Exception as e:  # shown on the page; the server keeps running
        traceback.print_exc()
        job["error"] = f"{type(e).__name__}: {e}"
    finally:
        job["running"] = False
        job["what"] = ""


class Handler(BaseHTTPRequestHandler):
    server_version = "8gb-image-lab"

    def _send(self, code: int, body: bytes, ctype: str) -> None:
        self.send_response(code)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(body)

    def _json(self, code: int, obj) -> None:
        self._send(code, json.dumps(obj).encode(), "application/json")

    def _local(self) -> bool:
        """Only this PC's browser: refuses requests that name another host (DNS rebinding)."""
        host = (self.headers.get("Host") or "").split(":")[0]
        if host not in ("127.0.0.1", "localhost"):
            self._json(403, {"error": "local use only"})
            return False
        return True

    def do_GET(self):
        if not self._local():
            return
        path = self.path.split("?")[0]
        if path == "/":
            self._send(200, (ROOT / "static" / "index.html").read_bytes(), "text/html; charset=utf-8")
        elif path == "/api/status":
            self._json(200, _status())
        elif path == "/api/info":
            self._json(200, {"models": _models(), "sizes": SIZES, "versions": versions()})
        elif path.startswith("/outputs/"):
            name = path.rsplit("/", 1)[-1]
            file = OUT / name
            if "/" in name or "\\" in name or ".." in name or not file.is_file() or file.suffix != ".png":
                self._json(404, {"error": "not found"})
            else:
                self._send(200, file.read_bytes(), "image/png")
        else:
            self._json(404, {"error": "not found"})

    def do_POST(self):
        if not self._local():
            return
        if self.headers.get("Content-Type", "").split(";")[0].strip() != "application/json":
            return self._json(415, {"error": "send application/json"})
        try:
            length = int(self.headers.get("Content-Length", "0"))
        except ValueError:
            return self._json(400, {"error": "bad Content-Length"})
        if not 0 <= length <= MAX_BODY:
            return self._json(413, {"error": "too large"})
        try:
            req = json.loads(self.rfile.read(length) or b"{}")
        except json.JSONDecodeError:
            return self._json(400, {"error": "bad JSON"})
        if self.path == "/api/generate":
            with job_lock:
                if job["running"]:
                    return self._json(409, {"error": "busy: one picture at a time"})
                job.update(running=True, started=time.time(), what="starting", error="")
            threading.Thread(target=_run, args=(req,), daemon=True).start()
            self._json(202, {"ok": True})
        elif self.path == "/api/unload":
            with job_lock:
                if job["running"]:
                    return self._json(409, {"error": "busy"})
                engine.unload()
            self._json(200, {"ok": True})
        else:
            self._json(404, {"error": "not found"})

    def log_message(self, fmt, *args):
        pass


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, default=8188)
    ap.add_argument("--open", action="store_true", help="open the page in your browser")
    a = ap.parse_args()
    server = ThreadingHTTPServer(("127.0.0.1", a.port), Handler)
    print(f"\n  Open  http://127.0.0.1:{a.port}  in your browser.  Ctrl+C here to stop.\n", flush=True)
    if a.open:
        import webbrowser

        threading.Timer(1.0, webbrowser.open, args=(f"http://127.0.0.1:{a.port}",)).start()
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass


if __name__ == "__main__":
    main()
