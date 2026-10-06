"""Vercel serverless function: POST /api/chat  {"text": "..."}  ->  intent + reply.
GET /api/chat returns model information and evaluation metrics."""
import json
import os
import sys
import time
from http.server import BaseHTTPRequestHandler

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "_lib"))

from intent_np import IntentModel  # noqa: E402
from responses import respond      # noqa: E402

MODEL_DIR = os.path.join(HERE, "_model")
MODEL = IntentModel(MODEL_DIR)     # loaded once per warm serverless instance
SUMMARY = json.load(open(os.path.join(MODEL_DIR, "summary.json")))
MAX_CHARS = 300


class handler(BaseHTTPRequestHandler):
    def _send(self, code, payload):
        body = json.dumps(payload).encode()
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Cache-Control", "no-store")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):
        self._send(200, {"status": "ok", "model": "BiLSTM + attention intent classifier",
                         "intents": len(MODEL.labels) - 1,
                         "oos_rule": {"threshold": MODEL.threshold, "ratio": MODEL.ratio},
                         "metrics": SUMMARY})

    def do_POST(self):
        try:
            length = int(self.headers.get("Content-Length") or 0)
            data = json.loads(self.rfile.read(length) or b"{}")
            text = str(data.get("text", "")).strip()[:MAX_CHARS]
        except (ValueError, json.JSONDecodeError):
            return self._send(400, {"error": "Send JSON like {\"text\": \"hello\"}"})
        if not text:
            return self._send(400, {"error": "Empty text"})
        t0 = time.perf_counter()
        result = MODEL.predict(text)
        result["reply"] = respond(result["intent"], text)
        result["text"] = text
        result["latency_ms"] = round((time.perf_counter() - t0) * 1000, 1)
        self._send(200, result)
