"""Run VoiceBot locally (e.g. from VS Code):  python dev_server.py
Serves the web page and the same /api/chat code that Vercel runs.
Open http://localhost:8000 in Chrome or Edge (the mic works on localhost)."""
import os
import sys
from functools import partial
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer

ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(ROOT, "api"))
import chat  # noqa: E402  (loads the model once)


class DevHandler(chat.handler, SimpleHTTPRequestHandler):
    def do_GET(self):
        if self.path.startswith("/api/chat"):
            return chat.handler.do_GET(self)
        return SimpleHTTPRequestHandler.do_GET(self)

    def do_POST(self):
        if self.path.startswith("/api/chat"):
            return chat.handler.do_POST(self)
        self.send_error(404)

    def end_headers(self):
        if self.path.endswith(".js"):
            self.send_header("Cache-Control", "no-store")
        super().end_headers()


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8000))
    server = ThreadingHTTPServer(("127.0.0.1", port), partial(DevHandler, directory=ROOT))
    print(f"VoiceBot running at http://localhost:{port}  (Ctrl+C to stop)")
    server.serve_forever()
