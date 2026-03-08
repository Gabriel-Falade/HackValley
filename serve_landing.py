"""
Serves landing.html on http://localhost:8080
Run with: python serve_landing.py
"""
import http.server, webbrowser, os, threading

PORT = 8080
DIR  = os.path.dirname(os.path.abspath(__file__))

class Handler(http.server.SimpleHTTPRequestHandler):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, directory=DIR, **kwargs)
    def log_message(self, *args):
        pass  # silence request logs

def _open():
    webbrowser.open(f"http://localhost:{PORT}/landing.html")

threading.Timer(0.4, _open).start()
print(f"FacePlay landing page running at http://localhost:{PORT}/landing.html")
print("Press Ctrl+C to stop.")
with http.server.HTTPServer(("", PORT), Handler) as srv:
    srv.serve_forever()
