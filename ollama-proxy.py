# ollama-proxy.py — run on Windows (python3)
# listens on 0.0.0.0:11434 and forwards requests to 127.0.0.1:11434
from http.server import BaseHTTPRequestHandler, HTTPServer
import http.client, sys, urllib.parse

LISTEN_HOST = '0.0.0.0'
LISTEN_PORT = 11434
TARGET_HOST = '127.0.0.1'
TARGET_PORT = 11434

class ProxyHandler(BaseHTTPRequestHandler):
    def do_ANY(self):
        # forward request to target
        method = self.command
        path = self.path
        content_length = int(self.headers.get('Content-Length') or 0)
        body = self.rfile.read(content_length) if content_length else None

        conn = http.client.HTTPConnection(TARGET_HOST, TARGET_PORT, timeout=30)
        # rebuild headers
        headers = {k: v for k, v in self.headers.items() if k.lower() != 'host'}
        conn.request(method, path, body=body, headers=headers)
        resp = conn.getresponse()
        data = resp.read()

        self.send_response(resp.status)
        for k, v in resp.getheaders():
            # do not forward hop-by-hop headers
            if k.lower() in ('transfer-encoding','connection','keep-alive','proxy-authenticate','proxy-authorization','te','trailers','upgrade'):
                continue
            self.send_header(k, v)
        self.end_headers()
        if data:
            self.wfile.write(data)

    do_GET = do_ANY
    do_POST = do_ANY
    do_PUT = do_ANY
    do_DELETE = do_ANY
    do_PATCH = do_ANY
    do_OPTIONS = do_ANY

def run():
    server = HTTPServer((LISTEN_HOST, LISTEN_PORT), ProxyHandler)
    print(f"Forwarding 0.0.0.0:{LISTEN_PORT} -> {TARGET_HOST}:{TARGET_PORT}")
    server.serve_forever()

if __name__=='__main__':
    run()
