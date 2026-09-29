import json
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

from benchmarks.benchmark_vllm_server import (
    _parse_sse_line,
    percentile,
    run_one,
    summarize,
)


def test_percentiles_are_interpolated_and_tail_visible():
    values = [1.0, 2.0, 3.0, 100.0]
    summary = summarize(values)
    assert summary["p50"] == 2.5
    assert summary["p95"] > 80.0
    assert summary["p99"] > summary["p95"]
    assert percentile([], 0.5) is None


def test_sse_parser_ignores_done_and_reads_json():
    assert _parse_sse_line(b"data: [DONE]\n") is None
    assert _parse_sse_line(b"event: ping\n") is None
    assert _parse_sse_line(b'data: {"choices":[{"text":"7"}]}\n')["choices"][0]["text"] == "7"


class _Handler(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"

    def log_message(self, *_args):
        pass

    def do_POST(self):
        length = int(self.headers["Content-Length"])
        payload = json.loads(self.rfile.read(length))
        assert payload["stream"] is True
        rows = [
            {"choices": [{"text": "12"}], "usage": None},
            {"choices": [{"text": "34"}], "usage": None},
            {"choices": [{"text": "5678"}], "usage": {"completion_tokens": 3}},
        ]
        body = "".join("data: " + json.dumps(row) + "\n\n" for row in rows)
        body += "data: [DONE]\n\n"
        encoded = body.encode()
        self.send_response(200)
        self.send_header("Content-Type", "text/event-stream")
        self.send_header("Content-Length", str(len(encoded)))
        self.end_headers()
        # Split the response so the client sees real streaming timestamps.
        midpoint = encoded.find(b"\n\n") + 2
        self.wfile.write(encoded[:midpoint])
        self.wfile.flush()
        time.sleep(0.01)
        self.wfile.write(encoded[midpoint:])
        self.wfile.flush()


def test_run_one_requires_real_usage_and_measures_streaming_timing():
    server = ThreadingHTTPServer(("127.0.0.1", 0), _Handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        result = run_one(
            0,
            "hello",
            url=f"http://127.0.0.1:{server.server_port}",
            model="test-model",
            max_tokens=3,
            timeout=5.0,
        )
    finally:
        server.shutdown()
        thread.join(timeout=2)
        server.server_close()
    assert result.ok is True
    assert result.prompt_tokens is None
    assert result.completion_tokens == 3
    assert result.ttft_s is not None and result.ttft_s >= 0
    assert result.tpot_s is not None and result.tpot_s >= 0
    assert result.e2e_s >= result.ttft_s
