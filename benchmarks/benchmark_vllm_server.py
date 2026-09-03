#!/usr/bin/env python3
"""Matched OpenAI-compatible serving benchmark for QCC and KV baselines.

Run this client against a separately started *stock vLLM* server. Streaming timestamps
measure TTFT and per-request TPOT without relying on internal vLLM profiler APIs. The
same script/workload must be used for QCC, FP8 Full-KV and compression baselines.
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import queue
import statistics
import subprocess
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import asdict, dataclass
from typing import Any
from urllib import request


@dataclass
class RequestResult:
    index: int
    ok: bool
    prompt_chars: int
    completion_tokens: int
    ttft_s: float | None
    e2e_s: float
    tpot_s: float | None
    error: str | None = None


def percentile(values: list[float], q: float) -> float | None:
    if not values:
        return None
    xs = sorted(values)
    if len(xs) == 1:
        return xs[0]
    pos = (len(xs) - 1) * q
    lo = int(math.floor(pos))
    hi = int(math.ceil(pos))
    if lo == hi:
        return xs[lo]
    frac = pos - lo
    return xs[lo] * (1.0 - frac) + xs[hi] * frac


def summarize(values: list[float]) -> dict[str, float | None]:
    return {
        "mean": statistics.fmean(values) if values else None,
        "p50": percentile(values, 0.50),
        "p95": percentile(values, 0.95),
        "p99": percentile(values, 0.99),
        "max": max(values) if values else None,
    }


def load_prompts(path: Path, limit: int | None) -> list[str]:
    prompts: list[str] = []
    with path.open() as handle:
        for line in handle:
            if not line.strip():
                continue
            row = json.loads(line)
            if isinstance(row, str):
                prompt = row
            elif isinstance(row, dict):
                prompt = row.get("prompt") or row.get("text") or row.get("input")
            else:
                prompt = None
            if not isinstance(prompt, str) or not prompt:
                raise ValueError("each JSONL row must contain non-empty prompt/text/input")
            prompts.append(prompt)
            if limit is not None and len(prompts) >= limit:
                break
    if not prompts:
        raise ValueError("workload contains no prompts")
    return prompts


def _parse_sse_line(line: bytes) -> dict[str, Any] | None:
    text = line.decode("utf-8", errors="replace").strip()
    if not text.startswith("data:"):
        return None
    payload = text[5:].strip()
    if not payload or payload == "[DONE]":
        return None
    return json.loads(payload)


def run_one(
    index: int,
    prompt: str,
    *,
    url: str,
    model: str,
    max_tokens: int,
    timeout: float,
) -> RequestResult:
    payload = {
        "model": model,
        "prompt": prompt,
        "max_tokens": max_tokens,
        "temperature": 0,
        "stream": True,
        "stream_options": {"include_usage": True},
    }
    body = json.dumps(payload).encode()
    req = request.Request(
        url.rstrip("/") + "/v1/completions",
        data=body,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    start = time.perf_counter()
    first_content: float | None = None
    last_content: float | None = None
    completion_tokens = 0
    try:
        with request.urlopen(req, timeout=timeout) as response:
            for raw in response:
                row = _parse_sse_line(raw)
                if row is None:
                    continue
                now = time.perf_counter()
                usage = row.get("usage")
                if isinstance(usage, dict) and usage.get("completion_tokens") is not None:
                    completion_tokens = int(usage["completion_tokens"])
                choices = row.get("choices") or []
                has_content = False
                for choice in choices:
                    text = choice.get("text") if isinstance(choice, dict) else None
                    if isinstance(text, str) and text:
                        has_content = True
                        break
                if has_content:
                    if first_content is None:
                        first_content = now
                    last_content = now
        end = time.perf_counter()
        ttft = None if first_content is None else first_content - start
        if completion_tokens <= 0 and first_content is not None:
            # Token usage should be present on stock vLLM with include_usage. Treat
            # missing usage as a benchmark failure rather than counting SSE chunks.
            raise RuntimeError("server omitted completion token usage from streamed response")
        tpot = None
        if completion_tokens > 1 and first_content is not None and last_content is not None:
            tpot = (last_content - first_content) / (completion_tokens - 1)
        return RequestResult(index, True, len(prompt), completion_tokens, ttft, end - start, tpot)
    except Exception as exc:  # keep all failures auditable in output JSON
        return RequestResult(
            index=index,
            ok=False,
            prompt_chars=len(prompt),
            completion_tokens=0,
            ttft_s=None,
            e2e_s=time.perf_counter() - start,
            tpot_s=None,
            error=f"{type(exc).__name__}: {exc}",
        )


class NvidiaMemorySampler:
    def __init__(self, pid: int | None, interval_s: float = 0.05):
        self.pid = pid
        self.interval_s = interval_s
        self.samples_mib: list[float] = []
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None

    def _sample(self) -> None:
        while not self._stop.is_set():
            try:
                proc = subprocess.run(
                    [
                        "nvidia-smi",
                        "--query-compute-apps=pid,used_memory",
                        "--format=csv,noheader,nounits",
                    ],
                    capture_output=True,
                    text=True,
                    timeout=2,
                    check=False,
                )
                total = 0.0
                for line in proc.stdout.splitlines():
                    fields = [part.strip() for part in line.split(",")]
                    if len(fields) != 2:
                        continue
                    sample_pid, memory = int(fields[0]), float(fields[1])
                    if self.pid is None or sample_pid == self.pid:
                        total += memory
                if total > 0:
                    self.samples_mib.append(total)
            except Exception:
                pass
            self._stop.wait(self.interval_s)

    def __enter__(self):
        self._thread = threading.Thread(target=self._sample, daemon=True)
        self._thread.start()
        return self

    def __exit__(self, *_):
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=2)

    @property
    def peak_mib(self) -> float | None:
        return max(self.samples_mib) if self.samples_mib else None


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--url", default="http://127.0.0.1:8000")
    parser.add_argument("--model", required=True)
    parser.add_argument("--workload", type=Path, required=True)
    parser.add_argument("--label", required=True, help="qcc/fp8-fullkv/baseline-name")
    parser.add_argument("--max-tokens", type=int, default=32)
    parser.add_argument("--concurrency", type=int, default=1)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--timeout", type=float, default=900.0)
    parser.add_argument("--warmup", type=int, default=1)
    parser.add_argument("--server-pid", type=int)
    parser.add_argument("--context-length", type=int)
    parser.add_argument("--gpu", default=None)
    parser.add_argument("--vllm-version", default=None)
    parser.add_argument("--model-id", default=None)
    parser.add_argument("--run-id", default=None)
    parser.add_argument(
        "--real-model", action=argparse.BooleanOptionalAction, default=None
    )
    parser.add_argument(
        "--synthetic", action=argparse.BooleanOptionalAction, default=None
    )
    parser.add_argument(
        "--protocol-locked", action=argparse.BooleanOptionalAction, default=None
    )
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.concurrency <= 0 or args.max_tokens <= 0 or args.warmup < 0:
        raise ValueError("concurrency/max-tokens must be positive and warmup non-negative")

    prompts = load_prompts(args.workload, args.limit)
    # Unmeasured warmup requests remove startup/graph-capture effects from the primary run.
    for index in range(min(args.warmup, len(prompts))):
        warm = run_one(
            -index - 1,
            prompts[index],
            url=args.url,
            model=args.model,
            max_tokens=min(args.max_tokens, 4),
            timeout=args.timeout,
        )
        if not warm.ok:
            raise RuntimeError(f"warmup failed: {warm.error}")

    started = time.perf_counter()
    results: list[RequestResult] = []
    with NvidiaMemorySampler(args.server_pid) as memory:
        with ThreadPoolExecutor(max_workers=args.concurrency) as pool:
            futures = [
                pool.submit(
                    run_one,
                    index,
                    prompt,
                    url=args.url,
                    model=args.model,
                    max_tokens=args.max_tokens,
                    timeout=args.timeout,
                )
                for index, prompt in enumerate(prompts)
            ]
            for future in as_completed(futures):
                results.append(future.result())
    wall = time.perf_counter() - started
    results.sort(key=lambda item: item.index)
    successful = [item for item in results if item.ok]
    failures = [item for item in results if not item.ok]
    ttft = [item.ttft_s for item in successful if item.ttft_s is not None]
    tpot = [item.tpot_s for item in successful if item.tpot_s is not None]
    e2e = [item.e2e_s for item in successful]
    completion_tokens = sum(item.completion_tokens for item in successful)
    report = {
        "schema": "qcc-vllm-serving-v1",
        "label": args.label,
        "model": args.model,
        "model_id": args.model_id or args.model,
        "run_id": args.run_id,
        "real_model": args.real_model,
        "synthetic": args.synthetic,
        "protocol_locked": args.protocol_locked,
        "context_length": args.context_length,
        "gpu": args.gpu,
        "vllm_version": args.vllm_version,
        "stock_vllm": True,
        "streaming": True,
        "workload": str(args.workload),
        "num_requests": len(results),
        "successful_requests": len(successful),
        "failed_requests": len(failures),
        "failure_rate": len(failures) / len(results),
        "concurrency": args.concurrency,
        "max_tokens": args.max_tokens,
        "wall_s": wall,
        "completion_tokens": completion_tokens,
        "throughput_tokens_per_s": completion_tokens / wall if wall > 0 else None,
        "request_throughput_per_s": len(successful) / wall if wall > 0 else None,
        "ttft_s": summarize(ttft),
        "tpot_s": summarize(tpot),
        "e2e_s": summarize(e2e),
        "server_peak_gpu_memory_mib": memory.peak_mib,
        "requests": [asdict(item) for item in results],
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, sort_keys=True))
    print(json.dumps({k: report[k] for k in (
        "label", "num_requests", "failed_requests", "throughput_tokens_per_s",
        "ttft_s", "tpot_s", "e2e_s", "server_peak_gpu_memory_mib"
    )}, indent=2))
    if failures:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
