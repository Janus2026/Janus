#!/usr/bin/env python3
"""Async traffic sender for high-concurrency SSE streaming load testing.

Replays pruned log traces against the service with thousands of concurrent
streaming SSE connections using asyncio + aiohttp.

Usage:
    python async_traffic_sender.py <pruned_log.txt> [options]

    python async_traffic_sender.py pruned_combined_200_3000_20.txt \
        --url http://10.0.0.1:9888/v1/completions \
        --max-concurrency 5000 \
        --time-limit 30

Input format (one request per line, from parse_and_prune_logs.py):
    <service_name> <HH:MM:SS.ffffff> <input_tokens> <output_tokens>
"""

import argparse
import asyncio
import json
import math
import os
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path

# Add janus_data to path for model_config import
JANUS_DATA_DIR = Path.home() / "janus_data"
if JANUS_DATA_DIR.exists():
    sys.path.insert(0, str(JANUS_DATA_DIR))

from model_config import SERVICE_TO_MODEL_MAP, DEFAULT_MODEL

try:
    import aiohttp
except ImportError:
    print("ERROR: aiohttp is required. Install with: pip install aiohttp")
    sys.exit(1)


# ── Data structures ──────────────────────────────────────────────────────────

@dataclass
class RequestSpec:
    """A single request parsed from the pruned log."""
    offset_secs: float
    model: str
    input_tokens: int
    output_tokens: int
    service: str


@dataclass
class RequestResult:
    """Result of a single completed request."""
    req_id: int
    model: str
    input_tokens: int
    status: str          # "success", "timeout", "severe_timeout", "fail"
    latency: float       # total latency in seconds
    ttft: float = 0.0    # time to first token (seconds), 0 if no token received
    token_count: int = 0  # number of SSE tokens received


@dataclass
class Stats:
    """Aggregate statistics collected during the test."""
    results: list = field(default_factory=list)
    success: int = 0
    timeout: int = 0
    severe_timeout: int = 0
    failed: int = 0
    submitted: int = 0
    lock: asyncio.Lock = field(default_factory=asyncio.Lock)

    async def record(self, result: RequestResult):
        async with self.lock:
            self.results.append(result)
            if result.status == "success":
                self.success += 1
            elif result.status == "severe_timeout":
                self.severe_timeout += 1
                self.timeout += 1
            elif result.status == "timeout":
                self.timeout += 1
            else:
                self.failed += 1


# ── Log parsing (reuses send_pruned_logs.py logic) ──────────────────────────

def parse_pruned_log(path: str) -> list[RequestSpec]:
    """Parse pruned log file.

    Format: <service_name> <HH:MM:SS.ffffff> <input_tokens> <output_tokens>
    Returns sorted list of RequestSpec.
    """
    requests = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            parts = line.split()
            if len(parts) < 4:
                continue
            service = parts[0]
            time_str = parts[1]
            input_tokens = max(1, int(float(parts[2])))
            output_tokens = max(1, int(float(parts[3])))

            # HH:MM:SS.ffffff -> seconds
            h, m, s = time_str.split(":")
            offset_secs = int(h) * 3600 + int(m) * 60 + float(s)

            model = SERVICE_TO_MODEL_MAP.get(service, DEFAULT_MODEL)
            requests.append(RequestSpec(
                offset_secs=offset_secs,
                model=model,
                input_tokens=input_tokens,
                output_tokens=output_tokens,
                service=service,
            ))
    requests.sort(key=lambda r: r.offset_secs)
    return requests


# ── Request sending ──────────────────────────────────────────────────────────

async def send_streaming_request(
    session: aiohttp.ClientSession,
    url: str,
    spec: RequestSpec,
    req_id: int,
    stats: Stats,
    endpoint: str,
):
    """Send a single streaming request, consume all SSE tokens, record metrics."""
    repeats = spec.input_tokens
    max_tokens = spec.output_tokens
    slo_ms = int(100 + repeats * 0.2)
    slo_s = slo_ms / 1000.0

    # Build prompt (same as send_pruned_logs.py)
    prompt = "hello janus " * (repeats // 4 + 1)

    if endpoint == "completions":
        payload = {
            "model": spec.model,
            "prompt": prompt,
            "max_tokens": max_tokens,
            "temperature": 0,
            "stream": True,
            "ttft_slo": slo_ms,
        }
    else:
        payload = {
            "model": spec.model,
            "messages": [{"role": "user", "content": prompt}],
            "max_tokens": max_tokens,
            "temperature": 0,
            "stream": True,
            "ttft_slo": slo_ms,
        }

    start = time.monotonic()
    first_token_time = 0.0
    token_count = 0

    try:
        async with session.post(url, json=payload) as resp:
            if resp.status != 200:
                body = await resp.text()
                latency = time.monotonic() - start
                print(f"[#{req_id}][{spec.model}] HTTP {resp.status}: {body[:200]}")
                result = RequestResult(
                    req_id=req_id, model=spec.model,
                    input_tokens=repeats, status="fail",
                    latency=latency,
                )
                await stats.record(result)
                return

            # Consume SSE stream
            async for line in resp.content:
                decoded = line.decode("utf-8", errors="replace").strip()
                if not decoded or not decoded.startswith("data:"):
                    continue
                token_count += 1
                if token_count == 1:
                    first_token_time = time.monotonic() - start

        latency = time.monotonic() - start
        ttft = first_token_time if first_token_time > 0 else latency

        # Classify result
        if latency > 4 * slo_s:
            status = "severe_timeout"
            tag = "SEVERE_TIMEOUT"
        elif latency > slo_s:
            status = "timeout"
            tag = "TIMEOUT"
        else:
            status = "success"
            tag = "OK"

        print(f"[#{req_id}][{spec.model}] Tokens: {repeats}, {tag}, "
              f"Latency: {latency*1000:.1f}ms, TTFT: {ttft*1000:.1f}ms, "
              f"SSE chunks: {token_count}")

        result = RequestResult(
            req_id=req_id, model=spec.model,
            input_tokens=repeats, status=status,
            latency=latency, ttft=ttft, token_count=token_count,
        )
        await stats.record(result)

    except asyncio.TimeoutError:
        latency = time.monotonic() - start
        print(f"[#{req_id}][{spec.model}] Tokens: {repeats}, TIMEOUT (aiohttp)")
        result = RequestResult(
            req_id=req_id, model=spec.model,
            input_tokens=repeats, status="fail",
            latency=latency,
        )
        await stats.record(result)
    except Exception as e:
        latency = time.monotonic() - start
        print(f"[#{req_id}][{spec.model}] Tokens: {repeats}, FAIL: {e}")
        result = RequestResult(
            req_id=req_id, model=spec.model,
            input_tokens=repeats, status="fail",
            latency=latency,
        )
        await stats.record(result)


# ── Main driver ──────────────────────────────────────────────────────────────

async def run(args):
    requests_list = parse_pruned_log(args.input)
    total = len(requests_list)

    if total == 0:
        print("No requests found in input file.")
        return

    total_tokens = sum(r.input_tokens for r in requests_list)
    duration = requests_list[-1].offset_secs - requests_list[0].offset_secs

    print(f"Loaded {total} requests from {args.input}")
    print(f"Scheduled duration: {duration:.2f}s")
    print(f"Total input tokens: {total_tokens}")
    models = sorted(set(r.model for r in requests_list))
    print(f"Models: {models}")
    print(f"Max concurrency: {args.max_concurrency}")
    print(f"Time limit: {args.time_limit}s | "
          f"Severe timeout limit: {args.severe_timeout_limit}")
    print()

    # aiohttp session with large connection pool
    connector = aiohttp.TCPConnector(
        limit=args.max_concurrency,
        keepalive_timeout=300,
        enable_cleanup_closed=True,
    )
    timeout = aiohttp.ClientTimeout(total=args.request_timeout)

    stats = Stats()
    stop_event = asyncio.Event()

    # Determine API endpoint
    if "chat/completions" in args.url:
        endpoint = "chat"
    else:
        endpoint = "completions"

    async with aiohttp.ClientSession(
        connector=connector, timeout=timeout
    ) as session:
        tasks: list[asyncio.Task] = []
        wall_start = time.monotonic()
        time_base = requests_list[0].offset_secs

        for i, spec in enumerate(requests_list):
            # Check time limit
            elapsed = time.monotonic() - wall_start
            if elapsed >= args.time_limit:
                stats.submitted = i
                print(f"\n*** STOP: time limit reached ({args.time_limit}s), "
                      f"{i}/{total} submitted ***")
                stop_event.set()
                break

            # Check severe timeout limit
            if stats.severe_timeout >= args.severe_timeout_limit:
                stats.submitted = i
                print(f"\n*** STOP: severe timeout limit reached "
                      f"({stats.severe_timeout} >= {args.severe_timeout_limit}) ***")
                stop_event.set()
                break

            # Wait until scheduled time
            target_wall = wall_start + (spec.offset_secs - time_base)
            now = time.monotonic()
            if target_wall > now:
                delay = target_wall - now
                # Sleep, but check stop conditions every 100ms
                while delay > 0 and not stop_event.is_set():
                    await asyncio.sleep(min(delay, 0.1))
                    delay = target_wall - time.monotonic()

            if stop_event.is_set():
                stats.submitted = i
                break

            # Launch request as async task
            task = asyncio.create_task(
                send_streaming_request(
                    session, args.url, spec, i + 1, stats, endpoint
                )
            )
            tasks.append(task)
            stats.submitted = i + 1
        else:
            stats.submitted = total

        # Wait for all in-flight requests to complete
        if tasks:
            in_flight = sum(1 for t in tasks if not t.done())
            if stop_event.is_set():
                print(f"Waiting for {in_flight} in-flight requests to complete...")
            await asyncio.gather(*tasks, return_exceptions=True)

    total_time = time.monotonic() - wall_start
    print_summary(stats, total, total_time, requests_list)


# ── Summary reporting ────────────────────────────────────────────────────────

def percentile(sorted_vals, pct):
    if not sorted_vals:
        return 0.0
    idx = max(0, min(len(sorted_vals) - 1,
                     int(math.ceil((pct / 100.0) * len(sorted_vals))) - 1))
    return sorted_vals[idx]


def print_summary(stats: Stats, total: int, total_time: float,
                   requests_list: list[RequestSpec]):
    print(f"\n{'='*60}")
    print(f"Test completed in {total_time:.2f}s")
    print(f"Requests submitted: {stats.submitted}/{total}")

    if total_time > 0:
        submitted_tokens = sum(
            r.input_tokens for r in requests_list[:stats.submitted]
        )
        print(f"Throughput: {stats.submitted / total_time:.2f} req/s")
        print(f"Input tokens submitted: {submitted_tokens}")
        print(f"Avg tokens/s: {submitted_tokens / total_time:.1f}")

    completed = stats.success + stats.timeout + stats.failed
    print(f"\nSuccess: {stats.success}, "
          f"Timeout: {stats.timeout} (severe: {stats.severe_timeout}), "
          f"Failed: {stats.failed}")
    if completed > 0:
        print(f"SLO attainment: {stats.success}/{completed} "
              f"({stats.success / completed * 100:.1f}%)")

    # Latency percentiles
    latencies = sorted(r.latency for r in stats.results if r.status != "fail")
    ttfts = sorted(r.ttft for r in stats.results
                   if r.status != "fail" and r.ttft > 0)

    if latencies:
        print(f"\nLatency (total):")
        print(f"  p50:  {percentile(latencies, 50)*1000:.2f} ms")
        print(f"  p90:  {percentile(latencies, 90)*1000:.2f} ms")
        print(f"  p95:  {percentile(latencies, 95)*1000:.2f} ms")
        print(f"  p99:  {percentile(latencies, 99)*1000:.2f} ms")
        print(f"  max:  {latencies[-1]*1000:.2f} ms")

    if ttfts:
        print(f"\nTTFT (time to first token):")
        print(f"  p50:  {percentile(ttfts, 50)*1000:.2f} ms")
        print(f"  p90:  {percentile(ttfts, 90)*1000:.2f} ms")
        print(f"  p95:  {percentile(ttfts, 95)*1000:.2f} ms")
        print(f"  p99:  {percentile(ttfts, 99)*1000:.2f} ms")
        print(f"  max:  {ttfts[-1]*1000:.2f} ms")

    # Per-model breakdown
    model_results: dict[str, list[RequestResult]] = {}
    for r in stats.results:
        model_results.setdefault(r.model, []).append(r)

    if len(model_results) > 1:
        print(f"\nPer-model breakdown:")
        for model in sorted(model_results.keys()):
            results = model_results[model]
            ok = sum(1 for r in results if r.status == "success")
            tot = len(results)
            model_latencies = sorted(r.latency for r in results
                                     if r.status != "fail")
            p50 = percentile(model_latencies, 50) * 1000 if model_latencies else 0
            p99 = percentile(model_latencies, 99) * 1000 if model_latencies else 0
            print(f"  {model:20s}: {ok}/{tot} OK, "
                  f"p50={p50:.0f}ms, p99={p99:.0f}ms")
    print(f"{'='*60}")


# ── CLI ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Async traffic sender for high-concurrency SSE load testing"
    )
    parser.add_argument("input", help="Path to pruned log file")
    parser.add_argument(
        "--url", default="http://127.0.0.1:9888/v1/completions",
        help="Target URL (default: http://127.0.0.1:9888/v1/completions)")
    parser.add_argument(
        "--max-concurrency", type=int, default=5000,
        help="Max concurrent connections (default: 5000)")
    parser.add_argument(
        "--time-limit", type=float, default=30.0,
        help="Hard time limit in seconds (default: 30)")
    parser.add_argument(
        "--severe-timeout-limit", type=int, default=10,
        help="Stop after N requests with latency > 4x SLO (default: 10)")
    parser.add_argument(
        "--request-timeout", type=float, default=120.0,
        help="Per-request timeout in seconds (default: 120)")
    args = parser.parse_args()

    # Bypass proxy for local connections
    os.environ["no_proxy"] = "*"

    asyncio.run(run(args))


if __name__ == "__main__":
    main()
