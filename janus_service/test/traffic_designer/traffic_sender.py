"""
Traffic sender script that reads pickle configuration and sends requests
according to the designed traffic curves.

Usage:
    python traffic_sender.py --config traffic_config.pkl --url http://127.0.0.1:28888/v1/completions
"""

import argparse
import json
import math
import os
import pickle
import random
import time
import urllib.request
import urllib.error
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Dict, List, Tuple

import matplotlib.pyplot as plt

# Set no_proxy to bypass proxies for local connections
os.environ['no_proxy'] = '*'

# Configuration
AVERAGE_PROMPT_LEN = 1000
CONCURRENCY = 50

random.seed(42)


def generate_poisson(lam: float) -> int:
    """Generate a Poisson-distributed random number."""
    if lam > 30:
        return max(0, int(random.gauss(lam, math.sqrt(lam))))
    else:
        L = math.exp(-lam)
        k = 0
        p = 1
        while p > L:
            k += 1
            p *= random.random()
        return k - 1


def interpolate_curve(points: List[dict], x: float) -> float:
    """
    Interpolate y value at given x from curve points.
    Points are sorted by x coordinate.
    """
    if not points:
        return 0

    # Find surrounding points
    if x <= points[0]['x']:
        return points[0]['y']
    if x >= points[-1]['x']:
        return points[-1]['y']

    for i in range(len(points) - 1):
        if points[i]['x'] <= x <= points[i + 1]['x']:
            # Linear interpolation
            x0, y0 = points[i]['x'], points[i]['y']
            x1, y1 = points[i + 1]['x'], points[i + 1]['y']
            if x1 == x0:
                return y0
            return y0 + (y1 - y0) * (x - x0) / (x1 - x0)

    return 0


def get_rate_at_time(curves: Dict[str, List[dict]], model: str,
                     t: float, duration: float, max_rate: float) -> float:
    """
    Get the request rate for a model at time t.
    Returns rate in requests per second.
    """
    if model not in curves or not curves[model]:
        return 0

    # Normalize time to 0-1 range
    normalized_t = t / duration
    # Get normalized y value (0-1)
    normalized_y = interpolate_curve(curves[model], normalized_t)
    # Convert to actual rate
    return normalized_y * max_rate


def send_request(url: str, model: str) -> Tuple[bool, float, int]:
    """Send a single request and return (success, latency, token_count)."""
    repeats = generate_poisson(AVERAGE_PROMPT_LEN)
    token_count = repeats

    prompt = "hello janus " * repeats
    payload = {
        "model": model,
        "prompt": prompt,
        "max_tokens": 10,
        "temperature": 0,
        "stream": False
    }
    data = json.dumps(payload).encode('utf-8')
    headers = {"Content-Type": "application/json"}

    req = urllib.request.Request(url, data=data, headers=headers)

    start = time.time()
    try:
        with urllib.request.urlopen(req, timeout=120) as response:
            response.read()
            latency = time.time() - start
            print(f"[{model}] Status: {response.status}, Latency: {latency:.4f}s, Tokens: {token_count}")
            return True, latency, token_count
    except Exception as e:
        print(f"[{model}] Failed: {e}")
        return False, 0, 0


def load_config(config_path: str) -> dict:
    """Load traffic configuration from pickle file."""
    with open(config_path, 'rb') as f:
        return pickle.load(f)


def generate_scheduled_rate_plot(requests_per_second: dict, models: List[str], duration: int):
    """Generate scheduled request rate plot for dry-run mode."""
    print("\nGenerating scheduled request rate plot...")

    colors = plt.cm.get_cmap('tab10', len(models))

    plt.figure(figsize=(12, 6))

    x_axis = list(range(duration))

    for idx, model_name in enumerate(models):
        y_axis = [requests_per_second[t][model_name] for t in x_axis]
        plt.plot(x_axis, y_axis, label=model_name, marker='o', markersize=3,
                 linewidth=2, color=colors(idx))

    plt.title("Scheduled Request Rate Per Second (Dry Run)")
    plt.xlabel("Time (s)")
    plt.ylabel("Requests per second")
    plt.legend(title="Model")
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.tight_layout()
    plt.savefig("scheduled_request_rate.png")
    print("Saved scheduled_request_rate.png")


def run_traffic_test(config_path: str, url: str, dry_run: bool = False):
    """Run traffic test based on configuration."""
    print(f"Loading configuration from {config_path}...")
    config = load_config(config_path)

    models = config['models']
    curves = config['curves']
    duration = config['duration']
    max_rate = config['max_rate']

    print(f"Models: {models}")
    print(f"Duration: {duration}s")
    print(f"Max Rate: {max_rate} req/s")
    print()

    # Generate request schedule
    # For each second, determine how many requests to send for each model
    schedule = []  # List of (time, model) tuples

    for t in range(duration):
        for model in models:
            rate = get_rate_at_time(curves, model, t, duration, max_rate)
            # Number of requests to send this second for this model
            num_requests = int(round(rate))
            for _ in range(num_requests):
                # Distribute requests within the second
                request_time = t + random.random()
                schedule.append((request_time, model))

    # Sort schedule by time
    schedule.sort(key=lambda x: x[0])

    total_requests = len(schedule)
    print(f"Total scheduled requests: {total_requests}")

    if dry_run:
        print("\nDry run mode - showing schedule preview:")
        # Show requests per second per model
        requests_per_second = defaultdict(lambda: defaultdict(int))
        for t, model in schedule:
            requests_per_second[int(t)][model] += 1

        for sec in range(min(10, duration)):
            counts = requests_per_second[sec]
            print(f"  Second {sec}: {dict(counts)}")
        print("  ...")

        # Generate scheduled request rate plot
        generate_scheduled_rate_plot(requests_per_second, models, duration)
        return

    if total_requests == 0:
        print("No requests scheduled. Please draw curves in the web interface.")
        return

    # Execute requests
    print("\nStarting traffic test...")
    request_records = []
    success = 0
    failed = 0
    start_time_global = time.time()

    def task(scheduled_time: float, model: str):
        # Wait until scheduled time
        now = time.time() - start_time_global
        if scheduled_time > now:
            time.sleep(scheduled_time - now)

        t0 = time.time()
        ok, latency, tokens = send_request(url, model)
        rel_start = t0 - start_time_global
        return ok, latency, tokens, rel_start, model

    with ThreadPoolExecutor(max_workers=CONCURRENCY) as executor:
        futures = [executor.submit(task, t, m) for t, m in schedule]
        for f in as_completed(futures):
            try:
                ok, latency, tokens, rel_start, model_name = f.result()
                if ok:
                    success += 1
                    request_records.append({
                        "model": model_name,
                        "start_time_rel": rel_start,
                        "latency": latency,
                        "tokens": tokens
                    })
                else:
                    failed += 1
            except Exception as e:
                print(f"Task error: {e}")
                failed += 1

    total_time = time.time() - start_time_global
    print(f"\nTraffic test completed in {total_time:.2f}s")
    print(f"Success: {success}, Failed: {failed}")

    if not request_records:
        print("No successful requests to analyze.")
        return

    # Statistics
    all_latencies = sorted([r['latency'] for r in request_records])

    def get_percentile(sorted_vals, pct):
        idx = max(0, min(len(sorted_vals) - 1, int(math.ceil((pct / 100.0) * len(sorted_vals))) - 1))
        return sorted_vals[idx]

    p50 = get_percentile(all_latencies, 50)
    p90 = get_percentile(all_latencies, 90)
    p95 = get_percentile(all_latencies, 95)
    p99 = get_percentile(all_latencies, 99)
    max_lat = all_latencies[-1]

    print(f"\nLatency Statistics:")
    print(f"  p50: {p50 * 1000:.2f} ms")
    print(f"  p90: {p90 * 1000:.2f} ms")
    print(f"  p95: {p95 * 1000:.2f} ms")
    print(f"  p99: {p99 * 1000:.2f} ms")
    print(f"  max: {max_lat * 1000:.2f} ms")

    # Generate plots
    generate_plots(request_records, models, duration)


def generate_plots(request_records: List[dict], models: List[str], duration: int):
    """Generate visualization plots."""
    print("\nGenerating plots...")

    colors = plt.cm.get_cmap('tab10', len(models))

    # 1. Request rate per second (actual)
    plt.figure(figsize=(12, 6))

    max_time_rel = max(r['start_time_rel'] for r in request_records)
    total_seconds = int(math.ceil(max_time_rel))

    request_counts = defaultdict(lambda: defaultdict(int))
    for r in request_records:
        sec = int(r['start_time_rel'])
        request_counts[r['model']][sec] += 1

    x_axis = list(range(total_seconds + 1))

    for idx, model_name in enumerate(models):
        y_axis = [request_counts[model_name][t] for t in x_axis]
        plt.plot(x_axis, y_axis, label=model_name, marker='o', markersize=3,
                 linewidth=2, color=colors(idx))

    plt.title("Actual Request Rate Per Second")
    plt.xlabel("Time (s)")
    plt.ylabel("Requests per second")
    plt.legend(title="Model")
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.tight_layout()
    plt.savefig("actual_request_rate.png")
    print("Saved actual_request_rate.png")

    # 2. Token count per second
    plt.figure(figsize=(12, 6))

    token_sums = defaultdict(lambda: defaultdict(int))
    for r in request_records:
        sec = int(r['start_time_rel'])
        token_sums[r['model']][sec] += r['tokens']

    for idx, model_name in enumerate(models):
        y_axis = [token_sums[model_name][t] for t in x_axis]
        plt.plot(x_axis, y_axis, label=model_name, marker='o', markersize=3,
                 linewidth=2, color=colors(idx))

    plt.title("Total Request Tokens Per Second")
    plt.xlabel("Time (s)")
    plt.ylabel("Sum of Request Tokens")
    plt.legend(title="Model")
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.tight_layout()
    plt.savefig("request_token_num.png")
    print("Saved request_token_num.png")

    # 3. Latency CDF
    plt.figure(figsize=(10, 6))

    all_latencies = sorted([r['latency'] for r in request_records])
    n = len(all_latencies)
    y = [(i + 1) / n for i in range(n)]
    x_ms = [val * 1000 for val in all_latencies]

    plt.plot(x_ms, y, marker='.', linestyle='-', label='All Models',
             color='black', linewidth=2)

    for idx, model_name in enumerate(models):
        model_lats = sorted([r['latency'] for r in request_records if r['model'] == model_name])
        if not model_lats:
            continue
        nm = len(model_lats)
        ym = [(i + 1) / nm for i in range(nm)]
        xm_ms = [val * 1000 for val in model_lats]
        plt.plot(xm_ms, ym, linestyle='--', label=model_name, alpha=0.8, color=colors(idx))

    plt.title("Latency CDF")
    plt.xlabel("Latency (ms)")
    plt.ylabel("Cumulative Probability")
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.tight_layout()
    plt.savefig("latency_cdf.png")
    print("Saved latency_cdf.png")


def main():
    parser = argparse.ArgumentParser(description='Send traffic based on designed curves')
    parser.add_argument('--config', type=str, default='traffic_config.pkl',
                        help='Path to pickle configuration file')
    parser.add_argument('--url', type=str, default='http://127.0.0.1:28888/v1/completions',
                        help='Target URL for requests')
    parser.add_argument('--dry-run', action='store_true',
                        help='Show schedule without sending requests')
    parser.add_argument('--concurrency', type=int, default=50,
                        help='Maximum concurrent requests')
    parser.add_argument('--prompt-len', type=int, default=1000,
                        help='Average prompt length (Poisson lambda)')

    args = parser.parse_args()

    global CONCURRENCY, AVERAGE_PROMPT_LEN
    CONCURRENCY = args.concurrency
    AVERAGE_PROMPT_LEN = args.prompt_len

    run_traffic_test(args.config, args.url, args.dry_run)


if __name__ == "__main__":
    main()
