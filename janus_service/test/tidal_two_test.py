import json
import time
import random
import math
import os
import urllib.request
import urllib.error
from concurrent.futures import ThreadPoolExecutor
from collections import defaultdict
import matplotlib.pyplot as plt

# Set no_proxy to bypass proxies for local connections
os.environ['no_proxy'] = '*'

# Configuration
URL = "http://127.0.0.1:28888/v1/completions"
MODELS = ["Qwen2-7B", "Qwen3-4B", "Qwen3-8B", "Qwen2.5-14B"]
AVERAGE_PROMPT_LEN = 1000
CONCURRENCY = 20
TOTAL_REQUESTS = 1000

random.seed(42)

def generate_poisson(lam):
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

def send_request(model):
    repeats = generate_poisson(AVERAGE_PROMPT_LEN)
    # Simple estimation: assuming "hello janus " is roughly 3 tokens
    # For plotting purposes, we record repeats or the estimated length
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
    
    req = urllib.request.Request(URL, data=data, headers=headers)
    
    start = time.time()
    try:
        with urllib.request.urlopen(req, timeout=60) as response:
            response.read()
            latency = time.time() - start
            print(f"[{model}] Status: {response.status}, Latency: {latency:.4f}s, Tokens: {token_count}")
            # Returns: (Success, Latency, Token Count)
            return True, latency, token_count
    except Exception as e:
        print(f"[{model}] Failed: {e}")
        return False, 0, 0

def run_tidal_test_with_latency():
    num_models = len(MODELS)
    if num_models < 1:
        raise SystemExit("No models configured")
    
    # --- Build Dual-Model Tidal Sequence ---
    pairs = []
    if num_models == 1:
        pairs = [(MODELS[0], MODELS[0])]
    else:
        for i in range(num_models):
            m1 = MODELS[i]
            m2 = MODELS[(i + 1) % num_models]
            pairs.append((m1, m2))
    
    num_pairs = len(pairs)
    centers = [(i + 1) * TOTAL_REQUESTS / (num_pairs + 1) for i in range(num_pairs)]
    sigma = max(1.0, TOTAL_REQUESTS / (num_pairs * 2.5))

    model_sequence = []
    print(f"Generating sequence with {num_pairs} shifting hot pairs: {pairs}")

    for i in range(TOTAL_REQUESTS):
        weights = [math.exp(-((i - c) ** 2) / (2 * sigma * sigma)) for c in centers]
        best_pair_idx = max(range(num_pairs), key=lambda j: weights[j])
        current_hot_pair = pairs[best_pair_idx]
        chosen = random.choice(current_hot_pair)
        model_sequence.append(chosen)

    print("Model sequence (first 20):", model_sequence[:20])
    print("\nStarting tidal pressure test...")

    # List to store detailed data
    # Format: {"model": str, "start_time_rel": float, "latency": float, "tokens": int}
    request_records = []
    
    success = 0
    failed = 0
    start_time_global = time.time()

    def task(model):
        t0 = time.time()
        ok, latency, tokens = send_request(model)
        # Record time relative to start
        rel_start = t0 - start_time_global
        return ok, latency, tokens, rel_start, model

    with ThreadPoolExecutor(max_workers=CONCURRENCY) as executor:
        futures = [executor.submit(task, model_sequence[i]) for i in range(TOTAL_REQUESTS)]
        for f in futures:
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

    total_time = time.time() - start_time_global
    print(f"\nTidal test completed in {total_time:.2f}s")
    print(f"Success: {success}, Failed: {failed}")

    if not request_records:
        print("No successful requests to plot.")
        return

    # --- Statistics & Output ---
    all_latencies = [r['latency'] for r in request_records]
    all_latencies.sort()
    
    def get_percentile(sorted_vals, pct):
        idx = max(0, min(len(sorted_vals)-1, int(math.ceil((pct/100.0) * len(sorted_vals))) - 1))
        return sorted_vals[idx]

    p50 = get_percentile(all_latencies, 50)
    p90 = get_percentile(all_latencies, 90)
    p95 = get_percentile(all_latencies, 95)
    p99 = get_percentile(all_latencies, 99)
    max_lat = all_latencies[-1]

    print(f"p50 latency: {p50*1000:.2f} ms")
    print(f"p90 latency: {p90*1000:.2f} ms")
    print(f"p95 latency: {p95*1000:.2f} ms")
    print(f"p99 latency: {p99*1000:.2f} ms")
    print(f"max latency: {max_lat*1000:.2f} ms")

    # --- Plotting Section ---
    print("\nGenerating plots...")
    
    # 1. Plot Request Token Num per Second (Line Chart)
    # Changed from Scatter to Line chart aggregating tokens per second
    plt.figure(figsize=(12, 6))
    
    colors = plt.cm.get_cmap('tab10', len(MODELS))
    
    # Calculate the maximum time (seconds) to define the X-axis range
    max_time_rel = max(r['start_time_rel'] for r in request_records)
    total_seconds = int(math.ceil(max_time_rel))
    
    # Data aggregation: Sum tokens per second per model
    # token_sums[model_name][second] = total_tokens
    token_sums = defaultdict(lambda: defaultdict(int))
    
    for r in request_records:
        sec = int(r['start_time_rel'])
        token_sums[r['model']][sec] += r['tokens']
    
    # Generate X axis (seconds)
    x_axis = list(range(total_seconds + 1))
    
    for idx, model_name in enumerate(MODELS):
        # Generate Y axis (sum of tokens for each second, 0 if no requests)
        y_axis = [token_sums[model_name][t] for t in x_axis]
        
        # Plot line
        plt.plot(x_axis, y_axis, label=model_name, marker='o', markersize=3, linewidth=2, color=colors(idx))

    plt.title("Total New Request Tokens Per Second (Tidal Load)")
    plt.xlabel("Time since start (s)")
    plt.ylabel("Sum of Request Tokens")
    plt.legend(title="Model")
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.tight_layout()
    plt.savefig("request_token_num.png")
    print("Saved request_token_num.png")

    # 2. Plot Latency CDF (latency_cdf.png)
    plt.figure(figsize=(10, 6))
    
    # Global CDF
    n = len(all_latencies)
    y = [ (i+1)/n for i in range(n) ]
    # Convert seconds to ms for readability
    x_ms = [val * 1000 for val in all_latencies]
    
    plt.plot(x_ms, y, marker='.', linestyle='-', label='All Models', color='black', linewidth=2)

    # Individual Model CDFs
    for idx, model_name in enumerate(MODELS):
        model_lats = sorted([r['latency'] for r in request_records if r['model'] == model_name])
        if not model_lats:
            continue
        nm = len(model_lats)
        ym = [(i+1)/nm for i in range(nm)]
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

if __name__ == "__main__":
    run_tidal_test_with_latency()
