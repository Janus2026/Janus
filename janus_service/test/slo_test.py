import json
import time
import random
import math
import os
import bisect
import urllib.request
import urllib.error
import threading
from concurrent.futures import ThreadPoolExecutor

# Set no_proxy to bypass proxies for local connections (equivalent to curl --noproxy "*")
os.environ['no_proxy'] = '*'

# Configuration
URL = "http://127.0.0.1:27888/v1/completions"
MODELS = ["Qwen3-8B"]
# MODELS = ["Qwen3-8B", "Qwen2-7B"]
AVG_TOKENS_PER_SECOND = 15000  # Target average input token throughput
TOTAL_REQUESTS = 300
MAX_WORKERS = 200  # Max concurrent threads (large enough to not bottleneck)

# Prompt length distribution (Zipf: short prompts more frequent, long prompts rare)
MIN_PROMPT_LEN = 20
MAX_PROMPT_LEN = 8000
ZIPF_EXPONENT = 1.0  # Higher = more skewed toward short prompts

random.seed(42)

def build_zipf_cdf(min_val, max_val, s):
    """Build CDF for Zipf distribution over [min_val, max_val]."""
    n = max_val - min_val + 1
    # weight for value (min_val + i) is 1/(i+1)^s
    weights = [1.0 / ((i + 1) ** s) for i in range(n)]
    total = sum(weights)
    cdf = []
    cumsum = 0.0
    for w in weights:
        cumsum += w / total
        cdf.append(cumsum)
    avg = sum((min_val + i) * weights[i] for i in range(n)) / total
    return cdf, avg

# Pre-build Zipf CDF and compute actual average prompt length
_zipf_cdf, AVG_PROMPT_LEN = build_zipf_cdf(MIN_PROMPT_LEN, MAX_PROMPT_LEN, ZIPF_EXPONENT)

def generate_zipf_prompt_len():
    """Sample a prompt length from Zipf distribution."""
    u = random.random()
    idx = bisect.bisect_left(_zipf_cdf, u)
    return MIN_PROMPT_LEN + min(idx, MAX_PROMPT_LEN - MIN_PROMPT_LEN)

def compute_slo(repeats):
    return (100 + repeats * 0.2) / 1000.0  # SLO in seconds

def send_request(model, repeats, req_id):
    prompt = "hello janus " * (repeats // 4 + 1)
    slo_ms = int(100 + repeats * 0.2)
    slo_s = slo_ms / 1000.0
    payload = {
        "model": model,
        "prompt": prompt,
        "max_tokens": 1,
        "temperature": 0,
        "stream": False,
        "ttft_slo": slo_ms
    }
    data = json.dumps(payload).encode('utf-8')
    headers = {"Content-Type": "application/json"}

    req = urllib.request.Request(URL, data=data, headers=headers)

    start = time.time()
    try:
        with urllib.request.urlopen(req, timeout=60) as response:
            body = response.read()
            latency = time.time() - start
            if latency > slo_s:
                print(f"[#{req_id}][{model}] Tokens: {repeats}, TIMEOUT (latency {latency*1000:.1f}ms > slo {slo_ms}ms)")
                return 'timeout', latency
            print(f"[#{req_id}][{model}] Tokens: {repeats}, OK, Latency: {latency*1000:.1f}ms")
            return 'success', latency
    except Exception as e:
        latency = time.time() - start
        print(f"[#{req_id}][{model}] Tokens: {repeats}, FAIL: {e}")
        return 'fail', latency

def run_pressure_test_with_latency():
    # Poisson process: average request arrival rate derived from token rate
    avg_request_rate = AVG_TOKENS_PER_SECOND / AVG_PROMPT_LEN  # requests per second
    avg_interval = 1.0 / avg_request_rate  # mean seconds between requests

    print("\nStarting Poisson-rate pressure test...")
    print(f"Target avg tokens/s: {AVG_TOKENS_PER_SECOND}")
    print(f"Avg request rate: {avg_request_rate:.2f} req/s (mean interval: {avg_interval:.4f}s)")
    print(f"Prompt length distribution: Zipf(s={ZIPF_EXPONENT}), range=[{MIN_PROMPT_LEN}, {MAX_PROMPT_LEN}], avg={AVG_PROMPT_LEN:.1f}")
    print(f"Total requests: {TOTAL_REQUESTS}")

    # Pre-generate all requests
    requests_list = []
    for i in range(TOTAL_REQUESTS):
        model = random.choice(MODELS)
        repeats = generate_zipf_prompt_len()
        requests_list.append((model, repeats))

    latencies = []
    results_lock = threading.Lock()
    success = 0
    timeout = 0
    failed = 0

    def task(model, repeats, req_id):
        nonlocal success, timeout, failed
        status, elapsed = send_request(model, repeats, req_id)
        with results_lock:
            if status == 'success':
                latencies.append(elapsed)
                success += 1
            elif status == 'timeout':
                latencies.append(elapsed)
                timeout += 1
            else:
                failed += 1

    start_time = time.time()
    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        for i in range(TOTAL_REQUESTS):
            # Poisson process: exponentially distributed inter-arrival times
            if i > 0:
                interval = random.expovariate(avg_request_rate)
                time.sleep(interval)
            model, repeats = requests_list[i]
            executor.submit(task, model, repeats, i + 1)
        # ThreadPoolExecutor.__exit__ waits for all submitted tasks to complete

    total_time = time.time() - start_time
    total_tokens = sum(r[1] for r in requests_list)

    print(f"\nTest completed in {total_time:.2f}s")
    if total_time > 0:
        print(f"Overall throughput: {TOTAL_REQUESTS / total_time:.2f} requests/s")
        print(f"Total tokens submitted: {total_tokens}")
        print(f"Actual avg tokens/s: {total_tokens / total_time:.2f}")
    print(f"Success: {success}, Timeout (SLO exceeded): {timeout}, Failed: {failed}")
    print(f"SLO attainment: {success}/{success + timeout + failed} ({success/(success + timeout + failed)*100:.1f}%)")

    if not latencies:
        print("No successful requests to compute latency percentiles.")
        return

    latencies.sort()
    def percentile(sorted_vals, pct):
        idx = max(0, min(len(sorted_vals)-1, int(math.ceil((pct/100.0) * len(sorted_vals))) - 1))
        return sorted_vals[idx]

    p50 = percentile(latencies, 50)
    p90 = percentile(latencies, 90)
    p95 = percentile(latencies, 95)
    p99 = percentile(latencies, 99)
    max_lat = latencies[-1]

    print(f"p50 latency: {p50*1000:.2f} ms")
    print(f"p90 latency: {p90*1000:.2f} ms")
    print(f"p95 latency: {p95*1000:.2f} ms")
    print(f"p99 latency: {p99*1000:.2f} ms")
    print(f"max latency: {max_lat*1000:.2f} ms")

# Run the test
if __name__ == "__main__":
    run_pressure_test_with_latency()