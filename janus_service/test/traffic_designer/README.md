# Traffic Designer

A web-based tool for designing and executing custom model traffic patterns.

## Overview

1. **Web Frontend (`app.py`)**: Draw traffic curves for different models using your mouse
2. **Traffic Sender (`traffic_sender.py`)**: Execute requests based on saved curves

## Installation

```bash
pip install flask matplotlib
```

## Usage

### Step 1: Design Traffic Curves

Start the web server:

```bash
cd test/traffic_designer
python app.py
```

Open http://127.0.0.1:5000 in your browser.

**Features:**
- Add/remove models from the model list
- Select a model and draw its traffic curve by clicking and dragging on the canvas
- X-axis: time (seconds), Y-axis: request rate (requests/second)
- Configure duration and max rate
- Save configuration to a pickle file

### Step 2: Send Traffic

```bash
python traffic_sender.py --config traffic_config.pkl --url http://127.0.0.1:28888/v1/completions
```

**Options:**
- `--config`: Path to pickle configuration file (default: `traffic_config.pkl`)
- `--url`: Target URL for requests (default: `http://127.0.0.1:28888/v1/completions`)
- `--dry-run`: Show schedule without sending requests
- `--concurrency`: Maximum concurrent requests (default: 50)
- `--prompt-len`: Average prompt length in tokens (default: 1000)

### Example

```bash
# Dry run to preview the schedule
python traffic_sender.py --config traffic_config.pkl --dry-run

# Execute with custom settings
python traffic_sender.py \
    --config traffic_config.pkl \
    --url http://127.0.0.1:28888/v1/completions \
    --concurrency 100 \
    --prompt-len 500
```

## Output

The traffic sender generates three plots:
- `actual_request_rate.png`: Actual requests per second for each model
- `request_token_num.png`: Total tokens per second for each model
- `latency_cdf.png`: Latency cumulative distribution function

## Pickle File Format

```python
{
    'models': ['Qwen2-7B', 'Qwen3-4B', ...],
    'curves': {
        'Qwen2-7B': [{'x': 0.0, 'y': 0.5}, {'x': 0.5, 'y': 1.0}, ...],
        ...
    },
    'duration': 60,      # Total duration in seconds
    'max_rate': 100,     # Maximum request rate (req/s)
}
```

Curve points are normalized to 0-1 range for both x (time) and y (rate).
