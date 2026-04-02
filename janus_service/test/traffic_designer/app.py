"""
Flask web application for designing model traffic curves.
Draw traffic curves with mouse, manage model list, and save to pickle.
"""

import os
import pickle
from flask import Flask, render_template, request, jsonify

app = Flask(__name__)

# Default save path
DEFAULT_SAVE_PATH = os.path.join(os.path.dirname(__file__), 'traffic_config.pkl')


@app.route('/')
def index():
    return render_template('index.html')


@app.route('/api/save', methods=['POST'])
def save_config():
    """Save traffic configuration to pickle file."""
    data = request.get_json()

    save_path = data.get('save_path', DEFAULT_SAVE_PATH)
    config = {
        'models': data.get('models', []),
        'curves': data.get('curves', {}),
        'duration': data.get('duration', 60),
        'max_rate': data.get('max_rate', 100),
    }

    try:
        os.makedirs(os.path.dirname(save_path) or '.', exist_ok=True)
        with open(save_path, 'wb') as f:
            pickle.dump(config, f)
        return jsonify({'success': True, 'path': save_path})
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 500


@app.route('/api/load', methods=['POST'])
def load_config():
    """Load traffic configuration from pickle file."""
    data = request.get_json()
    load_path = data.get('load_path', DEFAULT_SAVE_PATH)

    try:
        if not os.path.exists(load_path):
            return jsonify({'success': False, 'error': 'File not found'}), 404

        with open(load_path, 'rb') as f:
            config = pickle.load(f)
        return jsonify({'success': True, 'config': config})
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 500


if __name__ == '__main__':
    app.run(host='127.0.0.1', port=5000, debug=True)
