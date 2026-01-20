#!/bin/bash
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

export PIPER_VOICE="$SCRIPT_DIR/models/es_ES-sharvard-medium.onnx"

# Start Ollama if not running
if ! pgrep -x "ollama" > /dev/null; then
    echo "Starting Ollama..."
    ollama serve &>/dev/null &
    sleep 2
fi

cd "$SCRIPT_DIR"
uv run python osito.py
