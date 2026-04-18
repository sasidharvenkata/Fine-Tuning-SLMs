#!/usr/bin/env bash
# Launch supervised fine-tuning inside a tmux session on a RunPod Linux box.
#
# This script is intentionally verbose and defensive so you can use it as both
# an execution entrypoint and operational documentation.
#
# Responsibilities:
# 1. Create or reuse a Python virtual environment.
# 2. Install the libraries needed for either Unsloth or the Hugging Face fallback path.
# 3. Start a dedicated tmux session.
# 4. Execute the Python trainer and stream logs to a file.
#
# Usage:
#   bash run_training.sh
#   TMUX_SESSION=ft_mistral bash run_training.sh
#
# Helpful tmux commands:
#   tmux attach -t ft_mistral
#   tmux ls
#   tmux kill-session -t ft_mistral

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="${PROJECT_DIR:-$SCRIPT_DIR}"
CONFIG_PATH="${CONFIG_PATH:-$PROJECT_DIR/train_settings.yml}"
PYTHON_SCRIPT="${PYTHON_SCRIPT:-$PROJECT_DIR/train_llm.py}"
TMUX_SESSION="${TMUX_SESSION:-ft_mistral}"
VENV_DIR="${VENV_DIR:-$PROJECT_DIR/.venv}"
LOG_DIR="${LOG_DIR:-$PROJECT_DIR/logs}"
TIMESTAMP="$(date +%Y%m%d_%H%M%S)"
LOG_FILE="$LOG_DIR/train_${TIMESTAMP}.log"

mkdir -p "$LOG_DIR"

if ! command -v tmux >/dev/null 2>&1; then
  echo "[INFO] tmux not found. Installing via apt-get..."
  sudo apt-get update -y
  sudo apt-get install -y tmux
fi

if ! command -v python3 >/dev/null 2>&1; then
  echo "[ERROR] python3 is required but was not found."
  exit 1
fi

if [ ! -d "$VENV_DIR" ]; then
  echo "[INFO] Creating virtual environment at $VENV_DIR"
  python3 -m venv "$VENV_DIR"
fi

# shellcheck disable=SC1091
source "$VENV_DIR/bin/activate"

python -m pip install --upgrade pip setuptools wheel

# Install core training stack. Unsloth may fail on some environments; that is okay because
# the Python trainer can fall back to standard Hugging Face libraries.
pip install \
  "torch" \
  "transformers>=4.40.0" \
  "datasets>=2.18.0" \
  "accelerate>=0.28.0" \
  "trl>=0.8.6" \
  "peft>=0.10.0" \
  "bitsandbytes>=0.43.1" \
  "sentencepiece" \
  "pyyaml" || true

pip install "unsloth" || true

if tmux has-session -t "$TMUX_SESSION" 2>/dev/null; then
  echo "[ERROR] tmux session '$TMUX_SESSION' already exists."
  echo "        Attach with: tmux attach -t $TMUX_SESSION"
  echo "        Or kill it with: tmux kill-session -t $TMUX_SESSION"
  exit 1
fi

TRAIN_CMD="cd '$PROJECT_DIR' && source '$VENV_DIR/bin/activate' && python '$PYTHON_SCRIPT' --config '$CONFIG_PATH' 2>&1 | tee '$LOG_FILE'"

echo "[INFO] Starting tmux session: $TMUX_SESSION"
echo "[INFO] Log file: $LOG_FILE"
tmux new-session -d -s "$TMUX_SESSION" "$TRAIN_CMD"

echo "[INFO] Training launched successfully."
echo "[INFO] Attach: tmux attach -t $TMUX_SESSION"
echo "[INFO] Tail log: tail -f '$LOG_FILE'"
