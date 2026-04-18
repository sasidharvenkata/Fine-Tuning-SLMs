#!/usr/bin/env bash
# Launch YAML-driven quantization for a merged LLM checkpoint.
#
# This script reads the quantization configuration from train_settings.yml and
# invokes quantize_llm.py. It supports both AWQ and AutoRound depending on the
# selected quantization.method value in the YAML.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="${PROJECT_DIR:-$SCRIPT_DIR}"
CONFIG_PATH="${CONFIG_PATH:-$PROJECT_DIR/train_settings.yml}"
PYTHON_SCRIPT="${PYTHON_SCRIPT:-$PROJECT_DIR/quantize_llm.py}"
VENV_DIR="${VENV_DIR:-$PROJECT_DIR/.venv}"
LOG_DIR="${LOG_DIR:-$PROJECT_DIR/logs}"
TIMESTAMP="$(date +%Y%m%d_%H%M%S)"
LOG_FILE="$LOG_DIR/quantize_${TIMESTAMP}.log"

mkdir -p "$LOG_DIR"

if [[ ! -f "$CONFIG_PATH" ]]; then
  echo "[ERROR] Config not found at $CONFIG_PATH"
  exit 1
fi

if [[ ! -d "$VENV_DIR" ]]; then
  echo "[ERROR] Virtual environment not found at $VENV_DIR"
  exit 1
fi

# shellcheck disable=SC1091
source "$VENV_DIR/bin/activate"

echo "[INFO] Using config: $CONFIG_PATH"
echo "[INFO] Python script: $PYTHON_SCRIPT"
echo "[INFO] Log file: $LOG_FILE"
echo "[INFO] Starting quantization job"
python "$PYTHON_SCRIPT" --config "$CONFIG_PATH" 2>&1 | tee "$LOG_FILE"
echo "[DONE] Quantization script completed"
