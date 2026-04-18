#!/usr/bin/env bash
# Launch merged-model export using settings from train_settings.yml.
#
# This script reads values from train_settings.yml and calls merge_adapter.py.
# It is intended for use after training completes and adapter weights are saved.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="${PROJECT_DIR:-$SCRIPT_DIR}"
CONFIG_PATH="${CONFIG_PATH:-$PROJECT_DIR/train_settings.yml}"
PYTHON_SCRIPT="${PYTHON_SCRIPT:-$PROJECT_DIR/merge_adapter.py}"
VENV_DIR="${VENV_DIR:-$PROJECT_DIR/.venv}"
LOG_DIR="${LOG_DIR:-$PROJECT_DIR/logs}"
TIMESTAMP="$(date +%Y%m%d_%H%M%S)"
LOG_FILE="$LOG_DIR/merge_${TIMESTAMP}.log"

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

read_yaml() {
  python - "$CONFIG_PATH" "$1" <<'PY'
import sys, yaml
config_path, dotted = sys.argv[1], sys.argv[2]
with open(config_path, 'r', encoding='utf-8') as f:
    data = yaml.safe_load(f)
cur = data
for part in dotted.split('.'):
    cur = cur[part]
print(cur)
PY
}

BASE_MODEL="$(read_yaml model.name)"
ADAPTER_DIR="$(read_yaml merge.adapter_dir)"
OUTPUT_DIR="$(read_yaml merge.output_dir)"
DTYPE="$(read_yaml merge.dtype)"

mkdir -p "$(dirname "$OUTPUT_DIR")"
echo "[INFO] Base model: $BASE_MODEL"
echo "[INFO] Adapter dir: $ADAPTER_DIR"
echo "[INFO] Output dir: $OUTPUT_DIR"
echo "[INFO] Dtype: $DTYPE"
echo "[INFO] Log file: $LOG_FILE"

echo "[INFO] Starting merge job"
python "$PYTHON_SCRIPT" \
  --base_model "$BASE_MODEL" \
  --adapter_dir "$ADAPTER_DIR" \
  --output_dir "$OUTPUT_DIR" \
  --dtype "$DTYPE" 2>&1 | tee "$LOG_FILE"

echo "[DONE] Merge script completed"
