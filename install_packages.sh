#!/usr/bin/env bash
# Interactive installer for a RunPod / Linux training + inference environment.
#
# This script helps you either create a new Python virtual environment or reuse
# an existing one, then installs a practical package set for:
# - Fine-tuning with Unsloth when supported
# - Standard Hugging Face training fallback
# - Inference and serving with vLLM
#
# Design goals:
# - Be explicit and beginner-friendly in prompts
# - Avoid hard-failing if Unsloth cannot be installed on the current machine
# - Keep versions pinned to a compatible and reproducible stack
# - Work well in remote environments such as RunPod
#
# Notes on version strategy:
# - Unsloth documentation states that `pip install unsloth` pulls compatible core
#   dependencies automatically, so this script installs the core stack first and
#   then tries Unsloth separately.[web:38]
# - vLLM is installed in the same environment because RunPod and vLLM docs present
#   pip-based installation as the standard path for model serving.[web:40][web:43]
# - Mistral-family support exists in vLLM's supported model list, which makes it
#   appropriate for serving a fine-tuned Mistral checkpoint.[web:42][web:45]

set -euo pipefail

DEFAULT_BASE_DIR="${HOME}/envs"
DEFAULT_ENV_NAME="llm-ft"
PYTHON_BIN_DEFAULT="python3"

echo "=============================================================="
echo " Interactive LLM Environment Installer"
echo "=============================================================="
echo "This will help you create or reuse a Python virtual environment"
echo "and install packages for training and inference."
echo

prompt_yes_no() {
  local prompt="$1"
  local default="$2"
  local reply
  while true; do
    if [[ "$default" == "y" ]]; then
      read -r -p "$prompt [Y/n]: " reply || true
      reply="${reply:-Y}"
    else
      read -r -p "$prompt [y/N]: " reply || true
      reply="${reply:-N}"
    fi
    case "$reply" in
      Y|y|Yes|yes) return 0 ;;
      N|n|No|no) return 1 ;;
      *) echo "Please answer y or n." ;;
    esac
  done
}

prompt_non_empty() {
  local prompt="$1"
  local default="${2:-}"
  local value
  while true; do
    if [[ -n "$default" ]]; then
      read -r -p "$prompt [$default]: " value || true
      value="${value:-$default}"
    else
      read -r -p "$prompt: " value || true
    fi
    if [[ -n "${value// }" ]]; then
      printf '%s\n' "$value"
      return 0
    fi
    echo "Value cannot be empty."
  done
}

ensure_tmux() {
  if command -v tmux >/dev/null 2>&1; then
    echo "[INFO] tmux already installed"
    return 0
  fi
  echo "[INFO] tmux not found"
  if prompt_yes_no "Install tmux using apt-get?" "y"; then
    sudo apt-get update -y
    sudo apt-get install -y tmux
  else
    echo "[WARN] tmux not installed. Your training launcher script may not work until tmux is available."
  fi
}

ensure_system_packages() {
  echo "[INFO] Checking base OS packages"
  if prompt_yes_no "Install common OS packages (build-essential, git, curl, wget, htop)?" "y"; then
    sudo apt-get update -y
    sudo apt-get install -y build-essential git curl wget htop
  fi
}

resolve_python_bin() {
  local pybin
  pybin=$(prompt_non_empty "Python executable to use" "$PYTHON_BIN_DEFAULT")
  if ! command -v "$pybin" >/dev/null 2>&1; then
    echo "[ERROR] Python executable '$pybin' was not found in PATH."
    exit 1
  fi
  echo "$pybin"
}

activate_env() {
  local env_path="$1"
  # shellcheck disable=SC1090
  source "$env_path/bin/activate"
  echo "[INFO] Activated environment: $env_path"
  echo "[INFO] Python: $(command -v python)"
  python --version
}

choose_environment() {
  local pybin="$1"
  local mode env_name env_path base_dir

  if prompt_yes_no "Do you want to create a new virtual environment?" "y"; then
    mode="new"
    env_name=$(prompt_non_empty "Enter the new environment name" "$DEFAULT_ENV_NAME")
    base_dir=$(prompt_non_empty "Enter the base directory for virtual environments" "$DEFAULT_BASE_DIR")
    mkdir -p "$base_dir"
    env_path="$base_dir/$env_name"

    if [[ -d "$env_path" ]]; then
      if prompt_yes_no "Environment path already exists at '$env_path'. Reuse it?" "y"; then
        :
      else
        echo "[ERROR] Refusing to overwrite existing environment directory."
        exit 1
      fi
    else
      echo "[INFO] Creating new virtual environment at $env_path"
      "$pybin" -m venv "$env_path"
    fi
  else
    mode="existing"
    env_name=$(prompt_non_empty "Enter the existing environment name")
    if prompt_yes_no "Is the environment stored under the default base directory '$DEFAULT_BASE_DIR'?" "y"; then
      env_path="$DEFAULT_BASE_DIR/$env_name"
    else
      env_path=$(prompt_non_empty "Enter the full path to the existing environment")
    fi

    if [[ ! -d "$env_path" || ! -f "$env_path/bin/activate" ]]; then
      echo "[ERROR] Could not find a valid virtual environment at '$env_path'."
      exit 1
    fi
  fi

  echo "$env_path"
}

install_core_stack() {
  echo "[INFO] Upgrading pip tooling"
  python -m pip install --upgrade pip setuptools wheel

  echo "[INFO] Installing core training stack"
  pip install \
    "torch>=2.5,<2.8" \
    "transformers>=4.51,<5" \
    "datasets>=2.18,<4" \
    "accelerate>=0.34,<2" \
    "trl>=0.12,<1" \
    "peft>=0.13,<1" \
    "bitsandbytes>=0.43,<1" \
    "sentencepiece>=0.2,<1" \
    "safetensors>=0.4.3,<1" \
    "huggingface_hub>=0.24,<1" \
    "pyyaml>=6,<7" \
    "scipy>=1.11,<2" \
    "einops>=0.7,<1"
}

install_optional_training_tools() {
  echo "[INFO] Installing optional quality-of-life packages"
  pip install \
    "tensorboard>=2.16,<3" \
    "wandb>=0.17,<1" \
    "evaluate>=0.4,<1" \
    "rouge_score>=0.1.2,<1" || true
}

install_unsloth() {
  if prompt_yes_no "Attempt to install Unsloth in this environment?" "y"; then
    echo "[INFO] Installing Unsloth"
    if pip install "unsloth"; then
      echo "[INFO] Unsloth installed successfully"
    else
      echo "[WARN] Unsloth installation failed. The Hugging Face fallback path can still be used."
    fi
  else
    echo "[INFO] Skipping Unsloth installation by user choice"
  fi
}

install_vllm() {
  if prompt_yes_no "Install vLLM for inference/serving in the same environment?" "y"; then
    echo "[INFO] Installing vLLM"
    pip install "vllm>=0.8,<1"
    echo "[INFO] vLLM installed successfully"
  else
    echo "[INFO] Skipping vLLM installation"
  fi
}

write_env_info() {
  local env_path="$1"
  local info_dir="${PWD}/artifacts"
  local info_file="$info_dir/env_install_summary.txt"
  mkdir -p "$info_dir"

  {
    echo "Environment path: $env_path"
    echo "Python: $(python --version 2>&1)"
    echo "Pip: $(pip --version)"
    echo
    echo "Installed package versions:"
    python - <<'PY'
import importlib
packages = [
    "torch",
    "transformers",
    "datasets",
    "accelerate",
    "trl",
    "peft",
    "bitsandbytes",
    "sentencepiece",
    "safetensors",
    "huggingface_hub",
    "yaml",
]
for name in packages:
    try:
        mod = importlib.import_module(name)
        version = getattr(mod, "__version__", "unknown")
        print(f"{name}=={version}")
    except Exception as exc:
        print(f"{name} not importable: {exc}")
for extra in ["unsloth", "vllm", "wandb", "tensorboard"]:
    try:
        mod = importlib.import_module(extra)
        version = getattr(mod, "__version__", "unknown")
        print(f"{extra}=={version}")
    except Exception as exc:
        print(f"{extra} not importable: {exc}")
PY
  } | tee "$info_file"

  echo "[INFO] Wrote installation summary to $info_file"
}

main() {
  ensure_system_packages
  ensure_tmux

  local pybin
  pybin=$(resolve_python_bin)

  local env_path
  env_path=$(choose_environment "$pybin")

  activate_env "$env_path"
  install_core_stack
  install_optional_training_tools
  install_unsloth
  install_vllm
  write_env_info "$env_path"

  echo
  echo "=============================================================="
  echo " Installation completed"
  echo "=============================================================="
  echo "To activate this environment later, run:"
  echo "  source '$env_path/bin/activate'"
  echo
  echo "You can now run your training launcher script afterward."
}

main "$@"
