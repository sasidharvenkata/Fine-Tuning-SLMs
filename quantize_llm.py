#!/usr/bin/env python3
"""Quantize a merged LLM checkpoint based on train_settings.yml.

Supported methods
-----------------
- awq: Uses AutoAWQ-style calibration/export flow.
- autoround: Uses Intel AutoRound for post-training weight-only quantization.

The script reads all required settings from the ``quantization`` section in the
YAML config file so that quantization remains part of the same experiment config
used for training and merging.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict

import yaml
from transformers import AutoTokenizer


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Quantize an LLM using YAML settings.")
    parser.add_argument("--config", default="train_settings.yml", help="Path to YAML settings file.")
    return parser.parse_args()


def load_config(config_path: str) -> Dict[str, Any]:
    with open(config_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def save_metadata(output_dir: Path, payload: Dict[str, Any]) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    metadata_path = output_dir / "quantization_metadata.json"
    print(f"[SAVE-INFO] Writing quantization metadata to: {metadata_path}")
    with open(metadata_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)


def run_awq(qcfg: Dict[str, Any]) -> None:
    from awq import AutoAWQForCausalLM

    model_dir = qcfg["input_model_dir"]
    output_dir = Path(qcfg["output_dir"])
    quant_config = {
        "zero_point": bool(qcfg.get("zero_point", True)),
        "q_group_size": int(qcfg.get("group_size", 128)),
        "w_bit": int(qcfg.get("bits", 4)),
        "version": str(qcfg.get("version", "GEMM")).upper(),
    }

    print(f"[INFO] Loading tokenizer from: {model_dir}")
    tokenizer = AutoTokenizer.from_pretrained(model_dir, trust_remote_code=bool(qcfg.get("trust_remote_code", False)))

    print(f"[INFO] Loading model for AWQ quantization from: {model_dir}")
    model = AutoAWQForCausalLM.from_pretrained(
        model_dir,
        low_cpu_mem_usage=bool(qcfg.get("low_cpu_mem_usage", True)),
        use_cache=False,
    )

    print(f"[INFO] Starting AWQ quantization with config: {quant_config}")
    model.quantize(
        tokenizer,
        quant_config=quant_config,
        calib_data=qcfg.get("calib_data", "pileval"),
        max_calib_samples=int(qcfg.get("max_calib_samples", 128)),
        max_calib_seq_len=int(qcfg.get("max_calib_seq_len", 512)),
    )

    output_dir.mkdir(parents=True, exist_ok=True)
    print(f"[SAVE-INFO] Saving AWQ-quantized model to: {output_dir}")
    model.save_quantized(str(output_dir))
    print(f"[SAVE-INFO] Saving tokenizer to: {output_dir}")
    tokenizer.save_pretrained(str(output_dir))

    save_metadata(output_dir, {
        "method": "awq",
        "input_model_dir": model_dir,
        "output_dir": str(output_dir),
        "quant_config": quant_config,
        "calib_data": qcfg.get("calib_data", "pileval"),
        "max_calib_samples": int(qcfg.get("max_calib_samples", 128)),
        "max_calib_seq_len": int(qcfg.get("max_calib_seq_len", 512)),
    })
    print("[DONE] AWQ quantization completed successfully")


def run_autoround(qcfg: Dict[str, Any]) -> None:
    from auto_round import AutoRound
    from transformers import AutoModelForCausalLM

    model_dir = qcfg["input_model_dir"]
    output_dir = Path(qcfg["output_dir"])
    bits = int(qcfg.get("bits", 4))
    group_size = int(qcfg.get("group_size", 128))
    sym = bool(qcfg.get("sym", True))
    export_format = str(qcfg.get("autoround_export_format", "auto_round"))

    print(f"[INFO] Loading tokenizer from: {model_dir}")
    tokenizer = AutoTokenizer.from_pretrained(model_dir, trust_remote_code=bool(qcfg.get("trust_remote_code", False)))

    print(f"[INFO] Loading model for AutoRound quantization from: {model_dir}")
    model = AutoModelForCausalLM.from_pretrained(
        model_dir,
        torch_dtype="auto",
        trust_remote_code=bool(qcfg.get("trust_remote_code", False)),
        device_map=qcfg.get("device_map", "auto"),
    )

    print("[INFO] Building AutoRound quantizer")
    autoround = AutoRound(
        model,
        tokenizer,
        bits=bits,
        group_size=group_size,
        sym=sym,
        nsamples=int(qcfg.get("max_calib_samples", 128)),
        iters=int(qcfg.get("autoround_iters", 200)),
        seqlen=int(qcfg.get("max_calib_seq_len", 512)),
        batch_size=int(qcfg.get("autoround_batch_size", 8)),
        low_gpu_mem_usage=bool(qcfg.get("low_gpu_mem_usage", False)),
    )

    print(f"[INFO] Starting AutoRound quantization with export format: {export_format}")
    autoround.quantize_and_save(str(output_dir), format=export_format)

    print(f"[SAVE-INFO] Saving tokenizer to: {output_dir}")
    tokenizer.save_pretrained(str(output_dir))
    save_metadata(output_dir, {
        "method": "autoround",
        "input_model_dir": model_dir,
        "output_dir": str(output_dir),
        "bits": bits,
        "group_size": group_size,
        "sym": sym,
        "max_calib_samples": int(qcfg.get("max_calib_samples", 128)),
        "max_calib_seq_len": int(qcfg.get("max_calib_seq_len", 512)),
        "autoround_iters": int(qcfg.get("autoround_iters", 200)),
        "autoround_batch_size": int(qcfg.get("autoround_batch_size", 8)),
        "autoround_export_format": export_format,
    })
    print("[DONE] AutoRound quantization completed successfully")


def main() -> None:
    args = parse_args()
    cfg = load_config(args.config)
    qcfg = cfg["quantization"]
    method = str(qcfg.get("method", "awq")).lower()

    print(f"[INFO] Quantization method selected: {method}")
    if method == "awq":
        run_awq(qcfg)
    elif method == "autoround":
        run_autoround(qcfg)
    else:
        raise ValueError(f"Unsupported quantization method: {method}")


if __name__ == "__main__":
    main()
