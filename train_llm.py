#!/usr/bin/env python3
"""Train a supervised fine-tuning adapter for Mistral-7B on yahma/alpaca-cleaned.

This script is designed for single-GPU Linux environments such as RunPod.
It prefers Unsloth for faster and lower-memory fine-tuning when the requested
model and runtime are compatible. If Unsloth is unavailable or initialization
fails, it automatically falls back to a standard Hugging Face stack using
Transformers + PEFT + TRL.

Main features
-------------
- Reads all configuration from ``train_settings.yml``.
- Formats ``yahma/alpaca-cleaned`` into a single text field for SFT.
- Supports LoRA/QLoRA style fine-tuning.
- Saves adapter weights, tokenizer, trainer state, and optional merged weights.
- Runs an optional post-training generation smoke test.

Expected files
--------------
- train_settings.yml : YAML configuration for model, data, training, and save options.

Typical usage
-------------
    python3 train_llm.py --config train_settings.yml

The companion shell script launches this inside a tmux session.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import random
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Tuple

import yaml

import torch
from datasets import DatasetDict, load_dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    TrainingArguments,
    set_seed,
)

LOGGER = logging.getLogger("train_llm")


def setup_logging() -> None:
    """Configure process-wide structured logging.

    The format is intentionally concise for tmux and file-based tailing on
    remote GPU boxes.
    """
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )


@dataclass
class RuntimeChoice:
    """Container describing the resolved training backend.

    Attributes
    ----------
    backend:
        Either ``unsloth`` or ``huggingface``.
    reason:
        Human-readable explanation for logs and reproducibility.
    """

    backend: str
    reason: str


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description="Fine-tune an LLM using Unsloth or Hugging Face.")
    parser.add_argument("--config", default="train_settings.yml", help="Path to YAML training config.")
    return parser.parse_args()


def load_config(config_path: str | os.PathLike[str]) -> Dict[str, Any]:
    """Load YAML configuration into a Python dictionary.

    Parameters
    ----------
    config_path:
        Path to the YAML file.

    Returns
    -------
    dict
        Parsed configuration.
    """
    with open(config_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def ensure_dirs(cfg: Dict[str, Any]) -> None:
    """Create output directories declared in the config."""
    Path(cfg["training"]["output_dir"]).mkdir(parents=True, exist_ok=True)
    Path(cfg["training"]["logging_dir"]).mkdir(parents=True, exist_ok=True)
    Path(cfg["save"]["final_model_dir"]).mkdir(parents=True, exist_ok=True)
    if cfg["save"].get("merged_model", False):
        Path(cfg["save"]["merged_model_dir"]).mkdir(parents=True, exist_ok=True)


def configure_environment(cfg: Dict[str, Any]) -> None:
    """Set reproducibility and cache-related environment variables."""
    runtime_cfg = cfg.get("runtime", {})
    if runtime_cfg.get("hf_home"):
        os.environ["HF_HOME"] = str(runtime_cfg["hf_home"])
    if runtime_cfg.get("transformers_cache"):
        os.environ["TRANSFORMERS_CACHE"] = str(runtime_cfg["transformers_cache"])
    seed = int(runtime_cfg.get("seed", cfg["training"].get("seed", 3407)))
    random.seed(seed)
    set_seed(seed)
    if torch.cuda.is_available() and cfg["training"].get("tf32", True):
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True




def configure_wandb(cfg: Dict[str, Any]) -> None:
    """Configure Weights & Biases environment variables from YAML settings.

    Transformers integrates with W&B when ``report_to="wandb"`` and the
    ``wandb`` package is installed. This helper keeps the Python entrypoint
    fully driven by the YAML file instead of requiring manual shell exports.
    """
    tcfg = cfg.get("training", {})
    report_to = str(tcfg.get("report_to", "none"))
    if report_to != "wandb":
        return

    os.environ.setdefault("WANDB_PROJECT", str(tcfg.get("wandb_project", "llm-finetune")))
    if tcfg.get("wandb_api_key") not in (None, "", "null"):
        os.environ["WANDB_API_KEY"] = str(tcfg.get("wandb_api_key"))
    if tcfg.get("wandb_entity") not in (None, "", "null"):
        os.environ.setdefault("WANDB_ENTITY", str(tcfg.get("wandb_entity")))
    if tcfg.get("run_name"):
        os.environ.setdefault("WANDB_NAME", str(tcfg.get("run_name")))
    if tcfg.get("wandb_log_model") not in (None, False, "false"):
        os.environ.setdefault("WANDB_LOG_MODEL", str(tcfg.get("wandb_log_model")))

    wandb_mode = str(tcfg.get("wandb_mode", "online"))
    if wandb_mode == "disabled":
        os.environ["WANDB_DISABLED"] = "true"
    else:
        os.environ.setdefault("WANDB_MODE", wandb_mode)


def resolve_precision(training_cfg: Dict[str, Any]) -> Tuple[bool, bool]:
    """Resolve fp16/bf16 settings from ``auto`` or explicit booleans.

    Returns
    -------
    tuple[bool, bool]
        ``(fp16, bf16)`` flags suitable for TrainingArguments / SFTConfig.
    """
    bf16_supported = torch.cuda.is_available() and torch.cuda.is_bf16_supported()

    bf16_cfg = training_cfg.get("bf16", "auto")
    fp16_cfg = training_cfg.get("fp16", "auto")

    if bf16_cfg == "auto":
        bf16 = bf16_supported
    else:
        bf16 = bool(bf16_cfg)

    if fp16_cfg == "auto":
        fp16 = not bf16
    else:
        fp16 = bool(fp16_cfg)

    if bf16:
        fp16 = False

    return fp16, bf16


def resolve_dtype(model_cfg: Dict[str, Any]) -> torch.dtype | None:
    """Map config dtype string to torch dtype.

    ``auto`` resolves to ``None`` so downstream libraries can infer the best
    available precision for the hardware.
    """
    name = str(model_cfg.get("dtype", "auto")).lower()
    if name == "auto":
        return None
    mapping = {
        "float16": torch.float16,
        "fp16": torch.float16,
        "bfloat16": torch.bfloat16,
        "bf16": torch.bfloat16,
        "float32": torch.float32,
        "fp32": torch.float32,
    }
    if name not in mapping:
        raise ValueError(f"Unsupported dtype setting: {name}")
    return mapping[name]


def build_prompt(example: Dict[str, Any], cfg: Dict[str, Any]) -> Dict[str, str]:
    """Format a raw Alpaca sample into prompt/completion fields for completion-only SFT.

    This function intentionally separates the prompt from the completion so the
    trainer can compute loss only on the response tokens. This is the preferred
    pattern for prompt-completion datasets in modern TRL workflows.

    Returns
    -------
    dict
        A mapping with ``prompt``, ``completion``, and rendered ``text`` fields.
    """
    dcfg = cfg["dataset"]
    pcfg = cfg["prompt"]
    instruction = (example.get(dcfg["instruction_field"], "") or "").strip()
    input_text = (example.get(dcfg["input_field"], "") or "").strip()
    output_text = (example.get(dcfg["output_field"], "") or "").strip()

    prompt_text = pcfg["prompt_template"].format(
        instruction=instruction,
        input=input_text if input_text else "",
    )
    completion_text = pcfg["response_prefix"] + output_text

    if pcfg.get("add_eos_token", True):
        completion_text = completion_text.rstrip() + "</s>"

    full_text = prompt_text + completion_text

    return {
        "prompt": prompt_text,
        "completion": completion_text,
        dcfg.get("text_field", "text"): full_text,
    }


def load_and_prepare_dataset(cfg: Dict[str, Any]):
    """Load the dataset, split train/eval, and create the text field for SFT."""
    dcfg = cfg["dataset"]
    LOGGER.info("Loading dataset: %s", dcfg["name"])
    ds = load_dataset(dcfg["name"], split=dcfg["split"])

    if dcfg.get("shuffle", True):
        ds = ds.shuffle(seed=int(dcfg.get("seed", 3407)))

    val_size = float(dcfg.get("val_size", 0.0))
    if val_size > 0:
        split_ds = ds.train_test_split(test_size=val_size, seed=int(dcfg.get("seed", 3407)))
        dataset = DatasetDict(train=split_ds["train"], test=split_ds["test"])
    else:
        dataset = DatasetDict(train=ds)

    num_proc = int(dcfg.get("num_proc", 1))
    text_field = dcfg.get("text_field", "text")
    dataset = dataset.map(
        lambda ex: build_prompt(ex, cfg),
        num_proc=num_proc,
        desc="Formatting alpaca samples into SFT prompts",
    )

    LOGGER.info("Prepared dataset with text field '%s'", text_field)
    return dataset


def choose_backend(cfg: Dict[str, Any]) -> RuntimeChoice:
    """Resolve the training backend with an Unsloth-first orchestration policy.

    Policy
    ------
    1. Always attempt Unsloth first unless the user explicitly disables it.
    2. Use Hugging Face only when Unsloth cannot be imported, the chosen model is
       not supported by the requested Unsloth loading path, or Unsloth runtime
       initialization/training fails.

    This matches an operational strategy where Unsloth is the primary engine on
    RunPod GPUs and Transformers acts only as a safety fallback.
    """
    use_unsloth = cfg["model"].get("use_unsloth", True)
    if use_unsloth is False or str(use_unsloth).lower() == "false":
        return RuntimeChoice("huggingface", "Config explicitly disabled Unsloth.")

    try:
        import unsloth  # noqa: F401
    except Exception as exc:
        return RuntimeChoice("huggingface", f"Unsloth import failed, so falling back to Hugging Face: {exc}")

    unsloth_model_name = str(cfg["model"].get("unsloth_model_name") or "").strip()
    base_model_name = str(cfg["model"].get("name") or "").strip()

    if unsloth_model_name:
        return RuntimeChoice("unsloth", f"Using Unsloth-first path with configured Unsloth model '{unsloth_model_name}'.")

    return RuntimeChoice("unsloth", f"Using Unsloth-first path with base model '{base_model_name}'. If unsupported at runtime, Hugging Face fallback will be used.")


def prepare_tokenizer(tokenizer, cfg: Dict[str, Any]):
    """Ensure tokenizer has a valid padding token for causal LM training."""
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"
    return tokenizer


def train_with_unsloth(cfg: Dict[str, Any], dataset) -> Tuple[Any, Any, Any, str]:
    """Train using Unsloth + TRL SFTTrainer.

    Returns
    -------
    tuple
        ``(model, tokenizer, trainer, backend_name)``.
    """
    from trl import SFTConfig, SFTTrainer
    from unsloth import FastLanguageModel

    model_cfg = cfg["model"]
    train_cfg = cfg["training"]
    lora_cfg = cfg["lora"]

    fp16, bf16 = resolve_precision(train_cfg)
    dtype = resolve_dtype(model_cfg)

    model_name = model_cfg.get("unsloth_model_name") or model_cfg["name"]
    LOGGER.info("Starting Unsloth backend with model: %s", model_name)

    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=model_name,
        max_seq_length=int(model_cfg["max_sequence_length"]),
        dtype=dtype,
        load_in_4bit=bool(model_cfg.get("load_in_4bit", True)),
    )
    tokenizer = prepare_tokenizer(tokenizer, cfg)

    if lora_cfg.get("enabled", True):
        gckpt = train_cfg.get("gradient_checkpointing_mode", "unsloth")
        if str(gckpt).lower() == "false":
            gckpt = False
        elif str(gckpt).lower() == "true":
            gckpt = True

        model = FastLanguageModel.get_peft_model(
            model,
            r=int(lora_cfg["r"]),
            target_modules=list(lora_cfg["target_modules"]),
            lora_alpha=int(lora_cfg["alpha"]),
            lora_dropout=float(lora_cfg["dropout"]),
            bias=str(lora_cfg["bias"]),
            use_gradient_checkpointing=gckpt,
            random_state=int(train_cfg.get("seed", 3407)),
            max_seq_length=int(model_cfg["max_sequence_length"]),
        )

    sft_args = SFTConfig(
        output_dir=train_cfg["output_dir"],
        overwrite_output_dir=bool(train_cfg.get("overwrite_output_dir", False)),
        per_device_train_batch_size=int(train_cfg["per_device_train_batch_size"]),
        per_device_eval_batch_size=int(train_cfg["per_device_eval_batch_size"]),
        gradient_accumulation_steps=int(train_cfg["gradient_accumulation_steps"]),
        learning_rate=float(train_cfg["learning_rate"]),
        num_train_epochs=float(train_cfg["num_train_epochs"]),
        max_steps=int(train_cfg["max_steps"]),
        logging_steps=int(train_cfg["logging_steps"]),
        save_steps=int(train_cfg["save_steps"]),
        save_strategy=str(train_cfg["save_strategy"]),
        save_total_limit=int(train_cfg["save_total_limit"]),
        eval_steps=int(train_cfg["eval_steps"]),
        eval_strategy=str(train_cfg["evaluation_strategy"]),
        warmup_ratio=float(train_cfg["warmup_ratio"]),
        lr_scheduler_type=str(train_cfg["lr_scheduler_type"]),
        optim=str(train_cfg["optim"]),
        weight_decay=float(train_cfg["weight_decay"]),
        max_grad_norm=float(train_cfg["max_grad_norm"]),
        bf16=bf16,
        fp16=fp16,
        logging_dir=train_cfg["logging_dir"],
        report_to=None if str(train_cfg.get("report_to", "none")) == "none" else train_cfg.get("report_to"),
        run_name=train_cfg.get("run_name"),
        seed=int(train_cfg.get("seed", 3407)),
        dataset_num_proc=int(cfg["dataset"].get("num_proc", 1)),
        max_length=int(model_cfg["max_sequence_length"]),
        packing=bool(train_cfg.get("packing", False)),
        completion_only_loss=bool(train_cfg.get("completion_only_loss", True)),
        group_by_length=bool(train_cfg.get("group_by_length", True)),
        dataloader_num_workers=int(train_cfg.get("dataloader_num_workers", 0)),
        remove_unused_columns=bool(train_cfg.get("remove_unused_columns", False)),
        ddp_find_unused_parameters=bool(train_cfg.get("ddp_find_unused_parameters", False)),
    )

    trainer = SFTTrainer(
        model=model,
        tokenizer=tokenizer,
        args=sft_args,
        train_dataset=dataset["train"],
        eval_dataset=dataset.get("test"),
        dataset_text_field=cfg["dataset"].get("text_field", "text"),
    )

    trainer.train()
    return model, tokenizer, trainer, "unsloth"


def train_with_huggingface(cfg: Dict[str, Any], dataset) -> Tuple[Any, Any, Any, str]:
    """Train using Transformers + PEFT + TRL as the fallback backend."""
    from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
    from trl import SFTConfig, SFTTrainer

    model_cfg = cfg["model"]
    train_cfg = cfg["training"]
    lora_cfg = cfg["lora"]

    fp16, bf16 = resolve_precision(train_cfg)
    dtype = resolve_dtype(model_cfg)

    quant_config = None
    if model_cfg.get("load_in_4bit", False) or model_cfg.get("load_in_8bit", False):
        quant_config = BitsAndBytesConfig(
            load_in_4bit=bool(model_cfg.get("load_in_4bit", False)),
            load_in_8bit=bool(model_cfg.get("load_in_8bit", False)),
            bnb_4bit_compute_dtype=torch.bfloat16 if bf16 else torch.float16,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True,
        )

    LOGGER.info("Starting Hugging Face backend with model: %s", model_cfg["name"])
    tokenizer = AutoTokenizer.from_pretrained(
        model_cfg["name"],
        trust_remote_code=bool(model_cfg.get("trust_remote_code", False)),
        revision=model_cfg.get("revision"),
        use_fast=True,
    )
    tokenizer = prepare_tokenizer(tokenizer, cfg)

    model = AutoModelForCausalLM.from_pretrained(
        model_cfg["name"],
        trust_remote_code=bool(model_cfg.get("trust_remote_code", False)),
        revision=model_cfg.get("revision"),
        torch_dtype=dtype,
        device_map=model_cfg.get("device_map", "auto"),
        quantization_config=quant_config,
        attn_implementation=model_cfg.get("attn_implementation", "sdpa"),
    )

    if lora_cfg.get("enabled", True):
        if quant_config is not None:
            model = prepare_model_for_kbit_training(
                model,
                use_gradient_checkpointing=bool(train_cfg.get("gradient_checkpointing", True)),
            )
        peft_config = LoraConfig(
            r=int(lora_cfg["r"]),
            lora_alpha=int(lora_cfg["alpha"]),
            lora_dropout=float(lora_cfg["dropout"]),
            bias=str(lora_cfg["bias"]),
            target_modules=list(lora_cfg["target_modules"]),
            task_type=str(lora_cfg["task_type"]),
        )
        model = get_peft_model(model, peft_config)

    sft_args = SFTConfig(
        output_dir=train_cfg["output_dir"],
        overwrite_output_dir=bool(train_cfg.get("overwrite_output_dir", False)),
        num_train_epochs=float(train_cfg["num_train_epochs"]),
        max_steps=int(train_cfg["max_steps"]),
        per_device_train_batch_size=int(train_cfg["per_device_train_batch_size"]),
        per_device_eval_batch_size=int(train_cfg["per_device_eval_batch_size"]),
        gradient_accumulation_steps=int(train_cfg["gradient_accumulation_steps"]),
        learning_rate=float(train_cfg["learning_rate"]),
        weight_decay=float(train_cfg["weight_decay"]),
        warmup_ratio=float(train_cfg["warmup_ratio"]),
        lr_scheduler_type=str(train_cfg["lr_scheduler_type"]),
        optim=str(train_cfg["optim"]),
        logging_steps=int(train_cfg["logging_steps"]),
        save_strategy=str(train_cfg["save_strategy"]),
        save_steps=int(train_cfg["save_steps"]),
        save_total_limit=int(train_cfg["save_total_limit"]),
        eval_strategy=str(train_cfg["evaluation_strategy"]),
        eval_steps=int(train_cfg["eval_steps"]),
        max_grad_norm=float(train_cfg["max_grad_norm"]),
        fp16=fp16,
        bf16=bf16,
        logging_dir=train_cfg["logging_dir"],
        report_to=None if str(train_cfg.get("report_to", "none")) == "none" else train_cfg.get("report_to"),
        run_name=train_cfg.get("run_name"),
        seed=int(train_cfg.get("seed", 3407)),
        group_by_length=bool(train_cfg.get("group_by_length", True)),
        dataloader_num_workers=int(train_cfg.get("dataloader_num_workers", 0)),
        remove_unused_columns=bool(train_cfg.get("remove_unused_columns", False)),
        ddp_find_unused_parameters=bool(train_cfg.get("ddp_find_unused_parameters", False)),
        max_length=int(model_cfg["max_sequence_length"]),
        packing=bool(train_cfg.get("packing", False)),
        completion_only_loss=bool(train_cfg.get("completion_only_loss", True)),
    )

    trainer = SFTTrainer(
        model=model,
        tokenizer=tokenizer,
        args=sft_args,
        train_dataset=dataset["train"],
        eval_dataset=dataset.get("test"),
    )

    trainer.train()
    return model, tokenizer, trainer, "huggingface"


def save_outputs(model, tokenizer, trainer, cfg: Dict[str, Any], backend_name: str) -> None:
    """Persist fine-tuned weights, tokenizer, metadata, and optional merged model."""
    save_cfg = cfg["save"]
    final_dir = Path(save_cfg["final_model_dir"])
    final_dir.mkdir(parents=True, exist_ok=True)

    LOGGER.info("Saving final model artifacts to %s", final_dir)
    print(f"[SAVE-INFO] Saving trainer model artifacts to: {final_dir}")
    trainer.save_model(str(final_dir))
    if save_cfg.get("save_tokenizer", True):
        print(f"[SAVE-INFO] Saving tokenizer to: {final_dir}")
        tokenizer.save_pretrained(str(final_dir))
    if save_cfg.get("save_training_state", True):
        print("[SAVE-INFO] Saving trainer state")
        trainer.save_state()

    metadata = {
        "backend": backend_name,
        "base_model": cfg["model"]["name"],
        "dataset": cfg["dataset"]["name"],
        "max_sequence_length": cfg["model"]["max_sequence_length"],
        "output_dir": str(final_dir),
    }
    with open(final_dir / "training_metadata.json", "w", encoding="utf-8") as f:
        print(f"[SAVE-INFO] Writing training metadata to: {final_dir / 'training_metadata.json'}")
        json.dump(metadata, f, indent=2)

    if save_cfg.get("merged_model", False):
        merged_dir = Path(save_cfg["merged_model_dir"])
        merged_dir.mkdir(parents=True, exist_ok=True)
        LOGGER.info("Attempting to save merged model to %s", merged_dir)
        try:
            if hasattr(model, "save_pretrained_merged"):
                print(f"[SAVE-INFO] Saving merged model to: {merged_dir}")
                model.save_pretrained_merged(
                    str(merged_dir),
                    tokenizer,
                    save_method="merged_16bit",
                )
            else:
                merged_model = model.merge_and_unload() if hasattr(model, "merge_and_unload") else model
                print(f"[SAVE-INFO] Saving merged model to: {merged_dir}")
                merged_model.save_pretrained(str(merged_dir))
                print(f"[SAVE-INFO] Saving tokenizer to: {merged_dir}")
                tokenizer.save_pretrained(str(merged_dir))
        except Exception as exc:
            LOGGER.warning("Merged model save skipped due to error: %s", exc)


@torch.inference_mode()
def run_smoke_test(model, tokenizer, cfg: Dict[str, Any], backend_name: str) -> None:
    """Run a minimal generation test after training to validate saved weights."""
    infer_cfg = cfg.get("inference", {})
    if not infer_cfg.get("run_after_train", False):
        return

    prompt = infer_cfg["prompt"]
    LOGGER.info("Running post-training generation smoke test.")

    if backend_name == "unsloth":
        try:
            from unsloth import FastLanguageModel
            FastLanguageModel.for_inference(model)
        except Exception as exc:
            LOGGER.warning("Unsloth inference optimization unavailable: %s", exc)

    device = model.device if hasattr(model, "device") else torch.device("cuda" if torch.cuda.is_available() else "cpu")
    inputs = tokenizer(prompt, return_tensors="pt").to(device)
    outputs = model.generate(
        **inputs,
        max_new_tokens=int(infer_cfg.get("max_new_tokens", 128)),
        do_sample=bool(infer_cfg.get("do_sample", True)),
        temperature=float(infer_cfg.get("temperature", 0.7)),
        top_p=float(infer_cfg.get("top_p", 0.9)),
        pad_token_id=tokenizer.eos_token_id,
    )
    text = tokenizer.decode(outputs[0], skip_special_tokens=True)
    LOGGER.info("Smoke test output preview:\n%s", text[:1000])


def main() -> None:
    """Program entry point."""
    setup_logging()
    args = parse_args()
    cfg = load_config(args.config)
    ensure_dirs(cfg)
    configure_environment(cfg)
    configure_wandb(cfg)

    backend_choice = choose_backend(cfg)
    LOGGER.info("Resolved backend: %s (%s)", backend_choice.backend, backend_choice.reason)

    dataset = load_and_prepare_dataset(cfg)

    try:
        if backend_choice.backend == "unsloth":
            model, tokenizer, trainer, backend_name = train_with_unsloth(cfg, dataset)
        else:
            model, tokenizer, trainer, backend_name = train_with_huggingface(cfg, dataset)
    except Exception as exc:
        if backend_choice.backend == "unsloth":
            LOGGER.exception("Unsloth path failed; retrying with Hugging Face fallback. Error: %s", exc)
            model, tokenizer, trainer, backend_name = train_with_huggingface(cfg, dataset)
        else:
            raise

    save_outputs(model, tokenizer, trainer, cfg, backend_name)
    run_smoke_test(model, tokenizer, cfg, backend_name)
    LOGGER.info("Training completed successfully using backend: %s", backend_name)


if __name__ == "__main__":
    main()
