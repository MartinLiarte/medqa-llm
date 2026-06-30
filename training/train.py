"""
Fine-tune Llama 3.1 70B Instruct with QLoRA on MedQA + PubMedQA.

Usage (via SLURM job.sh):
    torchrun --nproc_per_node=4 train.py --config config.yaml
"""

import os
import sys
import json
import argparse
from pathlib import Path

import yaml
import torch
from datasets import Dataset, load_dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
)
from peft import LoraConfig, get_peft_model
from trl import SFTTrainer, SFTConfig


def load_config(path: str) -> dict:
    with open(path) as f:
        config = yaml.safe_load(f)
    # Expand env vars in paths
    for key, val in config.get("paths", {}).items():
        config["paths"][key] = os.path.expandvars(val)
    return config


def load_jsonl(path: str) -> list:
    examples = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                examples.append(json.loads(line))
    return examples


def format_prompt(example: dict, tokenizer) -> str:
    """Format as Llama 3.1 chat template."""
    messages = [
        {"role": "system", "content": example["instruction"]},
        {"role": "user", "content": example["input"]},
        {"role": "assistant", "content": example["output"]},
    ]
    return tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=False,
    )


def build_dataset(data_dir: str, tokenizer, max_seq_length: int) -> tuple:
    train_examples = load_jsonl(os.path.join(data_dir, "train.jsonl"))
    val_examples = load_jsonl(os.path.join(data_dir, "val.jsonl"))

    def to_hf_dataset(examples):
        texts = [format_prompt(ex, tokenizer) for ex in examples]
        return Dataset.from_dict({"text": texts})

    train_dataset = to_hf_dataset(train_examples)
    val_dataset = to_hf_dataset(val_examples)

    print(f"Train: {len(train_dataset)} | Val: {len(val_dataset)}")
    return train_dataset, val_dataset


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="config.yaml")
    parser.add_argument("--deepspeed", default=None)
    args = parser.parse_args()

    cfg = load_config(args.config)

    local_rank = int(os.environ.get("LOCAL_RANK", 0))

    # W&B config via env vars — the Trainer initializes wandb automatically
    os.environ["WANDB_PROJECT"] = cfg["wandb"]["project"]
    os.environ["WANDB_RUN_ID"] = cfg["training"]["run_name"]
    if cfg["wandb"].get("entity"):
        os.environ["WANDB_ENTITY"] = cfg["wandb"]["entity"]

    # BitsAndBytes 4-bit config
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=cfg["model"]["load_in_4bit"],
        bnb_4bit_quant_type=cfg["model"]["bnb_4bit_quant_type"],
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=cfg["model"]["bnb_4bit_use_double_quant"],
    )

    print(f"Loading base model: {cfg['model']['base_model']}")
    model = AutoModelForCausalLM.from_pretrained(
        cfg["model"]["base_model"],
        quantization_config=bnb_config,
        device_map={"": local_rank},
        dtype=torch.bfloat16,
        trust_remote_code=False,
    )
    model.enable_input_require_grads()

    tokenizer = AutoTokenizer.from_pretrained(
        cfg["model"]["base_model"],
        trust_remote_code=False,
    )
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"

    # LoRA
    lora_cfg = cfg["lora"]
    peft_config = LoraConfig(
        r=lora_cfg["r"],
        lora_alpha=lora_cfg["lora_alpha"],
        lora_dropout=lora_cfg["lora_dropout"],
        target_modules=lora_cfg["target_modules"],
        bias=lora_cfg["bias"],
        task_type=lora_cfg["task_type"],
    )

    train_dataset, val_dataset = build_dataset(
        cfg["paths"]["data_dir"],
        tokenizer,
        cfg["training"]["max_seq_length"],
    )

    t = cfg["training"]
    # SFTConfig extends TrainingArguments with SFT-specific args (TRL 1.7+)
    training_args = SFTConfig(
        output_dir=cfg["paths"]["output_dir"],
        deepspeed=args.deepspeed,
        num_train_epochs=t["num_train_epochs"],
        per_device_train_batch_size=t["per_device_train_batch_size"],
        per_device_eval_batch_size=t["per_device_eval_batch_size"],
        gradient_accumulation_steps=t["gradient_accumulation_steps"],
        learning_rate=float(t["learning_rate"]),
        lr_scheduler_type=t["lr_scheduler_type"],
        warmup_steps=int(t["warmup_ratio"] * t["num_train_epochs"] * 9960 // (t["per_device_train_batch_size"] * t["gradient_accumulation_steps"] * 4)),
        fp16=t["fp16"],
        bf16=t["bf16"],
        gradient_checkpointing=t["gradient_checkpointing"],
        logging_steps=t["logging_steps"],
        eval_strategy="steps",
        eval_steps=t["eval_steps"],
        save_strategy="steps",
        save_steps=t["save_steps"],
        save_total_limit=t["save_total_limit"],
        load_best_model_at_end=t["load_best_model_at_end"],
        metric_for_best_model=t["metric_for_best_model"],
        report_to=t["report_to"],
        run_name=t["run_name"],
        ddp_find_unused_parameters=False,
        dataloader_num_workers=4,
        max_length=t["max_seq_length"],
        dataset_text_field="text",
    )

    trainer = SFTTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        peft_config=peft_config,
        processing_class=tokenizer,
    )

    print("Starting training...")
    trainer.train()

    if local_rank == 0:
        print("Saving final model...")
        trainer.save_model(cfg["paths"]["output_dir"] + "/final")
        tokenizer.save_pretrained(cfg["paths"]["output_dir"] + "/final")


if __name__ == "__main__":
    main()
