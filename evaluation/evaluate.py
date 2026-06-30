"""
Evaluate fine-tuned Llama 3.3 70B on MedQA-USMLE test set.

Usage:
    python evaluation/evaluate.py \
        --model_path ~/W/checkpoints/medrag/final \
        --test_data ~/W/data/processed/combined/test.jsonl \
        --output evaluation/results/run1.json
"""

import os
import json
import argparse
import re
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from peft import PeftModel
from tqdm import tqdm


SYSTEM_PROMPT = (
    "You are a medical expert. Answer the following multiple-choice medical question "
    "by selecting the single best answer. Provide your answer as the letter (A, B, C, or D) "
    "followed by a brief explanation."
)


def load_jsonl(path: str) -> list:
    examples = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                examples.append(json.loads(line))
    return examples


def extract_answer_letter(text: str) -> str:
    """Extract answer letter (A/B/C/D) from model output."""
    text = text.strip()

    # Match "Answer: X" pattern first
    m = re.search(r'Answer:\s*([A-D])', text, re.IGNORECASE)
    if m:
        return m.group(1).upper()

    # Match standalone letter at start
    m = re.match(r'^([A-D])[\.:\)]\s', text, re.IGNORECASE)
    if m:
        return m.group(1).upper()

    # Match any first A/B/C/D in the text
    m = re.search(r'\b([A-D])\b', text, re.IGNORECASE)
    if m:
        return m.group(1).upper()

    return ""


def build_prompt(example: dict, tokenizer) -> str:
    messages = [
        {"role": "system", "content": example["instruction"]},
        {"role": "user", "content": example["input"]},
    ]
    return tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--base_model", default="meta-llama/Llama-3.3-70B-Instruct")
    parser.add_argument("--model_path", default=None, help="Path to LoRA adapter directory (omit with --no_lora)")
    parser.add_argument("--no_lora", action="store_true", help="Evaluate base model without any LoRA adapters")
    parser.add_argument("--test_data", required=True)
    parser.add_argument("--output", default="evaluation/results/results.json")
    parser.add_argument("--max_new_tokens", type=int, default=128)
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--limit", type=int, default=None, help="Evaluate only N examples (for testing)")
    args = parser.parse_args()

    if not args.no_lora and args.model_path is None:
        parser.error("Either --model_path or --no_lora is required")

    Path(args.output).parent.mkdir(parents=True, exist_ok=True)

    print(f"Loading tokenizer from {args.base_model}...")
    tokenizer = AutoTokenizer.from_pretrained(args.base_model, trust_remote_code=False)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"  # left-pad for generation

    print("Loading base model in 4-bit...")
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=True,
    )
    base_model = AutoModelForCausalLM.from_pretrained(
        args.base_model,
        quantization_config=bnb_config,
        device_map="auto",
        dtype=torch.bfloat16,
        trust_remote_code=False,
    )

    if args.no_lora:
        print("Evaluating base model without LoRA adapters.")
        model = base_model
    else:
        print(f"Loading LoRA adapters from {args.model_path}...")
        model = PeftModel.from_pretrained(base_model, args.model_path)
    model.eval()

    print(f"Loading test data from {args.test_data}...")
    examples = load_jsonl(args.test_data)

    # Filter only MedQA examples for accuracy comparison with GPT-4o
    medqa_examples = [ex for ex in examples if ex.get("source") == "medqa"]
    all_examples = medqa_examples  # primary evaluation on MedQA

    if args.limit:
        all_examples = all_examples[: args.limit]

    print(f"Evaluating on {len(all_examples)} MedQA examples...")

    results = []
    correct = 0
    total = 0
    skipped = 0

    for i in tqdm(range(0, len(all_examples), args.batch_size)):
        batch = all_examples[i : i + args.batch_size]

        prompts = [build_prompt(ex, tokenizer) for ex in batch]
        inputs = tokenizer(
            prompts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=1024,
        ).to(model.device)

        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=args.max_new_tokens,
                do_sample=False,
                temperature=1.0,
                pad_token_id=tokenizer.eos_token_id,
            )

        for j, (ex, output) in enumerate(zip(batch, outputs)):
            input_len = inputs["input_ids"].shape[1]
            generated = tokenizer.decode(output[input_len:], skip_special_tokens=True)
            predicted = extract_answer_letter(generated)
            ground_truth = ex.get("answer", "").strip().upper()

            if not predicted:
                skipped += 1

            is_correct = predicted == ground_truth
            if predicted:
                correct += int(is_correct)
                total += 1

            results.append({
                "input": ex["input"][:200],  # truncate for readability
                "ground_truth": ground_truth,
                "predicted": predicted,
                "generated": generated[:300],
                "correct": is_correct,
                "source": ex.get("source", ""),
            })

        if (i // args.batch_size) % 10 == 0:
            current_acc = correct / total if total > 0 else 0
            print(f"  Step {i+args.batch_size}/{len(all_examples)} — running accuracy: {current_acc:.3f}")

    accuracy = correct / total if total > 0 else 0

    summary = {
        "model_path": "base_model_no_lora" if args.no_lora else args.model_path,
        "test_file": args.test_data,
        "total_examples": len(all_examples),
        "evaluated": total,
        "skipped": skipped,
        "correct": correct,
        "accuracy": accuracy,
        "gpt4o_baseline": 0.87,
        "delta_vs_gpt4o": accuracy - 0.87,
    }

    print("\n" + "="*50)
    print(f"ACCURACY: {accuracy:.4f} ({correct}/{total})")
    print(f"GPT-4o baseline: 0.8700")
    print(f"Delta: {accuracy - 0.87:+.4f}")
    print(f"Skipped (no answer extracted): {skipped}")
    print("="*50)

    output_data = {"summary": summary, "results": results}
    with open(args.output, "w") as f:
        json.dump(output_data, f, indent=2)
    print(f"\nResultados guardados en {args.output}")


if __name__ == "__main__":
    main()
