"""
Evaluate fine-tuned Llama 3.3 70B + RAG on MedQA-USMLE test set.

Usage:
    python evaluation/evaluate_rag.py \
        --model_path ~/W/checkpoints/medrag_v2/final \
        --vectorstore ~/W/vectorstore \
        --test_data ~/W/data/processed/medqa/test.jsonl \
        --output evaluation/results/rag.json
"""

import os
import json
import argparse
import re
import sys
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from peft import PeftModel
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).parent.parent))
from rag.pipeline import MedRAGPipeline


def load_jsonl(path: str) -> list:
    examples = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                examples.append(json.loads(line))
    return examples


def extract_answer_letter(text: str) -> str:
    text = text.strip()
    m = re.search(r'Answer:\s*([A-D])', text, re.IGNORECASE)
    if m:
        return m.group(1).upper()
    m = re.match(r'^([A-D])[\.:\)]\s', text, re.IGNORECASE)
    if m:
        return m.group(1).upper()
    m = re.search(r'\b([A-D])\b', text, re.IGNORECASE)
    if m:
        return m.group(1).upper()
    return ""


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--base_model", default="meta-llama/Llama-3.3-70B-Instruct")
    parser.add_argument("--model_path", default=None, help="LoRA adapter path (omit for base model)")
    parser.add_argument("--no_lora", action="store_true")
    parser.add_argument("--vectorstore", required=True, help="Path to vectorstore directory")
    parser.add_argument("--top_k", type=int, default=3, help="Number of chunks to retrieve")
    parser.add_argument("--test_data", required=True)
    parser.add_argument("--output", default="evaluation/results/rag_results.json")
    parser.add_argument("--max_new_tokens", type=int, default=256)
    parser.add_argument("--batch_size", type=int, default=2)
    parser.add_argument("--limit", type=int, default=None)
    args = parser.parse_args()

    if not args.no_lora and args.model_path is None:
        parser.error("Either --model_path or --no_lora is required")

    Path(args.output).parent.mkdir(parents=True, exist_ok=True)

    # Load RAG pipeline first (uses GPU for embeddings — share device with LLM)
    rag = MedRAGPipeline(vectorstore_dir=args.vectorstore, top_k=args.top_k)

    print(f"Loading tokenizer from {args.base_model}...")
    tokenizer = AutoTokenizer.from_pretrained(args.base_model, trust_remote_code=False)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"

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
        print("Using base model without LoRA.")
        model = base_model
    else:
        print(f"Loading LoRA adapters from {args.model_path}...")
        model = PeftModel.from_pretrained(base_model, args.model_path)
    model.eval()

    print(f"Loading test data from {args.test_data}...")
    examples = load_jsonl(args.test_data)
    medqa_examples = [ex for ex in examples if ex.get("source") == "medqa"]
    if args.limit:
        medqa_examples = medqa_examples[: args.limit]
    print(f"Evaluating on {len(medqa_examples)} MedQA examples with RAG (top_k={args.top_k})...")

    results = []
    correct = 0
    total = 0
    skipped = 0

    for i in tqdm(range(0, len(medqa_examples), args.batch_size)):
        batch = medqa_examples[i : i + args.batch_size]

        # Build RAG-augmented prompts (retrieval happens here)
        prompts = [rag.build_prompt(ex, tokenizer) for ex in batch]
        inputs = tokenizer(
            prompts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=2048,
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
                "input": ex["input"][:200],
                "ground_truth": ground_truth,
                "predicted": predicted,
                "generated": generated[:400],
                "correct": is_correct,
            })

        if (i // args.batch_size) % 10 == 0 and total > 0:
            print(f"  Step {i+args.batch_size}/{len(medqa_examples)} — running accuracy: {correct/total:.3f}")

    accuracy = correct / total if total > 0 else 0

    summary = {
        "model_path": "base_model_no_lora" if args.no_lora else args.model_path,
        "vectorstore": args.vectorstore,
        "top_k": args.top_k,
        "test_file": args.test_data,
        "total_examples": len(medqa_examples),
        "evaluated": total,
        "skipped": skipped,
        "correct": correct,
        "accuracy": accuracy,
        "gpt4_baseline_xiong2024": 0.8397,
        "delta_vs_gpt4": accuracy - 0.8397,
    }

    print("\n" + "=" * 50)
    print(f"ACCURACY (RAG top_k={args.top_k}): {accuracy:.4f} ({correct}/{total})")
    print(f"GPT-4 baseline (Xiong et al. ACL 2024): 0.8397")
    print(f"Delta: {accuracy - 0.8397:+.4f}")
    print(f"Skipped: {skipped}")
    print("=" * 50)

    with open(args.output, "w") as f:
        json.dump({"summary": summary, "results": results}, f, indent=2)
    print(f"\nResultados guardados en {args.output}")


if __name__ == "__main__":
    main()
