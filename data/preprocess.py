"""
Preprocess MedQA and PubMedQA into instruction-tuning format for Llama 3.1 70B.

Output format per example:
{
    "instruction": "Answer the following medical question...",
    "input": "Question + options",
    "output": "Answer letter + explanation",
    "source": "medqa" | "pubmedqa"
}

Saved to $HOME/W/data/processed/
"""

import json
import random
from pathlib import Path
import os

RAW_DIR = Path(os.environ.get("HOME", ".")) / "W" / "data" / "raw"
PROCESSED_DIR = Path(os.environ.get("HOME", ".")) / "W" / "data" / "processed"

MEDQA_INSTRUCTION = (
    "You are a medical expert. Answer the following multiple-choice medical question "
    "by selecting the single best answer. Provide your answer as the letter (A, B, C, or D) "
    "followed by a brief explanation."
)

PUBMEDQA_INSTRUCTION = (
    "You are a medical expert. Based on the provided research context, "
    "answer the following biomedical question with 'yes', 'no', or 'maybe', "
    "followed by a brief justification."
)

OPTION_KEYS = ["A", "B", "C", "D", "E"]


def format_medqa_example(example: dict) -> dict:
    options = example.get("options", {})
    options_text = "\n".join(
        f"{k}: {v}" for k, v in options.items() if v
    )
    answer_key = example.get("answer_idx", example.get("answer", ""))

    input_text = f"Question: {example['question']}\n\nOptions:\n{options_text}"
    output_text = f"Answer: {answer_key}\n\nExplanation: The correct answer is {answer_key}: {options.get(answer_key, '')}."

    return {
        "instruction": MEDQA_INSTRUCTION,
        "input": input_text,
        "output": output_text,
        "source": "medqa",
        "answer": answer_key,
    }


def format_pubmedqa_example(example: dict) -> dict:
    contexts = example.get("context", {})
    if isinstance(contexts, dict):
        context_text = "\n\n".join(contexts.get("contexts", []))
    else:
        context_text = str(contexts)

    answer = example.get("final_decision", "")
    long_answer = example.get("long_answer", "")

    input_text = f"Context:\n{context_text}\n\nQuestion: {example['question']}"
    output_text = f"Answer: {answer}\n\nJustification: {long_answer}"

    return {
        "instruction": PUBMEDQA_INSTRUCTION,
        "input": input_text,
        "output": output_text,
        "source": "pubmedqa",
        "answer": answer,
    }


def load_jsonl(path: Path) -> list:
    examples = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                examples.append(json.loads(line))
    return examples


def save_jsonl(examples: list, path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        for ex in examples:
            f.write(json.dumps(ex) + "\n")
    print(f"  Saved {len(examples)} examples → {path}")


def process_medqa():
    print("Processing MedQA...")
    medqa_dir = RAW_DIR / "medqa"
    splits = {}

    for split_file in sorted(medqa_dir.glob("*.jsonl")):
        split_name = split_file.stem
        raw = load_jsonl(split_file)
        processed = [format_medqa_example(ex) for ex in raw]
        splits[split_name] = processed
        save_jsonl(processed, PROCESSED_DIR / "medqa" / f"{split_name}.jsonl")

    return splits


def process_pubmedqa():
    print("Processing PubMedQA...")
    pubmedqa_dir = RAW_DIR / "pubmedqa"
    splits = {}

    for split_file in sorted(pubmedqa_dir.glob("*.jsonl")):
        split_name = split_file.stem
        raw = load_jsonl(split_file)
        processed = [format_pubmedqa_example(ex) for ex in raw]
        splits[split_name] = processed
        save_jsonl(processed, PROCESSED_DIR / "pubmedqa" / f"{split_name}.jsonl")

    return splits


def merge_and_shuffle(medqa_splits: dict, pubmedqa_splits: dict):
    """Merge both datasets into combined train/val/test splits."""
    print("Merging datasets...")

    random.seed(42)

    # PubMedQA: split manually (80/10/10)
    pubmedqa_all = []
    for examples in pubmedqa_splits.values():
        pubmedqa_all.extend(examples)
    random.shuffle(pubmedqa_all)
    n = len(pubmedqa_all)
    pubmedqa_train = pubmedqa_all[: int(0.8 * n)]
    pubmedqa_val   = pubmedqa_all[int(0.8 * n) : int(0.9 * n)]
    pubmedqa_test  = pubmedqa_all[int(0.9 * n) :]

    # MedQA: carve 10% of train as validation (no official dev split)
    medqa_train_all = medqa_splits.get("train", [])
    random.shuffle(medqa_train_all)
    n_med = len(medqa_train_all)
    split_idx = int(0.9 * n_med)
    medqa_train = medqa_train_all[:split_idx]
    medqa_val   = medqa_train_all[split_idx:]

    train = medqa_train + pubmedqa_train
    val   = medqa_val  + pubmedqa_val
    test  = medqa_splits.get("test", []) + pubmedqa_test

    random.shuffle(train)

    save_jsonl(train, PROCESSED_DIR / "combined" / "train.jsonl")
    save_jsonl(val,   PROCESSED_DIR / "combined" / "val.jsonl")
    save_jsonl(test,  PROCESSED_DIR / "combined" / "test.jsonl")

    print(f"\nCombined splits: train={len(train)}, val={len(val)}, test={len(test)}")


if __name__ == "__main__":
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

    medqa_splits = process_medqa()
    pubmedqa_splits = process_pubmedqa()
    merge_and_shuffle(medqa_splits, pubmedqa_splits)

    print("\nPreprocessing complete.")
