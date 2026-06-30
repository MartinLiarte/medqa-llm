"""
Preprocess MedQA into instruction-tuning format for Llama 3.3 70B.

Output format per example:
{
    "instruction": "Answer the following medical question...",
    "input": "Question + options",
    "output": "Answer letter + explanation",
    "source": "medqa",
    "answer": "A" | "B" | "C" | "D"
}

Saved to $HOME/W/data/processed/medqa/
"""

import json
from pathlib import Path
import os

RAW_DIR = Path(os.environ.get("HOME", ".")) / "W" / "data" / "raw"
PROCESSED_DIR = Path(os.environ.get("HOME", ".")) / "W" / "data" / "processed"

MEDQA_INSTRUCTION = (
    "You are a medical expert. Answer the following multiple-choice medical question "
    "by selecting the single best answer. Provide your answer as the letter (A, B, C, or D) "
    "followed by a brief explanation."
)


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

    for split_file in sorted(medqa_dir.glob("*.jsonl")):
        split_name = split_file.stem
        raw = load_jsonl(split_file)
        processed = [format_medqa_example(ex) for ex in raw]
        save_jsonl(processed, PROCESSED_DIR / "medqa" / f"{split_name}.jsonl")
        print(f"  {split_name}: {len(processed)} examples")


if __name__ == "__main__":
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    process_medqa()
    print("\nPreprocessing complete.")
    print(f"Output: {PROCESSED_DIR / 'medqa'}/")
