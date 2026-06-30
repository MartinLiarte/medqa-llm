"""
Model inference engine: loads Llama 3.3 70B in 4-bit + optional LoRA adapters.
Shared across the FastAPI app (singleton pattern via lifespan).
"""

import re
import sys
from pathlib import Path

import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

sys.path.insert(0, str(Path(__file__).parent.parent))
from rag.pipeline import MedRAGPipeline

SYSTEM_PROMPT = (
    "You are a medical expert. Answer the following multiple-choice medical question "
    "by selecting the single best answer. Provide your answer as the letter (A, B, C, or D) "
    "followed by a brief explanation."
)


class MedRAGInferenceEngine:
    def __init__(
        self,
        base_model: str,
        adapter_path: str | None = None,
        vectorstore_path: str | None = None,
        top_k: int = 3,
    ):
        self.top_k = top_k

        print(f"Loading tokenizer from {base_model}...")
        self.tokenizer = AutoTokenizer.from_pretrained(base_model, trust_remote_code=False)
        self.tokenizer.pad_token = self.tokenizer.eos_token
        self.tokenizer.padding_side = "left"

        print("Loading model in 4-bit NF4...")
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_use_double_quant=True,
        )
        base = AutoModelForCausalLM.from_pretrained(
            base_model,
            quantization_config=bnb_config,
            device_map="auto",
            dtype=torch.bfloat16,
            trust_remote_code=False,
        )

        if adapter_path:
            print(f"Loading LoRA adapters from {adapter_path}...")
            self.model = PeftModel.from_pretrained(base, adapter_path)
        else:
            self.model = base
        self.model.eval()

        self.rag = None
        if vectorstore_path:
            print(f"Loading RAG pipeline from {vectorstore_path}...")
            self.rag = MedRAGPipeline(vectorstore_dir=vectorstore_path, top_k=top_k)

        print("Inference engine ready.")

    def _build_prompt(self, question: str, use_rag: bool) -> str:
        if use_rag and self.rag:
            example = {"input": question}
            return self.rag.build_prompt(example, self.tokenizer)

        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": question},
        ]
        return self.tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )

    @torch.no_grad()
    def generate(
        self,
        question: str,
        use_rag: bool = False,
        max_new_tokens: int = 256,
    ) -> dict:
        if use_rag and not self.rag:
            raise ValueError("RAG requested but vectorstore not loaded.")

        prompt = self._build_prompt(question, use_rag)
        inputs = self.tokenizer(
            prompt, return_tensors="pt", truncation=True, max_length=2048
        ).to(self.model.device)

        output = self.model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            temperature=1.0,
            pad_token_id=self.tokenizer.eos_token_id,
        )

        input_len = inputs["input_ids"].shape[1]
        generated = self.tokenizer.decode(output[0][input_len:], skip_special_tokens=True)
        answer = self._extract_answer(generated)

        return {"answer": answer, "explanation": generated, "rag_used": use_rag}

    def _extract_answer(self, text: str) -> str:
        m = re.search(r'Answer:\s*([A-D])', text, re.IGNORECASE)
        if m:
            return m.group(1).upper()
        m = re.match(r'^([A-D])[\.:\)]\s', text.strip(), re.IGNORECASE)
        if m:
            return m.group(1).upper()
        m = re.search(r'\b([A-D])\b', text, re.IGNORECASE)
        if m:
            return m.group(1).upper()
        return ""
