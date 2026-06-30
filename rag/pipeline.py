"""
RAG pipeline: retrieve relevant medical text chunks given a question,
then augment the prompt before generation.

Usage:
    from rag.pipeline import MedRAGPipeline
    rag = MedRAGPipeline(vectorstore_dir="~/W/vectorstore")
    augmented_prompt = rag.build_prompt(question_text, tokenizer)
"""

import json
from pathlib import Path

import faiss
import numpy as np
import torch
from sentence_transformers import SentenceTransformer

EMBEDDING_MODEL = "BAAI/bge-large-en-v1.5"
DEFAULT_TOP_K = 3
MAX_CONTEXT_CHARS = 1500  # truncate each chunk to keep prompt manageable


class MedRAGPipeline:
    def __init__(self, vectorstore_dir: str, top_k: int = DEFAULT_TOP_K, device: str = None):
        self.top_k = top_k
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")

        vs_dir = Path(vectorstore_dir).expanduser()
        index_path = vs_dir / "faiss_index.bin"
        docs_path = vs_dir / "documents.jsonl"

        print(f"Loading FAISS index from {index_path}...")
        self.index = faiss.read_index(str(index_path))

        print(f"Loading document metadata from {docs_path}...")
        self.documents = []
        with open(docs_path) as f:
            for line in f:
                self.documents.append(json.loads(line.strip()))

        print(f"Loading embedding model ({EMBEDDING_MODEL})...")
        self.embedder = SentenceTransformer(EMBEDDING_MODEL, device=self.device)
        print(f"RAG pipeline ready: {self.index.ntotal} indexed chunks, top_k={top_k}")

    def retrieve(self, query: str) -> list[dict]:
        query_emb = self.embedder.encode(
            [query], normalize_embeddings=True, show_progress_bar=False
        ).astype("float32")
        scores, indices = self.index.search(query_emb, self.top_k)
        results = []
        for score, idx in zip(scores[0], indices[0]):
            if idx < 0:
                continue
            doc = self.documents[idx].copy()
            doc["score"] = float(score)
            results.append(doc)
        return results

    def build_prompt(self, example: dict, tokenizer) -> str:
        """Build a RAG-augmented chat prompt for a MedQA example."""
        query = example["input"]
        chunks = self.retrieve(query)

        context_parts = []
        for i, chunk in enumerate(chunks, 1):
            text = chunk["text"][:MAX_CONTEXT_CHARS]
            title = chunk.get("title", "")
            header = f"[{i}] {title}" if title else f"[{i}]"
            context_parts.append(f"{header}\n{text}")

        context_block = "\n\n".join(context_parts)

        system = (
            "You are a medical expert. Use the following reference passages to help answer "
            "the multiple-choice question. Select the single best answer (A, B, C, or D) "
            "and provide a brief explanation.\n\n"
            f"Reference passages:\n{context_block}"
        )

        messages = [
            {"role": "system", "content": system},
            {"role": "user", "content": example["input"]},
        ]
        return tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
