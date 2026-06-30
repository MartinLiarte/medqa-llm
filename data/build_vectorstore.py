"""
Build a FAISS vector store from medical textbook chunks (MedRAG/textbooks).
Uses BAAI/bge-large-en-v1.5 embeddings (1024-dim).

Saves:
  $HOME/W/vectorstore/faiss_index.bin   — FAISS index
  $HOME/W/vectorstore/documents.jsonl   — parallel document metadata

Usage:
    python data/build_vectorstore.py [--corpus textbooks] [--batch_size 256]
"""

import argparse
import json
import os
from pathlib import Path

import faiss
import numpy as np
import torch
from datasets import load_dataset
from sentence_transformers import SentenceTransformer
from tqdm import tqdm

VECTORSTORE_DIR = Path(os.environ.get("HOME", ".")) / "W" / "vectorstore"
EMBEDDING_MODEL = "BAAI/bge-large-en-v1.5"


def load_corpus(corpus: str) -> list[dict]:
    docs = []

    if corpus in ("textbooks", "all"):
        print("Loading MedRAG/textbooks...")
        ds = load_dataset("MedRAG/textbooks", split="train")
        for ex in tqdm(ds, desc="textbooks"):
            docs.append({
                "text": ex["content"],
                "title": ex.get("title", ""),
                "source": "textbooks",
            })
        print(f"  → {len(docs)} chunks from textbooks")

    if corpus in ("statpearls", "all"):
        print("Loading MedRAG/statpearls...")
        n_before = len(docs)
        ds = load_dataset("MedRAG/statpearls", split="train")
        for ex in tqdm(ds, desc="statpearls"):
            docs.append({
                "text": ex["content"],
                "title": ex.get("title", ""),
                "source": "statpearls",
            })
        print(f"  → {len(docs) - n_before} chunks from statpearls")

    print(f"Total documents: {len(docs)}")
    return docs


def build_index(docs: list[dict], batch_size: int):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Embedding device: {device}")
    print(f"Loading {EMBEDDING_MODEL}...")
    model = SentenceTransformer(EMBEDDING_MODEL, device=device)

    texts = [d["text"] for d in docs]
    all_embeddings = []
    for i in tqdm(range(0, len(texts), batch_size), desc="Encoding"):
        batch = texts[i : i + batch_size]
        emb = model.encode(batch, normalize_embeddings=True, show_progress_bar=False)
        all_embeddings.append(emb)

    matrix = np.vstack(all_embeddings).astype("float32")
    dim = matrix.shape[1]
    print(f"Embedding dim: {dim}, total vectors: {len(matrix)}")

    if len(docs) < 200_000:
        index = faiss.IndexFlatIP(dim)
    else:
        nlist = 512
        quantizer = faiss.IndexFlatIP(dim)
        index = faiss.IndexIVFFlat(quantizer, dim, nlist, faiss.METRIC_INNER_PRODUCT)
        index.train(matrix)

    index.add(matrix)
    print(f"Index built: {index.ntotal} vectors")

    VECTORSTORE_DIR.mkdir(parents=True, exist_ok=True)

    faiss.write_index(index, str(VECTORSTORE_DIR / "faiss_index.bin"))
    print(f"Index saved → {VECTORSTORE_DIR / 'faiss_index.bin'}")

    with open(VECTORSTORE_DIR / "documents.jsonl", "w") as f:
        for doc in docs:
            f.write(json.dumps(doc) + "\n")
    print(f"Metadata saved → {VECTORSTORE_DIR / 'documents.jsonl'}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--corpus", default="textbooks",
                        choices=["textbooks", "statpearls", "all"],
                        help="Which corpus to index")
    parser.add_argument("--batch_size", type=int, default=256)
    args = parser.parse_args()

    docs = load_corpus(args.corpus)
    build_index(docs, args.batch_size)
    print("\nVector store build complete.")


if __name__ == "__main__":
    main()
