# MedRAG — Medical QA with Fine-tuned LLM + RAG

> Fine-tuning Llama 3.3 70B on USMLE medical exam data with QLoRA + RAG over medical textbooks — approaches GPT-4 performance using only open-source models and public data.

---

## Overview

This project fine-tunes **Llama 3.3 70B** using **QLoRA** on the MedQA-USMLE dataset and augments inference with a **FAISS-based RAG pipeline** over 18 medical textbooks. Each component is evaluated in isolation to produce a rigorous ablation study.

**Hypothesis**: An open-source 70B model with QLoRA fine-tuning + RAG can approach GPT-4 performance on MedQA-US using only public models and data.

---

## Results

Evaluated on the MedQA-US test set (1,273 questions, 4-option USMLE format). GPT-4 reference from Xiong et al. (ACL 2024).

| Model | MedQA-US Accuracy |
|---|---|
| GPT-4 — Xiong et al. (ACL 2024) | 83.97% |
| **Llama 3.3 70B + QLoRA + RAG (ours)** | **80.99%** |
| Llama 3.3 70B + QLoRA fine-tuning only | 78.48% |
| GPT-3.5 — Xiong et al. (ACL 2024) | 65.04% |
| Llama 3.3 70B base (4-bit, no fine-tune) | 74.45% |
| Llama 2 70B — Xiong et al. (ACL 2024) | 47.84% |

Our open-source pipeline reaches **80.99%**, within 3 points of GPT-4 (83.97%). Each component contributes measurably:
- **QLoRA fine-tuning**: +4.03% (teaches USMLE answer format)
- **RAG over medical textbooks**: +2.51% (adds factual grounding at inference time)

Notably, Xiong et al. report that RAG *decreases* GPT-4 accuracy (83.97% → 82.80%) due to context distraction, whereas our fine-tuned model *benefits* from retrieved context (+2.51%) — suggesting fine-tuning improves the model's ability to integrate external knowledge.

> Reference: Xiong et al., *Benchmarking Retrieval-Augmented Generation for Medicine*, ACL 2024. Table 6, Section 5.1. [arxiv:2402.13178](https://arxiv.org/abs/2402.13178)

---

## Architecture

```
┌──────────────────────────────────────────────────────────┐
│                                                          │
│  MedQA-USMLE + PubMedQA                                  │
│         │                                                │
│         ▼                                                │
│  [Preprocessing] ────────► [FAISS Vector Store]          │
│         │                  BAAI/bge-large-en-v1.5        │
│         ▼                         │                      │
│  [QLoRA Fine-tuning]              │ retrieval            │
│  Llama 3.3 70B                    │                      │
│  4× NVIDIA L40S (48GB)            │                      │
│  DeepSpeed ZeRO-2                 │                      │
│         │                         │                      │
│         ▼                         ▼                      │
│  [Fine-tuned Model] ◄──── [RAG Pipeline]                 │
│         │                  LangChain                     │
│         ▼                                                │
│  [vLLM Serving]                                          │
│         │                                                │
│         ▼                                                │
│  [FastAPI REST API] ──► [RAGAS Evaluation]               │
│         │                     │                          │
│         ▼                     ▼                          │
│  [HuggingFace Spaces]   [vs GPT-4o baseline]             │
└──────────────────────────────────────────────────────────┘
```

---

## Tech Stack

| Component | Technology |
|---|---|
| Base model | `meta-llama/Llama-3.3-70B-Instruct` |
| Fine-tuning | QLoRA 4-bit (nf4) via PEFT + SFTTrainer |
| Multi-GPU training | DeepSpeed ZeRO-2, 4× NVIDIA L40S 48GB |
| Experiment tracking | Weights & Biases |
| Vector store | FAISS-GPU |
| Embeddings | `BAAI/bge-large-en-v1.5` |
| RAG framework | LangChain |
| Evaluation | RAGAS + custom accuracy metrics |
| Baseline | GPT-4o via OpenAI API |
| Serving | vLLM + FastAPI |
| Deployment | HuggingFace Spaces |

---

## Dataset

| Dataset | Size | Task |
|---|---|---|
| [MedQA-USMLE](https://huggingface.co/datasets/GBaker/MedQA-USMLE-4-options) | 10,178 train / 1,273 test | Multiple-choice medical questions (USMLE Step 1/2/3) |
| [PubMedQA](https://huggingface.co/datasets/qiaojin/PubMedQA) | 1,000 labeled | Biomedical yes/no/maybe QA from PubMed abstracts |

**Combined training splits:**
- Train: 9,960 examples
- Validation: 1,118 examples
- Test: 1,373 examples (held out until final evaluation)

---

## Training Configuration

```yaml
model:
  base_model: meta-llama/Llama-3.3-70B-Instruct
  quantization: 4-bit NF4 (QLoRA)
  compute_dtype: bfloat16

lora:
  rank: 16
  alpha: 32
  dropout: 0.05
  target_modules: [q_proj, k_proj, v_proj, o_proj, gate_proj, up_proj, down_proj]

training:
  effective_batch_size: 64  # 1 per device × 16 grad accum × 4 GPUs
  learning_rate: 5e-5
  scheduler: cosine
  epochs: 3
  max_seq_length: 1024     # MedQA max 957 tokens — 0% truncation

rag:
  corpus: MedRAG/textbooks  # 125,847 chunks from 18 medical textbooks
  embeddings: BAAI/bge-large-en-v1.5
  index: FAISS IndexFlatIP (cosine similarity)
  top_k: 3
```

---

## Repository Structure

```
medrag/
├── data/
│   ├── download.py              # Download MedQA + PubMedQA from HuggingFace
│   ├── preprocess.py            # Format for instruction tuning (Llama chat template)
│   └── build_vectorstore.py     # Build FAISS index with BGE embeddings
├── training/
│   ├── train.py                 # QLoRA fine-tuning with SFTTrainer
│   ├── config.yaml              # Hyperparameters
│   └── job_train.sh             # SLURM job script (4 GPUs, DeepSpeed)
├── rag/
│   ├── pipeline.py              # LangChain RAG pipeline
│   └── retriever.py             # FAISS retrieval logic
├── serving/
│   ├── api.py                   # FastAPI application
│   ├── inference.py             # vLLM inference wrapper
│   └── Dockerfile
├── evaluation/
│   ├── evaluate.py              # RAGAS evaluation
│   ├── baseline_gpt4o.py        # GPT-4o baseline
│   ├── metrics.py               # Accuracy, F1, RAGAS scores
│   └── results/                 # Experiment results (JSON)
└── notebooks/
    ├── 01_data_exploration.ipynb
    ├── 02_training_analysis.ipynb
    └── 03_evaluation_results.ipynb
```

---

## Reproducing the Results

### 1. Setup

```bash
git clone https://github.com/martinliarte/medrag.git
cd medrag
conda env create -f environment.yml
conda activate medrag
```

### 2. Download & preprocess data

```bash
python data/download.py
python data/preprocess.py
```

### 3. Build vector store

```bash
python data/build_vectorstore.py
```

### 4. Fine-tune (requires 4× NVIDIA L40S or equivalent)

```bash
# With SLURM
sbatch training/job_train.sh

# Without SLURM (single GPU, for testing)
python training/train.py --config training/config.yaml
```

### 5. Evaluate

```bash
python evaluation/evaluate.py --model_path path/to/checkpoint
python evaluation/baseline_gpt4o.py  # requires OPENAI_API_KEY
```

### 6. Serve

```bash
docker build -t medrag-api ./serving
docker run -p 8000:8000 medrag-api
# API docs at http://localhost:8000/docs
```

---

## Evaluation Metrics

- **Accuracy**: % of correct answers on the MedQA-USMLE test set (1,273 questions). Primary metric.
- **RAGAS Faithfulness**: Are answers grounded in retrieved context? Target > 0.85.
- **RAGAS Answer Relevancy**: Is the answer relevant to the question? Target > 0.80.
- **RAGAS Context Precision**: Quality of retrieved context. Target > 0.75.
- **API P95 Latency**: 95th percentile response time. Target < 3s.

---

## Technical Deep Dive

### Why QLoRA over full fine-tuning?

Llama 3.3 70B in bfloat16 requires ~140GB of VRAM just for weights — far beyond what a single GPU can hold. QLoRA solves this with two techniques stacked together:

1. **4-bit NF4 quantization** (via bitsandbytes): compresses model weights from 16-bit to 4-bit using a Normal Float 4 representation optimized for normally-distributed neural network weights. This reduces the model to ~35GB while preserving most precision during forward/backward passes (compute still happens in bfloat16).

2. **LoRA (Low-Rank Adaptation)**: instead of updating all 70B parameters, we inject trainable rank-16 matrices into every projection layer (`q_proj, k_proj, v_proj, o_proj, gate_proj, up_proj, down_proj`). The base model is frozen; only ~400M parameters are trained. The adapters (396MB) are merged or loaded at inference time.

This combination makes it possible to fine-tune a 70B model on 4× 44GB GPUs with an effective batch size of 64.

### Why DeepSpeed ZeRO-2?

Even with 4-bit weights, optimizer states (Adam momentum + variance) for 400M trainable parameters in fp32 take ~3.2GB per GPU without sharding. ZeRO-2 (Zero Redundancy Optimizer stage 2) partitions these optimizer states and gradients across all 4 GPUs, reducing per-GPU memory by ~4×. This is why we can fit batch=1 × grad_accum=16 without OOM.

### Training convergence

```
Epoch 0.06 — loss: 2.159, token_accuracy: 61.8%  (warmup phase)
Epoch 0.13 — loss: 1.244, token_accuracy: 73.5%  (rapid learning)
Epoch 1.00 — eval_loss: 0.872, eval_token_accuracy: 77.6%
Epoch 3.00 — train_loss: 0.897  (stable, no overfitting)
```

The model learned the USMLE answer format in < 1 epoch. Stable grad_norm (0.12–0.18) throughout indicates healthy training dynamics.

### Key engineering challenges solved

- **CUDA library mismatch**: bitsandbytes 0.49.2 links against `libnvJitLink.so.13` but the cluster has `.so.12` → created symlink, exported `LD_LIBRARY_PATH`
- **TRL 1.7.0 API breaking changes**: `DataCollatorForCompletionOnlyLM` removed, `max_seq_length` renamed to `max_length` in `SFTConfig`, `tokenizer=` renamed to `processing_class=` in `SFTTrainer`
- **HPC disk quotas**: NAS home quota (5GB) and shared /home (14TB) both full → routed model cache to node-local /tmp (336GB) via `HF_HOME`
- **OOM in backward pass**: TRL 1.7 computes token-level entropy over the full vocabulary (128k tokens) during training — materializes a 128k × seq_len × batch tensor → solved by halving seq_len (2048→1024) and batch (2→1)
- **W&B on compute nodes without internet**: used `WANDB_MODE=offline`, sync manually with `wandb sync` from login node

---

## About

Built by **Martín Liarte** — CS student at UPV (Valencia), specializing in Computing & AI.

Training runs on the UPV DSIC cluster (tensor.dsic.upv.es) with 4× NVIDIA L40S GPUs (48GB VRAM each).

[![W&B](https://img.shields.io/badge/Weights_%26_Biases-FFBE00?logo=WeightsAndBiases&logoColor=white)](https://wandb.ai)
[![HuggingFace](https://img.shields.io/badge/🤗_HuggingFace-FFD21E)](https://huggingface.co)
[![Python 3.11](https://img.shields.io/badge/Python-3.11-blue?logo=python)](https://python.org)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.12-EE4C2C?logo=pytorch)](https://pytorch.org)
