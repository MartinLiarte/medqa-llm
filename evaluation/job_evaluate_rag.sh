#!/bin/bash
#SBATCH -p docencia
#SBATCH --gres=gpu:2
#SBATCH --cpus-per-task=8
#SBATCH --time=04:00:00
#SBATCH --job-name=medrag_eval_rag
#SBATCH -o logs/eval_rag_%j.log

echo "=== Eval RAG start: $(date) ==="
echo "=== Node: $(hostname) ==="

source /opt/miniconda3/etc/profile.d/conda.sh
conda activate medrag
export LD_LIBRARY_PATH=$CONDA_PREFIX/lib:$LD_LIBRARY_PATH

export HF_TOKEN=$(cat $HOME/.cache/huggingface/token)
export HF_HOME=/tmp/hf_cache_$USER

cd $HOME/medrag
mkdir -p logs evaluation/results

python evaluation/evaluate_rag.py \
    --base_model meta-llama/Llama-3.3-70B-Instruct \
    --model_path $HOME/W/checkpoints/medrag_v2/final \
    --vectorstore $HOME/W/vectorstore \
    --top_k 3 \
    --test_data $HOME/W/data/processed/combined/test.jsonl \
    --output evaluation/results/rag_v2_top3.json \
    --batch_size 2 \
    --max_new_tokens 256

echo "=== Eval RAG end: $(date) ==="
