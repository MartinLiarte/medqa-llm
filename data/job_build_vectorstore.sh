#!/bin/bash
#SBATCH -p docencia
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --time=02:00:00
#SBATCH --job-name=medrag_vs
#SBATCH -o logs/vectorstore_%j.log

echo "=== Build vectorstore start: $(date) ==="
echo "=== Node: $(hostname) ==="

source /opt/miniconda3/etc/profile.d/conda.sh
conda activate medrag
export LD_LIBRARY_PATH=$CONDA_PREFIX/lib:$LD_LIBRARY_PATH

export HF_TOKEN=$(cat $HOME/.cache/huggingface/token)
export HF_HOME=/tmp/hf_cache_$USER
export HF_DATASETS_CACHE=/tmp/hf_datasets_$USER

mkdir -p /tmp/hf_datasets_$USER
mkdir -p $HOME/W/vectorstore

cd $HOME/medrag

python data/build_vectorstore.py \
    --corpus textbooks \
    --batch_size 256

echo "Vectorstore size: $(du -sh $HOME/W/vectorstore/)"
echo "=== Build vectorstore end: $(date) ==="
