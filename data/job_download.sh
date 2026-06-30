#!/bin/bash
#SBATCH -p docencia
#SBATCH --gres=gpu:0
#SBATCH --cpus-per-task=4
#SBATCH --time=02:00:00
#SBATCH --job-name=medrag_data
#SBATCH -o logs/download_%j.log

echo "=== Download start: $(date) ==="

source /opt/miniconda3/etc/profile.d/conda.sh
conda activate medrag
export LD_LIBRARY_PATH=$CONDA_PREFIX/lib:$LD_LIBRARY_PATH

cd $HOME/medrag

echo "=== Downloading datasets ==="
python data/download.py

echo "=== Preprocessing ==="
python data/preprocess.py

echo "=== Done: $(date) ==="
