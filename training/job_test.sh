#!/bin/bash
#SBATCH -p docencia
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --time=01:00:00
#SBATCH --job-name=medrag_test
#SBATCH -o logs/test_%j.log

echo "=== GPU check ==="
nvidia-smi

source /opt/miniconda3/etc/profile.d/conda.sh
conda activate medrag
export LD_LIBRARY_PATH=$CONDA_PREFIX/lib:$LD_LIBRARY_PATH

# HuggingFace token (configurado con huggingface-cli login)
if [ -f "$HOME/.cache/huggingface/token" ]; then
    export HF_TOKEN=$(cat $HOME/.cache/huggingface/token)
fi

cd $HOME/medrag

echo "=== Python check ==="
python -c "
import torch
print(f'PyTorch: {torch.__version__}')
print(f'CUDA available: {torch.cuda.is_available()}')
print(f'GPU count: {torch.cuda.device_count()}')
for i in range(torch.cuda.device_count()):
    print(f'  GPU {i}: {torch.cuda.get_device_name(i)} ({torch.cuda.get_device_properties(i).total_memory / 1e9:.1f} GB)')
"

echo "=== Package check ==="
python -c "
import transformers, peft, trl, bitsandbytes, datasets, accelerate, deepspeed
print('All packages OK')
print(f'  transformers: {transformers.__version__}')
print(f'  peft: {peft.__version__}')
print(f'  trl: {trl.__version__}')
"

echo "=== Quick model load test (small) ==="
python -c "
from transformers import AutoTokenizer
tok = AutoTokenizer.from_pretrained('meta-llama/Llama-3.1-8B-Instruct')
print(f'Tokenizer loaded OK, vocab size: {tok.vocab_size}')
"

echo "Test job finished: $(date)"
