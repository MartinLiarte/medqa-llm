#!/bin/bash
#SBATCH -p long
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-task=16
#SBATCH --time=48:00:00
#SBATCH --job-name=medrag_ft_v2
#SBATCH -o logs/train_%j.log

echo "=== Job start: $(date) ==="
echo "=== Node: $(hostname) ==="
nvidia-smi

source /opt/miniconda3/etc/profile.d/conda.sh
conda activate medrag
export LD_LIBRARY_PATH=$CONDA_PREFIX/lib:$LD_LIBRARY_PATH

if [ -f "$HOME/.cache/huggingface/token" ]; then
    export HF_TOKEN=$(cat $HOME/.cache/huggingface/token)
else
    echo "ERROR: HuggingFace token no encontrado"; exit 1
fi

cd $HOME/medrag
mkdir -p logs

export WANDB_PROJECT=medrag
export WANDB_API_KEY=$(python3 -c "import netrc; print(netrc.netrc().authenticators('api.wandb.ai')[2])")
export WANDB_MODE=offline

export HF_HOME=/tmp/hf_cache_$USER
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export HF_HUB_DISABLE_SYMLINKS_WARNING=1
mkdir -p $HF_HOME
rm -f $HF_HOME/hub/.locks/models--meta-llama*/*.lock 2>/dev/null || true

# Crear dataset MedQA-only si no existe
if [ ! -f "$HOME/W/data/processed/medqa_only/train.jsonl" ]; then
    echo "Creando dataset MedQA-only..."
    python3 -c "
import json, random, os
from pathlib import Path

random.seed(42)
data = []
with open(os.path.expanduser('~/W/data/processed/medqa/train.jsonl')) as f:
    for line in f:
        data.append(json.loads(line))

random.shuffle(data)
n = len(data)
split = int(0.9 * n)
train, val = data[:split], data[split:]

Path(os.path.expanduser('~/W/data/processed/medqa_only')).mkdir(parents=True, exist_ok=True)
with open(os.path.expanduser('~/W/data/processed/medqa_only/train.jsonl'), 'w') as f:
    for ex in train: f.write(json.dumps(ex) + '\n')
with open(os.path.expanduser('~/W/data/processed/medqa_only/val.jsonl'), 'w') as f:
    for ex in val: f.write(json.dumps(ex) + '\n')
print(f'MedQA-only — Train: {len(train)}, Val: {len(val)}')
"
fi

# DeepSpeed ZeRO-2 config
cat > /tmp/ds_config_v2.json << 'EOF'
{
  "zero_optimization": {
    "stage": 2,
    "allgather_partitions": true,
    "allgather_bucket_size": 2e8,
    "overlap_comm": true,
    "reduce_scatter": true,
    "reduce_bucket_size": 2e8,
    "contiguous_gradients": true
  },
  "bf16": {
    "enabled": true
  },
  "gradient_clipping": 1.0,
  "steps_per_print": 10,
  "train_micro_batch_size_per_gpu": 1,
  "gradient_accumulation_steps": 16,
  "wall_clock_breakdown": false
}
EOF

torchrun \
  --nproc_per_node=4 \
  --master_port=29501 \
  training/train.py \
  --config training/config_v2.yaml \
  --deepspeed /tmp/ds_config_v2.json

# Copiar adaptadores LoRA a W/
echo "Copiando modelo v2 a W/..."
mkdir -p $HOME/W/checkpoints/medrag_v2
cp -r /tmp/medrag_checkpoints_v2/final $HOME/W/checkpoints/medrag_v2/final
echo "Copia completada: $(du -sh $HOME/W/checkpoints/medrag_v2/final)"

echo "=== Job end: $(date) ==="
