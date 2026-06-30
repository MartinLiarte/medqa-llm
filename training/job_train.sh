#!/bin/bash
#SBATCH -p long
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-task=16
#SBATCH --time=48:00:00
#SBATCH --job-name=medrag_ft
#SBATCH -o logs/train_%j.log

echo "=== Job start: $(date) ==="
echo "=== Node: $(hostname) ==="
nvidia-smi

source /opt/miniconda3/etc/profile.d/conda.sh
conda activate medrag
export LD_LIBRARY_PATH=$CONDA_PREFIX/lib:$LD_LIBRARY_PATH

# HuggingFace token
if [ -f "$HOME/.cache/huggingface/token" ]; then
    export HF_TOKEN=$(cat $HOME/.cache/huggingface/token)
else
    echo "ERROR: HuggingFace token no encontrado. Ejecuta huggingface-cli login primero."
    exit 1
fi

cd $HOME/medrag

mkdir -p logs

export WANDB_PROJECT=medrag
export WANDB_API_KEY=$(python3 -c "import netrc; print(netrc.netrc().authenticators('api.wandb.ai')[2])")
export WANDB_MODE=offline   # compute nodes may lack internet; sync with: wandb sync wandb/

# Cache del modelo en /tmp (disco local del nodo, 336GB libres)
# Los checkpoints LoRA (~1-2GB) van a W/ que sí tiene cuota suficiente
export HF_HOME=/tmp/hf_cache_$USER
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export HF_HUB_DISABLE_SYMLINKS_WARNING=1
mkdir -p $HF_HOME

# Limpiar lock files obsoletos de intentos anteriores
rm -f $HF_HOME/hub/.locks/models--meta-llama*/*.lock 2>/dev/null || true

# DeepSpeed ZeRO-2 config
cat > /tmp/ds_config.json << 'EOF'
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
  "gradient_accumulation_steps": 16
}
EOF

torchrun \
  --nproc_per_node=4 \
  --master_port=29500 \
  training/train.py \
  --config training/config.yaml \
  --deepspeed /tmp/ds_config.json

# Copiar adaptadores LoRA finales a W/ (solo ~500MB, cabe en la cuota)
echo "Copiando modelo final a W/..."
mkdir -p $HOME/W/checkpoints/medrag
cp -r /tmp/medrag_checkpoints/final $HOME/W/checkpoints/medrag/final
echo "Copia completada: $(du -sh $HOME/W/checkpoints/medrag/final)"

echo "=== Job end: $(date) ==="
