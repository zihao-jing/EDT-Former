#!/bin/bash
# Molecule Generation/Captioning Training Script

# Source environment setup
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"

if [ -f "${PROJECT_ROOT}/local.env.sh" ]; then
    echo "Sourcing local.env.sh..."
    source "${PROJECT_ROOT}/local.env.sh"
else
    echo "Warning: local.env.sh not found. Please ensure environment variables are set."
    : "${BASE_DIR:?Environment variable BASE_DIR not set}"
    : "${DATA_DIR:?Environment variable DATA_DIR not set}"
    export PYTHONPATH=${BASE_DIR}:${PYTHONPATH}
fi

# Configuration
export GPUs="0"  # GPU IDs to use for Molecule Generation training
export MASTER_PORT=29502  # Master port for distributed training

# Set CUDA architecture to avoid compilation warnings
# Common options: "7.0" (V100), "8.0" (A100), "8.6" (RTX 3090), "8.9" (RTX 4090), "9.0" (H100)
# Set to your GPU architecture or leave commented to auto-detect
export TORCH_CUDA_ARCH_LIST="8.0"

# Get deepspeed stage from argument or default to 2
DEEPSPEED_STAGE=${1:-2}

# Check for --clear-cache flag
CLEAR_CACHE_FLAG=""
if [[ "$*" == *"--clear-cache"* ]]; then
    CLEAR_CACHE_FLAG="--clear_cache"
    echo "Cache clearing enabled"
fi

echo "=========================================="
echo "Molecule Generation EDT-Former Training"
echo "=========================================="
echo "BASE_DIR: $BASE_DIR"
echo "GPUs: $GPUs"
echo "MASTER_PORT: $MASTER_PORT"
echo "DeepSpeed Stage: $DEEPSPEED_STAGE"
echo "=========================================="

# Launch training with DeepSpeed
deepspeed --master_port ${MASTER_PORT} --include localhost:${GPUs} \
    ${BASE_DIR}/runner/qa_finetuning.py \
    --model_config_path ${BASE_DIR}/configs/qa/mol_gen/model_config.yaml \
    --training_config_path ${BASE_DIR}/configs/qa/mol_gen/training_config.yaml \
    --data_config_path ${BASE_DIR}/configs/qa/mol_gen/data_config_preprocessed.yaml \
    --deepspeed_stage $DEEPSPEED_STAGE \
    $CLEAR_CACHE_FLAG

echo "=========================================="
echo "Training completed!"
echo "=========================================="


