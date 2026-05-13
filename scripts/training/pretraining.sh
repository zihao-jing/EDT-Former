#!/bin/bash

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
export GPUs="0"  # GPU IDs to use
export MASTER_PORT=29500  # Master port for distributed training

# Set CUDA architecture to avoid compilation warnings
# Common options: "7.0" (V100), "8.0" (A100), "8.6" (RTX 3090), "8.9" (RTX 4090), "9.0" (H100)
# Set to your GPU architecture or leave commented to auto-detect
# export TORCH_CUDA_ARCH_LIST="8.0"  # Uncomment and set for your GPU
export TORCH_CUDA_ARCH_LIST="8.0"

# # Launch with DeepSpeed
deepspeed --master_port ${MASTER_PORT} --include localhost:${GPUs} \
    ${BASE_DIR}/runner/pretrain.py -- \
    --model_config_path ${BASE_DIR}/configs/stage1/model_config.yaml \
    --training_config_path ${BASE_DIR}/configs/stage1/training_config.yaml \
    --data_config_path ${BASE_DIR}/configs/stage1/data_config_preprocessed.yaml

