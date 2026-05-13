#!/bin/bash
# Description-Guided Molecule Design Finetuning Training Script

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
export GPUs="0"  # GPU IDs to use for molecule design training
export MASTER_PORT=29500  # Master port for distributed training

# Set CUDA architecture to avoid compilation warnings
# Common options: "7.0" (V100), "8.0" (A100), "8.6" (RTX 3090), "8.9" (RTX 4090), "9.0" (H100)
# Set to your GPU architecture or leave commented to auto-detect
export TORCH_CUDA_ARCH_LIST="8.0"

# Get deepspeed stage from argument or default to 2
# Note: 
#   - When num_train_epochs=0: DeepSpeed is AUTOMATICALLY DISABLED (stage ignored)
#   - ZeRO Stage 3: For training WITH evaluation (eval_strategy="steps"/"epoch")
#   - ZeRO Stage 2: For training WITHOUT evaluation (eval_strategy="no")
# Usage: ./mol_design.sh     (default - auto-detects based on num_train_epochs)
#        ./mol_design.sh 3   (training with eval)
#        ./mol_design.sh 2   (training without eval)
DEEPSPEED_STAGE=${1:-2}


echo "=========================================="
echo "Description-Guided Molecule Design EDT-Former Training"
echo "=========================================="
echo "BASE_DIR: $BASE_DIR"
echo "GPUs: $GPUs"
echo "MASTER_PORT: $MASTER_PORT"
echo "DeepSpeed Stage: $DEEPSPEED_STAGE"
echo "=========================================="

# Launch training with DeepSpeed
deepspeed --master_port ${MASTER_PORT} --include localhost:${GPUs} \
    ${BASE_DIR}/runner/qa_finetuning.py \
    --model_config_path ${BASE_DIR}/configs/qa/description_guided_molecule_design/model_config.yaml \
    --training_config_path ${BASE_DIR}/configs/qa/description_guided_molecule_design/training_config.yaml \
    --data_config_path ${BASE_DIR}/configs/qa/description_guided_molecule_design/data_config_preprocessed.yaml \
    --deepspeed_stage ${DEEPSPEED_STAGE}

echo "=========================================="
echo "Training completed!"
echo "=========================================="


