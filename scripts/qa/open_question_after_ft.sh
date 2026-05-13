#!/bin/bash
# Description-Guided Molecule Design Finetuning Training Script
# 2hours to train 5 epochs
# 2hours zero-shot evaluation
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

# Set CUDA architecture to avoid compilation warnings
# Common options: "7.0" (V100), "8.0" (A100), "8.6" (RTX 3090), "8.9" (RTX 4090), "9.0" (H100)
# Set to your GPU architecture or leave commented to auto-detect
export TORCH_CUDA_ARCH_LIST="8.0"

# Get deepspeed stage from argument or default to 2
# Note: 
#   - When num_train_epochs=0: DeepSpeed is AUTOMATICALLY DISABLED (stage ignored)
#   - ZeRO Stage 3: For training WITH evaluation (eval_strategy="steps"/"epoch")
#   - ZeRO Stage 2: For training WITHOUT evaluation (eval_strategy="no")
# Usage: ./open_question.sh     (default - auto-detects based on num_train_epochs)
#        ./open_question.sh 3   (training with eval)
#        ./open_question.sh 2   (training without eval)


echo "=========================================="
echo "Open Question EDT-Former After Finetuning"
echo "=========================================="
echo "BASE_DIR: $BASE_DIR"
echo "=========================================="

# Launch training with DeepSpeed
CUDA_VISIBLE_DEVICES=0 python ${BASE_DIR}/runner/qa_finetuning.py \
    --model_config_path ${BASE_DIR}/configs/qa/open_question_after_ft/model_config.yaml \
    --training_config_path ${BASE_DIR}/configs/qa/open_question_after_ft/training_config.yaml \
    --data_config_path ${BASE_DIR}/configs/qa/open_question_after_ft/data_config_preprocessed.yaml

echo "=========================================="
echo "Training completed!"
echo "=========================================="


