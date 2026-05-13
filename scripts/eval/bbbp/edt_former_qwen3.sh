#! /bin/bash
: "${BASE_DIR:?Environment variable BASE_DIR not set}"
: "${DATA_DIR:?Environment variable DATA_DIR not set}"

export PYTHONPATH=${BASE_DIR}:${PYTHONPATH}
export CUDA_VISIBLE_DEVICES=0

for prompt_type in default rationale task_info; do

    python ${BASE_DIR}/evaluation/inference.py \
        --pretrained_model_name_or_path unsloth/Qwen3-8B \
        --data_dir ${DATA_DIR} \
        --task_name bbbp \
        --qformer_path ${BASE_DIR}/checkpoints/stage2_dqw2d_qwen3_v2/last.ckpt \
        --prompt_type ${prompt_type} \
        --output_name edt_former_qwen3 \
        --use_dq_encoder \
        --enable_blending \
        --llm_baseline
done