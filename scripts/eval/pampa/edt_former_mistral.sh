#! /bin/bash
: "${BASE_DIR:?Environment variable BASE_DIR not set}"
: "${DATA_DIR:?Environment variable DATA_DIR not set}"

export PYTHONPATH=${BASE_DIR}:${PYTHONPATH}
export CUDA_VISIBLE_DEVICES=0

for prompt_type in default rationale task_info; do

    python ${BASE_DIR}/evaluation/inference.py \
        --pretrained_model_name_or_path mistralai/Ministral-8B-Instruct-2410 \
        --data_dir ${DATA_DIR} \
        --task_name pampa \
        --qformer_path ${BASE_DIR}/checkpoints/stage2_dqw2d_mistral8b_v2_dusky/last.ckpt \
        --prompt_type ${prompt_type} \
        --output_name edt_former_mistral \
        --use_dq_encoder \
        --enable_blending \
        --llm_baseline
done