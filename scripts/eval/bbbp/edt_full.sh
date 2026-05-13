#! /bin/bash
: "${BASE_DIR:?Environment variable BASE_DIR not set}"
: "${DATA_DIR:?Environment variable DATA_DIR not set}"

export PYTHONPATH=${BASE_DIR}:${PYTHONPATH}
export CUDA_VISIBLE_DEVICES=1

python ${BASE_DIR}/zeroshot/inference.py \
    --pretrained_model_name_or_path unsloth/Llama-3.1-8B-Instruct \
    --tokenizer_path DongkiKim/Mol-Llama-3.1-8B-Instruct \
    --data_dir ${DATA_DIR} \
    --task_name bbbp \
    --use_dq_encoder \
    --qformer_path ${BASE_DIR}/checkpoints/stage2/stage2_dqformer_full/last.ckpt \
    --prompt_type task_info