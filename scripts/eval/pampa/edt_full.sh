#! /bin/bash
: "${BASE_DIR:?Environment variable BASE_DIR not set}"
: "${DATA_DIR:?Environment variable DATA_DIR not set}"

export PYTHONPATH=${BASE_DIR}:${PYTHONPATH}
export CUDA_VISIBLE_DEVICES=0

python ${BASE_DIR}/zeroshot/inference.py \
    --pretrained_model_name_or_path unsloth/Llama-3.1-8B-Instruct \
    --tokenizer_path DongkiKim/Mol-Llama-3.1-8B-Instruct \
    --data_dir ${DATA_DIR} \
    --task_name pampa \
    --qformer_path ${BASE_DIR}/checkpoints/stage2/stage2_dqformer_full/last.ckpt \
    --prompt_type default \
    --use_dq_encoder