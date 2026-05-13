export HF_HOME=your-hf-home
export BASE_DIR=your-project-dir
export DATA_DIR=your-data-dir
export DATA_CACHE_DIR=your-data-cache-dir
export CHECKPOINT_DIR=your-checkpoint-dir
export PYTHONPATH=${BASE_DIR}:${PYTHONPATH}

# Create cache directory if it doesn't exist
mkdir -p ${DATA_CACHE_DIR}
mkdir -p ${CHECKPOINT_DIR}
source ~/miniconda3/bin/activate
conda activate your-conda-env

# export OPENAI_API_KEY=your-openai-api-key
# export PATH="/path/to/your/conda/env/bin:$PATH"

# rclone sync gl:/FS/EDT-Former/checkpoints/ ./checkpoints/ --progress
# rclone sync gl:/FS/EDT-Former/data/ ./data/ --progress
