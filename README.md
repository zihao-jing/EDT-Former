# EDT-Former: Entropy-Guided Dynamic Tokens for Graph-LLM Alignment in Molecular Understanding

[![ICLR 2026](https://img.shields.io/badge/ICLR-2026-blue.svg)](https://openreview.net/forum?id=yzwSzhqLpH&referrer=%5BAuthor%20Console%5D(%2Fgroup%3Fid%3DICLR.cc%2F2026%2FConference%2FAuthors%23your-submissions))
[![Paper](https://img.shields.io/badge/Paper-PDF-red.svg)](https://www.arxiv.org/abs/2602.02742)
[![Pretrain Data](https://img.shields.io/badge/🤗%20Dataset-Pretrain-yellow.svg)](https://huggingface.co/datasets/zihaojing/EDT-Former-pretrain-data)
[![SFT Data](https://img.shields.io/badge/🤗%20Dataset-SFT-yellow.svg)](https://huggingface.co/datasets/zihaojing/EDT-Former-sft-data)
[![Encoder](https://img.shields.io/badge/🤗%20Model-Encoder-green.svg)](https://huggingface.co/zihaojing/EDT-Former-encoder)
[![Full Model](https://img.shields.io/badge/🤗%20Model-Full-green.svg)](https://huggingface.co/zihaojing/EDT-Former-model)

> **Accepted at ICLR 2026**
>
> **Authors:** Zihao Jing, Qiuhao Zeng, Ruiyi Fang, Yan Sun, Boyu Wang, Pingzhao Hu

### Abstract
Molecular understanding is central to advancing areas such as scientific and drug discovery, yet Large Language Models (LLMs) struggle to understand molecular graphs effectively. Existing graph–LLM bridges often adapt the Q-Former-style connector with fixed-length static tokens, which is originally designed for vision tasks. These designs overlook stereochemistry and substructural context and typically require costly LLM-backbone fine-tuning, limiting efficiency and generalization. We introduce EDT-Former, an Entropy-guided Dynamic Token Transformer that generates tokens aligned with informative molecular patches, thereby preserving both local and global structural features for molecular graph understanding. Beyond prior approaches, EDT-Former enables alignment between frozen graph encoders and LLMs without tuning the LLM backbone (excluding the embedding layer), resulting in computationally efficient finetuning, and achieves state-of-the-art results on MoleculeQA, Mol-Instructions, and property prediction benchmarks (TDC, MoleculeNet), underscoring its effectiveness for scalable and generalizable multimodal molecular understanding.

![EDT-Former Architecture](figs/arch.png)

### Environment Setup (conda)
- **Prerequisites**: CUDA-ready PyTorch, Conda (miniconda/anaconda), and git.
- Create and activate the environment:

**Using environment.yml (Recommended)**
```bash
conda env create -f environment.yml
conda activate edtformer
pip install --no-deps --no-build-isolation torch-geometric
pip install --no-deps --no-build-isolation flash-attn
pip install --no-deps --no-build-isolation torch-scatter
# The torch-scatter installation will take a long time, please kindly wait for it.
```

If you want to use Uni-Mol model as a encoder, please run:
```bash
git clone https://github.com/dptech-corp/Uni-Core.git
pip install --no-deps --no-build-isolation ./Uni-Core/
```

- Initialize environment variables (copy and edit paths first):
```bash
cp env.sh local.env.sh
# Edit local.env.sh: set HF_HOME, BASE_DIR, DATA_DIR, DATA_CACHE_DIR, CHECKPOINT_DIR
source local.env.sh
```

### Models and Datasets

Pre-trained models and datasets are available on HuggingFace:

| Resource | HuggingFace Link | Description |
|----------|-----------------|-------------|
| Pretrain Data | [zihaojing/EDT-Former-pretrain-data](https://huggingface.co/datasets/zihaojing/EDT-Former-pretrain-data) | Stage 1 encoder pretraining corpus (~12 GB, from PubChem) |
| SFT Data | [zihaojing/EDT-Former-sft-data](https://huggingface.co/datasets/zihaojing/EDT-Former-sft-data) | Stage 2 instruction-tuning corpus (~12 GB, from Mol-LLaMA-Instruct) |
| Encoder | [zihaojing/EDT-Former-encoder](https://huggingface.co/zihaojing/EDT-Former-encoder) | Stage 1 EDT-Former encoder checkpoint (~699 MB) |
| Full Model | [zihaojing/EDT-Former-model](https://huggingface.co/zihaojing/EDT-Former-model) | Stage 2 full EDT-Former model (encoder + Llama-3.1-8B-Instruct, ~16 GB) |

**Quick download:**
```python
from huggingface_hub import snapshot_download

# Download pretrained encoder (Stage 1)
snapshot_download("zihaojing/EDT-Former-encoder", local_dir="checkpoints/edt_former_s1_large/final_model")

# Download full model (Stage 2)
snapshot_download("zihaojing/EDT-Former-model", local_dir="checkpoints/edt_former_s2_large/final_model")

# Download pretraining data
snapshot_download("zihaojing/EDT-Former-pretrain-data", repo_type="dataset", local_dir="data/pretrain")

# Download SFT data
snapshot_download("zihaojing/EDT-Former-sft-data", repo_type="dataset", local_dir="data/finetune")
```

### Data
- Datasets can be downloaded from HuggingFace (see above) or placed manually under `data/`.
- Config files under `configs/*/data_config*.yaml` accept either a local path or a HuggingFace repo ID for `preprocessed_data`.
- Preprocessing helpers are in `scripts/preprocess/` and `data_provider/preprocess/`.

### Pretraining (Stage 1)
The main entrypoint is `runner/pretrain.py`, launched via DeepSpeed.
```bash
deepspeed --include localhost:0 \
    runner/pretrain.py -- \
    --model_config_path    configs/stage1/model_config.yaml \
    --training_config_path configs/stage1/training_config.yaml \
    --data_config_path     configs/stage1/data_config_preprocessed.yaml
```
Or use the provided helper script (edit `local.env.sh` first):
```bash
bash scripts/training/pretraining.sh
```

### Finetuning / Instruction Tuning (Stage 2)
The main entrypoint is `runner/finetuning.py`.
```bash
deepspeed --include localhost:0 \
    runner/finetuning.py \
    --model_config_path    configs/stage2/model_config.yaml \
    --training_config_path configs/stage2/training_config.yaml \
    --data_config_path     configs/stage2/data_config_preprocessed.yaml
```
Or:
```bash
bash scripts/training/finetuning.sh
```
Set the pretrained checkpoint path in `model_config.yaml` (`model_name_or_path`).

### Downstream Tasks (MoleculeQA / Mol-Instructions)
QA training uses `runner/qa_finetuning.py`. Example for MoleculeQA:
```bash
deepspeed --include localhost:0,1 \
    runner/qa_finetuning.py \
    --model_config_path    configs/qa/mol_qa/model_config.yaml \
    --training_config_path configs/qa/mol_qa/training_config.yaml \
    --data_config_path     configs/qa/mol_qa/data_config_preprocessed.yaml \
    --deepspeed_stage 2
```
Helper scripts for all downstream tasks are in `scripts/qa/`.


### License
See `LICENSE`.

### Citation
If you find this work helpful, please consider citing:
```bibtex
@inproceedings{jing2026edtformer,
  title={Entropy-Guided Dynamic Tokens for Graph-LLM Alignment in Molecular Understanding},
  author={Jing, Zihao and Zeng, Qiuhao and Fang, Ruiyi and Sun, Yan and Wang, Boyu and Hu, Pingzhao},
  booktitle={International Conference on Learning Representations (ICLR)},
  year={2026}
}
```

