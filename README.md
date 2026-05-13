# EDT-Former: Entropy-Guided Dynamic Tokens for Graph-LLM Alignment in Molecular Understanding

[![ICLR 2026](https://img.shields.io/badge/ICLR-2026-blue.svg)](https://openreview.net/forum?id=yzwSzhqLpH&referrer=%5BAuthor%20Console%5D(%2Fgroup%3Fid%3DICLR.cc%2F2026%2FConference%2FAuthors%23your-submissions))
[![Paper](https://img.shields.io/badge/Paper-PDF-red.svg)](https://www.arxiv.org/abs/2602.02742)

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

### Data
- Place datasets under `data/` (see `configs/*/data_config*.yaml`).
- Preprocessing helpers are in `scripts/preprocess/` and `data_provider/preprocess/`.

### Pretraining (Stage 1)
The main entrypoint is `runner/pretrain.py`, launched via DeepSpeed.
```bash
deepspeed --include localhost:0 \
    runner/pretrain.py -- \
    --model_config_path    configs/stage1_dqw2d/model_config.yaml \
    --training_config_path configs/stage1_dqw2d/training_config.yaml \
    --data_config_path     configs/stage1_dqw2d/data_config_preprocessed.yaml
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
    --model_config_path    configs/stage2_dqw2d/model_config.yaml \
    --training_config_path configs/stage2_dqw2d/training_config.yaml \
    --data_config_path     configs/stage2_dqw2d/data_config_preprocessed.yaml
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

