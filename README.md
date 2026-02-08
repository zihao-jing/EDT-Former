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

- Initialize environment variables (edit paths first):
```bash
# copy and edit
cp init_env.sh init_env.local.sh
# In init_env.local.sh set HF_HOME, BASE_DIR, DATA_DIR, CUDA_VISIBLE_DEVICES
source init_env.local.sh
```

### Data
- Place datasets under `data/` (see `data/`, `configs/*/data_config.yaml`).
- Preprocessing helpers are in `scripts/preprocess/`.

### Pretraining (Stage 1)
The main entrypoint is `stage1.py`.
```bash
python stage1.py \
  --train_config_path configs/stage1/train_config.yaml \
  --data_config_path  configs/stage1/data_config.yaml
```
Notes:
- Checkpoints are saved to `checkpoints/<filename>/` as defined in the train config.
- Use `--test_mode` for a quick smoke test.

### Finetuning / Instruction Tuning (Stage 2)
The main entrypoint is `stage2.py`.
```bash
python stage2.py \
  --train_config_path configs/stage2/train_config.yaml \
  --data_config_path  configs/stage2/data_config.yaml
```
Common options:
- `--test_mode`: small subset run
- `--resume_from last` or a path: resume training

Ensure the pretrained checkpoint path is set in the model config (e.g., `model_config.model_name_or_path`).

### Downstream Example: MoleculeQA
Train/evaluate MoleculeQA with the main script in `evaluation/moleculeqa.py`:
```bash
python evaluation/moleculeqa.py \
  --train_config_path configs/moleculeqa/train_config.yaml \
  --data_config_path  configs/moleculeqa/data_config.yaml
```
Or use the provided helper script (edit `BASE_DIR`/`DATA_DIR` first):
```bash
bash scripts/MoleculeQA/dqw2d.sh
```


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

