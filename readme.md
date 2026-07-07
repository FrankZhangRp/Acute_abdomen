# Acute abdomen on non-contrast CT: a foundation model for diagnosis, risk stratification and emergency triage

[![DINOv2 License: Apache-2.0](https://img.shields.io/badge/DINOv2%20License-Apache--2.0-green.svg)](./pretrain/LICENSE)
[![Python 3.10](https://img.shields.io/badge/Python-3.10-blue.svg)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.x-ee4c2c.svg)](https://pytorch.org/)
![Patient data not included](https://img.shields.io/badge/Patient%20Data-Not%20Included-lightgrey.svg)

This is the official repository for the paper “Acute abdomen on non-contrast CT: a foundation model for diagnosis, risk stratification and emergency triage.”

The repo contains:

- Pre-training: DINOv2-based ViT backbone pretraining code in `pretrain/`.
- Fine-tuning & evaluation: Multi-task classification on non-contrast abdominal CT in `finetune/`, using a 2.5D Transformer decoder to model full volumes and supporting multi-label/multi-task setups.
- Statistical analysis utilities: bootstrap CIs, paired DeLong, MRMC GLMM, Wilcoxon signed-rank, and Mann-Whitney U scripts in `statistics/`.

## Repository Structure

The repository is organized into two main parts:

- Pre-training (DINOv2)
  - Directory: `pretrain/`
  - Contents: DINOv2 training/eval utilities, configs, and scripts (PyTorch ≥ 2.1).

- Fine-tuning & Evaluation
  - Directory: `finetune/`
  - Contents: Datasets (`data/`), models & layers (`models/`, `layers/`), trainers (`trainer/`), configs (`configs/`), runner (`run/run_trainer.py`), and utilities (`utils/`).

## Install Environment

1) Create a Conda environment

```sh
conda create -n abdomen python=3.10 -y
conda activate abdomen
```

1) Download repo

```sh
git clone <repository-url>
cd Acute_abdomen
```

Alternative download commands:

```sh
wget <repository-zip-url> -O Acute_abdomen.zip
python -m zipfile -e Acute_abdomen.zip Acute_abdomen
cd Acute_abdomen
```

1) Install PyTorch (choose one)

- Conda (CUDA 12.4 example):

```sh
conda install pytorch torchvision torchaudio pytorch-cuda=12.4 -c pytorch -c nvidia
```

- Pip (CUDA 12.4 example):

```sh
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124
```

- CPU only (e.g., macOS/no GPU):

```sh
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu
```

1) Install additional dependencies for fine-tuning

```sh
pip install omegaconf tqdm pillow scikit-learn nibabel numpy torchvision tensorboard
```

1) [Optional] Install dependencies for the statistical-analysis scripts

```sh
pip install pandas scipy statsmodels scikit-learn
```

1) [Optional] Install DINOv2 pre-training environment

Only required if you plan to run DINOv2 pre-training:

```sh
conda env create -f pretrain/conda.yaml
conda activate dinov2_new
cd pretrain
pip install -e .
```

After the above steps, you can run fine-tuning/evaluation directly; use the DINOv2 environment only if you run pre-training.

## Assets, Checkpoints and Data Preparation

1) Data files and lists

- Fine-tuning/evaluation uses `finetune/data/volume_dataset.py` and supports volumes as `.npy` or `.nii.gz`.
- Dataset lists can be `.txt` or `.csv`:
  - The first column is the path to the volume; subsequent columns are labels.
  - For multi-task/multi-label, provide multiple 0/1 labels separated by spaces or commas.

Example (multi-task/multi-label):

```text
data/examples/study_0001.npy 1 0 0 1 0 1 0 0 1 0 1
data/examples/study_0002.npy 0 1 1 0 0 0 1 0 0 1 0
```

Example (single-task binary):

```text
data/examples/study_0003.npy 1
data/examples/study_0004.npy 0
```

Notes:

- NIfTI `.nii.gz` volumes are loaded and converted to grayscale slices along the z-axis, then stacked to 3-channel images for augmentation/normalization.
- Training samples up to 300 slices per volume with a stride; evaluation can set different `max_slices` via evaluation trainers.

1) CT intensity preprocessing

- For both **pre-training** and **fine-tuning**, the raw CT volumes were normalized using a **soft-tissue window**:
  - **window width = 350 HU**
  - **window level = 40 HU**
- In practical terms, this corresponds to clipping intensities to approximately **[-135, 215] HU** and then rescaling to the image range used by the downstream pipeline.
- This preprocessing choice was applied consistently across the AbdomenNet pretraining and downstream classification experiments unless a specific external baseline required its own author-recommended preprocessing pipeline.

1) Model checkpoints

- Model weights and patient cases are not tracked in this repository. Obtain checkpoints only from an authorized project release channel and keep local artifact paths out of git.
- Pretrained backbone (optional): set `model.pretrained_weights` to a ViT DINOv2-style checkpoint. The loader supports either:
  - A direct `pos_embed` state dict; or
  - A dict with `teacher.backbone.*` keys (automatically mapped to the model backbone).
- Fine-tuning checkpoints: saved under `--output-dir/checkpoints/epoch_*.pth`. For evaluation, set `model.ckpt_path` to the desired checkpoint.
- A backbone-only checkpoint can initialize the model but is not a complete classification release by itself. Classification evaluation also requires a matching fine-tuned checkpoint with the task head.

### Local inference tutorial

- No patient cases or private metadata are distributed with this repository.
- Put locally permitted sample files under:
  - `demo/release/local_cases/`
- Validate a local CSV before running the notebook:
  - `python demo/release/validate_case_csv.py demo/release/local_cases/cases.csv`
- The local tutorial template in this repo uses:
  - `demo/release/local_cases/cases.csv` (user-provided; ignored by git)
  - `demo/release/README.md`
  - `demo/release/abdomennet_local_inference.ipynb`

## Training & Evaluation

Entry point: `finetune/run/run_trainer.py`, configured entirely by a YAML config.

1) Prepare a config

- Template: `finetune/configs/abdomennet.yaml`. Key fields include:
  - `data.train_dataset / val_dataset / test_dataset`: local paths to the list files (.txt/.csv)
  - `model.num_classes`: number of tasks/labels (must match your data)
  - `model.pretrained_weights`: optional ViT pretrained path (empty = random init of backbone)
  - `optim.loss_type`: BCE (multi-label) / CE (multi-class) / CE_Focal / BCE_Focal
  - `trainer`: trainer/evaluator, default `FinetuneTrans25D_Trainer`

You can also copy and customize:

```yaml
# my_abdomen.yaml (example)
data:
  train_dataset: "data/train_list.txt"
  val_dataset:   "data/val_list.txt"
  test_dataset:  "data/test_list.txt"
  batch_size: 16
  num_workers: 8
crops:
  global_crops_size: 518
model:
  arch: vit_giant2
  patch_size: 14
  num_classes: 11
  use_n_blocks: 4
  use_avgpool: true
  num_decoder_layers: 3
  trans_nhead: 16
  trans_dim_feedforward_ratio: 4
  pretrained_weights: "checkpoints/backbone.pth"   # optional
optim:
  loss_type: BCE
  epochs: 30
  lr: 5e-4
  weight_decay: 5e-4
  lr_scheduler: cosine
trainer: FinetuneTrans25D_Trainer
```

1) Start training

```sh
python finetune/run/run_trainer.py \
  --config-file finetune/configs/abdomennet.yaml \
  --output-dir outputs/abdomennet
```

Logs, TensorBoard, and prediction `.npz` files are written under `--output-dir` (subdirs: `tensorboard/`, `pred_npz/`).

Resume training:

```sh
python finetune/run/run_trainer.py \
  --config-file finetune/configs/abdomennet.yaml \
  --output-dir outputs/abdomennet \
  --resume
```

1) Evaluate a checkpoint

Set `trainer: EvalTrans25D_Trainer` and provide `model.ckpt_path`:

```yaml
trainer: EvalTrans25D_Trainer
model:
  ckpt_path: "outputs/abdomennet/checkpoints/epoch_XX.pth"
```

Run:

```sh
python finetune/run/run_trainer.py \
  --config-file configs/my_eval.yaml \
  --output-dir outputs/eval_run
```

The log will print per-task metrics (e.g., AUC, mAP, F1, ACC) and their averages across tasks.

## Config Overview

- `data.*`: dataset list paths, batch size, workers, optional `random_z_flip` augmentation
- `crops.global_crops_size`: input resolution for slices (random resized crop for training; resize for eval)
- `model.*`: ViT backbone + 2.5D decoder (last `use_n_blocks` CLS tokens + optional mean patch token → linear reduction → Transformer decoder → classifier)
- `optim.*`: optimizer, LR/scheduler, loss types (Focal and weighted BCE supported)
- `trainer`: `FinetuneTrans25D_Trainer` (train/val/test) or `EvalTrans25D_Trainer` (test only)

## Acknowledgements

The pre-training code in `pretrain/` builds on the open-source
[DINOv2](https://github.com/facebookresearch/dinov2) project from Meta AI
Research/FAIR. DINOv2 code and model weights are released under the Apache
License 2.0, and the redistributed DINOv2-derived source files retain the
original Meta copyright and Apache-2.0 license notices.

## License

The DINOv2-derived pre-training code and any DINOv2 initialization weights are
governed by the Apache License 2.0; see `pretrain/LICENSE` and the copyright
headers retained in `pretrain/dinov2/`. Please also cite or acknowledge the
original DINOv2 work when using this pre-training component.

## Statistical Analysis

Reusable statistical-analysis utilities that match the paper protocol are provided under `statistics/`.

- `statistics/bootstrap_ci.py`
  - Bootstrap 95% CIs (default `10,000` iterations) for AUROC, sensitivity, and specificity.
- `statistics/delong.py`
  - Paired DeLong test for comparing AUROCs from unaided vs AI-assisted sessions.
- `statistics/mrmc_glmm.py`
  - Reader-study generalized linear mixed-effects analysis with reader and case random effects.
- `statistics/nonparametric_tests.py`
  - Wilcoxon signed-rank test for paired reading-time comparisons.
  - Mann-Whitney U test for independent workflow-turnaround cohorts.
- `statistics/README.md`
  - Expected CSV schemas and example commands.

The statistical-analysis scripts intentionally do **not** include any experimental results. They provide templates and CLI utilities for reproducing the analysis workflow described in the paper.
