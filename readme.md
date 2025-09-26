# A multi‑task deep learning system for diagnosis, risk stratification, and emergency triage of acute abdomen on non‑contrast CT

This is the official repository for the paper “A multi‑task deep learning system for diagnosis, risk stratification, and emergency triage of acute abdomen on non‑contrast CT.”

The repo contains:

- Pre-training: DINOv2-based ViT backbone pretraining code in `pretrain/`.
- Fine-tuning & evaluation: Multi-task classification on non-contrast abdominal CT in `finetune/`, using a 2.5D Transformer decoder to model full volumes and supporting multi-label/multi-task setups.

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
# Download and unzip from anonymous repository (temporary link before publication)
wget https://anonymous.4open.science/r/Acute_abdomen/download -O Acute_abdomen.zip
unzip Acute_abdomen.zip
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
/abs/path/to/study_0001.npy 1 0 0 1 0 1 0 0 1 0 1
/abs/path/to/study_0002.npy 0 1 1 0 0 0 1 0 0 1 0
```

Example (single-task binary):

```text
/abs/path/to/study_0003.npy 1
/abs/path/to/study_0004.npy 0
```

Notes:

- NIfTI `.nii.gz` volumes are loaded and converted to grayscale slices along the z-axis, then stacked to 3-channel images for augmentation/normalization.
- Training samples up to 300 slices per volume with a stride; evaluation can set different `max_slices` via evaluation trainers.

1) Model checkpoints

- Pretrained backbone (optional): set `model.pretrained_weights` to a ViT DINOv2-style checkpoint. The loader supports either:
  - A direct `pos_embed` state dict; or
  - A dict with `teacher.backbone.*` keys (automatically mapped to the model backbone).
- Fine-tuning checkpoints: saved under `--output-dir/checkpoints/epoch_*.pth`. For evaluation, set `model.ckpt_path` to the desired checkpoint.

## Training & Evaluation

Entry point: `finetune/run/run_trainer.py`, configured entirely by a YAML config.

1) Prepare a config

- Template: `finetune/configs/abdomennet.yaml`. Key fields include:
  - `data.train_dataset / val_dataset / test_dataset`: absolute paths to the list files (.txt/.csv)
  - `model.num_classes`: number of tasks/labels (must match your data)
  - `model.pretrained_weights`: optional ViT pretrained path (empty = random init of backbone)
  - `optim.loss_type`: BCE (multi-label) / CE (multi-class) / CE_Focal / BCE_Focal
  - `trainer`: trainer/evaluator, default `FinetuneTrans25D_Trainer`

You can also copy and customize:

```yaml
# my_abdomen.yaml (example)
data:
  train_dataset: "/abs/path/train_list.txt"
  val_dataset:   "/abs/path/val_list.txt"
  test_dataset:  "/abs/path/test_list.txt"
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
  pretrained_weights: "/abs/path/to/dinov2_vitg14.pth"   # optional
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
  --output-dir /abs/path/to/outputs/abdomennet
```

Logs, TensorBoard, and prediction `.npz` files are written under `--output-dir` (subdirs: `tensorboard/`, `pred_npz/`).

Resume training:

```sh
python finetune/run/run_trainer.py \
  --config-file finetune/configs/abdomennet.yaml \
  --output-dir /abs/path/to/outputs/abdomennet \
  --resume
```

1) Evaluate a checkpoint

Set `trainer: EvalTrans25D_Trainer` and provide `model.ckpt_path`:

```yaml
trainer: EvalTrans25D_Trainer
model:
  ckpt_path: "/abs/path/to/outputs/abdomennet/checkpoints/epoch_29.pth"
```

Run:

```sh
python finetune/run/run_trainer.py \
  --config-file /abs/path/to/my_eval.yaml \
  --output-dir /abs/path/to/outputs/eval_run
```

The log will print per-task metrics (e.g., AUC, mAP, F1, ACC) and their averages across tasks.

## Config Overview

- `data.*`: dataset list paths, batch size, workers, optional `random_z_flip` augmentation
- `crops.global_crops_size`: input resolution for slices (random resized crop for training; resize for eval)
- `model.*`: ViT backbone + 2.5D decoder (last `use_n_blocks` CLS tokens + optional mean patch token → linear reduction → Transformer decoder → classifier)
- `optim.*`: optimizer, LR/scheduler, loss types (Focal and weighted BCE supported)
- `trainer`: `FinetuneTrans25D_Trainer` (train/val/test) or `EvalTrans25D_Trainer` (test only)

## License

MIT License
