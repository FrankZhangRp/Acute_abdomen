# AbdomenNet Release Demo

This folder contains a minimal public demo for:

- checkpoint loading
- inference on five sample NIfTI cases
- a tiny smoke-training example that verifies the training entry point

## Download demo cases

The current default demo download source is Google Drive:

- https://drive.google.com/drive/folders/1uNKexIEE11j7Nf5LiFkiIGaTdCjW_nEm?usp=sharing

Place the downloaded files inside:

```text
papers/revision_nc_20260329/release_demo/test_cases/
```

Expected files:

- `case1.nii.gz`
- `case2.nii.gz`
- `case3.nii.gz`
- `case4.nii.gz`
- `case5.nii.gz`
- `test_cases.csv`
- `abdo_trans_20250812_eval.yaml`

The plan is to mirror the formal model release on Hugging Face after paper acceptance.

## Notebook tutorial

Use the notebook:

- `abdomennet_release_demo.ipynb`

It walks through:

1. preparing relative-path demo CSV files
2. writing a runnable inference config
3. running release-checkpoint inference
4. collecting `pred_npz/test_epoch_0.npz`
5. preparing a tiny smoke-training config

## Notes

- The notebook is designed as a reproducibility/tutorial artifact.
- The smoke-training section is intentionally tiny and exists only to validate the public training entry point.
- For real training, replace the demo CSV with your own train/val/test splits and use the standard `finetune/configs/abdomennet.yaml` template.
