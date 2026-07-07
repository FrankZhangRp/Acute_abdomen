# AbdomenNet Local Inference Example

This folder contains a minimal local tutorial for:

- checkpoint loading
- inference on locally provided NIfTI or NumPy volumes
- a tiny smoke-training example that verifies the training entry point

## Local case files

No patient cases or private metadata are distributed from this repository.

Place only local, permitted sample files inside:

```text
demo/release/local_cases/
```

Use a CSV where the first column is the image path and the remaining columns
are 0/1 labels. Relative image paths are resolved from the CSV directory.

Example:

```text
image_path,label_0,label_1,label_2
sample_001.nii.gz,1,0,0
sample_002.nii.gz,0,1,0
```

Validate a local CSV before running the notebook:

```bash
python demo/release/validate_case_csv.py demo/release/local_cases/cases.csv
```

## Notebook tutorial

Use the notebook:

- `demo/release/abdomennet_local_inference.ipynb`

It walks through:

1. preparing relative-path local CSV files
2. writing a runnable inference config
3. running checkpoint inference
4. collecting `pred_npz/test_epoch_0.npz`
5. preparing a tiny smoke-training config

## Notes

- The notebook is designed as a reproducibility/tutorial artifact.
- The smoke-training section is intentionally tiny and exists only to validate the training entry point.
- For real training, replace the local CSV with your own train/val/test splits and use the standard `finetune/configs/abdomennet.yaml` template.
