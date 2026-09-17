# Tractomamba

Tractomamba is a tractography parcellation inference package based on the
Tractomamba model architecture. It takes a registered whole-brain tractography
file in `.vtp` or `.vtk` format and writes the predicted tract bundles as
separate `.vtp` files.

This GitHub release includes the inference code, model configuration, label
metadata, and a trained checkpoint for direct inference.

## Repository Structure

```text
Tractomamba/
├── configs/
│   └── model_config.json
├── datasets/
│   ├── dataset.py
│   └── tract_cluster_labels.xlsx
├── models/
│   └── Tractomamba.py
├── trainedmodel/
│   └── best_tract_f1_model.pth
├── utils/
├── run_inference.py
├── requirements.txt
├── REPRODUCIBILITY.md
└── README.md
```

## Included Model

The trained checkpoint is included at:

```text
trainedmodel/best_tract_f1_model.pth
```

`run_inference.py` uses this checkpoint by default. You only need to pass
`--weight_path` if you want to use another checkpoint.

## Requirements

This package is intended for a CUDA-enabled Python environment. CPU inference
is not supported because the `mamba-ssm` kernels used by this model require
CUDA.

Install dependencies from this folder:

```bash
pip install -r requirements.txt
```

Main dependencies:

- PyTorch
- NumPy
- pandas and openpyxl
- VTK
- whitematteranalysis
- pytorch3d
- mamba-ssm

Depending on your CUDA and PyTorch versions, `pytorch3d` and `mamba-ssm` may
need version-specific installation commands.

## Input Data

The inference script expects a whole-brain tractography file:

- Format: `.vtp` or `.vtk`
- Coordinate convention: RAS
- The tractography should already be registered/aligned to the model's expected
  space
- The script resamples streamline features internally according to
  `configs/model_config.json`
- The script does not recenter the input tractography

## Run Inference

From the `Tractomamba` folder:

```bash
python run_inference.py \
  --tractography_path /path/to/input.vtp \
  --out_path ./outputs/example \
  --device auto
```

To select a specific GPU:

```bash
python run_inference.py \
  --tractography_path /path/to/input.vtp \
  --out_path ./outputs/example \
  --device cuda:0
```

To use a different checkpoint:

```bash
python run_inference.py \
  --tractography_path /path/to/input.vtp \
  --weight_path /path/to/another_checkpoint.pth \
  --out_path ./outputs/example \
  --device cuda:0
```

## Output

Predicted tract files are written to:

```text
outputs/example/predictions/
```

Runtime logs are written to:

```text
outputs/example/log/
```

Generated outputs and logs are ignored by Git and should not be committed.

## Reproducibility

Inference can be reproduced with:

- The included checkpoint: `trainedmodel/best_tract_f1_model.pth`
- The included architecture config: `configs/model_config.json`
- The same registered input tractography file
- A compatible CUDA/PyTorch/`mamba-ssm`/`pytorch3d` environment

See `REPRODUCIBILITY.md` for more details.

## Notes

- This package is for research use.
- This repository provides inference code and a trained inference checkpoint.
- Training and validation datasets are not included in this repository.
- Training from scratch requires the original training data, train/validation
  split files, training script, hyperparameters, preprocessing settings, and
  compatible CUDA/PyTorch/library versions.

## License

This package follows the license in `LICENSE`.

## Citation

If you use atlas-derived training data or label resources, please cite the
white matter atlas work:

```text
Zhang, F., Wu, Y., Norton, I., Rathi, Y., Makris, N.,
O'Donnell, L. J.
An anatomically curated fiber clustering white matter atlas for
consistent white matter tract parcellation across the lifespan.
NeuroImage, 179:429-447, 2018.
```
