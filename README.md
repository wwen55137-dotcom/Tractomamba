# Tractomamba

Tractomamba is a sequence-aware algorithm for whole-brain parcellation of
streamlines from diffusion MRI tractography into anatomically defined white
matter fiber bundles. It assigns one label to each streamline in a registered
tractogram and writes the predicted tract bundles as separate `.vtp` files for
efficient large-scale inference.

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
│   └── tractomamba_pretrained.pth
├── utils/
├── run_inference.py
├── pyproject.toml
├── requirements.txt
└── README.md
```

## Included Model

The trained checkpoint is included at:

```text
trainedmodel/tractomamba_pretrained.pth
```

`run_inference.py` uses this checkpoint by default. You only need to pass
`--weight_path` if you want to use another checkpoint.

## Quick Start

Clone the repository and install the package:

```bash
git clone https://github.com/wwen55137-dotcom/Tractomamba.git
cd Tractomamba
pip install -e .
```

Prepare a registered whole-brain tractography file in `.vtp` or `.vtk` format
using the LPS coordinate convention. Then run inference with the included
checkpoint:

```bash
tractomamba \
  --tractography_path /path/to/input.vtp \
  --out_path ./outputs/example \
  --device auto
```

The predicted tract bundles will be saved in:

```text
outputs/example/predictions/
```

## Requirements

This package is intended for a CUDA-enabled Python environment. CPU inference
is not supported because the `mamba-ssm` kernels used by this model require
CUDA.

Install Tractomamba from this folder:

```bash
pip install -e .
```

This installs the `tractomamba` command-line tool and the main dependencies:

```bash
pip install torch numpy pandas openpyxl vtk whitematteranalysis pytorch3d mamba-ssm
```

Depending on your CUDA and PyTorch versions, `pytorch3d` and `mamba-ssm` may
need version-specific installation commands.

## Input Data

The inference script expects a whole-brain tractography file:

- Format: `.vtp` or `.vtk`
- Coordinate convention: LPS
- The tractography should already be registered/aligned to the model's expected
  space
- The script resamples streamline features internally according to
  `configs/model_config.json`

## Atlas Data

The ORG atlas used in training is available at:

```text
http://dmri.slicer.org/atlases/
```

## Output

Predicted tract files are written to:

```text
outputs/example/predictions/
```

The output tract files preserve the input tractogram coordinates and are written
in LPS space.

Runtime logs are written to:

```text
outputs/example/log/
```

Generated outputs and logs are ignored by Git and should not be committed.

## Notes

- This package is for research use.
- This repository provides inference code and a trained inference checkpoint.
- Training and validation datasets are not included in this repository.
- Training from scratch requires the original training data, train/validation
  split files, training script, hyperparameters, preprocessing settings, and
  compatible CUDA/PyTorch/library versions.

## License

This package follows the license in `LICENSE`.
