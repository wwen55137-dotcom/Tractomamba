# Tractomamba

Tractomamba is an inference package for tractography parcellation using the Tractomamba model architecture. It takes a registered whole-brain tractography file in VTK/VTP format and writes predicted tract bundles as separate `.vtp` files.

This repository publishes the model architecture, inference code, runtime configuration, and label metadata. Trained model checkpoints are not included. Pass a private or separately distributed checkpoint at inference time with `--weight_path`.

## Contents

- `run_inference.py`: command-line inference entry point for registered tractography data
- `models/Tractomamba.py`: Tractomamba model definition
- `configs/model_config.json`: architecture settings used to rebuild the model
- `weights/README.md`: note about checkpoint placement
- `datasets/tract_cluster_labels.xlsx`: cluster-to-tract label metadata
- `utils/`: runtime utilities for feature extraction, label mapping, logging, and output writing

## Public Release Scope

This repository is intended to release code only:

- Included: inference code, model definition, configuration, label metadata, requirements, and documentation
- Not included: trained checkpoints, local experiment outputs, logs, cache files, or private model artifacts
- Training and validation data: available from the TractCloud GitHub release/dataset repository

With the same checkpoint, configuration, environment, and registered input tractography, this package can reproduce the inference workflow. To reproduce training from scratch, use the public TractCloud training/validation data together with the original training protocol and hyperparameters.

## Requirements

The package has been tested in a CUDA environment. CPU inference is not supported because the installed `mamba_ssm` kernels require CUDA.

Install the Python dependencies from this folder:

```bash
pip install -r requirements.txt
```

The main dependencies are:

- PyTorch
- NumPy
- pandas and openpyxl
- VTK
- whitematteranalysis
- pytorch3d
- mamba-ssm

Depending on your CUDA and PyTorch versions, `pytorch3d` and `mamba-ssm` may need version-specific installation commands.

For stricter reproducibility, record the exact Python, CUDA, PyTorch, `mamba-ssm`, and `pytorch3d` versions used in your environment.

## Input

The inference script expects a whole-brain tractography file:

- Format: `.vtp` or `.vtk`
- Streamline coordinates: RAS coordinate convention
- Input data should already be registered/aligned to the model's expected space
- The script resamples streamline features internally according to the saved model configuration
- The script does not recenter the input tractography

## Run Inference

From this folder:

```bash
python run_inference.py \
  --tractography_path /path/to/input.vtp \
  --weight_path /path/to/best_tract_f1_model.pth \
  --out_path ./outputs/example \
  --device cuda:0
```

You can also let the script select CUDA automatically:

```bash
python run_inference.py \
  --tractography_path /path/to/input.vtp \
  --weight_path /path/to/best_tract_f1_model.pth \
  --out_path ./outputs/example \
  --device auto
```

## Output

The predicted tract `.vtp` files are written under:

```text
outputs/example/predictions/
```

The script also writes logs under:

```text
outputs/example/log/
```

## Notes

- This package is for research use.
- `configs/model_config.json` stores only the architecture settings needed to rebuild the model for inference.
- Trained checkpoints (`*.pth`, `*.pt`, `*.ckpt`, `*.safetensors`, and similar files) are intentionally excluded from this public repository.
- Training/validation datasets are provided through the TractCloud GitHub resources, not duplicated in this repository.
- Generated files such as `outputs/`, `__pycache__/`, and `*.pyc` should not be committed.

## License

This package follows the license in `LICENSE`.

## Citation

If you use atlas-derived training data or label resources, please cite the white matter atlas work:

```text
Zhang, F., Wu, Y., Norton, I., Rathi, Y., Makris, N.,
O'Donnell, L. J.
An anatomically curated fiber clustering white matter atlas for
consistent white matter tract parcellation across the lifespan.
NeuroImage, 179:429-447, 2018.
```
