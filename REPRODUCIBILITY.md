# Reproducibility

This repository includes the Tractomamba inference code and a trained
checkpoint for direct inference.

## Inference Reproducibility

Inference can be reproduced when the following are fixed:

- Checkpoint: `trainedmodel/tractomamba_pretrained.pth`
- Model configuration: `configs/model_config.json`
- Registered input tractography file
- Python, CUDA, PyTorch, `mamba-ssm`, and `pytorch3d` versions
- Inference arguments such as `--batch_size`, `--k_ds_rate`, and `--device`

Input files should already be registered/aligned to the model's expected space.

## Training Reproducibility

This repository is packaged for inference. Training from scratch requires
resources outside this release:

- TractCloud training and validation datasets
- The original train/validation split files
- The original training script and hyperparameters
- The same preprocessing and augmentation settings
- Fixed random seeds and compatible CUDA/PyTorch/library versions

Because CUDA kernels and data loading can introduce nondeterminism, retraining
may produce slightly different metrics even with the same configuration.
