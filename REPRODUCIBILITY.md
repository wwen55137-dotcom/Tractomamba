# Reproducibility

This repository is a public code release without trained model checkpoints.

## Inference Reproducibility

Inference can be reproduced when the following are available:

- The same Tractomamba checkpoint passed with `--weight_path`
- The same `configs/model_config.json`
- The same registered input tractography file
- A compatible CUDA environment with PyTorch, `mamba-ssm`, and `pytorch3d`

The public repository does not include pretrained weights.

## Training Reproducibility

Training from scratch requires resources outside this inference package:

- TractCloud training and validation datasets from the TractCloud GitHub resources
- The same train/validation split files
- The original training script and hyperparameters
- The same preprocessing and augmentation settings
- Fixed random seeds and compatible CUDA/PyTorch/library versions

Because CUDA kernels and data loading can introduce nondeterminism, retraining may produce slightly different metrics even with the same configuration.

