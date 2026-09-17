# Weights

This directory is intentionally kept empty in the public repository.

Place trained model checkpoints here only in your private local copy, or pass a checkpoint from another local path. Do not commit model checkpoints to GitHub.

To run inference, pass the checkpoint explicitly:

```bash
python run_inference.py \
  --tractography_path /path/to/input.vtp \
  --weight_path /path/to/best_tract_f1_model.pth \
  --out_path ./outputs/example
```
