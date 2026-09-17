# Trained Model

This directory contains the trained Tractomamba checkpoint used by default for
inference:

```text
best_tract_f1_model.pth
```

`run_inference.py` loads this checkpoint automatically unless another path is
provided with `--weight_path`.
