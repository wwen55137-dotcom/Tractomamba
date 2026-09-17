# Publishing Checklist

Use this checklist before uploading this folder to GitHub.

- Keep source code, configuration, metadata, and documentation.
- Keep `weights/README.md` as the only file under `weights/`.
- Do not publish trained checkpoints such as `*.pth`, `*.pt`, `*.ckpt`, `*.safetensors`, or `*.onnx`.
- Do not publish generated outputs, logs, cache files, or local experiment data.
- If using Git, run `git status --ignored` before committing and confirm that model weights are ignored.

The trained checkpoint that was previously under `weights/` has been moved out of this public package.
