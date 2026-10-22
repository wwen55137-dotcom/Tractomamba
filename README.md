# Tractomamba

Tractomamba is a sequence-aware algorithm for whole-brain parcellation of
streamlines from diffusion MRI tractography into anatomically defined white
matter fiber bundles.

## Release Status

**The core model implementation, trained model, and pretrained weights will be
made publicly available after the associated paper is accepted.**

Until then, these materials are withheld from the public release. The current
repository contains supporting code, configuration, and label metadata only.
Training and inference require the unreleased model implementation and weights
and cannot be run using the public repository alone. Installation and usage
instructions will be provided with the full release after paper acceptance.

## Public Repository Contents

- `configs/`: model configuration.
- `datasets/`: data-loading utilities and tract label metadata.
- `models/README.md`: core model release notice.
- `trainedmodel/README.md`: trained model and weights release notice.
- `utils/`: supporting utilities.
- `run_inference.py`: inference entry point; requires the forthcoming model release.
- `train.py`: training entry point; requires the forthcoming model release.
- `pyproject.toml` and `requirements.txt`: package metadata and dependencies.

## Data Availability

Training and validation datasets are not included in this repository.
The ORG atlas used in training is available at <http://dmri.slicer.org/atlases/>.

## License

The currently published files are subject to `LICENSE`.
