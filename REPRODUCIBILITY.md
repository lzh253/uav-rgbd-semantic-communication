# Reproducibility Notes

The repository fixes the data split, preprocessing assumptions, training configurations, and evaluation scripts used in the reported experiments. It is intended to make the protocol inspectable and rerunnable on a local AeroScapes installation.

## What Can Be Reproduced

- binary traversability split construction from AeroScapes labels;
- monocular-depth prior generation for RGB-D experiments;
- RGB-only, depth-only, and naive RGB-D concatenation baselines;
- RGB-D semantic bottleneck training and channel stress tests;
- A* path-planning evaluation and Dijkstra consistency checks;
- calibration, spatial-risk, post-processing, and multi-seed diagnostic analyses.

## Local Setup

1. Install the environment from `environment.yml` or `requirements.txt`.
2. Download AeroScapes and update `configs/experiment_protocol.json`.
3. Run the data audit and depth generation scripts.
4. Train the required baseline and semantic models.
5. Run the evaluation scripts for segmentation, channel stress, calibration, and path planning.

## Paths and Checkpoints

Configuration files use portable placeholders such as `AeroScapes` and `REPLACE_WITH_LOCAL_OUTPUT_ROOT`. Replace them with local paths before running.

The selected paper checkpoints are published as release assets under tag `v1.0.0`. Download and extract the two checkpoint archives, then update the `checkpoint` fields in the relevant configs to the extracted local paths.

## Expected Variation

Training is stochastic. Exact values may vary with hardware, CUDA/PyTorch versions, and random seeds. The manuscript reports validation-selected checkpoints and additional stability diagnostics to make this boundary explicit.
