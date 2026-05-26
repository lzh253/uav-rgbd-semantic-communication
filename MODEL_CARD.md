# Model Notes

This repository documents the model family used in the experiments. The selected paper checkpoints are distributed as GitHub release assets instead of regular Git-tracked files.

## Task

The task is binary ground traversability mapping from UAV imagery. AeroScapes road labels are converted into passable-road masks, and RGB images are paired with generated monocular-depth priors for RGB-D experiments.

## Model Families

- RGB-only segmentation baseline.
- Depth-only segmentation baseline.
- Naive RGB-D concatenation baseline.
- Dual-branch RGB-D semantic bottleneck model with a compact intermediate representation.
- Channel-stress variants with feature quantization, additive noise, and dropout.

## Output

The models output a road-traversability probability map. A threshold converts the probability map into a binary grid for downstream route planning.

## Intended Use

The code is intended for academic reproduction of the reported segmentation and path-planning experiments. It is useful for studying how task-oriented semantic representations affect downstream route safety.

## Limitations

The models are not a complete autonomous navigation stack. The experiments do not model real UAV radio propagation, real UGV dynamics, fine obstacle avoidance, or cross-dataset generalization. The 128 x 128 setting is a compact route-level representation for communication-oriented evaluation, not a deployment-ready perception resolution.
