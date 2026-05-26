# Task-Oriented RGB-D Semantic Communication for UAV-to-UGV Traversability

This repository provides the code and fixed evaluation protocol for a UAV-to-UGV traversability mapping study. The experiments test whether a compact RGB-D semantic bottleneck can preserve information that is useful for downstream ground-route planning, rather than optimizing only pixel-level reconstruction or segmentation scores.

The main experimental pipeline compares RGB-only, depth-only, naive RGB-D concatenation, and task-oriented RGB-D semantic bottleneck models on AeroScapes road traversability masks. Model outputs are evaluated with segmentation metrics and closed-loop path-planning safety metrics based on A* and Dijkstra search.

## Repository Layout

- `configs/`: training, evaluation, channel-stress, calibration, and path-planning configurations.
- `protocol/`: fixed train/validation/test split and sample index used in the experiments.
- `scripts/`: data preparation, monocular-depth generation, model training, evaluation, path planning, and diagnostic scripts.
- `reports/`: compact machine-readable summaries retained for reproducibility checks.
- `results/planner_runs/dijkstra_confirm_v1/`: Dijkstra consistency tables used to verify that planner ranking is not only an A* heuristic artifact.

## Data Preparation

Download AeroScapes from its official source and edit `configs/experiment_protocol.json`:

```json
{
  "dataset_root": "path/to/AeroScapes",
  "outputs_root": "path/to/local/output"
}
```

The expected dataset layout follows the standard AeroScapes folders, including `JPEGImages`, `SegmentationClass`, `Visualizations`, and `ImageSets`.

Depth priors are generated locally from the RGB images:

```bash
python scripts/dataset_audit.py --config configs/experiment_protocol.json
python scripts/generate_raw_depth.py --config configs/experiment_protocol.json --split all
python scripts/audit_raw_depth.py --config configs/experiment_protocol.json
```

## Training and Evaluation

Create the environment:

```bash
conda env create -f environment.yml
conda activate rgbd-semantic-uav
```

Typical runs:

```bash
python scripts/train_baselines.py --mode rgb --config configs/baseline_training_long_cuda_safe.json
python scripts/train_baselines.py --mode rgbd --config configs/baseline_training_long_cuda_safe.json
python scripts/train_semantic_comm.py --variant clean --config configs/semantic_comm_training.json
python scripts/evaluate_path_planning.py --config configs/path_planning_eval.json
```

Additional scripts reproduce the threshold sweeps, channel stress tests, calibration analysis, spatial-risk diagnostics, post-processing ablations, multi-seed checks, and Dijkstra consistency evaluation reported in the manuscript.

## Checkpoints

The selected paper checkpoints are distributed through the GitHub release page:

`https://github.com/lzh253/uav-rgbd-semantic-communication/releases/tag/v1.0.0`

Download:

- `uav-rgbd-baseline-checkpoints-v1.0.0.zip`
- `uav-rgbd-semantic-checkpoints-v1.0.0.zip`

After extracting the archives, update the `checkpoint` fields in the evaluation configs to point to the local `.pth` files. The checkpoint archives are kept outside Git history so that the repository remains quick to clone.

## Notes

The 128 x 128 input setting should be read as a compact route-level semantic message for low-bandwidth UAV-to-UGV communication experiments, not as a final obstacle-avoidance map for deployment. The channel variants in this codebase are feature-level stress tests; they are not a complete physical UAV radio-link model.
