# Data Availability

The experiments use the publicly available AeroScapes dataset. This repository does not redistribute AeroScapes images or labels; users should obtain the dataset from the official source and set the local paths in `configs/experiment_protocol.json`.

This repository contains the fixed split files, experiment configurations, source code, compact summary tables, and Dijkstra consistency results needed to inspect and rerun the main protocol.

The selected trained checkpoints are provided separately as GitHub release assets under tag `v1.0.0`:

`https://github.com/lzh253/uav-rgbd-semantic-communication/releases/tag/v1.0.0`

Generated monocular-depth arrays, prediction previews, full per-sample outputs, and historical training runs are not tracked in the repository. They are derived artifacts and can be regenerated from the public dataset with the provided scripts.
