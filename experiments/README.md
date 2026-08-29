# Experiments

The experiment suite records the controlled decisions that define the detector
pipeline. The five numbered experiment names are consistent across `scripts/`,
`outputs/`, `figures/`, and `reports/` so each entry point, its generated
evidence, its accepted visuals, and its engineering interpretation are easy to
trace.

## Connected experiment sequence

| Stage | Engineering decision | Primary entry point | Report |
|---|---|---|---|
| 01 · Model selection | Select the checkpoint that establishes the quality and runtime baseline. | `scripts/01_model_selection/01_compare_model_quality.py` | [Report](reports/01_model_selection/REPORT.md) |
| 02 · Dataset analysis | Define a bounded workload that preserves class, density, and crowding characteristics. | `scripts/02_dataset_analysis/03_select_analysis_workload.py` | [Report](reports/02_dataset_analysis/REPORT.md) |
| 03 · NMS thresholding | Select the class-aware NMS IoU operating point. | `scripts/03_nms_thresholding/01_sweep_nms_thresholds.py` | [Report](reports/03_nms_thresholding/REPORT.md) |
| 04 · Input-shift diagnostics | Measure sensitivity to controlled image transformations. | `scripts/04_augmentation_robustness/02_measure_augmentation_robustness.py` | [Report](reports/04_augmentation_robustness/REPORT.md) |
| 05 · Hard-negative mining | Build deterministic, purpose-specific image review queues. | `scripts/05_hard_negative_mining/02_build_error_review_queues.py` | [Report](reports/05_hard_negative_mining/REPORT.md) |

The shared corpus inventory is produced by
`scripts/02_dataset_analysis/01_build_dataset_inventory.py` and stored under
output stage `00_dataset_inventory/`. It is a prerequisite consumed by multiple
experiments, not a sixth experiment.

## Directory contract

- `scripts/<numbered_experiment>/` contains that experiment's analytical entry
  points and figure builder.
- `outputs/<numbered_experiment>/` contains generated local evidence; see the
  [output contract](OUTPUTS.md).
- `figures/<numbered_experiment>/` contains the accepted public figure package.
- `reports/<numbered_experiment>/REPORT.md` contains the controlled comparison,
  decision, limitations, and implementation map.
- [`reports/README.md`](reports/README.md) is the connected report index.

Run entry points from the repository root after placing the documented external
dataset and model assets under `detector_service/storage/`. Use each script's
`--help` output for its required inputs and controls, and validate a bounded
pilot before starting an expensive full-corpus run.
