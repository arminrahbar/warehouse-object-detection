# Experiment output contract

`experiments/outputs/` contains generated analytical evidence. The directory is
ignored because full experiment results can be large, can include per-image
records, and depend on externally managed datasets and model assets. Public
reports and accepted figures summarize the verified decisions without exposing
machine-local paths or redistributing those external inputs.

Numeric prefixes align every output with the experiment that owns it. Stage
`00_dataset_inventory/` is a shared prerequisite consumed by Experiments 01 and
02; later experiments consume the bounded workload and decisions produced by
the preceding stages.

```text
experiments/outputs/
|-- 00_dataset_inventory/
|-- 01_model_selection/
|   |-- 01_quality_comparison/
|   |-- 02_runtime_benchmark/
|   `-- 03_checkpoint_decision/
|-- 02_dataset_analysis/
|   |-- 01_dataset_summary/
|   |-- 02_sample_selection/
|   `-- 03_overlap_analysis/
|-- 03_nms_thresholding/
|   `-- 01_threshold_sweep/
|-- 04_augmentation_robustness/
|   `-- 01_condition_evaluation/
`-- 05_hard_negative_mining/
    |-- 01_error_components/
    `-- 02_review_queues/
```

## Stage ownership

All paths in the final column are relative to `experiments/outputs/`.

| Stage | Producing entry point | Output directory |
|---|---|---|
| Dataset inventory | `scripts/02_dataset_analysis/00_build_dataset_inventory.py` | `00_dataset_inventory/` |
| Checkpoint quality | `scripts/01_model_selection/01_model_comparison.py` | `01_model_selection/01_quality_comparison/` |
| Checkpoint runtime | `scripts/01_model_selection/01_benchmark_inference.py` | `01_model_selection/02_runtime_benchmark/` |
| Checkpoint decision | `scripts/01_model_selection/01_select_checkpoint.py` | `01_model_selection/03_checkpoint_decision/` |
| Dataset summary | `scripts/02_dataset_analysis/02_summarize_dataset.py` | `02_dataset_analysis/01_dataset_summary/` |
| Sample selection | `scripts/02_dataset_analysis/02_dataset_sampling.py` | `02_dataset_analysis/02_sample_selection/` |
| Overlap analysis | `scripts/02_dataset_analysis/02_overlap_analysis.py` | `02_dataset_analysis/03_overlap_analysis/` |
| NMS threshold sweep | `scripts/03_nms_thresholding/03_nms_threshold_sweep.py` | `03_nms_thresholding/01_threshold_sweep/` |
| Input-condition evaluation | `scripts/04_augmentation_robustness/04_augmentation_robustness.py` | `04_augmentation_robustness/01_condition_evaluation/` |
| Error-component construction | `scripts/05_hard_negative_mining/05_build_hnm_components.py` | `05_hard_negative_mining/01_error_components/` |
| Review-queue construction | `scripts/05_hard_negative_mining/05_build_error_review_queues.py` | `05_hard_negative_mining/02_review_queues/` |

## Shared dataset inventory

`00_dataset_inventory/dataset_index.csv` records one validated image-and-label
pair per row together with object counts and classes present. Its companion
tables summarize class frequency and image-density counts. Keeping this shared
inventory in stage `00` avoids assigning a cross-experiment prerequisite to a
single model or operating-point decision.

## Evidence and publication boundary

Each producer validates the schema, expected population, deterministic ordering,
and numerical invariants appropriate to its stage. Downstream stages verify the
upstream decision and population they consume rather than trusting filenames
alone.

The following remain local and ignored:

- external datasets, model weights, configuration files, videos, and class
  vocabularies;
- per-image predictions, ground-truth records, inference ledgers, and model
  caches;
- temporary logs, pilots, and other regenerable intermediate artifacts; and
- generated tables that contain machine-local asset paths or are too large for
  a source repository.

The following are tracked public artifacts:

- this output-ownership contract;
- the connected experiment index and numbered engineering reports;
- accepted figures referenced by those reports; and
- the integrated project report and its curated figure package.

Run experiment entry points from the repository root. Once the documented
external inputs are available, their default output locations follow the stage
ownership table above. The public reports state the fixed populations, controls,
metrics, decisions, and limitations required to interpret the reproduced
results.
