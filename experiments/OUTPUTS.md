# Experiment output contract

`experiments/outputs/` contains generated analytical evidence. It is excluded
from version control because the tables can be large, may contain machine-local
asset paths, and are reproducible from externally managed data and model assets.
This document is the tracked contract for that ignored directory.

The numeric prefixes align outputs with the experiment reports and figures. A
second prefix identifies a producing stage when an experiment has more than one
independent entry point.

```text
experiments/outputs/
|-- 00_dataset_inventory/
|   |-- dataset_index.csv
|   |-- class_distribution.csv
|   `-- object_count_distribution.csv
|-- 01_model_selection/
|   |-- 01_quality_comparison/
|   |   `-- <run-id>/
|   |-- 02_runtime_benchmark/
|   |   `-- <run-id>/
|   `-- 03_checkpoint_decision/
|       `-- <run-id>/
|-- 02_dataset_analysis/
|   |-- 01_dataset_summary/
|   |-- 02_sample_selection/
|   `-- 03_overlap_analysis/
|-- 03_nms_thresholding/
|   `-- 01_threshold_sweep/
|       |-- operating_point.json
|       `-- <derived threshold tables and optional managed cache package>
|-- 04_augmentation_robustness/
|   `-- 01_condition_evaluation/
`-- 05_hard_negative_mining/
    |-- 01_error_components/
    `-- 02_review_queues/
```

## Stage ownership

| Stage | Producing entry point | Canonical output directory |
|---|---|---|
| Dataset inventory | `00_build_dataset_inventory.py` | `00_dataset_inventory/` |
| Checkpoint quality | `01_model_comparison.py` | `01_model_selection/01_quality_comparison/<run-id>/` |
| Checkpoint runtime | `01_benchmark_inference.py` | `01_model_selection/02_runtime_benchmark/<run-id>/` |
| Checkpoint decision | `01_select_checkpoint.py` | `01_model_selection/03_checkpoint_decision/<run-id>/` |
| Dataset summary | `02_summarize_dataset.py` | `02_dataset_analysis/01_dataset_summary/` |
| Sample selection | `02_dataset_sampling.py` | `02_dataset_analysis/02_sample_selection/` |
| Overlap analysis | `02_overlap_analysis.py` | `02_dataset_analysis/03_overlap_analysis/` |
| NMS threshold sweep | `03_nms_threshold_sweep.py` | `03_nms_thresholding/01_threshold_sweep/` |
| Input-condition evaluation | `04_augmentation_robustness.py` | `04_augmentation_robustness/01_condition_evaluation/` |
| Error-component construction | `05_build_hnm_components.py` | `05_hard_negative_mining/01_error_components/` |
| Review-queue construction | `05_build_error_review_queues.py` | `05_hard_negative_mining/02_review_queues/` |

Paths in the final column are relative to `experiments/outputs/`. All three
model-selection stages use immutable run-ID children so a completed quality,
runtime, or checkpoint-decision package is never silently overwritten. Their
CLIs accept an output root and validate a new run ID before writing the child
package. Each stage assembles its candidate below a uniquely named
`.incomplete` child and promotes it only after the complete package passes its
schema, identity, and hash checks.

## What `dataset_index.csv` represents

`00_dataset_inventory/dataset_index.csv` is a generated manifest of the source
corpus, not the result of a model experiment. It records one validated image and
label pairing per row, including their paths, object count, and classes present.
The companion tables aggregate class frequency and image-density counts.

The inventory is stage `00` because it is shared input:

- Experiment 01 uses it to evaluate both checkpoints on the same corpus.
- Experiment 02 uses it to characterize the corpus and select a bounded
  analysis workload.
- Experiments 03–05 consume the selected-workload and derived evidence produced
  by Experiment 02 rather than rebuilding the inventory.

Keeping the inventory in its own prerequisite directory avoids implying that it
belongs exclusively to model selection or dataset analysis.

## Generated evidence versus presentation artifacts

The output contract covers derived CSV evidence, manifests, and any inference
caches retained by a particular run. It does not require one output directory
per Python file. Entry points that do not own analytical evidence map as follows:

| Entry point | Artifact role | Destination contract |
|---|---|---|
| `01_build_selection_figures.py` | Curated Experiment 01 figure package | New explicit `--output-dir`; accepted package belongs under `experiments/figures/01_model_selection/` |
| `02_build_report_figures.py` | Curated Experiment 02 figure package | New explicit `--output-dir`; accepted package belongs under `experiments/figures/02_dataset_analysis/` |
| `03_build_report_figures.py` | Curated Experiment 03 figure package | New explicit `--output-dir`; accepted package belongs under `experiments/figures/03_nms_thresholding/` |
| `04_build_report_figures.py` | Curated Experiment 04 figure package | New explicit `--output-dir`; accepted package belongs under `experiments/figures/04_augmentation_robustness/` |
| `05_build_report_figures.py` | Curated Experiment 05 figure package | New explicit `--output-dir`; accepted package belongs under `experiments/figures/05_hard_negative_mining/` |
| `build_project_report_figures.py` | Integrated public-report figure package | New explicit `--output-dir`; accepted package belongs under `docs/figures/` |
| `04_augmentation_demo.py` | Analytical diagnostic preview | `scratch/diagnostic-figures/04_augmentation_robustness/` by default |
| Experiment CLI plotting and `--figures-only` modes | Analytical diagnostic plots | Stage-specific directories under `scratch/diagnostic-figures/` |
| `report_figure_style.py` | Shared rendering library | No standalone artifact |

Bounded smoke runs and temporary investigations also belong in `scratch/`.
Report-figure builders refuse to overwrite an existing package.

Canonical report evidence belongs in the numbered output directories above.
Temporary `scratch/` paths should not be used as the long-term source named by a
report.

## Reproduction and provenance

Run entry points from the repository root. Their defaults implement this
directory contract; command-line path options may still redirect a run when an
external evidence store is required. Where a stage emits a manifest, treat the
manifest and its hashes as the identity of that package. Moving a completed
package does not rewrite historical absolute paths embedded in an existing
manifest; create a new verified run when a fully self-consistent relocated
manifest is required.

The output directory is intentionally ignored. Verify its local contents with
the producing script's schema and manifest checks rather than treating directory
presence alone as evidence that a run completed successfully.

Under the standard repository layout, new inventories serialize asset paths
with the canonical `detector_service/storage` prefix. With a custom
`--asset-root`, they serialize portable paths relative to that root. Producers
and readers share this canonical contract; host-specific and historical
repository prefixes are not valid evidence paths.

### Evidence guarantees by stage

| Evidence boundary | Enforced contract |
|---|---|
| Experiment 01 quality, paired runtime, and checkpoint decision | Immutable run-ID packages; exact artifact sets; schema, row-count, byte-size, and SHA-256 verification; locked policies; cross-stage recomputation; incomplete-directory staging and atomic promotion |
| Experiment 03 operating point | The selected threshold and recorded metrics are recomputed from exactly two package-local, hashed seven-row evidence tables; the record is linked to the verified checkpoint decision |
| Experiment 03 raw-inference replay | Ground truth, raw predictions, and a one-row-per-image completeness ledger require a content-addressed manifest binding the selected-index bytes and rows, checkpoint decision, model assets, ordered vocabulary, and candidate policy |
| Experiment 04 raw-inference replay | Every condition's ground truth, raw predictions, and completeness ledger require one manifest binding the exact five-condition contract, selected index, checkpoint, operating point, vocabulary, policy, artifact hashes, and row counts |
| Experiments 02, 04 derived evidence, and 05 review evidence | Exact schemas, population/count reconciliation, deterministic ordering, numerical invariants, and report-builder source hashes; these historical retained folders are not immutable run-ID packages |

The migrated canonical evidence intentionally omits regenerable raw-prediction
caches captured before post-processing. It therefore does not contain the new
Experiment 03 inference-cache manifest or Experiment 04 condition-cache
manifest. `--refresh-postprocessing` avoids model inference only when a complete
matching manifest-backed cache package is supplied or rebuilt. A partial or
unmanifested historical cache is rejected rather than trusted by filename.

By contrast, `--figures-only` and the report-figure builders read the retained
derived tables and can render diagnostic or curated figures without model
inference. They validate exact schemas, expected row populations, locked policy
fields, and cross-table invariants before rendering; figure builders
additionally lock the accepted evidence bytes in their generated manifests.
