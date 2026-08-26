# Experiment Report Guide

These reports form one evidence chain for the warehouse object-detection
system. Read them in order to follow the project's engineering decisions.

Their numbered directories align with the generated-evidence directories in the
[experiment output contract](../OUTPUTS.md). Stage `00` is a shared dataset
inventory and therefore has no standalone experiment report.

| Stage | Engineering question | Report | Figures |
|---:|---|---|---|
| 1 | Which checkpoint is the stronger baseline on the locked corpus, and what does it cost to run? | [Model selection](01_model_selection/REPORT.md) | [5 figures](../figures/01_model_selection/) |
| 2 | How can the corpus be converted into a coverage-preserving analysis workload? | [Dataset analysis](02_dataset_analysis/REPORT.md) | [6 figures](../figures/02_dataset_analysis/) |
| 3 | Which NMS threshold wins the locked quality-first rule, and what output-volume and redundancy trade-off does that choice imply? | [NMS thresholding](03_nms_thresholding/REPORT.md) | [5 figures](../figures/03_nms_thresholding/) |
| 4 | How stable is the selected pipeline under controlled input shifts? | [Input-shift diagnostics](04_augmentation_robustness/REPORT.md) | [6 figures](../figures/04_augmentation_robustness/) |
| 5 | Which images should enter targeted error-review queues for diagnosis and future improvement? | [Error-review prioritization](05_hard_negative_mining/REPORT.md) | [7 figures](../figures/05_hard_negative_mining/) |

## How the experiments connect

1. Compare both checkpoints under one locked quality policy and a paired runtime
   benchmark; select Checkpoint B as the fixed model baseline.
2. Characterize the full corpus and construct a deterministic 5,000-image sample
   for the more expensive downstream evaluations.
3. Hold Checkpoint B and the sample fixed while selecting the class-aware NMS
   operating point.
4. Stress the selected pipeline with controlled image perturbations.
5. Convert the resulting image-level errors into prioritized diagnostic review
   queues.

The numerical source of truth remains the generated CSV evidence cited in each
report. The figures are presentation views of that evidence, and every report
states the limits of the conclusions it supports.

The reports and their referenced PNG figures are tracked public engineering
artifacts. Large raw predictions, caches, external assets, and regenerable
intermediate tables remain outside the public repository. Every report is
written so its code and figure links resolve from a clean repository clone.
