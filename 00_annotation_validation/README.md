# Annotation Validation — `00_annotation_validation`

## Purpose

This folder contains the full, reproducible pipeline used to validate the
reliability of the ground-truth labels in the five candidate Persian
sentiment/emotion datasets evaluated in this study (see Section 3.1.2,
*Sampling and Manual Accuracy Evaluation*, and Table 1 of the paper).

For each dataset, a fixed-seed, class-stratified sample of texts was drawn
and independently labeled — blind to the original dataset label — by two
annotators, following a written annotation guideline. Inter-annotator
agreement (Cohen's kappa) was computed on the independent labels, after
which the two annotators reconciled every disagreement into a single
consensus label, compared against the original dataset label to produce
the final labeling accuracy reported in the paper.

The files below reflect the **completed** pipeline: the CSVs in
`3_cohen_kappa_files/` and `4_final_accuracy_files/` are the final, fully
filled-in annotations, not blank templates. (Note for anyone re-running
the scripts: `*_generate_samples.py` and `*_compute_kappa.py` will
regenerate *blank* versions of these files — annotator columns and the
`consensus` column empty, ready to be filled in — as the first step of
reproducing the process from scratch.)


## The pipeline, in three phases

The same three-step process was run once per dataset (d1–d5), which is
why the scripts are numbered `01`–`15`: three scripts per dataset, in
order.

1. **`*_generate_samples.py`** — draws a fixed-seed, class-stratified
   sample from the raw dataset (`1_datasets_files/`). Deterministic and
   checksummed, so the sample is independently reproducible.
2. **`*_compute_kappa.py`** — after the two annotators label the sample
   independently and blind (using the guideline in
   `2_annotation_guidelines/`), this validates their files and computes
   Cohen's kappa and percent agreement, both between annotators and
   against the original label. Outputs the combined file used in step 3.
3. **`*_final_accuracy.py`** — after the annotators discuss every
   disagreement and record a final `consensus` label together, this
   validates that column and computes final labeling accuracy
   (`consensus` vs. `ground_truth`) — the figure reported in the paper.

## Why this design

- **Fixed seed** make the sampling step independently
  reproducible.
- **Blind, independent labeling before reconciliation** is what makes
  Cohen's kappa meaningful — it measures genuine agreement, not agreement
  inflated by anchoring on the original label or on each other.
- **A written guideline per dataset** ensures label decisions follow
  explicit, documented criteria rather than ad hoc judgment.
- **Separating agreement from adjudication** preserves and reports the
  disagreement rate itself, instead of it disappearing into a merged
  label.

## Annotators

Two native Persian speakers labeled every dataset (Annotator 1, Annotator
2). A signed agreement documenting voluntary participation and
consent to data use is retained by the corresponding author and is not
included in this repository.

## How to reproduce

For each dataset `dN` (`d1`–`d5`):

1. Run `0X_dN_generate_samples.py` to regenerate the sample from
   `1_datasets_files/datasetN.csv`.
2. Annotators label independently, using
   `2_annotation_guidelines/Annotation_Guidelines_DatasetN.md`.
3. Run `0X_dN_compute_kappa.py` to validate the annotations and compute
   Cohen's kappa.
4. Annotators discuss disagreements and fill in the `consensus` column
   together.
5. Run `0X_dN_final_accuracy.py` to compute final labeling accuracy.

## Reference

This folder supports the dataset validation described in Section 3.1.2 of
the paper and is linked from the paper's Data Availability Statement.
