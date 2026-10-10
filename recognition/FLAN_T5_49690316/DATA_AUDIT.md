# LaymanRRG data audit

## Dataset
BioLaySumm/BioLaySumm2025-LaymanRRG-opensource-track

Fields: source, images_path, radiology_report, layman_report.

## Original splits
- Training: 150,454 rows.
- Validation: 10,000 rows.
- Test: 10,537 rows.

Training and validation have no missing or blank reports or summaries.
All test reference summaries are blank, so local reference-based
evaluation cannot use the supplied test split.

## Overlap findings
- Training and validation share 2,148 distinct report texts.
- Many shared texts are routine findings, such as "No abnormalities."
- Training and validation share 138 individual image identifiers.
- No individual identifiers overlap with test in our checks.

## Preparation
Remove training rows sharing any source-qualified image or session
identifier with validation. Parse OpenI image lists before comparison.

Filtered training size: 150,316 rows.
Remaining training/validation identifier overlap: 0.

Retain repeated report wording because identical text alone does not
establish that records belong to the same patient.

## Remaining work
- Investigate patient-level identifiers or metadata.
- Create a development split that keeps related records together.
- Keep supplied validation untouched for final scored evaluation.
- Record the dataset revision for reproducibility.

Image/session identifier separation has been verified.
Patient-level separation has not been verified.