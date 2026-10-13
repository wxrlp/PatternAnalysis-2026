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



## Untuned FLAN-T5-small smoke test

- Device: Apple MPS on an M4 Mac with 16 GB unified memory.
- Model: google/flan-t5-small.
- Total parameters: 76,961,152.
- Generation: do_sample=False, max_new_tokens=128.
- Example: first row of filtered training data.

Input report:
No infiltrates or consolidations are observed in the study.

Generated output:
Report: No infiltrates or consolidations are observed in the study.

The pipeline successfully generated text, but this example was copied
rather than rewritten in plain language. This is a single-example
functional check, not a held-out performance evaluation.



## Training computation smoke test

FLAN-T5-small successfully processed a two-example batch on Apple MPS.

- Target padding used -100 so padded positions were excluded from loss.
- Evaluation-mode forward loss: 3.2585.
- Training-mode backward loss: 3.4796.
- Shared embedding gradient norm: 0.01508.
- One AdamW update completed with learning rate 0.0001.
- Encoder final-layer normalisation parameter change norm: 0.002263.

These checks demonstrate working forward, backward, and optimiser
operations. They do not measure held-out performance or establish
that summary quality improved. Updated weights were not saved.