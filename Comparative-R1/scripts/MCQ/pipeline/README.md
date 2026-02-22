# ISIC MCQ Pipeline README

This directory contains a reproducible pipeline for generating ISIC offline-RFT datasets from few-shot JSONL sources, then splitting by task type.

## Files and responsibilities

- `generate_isic_16shot_all.sh`
  - Generates all base and MCQ variants for 16-shot train/test.
  - True source is `Comparative-R1/scripts/ominimed_expertv2.py` (`build-fewshot-dtd`), not pre-existing JSONL snapshots.
  - It first writes DTD fewshot files into `data/offline_rft/isic/16shot/source_from_expertv2/`, then converts them to offline-RFT format.
  - Output root: `data/offline_rft/isic/16shot`.

- `split_isic_task_types.py`
  - Splits one JSONL file into:
    - `*.binary.jsonl` (Benign/Malignant tasks)
    - `*.8class.jsonl` (8-class ISIC diagnosis tasks)
  - Supports both schemas:
    - `answer.candidate_labels`
    - `answer.option_A/B/C/D`

- `run_isic_16shot_pipeline.sh`
  - One-command end-to-end runner:
    - generate all variants
    - split all variants into binary/8class

## Dataset variants generated

For both train and test sets, the pipeline generates:

- Base text dataset
  - `16shot_nothinking.jsonl`
  - `test_16shot_nothinking.jsonl`

- MCQ letter-answer dataset (A/B/C/D)
  - `MCQ_16shot_nothinking.jsonl`
  - `MCQ_test_16shot_nothinking.jsonl`

- MCQ letter-answer with full candidate labels
  - `MCQ_16shot_nothinking.full_labels.jsonl`
  - `MCQ_test_16shot_nothinking.full_labels.jsonl`

- MCQ text-answer with full candidate labels
  - `MCQ_16shot_nothinking.full_labels_text.jsonl`
  - `MCQ_test_16shot_nothinking.full_labels_text.jsonl`

- MCQ text-answer with 4-option list prompt
  - `MCQ_16shot_nothinking.4labels_text.jsonl`
  - `MCQ_test_16shot_nothinking.4labels_text.jsonl`

Each file above is also split into:

- `*.binary.jsonl`
- `*.8class.jsonl`

## How to run

Run from anywhere:

```bash
bash EasyR1/Comparative-R1/scripts/MCQ/pipeline/run_isic_16shot_pipeline.sh
```

To override Open-access location:

```bash
QA_OPEN_ACCESS_DIR=/path/to/OmniMedVQA/QA_information/Open-access \
bash EasyR1/Comparative-R1/scripts/MCQ/pipeline/run_isic_16shot_pipeline.sh
```

Optional overrides:

```bash
OMNI_ROOT=/path/to/OmniMedVQA \
SEED=42 \
bash EasyR1/Comparative-R1/scripts/MCQ/pipeline/run_isic_16shot_pipeline.sh
```

## Naming convention

- Base:
  - `{split}_nothinking.jsonl`
- MCQ:
  - `MCQ_{split}_nothinking.jsonl`
  - `MCQ_{split}_nothinking.full_labels.jsonl`
  - `MCQ_{split}_nothinking.full_labels_text.jsonl`
  - `MCQ_{split}_nothinking.4labels_text.jsonl`
- Task split suffix:
  - `.binary.jsonl`
  - `.8class.jsonl`

Where `{split}` is:

- `16shot` for training
- `test_16shot` for test

## Adapting to other datasets

If you want to apply this pipeline to another dataset (for example 4-shot, 32-shot, or another organ/task):

1. Copy `generate_isic_16shot_all.sh` to a new dataset-specific script name.
2. Update `build-fewshot-dtd` arguments in that script:
   - `--dataset-regex` / `--question-type`
   - `--shots`
   - `--min-option-count` / `--max-option-count`
   - `--candidate-source`
3. Update output variables:
   - `OUT_DIR` (target output dir)
   - `--out-stem` (DTD source filename stem)
3. Keep downstream builders unchanged unless output schema requirements differ.
4. Reuse `split_isic_task_types.py` if target labels still follow binary vs 8-class ISIC semantics.
5. If class taxonomy changes, update label sets in `split_isic_task_types.py`:
   - `EIGHT_CLASS_LABELS`
   - `BINARY_LABELS`

## Validation checklist

After generation, verify:

- Conversion stats show `written == total` for each builder.
- Split stats show `unknown == 0`.
- Row counts satisfy:
  - `original == binary + 8class`
