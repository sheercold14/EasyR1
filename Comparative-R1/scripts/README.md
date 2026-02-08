# Comparative-R1 Scripts

## `ominimed_expertv2.py`

`ominimed_expertv2.py` is a migrated copy of:

- `scripts/OminiExpert/omnimed_expert.py`

and is extended in this directory with:

- per-label K-shot few-shot export (`--fewshot-shots`)
- few-shot train + rest-as-test mode (`--fewshot-as-train-rest-as-test`)
- DTD-style optionless prompt rewrite (`--prompt-style dtd`)

The original `scripts/OminiExpert/omnimed_expert.py` is not modified by this script.

## Quick Start

### 1) Build base dataset + split + 4-shot

```bash
python3 Comparative-R1/scripts/ominimed_expertv2.py build-base \
  --omni_root /path/to/OmniMedVQA \
  --dataset-regex ISIC2018 ISIC2019 ISIC2020 \
  --question-type "Disease Diagnosis" \
  --seed 42 \
  --fewshot-as-train-rest-as-test \
  --fewshot-shots 4 \
  --prompt-style dtd \
  --out-dir /path/to/output_dir
```

Outputs include:

- `train.jsonl`
- `val.jsonl`
- `test.jsonl`
- `train_fewshot_4shot.jsonl`
- `summary.json`

Notes:

- `--fewshot-ratio` (original behavior) writes an extra few-shot subset from `train.jsonl`.
- `--fewshot-shots` writes `train_fewshot_{K}shot.jsonl` from `train.jsonl`.
- If `--fewshot-as-train-rest-as-test` is enabled, `train/test` are rebuilt directly from all rows:
  - `train.jsonl`: K-shot per label
  - `test.jsonl`: remaining rows
  - `val.jsonl`: empty
  - in this mode no extra `train_fewshot_{K}shot.jsonl` is produced.

If you want few-shot train with test as all remaining samples (instead of split-based test), use:

```bash
python3 Comparative-R1/scripts/ominimed_expertv2.py build-base \
  --omni_root /path/to/OmniMedVQA \
  --dataset-regex "ISIC" \
  --question-type "Disease Diagnosis" \
  --seed 42 \
  --fewshot-shots 4 \
  --fewshot-as-train-rest-as-test \
  --out-dir /path/to/output_dir
```

In this mode:

- `train.jsonl` = per-label 4-shot subset
- `test.jsonl` = all remaining samples
- `val.jsonl` = empty

### 2) Rewrite single-image MCQ rows to optionless text labels (DTD style prompt)

```bash
python3 Comparative-R1/scripts/ominimed_expertv2.py build-optionless \
  --input /path/to/output_dir/train_fewshot_4shot.jsonl \
  --output /path/to/output_dir/train_fewshot_4shot_optionless_dtd.jsonl \
  --prompt-style dtd
```

Prompt style choices:

- `omnimed` (default)
- `dtd` (question + `Please choose one from list [ ... ]`)

## CLI Additions vs Original Script

- `build-base`:
  - `--fewshot-shots <int>`: sample exactly K instances per label from `train.jsonl`
  - `--fewshot-as-train-rest-as-test`: use K-shot per label as `train`, and put remaining rows into `test`
- `build-optionless`:
  - `--prompt-style {omnimed,dtd}`
- `build-train-mix` config:
  - `optionless_prompt_style: dtd` (optional)

## Sanity Checks

```bash
python3 -m py_compile Comparative-R1/scripts/ominimed_expertv2.py
python3 Comparative-R1/scripts/ominimed_expertv2.py -h
python3 Comparative-R1/scripts/ominimed_expertv2.py build-base -h
python3 Comparative-R1/scripts/ominimed_expertv2.py build-optionless -h
```
