# OminiExpert: OmniMedVQA Benchmark Builder + Comparative-RFT (B1–B7)

This folder contains a small dataset construction engine for turning **OmniMedVQA** JSON files into **EasyR1**-style `jsonl` training data, plus a generator for the **multi-image contrastive task suite (B1–B7)**.

## Script

- `EasyR1/scripts/OminiExpert/omnimed_expert.py`

Default paths assume this repo layout:

- OmniMedVQA root: `data/OmniMedVQA`
- Output root (suggested): `data/OminiMedExpert`

You can override with `--omni_root ...`.

## 1) List available datasets

```bash
python EasyR1/scripts/OminiExpert/omnimed_expert.py list
```

Dataset names correspond to `QA_information/*/*.json` **file stems** (e.g., `ISIC2018` for `ISIC2018.json`).

## 2) Inspect question_type / modality_type distributions

```bash
python EasyR1/scripts/OminiExpert/omnimed_expert.py inspect --datasets ISIC2018 ISIC2019 ISIC2020
```

## 3) Build base VQA-style JSONL + split + few-shot (steps 1–5)

Example: merge ISIC2018-2020, keep only `Disease Diagnosis`, split into train/val/test, and build a few-shot subset from train.

```bash
python EasyR1/scripts/OminiExpert/omnimed_expert.py build-base \
  --datasets ISIC2018 ISIC2019 ISIC2020 \
  --question-type "Disease Diagnosis" \
  --min-option-count 2 --max-option-count 4 \
  --split 0.7,0.0,0.3 \
  --seed 42 \
  --skip-missing-images \
  --fewshot-ratio 0.5 \
  --out-dir data/OminiMedExpert/isic_disease_diagnosis_v1_0.05
```

Outputs:

- `data/OminiMedExpert/isic2018_2020_disease_diagnosis/train.jsonl`
- `data/OminiMedExpert/isic2018_2020_disease_diagnosis/val.jsonl`
- `data/OminiMedExpert/isic2018_2020_disease_diagnosis/test.jsonl`
- `data/OminiMedExpert/isic2018_2020_disease_diagnosis/train_fewshot_0.05.jsonl` (if enabled)
- `data/OminiMedExpert/isic2018_2020_disease_diagnosis/summary.json`

Prompt format for MCQ rows asks the model to answer with the **option letter** (A/B/C/D); `answer.correct_answer` is also filled for verification.

### Notes on leakage-aware splitting

The builder writes `answer.group_id` and splits by this key:

1. If the raw JSON contains patient/subject/case-like fields (`patient_id`, `subject_id`, `case_id`, ...), it uses them.
2. Else, it groups by **3D-slice prefix** naming convention (`{ori}_{x|y|z}_{slice}`).
3. Else, it groups by **image stem** (so multiple QA items for the same image never leak across splits).

## 4) Generate Comparative-RFT B1–B7 data from train (step 6)

Generate a new `jsonl` with B1–B7 tasks from a base `train.jsonl` (or the few-shot train file):

```bash
python EasyR1/scripts/OminiExpert/omnimed_expert.py build-comparative \
  --input EasyR1/data/OminiMedExpert/isic_disease_diagnosis_v1_0.05/test.jsonl \
  --output EasyR1/data/OminiMedExpert/isic_disease_diagnosis_v1_0.05/comparative/test_b1_tasks_200.jsonl \
  --label-space-by question_type+optioncount \
  --task B1=200 --task B2=0 --task B3=0 --task B4=0 --task B5=0 --task B6=0 --task B7=0 \
  --k 4 \
  --b4-candidates 3 \
  --b7-nway 3 \
  --seed 123
```

Each generated sample includes:

- `answer.task_suite = "B"`
- `answer.task_type = "B1_target_search" | ... | "B7_support_set_nway"`
- `answer.correct_answer` (verifiable discrete target)
- `answer.label_space_key` and `answer.label_space_by` (for tracking performance by task + label space)

Note on prompts: B1/B2/B4/B5/B6 now ask the model to first output each image's label (e.g., `A: Melanoma`),
then provide a final answer line (`Final: ...`) inside `<answer>...</answer>`. B3/B7 remain single-letter.
