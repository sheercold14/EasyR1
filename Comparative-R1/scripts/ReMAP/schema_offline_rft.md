# Offline RFT Schema (ReMAP v1)

## Row format

Each JSONL row should be:

```json
{
  "prompt": "...",
  "images": ["Images/ISIC2019/train/xxx.jpg"],
  "answer": {
    "task_type": "attr",
    "source_type": "attr",
    "answer_type": "bool",
    "correct_answer": "yes"
  },
  "meta": {
    "source_type": "attr",
    "sample_id": "attr_000001_00",
    "teacher_model": "your-model",
    "prompt_version": "attr_v1"
  }
}
```

`images` can be omitted for text-only tasks (`text_rule`).

## Task types

1. `attr`
- Source: image + task prompt.
- Answers: `bool`, `short_text`, `list`.
- Deterministic key: `answer.correct_answer`.

2. `text_rule`
- Source: guideline text + label set.
- Answers: `bool`, `short_text`, `list`.
- Optional: `answer.keywords`, `answer.evidence_snippet`.

3. `cls` (optional passthrough)
- Existing classification rows can be mixed if they follow same `answer.correct_answer`.

## Reward routing

- Route by `ground_truth.task_type`.
- Reward parser reads only `ground_truth` and model `<answer>...</answer>` text.
- `text_rule` has non-negative shaping in v1 (`r_text >= 0`).
