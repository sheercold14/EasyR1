# ReMAP Offline Builders (Attr + Text Rule)

This folder contains minimal offline builders for Visual RFT style data:

- `build_attr_samples.py`: generate image-grounded attribute QA using a teacher API.
- `build_text_rule_samples.py`: generate text-only rule QA from label set + guideline text.
- `teacher_api.py`: OpenAI-compatible API client with local cache and retries.
- `reward_offline_mixed.py`: reward scaffold for `cls`, `attr`, `text_rule`.

## Data contract

Each output row follows EasyR1 JSONL style:

- `prompt`: user-facing question for student model.
- `images` (optional): image paths (relative or absolute).
- `answer`: deterministic verifier payload.
  - `task_type`: `attr` or `text_rule` (and optional `cls` passthrough).
  - `source_type`: same as task source.
  - `answer_type`: `bool` / `short_text` / `list`.
  - `correct_answer`: ground-truth answer.
  - `candidate_answers` (optional): closed answer set.
  - `keywords` (optional): positive bonus match list.
- `meta`: trace info (`sample_id`, `teacher_model`, `prompt_version`...).

## Env vars

- `REMAP_API_BASE`: OpenAI-compatible base URL.
  - Default: `https://dashscope.aliyuncs.com/compatible-mode/v1`
- `REMAP_API_MODEL`: model name.
  - Default: `qwen-vl-max-latest`
- API key options:
  - `REMAP_API_KEY`
  - or `REMAP_API_KEY_JSONL` (first non-empty JSONL row with field `key`)

## Quick start

```bash
python3 Comparative-R1/scripts/ReMAP/build_attr_samples.py
python3 Comparative-R1/scripts/ReMAP/build_text_rule_samples.py
```

Default configs:
- `Comparative-R1/scripts/ReMAP/config/attr_config.json`
- `Comparative-R1/scripts/ReMAP/config/text_rule_config.json`

Override config path with env var:
- `REMAP_ATTR_CONFIG=/path/to/attr_config.json`
- `REMAP_TEXT_RULE_CONFIG=/path/to/text_rule_config.json`
