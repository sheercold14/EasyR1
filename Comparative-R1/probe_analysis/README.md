# Probe Analysis (Visual Capability Layer-1)

Minimal framework for testing whether visual separability improves before/after RFT.

## What it provides

- JSONL loader for EasyR1-style samples (`images`, `answer.*`)
- Feature extractor interface (built-in + plugin)
- Probe training (`linear`, `knn`, `mlp`)
- Metrics + bootstrap CI output to `summary.json`

## Directory

- `run_probe.py`: single-entry CLI
- `datasets.py`: sample loading
- `extractors.py`: feature extraction interfaces
- `probes.py`: probe models
- `stats.py`: metrics and CI
- `custom_extractor_template.py`: plugin template

## Quick start

Run from `Comparative-R1` parent so imports work.

```bash
cd /data/shichao/EasyR1/Comparative-R1

python3 probe_analysis/run_probe.py \
  --data /data/shichao/EasyR1/data/offline_rft/isic/v1/MCQ_test_4shot_nothinking.full_labels_text.jsonl \
  --output-dir /tmp/probe_meanrgb \
  --image-root /data/shichao/data/OmniMedVQA \
  --extractor mean_rgb \
  --probes linear,knn,mlp
```

## Using precomputed features (`npz`)

`npz` must contain:

- `features`: `[N, D]`
- optional `ids`: `[N]` matching `sample_id`

If `ids` is missing, `features` must align with JSONL row order.

```bash
python3 probe_analysis/run_probe.py \
  --data /path/to/data.jsonl \
  --output-dir /tmp/probe_npz \
  --extractor npz \
  --features-npz /path/to/features.npz
```

## Plugin extractor interface

Pass `--extractor module:function`, where function returns an object with:

- `extract(samples: list[Sample]) -> np.ndarray` of shape `[N, D]`

Example template:

```bash
python3 probe_analysis/run_probe.py \
  --data /path/to/data.jsonl \
  --output-dir /tmp/probe_custom \
  --extractor probe_analysis.custom_extractor_template:build_extractor \
  --plugin-kwargs '{"dim": 128}'
```

## Recommended workflow for pre/post RFT

1. Use same dataset split seed for both checkpoints.
2. Extract features with your pre-model and run probe (`out_pre`).
3. Extract features with your post-model and run probe (`out_post`).
4. Compare `summary.json` metrics and CI.

Keep probe settings identical across runs (`--probes`, `--seed`, `--test-size`).

## Directly extract features from checkpoint

For Qwen2/2.5-VL merged actor checkpoints (`.../actor/huggingface`):

```bash
cd /data/shichao/EasyR1/Comparative-R1

python3 probe_analysis/extract_features_from_checkpoint.py   --data /mnt/cache/wuruixiao/users/lsc/EasyR1/data/offline_rft/isic/v1/MCQ_test_4shot_nothinking.4labels_text.jsonl   --image-root /mnt/cache/wuruixiao/users/lsc/data/OmniMedVQA   --checkpoint /mnt/cache/wuruixiao/users/lsc/EasyR1/checkpoints/MCQ_failure/4label_text_list_eval2_nothinking_isic_rollout_log/global_step_170/actor/huggingface   --output-npz /mnt/cache/wuruixiao/users/lsc/data/OminiMedExpert/probe_feat/isic_post_features.npz   --dtype bf16   --device auto   --trust-remote-code

python3 probe_analysis/extract_features_from_checkpoint.py   --data /mnt/cache/wuruixiao/users/lsc/EasyR1/data/offline_rft/isic/v1/MCQ_test_4shot_nothinking.4labels_text.jsonl   --image-root /mnt/cache/wuruixiao/users/lsc/data/OmniMedVQA   --checkpoint /mnt/cache/wuruixiao/users/lsc/qwen25-vl-7b  --output-npz /mnt/cache/wuruixiao/users/lsc/data/OminiMedExpert/probe_feat/isic_pre_features.npz   --dtype bf16   --device auto   --trust-remote-code
```

Then run probe from that npz:

```bash
python3 probe_analysis/run_probe.py \
  --data /mnt/cache/wuruixiao/users/lsc/EasyR1/data/offline_rft/isic/v1/MCQ_test_4shot_nothinking.4labels_text.jsonl \
  --output-dir /mnt/cache/wuruixiao/users/lsc/data/OminiMedExpert/probe_feat/probe \
  --extractor npz \
  --features-npz /mnt/cache/wuruixiao/users/lsc/data/OminiMedExpert/probe_feat/isic_post_features.npz \
  --probes linear,knn,mlp
```

```bash
  python3 -m probe_analysis.run_probe --data /mnt/cache/wuruixiao/users/lsc/EasyR1/data/offline_rft/isic/v1/MCQ_test_4shot_nothinking.full_labels_text.jsonl --output-dir /mnt/cache/wuruixiao/users/lsc/data/OminiMedExpert/probe_feat/probe --extractor npz --features-npz /mnt/cache/wuruixiao/users/lsc/data/OminiMedExpert/probe_feat/isic_pre_features.npz --probes linear,knn,mlp --group-by-candidate-labels --candidate-labels-key answer.candidate_labels --min-group-size 20 --ids-key image_paths --sample-id-key images.0 
```

Repeat with pre-RFT checkpoint and compare:

```bash
python3 probe_analysis/compare_runs.py \
  --pre /tmp/probe_pre/summary.json \
  --post /tmp/probe_post/summary.json
```
