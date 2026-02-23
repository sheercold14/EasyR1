# Probe Analysis (RFT Capability Diagnostics)

Minimal framework to diagnose what changes before/after RFT, and where (vision vs fusion vs mid/late LLM).

This folder supports:
- Extract features from a merged HF checkpoint (`.../actor/huggingface`) into `.npz`
- Run lightweight probes (`linear`, `knn`, `mlp`) with bootstrap CIs
- Optionally group evaluation by `answer.candidate_labels` schema (e.g. 2-way vs 8-way vs 4-way)

All command examples below use the paths on the `/mnt/cache/...` machine.

## Scripts

- `probe_analysis/extract_features_from_checkpoint.py`: checkpoint -> `npz`
- `probe_analysis/hf_extractors.py`: Qwen2/2.5-VL feature taps
- `probe_analysis/run_probe.py`: `jsonl + npz` -> `summary.json`
- `probe_analysis/compare_runs.py`: compare two summaries (pre vs post)

## Alignment keys (important)

If your `.npz` and `.jsonl` are not the same file, align by a stable key.

Recommended:
- `--sample-id-key images.0` (use image path as sample_id)
- `--ids-key image_paths` (use `npz['image_paths']` for lookup)

This lets you run probe on a different JSONL as long as `images.0` matches `npz['image_paths']`.

## Step 1: Extract features from checkpoint (to npz)

Run from the `Comparative-R1` root and prefer module mode (`-m`).

```bash
cd /mnt/cache/wuruixiao/users/lsc/EasyR1/Comparative-R1

# Tap A: vision tower output (mean-pool visual tokens)
python3 -m probe_analysis.extract_features_from_checkpoint \
  --data /mnt/cache/wuruixiao/users/lsc/EasyR1/data/offline_rft/isic/v1/MCQ_test_4shot_nothinking.full_labels_text.jsonl \
  --sample-id-key images.0 \
  --image-root /data/shichao/data/OmniMedVQA \
  --checkpoint /mnt/cache/wuruixiao/users/lsc/EasyR1/checkpoints/.../global_step_xxx/actor/huggingface \
  --output-npz /mnt/cache/wuruixiao/users/lsc/data/OminiMedExpert/probe_feat/isic_post_vision_mean.npz \
  --tap vision_mean \
  --prompt-key prompt \
  --dtype bf16 \
  --device auto \
  --trust-remote-code
```

### Extra taps (more comprehensive)

`hidden_mean` runs a forward pass with `output_hidden_states=True`, then mean-pools selected tokens at a chosen layer.
If you want to compute multiple hidden-state taps in one run, use `--taps` (repeatable). Hidden-state taps share one forward pass per image.

Typical use:
- Tap B: fusion embedding (layer 0, image tokens)
- Tap C: mid/late LLM (layer 16 or -1, image tokens)

```bash
# Multi-tap in one run (shared forward for all hs:* taps)
python3 -m probe_analysis.extract_features_from_checkpoint \
  --data /mnt/cache/wuruixiao/users/lsc/EasyR1/data/offline_rft/isic/v1/MCQ_test_4shot_nothinking.4labels_text.jsonl \
  --sample-id-key images.0 \
  --image-root /mnt/cache/wuruixiao/users/lsc/data/OmniMedVQA \
  --checkpoint /mnt/cache/wuruixiao/users/lsc/EasyR1/checkpoints/MCQ_failure/4label_text_list_eval2_nothinking_isic_rollout_log/global_step_170/actor/huggingface \
  --output-npz /mnt/cache/wuruixiao/users/lsc/data/OminiMedExpert/probe_feat/isic_4labellist_post_multitap.npz \
  --taps vision_mean \
  --taps hs:0:image \
  --taps hs:16:image \
  --taps hs:-1:image \
  --taps last:-1 \
  --prompt-key prompt \
  --dtype bf16 --device auto --trust-remote-code \
  --verbose \
  --progress-every 50 

# Tap B: embedding output (layer=0), pool image-token hidden states
python3 -m probe_analysis.extract_features_from_checkpoint \
  --data /mnt/cache/wuruixiao/users/lsc/EasyR1/data/offline_rft/isic/v1/MCQ_test_4shot_nothinking.full_labels_text.jsonl \
  --sample-id-key images.0 \
  --image-root /data/shichao/data/OmniMedVQA \
  --checkpoint /mnt/cache/wuruixiao/users/lsc/EasyR1/checkpoints/.../global_step_xxx/actor/huggingface \
  --output-npz /mnt/cache/wuruixiao/users/lsc/data/OminiMedExpert/probe_feat/isic_post_hs0_image.npz \
  --tap hidden_mean --layer 0 --token-scope image \
  --dtype bf16 --device auto --trust-remote-code

# Tap C: last layer (-1), pool image-token hidden states
python3 -m probe_analysis.extract_features_from_checkpoint \
  --data /mnt/cache/wuruixiao/users/lsc/EasyR1/data/offline_rft/isic/v1/MCQ_test_4shot_nothinking.full_labels_text.jsonl \
  --sample-id-key images.0 \
  --image-root /data/shichao/data/OmniMedVQA \
  --checkpoint /mnt/cache/wuruixiao/users/lsc/EasyR1/checkpoints/.../global_step_xxx/actor/huggingface \
  --output-npz /mnt/cache/wuruixiao/users/lsc/data/OminiMedExpert/probe_feat/isic_post_hs-1_image.npz \
  --tap hidden_mean --layer -1 --token-scope image \
  --dtype bf16 --device auto --trust-remote-code
```

## Step 2: Run probe (jsonl + npz -> summary.json)

```bash
python3 -m probe_analysis.run_probe \
  --data /mnt/cache/wuruixiao/users/lsc/EasyR1/data/offline_rft/isic/v1/MCQ_test_4shot_nothinking.full_labels_text.jsonl \
  --sample-id-key images.0 \
  --output-dir /mnt/cache/wuruixiao/users/lsc/data/OminiMedExpert/probe_feat/probe_full_labels_text \
  --extractor npz \
  --features-npz /mnt/cache/wuruixiao/users/lsc/data/OminiMedExpert/probe_feat/isic_post_multitap.npz \
  --features-key features_vision_mean \
  --ids-key image_paths \
  --label-key answer.label \
  --probes linear,knn,mlp \
  --test-size 0.4 \
  --seed 42 \
  --bootstrap 1000
```

### Selecting a tap from a multi-tap npz

If you extracted multiple taps into a single `.npz`, choose which tap to probe by `--features-key`.

Examples:
- `--features-key features_vision_mean`
- `--features-key features_hs0_image`
- `--features-key features_hs16_image`
- `--features-key features_hs-1_image`
- `--features-key features_last-1`

To list keys inside an npz:

```bash
python3 - <<'PY'
import numpy as np
z=np.load('/mnt/cache/wuruixiao/users/lsc/data/OminiMedExpert/probe_feat/isic_post_multitap.npz', allow_pickle=True)
print(z.files)
print('default:', z.get('features_key_default', None))
PY
```

### Grouped probe by candidate label schema

Use this when your dataset mixes 2-way, 8-way, 4-way, etc. It will emit per-group results into `summary.json` under `groups`.

```bash
python3 -m probe_analysis.run_probe \
  --data /mnt/cache/wuruixiao/users/lsc/EasyR1/data/offline_rft/isic/v1/MCQ_test_4shot_nothinking.full_labels_text.jsonl \
  --sample-id-key images.0 \
  --output-dir /mnt/cache/wuruixiao/users/lsc/data/OminiMedExpert/probe_feat/probe_grouped \
  --extractor npz \
  --features-npz /mnt/cache/wuruixiao/users/lsc/data/OminiMedExpert/probe_feat/isic_post_multitap.npz \
  --features-key features_hs0_image \
  --ids-key image_paths \
  --label-key answer.label \
  --probes linear,knn,mlp \
  --group-by-candidate-labels \
  --candidate-labels-key answer.candidate_labels \
  --test-size 0.4 \
  --min-group-size 20 \
  --verbose \
  --log-every 50 \
  --bootstrap 1000 \
  --auto-summary-suffix
```

## Step 3: Compare pre vs post

```bash
python3 -m probe_analysis.compare_runs \
  --pre /mnt/cache/wuruixiao/users/lsc/data/OminiMedExpert/probe_feat/probe_pre/summary.json \
  --post /mnt/cache/wuruixiao/users/lsc/data/OminiMedExpert/probe_feat/probe_post/summary.json
```
