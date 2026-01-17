# Comparative-R1: Few-Shot Learning with GSPO

Training framework for few-shot medical image classification using mixed single-image and multi-image comparative learning with Group Sparse Policy Optimization (GSPO).

## Overview

Two-stage pipeline:
1. **Data Preprocessing**: Generate mixed/comparative samples from original data
2. **GSPO Training**: Train with standard EasyR1 pipeline

### Training Data Format
- **Single-image samples**: "Classify this image as benign or malignant"
- **Multi-image comparative samples**: "Which image shows [target class]?" (random target class)
- **Validation data**: Single-image only for proper evaluation

### Key Features
- **Random target class**: Comparative task asks "which is benign?" OR "which is malignant?" randomly
- **N-class support**: Works with any number of classes
- **Guaranteed unique answer**: Always exactly 1 target image in comparative samples

## Project Structure

```
Comparative-R1/
├── reward/
│   └── comparative_reward.py     # Reward function for both task types
├── prompts/
│   └── comparative.jinja         # Reference prompt template
├── configs/
│   └── comparative_gspo.yaml     # Training configuration
├── train_comparative.sh           # Launch script (data gen + training)
└── README.md
```

Data preprocessing:
```
scripts/generate_comparative_data.py
```

## Dataset

**Thyroid Ultrasound Classification**
- Source: `/data/shichao/EasyR1/data/thyroid/all.jsonl`
- Total samples: 316 (128 benign, 188 malignant)
- 5-fold cross-validation splits (~64 samples per fold)
- Default: Train on splits 0,1,2 (~190 samples) | Val on splits 3,4 (~126 samples)

## Quick Start

### 1. Generate Data + Train (Default)

```bash
cd /data/shichao/EasyR1
bash Comparative-R1/train_comparative.sh
```

This will:
1. Generate `train.jsonl` and `val.jsonl` in `/data/shichao/EasyR1/data/thyroid/`
2. Train the model with GSPO

### 2. Data Generation Only

```bash
python3 /data/shichao/EasyR1/scripts/generate_comparative_data.py \
    --data_path /data/shichao/EasyR1/data/thyroid/all.jsonl \
    --output_dir /data/shichao/EasyR1/data/thyroid \
    --train_splits "0,1,2" \
    --val_splits "3,4" \
    --num_images 2 \
    --target_class_strategy random \
    --num_comparative_samples 1000
```

### 3. Training Only (Skip Data Gen)

```bash
python3 -m verl.trainer.main \
    config=Comparative-R1/configs/comparative_gspo.yaml \
    ...
```

## Parameters

### Data Generation

| Parameter | Default | Description |
|-----------|---------|-------------|
| `--train_splits` | 0,1,2 | Training folds (comma-separated) |
| `--val_splits` | 3,4 | Validation folds (comma-separated) |
| `--num_images` | 2 | Images per comparison |
| `--target_class_strategy` | random | Target selection: random, fixed_first |
| `--num_comparative_samples` | 1000 | Comparative samples per split |

### Training

| Parameter | Default | Description |
|-----------|---------|-------------|
| `NUM_GPUS` | 4 | Number of GPUs |
| `BATCH_SIZE` | 32 | Rollout batch size |
| `N_SAMPLES` | 4 | Samples per prompt for GSPO |
| `TEMPERATURE` | 0.7 | Sampling temperature |
| `SAVE_FREQ` | 5 | Save checkpoint every N steps |

## Custom Training

```bash
# Custom splits
TRAIN_SPLITS=0,1 VAL_SPLITS=2,3,4 bash Comparative-R1/train_comparative.sh

# More images per comparison
NUM_IMAGES=3 bash Comparative-R1/train_comparative.sh

# More comparative samples
NUM_COMP_SAMPLES=2000 bash Comparative-R1/train_comparative.sh

# Always ask "which is benign?"
TARGET_STRATEGY=fixed_first bash Comparative-R1/train_comparative.sh
```

## Model Output Format

### Single-Image Task

```
<thinking>
[Analysis of image features...]
[Explanation of what these features indicate...]
</thinking>

<answer> benign or malignant </answer>
```

### Comparative Task (Random Target)

```
<thinking>
[Comparative analysis of each image...]
[Comparison between images...]
[Reasoning for which is target_class...]
</thinking>

<answer> A or B </answer>
```

Target class (benign/malignant) changes randomly per sample!

## Reward Function

The reward has three components:

1. **Correctness** (`+2.0` / `-1.5` / `-0.5`)
   - Correct: +2.0
   - Wrong: -1.5
   - Unable to parse: -0.5

2. **Structure** (`0.0` to `0.5`)
   - Proper `<thinking>` tags
   - Proper `<answer>` tags
   - Comparative language

3. **Reasoning Quality** (`0.0` to `0.8`)
   - Medical feature mentions
   - Comparative reasoning
   - Explanation of decision

## Checkpoint Strategy

- **Latest checkpoint** (most recent)
- **Best checkpoint** (highest validation reward, protected from deletion)
- `save_limit: 1` keeps only latest + best

## Configuration

Key settings in `configs/comparative_gspo.yaml`:

```yaml
data:
  train_files: /data/shichao/EasyR1/data/thyroid/train.jsonl
  val_files: /data/shichao/EasyR1/data/thyroid/val.jsonl
  limit_images: 4  # Support up to 4 images

algorithm:
  adv_estimator: grpo
  disable_kl: true

worker:
  actor:
    loss_type: gspo_token
    loss_avg_mode: seq
```

## Model Support

- **Qwen3-VL-2B-Instruct** (default)
- Modify `MODEL_PATH` in training script for other models

## Citation

If you use this framework, please cite EasyR1.
