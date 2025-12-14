# -*- coding: utf-8 -*-
"""
Aligned validation script.
- Uses the same template as training (pathology.jinja) via chat template.
- Loads merged actor weights for plain HF generation.
- Uses the same answer parsing/ambiguous logic as pathology_answer.py.
- Outputs per-sample JSONL with image path, gt label, pred, ambiguous flag, right, and full output.
"""

import json
import pathlib
import re
from typing import Optional

import torch
from PIL import Image
from transformers import AutoModelForImageTextToText, AutoProcessor


# === Paths to adjust if needed ===
BASE_MODEL = "/tmp/shared-storage/lishichao/cache/models--qwen--qwen2.5-vl-7b-instruct/snapshots/cc594898137f460bfe9f0759e9844b3ce807cfb5"
CKPT_ACTOR = "/tmp/shared-storage/lishichao/EasyR1/checkpoints/easy_r1/qwen2_5_vl_7b_geo_grpo_pathology_1210_n=4_t=0.7_p=0.9_correct_rewards/global_step_1225/actor/huggingface"
VAL_PATH = "/tmp/shared-storage/lishichao/EasyR1/data/thyroid/val.jsonl"
TEMPLATE_PATH = "/tmp/shared-storage/lishichao/EasyR1/examples/format_prompt/pathology.jinja"
OUT_PATH = "/tmp/shared-storage/lishichao/EasyR1/scripts/val_with_paths_codex.jsonl"

TEMPERATURE = 0.6  # match val_override_config
TOP_P = 0.95
MAX_NEW_TOKENS = 512  # align with max_response_length to avoid truncation


# === Helpers matching pathology_answer.py ===
AMBIGUOUS = [
    r"uncertain",
    r"cannot\s+determine",
    r"cannot\s+rule\s+out",
    r"indeterminate",
    r"need\s+biopsy",
    r"need\s+histopath",
    r"not\s+sure",
    r"further\s+exam",
]


def _is_ambiguous(text: str) -> bool:
    t = text.lower()
    return any(re.search(pat, t) for pat in AMBIGUOUS)


def _extract_answer(text: str) -> str:
    m = re.search(r"<answer>\s*(malignant|benign)\s*</answer>", text, flags=re.IGNORECASE)
    if m:
        return m.group(1).strip().lower()
    m = re.search(r"answer\s*:\s*(malignant|benign)", text, flags=re.IGNORECASE)
    if m:
        return m.group(1).strip().lower()
    return "unknown"


def main(
    base_model: str = BASE_MODEL,
    ckpt_actor: str = CKPT_ACTOR,
    val_path: str = VAL_PATH,
    template_path: str = TEMPLATE_PATH,
    out_path: str = OUT_PATH,
    temperature: float = TEMPERATURE,
    top_p: float = TOP_P,
    max_new_tokens: int = MAX_NEW_TOKENS,
    device_map: Optional[str] = "auto",
    dtype: torch.dtype = torch.float16,
):
    template = pathlib.Path(template_path).read_text(encoding="utf-8")
    processor = AutoProcessor.from_pretrained(base_model, trust_remote_code=False)
    model = AutoModelForImageTextToText.from_pretrained(
        ckpt_actor, trust_remote_code=False, torch_dtype=dtype, device_map=device_map
    ).eval()
    eos_id = processor.tokenizer.eos_token_id
    pad_id = processor.tokenizer.pad_token_id

    correct_total = 0
    sample_total = 0

    with open(val_path) as fin, open(out_path, "w", encoding="utf-8") as fout:
        for idx, line in enumerate(fin):
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            img_path = obj["images"][0]
            gt_label = obj.get("answer", {}).get("label")
            image = Image.open(img_path).convert("RGB")

            messages = [
                {
                    "role": "user",
                    "content": [
                        {"type": "image"},
                        {"type": "text", "text": template},
                    ],
                }
            ]
            prompt = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
            inputs = processor(text=prompt, images=[image], return_tensors="pt").to(model.device)

            with torch.no_grad():
                out_ids = model.generate(
                    **inputs,
                    max_new_tokens=max_new_tokens,
                    temperature=temperature,
                    top_p=top_p,
                    do_sample=True,
                    eos_token_id=eos_id,
                    pad_token_id=pad_id,
                )
            out_text = processor.batch_decode(out_ids, skip_special_tokens=True)[0]

            pred_label = _extract_answer(out_text)
            ambiguous = _is_ambiguous(out_text)
            right = 1 if (gt_label is not None and pred_label == str(gt_label).lower() and not ambiguous) else 0

            correct_total += right
            sample_total += 1

            fout.write(
                json.dumps(
                    {
                        "idx": idx,
                        "image": img_path,
                        "label": gt_label,
                        "output": out_text,
                        "pred": pred_label,
                        "ambiguous": int(ambiguous),
                        "right": right,
                    },
                    ensure_ascii=False,
                )
                + "\n"
            )

    acc = correct_total / sample_total if sample_total > 0 else 0.0
    print(f"Wrote {out_path}, total={sample_total}, acc={acc:.4f}")


if __name__ == "__main__":
    main()
