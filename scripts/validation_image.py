# infer_val.py
import json, pathlib, re
from PIL import Image
import torch
from transformers import AutoProcessor, AutoModelForImageTextToText

base_model = "/tmp/shared-storage/lishichao/cache/models--qwen--qwen2.5-vl-7b-instruct/snapshots/cc594898137f460bfe9f0759e9844b3ce807cfb5"
ckpt_actor = "/tmp/shared-storage/lishichao/EasyR1/checkpoints/easy_r1/qwen2_5_vl_7b_geo_grpo_pathology_1210_n=4_t=0.7_p=0.9_correct_rewards/global_step_1225/actor/huggingface"
val_path = "/tmp/shared-storage/lishichao/EasyR1/data/thyroid/val.jsonl"
out_path = "val_with_paths.jsonl"

template = """You are a pathology assistant. Read the gross description and the associated frozen section image, then produce a concise diagnosis.

<image>
Follow the exact output format:
<think>
- Summarize gross features: solidity vs cystic change, hemorrhage, necrosis (if any), margin (regular/irregular), capsule status.
- Explain how these features support the diagnosis.
</think>
<answer> benign or malignant </answer>
"""

device = "cuda"
dtype = torch.float16

# Processor 从基座加载（含视觉处理）
processor = AutoProcessor.from_pretrained(base_model, trust_remote_code=False)

# 提取 <answer> 中的标签
def _extract_answer(text: str) -> str:
    m = re.findall(r"<answer>\s*(.*?)\s*</answer>", text, flags=re.IGNORECASE|re.DOTALL)
    m = m[1].strip().lower() if len(m) >= 2 else None
    if m:
        return m

    return "no answer"

# 模型优先直接从 actor ckpt 目录加载；如失败，可用 base_model 再 load_state_dict
model = AutoModelForImageTextToText.from_pretrained(
    ckpt_actor, trust_remote_code=False, torch_dtype=dtype, device_map="auto"
).eval()

with open(val_path) as fin, open(out_path, "w") as fout:
    correct_total = 0
    sample_total = 0
    for idx, line in enumerate(fin):
        obj = json.loads(line)
        img_path = obj["images"][0]
        image = Image.open(img_path).convert("RGB")
        gt_label = obj.get("answer", {}).get("label")

        # 使用 chat template 注入图像占位符，保持原有文本内容
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
                **inputs, max_new_tokens=2048, temperature=0.6, top_p=0.95, do_sample=True
            )
        out_text = processor.batch_decode(out_ids, skip_special_tokens=True)[0]

        pred_label = _extract_answer(out_text)
        right = 1 if gt_label is not None and pred_label == str(gt_label).lower() else 0
        correct_total += right
        sample_total += 1

        fout.write(json.dumps({
            "idx": idx,
            "image": img_path,
            "label": gt_label,
            "output": out_text,
            "pred": pred_label,
            "right": right
        }, ensure_ascii=False) + "\n")

acc = correct_total / sample_total if sample_total > 0 else 0.0
print(f"Wrote {out_path}, total={sample_total}, acc={acc:.4f}")
