# save as build_food101_fixed.py
import json, pathlib
from datasets import load_dataset

OUT_DIR = "/tmp/shared-storage/lishichao/EasyR1/data/food101"  # 修改为你的输出目录

out_dir = pathlib.Path(OUT_DIR)
out_dir.mkdir(parents=True, exist_ok=True)

train_path = out_dir / "train.jsonl"
val_path = out_dir / "val.jsonl"
out_img = out_dir / "images"
out_img.mkdir(parents=True, exist_ok=True)

def dump_split(ds, json_path):
    ds = load_dataset("food101", split=ds)
    with json_path.open("w", encoding="utf-8") as f:
        for i, ex in enumerate(ds):
            label = ds.features["label"].int2str(ex["label"])
            img_path = out_img / f"{json_path.stem}_{i:06d}.jpg"
            ex["image"].save(img_path, format="JPEG")
            entry = {
                "prompt": "",
                "images": [str(img_path)],
                "answer": {"label": label},
            }
            f.write(json.dumps(entry, ensure_ascii=False) + "\n")

dump_split("train", train_path)
dump_split("validation", val_path)

print("written:", train_path, val_path)
