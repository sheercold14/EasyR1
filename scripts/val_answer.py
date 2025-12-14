import json, pathlib

src = pathlib.Path("/tmp/shared-storage/lishichao/EasyR1/data/thyroid/val.jsonl")
dst = pathlib.Path("/tmp/shared-storage/lishichao/EasyR1/data/thyroid/val_with_img.jsonl")

with src.open() as fin, dst.open("w") as fout:
    for line in fin:
        line = line.strip()
        if not line:
            continue
        obj = json.loads(line)
        imgs = obj.get("images", [])
        img_path = imgs[0] if imgs else ""
        ans = obj.get("answer", {})
        if isinstance(ans, dict):
            ans["image"] = img_path
        else:
            ans = {"label": ans, "image": img_path}
        obj["answer"] = ans
        fout.write(json.dumps(obj, ensure_ascii=False) + "\n")

print(f"written: {dst}")
