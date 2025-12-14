import json, random, pathlib, collections

src = pathlib.Path("/tmp/shared-storage/lishichao/EasyR1/data/food101/val.jsonl")
dst = pathlib.Path("/tmp/shared-storage/lishichao/EasyR1/data/food101/val_subset.jsonl")
target_total = 512
seed = 42

random.seed(seed)
groups = collections.defaultdict(list)
with src.open() as f:
    for line in f:
        line = line.strip()
        if not line:
            continue
        obj = json.loads(line)
        label = obj.get("answer", {}).get("label", "<none>")
        groups[label].append(line)

num_labels = len(groups)
per_label = max(1, round(target_total / num_labels))  # ≈5/类
subset = []
for label, lines in groups.items():
    k = min(per_label, len(lines))
    subset.extend(random.sample(lines, k))

# 如果总数略大于/小于 target_total，可以再截断或补齐，这里保持均匀采样结果
with dst.open("w") as f:
    for line in subset:
        f.write(line + "\n")

print(f"labels: {num_labels}, per_label: {per_label}, total written: {len(subset)} -> {dst}")

