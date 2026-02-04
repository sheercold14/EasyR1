import json
from pathlib import Path
inp = Path("/mnt/cache/wuruixiao/users/lsc/EasyR1/data/OminiMedExpert/isic_disease_diagnosis_v0_0.05/train_fewshot_0.05.jsonl")
out = inp.with_name(inp.stem + "_relpath.jsonl")
with inp.open("r", encoding="utf-8") as fi, out.open("w", encoding="utf-8") as fo:
    for line in fi:
        if not line.strip(): continue
        obj = json.loads(line)
        if isinstance(obj.get("images"), list):
            obj["images"] = [x.lstrip("/") if isinstance(x, str) else x for x in obj["images"]]
        fo.write(json.dumps(obj, ensure_ascii=False) + "\n")
print(out)