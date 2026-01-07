import json
from collections import defaultdict
from pathlib import Path
from typing import DefaultDict, List

BASE_DIR = Path("/tmp/shared-storage/lishichao/EasyR1/data/thyroid")
INPUT_FILE = BASE_DIR / "all.jsonl"
OUTPUT_ROOT = BASE_DIR / "folds"


def load_lines(path: Path) -> list[str]:
    lines: list[str] = []
    with path.open("r", encoding="utf-8") as f:
        for raw in f:
            if not raw.strip():
                continue
            lines.append(raw if raw.endswith("\n") else raw + "\n")
    return lines


def group_by_split(lines: list[str]) -> DefaultDict[str, List[str]]:
    groups: DefaultDict[str, List[str]] = defaultdict(list)
    for line in lines:
        data = json.loads(line)
        split_value = data.get("split")
        if split_value is None:
            raise ValueError("Found sample without `split` field.")
        groups[str(split_value)].append(line)
    return groups


def write_folds() -> None:
    lines = load_lines(INPUT_FILE)
    groups = group_by_split(lines)
    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)

    total = len(lines)
    print(f"Total samples: {total}")
    print(f"Folds: {sorted(groups.keys())}")

    for fold, val_lines in groups.items():
        if fold not in ['4']:
            train_lines: list[str] = []
            for other_fold, other_lines in groups.items():
                if other_fold != fold:
                    train_lines.extend(other_lines)

            fold_dir = OUTPUT_ROOT / fold
            fold_dir.mkdir(parents=True, exist_ok=True)
            (fold_dir / "train.jsonl").write_text("".join(train_lines), encoding="utf-8")
            (fold_dir / "val.jsonl").write_text("".join(val_lines), encoding="utf-8")
            print(f"[fold={fold}] train={len(train_lines)} val={len(val_lines)}")


if __name__ == "__main__":
    write_folds()
