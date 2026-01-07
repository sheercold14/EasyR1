import json
from pathlib import Path
from typing import List


OMNI_ROOT = Path("/tmp/shared-storage/lishichao/data/OmniMedVQA")
ISIC_DIR = Path("/tmp/shared-storage/lishichao/data/OminiMedVQA-Expert/QA_information/ISIC/fewshot/16-shot")
OUT_DIR = Path("/tmp/shared-storage/lishichao/EasyR1/data/omnimedvqa/ISIC/few_shot/16-shot")


def build_prompt(item: dict) -> str:
    question = str(item.get("question", "")).strip()
    options: List[str] = []
    for key in ["option_A", "option_B", "option_C", "option_D"]:
        if key in item and item[key] is not None:
            label = key.split("_")[1]
            options.append(f"{label}. {str(item[key]).strip()}")
    if options:
        return (
            f"Question: {question}\n"
            "Options:\n"
            + "\n".join(options)
            + "\nAnswer with the exact option text."
        )
    return f"Question: {question}\nAnswer succinctly."


def resolve_image_path(root: Path, image_path: str) -> Path:
    return root / image_path


def write_jsonl(path: Path, rows: List[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def main() -> None:
    train_src = ISIC_DIR / "ISIC_train.json"
    test_src = ISIC_DIR / "ISIC_test.json"

    train_items = json.loads(train_src.read_text(encoding="utf-8"))
    test_items = json.loads(test_src.read_text(encoding="utf-8"))

    def convert(items: List[dict]) -> tuple[List[dict], int]:
        rows: List[dict] = []
        missing_images = 0
        for item in items:
            image_path = str(item.get("image_path", "")).strip()
            abs_path = resolve_image_path(OMNI_ROOT, image_path)
            if not abs_path.exists():
                missing_images += 1
                continue
            rows.append(
                {
                    "prompt": build_prompt(item),
                    "images": [str(abs_path)],
                    "answer": {
                        "label": str(item.get("gt_answer", "")).strip(),
                        "question_id": str(item.get("question_id", "")).strip(),
                        "dataset": str(item.get("dataset", "")).strip(),
                        "question_type": str(item.get("question_type", "")).strip(),
                        "modality_type": str(item.get("modality_type", "")).strip(),
                        "option_A": item.get("option_A"),
                        "option_B": item.get("option_B"),
                        "option_C": item.get("option_C"),
                        "option_D": item.get("option_D"),
                    },
                }
            )
        return rows, missing_images

    train_rows, train_missing = convert(train_items)
    test_rows, test_missing = convert(test_items)

    write_jsonl(OUT_DIR / "train.jsonl", train_rows)
    write_jsonl(OUT_DIR / "val.jsonl", test_rows)

    summary = {
        "train_total": len(train_items),
        "train_written": len(train_rows),
        "train_missing_images": train_missing,
        "test_total": len(test_items),
        "test_written": len(test_rows),
        "test_missing_images": test_missing,
        "output_dir": str(OUT_DIR),
    }

    (OUT_DIR / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
