#!/usr/bin/env python3

from __future__ import annotations

from reward_offline_mixed import compute_score


def main() -> None:
    reward_inputs = [
        {
            "response": "<answer>yes</answer>",
            "ground_truth": {"task_type": "attr", "answer_type": "bool", "correct_answer": "yes"},
        },
        {
            "response": "<answer>A, B</answer>",
            "ground_truth": {"task_type": "text_rule", "answer_type": "list", "correct_answer": ["a", "b"], "keywords": ["a"]},
        },
        {
            "response": "<answer>melanoma</answer>",
            "ground_truth": {"task_type": "cls", "answer_type": "short_text", "correct_answer": "Melanoma"},
        },
    ]
    scores = compute_score(reward_inputs)
    for item in scores:
        print(item)


if __name__ == "__main__":
    main()
