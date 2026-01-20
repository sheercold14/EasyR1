"""
Reward function for thyroid image analysis with jinja-generated prompts.

Supports two task types based on data:
1. Single-image: "Classify as [class]"
2. Multi-image comparative: "Which image shows [target_class]?"

The data only contains sample info. Prompts are added dynamically via jinja template.
"""

import re
from typing import Any, List

REWARD_NAME = "thyroid_jinja_mixed"
REWARD_TYPE = "batch"

STRUCTURE_WEIGHT = 0.0


def _extract_single_image_label(text: str, class_names: List[str]) -> str:
    """Extract the classification label from single-image response."""
    t = text.lower()

    # Prefer <answer>...</answer>
    for class_name in class_names:
        pattern = rf"<answer>\s*{re.escape(class_name.lower())}\s*</answer>"
        if re.search(pattern, t):
            return class_name

    # General <answer>pattern
    m = re.search(r"<answer>\s*(\w+)\s*</answer>", t)
    if m:
        cand = m.group(1)
        if cand in class_names:
            return cand

    # Keywords fallback - find which class is mentioned
    mentioned_classes = [c for c in class_names if c.lower() in t]
    if len(mentioned_classes) == 1:
        return mentioned_classes[0]

    # Check conclusion patterns
    for pattern_template in [
        r"(?:conclusion|diagnosis|classified as|is)\s*:\s*(\w+)",
        r"(?:the\s+(?:image|case)\s+is)\s+(\w+)",
    ]:
        m = re.search(pattern_template, t)
        if m:
            cand = m.group(1)
            if cand in class_names:
                return cand

    return "unknown"


def _extract_comparative_answer(text: str, correct_answer: str) -> str:
    """
    Extract the answer letter from comparative response.

    The correct_answer is pre-determined in the data (e.g., "A", "B").
    """
    t = text.lower()

    # Prefer <answer>...</answer>
    m = re.search(r"<answer>\s*([a-z])\s*</answer>", t)
    if m:
        return m.group(1).upper()

    # Look for "Answer: X" pattern
    m = re.search(r"answer\s*:\s*([a-z])", t)
    if m:
        return m.group(1).upper()

    # Look for "The [target_class] image is X" pattern
    m = re.search(r"(?:image|answer|correct|target)\s*(?:is|:)\s*([a-z])", t)
    if m:
        return m.group(1).upper()

    # Look for standalone letter at the end
    m = re.search(r"\s([a-z])\s*$", t)
    if m:
        return m.group(1).upper()

    # Fallback: find any valid letter mentioned
    for letter in [chr(ord("a") + i) for i in range(4)]:  # up to 4 images
        if f"({letter})" in t or f" {letter} " in t or f"image {letter}" in t:
            return letter.upper()

    return "UNKNOWN"


def _structure_score(text: str) -> float:
    """Reward for proper response structure."""
    score = 0.0
    t = text.lower()

    # Has thinking/reasoning section
    if "<thinking>" in t and "</thinking>" in t:
        score += 0.1

    # Has answer tags
    if "<answer>" in t and "</answer>" in t:
        score += 0.1

    # Has comparative language
    comparative_keywords = [
        "compare", "compared", "difference", "similar", "unlike",
        "whereas", "while", "versus", "both", "however",
    ]
    if any(kw in t for kw in comparative_keywords):
        score += 0.2

    # Mentions multiple images
    if re.search(r"(image\s*[a-z]|image\s*\d)", t):
        score += 0.1

    return min(score, 0.5)

def compute_score_single(response: str, ground_truth: dict) -> dict[str, Any]:
    """
    Compute reward for single-image classification task.

    Args:
        response: Model's response text
        ground_truth: Contains 'label'
    """
    class_names = ["benign", "malignant"]
    true_label = ground_truth.get("label", "unknown")
    predicted_label = _extract_single_image_label(response, class_names)

    # Correctness score
    if predicted_label == "unknown":
        r_correct = -0.5
        acc = 0.0
    elif predicted_label == true_label:
        r_correct = 2.0
        acc = 1.0
    else:
        r_correct = -1.5
        acc = 0.0

    r_struct = _structure_score(response)
    total_score = r_correct + STRUCTURE_WEIGHT * r_struct

    return {
        "overall": float(total_score),
        "correct": float(r_correct),
        "structure": float(r_struct),
        "acc": float(acc),
        "predicted": predicted_label,
        "correct_answer": true_label,
    }


def compute_score_comparative(response: str, ground_truth: dict) -> dict[str, Any]:
    """
    Compute reward for multi-image comparative task.

    Args:
        response: Model's response text
        ground_truth: Contains 'target_class', 'correct_answer', 'labels', 'num_images'
    """
    correct_answer = ground_truth.get("correct_answer", "A").upper()
    target_class = ground_truth.get("target_class", "")

    predicted_answer = _extract_comparative_answer(response, correct_answer)

    # Correctness score
    if predicted_answer == "UNKNOWN":
        r_correct = -0.5
        acc = 0.0
    elif predicted_answer == correct_answer:
        r_correct = 2.0
        acc = 1.0
    else:
        r_correct = -1.5
        acc = 0.0

    r_struct = _structure_score(response)
    total_score = r_correct + STRUCTURE_WEIGHT * r_struct

    return {
        "overall": float(total_score),
        "correct": float(r_correct),
        "structure": float(r_struct),
        "acc": float(acc),
        "predicted": predicted_answer,
        "correct_answer": correct_answer,
        "target_class": target_class,
    }


def compute_score(reward_inputs: List[dict[str, Any]]) -> List[dict[str, float]]:
    """
    Compute rewards for thyroid image analysis.

    The data contains a `task_type` field that determines which task:
    - "single": Single-image classification
    - "comparative": Multi-image comparison

    Args:
        reward_inputs: List of dicts with keys:
            - response: str, model's response
            - ground_truth: dict with task-specific fields

    Returns:
        List of dicts with ONLY numeric reward components (float values)
    """
    scores = []

    for item in reward_inputs:
        response = item["response"]
        ground_truth = item["ground_truth"]

        if not isinstance(ground_truth, dict):
            ground_truth = {"label": str(ground_truth)}

        task_type = ground_truth.get("task_type", None)
        is_comparative = task_type == "comparative" or "correct_answer" in ground_truth

        if is_comparative:
            result = compute_score_comparative(response, ground_truth)
        else:
            result = compute_score_single(response, ground_truth)

        # Add task_type for tracking (but will be filtered out below)
        result["task_type"] = "comparative" if is_comparative else "single"

        # Filter to ONLY numeric values for metrics aggregation
        # Remove string keys: 'predicted', 'correct_answer', 'target_class', 'task_type'
        numeric_result = {k: v for k, v in result.items() if isinstance(v, (int, float))}

        scores.append(numeric_result)

    return scores
