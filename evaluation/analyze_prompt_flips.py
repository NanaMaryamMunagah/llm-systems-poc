import json
from pathlib import Path

RESULTS_DIR = Path("evaluation/results/prompt_ablation")


def load_predictions(filename):
    with open(RESULTS_DIR / filename) as f:
        return [json.loads(line) for line in f]


def compare(original, variant, variant_name):
    wrong_to_right = []
    right_to_wrong = []
    changed_answer = []

    for p0, other in zip(original, variant):
        if p0["prediction"] != other["prediction"]:
            changed_answer.append((p0, other))

        if not p0["correct"] and other["correct"]:
            wrong_to_right.append((p0, other))

        elif p0["correct"] and not other["correct"]:
            right_to_wrong.append((p0, other))

    print(f"\nP0_original -> {variant_name}")
    print(f"Predictions changed: {len(changed_answer)}")
    print(f"Wrong -> right: {len(wrong_to_right)}")
    print(f"Right -> wrong: {len(right_to_wrong)}")
    print(
        f"Net accuracy change: "
        f"{len(wrong_to_right) - len(right_to_wrong)} questions"
    )

    return {
        "predictions_changed": len(changed_answer),
        "wrong_to_right": len(wrong_to_right),
        "right_to_wrong": len(right_to_wrong),
    }


def main():
    p0 = load_predictions(
        "P0_original_predictions.jsonl"
    )

    p1 = load_predictions(
        "P1_explicit_predictions.jsonl"
    )

    p2 = load_predictions(
        "P2_constrained_predictions.jsonl"
    )

    compare(
        p0,
        p1,
        "P1_explicit",
    )

    compare(
        p0,
        p2,
        "P2_constrained",
    )


if __name__ == "__main__":
    main()