import json
from collections import defaultdict
from pathlib import Path

import torch
from transformers import AutoProcessor, Gemma3ForConditionalGeneration


MODEL_ID = "google/gemma-3-4b-pt"
TEST_FILE = Path("training/data/test.jsonl")
RESULTS_DIR = Path("evaluation/results")
LETTERS = ["A", "B", "C", "D"]


def load_test_data():
    #Load the held-out MMLU test examples.
    with open(TEST_FILE) as f:
        return [json.loads(line) for line in f]


def build_prompt(example):
    #Format one MMLU example as a multiple-choice question.
    choices = example["choices"]

    return (
        f"Question: {example['question']}\n\n"
        f"A. {choices[0]}\n"
        f"B. {choices[1]}\n"
        f"C. {choices[2]}\n"
        f"D. {choices[3]}\n\n"
        "Answer:"
    )


def predict(model, processor, example):
    #Choose the answer letter with the highest next-token score.
    prompt = build_prompt(example)

    inputs = processor(
        text=prompt,
        return_tensors="pt",
    ).to("cuda")

    with torch.no_grad():
        outputs = model(**inputs)

    next_token_logits = outputs.logits[0, -1]

    scores = {}

    for letter in LETTERS:
        token_ids = processor.tokenizer.encode(
            letter,
            add_special_tokens=False,
        )

        if len(token_ids) != 1:
            raise ValueError(
                f"Expected {letter} to be one token, got {token_ids}"
            )

        scores[letter] = next_token_logits[token_ids[0]].item()

    prediction = max(scores, key=scores.get)

    return prediction, scores


def main():
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    print("Loading test data...")
    examples = load_test_data()
    print(f"Examples: {len(examples)}")

    print("\nLoading model...")
    processor = AutoProcessor.from_pretrained(MODEL_ID)

    model = Gemma3ForConditionalGeneration.from_pretrained(
        MODEL_ID,
        torch_dtype=torch.bfloat16,
    ).to("cuda")

    model.eval()

    correct = 0
    subject_correct = defaultdict(int)
    subject_total = defaultdict(int)
    predictions = []

    print("\nRunning baseline evaluation...")

    for i, example in enumerate(examples, start=1):
        prediction, scores = predict(
            model,
            processor,
            example,
        )

        correct_letter = LETTERS[example["answer"]]
        is_correct = prediction == correct_letter
        subject = example["subject"]

        correct += int(is_correct)
        subject_correct[subject] += int(is_correct)
        subject_total[subject] += 1

        predictions.append(
            {
                "question": example["question"],
                "subject": subject,
                "prediction": prediction,
                "correct_answer": correct_letter,
                "correct": is_correct,
                "scores": scores,
            }
        )

        if i % 25 == 0 or i == len(examples):
            print(f"Completed {i}/{len(examples)}")

    overall_accuracy = correct / len(examples)

    per_subject = {}

    for subject in sorted(subject_total):
        accuracy = (
            subject_correct[subject]
            / subject_total[subject]
        )

        per_subject[subject] = {
            "correct": subject_correct[subject],
            "total": subject_total[subject],
            "accuracy": accuracy,
        }

    summary = {
        "model": MODEL_ID,
        "evaluation_split": "custom held-out MMLU math test split",
        "total_examples": len(examples),
        "correct": correct,
        "accuracy": overall_accuracy,
        "per_subject": per_subject,
    }

    with open(RESULTS_DIR / "baseline_predictions.jsonl", "w") as f:
        for row in predictions:
            f.write(json.dumps(row) + "\n")

    with open(RESULTS_DIR / "baseline_summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    print("\nBASELINE RESULTS")
    print(f"Overall: {correct}/{len(examples)}")
    print(f"Accuracy: {overall_accuracy:.2%}")

    print("\nPer subject:")

    for subject, result in per_subject.items():
        print(
            f"{subject}: "
            f"{result['correct']}/{result['total']} "
            f"({result['accuracy']:.2%})"
        )

    print("\nResults saved to:")
    print(RESULTS_DIR / "baseline_summary.json")
    print(RESULTS_DIR / "baseline_predictions.jsonl")


if __name__ == "__main__":
    main()