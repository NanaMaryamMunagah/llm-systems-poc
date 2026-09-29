import json
from collections import defaultdict
from pathlib import Path

import torch
from transformers import AutoProcessor, Gemma3ForConditionalGeneration


BASE_MODEL_PATH = "/workspace/models/gemma-3-4b-pt"
MODEL_PATH = "training/results/gemma3-4b-mmlu-math-full-lr2e-6-10-ep/final_model"
TEST_FILE = Path("training/data/test.jsonl")
RESULTS_DIR = Path("evaluation/results/full_finetune_low_lr_prompt_ablation")

LETTERS = ["A", "B", "C", "D"]

PROMPT_VARIANTS = [
    "P0_original",
    "P1_explicit",
    "P2_constrained",
]


def load_test_data():
    #Load the held-out MMLU test examples.
    with open(TEST_FILE) as f:
        return [json.loads(line) for line in f]


def build_prompt(example, prompt_variant):
    #Format one MMLU example using the prompt we want to test.
    choices = example["choices"]

    question_block = (
        f"Question: {example['question']}\n\n"
        f"A. {choices[0]}\n"
        f"B. {choices[1]}\n"
        f"C. {choices[2]}\n"
        f"D. {choices[3]}\n\n"
        "Answer:"
    )

    if prompt_variant == "P0_original":
        return question_block

    elif prompt_variant == "P1_explicit":
        return (
            "Choose the correct answer from A, B, C, or D.\n\n"
            + question_block
        )

    elif prompt_variant == "P2_constrained":
        return (
            "Answer the following multiple-choice question. "
            "Respond with only the letter of the correct answer: "
            "A, B, C, or D.\n\n"
            + question_block
        )

    else:
        raise ValueError(
            f"Unknown prompt variant: {prompt_variant}"
        )


def predict(model, processor, example, prompt_variant):
    #Choose the answer letter with the highest score.
    prompt = build_prompt(
        example,
        prompt_variant,
    )

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

        scores[letter] = next_token_logits[
            token_ids[0]
        ].item()

    prediction = max(
        scores,
        key=scores.get,
    )

    return prediction, scores


def evaluate_prompt(
    model,
    processor,
    examples,
    prompt_variant,
):
    #Evaluate one prompt on the full test set.
    correct = 0
    subject_correct = defaultdict(int)
    subject_total = defaultdict(int)
    predictions = []

    print(f"\nEvaluating {prompt_variant}...")

    for i, example in enumerate(examples, start=1):
        prediction, scores = predict(
            model,
            processor,
            example,
            prompt_variant,
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
                "prompt_variant": prompt_variant,
                "prediction": prediction,
                "correct_answer": correct_letter,
                "correct": is_correct,
                "scores": scores,
            }
        )

        if i % 25 == 0 or i == len(examples):
            print(
                f"{prompt_variant}: "
                f"{i}/{len(examples)}"
            )

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

    return {
        "prompt_variant": prompt_variant,
        "correct": correct,
        "total": len(examples),
        "accuracy": overall_accuracy,
        "per_subject": per_subject,
        "predictions": predictions,
    }


def main():
    RESULTS_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    print("Loading test data...")
    examples = load_test_data()
    print(f"Test examples: {len(examples)}")

    print("\nLoading fully fine-tuned model...")

    processor = AutoProcessor.from_pretrained(
        BASE_MODEL_PATH,
        local_files_only=True,
    )

    model = Gemma3ForConditionalGeneration.from_pretrained(
        MODEL_PATH,
        torch_dtype=torch.bfloat16,
        local_files_only=True,
    ).to("cuda")

    model.eval()

    summaries = {}

    #Run the same model with each prompt so the prompt is the only thing changing.
    for prompt_variant in PROMPT_VARIANTS:
        result = evaluate_prompt(
            model,
            processor,
            examples,
            prompt_variant,
        )

        summaries[prompt_variant] = {
            "correct": result["correct"],
            "total": result["total"],
            "accuracy": result["accuracy"],
            "per_subject": result["per_subject"],
        }

        prediction_file = (
            RESULTS_DIR
            / f"{prompt_variant}_predictions.jsonl"
        )

        with open(prediction_file, "w") as f:
            for row in result["predictions"]:
                f.write(json.dumps(row) + "\n")

    summary = {
        "model": MODEL_PATH,
        "evaluation_split": "custom held-out MMLU math test split",
        "prompt_variants": summaries,
    }

    summary_file = RESULTS_DIR / "summary.json"

    with open(summary_file, "w") as f:
        json.dump(summary, f, indent=2)

    print("\nFULL FINE-TUNING PROMPT ABLATION RESULTS")

    for prompt_variant in PROMPT_VARIANTS:
        result = summaries[prompt_variant]

        print(
            f"{prompt_variant}: "
            f"{result['correct']}/{result['total']} "
            f"({result['accuracy']:.2%})"
        )

    print("\nResults saved to:")
    print(summary_file)


if __name__ == "__main__":
    main()