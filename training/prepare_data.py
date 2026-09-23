from datasets import load_dataset, Dataset
import json
import random
from pathlib import Path

SUBJECTS = [
    "elementary_mathematics",
    "high_school_mathematics",
    "high_school_statistics",
    "college_mathematics",
    "abstract_algebra",
]

SEED = 1234

OUTPUT_DIR = Path("training/data")

def question_key(example):
    """Identify an MMLU problem using its question and answer choices."""
    question = " ".join(example["question"].lower().split())
    choices = tuple(
        " ".join(choice.lower().split())
        for choice in example["choices"]
    )
    return question, choices

def main():
    random.seed(SEED)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    all_train = []
    all_validation = []
    all_test = []

    manifest = {
        "seed": SEED,
        "split_ratio": {
            "train": 0.60,
            "validation": 0.10,
            "test": 0.30,
        },
        "subjects": {},
    }

    for subject in SUBJECTS:
        dataset = load_dataset("cais/mmlu", subject)

        # Pool the original MMLU splits because the official dev split
        # only contains five examples per subject.
        examples = []

        for split in ["dev", "validation", "test"]:
            for example in dataset[split]:
                examples.append({
                    "question": example["question"],
                    "choices": example["choices"],
                    "answer": example["answer"],
                    "subject": subject,
                    "original_split": split,
                })

        original_count = len(examples)

        # Remove exact duplicate questions before splitting.
        unique_examples = []
        seen = set()

        for example in examples:
            key = question_key(example)

            if key not in seen:
                seen.add(key)
                unique_examples.append(example)

        # Fixed seed makes it reproducible.
        random.shuffle(unique_examples)

        n = len(unique_examples)

        train_end = int(n * 0.60)
        validation_end = train_end + int(n * 0.10)

        train_examples = unique_examples[:train_end]
        validation_examples = unique_examples[train_end:validation_end]
        test_examples = unique_examples[validation_end:]

        all_train.extend(train_examples)
        all_validation.extend(validation_examples)
        all_test.extend(test_examples)

        manifest["subjects"][subject] = {
            "original_count": original_count,
            "unique_count": n,
            "duplicates_removed": original_count - n,
            "train": len(train_examples),
            "validation": len(validation_examples),
            "test": len(test_examples),
        }

    # Shuffle the combined sets so the subjects are mixed up during training.
    random.shuffle(all_train)
    random.shuffle(all_validation)
    random.shuffle(all_test)

    Dataset.from_list(all_train).to_json(OUTPUT_DIR / "train.jsonl")
    Dataset.from_list(all_validation).to_json(
        OUTPUT_DIR / "validation.jsonl"
    )
    Dataset.from_list(all_test).to_json(OUTPUT_DIR / "test.jsonl")

    manifest["totals"] = {
        "train": len(all_train),
        "validation": len(all_validation),
        "test": len(all_test),
    }

    with open(OUTPUT_DIR / "manifest.json", "w") as f:
        json.dump(manifest, f, indent=2)

    print("\nDataset preparation complete.")
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
