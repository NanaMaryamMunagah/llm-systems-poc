import argparse
import json
from pathlib import Path

import torch
import torch.distributed as dist
from peft import LoraConfig, get_peft_model
from torch.utils.data import Dataset
from transformers import (
    AutoProcessor,
    Gemma3ForConditionalGeneration,
    Trainer,
    TrainingArguments,
)

MODEL_PATH = "/workspace/models/gemma-3-4b-pt"
LETTERS = ["A", "B", "C", "D"]

TRAIN_FILE = Path("training/data/train.jsonl")
VALIDATION_FILE = Path("training/data/validation.jsonl")
OUTPUT_DIR = Path("training/results/gemma3-4b-mmlu-math-lora")

def build_prompt(example):
    """Format an MMLU question the same way we do during evaluation."""
    choices = example["choices"]

    return (
        f"Question: {example['question']}\n\n"
        f"A. {choices[0]}\n"
        f"B. {choices[1]}\n"
        f"C. {choices[2]}\n"
        f"D. {choices[3]}\n\n"
        "Answer:"
    )

class MMLUDataset(Dataset):
    #Prepare MMLU examples for answer-only supervised fine-tuning.

    def __init__(self, path, processor, max_examples=None):
        with open(path) as f:
            examples = [json.loads(line) for line in f]

        if max_examples is not None:
            examples = examples[:max_examples]

        self.examples = examples
        self.tokenizer = processor.tokenizer

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, index):
        example = self.examples[index]

        prompt = build_prompt(example)
        answer = LETTERS[example["answer"]]

        prompt_ids = self.tokenizer.encode(
            prompt,
            add_special_tokens=True,
        )

        answer_ids = self.tokenizer.encode(
            answer,
            add_special_tokens=False,
        )

        input_ids = prompt_ids + answer_ids

        # Ignore the prompt when calculating training loss
        # Training only on the correct answer token
        labels = [-100] * len(prompt_ids) + answer_ids

        return {
            "input_ids": input_ids,
            "attention_mask": [1] * len(input_ids),
            "labels": labels,
        }

class DataCollator:
    #Pad examples in a batch to the same sequence length

    def __init__(self, tokenizer):
        self.tokenizer = tokenizer

    def __call__(self, features):
        max_length = max(
            len(feature["input_ids"])
            for feature in features
        )

        input_ids = []
        attention_masks = []
        labels = []

        pad_token_id = self.tokenizer.pad_token_id

        for feature in features:
            padding_length = (
                max_length - len(feature["input_ids"])
            )

            input_ids.append(
                feature["input_ids"]
                + [pad_token_id] * padding_length
            )

            attention_masks.append(
                feature["attention_mask"]
                + [0] * padding_length
            )

            # Padding should not contribute to the loss.
            labels.append(
                feature["labels"]
                + [-100] * padding_length
            )

        return {
            "input_ids": torch.tensor(
                input_ids,
                dtype=torch.long,
            ),
            "attention_mask": torch.tensor(
                attention_masks,
                dtype=torch.long,
            ),
            "labels": torch.tensor(
                labels,
                dtype=torch.long,
            ),
        }

def load_model():
    #Load Gemma from shared storage and attach LoRA adapters

    processor = AutoProcessor.from_pretrained(
        MODEL_PATH,
        local_files_only=True,
    )

    model = Gemma3ForConditionalGeneration.from_pretrained(
        MODEL_PATH,
        torch_dtype=torch.bfloat16,
        local_files_only=True,
    )

    lora_config = LoraConfig(
        r=16,
        lora_alpha=32,
        lora_dropout=0.05,
        target_modules=[
            "q_proj",
            "k_proj",
            "v_proj",
            "o_proj",
        ],
        bias="none",
        task_type="CAUSAL_LM",
    )

    model = get_peft_model(
        model,
        lora_config,
    )

    return model, processor


def parse_args():
    #Read options for smoke tests and full training runs
    
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--max-train-examples",
        type=int,
        default=None,
    )

    parser.add_argument(
        "--max-val-examples",
        type=int,
        default=None,
    )

    parser.add_argument(
        "--epochs",
        type=float,
        default=3.0,
    )

    parser.add_argument(
        "--output-dir",
        type=str,
        default=str(OUTPUT_DIR),
    )

    return parser.parse_args()


def main():
    args = parse_args()

    model, processor = load_model()

    print("\nTrainable parameters:")
    model.print_trainable_parameters()

    train_dataset = MMLUDataset(
        TRAIN_FILE,
        processor,
        max_examples=args.max_train_examples,
    )

    validation_dataset = MMLUDataset(
        VALIDATION_FILE,
        processor,
        max_examples=args.max_val_examples,
    )

    print(
        f"\nTraining examples: "
        f"{len(train_dataset)}"
    )

    print(
        f"Validation examples: "
        f"{len(validation_dataset)}"
    )

    training_args = TrainingArguments(
        output_dir=args.output_dir,

        # Training settings
        num_train_epochs=args.epochs,
        per_device_train_batch_size=4,
        per_device_eval_batch_size=4,
        gradient_accumulation_steps=4,
        learning_rate=2e-4,
        weight_decay=0.01,
        warmup_steps=2,

        # H200 supports BF16.
        bf16=True,
        fp16=False,

        # Evaluate and save after every epoch.
        eval_strategy="epoch",
        save_strategy="epoch",
        logging_steps=5,

        # Distributed training setting.
        ddp_find_unused_parameters=True,

        # Keep experiment reporting local.
        report_to="none",

        seed=1234,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=validation_dataset,
        data_collator=DataCollator(
            processor.tokenizer
        ),
    )

    trainer.train()

    final_dir = (
        Path(args.output_dir)
        / "final_adapter"
    )

    trainer.save_model(final_dir)

    print(
        f"\nFinal LoRA adapter saved to: "
        f"{final_dir}"
    )

    if dist.is_available() and dist.is_initialized():
        dist.destroy_process_group()

if __name__ == "__main__":
    main()