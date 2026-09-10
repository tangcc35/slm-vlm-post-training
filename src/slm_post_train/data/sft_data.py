import os
from typing import Optional, Union
import pathlib

import unsloth
from datasets import load_dataset, Dataset
from unsloth.chat_templates import get_chat_template, standardize_data_formats


def prepare_sft_dataset(
    dataset_path_or_id: Union[str, pathlib.Path],
    tokenizer,
    chat_template: str = "chatml",
    split: str = "train",
    max_samples: Optional[int] = None,
) -> Dataset:
    """Loads and standardizes chat dataset for SFT training."""
    dataset_path_or_id = str(dataset_path_or_id)

    if os.path.exists(dataset_path_or_id):
        if dataset_path_or_id.endswith(".jsonl") or dataset_path_or_id.endswith(".json"):
            dataset = load_dataset("json", data_files=dataset_path_or_id, split=split)
        elif dataset_path_or_id.endswith(".csv"):
            dataset = load_dataset("csv", data_files=dataset_path_or_id, split=split)
        else:
            raise ValueError(f"Unsupported file format: {dataset_path_or_id}")
    else:
        dataset = load_dataset(dataset_path_or_id, split=split)

    if max_samples is not None:
        dataset = dataset.select(range(min(len(dataset), max_samples)))

    # Normalize tabular instruction/prompt formats to conversations if not already present
    if "conversations" not in dataset.column_names:
        if "messages" in dataset.column_names:
            dataset = dataset.rename_column("messages", "conversations")
        elif "instruction" in dataset.column_names:
            output_col = "output" if "output" in dataset.column_names else "response"

            def to_conversations(example):
                user_content = example["instruction"]
                if "input" in example and example["input"]:
                    user_content = f"{user_content}\n{example['input']}"
                return {
                    "conversations": [
                        {"role": "user", "content": user_content},
                        {"role": "assistant", "content": example.get(output_col, "")},
                    ]
                }

            dataset = dataset.map(to_conversations)
        elif "prompt" in dataset.column_names:
            resp_col = "response" if "response" in dataset.column_names else "completion"

            def to_conversations(example):
                return {
                    "conversations": [
                        {"role": "user", "content": example["prompt"]},
                        {"role": "assistant", "content": example.get(resp_col, "")},
                    ]
                }

            dataset = dataset.map(to_conversations)

    tokenizer = get_chat_template(tokenizer, chat_template=chat_template)
    dataset = standardize_data_formats(dataset)

    def formatting_prompts_func(examples):
        convos = examples["conversations"]
        texts = [
            tokenizer.apply_chat_template(convo, tokenize=False, add_generation_prompt=False)
            for convo in convos
        ]
        return {"text": texts}

    dataset = dataset.map(formatting_prompts_func, batched=True)
    return dataset
