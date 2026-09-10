import os
from typing import Optional
from datasets import load_dataset, Dataset


def prepare_grpo_dataset(
    dataset_path_or_id: str,
    split: str = "train",
    max_samples: Optional[int] = None,
) -> Dataset:
    """Loads and formats prompt dataset for GRPO reinforcement learning."""
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

    def format_row(example):
        prompt = example["prompt"]
        if isinstance(prompt, str):
            formatted_prompt = [{"role": "user", "content": prompt}]
        else:
            formatted_prompt = prompt
        result = {"prompt": formatted_prompt}
        if "answer" in example:
            result["answer"] = str(example["answer"]) if example["answer"] is not None else None
        return result

    return dataset.map(format_row)
