import os
from typing import Optional, Union
import pathlib
from datasets import load_dataset, Dataset


def prepare_grpo_dataset(
    dataset_path_or_id: Union[str, pathlib.Path],
    split: str = "train",
    max_samples: Optional[int] = None,
) -> Dataset:
    """Loads and formats prompt dataset for GRPO reinforcement learning."""
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

    def format_row(example):
        prompt = None
        for key in ("prompt", "question", "problem"):
            if example.get(key) is not None:
                prompt = example[key]
                break

        if prompt is None:
            raise ValueError(
                f"Dataset row does not contain a recognized prompt column ('prompt', 'question', 'problem'). Available columns: {list(example.keys())}"
            )

        if isinstance(prompt, str):
            formatted_prompt = [{"role": "user", "content": prompt}]
        else:
            formatted_prompt = prompt

        result = {"prompt": formatted_prompt}

        raw_answer = None
        for key in ("answer", "solution", "ground_truth"):
            if example.get(key) is not None:
                raw_answer = example[key]
                break

        if raw_answer is not None:
            result["answer"] = str(raw_answer)

        return result

    return dataset.map(format_row)
