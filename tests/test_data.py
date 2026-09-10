import os
import pathlib
import tempfile
from unittest.mock import patch

import unsloth
import pytest
from tokenizers import Tokenizer
from tokenizers.models import WordLevel
from transformers import PreTrainedTokenizerFast
from datasets import Dataset

from slm_post_train.data.grpo_data import prepare_grpo_dataset
from slm_post_train.data.sft_data import prepare_sft_dataset
from slm_post_train.data import (
    prepare_grpo_dataset as grpo_export,
    prepare_sft_dataset as sft_export,
)


@pytest.fixture
def fast_tokenizer():
    tok = Tokenizer(
        WordLevel(
            vocab={"<unk>": 0, "<|im_start|>": 1, "<|im_end|>": 2},
            unk_token="<unk>",
        )
    )
    return PreTrainedTokenizerFast(
        tokenizer_object=tok,
        unk_token="<unk>",
        pad_token="<unk>",
        eos_token="<unk>",
    )


# --- GRPO Dataset Tests ---

def test_prepare_grpo_dataset_dummy():
    dataset = prepare_grpo_dataset("data/dummy/grpo_sample.jsonl")
    assert len(dataset) == 5
    sample = dataset[0]
    assert "prompt" in sample
    assert "answer" in sample
    assert isinstance(sample["prompt"], list)
    assert sample["prompt"][0]["role"] == "user"
    assert "15 + 27" in sample["prompt"][0]["content"]
    assert sample["answer"] == "42"


def test_prepare_grpo_dataset_pathlib():
    path = pathlib.Path("data/dummy/grpo_sample.jsonl")
    dataset = prepare_grpo_dataset(path)
    assert len(dataset) == 5
    assert dataset[0]["answer"] == "42"


def test_prepare_grpo_dataset_aliases():
    with tempfile.NamedTemporaryFile(mode="w", suffix=".jsonl", delete=False) as f:
        f.write('{"question": "What is 3*3?", "solution": "9"}\n')
        f.write('{"problem": "Solve for x: x=10", "ground_truth": "10"}\n')
        f_path = f.name
    try:
        dataset = prepare_grpo_dataset(f_path)
        assert len(dataset) == 2
        # First row: question -> prompt, solution -> answer
        assert dataset[0]["prompt"] == [{"role": "user", "content": "What is 3*3?"}]
        assert dataset[0]["answer"] == "9"
        # Second row: problem -> prompt, ground_truth -> answer
        assert dataset[1]["prompt"] == [{"role": "user", "content": "Solve for x: x=10"}]
        assert dataset[1]["answer"] == "10"
    finally:
        os.remove(f_path)


def test_prepare_grpo_dataset_missing_prompt_raises():
    with tempfile.NamedTemporaryFile(mode="w", suffix=".jsonl", delete=False) as f:
        f.write('{"unrecognized_col": "some input", "answer": "42"}\n')
        f_path = f.name
    try:
        with pytest.raises(ValueError, match="prompt"):
            prepare_grpo_dataset(f_path)
    finally:
        os.remove(f_path)


def test_prepare_grpo_dataset_max_samples():
    dataset = prepare_grpo_dataset("data/dummy/grpo_sample.jsonl", max_samples=2)
    assert len(dataset) == 2

    # max_samples exceeding dataset size should return all samples
    dataset_large = prepare_grpo_dataset("data/dummy/grpo_sample.jsonl", max_samples=100)
    assert len(dataset_large) == 5


def test_prepare_grpo_dataset_preformatted_prompt():
    with tempfile.NamedTemporaryFile(mode="w", suffix=".jsonl", delete=False) as f:
        f.write('{"prompt": [{"role": "user", "content": "hello"}], "answer": "world"}\n')
        f_path = f.name
    try:
        dataset = prepare_grpo_dataset(f_path)
        assert len(dataset) == 1
        assert dataset[0]["prompt"] == [{"role": "user", "content": "hello"}]
        assert dataset[0]["answer"] == "world"
    finally:
        os.remove(f_path)


def test_prepare_grpo_dataset_csv():
    with tempfile.NamedTemporaryFile(mode="w", suffix=".csv", delete=False) as f:
        f.write("prompt,answer\nWhat is 2+2?,4\nWhat is 3+3?,6\n")
        f_path = f.name
    try:
        dataset = prepare_grpo_dataset(f_path)
        assert len(dataset) == 2
        assert dataset[0]["prompt"] == [{"role": "user", "content": "What is 2+2?"}]
        assert str(dataset[0]["answer"]) == "4"
    finally:
        os.remove(f_path)


def test_prepare_grpo_dataset_unsupported_format():
    with tempfile.NamedTemporaryFile(suffix=".txt", delete=False) as f:
        f.write(b"unsupported content")
        f_path = f.name
    try:
        with pytest.raises(ValueError, match="Unsupported file format"):
            prepare_grpo_dataset(f_path)
    finally:
        os.remove(f_path)


def test_prepare_grpo_dataset_hf_id():
    mock_ds = Dataset.from_dict({
        "prompt": ["What is 5*5?"],
        "answer": ["25"],
    })
    with patch("slm_post_train.data.grpo_data.load_dataset", return_value=mock_ds) as mock_load:
        dataset = prepare_grpo_dataset("org/hf_grpo_dataset", split="train")
        mock_load.assert_called_once_with("org/hf_grpo_dataset", split="train")
        assert len(dataset) == 1
        assert dataset[0]["prompt"] == [{"role": "user", "content": "What is 5*5?"}]
        assert dataset[0]["answer"] == "25"


# --- SFT Dataset Tests ---

def test_prepare_sft_dataset_dummy(fast_tokenizer):
    dataset = prepare_sft_dataset("data/dummy/sft_sample.jsonl", tokenizer=fast_tokenizer)
    assert len(dataset) == 5
    sample = dataset[0]
    assert "conversations" in sample
    assert "text" in sample
    assert "<|im_start|>user" in sample["text"]
    assert "What is the capital of France?" in sample["text"]
    assert "<|im_start|>assistant" in sample["text"]
    assert "Paris" in sample["text"]


def test_prepare_sft_dataset_pathlib(fast_tokenizer):
    path = pathlib.Path("data/dummy/sft_sample.jsonl")
    dataset = prepare_sft_dataset(path, tokenizer=fast_tokenizer)
    assert len(dataset) == 5
    assert "text" in dataset[0]


def test_prepare_sft_dataset_max_samples(fast_tokenizer):
    dataset = prepare_sft_dataset(
        "data/dummy/sft_sample.jsonl",
        tokenizer=fast_tokenizer,
        max_samples=3,
    )
    assert len(dataset) == 3


def test_prepare_sft_dataset_csv(fast_tokenizer):
    with tempfile.NamedTemporaryFile(mode="w", suffix=".csv", delete=False) as f:
        f.write('instruction,response\n"Say hi","Hello there!"\n"Solve: 2+2","4"\n')
        f_path = f.name
    try:
        dataset = prepare_sft_dataset(f_path, tokenizer=fast_tokenizer)
        assert len(dataset) == 2
        assert "text" in dataset[0]
        assert "Say hi" in dataset[0]["text"]
        assert "Hello there!" in dataset[0]["text"]
    finally:
        os.remove(f_path)


def test_prepare_sft_dataset_unsupported_format(fast_tokenizer):
    with tempfile.NamedTemporaryFile(suffix=".txt", delete=False) as f:
        f.write(b"unsupported")
        f_path = f.name
    try:
        with pytest.raises(ValueError, match="Unsupported file format"):
            prepare_sft_dataset(f_path, tokenizer=fast_tokenizer)
    finally:
        os.remove(f_path)


def test_prepare_sft_dataset_hf_id(fast_tokenizer):
    mock_ds = Dataset.from_dict({
        "conversations": [
            [
                {"role": "user", "content": "HF prompt 1"},
                {"role": "assistant", "content": "HF response 1"},
            ],
            [
                {"role": "user", "content": "HF prompt 2"},
                {"role": "assistant", "content": "HF response 2"},
            ],
        ],
    })
    with patch("slm_post_train.data.sft_data.load_dataset", return_value=mock_ds) as mock_load:
        dataset = prepare_sft_dataset("org/hf_sft_dataset", tokenizer=fast_tokenizer, split="train")
        mock_load.assert_called_once_with("org/hf_sft_dataset", split="train")
        assert len(dataset) == 2
        assert "text" in dataset[0]
        assert "HF prompt 1" in dataset[0]["text"]
        assert "HF response 1" in dataset[0]["text"]


# --- Module exports test ---

def test_data_package_exports():
    assert grpo_export is prepare_grpo_dataset
    assert sft_export is prepare_sft_dataset
