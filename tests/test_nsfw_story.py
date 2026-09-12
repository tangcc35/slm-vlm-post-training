import json
import pytest
from slm_post_train.data.nsfw_story import (
    clean_prompt,
    clean_story,
    format_story_sample,
    process_nsfw_story_dataset,
    curate_and_save_dataset,
    curate_from_config,
    DEFAULT_STORY_SYSTEM_PROMPT,
)


def test_clean_prompt_removes_reddit_tags():
    raw_prompt = "[WP] [NSFW] As a secret agent, your cover is blown in the bedroom."
    cleaned = clean_prompt(raw_prompt)
    assert cleaned == "As a secret agent, your cover is blown in the bedroom."


def test_clean_prompt_removes_urls_and_subreddits():
    raw_prompt = "Write a story based on this r/WritingPrompts idea: Two lovers meet again. http://reddit.com/r/test"
    cleaned = clean_prompt(raw_prompt)
    assert "http" not in cleaned
    assert "Two lovers meet again." in cleaned


def test_clean_story_removes_reddit_trailers():
    story = (
        "The candles flickered against the oak walls as they embraced.\n\n"
        "Her fingers traced the line of his collarbone with quiet reverence.\n\n"
        "***\n\n"
        "Edit: Thanks for the gold everyone! Part 2 will be posted on my profile u/author."
    )
    cleaned = clean_story(story)
    assert "Thanks for the gold" not in cleaned
    assert "u/author" not in cleaned
    assert "Her fingers traced the line of his collarbone" in cleaned


def test_clean_story_normalizes_formatting():
    story = "He whispered,  \"stay with me.\"\n\n\n\n*****\n\nShe leaned closer."
    cleaned = clean_story(story)
    assert "*****" not in cleaned
    assert "stay with me" in cleaned
    assert "\n\n\n\n" not in cleaned


def test_format_story_sample_structure():
    prompt = "Write an intimate reunion scene."
    story = "After years apart, their eyes met across the dimly lit study."
    sample = format_story_sample(prompt, story, system_prompt=DEFAULT_STORY_SYSTEM_PROMPT)

    assert "conversations" in sample
    convo = sample["conversations"]
    assert len(convo) == 3
    assert convo[0]["role"] == "system"
    assert convo[0]["content"] == DEFAULT_STORY_SYSTEM_PROMPT
    assert convo[1]["role"] == "user"
    assert convo[1]["content"] == prompt
    assert convo[2]["role"] == "assistant"
    assert convo[2]["content"] == story


def test_format_story_sample_without_system():
    prompt = "Write an intimate reunion scene."
    story = "After years apart, their eyes met across the dimly lit study."
    sample = format_story_sample(prompt, story, system_prompt=None)

    assert len(sample["conversations"]) == 2
    assert sample["conversations"][0]["role"] == "user"
    assert sample["conversations"][1]["role"] == "assistant"


def test_process_nsfw_story_dataset_filtering():
    raw_records = [
        # Valid sample
        {
            "conversations": [
                {"from": "human", "value": "[WP] Write a slow romance."},
                {"from": "gpt", "value": "They shared wine by the terrace. " * 30},
            ]
        },
        # Too short (< 20 words in this test setting)
        {
            "conversations": [
                {"from": "human", "value": "Short prompt"},
                {"from": "gpt", "value": "Too short story."},
            ]
        },
        # Prompt too short (< 3 words)
        {
            "conversations": [
                {"from": "human", "value": "Hi"},
                {"from": "gpt", "value": "A story passage. " * 30},
            ]
        },
    ]

    processed = process_nsfw_story_dataset(
        raw_records,
        min_words=20,
        max_words=2000,
        include_system_ratio=1.0,
    )
    assert len(processed) == 1
    assert processed[0]["conversations"][1]["content"] == "Write a slow romance."



def test_curate_and_save_dataset_local(tmp_path):
    from slm_post_train.data.nsfw_story import curate_and_save_dataset

    raw_file = tmp_path / "raw_data.jsonl"
    with open(raw_file, "w", encoding="utf-8") as f:
        for i in range(10):
            sample = {
                "conversations": [
                    {"from": "human", "value": f"[WP] Romance prompt number {i}."},
                    {"from": "gpt", "value": f"A beautiful romantic story line {i}. " * 15},
                ]
            }
            f.write(json.dumps(sample) + "\n")

    out_dir = tmp_path / "processed"
    train_path, val_path = curate_and_save_dataset(
        dataset_name_or_path=str(raw_file),
        output_dir=out_dir,
        min_words=10,
        max_words=1000,
        val_split=0.2,
        seed=42,
    )

    assert train_path.exists()
    assert val_path.exists()

    with open(train_path, "r") as f:
        train_lines = [json.loads(line) for line in f]
    with open(val_path, "r") as f:
        val_lines = [json.loads(line) for line in f]

    assert len(train_lines) == 8
    assert len(val_lines) == 2
    assert "conversations" in train_lines[0]


def test_curate_from_config_local(tmp_path):
    raw_file = tmp_path / "raw_data.jsonl"
    with open(raw_file, "w", encoding="utf-8") as f:
        for i in range(10):
            sample = {
                "conversations": [
                    {"from": "human", "value": f"[WP] Romance prompt {i}."},
                    {"from": "gpt", "value": f"Story content line {i}. " * 15},
                ]
            }
            f.write(json.dumps(sample) + "\n")

    out_dir = tmp_path / "cfg_processed"
    config = {
        "source": str(raw_file),
        "output_dir": str(out_dir),
        "min_words": 10,
        "max_words": 1000,
        "val_split": 0.3,
        "seed": 42,
    }
    train_path, val_path = curate_from_config(config)
    assert train_path.exists()
    assert val_path.exists()

    with open(train_path, "r") as f:
        train_lines = [json.loads(line) for line in f]
    with open(val_path, "r") as f:
        val_lines = [json.loads(line) for line in f]

    assert len(train_lines) == 7
    assert len(val_lines) == 3


