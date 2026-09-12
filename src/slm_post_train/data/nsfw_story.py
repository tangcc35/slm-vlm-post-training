import json
import logging
import os
import pathlib
import random
import re
from typing import Any, Dict, Iterable, List, Optional, Tuple, Union

logger = logging.getLogger(__name__)

DEFAULT_STORY_SYSTEM_PROMPT = (
    "You are an uncensored, immersive creative writer specializing in atmospheric adult fiction "
    "with rich sensory detail, engaging dialogue, and vivid scene descriptions."
)

# Common tag prefixes in prompt datasets (e.g., Reddit WritingPrompts)
PROMPT_TAG_PATTERN = re.compile(
    r"^\s*\[\s*(?:wp|nsfw|prompt|cw|eu|rf|ip|pi|ot|mp|pm)\s*\]\s*",
    re.IGNORECASE,
)

# End-of-story trailers, Reddit-specific notes, self-promo
TRAILER_PATTERNS = [
    re.compile(r"(?i)\b(?:edit\s*\d*|update\s*\d*)\s*:.*$", re.DOTALL),
    re.compile(r"(?i)\b(?:thanks for reading|thank you for reading|part \d+ will be|read more at|find more stories).*$", re.DOTALL),
    re.compile(r"(?i)check out my (?:profile|patreon|subreddit|sub)\b.*$", re.DOTALL),
    re.compile(r"(?i)posted on my profile\b.*$", re.DOTALL),
]

# Patterns for links, usernames, and subreddits
URL_PATTERN = re.compile(r"https?://\S+|www\.\S+")
REDDIT_USER_SUB_PATTERN = re.compile(r"(?:\b[ru]/[A-Za-z0-9_/-]+|\b[ru]\\[A-Za-z0-9_/-]+)")


def clean_prompt(prompt: str) -> str:
    """Cleans writing prompt text by stripping tags, URLs, and artifacts."""
    if not prompt:
        return ""
    text = prompt.strip()
    # Strip multiple consecutive tag prefixes (e.g. [WP] [NSFW])
    while True:
        m = PROMPT_TAG_PATTERN.match(text)
        if m:
            text = text[m.end():].strip()
        else:
            break

    # Strip URLs
    text = URL_PATTERN.sub("", text)
    # Strip r/... and u/... mentions
    text = REDDIT_USER_SUB_PATTERN.sub("", text)

    # Normalize whitespace
    text = re.sub(r"[ \t]+", " ", text).strip()
    return text


def clean_story(story: str) -> str:
    """Cleans narrative prose by removing author trailers, edit notes, and formatting glitches."""
    if not story:
        return ""
    text = story.strip()

    # Strip author trailers / edit notes
    for pat in TRAILER_PATTERNS:
        text = pat.sub("", text).strip()

    # Strip URLs and usernames/subreddits
    text = URL_PATTERN.sub("", text)
    text = REDDIT_USER_SUB_PATTERN.sub("", text)

    # Normalize repetitive scene dividers (like ***** or -----)
    text = re.sub(r"[\*\-_=]{3,}", "***", text)
    # Remove divider if at start or end of text
    text = re.sub(r"^\s*\*\*\*\s*", "", text)
    text = re.sub(r"\s*\*\*\*\s*$", "", text)

    # Normalize excessive newlines and whitespace
    text = re.sub(r"\n{3,}", "\n\n", text)
    text = re.sub(r"[ \t]+", " ", text)
    return text.strip()


def format_story_sample(
    prompt: str,
    story: str,
    system_prompt: Optional[str] = DEFAULT_STORY_SYSTEM_PROMPT,
) -> Dict[str, List[Dict[str, str]]]:
    """Formats prompt and story into a standard ChatML conversation dict."""
    conversations: List[Dict[str, str]] = []
    if system_prompt:
        conversations.append({"role": "system", "content": system_prompt})
    conversations.append({"role": "user", "content": prompt})
    conversations.append({"role": "assistant", "content": story})
    return {"conversations": conversations}


def extract_prompt_and_story(record: Dict[str, Any]) -> Tuple[Optional[str], Optional[str]]:
    """Extracts prompt and story text from various dataset formats."""
    # Format 1: ShareGPT conversations [{'from': 'human', 'value': ...}, {'from': 'gpt', ...}]
    # Format 2: Chat conversations [{'role': 'user', 'content': ...}, {'role': 'assistant', ...}]
    convo = record.get("conversations") or record.get("conversation") or record.get("messages")
    if convo and isinstance(convo, list) and len(convo) >= 2:
        user_msg = None
        assistant_msg = None
        for turn in convo:
            role = turn.get("from") or turn.get("role")
            content = turn.get("value") or turn.get("content")
            if role in ["human", "user"] and not user_msg:
                user_msg = content
            elif role in ["gpt", "assistant", "bot"] and not assistant_msg:
                assistant_msg = content
        if user_msg and assistant_msg:
            return str(user_msg), str(assistant_msg)

    # Format 3: Explicit prompt & response keys
    if "prompt" in record and ("response" in record or "completion" in record or "story" in record):
        story = record.get("response") or record.get("completion") or record.get("story")
        return str(record["prompt"]), str(story)

    # Format 4: instruction & output
    if "instruction" in record and "output" in record:
        inst = record["instruction"]
        if record.get("input"):
            inst = f"{inst}\n{record['input']}"
        return str(inst), str(record["output"])

    return None, None


def process_nsfw_story_dataset(
    records: Iterable[Dict[str, Any]],
    min_words: int = 200,
    max_words: int = 2500,
    system_prompt: Optional[str] = DEFAULT_STORY_SYSTEM_PROMPT,
    include_system_ratio: float = 0.85,
    seed: int = 42,
) -> List[Dict[str, Any]]:
    """Filters, cleans, and standardizes raw dataset records for training."""
    rng = random.Random(seed)
    processed_samples: List[Dict[str, Any]] = []
    seen_prompts = set()

    for item in records:
        raw_prompt, raw_story = extract_prompt_and_story(item)
        if not raw_prompt or not raw_story:
            continue

        p_clean = clean_prompt(raw_prompt)
        s_clean = clean_story(raw_story)

        # Word count checks on story
        words = len(s_clean.split())
        if words < min_words or words > max_words:
            continue

        # Check prompt length (must have substance)
        if len(p_clean.split()) < 3:
            continue

        # Deduplication on prompt
        prompt_norm = re.sub(r"\W+", "", p_clean.lower())
        if prompt_norm in seen_prompts:
            continue
        seen_prompts.add(prompt_norm)

        # Randomly choose whether to include system prompt
        use_sys = system_prompt if rng.random() < include_system_ratio else None
        formatted = format_story_sample(p_clean, s_clean, system_prompt=use_sys)
        processed_samples.append(formatted)

    return processed_samples


def save_dataset_jsonl(samples: List[Dict[str, Any]], output_path: Union[str, pathlib.Path]) -> None:
    """Saves formatted dataset samples to a JSONL file."""
    path = pathlib.Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for s in samples:
            f.write(json.dumps(s, ensure_ascii=False) + "\n")
    logger.info(f"Saved {len(samples)} samples to {path}")


def curate_and_save_dataset(
    dataset_name_or_path: Union[str, pathlib.Path] = "ChaoticNeutrals/Reddit-NSFW-Writing_Prompts_ShareGPT",
    output_dir: Union[str, pathlib.Path] = "data/processed/nsfw_story",
    split: str = "train",
    min_words: int = 200,
    max_words: int = 2500,
    val_split: float = 0.05,
    system_prompt: Optional[str] = DEFAULT_STORY_SYSTEM_PROMPT,
    include_system_ratio: float = 0.85,
    max_samples: Optional[int] = None,
    seed: int = 42,
) -> Tuple[pathlib.Path, pathlib.Path]:
    """Curates, filters, and splits an NSFW story dataset, saving to JSONL format."""
    from datasets import load_dataset

    source_path = str(dataset_name_or_path)
    output_path = pathlib.Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    if os.path.exists(source_path):
        if source_path.endswith(".jsonl") or source_path.endswith(".json"):
            raw_data = load_dataset("json", data_files=source_path, split=split)
        elif source_path.endswith(".csv"):
            raw_data = load_dataset("csv", data_files=source_path, split=split)
        else:
            raise ValueError(f"Unsupported file format: {source_path}")
    else:
        raw_data = load_dataset(source_path, split=split)

    if max_samples is not None:
        raw_data = raw_data.select(range(min(len(raw_data), max_samples)))

    logger.info(f"Processing {len(raw_data)} raw records...")
    processed = process_nsfw_story_dataset(
        records=raw_data,
        min_words=min_words,
        max_words=max_words,
        system_prompt=system_prompt,
        include_system_ratio=include_system_ratio,
        seed=seed,
    )
    logger.info(f"Retained {len(processed)} samples after filtering and deduplication.")

    rng = random.Random(seed)
    rng.shuffle(processed)

    val_count = int(len(processed) * val_split)
    val_count = max(1, val_count) if (0 < val_split < 1 and len(processed) > 1) else 0

    val_samples = processed[:val_count]
    train_samples = processed[val_count:]

    train_file = output_path / "train.jsonl"
    val_file = output_path / "val.jsonl"

    save_dataset_jsonl(train_samples, train_file)
    save_dataset_jsonl(val_samples, val_file)

    logger.info(f"Dataset split: {len(train_samples)} train, {len(val_samples)} validation.")
    return train_file, val_file


def curate_from_config(config: Dict[str, Any]) -> Tuple[pathlib.Path, pathlib.Path]:
    """Curates, filters, and splits dataset using configuration dictionary loaded from YAML."""
    dataset_source = (
        config.get("source")
        or config.get("dataset")
        or config.get("dataset_name_or_path")
        or "ChaoticNeutrals/Reddit-NSFW-Writing_Prompts_ShareGPT"
    )
    output_dir = config.get("output_dir", "data/processed/nsfw_story")
    split = config.get("split", "train")
    min_words = int(config.get("min_words", 200))
    max_words = int(config.get("max_words", 2500))
    val_split = float(config.get("val_split", 0.05))
    system_prompt = config.get("system_prompt", DEFAULT_STORY_SYSTEM_PROMPT)
    include_system_ratio = float(config.get("include_system_ratio", 0.85))
    max_samples = config.get("max_samples")
    if max_samples is not None:
        max_samples = int(max_samples)
    seed = int(config.get("seed", 42))

    return curate_and_save_dataset(
        dataset_name_or_path=dataset_source,
        output_dir=output_dir,
        split=split,
        min_words=min_words,
        max_words=max_words,
        val_split=val_split,
        system_prompt=system_prompt,
        include_system_ratio=include_system_ratio,
        max_samples=max_samples,
        seed=seed,
    )


def main():
    import argparse
    from slm_post_train.cli import load_yaml_config

    parser = argparse.ArgumentParser(description="Curate and format NSFW story writing dataset from YAML config.")
    parser.add_argument("--config", required=True, help="Path to YAML training configuration file")
    args = parser.parse_args()

    config = load_yaml_config(args.config)
    curation_cfg = config.get("dataset", {}).get("curation") or config.get("curation")
    if not curation_cfg:
        raise ValueError(
            f"No curation configuration found in {args.config}. "
            "Please specify a 'curation' section under 'dataset' (or at root level)."
        )

    train_p, val_p = curate_from_config(curation_cfg)
    print(f"Curation complete!\nTrain: {train_p}\nValidation: {val_p}")


if __name__ == "__main__":
    main()

