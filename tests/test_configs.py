import os
import pathlib
import pytest
import yaml

ROOT_DIR = pathlib.Path(__file__).resolve().parent.parent

CONFIG_PATHS = [
    "configs/sft/smoke_test.yaml",
    "configs/grpo/smoke_test.yaml",
    "configs/sft/gemma_text_sft.yaml",
    "configs/grpo/gemma_sudoku_rl.yaml",
    "configs/sft/qwen35_08b_story_writing.yaml",
]



@pytest.mark.parametrize("rel_path", CONFIG_PATHS)
def test_config_file_exists_and_is_valid_yaml(rel_path):
    full_path = ROOT_DIR / rel_path
    assert full_path.is_file(), f"Configuration file missing: {rel_path}"

    with open(full_path, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    assert isinstance(config, dict), f"Config in {rel_path} must be a dictionary"
    assert "stage" in config, f"Config in {rel_path} missing 'stage' key"
    assert config["stage"] in ("sft", "grpo")
    assert "model" in config
    assert "dataset" in config
    assert "training" in config
    assert "output" in config


def test_sft_smoke_test_config_keys():
    config_path = ROOT_DIR / "configs/sft/smoke_test.yaml"
    with open(config_path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    assert cfg["stage"] == "sft"
    assert cfg["model"]["name_or_path"] == "unsloth/gemma-2-2b-it"
    assert cfg["model"]["max_seq_length"] == 512
    assert cfg["model"]["load_in_4bit"] is True
    assert cfg["lora"]["r"] == 8
    assert "q_proj" in cfg["lora"]["target_modules"]
    assert cfg["dataset"]["path"] == "data/dummy/sft_sample.jsonl"
    assert cfg["training"]["max_steps"] == 2
    assert cfg["output"]["output_dir"] == "outputs/sft_smoke_test"


def test_sft_gemma_recipe_config_keys():
    config_path = ROOT_DIR / "configs/sft/gemma_text_sft.yaml"
    with open(config_path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    assert cfg["stage"] == "sft"
    assert cfg["model"]["name_or_path"] == "unsloth/gemma-2-2b-it"
    assert cfg["model"]["max_seq_length"] == 2048
    assert cfg["lora"]["r"] == 16
    assert cfg["lora"]["lora_alpha"] == 32
    assert "gate_proj" in cfg["lora"]["target_modules"]
    assert cfg["dataset"]["path"] == "mlabonne/FineTome-100k"
    assert cfg["training"]["max_steps"] == 100
    assert cfg["output"]["output_dir"] == "outputs/gemma_sft"


def test_grpo_smoke_test_config_keys():
    config_path = ROOT_DIR / "configs/grpo/smoke_test.yaml"
    with open(config_path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    assert cfg["stage"] == "grpo"
    assert cfg["model"]["name_or_path"] == "unsloth/gemma-2-2b-it"
    assert cfg["model"]["max_seq_length"] == 512
    assert cfg["lora"]["r"] == 8
    assert "rewards" in cfg
    assert isinstance(cfg["rewards"], list)
    assert "xml_format" in cfg["rewards"]
    assert "exact_match" in cfg["rewards"]
    assert cfg["training"]["num_generations"] == 2
    assert cfg["output"]["output_dir"] == "outputs/grpo_smoke_test"


def test_grpo_sudoku_recipe_config_keys():
    config_path = ROOT_DIR / "configs/grpo/gemma_sudoku_rl.yaml"
    with open(config_path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    assert cfg["stage"] == "grpo"
    assert cfg["model"]["name_or_path"] == "unsloth/gemma-2-2b-it"
    assert cfg["model"]["max_seq_length"] == 1024
    assert cfg["lora"]["r"] == 16
    assert "rewards" in cfg
    assert isinstance(cfg["rewards"], list)
    assert set(cfg["rewards"]) == {"xml_format", "exact_match", "code_execution"}
    assert cfg["training"]["num_generations"] == 4
    assert cfg["training"]["max_steps"] == 50
    assert cfg["output"]["output_dir"] == "outputs/gemma_sudoku_grpo"
