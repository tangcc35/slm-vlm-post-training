import sys
from unittest.mock import MagicMock, patch
import pytest
from slm_post_train.cli import load_yaml_config, main


def test_load_yaml_config_valid(tmp_path):
    p = tmp_path / "test.yaml"
    p.write_text("stage: sft\nlearning_rate: 0.001\n", encoding="utf-8")
    data = load_yaml_config(str(p))
    assert data == {"stage": "sft", "learning_rate": 0.001}


def test_load_yaml_config_empty(tmp_path):
    p = tmp_path / "empty.yaml"
    p.write_text("", encoding="utf-8")
    data = load_yaml_config(str(p))
    assert data == {}


def test_load_yaml_config_nonexistent():
    with pytest.raises(FileNotFoundError):
        load_yaml_config("nonexistent_file_path.yaml")


def test_cli_no_arguments():
    with pytest.raises(SystemExit):
        main([])


def test_cli_train_missing_config():
    with pytest.raises(SystemExit):
        main(["train"])


def test_cli_train_sft(tmp_path):
    config_path = tmp_path / "sft_test.yaml"
    config_path.write_text("stage: sft\nlearning_rate: 0.001\n", encoding="utf-8")

    with patch("slm_post_train.trainers.sft_runner.run_sft") as mock_run_sft:
        main(["train", "--config", str(config_path)])
        mock_run_sft.assert_called_once_with({"stage": "sft", "learning_rate": 0.001})


def test_cli_train_grpo(tmp_path):
    config_path = tmp_path / "grpo_test.yaml"
    config_path.write_text("stage: grpo\nrewards:\n  - xml_format\n", encoding="utf-8")

    with patch("slm_post_train.trainers.grpo_runner.run_grpo") as mock_run_grpo:
        main(["train", "--config", str(config_path)])
        mock_run_grpo.assert_called_once_with({"stage": "grpo", "rewards": ["xml_format"]})


def test_cli_train_invalid_stage(tmp_path):
    config_path = tmp_path / "invalid_stage.yaml"
    config_path.write_text("stage: dpo\n", encoding="utf-8")

    with pytest.raises(ValueError, match="Unknown training stage: 'dpo'"):
        main(["train", "--config", str(config_path)])


def test_cli_train_default_stage_is_sft(tmp_path):
    config_path = tmp_path / "default_stage.yaml"
    config_path.write_text("learning_rate: 0.001\n", encoding="utf-8")

    with patch("slm_post_train.trainers.sft_runner.run_sft") as mock_run_sft:
        main(["train", "--config", str(config_path)])
        mock_run_sft.assert_called_once_with({"learning_rate": 0.001})


def test_cli_export_defaults():
    mock_model = MagicMock()
    mock_tokenizer = MagicMock()

    with patch("unsloth.FastLanguageModel.from_pretrained", return_value=(mock_model, mock_tokenizer)) as mock_load, \
         patch("slm_post_train.export.exporter.export_model") as mock_export:
        main(["export", "--model-path", "models/my_model", "--output-dir", "outputs/exported"])
        mock_load.assert_called_once_with("models/my_model")
        mock_export.assert_called_once_with(
            mock_model,
            mock_tokenizer,
            "outputs/exported",
            export_format="lora",
            quantization_method="q4_k_m",
        )


def test_cli_export_custom_format_and_quant():
    mock_model = MagicMock()
    mock_tokenizer = MagicMock()

    with patch("unsloth.FastLanguageModel.from_pretrained", return_value=(mock_model, mock_tokenizer)) as mock_load, \
         patch("slm_post_train.export.exporter.export_model") as mock_export:
        main([
            "export",
            "--model-path", "models/my_model",
            "--output-dir", "outputs/exported",
            "--format", "gguf",
            "--quant", "q8_0",
        ])
        mock_load.assert_called_once_with("models/my_model")
        mock_export.assert_called_once_with(
            mock_model,
            mock_tokenizer,
            "outputs/exported",
            export_format="gguf",
            quantization_method="q8_0",
        )


def test_cli_export_invalid_format():
    with pytest.raises(SystemExit):
        main([
            "export",
            "--model-path", "models/my_model",
            "--output-dir", "outputs/exported",
            "--format", "invalid_format",
        ])


def test_cli_export_missing_required_args():
    with pytest.raises(SystemExit):
        main(["export", "--model-path", "models/my_model"])


def test_cli_main_sys_argv(tmp_path):
    config_path = tmp_path / "sft_argv.yaml"
    config_path.write_text("stage: sft\n", encoding="utf-8")

    test_args = ["slm-post-train", "train", "--config", str(config_path)]
    with patch.object(sys, "argv", test_args):
        with patch("slm_post_train.trainers.sft_runner.run_sft") as mock_run_sft:
            main()
            mock_run_sft.assert_called_once_with({"stage": "sft"})
