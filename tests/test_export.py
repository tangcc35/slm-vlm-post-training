import os
from unittest.mock import MagicMock
import pytest


def test_export_package_exports():
    from slm_post_train.export import export_model
    assert callable(export_model)


def test_export_lora_default(tmp_path):
    from slm_post_train.export import export_model

    mock_model = MagicMock()
    mock_tokenizer = MagicMock()
    out_dir = str(tmp_path / "lora_default")

    export_model(mock_model, mock_tokenizer, out_dir)

    assert os.path.exists(out_dir)
    mock_model.save_pretrained.assert_called_once_with(out_dir)
    mock_tokenizer.save_pretrained.assert_called_once_with(out_dir)


def test_export_lora_explicit(tmp_path):
    from slm_post_train.export import export_model

    mock_model = MagicMock()
    mock_tokenizer = MagicMock()
    out_dir = str(tmp_path / "lora_explicit")

    export_model(mock_model, mock_tokenizer, out_dir, export_format="lora")

    assert os.path.exists(out_dir)
    mock_model.save_pretrained.assert_called_once_with(out_dir)
    mock_tokenizer.save_pretrained.assert_called_once_with(out_dir)


def test_export_merged_16bit(tmp_path):
    from slm_post_train.export import export_model

    mock_model = MagicMock()
    mock_tokenizer = MagicMock()
    out_dir = str(tmp_path / "merged_16bit")

    export_model(mock_model, mock_tokenizer, out_dir, export_format="merged_16bit")

    assert os.path.exists(out_dir)
    mock_model.save_pretrained_merged.assert_called_once_with(
        out_dir, mock_tokenizer, save_method="merged_16bit"
    )
    mock_model.save_pretrained.assert_not_called()
    mock_tokenizer.save_pretrained.assert_not_called()


def test_export_merged_4bit(tmp_path):
    from slm_post_train.export import export_model

    mock_model = MagicMock()
    mock_tokenizer = MagicMock()
    out_dir = str(tmp_path / "merged_4bit")

    export_model(mock_model, mock_tokenizer, out_dir, export_format="merged_4bit")

    assert os.path.exists(out_dir)
    mock_model.save_pretrained_merged.assert_called_once_with(
        out_dir, mock_tokenizer, save_method="merged_4bit"
    )
    mock_model.save_pretrained.assert_not_called()
    mock_tokenizer.save_pretrained.assert_not_called()


def test_export_gguf_default_quantization(tmp_path):
    from slm_post_train.export import export_model

    mock_model = MagicMock()
    mock_tokenizer = MagicMock()
    out_dir = str(tmp_path / "gguf_default")

    export_model(mock_model, mock_tokenizer, out_dir, export_format="gguf")

    assert os.path.exists(out_dir)
    mock_model.save_pretrained_gguf.assert_called_once_with(
        out_dir, mock_tokenizer, quantization_method="q4_k_m"
    )
    mock_model.save_pretrained.assert_not_called()


def test_export_gguf_custom_quantization(tmp_path):
    from slm_post_train.export import export_model

    mock_model = MagicMock()
    mock_tokenizer = MagicMock()
    out_dir = str(tmp_path / "gguf_q8_0")

    export_model(
        mock_model,
        mock_tokenizer,
        out_dir,
        export_format="gguf",
        quantization_method="q8_0",
    )

    assert os.path.exists(out_dir)
    mock_model.save_pretrained_gguf.assert_called_once_with(
        out_dir, mock_tokenizer, quantization_method="q8_0"
    )


def test_export_invalid_format(tmp_path):
    from slm_post_train.export import export_model

    mock_model = MagicMock()
    mock_tokenizer = MagicMock()
    out_dir = str(tmp_path / "invalid")

    with pytest.raises(ValueError, match="Unknown export format: 'invalid_format'"):
        export_model(mock_model, mock_tokenizer, out_dir, export_format="invalid_format")

    mock_model.save_pretrained.assert_not_called()
    mock_model.save_pretrained_merged.assert_not_called()
    mock_model.save_pretrained_gguf.assert_not_called()
