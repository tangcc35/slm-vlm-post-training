import os
from unittest.mock import MagicMock
import pytest


def test_export_package_exports():
    from slm_post_train.export import export_model
    assert callable(export_model)


def test_export_lora_default(tmp_path):
    from slm_post_train.export.exporter import DT_STR, export_model

    mock_model = MagicMock()
    mock_tokenizer = MagicMock()
    out_dir = str(tmp_path / "lora_default")
    saved_dir = os.path.join(out_dir, DT_STR)

    export_model(mock_model, mock_tokenizer, out_dir)

    assert os.path.exists(saved_dir)
    mock_model.save_pretrained.assert_called_once_with(saved_dir)
    mock_tokenizer.save_pretrained.assert_called_once_with(saved_dir)


def test_export_lora_explicit(tmp_path):
    from slm_post_train.export.exporter import DT_STR, export_model

    mock_model = MagicMock()
    mock_tokenizer = MagicMock()
    out_dir = str(tmp_path / "lora_explicit")
    saved_dir = os.path.join(out_dir, DT_STR)

    export_model(mock_model, mock_tokenizer, out_dir, export_format="lora")

    assert os.path.exists(saved_dir)
    mock_model.save_pretrained.assert_called_once_with(saved_dir)
    mock_tokenizer.save_pretrained.assert_called_once_with(saved_dir)


def test_export_merged_16bit(tmp_path):
    from slm_post_train.export.exporter import DT_STR, export_model

    mock_model = MagicMock()
    mock_tokenizer = MagicMock()
    out_dir = str(tmp_path / "merged_16bit")
    saved_dir = os.path.join(out_dir, DT_STR)

    export_model(mock_model, mock_tokenizer, out_dir, export_format="merged_16bit")

    assert os.path.exists(saved_dir)
    mock_model.save_pretrained_merged.assert_called_once_with(
        saved_dir, mock_tokenizer, save_method="merged_16bit"
    )
    mock_model.save_pretrained.assert_not_called()
    mock_tokenizer.save_pretrained.assert_not_called()


def test_export_merged_4bit(tmp_path):
    from slm_post_train.export.exporter import DT_STR, export_model

    mock_model = MagicMock()
    mock_tokenizer = MagicMock()
    out_dir = str(tmp_path / "merged_4bit")
    saved_dir = os.path.join(out_dir, DT_STR)

    export_model(mock_model, mock_tokenizer, out_dir, export_format="merged_4bit")

    assert os.path.exists(saved_dir)
    mock_model.save_pretrained_merged.assert_called_once_with(
        saved_dir, mock_tokenizer, save_method="merged_4bit"
    )
    mock_model.save_pretrained.assert_not_called()
    mock_tokenizer.save_pretrained.assert_not_called()


def test_export_gguf_default_quantization(tmp_path):
    from slm_post_train.export.exporter import DT_STR, export_model

    mock_model = MagicMock()
    mock_tokenizer = MagicMock()
    out_dir = str(tmp_path / "gguf_default")
    saved_dir = os.path.join(out_dir, DT_STR)

    export_model(mock_model, mock_tokenizer, out_dir, export_format="gguf")

    assert os.path.exists(saved_dir)
    mock_model.save_pretrained_gguf.assert_called_once_with(
        saved_dir, mock_tokenizer, quantization_method="q4_k_m"
    )
    mock_model.save_pretrained.assert_not_called()


def test_export_gguf_custom_quantization(tmp_path):
    from slm_post_train.export.exporter import DT_STR, export_model

    mock_model = MagicMock()
    mock_tokenizer = MagicMock()
    out_dir = str(tmp_path / "gguf_q8_0")
    saved_dir = os.path.join(out_dir, DT_STR)

    export_model(
        mock_model,
        mock_tokenizer,
        out_dir,
        export_format="gguf",
        quantization_method="q8_0",
    )

    assert os.path.exists(saved_dir)
    mock_model.save_pretrained_gguf.assert_called_once_with(
        saved_dir, mock_tokenizer, quantization_method="q8_0"
    )


def test_setup_llama_cpp_env(tmp_path):
    import sys
    from slm_post_train.export import setup_llama_cpp_env

    # Non-existent dir returns None
    assert setup_llama_cpp_env(str(tmp_path / "nonexistent")) is None

    # Existing dir is added to sys.path and PYTHONPATH
    fake_llama = tmp_path / "fake_llama_cpp"
    fake_llama.mkdir()
    res = setup_llama_cpp_env(str(fake_llama))
    assert res == str(fake_llama)
    assert str(fake_llama) in sys.path
    assert str(fake_llama) in os.environ["PYTHONPATH"]


def test_export_gguf_copies_artifacts(tmp_path):
    from slm_post_train.export.exporter import DT_STR, export_model

    mock_model = MagicMock()
    mock_tokenizer = MagicMock()
    out_dir = str(tmp_path / "gguf_out")
    saved_dir = os.path.join(out_dir, DT_STR)

    def fake_save_gguf(output_dir, tokenizer, quantization_method):
        gguf_sidecar = f"{output_dir}_gguf"
        os.makedirs(gguf_sidecar, exist_ok=True)
        with open(os.path.join(gguf_sidecar, "model.Q4_K_M.gguf"), "w") as f:
            f.write("dummy gguf")
        with open(os.path.join(gguf_sidecar, "Modelfile"), "w") as f:
            f.write("dummy modelfile")

    mock_model.save_pretrained_gguf.side_effect = fake_save_gguf

    export_model(mock_model, mock_tokenizer, out_dir, export_format="gguf")

    assert os.path.exists(os.path.join(saved_dir, "model.Q4_K_M.gguf"))
    assert os.path.exists(os.path.join(saved_dir, "Modelfile"))


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
