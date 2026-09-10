import pytest
from unittest.mock import MagicMock, patch
import torch


def test_models_package_exports():
    from slm_post_train.models import load_model_and_tokenizer
    assert callable(load_model_and_tokenizer)


@patch("unsloth.FastLanguageModel")
def test_load_text_model_default_args(mock_flm):
    from slm_post_train.models.loader import load_model_and_tokenizer

    fake_model = MagicMock(name="base_model")
    fake_tokenizer = MagicMock(name="tokenizer")
    mock_flm.from_pretrained.return_value = (fake_model, fake_tokenizer)

    model, tokenizer = load_model_and_tokenizer({})

    mock_flm.from_pretrained.assert_called_once_with(
        model_name="unsloth/gemma-2-2b-it",
        max_seq_length=2048,
        load_in_4bit=True,
        dtype=None,
    )
    mock_flm.get_peft_model.assert_not_called()
    assert model == fake_model
    assert tokenizer == fake_tokenizer


@patch("unsloth.FastLanguageModel")
def test_load_text_model_with_default_lora(mock_flm):
    from slm_post_train.models.loader import load_model_and_tokenizer

    fake_base_model = MagicMock(name="base_model")
    fake_peft_model = MagicMock(name="peft_model")
    fake_tokenizer = MagicMock(name="tokenizer")
    mock_flm.from_pretrained.return_value = (fake_base_model, fake_tokenizer)
    mock_flm.get_peft_model.return_value = fake_peft_model

    model_cfg = {
        "name_or_path": "unsloth/gemma-2-2b-it",
        "max_seq_length": 1024,
        "load_in_4bit": True,
        "dtype": "bfloat16",
    }
    lora_cfg = {"r": 16}

    model, tokenizer = load_model_and_tokenizer(model_cfg=model_cfg, lora_cfg=lora_cfg, modality="text")

    mock_flm.from_pretrained.assert_called_once_with(
        model_name="unsloth/gemma-2-2b-it",
        max_seq_length=1024,
        load_in_4bit=True,
        dtype=torch.bfloat16,
    )
    mock_flm.get_peft_model.assert_called_once_with(
        fake_base_model,
        r=16,
        lora_alpha=32,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
        lora_dropout=0.0,
        bias="none",
        random_state=3407,
        use_gradient_checkpointing="unsloth",
    )
    assert model == fake_peft_model
    assert tokenizer == fake_tokenizer


@patch("unsloth.FastLanguageModel")
def test_load_text_model_with_custom_lora_cfg(mock_flm):
    from slm_post_train.models.loader import load_model_and_tokenizer

    fake_base_model = MagicMock(name="base_model")
    fake_peft_model = MagicMock(name="peft_model")
    fake_tokenizer = MagicMock(name="tokenizer")
    mock_flm.from_pretrained.return_value = (fake_base_model, fake_tokenizer)
    mock_flm.get_peft_model.return_value = fake_peft_model

    lora_cfg = {
        "r": 32,
        "lora_alpha": 64,
        "target_modules": ["q_proj", "v_proj"],
        "lora_dropout": 0.05,
        "bias": "all",
        "random_state": 42,
        "use_gradient_checkpointing": True,
    }

    model, tokenizer = load_model_and_tokenizer(
        model_cfg={"name_or_path": "custom/model"},
        lora_cfg=lora_cfg,
    )

    mock_flm.get_peft_model.assert_called_once_with(
        fake_base_model,
        r=32,
        lora_alpha=64,
        target_modules=["q_proj", "v_proj"],
        lora_dropout=0.05,
        bias="all",
        random_state=42,
        use_gradient_checkpointing=True,
    )
    assert model == fake_peft_model


@patch("unsloth.FastVisionModel")
@patch("unsloth.FastLanguageModel")
def test_load_vision_model_with_lora(mock_flm, mock_fvm):
    from slm_post_train.models.loader import load_model_and_tokenizer

    fake_v_model = MagicMock(name="v_base_model")
    fake_v_peft_model = MagicMock(name="v_peft_model")
    fake_tokenizer = MagicMock(name="tokenizer")
    mock_fvm.from_pretrained.return_value = (fake_v_model, fake_tokenizer)
    mock_fvm.get_peft_model.return_value = fake_v_peft_model

    model_cfg = {
        "name_or_path": "unsloth/Qwen2-VL-7B-Instruct",
        "max_seq_length": 4096,
        "load_in_4bit": False,
        "dtype": "float16",
    }
    lora_cfg = {"r": 8}

    model, tokenizer = load_model_and_tokenizer(
        model_cfg=model_cfg,
        lora_cfg=lora_cfg,
        modality="vision",
    )

    mock_fvm.from_pretrained.assert_called_once_with(
        model_name="unsloth/Qwen2-VL-7B-Instruct",
        max_seq_length=4096,
        load_in_4bit=False,
        dtype=torch.float16,
    )
    mock_fvm.get_peft_model.assert_called_once_with(
        fake_v_model,
        r=8,
        lora_alpha=16,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
        lora_dropout=0.0,
        bias="none",
        random_state=3407,
        use_gradient_checkpointing="unsloth",
    )
    mock_flm.from_pretrained.assert_not_called()
    mock_flm.get_peft_model.assert_not_called()
    assert model == fake_v_peft_model
    assert tokenizer == fake_tokenizer


@patch("unsloth.FastVisionModel")
def test_load_vision_model_no_lora(mock_fvm):
    from slm_post_train.models.loader import load_model_and_tokenizer

    fake_v_model = MagicMock(name="v_base_model")
    fake_tokenizer = MagicMock(name="tokenizer")
    mock_fvm.from_pretrained.return_value = (fake_v_model, fake_tokenizer)

    model, tokenizer = load_model_and_tokenizer(
        model_cfg={"name_or_path": "unsloth/Qwen2-VL-7B-Instruct"},
        lora_cfg=None,
        modality="vision",
    )

    mock_fvm.from_pretrained.assert_called_once()
    mock_fvm.get_peft_model.assert_not_called()
    assert model == fake_v_model
    assert tokenizer == fake_tokenizer


@pytest.mark.parametrize(
    "input_dtype,expected_dtype",
    [
        ("bfloat16", torch.bfloat16),
        ("float16", torch.float16),
        ("float32", torch.float32),
        (torch.bfloat16, torch.bfloat16),
        (None, None),
    ],
)
@patch("unsloth.FastLanguageModel")
def test_dtype_conversions(mock_flm, input_dtype, expected_dtype):
    from slm_post_train.models.loader import load_model_and_tokenizer

    mock_flm.from_pretrained.return_value = (MagicMock(), MagicMock())

    load_model_and_tokenizer(model_cfg={"dtype": input_dtype})

    call_kwargs = mock_flm.from_pretrained.call_args.kwargs
    assert call_kwargs["dtype"] == expected_dtype


@patch("unsloth.FastLanguageModel")
def test_load_model_none_cfg(mock_flm):
    from slm_post_train.models.loader import load_model_and_tokenizer

    mock_flm.from_pretrained.return_value = (MagicMock(), MagicMock())

    model, tokenizer = load_model_and_tokenizer(model_cfg=None)

    mock_flm.from_pretrained.assert_called_once_with(
        model_name="unsloth/gemma-2-2b-it",
        max_seq_length=2048,
        load_in_4bit=True,
        dtype=None,
    )
