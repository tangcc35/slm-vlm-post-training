from unittest.mock import MagicMock, patch
import pytest
import torch


# ============================================================================
# Package Export Tests
# ============================================================================

def test_trainers_package_exports():
    from slm_post_train.trainers import run_sft, run_grpo
    assert callable(run_sft)
    assert callable(run_grpo)


# ============================================================================
# SFT Runner Tests
# ============================================================================

@patch("slm_post_train.trainers.sft_runner.train_on_responses_only")
@patch("slm_post_train.trainers.sft_runner.SFTTrainer")
@patch("slm_post_train.trainers.sft_runner.SFTConfig")
@patch("slm_post_train.trainers.sft_runner.prepare_sft_dataset")
@patch("slm_post_train.trainers.sft_runner.load_model_and_tokenizer")
def test_run_sft_default_config(
    mock_load_model,
    mock_prep_dataset,
    mock_sft_config_cls,
    mock_sft_trainer_cls,
    mock_train_on_responses,
):
    from slm_post_train.trainers.sft_runner import run_sft

    mock_model = MagicMock(name="model")
    mock_tokenizer = MagicMock(name="tokenizer")
    mock_load_model.return_value = (mock_model, mock_tokenizer)

    mock_dataset = MagicMock(name="sft_dataset")
    mock_prep_dataset.return_value = mock_dataset

    mock_sft_config = MagicMock(name="sft_config")
    mock_sft_config_cls.return_value = mock_sft_config

    mock_raw_trainer = MagicMock(name="raw_trainer")
    mock_sft_trainer_cls.return_value = mock_raw_trainer

    mock_masked_trainer = MagicMock(name="masked_trainer")
    mock_train_stats = MagicMock(name="train_stats")
    mock_train_stats.metrics = {"train_runtime": 42.5}
    mock_masked_trainer.train.return_value = mock_train_stats
    mock_train_on_responses.return_value = mock_masked_trainer

    result = run_sft({})

    # 1. Verify model loader called with defaults
    mock_load_model.assert_called_once_with(
        model_cfg={},
        lora_cfg={},
        modality="text",
    )

    # 2. Verify dataset preparation called with defaults
    mock_prep_dataset.assert_called_once_with(
        dataset_path_or_id="data/dummy/sft_sample.jsonl",
        tokenizer=mock_tokenizer,
        chat_template="chatml",
        split="train",
        max_samples=None,
    )

    # 3. Verify SFTConfig instantiations with defaults
    mock_sft_config_cls.assert_called_once_with(
        dataset_text_field="text",
        max_seq_length=2048,
        dataset_num_proc=2,
        packing=False,
        per_device_train_batch_size=1,
        gradient_accumulation_steps=2,
        warmup_steps=5,
        max_steps=60,
        learning_rate=2e-4,
        logging_steps=1,
        optim="adamw_8bit",
        weight_decay=0.01,
        lr_scheduler_type="linear",
        seed=3407,
        output_dir="outputs/sft_model",
        report_to="none",
    )

    # 4. Verify SFTTrainer creation
    mock_sft_trainer_cls.assert_called_once_with(
        model=mock_model,
        tokenizer=mock_tokenizer,
        train_dataset=mock_dataset,
        args=mock_sft_config,
    )

    # 5. Verify responses-only masking applied
    mock_train_on_responses.assert_called_once_with(
        mock_raw_trainer,
        instruction_part="<|im_start|>user\n",
        response_part="<|im_start|>assistant\n",
    )

    # 6. Verify training initiated and returned
    mock_masked_trainer.train.assert_called_once()
    assert result == mock_train_stats

    # 7. Verify model & tokenizer saving
    mock_model.save_pretrained.assert_called_once_with("outputs/sft_model")
    mock_tokenizer.save_pretrained.assert_called_once_with("outputs/sft_model")


@patch("slm_post_train.trainers.sft_runner.train_on_responses_only")
@patch("slm_post_train.trainers.sft_runner.SFTTrainer")
@patch("slm_post_train.trainers.sft_runner.SFTConfig")
@patch("slm_post_train.trainers.sft_runner.prepare_sft_dataset")
@patch("slm_post_train.trainers.sft_runner.load_model_and_tokenizer")
def test_run_sft_explicit_delimiters(
    mock_load_model,
    mock_prep_dataset,
    mock_sft_config_cls,
    mock_sft_trainer_cls,
    mock_train_on_responses,
):
    from slm_post_train.trainers.sft_runner import run_sft

    mock_load_model.return_value = (MagicMock(), MagicMock())
    mock_raw_trainer = MagicMock()
    mock_sft_trainer_cls.return_value = mock_raw_trainer
    mock_masked_trainer = MagicMock()
    mock_train_on_responses.return_value = mock_masked_trainer

    cfg = {
        "training": {
            "train_on_responses_only": True,
            "instruction_part": "Human: ",
            "response_part": "Assistant: ",
        }
    }
    run_sft(cfg)

    mock_train_on_responses.assert_called_once_with(
        mock_raw_trainer,
        instruction_part="Human: ",
        response_part="Assistant: ",
    )


@patch("slm_post_train.trainers.sft_runner.train_on_responses_only")
@patch("slm_post_train.trainers.sft_runner.SFTTrainer")
@patch("slm_post_train.trainers.sft_runner.SFTConfig")
@patch("slm_post_train.trainers.sft_runner.prepare_sft_dataset")
@patch("slm_post_train.trainers.sft_runner.load_model_and_tokenizer")
def test_run_sft_llama_chat_template_delimiters(
    mock_load_model,
    mock_prep_dataset,
    mock_sft_config_cls,
    mock_sft_trainer_cls,
    mock_train_on_responses,
):
    from slm_post_train.trainers.sft_runner import run_sft

    mock_load_model.return_value = (MagicMock(), MagicMock())
    mock_raw_trainer = MagicMock()
    mock_sft_trainer_cls.return_value = mock_raw_trainer
    mock_masked_trainer = MagicMock()
    mock_train_on_responses.return_value = mock_masked_trainer

    cfg = {
        "dataset": {
            "chat_template": "llama-3",
        },
        "training": {
            "train_on_responses_only": True,
        }
    }
    run_sft(cfg)

    mock_train_on_responses.assert_called_once_with(
        mock_raw_trainer,
        instruction_part="<|start_header_id|>user<|end_header_id|>\n\n",
        response_part="<|start_header_id|>assistant<|end_header_id|>\n\n",
    )


@patch("slm_post_train.trainers.sft_runner.train_on_responses_only")
@patch("slm_post_train.trainers.sft_runner.SFTTrainer")
@patch("slm_post_train.trainers.sft_runner.SFTConfig")
@patch("slm_post_train.trainers.sft_runner.prepare_sft_dataset")
@patch("slm_post_train.trainers.sft_runner.load_model_and_tokenizer")
def test_run_sft_tokenizer_with_existing_unsloth_parts(
    mock_load_model,
    mock_prep_dataset,
    mock_sft_config_cls,
    mock_sft_trainer_cls,
    mock_train_on_responses,
):
    from slm_post_train.trainers.sft_runner import run_sft

    mock_tok = MagicMock()
    mock_tok._unsloth_input_part = "<|user|>"
    mock_tok._unsloth_output_part = "<|bot|>"
    del mock_tok.image_processor
    del mock_tok.tokenizer

    mock_load_model.return_value = (MagicMock(), mock_tok)
    mock_raw_trainer = MagicMock()
    mock_raw_trainer.tokenizer = mock_tok
    del mock_raw_trainer.processing_class
    mock_sft_trainer_cls.return_value = mock_raw_trainer
    mock_masked_trainer = MagicMock()
    mock_train_on_responses.return_value = mock_masked_trainer

    run_sft({})

    mock_train_on_responses.assert_called_once_with(mock_raw_trainer)


@patch("slm_post_train.trainers.sft_runner.train_on_responses_only")
@patch("slm_post_train.trainers.sft_runner.SFTTrainer")
@patch("slm_post_train.trainers.sft_runner.SFTConfig")
@patch("slm_post_train.trainers.sft_runner.prepare_sft_dataset")
@patch("slm_post_train.trainers.sft_runner.load_model_and_tokenizer")
def test_run_sft_unknown_template_raises_error(
    mock_load_model,
    mock_prep_dataset,
    mock_sft_config_cls,
    mock_sft_trainer_cls,
    mock_train_on_responses,
):
    from slm_post_train.trainers.sft_runner import run_sft

    mock_load_model.return_value = (MagicMock(), MagicMock())
    mock_raw_trainer = MagicMock()
    mock_sft_trainer_cls.return_value = mock_raw_trainer

    cfg = {
        "dataset": {
            "chat_template": "unknown_custom_template",
        },
        "training": {
            "train_on_responses_only": True,
        },
    }

    with pytest.raises(ValueError, match="Could not automatically determine instruction/response delimiters"):
        run_sft(cfg)


@patch("slm_post_train.trainers.sft_runner.train_on_responses_only")
@patch("slm_post_train.trainers.sft_runner.SFTTrainer")
@patch("slm_post_train.trainers.sft_runner.SFTConfig")
@patch("slm_post_train.trainers.sft_runner.prepare_sft_dataset")
@patch("slm_post_train.trainers.sft_runner.load_model_and_tokenizer")
def test_run_sft_unknown_template_with_explicit_delimiters(
    mock_load_model,
    mock_prep_dataset,
    mock_sft_config_cls,
    mock_sft_trainer_cls,
    mock_train_on_responses,
):
    from slm_post_train.trainers.sft_runner import run_sft

    mock_load_model.return_value = (MagicMock(), MagicMock())
    mock_raw_trainer = MagicMock()
    mock_sft_trainer_cls.return_value = mock_raw_trainer
    mock_masked_trainer = MagicMock()
    mock_train_on_responses.return_value = mock_masked_trainer

    cfg = {
        "dataset": {
            "chat_template": "unknown_custom_template",
        },
        "training": {
            "train_on_responses_only": True,
            "instruction_part": "USER>>",
            "response_part": "ASSISTANT>>",
        },
    }

    run_sft(cfg)
    mock_train_on_responses.assert_called_once_with(
        mock_raw_trainer,
        instruction_part="USER>>",
        response_part="ASSISTANT>>",
    )


@patch("slm_post_train.trainers.sft_runner.train_on_responses_only")
@patch("slm_post_train.trainers.sft_runner.SFTTrainer")
@patch("slm_post_train.trainers.sft_runner.SFTConfig")
@patch("slm_post_train.trainers.sft_runner.prepare_sft_dataset")
@patch("slm_post_train.trainers.sft_runner.load_model_and_tokenizer")
def test_run_sft_custom_config(
    mock_load_model,
    mock_prep_dataset,
    mock_sft_config_cls,
    mock_sft_trainer_cls,
    mock_train_on_responses,
):
    from slm_post_train.trainers.sft_runner import run_sft

    mock_model = MagicMock(name="model")
    mock_tokenizer = MagicMock(name="tokenizer")
    mock_load_model.return_value = (mock_model, mock_tokenizer)
    mock_raw_trainer = MagicMock(name="raw_trainer")
    mock_sft_trainer_cls.return_value = mock_raw_trainer

    custom_cfg = {
        "model": {
            "name_or_path": "unsloth/Qwen2-VL-7B-Instruct",
            "max_seq_length": 1024,
            "modality": "vision",
        },
        "lora": {
            "r": 32,
            "lora_alpha": 64,
        },
        "dataset": {
            "path": "custom/data.jsonl",
            "chat_template": "qwen-2.5",
            "split": "train[:100]",
            "max_samples": 50,
            "dataset_num_proc": 4,
        },
        "training": {
            "batch_size": 4,
            "gradient_accumulation_steps": 8,
            "warmup_steps": 10,
            "max_steps": 120,
            "learning_rate": 1e-4,
            "optim": "adamw_torch",
            "weight_decay": 0.05,
            "lr_scheduler_type": "cosine",
            "seed": 42,
            "packing": True,
            "train_on_responses_only": False,
        },
        "output": {
            "output_dir": "custom_outputs/sft_experiment",
            "logging_steps": 10,
        },
    }

    result = run_sft(custom_cfg)

    # 1. Model loader with custom parameters
    mock_load_model.assert_called_once_with(
        model_cfg=custom_cfg["model"],
        lora_cfg=custom_cfg["lora"],
        modality="vision",
    )

    # 2. Dataset preparation with custom parameters
    mock_prep_dataset.assert_called_once_with(
        dataset_path_or_id="custom/data.jsonl",
        tokenizer=mock_tokenizer,
        chat_template="qwen-2.5",
        split="train[:100]",
        max_samples=50,
    )

    # 3. SFTConfig with custom parameters
    mock_sft_config_cls.assert_called_once_with(
        dataset_text_field="text",
        max_seq_length=1024,
        dataset_num_proc=4,
        packing=True,
        per_device_train_batch_size=4,
        gradient_accumulation_steps=8,
        warmup_steps=10,
        max_steps=120,
        learning_rate=1e-4,
        logging_steps=10,
        optim="adamw_torch",
        weight_decay=0.05,
        lr_scheduler_type="cosine",
        seed=42,
        output_dir="custom_outputs/sft_experiment",
        report_to="none",
    )

    # 4. When train_on_responses_only is False, wrapper should not be called
    mock_train_on_responses.assert_not_called()
    mock_raw_trainer.train.assert_called_once()

    # 5. Model & tokenizer saved to custom output dir
    mock_model.save_pretrained.assert_called_once_with("custom_outputs/sft_experiment")
    mock_tokenizer.save_pretrained.assert_called_once_with("custom_outputs/sft_experiment")


@patch("slm_post_train.trainers.sft_runner.torch.cuda")
@patch("slm_post_train.trainers.sft_runner.train_on_responses_only")
@patch("slm_post_train.trainers.sft_runner.SFTTrainer")
@patch("slm_post_train.trainers.sft_runner.SFTConfig")
@patch("slm_post_train.trainers.sft_runner.prepare_sft_dataset")
@patch("slm_post_train.trainers.sft_runner.load_model_and_tokenizer")
def test_run_sft_cuda_vram_logging(
    mock_load_model,
    mock_prep_dataset,
    mock_sft_config_cls,
    mock_sft_trainer_cls,
    mock_train_on_responses,
    mock_cuda,
):
    from slm_post_train.trainers.sft_runner import run_sft

    mock_load_model.return_value = (MagicMock(), MagicMock())
    mock_cuda.is_available.return_value = True
    mock_cuda.max_memory_reserved.return_value = 8 * (1024 ** 3)  # 8 GB

    run_sft({})

    mock_cuda.is_available.assert_called_once()
    mock_cuda.max_memory_reserved.assert_called_once()


@patch("slm_post_train.trainers.sft_runner.train_on_responses_only")
@patch("slm_post_train.trainers.sft_runner.SFTTrainer")
@patch("slm_post_train.trainers.sft_runner.SFTConfig")
@patch("slm_post_train.trainers.sft_runner.prepare_sft_dataset")
@patch("slm_post_train.trainers.sft_runner.load_model_and_tokenizer")
def test_run_sft_none_config(
    mock_load_model,
    mock_prep_dataset,
    mock_sft_config_cls,
    mock_sft_trainer_cls,
    mock_train_on_responses,
):
    from slm_post_train.trainers.sft_runner import run_sft

    mock_load_model.return_value = (MagicMock(), MagicMock())
    # Should not raise AttributeError when passed None
    run_sft(None)
    mock_load_model.assert_called_once_with(
        model_cfg={},
        lora_cfg={},
        modality="text",
    )


@patch("slm_post_train.trainers.sft_runner.train_on_responses_only")
@patch("slm_post_train.trainers.sft_runner.SFTTrainer")
@patch("slm_post_train.trainers.sft_runner.SFTConfig")
@patch("slm_post_train.trainers.sft_runner.prepare_sft_dataset")
@patch("slm_post_train.trainers.sft_runner.load_model_and_tokenizer")
def test_run_sft_packing_disables_train_on_responses_only(
    mock_load_model,
    mock_prep_dataset,
    mock_sft_config_cls,
    mock_sft_trainer_cls,
    mock_train_on_responses,
):
    from slm_post_train.trainers.sft_runner import run_sft

    mock_load_model.return_value = (MagicMock(), MagicMock())
    mock_raw_trainer = MagicMock(name="raw_trainer")
    mock_sft_trainer_cls.return_value = mock_raw_trainer

    cfg = {
        "training": {
            "packing": True,
            "train_on_responses_only": True,
        }
    }
    run_sft(cfg)

    # When packing=True and train_on_responses_only=True, train_on_responses_only must be disabled to avoid conflict
    mock_train_on_responses.assert_not_called()
    mock_raw_trainer.train.assert_called_once()


# ============================================================================
# GRPO Runner Tests
# ============================================================================

@patch("slm_post_train.trainers.grpo_runner.GRPOTrainer")
@patch("slm_post_train.trainers.grpo_runner.GRPOConfig")
@patch("slm_post_train.trainers.grpo_runner.get_reward_function")
@patch("slm_post_train.trainers.grpo_runner.prepare_grpo_dataset")
@patch("slm_post_train.trainers.grpo_runner.load_model_and_tokenizer")
def test_run_grpo_default_config(
    mock_load_model,
    mock_prep_dataset,
    mock_get_reward,
    mock_grpo_config_cls,
    mock_grpo_trainer_cls,
):
    from slm_post_train.trainers.grpo_runner import run_grpo

    mock_model = MagicMock(name="model")
    mock_tokenizer = MagicMock(name="tokenizer")
    mock_load_model.return_value = (mock_model, mock_tokenizer)

    mock_dataset = MagicMock(name="grpo_dataset")
    mock_prep_dataset.return_value = mock_dataset

    mock_reward_fn = MagicMock(name="xml_format_fn")
    mock_get_reward.return_value = mock_reward_fn

    mock_grpo_config = MagicMock(name="grpo_config")
    mock_grpo_config_cls.return_value = mock_grpo_config

    mock_trainer = MagicMock(name="grpo_trainer")
    mock_train_stats = MagicMock(name="train_stats")
    mock_trainer.train.return_value = mock_train_stats
    mock_grpo_trainer_cls.return_value = mock_trainer

    result = run_grpo({})

    # 1. Model loader called with defaults
    mock_load_model.assert_called_once_with(
        model_cfg={},
        lora_cfg={},
        modality="text",
    )

    # 2. Dataset preparation called with defaults
    mock_prep_dataset.assert_called_once_with(
        dataset_path_or_id="data/dummy/grpo_sample.jsonl",
        split="train",
        max_samples=None,
    )

    # 3. Default reward function resolved
    mock_get_reward.assert_called_once_with("xml_format")

    # 4. GRPOConfig created with defaults
    mock_grpo_config_cls.assert_called_once_with(
        use_vllm=False,
        learning_rate=5e-6,
        adam_beta1=0.9,
        adam_beta2=0.99,
        weight_decay=0.1,
        warmup_ratio=0.1,
        lr_scheduler_type="cosine",
        optim="paged_adamw_8bit",
        logging_steps=1,
        per_device_train_batch_size=1,
        gradient_accumulation_steps=2,
        num_generations=2,
        max_prompt_length=256,
        max_completion_length=256,
        max_steps=10,
        output_dir="outputs/grpo_model",
        report_to="none",
    )

    # 5. GRPOTrainer instantiation
    mock_grpo_trainer_cls.assert_called_once_with(
        model=mock_model,
        processing_class=mock_tokenizer,
        reward_funcs=[mock_reward_fn],
        args=mock_grpo_config,
        train_dataset=mock_dataset,
    )

    # 6. Training initiated and returned
    mock_trainer.train.assert_called_once()
    assert result == mock_train_stats

    # 7. Model and tokenizer saved
    mock_model.save_pretrained.assert_called_once_with("outputs/grpo_model")
    mock_tokenizer.save_pretrained.assert_called_once_with("outputs/grpo_model")


@patch("slm_post_train.trainers.grpo_runner.GRPOTrainer")
@patch("slm_post_train.trainers.grpo_runner.GRPOConfig")
@patch("slm_post_train.trainers.grpo_runner.get_reward_function")
@patch("slm_post_train.trainers.grpo_runner.prepare_grpo_dataset")
@patch("slm_post_train.trainers.grpo_runner.load_model_and_tokenizer")
def test_run_grpo_multiple_rewards(
    mock_load_model,
    mock_prep_dataset,
    mock_get_reward,
    mock_grpo_config_cls,
    mock_grpo_trainer_cls,
):
    from slm_post_train.trainers.grpo_runner import run_grpo

    mock_load_model.return_value = (MagicMock(), MagicMock())
    fn_xml = MagicMock(name="fn_xml")
    fn_em = MagicMock(name="fn_em")
    fn_code = MagicMock(name="fn_code")
    mock_get_reward.side_effect = [fn_xml, fn_em, fn_code]

    cfg = {"rewards": ["xml_format", "exact_match", "code_execution"]}
    run_grpo(cfg)

    assert mock_get_reward.call_count == 3
    mock_get_reward.assert_any_call("xml_format")
    mock_get_reward.assert_any_call("exact_match")
    mock_get_reward.assert_any_call("code_execution")

    call_kwargs = mock_grpo_trainer_cls.call_args.kwargs
    assert call_kwargs["reward_funcs"] == [fn_xml, fn_em, fn_code]


@patch("slm_post_train.trainers.grpo_runner.GRPOTrainer")
@patch("slm_post_train.trainers.grpo_runner.GRPOConfig")
@patch("slm_post_train.trainers.grpo_runner.get_reward_function")
@patch("slm_post_train.trainers.grpo_runner.prepare_grpo_dataset")
@patch("slm_post_train.trainers.grpo_runner.load_model_and_tokenizer")
def test_run_grpo_custom_config(
    mock_load_model,
    mock_prep_dataset,
    mock_get_reward,
    mock_grpo_config_cls,
    mock_grpo_trainer_cls,
):
    from slm_post_train.trainers.grpo_runner import run_grpo

    mock_model = MagicMock()
    mock_tokenizer = MagicMock()
    mock_load_model.return_value = (mock_model, mock_tokenizer)

    custom_cfg = {
        "model": {
            "name_or_path": "unsloth/gemma-2-2b-it",
            "modality": "text",
        },
        "lora": {
            "r": 16,
            "lora_alpha": 32,
        },
        "dataset": {
            "path": "custom/grpo_data.jsonl",
            "split": "train[:100]",
            "max_samples": 25,
        },
        "rewards": ["exact_match"],
        "training": {
            "use_vllm": True,
            "learning_rate": 1e-5,
            "adam_beta1": 0.92,
            "adam_beta2": 0.98,
            "weight_decay": 0.05,
            "warmup_ratio": 0.05,
            "lr_scheduler_type": "linear",
            "optim": "adamw_8bit",
            "batch_size": 2,
            "gradient_accumulation_steps": 4,
            "num_generations": 4,
            "max_prompt_length": 512,
            "max_completion_length": 512,
            "max_steps": 50,
        },
        "output": {
            "output_dir": "custom_outputs/grpo_experiment",
            "logging_steps": 5,
        },
    }

    run_grpo(custom_cfg)

    # 1. Loader
    mock_load_model.assert_called_once_with(
        model_cfg=custom_cfg["model"],
        lora_cfg=custom_cfg["lora"],
        modality="text",
    )

    # 2. Dataset
    mock_prep_dataset.assert_called_once_with(
        dataset_path_or_id="custom/grpo_data.jsonl",
        split="train[:100]",
        max_samples=25,
    )

    # 3. GRPOConfig custom kwargs
    mock_grpo_config_cls.assert_called_once_with(
        use_vllm=True,
        learning_rate=1e-5,
        adam_beta1=0.92,
        adam_beta2=0.98,
        weight_decay=0.05,
        warmup_ratio=0.05,
        lr_scheduler_type="linear",
        optim="adamw_8bit",
        logging_steps=5,
        per_device_train_batch_size=2,
        gradient_accumulation_steps=4,
        num_generations=4,
        max_prompt_length=512,
        max_completion_length=512,
        max_steps=50,
        output_dir="custom_outputs/grpo_experiment",
        report_to="none",
    )

    # 4. Save
    mock_model.save_pretrained.assert_called_once_with("custom_outputs/grpo_experiment")
    mock_tokenizer.save_pretrained.assert_called_once_with("custom_outputs/grpo_experiment")


@patch("slm_post_train.trainers.grpo_runner.torch.cuda")
@patch("slm_post_train.trainers.grpo_runner.GRPOTrainer")
@patch("slm_post_train.trainers.grpo_runner.GRPOConfig")
@patch("slm_post_train.trainers.grpo_runner.get_reward_function")
@patch("slm_post_train.trainers.grpo_runner.prepare_grpo_dataset")
@patch("slm_post_train.trainers.grpo_runner.load_model_and_tokenizer")
def test_run_grpo_cuda_vram_logging(
    mock_load_model,
    mock_prep_dataset,
    mock_get_reward,
    mock_grpo_config_cls,
    mock_grpo_trainer_cls,
    mock_cuda,
):
    from slm_post_train.trainers.grpo_runner import run_grpo

    mock_load_model.return_value = (MagicMock(), MagicMock())
    mock_cuda.is_available.return_value = True
    mock_cuda.max_memory_reserved.return_value = 12 * (1024 ** 3)  # 12 GB

    run_grpo({})

    mock_cuda.is_available.assert_called_once()
    mock_cuda.max_memory_reserved.assert_called_once()


@patch("slm_post_train.trainers.grpo_runner.GRPOTrainer")
@patch("slm_post_train.trainers.grpo_runner.GRPOConfig")
@patch("slm_post_train.trainers.grpo_runner.get_reward_function")
@patch("slm_post_train.trainers.grpo_runner.prepare_grpo_dataset")
@patch("slm_post_train.trainers.grpo_runner.load_model_and_tokenizer")
def test_run_grpo_none_config(
    mock_load_model,
    mock_prep_dataset,
    mock_get_reward,
    mock_grpo_config_cls,
    mock_grpo_trainer_cls,
):
    from slm_post_train.trainers.grpo_runner import run_grpo

    mock_load_model.return_value = (MagicMock(), MagicMock())
    # Should not raise AttributeError when passed None
    run_grpo(None)
    mock_load_model.assert_called_once_with(
        model_cfg={},
        lora_cfg={},
        modality="text",
    )
