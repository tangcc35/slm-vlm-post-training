import unsloth  # Must precede trl/transformers imports
import logging
import torch
from trl import GRPOConfig, GRPOTrainer
from slm_post_train.models.loader import load_model_and_tokenizer
from slm_post_train.data.grpo_data import prepare_grpo_dataset
from slm_post_train.rewards.registry import get_reward_function

logger = logging.getLogger(__name__)


def run_grpo(config: dict = None):
    """Executes GRPO Reinforcement Learning."""
    if config is None:
        config = {}

    model_cfg = config.get("model") or {}
    lora_cfg = config.get("lora") if config.get("lora") is not None else {}
    data_cfg = config.get("dataset") or {}
    training_cfg = config.get("training") or {}
    output_cfg = config.get("output") or {}

    output_dir = output_cfg.get("output_dir") or "outputs/grpo_model"

    logger.info("Initializing model and tokenizer...")
    model, tokenizer = load_model_and_tokenizer(
        model_cfg=model_cfg,
        lora_cfg=lora_cfg,
        modality=model_cfg.get("modality", "text"),
    )

    logger.info("Preparing GRPO dataset...")
    dataset = prepare_grpo_dataset(
        dataset_path_or_id=data_cfg.get("path", "data/dummy/grpo_sample.jsonl"),
        split=data_cfg.get("split", "train"),
        max_samples=data_cfg.get("max_samples", None),
    )

    reward_names = config.get("rewards") or ["xml_format"]
    reward_funcs = [get_reward_function(name) for name in reward_names]
    logger.info(f"Loaded reward functions: {reward_names}")

    grpo_args = GRPOConfig(
        use_vllm=training_cfg.get("use_vllm", False),
        learning_rate=training_cfg.get("learning_rate", 5e-6),
        adam_beta1=training_cfg.get("adam_beta1", 0.9),
        adam_beta2=training_cfg.get("adam_beta2", 0.99),
        weight_decay=training_cfg.get("weight_decay", 0.1),
        warmup_ratio=training_cfg.get("warmup_ratio", 0.1),
        lr_scheduler_type=training_cfg.get("lr_scheduler_type", "cosine"),
        optim=training_cfg.get("optim", "paged_adamw_8bit"),
        logging_steps=output_cfg.get("logging_steps", 1),
        per_device_train_batch_size=training_cfg.get("batch_size", 1),
        gradient_accumulation_steps=training_cfg.get("gradient_accumulation_steps", 2),
        num_generations=training_cfg.get("num_generations", 2),
        max_prompt_length=training_cfg.get("max_prompt_length", 256),
        max_completion_length=training_cfg.get("max_completion_length", 256),
        max_steps=training_cfg.get("max_steps", 10),
        output_dir=output_dir,
        report_to="none",
    )

    trainer = GRPOTrainer(
        model=model,
        processing_class=tokenizer,
        reward_funcs=reward_funcs,
        args=grpo_args,
        train_dataset=dataset,
    )

    logger.info("Starting GRPO training...")
    stats = trainer.train()
    logger.info("GRPO training completed.")

    if torch.cuda.is_available():
        peak_vram = torch.cuda.max_memory_reserved() / (1024 ** 3)
        logger.info(f"Peak VRAM reserved: {peak_vram:.2f} GB")

    model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)
    logger.info(f"Model policy saved to {output_dir}")
    return stats
