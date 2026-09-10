import unsloth  # Must precede trl/transformers imports
import logging
import torch
from trl import SFTTrainer, SFTConfig
from unsloth.chat_templates import train_on_responses_only
from slm_post_train.models.loader import load_model_and_tokenizer
from slm_post_train.data.sft_data import prepare_sft_dataset

logger = logging.getLogger(__name__)


def run_sft(config: dict = None):
    """Executes SFT instruction training."""
    if config is None:
        config = {}

    model_cfg = config.get("model") or {}
    lora_cfg = config.get("lora") if config.get("lora") is not None else {}
    data_cfg = config.get("dataset") or {}
    training_cfg = config.get("training") or {}
    output_cfg = config.get("output") or {}

    output_dir = output_cfg.get("output_dir") or "outputs/sft_model"

    logger.info("Initializing model and tokenizer...")
    model, tokenizer = load_model_and_tokenizer(
        model_cfg=model_cfg,
        lora_cfg=lora_cfg,
        modality=model_cfg.get("modality", "text"),
    )

    logger.info("Preparing SFT dataset...")
    dataset = prepare_sft_dataset(
        dataset_path_or_id=data_cfg.get("path", "data/dummy/sft_sample.jsonl"),
        tokenizer=tokenizer,
        chat_template=data_cfg.get("chat_template", "chatml"),
        split=data_cfg.get("split", "train"),
        max_samples=data_cfg.get("max_samples", None),
    )

    packing = training_cfg.get("packing", False)
    train_on_responses = training_cfg.get("train_on_responses_only", True)

    if packing and train_on_responses:
        logger.warning(
            "train_on_responses_only is incompatible with packing=True. "
            "Disabling train_on_responses_only to avoid conflicts."
        )
        train_on_responses = False

    sft_args = SFTConfig(
        dataset_text_field="text",
        max_seq_length=model_cfg.get("max_seq_length", 2048),
        dataset_num_proc=data_cfg.get("dataset_num_proc", 2),
        packing=packing,
        per_device_train_batch_size=training_cfg.get("batch_size", 1),
        gradient_accumulation_steps=training_cfg.get("gradient_accumulation_steps", 2),
        warmup_steps=training_cfg.get("warmup_steps", 5),
        max_steps=training_cfg.get("max_steps", 60),
        learning_rate=training_cfg.get("learning_rate", 2e-4),
        logging_steps=output_cfg.get("logging_steps", 1),
        optim=training_cfg.get("optim", "adamw_8bit"),
        weight_decay=training_cfg.get("weight_decay", 0.01),
        lr_scheduler_type=training_cfg.get("lr_scheduler_type", "linear"),
        seed=training_cfg.get("seed", 3407),
        output_dir=output_dir,
        report_to="none",
    )

    trainer = SFTTrainer(
        model=model,
        tokenizer=tokenizer,
        train_dataset=dataset,
        args=sft_args,
    )

    if train_on_responses:
        trainer = train_on_responses_only(trainer)

    logger.info("Starting SFT training...")
    stats = trainer.train()
    try:
        runtime = float(stats.metrics.get("train_runtime", 0))
        logger.info(f"Training completed. Runtime: {runtime:.2f}s")
    except (TypeError, ValueError, AttributeError):
        logger.info("Training completed.")

    if torch.cuda.is_available():
        peak_vram = torch.cuda.max_memory_reserved() / (1024 ** 3)
        logger.info(f"Peak VRAM reserved: {peak_vram:.2f} GB")

    model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)
    logger.info(f"LoRA adapter saved to {output_dir}")
    return stats
