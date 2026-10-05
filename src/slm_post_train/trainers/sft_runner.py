import os
import unsloth  # Must precede trl/transformers imports
import logging
import torch
import wandb
from datetime import datetime
from trl import SFTTrainer, SFTConfig
from unsloth.chat_templates import train_on_responses_only
from slm_post_train.models.loader import load_model_and_tokenizer
from slm_post_train.data.sft_data import prepare_sft_dataset


logger = logging.getLogger(__name__)


DT_STR = datetime.now().strftime("%Y%m%d-%H%M%S")
CHAT_TEMPLATE_DELIMITERS = {
    "chatml": ("<|im_start|>user\n", "<|im_start|>assistant\n"),
    "qwen-2.5": ("<|im_start|>user\n", "<|im_start|>assistant\n"),
    "qwen-25": ("<|im_start|>user\n", "<|im_start|>assistant\n"),
    "qwen2.5": ("<|im_start|>user\n", "<|im_start|>assistant\n"),
    "qwen-3": ("<|im_start|>user\n", "<|im_start|>assistant\n"),
    "qwen3": ("<|im_start|>user\n", "<|im_start|>assistant\n"),
    "qwen3-instruct": ("<|im_start|>user\n", "<|im_start|>assistant\n"),
    "llama-3": ("<|start_header_id|>user<|end_header_id|>\n\n", "<|start_header_id|>assistant<|end_header_id|>\n\n"),
    "llama3": ("<|start_header_id|>user<|end_header_id|>\n\n", "<|start_header_id|>assistant<|end_header_id|>\n\n"),
    "llama-3.1": ("<|start_header_id|>user<|end_header_id|>\n\n", "<|start_header_id|>assistant<|end_header_id|>\n\n"),
    "llama-31": ("<|start_header_id|>user<|end_header_id|>\n\n", "<|start_header_id|>assistant<|end_header_id|>\n\n"),
    "llama-3.2": ("<|start_header_id|>user<|end_header_id|>\n\n", "<|start_header_id|>assistant<|end_header_id|>\n\n"),
    "llama-32": ("<|start_header_id|>user<|end_header_id|>\n\n", "<|start_header_id|>assistant<|end_header_id|>\n\n"),
    "llama-3.3": ("<|start_header_id|>user<|end_header_id|>\n\n", "<|start_header_id|>assistant<|end_header_id|>\n\n"),
    "llama-33": ("<|start_header_id|>user<|end_header_id|>\n\n", "<|start_header_id|>assistant<|end_header_id|>\n\n"),
    "gemma": ("<start_of_turn>user\n", "<start_of_turn>model\n"),
    "gemma_chatml": ("<|im_start|>user\n", "<|im_start|>assistant\n"),
    "gemma2": ("<start_of_turn>user\n", "<start_of_turn>model\n"),
    "gemma2_chatml": ("<|im_start|>user\n", "<|im_start|>assistant\n"),
    "gemma-3": ("<start_of_turn>user\n", "<start_of_turn>model\n"),
    "gemma3": ("<start_of_turn>user\n", "<start_of_turn>model\n"),
    "gemma-3n": ("<start_of_turn>user\n", "<start_of_turn>model\n"),
    "gemma3n": ("<start_of_turn>user\n", "<start_of_turn>model\n"),
    "phi-3": ("<|user|>\n", "<|assistant|>\n"),
    "phi-35": ("<|user|>\n", "<|assistant|>\n"),
    "phi-3.5": ("<|user|>\n", "<|assistant|>\n"),
    "phi-4": ("<|user|>\n", "<|assistant|>\n"),
    "mistral": ("[INST] ", " [/INST]"),
    "zephyr": ("<|user|>\n", "<|assistant|>\n"),
    "alpaca": ("### Instruction:\n", "### Response:\n"),
    "vicuna": ("USER: ", "ASSISTANT: "),
    "vicuna_old": ("USER: ", "ASSISTANT: "),
    "vicuna old": ("USER: ", "ASSISTANT: "),
    "starling": ("GPT4 Correct User: ", "<|end_of_turn|>GPT4 Correct Assistant: "),
    "yi-chat": ("<|im_start|>user\n", "<|im_start|>assistant\n"),
}


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
    output_dir = os.path.join(output_dir, DT_STR)

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

    # init wandb with project name and run name
    wandb.init(
        project="slm_post_train",
        name=f'{model_cfg.get("name_or_path")}-{DT_STR}',
        config=config,
    )

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
        report_to="wandb",
    )

    trainer = SFTTrainer(
        model=model,
        tokenizer=tokenizer,
        train_dataset=dataset,
        args=sft_args,
    )

    if train_on_responses:
        tok = trainer.processing_class if hasattr(trainer, "processing_class") else trainer.tokenizer
        if hasattr(tok, "image_processor") or hasattr(tok, "tokenizer"):
            tok = tok.tokenizer

        has_unsloth_parts = isinstance(
            getattr(tok, "_unsloth_input_part", None), str
        ) and isinstance(getattr(tok, "_unsloth_output_part", None), str)

        if has_unsloth_parts:
            trainer = train_on_responses_only(trainer)
        else:
            instruction_part = training_cfg.get("instruction_part") or data_cfg.get("instruction_part")
            response_part = training_cfg.get("response_part") or data_cfg.get("response_part")
            if not instruction_part or not response_part:
                chat_template = data_cfg.get("chat_template", "chatml")
                template_key = str(chat_template).lower()
                if template_key not in CHAT_TEMPLATE_DELIMITERS:
                    raise ValueError(
                        f"Could not automatically determine instruction/response delimiters for chat_template '{chat_template}'. "
                        f"Supported templates are: {sorted(CHAT_TEMPLATE_DELIMITERS.keys())}. "
                        "Please explicitly configure 'instruction_part' and 'response_part' under 'training' (or 'dataset'), "
                        "or set 'train_on_responses_only: false'."
                    )
                default_inst, default_resp = CHAT_TEMPLATE_DELIMITERS[template_key]
                instruction_part = instruction_part or default_inst
                response_part = response_part or default_resp

            trainer = train_on_responses_only(
                trainer,
                instruction_part=instruction_part,
                response_part=response_part,
            )

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
