import logging
import os

logger = logging.getLogger(__name__)


def export_model(
    model,
    tokenizer,
    output_dir: str,
    export_format: str = "lora",
    quantization_method: str = "q4_k_m",
):
    """Exports model as LoRA adapter, 16-bit/4-bit merged model, or GGUF binary."""
    os.makedirs(output_dir, exist_ok=True)
    logger.info(f"Exporting model to {output_dir} using format: {export_format}")

    if export_format == "lora":
        model.save_pretrained(output_dir)
        tokenizer.save_pretrained(output_dir)
        logger.info(f"Saved LoRA adapter weights to {output_dir}")

    elif export_format == "merged_16bit":
        model.save_pretrained_merged(output_dir, tokenizer, save_method="merged_16bit")
        logger.info(f"Saved 16-bit merged model to {output_dir}")

    elif export_format == "merged_4bit":
        model.save_pretrained_merged(output_dir, tokenizer, save_method="merged_4bit")
        logger.info(f"Saved 4-bit merged model to {output_dir}")

    elif export_format == "gguf":
        model.save_pretrained_gguf(output_dir, tokenizer, quantization_method=quantization_method)
        logger.info(f"Saved GGUF ({quantization_method}) to {output_dir}")

    else:
        raise ValueError(
            f"Unknown export format: '{export_format}'. Supported: lora, merged_16bit, merged_4bit, gguf"
        )
