import logging
import os
import shutil
import sys
from datetime import datetime

logger = logging.getLogger(__name__)


DT_STR = datetime.now().strftime("%Y%m%d-%H%M%S")


def setup_llama_cpp_env(llama_cpp_dir: str | None = None) -> str | None:
    """Ensures llama.cpp and its internal conversion package are on sys.path and PYTHONPATH."""
    if llama_cpp_dir is None:
        llama_cpp_dir = os.path.expanduser("~/.unsloth/llama.cpp")

    if os.path.exists(llama_cpp_dir):
        if llama_cpp_dir not in sys.path:
            sys.path.insert(0, llama_cpp_dir)
        current_pp = os.environ.get("PYTHONPATH", "")
        if llama_cpp_dir not in current_pp.split(os.pathsep):
            os.environ["PYTHONPATH"] = (
                f"{llama_cpp_dir}{os.pathsep}{current_pp}" if current_pp else llama_cpp_dir
            )
        return llama_cpp_dir
    return None


def export_model(
    model,
    tokenizer,
    output_dir: str,
    export_format: str = "lora",
    quantization_method: str = "q4_k_m",
):
    """Exports model as LoRA adapter, 16-bit/4-bit merged model, or GGUF binary."""
    output_dir = os.path.join(output_dir, DT_STR)

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
        setup_llama_cpp_env()
        model.save_pretrained_gguf(output_dir, tokenizer, quantization_method=quantization_method)

        # Unsloth saves GGUF files to f"{output_dir}_gguf"; copy artifacts into output_dir
        # so they reside directly where the user specified.
        gguf_dir = f"{output_dir.rstrip('/')}_gguf"
        if os.path.exists(gguf_dir) and os.path.abspath(gguf_dir) != os.path.abspath(output_dir):
            for fname in os.listdir(gguf_dir):
                src_path = os.path.join(gguf_dir, fname)
                dst_path = os.path.join(output_dir, fname)
                if os.path.isfile(src_path) and not os.path.exists(dst_path):
                    shutil.copy2(src_path, dst_path)
                    logger.info(f"Copied {fname} to {output_dir}")

        logger.info(f"Saved GGUF ({quantization_method}) to {output_dir}")

    else:
        raise ValueError(
            f"Unknown export format: '{export_format}'. Supported: lora, merged_16bit, merged_4bit, gguf"
        )
