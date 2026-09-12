import unsloth  # Must precede trl/transformers imports
import argparse
import logging
import sys
import yaml

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("slm_post_train")


def load_yaml_config(path: str) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)
    return config if config is not None else {}


def main(argv: list[str] | None = None):
    parser = argparse.ArgumentParser(prog="slm-post-train", description="SLM/VLM Post-Training CLI")
    subparsers = parser.add_subparsers(dest="subcommand", required=True)

    # Train command
    train_parser = subparsers.add_parser("train", help="Run a training job")
    train_parser.add_argument("--config", required=True, help="Path to YAML training configuration")

    # Export command
    export_parser = subparsers.add_parser("export", help="Export or quantize a model checkpoint")
    export_parser.add_argument("--model-path", required=True, help="Path to trained model or adapter")
    export_parser.add_argument("--output-dir", required=True, help="Path to save exported model")
    export_parser.add_argument(
        "--format",
        default="lora",
        choices=["lora", "merged_16bit", "merged_4bit", "gguf"],
        help="Export target format",
    )
    export_parser.add_argument("--quant", default="q4_k_m", help="Quantization method for GGUF")

    # Curate data command
    curate_parser = subparsers.add_parser("curate-data", help="Curate and process dataset using a YAML config")
    curate_parser.add_argument("--config", required=True, help="Path to YAML configuration")

    args = parser.parse_args(argv)

    if args.subcommand == "train":
        config = load_yaml_config(args.config)
        stage = config.get("stage", "sft").lower()
        if stage == "sft":
            from slm_post_train.trainers.sft_runner import run_sft

            run_sft(config)
        elif stage == "grpo":
            from slm_post_train.trainers.grpo_runner import run_grpo

            run_grpo(config)
        else:
            raise ValueError(f"Unknown training stage: '{stage}'. Must be 'sft' or 'grpo'.")

    elif args.subcommand == "curate-data":
        config = load_yaml_config(args.config)
        curation_cfg = config.get("dataset", {}).get("curation") or config.get("curation")
        if not curation_cfg:
            raise ValueError(
                f"No curation configuration found in {args.config}. "
                "Please add a 'curation' section under 'dataset' (or at root level)."
            )

        from slm_post_train.data.nsfw_story import curate_from_config

        train_p, val_p = curate_from_config(curation_cfg)
        logger.info("Dataset curation completed successfully.")
        logger.info(f"Train file: {train_p}")
        logger.info(f"Validation file: {val_p}")


    elif args.subcommand == "export":
        from unsloth import FastLanguageModel
        from slm_post_train.export.exporter import export_model

        logger.info(f"Loading model from {args.model_path} for export...")
        model, tokenizer = FastLanguageModel.from_pretrained(args.model_path)
        export_model(
            model,
            tokenizer,
            args.output_dir,
            export_format=args.format,
            quantization_method=args.quant,
        )



if __name__ == "__main__":
    main()
