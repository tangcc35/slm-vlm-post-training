import torch


def load_model_and_tokenizer(model_cfg: dict = None, lora_cfg: dict = None, modality: str = "text"):
    """Loads model and tokenizer with Unsloth FastModel and attaches LoRA adapters."""
    if model_cfg is None:
        model_cfg = {}

    model_name = model_cfg.get("name_or_path", "unsloth/gemma-2-2b-it")
    max_seq_length = model_cfg.get("max_seq_length", 2048)
    load_in_4bit = model_cfg.get("load_in_4bit", True)
    dtype = model_cfg.get("dtype", None)
    if dtype == "bfloat16":
        dtype = torch.bfloat16
    elif dtype == "float16":
        dtype = torch.float16
    elif dtype == "float32":
        dtype = torch.float32

    if modality == "vision":
        from unsloth import FastVisionModel

        model, tokenizer = FastVisionModel.from_pretrained(
            model_name=model_name,
            max_seq_length=max_seq_length,
            load_in_4bit=load_in_4bit,
            dtype=dtype,
        )
    else:
        from unsloth import FastLanguageModel

        model, tokenizer = FastLanguageModel.from_pretrained(
            model_name=model_name,
            max_seq_length=max_seq_length,
            load_in_4bit=load_in_4bit,
            dtype=dtype,
        )

    if lora_cfg:
        r = lora_cfg.get("r", 16)
        lora_alpha = lora_cfg.get("lora_alpha", r * 2)
        target_modules = lora_cfg.get(
            "target_modules",
            ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
        )
        lora_dropout = lora_cfg.get("lora_dropout", 0.0)
        bias = lora_cfg.get("bias", "none")
        random_state = lora_cfg.get("random_state", 3407)
        use_gradient_checkpointing = lora_cfg.get("use_gradient_checkpointing", "unsloth")

        if modality == "vision":
            model = FastVisionModel.get_peft_model(
                model,
                r=r,
                lora_alpha=lora_alpha,
                target_modules=target_modules,
                lora_dropout=lora_dropout,
                bias=bias,
                random_state=random_state,
                use_gradient_checkpointing=use_gradient_checkpointing,
            )
        else:
            model = FastLanguageModel.get_peft_model(
                model,
                r=r,
                lora_alpha=lora_alpha,
                target_modules=target_modules,
                lora_dropout=lora_dropout,
                bias=bias,
                random_state=random_state,
                use_gradient_checkpointing=use_gradient_checkpointing,
            )

    return model, tokenizer
