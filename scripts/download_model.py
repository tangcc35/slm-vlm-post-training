#!/usr/bin/env python3
"""Downloads Hugging Face models into the outputs directory using credentials from .env.

Usage:
    # Run with defaults (downloads the example Qwen3.5-2B model to outputs/)
    python scripts/download_model.py

    # Download a specific model repository or Hugging Face URL
    python scripts/download_model.py --model "https://huggingface.co/HauhauCS/Qwen3.5-2B-Uncensored-HauhauCS-Aggressive"

    # Download only a specific file pattern (e.g., a specific GGUF quant or safetensors)
    python scripts/download_model.py --pattern "*Q4_K_M*"

    # Specify custom output path
    python scripts/download_model.py --output-dir "outputs/custom_model_dir"
"""

import argparse
import os
import re
from pathlib import Path
from urllib.parse import urlparse

from huggingface_hub import snapshot_download

# =====================================================================
# DEFAULT CONFIGURATION
# =====================================================================
DEFAULT_MODEL = "https://huggingface.co/HauhauCS/Qwen3.5-2B-Uncensored-HauhauCS-Aggressive"
DEFAULT_OUTPUT_BASE = "outputs"
DEFAULT_ENV_FILE = ".env"
# =====================================================================


def load_env(env_path: Path | str = DEFAULT_ENV_FILE) -> dict[str, str]:
    """Loads key-value pairs from a .env file into os.environ.
    
    Does not require external packages like python-dotenv.
    """
    env_file = Path(env_path)
    if not env_file.is_file():
        # Fallback to repo root if run from a subdirectory
        repo_root_env = Path(__file__).resolve().parent.parent / ".env"
        if repo_root_env.is_file():
            env_file = repo_root_env
        else:
            return {}

    loaded = {}
    with open(env_file, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            key, val = line.split("=", 1)
            key = key.strip()
            val = val.strip().strip("'\"")
            os.environ[key] = val
            loaded[key] = val
    return loaded


def normalize_repo_id(model_input: str) -> str:
    """Extracts the Hugging Face repo ID ('author/model') from a URL or repo ID string."""
    model_input = model_input.strip()
    if model_input.startswith("http://") or model_input.startswith("https://"):
        path = urlparse(model_input).path.strip("/")
        # Remove any leading tree/main, blob/main, etc. if someone pasted a deep link
        path = re.sub(r"^(tree|blob)/[^/]+/", "", path)
        parts = path.split("/")
        if len(parts) >= 2:
            return f"{parts[0]}/{parts[1]}"
        return path
    return model_input


def download_hf_model(
    model_input: str,
    output_dir: str | Path | None = None,
    allow_patterns: list[str] | str | None = None,
    ignore_patterns: list[str] | str | None = None,
    env_path: str = DEFAULT_ENV_FILE,
) -> Path:
    """Loads environment variables, authenticates with HF_TOKEN, and downloads model."""
    env_vars = load_env(env_path)
    hf_token = os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN")

    if hf_token:
        masked_token = f"{hf_token[:4]}...{hf_token[-4:]}" if len(hf_token) > 8 else "***"
        print(f"[INFO] Loaded HF_TOKEN from environment: {masked_token}")
    else:
        print("[WARN] No HF_TOKEN found in .env or environment; proceeding as anonymous/public access.")

    repo_id = normalize_repo_id(model_input)
    model_folder_name = repo_id.split("/")[-1]

    if output_dir:
        destination = Path(output_dir)
    else:
        destination = Path(DEFAULT_OUTPUT_BASE) / model_folder_name

    destination.mkdir(parents=True, exist_ok=True)

    print(f"[INFO] Hugging Face Repo ID: {repo_id}")
    print(f"[INFO] Target Output Directory: {destination.resolve()}")
    if allow_patterns:
        print(f"[INFO] Filtering allowed patterns: {allow_patterns}")
    if ignore_patterns:
        print(f"[INFO] Filtering ignored patterns: {ignore_patterns}")

    print(f"[INFO] Starting download (resumable)...")
    downloaded_path = snapshot_download(
        repo_id=repo_id,
        local_dir=str(destination),
        token=hf_token,
        allow_patterns=allow_patterns,
        ignore_patterns=ignore_patterns,
        max_workers=4,
    )

    print(f"[SUCCESS] Model successfully downloaded to: {downloaded_path}")
    return Path(downloaded_path)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Download Hugging Face models to the outputs directory using .env credentials."
    )
    parser.add_argument(
        "-m",
        "--model",
        type=str,
        default=DEFAULT_MODEL,
        help=f"Hugging Face model repository ID or URL (default: '{DEFAULT_MODEL}')",
    )
    parser.add_argument(
        "-o",
        "--output-dir",
        type=str,
        default=None,
        help="Custom destination directory (default: outputs/<model-name>)",
    )
    parser.add_argument(
        "-p",
        "--pattern",
        type=str,
        nargs="+",
        default=None,
        help="Optional glob pattern(s) to restrict download (e.g. '*Q4_K_M*' or '*.safetensors')",
    )
    parser.add_argument(
        "--ignore",
        type=str,
        nargs="+",
        default=None,
        help="Optional glob pattern(s) to ignore during download",
    )
    parser.add_argument(
        "--env-file",
        type=str,
        default=DEFAULT_ENV_FILE,
        help=f"Path to .env file (default: '{DEFAULT_ENV_FILE}')",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    download_hf_model(
        model_input=args.model,
        output_dir=args.output_dir,
        allow_patterns=args.pattern,
        ignore_patterns=args.ignore,
        env_path=args.env_file,
    )


if __name__ == "__main__":
    main()
