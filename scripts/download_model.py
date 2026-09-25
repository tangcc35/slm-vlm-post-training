#!/usr/bin/env python3
"""Downloads Hugging Face models into the outputs directory using credentials from .env.

Usage:
    # Run with defaults (downloads Qwen3.5-9B Q8_0 GGUF to outputs/)
    uv run python scripts/download_model.py

    # Download directly into outputs/ without a nested folder
    uv run python scripts/download_model.py --output-dir outputs

    # Download by specifying the model repository and specific file
    uv run python scripts/download_model.py \
        --model "DavidAU/Qwen3.5-9B-The-Defiant-Fable-Uncensored-Heretic-NEO-IMATRIX-MAX-MTP-GGUF" \
        --pattern "Qwen3.5-9B-The-Defiant-Fable-Uncnr-Heretic-NEO-MAX-Q8_0.gguf"
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
DEFAULT_MODEL = "https://huggingface.co/DavidAU/Qwen3.5-9B-The-Defiant-Fable-Uncensored-Heretic-NEO-IMATRIX-MAX-MTP-GGUF"
DEFAULT_PATTERN = ["Qwen3.5-9B-The-Defiant-Fable-Uncnr-Heretic-NEO-MAX-Q8_0.gguf"]
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


def normalize_repo_id(model_input: str) -> tuple[str, str | None]:
    """Extracts the Hugging Face repo ID ('author/model') and any embedded filename from URL or input."""
    model_input = model_input.strip()

    # Handle URLs
    if model_input.startswith("http://") or model_input.startswith("https://"):
        path = urlparse(model_input).path.strip("/")
        # Check for deep link to a specific file: e.g. /blob/main/<file> or /tree/main/<file>
        m = re.match(r"^([^/]+/[^/]+)/(?:blob|tree)/[^/]+/(.+)$", path)
        if m:
            return m.group(1), m.group(2)
        parts = path.split("/")
        if len(parts) >= 2:
            return f"{parts[0]}/{parts[1]}", None
        return path, None

    parts = model_input.split("/")
    # e.g., author/repo/filename.gguf
    if len(parts) == 3:
        return f"{parts[0]}/{parts[1]}", parts[2]
    # e.g., repo/filename.gguf (missing author prefix)
    elif len(parts) == 2 and any(parts[1].endswith(ext) for ext in [".gguf", ".safetensors", ".bin", ".json"]):
        repo_name = parts[0]
        if "Qwen3.5-9B-The-Defiant-Fable" in repo_name and not repo_name.startswith("DavidAU/"):
            repo_id = f"DavidAU/{repo_name}"
        else:
            repo_id = repo_name
        return repo_id, parts[1]

    return model_input, None


def download_hf_model(
    model_input: str,
    output_dir: str | Path | None = None,
    allow_patterns: list[str] | str | None = None,
    ignore_patterns: list[str] | str | None = None,
    env_path: str = DEFAULT_ENV_FILE,
) -> Path:
    """Loads environment variables, authenticates with HF_TOKEN, and downloads model."""
    # Fast transfer using hf_transfer if installed
    if "HF_HUB_ENABLE_HF_TRANSFER" not in os.environ:
        try:
            import hf_transfer  # noqa: F401
            os.environ["HF_HUB_ENABLE_HF_TRANSFER"] = "1"
        except ImportError:
            pass

    env_vars = load_env(env_path)
    hf_token = os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN")

    if hf_token:
        masked_token = f"{hf_token[:4]}...{hf_token[-4:]}" if len(hf_token) > 8 else "***"
        print(f"[INFO] Loaded HF_TOKEN from environment: {masked_token}")
    else:
        print("[WARN] No HF_TOKEN found in .env or environment; proceeding as anonymous/public access.")

    repo_id, extracted_pattern = normalize_repo_id(model_input)
    if extracted_pattern:
        if allow_patterns is None:
            allow_patterns = [extracted_pattern]
        elif isinstance(allow_patterns, list) and extracted_pattern not in allow_patterns:
            allow_patterns.append(extracted_pattern)
        elif isinstance(allow_patterns, str) and extracted_pattern != allow_patterns:
            allow_patterns = [allow_patterns, extracted_pattern]

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
        help=f"Hugging Face model repository ID, URL, or model/file path (default: '{DEFAULT_MODEL}')",
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
        default=DEFAULT_PATTERN,
        help=f"Optional glob pattern(s) to restrict download (default: {DEFAULT_PATTERN})",
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
