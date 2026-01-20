#!/usr/bin/env python3
"""Download models for Osito."""

import sys
import shutil
from pathlib import Path
from huggingface_hub import hf_hub_download

MODELS_DIR = Path(__file__).parent / "models"

PIPER_REPO = "rhasspy/piper-voices"
PIPER_VOICE_NAME = "es_ES-sharvard-medium"
PIPER_VOICE_PATH = f"es/es_ES/sharvard/medium/{PIPER_VOICE_NAME}.onnx"
PIPER_JSON_PATH = f"es/es_ES/sharvard/medium/{PIPER_VOICE_NAME}.onnx.json"


def download_piper_voice():
    """Download Piper Spanish voice."""
    voice_path = MODELS_DIR / f"{PIPER_VOICE_NAME}.onnx"
    json_path = MODELS_DIR / f"{PIPER_VOICE_NAME}.onnx.json"

    if voice_path.exists() and voice_path.stat().st_size > 1000:
        print("  Piper voice already exists")
        return

    print("  Downloading Piper Spanish voice (~63MB)...")

    temp_dir = MODELS_DIR / "piper-temp"
    hf_hub_download(repo_id=PIPER_REPO, filename=PIPER_VOICE_PATH, local_dir=temp_dir)
    hf_hub_download(repo_id=PIPER_REPO, filename=PIPER_JSON_PATH, local_dir=temp_dir)

    src_onnx = temp_dir / PIPER_VOICE_PATH
    src_json = temp_dir / PIPER_JSON_PATH

    shutil.copy(src_onnx, voice_path)
    shutil.copy(src_json, json_path)
    shutil.rmtree(temp_dir, ignore_errors=True)

    print("  Piper voice downloaded")


def main():
    """Download models."""
    MODELS_DIR.mkdir(exist_ok=True)

    # Check for --piper-only flag
    piper_only = "--piper-only" in sys.argv

    print("\nDownloading Piper Spanish voice...")
    download_piper_voice()

    if not piper_only:
        print("\nNote: LLM is handled by Ollama (gpt-oss:20b)")

    print("\nDone!")


if __name__ == "__main__":
    main()
