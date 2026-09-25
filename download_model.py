"""Download compiled Gemma 3 Torq VMFB model artifacts for the Coralboard NPU.

Downloads the official Synaptics/gemma-3-270m-it-torq model artifacts compiled
specifically for execution on the Torq Coral NPU (f7600000.synpu).
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

DEFAULT_REPO = "Synaptics/gemma-3-270m-it-torq"
MODELS_DIR = Path(__file__).resolve().parent / "models"

# Key files required for NPU execution
CANDIDATE_VMFB_FILES = [
    "model.vmfb.trim",
    "model.vmfb",
    "transformer.vmfb",
    "lm_head.vmfb.trim",
    "lm_head.vmfb",
]

SUPPORTING_FILES = [
    "config.json",
    "tokenizer.json",
    "tokenizer_config.json",
    "token_embeddings.npy",
    "token_id_lut.npy",
    "generation_config.json",
]


def download_npu_model(repo_id: str = DEFAULT_REPO, dest_base: Path = MODELS_DIR) -> Path:
    dest_dir = dest_base / repo_id
    dest_dir.mkdir(parents=True, exist_ok=True)

    print(f"[NPU Setup] Downloading Gemma 3 NPU model artifacts from Hugging Face: {repo_id}")
    print(f"[NPU Setup] Destination directory: {dest_dir}")

    try:
        from huggingface_hub import snapshot_download

        local_dir = snapshot_download(
            repo_id=repo_id,
            local_dir=str(dest_dir),
            local_dir_use_symlinks=False,
            ignore_patterns=["*.git*", "*.md"],
        )
        print(f"\n[Success] All Torq Coral NPU artifacts downloaded to: {local_dir}")
        return Path(local_dir)
    except ImportError:
        print("huggingface_hub is required. Install with: pip install huggingface_hub")
        sys.exit(1)
    except Exception as e:
        print(f"[Error] Failed to download from Hugging Face: {e}")
        print("Note: If the repository requires Hugging Face authentication, export HF_TOKEN=hf_...")
        sys.exit(1)


def main():
    parser = argparse.ArgumentParser(description="Download Gemma 3 NPU VMFB model for Coralboard")
    parser.add_argument("--repo", default=DEFAULT_REPO, help=f"Hugging Face repository (default: {DEFAULT_REPO})")
    parser.add_argument("--dest", type=Path, default=MODELS_DIR, help="Base destination folder")
    args = parser.parse_args()

    download_npu_model(repo_id=args.repo, dest_base=args.dest)


if __name__ == "__main__":
    main()
