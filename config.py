"""Runtime configuration for Gemma 3 on Coralboard SL2619 Torq NPU."""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path


@dataclass
class Settings:
    # Server network settings
    host: str = os.getenv("HOST", "0.0.0.0")
    port: int = int(os.getenv("PORT", "8000"))
    api_key: str = os.getenv("API_KEY", "")  # Empty string means no auth required

    # Model and NPU parameters
    models_dir: Path = Path(os.getenv("MODELS_DIR", "./models"))
    model_repo: str = os.getenv("MODEL_REPO", "Synaptics/gemma-3-270m-it-torq")
    model_name: str = os.getenv("MODEL_NAME", "gemma-3-270m-it")
    device_uri: str = "torq"  # Hardcoded: NEVER fallback to CPU

    # Inference parameters
    max_seq_len: int = int(os.getenv("MAX_SEQ_LEN", "1024"))
    max_tokens: int = int(os.getenv("MAX_TOKENS", "128"))
    temperature: float = float(os.getenv("TEMPERATURE", "0.7"))
    top_p: float = float(os.getenv("TOP_P", "0.9"))
    n_threads: int = int(os.getenv("N_THREADS", "2"))

    def resolve_model_dir(self) -> Path:
        """Finds or validates the directory containing the compiled Torq NPU model."""
        # 1. Direct path check
        target = self.models_dir / self.model_repo
        if target.is_dir():
            return target

        # 2. Look inside models_dir for any directory containing .vmfb
        if self.models_dir.is_dir():
            for sub in self.models_dir.iterdir():
                if sub.is_dir() and any(sub.glob("*.vmfb*")):
                    return sub
                if sub.is_dir():
                    for nested in sub.iterdir():
                        if nested.is_dir() and any(nested.glob("*.vmfb*")):
                            return nested

        return target


settings = Settings()
