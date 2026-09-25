"""Coralboard SL2619 Torq Coral NPU Inference Runner for Gemma 3.

Strictly executes model inference on the Coralboard Torq Coral NPU hardware
(f7600000.synpu) via Gemma3Static, which manages the 20-tensor input signature
(input token/position + 18 KV-cache state tensors) required by the compiled Torq
VMFB model.

CPU fallback is completely disabled.
"""

from __future__ import annotations

import logging
import os
import sys
import time
from pathlib import Path
from typing import Any, Dict, Generator, List, Optional

from npu_control import configure_npu_max_frequency, verify_npu_hardware
from runner import Gemma3Static

logger = logging.getLogger("coral-npu-runner")


class CoralGemma3NPURunner:
    """Orchestrates Gemma 3 LLM execution exclusively on the Coral NPU."""

    def __init__(
        self,
        model_dir: Path,
        n_threads: int = 2,
        max_seq_len: int = 1024,
        temperature: float = 0.7,
        top_p: float = 0.9,
    ):
        self.model_dir = Path(model_dir).resolve()
        self.max_seq_len = max_seq_len
        self.n_threads = n_threads

        # 1. Verify physical NPU hardware exists before proceeding
        verify_npu_hardware()
        configure_npu_max_frequency()

        # 2. Locate model artifacts
        self.vmfb_path = self._find_vmfb_file(self.model_dir)

        # 3. Check for separate LM head (if split)
        lm_head = self.model_dir / "lm_head.vmfb.trim"
        if not lm_head.exists():
            lm_head = self.model_dir / "lm_head.vmfb"
        lm_head_path = str(lm_head) if lm_head.exists() else None

        logger.info(
            "Initializing Gemma3Static on Torq Coral NPU from %s (LM Head: %s)...",
            self.vmfb_path.name,
            Path(lm_head_path).name if lm_head_path else "None (integrated)",
        )

        # 4. Instantiate Gemma3Static (which binds to Torq NPU and manages KV cache tensors)
        self._engine = Gemma3Static(
            model_path=str(self.vmfb_path),
            max_seq_len=self.max_seq_len,
            n_threads=self.n_threads,
            instruct_model=True,
            temperature=temperature,
            top_p=top_p,
            runtime_flags=["--torq_device_allocator=system"],
            lm_head_path=lm_head_path,
        )
        logger.info("Successfully initialized Gemma 3 on Torq Coral NPU!")

    def _find_vmfb_file(self, directory: Path) -> Path:
        """Finds the primary compiled VMFB artifact for the Torq NPU."""
        candidates = [
            directory / "model.vmfb.trim",
            directory / "model.vmfb",
            directory / "transformer.vmfb",
        ]
        for c in candidates:
            if c.is_file():
                logger.info("Selected Torq NPU model binary: %s", c)
                return c

        all_vmfb = [p for p in directory.glob("*.vmfb*") if "lm_head" not in p.name]
        if all_vmfb:
            logger.info("Selected Torq NPU model binary: %s", all_vmfb[0])
            return all_vmfb[0]

        raise FileNotFoundError(
            f"No compiled Torq NPU model (.vmfb) found in {directory}!\n"
            "Please run `python3 download_model.py` to download the compiled NPU model."
        )

    def _prepare_prompt(self, messages: List[Dict[str, str]]) -> str:
        """Extracts and formats system instructions and user input."""
        system_parts: List[str] = []
        conversation: List[str] = []
        last_user_query = ""

        for msg in messages:
            role = msg.get("role", "user")
            content = msg.get("content", "").strip()
            if role == "system":
                if content:
                    system_parts.append(content)
            elif role == "user":
                last_user_query = content
                conversation.append(f"User: {content}")
            elif role == "assistant":
                conversation.append(f"Assistant: {content}")

        if len(conversation) > 1:
            full_prompt = "\n".join(conversation)
        else:
            full_prompt = last_user_query or "Hello"

        if system_parts:
            full_prompt = f"{' '.join(system_parts)}\n\n{full_prompt}"

        return full_prompt

    def generate(
        self,
        messages: List[Dict[str, str]],
        max_tokens: int = 128,
        temperature: float = 0.7,
        top_p: float = 0.9,
    ) -> Dict[str, Any]:
        """Runs full inference on the Torq Coral NPU."""
        prompt = self._prepare_prompt(messages)
        self._engine._temperature = temperature
        self._engine._top_p = top_p

        t0 = time.time()
        response_text = self._engine.run(prompt)
        elapsed_ms = (time.time() - t0) * 1000

        output_tokens = getattr(self._engine, "generated_tokens", 0)
        prompt_tokens = len(self._engine.tokenizer.encode(prompt).ids)

        return {
            "text": response_text.strip(),
            "prompt_tokens": prompt_tokens,
            "completion_tokens": output_tokens,
            "latency_ms": elapsed_ms,
        }

    def stream_generate(
        self,
        messages: List[Dict[str, str]],
        max_tokens: int = 128,
        temperature: float = 0.7,
        top_p: float = 0.9,
    ) -> Generator[str, None, None]:
        """Streams generated tokens directly from the Torq Coral NPU."""
        prompt = self._prepare_prompt(messages)
        self._engine._temperature = temperature
        self._engine._top_p = top_p

        for chunk in self._engine.run_stream(prompt):
            yield chunk
