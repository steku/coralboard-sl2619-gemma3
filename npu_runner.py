"""Coralboard SL2619 Torq Coral NPU Inference Runner for Gemma 3.

Strictly executes model inference on the Coralboard Torq Coral NPU hardware
(f7600000.synpu). CPU fallback is completely disabled.
"""

from __future__ import annotations

import json
import logging
import os
import sys
import time
from pathlib import Path
from typing import Any, Dict, Generator, List, Optional

import numpy as np

from npu_control import configure_npu_max_frequency, verify_npu_hardware

logger = logging.getLogger("coral-npu-runner")

# Enforce hardware device URI strictly to the Coral NPU
NPU_DEVICE_URI = "torq"


class CoralGemma3NPURunner:
    """Orchestrates Gemma 3 LLM execution exclusively on the Coral NPU."""

    def __init__(self, model_dir: Path, n_threads: int = 2, max_seq_len: int = 1024):
        self.model_dir = Path(model_dir).resolve()
        self.max_seq_len = max_seq_len
        self.n_threads = n_threads

        # 1. Verify physical NPU hardware exists before doing anything
        verify_npu_hardware()
        configure_npu_max_frequency()

        # 2. Locate model artifacts
        self.vmfb_path = self._find_vmfb_file(self.model_dir)
        self.tokenizer_path = self.model_dir / "tokenizer.json"
        self.config_path = self.model_dir / "config.json"

        if not self.tokenizer_path.exists():
            raise FileNotFoundError(f"Missing tokenizer.json in {self.model_dir}")

        # 3. Load tokenizer
        from tokenizers import Tokenizer
        self.tokenizer = Tokenizer.from_file(str(self.tokenizer_path))
        logger.info("Loaded tokenizer from %s", self.tokenizer_path)

        # 4. Initialize NPU engine
        self._init_npu_engine()

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

        # Scan for any .vmfb
        all_vmfb = list(directory.glob("*.vmfb*"))
        if all_vmfb:
            logger.info("Selected Torq NPU model binary: %s", all_vmfb[0])
            return all_vmfb[0]

        raise FileNotFoundError(
            f"No compiled Torq NPU model (.vmfb) found in {directory}!\n"
            "Please run `python3 download_model.py` to download the compiled NPU model."
        )

    def _init_npu_engine(self) -> None:
        """Loads and binds the model to the Torq Coral NPU."""
        logger.info(
            "Initializing Torq Coral NPU runtime with device_uri='%s'...",
            NPU_DEVICE_URI,
        )

        # Check if Gemma3Static from torq-examples is importable
        try:
            from runner import Gemma3Static

            logger.info("Using torq-examples Gemma3Static NPU runner.")
            lm_head = self.model_dir / "lm_head.vmfb.trim"
            if not lm_head.exists():
                lm_head = self.model_dir / "lm_head.vmfb"
            lm_head_path = str(lm_head) if lm_head.exists() else None

            self._engine = Gemma3Static(
                model_path=str(self.vmfb_path),
                max_seq_len=self.max_seq_len,
                n_threads=self.n_threads,
                instruct_model=True,
                runtime_flags=[f"--torq_device_allocator=system"],
                lm_head_path=lm_head_path,
            )
            self._mode = "gemma3_static"
            return
        except ImportError:
            logger.debug("torq-examples runner not directly in PYTHONPATH. Initializing VMFBInferenceRunner.")

        # Fallback to direct torq.runtime VMFBInferenceRunner
        try:
            from torq.runtime import VMFBInferenceRunner
        except ImportError as exc:
            msg = (
                "FATAL: 'torq-runtime' is not installed or accessible!\n"
                "Inference CANNOT run on the NPU without the Synaptics Torq runtime.\n"
                "CPU execution is forbidden. Please install the torq-runtime wheel."
            )
            logger.critical(msg)
            raise RuntimeError(msg) from exc

        try:
            # Strictly instantiate on device_uri="torq"
            self._npu_runner = VMFBInferenceRunner(
                str(self.vmfb_path),
                device_uri=NPU_DEVICE_URI,
                function="main",
                load_model_to_mem=True,
                device_outputs=True,
            )
            logger.info("Successfully initialized VMFBInferenceRunner on Torq Coral NPU!")
            self._mode = "vmfb_runner"
        except Exception as exc:
            msg = (
                f"FATAL: Failed to initialize model on Torq Coral NPU device '{NPU_DEVICE_URI}': {exc}\n"
                "CPU fallback is disabled by configuration. Aborting."
            )
            logger.critical(msg)
            raise RuntimeError(msg) from exc

    def format_chat_prompt(self, messages: List[Dict[str, str]]) -> str:
        """Constructs Gemma 3 instruction turn prompt."""
        formatted_parts: List[str] = []
        for msg in messages:
            role = msg.get("role", "user")
            content = msg.get("content", "").strip()
            if role == "system":
                # Prepend system instruction to prompt context
                formatted_parts.append(f"<start_of_turn>user\n[System Instructions]\n{content}<end_of_turn>")
            elif role == "user":
                formatted_parts.append(f"<start_of_turn>user\n{content}<end_of_turn>")
            elif role == "assistant":
                formatted_parts.append(f"<start_of_turn>model\n{content}<end_of_turn>")

        # Final prompt primes model response
        formatted_parts.append("<start_of_turn>model\n")
        return "\n".join(formatted_parts)

    def generate(
        self,
        messages: List[Dict[str, str]],
        max_tokens: int = 128,
        temperature: float = 0.7,
        top_p: float = 0.9,
    ) -> Dict[str, Any]:
        """Runs complete inference on the Torq Coral NPU."""
        prompt = self.format_chat_prompt(messages)
        t0 = time.time()

        if self._mode == "gemma3_static":
            # Delegate to Gemma3Static NPU runner
            response_text = self._engine.infer(prompt)
        else:
            # Stream tokens from VMFB NPU runner and assemble
            tokens = []
            for tok in self.stream_generate(messages, max_tokens, temperature, top_p):
                tokens.append(tok)
            response_text = "".join(tokens)

        elapsed = time.time() - t0
        output_tokens = len(self.tokenizer.encode(response_text).ids)
        prompt_tokens = len(self.tokenizer.encode(prompt).ids)

        return {
            "text": response_text.strip(),
            "prompt_tokens": prompt_tokens,
            "completion_tokens": output_tokens,
            "latency_ms": elapsed * 1000,
        }

    def stream_generate(
        self,
        messages: List[Dict[str, str]],
        max_tokens: int = 128,
        temperature: float = 0.7,
        top_p: float = 0.9,
    ) -> Generator[str, None, None]:
        """Streams tokens generated on the Torq Coral NPU."""
        prompt = self.format_chat_prompt(messages)

        if self._mode == "gemma3_static":
            if hasattr(self._engine, "stream"):
                for chunk in self._engine.stream(prompt):
                    yield chunk
            else:
                full_text = self._engine.infer(prompt)
                yield full_text
            return

        # Direct token generation via VMFB NPU execution
        encoded = self.tokenizer.encode(prompt)
        input_ids = list(encoded.ids)

        curr_ids = list(input_ids)
        eos_id = self.tokenizer.token_to_id("<end_of_turn>") or 1

        for _ in range(max_tokens):
            inp_array = np.array([curr_ids[-1:]], dtype=np.int32)
            # Execute on Torq NPU
            raw_out = self._npu_runner.infer([inp_array])
            logits = raw_out[0] if isinstance(raw_out, (list, tuple)) else raw_out

            if hasattr(logits, "to_host"):
                logits = logits.to_host()

            # Greedy or temperature sampling on next token
            next_logits = logits[0, -1, :]
            if temperature <= 0.01:
                next_id = int(np.argmax(next_logits))
            else:
                scaled = next_logits / temperature
                exp_logits = np.exp(scaled - np.max(scaled))
                probs = exp_logits / np.sum(exp_logits)
                next_id = int(np.random.choice(len(probs), p=probs))

            if next_id == eos_id:
                break

            curr_ids.append(next_id)
            piece = self.tokenizer.decode([next_id])
            yield piece
