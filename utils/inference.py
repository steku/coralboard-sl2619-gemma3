# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright © 2026 Synaptics Incorporated.

from __future__ import annotations

import os
from abc import abstractmethod
from collections.abc import Iterable, Mapping
from time import perf_counter_ns
from typing import Any

import numpy as np
import numpy.typing as npt

from torq.runtime import VMFBInferenceRunner
from iree.runtime import DeviceArray


class SimpleVMFBInferenceRunner:
    """Wrapper for simple VMFB models with optional device I/O."""

    def __init__(
        self,
        model_path: str | os.PathLike,
        *,
        device_uri: str = "torq",
        function: str = "main",
        runtime_flags: list[str] | None = None,
        device_io: bool = False,
        device_outputs: bool = False,
        load_model_to_mem: bool = True,
    ):
        runner_kwargs = {
            "device_uri": device_uri,
            "function": function,
            "load_model_to_mem": load_model_to_mem,
            "device_outputs": device_outputs or device_io,
        }
        if runtime_flags:
            runner_kwargs["runtime_flags"] = runtime_flags

        self.runner = VMFBInferenceRunner(model_path, **runner_kwargs)
        self.device_io = device_io

    @property
    def infer_time_ms(self):
        return self.runner.infer_time_ms

    def prepare_input(self, input_data):
        if not self.device_io:
            return input_data
        return self.runner.allocate_device_array(input_data)

    @staticmethod
    def prepare_output(output):
        if hasattr(output, "to_host"):
            return output.to_host()
        return output

    def infer(self, input_data):
        runner_input = self.prepare_input(input_data)
        outputs = self.runner.infer([runner_input])

        if isinstance(outputs, (list, tuple)):
            if len(outputs) != 1:
                raise RuntimeError(
                    f"Expected a single output tensor from the model, but got {len(outputs)} outputs. "
                    "This runner currently supports only single-output models. "
                    "Please update the code to select the desired output tensor."
                )
            outputs = outputs[0]

        return self.prepare_output(outputs)


class BaseManagedCacheRunner(VMFBInferenceRunner):
    """Abstract base for inference runners with managed KV caches."""

    def __init__(
        self,
        model_path: str | os.PathLike,
        cache_start_idx: int = 1,
        **kwargs,
    ) -> None:
        kwargs["device_outputs"] = True
        super().__init__(model_path, **kwargs)

        if self.inputs_info is None or self.outputs_info is None:
            raise ValueError(
                f"Model '{model_path}' is missing input/output metadata "
                "required for KV cache management."
            )

        self._cache_start_idx = cache_start_idx

    @abstractmethod
    def reset_kv(self) -> None:
        """Reset mutable KV caches to their initial state."""
        ...

    @abstractmethod
    def save_kv_state(self):
        """Snapshot the current KV-cache state to host NumPy arrays."""
        ...

    @abstractmethod
    def restore_kv_state(self, state) -> None:
        """Restore KV caches from a previously saved snapshot."""
        ...


class ManagedSelfAttnCacheRunner(BaseManagedCacheRunner):
    """VMFBInferenceRunner with managed self-attention KV cache.

    For decoder-only architectures (e.g. Gemma, LLaMA).
    Assumes model outputs layout: [..., self_v0, self_k0, self_v1, self_k1, ..., self_vN, self_kN].
    """

    def __init__(
        self,
        model_path: str | os.PathLike,
        cache_start_idx: int = 1,
        device_io: bool = False,
        **kwargs,
    ) -> None:
        self._device_io = device_io
        super().__init__(model_path, cache_start_idx=cache_start_idx, **kwargs)

        self._n_kv = len(self.outputs_info) - self._cache_start_idx

        in_info = self.inputs_info
        self._kv_init = [
            np.zeros(in_info[i].shape, dtype=np.dtype(in_info[i].dtype))
            for i in range(len(in_info) - self._n_kv, len(in_info))
        ]
        self._kv_cache = [self.allocate_device_array(z) for z in self._kv_init]

    def _infer(self, inputs: Iterable[npt.NDArray] | Mapping[str, npt.NDArray]) -> list:
        if isinstance(inputs, Mapping):
            user_inputs = list(inputs.values())
        else:
            user_inputs = list(inputs)

        if self._device_io:
            user_inputs = [
                self.allocate_device_array(x) if isinstance(x, np.ndarray) else x
                for x in user_inputs
            ]

        full_inputs = user_inputs + self._kv_cache

        results = super()._infer(full_inputs)

        # Outputs from cache_start_idx onward are updated KV caches.
        for i in range(self._n_kv):
            self._kv_cache[i] = results[self._cache_start_idx + i]

        return results[:self._cache_start_idx]

    def reset_kv(self) -> None:
        """Reset all KV caches to zeros."""
        self._kv_cache = [self.allocate_device_array(z) for z in self._kv_init]

    def save_kv_state(self) -> list[np.ndarray]:
        """Snapshot the current KV-cache state to host NumPy arrays."""
        return [kv.to_host().copy() for kv in self._kv_cache]

    def restore_kv_state(self, state: list[np.ndarray]) -> None:
        """Restore KV caches from a previously saved snapshot."""
        self._kv_cache = [self.allocate_device_array(arr) for arr in state]

    def shift_kv(self, keep_last_n: int, seq_axis: int = 2, protect_first_n: int = 0) -> None:
        """Shift the last *keep_last_n* entries to just after protected tokens."""
        for i in range(self._n_kv):
            host = self._kv_cache[i].to_host()
            seq_len = host.shape[seq_axis]
            dest_start = protect_first_n
            if dest_start + keep_last_n >= seq_len:
                continue
            new = np.zeros_like(host)
            if protect_first_n > 0:
                pfx = [slice(None)] * host.ndim
                pfx[seq_axis] = slice(0, protect_first_n)
                new[tuple(pfx)] = host[tuple(pfx)]

            src = [slice(None)] * host.ndim
            dst = [slice(None)] * host.ndim
            src[seq_axis] = slice(seq_len - keep_last_n, seq_len)
            dst[seq_axis] = slice(dest_start, dest_start + keep_last_n)
            new[tuple(dst)] = host[tuple(src)]
            self._kv_cache[i] = self.allocate_device_array(new)


def _canonical_dtype(dtype):
    try:
        return np.dtype(dtype)
    except (AttributeError, TypeError, ValueError):
        return str(dtype)


def _tensor_info_summary(info) -> str:
    return f"shape={getattr(info, 'shape', None)}, dtype={getattr(info, 'dtype', None)}"


class SplitLMHeadRunner:
    """Adapter for split body + lm_head inference on Torq NPU."""

    def __init__(
        self,
        body: BaseManagedCacheRunner,
        lm_head_path: str | os.PathLike,
        **kwargs,
    ) -> None:
        self._body = body
        self._infer_time_ms = 0.0
        lm_head_kwargs = {
            k: kwargs[k] for k in ("n_threads", "runtime_flags") if k in kwargs
        }
        self._lm_head = VMFBInferenceRunner(
            lm_head_path,
            device_outputs=True,
            **lm_head_kwargs,
        )
        self._validate_lm_head_io(lm_head_path)

    def _validate_lm_head_io(self, lm_head_path: str | os.PathLike) -> None:
        body_outputs = self._body.outputs_info
        lm_head_inputs = self._lm_head.inputs_info
        if not body_outputs:
            raise ValueError(
                f"Body model '{self._body.model_path}' is missing output metadata."
            )
        if not lm_head_inputs:
            raise ValueError(
                f"LM head model '{lm_head_path}' is missing input metadata."
            )

        body_hidden = body_outputs[0]
        lm_head_input = lm_head_inputs[0]
        body_shape = tuple(body_hidden.shape)
        lm_head_shape = tuple(lm_head_input.shape)
        if body_shape != lm_head_shape:
            raise ValueError(
                f"Split LM head input shape does not match body output shape: "
                f"body output[0] {_tensor_info_summary(body_hidden)}, "
                f"LM head input[0] {_tensor_info_summary(lm_head_input)}."
            )

        body_dtype = _canonical_dtype(body_hidden.dtype)
        lm_head_dtype = _canonical_dtype(lm_head_input.dtype)
        if body_dtype != lm_head_dtype:
            raise ValueError(
                f"Split LM head input dtype does not match body output dtype: "
                f"body output[0] {_tensor_info_summary(body_hidden)}, "
                f"LM head input[0] {_tensor_info_summary(lm_head_input)}."
            )

    @property
    def model_path(self) -> str | os.PathLike:
        return self._body.model_path

    @property
    def infer_time_ms(self) -> float:
        return self._infer_time_ms

    @property
    def inputs_info(self):
        return self._body.inputs_info

    @property
    def outputs_info(self):
        body_outputs = self._body.outputs_info
        lm_head_outputs = self._lm_head.outputs_info
        if not body_outputs or not lm_head_outputs:
            return body_outputs
        return [lm_head_outputs[0], *body_outputs[1:]]

    @property
    def device(self):
        return self._body.device

    def infer(
        self,
        inputs: Iterable[npt.NDArray] | Mapping[str, npt.NDArray],
        *,
        skip_lm_head: bool = False,
    ) -> list:
        start = perf_counter_ns()
        results = self._body.infer(inputs)
        if skip_lm_head:
            self._infer_time_ms = (perf_counter_ns() - start) / 1e6
            return results
        lm_out = self._lm_head.infer([results[0]])
        self._infer_time_ms = (perf_counter_ns() - start) / 1e6
        return [lm_out[0], *results[1:]]

    def allocate_device_array(self, array: npt.NDArray) -> DeviceArray:
        return self._body.allocate_device_array(array)

    def reset_kv(self) -> None:
        self._body.reset_kv()

    def save_kv_state(self):
        return self._body.save_kv_state()

    def restore_kv_state(self, state) -> None:
        self._body.restore_kv_state(state)

    def shift_kv(self, *args, **kwargs) -> None:
        self._body.shift_kv(*args, **kwargs)
