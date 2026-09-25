"""OpenAI-compatible LLM Server for Gemma 3 on Coralboard SL2619 NPU.

Strictly runs on the Synaptics Torq Coral NPU (device_uri='torq').
CPU execution is completely disabled.

Exposes /v1/chat/completions and /v1/models for Home Assistant Assist.
"""

from __future__ import annotations

import asyncio
import json
import logging
import time
from contextlib import asynccontextmanager
from typing import Any, AsyncGenerator, Dict, List, Optional, Union

import uvicorn
from fastapi import Depends, FastAPI, Header, HTTPException, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, StreamingResponse
from pydantic import BaseModel

from config import settings
from npu_control import configure_npu_max_frequency, verify_npu_hardware
from npu_runner import CoralGemma3NPURunner

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("coral-gemma3-npu")

# Global NPU runner and inference lock
_npu_runner: Optional[CoralGemma3NPURunner] = None
_inference_lock = asyncio.Lock()


# -----------------------------------------------------------------------------
# Data Models
# -----------------------------------------------------------------------------
class ChatMessage(BaseModel):
    role: str
    content: Optional[str] = ""
    name: Optional[str] = None


class ChatCompletionRequest(BaseModel):
    model: Optional[str] = None
    messages: List[ChatMessage]
    temperature: Optional[float] = None
    top_p: Optional[float] = None
    max_tokens: Optional[int] = None
    stream: Optional[bool] = False
    tools: Optional[List[Dict[str, Any]]] = None
    tool_choice: Optional[Union[str, Dict[str, Any]]] = None


# -----------------------------------------------------------------------------
# Lifespan: Strictly Validate and Initialize NPU
# -----------------------------------------------------------------------------
@asynccontextmanager
async def lifespan(app: FastAPI):
    global _npu_runner

    logger.info("Verifying Coralboard Torq Coral NPU hardware...")
    # Will throw RuntimeError if NPU node /sys/class/devfreq/f7600000.synpu is missing
    verify_npu_hardware()
    configure_npu_max_frequency()

    model_dir = settings.resolve_model_dir()
    logger.info("Initializing Gemma 3 on Coral NPU from %s...", model_dir)

    try:
        _npu_runner = CoralGemma3NPURunner(
            model_dir=model_dir,
            n_threads=settings.n_threads,
            max_seq_len=settings.max_seq_len,
        )
        logger.info("[SUCCESS] Gemma 3 is running strictly on Coralboard Torq Coral NPU!")
    except Exception as exc:
        logger.critical("FAILED to load Gemma 3 on NPU: %s", exc)
        logger.critical("CPU execution is disabled by design. Exiting.")
        raise RuntimeError(f"NPU initialization failed: {exc}") from exc

    yield

    logger.info("Shutting down Coral NPU server...")


# -----------------------------------------------------------------------------
# FastAPI App
# -----------------------------------------------------------------------------
app = FastAPI(
    title="Coralboard SL2619 Torq NPU Gemma 3 Service",
    description="Pure Coral NPU inference server for Home Assistant Assist",
    version="2.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


def verify_api_key(authorization: Optional[str] = Header(None)):
    """Verifies optional API key. Allows any string if settings.api_key is empty."""
    if not settings.api_key:
        return True

    if not authorization:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Missing Authorization header",
        )

    scheme, _, token = authorization.partition(" ")
    if scheme.lower() != "bearer" or token != settings.api_key:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid API key",
        )
    return True


@app.get("/health")
async def health_check():
    return {
        "status": "ok" if _npu_runner is not None else "npu_uninitialized",
        "backend": "torq-coral-npu",
        "device": "f7600000.synpu",
        "cpu_execution": "DISABLED",
        "model": settings.model_name,
    }


@app.get("/v1/models", dependencies=[Depends(verify_api_key)])
async def list_models():
    return {
        "object": "list",
        "data": [
            {
                "id": settings.model_name,
                "object": "model",
                "created": int(time.time()),
                "owned_by": "coral-npu",
                "permission": [],
                "root": settings.model_name,
                "parent": None,
            }
        ],
    }


@app.post("/v1/chat/completions", dependencies=[Depends(verify_api_key)])
async def chat_completions(req: ChatCompletionRequest):
    if _npu_runner is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Coral NPU is not initialized.",
        )

    raw_msgs = [m.model_dump(exclude_none=True) for m in req.messages]
    temperature = req.temperature if req.temperature is not None else settings.temperature
    top_p = req.top_p if req.top_p is not None else settings.top_p
    max_tokens = req.max_tokens if req.max_tokens is not None else settings.max_tokens

    async with _inference_lock:
        if req.stream:
            async def event_generator() -> AsyncGenerator[str, None]:
                loop = asyncio.get_running_loop()
                created = int(time.time())

                def get_chunks():
                    return list(
                        _npu_runner.stream_generate(
                            raw_msgs,
                            max_tokens=max_tokens,
                            temperature=temperature,
                            top_p=top_p,
                        )
                    )

                token_pieces = await loop.run_in_executor(None, get_chunks)
                for piece in token_pieces:
                    chunk = {
                        "id": f"chatcmpl-{int(time.time() * 1000)}",
                        "object": "chat.completion.chunk",
                        "created": created,
                        "model": settings.model_name,
                        "choices": [
                            {
                                "index": 0,
                                "delta": {"content": piece},
                                "finish_reason": None,
                            }
                        ],
                    }
                    yield f"data: {json.dumps(chunk)}\n\n"
                    await asyncio.sleep(0)

                # Send final stop chunk
                final_chunk = {
                    "id": f"chatcmpl-{int(time.time() * 1000)}",
                    "object": "chat.completion.chunk",
                    "created": created,
                    "model": settings.model_name,
                    "choices": [
                        {
                            "index": 0,
                            "delta": {},
                            "finish_reason": "stop",
                        }
                    ],
                }
                yield f"data: {json.dumps(final_chunk)}\n\n"
                yield "data: [DONE]\n\n"

            return StreamingResponse(event_generator(), media_type="text/event-stream")

        # Non-streaming execution on Coral NPU
        loop = asyncio.get_running_loop()
        res = await loop.run_in_executor(
            None,
            lambda: _npu_runner.generate(
                raw_msgs,
                max_tokens=max_tokens,
                temperature=temperature,
                top_p=top_p,
            ),
        )

        response = {
            "id": f"chatcmpl-{int(time.time() * 1000)}",
            "object": "chat.completion",
            "created": int(time.time()),
            "model": settings.model_name,
            "choices": [
                {
                    "index": 0,
                    "message": {
                        "role": "assistant",
                        "content": res["text"],
                    },
                    "finish_reason": "stop",
                }
            ],
            "usage": {
                "prompt_tokens": res["prompt_tokens"],
                "completion_tokens": res["completion_tokens"],
                "total_tokens": res["prompt_tokens"] + res["completion_tokens"],
            },
        }

        tps = (
            (res["completion_tokens"] / (res["latency_ms"] / 1000))
            if res["latency_ms"] > 0
            else 0
        )
        logger.info(
            "Coral NPU Turn: %.0f ms (%d tokens, %.1f tok/s)",
            res["latency_ms"],
            res["completion_tokens"],
            tps,
        )

        return JSONResponse(content=response)


def main():
    logger.info("Starting Coralboard SL2619 Torq Coral NPU Server on %s:%d", settings.host, settings.port)
    uvicorn.run(
        app,
        host=settings.host,
        port=settings.port,
        log_level="info",
        access_log=False,
    )


if __name__ == "__main__":
    main()
