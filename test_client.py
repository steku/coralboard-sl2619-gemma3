"""CLI test client to verify the Gemma 3 API server from terminal or LAN."""

from __future__ import annotations

import argparse
import json
import sys
import time
import urllib.error
import urllib.request


def make_request(url: str, data: dict | None = None, headers: dict | None = None) -> dict:
    req_headers = {"Content-Type": "application/json"}
    if headers:
        req_headers.update(headers)

    body = json.dumps(data).encode("utf-8") if data is not None else None
    req = urllib.request.Request(url, data=body, headers=req_headers)

    with urllib.request.urlopen(req) as resp:
        return json.loads(resp.read().decode("utf-8"))


def test_streaming(url: str, data: dict, headers: dict | None = None):
    req_headers = {"Content-Type": "application/json"}
    if headers:
        req_headers.update(headers)

    body = json.dumps(data).encode("utf-8")
    req = urllib.request.Request(url, data=body, headers=req_headers)

    print("\n[Streaming Response]: ", end="", flush=True)
    with urllib.request.urlopen(req) as resp:
        for line in resp:
            decoded = line.decode("utf-8").strip()
            if not decoded or not decoded.startswith("data: "):
                continue
            payload = decoded[6:]
            if payload == "[DONE]":
                break
            try:
                chunk = json.loads(payload)
                delta = chunk["choices"][0].get("delta", {})
                content = delta.get("content", "")
                print(content, end="", flush=True)
            except Exception:
                pass
    print("\n")


def main():
    parser = argparse.ArgumentParser(description="Test Coralboard Gemma 3 Server")
    parser.add_argument("--host", default="localhost", help="Server host (default: localhost)")
    parser.add_argument("--port", type=int, default=8000, help="Server port (default: 8000)")
    parser.add_argument("--api-key", default="", help="Optional API key")
    parser.add_argument("--prompt", default="Turn off the porch light and confirm.", help="Test prompt")
    parser.add_argument("--stream", action="store_true", help="Test streaming output")
    args = parser.parse_args()

    base_url = f"http://{args.host}:{args.port}"
    headers = {"Authorization": f"Bearer {args.api_key}"} if args.api_key else {}

    print(f"Testing server at {base_url}...")

    # 1. Health check
    try:
        health = make_request(f"{base_url}/health")
        print(f"[Health Check] OK: {health}")
    except Exception as e:
        print(f"[Health Check] FAILED: {e}")
        sys.exit(1)

    # 2. Models list
    try:
        models = make_request(f"{base_url}/v1/models", headers=headers)
        model_id = models["data"][0]["id"]
        print(f"[Models List] Found model ID: {model_id}")
    except Exception as e:
        print(f"[Models List] FAILED: {e}")
        sys.exit(1)

    # 3. Chat completion
    payload = {
        "model": model_id,
        "messages": [
            {
                "role": "system",
                "content": "You are Home Assistant's smart home voice assistant. Keep answers brief and direct.",
            },
            {"role": "user", "content": args.prompt},
        ],
        "temperature": 0.3,
        "max_tokens": 128,
        "stream": args.stream,
    }

    if args.stream:
        test_streaming(f"{base_url}/v1/chat/completions", payload, headers=headers)
    else:
        print(f"\n[Prompt]: {args.prompt}")
        t0 = time.time()
        resp = make_request(f"{base_url}/v1/chat/completions", data=payload, headers=headers)
        elapsed = (time.time() - t0) * 1000

        content = resp["choices"][0]["message"]["content"]
        usage = resp.get("usage", {})
        print(f"[Response] ({elapsed:.0f} ms, {usage.get('total_tokens', 0)} tokens):")
        print(f"  {content}\n")


if __name__ == "__main__":
    main()
