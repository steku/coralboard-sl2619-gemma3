# Gemma 3 on Coralboard SL2619 Torq Coral NPU (Pure NPU Inference)

This application runs Google's **Gemma 3** (270M IT) **exclusively on the Synaptics Coralboard SL2619 Torq Coral NPU** (`f7600000.synpu`).

**CPU fallback is completely disabled.** If the Torq Coral NPU hardware or driver is unavailable, the application aborts immediately to ensure no inference workloads ever execute on the CPU.

It exposes an OpenAI-compatible API (`/v1/chat/completions`) for use as a local, private **Conversation Agent** in **Home Assistant Assist**.

---

## Hardware & Architecture

```
 ┌──────────────────────┐        HTTP (OpenAI API)         ┌──────────────────────────────────────┐
 │    Home Assistant    │ ───────────────────────────────> │  Synaptics Coralboard SL2619         │
 │                      │  POST /v1/chat/completions       │                                      │
 │  - Assist Pipeline   │                                  │  - FastAPI Server (:8000)            │
 │  - Voice Satellites  │ <─────────────────────────────── │  - torq-runtime (device_uri='torq')  │
 │                      │        SSE / JSON Response       │  - Model: Synaptics Torq VMFB (.trim)│
 └──────────────────────┘                                  │  - HARDWARE: Torq Coral NPU (1 TOPS) │
                                                           │    (/sys/class/devfreq/f7600000.synpu)
                                                           └──────────────────────────────────────┘
```

- **NPU Device**: Synaptics Torq Coral NPU (Google RISC-V Coral NPU, 1 TOPS) at `/sys/class/devfreq/f7600000.synpu`.
- **Runtime**: `torq.runtime.VMFBInferenceRunner` with `device_uri="torq"` (IREE/MLIR runtime).
- **Model**: `Synaptics/gemma-3-270m-it-torq` compiled into `.vmfb.trim` binary for the Coral NPU.
- **CPU Policy**: **Disabled**. The server verifies physical NPU devfreq nodes and asserts `device_uri='torq'`.

---

## 1. Environment Setup on the Coralboard SL2619

Clone or transfer this project to your Coralboard (e.g. `/home/root/coral-sl2619-gemma3`):

```bash
# 1. Create and activate a Python 3.12 virtual environment with system site packages
python3 -m venv .venv --system-site-packages
source .venv/bin/activate

# 2. Install dependencies (including torq-runtime)
pip install -r requirements.txt
```

> **Note on `torq-runtime`:**  
> The Synaptics Torq runtime is provided as a prebuilt wheel (`torq_runtime-*.whl`) in the Synaptics Torq Compiler release or as part of the board's SDK image. If not installed from PyPI, install your board's wheel directly:
> ```bash
> pip install /path/to/torq_runtime-*-linux_aarch64.whl
> ```

---

## 2. Download the Compiled Torq Coral NPU Model

Download the compiled Torq NPU artifacts from Hugging Face:

```bash
# Downloads compiled model.vmfb.trim, tokenizer.json, and config.json into ./models/
python3 download_model.py
```

If Hugging Face authentication is required:
```bash
export HF_TOKEN="hf_your_token_here"
python3 download_model.py
```

---

## 3. Verify NPU Hardware & Run the Server

### Verify NPU Hardware and Clock
```bash
python3 npu_control.py
```
This confirms `/sys/class/devfreq/f7600000.synpu` is detected and sets the NPU frequency governor to `userspace` at `max_freq`.

### Start the NPU Server
```bash
python3 server.py
```

On startup, the server:
1. Validates that `/sys/class/devfreq/f7600000.synpu` is present.
2. Initializes `VMFBInferenceRunner` with `device_uri="torq"`.
3. Rejects startup if the Coral NPU cannot be bound.

### Test the Endpoint
From another terminal or machine on your LAN:
```bash
# Health check (confirms backend is torq-coral-npu and CPU execution is DISABLED)
curl http://<CORALBOARD_IP>:8000/health

# Test chat prompt
python3 test_client.py --host <CORALBOARD_IP> --port 8000 --prompt "Turn off the patio lights."
```

---

## 4. Home Assistant Configuration

### Step A: Add the OpenAI Conversation Integration
1. In Home Assistant, go to **Settings** → **Devices & Services**.
2. Click **Add Integration** and choose **OpenAI Conversation**.
3. Enter:
   - **API Key**: Any string (e.g., `coral-npu`).
   - **Base URL**: `http://<CORALBOARD_IP>:8000/v1`
4. Click **Submit**.

### Step B: Configure Options
1. Under **Configure** on the integration:
   - **Model**: `gemma-3-270m-it`
   - **Max response tokens**: `128`
   - **Prompt**:
     ```
     You are Home Assistant's local voice assistant powered by the Coral NPU.
     Keep responses brief and direct in 1-2 sentences.
     ```

### Step C: Set as Assist Conversation Agent
1. In Home Assistant, go to **Settings** → **Voice Assistants** → select your pipeline (e.g. "Preferred").
2. Set **Conversation Agent** to your newly created OpenAI Conversation (Gemma 3 NPU) instance.
3. Use the Assist dialog or your voice satellites to speak to Gemma 3 on your Coral NPU!

---

## 5. Auto-start on Boot (systemd)

```bash
sudo cp coral-gemma.service /etc/systemd/system/
sudo systemctl daemon-reload
sudo systemctl enable --now coral-gemma.service
sudo systemctl status coral-gemma.service
```
