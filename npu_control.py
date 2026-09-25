"""Hardware verification and clock control for the Synaptics Torq Coral NPU.

This module ensures that the Coralboard SL2619 Torq Coral NPU hardware
(/sys/class/devfreq/f7600000.synpu) is detected, its hardware clock is un-gated
via the clock control register (0xf7e104b0 = 0x216), and clocked at maximum
frequency for pure NPU inference.

CPU execution is strictly prohibited.
"""

from __future__ import annotations

import logging
import mmap
import os
import struct
import subprocess
from pathlib import Path

logger = logging.getLogger("coral-npu-control")

NPU_DEVFREQ_PATH = Path("/sys/class/devfreq/f7600000.synpu")
NPU_GOVERNOR_PATH = NPU_DEVFREQ_PATH / "governor"
NPU_SET_FREQ_PATH = NPU_DEVFREQ_PATH / "userspace" / "set_freq"
NPU_MAX_FREQ_PATH = NPU_DEVFREQ_PATH / "max_freq"
NPU_CUR_FREQ_PATH = NPU_DEVFREQ_PATH / "cur_freq"

# Physical address of the Torq Coral NPU clock gate register on Astra SL2619
NPU_CLK_REG_ADDR = 0xF7E104B0
NPU_CLK_ENABLE_VAL = 0x216


def verify_npu_hardware() -> None:
    """Verifies that the Coralboard NPU device exists in sysfs.

    Raises RuntimeError if the NPU device node is not found, ensuring CPU
    execution can never occur.
    """
    if not NPU_DEVFREQ_PATH.exists():
        msg = (
            f"FATAL: Coralboard NPU device node not found at {NPU_DEVFREQ_PATH}!\n"
            "This application is strictly configured to run only on the Coral NPU. "
            "CPU fallback is disabled. Please verify you are running on the Synaptics "
            "Coralboard SL2619 with the Torq NPU driver loaded."
        )
        logger.critical(msg)
        raise RuntimeError(msg)

    logger.info("Verified Coralboard Torq Coral NPU hardware: %s", NPU_DEVFREQ_PATH)


def enable_npu_clock() -> tuple[bool, str]:
    """Un-gates the Torq NPU clock register (0xf7e104b0 = 0x216) to allow XRAM writes."""
    # 1. Try devmem utility
    try:
        res = subprocess.run(
            ["devmem", f"0x{NPU_CLK_REG_ADDR:x}", "32", f"0x{NPU_CLK_ENABLE_VAL:x}"],
            capture_output=True,
            timeout=5,
            check=False,
        )
        if res.returncode == 0:
            logger.info(
                "NPU clock enabled via devmem (0x%x = 0x%x)",
                NPU_CLK_REG_ADDR,
                NPU_CLK_ENABLE_VAL,
            )
            return True, "NPU clock enabled via devmem"
    except FileNotFoundError:
        logger.debug("devmem command not found in PATH, trying direct /dev/mem...")
    except Exception as exc:
        logger.debug("devmem failed: %s, trying direct /dev/mem...", exc)

    # 2. Fallback: Direct memory map via /dev/mem
    try:
        page_size = os.sysconf("SC_PAGE_SIZE")
        page_addr = NPU_CLK_REG_ADDR & ~(page_size - 1)
        offset_in_page = NPU_CLK_REG_ADDR - page_addr

        with open("/dev/mem", "r+b", buffering=0) as f:
            with mmap.mmap(f.fileno(), page_size, offset=page_addr) as mm:
                mm.seek(offset_in_page)
                mm.write(struct.pack("<I", NPU_CLK_ENABLE_VAL))
                mm.flush()
        logger.info(
            "NPU clock enabled via /dev/mem mmap (0x%x = 0x%x)",
            NPU_CLK_REG_ADDR,
            NPU_CLK_ENABLE_VAL,
        )
        return True, "NPU clock enabled via /dev/mem"
    except PermissionError:
        msg = "Permission denied opening /dev/mem. Must run as root to un-gate NPU clock."
        logger.warning(msg)
        return False, msg
    except Exception as exc:
        msg = f"NPU clock register setup failed: {exc}"
        logger.warning(msg)
        return False, msg


def configure_npu_max_frequency() -> bool:
    """Sets the Torq NPU to its maximum clock speed using the userspace governor."""
    verify_npu_hardware()
    enable_npu_clock()

    try:
        if NPU_GOVERNOR_PATH.exists():
            current_gov = NPU_GOVERNOR_PATH.read_text().strip()
            if current_gov != "userspace":
                NPU_GOVERNOR_PATH.write_text("userspace")
                logger.info("NPU governor changed from '%s' to 'userspace'", current_gov)

        if NPU_MAX_FREQ_PATH.exists() and NPU_SET_FREQ_PATH.exists():
            max_freq = NPU_MAX_FREQ_PATH.read_text().strip()
            NPU_SET_FREQ_PATH.write_text(max_freq)
            logger.info("Set Torq Coral NPU clock frequency to maximum: %s Hz", max_freq)
            return True
    except PermissionError:
        logger.warning(
            "Permission denied configuring NPU frequency. Run as root or:\n"
            "  echo userspace | sudo tee /sys/class/devfreq/f7600000.synpu/governor\n"
            "  cat /sys/class/devfreq/f7600000.synpu/max_freq | sudo tee /sys/class/devfreq/f7600000.synpu/userspace/set_freq"
        )
        return False
    except OSError as e:
        logger.warning("Could not set NPU clock frequency: %s", e)
        return False

    return True


def setup_npu_for_inference() -> None:
    """Complete pre-flight hardware setup for Coral NPU inference."""
    verify_npu_hardware()
    enable_npu_clock()
    configure_npu_max_frequency()


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    setup_npu_for_inference()
    print("Coralboard NPU verification and clock setup passed.")
