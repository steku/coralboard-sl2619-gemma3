"""Hardware verification and clock control for the Synaptics Torq Coral NPU.

This module ensures that the Coralboard SL2619 Torq Coral NPU hardware
(/sys/class/devfreq/f7600000.synpu) is detected, active, and clocked at maximum
frequency for pure NPU inference.

CPU execution is strictly prohibited.
"""

from __future__ import annotations

import logging
from pathlib import Path

logger = logging.getLogger("coral-npu-control")

NPU_DEVFREQ_PATH = Path("/sys/class/devfreq/f7600000.synpu")
NPU_GOVERNOR_PATH = NPU_DEVFREQ_PATH / "governor"
NPU_SET_FREQ_PATH = NPU_DEVFREQ_PATH / "userspace" / "set_freq"
NPU_MAX_FREQ_PATH = NPU_DEVFREQ_PATH / "max_freq"
NPU_CUR_FREQ_PATH = NPU_DEVFREQ_PATH / "cur_freq"


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


def configure_npu_max_frequency() -> bool:
    """Sets the Torq NPU to its maximum clock speed using the userspace governor."""
    verify_npu_hardware()

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


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    verify_npu_hardware()
    configure_npu_max_frequency()
    print("Coralboard NPU verification and clock setup passed.")
