"""Scalar reference files per platform (plan_evotorch.md Step 8, CUDA fixes, item 4).

The scalar simulator (mvb/) is not bit-reproducible across platforms: it uses Python's
`math.tanh`, i.e. the operating system's C math library, and macOS and Linux differ in
the last bit (tests/platform_probe.py checksums: macOS/arm64 4f1ca5625433c9e4, Linux/
x86_64 0dc3b1ca96cc45a4). A reference made on one platform therefore cannot certify a
scalar run on another. References live in tests/refs/<platform>/; the original files in
tests/refs/ were made on macOS/arm64 and count as that platform.
"""

import platform
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
PLATFORM = f"{sys.platform}-{platform.machine()}"        # e.g. darwin-arm64, linux-x86_64

# How each reference is generated (ea_drift on the unchanged scalar code).
GENERATE = {
    "test_ea_ref.h5": "",
    "test_behaviour_ref.h5": (" --runner simulate.run_batch --config test_behaviour "
                              "--sim-name test_behaviour"),
}


def ref_path(name: str) -> Path:
    """Repository-relative path of reference `name` for this platform."""
    if PLATFORM == "darwin-arm64":
        return Path("tests/refs") / name
    return Path("tests/refs") / PLATFORM / name


def missing_message(name: str) -> str:
    p = ref_path(name)
    return (f"no scalar reference for platform {PLATFORM} ({p} is missing). Generate it once "
            f"on this machine from the unchanged scalar code:\n"
            f"        python -m tests.ea_drift --generate-reference {p}{GENERATE[name]}")
