"""Which devices the tests run on: the CPU plus every accelerator this machine has.

The production machines are NVIDIA/CUDA; development happened on Apple MPS. Every
device-parametrised test uses these helpers, so the same suite exercises whichever of
the two is present (plan_evotorch.md Step 8, CUDA validation).
"""

import torch


def accelerators():
    """Available GPU backends, e.g. ["cuda"] or ["mps"] (or both / neither)."""
    out = []
    if torch.cuda.is_available():
        out.append("cuda")
    if torch.backends.mps.is_available():
        out.append("mps")
    return out


def all_devices():
    return ["cpu"] + accelerators()


def sync(device):
    device = torch.device(device)
    if device.type == "cuda":
        torch.cuda.synchronize()
    elif device.type == "mps":
        torch.mps.synchronize()


def reset_peak(device):
    """Start a fresh peak-memory measurement (CUDA has exact peak counters)."""
    if torch.device(device).type == "cuda":
        torch.cuda.reset_peak_memory_stats()


def allocator_bytes(device):
    """Memory the PyTorch allocator holds from the device: live tensors + cache."""
    device = torch.device(device)
    if device.type == "cuda":
        return int(torch.cuda.memory_reserved())
    if device.type == "mps":
        return int(torch.mps.driver_allocated_memory())
    raise ValueError(f"no allocator statistics for {device}")


def live_bytes(device):
    device = torch.device(device)
    if device.type == "cuda":
        return int(torch.cuda.memory_allocated())
    if device.type == "mps":
        return int(torch.mps.current_allocated_memory())
    raise ValueError(f"no allocator statistics for {device}")


def peak_allocator_bytes(device):
    """Peak since `reset_peak` on CUDA; on MPS (no peak counter) the current value --
    callers sample it during the run instead."""
    device = torch.device(device)
    if device.type == "cuda":
        return int(torch.cuda.max_memory_reserved())
    return allocator_bytes(device)


# An independent copy of mvb_torch.generation.physical_ram, so the test of the width
# rule's memory source is not circular.
def physical_ram() -> int:
    """Total physical RAM in bytes, on Linux, macOS and Windows."""
    import os
    import sys
    if hasattr(os, "sysconf") and "SC_PHYS_PAGES" in os.sysconf_names:
        return int(os.sysconf("SC_PAGE_SIZE") * os.sysconf("SC_PHYS_PAGES"))
    if sys.platform == "win32":                 # no os.sysconf on Windows
        import ctypes

        class _MemoryStatusEx(ctypes.Structure):
            _fields_ = [("dwLength", ctypes.c_ulong), ("dwMemoryLoad", ctypes.c_ulong),
                        ("ullTotalPhys", ctypes.c_ulonglong),
                        ("ullAvailPhys", ctypes.c_ulonglong),
                        ("ullTotalPageFile", ctypes.c_ulonglong),
                        ("ullAvailPageFile", ctypes.c_ulonglong),
                        ("ullTotalVirtual", ctypes.c_ulonglong),
                        ("ullAvailVirtual", ctypes.c_ulonglong),
                        ("ullAvailExtendedVirtual", ctypes.c_ulonglong)]
        status = _MemoryStatusEx()
        status.dwLength = ctypes.sizeof(status)
        if not ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(status)):
            raise OSError("GlobalMemoryStatusEx failed")
        return int(status.ullTotalPhys)
    raise RuntimeError(f"no physical-memory query for platform {sys.platform!r}")
