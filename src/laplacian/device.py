"""Select the RTX 5070 by name and refuse to cross the 11.2 GB guard."""

from __future__ import annotations

import subprocess

GUARD_MIB = 11.2 * 1024


class VramGuard(RuntimeError):
    pass


def pick_5070():
    import torch

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is not available")
    for index in range(torch.cuda.device_count()):
        name = torch.cuda.get_device_name(index)
        if "5070" in name:
            return torch.device(f"cuda:{index}"), name
    names = [torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())]
    raise RuntimeError(f"RTX 5070 not found among {names}")


def memory_mib(name_substring="5070"):
    """Device memory used, from nvidia-smi, for the named GPU."""
    query = subprocess.run(
        ["nvidia-smi", "--query-gpu=name,memory.used", "--format=csv,noheader,nounits"],
        check=True,
        capture_output=True,
        text=True,
    )
    for line in query.stdout.splitlines():
        gpu_name, used = [part.strip() for part in line.split(",")]
        if name_substring in gpu_name:
            return float(used)
    raise RuntimeError("nvidia-smi did not list the requested GPU")


def enforce_guard(name_substring="5070"):
    used = memory_mib(name_substring)
    if used > GUARD_MIB:
        raise VramGuard(f"{name_substring} memory {used:.0f} MiB exceeds {GUARD_MIB:.0f} MiB")
    return used
