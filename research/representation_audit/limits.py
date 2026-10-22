"""OS memory containment applied before importing the model runtime."""
import ctypes
import os


def establish(memory_bytes, process_handle=None):
    if os.name != "nt":
        if process_handle is not None:
            raise OSError("Parent-owned audit containment currently requires Windows")
        import resource
        resource.setrlimit(resource.RLIMIT_AS, (memory_bytes, memory_bytes))
        return {"backend": "rlimit_as", "memory_bytes": memory_bytes}
    from ctypes import wintypes

    class BASIC(ctypes.Structure):
        _fields_ = [("per_process", ctypes.c_int64), ("per_job", ctypes.c_int64),
                    ("flags", wintypes.DWORD), ("min_ws", ctypes.c_size_t), ("max_ws", ctypes.c_size_t),
                    ("active_limit", wintypes.DWORD), ("affinity", ctypes.c_size_t),
                    ("priority", wintypes.DWORD), ("scheduling", wintypes.DWORD)]

    class IO(ctypes.Structure):
        _fields_ = [(name, ctypes.c_uint64) for name in ("read_ops", "write_ops", "other_ops", "read_bytes", "write_bytes", "other_bytes")]

    class EXTENDED(ctypes.Structure):
        _fields_ = [("basic", BASIC), ("io", IO), ("process_memory", ctypes.c_size_t),
                    ("job_memory", ctypes.c_size_t), ("peak_process", ctypes.c_size_t), ("peak_job", ctypes.c_size_t)]

    kernel = ctypes.WinDLL("kernel32", use_last_error=True)
    kernel.CreateJobObjectW.argtypes = [ctypes.c_void_p, wintypes.LPCWSTR]
    kernel.CreateJobObjectW.restype = wintypes.HANDLE
    kernel.SetInformationJobObject.argtypes = [wintypes.HANDLE, ctypes.c_int, ctypes.c_void_p, wintypes.DWORD]
    kernel.AssignProcessToJobObject.argtypes = [wintypes.HANDLE, wintypes.HANDLE]
    kernel.GetCurrentProcess.restype = wintypes.HANDLE
    kernel.GetProcessAffinityMask.argtypes = [wintypes.HANDLE, ctypes.POINTER(ctypes.c_size_t), ctypes.POINTER(ctypes.c_size_t)]
    kernel.CloseHandle.argtypes = [wintypes.HANDLE]
    handle = kernel.CreateJobObjectW(None, None)
    process_mask, system_mask = ctypes.c_size_t(), ctypes.c_size_t()
    process = process_handle if process_handle is not None else kernel.GetCurrentProcess()
    if not handle or not kernel.GetProcessAffinityMask(process, ctypes.byref(process_mask), ctypes.byref(system_mask)):
        raise OSError(ctypes.get_last_error(), "Cannot establish audit job/affinity")
    affinity = process_mask.value & -process_mask.value
    limits = EXTENDED()
    limits.basic.flags = 0x2000 | 0x100 | 0x10  # kill on close, process memory, CPU affinity
    limits.basic.affinity = affinity
    limits.process_memory = memory_bytes
    if (not kernel.SetInformationJobObject(handle, 9, ctypes.byref(limits), ctypes.sizeof(limits))
            or not kernel.AssignProcessToJobObject(handle, process)):
        kernel.CloseHandle(handle)
        raise OSError(ctypes.get_last_error(), "Cannot apply audit memory/job limit")
    # Do not close: the process owns the job for its whole lifetime.
    return {"backend": "windows_job", "memory_bytes": memory_bytes, "affinity": affinity, "handle": int(handle)}


def close(held):
    if held and "handle" in held:
        from ctypes import wintypes
        kernel = ctypes.WinDLL("kernel32", use_last_error=True)
        kernel.CloseHandle.argtypes = [wintypes.HANDLE]
        kernel.CloseHandle(held["handle"])
