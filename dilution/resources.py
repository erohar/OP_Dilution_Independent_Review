from __future__ import annotations

import ctypes
import logging
import os
import subprocess
import sys
from dataclasses import dataclass
from typing import Optional

log = logging.getLogger(__name__)

GB = 1024 ** 3
_WORKER_BASE = 300 * 1024 ** 2
_JOB_BYTES = 3 * 1024 ** 3 // 2

@dataclass
class Resources:
    threads: int
    total_ram: Optional[int]
    free_ram: Optional[int]

    def describe(self) -> str:
        ram = ("unknown RAM" if self.total_ram is None else
               f"{self.total_ram / GB:.1f} GB RAM"
               + ("" if self.free_ram is None else f" ({self.free_ram / GB:.1f} GB free)"))
        return f"{self.threads} CPU threads, {ram}"

def cpu_threads() -> int:
    try:
        return len(os.sched_getaffinity(0))
    except (AttributeError, OSError):
        return os.cpu_count() or 1

def memory() -> tuple[Optional[int], Optional[int]]:
    try:
        import psutil
        vm = psutil.virtual_memory()
        return int(vm.total), int(vm.available)
    except ImportError:
        pass
    except Exception as e:
        log.debug("psutil could not read memory: %s", e)
    try:
        if sys.platform == "win32":
            class MEMORYSTATUSEX(ctypes.Structure):
                _fields_ = [("dwLength", ctypes.c_ulong),
                            ("dwMemoryLoad", ctypes.c_ulong),
                            ("ullTotalPhys", ctypes.c_ulonglong),
                            ("ullAvailPhys", ctypes.c_ulonglong),
                            ("ullTotalPageFile", ctypes.c_ulonglong),
                            ("ullAvailPageFile", ctypes.c_ulonglong),
                            ("ullTotalVirtual", ctypes.c_ulonglong),
                            ("ullAvailVirtual", ctypes.c_ulonglong),
                            ("ullAvailExtendedVirtual", ctypes.c_ulonglong)]
            st = MEMORYSTATUSEX()
            st.dwLength = ctypes.sizeof(MEMORYSTATUSEX)
            ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(st))
            return int(st.ullTotalPhys), int(st.ullAvailPhys)
        if os.path.exists("/proc/meminfo"):
            info = {}
            with open("/proc/meminfo") as f:
                for line in f:
                    k, v = line.split(":", 1)
                    info[k] = int(v.split()[0]) * 1024
            return info.get("MemTotal"), info.get("MemAvailable", info.get("MemFree"))
        if sys.platform == "darwin":
            total = int(subprocess.check_output(["sysctl", "-n", "hw.memsize"]))
            return total, None
    except Exception as e:
        log.debug("Could not read memory from the operating system: %s", e)
    return None, None

def detect() -> Resources:
    total, free = memory()
    return Resources(cpu_threads(), total, free)

def recommend_threads(res: Resources, model_bytes: Optional[int] = None) -> int:
    n = max(1, res.threads)
    if res.free_ram and model_bytes:
        main = model_bytes // 5
        per_worker = _WORKER_BASE + _JOB_BYTES
        budget = 0.9 * res.free_ram - main
        n = max(1, min(n, int(budget // per_worker)))
    return n
