"""Samples the graphics card and system memory while a job runs.

What it reads:
- the whole card's dedicated memory (NVML: what nvidia-smi shows), every 0.1 s
- system RAM in use and this process's own memory (psutil)
- on Windows, the card's *shared* memory (typeperf): when the 8 GB are full the
  NVIDIA driver moves the overflow to system RAM instead of failing, and the job
  runs many times slower. Growth in shared memory during a job is how we see it.
"""

from __future__ import annotations

import os
import subprocess
import sys
import threading
import time

import psutil

try:
    import pynvml

    pynvml.nvmlInit()
    _HANDLE = pynvml.nvmlDeviceGetHandleByIndex(int(os.environ.get("EIGHTGB_GPU", "0")))
except Exception:  # no NVIDIA card or driver: the readouts stay empty
    pynvml = None
    _HANDLE = None

MB = 2**20


def gpu_name() -> str:
    if _HANDLE is None:
        return "no NVIDIA GPU found"
    name = pynvml.nvmlDeviceGetName(_HANDLE)
    return name.decode() if isinstance(name, bytes) else name


def driver_version() -> str:
    if _HANDLE is None:
        return ""
    v = pynvml.nvmlSystemGetDriverVersion()
    return v.decode() if isinstance(v, bytes) else v


def card_mb() -> tuple[int, int]:
    """(dedicated memory in use on the whole card, the card's total), in MB."""
    if _HANDLE is None:
        return 0, 0
    m = pynvml.nvmlDeviceGetMemoryInfo(_HANDLE)
    return int(m.used / MB), int(m.total / MB)


def gpu_util() -> int:
    if _HANDLE is None:
        return 0
    try:
        return int(pynvml.nvmlDeviceGetUtilizationRates(_HANDLE).gpu)
    except Exception:
        return 0


def ram_mb() -> tuple[int, int]:
    """(system RAM in use, total), in MB."""
    v = psutil.virtual_memory()
    return int((v.total - v.available) / MB), int(v.total / MB)


def commit_mb() -> tuple[int, int]:
    """(committed memory, commit limit) in MB: RAM plus page file. Windows only, else zeros."""
    if sys.platform != "win32":
        return 0, 0
    import ctypes

    class PERF(ctypes.Structure):
        _fields_ = [("cb", ctypes.c_ulong)] + [(n, ctypes.c_size_t) for n in (
            "CommitTotal", "CommitLimit", "CommitPeak", "PhysicalTotal", "PhysicalAvailable",
            "SystemCache", "KernelTotal", "KernelPaged", "KernelNonpaged", "PageSize")] + [
            ("HandleCount", ctypes.c_ulong), ("ProcessCount", ctypes.c_ulong), ("ThreadCount", ctypes.c_ulong)]

    p = PERF()
    p.cb = ctypes.sizeof(p)
    if not ctypes.windll.psapi.GetPerformanceInfo(ctypes.byref(p), p.cb):
        return 0, 0
    return int(p.CommitTotal * p.PageSize / MB), int(p.CommitLimit * p.PageSize / MB)


_PROC = psutil.Process()


def process_mb() -> int:
    """This process's memory in RAM right now (working set), in MB."""
    return int(_PROC.memory_info().rss / MB)


class SharedGpuMemory:
    """The card's shared (system RAM) memory, read once a second from typeperf. Windows only."""

    def __init__(self):
        self.mb = None
        self._proc = None
        if sys.platform != "win32":
            return
        try:
            self._proc = subprocess.Popen(
                # -sc bounds the run: if this program is killed hard, the helper still ends by itself.
                ["typeperf", r"\GPU Adapter Memory(*)\Shared Usage", "-si", "1", "-sc", "7200"],
                stdout=subprocess.PIPE, stderr=subprocess.DEVNULL, text=True,
                creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0))
        except OSError:
            return
        threading.Thread(target=self._read, daemon=True).start()

    def _read(self):
        for line in self._proc.stdout:
            cells = [c.strip('" \n') for c in line.split(",")]
            if len(cells) < 2:
                continue
            try:
                # one column per adapter; the NVIDIA card is the one that holds memory
                self.mb = int(max(float(c) for c in cells[1:] if c) / MB)
            except ValueError:
                continue  # the header line

    def close(self):
        if self._proc is not None:
            self._proc.kill()


_SHARED: SharedGpuMemory | None = None
_SHARED_LOCK = threading.Lock()


def shared() -> SharedGpuMemory:
    """The one shared-memory reader for this process, started on first use (never at import:
    a program that only imports this module must not leave a helper process behind)."""
    global _SHARED
    with _SHARED_LOCK:
        if _SHARED is None:
            import atexit

            _SHARED = SharedGpuMemory()
            atexit.register(_SHARED.close)
            deadline = time.perf_counter() + 2.5  # typeperf prints once a second
            while _SHARED._proc is not None and _SHARED.mb is None and time.perf_counter() < deadline:
                time.sleep(0.1)
        return _SHARED


def snapshot() -> dict:
    used, total = card_mb()
    ram, ram_total = ram_mb()
    return {"vram_mb": used, "vram_total_mb": total, "gpu_util": gpu_util(), "shared_mb": shared().mb,
            "ram_mb": ram, "ram_total_mb": ram_total, "process_mb": process_mb()}


class Peak:
    """Use as `with Peak() as p:`; afterwards p.result has the peaks seen while the block ran."""

    def __init__(self, interval: float = 0.1):
        self.interval = interval
        self.result: dict = {}

    def __enter__(self):
        self._stop = threading.Event()
        first = snapshot()
        self._start = first
        self._peak = dict(first)
        self._low_shared = first["shared_mb"]
        self._commit = commit_mb()[0]
        self._t0 = time.perf_counter()
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()
        return self

    def _run(self):
        n = 0
        while not self._stop.wait(self.interval):
            s = snapshot()
            for k in ("vram_mb", "ram_mb", "process_mb", "gpu_util"):
                self._peak[k] = max(self._peak[k], s[k])
            if s["shared_mb"] is not None:
                self._peak["shared_mb"] = max(self._peak["shared_mb"] or 0, s["shared_mb"])
            n += 1
            if n % 10 == 0:
                self._commit = max(self._commit, commit_mb()[0])

    def __exit__(self, *exc):
        self._stop.set()
        self._thread.join()
        start, peak = self._start, self._peak
        shared_growth = None
        if start["shared_mb"] is not None and peak["shared_mb"] is not None:
            shared_growth = peak["shared_mb"] - start["shared_mb"]
        self.result = {
            "seconds": round(time.perf_counter() - self._t0, 2),
            "vram_start_mb": start["vram_mb"], "vram_peak_mb": peak["vram_mb"], "vram_total_mb": start["vram_total_mb"],
            "shared_start_mb": start["shared_mb"], "shared_peak_mb": peak["shared_mb"], "shared_growth_mb": shared_growth,
            "ram_start_mb": start["ram_mb"], "ram_peak_mb": peak["ram_mb"], "ram_total_mb": start["ram_total_mb"],
            "process_peak_mb": peak["process_mb"], "commit_peak_mb": self._commit,
        }
        return False
