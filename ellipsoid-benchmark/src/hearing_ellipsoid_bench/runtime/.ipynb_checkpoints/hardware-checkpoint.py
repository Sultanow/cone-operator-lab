# src/hearing_ellipsoid_bench/runtime/hardware.py

from __future__ import annotations

import os
import platform
import subprocess
from pathlib import Path


def get_hardware_specs() -> dict:
    specs = {
        "platform": platform.platform(),
        "python": platform.python_version(),
        "cwd": str(Path.cwd()),
    }

    try:
        import psutil

        specs["logical_cpus"] = psutil.cpu_count(logical=True)
        specs["physical_cpus"] = psutil.cpu_count(logical=False)
        specs["ram_gb"] = round(psutil.virtual_memory().total / 1024**3, 2)
    except Exception:
        specs["logical_cpus"] = os.cpu_count()
        specs["physical_cpus"] = None
        specs["ram_gb"] = None

    try:
        specs["cpu_model"] = subprocess.check_output(
            "lscpu | grep 'Model name' | cut -d ':' -f2 | xargs",
            shell=True,
            text=True,
        ).strip()
    except Exception:
        specs["cpu_model"] = "unknown"

    try:
        specs["kernel"] = subprocess.check_output(
            "uname -a",
            shell=True,
            text=True,
        ).strip()
    except Exception:
        specs["kernel"] = "unknown"

    try:
        specs["available_logical_cpus"] = int(
            subprocess.check_output("nproc", shell=True, text=True).strip()
        )
        specs["node_logical_cpus"] = int(
            subprocess.check_output("nproc --all", shell=True, text=True).strip()
        )
    except Exception:
        pass

    return specs