from __future__ import annotations

import ctypes
import os
import subprocess

import cpuinfo
import numpy as np
import torch
from pelutils import c_ptr


cpu = torch.device("cpu")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def ptr(arr: torch.Tensor) -> ctypes:
    return ctypes.c_void_p(arr.data_ptr())

def tensor_size(x: np.ndarray | torch.Tensor) -> int:
    if isinstance(x, np.ndarray):
        return c_ptr(x)
    return x.element_size() * x.numel()

class HardwareInfo:

    cpu     = cpuinfo.get_cpu_info()["brand_raw"]
    sockets = int(subprocess.check_output('cat /proc/cpuinfo | grep "physical id" | sort -u | wc -l', shell=True))
    cores   = os.cpu_count()
    gpu     = torch.cuda.get_device_name(device) if torch.cuda.is_available() else None
