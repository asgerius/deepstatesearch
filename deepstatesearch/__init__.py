from __future__ import annotations

import ctypes
import os
import subprocess

import cpuinfo
import numpy as np
import torch
from pelutils import TickTock, TickTockException


def _reset(self: TickTock):
    """ Stops all timing and profiling. """
    if self._profile_stack:
        raise TickTockException("Cannot reset TickTock while profiling is active")
    self.__init__()

TickTock.reset = _reset

cpu = torch.device("cpu")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

LIBDSS = ctypes.cdll.LoadLibrary("lib/libdss.so")

LIBDSS.heap_extract_min.restype = ctypes.c_size_t
LIBDSS.astar_init.restype = ctypes.c_void_p
LIBDSS.astar_frontier_ptr.restype = ctypes.c_void_p
LIBDSS.astar_longest_path.restype = ctypes.c_size_t
LIBDSS.astar_retrace_path.restype = ctypes.c_size_t
LIBDSS.astar_num_states.restype = ctypes.c_size_t
LIBDSS.values_node_map_from_states.restype = ctypes.c_void_p

LIBDSS.astar_add_initial_state.argtypes = ctypes.c_float, ctypes.c_void_p, ctypes.c_void_p

def ptr(arr: torch.Tensor) -> ctypes:
    return ctypes.c_void_p(arr.data_ptr())

def tensor_size(x: np.ndarray | torch.Tensor) -> int:
    if isinstance(x, np.ndarray):
        x = torch.from_numpy(x)
    return x.element_size() * x.numel()

def unique(x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """ Returns a tensor with the indices of unique elements in x
    as well as a tensor of inverse indices. """
    index = torch.empty(len(x), dtype=torch.long)
    inverse = torch.empty(len(x), dtype=torch.long)

    numel_per_element = np.cumprod((*x.shape[1:], 1))[-1]

    num_unique = LIBDSS.unique(
        ptr(x),
        ctypes.c_size_t(len(x)),
        ctypes.c_size_t(x.element_size() * numel_per_element),
        ptr(index),
        ptr(inverse),
    )

    index = index[:num_unique]

    return index, inverse

class HardwareInfo:

    cpu     = cpuinfo.get_cpu_info()["brand_raw"]
    sockets = int(subprocess.check_output('cat /proc/cpuinfo | grep "physical id" | sort -u | wc -l', shell=True))
    cores   = os.cpu_count()
    gpu     = torch.cuda.get_device_name(device) if torch.cuda.is_available() else None
