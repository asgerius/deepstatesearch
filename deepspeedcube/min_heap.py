from __future__ import annotations

import ctypes

import torch

from deepspeedcube import LIBDSC, ptr


class MinHeap:

    _init_heap_size = 10_000

    def __init__(self, array_shape: torch.Size, dtype: torch.dtype):

        # Heap size including null element
        self._num_elems = 1

        self._array_shape = array_shape
        self._dtype = dtype

        self._keys = torch.empty(self._init_heap_size)
        self._data = torch.empty(
            (self._init_heap_size, *array_shape),
            dtype=dtype,
        )

        self._heap_ptr = ctypes.c_void_p(LIBDSC.heap_alloc(
            ptr(self._keys),
            ptr(self._data),
            torch.tensor([], dtype=dtype).element_size() * array_shape.numel(),
        ))

    def min(self) -> tuple[float, torch.Tensor]:

        if len(self) == 0:
            raise ValueError("Heap is empty")

        return self._keys[1].item(), self._data[1]

    def extract_min(self) -> tuple[float, torch.Tensor]:

        if self._num_elems == 1:
            raise ValueError("Heap is empty")

        self._num_elems -= 1

        key = torch.empty(1)
        data = torch.empty(self._array_shape, dtype=self._dtype)

        LIBDSC.heap_extract_min(
            self._heap_ptr,
            1,
            ptr(key),
            ptr(data),
        )

        return key[0].item(), data

    def extract_min_multiple(self, n: int) -> tuple[torch.FloatTensor, torch.Tensor]:

        n = min(n, len(self))
        if n == 0:
            raise ValueError("Heap is empty")

        self._num_elems -= n

        keys = torch.empty(n)
        data = torch.empty(
            (n, *self._array_shape),
            dtype=self._dtype,
        )

        LIBDSC.heap_extract_min(
            self._heap_ptr,
            n,
            ptr(keys),
            ptr(data),
        )

        return keys, data

    def insert(self, key: float, array: torch.Tensor):

        self._num_elems += 1
        while len(self._keys) < self._num_elems:
            self._expand_heap()

        key = torch.tensor([key], dtype=torch.float)

        LIBDSC.heap_insert(
            self._heap_ptr,
            1,
            ptr(key),
            ptr(array),
        )

    def insert_multiple(self, keys: torch.FloatTensor, arrays: torch.Tensor):

        self._num_elems += len(keys)
        while len(self._keys) < self._num_elems:
            self._expand_heap()

        LIBDSC.heap_insert(
            self._heap_ptr,
            len(keys),
            ptr(keys),
            ptr(arrays),
        )

    def _expand_heap(self):
        """ Doubles the amount of memory allocated to the heap. """

        self._keys = torch.concat((self._keys, torch.empty_like(self._keys)))
        self._data = torch.concat((self._data, torch.empty_like(self._data)))

        LIBDSC.heap_update_ptrs(self._heap_ptr, ptr(self._keys), ptr(self._data))

    def __len__(self) -> int:
        """ Heap size excluding null element """

        return self._num_elems - 1

    def __del__(self):

        LIBDSC.heap_free(self._heap_ptr)
