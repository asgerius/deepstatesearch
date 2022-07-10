from __future__ import annotations

import ctypes

import torch

from deepspeedcube import LIBDSC, ptr, tensor_size


class MinHeap:

    _init_heap_size = 10_000

    def __init__(self, sample_array: torch.Tensor):

        # Heap size including null element
        self._num_elems = 1

        self._sample_array = sample_array.clone()

        self._keys = torch.empty(self._init_heap_size)
        self._data = torch.empty(
            (self._init_heap_size, *sample_array.shape),
            dtype=sample_array.dtype,
        )

        self._heap_ptr = ctypes.c_void_p(LIBDSC.heap_alloc(
            ptr(self._keys),
            ptr(self._data),
            tensor_size(sample_array),
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
        data = torch.empty_like(self._sample_array)

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
            (n, *self._sample_array.shape),
            dtype=self._sample_array.shape,
        )

        LIBDSC.heap_extract_min(
            self._heap_ptr,
            n,
            ptr(keys),
            ptr(data),
        )

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
