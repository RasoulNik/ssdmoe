from __future__ import annotations

import ctypes
from pathlib import Path


class NativeExpertReader:
    def __init__(self, dylib_path: Path):
        dylib_path = Path(dylib_path).expanduser().resolve()
        self.lib = ctypes.CDLL(str(dylib_path))
        self.lib.read_component_batch.argtypes = [
            ctypes.c_int,
            ctypes.c_uint64,
            ctypes.c_uint64,
            ctypes.c_uint64,
            ctypes.POINTER(ctypes.c_int32),
            ctypes.c_int32,
            ctypes.POINTER(ctypes.c_uint8),
        ]
        self.lib.read_component_batch.restype = ctypes.c_int
        self.lib.read_component_batches.argtypes = [
            ctypes.c_int32,
            ctypes.POINTER(ctypes.c_int),
            ctypes.POINTER(ctypes.c_uint64),
            ctypes.POINTER(ctypes.c_uint64),
            ctypes.POINTER(ctypes.c_uint64),
            ctypes.POINTER(ctypes.c_uint64),
            ctypes.POINTER(ctypes.c_int32),
            ctypes.c_int32,
            ctypes.POINTER(ctypes.c_uint8),
        ]
        self.lib.read_component_batches.restype = ctypes.c_int

    def read_component_batch(
        self,
        fd: int,
        abs_offset: int,
        expert_stride: int,
        expert_size: int,
        expert_indices: list[int],
    ) -> memoryview:
        count = len(expert_indices)
        out_size = expert_size * count
        out = (ctypes.c_uint8 * out_size)()
        idx_arr = (ctypes.c_int32 * count)(*expert_indices)
        rc = self.lib.read_component_batch(
            fd,
            abs_offset,
            expert_stride,
            expert_size,
            idx_arr,
            count,
            out,
        )
        if rc != 0:
            raise RuntimeError(f"native read_component_batch failed: {rc}")
        return memoryview(out)

    def read_component_batches(
        self,
        specs: list[tuple[str, int, int, int, int]],
        expert_indices: list[int],
    ) -> dict[str, memoryview]:
        component_count = len(specs)
        expert_count = len(expert_indices)
        if component_count <= 0 or expert_count <= 0:
            return {}

        names = [name for name, *_ in specs]
        sizes = [expert_size * expert_count for _, _, _, _, expert_size in specs]
        offsets = []
        cursor = 0
        for size in sizes:
            offsets.append(cursor)
            cursor += size

        out = (ctypes.c_uint8 * cursor)()
        fds = (ctypes.c_int * component_count)(*[fd for _, fd, _, _, _ in specs])
        abs_offsets = (ctypes.c_uint64 * component_count)(
            *[abs_offset for _, _, abs_offset, _, _ in specs]
        )
        expert_strides = (ctypes.c_uint64 * component_count)(
            *[expert_stride for _, _, _, expert_stride, _ in specs]
        )
        expert_sizes = (ctypes.c_uint64 * component_count)(
            *[expert_size for _, _, _, _, expert_size in specs]
        )
        component_offsets = (ctypes.c_uint64 * component_count)(*offsets)
        idx_arr = (ctypes.c_int32 * expert_count)(*expert_indices)
        rc = self.lib.read_component_batches(
            component_count,
            fds,
            abs_offsets,
            expert_strides,
            expert_sizes,
            component_offsets,
            idx_arr,
            expert_count,
            out,
        )
        if rc != 0:
            raise RuntimeError(f"native read_component_batches failed: {rc}")

        blob = memoryview(out)
        result = {}
        for name, offset, size in zip(names, offsets, sizes):
            result[name] = blob[offset : offset + size]
        return result
