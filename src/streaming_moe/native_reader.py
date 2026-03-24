from __future__ import annotations

import ctypes
from pathlib import Path


class NativeBuffer:
    def __init__(self, reader: "NativeExpertReader", ptr: int, size: int):
        self._reader = reader
        self.ptr = int(ptr)
        self.size = int(size)
        self._freed = False

    def free(self) -> None:
        if not self._freed and self.ptr:
            self._reader.lib.free_buffer(ctypes.c_void_p(self.ptr))
            self._freed = True
            self.ptr = 0

    def view(self) -> memoryview:
        if self._freed or not self.ptr:
            raise RuntimeError("native buffer has been freed")
        array_type = ctypes.c_uint8 * self.size
        arr = array_type.from_address(self.ptr)
        return memoryview(arr)

    def __del__(self) -> None:
        self.free()


class NativeSlab:
    """Contiguous aligned allocation holding one sub-region per component.

    Created by NativeExpertReader.alloc_slab(); freed via free() / __del__.
    Python owns: offset accounting and component routing.
    C owns:      the aligned allocation (posix_memalign / free).
    """

    def __init__(
        self,
        reader: "NativeExpertReader",
        base_ptr: int,
        offsets: dict[str, int],
        sizes: dict[str, int],
    ):
        self._reader = reader
        self.base_ptr = int(base_ptr)
        self._offsets = offsets   # {component: byte offset from base}
        self._sizes = sizes       # {component: byte size}
        self.total_size: int = sum(sizes.values())
        self._freed = False

    def component_ptr(self, name: str) -> int:
        """Raw pointer (int) to the start of component ``name`` in the slab."""
        return self.base_ptr + self._offsets[name]

    def view(self, name: str) -> memoryview:
        """Zero-copy memoryview of component ``name``."""
        if self._freed or not self.base_ptr:
            raise RuntimeError("slab has been freed")
        offset = self._offsets[name]
        size = self._sizes[name]
        array_type = ctypes.c_uint8 * size
        arr = array_type.from_address(self.base_ptr + offset)
        return memoryview(arr)

    def free(self) -> None:
        if not self._freed and self.base_ptr:
            self._reader.lib.free_slab(ctypes.c_void_p(self.base_ptr))
            self._freed = True
            self.base_ptr = 0

    def __del__(self) -> None:
        self.free()


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

        self.lib.alloc_buffer.argtypes = [ctypes.c_uint64]
        self.lib.alloc_buffer.restype = ctypes.c_void_p
        self.lib.free_buffer.argtypes = [ctypes.c_void_p]
        self.lib.free_buffer.restype = None

        self.lib.alloc_slab.argtypes = [ctypes.c_uint64, ctypes.c_uint64]
        self.lib.alloc_slab.restype = ctypes.c_void_p
        self.lib.free_slab.argtypes = [ctypes.c_void_p]
        self.lib.free_slab.restype = None

        self.lib.read_component_batches_into_slots.argtypes = [
            ctypes.c_int32,
            ctypes.POINTER(ctypes.c_int),
            ctypes.POINTER(ctypes.c_uint64),
            ctypes.POINTER(ctypes.c_uint64),
            ctypes.POINTER(ctypes.c_uint64),
            ctypes.POINTER(ctypes.c_void_p),
            ctypes.POINTER(ctypes.c_int32),
            ctypes.POINTER(ctypes.c_int32),
            ctypes.c_int32,
        ]
        self.lib.read_component_batches_into_slots.restype = ctypes.c_int

        self.lib.copy_component_batches_mmap.argtypes = [
            ctypes.c_int32,                   # num_components
            ctypes.POINTER(ctypes.c_void_p),  # mmap_bases
            ctypes.POINTER(ctypes.c_uint64),  # abs_offsets
            ctypes.POINTER(ctypes.c_uint64),  # expert_strides
            ctypes.POINTER(ctypes.c_uint64),  # expert_sizes
            ctypes.POINTER(ctypes.c_uint64),  # component_output_offsets
            ctypes.POINTER(ctypes.c_int32),   # expert_indices
            ctypes.c_int32,                   # num_experts
            ctypes.POINTER(ctypes.c_uint8),   # out_buffer
        ]
        self.lib.copy_component_batches_mmap.restype = ctypes.c_int

        self.lib.copy_experts_multi.argtypes = [
            ctypes.c_int32,                   # num_components
            ctypes.POINTER(ctypes.c_void_p),  # src_ptrs
            ctypes.POINTER(ctypes.c_void_p),  # dst_ptrs
            ctypes.POINTER(ctypes.c_uint64),  # expert_sizes
            ctypes.POINTER(ctypes.c_int32),   # src_slots
            ctypes.POINTER(ctypes.c_int32),   # dst_slots
            ctypes.c_int32,                   # num_copies
        ]
        self.lib.copy_experts_multi.restype = ctypes.c_int

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

    def alloc_buffer(self, size: int) -> NativeBuffer:
        ptr = self.lib.alloc_buffer(size)
        if not ptr:
            raise MemoryError(f"native alloc_buffer failed for size={size}")
        return NativeBuffer(self, int(ptr), int(size))

    def alloc_slab(
        self,
        components: list[str],
        sizes: dict[str, int],
        alignment: int = 64,
    ) -> NativeSlab:
        """Allocate one aligned slab covering all components.

        ``sizes`` maps component name → byte count for that component.
        Components are laid out contiguously in the order given by ``components``.
        """
        total = sum(sizes[c] for c in components)
        ptr = self.lib.alloc_slab(ctypes.c_uint64(total), ctypes.c_uint64(alignment))
        if not ptr:
            raise MemoryError(f"alloc_slab failed for total={total} alignment={alignment}")
        offsets: dict[str, int] = {}
        cursor = 0
        for c in components:
            offsets[c] = cursor
            cursor += sizes[c]
        return NativeSlab(self, int(ptr), offsets, dict(sizes))

    def read_component_batches_into_slots(
        self,
        specs: list[tuple[str, int, int, int, int]],
        expert_indices: list[int],
        slot_indices: list[int],
        out_buffers: dict[str, "NativeBuffer | int"],
    ) -> None:
        """Write experts into specific slots of output buffers.

        ``out_buffers`` maps component name to either a NativeBuffer or a raw
        integer pointer (e.g. NativeSlab.component_ptr(name)).
        """
        component_count = len(specs)
        expert_count = len(expert_indices)
        if component_count <= 0 or expert_count <= 0:
            return
        if len(slot_indices) != expert_count:
            raise ValueError("slot_indices length must match expert_indices length")

        names = [name for name, *_ in specs]
        missing = [name for name in names if name not in out_buffers]
        if missing:
            raise KeyError(f"missing output buffers for components: {missing}")

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
        # Accept both NativeBuffer (has .ptr) and raw int pointer.
        out_ptrs = (ctypes.c_void_p * component_count)(
            *[
                ctypes.c_void_p(
                    out_buffers[name].ptr
                    if hasattr(out_buffers[name], "ptr")
                    else int(out_buffers[name])
                )
                for name in names
            ]
        )
        idx_arr = (ctypes.c_int32 * expert_count)(*expert_indices)
        slot_arr = (ctypes.c_int32 * expert_count)(*slot_indices)
        rc = self.lib.read_component_batches_into_slots(
            component_count,
            fds,
            abs_offsets,
            expert_strides,
            expert_sizes,
            out_ptrs,
            idx_arr,
            slot_arr,
            expert_count,
        )
        if rc != 0:
            raise RuntimeError(f"native read_component_batches_into_slots failed: {rc}")

    def copy_component_batches_mmap(
        self,
        specs: list[tuple[str, int, int, int, int]],
        expert_indices: list[int],
    ) -> dict[str, memoryview]:
        """Copy expert batches from mmap regions via parallel memcpy.

        specs: list of (name, mmap_base_ptr, abs_offset, expert_stride, expert_size)
        Returns dict[component_name -> memoryview] into a contiguous output buffer.
        """
        component_count = len(specs)
        expert_count = len(expert_indices)
        if component_count <= 0 or expert_count <= 0:
            return {}

        names = [name for name, *_ in specs]
        sizes = [expert_size * expert_count for _, _, _, _, expert_size in specs]
        out_offsets = []
        cursor = 0
        for size in sizes:
            out_offsets.append(cursor)
            cursor += size

        out = (ctypes.c_uint8 * cursor)()
        mmap_bases = (ctypes.c_void_p * component_count)(
            *[mmap_base for _, mmap_base, _, _, _ in specs]
        )
        abs_offsets = (ctypes.c_uint64 * component_count)(
            *[abs_offset for _, _, abs_offset, _, _ in specs]
        )
        expert_strides = (ctypes.c_uint64 * component_count)(
            *[expert_stride for _, _, _, expert_stride, _ in specs]
        )
        expert_sizes = (ctypes.c_uint64 * component_count)(
            *[expert_size for _, _, _, _, expert_size in specs]
        )
        component_offsets = (ctypes.c_uint64 * component_count)(*out_offsets)
        idx_arr = (ctypes.c_int32 * expert_count)(*expert_indices)
        rc = self.lib.copy_component_batches_mmap(
            component_count,
            mmap_bases,
            abs_offsets,
            expert_strides,
            expert_sizes,
            component_offsets,
            idx_arr,
            expert_count,
            out,
        )
        if rc != 0:
            raise RuntimeError(f"copy_component_batches_mmap failed: {rc}")

        blob = memoryview(out)
        return {
            name: blob[offset : offset + size]
            for name, offset, size in zip(names, out_offsets, sizes)
        }

    def copy_experts_multi(
        self,
        *,
        src_ptrs: list[int],
        dst_ptrs: list[int],
        expert_sizes: list[int],
        src_slots: list[int],
        dst_slots: list[int],
    ) -> None:
        """Copy hit experts across all components in a single dispatch_apply call.

        src_ptrs[c] / dst_ptrs[c] — base pointer for component c (from NativeSlab.component_ptr).
        expert_sizes[c]            — bytes per expert for component c.
        src_slots[i] / dst_slots[i] — slot index in source / destination for expert i.
        Total tasks: len(src_ptrs) * len(src_slots).
        """
        num_components = len(src_ptrs)
        num_copies = len(src_slots)
        if num_components <= 0 or num_copies <= 0:
            return
        c_src = (ctypes.c_void_p * num_components)(*src_ptrs)
        c_dst = (ctypes.c_void_p * num_components)(*dst_ptrs)
        c_esizes = (ctypes.c_uint64 * num_components)(*expert_sizes)
        c_src_slots = (ctypes.c_int32 * num_copies)(*src_slots)
        c_dst_slots = (ctypes.c_int32 * num_copies)(*dst_slots)
        rc = self.lib.copy_experts_multi(
            ctypes.c_int32(num_components),
            c_src,
            c_dst,
            c_esizes,
            c_src_slots,
            c_dst_slots,
            ctypes.c_int32(num_copies),
        )
        if rc != 0:
            raise RuntimeError(f"copy_experts_multi failed: {rc}")
