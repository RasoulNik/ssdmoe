from __future__ import annotations

import ctypes
import json
import mmap
import os
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import fcntl

from .native_reader import NativeExpertReader

class ExpertStore:
    """Read routed experts directly from original safetensors shards.

    This mirrors the flash-moe design choice of using explicit byte offsets and
    `pread()` instead of materializing a second packed copy on disk.
    """

    def __init__(
        self,
        index_path: Path,
        use_nocache: bool = False,
        native_reader_path: Path | None = None,
        resident_components: dict[int, dict[str, object]] | None = None,
        component_workers: int = 3,
    ):
        index_path = Path(index_path).expanduser().resolve()
        with index_path.open() as f:
            self.index = json.load(f)

        self.model_path = Path(self.index["model_path"]).expanduser().resolve()
        self.expert_reads = self.index["expert_reads"]
        self._fds: dict[str, int] = {}
        self._mmaps: dict[str, mmap.mmap] = {}
        self._mmap_bases: dict[str, int] = {}  # file_name -> base address of mmap region
        self.use_nocache = use_nocache
        self.native_reader = (
            NativeExpertReader(native_reader_path) if native_reader_path else None
        )
        self.resident_components = resident_components or {}
        self.reset_stats()

    def reset_stats(self) -> None:
        self.stats = {
            "component_reads": 0,
            "expert_reads": 0,
            "bytes_read": 0,
            "read_seconds": 0.0,
            "parallel_batches": 0,
        }

    def open(self) -> None:
        needed = set()
        for layer_info in self.expert_reads.values():
            for component in layer_info.values():
                needed.add(component["file"])

        for file_name in sorted(needed):
            if file_name not in self._fds:
                fd = os.open(
                    self.model_path / file_name,
                    os.O_RDONLY,
                )
                if self.use_nocache:
                    fcntl.fcntl(fd, fcntl.F_NOCACHE, 1)
                self._fds[file_name] = fd
                # ACCESS_COPY = MAP_PRIVATE: writable mapping, but pages are only
                # copied on actual write (which we never do).  Reads serve from
                # the same OS page cache as pread, and from_buffer works because
                # the mapping is technically writable.
                mm = mmap.mmap(fd, 0, access=mmap.ACCESS_COPY)
                self._mmaps[file_name] = mm
                # Compute base pointer once; valid for the lifetime of mm.
                self._mmap_bases[file_name] = ctypes.addressof(
                    (ctypes.c_char * 1).from_buffer(mm)
                )

    def close(self) -> None:
        self._mmap_bases.clear()
        for mm in self._mmaps.values():
            mm.close()
        self._mmaps.clear()
        for fd in self._fds.values():
            os.close(fd)
        self._fds.clear()

    def __enter__(self) -> "ExpertStore":
        self.open()
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self.close()

    def _component_info(self, layer_idx: int, component: str) -> dict:
        return self.expert_reads[str(layer_idx)][component]

    def has_resident_component(self, layer_idx: int, component: str) -> bool:
        return component in self.resident_components.get(layer_idx, {})

    def get_resident_component(self, layer_idx: int, component: str):
        return self.resident_components[layer_idx][component]

    def read_component(self, layer_idx: int, component: str, expert_idx: int) -> bytes:
        info = self._component_info(layer_idx, component)
        fd = self._fds[info["file"]]
        offset = info["abs_offset"] + expert_idx * info["expert_stride"]
        t0 = time.perf_counter()
        data = os.pread(fd, info["expert_size"], offset)
        self.stats["component_reads"] += 1
        self.stats["bytes_read"] += len(data)
        self.stats["read_seconds"] += time.perf_counter() - t0
        return data

    def read_expert(self, layer_idx: int, expert_idx: int) -> dict[str, bytes]:
        self.stats["expert_reads"] += 1
        out = {}
        for component in self.expert_reads[str(layer_idx)].keys():
            out[component] = self.read_component(layer_idx, component, expert_idx)
        return out

    def read_experts_parallel(
        self, layer_idx: int, expert_indices: list[int], max_workers: int | None = None
    ) -> dict[int, dict[str, bytes]]:
        self.stats["parallel_batches"] += 1

        def _read(expert_idx: int) -> tuple[int, dict[str, bytes]]:
            return expert_idx, self.read_expert(layer_idx, expert_idx)

        with ThreadPoolExecutor(max_workers=max_workers or len(expert_indices)) as pool:
            return dict(pool.map(_read, expert_indices))

    def read_components_batched(
        self,
        layer_idx: int,
        expert_indices: list[int],
        components: list[str] | None = None,
    ) -> dict[str, memoryview]:
        if self.native_reader is None:
            raise RuntimeError("native reader not configured")
        layer_info = self.expert_reads[str(layer_idx)]
        selected_components = components or list(layer_info.keys())
        t0 = time.perf_counter()

        specs = []
        total_bytes = 0
        for component in selected_components:
            info = layer_info[component]
            specs.append((
                component,
                self._fds[info["file"]],
                info["abs_offset"],
                info["expert_stride"],
                info["expert_size"],
            ))
            total_bytes += info["expert_size"] * len(expert_indices)

        out = self.native_reader.read_component_batches(specs, expert_indices)
        self.stats["component_reads"] += len(selected_components) * len(expert_indices)
        self.stats["bytes_read"] += total_bytes
        self.stats["expert_reads"] += len(expert_indices)
        self.stats["read_seconds"] += time.perf_counter() - t0
        return out

    def read_components_mmap_native(
        self,
        layer_idx: int,
        expert_indices: list[int],
        components: list[str] | None = None,
    ) -> dict[str, memoryview]:
        """Read experts via parallel memcpy from mmap regions (no pread syscalls).

        For data in the OS page cache this avoids all kernel-mode transitions —
        dispatch_apply threads copy directly from mapped pages.  Cold pages still
        trigger a page fault but no system-call overhead on top.
        """
        if self.native_reader is None:
            raise RuntimeError("native reader not configured")
        layer_info = self.expert_reads[str(layer_idx)]
        selected_components = components or list(layer_info.keys())
        t0 = time.perf_counter()

        specs = [
            (
                component,
                self._mmap_bases[layer_info[component]["file"]],
                layer_info[component]["abs_offset"],
                layer_info[component]["expert_stride"],
                layer_info[component]["expert_size"],
            )
            for component in selected_components
        ]
        total_bytes = sum(
            layer_info[c]["expert_size"] * len(expert_indices)
            for c in selected_components
        )
        out = self.native_reader.copy_component_batches_mmap(specs, expert_indices)
        self.stats["component_reads"] += len(selected_components) * len(expert_indices)
        self.stats["bytes_read"] += total_bytes
        self.stats["expert_reads"] += len(expert_indices)
        self.stats["read_seconds"] += time.perf_counter() - t0
        return out

    def read_components_mmap(
        self,
        layer_idx: int,
        expert_indices: list[int],
        components: list[str] | None = None,
    ) -> dict[str, bytes]:
        """Read experts via mmap — no dispatch_apply, no kernel copy.

        For data already in the OS page cache this is a pure RAM copy at
        memory-bandwidth speed.  For cold pages it triggers a page fault and
        the OS reads from SSD, similar to pread but without the GCD overhead.
        """
        layer_info = self.expert_reads[str(layer_idx)]
        selected_components = components or list(layer_info.keys())
        t0 = time.perf_counter()
        out: dict[str, bytes] = {}
        total_bytes = 0
        for component in selected_components:
            info = layer_info[component]
            mm = self._mmaps[info["file"]]
            size = info["expert_size"]
            base = info["abs_offset"]
            stride = info["expert_stride"]
            if len(expert_indices) == 1:
                # Single expert: return a direct memoryview slice — zero extra copy
                offset = base + expert_indices[0] * stride
                out[component] = mm[offset : offset + size]
            else:
                # Multiple experts: join slices into one contiguous buffer
                out[component] = b"".join(
                    mm[base + eidx * stride : base + eidx * stride + size]
                    for eidx in expert_indices
                )
            total_bytes += size * len(expert_indices)
        self.stats["component_reads"] += len(selected_components) * len(expert_indices)
        self.stats["bytes_read"] += total_bytes
        self.stats["expert_reads"] += len(expert_indices)
        self.stats["read_seconds"] += time.perf_counter() - t0
        return out

    def read_components_batched_into_slots(
        self,
        layer_idx: int,
        expert_indices: list[int],
        slot_indices: list[int],
        out_buffers: dict[str, object],
        components: list[str] | None = None,
    ) -> None:
        if self.native_reader is None:
            raise RuntimeError("native reader not configured")
        if len(expert_indices) != len(slot_indices):
            raise ValueError("slot_indices length must match expert_indices length")
        if not expert_indices:
            return

        layer_info = self.expert_reads[str(layer_idx)]
        selected_components = components or list(layer_info.keys())
        specs = [
            (
                component,
                self._fds[layer_info[component]["file"]],
                layer_info[component]["abs_offset"],
                layer_info[component]["expert_stride"],
                layer_info[component]["expert_size"],
            )
            for component in selected_components
        ]
        t0 = time.perf_counter()
        self.native_reader.read_component_batches_into_slots(
            specs=specs,
            expert_indices=expert_indices,
            slot_indices=slot_indices,
            out_buffers=out_buffers,
        )
        self.stats["component_reads"] += len(expert_indices) * len(selected_components)
        self.stats["bytes_read"] += sum(
            layer_info[component]["expert_size"] * len(expert_indices)
            for component in selected_components
        )
        self.stats["expert_reads"] += len(expert_indices)
        self.stats["read_seconds"] += time.perf_counter() - t0
