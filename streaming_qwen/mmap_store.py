"""Memory-mapped expert store.

Uses mmap for expert access which allows the OS to handle paging
more efficiently and potentially enables better I/O overlap.
"""
from __future__ import annotations

import json
import mmap
import os
import time
from pathlib import Path
from typing import Optional

import numpy as np


class MmapExpertStore:
    """Expert store using memory-mapped files.

    Memory mapping allows the OS kernel to manage paging, which can be
    more efficient than explicit pread() calls, especially for repeated
    access patterns.
    """

    def __init__(
        self,
        index_path: Path,
        use_madvise: bool = True,
    ):
        index_path = Path(index_path).expanduser().resolve()
        with index_path.open() as f:
            self.index = json.load(f)

        self.model_path = Path(self.index["model_path"]).expanduser().resolve()
        self.expert_reads = self.index["expert_reads"]
        self._mmaps: dict[str, tuple[int, mmap.mmap]] = {}
        self.use_madvise = use_madvise
        self.reset_stats()

    def reset_stats(self) -> None:
        self.stats = {
            "component_reads": 0,
            "expert_reads": 0,
            "bytes_read": 0,
            "read_seconds": 0.0,
        }

    def open(self) -> None:
        needed = set()
        for layer_info in self.expert_reads.values():
            for component in layer_info.values():
                needed.add(component["file"])

        for file_name in sorted(needed):
            if file_name not in self._mmaps:
                path = self.model_path / file_name
                fd = os.open(str(path), os.O_RDONLY)
                size = os.fstat(fd).st_size
                mm = mmap.mmap(fd, size, mmap.MAP_PRIVATE, mmap.PROT_READ)
                os.close(fd)  # fd can be closed after mmap
                self._mmaps[file_name] = (size, mm)

    def close(self) -> None:
        for _, mm in self._mmaps.values():
            mm.close()
        self._mmaps.clear()

    def __enter__(self) -> "MmapExpertStore":
        self.open()
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self.close()

    def _component_info(self, layer_idx: int, component: str) -> dict:
        return self.expert_reads[str(layer_idx)][component]

    def read_component(
        self, layer_idx: int, component: str, expert_idx: int
    ) -> memoryview:
        """Read a single component for a single expert."""
        info = self._component_info(layer_idx, component)
        _, mm = self._mmaps[info["file"]]
        offset = info["abs_offset"] + expert_idx * info["expert_stride"]
        size = info["expert_size"]

        t0 = time.perf_counter()
        data = mm[offset : offset + size]
        self.stats["component_reads"] += 1
        self.stats["bytes_read"] += size
        self.stats["read_seconds"] += time.perf_counter() - t0
        return memoryview(data)

    def read_components_batched(
        self,
        layer_idx: int,
        expert_indices: list[int],
        components: Optional[list[str]] = None,
    ) -> dict[str, bytes]:
        """Read multiple experts' components in batch.

        Returns concatenated bytes for each component.
        """
        layer_info = self.expert_reads[str(layer_idx)]
        selected_components = components or list(layer_info.keys())

        t0 = time.perf_counter()
        out = {}
        for component in selected_components:
            info = layer_info[component]
            _, mm = self._mmaps[info["file"]]
            expert_size = info["expert_size"]
            abs_offset = info["abs_offset"]
            stride = info["expert_stride"]

            # Concatenate all expert data for this component
            buffers = []
            for expert_idx in expert_indices:
                offset = abs_offset + expert_idx * stride
                buffers.append(mm[offset : offset + expert_size])
            out[component] = b"".join(buffers)

            self.stats["component_reads"] += len(expert_indices)
            self.stats["bytes_read"] += expert_size * len(expert_indices)

        self.stats["expert_reads"] += len(expert_indices)
        self.stats["read_seconds"] += time.perf_counter() - t0
        return out

    def read_expert(self, layer_idx: int, expert_idx: int) -> dict[str, bytes]:
        """Read all components for a single expert."""
        self.stats["expert_reads"] += 1
        out = {}
        for component in self.expert_reads[str(layer_idx)].keys():
            mv = self.read_component(layer_idx, component, expert_idx)
            out[component] = bytes(mv)
        return out
