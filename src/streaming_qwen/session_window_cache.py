from __future__ import annotations

import threading
import time
from collections import deque
from dataclasses import dataclass, field

import numpy as np
import mlx.core as mx


@dataclass
class LayerWindow:
    expert_ids: list[int]
    slot_by_expert: dict[int, int]
    tensors: dict[str, mx.array]   # {component: mx.array shape [K, ...]}
    total_bytes: int


@dataclass
class TokenWindow:
    layers: dict[int, LayerWindow] = field(default_factory=dict)
    total_bytes: int = 0


@dataclass
class SessionState:
    session_id: str
    windows: deque[TokenWindow] = field(default_factory=deque)
    current: TokenWindow | None = None
    pinned: int = 0
    last_access: float = 0.0
    ephemeral: bool = False
    # O(1) lookup: (layer_idx, expert_idx) -> (LayerWindow, slot_in_that_window)
    expert_lut: dict[tuple[int, int], tuple[LayerWindow, int]] = field(default_factory=dict)

    def current_bytes(self) -> int:
        total = sum(window.total_bytes for window in self.windows)
        if self.current is not None:
            total += self.current.total_bytes
        return total


def _bytes_to_mx(blob, info: dict, count: int) -> mx.array:
    component_shape = (count, *info["shape"][1:])
    dtype = info["dtype"]
    if dtype == "BF16":
        raw = np.frombuffer(blob, dtype=np.uint16).reshape(component_shape)
        return mx.view(mx.asarray(raw), mx.bfloat16)
    if dtype == "U32":
        raw = np.frombuffer(blob, dtype=np.uint32).reshape(component_shape)
        return mx.asarray(raw)
    raise ValueError(f"Unsupported dtype: {dtype}")


class SessionWindowNativeCache:
    def __init__(self, *, max_bytes: int, window_tokens: int):
        self.max_bytes = max(0, int(max_bytes))
        self.window_tokens = max(0, int(window_tokens))
        self._lock = threading.RLock()
        self._sessions: dict[str, SessionState] = {}
        self._active_session_id: str | None = None
        self._phase: str | None = None
        self._enabled = False
        self._current_bytes = 0
        self._peak_bytes = 0
        self._hit_experts = 0
        self._miss_experts = 0
        self._evicted_windows = 0

    def begin_request(
        self,
        *,
        session_id: str | None,
        phase: str,
        enabled: bool,
        ephemeral: bool,
    ) -> None:
        with self._lock:
            self._enabled = bool(enabled) and self.window_tokens > 0 and bool(session_id)
            self._phase = phase if self._enabled else None
            self._active_session_id = session_id if self._enabled else None
            if not self._enabled or session_id is None:
                return
            session = self._sessions.get(session_id)
            if session is None:
                session = SessionState(session_id=session_id)
                self._sessions[session_id] = session
            session.ephemeral = ephemeral
            session.pinned += 1
            session.last_access = time.time()
            if phase == "decode" and session.current is None:
                session.current = TokenWindow()

    def set_phase(self, phase: str) -> None:
        with self._lock:
            self._phase = phase if self._enabled else None
            if not self._enabled or self._active_session_id is None:
                return
            session = self._sessions[self._active_session_id]
            session.last_access = time.time()
            if phase == "decode" and session.current is None:
                session.current = TokenWindow()

    def end_request(self) -> None:
        with self._lock:
            session_id = self._active_session_id
            if session_id is not None and session_id in self._sessions:
                session = self._sessions[session_id]
                if session.current is not None and session.current.layers:
                    self._release_window(session.current, session)
                    session.current = None
                session.pinned = max(0, session.pinned - 1)
                session.last_access = time.time()
                if session.ephemeral:
                    self._drop_session_locked(session_id)
                elif session.current is None and self._phase == "decode":
                    session.current = TokenWindow()
            self._active_session_id = None
            self._phase = None
            self._enabled = False

    def complete_token(self) -> None:
        with self._lock:
            if not self._enabled or self._phase != "decode" or self._active_session_id is None:
                return
            session = self._sessions.get(self._active_session_id)
            if session is None or session.current is None or not session.current.layers:
                if session is not None and session.current is None:
                    session.current = TokenWindow()
                return
            session.windows.append(session.current)
            session.current = TokenWindow()
            session.last_access = time.time()
            while len(session.windows) > self.window_tokens:
                self._release_window(session.windows.popleft(), session)
            self._evict_global_locked()

    def load_components_for_layer(
        self,
        *,
        layer_idx: int,
        selected_experts: list[int],
        layer_info: dict,
        expert_store,
        streamed_components: list[str],
    ) -> dict[str, mx.array] | None:
        with self._lock:
            if (
                not self._enabled
                or self._phase != "decode"
                or self._active_session_id is None
                or self.window_tokens <= 0
                or not selected_experts
            ):
                return None
            session = self._sessions.get(self._active_session_id)
            if session is None:
                return None
            session.last_access = time.time()
            if session.current is None:
                session.current = TokenWindow()

            # O(1) LUT: classify each expert as hit or miss
            hit_map: dict[int, tuple[LayerWindow, int]] = {}  # expert_id -> (lw, src_slot)
            missing_experts: list[int] = []
            missing_dst_slots: list[int] = []
            for dst_slot, expert_id in enumerate(selected_experts):
                entry = session.expert_lut.get((layer_idx, expert_id))
                if entry is None:
                    missing_experts.append(expert_id)
                    missing_dst_slots.append(dst_slot)
                else:
                    hit_map[expert_id] = entry
                    self._hit_experts += 1
            self._miss_experts += len(missing_experts)

        # --- lock released: do I/O for misses ---

        miss_tensors: dict[str, mx.array] = {}
        if missing_experts:
            payload = expert_store.read_components_batched(
                layer_idx, missing_experts, components=streamed_components
            )
            for c in streamed_components:
                miss_tensors[c] = _bytes_to_mx(payload[c], layer_info[c], len(missing_experts))

        # Assemble output dict[str, mx.array] in selected_experts order
        out: dict[str, mx.array] = {}
        selected_mx = mx.array(selected_experts, dtype=mx.int32)

        for component, info in layer_info.items():
            if expert_store.has_resident_component(layer_idx, component):
                out[component] = mx.take(
                    expert_store.get_resident_component(layer_idx, component),
                    selected_mx,
                    axis=0,
                )
                continue

            if not hit_map:
                # All misses — use directly
                out[component] = miss_tensors[component]

            elif not missing_experts:
                # All hits — gather from cached mx.arrays, no I/O, no copy
                unique_windows = {id(hit_map[e][0]) for e in selected_experts}
                if len(unique_windows) == 1:
                    # Single source window: one mx.take
                    lw = hit_map[selected_experts[0]][0]
                    src_slots = mx.array(
                        [hit_map[e][1] for e in selected_experts], dtype=mx.int32
                    )
                    out[component] = mx.take(lw.tensors[component], src_slots, axis=0)
                else:
                    # Multiple source windows: stack row by row (K=4 iterations, all lazy)
                    rows = [hit_map[e][0].tensors[component][hit_map[e][1]]
                            for e in selected_experts]
                    out[component] = mx.stack(rows, axis=0)

            else:
                # Mixed hits and misses: interleave in selected_experts order
                rows = []
                miss_idx = 0
                for e in selected_experts:
                    if e in hit_map:
                        lw, src_slot = hit_map[e]
                        rows.append(lw.tensors[component][src_slot])
                    else:
                        rows.append(miss_tensors[component][miss_idx])
                        miss_idx += 1
                out[component] = mx.stack(rows, axis=0)

        # Store assembled tensors in session for future tokens
        with self._lock:
            if (
                self._enabled
                and self._phase == "decode"
                and self._active_session_id is not None
                and self._active_session_id in self._sessions
            ):
                session = self._sessions[self._active_session_id]
                if session.current is None:
                    session.current = TokenWindow()
                existing = session.current.layers.get(layer_idx)
                if existing is not None:
                    self._release_layer_window(existing, session, layer_idx)
                    session.current.total_bytes -= existing.total_bytes
                lw_tensors = {c: out[c] for c in streamed_components}
                total_bytes = sum(t.nbytes for t in lw_tensors.values())
                new_lw = LayerWindow(
                    expert_ids=list(selected_experts),
                    slot_by_expert={eid: slot for slot, eid in enumerate(selected_experts)},
                    tensors=lw_tensors,
                    total_bytes=total_bytes,
                )
                session.current.layers[layer_idx] = new_lw
                session.current.total_bytes += new_lw.total_bytes
                self._current_bytes += new_lw.total_bytes
                self._peak_bytes = max(self._peak_bytes, self._current_bytes)
                for slot, eid in enumerate(selected_experts):
                    session.expert_lut[(layer_idx, eid)] = (new_lw, slot)

        return out

    def stats(self) -> dict[str, object]:
        with self._lock:
            sessions = {}
            for session_id, session in self._sessions.items():
                sessions[session_id] = {
                    "bytes": session.current_bytes(),
                    "windows": len(session.windows),
                    "pinned": session.pinned,
                    "ephemeral": session.ephemeral,
                }
            total = self._hit_experts + self._miss_experts
            return {
                "current_bytes": self._current_bytes,
                "peak_bytes": self._peak_bytes,
                "session_count": len(self._sessions),
                "hit_experts": self._hit_experts,
                "miss_experts": self._miss_experts,
                "hit_rate": (self._hit_experts / total) if total else 0.0,
                "evicted_windows": self._evicted_windows,
                "sessions": sessions,
            }

    def _evict_global_locked(self) -> None:
        if self.max_bytes <= 0:
            return
        while self._current_bytes > self.max_bytes:
            candidates = [
                session
                for session in self._sessions.values()
                if session.windows and session.pinned == 0
            ]
            if not candidates:
                break
            victim = min(candidates, key=lambda session: session.last_access)
            self._release_window(victim.windows.popleft(), victim)
            self._evicted_windows += 1
            if not victim.windows and victim.current is None and victim.ephemeral:
                self._sessions.pop(victim.session_id, None)

    def _drop_session_locked(self, session_id: str) -> None:
        session = self._sessions.pop(session_id, None)
        if session is None:
            return
        while session.windows:
            self._release_window(session.windows.popleft(), None)
        if session.current is not None:
            self._release_window(session.current, None)
            session.current = None

    def _release_window(
        self, window: TokenWindow, session: SessionState | None
    ) -> None:
        for layer_idx, layer_window in window.layers.items():
            self._release_layer_window(layer_window, session, layer_idx)
        window.layers.clear()
        window.total_bytes = 0

    def _release_layer_window(
        self,
        layer_window: LayerWindow,
        session: SessionState | None = None,
        layer_idx: int = -1,
    ) -> None:
        self._current_bytes -= layer_window.total_bytes
        if session is not None and layer_idx >= 0:
            for expert_idx in layer_window.expert_ids:
                key = (layer_idx, expert_idx)
                entry = session.expert_lut.get(key)
                if entry is not None and entry[0] is layer_window:
                    del session.expert_lut[key]
        # mx.array refcounts drop naturally when LayerWindow is GC'd — no explicit free needed
