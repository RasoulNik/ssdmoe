from __future__ import annotations

import threading
import time
from collections import deque
from dataclasses import dataclass, field

from .native_reader import NativeSlab


@dataclass
class LayerWindow:
    expert_ids: list[int]
    slot_by_expert: dict[int, int]
    slab: NativeSlab
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
    ) -> dict[str, memoryview] | None:
        with self._lock:
            if (
                not self._enabled
                or self._active_session_id is None
                or self.window_tokens <= 0
                or not selected_experts
            ):
                return None
            session = self._sessions.get(self._active_session_id)
            if session is None:
                return None
            session.last_access = time.time()
            if self._phase == "decode" and session.current is None:
                session.current = TokenWindow()

            count = len(selected_experts)
            nr = expert_store.native_reader
            expert_sizes_by_comp = {
                c: int(layer_info[c]["expert_size"]) for c in streamed_components
            }
            # Step 2: single aligned slab for all components
            slab = nr.alloc_slab(
                components=streamed_components,
                sizes={c: expert_sizes_by_comp[c] * count for c in streamed_components},
            )

            # Step 1: O(1) LUT lookup; group hits by source LayerWindow
            # hits_by_window: id(lw) -> (lw, src_slots, dst_slots)
            hits_by_window: dict[int, tuple[LayerWindow, list[int], list[int]]] = {}
            missing_experts: list[int] = []
            missing_slots: list[int] = []

            for out_slot, expert_idx in enumerate(selected_experts):
                hit = session.expert_lut.get((layer_idx, expert_idx))
                if hit is None:
                    missing_experts.append(expert_idx)
                    missing_slots.append(out_slot)
                    continue
                layer_window, source_slot = hit
                self._hit_experts += 1
                lw_id = id(layer_window)
                if lw_id not in hits_by_window:
                    hits_by_window[lw_id] = (layer_window, [], [])
                hits_by_window[lw_id][1].append(source_slot)
                hits_by_window[lw_id][2].append(out_slot)

        # --- released lock for I/O and copy ---

        if missing_experts:
            expert_store.read_components_batched_into_slots(
                layer_idx,
                missing_experts,
                missing_slots,
                {c: slab.component_ptr(c) for c in streamed_components},
                components=streamed_components,
            )

        if hits_by_window:
            # Step 3: single C call per unique source window (1 call in common case)
            dst_ptrs = [slab.component_ptr(c) for c in streamed_components]
            esizes = [expert_sizes_by_comp[c] for c in streamed_components]
            for lw, src_slots_grp, dst_slots_grp in hits_by_window.values():
                nr.copy_experts_multi(
                    src_ptrs=[lw.slab.component_ptr(c) for c in streamed_components],
                    dst_ptrs=dst_ptrs,
                    expert_sizes=esizes,
                    src_slots=src_slots_grp,
                    dst_slots=dst_slots_grp,
                )

        with self._lock:
            self._miss_experts += len(missing_experts)
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
                new_lw = LayerWindow(
                    expert_ids=list(selected_experts),
                    slot_by_expert={eid: slot for slot, eid in enumerate(selected_experts)},
                    slab=slab,
                    total_bytes=slab.total_size,
                )
                session.current.layers[layer_idx] = new_lw
                session.current.total_bytes += new_lw.total_bytes
                self._current_bytes += new_lw.total_bytes
                self._peak_bytes = max(self._peak_bytes, self._current_bytes)
                # Upsert LUT: newest window wins for each (layer_idx, expert_idx) key
                for slot, eid in enumerate(selected_experts):
                    session.expert_lut[(layer_idx, eid)] = (new_lw, slot)

        return {c: slab.view(c) for c in streamed_components}

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

    def _lookup_expert_locked(
        self,
        session: SessionState,
        layer_idx: int,
        expert_idx: int,
    ) -> tuple[LayerWindow, int] | None:
        # Kept for reference; load_components_for_layer now uses expert_lut directly.
        return session.expert_lut.get((layer_idx, expert_idx))

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
        # Pass session=None: the LUT is part of the discarded object, no cleanup needed.
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
                # Guard: only remove if this is still the entry's source window.
                if entry is not None and entry[0] is layer_window:
                    del session.expert_lut[key]
        layer_window.slab.free()
