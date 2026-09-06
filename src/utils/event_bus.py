"""
Real-time Event Bus for Pre-Mortem Investigations.

Provides pub/sub event streaming for Server-Sent Events (SSE).
Supports:
1. Live subscribers (asyncio.Queue).
2. Historical event replay for late-joining EventSource listeners.
3. Automatic cleanup of stale prediction event buffers.
"""
from __future__ import annotations

import asyncio
import json
import time
from typing import Any, AsyncGenerator, Dict, List, Optional, Set


class PredictionEventBus:
    """Manages real-time event distribution and replay buffers for pre-mortem runs."""

    def __init__(self, buffer_ttl_sec: float = 600.0, max_events_per_run: int = 100):
        self.buffer_ttl_sec = buffer_ttl_sec
        self.max_events_per_run = max_events_per_run
        # prediction_id -> list of (event_type, data, timestamp)
        self._history: Dict[str, List[Dict[str, Any]]] = {}
        self._timestamps: Dict[str, float] = {}
        # prediction_id -> set of asyncio.Queue
        self._subscribers: Dict[str, Set[asyncio.Queue]] = {}

    def publish(self, prediction_id: str, event_type: str, data: Dict[str, Any]) -> None:
        """Publish an event to all active subscribers and record to history."""
        now = time.time()
        pid = str(prediction_id)

        event_payload = {
            "event": event_type,
            "data": data,
            "timestamp": now,
        }

        # Store in history
        if pid not in self._history:
            self._history[pid] = []
            self._timestamps[pid] = now
        self._history[pid].append(event_payload)
        if len(self._history[pid]) > self.max_events_per_run:
            self._history[pid].pop(0)

        # Broadcast to active queues
        queues = self._subscribers.get(pid, set())
        for q in list(queues):
            try:
                q.put_nowait(event_payload)
            except Exception:
                pass

        # Housekeeping
        self._cleanup_stale()

    def subscribe(self, prediction_id: str, replay_history: bool = True) -> asyncio.Queue:
        """Register a subscriber queue for a prediction ID."""
        pid = str(prediction_id)
        queue: asyncio.Queue = asyncio.Queue()
        if pid not in self._subscribers:
            self._subscribers[pid] = set()
        self._subscribers[pid].add(queue)

        # Enqueue historical events if requested
        if replay_history and pid in self._history:
            for past_event in self._history[pid]:
                queue.put_nowait(past_event)

        return queue

    def unsubscribe(self, prediction_id: str, queue: asyncio.Queue) -> None:
        """Unregister a subscriber queue."""
        pid = str(prediction_id)
        if pid in self._subscribers:
            self._subscribers[pid].discard(queue)
            if not self._subscribers[pid]:
                del self._subscribers[pid]

    async def sse_generator(
        self,
        prediction_id: str,
        timeout_sec: float = 120.0,
        replay_history: bool = True,
    ) -> AsyncGenerator[str, None]:
        """Yield SSE-formatted strings for consumption by FastAPI StreamingResponse."""
        queue = self.subscribe(prediction_id, replay_history=replay_history)
        start_time = time.time()
        try:
            while True:
                # Check timeout
                remaining = timeout_sec - (time.time() - start_time)
                if remaining <= 0:
                    break

                try:
                    event_payload = await asyncio.wait_for(queue.get(), timeout=min(remaining, 5.0))
                    event_type = event_payload.get("event", "message")
                    data_obj = event_payload.get("data", {})
                    data_str = json.dumps(data_obj)
                    yield f"event: {event_type}\ndata: {data_str}\n\n"

                    # If this is the terminal event, terminate the stream
                    if event_type in ("complete", "error"):
                        break
                except asyncio.TimeoutError:
                    # Send keep-alive comment so the HTTP connection remains open
                    yield ": keep-alive\n\n"
        finally:
            self.unsubscribe(prediction_id, queue)

    def has_history(self, prediction_id: str) -> bool:
        return str(prediction_id) in self._history

    def get_history(self, prediction_id: str) -> List[Dict[str, Any]]:
        return list(self._history.get(str(prediction_id), []))

    def _cleanup_stale(self) -> None:
        """Prune buffers older than buffer_ttl_sec."""
        cutoff = time.time() - self.buffer_ttl_sec
        stale = [pid for pid, ts in self._timestamps.items() if ts < cutoff]
        for pid in stale:
            self._history.pop(pid, None)
            self._timestamps.pop(pid, None)
            self._subscribers.pop(pid, None)


_instance: Optional[PredictionEventBus] = None


def get_event_bus() -> PredictionEventBus:
    global _instance
    if _instance is None:
        _instance = PredictionEventBus()
    return _instance
