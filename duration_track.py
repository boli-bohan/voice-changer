"""Shared helpers for logging audio duration of aiortc media tracks."""

from __future__ import annotations

import asyncio
import inspect
import logging
from typing import Awaitable

from aiortc.mediastreams import MediaStreamTrack
from av import AudioFrame


logger = logging.getLogger(__name__)


def _drain_async(awaitable: Awaitable[object]) -> None:
    """Ensure background cleanup coroutines run to completion."""

    async def _consume() -> None:
        try:
            await awaitable
        except Exception:  # pragma: no cover - best effort cleanup
            logger.exception("Exception while awaiting async cleanup")

    try:
        loop = asyncio.get_running_loop()
    except RuntimeError:
        asyncio.run(_consume())
    else:
        loop.create_task(_consume())


class DurationLoggingTrack(MediaStreamTrack):
    """Proxy track that logs cumulative audio duration at fixed intervals."""

    kind = "audio"

    def __init__(
        self,
        source: MediaStreamTrack,
        label: str,
        log_interval: float = 0.5,
    ) -> None:
        super().__init__()
        self._source = source
        self._label = label
        self._log_interval = max(log_interval, 0.1)
        self._total_samples = 0
        self._last_sample_rate = 0
        self._next_log_threshold = self._log_interval
        self._stopped = False

    async def recv(self) -> AudioFrame:  # type: ignore[override]
        frame = await self._source.recv()
        assert isinstance(frame, AudioFrame)

        samples = getattr(frame, "samples", None)
        if samples is None:
            array = frame.to_ndarray()
            samples = array.shape[-1] if array.size else 0

        self._total_samples += samples
        if frame.sample_rate:
            self._last_sample_rate = frame.sample_rate

        self._maybe_log()
        return frame

    def stop(self) -> None:  # pragma: no cover - passthrough cleanup
        if self._stopped:
            return
        self._stopped = True
        self._emit_final_log()

        maybe_awaitable = super().stop()
        if inspect.isawaitable(maybe_awaitable):
            _drain_async(maybe_awaitable)

        source_stop = getattr(self._source, "stop", None)
        if callable(source_stop):
            maybe_awaitable = source_stop()
            if inspect.isawaitable(maybe_awaitable):  # type: ignore[misc]
                _drain_async(maybe_awaitable)

    def _maybe_log(self) -> None:
        if not self._last_sample_rate:
            return

        duration = self._total_samples / float(self._last_sample_rate)
        while duration >= self._next_log_threshold:
            logger.info(
                "🎚️ %s duration: %.3f s",
                self._label,
                self._next_log_threshold,
            )
            self._next_log_threshold += self._log_interval

    def _emit_final_log(self) -> None:
        if not self._last_sample_rate or not self._total_samples:
            return
        duration = self._total_samples / float(self._last_sample_rate)
        logger.info("🏁 %s final duration: %.3f s", self._label, duration)
