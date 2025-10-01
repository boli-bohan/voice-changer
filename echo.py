#!/usr/bin/env python3
"""Simple WebRTC worker that echoes audio frames without modification."""

from __future__ import annotations

import asyncio
import logging
from aiortc import RTCPeerConnection, RTCSessionDescription, RTCDataChannel
from aiortc.mediastreams import MediaStreamTrack
from av import AudioFrame
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from duration_track import DurationLoggingTrack
from worker import WorkerContext, WorkerSettings, make_worker_app

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

# Suppress noisy HTTP client logs
logging.getLogger("httpx").setLevel(logging.WARNING)
logging.getLogger("httpcore").setLevel(logging.WARNING)

MAX_BUFFER_FRAMES = 2


class SDPModel(BaseModel):
    """Pydantic model representing a WebRTC SDP payload."""

    sdp: str = Field(description="Session description protocol body provided by the browser.")
    type: str = Field(description="SDP type such as 'offer' or 'answer'.")


class EchoTrack(MediaStreamTrack):
    """Media track that buffers inbound audio before forwarding it."""

    def __init__(
        self,
        source: MediaStreamTrack,
        max_buffer_frames: int = MAX_BUFFER_FRAMES,
    ) -> None:
        super().__init__()
        self._source = source
        self._max_buffer = max(1, max_buffer_frames)
        self._buffer: list[AudioFrame] = []
        self._should_flush = asyncio.Event()
        self._draining_batch = False
        self._flush_done = asyncio.Event()
        self._flush_done.set()

    async def flush(self) -> None:
        """Mark the buffer for flushing and wait until it drains."""

        self._should_flush.set()
        if self._flush_done.is_set():
            self._flush_done.clear()

        if not self._buffer:
            self._should_flush.clear()
            self._flush_done.set()
            return

        await self._flush_done.wait()

    async def recv(self) -> AudioFrame:
        """Receive and buffer audio frames from the source until the buffer is full or a flush is requested. Then return the buffered frames one by one.

        Returns:
            AudioFrame: The next audio frame from the buffer.
        """
        while True:
            # Check if we are draining the buffer because it was full.
            if self._draining_batch:
                frame = self._buffer.pop(0)
                if not self._buffer:
                    # We have drained the buffer.
                    self._draining_batch = False
                return frame

            # Check if the buffer should be flushed.
            if self._should_flush.is_set():
                frame = self._buffer.pop(0)
                if not self._buffer:
                    # We have flushed the buffer.
                    self._should_flush.clear()
                    self._flush_done.set()
                return frame

            # Wait for either a frame from source or the should_flush event to be set.
            done, pending = await asyncio.wait(
                [
                    asyncio.create_task(self._source.recv()),
                    asyncio.create_task(self._should_flush.wait()),
                ],
                return_when=asyncio.FIRST_COMPLETED,
            )

            # Cancel any pending tasks
            for task in pending:
                task.cancel()

            # Check which task completed first
            if self._should_flush.is_set():
                # Flush event was triggered, continue to flush logic
                continue

            # Source frame was received
            frame: AudioFrame = done.pop().result()
            self._buffer.append(frame)
            if len(self._buffer) >= self._max_buffer:
                # Reached max buffer size, drain the buffer before accepting new frames.
                self._draining_batch = True


SETTINGS = WorkerSettings(
    title="Echo WebRTC Worker",
    version="1.0.0",
    service_name="voice_changer_echo_worker",
    capabilities=["echo", "webrtc", "streaming_audio"],
)


def register_routes(app: FastAPI, ctx: WorkerContext) -> None:
    @app.post("/offer")
    async def handle_offer(payload: SDPModel) -> dict[str, str]:
        pc = RTCPeerConnection()
        ctx.active_peers.add(pc)
        logger.info("📡 Received new WebRTC offer; active peers: %d", len(ctx.active_peers))

        duration_wrappers: list[DurationLoggingTrack] = []
        echo_track: EchoTrack | None = None
        control_channel: RTCDataChannel | None = None
        cleanup_started = False

        async def cleanup() -> None:
            logger.info("🧹 Cleaning up echo worker")
            nonlocal cleanup_started, control_channel, echo_track
            if cleanup_started:
                return
            cleanup_started = True

            echo_track = None

            while duration_wrappers:
                duration_wrappers.pop().stop()

            if control_channel and control_channel.readyState == "open":
                try:
                    control_channel.close()
                except Exception:  # pragma: no cover - defensive close
                    logger.exception("Failed to close control data channel")

            await ctx.cleanup_peer(pc)

        @pc.on("connectionstatechange")
        async def on_connection_state_change() -> None:  # pragma: no cover - depends on network
            logger.info("Peer connection state: %s", pc.connectionState)
            if pc.connectionState in {"failed", "closed", "disconnected"}:
                await cleanup()

        track_ready = asyncio.Event()

        @pc.on("datachannel")
        def on_datachannel(channel: RTCDataChannel) -> None:
            nonlocal control_channel, echo_track
            control_channel = channel
            logger.info("🛰️ Control data channel established: %s", channel.label)

            @channel.on("message")
            async def on_message(message: object) -> None:
                if isinstance(message, bytes):
                    try:
                        message = message.decode("utf-8")
                    except Exception:
                        logger.warning("Received non-UTF8 control message; ignoring")
                        return

                if not isinstance(message, str):
                    logger.debug("Ignoring non-string control payload: %r", message)
                    return

                logger.info("📨 Control message received: %s", message)

                if message.strip().lower() == "flush":
                    # Flush the echo track and wait for the flush to complete.
                    await echo_track.flush()
                    channel.send("flush_done")
                    logger.info("🚰 Flush requested and completed")

        @pc.on("track")
        async def on_track(track: MediaStreamTrack) -> None:
            nonlocal echo_track
            logger.info("🎙️ Incoming track: %s", track.kind)
            if track.kind != "audio":
                logger.debug("Ignoring non-audio track of kind %s", track.kind)
                return

            recv_logger = DurationLoggingTrack(track, label="recv", log_interval=0.5)
            echo_track = EchoTrack(recv_logger, max_buffer_frames=MAX_BUFFER_FRAMES)
            send_logger = DurationLoggingTrack(echo_track, label="send", log_interval=0.5)
            duration_wrappers.extend([send_logger, recv_logger])
            pc.addTrack(send_logger)
            track_ready.set()

            @track.on("ended")
            async def on_track_ended() -> None:
                logger.info("🏁 Audio track ended")
                await cleanup()

        offer = RTCSessionDescription(sdp=payload.sdp, type=payload.type)

        try:
            await pc.setRemoteDescription(offer)
            try:
                await asyncio.wait_for(track_ready.wait(), timeout=5.0)
            except TimeoutError:
                logger.warning("⚠️ No audio track received before answer generation")
            answer = await pc.createAnswer()
            await pc.setLocalDescription(answer)
            await ctx.wait_for_ice_completion(pc)
        except Exception as exc:  # pragma: no cover - network failure handling
            await cleanup()
            raise HTTPException(status_code=500, detail=f"Failed to process SDP offer: {exc}")

        logger.info("✅ Generated SDP answer")
        assert pc.localDescription is not None
        return {"sdp": pc.localDescription.sdp, "type": pc.localDescription.type}


app = make_worker_app(settings=SETTINGS, logger=logger, register_routes=register_routes)


if __name__ == "__main__":
    import uvicorn

    logger.info("🚀 Starting Echo WebRTC Worker Service")
    uvicorn.run("echo:app", host="127.0.0.1", port=SETTINGS.worker_port, log_level="info")
