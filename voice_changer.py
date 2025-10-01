#!/usr/bin/env python3
"""Voice changer worker that pitch-shifts PCM audio frames over WebRTC."""

from __future__ import annotations

import asyncio
import logging
import librosa
import numpy as np
from aiortc import RTCPeerConnection, RTCSessionDescription, RTCDataChannel
from aiortc.mediastreams import MediaStreamTrack
from av import AudioFrame
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from duration_track import DurationLoggingTrack
from worker import WorkerContext, WorkerSettings, make_worker_app

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

for noisy_logger in ("httpx", "httpcore"):
    logging.getLogger(noisy_logger).setLevel(logging.WARNING)


class SDPModel(BaseModel):
    """Pydantic model representing a WebRTC SDP payload."""

    sdp: str = Field(description="Session description protocol body provided by the browser.")
    type: str = Field(description="SDP type such as 'offer' or 'answer'.")


TARGET_SAMPLE_RATE = 48_000
DEFAULT_PITCH_SHIFT = 4.0
MAX_BUFFER_FRAMES = 1


class PitchShiftTrack(MediaStreamTrack):
    """Buffering track that pitch-shifts audio in batches before forwarding."""

    kind = "audio"

    def __init__(
        self,
        source: MediaStreamTrack,
        pitch_shift_semitones: float = DEFAULT_PITCH_SHIFT,
        max_buffer_frames: int = MAX_BUFFER_FRAMES,
    ) -> None:
        super().__init__()
        self._source = source
        self._pitch_shift = pitch_shift_semitones
        self._max_buffer = max(1, max_buffer_frames)
        self._input_buffer: list[AudioFrame] = []
        self._output_buffer: list[AudioFrame] = []
        self._should_flush = asyncio.Event()
        self._flush_done = asyncio.Event()
        self._flush_done.set()

        self._frame_count = 0
        self._input_samples_total = 0
        self._output_samples_total = 0
        self._last_input_rate = TARGET_SAMPLE_RATE
        self._last_output_rate = TARGET_SAMPLE_RATE

    async def flush(self) -> None:
        """Flush any buffered audio and wait until completion."""

        self._should_flush.set()
        if self._flush_done.is_set():
            self._flush_done.clear()

        if self._input_buffer:
            self._process_buffer()

        if not self._output_buffer:
            self._should_flush.clear()
            self._flush_done.set()
            return

        await self._flush_done.wait()

    async def recv(self) -> AudioFrame:
        """Return the next processed frame, buffering input as needed."""

        while True:
            # Drain any frames that were already pitch-shifted.
            if self._output_buffer:
                frame = self._output_buffer.pop(0)
                if not self._output_buffer and self._should_flush.is_set():
                    self._should_flush.clear()
                    self._flush_done.set()
                return frame

            # If a flush was requested but no output is queued yet, process the input buffer.
            if self._should_flush.is_set():
                if self._input_buffer:
                    self._process_buffer()
                    continue
                self._should_flush.clear()
                self._flush_done.set()
                continue

            # Wait for either a new source frame or a flush notification.
            source_task = asyncio.create_task(self._source.recv())
            flush_task = asyncio.create_task(self._should_flush.wait())

            done, pending = await asyncio.wait(
                [source_task, flush_task],
                return_when=asyncio.FIRST_COMPLETED,
            )

            # Cancel whichever task did not finish so we avoid leaks.
            for task in pending:
                task.cancel()

            if flush_task in done and self._should_flush.is_set():
                if self._input_buffer:
                    self._process_buffer()
                else:
                    self._should_flush.clear()
                    self._flush_done.set()
                continue

            # Consume the newly received source frame and append it to the input buffer.
            frame: AudioFrame = source_task.result()
            self._input_buffer.append(frame)
            if len(self._input_buffer) >= self._max_buffer:
                self._process_buffer()

    def _process_buffer(self) -> None:
        if not self._input_buffer:
            return

        frames = self._input_buffer
        self._input_buffer = []

        reference_frame = frames[0]
        sample_rate = reference_frame.sample_rate or TARGET_SAMPLE_RATE
        self._last_input_rate = sample_rate

        channel_arrays: list[np.ndarray] = []
        frame_lengths: list[int] = []
        frame_meta: list[tuple[AudioFrame, int]] = []

        for frame in frames:
            array = frame.to_ndarray()
            array = np.asarray(array)
            if array.ndim == 1:
                array = array[np.newaxis, :]
            frame_lengths.append(array.shape[-1])
            frame_meta.append((frame, array.shape[-1]))
            if array.dtype != np.int16:
                array = array.astype(np.float32)
                array = np.clip(array, -1.0, 1.0)
                array = (array * 32767.0).astype(np.int16)
            channel_arrays.append(array)

        concatenated = np.concatenate(channel_arrays, axis=-1)
        total_samples = concatenated.shape[-1]
        self._input_samples_total += total_samples

        float_audio = concatenated.astype(np.float32) / 32767.0

        shifted_channels: list[np.ndarray] = []
        for ch in range(float_audio.shape[0]):
            shifted_channel = librosa.effects.pitch_shift(
                float_audio[ch], sr=sample_rate, n_steps=self._pitch_shift
            )
            shifted_channel = librosa.util.fix_length(shifted_channel, size=total_samples)
            shifted_channels.append(shifted_channel.astype(np.float32))

        shifted = np.stack(shifted_channels, axis=0)
        shifted = np.clip(shifted, -1.0, 1.0)
        shifted_pcm = (shifted * 32767.0).astype(np.int16)

        offset = 0
        for original_frame, length in frame_meta:
            slice_end = offset + length
            slice_array = shifted_pcm[:, offset:slice_end]
            offset = slice_end

            out_frame = AudioFrame.from_ndarray(
                np.ascontiguousarray(slice_array),
                format=original_frame.format.name,
                layout=original_frame.layout.name,
            )
            out_frame.sample_rate = sample_rate
            out_frame.pts = original_frame.pts
            out_frame.time_base = original_frame.time_base
            self._output_buffer.append(out_frame)

        self._last_output_rate = sample_rate
        self._output_samples_total += total_samples


SETTINGS = WorkerSettings(
    title="Voice Changer WebRTC Worker",
    version="3.0.0",
    service_name="voice_changer_webrtc_worker",
    capabilities=["pitch_shifting", "webrtc", "streaming_audio"],
)


def register_routes(app: FastAPI, ctx: WorkerContext) -> None:
    @app.post("/offer")
    async def handle_offer(payload: SDPModel) -> dict[str, str]:
        pc = RTCPeerConnection()
        ctx.active_peers.add(pc)
        logger.info("📡 Received new WebRTC offer; active peers: %d", len(ctx.active_peers))

        cleanup_started = False
        duration_wrappers: list[DurationLoggingTrack] = []
        control_channel: RTCDataChannel | None = None
        pitch_track: PitchShiftTrack | None = None
        track_ready = asyncio.Event()

        async def cleanup() -> None:
            nonlocal cleanup_started
            if cleanup_started:
                return
            cleanup_started = True

            while duration_wrappers:
                duration_wrappers.pop().stop()
            if pitch_track is not None:
                pitch_track.stop()
            if control_channel and control_channel.readyState == "open":
                try:
                    control_channel.close()
                except Exception:
                    logger.exception("Failed to close control data channel")
            await ctx.cleanup_peer(pc)

        @pc.on("connectionstatechange")
        async def on_connection_state_change() -> None:  # pragma: no cover - network dependent
            logger.info("Peer connection state: %s", pc.connectionState)
            if pc.connectionState in {"failed", "closed", "disconnected"}:
                await cleanup()

        @pc.on("datachannel")
        def on_datachannel(channel: RTCDataChannel) -> None:
            nonlocal control_channel, pitch_track
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
                    await pitch_track.flush()
                    channel.send("flush_done")
                    logger.info("🚰 Flush requested and completed")
                else:
                    logger.debug("Unhandled control message: %s", message)

        @pc.on("track")
        async def on_track(track: MediaStreamTrack) -> None:
            nonlocal pitch_track
            logger.info("🎙️ Incoming track: %s", track.kind)
            if track.kind != "audio":
                logger.debug("Ignoring non-audio track of kind %s", track.kind)
                return

            recv_logger = DurationLoggingTrack(track, label="recv", log_interval=0.5)
            pitch_track = PitchShiftTrack(recv_logger, max_buffer_frames=MAX_BUFFER_FRAMES)
            send_logger = DurationLoggingTrack(pitch_track, label="send", log_interval=0.5)
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

    logger.info("🚀 Starting Voice Changer WebRTC Worker Service")
    uvicorn.run("voice_changer:app", host="127.0.0.1", port=8001, log_level="info")
