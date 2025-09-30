#!/usr/bin/env python3
"""
Voice Changer Worker Service powered by WebRTC streaming.

This service accepts WebRTC peer connections, applies a pitch shift to
incoming audio in real time, and streams the transformed audio back to the
client. The HTTP API exposes an SDP offer endpoint that can be used by a
signalling service (the Go API) to negotiate connections.
"""

from __future__ import annotations

import asyncio
import logging
import os
import uuid
import wave
from contextlib import asynccontextmanager
from pathlib import Path

import httpx
import librosa
import numpy as np
from aiortc import RTCPeerConnection, RTCSessionDescription
from aiortc.mediastreams import MediaStreamTrack
from av import AudioFrame
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

# Worker configuration for heartbeat and load balancing
WORKER_ID = os.environ.get("HOSTNAME", str(uuid.uuid4()))
WORKER_URL = f"http://{os.environ.get('POD_IP', '127.0.0.1')}:8001"
API_URL = os.environ.get("API_URL", "http://voice-changer-api:8000")
HEARTBEAT_INTERVAL = 5  # seconds
MAX_CONNECTIONS = 4


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Initialize worker and start background tasks on application startup."""
    asyncio.create_task(heartbeat_loop())
    logger.info("✅ Worker initialized: id=%s url=%s", WORKER_ID, WORKER_URL)
    yield
    # Shutdown: ensure all active peer connections are closed
    await asyncio.gather(*(_cleanup_peer(pc) for pc in list(active_peers)), return_exceptions=True)


app = FastAPI(title="Voice Changer WebRTC Worker", version="3.0.0", lifespan=lifespan)

# Allow local development hosts to negotiate directly with the worker.
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


class SDPModel(BaseModel):
    """Pydantic model representing a WebRTC SDP payload."""

    sdp: str
    type: str


TARGET_SAMPLE_RATE = 48_000
TARGET_CHANNELS = 1


class PitchShiftTrack(MediaStreamTrack):
    """Wraps an audio track and applies a pitch shift to buffered frames."""

    kind = "audio"

    def __init__(self, source: MediaStreamTrack, pitch_shift_semitones: float = 0.0):
        """Initialise a processing track that applies a pitch shift.

        Args:
            source: Incoming audio track from the remote peer.
            pitch_shift_semitones: Number of semitones to shift the pitch.
        """
        super().__init__()
        self._source = source
        self._pitch_shift = pitch_shift_semitones

        # Buffer configuration for overlap-add smoothing.
        self._chunk_size = 4096
        self._overlap = 1024
        self._step_size = max(1, self._chunk_size - self._overlap)
        self._n_fft = 2048
        self._hop_length = 512
        self._input_buffer = np.empty(0, dtype=np.float32)
        self._output_queue: list[np.ndarray] = []
        self._output_available = 0
        self._prev_tail: np.ndarray | None = None
        if self._overlap > 0:
            self._fade_in = np.linspace(0.0, 1.0, self._overlap, dtype=np.float32)
            self._fade_out = 1.0 - self._fade_in
        else:
            self._fade_in = self._fade_out = None

        # Optional debug audio capture to inspect input/output pairs offline.
        self._debug_dir = Path(f"/tmp/audio/{WORKER_ID}")
        self._debug_dir.mkdir(parents=True, exist_ok=True)
        self._input_wav_path = self._debug_dir / "input.wav"
        self._output_wav_path = self._debug_dir / "output.wav"
        self._input_wav = wave.open(str(self._input_wav_path), "wb")
        self._input_wav.setnchannels(TARGET_CHANNELS)
        self._input_wav.setsampwidth(2)
        self._input_wav.setframerate(TARGET_SAMPLE_RATE)
        self._output_wav = wave.open(str(self._output_wav_path), "wb")
        self._output_wav.setnchannels(TARGET_CHANNELS)
        self._output_wav.setsampwidth(2)
        self._output_wav.setframerate(TARGET_SAMPLE_RATE)
        logger.info("🎙️ Debug audio streaming enabled: %s", self._debug_dir)

        # Debug counters
        self._total_input_samples = 0
        self._total_output_samples = 0
        self._frame_count = 0

    async def recv(self) -> AudioFrame:
        """Receive an audio frame, shift its pitch, and emit the result."""
        frame = await self._source.recv()
        self._frame_count += 1

        if frame.sample_rate and frame.sample_rate != TARGET_SAMPLE_RATE:
            logger.warning(
                "⚠️ Sample rate mismatch! Incoming=%d Hz, Expected=%d Hz",
                frame.sample_rate,
                TARGET_SAMPLE_RATE,
            )

        samples = frame.to_ndarray()
        mono = self._to_mono_float(samples)
        frame_length = mono.shape[0]

        # Log frame details periodically
        if self._frame_count % 100 == 1:
            logger.info(
                "📊 Frame #%d: samples.shape=%s, mono.shape=%s, frame.sample_rate=%s, frame_length=%d",
                self._frame_count,
                samples.shape,
                mono.shape,
                frame.sample_rate,
                frame_length,
            )

        if frame_length:
            self._input_buffer = np.concatenate((self._input_buffer, mono))

        input_pcm16 = (np.clip(mono, -1.0, 1.0) * 32767.0).astype(np.int16)
        if input_pcm16.size:
            self._input_wav.writeframes(input_pcm16.tobytes())
            self._total_input_samples += input_pcm16.size

        self._ensure_output(frame_length)
        processed = self._consume_output(frame_length)

        pcm = np.clip(processed, -1.0, 1.0)
        pcm16 = (pcm * 32767.0).astype(np.int16)
        if pcm16.size:
            self._output_wav.writeframes(pcm16.tobytes())
            self._total_output_samples += pcm16.size

        array = pcm16[np.newaxis, :]
        out_frame = AudioFrame.from_ndarray(array, format="s16", layout="mono")
        out_frame.sample_rate = TARGET_SAMPLE_RATE
        out_frame.pts = frame.pts
        out_frame.time_base = frame.time_base
        return out_frame

    def close_debug_files(self) -> None:
        """Close debug WAV files and log the saved file paths."""
        try:
            self._input_wav.close()
            self._output_wav.close()

            input_duration = self._total_input_samples / TARGET_SAMPLE_RATE
            output_duration = self._total_output_samples / TARGET_SAMPLE_RATE

            logger.info(
                "💾 Debug audio saved:\n"
                "   Input:  %s (%.2fs, %d samples, %d frames)\n"
                "   Output: %s (%.2fs, %d samples)",
                self._input_wav_path,
                input_duration,
                self._total_input_samples,
                self._frame_count,
                self._output_wav_path,
                output_duration,
                self._total_output_samples,
            )
        except Exception as exc:
            logger.warning("⚠️ Error closing debug WAV files: %s", exc)

    def _to_mono_float(self, samples: np.ndarray) -> np.ndarray:
        if samples.ndim == 1:
            mono = samples.astype(np.float32, copy=False)
        else:
            mono = samples[0].astype(np.float32, copy=False)

        if np.issubdtype(samples.dtype, np.integer):
            info = np.iinfo(samples.dtype)
            denom = float(max(abs(info.min), info.max)) or 1.0
            mono = mono / denom
        elif np.issubdtype(samples.dtype, np.floating):
            pass
        else:  # pragma: no cover - defensive fallback
            logger.warning(
                "Unsupported sample dtype %s; normalising via peak amplitude", samples.dtype
            )
            peak = float(np.max(np.abs(mono))) or 1.0
            mono = mono / peak

        return np.clip(mono, -1.0, 1.0)

    def _append_output(self, chunk: np.ndarray) -> None:
        if chunk.size == 0:
            return
        self._output_queue.append(chunk.astype(np.float32, copy=False))
        self._output_available += chunk.size

    def _consume_output(self, count: int) -> np.ndarray:
        if count <= 0:
            return np.empty(0, dtype=np.float32)

        out = np.zeros(count, dtype=np.float32)
        idx = 0
        remaining = count
        while remaining > 0 and self._output_queue:
            head = self._output_queue[0]
            take = min(head.size, remaining)
            out[idx : idx + take] = head[:take]
            if take == head.size:
                self._output_queue.pop(0)
            else:
                self._output_queue[0] = head[take:]
            idx += take
            remaining -= take

        consumed = count - remaining
        self._output_available = max(self._output_available - consumed, 0)
        return out

    def _ensure_output(self, required: int) -> None:
        if required <= 0:
            return

        while self._output_available < required and self._input_buffer.size >= self._chunk_size:
            chunk = self._input_buffer[: self._chunk_size]
            self._input_buffer = self._input_buffer[self._step_size :]
            self._process_chunk(chunk, final=False)

        if self._output_available < required and self._input_buffer.size:
            chunk = self._input_buffer
            self._input_buffer = np.empty(0, dtype=np.float32)
            self._process_chunk(chunk, final=True)

    def _process_chunk(self, chunk: np.ndarray, final: bool) -> None:
        if chunk.size == 0:
            return

        if chunk.size < 2:
            shifted = chunk
        else:
            try:
                shifted = librosa.effects.pitch_shift(
                    chunk,
                    sr=TARGET_SAMPLE_RATE,
                    n_steps=self._pitch_shift,
                    n_fft=self._n_fft,
                    hop_length=self._hop_length,
                )
            except Exception as exc:  # pragma: no cover - defensive logging
                logger.error("Pitch shifting error: %s", exc)
                shifted = chunk

        self._emit_shifted_chunk(np.asarray(shifted, dtype=np.float32), final)

    def _emit_shifted_chunk(self, shifted: np.ndarray, final: bool) -> None:
        if shifted.size == 0:
            return

        if self._overlap <= 0:
            if self._prev_tail is not None:
                self._append_output(self._prev_tail)
                self._prev_tail = None
            self._append_output(shifted)
            return

        if shifted.size <= self._overlap:
            if self._prev_tail is not None:
                fade_len = min(self._prev_tail.size, shifted.size)
                cross = (
                    self._prev_tail[:fade_len] * self._fade_out[:fade_len]
                    + shifted[:fade_len] * self._fade_in[:fade_len]
                )
                self._append_output(cross)
                remainder = shifted[fade_len:]
                if remainder.size:
                    self._append_output(remainder)
                self._prev_tail = None
            else:
                self._append_output(shifted)
            return

        head = shifted[: self._overlap]
        body = shifted[self._overlap : -self._overlap]
        tail = shifted[-self._overlap :]

        if self._prev_tail is None:
            self._append_output(shifted[: -self._overlap])
        else:
            cross = self._prev_tail * self._fade_out + head * self._fade_in
            self._append_output(cross)
            if body.size:
                self._append_output(body)

        if final:
            self._append_output(tail)
            self._prev_tail = None
        else:
            self._prev_tail = tail.copy()


active_peers: set[RTCPeerConnection] = set()


async def _wait_for_ice_completion(pc: RTCPeerConnection) -> None:
    """Poll until ICE gathering completes for the provided connection."""
    while pc.iceGatheringState != "complete":
        await asyncio.sleep(0.05)


async def _cleanup_peer(pc: RTCPeerConnection) -> None:
    """Remove a peer from the active set and close its connection."""
    if pc in active_peers:
        active_peers.remove(pc)
    await pc.close()
    logger.info("🔌 Closed peer connection (%s)", pc)


async def heartbeat_loop() -> None:
    """Periodically send heartbeat to API server with current capacity information.

    This allows the API server to maintain an up-to-date registry of available
    workers and their connection counts for load balancing.
    """
    logger.info(
        "🫀 Starting heartbeat loop to %s (worker_id=%s, worker_url=%s)",
        API_URL,
        WORKER_ID,
        WORKER_URL,
    )

    async with httpx.AsyncClient() as client:
        while True:
            try:
                await client.post(
                    f"{API_URL}/heartbeat",
                    json={
                        "worker_id": WORKER_ID,
                        "worker_url": WORKER_URL,
                        "connection_count": len(active_peers),
                        "max_connections": MAX_CONNECTIONS,
                    },
                    timeout=5.0,
                )
                logger.debug(
                    "💓 Heartbeat sent: %d/%d connections", len(active_peers), MAX_CONNECTIONS
                )
            except Exception as exc:
                logger.warning("⚠️ Heartbeat failed: %s", exc)

            await asyncio.sleep(HEARTBEAT_INTERVAL)


@app.get("/")
async def get_status() -> dict[str, str]:
    """Return a simple readiness response for health probes.

    Returns:
        dict[str, str]: Service identifier and status metadata.
    """
    return {"status": "running", "service": "voice_changer_webrtc_worker", "version": "3.0.0"}


@app.get("/health")
async def health_check() -> dict[str, object]:
    """Report the current health state and connection metrics.

    Returns:
        dict[str, object]: Details about service status and active peers.
    """
    return {
        "status": "healthy",
        "active_peers": len(active_peers),
        "service": "voice_changer_webrtc_worker",
        "version": "3.0.0",
        "capabilities": ["pitch_shifting", "webrtc", "streaming_audio"],
    }


@app.get("/capacity")
async def get_capacity() -> dict[str, object]:
    """Return current worker capacity and availability information.

    Returns:
        dict[str, object]: Worker identification, load, and availability status.
    """
    return {
        "worker_id": WORKER_ID,
        "worker_url": WORKER_URL,
        "connection_count": len(active_peers),
        "max_connections": MAX_CONNECTIONS,
        "available": len(active_peers) < MAX_CONNECTIONS,
    }


@app.post("/offer")
async def handle_offer(payload: SDPModel) -> dict[str, str]:
    """Accept an SDP offer and return the corresponding answer.

    Args:
        payload: The SDP model supplied by the signalling API.

    Returns:
        dict[str, str]: The generated SDP answer describing the worker session.
    """

    pc = RTCPeerConnection()
    active_peers.add(pc)
    logger.info("📡 Received new WebRTC offer; active peers: %d", len(active_peers))

    @pc.on("connectionstatechange")
    async def on_connection_state_change() -> None:  # pragma: no cover - relies on network state
        """Monitor peer state changes and trigger cleanup when disconnected."""
        logger.info("Peer connection state: %s", pc.connectionState)
        if pc.connectionState in {"failed", "closed", "disconnected"}:
            await _cleanup_peer(pc)

    track_ready = asyncio.Event()

    @pc.on("track")
    async def on_track(track: MediaStreamTrack) -> None:
        """Handle incoming tracks by applying pitch shifting to audio streams."""
        logger.info("🎙️ Incoming track: %s", track.kind)
        if track.kind == "audio":
            transformed = PitchShiftTrack(track)
            pc.addTrack(transformed)
            track_ready.set()

            @track.on("ended")
            async def on_track_ended() -> None:
                """Log track completion and dispose of the peer connection."""
                logger.info("🏁 Audio track ended")
                transformed.close_debug_files()
                await _cleanup_peer(pc)

    offer = RTCSessionDescription(sdp=payload.sdp, type=payload.type)

    try:
        await pc.setRemoteDescription(offer)
        try:
            await asyncio.wait_for(track_ready.wait(), timeout=5.0)
        except TimeoutError:
            logger.warning("⚠️ No audio track received before answer generation")
        answer = await pc.createAnswer()
        await pc.setLocalDescription(answer)
        await _wait_for_ice_completion(pc)
    except Exception as exc:  # pragma: no cover - network failure handling
        await _cleanup_peer(pc)
        raise HTTPException(status_code=500, detail=f"Failed to process SDP offer: {exc}")

    logger.info("✅ Generated SDP answer")
    assert pc.localDescription is not None
    return {"sdp": pc.localDescription.sdp, "type": pc.localDescription.type}


if __name__ == "__main__":
    import uvicorn

    logger.info("🚀 Starting Voice Changer WebRTC Worker Service")
    uvicorn.run("voice_changer:app", host="127.0.0.1", port=8001, log_level="info")
