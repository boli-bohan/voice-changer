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
from contextlib import asynccontextmanager

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
    """Wraps an audio track and applies a pitch shift to each frame."""

    kind = "audio"

    def __init__(self, source: MediaStreamTrack, pitch_shift_semitones: float = 4.0):
        """Initialise a processing track that applies a pitch shift.

        Args:
            source: Incoming audio track from the remote peer.
            pitch_shift_semitones: Number of semitones to shift the pitch.
        """
        super().__init__()
        self._source = source
        self._pitch_shift = pitch_shift_semitones

    async def recv(self) -> AudioFrame:
        """Receive an audio frame, shift its pitch, and emit the result."""
        frame = await self._source.recv()

        # Get audio data as int16 PCM (WebRTC's native format)
        samples = frame.to_ndarray(format="s16")

        # Convert stereo to mono if needed (take first channel)
        if samples.ndim == 2:
            mono = samples[0].astype(np.float32)
        else:
            mono = samples.astype(np.float32)

        # Normalize int16 range [-32768, 32767] to float [-1.0, 1.0]
        mono /= 32767.0

        original_length = mono.shape[0]
        if original_length < 2:
            shifted = mono
        else:
            n_fft = 1 << (original_length.bit_length() - 1)
            n_fft = max(2, min(n_fft, 2048))
            hop_length = max(1, n_fft // 4)
            try:
                shifted = librosa.effects.pitch_shift(
                    mono,
                    sr=TARGET_SAMPLE_RATE,
                    n_steps=self._pitch_shift,
                    n_fft=n_fft,
                    hop_length=hop_length,
                )
            except Exception as exc:  # pragma: no cover - defensive logging
                logger.error("Pitch shifting error: %s", exc)
                shifted = mono

        pcm = np.clip(shifted, -1.0, 1.0)
        pcm16 = (pcm * 32767.0).astype(np.int16)
        processed = pcm16[np.newaxis, :]

        out_frame = AudioFrame.from_ndarray(processed, format="s16", layout="mono")
        out_frame.sample_rate = TARGET_SAMPLE_RATE
        out_frame.pts = frame.pts
        out_frame.time_base = frame.time_base
        return out_frame


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
    logger.info("🫀 Starting heartbeat loop to %s (worker_id=%s, worker_url=%s)",
                API_URL, WORKER_ID, WORKER_URL)

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
                logger.debug("💓 Heartbeat sent: %d/%d connections",
                             len(active_peers), MAX_CONNECTIONS)
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
